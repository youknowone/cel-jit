//! What `dc9146c`'s loop-key resolution costs on the per-call JIT path.
//!
//! # The subject
//!
//! `try_function_entry_jit_f` runs on EVERY call on the JIT tier. `dc9146c`
//! changed its yield probe from
//!
//! ```text
//! pooled.loop_keys.iter().any(|key| driver.has_runnable_compiled_loop(*key))
//! ```
//!
//! to
//!
//! ```text
//! pooled.loop_keys.iter().any(|key| driver.has_runnable_compiled_loop(key.resolve(driver)))
//! ```
//!
//! The probe itself is common to both, so the delta is exactly one
//! `resolve_cell_key` per loop key per call — and nothing else. That is what
//! this file measures, in isolation, rather than hunting for it inside the
//! six-minute board where it is far below the floor.
//!
//! # Why not the board
//!
//! Task #27's two-run comparison measured this harness family's noise floor on
//! this box directly: absolute per-case columns drift 8.1-8.5% mean (max
//! 18-25%) between two runs of a byte-identical binary. The effect here is a
//! pair of hash lookups per call. A board A/B cannot resolve it and would
//! return "no detected difference", which is not the same claim as "no cost".
//!
//! This instrument keeps every control the board harness has — the null that
//! states the floor, the positive control, the sensitivity ladder, the closing
//! null — but points them at the one call that changed, so the floor is
//! nanoseconds per resolve rather than percent of a whole evaluation.
//!
//! # What is measured, and in which state
//!
//! `resolve_cell_key`'s cost is a function of the BUCKET the hash names, so a
//! single number would be a fiction. All three reachable states are measured:
//!
//! * `empty`   — no cell at that bucket. Two failed lookups.
//! * `single`  — one cell, the ordinary warm shape. Two successful lookups.
//! * `chained` — two cells, which is the ONLY state `dc9146c` changes an ANSWER
//!               in, and the only one that pays the typed key's two `Vec`
//!               allocations.
//!
//! Arm A is the pre-`dc9146c` shape (`13a1ac9`): probe the bare hash. Arm B is
//! HEAD: resolve, then probe. The harness reports `B - A`, so a positive delta
//! is the cost `dc9146c` added.
//!
//! # One process, both arms
//!
//! Both shapes are compiled into this binary and selected at run time, so there
//! is no second `cargo build` and therefore no compile drift or stale-binary
//! risk to be invisible in the numbers.
//!
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use majit_ir::{GreenKey, GreenType};
use majit_metainterp::warmstate::WarmEnterState;

// ---------------------------------------------------------------------------
// Allocation counting, for the positive control's self-check only
// ---------------------------------------------------------------------------

/// Allocator events seen while [`COUNTING`] is on.
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static DEALLOCS: AtomicUsize = AtomicUsize::new(0);
/// Off for the whole timed part of the run; on only for the startup
/// self-check.
static COUNTING: AtomicBool = AtomicBool::new(false);

/// The system allocator with an off-by-default event counter in front of it.
///
/// This exists for exactly one reason: to prove that the positive control's
/// "one heap allocation and free" really is one heap allocation and free.
/// That is not a safe assumption. Measured on this box, the obvious spelling —
/// `Box::new`, read the contents, drop — compiled to two stack stores and *no
/// allocator call*: LLVM saw the pointer never escape and promoted the heap
/// block to a stack slot. The positive control then passed, resolving 0.78
/// ns/iter with the correct sign, while measuring a function call and two
/// stores. A control that certifies the instrument by measuring nothing is
/// worse than no control, and nothing in its output distinguished the two
/// cases, so the check has to come from outside the timing.
///
/// The cost it imposes on everything else is one relaxed load of a static
/// bool and a never-taken branch per allocator event. `realloc` and
/// `alloc_zeroed` are forwarded to `System` rather than left to the trait's
/// defaults, which would decompose one in-place `realloc` into
/// alloc + copy + dealloc and change the allocation behaviour of the code
/// under test.
struct CountingAlloc;

// SAFETY: every method forwards to `System`, which is a correct allocator, and
// the counters are plain atomics that do not touch the returned memory.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if COUNTING.load(Ordering::Relaxed) {
            DEALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAlloc = CountingAlloc;

/// Run `body` with allocation counting on, returning `(allocs, deallocs)`.
///
/// Single-threaded by construction: the harness runs on one thread, so the
/// counters need no synchronisation beyond being atomic.
fn count_allocs(body: impl FnOnce()) -> (usize, usize) {
    ALLOCS.store(0, Ordering::Relaxed);
    DEALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    body();
    COUNTING.store(false, Ordering::Relaxed);
    (
        ALLOCS.load(Ordering::Relaxed),
        DEALLOCS.load(Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config {
    /// Paired rounds per comparison. Odd by default so the median of the
    /// differences is an observed value rather than an average of two.
    rounds: usize,
    /// Thread CPU time each arm should burn per round. The inner iteration
    /// count is calibrated to hit this, so the round length is a property of
    /// the harness rather than of how expensive the arm happens to be.
    batch: Duration,
    /// Bootstrap resamples behind each reported interval.
    resamples: usize,
    /// Bootstrap seed. Fixed, so the same binary on the same samples prints the
    /// same interval.
    seed: u64,
    /// Whether to run the sensitivity ladder.
    ladder: bool,
}

impl Config {
    fn from_env() -> Config {
        fn num<T: std::str::FromStr>(key: &str, default: T) -> T {
            std::env::var(key)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        Config {
            rounds: num("PAIRED_AB_ROUNDS", 81usize).max(5),
            batch: Duration::from_micros(num("PAIRED_AB_BATCH_US", 1000u64).max(50)),
            resamples: num("PAIRED_AB_RESAMPLES", 20_000usize).max(200),
            seed: num("PAIRED_AB_SEED", 0x5EED_1234_ABCD_0001u64),
            ladder: num("PAIRED_AB_LADDER", 1u32) != 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Clocks
// ---------------------------------------------------------------------------

/// CPU time burned by *this thread* so far.
///
/// Per-thread rather than per-process: the batches run on one thread, and a
/// process-wide clock would fold in whatever else the process happens to spawn,
/// which is a property of the program's structure and not of the code being
/// timed.
fn cpu_now() -> Duration {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: the call writes through the pointer and does nothing else, and
    // the pointer is to a live local of exactly the type it expects.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    // Checked, because both quiet failure modes are worse than a panic: a clock
    // stuck at zero leaves calibration growing the iteration count forever
    // hunting a batch that never gets long enough, and a stale one publishes a
    // per-iteration figure that looks ordinary and is invented.
    assert_eq!(
        rc,
        0,
        "clock_gettime(CLOCK_THREAD_CPUTIME_ID): {}",
        std::io::Error::last_os_error()
    );
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// Run `iters` iterations of one arm, returning `(thread CPU, wall)`.
fn timed(iters: usize, run: &mut dyn FnMut()) -> (Duration, Duration) {
    let wall_start = Instant::now();
    let cpu_start = cpu_now();
    for _ in 0..iters {
        run();
    }
    // Thread CPU time never runs backwards, so this cannot underflow.
    let cpu = cpu_now() - cpu_start;
    (cpu, wall_start.elapsed())
}

/// Choose an inner iteration count whose batch burns at least `target` thread
/// CPU time.
///
/// Aims straight at the target instead of doubling: an arm costing
/// milliseconds would otherwise spend most of calibration discovering that, and
/// one costing nanoseconds would spend twenty doublings.
fn calibrate(run: &mut dyn FnMut(), target: Duration) -> usize {
    let mut iters = 1usize;
    loop {
        let (cpu, _) = timed(iters, run);
        if cpu >= target {
            return iters;
        }
        let grown = if cpu.is_zero() {
            // The clock could not separate the batch from its own read. Jump
            // hard rather than crawl.
            iters.saturating_mul(16)
        } else {
            let ratio = target.as_secs_f64() / cpu.as_secs_f64();
            ((iters as f64) * ratio * 1.25) as usize
        };
        // Always make progress even when the estimate rounds back down.
        iters = grown.max(iters + 1).min(1 << 30);
    }
}

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

/// One side of a comparison: a name for the report and a closure to run.
struct Arm<'a> {
    name: String,
    run: Box<dyn FnMut() + 'a>,
}

impl<'a> Arm<'a> {
    fn new(name: impl Into<String>, run: impl FnMut() + 'a) -> Arm<'a> {
        Arm {
            name: name.into(),
            run: Box::new(run),
        }
    }
}

/// One unit of real, unfoldable work for the positive control: a heap
/// allocation and its matching free.
///
/// `#[inline(never)]` keeps the allocation from being reasoned about together
/// with the caller, but on its own that is not enough — the promotion that ate
/// this function's allocation the first time happened *within* the function,
/// because the pointer never escaped it. `black_box(&mut boxed)` fixes that by
/// making the box itself observable, not merely its contents: an opaque
/// consumer might read or replace the pointer, so it has to be a real one.
///
/// The startup self-check in `main` confirms this rather than trusting it.
#[inline(never)]
fn known_alloc_unit() {
    let mut boxed: Box<u64> = Box::new(black_box(0x5DEE_CE66_D000_0001));
    black_box(&mut boxed);
    drop(boxed);
}

/// `units` steps of a dependent integer chain — the sensitivity ladder's unit
/// of injected cost.
///
/// Each step depends on the previous one, so the steps cannot be run in
/// parallel by the machine and the cost really does scale with `units`. The
/// `black_box` per step is a compiler fence, not work: it stops the chain from
/// being closed-form folded, and the loop count is itself opaque so the loop
/// cannot be unrolled away for small rungs.
#[inline(never)]
fn spin(units: u32) -> u64 {
    let mut x = black_box(0x9E37_79B9_7F4A_7C15u64);
    for _ in 0..black_box(units) {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        x = black_box(x);
    }
    x
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// One arm's per-round cost, in nanoseconds per inner iteration.
struct Samples {
    cpu: Vec<f64>,
    wall: Vec<f64>,
}

impl Samples {
    fn with_capacity(n: usize) -> Samples {
        Samples {
            cpu: Vec::with_capacity(n),
            wall: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, cpu: Duration, wall: Duration, iters: usize) {
        let per = iters as f64;
        self.cpu.push(cpu.as_nanos() as f64 / per);
        self.wall.push(wall.as_nanos() as f64 / per);
    }
}

/// A completed comparison: both arms' per-round costs, plus the shared inner
/// iteration count that makes the rounds comparable.
struct PairRun {
    name_a: String,
    name_b: String,
    iters: usize,
    a: Samples,
    b: Samples,
}

/// Run one interleaved paired comparison.
///
/// Both arms use the *same* inner iteration count — calibrated on whichever
/// arm needs more of them to fill a batch — because a per-round difference
/// between two different iteration counts is not a difference.
fn run_pair<'a>(a: &mut Arm<'a>, b: &mut Arm<'a>, cfg: &Config) -> PairRun {
    let iters_a = calibrate(&mut *a.run, cfg.batch);
    let iters_b = calibrate(&mut *b.run, cfg.batch);
    let iters = iters_a.max(iters_b);

    // Warm both arms at the real iteration count before anything is recorded,
    // so the first round is not paying for cold caches and a cold branch
    // predictor on one arm only.
    let _ = timed(iters, &mut *a.run);
    let _ = timed(iters, &mut *b.run);

    let mut sa = Samples::with_capacity(cfg.rounds);
    let mut sb = Samples::with_capacity(cfg.rounds);

    for round in 0..cfg.rounds {
        // ABBA: alternate which arm goes first so that whatever the first arm
        // in a round pays for — a cold line, a migrated thread, a timer
        // interrupt landing at a fixed offset — is charged to A and B in equal
        // measure across the run rather than to one of them systematically.
        if round % 2 == 0 {
            let (ca, wa) = timed(iters, &mut *a.run);
            let (cb, wb) = timed(iters, &mut *b.run);
            sa.push(ca, wa, iters);
            sb.push(cb, wb, iters);
        } else {
            let (cb, wb) = timed(iters, &mut *b.run);
            let (ca, wa) = timed(iters, &mut *a.run);
            sa.push(ca, wa, iters);
            sb.push(cb, wb, iters);
        }
    }

    PairRun {
        name_a: a.name.clone(),
        name_b: b.name.clone(),
        iters,
        a: sa,
        b: sb,
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// A tiny xorshift64* generator.
///
/// Hand-written on purpose: the bootstrap needs a *reproducible* stream and no
/// new dependency, and those two requirements together rule out anything from
/// the ecosystem. Seeded from the config, so two runs of the same binary over
/// the same samples print the same interval.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        // A zero state is the one fixed point of xorshift; steer away from it.
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A uniform index below `n`, by the multiply-high method — no modulo, so
    /// no modulo bias.
    fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

fn sorted(values: &[f64]) -> Vec<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v
}

/// Median of an already-sorted slice.
fn median_sorted(v: &[f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The `p`-quantile of an already-sorted slice, by nearest rank.
fn quantile_sorted(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let idx = (p * v.len() as f64) as usize;
    v[idx.min(v.len() - 1)]
}

/// A 95% percentile-bootstrap interval on the median of the paired
/// differences.
///
/// Distribution-free: it assumes nothing about the shape of the differences,
/// which matters here because contention makes them heavily right-skewed —
/// a few rounds cost far more than typical, and none cost far less.
fn bootstrap_median_ci(diffs: &[f64], cfg: &Config) -> (f64, f64) {
    let n = diffs.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let mut rng = Rng::new(cfg.seed);
    let mut medians = Vec::with_capacity(cfg.resamples);
    let mut draw = vec![0.0f64; n];
    for _ in 0..cfg.resamples {
        for slot in draw.iter_mut() {
            *slot = diffs[rng.below(n)];
        }
        draw.sort_by(f64::total_cmp);
        medians.push(median_sorted(&draw));
    }
    medians.sort_by(f64::total_cmp);
    (
        quantile_sorted(&medians, 0.025),
        quantile_sorted(&medians, 0.975),
    )
}

/// Two-sided sign-test p-value: the probability that a fair coin would produce
/// a split at least this lopsided.
fn sign_test_p(pos: usize, neg: usize) -> f64 {
    let n = pos + neg;
    if n == 0 {
        return 1.0;
    }
    let k = pos.max(neg);
    // Accumulated in log space: the binomial coefficients overflow f64 well
    // before the round counts here become unreasonable.
    let mut ln_term = (n as f64) * -std::f64::consts::LN_2;
    let mut tail = 0.0f64;
    for i in 0..=n {
        if i >= k {
            tail += ln_term.exp();
        }
        if i < n {
            ln_term += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
        }
    }
    (2.0 * tail).min(1.0)
}

/// Everything the report needs about one arm-pair on one clock.
struct Stats {
    a_min: f64,
    a_median: f64,
    b_min: f64,
    b_median: f64,
    /// Median of the per-round differences `b - a`. The headline.
    median_diff: f64,
    ci_lo: f64,
    ci_hi: f64,
    /// `min(b) - min(a)`: an independent corroborator built from a different
    /// estimator, so agreement between it and `median_diff` means something.
    min_diff: f64,
    pos: usize,
    neg: usize,
    zero: usize,
    p: f64,
}

fn stats(a: &[f64], b: &[f64], cfg: &Config) -> Stats {
    let sa = sorted(a);
    let sb = sorted(b);
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| y - x).collect();
    let sd = sorted(&diffs);
    let (ci_lo, ci_hi) = bootstrap_median_ci(&diffs, cfg);
    let pos = diffs.iter().filter(|d| **d > 0.0).count();
    let neg = diffs.iter().filter(|d| **d < 0.0).count();
    Stats {
        a_min: sa.first().copied().unwrap_or(f64::NAN),
        a_median: median_sorted(&sa),
        b_min: sb.first().copied().unwrap_or(f64::NAN),
        b_median: median_sorted(&sb),
        median_diff: median_sorted(&sd),
        ci_lo,
        ci_hi,
        min_diff: sb.first().copied().unwrap_or(f64::NAN) - sa.first().copied().unwrap_or(f64::NAN),
        pos,
        neg,
        zero: diffs.len() - pos - neg,
        p: sign_test_p(pos, neg),
    }
}

// ---------------------------------------------------------------------------
// Verdicts
// ---------------------------------------------------------------------------

enum Verdict {
    /// The delta is smaller than the harness's own noise. No number with a
    /// sign may be quoted for it.
    BelowFloor,
    /// Above the floor, but the interval on the median still admits zero.
    SpansZero,
    /// The two estimators point opposite ways. Not a small result — a broken
    /// one.
    SignDisagreement,
    /// A real delta. `corroborated` records whether the per-arm minima agreed
    /// in sign at a magnitude above the floor, or were themselves too small to
    /// say.
    Resolved { corroborated: bool },
}

fn verdict(s: &Stats, floor: f64) -> Verdict {
    if !s.median_diff.is_finite() || s.median_diff.abs() < floor {
        return Verdict::BelowFloor;
    }
    if s.ci_lo <= 0.0 && s.ci_hi >= 0.0 {
        return Verdict::SpansZero;
    }
    if s.min_diff.abs() >= floor && s.min_diff.signum() != s.median_diff.signum() {
        return Verdict::SignDisagreement;
    }
    Verdict::Resolved {
        corroborated: s.min_diff.abs() >= floor,
    }
}

impl Verdict {
    fn resolved(&self) -> bool {
        matches!(self, Verdict::Resolved { .. })
    }

    fn render(&self, s: &Stats, floor: f64, name_a: &str, name_b: &str) -> String {
        match self {
            Verdict::BelowFloor => format!(
                "UNRESOLVED (below floor): |{:.3}| < {:.3} ns/iter",
                s.median_diff, floor
            ),
            Verdict::SpansZero => format!(
                "UNRESOLVED (interval spans zero): 95% CI [{:+.3}, {:+.3}]",
                s.ci_lo, s.ci_hi
            ),
            Verdict::SignDisagreement => format!(
                "!! SIGN DISAGREEMENT: paired median {:+.3} vs per-arm min {:+.3} — \
                 the estimators point opposite ways, which is contamination, not a result",
                s.median_diff, s.min_diff
            ),
            Verdict::Resolved { corroborated } => {
                let (faster, slower, mag) = if s.median_diff > 0.0 {
                    (name_a, name_b, s.median_diff)
                } else {
                    (name_b, name_a, -s.median_diff)
                };
                let note = if *corroborated {
                    "corroborated by per-arm min"
                } else {
                    "per-arm min below floor, uncorroborated"
                };
                format!(
                    "RESOLVED: `{slower}` is slower than `{faster}` by {mag:.3} ns/iter \
                     (95% CI [{:+.3}, {:+.3}], {note})",
                    s.ci_lo, s.ci_hi
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// The floor pair: one for each clock, so a wall-clock number is never graded
/// against a thread-CPU floor.
#[derive(Clone, Copy)]
struct Floors {
    cpu: f64,
    wall: f64,
}

/// The floor a null control establishes: the largest magnitude it could not
/// rule out. Taking the interval ends as well as the point estimate is
/// deliberate — a null whose median lands at 0.01 but whose interval reaches
/// 1.4 has not shown that 1.0 is measurable.
fn floor_from(s: &Stats) -> f64 {
    s.median_diff
        .abs()
        .max(s.ci_lo.abs())
        .max(s.ci_hi.abs())
        .max(0.001)
}

fn print_arm_line(tag: &str, name: &str, min: f64, median: f64) {
    println!("  {tag:<5} {min:>10.3} {median:>10.3}   {name}");
}

/// Print one comparison. `floors` is `None` for the null control itself, which
/// is what establishes them.
fn report(title: &str, run: &PairRun, cfg: &Config, floors: Option<Floors>) -> (Stats, Stats) {
    let cpu = stats(&run.a.cpu, &run.b.cpu, cfg);
    let wall = stats(&run.a.wall, &run.b.wall, cfg);

    println!();
    println!("== {title}");
    println!(
        "  rounds {} x {} inner iters/arm",
        run.a.cpu.len(),
        run.iters
    );
    println!("  {:<5} {:>10} {:>10}   arm", "clock", "min", "median");
    print_arm_line("cpu A", &run.name_a, cpu.a_min, cpu.a_median);
    print_arm_line("cpu B", &run.name_b, cpu.b_min, cpu.b_median);
    print_arm_line("wall A", &run.name_a, wall.a_min, wall.a_median);
    print_arm_line("wall B", &run.name_b, wall.b_min, wall.b_median);
    println!(
        "  paired median (cpu)  {:+.3} ns/iter ({:+.2}% of A)   95% CI [{:+.3}, {:+.3}]   \
         sign {}+/{}-/{}=  p={:.4}",
        cpu.median_diff,
        100.0 * cpu.median_diff / cpu.a_median,
        cpu.ci_lo,
        cpu.ci_hi,
        cpu.pos,
        cpu.neg,
        cpu.zero,
        cpu.p
    );
    println!(
        "  paired median (wall) {:+.3} ns/iter   95% CI [{:+.3}, {:+.3}]",
        wall.median_diff, wall.ci_lo, wall.ci_hi
    );
    println!(
        "  per-arm min delta    cpu {:+.3}   wall {:+.3}  (corroborator)",
        cpu.min_diff, wall.min_diff
    );

    match floors {
        None => {
            println!("  verdict              (this run defines the floor; see below)");
        }
        Some(f) => {
            let v = verdict(&cpu, f.cpu);
            println!(
                "  verdict (cpu)        {}",
                v.render(&cpu, f.cpu, &run.name_a, &run.name_b)
            );
            let vw = verdict(&wall, f.wall);
            println!(
                "  verdict (wall)       {}",
                vw.render(&wall, f.wall, &run.name_a, &run.name_b)
            );
        }
    }
    (cpu, wall)
}

// ---------------------------------------------------------------------------
// Load average
// ---------------------------------------------------------------------------

/// The 1/5/15-minute load averages.
///
/// Shelled out to `sysctl` rather than linked: `getloadavg` is not in every
/// `libc` binding this crate could be built against, and the whole point of
/// this file is that it adds no dependency. A failure to read is reported as
/// unknown, never as zero — a fabricated quiet box is the worst possible
/// reading.
fn loadavg() -> Option<[f64; 3]> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let nums: Vec<f64> = text
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    match nums.as_slice() {
        [a, b, c, ..] => Some([*a, *b, *c]),
        _ => None,
    }
}

fn render_load(l: Option<[f64; 3]>) -> String {
    match l {
        Some([a, b, c]) => format!("{a:.2} {b:.2} {c:.2}"),
        None => "unknown".to_string(),
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/// The real pair under test: two spellings of the same CEL evaluation, one
/// calling `size` as a global function and one as a receiver method. Same
/// answer, same public door, and the only difference is how the call is
/// dispatched — which is the kind of question this harness exists to settle.
/// Arm A: the shape before `dc9146c`.
const NAME_A: &str = "13a1ac9 (probe the bare hash)";
/// Arm B: HEAD.
const NAME_B: &str = "dc9146c (resolve, then probe)";

// ---------------------------------------------------------------------------
// The subject
// ---------------------------------------------------------------------------

/// The three green slots `can_enter_jit!` builds for a `greens = [pc, program]`
/// driver, in the order `green_key_at` folds them: the marker's own position
/// argument, then each declared green.
fn key_of(pc: i64, program: i64) -> GreenKey {
    GreenKey::with_types(
        vec![pc, pc, program],
        vec![GreenType::Int, GreenType::Int, GreenType::Int],
    )
}

/// Which shape the bucket the probe reads is in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bucket {
    Empty,
    Single,
    Chained,
}

impl Bucket {
    fn label(self) -> &'static str {
        match self {
            Bucket::Empty => "empty   (no cell at the bucket)",
            Bucket::Single => "single  (one cell — the ordinary warm shape)",
            Bucket::Chained => "chained (two cells — the state dc9146c changes an ANSWER in)",
        }
    }
}

/// A warmstate whose bucket for `key` is in the requested shape.
///
/// The chain is built the way one arises in production without any 64-bit
/// collision at all: a hash-form writer installs a cell with `comparekey: None`
/// (and `mark_as_being_traced` sets the `TRACING` flag, so `should_remove_jitcell`
/// keeps it), and the typed writer behind it cannot match a comparator-less cell,
/// so it installs a second one and `install_new_cell` links them.
fn fixture(state: Bucket) -> (WarmEnterState, GreenKey) {
    let mut warm = WarmEnterState::new(100);
    let key = key_of(0x11, 0x7f00_0000_1000);
    match state {
        Bucket::Empty => {}
        Bucket::Single => {
            warm.ensure_cell_key(&key);
        }
        Bucket::Chained => {
            warm.mark_as_being_traced(key.get_uhash());
            warm.ensure_cell_key(&key);
        }
    }
    (warm, key)
}

/// Arm A — the shape before `dc9146c` (`13a1ac9`): probe the bare bucket hash.
#[inline(never)]
fn probe_bare(warm: &WarmEnterState, hash: u64) -> bool {
    black_box(warm.get_procedure_token(black_box(hash)).is_some())
}

/// Arm B — HEAD: resolve the key, then probe the cell it names.
///
/// The closure reaches `resolve_cell_key` as a `&dyn Fn`, which is how the real
/// door delivers it (`JitDriver::resolve_cell_key` hands `MetaInterp` an
/// `Option<&dyn Fn() -> GreenKey>`), so the indirection is not optimized away
/// here in a way it is not there. It is CALLED only on a chained bucket.
#[inline(never)]
fn probe_resolved(warm: &WarmEnterState, hash: u64, make: &dyn Fn() -> GreenKey) -> bool {
    let key = warm.resolve_cell_key(black_box(hash), make);
    black_box(warm.get_procedure_token(key).is_some())
}

/// How many loop keys the door actually walks, per call, for each program shape
/// the board runs.
///
/// The multiplier the per-resolve cost above has to be scaled by, and it is NOT
/// "the number of loops a reader counts in the source". `loop_header_keys` is a
/// WORD SCAN: every word equal to `OP_JUMP_IF_ABOVE` whose `pc+3` names an
/// earlier position counts, so a register number or a constant that happens to
/// equal 16 contributes a key. The function's own doc records one such spurious
/// match. Quoting a per-call cost without this census would be quoting the
/// structural loop count, which is a different and smaller number.
#[cfg(feature = "loop-key-arm-probe")]
fn loop_key_census() {
    use cel::majit::batch::BatchProgram;
    use cel::majit::lower::{BatchReduce, Schema, ValType};
    use cel::Program;

    println!();
    println!("== LOOP-KEY CENSUS (the per-call multiplier)");
    println!("  keys  loops  program");
    // A LIST column is declared by its ELEMENT path, `name[]` — one schema
    // entry per field, which for a list of scalars is one entry.
    let scalar: Schema = [("x".to_string(), ValType::Int)].into_iter().collect();
    let list: Schema = [("list[]".to_string(), ValType::Int)].into_iter().collect();
    let items: Schema = [("items[]".to_string(), ValType::Int)]
        .into_iter()
        .collect();
    let cases: [(&str, &Schema, usize); 7] = [
        ("1 + 2 * 3 - 4 / 2", &scalar, 0),
        ("x > 10 ? x * 2 : x + 5", &scalar, 0),
        ("list[0] + list[5] + list[9]", &list, 0),
        ("list.map(x, x * 2)", &list, 1),
        ("list.filter(x, x % 2 == 0)", &list, 1),
        ("[1, 2, 3, 4, 5].map(x, x * 2)", &scalar, 1),
        ("items.filter(x, x % 2 == 0).map(x, x * 2)", &items, 2),
    ];
    for (src, schema, loops) in cases {
        let Ok(program) = Program::compile(src) else {
            println!("  {:>4}  {loops:>5}  {src}  (does not parse)", "-");
            continue;
        };
        match BatchProgram::from_program(&program, schema) {
            Ok(lowered) => {
                let shape = lowered.lowered().batch_shape(true, BatchReduce::PerRow);
                let keys = cel::majit::bytecode::float_bank::loop_key_count(&shape.code);
                let flag = if keys > loops {
                    "  <- MORE keys than loops"
                } else {
                    ""
                };
                println!("  {keys:>4}  {loops:>5}  {src}{flag}");
            }
            Err(_) => println!("  {:>4}  {loops:>5}  {src}  (majit declines it)", "-"),
        }
    }
    println!("  `loops` is what a reader counts in the source; `keys` is what the door walks.");
}

fn main() {
    let cfg = Config::from_env();
    let load_before = loadavg();
    let started = Instant::now();

    println!("paired_ab — interleaved paired A/B timing harness");
    println!("  primary clock   CLOCK_THREAD_CPUTIME_ID (this thread's CPU time)");
    println!("  secondary clock std::time::Instant (wall)");
    println!(
        "  rounds {}   batch {} us CPU/arm/round   bootstrap {} resamples, seed {:#x}",
        cfg.rounds,
        cfg.batch.as_micros(),
        cfg.resamples,
        cfg.seed
    );
    println!("  load before     {}", render_load(load_before));

    // The fixture the controls and the headline share: one warmstate whose
    // bucket holds exactly one cell, which is the shape a warm loop key is in.
    // Built once, outside every timed region.
    let (warm_s, key_s) = fixture(Bucket::Single);
    let hash_s = key_s.get_uhash();
    let make_s = || GreenKey::with_types(key_s.values.clone(), key_s.types.clone());
    let make_sd: &dyn Fn() -> GreenKey = &make_s;

    // A paired A/B of two arms that answer different questions is not a
    // measurement of anything, so this is checked rather than assumed. On an
    // unchained bucket the two shapes MUST agree -- that is the premise
    // dc9146c rests on, and if it failed here the delta below would be the
    // cost of a behaviour change rather than of a resolution.
    assert_eq!(
        probe_bare(&warm_s, hash_s),
        probe_resolved(&warm_s, hash_s, make_sd),
        "on a single-cell bucket both shapes must name the same cell"
    );
    println!(
        "  arms agree on   {:?} (single-cell bucket)",
        probe_bare(&warm_s, hash_s)
    );

    // The positive control certifies the instrument, so something has to
    // certify the positive control. Counting the allocator events of one call
    // is that: it answers "did this unit of work happen" from outside the
    // timing, where the timing itself cannot tell a real allocation from a
    // stack slot the optimizer left in its place.
    let (allocs, deallocs) = count_allocs(known_alloc_unit);
    let unit_is_real = allocs >= 1 && deallocs >= 1;
    println!("  control unit    {allocs} alloc / {deallocs} free per call");
    if !unit_is_real {
        println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        println!("!! POSITIVE CONTROL UNIT IS NOT REAL WORK");
        println!("!! The unit meant to cost one heap allocation performed none: the");
        println!("!! optimizer removed it. Whatever the positive control reports below");
        println!("!! is the cost of an empty call, so it certifies nothing.");
        println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }

    // -- NULL CONTROL -------------------------------------------------------
    // Arm A against an identical copy of arm A. The true delta is zero, so
    // every nanosecond it reports is the harness's own noise at this load.
    let null_run = {
        let mut a = Arm::new("A (probe the bare hash)", || {
            probe_bare(&warm_s, hash_s);
        });
        let mut b = Arm::new("A' (identical copy)", || {
            probe_bare(&warm_s, hash_s);
        });
        run_pair(&mut a, &mut b, &cfg)
    };
    let (null_cpu, null_wall) = report(
        "NULL CONTROL: arm A vs an identical copy of arm A",
        &null_run,
        &cfg,
        None,
    );
    let floors = Floors {
        cpu: floor_from(&null_cpu),
        wall: floor_from(&null_wall),
    };
    println!();
    println!("RESOLUTION FLOOR: ±{:.3} ns/iter", floors.cpu);
    println!(
        "  (thread CPU; max of |null median| {:.3}, |CI lo| {:.3}, |CI hi| {:.3}. \
         Wall-clock floor ±{:.3} ns/iter.)",
        null_cpu.median_diff.abs(),
        null_cpu.ci_lo.abs(),
        null_cpu.ci_hi.abs(),
        floors.wall
    );
    println!("  Every delta below is graded against this. Nothing smaller may be quoted.");

    // -- POSITIVE CONTROL ---------------------------------------------------
    // Arm A against arm A plus one heap allocation and free. The cost is real
    // and the sign is known in advance, so this is the check that the harness
    // is working at all right now.
    let pos_run = {
        let mut a = Arm::new("A", || {
            probe_bare(&warm_s, hash_s);
        });
        let mut b = Arm::new("A + one Box::new/drop", || {
            probe_bare(&warm_s, hash_s);
            known_alloc_unit();
        });
        run_pair(&mut a, &mut b, &cfg)
    };
    let (pos_cpu, _) = report(
        "POSITIVE CONTROL: arm A vs arm A + one heap alloc/free",
        &pos_run,
        &cfg,
        Some(floors),
    );
    let pos_v = verdict(&pos_cpu, floors.cpu);
    let positive_ok = unit_is_real && pos_v.resolved() && pos_cpu.median_diff > 0.0;
    println!();
    if positive_ok {
        println!(
            "POSITIVE CONTROL OK: a known real cost resolved with the correct sign \
             ({:+.3} ns/iter).",
            pos_cpu.median_diff
        );
    } else {
        println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        println!("!! POSITIVE CONTROL FAILED");
        println!("!! A known real cost did not resolve with the correct sign at this");
        println!("!! load. The harness is not working right now; treat every delta");
        println!("!! below as unproven regardless of what it says.");
        println!("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    }

    // -- SENSITIVITY LADDER -------------------------------------------------
    // How small an injected delta survives, in the same regime the real arms
    // live in. The smallest rung that resolves is this box's answer today; it
    // is measured, not assumed.
    let mut smallest_resolved: Option<(u32, f64)> = None;
    if cfg.ladder {
        for units in [1u32, 2, 4, 8, 16, 32] {
            let run = {
                let mut a = Arm::new("A + spin(0)", || {
                    probe_bare(&warm_s, hash_s);
                    black_box(spin(0));
                });
                let mut b = Arm::new(format!("A + spin({units})"), || {
                    probe_bare(&warm_s, hash_s);
                    black_box(spin(units));
                });
                run_pair(&mut a, &mut b, &cfg)
            };
            let (s, _) = report(
                &format!("SENSITIVITY LADDER: +{units} chained integer steps"),
                &run,
                &cfg,
                Some(floors),
            );
            let v = verdict(&s, floors.cpu);
            if smallest_resolved.is_none() && v.resolved() && s.median_diff > 0.0 {
                smallest_resolved = Some((units, s.median_diff));
            }
        }
    }

    // -- THE ARMS UNDER TEST ------------------------------------------------
    // Replace these two closures to measure something else. Nothing above needs
    // to change: the controls, the floor and the verdicts are all machinery.
    // The two states that are NOT the headline, reported for the record. Both
    // are run before the headline so the closing null still brackets them.
    //
    // `chained` is also the only place the two arms can DISAGREE, and whether
    // they do is printed rather than assumed: that disagreement is the whole
    // reason dc9146c exists, and a run where it did not appear would mean the
    // fixture failed to build a chain and the cost below was measured on the
    // wrong shape.
    for state in [Bucket::Empty, Bucket::Chained] {
        let (w, k) = fixture(state);
        let h = k.get_uhash();
        let mk = || GreenKey::with_types(k.values.clone(), k.types.clone());
        let mkd: &dyn Fn() -> GreenKey = &mk;
        // The fixture's own witness. Inferring "the chained branch was taken"
        // from the magnitude of the result would be reading the answer off the
        // measurement, so the shape is asserted before anything is timed.
        assert_eq!(
            w.bucket_is_chained(h),
            state == Bucket::Chained,
            "the fixture did not build the bucket shape it claims"
        );
        let bare = probe_bare(&w, h);
        let resolved = probe_resolved(&w, h, mkd);
        let run = {
            let mut a = Arm::new(NAME_A, || {
                probe_bare(&w, h);
            });
            let mut b = Arm::new(NAME_B, || {
                probe_resolved(&w, h, mkd);
            });
            run_pair(&mut a, &mut b, &cfg)
        };
        report(
            &format!("MEASUREMENT [{}]", state.label()),
            &run,
            &cfg,
            Some(floors),
        );
        // Reported, not asserted, and deliberately weakly: no cell in these
        // fixtures carries a procedure token, so BOTH arms read `None` and
        // their agreement is vacuous. It says the arms ran, not that they
        // decide alike — the behavioural claim belongs to the door's own tests,
        // and the timing above does not rest on it.
        println!(
            "  bucket chained: {}   probe answers: bare = {bare}, resolved = {resolved} \
             (both None-valued here: an agreement witness, not a behaviour one)",
            w.bucket_is_chained(h),
        );
    }

    // The headline is the ordinary warm shape: a bucket holding one cell.
    let real_run = {
        let mut a = Arm::new(NAME_A, || {
            probe_bare(&warm_s, hash_s);
        });
        let mut b = Arm::new(NAME_B, || {
            probe_resolved(&warm_s, hash_s, make_sd);
        });
        run_pair(&mut a, &mut b, &cfg)
    };
    let (real_cpu, real_wall) = report(
        &format!("MEASUREMENT [{}]", Bucket::Single.label()),
        &real_run,
        &cfg,
        Some(floors),
    );

    #[cfg(feature = "loop-key-arm-probe")]
    loop_key_census();

    // -- CLOSING NULL CONTROL -----------------------------------------------
    // The opening null measured the box at the start. Everything graded
    // against it assumed the box stayed that way, and on a shared machine that
    // is an assumption, not a fact. Repeating the null at the end measures how
    // far it moved: two nulls that disagree bound the drift the comparisons
    // between them silently absorbed. This is the contamination witness that
    // works at this timescale — the kernel's load average is sampled far too
    // coarsely to say anything about a run this short.
    let close_run = {
        let mut a = Arm::new("A (probe the bare hash)", || {
            probe_bare(&warm_s, hash_s);
        });
        let mut b = Arm::new("A' (identical copy)", || {
            probe_bare(&warm_s, hash_s);
        });
        run_pair(&mut a, &mut b, &cfg)
    };
    let (close_cpu, close_wall) = report(
        "CLOSING NULL CONTROL: the same null, re-run at the end",
        &close_run,
        &cfg,
        Some(floors),
    );
    let closing = Floors {
        cpu: floor_from(&close_cpu),
        wall: floor_from(&close_wall),
    };
    // Grade the headline against the worse of the two. A verdict is only as
    // good as the noisiest moment of the window it was taken in.
    let effective = Floors {
        cpu: floors.cpu.max(closing.cpu),
        wall: floors.wall.max(closing.wall),
    };

    // -- SUMMARY ------------------------------------------------------------
    let load_after = loadavg();
    let elapsed = started.elapsed();
    println!();
    println!("== SUMMARY");
    println!("  wall time        {:.1} s", elapsed.as_secs_f64());
    println!("  load before      {}", render_load(load_before));
    println!("  load after       {}", render_load(load_after));
    // How much of this thread's elapsed time it did not spend running. The
    // load average describes the machine; this describes the measurement. A
    // ratio at 1.00 means the box's other work never took the CPU away from
    // these batches, however high the load average was, and the two clocks
    // therefore agree.
    println!(
        "  wall/CPU         {:.4} on the null control (1.0000 = this thread was never preempted)",
        null_wall.a_median / null_cpu.a_median
    );
    match (load_before, load_after) {
        (Some(before), Some(after)) if after[0] > before[0] * 1.5 => {
            println!();
            println!("  ############################################################");
            println!(
                "  # CONTAMINATED: 1-minute load rose {:.2} -> {:.2} across the run",
                before[0], after[0]
            );
            println!("  # (more than +50%). The numbers above stay printed and are");
            println!("  # not discarded, but the machine was not the same machine at");
            println!("  # the start and the end, so the floor may understate the");
            println!("  # noise the later comparisons actually ran in.");
            println!("  ############################################################");
        }
        (Some(_), Some(_)) if elapsed < Duration::from_secs(5) => {
            // The kernel's load average is a decimated 5-second sampler. A run
            // shorter than one of its periods can read the same three numbers
            // at both ends without that meaning anything, so this says
            // "uninformative" rather than "clean".
            println!(
                "  contamination    NOT TESTED: the run ({:.1} s) was shorter than the load \
                 average's own update period, so before and after can be the same sample. \
                 Use the wall/CPU ratio above, or raise PAIRED_AB_ROUNDS / \
                 PAIRED_AB_BATCH_US until the run exceeds 5 s.",
                elapsed.as_secs_f64()
            );
        }
        (Some(_), Some(_)) => {
            println!("  contamination    none detected (1-minute load did not rise by >50%)");
        }
        _ => {
            println!("  contamination    UNKNOWN (load average unreadable)");
        }
    }
    println!(
        "  floor opening    ±{:.3} ns/iter     floor closing ±{:.3} ns/iter",
        floors.cpu, closing.cpu
    );
    println!(
        "  RESOLUTION FLOOR ±{:.3} ns/iter (thread CPU, the worse of the two nulls)",
        effective.cpu
    );
    if closing.cpu > floors.cpu * 2.0 {
        println!();
        println!("  ############################################################");
        println!(
            "  # DRIFT: the closing null is {:.1}x noisier than the opening one",
            closing.cpu / floors.cpu
        );
        println!(
            "  # (±{:.3} -> ±{:.3} ns/iter). The box did not hold still for the",
            floors.cpu, closing.cpu
        );
        println!("  # length of this run, so comparisons taken late in it were graded");
        println!("  # against a floor that had already stopped applying. Re-read every");
        println!("  # verdict above against the larger figure.");
        println!("  ############################################################");
    }
    println!(
        "  positive control {}",
        if positive_ok { "OK" } else { "FAILED" }
    );
    match smallest_resolved {
        Some((units, ns)) => {
            println!("  smallest resolved injected delta: spin({units}) at {ns:+.3} ns/iter")
        }
        None if cfg.ladder => {
            println!("  smallest resolved injected delta: NONE — no ladder rung up to 32 steps")
        }
        None => println!("  sensitivity ladder skipped (PAIRED_AB_LADDER=0)"),
    }
    println!(
        "  headline         {}",
        verdict(&real_cpu, effective.cpu).render(
            &real_cpu,
            effective.cpu,
            &real_run.name_a,
            &real_run.name_b
        )
    );
    // Quoted beside the nanoseconds because it is the more portable of the
    // two. The absolute figure moves with whatever clock frequency the box
    // happened to be running at; the ratio between two arms measured in the
    // same batches does not.
    println!(
        "  headline (ratio) {:+.2}% of `{}`'s own median cost",
        100.0 * real_cpu.median_diff / real_cpu.a_median,
        real_run.name_a
    );
    println!(
        "  wall corroborate {}",
        verdict(&real_wall, effective.wall).render(
            &real_wall,
            effective.wall,
            &real_run.name_a,
            &real_run.name_b
        )
    );
}
