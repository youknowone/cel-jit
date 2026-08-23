//! A self-calibrating, interleaved paired A/B timing harness that states its
//! own resolution and refuses claims below it.
//!
//! # What problem this solves
//!
//! The deltas worth measuring here are single-digit nanoseconds per
//! evaluation, and the box they have to be measured on carries other people's
//! work: load average in the tens, several compilers at full tilt. Waiting for
//! the box to go quiet is not a method — it is a wish, and it has already cost
//! whole measurement windows that produced nothing.
//!
//! So this harness does not try to be accurate. It tries to be *honest*: it
//! measures under whatever load exists, it measures its own noise floor in the
//! same conditions, and it reports any delta smaller than that floor as
//! `UNRESOLVED` rather than as a number with a sign. A constant that fails
//! loudly costs one probe. A constant that fails quietly costs the window.
//!
//! # How it is built
//!
//! * **One process, both arms compiled in.** The arms are boxed closures
//!   selected at run time, never two `cargo build` invocations. Separate builds
//!   admit compile drift and stale binaries, neither of which is visible in the
//!   numbers they produce.
//!
//! * **Interleaved paired sampling with order alternation.** Each round runs
//!   both arms back to back for the same inner iteration count and records the
//!   pair `(a, b)`. Even rounds run A first, odd rounds run B first, so
//!   first-mover and cache-warming effects cancel across rounds instead of
//!   accumulating into the sign of the delta. Because both halves of a pair see
//!   the same few milliseconds of machine weather, the per-round difference
//!   `b - a` is far quieter than either arm alone.
//!
//! * **Thread CPU time is the primary clock.** A wall clock on a loaded box
//!   measures this program plus everyone else's: a batch descheduled for 200 ms
//!   reports 200 ms it never spent, and nothing in the figure distinguishes
//!   that from real slowdown. `CLOCK_THREAD_CPUTIME_ID` charges only the cycles
//!   this thread actually ran, so a co-tenant's preemption costs the
//!   measurement nothing and only its cache and TLB damage survives — which is
//!   what the per-round minimum and the paired median are there to shed. Wall
//!   clock is still collected and printed beside it, because a large divergence
//!   between the two is itself the load's signature.
//!
//!   This clock costs no new dependency: `libc` is already a dev-dependency of
//!   this crate, so `clock_gettime` is available to an example as-is. (Had it
//!   needed a new crate, the rule was to fall back to `Instant` plus the
//!   min-of-K estimator and say so here. That fallback was not needed.)
//!
//! * **Robust estimators, printed side by side.** Per arm: minimum and median
//!   over the rounds. Paired: the median of the per-round differences, with a
//!   distribution-free 95% interval from a bootstrap over those differences
//!   (resampling written by hand, fixed seed, so a rerun of the same binary
//!   reproduces the interval exactly), plus a sign test. The paired median is
//!   the headline; the difference of the per-arm minima is an independent
//!   corroborator. When those two disagree in sign the harness says so loudly —
//!   that is a contamination signature, not a result.
//!
//! * **A null control, always.** Arm A against a second, identical copy of arm
//!   A. The true delta there is zero, so whatever magnitude the harness reports
//!   for it is the harness's own resolution at this load. That magnitude is
//!   printed as `RESOLUTION FLOOR`, and every later verdict is graded against
//!   it.
//!
//! * **A positive control, always.** Arm A against arm A plus one heap
//!   allocation and its matching free — real work, in a non-inlined function,
//!   with `black_box` on the value so it cannot be optimized away. It must come
//!   out with the correct sign and above the floor. If it does not, the harness
//!   is not working at this load and says so before any other result.
//!
//! * **A sensitivity ladder.** Beyond the mandatory pair of controls, the same
//!   machinery is run against arm A plus 1, 2, 4, 8, 16 and 32 units of a
//!   dependent integer chain. The smallest rung that resolves is the smallest
//!   delta this box actually resolved today, stated in nanoseconds rather than
//!   asserted.
//!
//! * **Load capture.** The 1/5/15-minute load averages are read before and
//!   after the run. If the 1-minute figure rose by more than half across the
//!   run, a `CONTAMINATED` banner is printed. The numbers stay printed —
//!   discarding them silently would be the same mistake in the other
//!   direction — but they are flagged.
//!
//! # Adding an arm
//!
//! Everything above is machinery. The measurement itself is the block at the
//! bottom of `main` marked `THE ARMS UNDER TEST`: two `Arm::new` calls, each
//! wrapping a closure. A later probe replaces those two closures and changes
//! nothing else.
//!
//! # Running it
//!
//! ```text
//! cargo run -p cel --example paired_ab --release
//! ```
//!
//! Release matters: a debug build measures the debug build. Tunables, all
//! optional:
//!
//! * `PAIRED_AB_ROUNDS` — paired rounds per comparison (default 81).
//! * `PAIRED_AB_BATCH_US` — thread-CPU microseconds per arm per round, which
//!   sets the calibrated inner iteration count (default 1000).
//! * `PAIRED_AB_RESAMPLES` — bootstrap resamples (default 20000).
//! * `PAIRED_AB_SEED` — bootstrap seed (default 0x5EED_1234_ABCD_0001).
//! * `PAIRED_AB_LADDER=0` — skip the sensitivity ladder.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cel::{Context, Program};

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

/// One evaluation through the public door, fenced on both sides.
///
/// `black_box` on the inputs stops the optimizer from folding the evaluation
/// against known-constant operands or hoisting it out of the inner loop, and
/// `black_box` on the result stops it from deleting the work as unused.
fn eval(program: &Program, ctx: &Context) {
    let out = black_box(program)
        .execute(black_box(ctx))
        .expect("the arm evaluates");
    black_box(out);
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
const SRC_A: &str = "size(list) > 0";
const SRC_B: &str = "list.size() > 0";

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

    // The fixtures both real arms evaluate against, built once so that neither
    // arm pays for setup inside the timed region.
    let program_a = Program::compile(SRC_A).expect("arm A source compiles");
    let program_b = Program::compile(SRC_B).expect("arm B source compiles");
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", (1i64..=8).collect::<Vec<i64>>());
    let ctx = ctx;

    // A paired A/B of two arms that compute different things is not a
    // measurement of anything, so this is checked rather than assumed.
    let out_a = program_a.execute(&ctx).expect("arm A evaluates");
    let out_b = program_b.execute(&ctx).expect("arm B evaluates");
    assert_eq!(
        out_a, out_b,
        "the two arms must compute the same answer to be comparable"
    );
    println!("  arms agree on   {out_a:?}");

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
        let mut a = Arm::new("A (size(list) > 0)", || eval(&program_a, &ctx));
        let mut b = Arm::new("A' (identical copy)", || eval(&program_a, &ctx));
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
        let mut a = Arm::new("A", || eval(&program_a, &ctx));
        let mut b = Arm::new("A + one Box::new/drop", || {
            eval(&program_a, &ctx);
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
                    eval(&program_a, &ctx);
                    black_box(spin(0));
                });
                let mut b = Arm::new(format!("A + spin({units})"), || {
                    eval(&program_a, &ctx);
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

    // -- THE PROBE ----------------------------------------------------------
    // Task #36's drop decomposition and the `IterAt` arm, behind
    // `--features drop-arm-probe`. Placed here so that the closing null
    // control below still brackets them; each section inside sets its own
    // floor, because the one above was measured on a much shorter arm.
    #[cfg(feature = "drop-arm-probe")]
    probe(&cfg);

    // -- THE ELEMENT-ATTRIBUTION PROBE --------------------------------------
    // Task #40: what the bytecode VM's flat per-element excess over the tree
    // walker on `list.map(x, x * 2)` is MADE OF, behind
    // `--features elem-attr-probe`. Same bracket, same rule about floors.
    #[cfg(feature = "elem-attr-probe")]
    element_attribution(&cfg);

    // -- THE LOOP-KEY PROBE -------------------------------------------------
    // Task #31: what the function-entry door's per-call loop-key resolution
    // costs, behind `--features jit-<backend>,loop-key-arm-probe`. Bracketed by
    // the same two null controls, and each section sets its own floor for the
    // same reason the drop probe's do.
    #[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
    loop_key_probe(&cfg);

    // -- THE ENCODE SPLIT ---------------------------------------------------
    // Task #58: what the columnar encoding is MADE OF for an activation with
    // no list and no string, behind `--features jit-<backend>,encode-stage-probe`.
    // Same bracket, and its own floor per size for the same reason the others
    // set theirs.
    #[cfg(all(feature = "jit", feature = "encode-stage-probe"))]
    encode_stage_split(&cfg);

    // -- THE ARMS UNDER TEST ------------------------------------------------
    // Replace these two closures to measure something else. Nothing above needs
    // to change: the controls, the floor and the verdicts are all machinery.
    let real_run = {
        let mut a = Arm::new(SRC_A, || eval(&program_a, &ctx));
        let mut b = Arm::new(SRC_B, || eval(&program_b, &ctx));
        run_pair(&mut a, &mut b, &cfg)
    };
    let (real_cpu, real_wall) = report(
        &format!("MEASUREMENT: `{SRC_A}` vs `{SRC_B}`"),
        &real_run,
        &cfg,
        Some(floors),
    );

    // -- CLOSING NULL CONTROL -----------------------------------------------
    // The opening null measured the box at the start. Everything graded
    // against it assumed the box stayed that way, and on a shared machine that
    // is an assumption, not a fact. Repeating the null at the end measures how
    // far it moved: two nulls that disagree bound the drift the comparisons
    // between them silently absorbed. This is the contamination witness that
    // works at this timescale — the kernel's load average is sampled far too
    // coarsely to say anything about a run this short.
    let close_run = {
        let mut a = Arm::new("A (size(list) > 0)", || eval(&program_a, &ctx));
        let mut b = Arm::new("A' (identical copy)", || eval(&program_a, &ctx));
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

// ---------------------------------------------------------------------------
// The probe (feature `drop-arm-probe`)
// ---------------------------------------------------------------------------
//
// Two questions, both answered as a difference taken inside ONE binary:
//
//   * task #36 — an operand the interpreter throws away goes through the
//     out-of-line drop glue for `Value`. Is the cost the CALL, or the work
//     inside it?
//   * `IterAt` — the arm was rewritten to index a known list instead of
//     routing through `value_index`, and landed explicitly unmeasured.
//
// Both are selected at run time by `cel::vm::ProbePolicy`, so every arm is the
// same compiled `Vm::step` taking a different branch. That branch costs one
// field read and one perfectly-predicted test at each site, present
// identically in every arm, so it cancels out of any difference between two of
// them — and it also means no arm's ABSOLUTE figure is what the shipping
// interpreter costs. Only differences are claims.

/// Elements per comprehension for the drop decomposition.
///
/// Large on purpose. The harness's floor is per ITERATION of an arm, and one
/// iteration is a whole comprehension, so the resolution available per DISCARD
/// is the floor divided by the discards one iteration performs. A thousand
/// elements buys three orders of magnitude of that division; the price is that
/// the figure is an average over a loop, which is what a per-element cost is.
#[cfg(feature = "drop-arm-probe")]
const DROP_PROBE_ELEMS: usize = 1000;

/// A context binding `xs` to `n` integers.
///
/// From ONE, not from zero. A comprehension whose predicate is false for the
/// first element short-circuits, and an arm that stops after one element is
/// still a perfectly well-behaved arm -- it just measures a loop that did not
/// run. The `assert` on each ladder's answer is what turns that from a thing
/// to remember into a thing that fails.
#[cfg(feature = "drop-arm-probe")]
fn int_list_ctx(n: usize) -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("xs", (1..=n as i64).collect::<Vec<i64>>());
    ctx
}

/// A context binding `xs` to `n` strings, each its own allocation.
#[cfg(feature = "drop-arm-probe")]
fn string_list_ctx(n: usize) -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value(
        "xs",
        (0..n)
            .map(|i| format!("element-{i:06}"))
            .collect::<Vec<String>>(),
    );
    ctx
}

/// Compile `src` to a code object the probe can run under an explicit policy.
#[cfg(any(feature = "drop-arm-probe", feature = "elem-attr-probe"))]
fn probe_code(src: &str) -> cel::vm::CelCode {
    let expr = cel::parser::Parser::default()
        .parse(src)
        .expect("the probe source parses");
    cel::vm::compile(&expr).expect("the probe source compiles")
}

/// Print what a resolved comparison says per unit of whatever it loops over.
///
/// The floor is per iteration of an arm and one iteration is a whole
/// comprehension, so BOTH the delta and the floor are divided by the same
/// count. Quoting a per-element delta against a per-iteration floor would
/// claim a resolution three orders of magnitude finer than the one measured.
/// An unresolved comparison prints no number at all: below the floor there is
/// nothing to divide.
// Shared by the probes; the cfg names each one exactly so that enabling any one
// alone leaves no unused item behind.
#[cfg(any(
    feature = "drop-arm-probe",
    feature = "elem-attr-probe",
    all(feature = "jit", feature = "encode-stage-probe"),
    all(feature = "jit", feature = "loop-key-arm-probe")
))]
fn per_unit(label: &str, s: &Stats, floor: f64, units: f64, unit: &str) {
    if verdict(s, floor).resolved() {
        println!(
            "  {label:<24} {:+.4} ns per {unit}   ({:+.2}% of arm A)   \
             floor/{unit} ±{:.4} ns over {units:.0}",
            s.median_diff / units,
            100.0 * s.median_diff / s.a_median,
            floor / units,
        );
    } else {
        println!(
            "  {label:<24} UNRESOLVED per iteration (|{:+.3}| vs floor ±{floor:.3} ns/iter) — \
             nothing may be quoted per {unit}",
            s.median_diff,
        );
    }
}

/// A null control on the arms actually under test, and the floor it sets.
///
/// The harness's own floor was measured on an arm costing tens of nanoseconds.
/// A comprehension over a thousand elements costs tens of microseconds, and
/// noise scales with the length of the batch it lands in, so grading one
/// against the other would understate the noise by the ratio of the two. Every
/// comparison below is graded against a null run on its own arms instead.
// Shared by the probes; the cfg names each one exactly so that enabling any one
// alone leaves no unused item behind.
#[cfg(any(
    feature = "drop-arm-probe",
    feature = "elem-attr-probe",
    all(feature = "jit", feature = "encode-stage-probe"),
    all(feature = "jit", feature = "loop-key-arm-probe")
))]
fn local_floor<F: FnMut()>(title: &str, mut make: impl FnMut() -> F, cfg: &Config) -> Floors {
    let run = {
        let mut a = Arm::new("arm under test", make());
        let mut b = Arm::new("identical copy", make());
        run_pair(&mut a, &mut b, cfg)
    };
    let (cpu, wall) = report(title, &run, cfg, None);
    let floors = Floors {
        cpu: floor_from(&cpu),
        wall: floor_from(&wall),
    };
    println!(
        "  LOCAL RESOLUTION FLOOR ±{:.3} ns/iter (thread CPU) — this section only",
        floors.cpu
    );
    floors
}

/// Task #36: split the cost of an out-of-line `Value` drop into the call and
/// the work.
///
/// Stated before any run:
///
/// * `Baseline − InlineDiscriminant` measures **(A) the call**. Both arms test
///   the discriminant; only Baseline does it behind a `call`/`ret`, with the
///   caller-saved clobber and the alias barrier that come with it.
/// * `InlineDiscriminant − ForgetUnsound` measures **(B) the work**. Neither
///   arm calls out on the trivial path; only Arm I executes the test.
/// * `Baseline − ForgetUnsound` is **A + B**, an additivity check and not a
///   third quantity.
///
/// #36's hypothesis — "the cost is the CALL, not the work inside it" —
/// predicts A > B. B ≥ A with both resolved refutes it.
///
/// What this does NOT cover: `compare_values` and `binary_values` take their
/// operands by value and drop them inside `objects.rs`, which the tree walker
/// shares. Each ladder below therefore measures a MAJORITY of its per-element
/// `Value` drops, not all of them.
#[cfg(feature = "drop-arm-probe")]
fn drop_decomposition(cfg: &Config) {
    use cel::vm::{cel_eval_loop_with_probe, DropArm, IterAtArm, ProbePolicy};
    use cel::Value;

    // Both ladders are integer-only, which is what makes `ForgetUnsound`
    // sound: every operand it forgets is an `Int` or a `Bool`, and forgetting
    // one releases nothing because it owns nothing.
    //
    // The discard counts are read off the lowering in `vm/compile.rs`, per
    // element. `Vm::discard` is reached from four places — the conditional
    // jumps, a short-circuit operator's left half, its merge, and
    // `Vm::store_slot`:
    //
    //   map:  IterBind, which stores the element over the previous one  = 1
    //   all:  that one, plus AndMerge and StoreLocal accu               = 3
    //
    // Both counts fell when the loop scaffolding was fused, because three of
    // the discards were in instructions that no longer exist: `IterGuard`
    // replaced a `JumpIfFalse` that discarded the guard's bool,
    // `AccuLoopCond` replaced another, and `AndLocal` replaced an `And` that
    // discarded the copy of the accumulator it had just been handed. None of
    // the fused forms puts a value on the operand stack, so none of them has
    // one to throw away.
    //
    // What that costs this section is independence: the two ladders used to
    // reach all four site families between them and now reach two, with
    // `all`'s sites a superset of `map`'s. They are still two lengths and two
    // programs, but no longer two different mixes of site.
    let ladders: [(&str, f64); 2] = [("xs.map(x, x * 2)", 1.0), ("xs.all(x, x > 0)", 3.0)];

    for (src, discards_per_elem) in ladders {
        let code = probe_code(src);
        let ctx = int_list_ctx(DROP_PROBE_ELEMS);
        let discards = discards_per_elem * DROP_PROBE_ELEMS as f64;

        // The discard count above is per element and assumes the loop reaches
        // every element. `all` stops at the first false, so this is a
        // precondition of the arithmetic, not a smoke test: an arm that
        // short-circuits after one element still runs, still times, and its
        // per-discard figure is then wrong by three orders of magnitude.
        let answer = cel_eval_loop_with_probe(&code, &ctx, ProbePolicy::default())
            .expect("the ladder evaluates");
        let elements = match &answer {
            Value::List(list) => list.len(),
            Value::Bool(true) => DROP_PROBE_ELEMS,
            other => panic!("`{src}` answered {other:?}, which cannot show a full traversal"),
        };
        assert_eq!(
            elements, DROP_PROBE_ELEMS,
            "`{src}` traversed {elements} of {DROP_PROBE_ELEMS} elements: the discard count \
             below is per element and assumes the whole sequence"
        );

        println!();
        println!("###########################################################");
        println!("# DROP DECOMPOSITION on `{src}` over {DROP_PROBE_ELEMS} integers");
        println!("# {discards:.0} discards inside `Vm::step` per iteration");
        println!("###########################################################");

        // Guard 1, static: nothing the program can put on the stack owns
        // anything.
        assert!(
            code.consts.iter().all(|c| matches!(
                c,
                Value::Int(_) | Value::UInt(_) | Value::Float(_) | Value::Bool(_) | Value::Null
            )),
            "the unsound arm needs a program whose constants own nothing: {:?}",
            code.consts
        );

        let policy = |drop_arm| ProbePolicy {
            drop_arm,
            iter_at: IterAtArm::KnownList,
        };

        // Two of the three arms are semantics-preserving and the third is
        // supposed to be, on this ladder, for the reason the guards above
        // state. All three answering the same thing is what says so.
        for other in [DropArm::InlineDiscriminant, DropArm::ForgetUnsound] {
            let out =
                cel_eval_loop_with_probe(&code, &ctx, policy(other)).expect("the arm evaluates");
            assert_eq!(
                out, answer,
                "`{src}` under {other:?} disagrees with the baseline"
            );
        }
        let arm = |drop_arm| {
            let code = &code;
            let ctx = &ctx;
            let policy = policy(drop_arm);
            move || {
                let out = cel_eval_loop_with_probe(black_box(code), black_box(ctx), policy)
                    .expect("the arm evaluates");
                black_box(out);
            }
        };

        // Warm the thread's buffer pool and the allocator's free lists before
        // counting, so the witness compares two steady-state evaluations
        // rather than one cold one against one warm one.
        for arm_policy in [DropArm::Baseline, DropArm::ForgetUnsound] {
            for _ in 0..4 {
                arm(arm_policy)();
            }
        }

        // Guard 2, dynamic, and the one that decides. If the arm forgot
        // anything the allocator handed out, the counts do not balance.
        // NECESSARY, NOT SUFFICIENT: a forgotten clone of something the
        // CONTEXT owns leaks a reference count and frees no less memory inside
        // the window, which is why guard 1 and the fixed sources above are not
        // redundant with it.
        let (base_allocs, base_frees) = count_allocs(arm(DropArm::Baseline));
        let (leak_allocs, leak_frees) = count_allocs(arm(DropArm::ForgetUnsound));
        assert_eq!(
            leak_allocs as i64 - leak_frees as i64,
            base_allocs as i64 - base_frees as i64,
            "DropArm::ForgetUnsound leaked on `{src}`: {leak_allocs} allocs / {leak_frees} \
             frees against a baseline of {base_allocs}/{base_frees}. Do not time this arm."
        );
        println!(
            "  leak witness     baseline {base_allocs} alloc / {base_frees} free, \
             forget {leak_allocs} alloc / {leak_frees} free — balanced"
        );

        let floors = local_floor(
            &format!("LOCAL NULL CONTROL: `{src}` Baseline vs an identical copy"),
            || arm(DropArm::Baseline),
            cfg,
        );

        let call_run = {
            let mut a = Arm::new("Baseline (out-of-line glue)", arm(DropArm::Baseline));
            let mut b = Arm::new(
                "Arm I (discriminant inline)",
                arm(DropArm::InlineDiscriminant),
            );
            run_pair(&mut a, &mut b, cfg)
        };
        let (call_cpu, _) = report(
            &format!("(A) the out-of-line drop CALL — Baseline vs Arm I, `{src}`"),
            &call_run,
            cfg,
            Some(floors),
        );
        per_unit(
            "(A) call, per discard",
            &call_cpu,
            floors.cpu,
            discards,
            "discard",
        );

        let work_run = {
            let mut a = Arm::new(
                "Arm I (discriminant inline)",
                arm(DropArm::InlineDiscriminant),
            );
            let mut b = Arm::new("Arm N (forget — LEAKS)", arm(DropArm::ForgetUnsound));
            run_pair(&mut a, &mut b, cfg)
        };
        let (work_cpu, _) = report(
            &format!("(B) the discriminant WORK — Arm I vs Arm N, `{src}`"),
            &work_run,
            cfg,
            Some(floors),
        );
        per_unit(
            "(B) work, per discard",
            &work_cpu,
            floors.cpu,
            discards,
            "discard",
        );

        let whole_run = {
            let mut a = Arm::new("Baseline (out-of-line glue)", arm(DropArm::Baseline));
            let mut b = Arm::new("Arm N (forget — LEAKS)", arm(DropArm::ForgetUnsound));
            run_pair(&mut a, &mut b, cfg)
        };
        let (whole_cpu, _) = report(
            &format!("(A+B) Baseline vs Arm N — must equal A + B, `{src}`"),
            &whole_run,
            cfg,
            Some(floors),
        );
        per_unit(
            "(A+B) per discard",
            &whole_cpu,
            floors.cpu,
            discards,
            "discard",
        );

        // Additivity. Two halves measured separately have to add up to the
        // whole measured directly; if they do not, the three arms were not one
        // binary's three branches and both halves are void. Reported against
        // the whole comparison's own interval rather than asserted, because a
        // disagreement is a result about the instrument and deserves to be
        // read, not to abort the run.
        let halves = -call_cpu.median_diff + -work_cpu.median_diff;
        let direct = -whole_cpu.median_diff;
        println!(
            "  additivity        A+B from the halves {halves:+.3} vs measured directly \
             {direct:+.3} ns/iter   (whole's 95% CI [{:+.3}, {:+.3}])",
            -whole_cpu.ci_hi, -whole_cpu.ci_lo
        );
    }
}

/// Price the `IterAt` change that landed in `338e0d6` with its effect
/// explicitly unmeasured.
///
/// The old arm handed both slots to `value_index`, which decides the
/// container's kind, then the key's kind, then bounds-checks, then answers in
/// `ExecutionError` — which `Vm::park` records on `&mut self`. The new arm is
/// two variant tests and one unsigned comparison answering in `CelErr`.
///
/// The read is `Vm::element_at`, which is what `IterAt` is and what `IterBind`
/// does before it stores. A compiled comprehension reaches it through
/// `IterBind`, so this prices the same element read the loop always ran.
///
/// Swept over three lengths so that an O(1) effect — anything paid once per
/// evaluation — separates from the O(N) one the instruction is, and over two
/// element types because the boxing on the way out of a list differs between
/// an unboxed integer column and an interned string.
#[cfg(feature = "drop-arm-probe")]
fn iter_at_sweep(cfg: &Config) {
    use cel::vm::{cel_eval_loop_with_probe, DropArm, IterAtArm, ProbePolicy};

    // The smallest body that still runs the instruction once per element, so
    // the loop's other work is as small a share of the arm as it gets.
    const SRC: &str = "xs.map(x, x)";
    let code = probe_code(SRC);

    for (kind, build) in [
        ("int", int_list_ctx as fn(usize) -> Context<'static>),
        ("string", string_list_ctx as fn(usize) -> Context<'static>),
    ] {
        for n in [10usize, 100, 1000] {
            let ctx = build(n);

            println!();
            println!("###########################################################");
            println!("# IterAt on `{SRC}` over {n} {kind} elements");
            println!("###########################################################");

            let arm = |iter_at| {
                let code = &code;
                let ctx = &ctx;
                let policy = ProbePolicy {
                    drop_arm: DropArm::Baseline,
                    iter_at,
                };
                move || {
                    let out = cel_eval_loop_with_probe(black_box(code), black_box(ctx), policy)
                        .expect("the arm evaluates");
                    black_box(out);
                }
            };

            // Both arms must compute the same answer to be comparable, and
            // here that is a real check rather than a formality: the two
            // lowerings disagree about which error an out-of-range index
            // raises, so agreement on the in-range path is what says the
            // rewrite preserved the value.
            let old = cel_eval_loop_with_probe(
                &code,
                &ctx,
                ProbePolicy {
                    drop_arm: DropArm::Baseline,
                    iter_at: IterAtArm::ViaValueIndex,
                },
            )
            .expect("the old arm evaluates");
            let new = cel_eval_loop_with_probe(&code, &ctx, ProbePolicy::default())
                .expect("the new arm evaluates");
            assert_eq!(old, new, "the two IterAt lowerings must agree");

            let floors = local_floor(
                &format!("LOCAL NULL CONTROL: `{SRC}`, {n} {kind}, new arm vs an identical copy"),
                || arm(IterAtArm::KnownList),
                cfg,
            );

            let run = {
                let mut a = Arm::new("old (via value_index)", arm(IterAtArm::ViaValueIndex));
                let mut b = Arm::new("new (known list)", arm(IterAtArm::KnownList));
                run_pair(&mut a, &mut b, cfg)
            };
            let (cpu, _) = report(
                &format!("IterAt: value_index vs known-list, {n} {kind} elements"),
                &run,
                cfg,
                Some(floors),
            );
            per_unit("IterAt, per element", &cpu, floors.cpu, n as f64, "element");
        }
    }
}

// ---------------------------------------------------------------------------
// The element-attribution probe (feature `elem-attr-probe`)
// ---------------------------------------------------------------------------
//
// Task #40. On the `map_list_scaling` ladder the bytecode VM's excess over the
// tree walker is a FLAT ~58-64 ns per element from n=1 to n=10000 — the ratio
// only climbs because the walker amortises a fixed cost, not because the VM's
// per-element excess grows. Nothing named accounted for it. This section takes
// it apart.
//
// The method is subtractive and stays inside one binary: each arm runs the same
// program through the same dispatch loop with one named GROUP of the
// four-instruction per-element block fused into a single step. An arm removes
// dispatches and operand-stack round trips; it does not remove work the walker
// also does, and `binary_values` — which both evaluators call — is still called
// by every arm with the same operands. The one exception, `compare_values`, is
// isolated by running the guard fusion twice, once with the helper put back.
//
// All four groups became single instructions — `IterGuard`, `IterBind`,
// `MulLocalConstAppend` and `IterAdvance` — so the stock arm already pays no
// operand-stack round trip for any of them, and what each of the four rungs
// prices is one dispatch. The block holds no push and no pop at all: the only
// operand it touches is the builder the append mutates in place, which is put
// on the stack before the loop and taken off after it.
//
// Every rung's answer is asserted equal to the stock arm's before any timing, so
// an arm that removed the wrong thing fails rather than prints a better number.

/// The ladder's source and its input, verbatim from `map_list_scaling`.
#[cfg(feature = "elem-attr-probe")]
const ELEM_SRC: &str = "list.map(x, x * 2)";

/// A context binding `list` to `n` integers, as the ladder binds it.
#[cfg(feature = "elem-attr-probe")]
fn elem_ctx(n: usize) -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", (0..n as i64).collect::<Vec<i64>>());
    ctx
}

/// The instructions the per-element block is made of, in order.
///
/// Read off `vm/compile.rs`'s appending-comprehension lowering and pinned by
/// `assert_element_block` below, which reads the actual instruction stream.
/// Every count this section reports about what an arm removed is derived from
/// this array rather than restated beside the arm: a restated count is one a
/// lowering change can leave behind, and one already had been -- the body
/// group was still labelled with the four instructions and six stack
/// operations it had before its operator absorbed the constant load.
#[cfg(feature = "elem-attr-probe")]
const ELEM_OPS: [cel::vm::OpCode; 4] = {
    use cel::vm::OpCode;
    [
        OpCode::IterGuard,
        OpCode::IterBind,
        OpCode::MulLocalConstAppend,
        OpCode::IterAdvance,
    ]
};

#[cfg(feature = "elem-attr-probe")]
const ELEM_BLOCK: usize = ELEM_OPS.len();

/// What fusing `ELEM_OPS[range]` removes, spelled for the report.
///
/// The dispatch count is the number of instructions; the stack counts are the
/// declared effect of each, which is the same table `Compiler::emit` sizes the
/// operand stack from. Nothing here is a second opinion about the block.
#[cfg(feature = "elem-attr-probe")]
fn removed_by(range: std::ops::Range<usize>) -> String {
    let (mut pushes, mut pops) = (0u32, 0u32);
    for op in &ELEM_OPS[range.clone()] {
        // No opcode in this block takes an arity operand, so the effect does
        // not depend on the operand words.
        let (popped, pushed) = op.stack_effect(&[]);
        pops += popped;
        pushes += pushed;
    }
    let counted = |n: u32, one: &str, many: &str| {
        if n == 1 {
            format!("1 {one}")
        } else {
            format!("{n} {many}")
        }
    };
    let mut parts = vec![counted(range.len() as u32, "dispatch", "dispatches")];
    if pushes > 0 {
        parts.push(counted(pushes, "push", "pushes"));
    }
    if pops > 0 {
        parts.push(counted(pops, "pop", "pops"));
    }
    parts.join(", ")
}

/// Refuse to measure a program that is not the block this section is about.
///
/// Two questions, and the second is the load-bearing one. The window below
/// says what the block IS, in a form a reader can check against a disassembly.
/// `map_loop_is_fusable` asks the recogniser itself whether it can fuse this
/// program — which is the only thing that decides whether any arm below fuses
/// anything. A window here that had drifted from the recogniser's would let
/// every arm quietly run the stock loop and still agree on the answer, so the
/// recogniser is the authority and this window is the description of it.
#[cfg(feature = "elem-attr-probe")]
fn assert_element_block(code: &cel::vm::CelCode) {
    use cel::vm::OpCode;
    let want = ELEM_OPS;
    let ops: Vec<OpCode> = code.instructions().map(|(_, op, _)| op).collect();
    assert!(
        ops.windows(ELEM_BLOCK).any(|w| w == want),
        "`{ELEM_SRC}` no longer lowers to the {ELEM_BLOCK}-instruction \
         per-element block this section attributes. Disassembly:\n{}",
        code.disassemble()
    );
    assert!(
        cel::vm::map_loop_is_fusable(code),
        "the window above matches `{ELEM_SRC}` but the probe's recogniser does \
         not, so every arm would run the stock dispatch loop and report \
         agreement. Disassembly:\n{}",
        code.disassemble()
    );
}

/// Walk the whole element ladder on both evaluators and print the excess per
/// element at each rung.
///
/// This is the measurement being attributed, re-taken here rather than quoted,
/// because everything below is a decomposition OF it and a decomposition of a
/// number this binary cannot reproduce is a decomposition of nothing.
#[cfg(feature = "elem-attr-probe")]
fn elem_ladder(cfg: &Config) {
    use cel::vm::{cel_eval_loop_with_fuse, FuseArm};

    let expr = cel::parser::Parser::default()
        .parse(ELEM_SRC)
        .expect("the ladder source parses");
    let code = probe_code(ELEM_SRC);
    assert_element_block(&code);

    println!();
    println!("###########################################################");
    println!("# THE LADDER, re-taken: `{ELEM_SRC}`, VM vs tree walker");
    println!("###########################################################");

    for n in [1usize, 10, 100, 1000, 10_000] {
        let ctx = elem_ctx(n);
        let vm = || {
            let code = &code;
            let ctx = &ctx;
            move || {
                let out = cel_eval_loop_with_fuse(black_box(code), black_box(ctx), FuseArm::None)
                    .expect("the VM arm evaluates");
                black_box(out);
            }
        };
        let walker = || {
            let out = cel::Value::resolve_value(black_box(&expr), black_box(&ctx))
                .expect("the walker arm evaluates");
            black_box(out);
        };

        assert_eq!(
            cel_eval_loop_with_fuse(&code, &ctx, FuseArm::None).expect("VM"),
            cel::Value::resolve_value(&expr, &ctx).expect("walker"),
            "the two evaluators must agree at n={n}"
        );

        let floors = local_floor(
            &format!("LOCAL NULL CONTROL: walker vs an identical copy, n={n}"),
            || walker,
            cfg,
        );
        let run = {
            let mut a = Arm::new("walker (resolve_value)", walker);
            let mut b = Arm::new("VM (cel_eval_loop)", vm());
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(&format!("VM minus walker, n={n}"), &run, cfg, Some(floors));
        println!(
            "  ABSOLUTE  walker {:9.1} ns/eval = {:7.3} ns/elem   \
             VM {:9.1} ns/eval = {:7.3} ns/elem",
            cpu.a_median,
            cpu.a_median / n as f64,
            cpu.b_median,
            cpu.b_median / n as f64,
        );
        per_unit("VM excess", &cpu, floors.cpu, n as f64, "element");
    }
}

/// The fusion ladder: what the per-element excess is made of.
#[cfg(feature = "elem-attr-probe")]
fn elem_fusion(cfg: &Config, n: usize) {
    use cel::vm::{cel_eval_loop_with_fuse, FuseArm};

    let expr = cel::parser::Parser::default()
        .parse(ELEM_SRC)
        .expect("the ladder source parses");
    let code = probe_code(ELEM_SRC);
    assert_element_block(&code);
    let ctx = elem_ctx(n);

    println!();
    println!("###########################################################");
    println!("# FUSION LADDER on `{ELEM_SRC}`, {n} elements");
    println!("###########################################################");

    // Every arm must answer what the stock arm answers. This is the check that
    // separates "removed a dispatch" from "removed the work".
    let stock = cel_eval_loop_with_fuse(&code, &ctx, FuseArm::None).expect("stock evaluates");
    for arm in [
        FuseArm::GuardKeepingCompare,
        FuseArm::Guard,
        FuseArm::Bind,
        FuseArm::Body,
        FuseArm::Advance,
        FuseArm::AdvanceOnly,
        FuseArm::AllButBody,
        FuseArm::AdvancePlusArcRoundTrip,
    ] {
        let got = cel_eval_loop_with_fuse(&code, &ctx, arm).expect("the arm evaluates");
        assert_eq!(
            got, stock,
            "{arm:?} answered differently from the stock arm"
        );
    }
    assert_eq!(
        cel::Value::resolve_value(&expr, &ctx).expect("walker"),
        stock,
        "the walker must answer what the VM answers"
    );
    println!("  ARMS AGREE: every fusion arm and the walker answer the stock arm's value");

    let arm = |fuse: FuseArm| {
        let code = &code;
        let ctx = &ctx;
        move || {
            let out = cel_eval_loop_with_fuse(black_box(code), black_box(ctx), fuse)
                .expect("the arm evaluates");
            black_box(out);
        }
    };
    let walker = || {
        let out = cel::Value::resolve_value(black_box(&expr), black_box(&ctx))
            .expect("the walker arm evaluates");
        black_box(out);
    };

    let floors = local_floor(
        &format!("LOCAL NULL CONTROL: stock VM vs an identical copy, n={n}"),
        || arm(FuseArm::None),
        cfg,
    );

    // (label, arm A, arm B, what B removes relative to A)
    //
    // A chain from the stock arm to the fully fused one, which is what makes
    // the SUM CHECK below an identity: the marginals telescope.
    // `GuardKeepingCompare` is not on the chain. It stopped being a rung when
    // the guard became one instruction deciding on two `i64`s, because an arm
    // that puts `compare_values` back now ADDS work to the stock arm rather
    // than keeping work the stock arm does; it is taken as a control below.
    // Each rung names the slice of `ELEM_OPS` it fuses, and what it removed is
    // derived from that slice rather than written out beside it.
    let steps: [(&str, FuseArm, FuseArm, std::ops::Range<usize>); 4] = [
        ("guard dispatch", FuseArm::None, FuseArm::Guard, 0..1),
        ("bind dispatch", FuseArm::Guard, FuseArm::Bind, 1..2),
        ("body+append dispatch", FuseArm::Bind, FuseArm::Body, 2..3),
        ("advance dispatch", FuseArm::Body, FuseArm::Advance, 3..4),
    ];

    for (label, a_arm, b_arm, fused) in steps {
        let removed = removed_by(fused);
        let removed = removed.as_str();
        let run = {
            let mut a = Arm::new(format!("{a_arm:?}"), arm(a_arm));
            let mut b = Arm::new(format!("{b_arm:?}"), arm(b_arm));
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(
            &format!("{label}: {a_arm:?} minus {b_arm:?} — removes {removed}"),
            &run,
            cfg,
            Some(floors),
        );
        per_unit(label, &cpu, floors.cpu, n as f64, "element");
    }

    // The order control. The ladder above is cumulative, so every marginal but
    // the first is taken against an already-fused arm. This takes the LAST
    // group's marginal from the stock end instead; agreement with the ladder's
    // own figure for it is what says the split does not depend on the order the
    // groups were removed in.
    {
        let run = {
            let mut a = Arm::new("None", arm(FuseArm::None));
            let mut b = Arm::new("AdvanceOnly", arm(FuseArm::AdvanceOnly));
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(
            "ORDER CONTROL: the advance group removed from the STOCK arm — compare with `advance dispatch` above",
            &run,
            cfg,
            Some(floors),
        );
        per_unit("advance, from stock", &cpu, floors.cpu, n as f64, "element");
    }

    // The price of the guard's `i64` comparison, taken from the other side.
    // `IterGuard` decides on two integers it reads out of their slots; this arm
    // puts `compare_values`, `as_bool` and the discard of their answer back, so
    // B is the slower one and the figure is what the four-instruction guard
    // paid to decide the same thing.
    {
        let run = {
            let mut a = Arm::new("Guard", arm(FuseArm::Guard));
            let mut b = Arm::new("Guard + compare_values", arm(FuseArm::GuardKeepingCompare));
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(
            "CONTROL: compare_values + as_bool + discard put back into the guard (sign must be NEGATIVE: B is slower)",
            &run,
            cfg,
            Some(floors),
        );
        per_unit(
            "compare_values+as_bool",
            &cpu,
            floors.cpu,
            n as f64,
            "element",
        );
    }

    // The positive control, and the calibration for hypothesis 1. This arm ADDS
    // one atomic increment and one atomic decrement per element to the fully
    // fused arm; the disassembly check in the report says whether it really
    // did. A resolved, correctly-signed answer here is what says the section's
    // instrument can see a per-element `Arc` clone at all.
    {
        let run = {
            let mut a = Arm::new("Advance", arm(FuseArm::Advance));
            let mut b = Arm::new(
                "Advance + Arc round trip",
                arm(FuseArm::AdvancePlusArcRoundTrip),
            );
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(
            "POSITIVE CONTROL: one Arc clone-and-release per element (sign must be NEGATIVE: B is slower)",
            &run,
            cfg,
            Some(floors),
        );
        per_unit("one Arc round trip", &cpu, floors.cpu, n as f64, "element");
    }

    // The design candidate, measured as ONE paired difference.
    //
    // ⭐ PRE-REGISTERED, written before the number existed. The ladder's own
    // marginals put this at 7.27..7.71 ns/element, by four spellings that
    // disagree:
    //
    //     SUM CHECK      - body                          7.53
    //     sum(4)         - body                          7.71
    //     guard + bind + advance (ladder)                 7.69
    //     guard + bind + advance (ORDER CONTROL's)        7.27
    //
    // The bracket is wide because the marginals are NOT independent: the order
    // control disagrees with the ladder's advance figure in 3 of 3 runs at
    // 3.1x-11.6x the floor, and the four marginals sum to MORE than the
    // measured end-to-end difference by up to 0.66. This arm is the fix for
    // both, because it is measured directly and recombines nothing.
    //
    // ⛔ REFUTES: a figure outside 7.0..8.0, or one that does not resolve.
    // Below the section floor is UNRESOLVED and is reported as such, never as a
    // signed number.
    // ⛔ SUSPICIOUS: anything at or past the SUM CHECK's own magnitude, which
    // would mean this arm removed the body's dispatch as well -- the one thing
    // it must not do. The `ARMS AGREE` assertion above cannot catch that, since
    // an arm that wrongly fused the body would still compute the right answer.
    {
        let run = {
            let mut a = Arm::new("None", arm(FuseArm::None));
            let mut b = Arm::new("AllButBody", arm(FuseArm::AllButBody));
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(
            "DESIGN CANDIDATE: guard+bind+advance fused, BODY still dispatched — \
             one paired difference, pre-registered at 7.27..7.71 ns/element",
            &run,
            cfg,
            Some(floors),
        );
        per_unit("all but body", &cpu, floors.cpu, n as f64, "element");
    }

    // The additivity check. Not a new term: the end-to-end difference has to
    // equal the four marginals summed, and where it does not, the marginals
    // are not measuring what their labels say.
    {
        let run = {
            let mut a = Arm::new("None", arm(FuseArm::None));
            let mut b = Arm::new("Advance", arm(FuseArm::Advance));
            run_pair(&mut a, &mut b, cfg)
        };
        let (cpu, _) = report(
            "SUM CHECK: stock minus fully fused — must equal the four marginals summed",
            &run,
            cfg,
            Some(floors),
        );
        per_unit("SUM CHECK", &cpu, floors.cpu, n as f64, "element");
    }

    // What is left when every dispatch and every operand-stack round trip is
    // gone: the fully fused VM against the walker. This is the UNATTRIBUTED
    // remainder of the excess, measured rather than inferred by subtraction.
    let run = {
        let mut a = Arm::new("walker (resolve_value)", walker);
        let mut b = Arm::new("VM, whole element fused", arm(FuseArm::Advance));
        run_pair(&mut a, &mut b, cfg)
    };
    let (cpu, _) = report(
        &format!("RESIDUAL: fully fused VM minus walker, n={n}"),
        &run,
        cfg,
        Some(floors),
    );
    println!(
        "  ABSOLUTE  walker {:9.1} ns/eval = {:7.3} ns/elem   \
         fused VM {:9.1} ns/eval = {:7.3} ns/elem",
        cpu.a_median,
        cpu.a_median / n as f64,
        cpu.b_median,
        cpu.b_median / n as f64,
    );
    per_unit(
        "RESIDUAL (unattributed)",
        &cpu,
        floors.cpu,
        n as f64,
        "element",
    );
}

/// Everything behind `elem-attr-probe`.
#[cfg(feature = "elem-attr-probe")]
fn element_attribution(cfg: &Config) {
    println!();
    println!("===========================================================");
    println!("=  ELEMENT ATTRIBUTION (feature `elem-attr-probe`)");
    println!("=  Each section sets its OWN floor. Every arm is the same");
    println!("=  dispatch loop taking a different branch, and every arm's");
    println!("=  answer is asserted equal to the stock arm's.");
    println!("===========================================================");
    elem_ladder(cfg);
    for n in [1000usize, 10_000] {
        elem_fusion(cfg, n);
    }
}

/// Everything behind `drop-arm-probe`, run inside the bracket the opening and
/// closing null controls form.
#[cfg(feature = "drop-arm-probe")]
fn probe(cfg: &Config) {
    println!();
    println!("===========================================================");
    println!("=  PROBE (feature `drop-arm-probe`)");
    println!("=  Every section below sets its OWN floor from a null");
    println!("=  control on its own arms. The floor printed above was");
    println!("=  measured on an arm costing tens of nanoseconds and does");
    println!("=  not apply to a comprehension costing tens of thousands.");
    println!("===========================================================");
    drop_decomposition(cfg);
    iter_at_sweep(cfg);
}

// ---------------------------------------------------------------------------
// The loop-key probe (feature `loop-key-arm-probe`, with a JIT backend)
// ---------------------------------------------------------------------------
//
// Task #31. `dc9146c` made the function-entry door's yield probe ask
// `has_runnable_compiled_loop` on a RESOLVED cell key rather than on the bare
// bucket hash, which buys one bucket walk per loop key on a path that runs on
// every call at the JIT tier. This prices that walk.
//
// The two arms are one compiled `try_function_entry_jit_f` taking a different
// branch off `float_bank::LoopKeyArm`, so nothing here compares two builds.
// Both arms DECIDE the same thing — `resolve` answers a bucket holding one
// cell or none from the walk alone and returns the raw hash — so the
// difference is the walk and nothing else.

/// The threshold the tier compiles at, matching `majit_ab`'s board.
#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
const LOOP_KEY_JIT_ON: u32 = 8;

/// Rows per call.
///
/// ONE, which is the shape the entry door exists for: the row loop is
/// bottom-tested, so a one-row call takes no back edge and the loop's own door
/// never counts and never opens. It is also the state in which every loop key
/// answers NO, so `any` walks the whole list instead of short-circuiting on
/// its first key — which is what makes the per-key divisor below the number of
/// walks actually performed.
#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
const LOOP_KEY_ROWS: usize = 1;

/// The columns the lowered program reads.
///
/// Owned by the caller for as long as the program runs: the words carry their
/// ADDRESSES as seeded registers, so dropping these while a program naming
/// them is still callable would leave the run reading freed memory.
#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
struct LoopKeyColumns {
    balance: Vec<i64>,
    amount: Vec<i64>,
    frozen: Vec<i64>,
}

#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
fn loop_key_columns(n: usize) -> LoopKeyColumns {
    LoopKeyColumns {
        balance: (0..n as i64).map(|i| 100 + i).collect(),
        amount: (0..n as i64).map(|i| 50 + i).collect(),
        frozen: vec![0; n],
    }
}

/// `balance >= amount && !frozen` lowered to a batch program over `cols`, with
/// `extra` further back edges appended after its `RETURN`.
///
/// The appended words are never executed — the program has already returned —
/// but [`float_bank::loop_key_count`] sees them, because the door's scan for
/// loop headers is word-wise rather than a decode and is documented to count
/// positions that are not instructions. That is exactly what makes them usable
/// here: they cost the door precisely what a real loop header costs it, one
/// key each, without changing a single instruction the call runs. The ladder
/// they build is what turns a per-call figure into a per-key one.
///
/// Each block is `[JUMP_IF_ABOVE, 0, 0, target]` with a target that is
/// backward, non-zero and distinct, since a target of `ENTRY_PC` is excluded
/// by the scan and equal targets collapse to one key. The count is asserted
/// rather than assumed, because a divisor derived from intent instead of from
/// the door's own answer is how a per-unit figure goes wrong by a factor.
/// Task #58: what the columnar ENCODE is made of, split by amplification.
///
/// `bind` is a resolution (one map lookup per declared path) followed by this
/// encoding, and the encoding is 62-97% of it -- 94-97% on the CHEAPEST
/// expressions, which carry no list and no string and so should have the least
/// to encode. This section asks what those rows are paying for.
///
/// ⭐ NO RECOMBINATION IN THE WHOLE FIGURE. `BatchProgram` exposes both halves
/// of `bind` as public entry points -- `resolve` and `bind_per_row_resolved` --
/// so encode is timed as its OWN arm rather than as `bind - resolve`. The
/// halves' doc comment says they exist for exactly this.
///
/// Each named stage IS a two-term subtraction (its own delta less the
/// barrier's), which is the amplification discipline
/// `float_bank::EntryStageRepeats` established and not the recombination that
/// ran +19.5% high on the fusion ladder: both terms are measured in THIS
/// section, against THIS section's floor, on the same two arms.
///
/// Both arms call `set_encode_stage_repeats` inside the timed closure, so that
/// cost is common to them and cancels out of every difference.
///
/// ⭐ PRE-REGISTERED, written before the number existed. Encode at n=1 is
/// expected to be mostly FIXED:
///
///     encode(n) ~ a + b*n   with a >= 30 ns, and a > b*n at n=1, 10 and 100
///
/// ⛔ REFUTES: a fixed term under 15 ns, or a proportional term that already
/// dominates at n=10. Either means encode SCALES, and then no amount of
/// fixed-cost work lets a tier-aware `execute` win at small n -- which bounds
/// the design to a threshold form and closes #58 as a lever rather than
/// qualifying it.
/// ⛔ SUSPICIOUS: a fixed term close to the whole of `bind`, which would mean
/// this timed a door that skips the work rather than encode itself.
///
/// ⛔⛔ EVERY STAGE FIGURE BELOW IS AN UPPER BOUND ON WHAT REMOVING THAT STAGE
/// WOULD BUY, and the gap is a factor rather than a rounding error. Amplifying
/// an operation k times back-to-back measures its ISOLATED, SERIALIZED cost;
/// deleting it from a real path measures its MARGINAL cost, with whatever the
/// surrounding code overlaps with it already discounted. Measured on this box:
/// one isolated 8-byte alloc+free is 9.16 ns, while removing one from the JIT
/// entry bought 2.62 ns -- both correct, 3.5x apart.
///
/// Two smaller over-counts stack on top, in the same direction: an allocating
/// stage prices the alloc AND the free together, and `black_box` forces a spill
/// and reload per pass that production never pays.
///
/// ⇒ Read a stage as "no more than this much is here", never as a budget for a
/// fix. The worked case is `StrDict::build` on a string-free batch: this probe
/// prices it at 12.47 ns and a standalone of its exact body costs 5.36 ns.
///
/// ⭐ SECOND PRE-REGISTRATION, about the SPLIT rather than the fit, because the
/// split is what decides whether #58 has a target at all:
///
///     the six named stages account for UNDER HALF of encode,
///     so the residual is > 50%
///
/// Registered pessimistically on the E3 precedent, where the residual dominated
/// at every level until it was split (entry->E 81.2%, E->E3 83.7%). A residual
/// that then comes back at 70% is a CONFIRMED PREDICTION and the follow-on
/// ("split the residual") is already justified by it -- not a disappointing run.
///
/// The residual is not opaque — `prepare_batch_reduce` continues past the six
/// named stages, and everything below is in it by construction. Read this as
/// the candidate list a large residual selects from, not as a measurement:
///
///     per bind, unconditional   `scalars` Vec, `regs_list`, `shape.code`
///                               clone, the `batch_shape` OnceLock probe, and
///                               a full re-collect of `str_ids` that runs even
///                               when there are no predicate tables to chain
///     per bind, PerRow + list   one `list_out` buffer per output field, sized
///                               by summing the source's `size(..)` column
///     per bind, string result   `dict.sorted()`, then ONE OWNED `String` PER
///                               DISTINCT STRING in the batch
///     per bind, projection      a copy or widening pass over all n rows
///
/// ⚠ NAMED IN ADVANCE so it does not read as inconclusive: both
/// pre-registrations can pass while #58 still fails as a lever. Encode can be
/// mostly FIXED (the fit holds) AND that fixed cost can sit entirely in the
/// RESIDUAL (no named stage is the target). On the E3 precedent that is the
/// most likely single outcome.
///
/// ⛔ THE RESIDUAL IS A REPORTED ROW, NOT A LEFTOVER. At every level of the JIT
/// entry's itemisation the residual dominated until it was split -- 81.2% of
/// the entry, then 83.7% of that. Assume the same here until shown otherwise,
/// and read a large residual as "not yet split", never as "the named stages are
/// the answer".
#[cfg(all(feature = "jit", feature = "encode-stage-probe"))]
fn encode_stage_split(cfg: &Config) {
    use cel::majit::batch::{Batch, BatchProgram, ColumnRef};
    use cel::majit::bytecode::{
        encode_stage_out_passes, reset_encode_stage_passes, set_encode_stage_repeats,
        EncodeStageRepeats,
    };
    use cel::majit::lower::{Schema, ValType};

    /// A scalar shape: no list, no string, nothing for the encoding to build.
    /// The question is what it pays anyway.
    const SRC: &str = "x * 2 + 1";
    /// Extra passes per amplified stage. Large enough that one stage's total
    /// clears the section floor, small enough that the loop stays in cache.
    const K: u32 = 64;
    const SIZES: &[usize] = &[1, 10, 100];

    println!();
    println!("###########################################################");
    println!("# ENCODE SPLIT: `{SRC}`, {K} extra passes per stage");
    println!("# pre-registered: mostly FIXED, a >= 30 ns");
    println!("###########################################################");

    let program = Program::compile(SRC).expect("compiles");
    let schema: Schema = [("x".to_string(), ValType::Int)].into_iter().collect();
    let lowered = BatchProgram::from_program(&program, &schema).expect("lowers");

    for &rows in SIZES {
        let vals: Vec<i64> = (0..rows as i64).collect();
        let batch = Batch::new(rows).column("x".to_string(), ColumnRef::Int(&vals));
        let resolved = lowered.resolve(&batch).expect("resolves");

        let arm = |r: EncodeStageRepeats| {
            let lowered = &lowered;
            let resolved = &resolved;
            move || {
                set_encode_stage_repeats(r);
                black_box(
                    lowered
                        .bind_per_row_resolved(resolved)
                        .expect("encodes")
                        .body_words(),
                );
            }
        };
        let zero = EncodeStageRepeats::default();

        println!();
        println!("== rows={rows}");
        let floors = local_floor(
            &format!("LOCAL NULL CONTROL: encode rows={rows} vs an identical copy"),
            || arm(zero),
            cfg,
        );

        // The barrier first: every stage below is reported net of it, so a
        // section that could not resolve the barrier cannot report a stage
        // either.
        let barrier_ns = {
            let run = {
                let mut a = Arm::new("repeats all zero", arm(zero));
                let mut b = Arm::new(
                    "barrier only",
                    arm(EncodeStageRepeats { barrier: K, ..zero }),
                );
                run_pair(&mut a, &mut b, cfg)
            };
            let (cpu, _) = report(
                &format!("BARRIER: an amplification loop with no stage in it, rows={rows}"),
                &run,
                cfg,
                Some(floors),
            );
            per_unit("barrier", &cpu, floors.cpu, K as f64, "pass");
            cpu.median_diff / K as f64
        };

        let stages: [(&str, fn(u32, EncodeStageRepeats) -> EncodeStageRepeats); 6] = [
            ("asserts", |k, z| EncodeStageRepeats { asserts: k, ..z }),
            ("temporal (1 of the 2)", |k, z| EncodeStageRepeats {
                temporal: k,
                ..z
            }),
            ("StrDict::build", |k, z| EncodeStageRepeats {
                strdict: k,
                ..z
            }),
            ("bases", |k, z| EncodeStageRepeats { bases: k, ..z }),
            ("trap Box (alloc+free)", |k, z| EncodeStageRepeats {
                trap: k,
                ..z
            }),
            ("out buffer (alloc+free)", |k, z| EncodeStageRepeats {
                out: k,
                ..z
            }),
        ];

        let mut named_total = 0.0;
        let mut whole = 0.0;
        for (label, make) in stages {
            reset_encode_stage_passes();
            let run = {
                let mut a = Arm::new("repeats all zero", arm(zero));
                let mut b = Arm::new(label, arm(make(K, zero)));
                run_pair(&mut a, &mut b, cfg)
            };
            let passes = encode_stage_out_passes();
            let (cpu, _) = report(
                &format!("STAGE `{label}` x{K}, rows={rows}"),
                &run,
                cfg,
                Some(floors),
            );
            per_unit(label, &cpu, floors.cpu, K as f64, "pass");
            whole = cpu.a_median;
            if verdict(&cpu, floors.cpu).resolved() {
                let net = cpu.median_diff / K as f64 - barrier_ns;
                println!("    net of barrier: {net:+.4} ns");
                named_total += net;
            } else {
                println!("    UNRESOLVED — contributes nothing to the named total");
            }
            // The `out` stage is zero-trip under `BatchReduce::Sum`, and an
            // ELIDED stage reads the same. Only the counter separates them.
            if label.starts_with("out buffer") {
                println!(
                    "    body ran {passes} times ({}); a near-zero figure with a \
                     NON-ZERO count means possibly ELIDED, and needs the disassembly",
                    if passes == 0 {
                        "zero-trip: this batch reduces by Sum, so the stage genuinely never runs"
                    } else {
                        "the stage really executed"
                    }
                );
            }
        }

        // The residual, reported rather than inferred. `whole` is arm A's own
        // median: one `bind_per_row_resolved`, which is the encode this section
        // is splitting.
        println!();
        println!("  WHOLE encode (arm A median)      {whole:9.4} ns");
        println!("  named stages, net of barrier     {named_total:9.4} ns");
        println!(
            "  RESIDUAL                         {:9.4} ns   ({:.1}% of the whole)",
            whole - named_total,
            100.0 * (whole - named_total) / whole,
        );
        println!(
            "  ⚠ a residual above ~50% means NOT YET SPLIT, not `the named stages are the answer`"
        );
        println!(
            "  ⛔ every stage above is an UPPER BOUND on what removing it buys. Amplification"
        );
        println!(
            "     prices an operation SERIALIZED and in ISOLATION; a removal prices it at the"
        );
        println!(
            "     margin. On this box an isolated 8-byte alloc+free is 9.16 ns while removing"
        );
        println!("     one bought 2.62 ns. Do not read a stage as a budget for a fix.");
    }
}

#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
fn loop_key_program(
    cols: &LoopKeyColumns,
    extra: usize,
) -> (std::sync::Arc<cel::majit::bytecode::Code>, Vec<i64>, usize) {
    use cel::majit::bytecode::OP_JUMP_IF_ABOVE;
    use cel::majit::lower::{lower_typed, Schema, ValType};

    let program =
        Program::compile("balance >= amount && !frozen").expect("the probe expression compiles");
    let schema: Schema = [
        ("balance".to_string(), ValType::Int),
        ("amount".to_string(), ValType::Int),
        ("frozen".to_string(), ValType::Bool),
    ]
    .into_iter()
    .collect();
    let lowered = lower_typed(program.expression(), &schema).expect("the probe expression lowers");
    let bases = [
        cols.balance.as_ptr() as i64,
        cols.amount.as_ptr() as i64,
        cols.frozen.as_ptr() as i64,
    ];
    let (shape, regs) = lowered.batch_sum_program(&bases, LOOP_KEY_ROWS as i64);

    let mut words = shape.code.to_vec();
    let base_len = words.len();
    for block in 0..extra {
        // Backward by construction: the target names the word just before this
        // block, which is inside the base program for the first block and
        // inside the previous block after that.
        let target = (base_len + 4 * block - 1) as i64;
        words.extend_from_slice(&[OP_JUMP_IF_ABOVE, 0, 0, target]);
    }
    (std::sync::Arc::from(words), regs, shape.num_float_regs)
}

/// Price one program's loop-key walk, and return the per-key figure it
/// resolved to.
///
/// `keys` is read back out of the door rather than passed in, so the divisor
/// is the number of walks the door performs and not the number a reader of
/// this file would expect it to.
#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
fn loop_key_section(cols: &LoopKeyColumns, extra: usize, cfg: &Config) {
    use cel::majit::bytecode::float_bank::{
        jit_stats, loop_key_count, reset_jit_stats, reset_persistent_state,
        run_jit_persistent_probe_f, LoopKeyArm,
    };

    let (code, regs, nf) = loop_key_program(cols, extra);
    let keys = loop_key_count(&code);

    println!();
    println!("###########################################################");
    println!("# ENTRY-DOOR LOOP KEYS: {keys} per call ({extra} appended)");
    println!("###########################################################");

    // A fresh driver, so the compiled artifact the timed calls run on is the
    // one this section's warmup minted rather than one an earlier section left
    // behind under a different program.
    reset_persistent_state();
    reset_jit_stats();

    let arm = |which| {
        let code = &code;
        let regs = &regs;
        move || {
            let out = run_jit_persistent_probe_f(
                black_box(code),
                black_box(regs),
                nf,
                LOOP_KEY_JIT_ON,
                which,
            );
            black_box(out);
        }
    };

    // Both arms must answer the same thing. They decide identically on every
    // bucket holding one cell or none, and that is every bucket a program here
    // produces — but "must" is what a check is for.
    let resolved =
        run_jit_persistent_probe_f(&code, &regs, nf, LOOP_KEY_JIT_ON, LoopKeyArm::Resolved);
    let bare = run_jit_persistent_probe_f(&code, &regs, nf, LOOP_KEY_JIT_ON, LoopKeyArm::BareHash);
    assert_eq!(
        resolved, bare,
        "the two loop-key arms disagree on the answer at {keys} keys"
    );

    // Warm both arms past the threshold, so the timed calls run on a compiled
    // entry rather than paying for a trace inside the window.
    for _ in 0..64 {
        arm(LoopKeyArm::Resolved)();
        arm(LoopKeyArm::BareHash)();
    }

    // WHICH TIER THE TIMED CALLS RUN ON, measured rather than assumed. A
    // section whose `compiled_entries` stayed at zero over a window of calls
    // priced the door in front of the INTERPRETER, which is a different
    // question from the one task #31 asks; printing it is what lets the answer
    // be read for what it is.
    const CENSUS_CALLS: usize = 64;
    let before = jit_stats();
    for _ in 0..CENSUS_CALLS {
        arm(LoopKeyArm::Resolved)();
    }
    let after = jit_stats();
    println!(
        "  tier census      {CENSUS_CALLS} calls -> compiled_entries +{}, loops_compiled +{}, \
         aborted +{}   (answer {resolved})",
        after.compiled_entries - before.compiled_entries,
        after.loops_compiled - before.loops_compiled,
        after.loops_aborted - before.loops_aborted,
    );

    let floors = local_floor(
        &format!("LOCAL NULL CONTROL: {keys} loop keys, resolved arm vs an identical copy"),
        || arm(LoopKeyArm::Resolved),
        cfg,
    );

    let run = {
        let mut a = Arm::new("bare hash (pre-dc9146c)", arm(LoopKeyArm::BareHash));
        let mut b = Arm::new("resolved key (HEAD)", arm(LoopKeyArm::Resolved));
        run_pair(&mut a, &mut b, cfg)
    };
    let (cpu, _) = report(
        &format!("LOOP-KEY RESOLUTION at {keys} keys per call"),
        &run,
        cfg,
        Some(floors),
    );
    // Per CALL first, because that is the quantity the task asks for: one call
    // is one iteration of this arm, so the paired median IS the per-call cost
    // and the floor grades it directly.
    if verdict(&cpu, floors.cpu).resolved() {
        println!(
            "  per call         {:+.4} ns ({:+.2}% of the bare-hash arm)   floor ±{:.4} ns/call",
            cpu.median_diff,
            100.0 * cpu.median_diff / cpu.a_median,
            floors.cpu,
        );
    } else {
        println!(
            "  per call         UNRESOLVED: |{:+.4}| ns below this section's floor of \
             ±{:.4} ns/call — the cost is bounded ABOVE by the floor, not shown to be zero",
            cpu.median_diff, floors.cpu,
        );
    }
    if keys > 0 {
        per_unit("per loop key", &cpu, floors.cpu, keys as f64, "key");
    }
}

/// A program with NO backward jump, which is the case the door is supposed to
/// charge nothing for.
///
/// Straight-line words the interpreter runs to a `RETURN`, and no word in them
/// is `OP_JUMP_IF_ABOVE`, so `loop_keys` is empty and the walk never begins.
/// The structural claim is that the two arms cannot differ here; the timing is
/// what says the harness agrees.
#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
fn loop_key_zero_section(cfg: &Config) {
    use cel::majit::bytecode::float_bank::{
        loop_key_count, reset_jit_stats, reset_persistent_state, run_jit_persistent_probe_f,
        LoopKeyArm,
    };
    use cel::majit::bytecode::{OP_ADD, OP_LOAD_CONST, OP_RETURN};

    // `LOAD_CONST 7 -> r1`, `ADD r0 r1 -> r2`, `RETURN r2`. No word in it is
    // `OP_JUMP_IF_ABOVE`, which is what makes the key list empty.
    let words: Vec<i64> = vec![OP_LOAD_CONST, 7, 1, OP_ADD, 0, 1, 2, OP_RETURN, 2];
    let code: std::sync::Arc<cel::majit::bytecode::Code> = std::sync::Arc::from(words);
    let regs = vec![0i64; 3];
    let keys = loop_key_count(&code);
    assert_eq!(keys, 0, "the no-back-edge control must have no loop keys");

    println!();
    println!("###########################################################");
    println!("# ENTRY-DOOR LOOP KEYS: 0 per call (no backward jump)");
    println!("###########################################################");

    reset_persistent_state();
    reset_jit_stats();

    let arm = |which| {
        let code = &code;
        let regs = &regs;
        move || {
            let out = run_jit_persistent_probe_f(
                black_box(code),
                black_box(regs),
                0,
                LOOP_KEY_JIT_ON,
                which,
            );
            black_box(out);
        }
    };
    let resolved =
        run_jit_persistent_probe_f(&code, &regs, 0, LOOP_KEY_JIT_ON, LoopKeyArm::Resolved);
    let bare = run_jit_persistent_probe_f(&code, &regs, 0, LOOP_KEY_JIT_ON, LoopKeyArm::BareHash);
    assert_eq!(resolved, bare, "the two arms disagree with no loop keys");
    for _ in 0..64 {
        arm(LoopKeyArm::Resolved)();
        arm(LoopKeyArm::BareHash)();
    }

    let floors = local_floor(
        "LOCAL NULL CONTROL: 0 loop keys, resolved arm vs an identical copy",
        || arm(LoopKeyArm::Resolved),
        cfg,
    );
    let run = {
        let mut a = Arm::new("bare hash (pre-dc9146c)", arm(LoopKeyArm::BareHash));
        let mut b = Arm::new("resolved key (HEAD)", arm(LoopKeyArm::Resolved));
        run_pair(&mut a, &mut b, cfg)
    };
    let (cpu, _) = report(
        "LOOP-KEY RESOLUTION at 0 keys per call (must not resolve)",
        &run,
        cfg,
        Some(floors),
    );
    if verdict(&cpu, floors.cpu).resolved() {
        println!(
            "  !! a comparison whose two arms run the SAME instructions RESOLVED at \
             {:+.4} ns/call against a floor of ±{:.4}. That is a statement about the \
             instrument, not about the door.",
            cpu.median_diff, floors.cpu,
        );
    } else {
        println!(
            "  per call         UNRESOLVED, as it must be: the walk never begins, so the \
             two arms are the same instructions (|{:+.4}| vs floor ±{:.4} ns/call)",
            cpu.median_diff, floors.cpu,
        );
    }
}

/// Everything behind `loop-key-arm-probe`, run inside the bracket the opening
/// and closing null controls form.
#[cfg(all(feature = "jit", feature = "loop-key-arm-probe"))]
fn loop_key_probe(cfg: &Config) {
    println!();
    println!("===========================================================");
    println!("=  LOOP-KEY PROBE (feature `loop-key-arm-probe`)");
    println!("=  One call is one iteration here, so the paired median IS");
    println!("=  a per-call figure. Every section sets its own floor from");
    println!("=  a null control on its own arms.");
    println!("===========================================================");

    let cols = loop_key_columns(LOOP_KEY_ROWS);
    loop_key_zero_section(cfg);
    // A ladder rather than one point: the walk is one per key, so a cost that
    // is really the walk has to grow with the key count. A per-key figure that
    // holds across the rungs is the evidence for that; one taken at a single
    // rung would be an assumption with a number attached.
    for extra in [0usize, 3, 15, 63] {
        loop_key_section(&cols, extra, cfg);
    }
}
