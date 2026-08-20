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
#[cfg(feature = "drop-arm-probe")]
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
#[cfg(feature = "drop-arm-probe")]
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
#[cfg(feature = "drop-arm-probe")]
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
    // element:
    //
    //   map:  StoreLocal iter_var (the old element) + JumpIfFalse (the
    //         exhaustion guard)                                        = 2
    //   all:  those two, plus JumpIfFalse (the loop condition), And,
    //         AndMerge, and StoreLocal accu                            = 6
    //
    // `map` reaches two of the four site families and `all` reaches all four,
    // so the two ladders are an independent pair of estimates rather than one
    // measurement run twice.
    let ladders: [(&str, f64); 2] = [("xs.map(x, x * 2)", 2.0), ("xs.all(x, x > 0)", 6.0)];

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
