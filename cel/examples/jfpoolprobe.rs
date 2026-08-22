//! What the per-entry jitframe `calloc`/`free` pair costs the JIT entry, as a
//! difference taken inside ONE binary.
//!
//! `entryprobe` splits the ~109 ns a `Tier::Jit` call spends ARRIVING in
//! compiled code into four measured stages and a residual, and the residual is
//! 79% of it. The leading candidate for what is inside the residual is the
//! frame itself: `run_compiled_code_inner` (majit-backend-cranelift) branches on
//! whether a JITFRAME type id is registered, cel never registers one — two heaps
//! would be two universes, and `install_cel_gc` deliberately has no caller — so
//! every compiled entry takes the `else` arm and builds its frame out of the
//! Rust heap. That is one `vec![0i64; 21..22]` in and one free out, ~170 bytes,
//! once per entry.
//!
//! This file prices that pair by running the door twice over the same shapes
//! with the frame coming from two different places:
//!
//! ```text
//!   owned    vec![0i64; words]  in,  free  out    — what ships
//!   pooled   a per-thread free list, cleared to `words` on the way out
//! ```
//!
//! # Two arms, one binary
//!
//! `drop-arm-probe`, `loop-key-arm-probe` and `entry-stage-probe` all exist
//! because two `cargo build` invocations cannot answer what one lowering costs
//! against another: they admit compile drift and stale binaries, and neither is
//! visible in the numbers they produce. The arm here is the same shape, one
//! layer down — `majit_metainterp::set_jitframe_pool` flips a process-global
//! selector that `run_compiled_code_inner` reads per entry, and the flip is done
//! from INSIDE each timed closure so both arms pay the store and it cancels out
//! of their difference.
//!
//! The arms INTERLEAVE for the same reason `entryprobe`'s do: on a loaded box,
//! timing all of one arm and then all of the other puts a load excursion
//! entirely inside one of them.
//!
//! # What is read, and what it is read against
//!
//! Three arms per swept point — `Tier::Clean`, `Tier::Jit` owned, `Tier::Jit`
//! pooled — and two OLS intercepts per shape. `entry = jit_fix - clean_fix` is
//! `routeprobe`'s reading and `entryprobe`'s, so the two entries printed here
//! land beside `batch::JIT_ENTRY_PS`. The DELTA between them is
//! `jit_fix_pooled - jit_fix_owned`: `clean_fix` is common to both and drops
//! out, which is what makes the delta the tighter of the two numbers.
//!
//! ⛔ The delta is a change in the WHOLE entry. `entryprobe`'s stage E is a
//! residual and stays one; nothing here measures it.
//!
//! ⛔ Pooling removes the `calloc`/`free` pair but NOT the zero-fill — the
//! pooled arm memsets the same `words` on the way out, because the frame header
//! must start with `jf_descr == 0` (`GuardNotForced` reads `jf_descr != 0`) and
//! a reused buffer carries the last entry's. So the delta is the allocator
//! round trip and not the cost of producing a zeroed frame.
//!
//! An allocation COUNT is the gate, not this: `cel/tests/allocs_per_eval.rs`
//! under `MAJIT_JITFRAME_POOL=1` is what proves the allocation disappeared
//! rather than moved. A timing win with an unchanged count is a measurement
//! error. The pool's own counters are printed at the end as the weaker witness
//! that the selector reached the allocation at all.
//!
//! RELEASE ONLY:
//!
//! ```text
//! cargo build --release -p cel --features jit-cranelift --example jfpoolprobe
//! target/release/examples/jfpoolprobe
//! ```

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, Tier, JIT_ENTRY_PS};
use cel::majit::bytecode::float_bank::{reset_persistent_state, COMPILED_ENTRIES};
use cel::majit::lower::{Schema, ValType};
use cel::Value;

/// One timed batch must last at least this long, so the clock's own resolution
/// is not what the measurement is against. `entryprobe`'s number, because the
/// entries here have to be comparable with its.
const MIN_BATCH: Duration = Duration::from_millis(20);
/// Timed batches per arm; the fastest is reported, since other work on the box
/// can only ever make one slower. `entryprobe`'s count: the difference being
/// read is a few nanoseconds inside a call of a few hundred.
const ROUNDS: usize = 21;
/// Calls in the probe that decides whether a point's `jit` cell is the compiled
/// tier's number or the tracing interpreter's.
const PROBE_CALLS: usize = 200;
/// The points the intercepts are read from — `entryprobe`'s region, at or below
/// four, where a fixed entry cost is the largest share of what is timed.
const SWEEP: [usize; 4] = [1, 2, 3, 4];

fn timed(iters: usize, run: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        run();
    }
    start.elapsed()
}

/// Time THREE calls against each other, INTERLEAVED: one iteration count is
/// calibrated so a batch of any of them lasts at least [`MIN_BATCH`], then the
/// three take turns for [`ROUNDS`] batches each and the fastest of each is
/// reported.
///
/// `entryprobe::per_call_pair` widened by one arm. Turn-taking and not one arm
/// after another, because the difference between two of them is the whole
/// measurement and a busy box does not hold still for the length of a sweep.
fn per_call_trio(mut a: impl FnMut(), mut b: impl FnMut(), mut c: impl FnMut()) -> (f64, f64, f64) {
    let mut iters = 1usize;
    while timed(iters, &mut a)
        .max(timed(iters, &mut b))
        .max(timed(iters, &mut c))
        < MIN_BATCH
    {
        iters = (iters * 2).max(1);
    }
    let (mut best_a, mut best_b, mut best_c) = (Duration::MAX, Duration::MAX, Duration::MAX);
    for _ in 0..ROUNDS {
        best_a = best_a.min(timed(iters, &mut a));
        best_b = best_b.min(timed(iters, &mut b));
        best_c = best_c.min(timed(iters, &mut c));
    }
    let ns = |d: Duration| d.as_secs_f64() * 1e9 / iters as f64;
    (ns(best_a), ns(best_b), ns(best_c))
}

/// Run the batch loop until the trace threshold is crossed, then report the
/// share of a fixed probe's calls that ENTERED compiled code.
///
/// A `Tier::Jit` call that did not enter is the tracing interpreter's and pays
/// no entry at all, so a point measured through one describes the harness
/// rather than the door. Everything downstream drops the points below one.
fn warm(bound: &BoundBatch<'_, '_>, out: &mut Vec<Value>) -> f64 {
    for _ in 0..512 {
        bound.collect_into_on(Tier::Jit, out).expect("warm run");
        black_box(out.as_slice());
    }
    let before = COMPILED_ENTRIES.load(Ordering::Relaxed);
    for _ in 0..PROBE_CALLS {
        bound.collect_into_on(Tier::Jit, out).expect("probe run");
        black_box(out.as_slice());
    }
    (COMPILED_ENTRIES.load(Ordering::Relaxed) - before) as f64 / PROBE_CALLS as f64
}

/// One swept point: `(n, clean, jit owned, jit pooled, entered share)`.
fn point(bound: &BoundBatch<'_, '_>, n: usize) -> (usize, f64, f64, f64, f64) {
    let mut out: Vec<Value> = Vec::new();
    let entered = warm(bound, &mut out);

    // The pooled arm reuses memory the previous entry wrote, so an arm that
    // stopped clearing it would still return `Ok` — and would return the last
    // entry's `jf_descr` to a guard that reads it. Both arms are checked
    // against the clean tier's answer before either is timed, which is the
    // convention `entryprobe::arms` and `elem-attr-probe` bind their arms under.
    let mut want: Vec<Value> = Vec::new();
    bound
        .collect_into_on(Tier::Clean, &mut want)
        .expect("clean witness");
    for pooled in [false, true] {
        majit_metainterp::set_jitframe_pool(pooled);
        let mut got: Vec<Value> = Vec::new();
        for _ in 0..64 {
            got.clear();
            bound
                .collect_into_on(Tier::Jit, &mut got)
                .expect("witness run");
            assert!(
                got == want,
                "the jitframe arm pooled={pooled} changed the answer at n={n}"
            );
        }
    }

    // Three buffers, not one: the arms alternate, and a shared buffer would
    // hand each arm the other's allocation state.
    let mut out_clean = out.clone();
    let mut out_owned = out.clone();
    let mut out_pooled = out;
    // Set from INSIDE each arm, so both JIT arms pay one atomic store per call
    // and it cancels out of their difference. Setting it once outside would
    // not: the arms interleave in batches, and each batch would then have to
    // restore what the other left.
    let jit = |pooled: bool, buf: &mut Vec<Value>| {
        majit_metainterp::set_jitframe_pool(pooled);
        bound.collect_into_on(Tier::Jit, buf).expect("jit run");
        black_box(buf.as_slice());
    };
    let (clean, owned, pooled) = per_call_trio(
        || {
            bound
                .collect_into_on(Tier::Clean, &mut out_clean)
                .expect("clean run");
            black_box(out_clean.as_slice());
        },
        || jit(false, &mut out_owned),
        || jit(true, &mut out_pooled),
    );
    (n, clean, owned, pooled, entered)
}

/// Ordinary least squares of `y` against `x`, as `(intercept, slope)`.
fn line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
    (my - slope * mx, slope)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One shape's two entries.
struct Arms {
    label: &'static str,
    entered: f64,
    clean_fix: f64,
    owned_fix: f64,
    pooled_fix: f64,
    /// The smallest swept point that entered compiled code, and its two raw
    /// per-call figures — the difference before any line is fitted through it.
    raw_n: usize,
    raw_owned: f64,
    raw_pooled: f64,
}

impl Arms {
    /// What a call pays to reach compiled code with the shipping allocation.
    fn entry_owned(&self) -> f64 {
        self.owned_fix - self.clean_fix
    }
    /// …and with the frame off the pool.
    fn entry_pooled(&self) -> f64 {
        self.pooled_fix - self.clean_fix
    }
    /// The A/B. `clean_fix` is common to both entries and drops out, so this is
    /// a difference between two fitted intercepts rather than between two
    /// differences of them.
    fn delta(&self) -> f64 {
        self.pooled_fix - self.owned_fix
    }
}

fn fit(label: &'static str, points: &[(usize, f64, f64, f64, f64)], entered: f64) -> Arms {
    let used: Vec<&(usize, f64, f64, f64, f64)> = points.iter().filter(|p| p.4 >= 1.0).collect();
    let xs: Vec<f64> = used.iter().map(|p| p.0 as f64).collect();
    let cs: Vec<f64> = used.iter().map(|p| p.1).collect();
    let os: Vec<f64> = used.iter().map(|p| p.2).collect();
    let ps: Vec<f64> = used.iter().map(|p| p.3).collect();
    let raw = used.first().copied();
    Arms {
        label,
        entered,
        clean_fix: line(&xs, &cs).0,
        owned_fix: line(&xs, &os).0,
        pooled_fix: line(&xs, &ps).0,
        raw_n: raw.map_or(0, |p| p.0),
        raw_owned: raw.map_or(f64::NAN, |p| p.2),
        raw_pooled: raw.map_or(f64::NAN, |p| p.3),
    }
}

/// The `n`-row prefix of two int columns, as the batch a row sweep binds.
///
/// A function and not a closure: what it returns borrows the columns, and a
/// closure would tie that borrow to the closure itself rather than to the
/// locals the columns live in.
fn rows_batch<'a>(x: &'a [i64], y: &'a [i64], n: usize) -> Batch<'a> {
    Batch::new(n)
        .column("x", ColumnRef::Int(&x[..n]))
        .column("y", ColumnRef::Int(&y[..n]))
}

/// One row carrying an `n`-element list, as the batch an element sweep binds.
fn elems_batch<'a>(lens: &'a [i64], flat: &'a [i64], n: usize) -> Batch<'a> {
    Batch::new(1).column(
        "list",
        ColumnRef::List {
            lens: &lens[n..n + 1],
            fields: vec![(None, ColumnRef::Int(&flat[..n]))],
        },
    )
}

fn sweep(label: &'static str, points: Vec<(usize, f64, f64, f64, f64)>) -> Arms {
    let entered = points.iter().map(|p| p.4).fold(0.0f64, f64::max);
    fit(label, &points, entered)
}

/// A straight-line program swept by batch HEIGHT.
fn arms_rows(label: &'static str, source: &'static str) -> Arms {
    let mut schema = Schema::new();
    schema.insert("x".to_string(), ValType::Int);
    schema.insert("y".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    let top = SWEEP[SWEEP.len() - 1];
    // Built once and SLICED per point, so every point binds the same columns
    // rather than an allocation that happens to hold equal values.
    let x: Vec<i64> = (0..top as i64).collect();
    let y: Vec<i64> = (0..top as i64).map(|v| v + 3).collect();
    let mut points = Vec::new();
    for &n in &SWEEP {
        let batch = rows_batch(&x, &y, n);
        // A fresh driver pool per point, the protocol both gated benches bind
        // under. Without it every point after the first is timed against a pool
        // this sweep itself grew.
        reset_persistent_state();
        let bound = program.bind_per_row(&batch).expect("binds");
        points.push(point(&bound, n));
    }
    sweep(label, points)
}

/// A comprehension at ONE row whose list carries a rising number of elements.
fn arms_elems(label: &'static str, source: &'static str) -> Arms {
    let mut schema = Schema::new();
    schema.insert("list[]".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    let top = SWEEP[SWEEP.len() - 1];
    let flat: Vec<i64> = (0..top as i64).collect();
    let lens: Vec<i64> = (0..=top as i64).collect();
    let mut points = Vec::new();
    for &n in &SWEEP {
        let batch = elems_batch(&lens, &flat, n);
        reset_persistent_state();
        let bound = program.bind_per_row(&batch).expect("binds");
        points.push(point(&bound, n));
    }
    sweep(label, points)
}

fn main() {
    println!("the jitframe allocation, priced: one calloc/free pair per compiled entry");
    println!(
        "best of {ROUNDS} interleaved batches of >= {} ms per arm; entry read as \
         jit_fix - clean_fix\nover n in {SWEEP:?}, a fresh driver pool per point; \
         batch.rs ships JIT_ENTRY_PS = {JIT_ENTRY_PS} ps.",
        MIN_BATCH.as_millis()
    );

    let arms = vec![
        arms_rows("rows: x + 1", "x + 1"),
        arms_rows(
            "rows: x > 3 && y < 90 || x == y",
            "x > 3 && y < 90 || x == y",
        ),
        arms_rows(
            "rows: x*2 + x*3 + y*4 + y*5 + x*6 + y*7",
            "x*2 + x*3 + y*4 + y*5 + x*6 + y*7",
        ),
        arms_elems("elems: list.all(e, e > 0)", "list.all(e, e > 0)"),
        arms_elems("elems: list.map(e, e + 1)", "list.map(e, e + 1)"),
        arms_elems(
            "elems: list.filter(e, e % 2 == 0)",
            "list.filter(e, e % 2 == 0)",
        ),
    ];

    println!(
        "\n  {:<42} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8}",
        "shape", "enter", "clean fix", "owned fix", "pool fix", "entry own", "entry pool", "delta"
    );
    for a in &arms {
        println!(
            "  {:<42} {:>6.2} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2}",
            a.label,
            a.entered,
            a.clean_fix,
            a.owned_fix,
            a.pooled_fix,
            a.entry_owned(),
            a.entry_pooled(),
            a.delta()
        );
    }

    println!("\n  the same difference BEFORE any line is fitted, at the smallest entered point");
    println!(
        "  {:<42} {:>4} {:>11} {:>11} {:>8}",
        "shape", "n", "owned ns", "pooled ns", "delta"
    );
    for a in &arms {
        println!(
            "  {:<42} {:>4} {:>11.1} {:>11.1} {:>8.2}",
            a.label,
            a.raw_n,
            a.raw_owned,
            a.raw_pooled,
            a.raw_pooled - a.raw_owned
        );
    }

    let owned = median(arms.iter().map(Arms::entry_owned).collect());
    let pooled = median(arms.iter().map(Arms::entry_pooled).collect());
    let delta = median(arms.iter().map(Arms::delta).collect());
    let spread = |f: fn(&Arms) -> f64| {
        arms.iter()
            .map(f)
            .fold((f64::MAX, f64::MIN), |(lo, hi), e| (lo.min(e), hi.max(e)))
    };
    let (olo, ohi) = spread(Arms::entry_owned);
    let (plo, phi) = spread(Arms::entry_pooled);
    let (dlo, dhi) = spread(Arms::delta);

    println!("\n  MEDIAN over {} shapes", arms.len());
    println!(
        "    entry, frame owned           {owned:8.2} ns   spread {olo:.1} .. {ohi:.1}  \
         ({:.2}x)",
        ohi / olo
    );
    println!(
        "    entry, frame pooled          {pooled:8.2} ns   spread {plo:.1} .. {phi:.1}  \
         ({:.2}x)",
        phi / plo
    );
    println!(
        "    delta (pooled - owned)       {delta:8.2} ns   spread {dlo:.2} .. {dhi:.2}   \
         {:+.1}% of the owned entry",
        100.0 * delta / owned
    );
    println!(
        "    JIT_ENTRY_PS, for reference  {:8.2} ns",
        JIT_ENTRY_PS as f64 / 1000.0
    );

    // The weaker of the two witnesses, and the only one this binary can take:
    // it says the selector reached the allocation, not that an allocation
    // stopped happening. `cel/tests/allocs_per_eval.rs` under
    // MAJIT_JITFRAME_POOL=1 is what says the second thing.
    let (n_owned, n_taken, n_missed) = majit_metainterp::jitframe_pool_counts();
    println!(
        "\n  frame buffers on this thread: {n_owned} allocated by the owned arm, \
         {n_taken} handed out by the pool\n  of which {n_missed} found it empty and \
         allocated after all -- {:.4}% miss.",
        if n_taken == 0 {
            0.0
        } else {
            100.0 * n_missed as f64 / n_taken as f64
        }
    );
    if n_owned == 0 || n_taken == 0 {
        println!(
            "  ⚠ ONE ARM NEVER RAN. Either the selector did not reach \
             `run_compiled_code_inner`,\n    or this build registered a JITFRAME type id and \
             takes the nursery arm instead --\n    in which case there is no per-entry \
             calloc to price and the delta above is noise."
        );
    }
}
