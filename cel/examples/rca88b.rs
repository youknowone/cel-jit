//! #88 probes A and B — the two experiments left that need no in-tree instrument.
//!
//! `cel/examples/rca88.rs` established that the wall is a per-call cost that
//! survives warmup (25-57x at n=10, both backends), that `gfails/call == 1.00`
//! is NOT the cause (it reads 0.00 exactly where the wall is), and that
//! `loops/call` and `aborts/call` are 0.00 in every cell. All four counters cel
//! exports read zero in the cells that matter, so the COUNTERS are exhausted.
//!
//! The experiment DESIGN is not. Both probes here vary something majit's
//! counters do not report:
//!
//! **Probe B — is it the artifact, or the boundary?** A third arm: majit
//! enabled, threshold set so high nothing ever compiles. `run_jit_persistent_f`
//! keys its driver pool on `(nregs, nfregs, threshold)`, so this arm gets its
//! own driver and cannot reach the compiled one.
//!   * wall present  => the cost is being on the majit call path at all, and
//!     has nothing to do with compiled code;
//!   * wall absent   => the cost requires a compiled artifact, and Probe A says
//!     whether it is per-artifact or per-population.
//!
//! **Probe A — per-call, or a scan over a population?** 62 µs is ~200 000
//! cycles with zero allocations, zero compiles and zero guard failures. One
//! shape that fits is a per-call linear pass over a population that was
//! CONSTANT during the rca88 sweep — in which case it would read as a fixed
//! per-call cost by construction. cel has exactly such a population and it is
//! reachable from outside: `run_jit_persistent_f` inserts every program it sees
//! into `pooled.programs`, a `HashMap` bounded by `MAX_PROGRAMS_PER_DRIVER`
//! (256). Hold n fixed and grow that map.
//!
//! For the population to grow, the filler programs must land in the SAME
//! driver, i.e. share `(nregs, nfregs, threshold)`. The assert in
//! `population_sweep` checks that rather than assuming it — a filler that lands
//! in its own driver would leave the population at 1 and the probe would report
//! "flat" for a reason that has nothing to do with the hypothesis. The `loops`
//! column is the second, independent proof it was not vacuous: it reads exactly
//! `pop` at every point, so the population really did grow.
//!
//! # RESULTS (both backends, host load 11-23)
//!
//! **Probe A — NEGATIVE. The wall does not track the population.** `warm/clean`
//! over pop = 1 / 8 / 24 / 100 / 200: 27.50 / 28.87 / 27.73 / 29.60 / 30.74
//! (cranelift) and 24.63 / 28.33 / 24.57 / 25.13 / 27.26 (dynasm) — flat, and
//! non-monotonic on dynasm, which is the noise signature rather than a trend.
//! Across a 200x change in population the wall does not move. It is genuinely a
//! per-call cost, and no lookup or scan over the interned set explains it.
//!
//! **Probe B — the wall SPLITS IN TWO, and the fixed µs belongs to the
//! ARTIFACT.** Fitting each arm over n (cranelift):
//!
//! | arm | fixed per call | per row |
//! |---|---|---|
//! | clean | ~0.1 µs | 8.4 ns |
//! | majit, never compiles | 0.77 µs | **166 ns** |
//! | majit, compiled | **~4.4 µs** | ~0 ns |
//!
//! 1. Running cel's bytecode through the majit portal with nothing compiled
//!    costs **166 ns/row (cranelift) / 173 ns/row (dynasm)** against the clean
//!    interpreter's ~8 ns/row — a **~20x per-ROW tax** with zero tracing and
//!    zero compilation (`loops@nvr == 0` in every row). This is a real cost and
//!    it is NOT #88's wall, which is fixed per call.
//!    ⭐ It is also backend-identical to 4% (165.8 vs 172.7 ns/row), which is
//!    the control this arm carries for free: it generates no code, so the two
//!    backends MUST agree, and they do.
//! 2. The compiled arm is **flat in n** (4907 / 5328 / 3923 ns at n = 10 / 100 /
//!    1000) and carries ~4.4 µs of fixed cost against the no-compile arm's
//!    0.77 µs. **The ~3.6 µs difference appears only when an artifact exists**,
//!    so #88's wall is the price of entering and leaving compiled code, not the
//!    price of being on the majit call path.
//!
//! ⇒ At n=10 the compiled arm (4907 ns) is **worse than never compiling at all**
//! (2424 ns). Break-even between the two majit arms is n ~= 22.
//!
//! **Probe C — trace length is a REAL but MINORITY term, and it does not
//! explain the two-loop shapes at all.**
//!
//! ⛔ READ THE SCOPE FIRST: only the `warm n=10` and `warm n=1000` columns are
//! trustworthy. `warm n=100` is contradicted by Probe B **in the same binary,
//! same run, same expression** — Probe B reads 4230 ns at n=100 for
//! `price + qty * 2` where Probe C reads 19289, a 4.6x disagreement, while the
//! two probes agree to 0.5% at n=10 (4433 vs 4454) and 6% at n=1000 (3417 vs
//! 3213). The `lin@100` residual column is what caught it: it reads 4.1-5.2 for
//! every single-loop row, and the fitted `ns/row` comes out NEGATIVE, which no
//! positive per-row cost can produce. **`fixed/call` and `ns/row` are therefore
//! derived from a bad point and must not be quoted.** Cause unknown; the two
//! probes differ only in harness structure (Probe B interleaves three arms,
//! Probe C two). Eliminated: a refused batch falling back to the other
//! evaluator (#54) — the tier-agreement assertion added here does not fire.
//!
//! On the cross-validated `warm n=10` column, all eight single-loop rows fit
//! one line across THREE shape families (arithmetic, boolean, single-loop
//! comprehension):
//!
//! **per-call cost ~= 3.83 µs + 27.3 ns x trace_ops**
//!
//! So the lead's hypothesis is half right, and the half that fails matters
//! more. Trace length is real — but over a 3.3x range in `ops_post` (15 -> 49)
//! the cost rises only 22%, so it explains ~0.93 µs of a 4.2-5.2 µs cost. The
//! dominant **3.83 µs is constant** in trace length.
//!
//! The two-loop rows sit 2.4x / 3.4x / 10.4x ABOVE that line, and `ops_post`
//! actively mispredicts them: `pr=8` has the second-SMALLEST trace in the whole
//! table (24 ops) and the LARGEST cost (46.8 µs). Within that family the cost
//! is ~585-622 ns per inner ELEMENT (12441/20, 18020/30, 46764/80), flat to 6%.
//! So what separates the shapes is `loops_compiled` 1 -> 2 and the per-element
//! crossing between the two artifacts (#76), **not** trace length.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// Allocations of exactly `s` bytes land in bucket `s`; anything larger lands in
/// bucket 0. A `HashMap` keyed by size would allocate from inside the allocator,
/// so this is a flat array of `Cell` and never allocates at all.
const SIZE_MAX_EXACT: usize = 1024;
const SIZE_BUCKETS: usize = SIZE_MAX_EXACT + 1;

std::thread_local! {
    static LOCAL_ALLOCS: Cell<u64> = const { Cell::new(0) };
    /// Off by default: every probe that reports ns/call would otherwise pay for
    /// the histogram, and the timing probes are the ones already at issue.
    static SIZES_ON: Cell<bool> = const { Cell::new(false) };
    static LOCAL_SIZES: [Cell<u64>; SIZE_BUCKETS] = [const { Cell::new(0) }; SIZE_BUCKETS];
}
static GLOBAL_ALLOCS: AtomicU64 = AtomicU64::new(0);

struct Counting;

#[inline]
fn bump(size: usize) {
    GLOBAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
    let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
    if SIZES_ON.try_with(Cell::get).unwrap_or(false) {
        let idx = if size > SIZE_MAX_EXACT { 0 } else { size };
        let _ = LOCAL_SIZES.try_with(|h| h[idx].set(h[idx].get() + 1));
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn metered<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = LOCAL_ALLOCS.with(Cell::get);
    let out = f();
    let after = LOCAL_ALLOCS.with(Cell::get);
    (out, after - before)
}

/// Reads the running per-size totals with recording switched off: the `Vec` this
/// builds is itself an allocation, and it would otherwise be counted into the
/// bucket it is reading.
fn sizes_snapshot() -> Vec<u64> {
    let was = SIZES_ON.with(|c| c.replace(false));
    let out = LOCAL_SIZES.with(|h| h.iter().map(Cell::get).collect());
    SIZES_ON.with(|c| c.set(was));
    out
}

/// `metered`, plus the per-size breakdown of the same interval. The caller is
/// expected to check that the histogram sums to the count — they are collected
/// by two independent paths through `bump`, so a mismatch means the interval was
/// not the one it claims to be.
fn metered_sizes<T>(f: impl FnOnce() -> T) -> (T, u64, Vec<u64>) {
    // Both snapshots allocate a 1025-element `Vec`, so the count has to be read
    // INSIDE them — bracketing the other way charges `f` for the instrument.
    let before = sizes_snapshot();
    let before_n = LOCAL_ALLOCS.with(Cell::get);
    SIZES_ON.with(|c| c.set(true));
    let out = f();
    SIZES_ON.with(|c| c.set(false));
    let after_n = LOCAL_ALLOCS.with(Cell::get);
    let after = sizes_snapshot();
    let delta = after.iter().zip(&before).map(|(a, b)| a - b).collect();
    (out, after_n - before_n, delta)
}

fn hist_diff(a: &[u64], b: &[u64]) -> Vec<i64> {
    a.iter()
        .zip(b)
        .map(|(x, y)| *x as i64 - *y as i64)
        .collect()
}

/// `None` when the histogram does not divide evenly. That is the interesting
/// answer: `d` identical units cannot produce a bucket count indivisible by `d`,
/// so a `None` refutes "the entries are alike" at the size level even where the
/// per-call totals fit perfectly.
fn hist_div(h: &[i64], d: i64) -> Option<Vec<i64>> {
    h.iter()
        .all(|c| c % d == 0)
        .then(|| h.iter().map(|c| c / d).collect())
}

fn hist_line(h: &[i64]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut total: i64 = 0;
    for (size, &c) in h.iter().enumerate() {
        if c == 0 {
            continue;
        }
        total += c;
        // Bucket 0 is the overflow bucket, not a zero-byte request.
        if size == 0 {
            parts.push(format!(">{SIZE_MAX_EXACT}B x{c}"));
        } else {
            parts.push(format!("{size}B x{c}"));
        }
    }
    if parts.is_empty() {
        return "total 0".to_string();
    }
    format!("total {total:>5}  =  {}", parts.join("  "))
}

fn hist_u(h: &[u64]) -> String {
    hist_line(&h.iter().map(|&c| c as i64).collect::<Vec<_>>())
}

const THRESHOLD: u32 = 8;
/// Probe B's arm: majit is fully engaged, but the back-edge counter can never
/// reach this, so nothing is ever compiled. Distinct from `THRESHOLD` in the
/// `DRIVERS` key, so the two arms cannot share a driver or an artifact.
const NEVER: u32 = u32::MAX;
/// Under `MAX_PROGRAMS_PER_DRIVER` (256) at every point, because crossing it
/// drops the whole driver and silently resets the population being swept.
const POPULATIONS: &[usize] = &[1, 8, 24, 100, 200];

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

fn flat_schema() -> Schema {
    [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect()
}

fn flat_columns(rows: usize) -> (Vec<i64>, Vec<i64>) {
    (
        (0..rows as i64).map(|i| (i * 37) % 200).collect(),
        (0..rows as i64).map(|i| (i * 11) % 100).collect(),
    )
}

fn loadavg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

/// Probe B. Three arms at each n, interleaved within a round so a load spike
/// moves all three together, min over rounds.
fn artifact_or_boundary(label: &str, lowered: &LoweredF) {
    println!("\nProbe B — {label}: is the wall the ARTIFACT or the BOUNDARY?");
    println!(
        "{:>8} {:>11} {:>13} {:>13} {:>12} {:>14} {:>7} {:>9} {:>5} {:>5} {:>8}",
        "n",
        "clean ns",
        "jit(compiled)",
        "jit(nocomp)",
        "compiled/cl",
        "nocompiled/cl",
        "loops",
        "loops@nvr",
        "dL",
        "dB",
        "dG"
    );

    for n in [10usize, 100, 1_000] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];
        let reps = (20_000 / n).max(1);
        let rounds = 41;

        // Warm both JIT arms on their own drivers before timing anything.
        reset_persistent_state();
        reset_jit_stats();
        black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
        let compiled_stats = jit_stats();

        reset_jit_stats();
        for _ in 0..64 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }
        // The non-vacuity check for this whole probe: if this is not 0 the
        // "never compiles" arm compiled something and the split is void.
        let never_stats = jit_stats();

        let mut best_clean = f64::MAX;
        let mut best_compiled = f64::MAX;
        let mut best_never = f64::MAX;
        // Counter deltas across the timed window — see Probe C's header note.
        let window_before = jit_stats();
        for _ in 0..rounds {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(clean_batch_sum_f(lowered, &columns, n));
            }
            best_clean = best_clean.min(t.elapsed().as_nanos() as f64 / reps as f64);

            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
            }
            best_compiled = best_compiled.min(t.elapsed().as_nanos() as f64 / reps as f64);

            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
            }
            best_never = best_never.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        let window_after = jit_stats();
        println!(
            "{n:>8} {best_clean:>11.1} {best_compiled:>13.1} {best_never:>13.1} \
             {:>12.3} {:>14.3} {:>7} {:>9} {:>5} {:>5} {:>8}",
            best_compiled / best_clean,
            best_never / best_clean,
            compiled_stats.loops_compiled,
            never_stats.loops_compiled,
            window_after.loops_compiled - window_before.loops_compiled,
            window_after.bridges_compiled - window_before.bridges_compiled,
            window_after.guard_failures - window_before.guard_failures,
        );
    }
    println!(
        "  loops@nvr must be 0 in every row, or the no-compile arm is not one \
         and the split is void."
    );
}

/// Probe A. n is pinned at the size where the wall is largest; what moves is
/// how many OTHER programs share the driver.
fn population_sweep(label: &str, target_src: &str) {
    println!("\nProbe A — {label}: does the wall track the interned POPULATION?");
    println!(
        "{:>6} {:>11} {:>11} {:>10} {:>7} {:>11}",
        "pop", "clean ns", "warm ns", "warm/clean", "loops", "gfails/call"
    );

    const N: usize = 10;
    let schema = flat_schema();
    let (price, qty) = flat_columns(N);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];
    let target = lower(target_src, &schema);

    for &pop in POPULATIONS {
        // A fresh process-equivalent state per point, so the population is
        // exactly what this iteration puts there and not what the last one left.
        reset_persistent_state();
        reset_jit_stats();

        // `pop - 1` fillers plus the target. Structurally identical to the
        // target so they lower to the same register counts and therefore land
        // in the same driver; only a literal differs, which is what makes them
        // distinct programs with distinct addresses.
        let fillers: Vec<LoweredF> = (1..pop)
            .map(|k| lower(&format!("price + qty * {}", k + 2), &schema))
            .collect();
        for f in &fillers {
            assert_eq!(
                (f.num_int_regs, f.num_float_regs),
                (target.num_int_regs, target.num_float_regs),
                "filler lowers to a different shape than the target, so it \
                 lands in a different driver and the population never grows"
            );
            black_box(eval_batch_sum_f(f, &columns, N, THRESHOLD));
        }

        // Warm the target AFTER the population is in place.
        for _ in 0..64 {
            black_box(eval_batch_sum_f(&target, &columns, N, THRESHOLD));
        }
        let stats = jit_stats();

        let reps = 2_000;
        let mut best_clean = f64::MAX;
        let mut best_warm = f64::MAX;
        for _ in 0..41 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(clean_batch_sum_f(&target, &columns, N));
            }
            best_clean = best_clean.min(t.elapsed().as_nanos() as f64 / reps as f64);

            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(&target, &columns, N, THRESHOLD));
            }
            best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        reset_jit_stats();
        for _ in 0..100 {
            black_box(eval_batch_sum_f(&target, &columns, N, THRESHOLD));
        }
        let per_call = jit_stats().guard_failures as f64 / 100.0;

        println!(
            "{pop:>6} {best_clean:>11.1} {best_warm:>11.1} {:>10.3} {:>7} {per_call:>11.2}",
            best_warm / best_clean,
            stats.loops_compiled,
        );
    }
    println!(
        "  A rising warm/clean means the wall is a scan over the population. \
         Flat across 200x means it is genuinely per-call."
    );
}

/// Probe C. Probe B showed the compiled arm carries a fixed per-call cost that
/// does not scale with n. This asks what it DOES scale with.
///
/// The prompt is a discrepancy already sitting in the rca88 data: at n=10 the
/// single-loop shapes cost ~3.3-5.6 µs per call and the nested shape ~50 µs.
/// **An 11x difference between shapes, in a quantity that is flat in n.** A
/// pure per-call boundary constant would be identical for both, so the cost is
/// not a constant — it scales with something about the artifact.
///
/// Everything needed to test that is already exported: `loops_compiled`,
/// `bridges_compiled` and `trace_ops_before` / `trace_ops_after` are in cel's
/// `JitStats`. Hold n at 10, walk a complexity ladder, and read the per-call
/// cost against each.
///
/// n is reported at 10 AND 100 because the claim being tested is about a term
/// that does not scale with work — if the two columns track each other, what is
/// tabulated is the fixed cost rather than the row work.
fn artifact_scaling() {
    println!("\nProbe C — does the per-call cost scale with the ARTIFACT?");
    println!(
        "{:<34} {:>6} {:>5} {:>8} {:>8} {:>10} {:>11} {:>11} {:>11} {:>10} {:>9} {:>8} \
         {:>11} {:>11} {:>9} {:>5} {:>5} {:>8} {:>5} {:>6} {:>5} {:>13}",
        "expression",
        "loops",
        "brdg",
        "ops_pre",
        "ops_post",
        "clean n=10",
        "warm n=10",
        "warm n=100",
        "warm n=1000",
        "fixed/call",
        "ns/row",
        "lin@100",
        "clean n=100",
        "clean n=1000",
        "cLin@100",
        "dL",
        "dB",
        "dG",
        "L@10",
        "L@100",
        "L@1k",
        "warm/nvr@100"
    );
    // The discriminator for the invalidated `warm n=100` column. Probe B and
    // Probe C disagree 4.6x on the SAME expression at n=100 in the same binary.
    // If the CLEAN arm — which never touches majit — disagrees the same way,
    // the defect is in the harness or the timer and #88 is untouched. If only
    // the JIT arm disagrees, it is majit state-dependence. The clean arm's own
    // `lin@100` residual is the load-independent form of that question, so it
    // is printed beside the JIT one rather than left to be reconstructed.
    //
    // `dL/dB/dG` are the counter DELTAS across the n=100 timed window. min-of-41
    // only removes a cost paid *sometimes*; if this harness makes EVERY sample
    // pay a compile or a bridge, the minimum removes nothing. #91's law makes
    // that unmissable — one extra bridge inside the window is 200 guard
    // failures — so a non-zero dB or a large dG is the whole explanation.

    let flat = flat_schema();
    let nested: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();

    // (label, source, schema, per_row) — per_row 0 means the flat shape.
    let ladder: Vec<(&str, &str, &Schema, i64)> = vec![
        ("price", "price", &flat, 0),
        ("price + qty", "price + qty", &flat, 0),
        ("price + qty * 2", "price + qty * 2", &flat, 0),
        ("+ price * 3", "price + qty * 2 + price * 3", &flat, 0),
        (
            "+ qty * 4",
            "price + qty * 2 + price * 3 + qty * 4",
            &flat,
            0,
        ),
        (
            "+ price * 5",
            "price + qty * 2 + price * 3 + qty * 4 + price * 5",
            &flat,
            0,
        ),
        (
            "price >= 100 && qty < 50",
            "price >= 100 && qty < 50",
            &flat,
            0,
        ),
        (
            "&& price < 300 && qty > 2",
            "price >= 100 && qty < 50 && price < 300 && qty > 2",
            &flat,
            0,
        ),
        (
            "all(i, i.price > 10) pr=1",
            "items.all(i, i.price > 10)",
            &nested,
            1,
        ),
        (
            "all(i, i.price > 10) pr=2",
            "items.all(i, i.price > 10)",
            &nested,
            2,
        ),
        (
            "all(i, i.price > 10) pr=3",
            "items.all(i, i.price > 10)",
            &nested,
            3,
        ),
        (
            "all(i, i.price > 10) pr=8",
            "items.all(i, i.price > 10)",
            &nested,
            8,
        ),
    ];

    for (label, src, schema, per_row) in ladder {
        let lowered = lower(src, schema);
        let mut warm_at = [0.0f64; 3];
        let mut clean_at = [0.0f64; 3];
        let mut window_delta = (0usize, 0usize, 0usize);
        // `loops` was captured at n=10 only, so the table could not say whether
        // the OTHER two columns were even measuring a compiled artifact.
        let mut loops_at = [0usize; 3];
        let mut never_100 = 0.0f64;
        let mut stats = None;

        for (slot, n) in [10usize, 100, 1_000].into_iter().enumerate() {
            let (lens, offsets, elems);
            // The flat ladder varies how many distinct fields the expression
            // touches, so the column count has to come from the lowering rather
            // than be assumed: `price` alone binds one slot, `price + qty` two,
            // and `eval_batch_sum_f` asserts the two agree.
            // Byte-for-byte the generator `flat_columns` uses, extended for a
            // third-or-later slot. Probe B and rca88 both feed exactly these
            // values, and Probe C initially did not — which is the one input
            // difference between two measurements of the same expression at the
            // same n that disagreed 5328 vs 17257 ns.
            let flat_data: Vec<Vec<i64>> = (0..lowered.slots.len())
                .map(|c| match c {
                    0 => (0..n as i64).map(|i| (i * 37) % 200).collect(),
                    1 => (0..n as i64).map(|i| (i * 11) % 100).collect(),
                    _ => (0..n as i64).map(|i| (i * (37 + c as i64)) % 200).collect(),
                })
                .collect();
            let columns: Vec<Column> = if per_row == 0 {
                flat_data.iter().map(|v| Column::Int(v)).collect()
            } else {
                lens = vec![per_row; n];
                let mut off = Vec::with_capacity(n);
                let mut total = 0i64;
                for &l in &lens {
                    off.push(total);
                    total += l;
                }
                offsets = off;
                elems = (0..total.max(1)).map(|k| (k * 7) % 40).collect::<Vec<_>>();
                vec![
                    Column::Int(&lens),
                    Column::Int(&offsets),
                    Column::Int(&elems),
                ]
            };

            reset_persistent_state();
            reset_jit_stats();

            // The control this probe was missing. `eval_batch_sum_f` returns an
            // Option, and a refused batch re-runs every row in the OTHER
            // evaluator (#54) — which would inflate a timing by exactly the
            // kind of factor seen here while every counter stayed plausible.
            // Assert the tiers agree BEFORE trusting any number below.
            let want = clean_batch_sum_f(&lowered, &columns, n);
            let got = eval_batch_sum_f(&lowered, &columns, n, THRESHOLD);
            assert_eq!(
                got, want,
                "{label} @ n={n}: jit tier disagrees with clean tier \
                 (None means the batch refused and fell back)"
            );

            for _ in 0..64 {
                black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
            }
            if slot == 0 {
                stats = Some(jit_stats());
            }
            loops_at[slot] = jit_stats().loops_compiled;
            // Probe B's never-compiles arm, brought inside Probe C's harness at
            // the one n the two probes disagree about. Same driver-pool keying
            // (`threshold` is part of the key), so it cannot reach the compiled
            // loop. If `warm/nvr@100` is ~1.0 the "warm" column at n=100 is not
            // running compiled code at all, and the 4.6x is a tier difference
            // rather than a timing artifact.
            if slot == 1 {
                for _ in 0..64 {
                    black_box(eval_batch_sum_f(&lowered, &columns, n, NEVER));
                }
                let mut best = f64::MAX;
                for _ in 0..41 {
                    let t = std::time::Instant::now();
                    for _ in 0..(20_000 / n).max(1) {
                        black_box(eval_batch_sum_f(&lowered, &columns, n, NEVER));
                    }
                    best = best.min(t.elapsed().as_nanos() as f64 / (20_000 / n).max(1) as f64);
                }
                never_100 = best;
            }

            let reps = (20_000 / n).max(1);
            let mut best_clean = f64::MAX;
            let mut best_warm = f64::MAX;
            let before = jit_stats();
            for _ in 0..41 {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    black_box(clean_batch_sum_f(&lowered, &columns, n));
                }
                best_clean = best_clean.min(t.elapsed().as_nanos() as f64 / reps as f64);

                let t = std::time::Instant::now();
                for _ in 0..reps {
                    black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
                }
                best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
            }
            let after = jit_stats();
            warm_at[slot] = best_warm;
            clean_at[slot] = best_clean;
            if slot == 1 {
                window_delta = (
                    after.loops_compiled - before.loops_compiled,
                    after.bridges_compiled - before.bridges_compiled,
                    after.guard_failures - before.guard_failures,
                );
            }
        }

        let s = stats.expect("stats captured at n=10");
        // Three points, so the per-call and per-row terms can be SEPARATED
        // instead of assumed. Fit over the outer two (n=10, n=1000); `warm
        // n=100` is then a residual check that the model is linear at all.
        let per_row = (warm_at[2] - warm_at[0]) / 990.0;
        let fixed = warm_at[0] - 10.0 * per_row;
        let predicted_100 = fixed + 100.0 * per_row;
        // Same two-point fit on the arm that never enters majit at all.
        let clean_per_row = (clean_at[2] - clean_at[0]) / 990.0;
        let clean_predicted_100 = clean_at[0] - 10.0 * clean_per_row + 100.0 * clean_per_row;
        println!(
            "{label:<34} {:>6} {:>5} {:>8} {:>8} {:>10.1} {:>11.1} {:>11.1} \
             {:>11.1} {:>10.1} {:>9.1} {:>8.2} {:>11.1} {:>11.1} {:>9.2} {:>5} {:>5} {:>8} \
             {:>5} {:>6} {:>5} {:>13.3}",
            s.loops_compiled,
            s.bridges_compiled,
            s.trace_ops_before,
            s.trace_ops_after,
            clean_at[0],
            warm_at[0],
            warm_at[1],
            warm_at[2],
            fixed,
            per_row,
            warm_at[1] / predicted_100,
            clean_at[1],
            clean_at[2],
            clean_at[1] / clean_predicted_100,
            window_delta.0,
            window_delta.1,
            window_delta.2,
            loops_at[0],
            loops_at[1],
            loops_at[2],
            warm_at[1] / never_100,
        );
    }
    println!(
        "  Read `warm n=10` against ops_post and against loops. Tracking ops_post \
         means the cost is trace-length-driven;"
    );
    println!(
        "  a step at loops 1->2 with ops_post flat means it is per-ARTIFACT. \
         `warm n=100` ~= `warm n=10` confirms the column is the fixed term."
    );
    println!(
        "  cLin@100 is the same residual on the arm that never enters majit. \
         cLin@100 ~= lin@100 => the n=100 defect is the HARNESS, not majit."
    );
    println!(
        "  dL/dB/dG are counter deltas across the n=100 timed window; any dB>0 \
         means min-of-41 removed nothing (one bridge = 200 gfails, #91)."
    );
    println!(
        "  warm/nvr@100 ~= 1.0 means the n=100 column never ran compiled code — \
         a TIER difference, not a timing artifact."
    );
}

/// Probe F. Probe C's `warm n=100` column reads ~4.4x Probe B's, on the same
/// expression, the same n and the same binary — and `L@100` says the loop IS
/// compiled in both while `warm/nvr@100` ~= 1.1-1.5 says Probe C's calls run at
/// the never-compiles rate. So a compiled artifact stops being entered, and
/// `loops_compiled` cannot see it (it counts compiles, not liveness — the gate
/// trap #88 already names).
///
/// ⛔ Warmup count is REFUTED as the difference. A first form of this probe
/// swept the compiled arm's warmup calls 1 -> 256 at n=100 and found no cliff:
/// `warm/nvr` = 1.54 / 1.63 / 1.59 / 1.49 / 1.72 / 1.68 / 1.99 / 2.13 / 2.51.
/// It is monotone-ish and never approaches Probe B's 0.24. **At warmup=1 —
/// Probe B's own warmup — the compiled arm still costs 33 µs against Probe B's
/// 4.6 µs.** So the slow reading is the one that reproduces in isolation, and
/// what needs explaining is why Probe B is FAST.
///
/// What Probe B has that a bare compiled-arm loop does not, all of it inside
/// the round: a 64-call **never-compiles warmup**, and a **clean** and a
/// **never** arm timed between successive compiled batches. This adds them back
/// one at a time. The last row is Probe B's structure exactly, so it must
/// reproduce Probe B's number or the reconstruction is wrong and says nothing.
///
/// Only the compiled arm's own `Instant` is read in every row, so the added
/// arms cannot enter the number being compared.
fn reconstruct_probe_b() {
    println!("\nProbe F — rebuild Probe B's structure around the compiled arm (n=100)");
    println!(
        "{:<34} {:>12} {:>12} {:>10} {:>7} {:>6} {:>13} {:>10}",
        "structure", "warm ns", "never ns", "warm/nvr", "loops", "brdg", "gfails/call", "load"
    );

    let schema = flat_schema();
    let lowered = lower("price + qty * 2", &schema);
    const N: usize = 100;
    let (price, qty) = flat_columns(N);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];
    let reps = 200;

    // (label, compiled warmup calls, warm the never arm before timing, run
    // clean in-round, run never in-round). Row 4 is Probe B exactly; row 6 is
    // Probe C exactly. Everything between them is the bisect.
    let structures = [
        ("compiled arm alone", 1usize, false, false, false),
        ("+ never warmup (64)", 1, true, false, false),
        ("+ clean in-round", 1, true, true, false),
        ("+ never in-round  (= Probe B)", 1, true, true, true),
        ("Probe B round, warmup 66", 66, true, true, true),
        ("clean-only round, warmup 66 (= C)", 66, false, true, false),
    ];

    for (label, warmups, never_warmup, clean_in_round, never_in_round) in structures {
        reset_persistent_state();
        reset_jit_stats();
        for _ in 0..warmups {
            black_box(eval_batch_sum_f(&lowered, &columns, N, THRESHOLD));
        }
        let s = jit_stats();
        if never_warmup {
            for _ in 0..64 {
                black_box(eval_batch_sum_f(&lowered, &columns, N, NEVER));
            }
        }

        let mut best_warm = f64::MAX;
        let mut best_never = f64::MAX;
        let before = jit_stats();
        let mut warm_calls = 0usize;
        for _ in 0..41 {
            if clean_in_round {
                for _ in 0..reps {
                    black_box(clean_batch_sum_f(&lowered, &columns, N));
                }
            }
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(&lowered, &columns, N, THRESHOLD));
            }
            best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
            warm_calls += reps;
            if never_in_round {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    black_box(eval_batch_sum_f(&lowered, &columns, N, NEVER));
                }
                best_never = best_never.min(t.elapsed().as_nanos() as f64 / reps as f64);
            }
        }
        let after = jit_stats();

        // The never arm is only timed where the structure calls for it; without
        // this the unmeasured rows print `f64::MAX` and a ratio of 0.000, which
        // reads like a measurement rather than an absence.
        let (never_col, ratio_col) = if never_in_round {
            (
                format!("{best_never:.1}"),
                format!("{:.3}", best_warm / best_never),
            )
        } else {
            ("-".to_string(), "-".to_string())
        };
        println!(
            "{label:<34} {best_warm:>12.1} {never_col:>12} {ratio_col:>10} {:>7} {:>6} {:>13.3} \
             {:>10}",
            s.loops_compiled,
            after.bridges_compiled - before.bridges_compiled,
            (after.guard_failures - before.guard_failures) as f64 / warm_calls as f64,
            loadavg(),
        );
    }
    println!(
        "  The last row must reproduce Probe B's jit(compiled) at n=100 (~4.6 us) \
         or the reconstruction is void."
    );
}

/// Probe G. Probe F terminated the harness bisect on ONE variable: the number
/// of compiled calls made before the timing window. At n=100, warmup 1 gives
/// ~5-6 µs and warmup 66 gives ~23-26 µs — `warm/nvr` 0.30 -> 1.15, i.e. the
/// artifact stops paying for itself — and it is independent of whether the
/// clean or never arm is interleaved. So the compiled artifact DEGRADES with
/// use, and #88's "fixed per-call cost" is conditioned on a warmup nobody was
/// varying deliberately.
///
/// ⛔ An earlier sweep of this same axis found no cliff and was WRONG: it read
/// `gfails/call` from an extra 100-call loop placed BETWEEN the warmup and the
/// timing, so every row was really warmup+100 and the whole axis sat past the
/// transition. Nothing here runs between the warmup and the clock.
///
/// The sweep is run at all three n because the degradation is not monotone in
/// rows executed: at warmup 66 the n=10 and n=1000 columns agree with Probe B
/// and only n=100 does not.
fn warmup_degradation() {
    println!("\nProbe G — how many compiled calls until the artifact stops paying?");
    println!(
        "{:>8} {:>10} {:>12} {:>12} {:>10} {:>5} {:>6} {:>13} {:>7} {:>9}",
        "n",
        "warmups",
        "warm ns",
        "never ns",
        "warm/nvr",
        "dL",
        "brdg",
        "gfails/call",
        "aborts",
        "abrt/call"
    );

    let schema = flat_schema();
    let lowered = lower("price + qty * 2", &schema);

    for n in [10usize, 100, 1_000] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];
        let reps = (20_000 / n).max(1);

        // One denominator per n: the never-compiles arm has no artifact to
        // degrade, so it is the fixed yardstick this column is read against.
        reset_persistent_state();
        for _ in 0..64 {
            black_box(eval_batch_sum_f(&lowered, &columns, n, NEVER));
        }
        let mut best_never = f64::MAX;
        for _ in 0..41 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(&lowered, &columns, n, NEVER));
            }
            best_never = best_never.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        for warmups in [
            1usize, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512,
        ] {
            reset_persistent_state();
            reset_jit_stats();
            for _ in 0..warmups {
                black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
            }
            let before = jit_stats();
            let mut best_warm = f64::MAX;
            for _ in 0..41 {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
                }
                best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
            }
            let after = jit_stats();
            println!(
                "{n:>8} {warmups:>10} {best_warm:>12.1} {best_never:>12.1} {:>10.3} {:>5} \
                 {:>6} {:>13.3} {:>7} {:>9.3}",
                best_warm / best_never,
                after.loops_compiled - before.loops_compiled,
                after.bridges_compiled - before.bridges_compiled,
                (after.guard_failures - before.guard_failures) as f64 / (41 * reps) as f64,
                // `loops_aborted` is the discriminator for "re-arms tracing and
                // never compiles": an artifact that stopped being entered leaves
                // the back edge to arm tracing on every call, and a walk that
                // cannot close shows up here and nowhere else. It is already
                // exported and has never been read on this axis.
                after.loops_aborted - before.loops_aborted,
                (after.loops_aborted - before.loops_aborted) as f64 / (41 * reps) as f64,
            );
        }
    }
    println!(
        "  warm/nvr crossing 1.0 is the artifact ceasing to pay for itself. \
         Nothing runs between the warmup and the clock."
    );
}

/// Probe H. The standing hypothesis for Probe G's degradation (team-lead) is
/// **green-key chain growth**: #106 leaves `max_age = 0` so `alive_loops` never
/// prunes and `gc_cells` has no production caller, #117 has cells chaining off
/// one bucket with `lookup_chain_with_key` walking the chain, and #90 builds
/// and hashes the key on **every back edge**. Composed: each back edge pays a
/// chain walk whose length grows with cumulative compiled population. That
/// predicts a per-ROW cost, growth with cumulative activity, invisibility to
/// every exported counter (`get_stats` reads chain HEADS), backend-neutrality,
/// and — because chains form only on hash collision — a step at large n where
/// one collision is amplified n-fold.
///
/// The clean external test: **hold the target's own warmup fixed and vary the
/// population that shares its driver.** If cost tracks cumulative population,
/// the transition moves earlier as `pop` grows; if it tracks calls-since-
/// compile only, the `pop` columns are flat and the hypothesis is wrong.
///
/// ⚠ Probe A already swept population and found FLAT — but at **n=10**, the one
/// size that does not degrade at any warmup, and at a single shallow warmup.
/// It could not have seen this. That is why the sweep is two-dimensional here:
/// a `pop` axis crossed with the `warmups` axis, at the n where the step is
/// sharp.
///
/// Fillers are Probe A's: structurally identical so they lower to the same
/// register counts and land in the same driver — asserted, not assumed, because
/// a filler in its own driver would leave the population at 1 and print
/// "flat" for a reason unrelated to the hypothesis.
fn population_times_warmup() {
    println!(
        "\nProbe H — does the degradation track cumulative POPULATION or calls-since-compile?"
    );
    println!(
        "{:>8} {:>6} {:>10} {:>12} {:>12} {:>10} {:>7} {:>9}",
        "n", "pop", "warmups", "warm ns", "never ns", "warm/nvr", "loops", "load"
    );

    let schema = flat_schema();
    let target = lower("price + qty * 2", &schema);

    for n in [1_000usize, 100] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];
        let reps = (20_000 / n).max(1);

        reset_persistent_state();
        for _ in 0..64 {
            black_box(eval_batch_sum_f(&target, &columns, n, NEVER));
        }
        let mut best_never = f64::MAX;
        for _ in 0..41 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(&target, &columns, n, NEVER));
            }
            best_never = best_never.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        for pop in [1usize, 8, 32, 128] {
            for warmups in [8usize, 64, 128, 192, 256] {
                reset_persistent_state();
                reset_jit_stats();

                // Population first, so the target's own warmup is the only
                // thing the `warmups` axis moves.
                let fillers: Vec<LoweredF> = (1..pop)
                    .map(|k| lower(&format!("price + qty * {}", k + 2), &schema))
                    .collect();
                for f in &fillers {
                    assert_eq!(
                        (f.num_int_regs, f.num_float_regs),
                        (target.num_int_regs, target.num_float_regs),
                        "filler lowers to a different shape than the target, so it \
                         lands in a different driver and the population never grows"
                    );
                    black_box(eval_batch_sum_f(f, &columns, n, THRESHOLD));
                }

                for _ in 0..warmups {
                    black_box(eval_batch_sum_f(&target, &columns, n, THRESHOLD));
                }
                // Non-vacuity: must read `pop`, or the fillers did not compile
                // into this driver and the population axis is a no-op.
                let loops = jit_stats().loops_compiled;

                let mut best_warm = f64::MAX;
                for _ in 0..41 {
                    let t = std::time::Instant::now();
                    for _ in 0..reps {
                        black_box(eval_batch_sum_f(&target, &columns, n, THRESHOLD));
                    }
                    best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
                }

                println!(
                    "{n:>8} {pop:>6} {warmups:>10} {best_warm:>12.1} {best_never:>12.1} \
                     {:>10.3} {loops:>7} {:>9}",
                    best_warm / best_never,
                    loadavg(),
                );
            }
        }
    }
    println!(
        "  `loops` must equal `pop` or the population axis is vacuous. If the \
         transition moves EARLIER as pop grows, it is cumulative population;"
    );
    println!(
        "  if the pop columns are flat at every warmup, calls-since-compile is \
         the axis and chain growth is refuted."
    );
}

/// Probe E. Probe C measures ~600 ns per inner ELEMENT on `pr=8` at n=10;
/// rca88 measures **2.275 ns per element** on what is written down as the same
/// `pr=8` shape at 320 000 elements. That is a 260x discrepancy, and the
/// standing instinct is to model it — a fixed per-call term amortizing, a
/// warmup effect, a cache argument.
///
/// ⭐ Before explaining a discrepancy, check whether the two measurements are of
/// the same thing. "Same shape" here means the same SOURCE and the same
/// `per_row`; it does not mean the same artifact. `loops_compiled`,
/// `bridges_compiled` and the trace op counts are already exported and already
/// distinguish "the inner loop was cut into its own artifact" from "the inner
/// loop was inlined into the outer trace" (#76's census). If those counts
/// differ between the two sizes, the two runs compiled different code and there
/// is no 260x to explain — it dissolves with no model at all.
///
/// This is a counter read, not a new timing design: the ns columns are here
/// only so the two rows can be recognised as the ones being reconciled.
fn same_shape_across_size() {
    println!("\nProbe E — is `pr=8` at n=10 the same ARTIFACT as at n=40000?");
    println!(
        "{:>8} {:>10} {:>7} {:>6} {:>9} {:>10} {:>13} {:>12} {:>11} {:>11}",
        "n",
        "elements",
        "loops",
        "brdg",
        "ops_pre",
        "ops_post",
        "gfails/call",
        "warm ns/call",
        "ns/elem",
        "warm/clean"
    );

    let nested: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &nested);
    const PER_ROW: i64 = 8;

    for n in [10usize, 40_000] {
        let lens = vec![PER_ROW; n];
        let mut off = Vec::with_capacity(n);
        let mut total = 0i64;
        for &l in &lens {
            off.push(total);
            total += l;
        }
        let elems = (0..total.max(1)).map(|k| (k * 7) % 40).collect::<Vec<_>>();
        let columns = vec![Column::Int(&lens), Column::Int(&off), Column::Int(&elems)];

        reset_persistent_state();
        reset_jit_stats();
        let want = clean_batch_sum_f(&lowered, &columns, n);
        let got = eval_batch_sum_f(&lowered, &columns, n, THRESHOLD);
        assert_eq!(
            got, want,
            "pr=8 @ n={n}: jit tier disagrees with clean tier"
        );

        let warm_calls = (200_000 / n).max(4);
        for _ in 0..warm_calls {
            black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
        }
        let s = jit_stats();

        reset_jit_stats();
        for _ in 0..100 {
            black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
        }
        let gfails = jit_stats().guard_failures as f64 / 100.0;

        let reps = (20_000 / n).max(1);
        let mut best_clean = f64::MAX;
        let mut best_warm = f64::MAX;
        for _ in 0..41 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(clean_batch_sum_f(&lowered, &columns, n));
            }
            best_clean = best_clean.min(t.elapsed().as_nanos() as f64 / reps as f64);

            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(&lowered, &columns, n, THRESHOLD));
            }
            best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        println!(
            "{n:>8} {total:>10} {:>7} {:>6} {:>9} {:>10} {gfails:>13.2} {best_warm:>12.1} \
             {:>11.3} {:>11.3}",
            s.loops_compiled,
            s.bridges_compiled,
            s.trace_ops_before,
            s.trace_ops_after,
            best_warm / total as f64,
            best_warm / best_clean,
        );
    }
    println!(
        "  If loops/brdg/ops differ between the rows, the two measurements are \
         of DIFFERENT artifacts and the 260x needs no model."
    );
}

/// #111's instrument. Allocations per row on the never-compiles arm, which is
/// exact and load-independent — and here the control arm genuinely varies:
/// clean cel is ~0 allocations/row (a 320 000-element batch allocates 4 in
/// total) against 3/row if #90 fires. Orders of magnitude of dynamic range,
/// unlike the degenerate allocation control this investigation started with.
///
/// The read that licenses pointing it at this arm: `jit_interp/mod.rs:2388`
/// emits the green key as an ARGUMENT to `back_edge_structured`, so Rust
/// evaluates it before the call and the counter check inside cannot
/// short-circuit it. `green_key_expr` (`:2060-2068`) builds a `vec![..]` and
/// unzips it into two more `Vec`s — three allocations, on every back edge,
/// including this arm where nothing is ever compiled.
///
/// Decisive in both directions: 0/row refutes #90 for this arm regardless of
/// what the code reads like.
fn allocations_per_row(label: &str, lowered: &LoweredF) {
    println!("\n#111 — allocations per row, by arm ({label})");
    println!(
        "{:>8} {:>12} {:>12} {:>14} {:>12} {:>12} {:>14}",
        "n", "clean", "compiled", "never-compiles", "clean/row", "comp/row", "nevercomp/row"
    );

    for n in [10usize, 100, 1_000] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];

        // Warm each arm on its own driver first, so what the window sees is
        // steady-state execution and not compilation.
        reset_persistent_state();
        for _ in 0..64 {
            black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
        }
        for _ in 0..64 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }

        let (_, clean_a) = metered(|| black_box(clean_batch_sum_f(lowered, &columns, n)));
        let (_, comp_a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD)));
        let (_, never_a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, NEVER)));

        println!(
            "{n:>8} {clean_a:>12} {comp_a:>12} {never_a:>14} {:>12.3} {:>12.3} {:>14.3}",
            clean_a as f64 / n as f64,
            comp_a as f64 / n as f64,
            never_a as f64 / n as f64,
        );
    }
    println!(
        "  3.000 nevercomp/row would be green_key_expr's vec! + two unzip Vecs (#90). \
         0.000 refutes #90 for this arm."
    );
}

/// Probe D. The exit path co-varies with n — non-guard below n=1000, a real
/// guard failure at and above n=10000 — and the wall is present exactly where
/// the exit is NOT a guard failure. Inverted, that reads: **finishing normally
/// costs more than failing a guard**, which is backwards from every intuition
/// about JIT exits and names the finish/jump-exit path as the suspect.
///
/// That is a correlation across n. This converts it to a controlled experiment:
/// hold n at 10 and change only the SHAPE so the artifact must leave by a real
/// guard failure, then re-measure.
///
/// * cost collapses when guard failures appear => the finish/jump-exit path is
///   the cost, and that is a pinpoint RCA from outside majit;
/// * cost survives => the cost is in ENTRY, the exit-path correlation is a
///   passenger, and the search redirects.
///
/// Trip-count variation is the lever because #91 established it produces guard
/// failures deterministically on this shape, on both backends. `ns/elem` is
/// reported beside `ns/call` because the variants do not do identical element
/// work (72-85 elements against the constant shape's 80), so a raw per-call
/// comparison would be reading a work difference.
fn forced_exit_path() {
    println!("\nProbe D — force a guard exit at n=10, hold everything else");
    println!(
        "{:<26} {:>7} {:>9} {:>11} {:>11} {:>10} {:>11}",
        "shape (n=10)", "elems", "loops", "gfails/call", "warm ns/call", "ns/elem", "warm/clean"
    );

    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &schema);
    const N: usize = 10;

    let variants: Vec<(&str, Vec<i64>)> = vec![
        ("constant 8", vec![8i64; N]),
        (
            "alternating 8/9",
            (0..N).map(|i| if i % 2 == 0 { 8 } else { 9 }).collect(),
        ),
        ("cycle 4..12", (0..N).map(|i| 4 + (i as i64 % 9)).collect()),
        (
            "spread 1..40",
            (0..N).map(|i| 1 + (i as i64 * 37) % 40).collect(),
        ),
    ];

    for (label, lens) in variants {
        let mut off = Vec::with_capacity(N);
        let mut total = 0i64;
        for &l in &lens {
            off.push(total);
            total += l;
        }
        let elems = (0..total.max(1)).map(|k| (k * 7) % 40).collect::<Vec<_>>();
        let columns = vec![Column::Int(&lens), Column::Int(&off), Column::Int(&elems)];

        reset_persistent_state();
        reset_jit_stats();
        let want = clean_batch_sum_f(&lowered, &columns, N);
        let got = eval_batch_sum_f(&lowered, &columns, N, THRESHOLD);
        assert_eq!(got, want, "{label}: jit tier disagrees with clean tier");

        for _ in 0..512 {
            black_box(eval_batch_sum_f(&lowered, &columns, N, THRESHOLD));
        }
        let loops = jit_stats().loops_compiled;

        let reps = 2_000;
        let mut best_clean = f64::MAX;
        let mut best_warm = f64::MAX;
        for _ in 0..41 {
            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(clean_batch_sum_f(&lowered, &columns, N));
            }
            best_clean = best_clean.min(t.elapsed().as_nanos() as f64 / reps as f64);

            let t = std::time::Instant::now();
            for _ in 0..reps {
                black_box(eval_batch_sum_f(&lowered, &columns, N, THRESHOLD));
            }
            best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        // The manipulation check: this column must MOVE across the variants, or
        // the probe never forced the exit path it claims to have forced.
        reset_jit_stats();
        for _ in 0..100 {
            black_box(eval_batch_sum_f(&lowered, &columns, N, THRESHOLD));
        }
        let gfails = jit_stats().guard_failures as f64 / 100.0;

        println!(
            "{label:<26} {total:>7} {loops:>9} {gfails:>11.2} {best_warm:>11.1} {:>10.1} {:>11.2}",
            best_warm / total as f64,
            best_warm / best_clean,
        );
    }
    println!(
        "  gfails/call must MOVE across rows or nothing was forced. If ns/elem \
         FALLS as gfails/call rises, the finish path is the cost."
    );
}

/// Probe J. Probe G's three `n` columns are not three phenomena — they are the
/// same curve sampled at three different `reps` per round (2000 / 200 / 20).
/// min-of-41 reports the FASTEST round, so a permanent step at some fixed call
/// index is visible only while some whole round still lies before it. That
/// predicts:
///
/// * n=1000 (reps 20): a sharp step, because a round is 20 calls wide;
/// * n=100 (reps 200): a smooth monotone ramp, because each round straddles the
///   step and the pre-step fraction shrinks with warmup;
/// * n=10 (reps 2000): perfectly flat AT THE POST-STEP VALUE, because no round
///   is ever mostly pre-step.
///
/// All three are Probe G's actual columns. So the instrument, not the artifact,
/// may be what differs across n — and the way to settle it is to delete the
/// instrument: time each call individually, against its own call index, with no
/// min and no aggregation across rounds.
///
/// ⛔ This must not be read as "Probe G was wrong". Probe G measured what it
/// says it measured; the question is whether `n` or `reps` is the axis its rows
/// are indexed by, and those were confounded because `reps = 20_000 / n`.
fn cost_by_call_index(label: &str, lowered: &LoweredF) {
    println!("\nProbe J — {label}: per-call cost against CALL INDEX, no min, no rounds");

    const CALLS: usize = 460;
    const BUCKET: usize = 20;

    for n in [10usize, 100, 1_000] {
        println!(
            "\n  n={n}  {:>12} {:>11} {:>11} {:>11} {:>6} {:>8}",
            "calls", "mean ns", "min ns", "max ns", "brdg", "gfails"
        );
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];

        reset_persistent_state();
        reset_jit_stats();
        let mut ns = Vec::with_capacity(CALLS);
        let mut marks = Vec::with_capacity(CALLS / BUCKET + 1);
        for k in 0..CALLS {
            let t = std::time::Instant::now();
            black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
            ns.push(t.elapsed().as_nanos() as f64);
            // Read the counters only at bucket boundaries: `jit_stats` does not
            // advance the call counter, but keeping it out of the inner path
            // keeps the timed sequence identical to an uninstrumented one.
            if (k + 1) % BUCKET == 0 {
                let s = jit_stats();
                marks.push((s.bridges_compiled, s.guard_failures));
            }
        }

        let mut baseline = f64::MAX;
        let mut step_at: Option<usize> = None;
        for (b, chunk) in ns.chunks(BUCKET).enumerate() {
            let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let lo = chunk.iter().cloned().fold(f64::MAX, f64::min);
            let hi = chunk.iter().cloned().fold(0.0, f64::max);
            if b < 3 {
                baseline = baseline.min(mean);
            } else if step_at.is_none() && mean > 4.0 * baseline {
                step_at = Some(b * BUCKET + 1);
            }
            let (brdg, gf) = marks.get(b).copied().unwrap_or((0, 0));
            println!(
                "       {:>12} {mean:>11.1} {lo:>11.1} {hi:>11.1} {brdg:>6} {gf:>8}",
                format!("{}-{}", b * BUCKET + 1, (b + 1) * BUCKET),
            );
        }
        match step_at {
            Some(c) => println!("       => 4x step first seen in the bucket starting at call {c}"),
            None => println!("       => no 4x step anywhere in {CALLS} calls"),
        }
    }
    println!(
        "  A step at the SAME call index at every n means Probe G's rows are indexed by \
         `reps`, not by `n`."
    );
}

/// Probe K. The external lever on Probe J's step. `THRESHOLD` is the back-edge
/// count at which the loop compiles, and back edges are per ROW — so at n rows
/// per call the loop compiles on call `ceil(THRESHOLD / n)`. Everything that
/// happens on the guard-failure schedule is then anchored to THAT call, not to
/// call 1: #91's law puts the first bridge `trace_eagerness / gfails-per-call`
/// calls later.
///
/// So raising `THRESHOLD` must slide the step by exactly the number of calls it
/// delays compilation by, and nothing else about the artifact changes.
///
/// * step moves 1:1 with the compile call => it is anchored to CALLS SINCE
///   COMPILE, and the schedule #122 eliminated as a passenger is back in play
///   as the clock the step runs on;
/// * step stays at a fixed absolute call index => it is anchored to process
///   history, not to the artifact, and the schedule is ruled out a second and
///   independent time.
///
/// n=10 is the sweep size because it is the only one where `THRESHOLD` can be
/// pushed past a single call's worth of back edges without also changing the
/// per-call work.
fn step_moves_with_threshold(label: &str, lowered: &LoweredF) {
    println!("\nProbe K — {label}: does Probe J's step follow the COMPILE call? (n=10)");
    println!(
        "{:>11} {:>14} {:>11} {:>12} {:>13} {:>11} {:>11}",
        "threshold",
        "compiles@call",
        "step@call",
        "step - cmp",
        "pre-step ns",
        "post ns",
        "post/pre"
    );

    const N: usize = 10;
    const CALLS: usize = 700;
    const BUCKET: usize = 20;
    let (price, qty) = flat_columns(N);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];

    for threshold in [8u32, 800, 1_600, 3_000] {
        reset_persistent_state();
        reset_jit_stats();
        let mut ns = Vec::with_capacity(CALLS);
        // The call on which `loops_compiled` first moves — measured, not
        // computed from `THRESHOLD / n`, so a different back-edge accounting
        // shows up as a disagreement instead of being assumed away.
        let mut compiled_at = 0usize;
        for k in 0..CALLS {
            let t = std::time::Instant::now();
            black_box(eval_batch_sum_f(lowered, &columns, N, threshold));
            ns.push(t.elapsed().as_nanos() as f64);
            if compiled_at == 0 && jit_stats().loops_compiled > 0 {
                compiled_at = k + 1;
            }
        }

        let means: Vec<f64> = ns
            .chunks(BUCKET)
            .map(|c| c.iter().sum::<f64>() / c.len() as f64)
            .collect();
        // Baseline from the buckets after the compile and before any step, so a
        // late compile does not put its own tracing cost in the baseline.
        let first = (compiled_at / BUCKET) + 1;
        let baseline = means[first..(first + 3).min(means.len())]
            .iter()
            .cloned()
            .fold(f64::MAX, f64::min);
        let step_b = means
            .iter()
            .enumerate()
            .skip(first + 3)
            .find(|(_, &m)| m > 4.0 * baseline)
            .map(|(b, _)| b);
        let post = means[means.len().saturating_sub(3)..].iter().sum::<f64>() / 3.0;

        let (step_at, delta) = match step_b {
            Some(b) => {
                let c = b * BUCKET + 1;
                (c.to_string(), (c as i64 - compiled_at as i64).to_string())
            }
            None => ("none".to_string(), "-".to_string()),
        };
        println!(
            "{threshold:>11} {compiled_at:>14} {step_at:>11} {delta:>12} \
             {baseline:>13.1} {post:>11.1} {:>11.2}",
            post / baseline,
        );
    }
    println!(
        "  `step - cmp` constant across thresholds => the step is CALLS SINCE COMPILE. \
         `step@call` constant => it is process history."
    );
}

/// Probe L. Probes J and K established **when** — the first guard bridge, at
/// `trace_eagerness` calls after the loop compiles. This asks **what runs
/// afterwards**, on the axis they fixed, and it needs no held file.
///
/// The discriminator is #111's allocation instrument, whose two reference
/// points are already measured and which is exact and load-independent: the
/// compiled artifact allocates **0 per row** (69 per call, flat in n) and the
/// never-compiles portal allocates **~12 per row**. So across call 200:
///
/// * allocations/row rising toward 12 ⇒ the rows are being **interpreted** and
///   the artifact is not executing;
/// * allocations/row staying ~0 ⇒ the rows are **not** being interpreted, so
///   the penalty lives inside compiled execution.
///
/// ⚠ Read the second case exactly as far as it goes: ~0 licenses *"not
/// interpretation"*, **not** *"executes cleanly"*. An artifact that is entered
/// and then bails per row through a path that does not allocate would also read
/// ~0, and this probe cannot separate that from a slower compiled body.
///
/// `trace_ops_before` / `trace_ops_after` ride along because #122 names them as
/// the next cheap external read and they have never been sampled across call
/// 200. Buckets are Probe J's (20 calls) so the two tables index alike.
fn execution_by_allocation(label: &str, lowered: &LoweredF) {
    println!("\nProbe L — {label}: WHAT executes after the first bridge? (allocations/row)");

    const CALLS: usize = 460;
    const BUCKET: usize = 20;

    for n in [10usize, 100, 1_000] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];

        // The two yardsticks, measured here rather than quoted, so the row is
        // read against this binary and this n instead of #111's table.
        reset_persistent_state();
        for _ in 0..8 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }
        let (_, never_a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, NEVER)));
        let (_, clean_a) = metered(|| black_box(clean_batch_sum_f(lowered, &columns, n)));

        println!(
            "\n  n={n}  never-compiles {:.3} allocs/row, clean {:.3} allocs/row",
            never_a as f64 / n as f64,
            clean_a as f64 / n as f64,
        );
        println!(
            "  {:>12} {:>12} {:>12} {:>6} {:>8} {:>9} {:>9}",
            "calls", "allocs/call", "allocs/row", "brdg", "gfails", "ops_pre", "ops_post"
        );

        reset_persistent_state();
        reset_jit_stats();
        let mut allocs: Vec<u64> = Vec::with_capacity(CALLS);
        let mut marks = Vec::with_capacity(CALLS / BUCKET);
        for k in 0..CALLS {
            let (_, a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD)));
            allocs.push(a);
            if (k + 1) % BUCKET == 0 {
                let s = jit_stats();
                marks.push((
                    k + 1,
                    s.bridges_compiled,
                    s.guard_failures,
                    s.trace_ops_before,
                    s.trace_ops_after,
                ));
            }
        }

        for &(end, brdg, gfails, ops_pre, ops_post) in &marks {
            let start = end - BUCKET;
            let sum: u64 = allocs[start..end].iter().sum();
            let per_call = sum as f64 / BUCKET as f64;
            println!(
                "  {:>12} {per_call:>12.1} {:>12.3} {brdg:>6} {gfails:>8} {ops_pre:>9} \
                 {ops_post:>9}",
                format!("{}-{}", start + 1, end),
                per_call / n as f64,
            );
        }
    }
    println!(
        "\n  allocs/row rising toward the never-compiles figure across call 200 means the \
         rows are being INTERPRETED;"
    );
    println!(
        "  staying ~0 means they are not, and the penalty is inside compiled execution \
         (which is NOT the same as executing cleanly)."
    );
}

/// Probe M. Probes J/K/L all stop at call 460, and #122's headline says the
/// first bridge slows the artifact **permanently**. Probe G's own rows say
/// otherwise, and the arithmetic is decidable without a new hypothesis.
///
/// Probe G at n=10 uses `reps = 2000` and 41 rounds, so **every one of its
/// 82 000 calls except the first ~199 is past call 200**. Under a two-regime
/// model (pre 3 870 ns, post 9 214 ns) the *lowest* value it could report is the
/// first round's mixture,
///
/// ```text
/// (199 * 3870 + 1801 * 9214) / 2000 = 8682 ns/call
/// ```
///
/// and every later round is a flat 9 214. It reports **4 984-5 386**. That is
/// 1.74x below the minimum the two-regime model permits, so the model is
/// incomplete: something keeps changing after call 460. The same gap sits in the
/// other two columns and in the same direction — n=100 reads ~50 000 against a
/// post-bridge 68 400, n=1000 reads ~880 000 against 1 040 000.
///
/// So this extends the call-index axis to 6 000 calls and carries **both**
/// instruments, because they answer different questions on the same run:
/// ns/row tests "permanent" for #122, allocs/row tracks the post-bridge
/// composition for #125. Windows are the 20 calls ending at each mark, so they
/// are Probe J's buckets placed where the axis is interesting rather than every
/// 20 calls forever.
fn long_horizon(label: &str, lowered: &LoweredF) {
    println!("\nProbe M — {label}: is the first bridge's step PERMANENT?");

    const MARKS: [usize; 16] = [
        20, 100, 180, 200, 220, 260, 320, 400, 420, 500, 700, 1_000, 1_500, 2_500, 4_000, 6_000,
    ];
    const WIN: usize = 20;

    for (n, calls) in [(10usize, 6_000usize), (100, 3_000), (1_000, 800)] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];

        // Both yardsticks measured here, against this binary and this n.
        reset_persistent_state();
        for _ in 0..8 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }
        let (_, never_a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, NEVER)));
        let t = std::time::Instant::now();
        for _ in 0..64 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }
        let never_ns = t.elapsed().as_nanos() as f64 / 64.0;

        println!(
            "\n  n={n}  never-compiles {never_ns:.0} ns/call ({:.1} ns/row, {:.3} allocs/row)",
            never_ns / n as f64,
            never_a as f64 / n as f64,
        );
        println!(
            "  {:>12} {:>11} {:>10} {:>11} {:>7} {:>7} {:>6} {:>8}",
            "calls", "ns/call", "ns/row", "allocs/row", "vs pre", "vs nvr", "brdg", "gfails"
        );

        reset_persistent_state();
        reset_jit_stats();
        let mut per_call: Vec<(f64, u64)> = Vec::with_capacity(calls);
        let mut marks = Vec::new();
        for k in 0..calls {
            let t = std::time::Instant::now();
            let (_, a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD)));
            per_call.push((t.elapsed().as_nanos() as f64, a));
            if MARKS.contains(&(k + 1)) {
                let s = jit_stats();
                marks.push((k + 1, s.bridges_compiled, s.guard_failures));
            }
        }

        let window = |end: usize| -> (f64, f64) {
            let slice = &per_call[end - WIN..end];
            let ns = slice.iter().map(|c| c.0).sum::<f64>() / WIN as f64;
            let al = slice.iter().map(|c| c.1).sum::<u64>() as f64 / WIN as f64;
            (ns, al)
        };
        // The last window before the first bridge is the reference every ratio
        // in this table is read against.
        let (pre_ns, _) = window(200);

        for &(end, brdg, gfails) in &marks {
            let (ns, al) = window(end);
            println!(
                "  {:>12} {ns:>11.0} {:>10.1} {:>11.3} {:>7.2} {:>7.2} {brdg:>6} {gfails:>8}",
                format!("{}-{}", end - WIN + 1, end),
                ns / n as f64,
                al / n as f64,
                ns / pre_ns,
                ns / never_ns,
            );
        }
    }
    println!(
        "\n  `vs pre` falling back toward 1.00 as calls grow means the step DECAYS and \
         'permanently' is wrong;"
    );
    println!("  staying flat at its call-220 value means the step is permanent as filed.");
}

/// Probe N. Probe M fitted `allocs/call = a*(n-1) + b` over n = 10/100/1000 and
/// got exact integers: portal `12(n-1)+4`, pre-bridge `69` (CL) / `66` (DYN),
/// post-bridge `23(n-1)+4` (CL) / `20(n-1)+4` (DYN). The load-bearing reading is
/// that **`b` falls 69 -> 4 across the bridge**, i.e. the compiled entry price
/// stops being paid.
///
/// ⛔ That reading is the *weakest* part of the fit. At n=1000 the constant term
/// is 0.02% of the total, so the n=1000 point barely constrains it; `b` is
/// carried almost entirely by n=10. **A fitted constant must be tested where it
/// dominates, not where it rounds away.**
///
/// So this runs the same axis at n = 2, 3, 5, 8, where `b` is 6-25% of the
/// total — and it yields a discriminator that needs **no model at all**:
///
/// ```text
/// n=2 post-bridge is predicted at 23*1 + 4 = 27 allocations/call,
/// which is BELOW the pre-bridge entry price of 69/call.
/// ```
///
/// If the artifact still paid its compiled entry after the bridge, the total
/// could not fall below 69 no matter what the per-back-edge term is. So a
/// post-bridge reading under 69 at n=2 refutes "the entry is still paid" by
/// inequality, independent of whether the linear model is right.
///
/// ⚠ `loops` is printed because the inference dies if the loop never compiles at
/// these row counts: an uncompiled arm would read the portal's `12(n-1)+4`,
/// which at n=2 is 16 — also under 69, and for the wrong reason.
fn small_n_constant(label: &str, lowered: &LoweredF) {
    println!("\nProbe N — {label}: test the per-call constant where it is not negligible");
    println!(
        "  {:>4} {:>9} {:>7} {:>9} {:>9} {:>9} {:>10} {:>6} {:>6} {:>7}",
        "n", "portal", "brdg@", "early", "pre", "post", "post pred", "loops", "brdg", "entry?"
    );

    const CALLS: usize = 700;

    for n in [2usize, 3, 5, 6, 7, 8, 10] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];

        reset_persistent_state();
        for _ in 0..8 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }
        let (_, portal) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, NEVER)));

        reset_persistent_state();
        reset_jit_stats();
        let mut allocs: Vec<u64> = Vec::with_capacity(CALLS);
        // ⛔ The first version of this probe read fixed windows and was wrong at
        // n=5: probe M already showed the first bridge lands at a different call
        // index for different n, so a fixed "pre" window silently sampled the
        // post-bridge regime. Locate the bridge instead of assuming it.
        let mut brdg_at = 0usize;
        for k in 0..CALLS {
            let (_, a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD)));
            allocs.push(a);
            if brdg_at == 0 && jit_stats().bridges_compiled > 0 {
                brdg_at = k + 1;
            }
        }
        let s = jit_stats();
        let mean = |r: std::ops::Range<usize>| -> f64 {
            let len = r.len() as f64;
            allocs[r].iter().sum::<u64>() as f64 / len
        };
        // 20 calls ending just before the bridge, and the last 20 of the run.
        let pre = if brdg_at > 21 {
            mean(brdg_at - 21..brdg_at - 1)
        } else {
            f64::NAN
        };
        let post = mean(CALLS - 20..CALLS);
        // Well after the loop compiles (threshold is 8 back edges) and well
        // before any bridge. At n=2 there is no `pre` window to read, because
        // no bridge is ever compiled — this is the only view of that arm.
        let early = mean(20..40);
        // The cranelift/dynasm slopes differ, so predict with whichever this
        // binary was built against rather than hard-coding one backend.
        let slope = if cfg!(feature = "jit-cranelift") {
            23.0
        } else {
            20.0
        };
        let pred = slope * (n as f64 - 1.0) + 4.0;

        println!(
            "  {n:>4} {portal:>9} {brdg_at:>7} {early:>9.1} {pre:>9.1} {post:>9.1} {pred:>10.1} {:>6} {:>6} \
             {:>7}",
            s.loops_compiled,
            s.bridges_compiled,
            if post < pre { "GONE" } else { "n/a" },
        );
    }
    println!(
        "\n  `entry? GONE` with loops>0 means the post-bridge total is below the pre-bridge \
         entry price,"
    );
    println!("  so the compiled entry cannot still be being paid — an inequality, not a fit.");
}

/// Probe O. Probe N found that **n=2 never compiles a bridge in 700 calls** and
/// yet sits at `23(n−1)+4` — the *steady-state* value — from call 21 onward,
/// where n=10 sits at the pre-bridge constant of 69 until its bridge at call
/// ~201. So at n=2 the per-back-edge regime exists with no bridge, which refutes
/// the bridge as *necessary* for it.
///
/// Probe N never looked below call 21, so "it changes at 21" was never measured;
/// the loop compiles at ~call 8 (threshold is 8 back edges, and n=2 takes one
/// back edge per call). This resolves every call from 1 to 40 individually.
///
/// The discriminator:
///
/// * n=2 reads 27 from the call `loops_compiled` first moves ⇒ **the regime is
///   installed at LOOP COMPILE, not at the bridge**, and at n≥3 the bridge is
///   where it becomes visible rather than what causes it.
/// * n=2 reads 69 for a while and steps later, with no bridge ⇒ a third
///   installer, and both #128's bridge and loop compile are off the hook.
///
/// n=10 rides along as the control: same axis, same instrument, and its regime
/// is known to change at the bridge instead.
fn regime_installed_at(label: &str, lowered: &LoweredF) {
    println!("\nProbe O — {label}: WHEN is the per-back-edge regime installed?");

    for n in [2usize, 10] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];

        reset_persistent_state();
        reset_jit_stats();
        println!(
            "\n  n={n}   (steady-state model predicts {})",
            23 * (n - 1) + 4
        );
        println!(
            "  {:>5} {:>8} {:>7} {:>6} {:>8}",
            "call", "allocs", "loops", "brdg", "gfails"
        );
        let mut prev = (usize::MAX, usize::MAX);
        for k in 0..40 {
            let (_, a) = metered(|| black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD)));
            let s = jit_stats();
            let now = (s.loops_compiled, s.bridges_compiled);
            // Print every call up to 16, then only where a counter moves or the
            // allocation count changes — the transition is what this is for.
            let moved = now.0 != prev.0 || now.1 != prev.1;
            if k < 16 || moved {
                println!(
                    "  {:>5} {a:>8} {:>7} {:>6} {:>8}{}",
                    k + 1,
                    s.loops_compiled,
                    s.bridges_compiled,
                    s.guard_failures,
                    if moved && k > 0 {
                        "  <- counter moved"
                    } else {
                        ""
                    },
                );
            }
            prev = now;
        }
    }
}

/// One measured regime: the calls that agree on `(loops, bridges, guard
/// failures this call, allocations)`.
struct Regime {
    loops: usize,
    brdg: usize,
    /// `guard_failures` DELTA for the call, not the cumulative counter. This is
    /// the model's `G` read off the instrument instead of solved for.
    gf: usize,
    allocs: u64,
    calls: usize,
    first_call: usize,
    hist: Vec<u64>,
}

/// One compiled loop, recovered as a per-call delta over the cumulative
/// counters. `COMPILES` / `TRACE_OPS_BEFORE` / `TRACE_OPS_AFTER` are all
/// `fetch_add` (`bytecode.rs:1868-1870`), so a run that compiles two loops
/// reports their SUM and a reader who takes the total for "the size of the
/// artifact" reads 13+25 as one 38-op trace.
struct Compile {
    /// The call the counters moved on. Two loops compiling within one call are
    /// one event with `loops == 2`: nothing finer is observable from outside,
    /// and splitting them would invent an ordering.
    call: usize,
    loops: usize,
    ops_before: usize,
    ops_after: usize,
}

/// Groups a run's calls by `(bridges_compiled, allocations)` and keeps one
/// histogram per group. No fixed windows: probe N reported 86 against a true 84
/// at n=5 by reading an index on an axis whose landmark moves.
fn regimes(lowered: &LoweredF, n: usize, calls: usize, threshold: u32) -> Vec<Regime> {
    census(lowered, n, calls, threshold).0
}

/// The regime table and the compile log off ONE run. They have to come from the
/// same run: the two are read against each other, and a second run would be a
/// second population of compiled artifacts.
fn census(
    lowered: &LoweredF,
    n: usize,
    calls: usize,
    threshold: u32,
) -> (Vec<Regime>, Vec<Compile>) {
    let (price, qty) = flat_columns(n);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];

    reset_persistent_state();
    reset_jit_stats();
    let mut out: Vec<Regime> = Vec::new();
    let mut compiles: Vec<Compile> = Vec::new();
    let mut prev_gf = jit_stats().guard_failures;
    let mut prev_c = (0usize, 0usize, 0usize);
    for k in 0..calls {
        let (_, a, hist) =
            metered_sizes(|| black_box(eval_batch_sum_f(lowered, &columns, n, threshold)));
        let s = jit_stats();
        let gf = s.guard_failures - prev_gf;
        prev_gf = s.guard_failures;
        let now_c = (s.loops_compiled, s.trace_ops_before, s.trace_ops_after);
        if now_c.0 != prev_c.0 {
            compiles.push(Compile {
                call: k + 1,
                loops: now_c.0 - prev_c.0,
                ops_before: now_c.1 - prev_c.1,
                ops_after: now_c.2 - prev_c.2,
            });
        }
        prev_c = now_c;
        // The two counters come through independent paths in `bump`, so a
        // disagreement means the interval measured is not the one claimed.
        let summed: u64 = hist.iter().sum();
        assert_eq!(
            summed,
            a,
            "n={n} call {}: histogram sums to {summed}, count says {a}",
            k + 1
        );
        match out
            .iter_mut()
            .find(|r| r.brdg == s.bridges_compiled && r.gf == gf && r.allocs == a)
        {
            Some(r) => r.calls += 1,
            None => out.push(Regime {
                loops: s.loops_compiled,
                brdg: s.bridges_compiled,
                gf,
                allocs: a,
                calls: 1,
                first_call: k + 1,
                hist,
            }),
        }
    }
    (out, compiles)
}

/// Picks the regime a run spends the most calls in, among those at `brdg`.
/// Selecting by population rather than by a predicted allocation count keeps the
/// model out of its own decomposition.
fn dominant(rs: &[Regime], brdg: usize) -> Option<&Regime> {
    rs.iter().filter(|r| r.brdg == brdg).max_by_key(|r| r.calls)
}

/// Probe P. The E/G model fits twelve cells exactly and still says nothing about
/// what is INSIDE a compiled entry: the fit is over per-call totals, so any
/// residual sitting within the per-entry price is invisible to it by
/// construction. Counting harder cannot reach it — the counts are already exact.
///
/// So this stops counting allocations and starts typing them. The allocator
/// already sees a `Layout` on every call and throws the size away; recording it
/// turns each of the three unexplained constants into a subtraction over
/// histograms rather than over scalars:
///
/// * the **42** = (n=10 pre-bridge) − (n=2), both at one entry per call;
/// * the **42 again** = (n=10 at brdg=1) − (n=10 at brdg=2), a different pair of
///   regimes entirely. If these two disagree, the 42 is two mechanisms that
///   happen to share a magnitude, and `G` is a label rather than a variable.
/// * the **per-entry 23** = [(n=10, 9 entries) − (n=2, 1 entry)] / 8;
/// * the **per-row portal 12** = [portal(10) − portal(2)] / 8, which decides
///   whether `23 = 12 + 11` is a nesting or just an arithmetic coincidence — if
///   the two signatures are disjoint by size, the additive story is dead
///   structurally and not merely unproven.
///
/// ⚠ This can decompose the model. It cannot confirm it.
fn alloc_size_signature(label: &str, lowered: &LoweredF) {
    println!("\nProbe P — {label}: what are the allocations, by size?");

    const CALLS: usize = 900;

    let mut portal: Vec<(usize, u64, Vec<u64>)> = Vec::new();
    for n in [2usize, 10] {
        let (price, qty) = flat_columns(n);
        let columns = vec![Column::Int(&price), Column::Int(&qty)];
        reset_persistent_state();
        for _ in 0..8 {
            black_box(eval_batch_sum_f(lowered, &columns, n, NEVER));
        }
        let (_, a, hist) =
            metered_sizes(|| black_box(eval_batch_sum_f(lowered, &columns, n, NEVER)));
        portal.push((n, a, hist));
    }

    println!("\n  portal arm (never compiles):");
    for (n, a, hist) in &portal {
        println!("    n={n:<4} allocs {a:>5}   {}", hist_u(hist));
    }

    let r2 = regimes(lowered, 2, CALLS, THRESHOLD);
    let r10 = regimes(lowered, 10, CALLS, THRESHOLD);

    for (n, rs) in [(2usize, &r2), (10usize, &r10)] {
        println!("\n  compiled arm, n={n} over {CALLS} calls — every distinct regime:");
        println!(
            "    {:>6} {:>5} {:>8} {:>7} {:>7}   histogram",
            "brdg", "gf", "allocs", "calls", "first"
        );
        for r in rs.iter() {
            println!(
                "    {:>6} {:>5} {:>8} {:>7} {:>7}   {}",
                r.brdg,
                r.gf,
                r.allocs,
                r.calls,
                r.first_call,
                hist_u(&r.hist)
            );
        }
    }

    // Two of cranelift's per-entry allocations read 64 B / 200 B at n=2 and
    // 96 B / 216 B at n=10, so a per-ENTRY structure is sized by the whole
    // batch. Two n points cannot tell a linear law from a coincidence, and they
    // cannot tell "sized by n" from "sized by the trace", so sweep n and carry
    // the trace op counts alongside.
    let slope = if cfg!(feature = "jit-cranelift") {
        23i64
    } else {
        20
    };
    println!("\n  size law — steady regimes only (>=20 calls), swept over n:");
    println!("  `E` is solved from allocs = {slope}E + 42*gf + 4 with the MEASURED gf, so it is");
    println!("  one unknown in one equation — a non-integer E refutes the model outright.");
    println!(
        "    {:>4} {:>6} {:>5} {:>5} {:>8} {:>7} {:>7} {:>8} {:>8}   histogram",
        "n", "loops", "brdg", "gf", "allocs", "calls", "E", "ops_bef", "ops_aft"
    );
    for n in [2usize, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
        let rs = regimes(lowered, n, CALLS, THRESHOLD);
        let s = jit_stats();
        for r in rs.iter().filter(|r| r.calls >= 20) {
            let rest = r.allocs as i64 - 42 * r.gf as i64 - 4;
            let e = if rest % slope == 0 {
                format!("{}", rest / slope)
            } else {
                format!("{:.2}!", rest as f64 / slope as f64)
            };
            println!(
                "    {n:>4} {:>6} {:>5} {:>5} {:>8} {:>7} {e:>7} {:>8} {:>8}   {}",
                r.loops,
                r.brdg,
                r.gf,
                r.allocs,
                r.calls,
                s.trace_ops_before,
                s.trace_ops_after,
                hist_u(&r.hist)
            );
        }
    }

    println!("\n  --- decomposition ---");

    let base = dominant(&r2, 0);
    let pre = dominant(&r10, 0);
    let post1 = dominant(&r10, 1);
    let post2 = r10.iter().filter(|r| r.brdg >= 2).max_by_key(|r| r.calls);

    if let (Some(base), Some(pre)) = (base, pre) {
        let d = hist_diff(&pre.hist, &base.hist);
        println!(
            "\n  guard exit, route 1 (n=10 pre-bridge {} − n=2 {}):",
            pre.allocs, base.allocs
        );
        println!("    {}", hist_line(&d));
    }
    if let (Some(p1), Some(p2)) = (post1, post2) {
        let d = hist_diff(&p1.hist, &p2.hist);
        println!(
            "\n  guard exit, route 2 (n=10 brdg=1 {} − brdg>=2 {}):",
            p1.allocs, p2.allocs
        );
        println!("    {}", hist_line(&d));
    }
    if let (Some(base), Some(p2)) = (base, post2) {
        // n=10 post-bridge runs n−1 = 9 entries per call against n=2's 1, so the
        // difference is 8 entries with the harness constant and the guard exit
        // both cancelled.
        let d = hist_diff(&p2.hist, &base.hist);
        println!(
            "\n  8 compiled entries (n=10 brdg>=2 {} − n=2 {}):",
            p2.allocs, base.allocs
        );
        println!("    {}", hist_line(&d));
        match hist_div(&d, 8) {
            Some(per) => println!("    per entry:   {}", hist_line(&per)),
            None => println!("    ⛔ NOT divisible by 8 — the 9 entries are not alike by size"),
        }
    }
    if portal.len() == 2 {
        let d = hist_diff(&portal[1].2, &portal[0].2);
        println!(
            "\n  8 portal back edges (portal n=10 {} − n=2 {}):",
            portal[1].1, portal[0].1
        );
        println!("    {}", hist_line(&d));
        match hist_div(&d, 8) {
            Some(per) => println!("    per row:     {}", hist_line(&per)),
            None => println!("    ⛔ NOT divisible by 8 — the 9 rows are not alike by size"),
        }
    }
}

/// Probe Q. Probe P's sweep reads `trace_ops_after` as 13 at n ∈ {2,3,5} and 25
/// everywhere else in 2..=12, and `(n−1)` divides the threshold 8 at exactly
/// {2,3,5,9}. Eleven values with no exception, identical on both backends — and
/// a fit with no exception, taken at ONE setting of the divisor, is the shape
/// that has one more variable in it. Two points determine a line.
///
/// The threshold is an external lever and `DRIVERS` keys on it, so each setting
/// gets its own driver and its own artifact. That makes the fit directly
/// testable:
///
/// * if the law is `(n−1) | threshold`, the flat set MOVES with the threshold —
///   {2,3,4,5,7,13} at 12, {2,4,10} at 9, {2,8} at 7;
/// * if 8 is a constant of the machinery that the threshold merely coincides
///   with, the flat set stays {2,3,5,9} at every setting.
///
/// t=9 and t=7 are the sharp settings: both predict n=3 and n=5 GROW, which the
/// constant-8 reading forbids. t=16 is deliberately included as the weak
/// control — inside n ≤ 17 it predicts {2,3,5,9,17}, which agrees with 8 on
/// every value but 17, so it separates "divides" from "is at most 8" and
/// nothing else.
///
/// The reading is `ops_after` against `ops_before` per compile, not against the
/// literal 13/25: the question is whether the optimizer GREW the trace, and a
/// hard-coded pair of numbers would only be readable on this one expression.
fn threshold_control(label: &str, lowered: &LoweredF) {
    println!("\nProbe Q — {label}: does the flat-trace set follow the THRESHOLD?");

    const CALLS: usize = 900;
    const NS: std::ops::RangeInclusive<usize> = 2..=17;

    // 3/4/5 and 18/24 are not about the divisor law — they separate the two
    // readings of MIXED. Every MIXED cell in 6..=16 has `t/(n−1) <= 2` AND
    // `n >= 7`, and those two are confounded there: the smallest n with a ratio
    // of 1 is n = t+1 >= 7. t=3/4/5 put a ratio of 1 at n = 4/5/6, and t=18/24
    // put a ratio of 3 and 4 at n = 7 and n = 7/9. If MIXED follows the ratio,
    // the first three flip and the last two do not; if it follows n, the
    // reverse.
    for threshold in [3u32, 4, 5, 6, 7, 8, 9, 10, 12, 16, 18, 24] {
        println!("\n  threshold {threshold} — predicted flat where (n−1) | {threshold}");
        println!(
            "    {:>4} {:>9} {:>7} {:>9} {:>8} {:>4} {:>4} {:>6}   {}",
            "n",
            "(n−1)|t",
            "loops",
            "verdict",
            "allocs",
            "gf",
            "E",
            "first",
            "compiles (call: before -> after)"
        );
        let mut predicted: Vec<usize> = Vec::new();
        let mut flat: Vec<usize> = Vec::new();
        let mut mixed: Vec<usize> = Vec::new();
        let mut model_breaks: Vec<usize> = Vec::new();
        for n in NS {
            let (rs, cs) = census(lowered, n, CALLS, threshold);
            let divides = (threshold as usize) % (n - 1) == 0;
            if divides {
                predicted.push(n);
            }
            let loops: usize = cs.iter().map(|c| c.loops).sum();
            // Read per compile, so a run that compiles one flat and one grown
            // loop is reported as neither — Probe P's n=9 summed them to 38 and
            // printed that on all five of its regimes.
            let any_flat = cs.iter().any(|c| c.ops_after == c.ops_before);
            let any_grown = cs.iter().any(|c| c.ops_after > c.ops_before);
            let verdict = match (any_flat, any_grown) {
                (true, false) => {
                    flat.push(n);
                    "FLAT"
                }
                (false, true) => "GROWN",
                (true, true) => {
                    mixed.push(n);
                    "MIXED"
                }
                (false, false) => "none",
            };
            let detail = cs
                .iter()
                .map(|c| {
                    let many = if c.loops > 1 {
                        format!(" [{} loops]", c.loops)
                    } else {
                        String::new()
                    };
                    format!("{}: {}->{}{many}", c.call, c.ops_before, c.ops_after)
                })
                .collect::<Vec<_>>()
                .join("  ");
            // #125's model, re-solved at every threshold. The artifact halving
            // is a fact about the TRACE; whether it moves the per-call
            // allocation bill is a separate question, and assuming it does not
            // would be assuming the answer.
            let slope = if cfg!(feature = "jit-cranelift") {
                23i64
            } else {
                20
            };
            let steady = rs
                .iter()
                .filter(|r| r.calls >= 20)
                .max_by_key(|r| r.calls)
                .expect("no regime holds 20 calls");
            let rest = steady.allocs as i64 - 42 * steady.gf as i64 - 4;
            let e = if rest % slope == 0 {
                format!("{}", rest / slope)
            } else {
                format!("{:.2}!", rest as f64 / slope as f64)
            };
            // NOT `E == n−1`: the dominant regime is whichever one holds the
            // most calls, and at (t=10, n=6) that is a gf=1 regime reading
            // 138 = 23*4 + 42*1 + 4 — an integer E of 4 against 5 back edges.
            // The invariant the model actually asserts is that every back edge
            // is accounted for, as a compiled entry OR as a guard exit.
            if rest % slope != 0 || rest / slope + steady.gf as i64 != (n - 1) as i64 {
                model_breaks.push(n);
            }
            println!(
                "    {n:>4} {:>9} {loops:>7} {verdict:>9} {:>8} {:>4} {e:>4} {:>6}   {detail}",
                if divides { "yes" } else { "no" },
                steady.allocs,
                steady.gf,
                steady.first_call,
            );
        }
        println!("    predicted flat: {predicted:?}");
        println!("    observed  flat: {flat:?}   mixed: {mixed:?}");
        // MIXED counts as agreement only where it is predicted: a run that
        // compiles both shapes has produced the flat artifact, which is what
        // the predicate claims. Where it is NOT predicted it is a miss.
        let agrees = predicted
            .iter()
            .all(|n| flat.contains(n) || mixed.contains(n))
            && flat.iter().all(|n| predicted.contains(n))
            && mixed.iter().all(|n| predicted.contains(n));
        println!(
            "    => {}",
            if agrees {
                "AGREES with (n−1) | threshold"
            } else {
                "⛔ DISAGREES with (n−1) | threshold"
            }
        );
        println!(
            "    => {}",
            if model_breaks.is_empty() {
                "E + gf = n−1 in the dominant steady regime at every n".to_string()
            } else {
                format!("⛔ E + gf != n−1 at {model_breaks:?}")
            }
        );
    }
}

fn main() {
    let schema = flat_schema();
    let arith = lower("price + qty * 2", &schema);

    println!("load before probes: {}", loadavg());
    // Probe P indexes by call index from a cold driver, same as M/N/O, and it is
    // the only probe that turns the size histogram on. `RCA88B_SIZES=1`.
    if std::env::var_os("RCA88B_SIZES").is_some() {
        alloc_size_signature("arith price + qty * 2", &arith);
        println!("\nload after probes:  {}", loadavg());
        return;
    }
    // Probe Q sweeps the threshold, which is part of the `DRIVERS` key, so it
    // interns a driver per setting — same cold-driver requirement as P, and the
    // same gate.
    if std::env::var_os("RCA88B_THRESHOLD").is_some() {
        threshold_control("arith price + qty * 2", &arith);
        println!("\nload after probes:  {}", loadavg());
        return;
    }
    // Probe M is the same cold-driver axis as J/K/L but 13x longer, so it gets
    // its own gate: the question it settles is #122's "permanently", which does
    // not need the three shorter probes to have run first.
    if std::env::var_os("RCA88B_LONG").is_some() {
        long_horizon("arith price + qty * 2", &arith);
        small_n_constant("arith price + qty * 2", &arith);
        regime_installed_at("arith price + qty * 2", &arith);
        println!("\nload after probes:  {}", loadavg());
        return;
    }
    // Probes H and I need a clean process: both index cost by CALL INDEX from a
    // cold driver, and every earlier probe leaves compiled programs interned in
    // the pool. `RCA88B_STEP=1` runs them alone.
    if std::env::var_os("RCA88B_STEP").is_some() {
        cost_by_call_index("arith price + qty * 2", &arith);
        step_moves_with_threshold("arith price + qty * 2", &arith);
        // Same gate for the same reason: Probe L indexes by call index from a
        // cold driver, so it cannot follow a probe that has already interned
        // programs in the pool.
        execution_by_allocation("arith price + qty * 2", &arith);
        println!("\nload after probes:  {}", loadavg());
        return;
    }
    // After eliminating warmup count, in-round interleaving and the clean arm,
    // the only difference left between Probe B and Probe F's last row — which
    // are the same structure and read 4.6 us and 48.7 us — is WHERE IN THE
    // PROCESS they run. `RCA88B_F_FIRST=1` swaps them. If the fast/slow reading
    // follows the position rather than the structure, the cost is a function of
    // what the process compiled earlier, and every per-call number in #88 is
    // conditioned on probe order.
    let f_first = std::env::var_os("RCA88B_F_FIRST").is_some();
    if f_first {
        reconstruct_probe_b();
    }
    artifact_or_boundary("arith price + qty * 2", &arith);
    allocations_per_row("arith price + qty * 2", &arith);
    if !f_first {
        reconstruct_probe_b();
    }
    // The order A/B needs only Probes B and F, and running the rest costs
    // minutes on a loaded box — which is also the window in which another
    // session's load can move under the two halves of the comparison.
    warmup_degradation();
    population_times_warmup();
    if std::env::var_os("RCA88B_FAST").is_some() {
        println!("\nload after probes:  {}", loadavg());
        return;
    }
    same_shape_across_size();
    forced_exit_path();
    artifact_scaling();
    population_sweep("arith price + qty * 2", "price + qty * 2");
    println!("\nload after probes:  {}", loadavg());
}
