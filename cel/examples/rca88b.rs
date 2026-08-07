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

std::thread_local! {
    static LOCAL_ALLOCS: Cell<u64> = const { Cell::new(0) };
}
static GLOBAL_ALLOCS: AtomicU64 = AtomicU64::new(0);

struct Counting;

#[inline]
fn bump() {
    GLOBAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
    let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
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
        "{:>8} {:>10} {:>12} {:>12} {:>10} {:>6} {:>13}",
        "n", "warmups", "warm ns", "never ns", "warm/nvr", "brdg", "gfails/call"
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

        for warmups in [1usize, 2, 4, 8, 16, 32, 64, 128, 512] {
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
                "{n:>8} {warmups:>10} {best_warm:>12.1} {best_never:>12.1} {:>10.3} {:>6} \
                 {:>13.3}",
                best_warm / best_never,
                after.bridges_compiled - before.bridges_compiled,
                (after.guard_failures - before.guard_failures) as f64 / (41 * reps) as f64,
            );
        }
    }
    println!(
        "  warm/nvr crossing 1.0 is the artifact ceasing to pay for itself. \
         Nothing runs between the warmup and the clock."
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

fn main() {
    let schema = flat_schema();
    let arith = lower("price + qty * 2", &schema);

    println!("load before probes: {}", loadavg());
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
