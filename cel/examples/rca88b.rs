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

use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

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
        "{:>8} {:>11} {:>13} {:>13} {:>12} {:>14} {:>7} {:>9}",
        "n",
        "clean ns",
        "jit(compiled)",
        "jit(nocomp)",
        "compiled/cl",
        "nocompiled/cl",
        "loops",
        "loops@nvr"
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

        println!(
            "{n:>8} {best_clean:>11.1} {best_compiled:>13.1} {best_never:>13.1} \
             {:>12.3} {:>14.3} {:>7} {:>9}",
            best_compiled / best_clean,
            best_never / best_clean,
            compiled_stats.loops_compiled,
            never_stats.loops_compiled,
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

fn main() {
    let schema = flat_schema();
    let arith = lower("price + qty * 2", &schema);

    println!("load before probes: {}", loadavg());
    artifact_or_boundary("arith price + qty * 2", &arith);
    population_sweep("arith price + qty * 2", "price + qty * 2");
    println!("\nload after probes:  {}", loadavg());
}
