//! #88 probe: is cel's inner element loop artifact ever ENTERED?
//!
//! `loops_compiled` cannot answer this — a loop that compiles and is never
//! entered reads exactly like one that compiles and runs. The #91 census gives
//! the suspect:
//!
//! ```text
//! per_row=2  loops=2  bridges=0  guard_fails=1   at 4000, 20000 AND 100000 rows
//! per_row=3  loops=2  bridges=2  guard_fails=401
//! per_row=8  loops=2  bridges=1  guard_fails=201
//! ```
//!
//! One guard failure over 100000 rows is the floor for the OUTER row loop
//! alone, so the inner loop contributed zero. That is consistent with never
//! entering it AND with entering and leaving by a non-guard path. This
//! separates them.
//!
//! ANSWER: entered — or at least the element work IS done in compiled code.
//! `jit/clean` per element at per_row=2 is **0.033 on cranelift and 0.040 on
//! dynasm**, i.e. the compiled tier is 25-30x faster per element than the
//! interpreter. A never-entered artifact cannot produce that. per_row=2 is in
//! fact the BEST case in the sweep, so it is not a reproducer of #88's wall.
//!
//! The zero guard failures are explained without "never entered": at per_row<=2
//! the element loop is inlined into the outer row trace (ops 38->53, #76), so
//! there is no separate inner-loop exit to fail. A second artifact is compiled
//! and the work does not depend on entering it.
//!
//! Method note — the first method here FAILED and is kept because the failure is
//! the useful part. The intended discriminator was allocations against a
//! known-interpreted control, on the reasoning that allocation counts are exact
//! and load-independent where a timing on this shared box is not. It does not
//! work: **the clean tier allocates 4 times for a 320_000-element batch**, so
//! there is no interpreted allocation signal to compare against, and the entire
//! JIT-side bill is compile-time — flat at 12470 / 12470 / 12471 across a 16x
//! row range. An exact instrument pointed at a quantity that does not vary is
//! still no instrument. Only then is the timing sweep justified, and it is built
//! as a WITHIN-ROUND `jit/clean` ratio so a load spike moves both arms together.
//!
//! `per_row * rows` is held constant so every row of the sweep does the same
//! total element work; otherwise this re-measures the size curve of #102
//! instead of the entry question.

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

/// Allocations this thread made inside the window.
fn metered<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let before = LOCAL_ALLOCS.with(Cell::get);
    let out = f();
    let after = LOCAL_ALLOCS.with(Cell::get);
    (out, after - before)
}

const THRESHOLD: u32 = 8;
/// Held constant across the sweep so each row does the same element work.
const TOTAL_ELEMS: usize = 320_000;

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

fn main() {
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &schema);

    println!(
        "{:<9} {:>7} {:>9} {:>11} {:>11} {:>8} {:>7} {:>7} {:>7}",
        "per_row",
        "rows",
        "elems",
        "clean allocs",
        "jit allocs",
        "jit/clean",
        "loops",
        "bridges",
        "guard"
    );

    for per_row in [1i64, 2, 3, 4, 8, 16, 32] {
        let rows = TOTAL_ELEMS / per_row as usize;
        let lens = vec![per_row; rows];
        let mut offsets = Vec::with_capacity(rows);
        let mut total = 0i64;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        let elems: Vec<i64> = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
        let columns = [
            Column::Int(&lens),
            Column::Int(&offsets),
            Column::Int(&elems),
        ];

        // Known-interpreted control: same batch, no JIT.
        reset_persistent_state();
        let (clean, clean_allocs) =
            metered(|| black_box(clean_batch_sum_f(&lowered, &columns, rows)));

        // The compiled tier, from cold.
        reset_persistent_state();
        reset_jit_stats();
        let (jit, jit_allocs) =
            metered(|| black_box(eval_batch_sum_f(&lowered, &columns, rows, THRESHOLD)));
        assert_eq!(clean, jit, "per_row={per_row}: miscompile, not a census");
        let s = jit_stats();

        println!(
            "{per_row:<9} {rows:>7} {:>9} {clean_allocs:>11} {jit_allocs:>11} {:>8.3} {:>7} {:>7} {:>7}",
            total,
            jit_allocs as f64 / clean_allocs.max(1) as f64,
            s.loops_compiled,
            s.bridges_compiled,
            s.guard_failures,
        );
    }

    println!(
        "\nclean = interpreter only, so it is the known-interpreted reference.\n\
         jit/clean near 1.00 means the compiled artifact did not do the work."
    );

    // Does ANY of the allocation bill scale with the rows? If it does not, the
    // number above is compile cost and says nothing about execution, so
    // allocations cannot answer the entry question at all.
    println!("\nrows-scaling at fixed per_row (flat => the bill is compile-time only):");
    println!(
        "{:<9} {:>8} {:>11} {:>11} {:>10} {:>7} {:>7} {:>7}",
        "per_row", "rows", "clean allocs", "jit allocs", "jit/row", "loops", "bridges", "guard"
    );
    for per_row in [2i64, 3, 8] {
        for rows in [10_000usize, 40_000, 160_000] {
            let lens = vec![per_row; rows];
            let mut offsets = Vec::with_capacity(rows);
            let mut total = 0i64;
            for &l in &lens {
                offsets.push(total);
                total += l;
            }
            let elems: Vec<i64> = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
            let columns = [
                Column::Int(&lens),
                Column::Int(&offsets),
                Column::Int(&elems),
            ];

            reset_persistent_state();
            let (clean, clean_allocs) =
                metered(|| black_box(clean_batch_sum_f(&lowered, &columns, rows)));
            reset_persistent_state();
            reset_jit_stats();
            let (jit, jit_allocs) =
                metered(|| black_box(eval_batch_sum_f(&lowered, &columns, rows, THRESHOLD)));
            assert_eq!(clean, jit, "per_row={per_row} rows={rows}: miscompile");
            let s = jit_stats();
            println!(
                "{per_row:<9} {rows:>8} {clean_allocs:>11} {jit_allocs:>11} {:>10.4} {:>7} {:>7} {:>7}",
                jit_allocs as f64 / rows as f64,
                s.loops_compiled,
                s.bridges_compiled,
                s.guard_failures,
            );
        }
    }

    timing_sweep(&lowered);
    cold_vs_warm(&lowered);

    println!("\nload before ratio_vs_n: {}", loadavg());
    ratio_vs_n("nested items.all(i, i.price > 10)", &lowered, 8, false);

    let flat_schema: Schema = [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let flat = lower("price >= 100 && qty < 50", &flat_schema);
    ratio_vs_n("flat price >= 100 && qty < 50", &flat, 0, true);

    // #88's own `a + b * 2` probe, on the n axis. It is the same shape CLASS as
    // the predicate above — straight-line body inside the row loop, one compiled
    // loop — but a pure-arithmetic body, so a wall here cannot be blamed on the
    // comparisons or the logical-and. It is the shape that reported
    // `gfails/call == 0.00` at rows=64, i.e. the case the terminal-loop-exit
    // hypothesis predicts should have NO wall.
    let arith = lower("price + qty * 2", &flat_schema);
    ratio_vs_n("arith price + qty * 2", &arith, 0, true);
    println!("load after ratio_vs_n:  {}", loadavg());
}

/// Metric 2, run only because metric 1 came back flat: per-element time.
///
/// Reported as the WITHIN-ROUND ratio `jit / clean`, not as absolute ns. Both
/// arms are measured back to back in the same round on the same data, so a load
/// spike moves numerator and denominator together and largely cancels; an
/// absolute ns column on this box does not survive its own variance. Minimum
/// across rounds, because interference can only ever make a round slower.
///
/// The batch is compiled BEFORE the timed region, so this is steady-state
/// execution and not the fixed compile cost metric 1 just measured.
fn timing_sweep(lowered: &LoweredF) {
    const ROUNDS: usize = 7;
    println!("\nper-element time, compile excluded, min of {ROUNDS} interleaved rounds:");
    println!("load before: {}", loadavg());
    println!(
        "{:<9} {:>8} {:>13} {:>13} {:>10} {:>7} {:>7} {:>7}",
        "per_row", "rows", "clean ns/elem", "jit ns/elem", "jit/clean", "loops", "bridges", "guard"
    );

    for per_row in [1i64, 2, 3, 4, 8, 16, 32] {
        let rows = TOTAL_ELEMS / per_row as usize;
        let lens = vec![per_row; rows];
        let mut offsets = Vec::with_capacity(rows);
        let mut total = 0i64;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        let elems: Vec<i64> = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
        let columns = [
            Column::Int(&lens),
            Column::Int(&offsets),
            Column::Int(&elems),
        ];

        // Compile once, outside every timed region.
        reset_persistent_state();
        reset_jit_stats();
        black_box(eval_batch_sum_f(lowered, &columns, rows, THRESHOLD));
        let s = jit_stats();

        let mut best_clean = f64::MAX;
        let mut best_jit = f64::MAX;
        for _ in 0..ROUNDS {
            let t = std::time::Instant::now();
            black_box(clean_batch_sum_f(lowered, &columns, rows));
            let clean = t.elapsed().as_nanos() as f64 / total as f64;

            let t = std::time::Instant::now();
            black_box(eval_batch_sum_f(lowered, &columns, rows, THRESHOLD));
            let jit = t.elapsed().as_nanos() as f64 / total as f64;

            best_clean = best_clean.min(clean);
            best_jit = best_jit.min(jit);
        }

        println!(
            "{per_row:<9} {rows:>8} {best_clean:>13.3} {best_jit:>13.3} {:>10.3} {:>7} {:>7} {:>7}",
            best_jit / best_clean,
            s.loops_compiled,
            s.bridges_compiled,
            s.guard_failures,
        );
    }
    println!("load after:  {}", loadavg());
    println!(
        "\njit/clean well under 1 means the compiled inner loop is doing the element\n\
         work; jit/clean at or above 1 means it is not."
    );
}

/// Is #88's "34-92 µs fixed per call" the compile, paid once, or a real
/// per-call cost that recurs?
///
/// Metric 1 showed the whole allocation bill is compile-time and lands at
/// 12k-45k allocations for one cold batch — the right order of magnitude for
/// tens of µs. If that is the fixed cost, then a batch that does NOT reset the
/// driver pays it once and never again, and #88's ladder is measuring cold
/// start rather than a per-call wall.
fn cold_vs_warm(lowered: &LoweredF) {
    const ROUNDS: usize = 9;
    println!("\ncold (compile inside the call) vs warm (driver kept), µs per call:");
    println!(
        "{:<7} {:>8} {:>11} {:>11} {:>11} {:>12} {:>10}",
        "n", "per_row", "clean µs", "cold µs", "warm µs", "cold-warm", "warm/clean"
    );

    for n in [10usize, 100, 1_000, 10_000] {
        let per_row = 8i64;
        let lens = vec![per_row; n];
        let mut offsets = Vec::with_capacity(n);
        let mut total = 0i64;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        let elems: Vec<i64> = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
        let columns = [
            Column::Int(&lens),
            Column::Int(&offsets),
            Column::Int(&elems),
        ];

        let mut best_clean = f64::MAX;
        let mut best_cold = f64::MAX;
        let mut best_warm = f64::MAX;
        for _ in 0..ROUNDS {
            reset_persistent_state();
            let t = std::time::Instant::now();
            black_box(clean_batch_sum_f(lowered, &columns, n));
            best_clean = best_clean.min(t.elapsed().as_nanos() as f64 / 1000.0);

            // Cold: a fresh driver, so this call pays the compile.
            reset_persistent_state();
            let t = std::time::Instant::now();
            black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
            best_cold = best_cold.min(t.elapsed().as_nanos() as f64 / 1000.0);

            // Warm: same driver, loop already compiled.
            let t = std::time::Instant::now();
            black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
            best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / 1000.0);
        }

        println!(
            "{n:<7} {per_row:>8} {best_clean:>11.3} {best_cold:>11.3} {best_warm:>11.3} \
             {:>12.3} {:>10.3}",
            best_cold - best_warm,
            best_warm / best_clean,
        );
    }
    println!(
        "\ncold-warm isolates the compile. ⚠ THIS TABLE IS INDICATIVE ONLY: at these\n\
         µs scales min-of-{ROUNDS} on a loaded box is not enough, and the warm column\n\
         has been observed non-monotonic in n (193 µs at n=100 against 19 µs at\n\
         n=1000), which is a contamination tell, not a size effect. Only the\n\
         order of magnitude of cold-warm survives that: ~700-2500 µs, which is\n\
         10-70x LARGER than #88's quoted 34-92 µs fixed per-call cost — so that\n\
         cost cannot be a full compile. Re-run on a quiet box before quoting any\n\
         individual cell."
    );
}

/// The measurement the two sweeps above are BLIND to.
///
/// Holding `per_row * rows` at 320_000 puts every point far past #88's stated
/// break-even of n ~= 3500-5000, where a 51 µs fixed cost is 0.16 ns/element
/// against ~15 ns/element of work — amortized to invisibility. A sweep that
/// shows the JIT winning 25-30x there is silent about the wall, not evidence
/// against it.
///
/// The wall is size-dependent, so it shows up as `jit/clean` crossing 1.0 and
/// blowing up as n shrinks. Reported as a ratio for the same reason as
/// `timing_sweep`: ratios cancel load, absolute µs on this box do not.
///
/// `warm` keeps the driver across calls, so a cost that appears there is a
/// genuine per-call cost that survives compilation rather than the compile.
///
/// Two shapes, because the hypothesis on the table is that the wall IS the
/// terminal loop exit — the `+1` in `guard_failures = trace_eagerness *
/// bridges + 1`. If so, the wall should track `gfails/call`.
///
/// RESULT — that hypothesis is REFUTED twice over.
///
/// (1) By its own falsifiable prediction. The hypothesis predicts that the
/// `a + b * 2` shape, which reports `gfails/call == 0.00`, has no wall. It has
/// one: `price + qty * 2` at n=10 measures **25.30x (cranelift) / 25.08x
/// (dynasm)** with `gfails/call == 0.00` on both. Within 1% of each other.
///
/// (2) By anti-correlation rather than by a null. Over four runs (both
/// backends, two rounds each, host load 20-80):
/// `gfails/call` reads 0.00 at n = 10/100/1000, which is exactly where the
/// wall is (warm ratio 22-57x at n=10), and reads 1.00 at n = 10000/100000,
/// which is exactly where the wall is gone (warm ratio 0.05-0.18, the JIT
/// winning 6-15x). The `+1` terminal exit occurs only where there is no wall,
/// so it cannot be the wall.
///
/// What the sweep does establish:
/// - the wall is a per-call cost that SURVIVES warmup, and it is size
///   dependent: on the flat shape `warm ns/call` is ~3.3-5.6 µs whether the
///   call does 10 rows or 1000, i.e. flat across a 100x change in work;
/// - at n=10 the nested shape costs 55.7-62.7 µs/call on both backends, which
///   reproduces #88's reported 34-92 µs band and locates it on the n axis;
/// - `loops/call` and `aborts/call` are 0.00 in every cell, so this is the
///   steady-state cost of a settled artifact, not the driver still tracing.
///   Those two columns exist only to close that alternative;
/// - break-even is n ~= 330-500 rows (flat) and n ~= 65-115 rows (nested).
///
/// THE FLOOR CHECK, and why the small-n end is the contamination-prone one: a
/// timer or setup floor added to BOTH arms drags the ratio toward 1.0, so it
/// would HIDE this blow-up rather than manufacture it. Two guards:
/// - `reps` sizes each timed sample (2000 calls at n=10); the smallest timed
///   region in the table is ~20 µs against a ~40 ns `Instant` granularity.
/// - `clean ns/el` is the tell. On the nested shape it reads 11.46 / 10.68 /
///   10.20 / 10.45 / 10.28 (cranelift) — flat within 12% across a 10 000x
///   range in n, so the clean arm is still measuring row work at n=10. The two
///   single-loop shapes instead show ~17.6-19.7 at n=10 against a ~8-9
///   plateau, i.e. the CLEAN tier has its own ~95 ns per-call constant.
///
/// That constant is shared by both arms, so correcting for it makes the wall
/// LARGER, not smaller: at arith n=10, `(4452.8 - 95) / (176.0 - 95)` = 53.8x
/// against the 25.3x reported. Every ratio here is a lower bound.
///
/// The residual: at n=10 nested, 62 µs/call buys 80 elements of work that the
/// clean tier does in 1.2 µs, with zero compiles, zero aborts, zero guard
/// failures and (per the allocation sweep above) zero allocations. Every
/// counter available from outside majit reads zero. Separating "never entered"
/// from "entered and left by a non-guard path" needs the in-tree instrument,
/// which is blocked under the #89/#96 hold on `pyjitpl.rs`.
fn ratio_vs_n(label: &str, lowered: &LoweredF, per_row: i64, flat: bool) {
    println!("\n{label}: jit/clean vs n (per_row={per_row}), min of interleaved rounds");
    println!(
        "{:>8} {:>10} {:>10} {:>11} {:>11} {:>7} {:>7} {:>11} {:>11} {:>11} {:>11} {:>6}",
        "n",
        "cold ratio",
        "warm ratio",
        "clean ns/call",
        "warm ns/call",
        "loops",
        "guard",
        "gfails/call",
        "loops/call",
        "aborts/call",
        "clean ns/el",
        "reps"
    );

    for n in [10usize, 100, 1_000, 10_000, 100_000] {
        let rows = n;
        let (lens, offsets, elems);
        let columns: Vec<Column> = if flat {
            lens = (0..rows as i64).map(|i| (i * 37) % 200).collect::<Vec<_>>();
            offsets = (0..rows as i64).map(|i| (i * 11) % 100).collect::<Vec<_>>();
            vec![Column::Int(&lens), Column::Int(&offsets)]
        } else {
            lens = vec![per_row; rows];
            let mut off = Vec::with_capacity(rows);
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

        // Each timed region covers at least ~20k rows of work, so the clock's
        // own resolution cannot dominate the small-n cells.
        let reps = (20_000 / n).max(1);
        let rounds = if n <= 1_000 { 41 } else { 11 };

        // Warm: compile once, then time repeated calls on the live driver.
        reset_persistent_state();
        reset_jit_stats();
        black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
        let compiled = jit_stats();

        let mut best_clean = f64::MAX;
        let mut best_warm = f64::MAX;
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
            best_warm = best_warm.min(t.elapsed().as_nanos() as f64 / reps as f64);
        }

        // Guard failures, compiles and aborts per warm call, over a window of
        // known call count. loops/call and aborts/call are the controls that
        // separate "steady-state cost of a settled artifact" from "the driver
        // is still tracing on every call" — a fixed per-call cost means very
        // different things in those two worlds.
        reset_jit_stats();
        for _ in 0..100 {
            black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
        }
        let window = jit_stats();
        let per_call = window.guard_failures as f64 / 100.0;
        let loops_per_call = window.loops_compiled as f64 / 100.0;
        let aborts_per_call = window.loops_aborted as f64 / 100.0;

        // Cold: driver reset before each call, so every call pays the compile.
        let mut best_cold_clean = f64::MAX;
        let mut best_cold = f64::MAX;
        for _ in 0..rounds.min(15) {
            reset_persistent_state();
            let t = std::time::Instant::now();
            black_box(clean_batch_sum_f(lowered, &columns, n));
            best_cold_clean = best_cold_clean.min(t.elapsed().as_nanos() as f64);
            reset_persistent_state();
            let t = std::time::Instant::now();
            black_box(eval_batch_sum_f(lowered, &columns, n, THRESHOLD));
            best_cold = best_cold.min(t.elapsed().as_nanos() as f64);
        }

        // The floor check. A timer/setup floor added to BOTH arms drags the
        // ratio toward 1.0 and would HIDE the blow-up, so the small-n end is
        // the most contamination-prone, not the least. `clean ns/el` is the
        // tell: if it stays in the same range as at large n, the clean arm is
        // still measuring row work rather than the harness. `reps` is the
        // inner repetition count each timed sample covers, reported so the
        // reader can size the timed region instead of trusting it.
        let elems_per_call = if flat { n } else { n * per_row as usize };
        println!(
            "{n:>8} {:>10.2} {:>10.3} {best_clean:>11.1} {best_warm:>11.1} {:>7} {:>7} \
             {:>11.2} {:>11.2} {:>11.2} {:>11.2} {reps:>6}",
            best_cold / best_cold_clean,
            best_warm / best_clean,
            compiled.loops_compiled,
            compiled.guard_failures,
            per_call,
            loops_per_call,
            aborts_per_call,
            best_clean / elems_per_call as f64,
        );
    }
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
