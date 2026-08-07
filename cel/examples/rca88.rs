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

fn loadavg() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
