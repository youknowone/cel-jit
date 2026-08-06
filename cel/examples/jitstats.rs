//! Print the majit tier's trace census for a sweep of CEL shapes.
//!
//! The counters exist already — every test in `tests/majit_trace_evidence.rs`
//! resets and reads them — but they were only ever emitted as `eprintln!` from
//! inside a test, which `cargo test` captures and shows on failure alone. So
//! the one question you actually ask of a tracing JIT ("what did it do on my
//! expression, on this backend?") had no way to be asked. This example is that
//! way:
//!
//! ```text
//! cargo run -p cel --release --features jit-dynasm    --example jitstats
//! cargo run -p cel --release --features jit-cranelift --example jitstats
//! ```
//!
//! Run both and diff the tables: the two backends compile the same traces from
//! the same frontend, so a row that differs is a backend divergence, which is
//! the defect — not the smaller number.
//!
//! Every shape is answered by the oracle tier first and asserted equal, so a
//! census is never printed off a miscompile.

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// Which backend `majit-metainterp` was taken with. `cel/Cargo.toml`'s
/// `jit-dynasm` / `jit-cranelift` selectors are the only spelling that names
/// one, and they are cel features, so the choice is readable from here.
const BACKEND: &str = match (
    cfg!(feature = "jit-dynasm"),
    cfg!(feature = "jit-cranelift"),
) {
    // Cargo features are additive, so both-on is reachable; majit then picks,
    // and the table would not say which. Name it rather than guess.
    (true, true) => "dynasm+cranelift (ambiguous)",
    (true, false) => "dynasm",
    (false, true) => "cranelift",
    // A bare `--features jit` does not link: `majit-metainterp/src/pyjitpl.rs`
    // has a `compile_error!` for it. Reaching this arm means the backend came
    // from somewhere else — a workspace feature union, most likely.
    (false, false) => "unnamed (not selected through cel)",
};

/// Compilation threshold used for every census below. Small enough that a
/// 4000-row batch is well past it, so "did not compile" means refused, not cold.
const THRESHOLD: u32 = 8;

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

/// Run one shape from cold and print its row.
///
/// `reset_persistent_state` drops this thread's drivers and interned programs,
/// which is also what zeroes the two driver-sourced counters — so it and
/// `reset_jit_stats` are called together, and the row covers exactly this shape.
fn census(label: &str, src: &str, lowered: &LoweredF, columns: &[Column], n: usize) {
    reset_persistent_state();
    reset_jit_stats();

    let clean = clean_batch_sum_f(lowered, columns, n);
    let jit = eval_batch_sum_f(lowered, columns, n, THRESHOLD);
    assert_eq!(
        clean, jit,
        "{label} (`{src}`): the compiled tier answered {jit:?} where the oracle \
         tier answered {clean:?} — this is a miscompile, not a census"
    );

    let s = jit_stats();
    // A tier that compiled nothing still answers correctly, so nothing else in
    // this run would notice. These two are what notice.
    assert_eq!(
        s.internal_compile_panics, 0,
        "{label}: {} trace(s) were dropped by a panic inside compilation, so the \
         JIT was silently disabled for them",
        s.internal_compile_panics
    );
    assert!(
        s.loops_compiled >= 1,
        "{label}: nothing compiled at threshold {THRESHOLD} over {n} rows"
    );

    println!(
        "{label:<26} {n:>7} {:>6} {:>7} {:>6} {:>6} {:>6} {:>7} {:>6}",
        s.loops_compiled,
        s.bridges_compiled,
        s.loops_aborted,
        s.guard_failures,
        s.internal_compile_panics,
        s.trace_ops_before,
        s.trace_ops_after,
    );
}

/// The three columns of `items.all(i, i.price > 10)` for `rows` rows of exactly
/// `per_row` elements each, in the lowerer's slot order.
fn list_columns(per_row: i64, rows: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let lens = vec![per_row; rows];
    let mut offsets = Vec::with_capacity(rows);
    let mut total = 0i64;
    for &l in &lens {
        offsets.push(total);
        total += l;
    }
    let elems = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
    (lens, offsets, elems)
}

fn main() {
    println!("cel majit trace census — backend: {BACKEND}, threshold {THRESHOLD}");
    println!(
        "{:<26} {:>7} {:>6} {:>7} {:>6} {:>6} {:>6} {:>7} {:>6}",
        "shape", "rows", "loops", "bridges", "abrt", "guard", "panic", "ops_in", "ops_out"
    );

    // Flat int predicate: the row loop is the only loop.
    {
        let n = 50_000usize;
        let schema: Schema = [
            ("price".to_string(), ValType::Int),
            ("qty".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        let src = "price >= 100 && qty < 50";
        let lowered = lower(src, &schema);
        let price: Vec<i64> = (0..n as i64).map(|i| (i * 37) % 200).collect();
        let qty: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 100).collect();
        census(
            "flat/int",
            src,
            &lowered,
            &[Column::Int(&price), Column::Int(&qty)],
            n,
        );
    }

    // The same predicate over the two-bank machine.
    {
        let n = 50_000usize;
        let schema: Schema = [
            ("price".to_string(), ValType::Float),
            ("qty".to_string(), ValType::Float),
        ]
        .into_iter()
        .collect();
        let src = "price >= 100.0 && qty < 50.0";
        let lowered = lower(src, &schema);
        let price: Vec<f64> = (0..n).map(|i| (i as f64 * 37.0) % 200.0).collect();
        let qty: Vec<f64> = (0..n).map(|i| (i as f64 * 11.0) % 100.0).collect();
        census(
            "flat/float",
            src,
            &lowered,
            &[Column::Float(&price), Column::Float(&qty)],
            n,
        );
    }

    // A comprehension over a runtime-length list column: an inner element loop
    // inside the row loop, each back-edge its own `can_enter_jit` point. Swept,
    // because the inner trip count is what decides whether the inner loop
    // compiles at all — below 2 its back-edge is never taken.
    {
        let rows = 4_000usize;
        let schema: Schema = [
            ("size(items)".to_string(), ValType::Int),
            ("offset(items)".to_string(), ValType::Int),
            ("items[].price".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        let src = "items.all(i, i.price > 10)";
        let lowered = lower(src, &schema);
        let order: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(
            order,
            ["size(items)", "offset(items)", "items[].price"],
            "the column order below assumes this slot order"
        );
        for per_row in [1i64, 2, 3, 8] {
            let (lens, offsets, elems) = list_columns(per_row, rows);
            census(
                &format!("nested/all per_row={per_row}"),
                src,
                &lowered,
                &[
                    Column::Int(&lens),
                    Column::Int(&offsets),
                    Column::Int(&elems),
                ],
                rows,
            );
        }
    }

    // The same census through the PUBLIC batch API — `BatchProgram` + `Batch` +
    // `sum_on(Tier::Jit)` — which is the path every benchmark and every caller
    // outside this crate takes. It reaches the mainloop through
    // `run_jit_persistent_f` (a pooled driver) where the section above uses
    // `eval_batch_sum_f` (a fresh one), so a loop that compiles under one and is
    // never entered under the other shows up as a difference between the two
    // halves of this table rather than as a silently slow benchmark.
    println!("\npublic batch API (BatchProgram::compile + Batch + sum_on)");
    println!(
        "{:<26} {:>7} {:>6} {:>7} {:>6} {:>6} {:>6} {:>7} {:>6}",
        "shape", "rows", "loops", "bridges", "abrt", "guard", "panic", "ops_in", "ops_out"
    );
    {
        let n = 50_000usize;
        let a: Vec<i64> = (0..n as i64).map(|i| (i * 37) % 1000).collect();
        let b: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 1000).collect();
        for (label, src) in [
            ("batch/simple_arithmetic", "a + b * 2"),
            ("batch/comparison", "a > b"),
        ] {
            let schema: Schema = [
                ("a".to_string(), ValType::Int),
                ("b".to_string(), ValType::Int),
            ]
            .into_iter()
            .collect();
            let bp = BatchProgram::compile(src, &schema)
                .unwrap_or_else(|e| panic!("{label}: lower `{src}`: {e:?}"));
            let batch = Batch::new(n)
                .column("a", ColumnRef::Int(&a))
                .column("b", ColumnRef::Int(&b));
            let bound = bp.bind(&batch).expect("bind");

            reset_persistent_state();
            reset_jit_stats();
            let clean = bound.sum_on(Tier::Clean).expect("clean tier");
            let jit = bound.sum_on(Tier::Jit).expect("jit tier");
            assert_eq!(
                format!("{clean:?}"),
                format!("{jit:?}"),
                "{label}: compiled tier diverged from the oracle tier"
            );
            let s = jit_stats();
            assert_eq!(
                s.internal_compile_panics, 0,
                "{label}: compilation panicked"
            );
            println!(
                "{label:<26} {n:>7} {:>6} {:>7} {:>6} {:>6} {:>6} {:>7} {:>6}",
                s.loops_compiled,
                s.bridges_compiled,
                s.loops_aborted,
                s.guard_failures,
                s.internal_compile_panics,
                s.trace_ops_before,
                s.trace_ops_after,
            );
        }
    }

    println!(
        "\nops_in/ops_out are trace lengths summed over every compiled loop, \
         before and after the optimizer. A `guard` count near the row count is a \
         per-row bail out of compiled code; a `loops`>=1 with `guard`==0 and no \
         speedup means the loop compiled and nothing entered it."
    );
}
