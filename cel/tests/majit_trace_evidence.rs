//! Evidence that the majit tier does what a tracing JIT is supposed to do on
//! the REAL cel mainloop: it compiles the hot loop, and the compiled code then
//! RUNS that loop rather than deopting back to the interpreter every iteration.
//!
//! This lives in its own test binary because the evidence counters
//! (`float_bank::COMPILES`, `float_bank::GUARD_FAILS`) are process-global: a
//! concurrently running unit test that drives the same mainloop would land
//! inside another test's reset-run-assert window. Its own binary plus the
//! module-local serial guard gives each measurement an exclusive window.
//!
//! It replaces the old `majit::smoke` mainloop, a second hand-written
//! `#[jit_interp]` register machine that proved the same properties on a
//! synthetic bytecode that evaluated no CEL.

#![cfg(feature = "jit")]

use std::sync::atomic::Ordering;

use cel::majit::bytecode::float_bank::{COMPILES, GUARD_FAILS, TRACE_ABORTS};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// Serializes the measurements, which all reset and then read the global
/// evidence counters. Poison-tolerant so one failure does not cascade.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// `(compiles, guard_fails, result)` for one compiled run of `lowered` over
/// `columns` — with the oracle tier's answer asserted equal first, so a
/// measurement is never taken off a miscompile.
fn measure(lowered: &LoweredF, columns: &[Column], n: usize) -> (usize, usize, usize, Option<i64>) {
    let clean = clean_batch_sum_f(lowered, columns, n);
    COMPILES.store(0, Ordering::Relaxed);
    GUARD_FAILS.store(0, Ordering::Relaxed);
    TRACE_ABORTS.store(0, Ordering::Relaxed);
    let jit = eval_batch_sum_f(lowered, columns, n, 8);
    let compiles = COMPILES.load(Ordering::Relaxed);
    let deopts = GUARD_FAILS.load(Ordering::Relaxed);
    let aborts = TRACE_ABORTS.load(Ordering::Relaxed);
    assert_eq!(clean, jit, "compiled tier diverged from the oracle tier");
    (compiles, deopts, aborts, jit)
}

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

/// A flat per-row predicate: the row loop is the only loop, so once it compiles
/// the trace should run all `n` rows itself and deopt a constant number of times
/// (the loop-exit side exit plus warmup), NOT once per row.
#[test]
fn flat_row_loop_stays_in_compiled_code() {
    let _serial = serial();
    let n = 50_000usize;
    let schema: Schema = [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("price >= 100 && qty < 50", &schema);

    let price: Vec<i64> = (0..n as i64).map(|i| (i * 37) % 200).collect();
    let qty: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 100).collect();
    let columns = [Column::Int(&price), Column::Int(&qty)];

    let (compiles, deopts, aborts, result) = measure(&lowered, &columns, n);
    eprintln!(
        "[flat] n={n} compiles={compiles} guard_fails={deopts} aborts={aborts} result={result:?}"
    );
    assert!(compiles >= 1, "the row loop must compile");
    assert!(
        deopts <= 16,
        "the compiled row loop must run the rows itself (a constant number of \
         side exits), got {deopts} deopts over {n} rows — that is a per-row bail \
         back to the interpreter"
    );
}

/// The same property for the two-bank machine: a `double` column keeps the row
/// loop compiled just as an `int` column does.
#[test]
fn float_row_loop_stays_in_compiled_code() {
    let _serial = serial();
    let n = 50_000usize;
    let schema: Schema = [
        ("price".to_string(), ValType::Float),
        ("qty".to_string(), ValType::Float),
    ]
    .into_iter()
    .collect();
    let lowered = lower("price >= 100.0 && qty < 50.0", &schema);

    let price: Vec<f64> = (0..n).map(|i| (i as f64 * 37.0) % 200.0).collect();
    let qty: Vec<f64> = (0..n).map(|i| (i as f64 * 11.0) % 100.0).collect();
    let columns = [Column::Float(&price), Column::Float(&qty)];

    let (compiles, deopts, aborts, result) = measure(&lowered, &columns, n);
    eprintln!(
        "[float] n={n} compiles={compiles} guard_fails={deopts} aborts={aborts} result={result:?}"
    );
    assert!(compiles >= 1, "the float row loop must compile");
    assert!(
        deopts <= 16,
        "the compiled float row loop must run the rows itself, got {deopts} \
         deopts over {n} rows"
    );
}

/// The nested case, and a KNOWN DEFECT pinned as it currently behaves: a
/// comprehension over a runtime-length list column puts an inner element loop
/// inside the row loop, each back-edge its own `can_enter_jit` point. The inner
/// element loop compiles; the outer row loop traces to `CloseLoop` and is then
/// REFUSED at optimize time:
///
/// ```text
/// abort trace (InvalidLoop: next_iteration_args longer than inputargs
///              (full-body-walk cross-loop cut over a forced heap virtual))
/// abort compile: root loop entry/jump arity mismatch input=3 jump=29
/// ```
///
/// (`majit-metainterp/src/optimizeopt/optimizer.rs`, the `inputarg_type_at`
/// tripwire.) So every row enters the compiled inner loop and leaves it through
/// a guard: one deopt per row instead of one per batch, which is the whole of
/// the measured per-row cost. Reproduce the reason with `MAJIT_LOG=1`.
///
/// If this test starts failing because `aborts` went to 0, the outer loop began
/// compiling — replace the pins below with the `deopts <= 16` bound the flat
/// cases use.
#[test]
fn nested_list_loop_deopt_census() {
    let _serial = serial();
    let rows = 4_000usize;
    let per_row = 8i64;
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &schema);
    let order: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
    eprintln!("[nested] slot order = {order:?}");

    let lens: Vec<i64> = vec![per_row; rows];
    let mut offsets = Vec::with_capacity(rows);
    let mut acc = 0i64;
    for &l in &lens {
        offsets.push(acc);
        acc += l;
    }
    let elems: Vec<i64> = (0..acc).map(|k| (k * 7) % 40).collect();

    let columns: Vec<Column> = order
        .iter()
        .map(|p| match *p {
            "size(items)" => Column::Int(&lens),
            "offset(items)" => Column::Int(&offsets),
            "items[].price" => Column::Int(&elems),
            other => panic!("unexpected slot `{other}`"),
        })
        .collect();

    let (compiles, deopts, aborts, result) = measure(&lowered, &columns, rows);
    eprintln!(
        "[nested] rows={rows} elems={acc} compiles={compiles} guard_fails={deopts} \
         aborts={aborts} result={result:?}  ({:.2} deopts/row)",
        deopts as f64 / rows as f64
    );
    assert_eq!(
        compiles, 1,
        "exactly one of the two loops compiles — the inner element loop"
    );
    assert_eq!(
        aborts, 1,
        "the outer row loop's trace is refused (entry/jump arity mismatch); if \
         this is now 0 the defect is fixed"
    );
    assert!(
        deopts >= rows - 16,
        "the known shape is one deopt per row (the inner loop's exit), got \
         {deopts} over {rows} rows"
    );
}
