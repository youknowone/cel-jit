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

/// Build the three columns of `items.all(i, i.price > 10)` for `rows` rows of
/// exactly `per_row` elements each, in `lowered`'s slot order.
fn list_columns(per_row: i64, rows: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let lens = vec![per_row; rows];
    let mut offsets = Vec::with_capacity(rows);
    let mut total = 0i64;
    for &l in &lens {
        offsets.push(total);
        total += l;
    }
    // `.max(1)` keeps the buffer non-empty at `per_row == 0`, where nothing
    // reads it but a column still has to have a base address.
    let elems = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
    (lens, offsets, elems)
}

/// The nested case: a comprehension over a runtime-length list column, which
/// puts an inner element loop inside the row loop with each back-edge its own
/// `can_enter_jit` point. Sweeping the inner trip count:
///
/// | elements/row | compiles | guard_fails | aborts |
/// |---|---|---|---|
/// | 0 | 1 | 1 | 0 |
/// | 1 | 1 | 1 | 0 |
/// | 2 | 2 | 1 | 0 |
/// | 3 | 2 | 6 | 0 |
/// | 8 | 2 | 9 | 0 |
///
/// At 0 and 1 the inner back-edge is never taken, so there is only one loop.
/// From 2 both loops compile and the batch deopts a constant number of times.
///
/// This used to cost **one deopt per row** from trip count 3 up (3996 / 3999
/// over 4000 rows), on two stacked `majit-metainterp` defects:
///
///  1. From 3 the outer row loop hits the inner merge point TWICE while tracing
///     and closes there — the cross-loop cut (`compile.py:269`), which peels the
///     outer prefix as preamble. That cut was REFUSED at optimize time, its
///     label carrying 3 inputargs against a 29-arg JUMP: the merge-point
///     registration built `original_boxes` from the scalar state fields alone
///     while the close expanded the whole virtualizable, and `VmStateF` is
///     `{ regs: [int; virt], fregs: [float; virt] }` with no scalars at all.
///     Both sides now go through one construction
///     (`JitCodeSym::loop_carried_boxes`, pyjitpl.py:2981-2989).
///  2. The cut then compiled but nothing entered it: it is stored under
///     `green_key_from_code_ptr(state.code_ptr(), pc)`, and
///     `JitState::code_ptr()` defaults to 0 for every `#[jit_interp]`
///     interpreter, so the key is a pc-only hash rather than the
///     `S::green_key([pc, program])` the interpreter presents there. Every row
///     still entered the inner loop and left through its exit guard.
///     `reached_loop_header` (pyjitpl.py:3001-3007) never cuts at a merge point
///     that already holds a compiled loop — it jumps into that loop's procedure
///     token instead — so the dispatch loop now declines the cut there and keeps
///     tracing to its own header, which inlines the inner loop into the outer
///     one. The remaining half of :3001-3007, the JUMP into an already-compiled
///     foreign loop, is still unimplemented (`compile_trace_entry_data` declines
///     an entry-bridge close for `header_pc != 0`); it is not reachable here
///     because declining the cut already closes at the outer header.
///
/// Reproduce the key mismatch with `MAJIT_LOG=1 MAJIT_MPTRACE=1`: compare
/// `add-mp ... inner_key=` against the `start tracing at key=` of the inner
/// loop's own trace.
#[test]
fn nested_list_loop_deopt_census() {
    let _serial = serial();
    let rows = 4_000usize;
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &schema);
    let order: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(
        order,
        ["size(items)", "offset(items)", "items[].price"],
        "column order below assumes this slot order"
    );

    for per_row in [0i64, 1, 2, 3, 8] {
        let (lens, offsets, elems) = list_columns(per_row, rows);
        let columns = [
            Column::Int(&lens),
            Column::Int(&offsets),
            Column::Int(&elems),
        ];
        let (compiles, deopts, aborts, result) = measure(&lowered, &columns, rows);
        eprintln!(
            "[nested] per_row={per_row} rows={rows} compiles={compiles} \
             guard_fails={deopts} aborts={aborts} result={result:?}"
        );
        assert_eq!(aborts, 0, "per_row={per_row}: no trace should be refused");
        assert!(
            deopts <= 16,
            "per_row={per_row}: the batch should deopt a constant number of \
             times, got {deopts} over {rows} rows — that is a per-row bail back \
             to the interpreter"
        );
        let expected_compiles = if per_row < 2 { 1 } else { 2 };
        assert_eq!(
            compiles, expected_compiles,
            "per_row={per_row}: the row loop compiles, and from 2 elements the \
             inner element loop's own back-edge gets hot and compiles too"
        );
    }
}
