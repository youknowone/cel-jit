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

/// The nested case, and a KNOWN DEFECT pinned as it currently behaves.
///
/// A comprehension over a runtime-length list column puts an inner element loop
/// inside the row loop, each back-edge its own `can_enter_jit` point. Sweeping
/// the inner trip count splits the behaviour cleanly at **3**:
///
/// | elements/row | compiles | guard_fails | aborts |
/// |---|---|---|---|
/// | 0 | 1 | 1 | 0 |
/// | 1 | 1 | 1 | 0 |
/// | 2 | **2** | **1** | 0 |
/// | 3 | 2 | 3996 | 1 |
/// | 8 | 2 | 3999 | 1 |
///
/// At 0 and 1 the inner back-edge is never taken, so there is only one loop. At
/// **2 both loops compile** and the whole batch still deopts once — so a
/// compiled inner loop inside a compiled outer loop is not itself the problem.
/// From 3 the outer row loop hits the inner merge point TWICE while tracing, and
/// closes there: the cross-loop cut (`compile.py:269`), which peels the outer
/// prefix as preamble.
///
/// That cut used to be REFUSED at optimize time — its label carried 3 inputargs
/// against a 29-arg JUMP, because the merge-point registration built
/// `original_boxes` from the scalar state fields alone while the close expanded
/// the whole virtualizable. `VmStateF` is `{ regs: [int; virt], fregs: [float;
/// virt] }`, no scalars at all, so the registration fell back to the greens plus
/// one unexpanded vable ref. Fixed in `majit-metainterp`: both sides now go
/// through one construction (`JitCodeSym::loop_carried_boxes`, pyjitpl.py:2981-
/// 2989), and `MAJIT_LOG=1` shows `cut_trace_from: original_boxes=29` against a
/// 29-arg JUMP, compiling where it used to abort — hence `compiles` 1 → 2 here.
///
/// The per-row cost survives that fix, on a SECOND and separate defect. The cut
/// loop is stored under `cut_inner_green_key` =
/// `green_key_from_code_ptr(state.code_ptr(), pc)`, and `JitState::code_ptr()`
/// defaults to 0 for every `#[jit_interp]` interpreter — so the key is a
/// pc-only hash, not the `S::green_key([pc, program])` hash the interpreter
/// presents at that merge point. The outer loop therefore compiles into a key
/// nothing enters, every row still enters the inner loop and leaves through its
/// exit guard, and the bridge attempt from that guard aborts. Reproduce with
/// `MAJIT_LOG=1 MAJIT_MPTRACE=1` and compare `add-mp ... inner_key=` against the
/// `start tracing at key=` of the inner loop's own trace.
///
/// RPython would not take the cut here at all: `reached_loop_header`
/// (pyjitpl.py:3001-3007) first does `ptoken = get_procedure_token(greenboxes)`
/// on the merge point just reached and, when it `has_compiled_targets`, ends the
/// trace with a JUMP into that procedure (`compile_trace`). The inner element
/// loop always compiles first here, so that is the branch the outer trace should
/// be taking; the dispatch loop implements neither it nor a greens-derived cut
/// key.
///
/// If this test starts failing because `deopts` dropped, that second defect is
/// fixed — replace the pins below with the `deopts <= 16` bound the flat cases
/// use.
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
        if per_row < 3 {
            assert_eq!(aborts, 0, "per_row={per_row}: no trace should be refused");
            assert!(
                deopts <= 16,
                "per_row={per_row}: the batch should deopt a constant number of \
                 times, got {deopts} over {rows} rows"
            );
        } else {
            assert_eq!(
                compiles, 2,
                "per_row={per_row}: both the inner element loop and the outer \
                 row loop's cross-loop cut compile; 1 would mean the cut's \
                 label/JUMP arity regressed"
            );
            assert!(
                deopts >= rows - 16,
                "per_row={per_row}: the known shape is one deopt per row (the \
                 inner loop's exit, because the cut loop is keyed where nothing \
                 enters), got {deopts} over {rows} rows"
            );
        }
    }
}
