//! Evidence that the majit tier does what a tracing JIT is supposed to do on
//! the REAL cel mainloop: it compiles the hot loop, and the compiled code then
//! RUNS that loop rather than deopting back to the interpreter every iteration.
//!
//! This lives in its own test binary because the evidence counters behind
//! `float_bank::jit_stats` are process-global: a
//! concurrently running unit test that drives the same mainloop would land
//! inside another test's reset-run-assert window. Its own binary plus the
//! module-local serial guard gives each measurement an exclusive window.
//!
//! It replaces the old `majit::smoke` mainloop, a second hand-written
//! `#[jit_interp]` register machine that proved the same properties on a
//! synthetic bytecode that evaluated no CEL.

#![cfg(feature = "jit")]

use cel::majit::bytecode::float_bank::{
    interned_program_count, jit_stats, reset_jit_stats, reset_persistent_state,
    MAX_INTERNED_PROGRAMS,
};
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
    // Census one data shape at a time. The driver and the interned program now
    // outlive a call, so a second shape of the same expression would reuse the
    // first one's compiled loop, take its exit guard until that guard is hot,
    // and attach a bridge — real behaviour, but not the per-shape trace census
    // these tests exist to pin. `same_expression_second_batch_reuses_the_loop`
    // covers the reuse path instead.
    reset_persistent_state();
    measure_warm(lowered, columns, n)
}

/// [`measure`] without the reset: the driver keeps whatever it compiled for an
/// earlier batch, which is how the tier actually runs.
fn measure_warm(
    lowered: &LoweredF,
    columns: &[Column],
    n: usize,
) -> (usize, usize, usize, Option<i64>) {
    let clean = clean_batch_sum_f(lowered, columns, n);
    reset_jit_stats();
    let jit = eval_batch_sum_f(lowered, columns, n, 8);
    let stats = jit_stats();
    assert_eq!(clean, jit, "compiled tier diverged from the oracle tier");
    // A trace dropped by a panic inside compilation leaves the tier answering
    // out of the interpreter, so every other assertion in this file still holds
    // — the deopt bounds especially, since there is no compiled loop to bail
    // out of. `internal_compile_panics` is the only counter that sees it.
    assert_eq!(
        stats.internal_compile_panics, 0,
        "{} trace(s) were dropped by a panic inside compilation",
        stats.internal_compile_panics
    );
    (
        stats.loops_compiled,
        stats.guard_failures,
        stats.loops_aborted,
        jit,
    )
}

/// One swept data shape: a label and the per-row element count it produces.
type Shape = (&'static str, fn(usize) -> i64);

/// `rlib/jit.py:590` / `warmstate.rs:259 DEFAULT_TRACE_EAGERNESS`.
///
/// A guard does not bridge on its first failure: each failure ticks its own
/// counter by `1/trace_eagerness` and the bridge is attached when that counter
/// crosses 1.0 (`compile.py:783-784`, ported at `pyjitpl.rs:10937` ->
/// `warmstate.rs:1191-1193`). The counter is keyed per guard, so a batch's whole
/// deopt bill is `trace_eagerness` for every guard that ever goes hot — a
/// constant in the ROWS, which is the property a deopt budget exists to pin.
///
/// Budgeting a fixed small number instead asserts that no guard in the shape
/// ever goes hot, which is unreachable for anything that bridges at all. That is
/// what the old constant-16 budgets asserted, and they were recorded when the
/// inner element loop was still inlined into the outer row trace and there was
/// no separate inner-loop exit guard to warm up.
const TRACE_EAGERNESS: usize = 200;

/// Guards that are still mid-warmup when the batch ends: they have ticked
/// without having attached a bridge yet, so they cost failures that
/// `bridges_compiled` does not yet account for.
///
/// 2 is read off the widest shape pinned below rather than chosen. `spread
/// 0..32` attaches 1 bridge at 4000 rows and 3 from 20000 rows up, so exactly
/// two more guards were part-warmed when the short batch ended — 451 deopts
/// against `1 * 200 + 2 * 200 + 1`. Every other shape here needs less.
///
/// It is NOT "a shape with many distinct trip counts leaves a couple", which is
/// the plausible-sounding version and is measured false: the inner loop's exit
/// is one guard however varied the lengths are, so breadth does not multiply
/// guards. Widening the spread makes the count go DOWN, not up — 0..32 gives
/// 451, 0..200 gives 240, 0..2000 gives 204, all at 4000 rows with 1 bridge.
/// What 2 tracks is how far behind `bridges_compiled` runs at a short batch
/// size, nothing about the data's shape.
const WARMING_SLACK: usize = 2;

/// The deopt budget for a batch that attached `bridges` bridges: every failure
/// is either warmup toward some guard's bridge or the batch's own final exit.
///
/// This is independent of the row count by construction, so a per-row bail —
/// the regression these tests exist to catch — blows it at every batch size.
///
/// **That bound has been shown to fire, not just argued to.** Forcing the defect
/// back in by making bridging impossible (a temporary `trace_eagerness` override
/// of 10_000_000 inside `float_bank::new_driver_f`, so no guard's counter can
/// ever cross 1.0 within the batch) turns every guard exit into a raw bail:
///
/// | shape | rows | bridges | deopts | budget | |
/// |---|---|---|---|---|---|
/// | per_row=3 | 4000 | 0 | 3997 | 401 | fires |
/// | per_row=3 | 100000 | 0 | 99997 | 401 | fires |
/// | per_row=8 | 4000 | 0 | 3999 | 401 | fires |
/// | per_row=8 | 100000 | 0 | 99999 | 401 | fires |
/// | cycle 4..12 | 100000 | 0 | 99998 | 401 | fires |
///
/// Deopts land on `rows - 1` to `rows - 3` and scale 1:1 with the batch while the
/// budget does not move at all, so the gate fires by 10x at the smallest size
/// these tests use and by 250x at the largest. Those forced counts also match the
/// signature this shape had when the defect was live — "3996 / 3999 over 4000
/// rows", recorded on `nested_list_loop_deopt_census` below — so what the control
/// reproduces is the real failure mode, not a synthetic one.
///
/// Two other levers were tried first and do NOT reproduce it, which is worth
/// knowing before reaching for them (`examples/rca91.rs` runs all of this):
///
///  * **Widening the trip-count spread**, on the theory that many distinct trip
///    counts means many distinct guards, none reaching `trace_eagerness`. It does
///    not — see [`WARMING_SLACK`].
///  * **`CEL_RETRACE_LIMIT`**, the one pre-existing env knob on this path. At 0
///    (the shipped default, `rlib/jit.py:595`) and 1 the counts are unchanged. At
///    5 they rise to 1401 with 5 bridges, which is *exactly* `warmup_budget(5)`:
///    it passes on equality, **margin zero**. That is a different tier
///    configuration rather than the defect, but it is the tightest this bound has
///    been observed, and it is why the slack is not raised to buy headroom — a
///    fitted constant would hide it, which is what the old 16 was. At 100 a
///    single case runs past 10 minutes.
///
/// **Why a product-side instrument was necessary rather than a shortcut.** A
/// guard bridges after `trace_eagerness` failures, so the batch's whole bill is
/// bounded by `trace_eagerness * n_guards` unless a guard fails *without ever
/// bridging*. This trace has a small fixed number of guards — the inner loop's
/// exit is one guard regardless of the data — so no input shape can multiply
/// them, and the two levers above are the only ones reachable from outside.
/// Reproducing the class therefore requires making bridging itself fail, i.e.
/// `trace_eagerness` above the batch size; nothing public reaches that parameter
/// (`cel/src/majit/bytecode.rs` exposes no `set_param` or driver accessor), so
/// the control has to be built where the driver is. It was added, measured,
/// reverted by content and checksummed against the pre-edit copy.
///
/// Note that `trace_eagerness` is not the only parameter that can render a whole
/// mechanism inert this way: `loop_longevity` is effectively 0 against an
/// upstream default of 1000 (`rlib/jit.py:594`), so compiled loops are never
/// retired at all. That is a separate defect, filed as #106, and it is mentioned
/// here only so a reader of this budget knows the answer to "what else is
/// parameterised like this".
fn warmup_budget(bridges: usize) -> usize {
    TRACE_EAGERNESS * (bridges + WARMING_SLACK) + 1
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
    // A loop that compiled and was never ENTERED passes every other assertion
    // here: the answers still come out of the interpreter, `compiles` is 1, and
    // `deopts` is 0, which is under the upper bound. That was the state for the
    // whole window in which the compiled loop was filed under a green key the
    // back edge does not enter by — the tier was off and this test was green.
    // Entering the loop once and leaving it once at the end of the batch is a
    // guard failure, so the honest floor for a single-entry batch is 1.
    assert!(
        deopts >= 1,
        "nothing entered the compiled row loop over {n} rows: {deopts} guard \
         failures. A loop that compiles and is never entered answers correctly \
         through the interpreter, so only this bound sees it"
    );
    assert!(
        deopts <= 16,
        "the compiled row loop must run the rows itself (a constant number of \
         side exits), got {deopts} deopts over {n} rows — that is a per-row bail \
         back to the interpreter"
    );
}

/// What keeping the driver and the interned program alive across calls buys:
/// a second batch of the same expression finds its loop already compiled and
/// does not compile it again.
///
/// It also pins the correctness half of moving the column bases into registers.
/// The second batch reads a different buffer at a different address through the
/// *same* compiled code, so if a base were still baked into the program words —
/// or promoted, and the trace specialised on it — this batch would answer the
/// first batch's question.
#[test]
fn same_expression_second_batch_reuses_the_loop() {
    let _serial = serial();
    let n = 4_000usize;
    let schema: Schema = [("a".to_string(), ValType::Int)].into_iter().collect();
    let lowered = lower("a > 10", &schema);

    let first: Vec<i64> = (0..n as i64).collect();
    let second: Vec<i64> = (0..n as i64).map(|v| v + 5).collect();

    let (compiles_1, _, aborts_1, r1) = measure(&lowered, &[Column::Int(&first)], n);
    assert_eq!(aborts_1, 0, "first batch: no trace should be refused");
    assert_eq!(compiles_1, 1, "the first batch must compile the row loop");

    let (compiles_2, deopts_2, aborts_2, r2) = measure_warm(&lowered, &[Column::Int(&second)], n);
    eprintln!("[reuse] first={r1:?} second={r2:?} compiles_2={compiles_2} deopts_2={deopts_2}");
    assert_eq!(aborts_2, 0, "second batch: no trace should be refused");
    assert_eq!(
        compiles_2, 0,
        "the second batch must reuse the compiled loop, not compile again"
    );
    assert!(
        deopts_2 <= 16,
        "the reused loop must run the rows itself, got {deopts_2} deopts over {n} rows"
    );
    // Both tiers agreed inside `measure_warm`; this pins that the answers are
    // genuinely the two different columns'.
    assert_ne!(r1, r2, "the two batches must not answer the same question");
}

/// Two expressions of different register shapes get their own pooled drivers,
/// and both must keep answering correctly while they take turns.
///
/// majit publishes the jitcode registry and packed liveness that guard and
/// resume metadata decode through into ONE slot per thread, written when a
/// driver registers its dispatch jitcode — i.e. once, when the driver is built.
/// Holding a driver per shape breaks the invariant that the slot describes the
/// driver about to run, and `JitDriverStaticData::frame_value_count_fn` records
/// that the wrong slot does not fail: it decodes an unrelated jitcode at the
/// same pc and returns a mistyped frame count. `run_jit_persistent_f` therefore
/// re-publishes on the way in.
///
/// The nested shape is the one that stresses it: its trip count changes each
/// round, so its reused driver keeps attaching bridges — compiling *after* the
/// other shape's driver was the last to write the slot.
#[test]
fn alternating_shapes_share_one_thread_state_field_store() {
    let _serial = serial();
    let rows = 4_000usize;

    let nested_schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let nested = lower("items.all(i, i.price > 10)", &nested_schema);

    let flat_schema: Schema = [
        ("a".to_string(), ValType::Int),
        ("b".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let flat = lower("a * 3 + b > 10 && a % 7 != 0", &flat_schema);
    let a: Vec<i64> = (0..rows as i64).collect();
    let b: Vec<i64> = (0..rows as i64).map(|v| (v * 3) % 97).collect();

    reset_persistent_state();
    for per_row in [1i64, 2, 3, 8, 2, 1] {
        let (lens, offsets, elems) = list_columns(per_row, rows);
        // Each `measure_warm` asserts the compiled tier against the oracle, so a
        // frame count decoded through the other shape's store shows up here.
        let (_, _, nested_aborts, nested_r) = measure_warm(
            &nested,
            &[
                Column::Int(&lens),
                Column::Int(&offsets),
                Column::Int(&elems),
            ],
            rows,
        );
        let (_, _, flat_aborts, flat_r) =
            measure_warm(&flat, &[Column::Int(&a), Column::Int(&b)], rows);
        eprintln!(
            "[shapes] per_row={per_row} nested={nested_r:?} ({nested_aborts} abrt) \
             flat={flat_r:?} ({flat_aborts} abrt)"
        );
        // The answers are the assertion: `measure_warm` compares each tier
        // against the oracle, so a frame count decoded through the other shape's
        // store shows up as a divergence there.
        //
        // Aborts are only bounded, not required to be zero. Revisiting a trip
        // count on a driver that has since bridged for other counts does throw
        // traces away — 2 for this sequence — which the cold per-shape census
        // never sees because it starts each shape on a fresh driver. That is a
        // property of the warm regime, not a regression: the flat shape, which
        // never changes its data, aborts nothing.
        assert_eq!(flat_aborts, 0, "per_row={per_row}: flat trace refused");
        assert!(
            nested_aborts <= 4,
            "per_row={per_row}: nested threw away {nested_aborts} traces"
        );
        assert!(
            flat_r.is_some(),
            "per_row={per_row}: the flat shape must keep answering"
        );
    }
}

/// Keeping programs and drivers alive is bounded, not a leak: a thread that
/// keeps evaluating NEW expressions recycles its caches instead of growing one
/// never-freed program and one never-retired compiled loop per expression.
///
/// Each `a > K` interns its own words (the literal rides in the prelude), so
/// this walks past the cap and back round. Every answer is still checked
/// against the oracle, since recycling frees program allocations that compiled
/// loops were keyed on — dropping the drivers in the same breath is what makes
/// that safe.
#[test]
fn interning_is_bounded_and_survives_recycling() {
    let _serial = serial();
    let rows = 64usize;
    let schema: Schema = [("a".to_string(), ValType::Int)].into_iter().collect();
    let a: Vec<i64> = (0..rows as i64).collect();

    reset_persistent_state();
    let mut recycled = false;
    for k in 0..(MAX_INTERNED_PROGRAMS + 8) {
        let lowered = lower(&format!("a > {k}"), &schema);
        let before = interned_program_count();
        let (_, _, aborts, result) = measure_warm(&lowered, &[Column::Int(&a)], rows);
        recycled |= interned_program_count() < before;
        let expected = (0..rows as i64).filter(|v| *v > k as i64).count() as i64;
        assert_eq!(
            result,
            Some(expected),
            "k={k}: wrong answer after recycling"
        );
        assert_eq!(aborts, 0, "k={k}: trace refused");
        assert!(
            interned_program_count() <= MAX_INTERNED_PROGRAMS,
            "k={k}: {} interned programs exceeds the cap",
            interned_program_count()
        );
    }
    assert!(
        recycled,
        "the cap never fired, so this asserted nothing about recycling"
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
/// | elements/row | compiles | bridges | guard_fails | aborts |
/// |---|---|---|---|---|
/// | 0 | 1 | 0 | 1   | 0 |
/// | 1 | 1 | 0 | 1   | 0 |
/// | 2 | 2 | 0 | 1   | 0 |
/// | 3 | 2 | 2 | 401 | 0 |
/// | 8 | 2 | 1 | 201 | 0 |
///
/// At 0 and 1 the inner back-edge is never taken, so there is only one loop.
/// From 2 both loops compile and the batch deopts a constant number of times.
///
/// `guard_fails` is `TRACE_EAGERNESS * bridges + 1` on every row of that table,
/// and `bridges` — not the trip count, not its parity — is the whole
/// discriminator: per_row=2 and per_row=3 both compile two loops and both
/// short-circuit at a varying element, and they differ only in that no guard
/// goes hot at 2 while two do at 3. The counts are identical at 4000, 20000 and
/// 100000 rows on both backends.
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
        let bridges = jit_stats().bridges_compiled;
        eprintln!(
            "[nested] per_row={per_row} rows={rows} compiles={compiles} \
             bridges={bridges} guard_fails={deopts} aborts={aborts} result={result:?}"
        );
        assert_eq!(aborts, 0, "per_row={per_row}: no trace should be refused");
        let budget = warmup_budget(bridges);
        assert!(
            deopts <= budget,
            "per_row={per_row}: the batch should deopt a number of times that is \
             a constant in the rows, got {deopts} over {rows} rows against a \
             warmup budget of {budget} ({bridges} bridge(s) x {TRACE_EAGERNESS} \
             + {WARMING_SLACK} part-warmed + 1 final exit) — that is a per-row \
             bail back to the interpreter"
        );
        let expected_compiles = if per_row < 2 { 1 } else { 2 };
        assert_eq!(
            compiles, expected_compiles,
            "per_row={per_row}: the row loop compiles, and from 2 elements the \
             inner element loop's own back-edge gets hot and compiles too"
        );
    }
}

/// The same nested shape with a trip count that VARIES row to row — the shape
/// real list columns have.
///
/// The constant-trip-count census above is satisfied by the outer trace
/// inlining the inner loop and guarding its trip count, so it says nothing
/// about what happens when that guard is wrong on the next row. Rows of
/// alternating lengths used to deopt on every second row, and a 0..32 spread on
/// most rows:
///
/// | lengths | bridges | guard_fails | aborts | was |
/// |---|---|---|---|---|
/// | 8, 8, 8, …      | 1 | 201 | 0 | 9 |
/// | 8, 9, 8, 9, …   | 1 | 201 | 0 | 50004 / 1 abort |
/// | 4..12 cycling   | 1 | 201 | 0 | 88888 / 8 aborts |
/// | 0..32 spread    | 1 | 451 | 0 | 165621 / 12 aborts |
///
/// (`was` = 100k rows before the bridge fix; the counts here are 4000 rows.)
///
/// Every row is `TRACE_EAGERNESS * bridges + 1` plus the failures of guards not
/// yet warm — the spread's 451 is one attached bridge and two guards still
/// warming, which do attach by 20000 rows. None of it scales with the rows: the
/// same four shapes give the same counts at 20000 and 100000.
/// The exit guard now forms a bridge instead of deopting forever:
/// `#[jit_interp]` states with a `[.. ; virt]` array never rebuilt
/// `virtualizable_boxes` at bridge entry (`pyjitpl.py:3449
/// rebuild_state_after_failure`), so `__trace_*` aborted on its first statement
/// — `standard_virtualizable_jitcode_argbox` had nothing to resolve — and every
/// guard exit fell back to the blackhole.
///
/// The 0..32 spread was bounded at 13000 while the *preamble's* copy of the exit
/// guard still gave the bridge a vable identity that did not resolve to the live
/// state (`compile.py:725-729`). That was the `[.. ; virt]` header synthesis:
/// `extract_live` named the virtualizable once per virt array, so the loop's
/// entry contract carried 29 boxes with refs at 0 and 2 while a bridge's
/// contract — decoded from the guard's vable section — carried 26 with a ref
/// only at 0, and the bridge's JUMP put an int where the preamble guard named
/// the identity. Carrying the virtualizable as ONE slot
/// (`warmspot.py:529-538`, `virtualizable.py:139-144`) makes both contracts 26
/// with a ref only at 0, and the spread drops to 959.
///
/// These counts are deterministic for a given majit revision; they are pinned
/// with a small margin so a regression that reintroduces per-row bailing is
/// caught rather than absorbed by a loose budget.
#[test]
fn nested_list_loop_varying_trip_count() {
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

    let cases: [Shape; 4] = [
        ("constant 8", |_| 8),
        ("alternating 8/9", |r| if r % 2 == 0 { 8 } else { 9 }),
        ("cycle 4..12", |r| 4 + (r % 9) as i64),
        ("spread 0..32", |r| ((r * 2654435761) % 32) as i64),
    ];

    for (label, len_of) in cases {
        let lens: Vec<i64> = (0..rows).map(len_of).collect();
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
        let (compiles, deopts, aborts, result) = measure(&lowered, &columns, rows);
        let bridges = jit_stats().bridges_compiled;
        eprintln!(
            "[varying] {label} rows={rows} compiles={compiles} bridges={bridges} \
             guard_fails={deopts} aborts={aborts} result={result:?}"
        );
        assert_eq!(compiles, 2, "{label}: both loops must compile");
        // A shape whose inner trip count varies must still bridge its exit
        // guard, and `bridges_compiled` is the only counter that says whether it
        // did. Without it a batch that bridges on schedule and one that bails
        // per row are both just "a lot of deopts".
        assert!(
            bridges >= 1,
            "{label}: {deopts} deopts over {rows} rows and not one bridge \
             attached — the exit guard is not bridging"
        );
        let budget = warmup_budget(bridges);
        assert!(
            deopts <= budget,
            "{label}: got {deopts} deopts over {rows} rows against a warmup \
             budget of {budget} ({bridges} bridge(s) x {TRACE_EAGERNESS} + \
             {WARMING_SLACK} part-warmed + 1 final exit) — that is a per-row bail"
        );
    }
}

/// The property that decides whether the tier is a speedup at all: the deopt
/// count must be a WARMUP cost, not a per-row one.
///
/// A budget checked at one batch size cannot tell those apart — 1609 deopts over
/// 4000 rows and 1609 over 200000 rows pass the same `deopts <= 2000`, but the
/// first is a fixed price the batch amortises and the second is a per-row bail
/// that never does. Before the defects above were fixed this shape deopted
/// exactly `rows - 1` times at every size and ran at a flat 0.04–0.05x of the
/// clean VM; afterwards it amortises to 2.8–9.2x by 640k rows
/// (`examples/majit_nested_bench.rs`).
///
/// So this compares the count at two batch sizes 50x apart and requires it to
/// stay essentially flat. It asserts the shape of the curve, not a wall-clock
/// ratio, so it does not flake on a loaded machine the way a timing gate would.
#[test]
fn nested_loop_deopts_are_a_warmup_cost_not_a_per_row_cost() {
    let _serial = serial();
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &schema);

    let cases: [Shape; 4] = [
        ("constant 8", |_| 8),
        ("alternating 8/9", |r| if r % 2 == 0 { 8 } else { 9 }),
        ("cycle 4..12", |r| 4 + (r % 9) as i64),
        ("spread 0..32", |r| ((r * 2654435761) % 32) as i64),
    ];
    const SMALL: usize = 4_000;
    const LARGE: usize = 200_000;

    for (label, len_of) in cases {
        let mut counts = Vec::with_capacity(2);
        for rows in [SMALL, LARGE] {
            let lens: Vec<i64> = (0..rows).map(len_of).collect();
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
            let (_, deopts, _, _) = measure(&lowered, &columns, rows);
            counts.push(deopts);
        }
        let (small, large) = (counts[0], counts[1]);
        eprintln!("[warmup] {label} deopts {SMALL}rows={small} {LARGE}rows={large}");
        // 50x the rows may not cost more than 2x the deopts plus a small slack
        // for the extra lengths a bigger batch happens to present first.
        assert!(
            large <= small * 2 + 64,
            "{label}: {small} deopts over {SMALL} rows but {large} over {LARGE} \
             — the deopt count scales with the batch, so it is a per-row bail \
             back to the interpreter and no batch size can amortise it"
        );
    }
}
