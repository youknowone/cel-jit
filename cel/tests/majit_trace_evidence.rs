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
//!
//! ⚠ The three address-ownership tests below replaced one that could not fail
//! in any tree: it swept a single comparison, so every program it built shared
//! one `DRIVERS` key. The invariant had no failing-capable test, which is
//! exactly why deleting the mechanism that enforced it looked safe.

#![cfg(feature = "jit")]

use cel::majit::bytecode::float_bank::{
    abort_reasons, abort_reasons_since, guard_census_summary, jit_stats, reset_jit_stats,
    reset_persistent_state, JitStats, MAX_PROGRAMS_PER_DRIVER,
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
    // Census one data shape at a time. The driver outlives a call and the
    // program words outlive it too — the `LoweredF` owns them — so a second
    // shape of the same expression would reuse the first one's compiled loop,
    // take its exit guard until that guard is hot,
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
    LAST_STATS.with(|s| s.set(stats));
    (
        stats.loops_compiled,
        stats.guard_failures,
        stats.loops_aborted,
        jit,
    )
}

thread_local! {
    static LAST_STATS: std::cell::Cell<JitStats> = std::cell::Cell::new(JitStats::default());
}

/// Every counter from the most recent [`measure`] / [`measure_warm`] window.
///
/// Those two return a four-tuple that predates `compiled_entries` and is
/// destructured at fifteen call sites; widening it would edit fourteen tests
/// that do not care. The whole record is kept here instead, so a test can ask
/// for a field the tuple does not carry. The window is the callee's — it resets
/// the counters itself — and the tests are serialized, so the value belongs to
/// the call that just returned.
fn last_measured_stats() -> JitStats {
    LAST_STATS.with(|s| s.get())
}

/// One swept data shape: a label and the per-row element count it produces.
type Shape = (&'static str, fn(usize) -> i64);

/// `rlib/jit.py:590`, read off `majit_metainterp::jit::PARAMETERS` rather than
/// restated: a copy here would keep passing after the engine moved the default,
/// and the budget it feeds would then be wrong by exactly that difference.
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
const TRACE_EAGERNESS: usize = majit_metainterp::jit::PARAMETERS.trace_eagerness as usize;

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

/// One cumulative row from this thread's OpRef-variant audit, for a run
/// launched with `MAJIT_OPREF_VARIANT_AUDIT=1`.
///
/// Print only. The audit measures a defect nobody has reproduced on a crate
/// that compiles a non-degenerate trace, so there is no number to pin yet, and
/// pinning one before measuring certifies whatever state the pin was written
/// in.
///
/// Three properties of the instrument decide the shape of this row.
///
/// `notes` is the reached-ness half of any verdict the module reports:
/// `collisions() == 0` is a result only when `notes() > 0`, because a detector
/// that never ran reports the same zero as a clean one. So the row leads with
/// `notes` and never prints a collision count on its own.
///
/// `enabled()` is here to force the environment read on THIS thread. The mode
/// is resolved lazily — only `note_key` and `enabled` ever populate it — so a
/// thread that never notes never reads `MAJIT_OPREF_VARIANT_AUDIT` at all, and
/// an absent summary then says "not configured" and "not reached" in the same
/// breath. Calling it makes the mode a reading instead of an assumption, and it
/// is the only exercise the environment path gets anywhere: every unit test of
/// the module arms itself with `set_mode_for_test`, which bypasses it.
///
/// [`majit_ir::opref_audit::report_summary`] is called rather than left to the
/// thread-local destructor, on the module's own advice: teardown is
/// best-effort, and a caller that needs the summary should ask for it.
///
/// ⚠ The counters are per thread and are never reset here, so a row is
/// cumulative over the test that prints it.
fn opref_audit_row(label: &str) {
    let enabled = majit_ir::opref_audit::enabled();
    eprintln!(
        "[opref-probe] {label} enabled={enabled} notes={} keys={} \
         revisits_same_variant={} distinct_collisions={} collision_occurrences={}",
        majit_ir::opref_audit::notes(),
        majit_ir::opref_audit::keys_seen(),
        majit_ir::opref_audit::revisits_same_variant(),
        majit_ir::opref_audit::distinct_collisions(),
        majit_ir::opref_audit::collisions(),
    );
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
    //
    // `guard_failures >= 1` used to stand in for entry, on the reasoning that
    // leaving the loop at the end of the batch is a side exit. It is unsound in
    // both directions: once a bridge covers the loop-exit guard the deopt stops
    // being recorded while entry continues every call, so the proxy reads zero
    // on a loop that is entered 50 000 times. `compiled_entries` is the fact
    // itself, counted where the compiled body is about to run.
    let entries = last_measured_stats().compiled_entries;
    eprintln!("[flat] compiled_entries={entries}");
    assert!(
        entries >= 1,
        "nothing entered the compiled row loop over {n} rows. A loop that \
         compiles and is never entered answers correctly through the \
         interpreter, so every other counter here stays plausible"
    );
    assert!(
        deopts <= 16,
        "the compiled row loop must run the rows itself (a constant number of \
         side exits), got {deopts} deopts over {n} rows — that is a per-row bail \
         back to the interpreter"
    );
}

/// Whether a ONE-row batch reaches compiled code on a driver that has already
/// compiled the row loop from a big batch.
///
/// This is the question a single-activation benchmark has to answer before it
/// can print a compiled-tier number, and until `compiled_entries` existed
/// nothing could answer it: the artifact is minted either way, and the answers
/// are identical because the interpreter produces them.
///
/// The answer is still NO, and it is worth being exact about which of two very
/// different reasons now gives it.
///
/// It used to be structural and unconditional. The row loop's back edge was the
/// only door; the loop is bottom-tested, so an `n`-row batch takes `n - 1` back
/// edges and a one-row batch takes none, and nothing a one-row call executed
/// ever consulted the JIT. There is now a second door, ahead of the first
/// instruction and counted per CALL
/// (`float_bank::try_function_entry_jit_f`), and
/// [`repeated_one_row_calls_reach_the_compiled_tier`] is a one-row workload that
/// does reach compiled code through it.
///
/// What shuts it here is the 50 000-row warm-up, deliberately. The door declines
/// for any program whose own loop is already compiled, because on such a program
/// it does not add a way in — it takes one away. Measured on
/// `items.all(i, i.price > 10)` over repeated 20 000-row calls: with the decline
/// removed, an entry artifact was minted and from then on every call entered the
/// ENTRY key exactly once and the loop header not at all, where before it
/// entered the loop header and ran the batch there. Both answer correctly, and
/// `majit_shape_change`'s "never worse than the untraced VM" bound fired on the
/// difference.
///
/// So this pins the boundary between the two doors rather than the absence of
/// one, and a mixed workload — big batches and single rows through one driver —
/// gets the row loop's door only. Lifting that would need the two artifacts to
/// coexist without the entry one displacing the loop's, which is a majit-side
/// question, not one this crate can answer by keying differently.
#[test]
fn a_compiled_row_loop_shuts_the_entry_door() {
    let _serial = serial();
    let schema: Schema = [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("price >= 100 && qty < 50", &schema);
    let n = 50_000usize;
    let price: Vec<i64> = (0..n as i64).map(|i| (i * 37) % 200).collect();
    let qty: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 100).collect();

    // Warm the pooled driver until the row loop is compiled and entered.
    reset_persistent_state();
    reset_jit_stats();
    let warm_cols = [Column::Int(&price), Column::Int(&qty)];
    let _ = eval_batch_sum_f(&lowered, &warm_cols, n, 8);
    let warm = jit_stats();
    assert!(
        warm.loops_compiled >= 1,
        "warm-up must compile the row loop"
    );
    assert!(warm.compiled_entries >= 1, "warm-up must enter it");

    // Same driver, same expression, one row — for longer than the entry door's
    // threshold, so a door that were open would be warm several times over.
    const ONE_ROW_CALLS: usize = 64;
    let one_cols = [Column::Int(&price[..1]), Column::Int(&qty[..1])];
    reset_jit_stats();
    for _ in 0..ONE_ROW_CALLS {
        assert_eq!(
            eval_batch_sum_f(&lowered, &one_cols, 1, 8),
            clean_batch_sum_f(&lowered, &one_cols, 1),
            "the one-row batch must answer what the oracle tier answers"
        );
    }
    let one = jit_stats();
    eprintln!(
        "[one-row] warm_entries={} one_row_entries={} one_row_compiles={}",
        warm.compiled_entries, one.compiled_entries, one.loops_compiled
    );
    assert_eq!(
        one.compiled_entries, 0,
        "a one-row batch entered compiled code {} time(s) on a driver whose row \
         loop is already compiled. The entry door declines exactly there, and \
         the decline is what keeps a batch workload entering its row loop; if \
         this fires, the door opened on a program that has a hot loop and the \
         batch tier's ns/row is the thing to re-measure",
        one.compiled_entries
    );
    assert_eq!(
        one.loops_compiled, 0,
        "{ONE_ROW_CALLS} one-row calls minted {} more loop(s) on a driver whose \
         row loop is compiled. Nothing should trace here at all",
        one.loops_compiled
    );
}

/// A program called over and over with ONE row per call — the way an expression
/// evaluated per record is used — reaches the compiled tier from cold, with no
/// big batch anywhere in its history.
///
/// The row-loop back edge cannot produce this: at one row it is never taken, so
/// its counter never moves and the workload stays in the interpreter for as many
/// calls as it is given. The measurement here is the whole point of counting at
/// the entry door instead.
///
/// Each call binds DIFFERENT values, so the answers are not one answer repeated:
/// a compiled artifact that had specialised on the first call's bindings — baked
/// them as constants rather than reading them as loop-invariant inputs — would
/// answer the first call's question for every later one, and the oracle
/// comparison per call is what catches that.
#[test]
fn repeated_one_row_calls_reach_the_compiled_tier() {
    let _serial = serial();
    let schema: Schema = [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("price >= 100 && qty < 50", &schema);

    reset_persistent_state();
    reset_jit_stats();

    const CALLS: usize = 256;
    for i in 0..CALLS as i64 {
        let price = [(i * 37) % 200];
        let qty = [(i * 11) % 100];
        let columns = [Column::Int(&price), Column::Int(&qty)];
        let jit = eval_batch_sum_f(&lowered, &columns, 1, 8);
        assert_eq!(
            jit,
            clean_batch_sum_f(&lowered, &columns, 1),
            "call {i} diverged from the oracle tier"
        );
    }

    let stats = jit_stats();
    eprintln!("[percall] calls={CALLS} {stats}");
    assert_eq!(
        stats.internal_compile_panics, 0,
        "a trace was dropped by a panic inside compilation, so the tier answered \
         out of the interpreter and the entry count below measures nothing"
    );
    assert!(
        stats.loops_compiled >= 1,
        "{CALLS} one-row calls must compile something: the entry door counts \
         calls, and nothing else in this workload counts at all"
    );
    assert!(
        stats.compiled_entries >= 1,
        "{CALLS} one-row calls compiled {} loop(s) and entered none. An artifact \
         that exists and is never entered leaves every answer coming out of the \
         interpreter, which is the state this door was added to end",
        stats.loops_compiled
    );
}

/// The entry door must not read its OWN artifact as evidence that another door
/// exists.
///
/// `float_bank::loop_header_keys` finds a program's loop headers by scanning for
/// `OP_JUMP_IF_ABOVE`'s value word-wise, so an OPERAND holding that value
/// contributes a position that is not an instruction. That is harmless for every
/// spurious position but one: `ENTRY_PC`, which is the position the entry door
/// itself arms at. `[1, 2, 3, 4, 5].map(x, x * 2)` produces exactly that — the
/// unrolled body's `OP_MUL_OVF` reads register 16 (`OP_JUMP_IF_ABOVE`'s value)
/// and traps to register 0, so its four words read as a back edge to 0 — and the
/// door then declines from the call after the one that minted its artifact,
/// forever. Measured before the exclusion: `loops_compiled=1`,
/// `compiled_entries=0` over 4096 calls, every answer out of the interpreter.
///
/// The expression is load-bearing and not an example: it is one of the few
/// spellings measured to fail. What selects the defect is a numeric
/// coincidence in the word stream — an operand holding 16 with a 0 three words
/// later — and no property of the source decides that. `.map` over a literal
/// unrolls on a six-register stride from a base of 4, so its THIRD element
/// takes first operand 16 and its `OP_MUL_OVF` traps to register 0; three
/// elements already suffice, and `[1, 2].map(x, x * 2)` stops at 10 and is
/// healthy. Neighbours that look like they should fail do not: `.all` over a
/// literal does not unroll at all, and `.filter` with the same multiply
/// allocates on a different stride and lands its 16 three words ahead of an
/// 18. `examples/rca_listmap.rs` runs that whole census.
#[test]
fn a_spurious_back_edge_to_entry_pc_does_not_shut_the_entry_door() {
    use cel::majit::batch::{Batch, BatchProgram, Tier};

    let _serial = serial();
    let schema: Schema = Schema::new();
    let lowered = BatchProgram::compile("[1, 2, 3, 4, 5].map(x, x * 2)", &schema)
        .expect("a literal-list map lowers");
    let batch = Batch::new(1);
    let bound = lowered.bind_per_row(&batch).expect("no columns to bind");
    let oracle = bound
        .collect_on(Tier::Clean)
        .expect("the clean tier answers");

    reset_persistent_state();
    reset_jit_stats();

    const CALLS: usize = 64;
    for call in 0..CALLS {
        let jit = bound.collect_on(Tier::Jit).expect("the jit tier answers");
        assert_eq!(jit, oracle, "call {call} diverged from the oracle tier");
    }

    let stats = jit_stats();
    eprintln!("[entry-pc-alias] calls={CALLS} {stats}");
    assert_eq!(
        stats.internal_compile_panics, 0,
        "a trace was dropped by a panic inside compilation, so the entry count \
         below measures nothing"
    );
    assert!(
        stats.loops_compiled >= 1,
        "{CALLS} one-row calls must compile something at the entry door"
    );
    assert!(
        stats.compiled_entries >= 1,
        "{CALLS} one-row calls compiled {} artifact(s) and entered none. The \
         entry key is in this program's scanned loop-header keys, so the door's \
         `has_compiled_loop` decline fires on the artifact the door itself just \
         minted",
        stats.loops_compiled
    );
}

/// What keeping the driver alive across calls, over a program the lowering
/// itself owns, buys:
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

/// Program identity is an ownership property, not a cache policy. The green key
/// stores the code pointer AS A NUMBER, so two programs that ever hold one
/// address build byte-identical keys and `comparekey` compares them equal — a
/// collision no lookup can resolve, because there is nothing left to compare.
/// What rules it out is that the [`LoweredF`] builds its words once and owns
/// them: the address cannot move while it is alive, and cannot be handed to a
/// second program that is also alive.
///
/// Both halves are asserted here. Every answer is still checked against the
/// oracle, since a program that moved under a live compiled loop surfaces as a
/// wrong answer rather than as a crash.
#[test]
fn a_programs_address_is_stable_and_two_programs_never_share_one() {
    let _serial = serial();
    let rows = 512usize;
    let schema: Schema = [("a".to_string(), ValType::Int)].into_iter().collect();
    let a: Vec<i64> = (0..rows as i64).collect();
    let count_gt = |k: i64| (0..rows as i64).filter(|v| *v > k).count() as i64;

    reset_persistent_state();

    // One expression, many warm batches: the address the tier binds must not
    // move, including across the first batch, which is what builds the words.
    let lowered = lower("a > 7", &schema);
    let first = lowered.batch_sum_shape(true).code.as_ptr() as usize;
    let mut compiled = false;
    for i in 0..8 {
        let (compiles, _, aborts, result) = measure_warm(&lowered, &[Column::Int(&a)], rows);
        compiled |= compiles >= 1;
        assert_eq!(result, Some(count_gt(7)), "batch {i}: wrong answer");
        assert_eq!(aborts, 0, "batch {i}: trace refused");
        assert_eq!(
            lowered.batch_sum_shape(true).code.as_ptr() as usize,
            first,
            "batch {i}: the program words moved under a live compiled loop"
        );
    }
    // Non-vacuity: an address nothing ever keyed on is stable for free. Only a
    // batch that actually compiled took a green key over this pointer.
    assert!(
        compiled,
        "nothing compiled, so no green key was ever taken over this address"
    );

    // Distinct expressions, all held alive at once: no two may be handed the
    // same address. This is the half a cap cannot provide — it freed program
    // words while the compiled loops keyed on them were still reachable.
    let mut live = Vec::new();
    let mut addrs = std::collections::HashSet::new();
    for k in 0..32i64 {
        let lowered = lower(&format!("a > {k}"), &schema);
        let (_, _, aborts, result) = measure_warm(&lowered, &[Column::Int(&a)], rows);
        assert_eq!(result, Some(count_gt(k)), "k={k}: wrong answer");
        assert_eq!(aborts, 0, "k={k}: trace refused");
        let shape = lowered.batch_sum_shape(true);
        assert!(!shape.code.is_empty(), "k={k}: empty program");
        assert!(
            addrs.insert(shape.code.as_ptr() as usize),
            "k={k}: this address is already held by another live program"
        );
        live.push(lowered);
    }
    assert_eq!(
        addrs.len(),
        live.len(),
        "every live program must hold its own address"
    );
}

/// The half the check above cannot see, and the one that actually bites: a
/// program that has **died**, whose address the allocator then hands to a
/// program with DIFFERENT words.
///
/// Distinctness among programs held alive together is free — the allocator
/// guarantees it. The green key names an address, the compiled loop it names
/// lives in `DRIVERS`, and `DRIVERS` outlives every `LoweredF`. So the question
/// is not whether two live programs can share an address; it is whether a
/// compiled loop can outlive the program its key names. Nothing retires it
/// today (`memmgr`'s `max_age` is 0), so the only thing that can rule it out is
/// an owner that outlives the key.
///
/// The sweep alternates the comparison as well as the literal, and both matter.
/// Over this schema every one of these lowers to 44 words differing in exactly
/// two positions: index 13 carries the literal and index 19 the comparison
/// opcode (`7` for `>`, `9` for `<`). Alternating both keeps the *seeded*
/// register shape identical — which is what `DRIVERS` is keyed on
/// (`bytecode.rs`, `(init_regs.len(), num_fregs, threshold)`) — so every program
/// here lands on ONE driver and can therefore reach another program's compiled
/// loop. Varying the arity instead (adding a conjunct, hence a second scalar
/// seed) splits the driver key and the collision cannot occur, which makes for
/// a test that passes without ever exercising the hazard.
///
/// Each round builds a program, runs it warm, and drops it before the next is
/// built — a host evaluating generated expressions one at a time. Every answer
/// is checked against the oracle, because this fault does not crash: it
/// silently answers an EARLIER expression's question. Without the pairing this
/// test fails on the second program, `a < 1` returning 509 — which is `a > 2`'s
/// answer over this column.
#[test]
fn a_dead_programs_address_does_not_carry_its_compiled_loop() {
    let _serial = serial();
    let rows = 512usize;
    let schema: Schema = [("a".to_string(), ValType::Int)].into_iter().collect();
    let a: Vec<i64> = (0..rows as i64).collect();

    const PROGRAMS: i64 = 24;

    reset_persistent_state();
    let mut addrs = std::collections::HashSet::new();
    let mut compiled = false;
    for k in 0..PROGRAMS {
        let (src, expected) = if k % 2 == 0 {
            let want = (0..rows as i64).filter(|v| *v > k).count() as i64;
            (format!("a > {k}"), want)
        } else {
            let want = (0..rows as i64).filter(|v| *v < k).count() as i64;
            (format!("a < {k}"), want)
        };
        // `lowered` dies at the end of this block, before the next is built.
        let addr = {
            let lowered = lower(&src, &schema);
            for round in 0..3 {
                let (compiles, _, aborts, result) =
                    measure_warm(&lowered, &[Column::Int(&a)], rows);
                compiled |= compiles >= 1;
                assert_eq!(
                    result,
                    Some(expected),
                    "`{src}` round {round}: wrong answer"
                );
                assert_eq!(aborts, 0, "`{src}` round {round}: trace refused");
            }
            lowered.batch_sum_shape(true).code.as_ptr() as usize
        };
        addrs.insert(addr);
    }
    // The invariant itself: every program got its OWN address even though each
    // was dropped before the next was built. That can only hold because the
    // driver that could still name them is holding the words — drop that edge
    // and the allocator hands the same address straight back, which is how this
    // test fails without the pairing (it fails twice over: a reused address
    // here, and a wrong answer above, on the second program).
    assert_eq!(
        addrs.len() as i64,
        PROGRAMS,
        "an address was handed to a second program while a driver could still \
         name the first"
    );
    // Non-vacuity, in the role `interned_program_count() < before` played:
    // addresses nothing ever keyed on stay distinct for free. Only a batch that
    // actually compiled put a green key on one.
    assert!(
        compiled,
        "nothing compiled, so no green key was ever taken over these addresses"
    );
}

/// Holding every program a driver has keyed on is unbounded on its own, so the
/// pool flushes a driver that passes [`MAX_PROGRAMS_PER_DRIVER`]. This is that
/// flush actually firing.
///
/// It has to be tested precisely because it is the kind of mechanism that goes
/// inert without moving a counter: nothing else in this file builds enough
/// distinct programs to reach it, so an off-by-one or a check on the wrong side
/// of the reinsert would leave the bound looking present and never running.
///
/// The observable is that a program which had already compiled has to compile
/// AGAIN once the flush drops the driver holding its loop. That also pins the
/// half that matters: the loops and the words go together, so a survivor is
/// impossible in either direction.
#[test]
fn the_per_driver_program_cap_flushes_loops_and_words_together() {
    let _serial = serial();
    let rows = 64usize;
    let schema: Schema = [("a".to_string(), ValType::Int)].into_iter().collect();
    let a: Vec<i64> = (0..rows as i64).collect();
    let expected = Some((0..rows as i64).filter(|v| *v > 7).count() as i64);

    reset_persistent_state();
    let pinned = lower("a > 7", &schema);
    let (first, _, _, result) = measure_warm(&pinned, &[Column::Int(&a)], rows);
    assert_eq!(result, expected, "pinned program: wrong answer");
    assert!(
        first >= 1,
        "the pinned program must compile on its first batch"
    );
    let (again, _, _, result) = measure_warm(&pinned, &[Column::Int(&a)], rows);
    assert_eq!(result, expected, "pinned program: wrong answer while warm");
    assert_eq!(
        again, 0,
        "a warm driver recompiled a program it had already compiled, so the \
         control for the flush below does not discriminate"
    );

    // Walk that same driver past its cap. One scalar seed each, exactly like
    // the pinned program, so every one of these lands on its driver rather
    // than minting a second.
    for k in 1000..(1000 + MAX_PROGRAMS_PER_DRIVER as i64 + 8) {
        let filler = lower(&format!("a > {k}"), &schema);
        let (_, _, aborts, r) = measure_warm(&filler, &[Column::Int(&a)], rows);
        assert_eq!(r, Some(0), "filler a > {k}: wrong answer");
        assert_eq!(aborts, 0, "filler a > {k}: trace refused");
    }

    let (after, _, _, result) = measure_warm(&pinned, &[Column::Int(&a)], rows);
    assert_eq!(
        result, expected,
        "pinned program: wrong answer after the flush"
    );
    assert!(
        after >= 1,
        "the cap never fired: the pinned loop survived {} further programs on \
         its own driver",
        MAX_PROGRAMS_PER_DRIVER + 8
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
        // `aborts` is a bare count; the reason lives only in majit's MC_DIAG
        // slots. Snapshot across this one shape so a non-zero count arrives
        // already attributed instead of sending the next reader on a sweep.
        let reasons_before = abort_reasons();
        let (compiles, deopts, aborts, result) = measure(&lowered, &columns, rows);
        let reasons = abort_reasons_since(&reasons_before);
        let bridges = jit_stats().bridges_compiled;
        eprintln!(
            "[nested] per_row={per_row} rows={rows} compiles={compiles} \
             bridges={bridges} guard_fails={deopts} aborts={aborts} result={result:?} \
             abort_reasons=[{reasons}]"
        );
        opref_audit_row(&format!("per_row={per_row}"));
        assert_eq!(
            aborts, 0,
            "per_row={per_row}: no trace should be refused (reasons: [{reasons}])"
        );
        // `bridges` has no callback: majit keeps the tally on the driver, so a
        // reader that only sums drivers it has already absorbed at their end of
        // life reports zero for a pool whose drivers are all still alive. The
        // budget below cannot notice that — a zero only makes it TIGHTER, and
        // 201 deopts still fit inside the 401 that `bridges = 0` allows — and
        // `allocs_per_eval` asserts `bridges == 0`, so it passes on a dead
        // reader too. This is the failing-capable direction.
        //
        // The pin is a biconditional rather than `bridges > 0`, because three of
        // these five shapes attach no bridge and are right not to: `deopts == 1`
        // is the loop's single final exit, with no guard failing repeatedly for
        // anything to bridge. Past that, a guard that keeps failing stops only
        // when a bridge attaches to it — `aborts == 0` is asserted above, so the
        // other way for it to keep failing is excluded here.
        assert_eq!(
            bridges > 0,
            deopts > 1,
            "per_row={per_row}: {deopts} deopt(s) and {bridges} bridge(s) — a \
             guard that failed more than the one final exit has a bridge \
             attached to it, and only a bridge stops it failing"
        );
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
    majit_ir::opref_audit::report_summary();
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
        opref_audit_row(label);
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
    majit_ir::opref_audit::report_summary();
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
            // Under MAJIT_GUARD_CENSUS this separates the two shapes a deopt
            // TOTAL cannot: one guard with no bridge attached versus a spread
            // of cold guards each below trace_eagerness. Cumulative and with no
            // reset, so read the increment between consecutive lines.
            eprintln!(
                "[warmup-census] {label} rows={rows} deopts={deopts} {}",
                guard_census_summary(4)
            );
            counts.push(deopts);
        }
        let (small, large) = (counts[0], counts[1]);
        eprintln!("[warmup] {label} deopts {SMALL}rows={small} {LARGE}rows={large}");
        // 50x the rows may not cost more than 2x the deopts plus a small slack
        // for the extra lengths a bigger batch happens to present first.
        let allowed = small * 2 + 64;
        // Report the PER-ROW RATE, not just the counts. Exceeding the budget
        // does not by itself say which of the two costs this is, and the rate
        // is what separates them: a rate that holds is a per-row bail, a rate
        // that falls is warmup that has not finished amortising. Reading a
        // failure as the former on the strength of the count alone is wrong
        // whenever the rate fell.
        let small_rate = small as f64 / SMALL as f64;
        let large_rate = large as f64 / LARGE as f64;
        assert!(
            large <= allowed,
            "{label}: deopts {small} over {SMALL} rows -> {large} over {LARGE}, \
             past the {allowed} this gate allows. Per-row rate {small_rate:.4} \
             -> {large_rate:.4} — a rate that HOLDS is a per-row bail that no \
             batch size amortises; a rate that FALLS is warmup still amortising, \
             which has not flattened by {LARGE} rows the way the other shapes \
             do. The rate says which; the count alone cannot."
        );
    }
}

/// A user function in scalar form is a residual call in the trace, not an
/// abort: the row loop compiles, is entered, and answers what the clean tier
/// answers — on the int convention and on the float one, whose values cross
/// the call as bits. Without this the parity test in `batch.rs` could pass
/// with the JIT tier quietly answering out of the interpreter.
#[test]
fn a_host_call_loop_compiles_and_is_entered() {
    use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
    use cel::Context;
    let _serial = serial();
    let mut ctx = Context::default();
    ctx.add_function("add", |a: i64, b: i64| a + b);
    ctx.add_function("multiply", |a: i64, b: i64| a * b);
    ctx.add_function("scale", |a: f64, b: f64| a * b + 0.5);
    ctx.add_function("half", |a: f64| a / 2.0);
    let schema: Schema = [
        ("x", ValType::Int),
        ("y", ValType::Int),
        ("f", ValType::Float),
    ]
    .into_iter()
    .map(|(p, t)| (p.to_string(), t))
    .collect();
    let n = 50_000usize;
    let xs: Vec<i64> = (0..n as i64).collect();
    let ys: Vec<i64> = (0..n as i64).map(|i| i * 3 - 7).collect();
    let fs: Vec<f64> = (0..n as i64).map(|i| i as f64 * 0.25).collect();
    let batch = Batch::new(n)
        .column("x", ColumnRef::Int(&xs))
        .column("y", ColumnRef::Int(&ys))
        .column("f", ColumnRef::Float(&fs));
    for src in ["add(x, y) + multiply(x, 3)", "scale(f, 2.0) + half(f)"] {
        let program = Program::compile(src).unwrap();
        let batched = BatchProgram::from_program_in(&program, &schema, &ctx).unwrap();
        let bound = batched.bind_per_row(&batch).unwrap();

        reset_persistent_state();
        reset_jit_stats();
        let before = abort_reasons();
        let clean = bound.collect_on(Tier::Clean).unwrap();
        let jit = bound.collect_on(Tier::Jit).unwrap();
        let stats = jit_stats();
        assert_eq!(
            clean, jit,
            "{src}: compiled tier diverged from the clean tier"
        );
        assert_eq!(stats.internal_compile_panics, 0, "{src}");
        assert_eq!(
            stats.loops_aborted,
            0,
            "{src}: the host call aborted the trace: {}",
            abort_reasons_since(&before)
        );
        assert!(
            stats.loops_compiled >= 1,
            "{src}: the host-call loop did not compile: {stats:?}"
        );
        assert!(
            stats.compiled_entries >= 1,
            "{src}: compiled but never entered: {stats:?}"
        );
    }
}

/// A power-of-two modulus is one mask in the compiled loop, not a residual
/// call per element.
///
/// `OP_MOD_CHK`'s dispatch arm computes the remainder on unsigned magnitudes
/// through the `majit_uint_mod` helper, and a helper is a residual call in the
/// trace -- `filter(x, x % 2 == 0)` paid one `CallI` per element, which is
/// what held `filter_list_scaling/10000` at 2.85x `map_list_scaling`'s
/// per-element cost (2.47 vs 0.86 ns/elem; 2.03x once the call was gone --
/// `examples/rca_filtercost.rs` is the instrument). The arm now answers a
/// power-of-two magnitude with a mask, so the loop keeps no call. The data
/// runs every sign combination and both `i64` extremes through the mask arm,
/// with the clean tier as the oracle.
///
/// The divisor arrives in a COLUMN, not as a literal: a literal divisor lowers
/// to `OP_MOD_CHK_K` instead, whose immediate is green and whose whole dispatch
/// therefore folds while tracing (see
/// `a_constant_divisor_keeps_no_residual_call_in_its_loop`). This arm is what a
/// divisor the lowering cannot see runs on, and the mask is what it buys there.
#[test]
fn a_power_of_two_modulus_filter_keeps_no_residual_call_in_its_loop() {
    use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
    let _serial = serial();
    let mut xs: Vec<i64> = (-17..=17).collect();
    xs.extend([i64::MIN, i64::MIN + 1, i64::MAX, i64::MAX - 1]);
    let lens = vec![xs.len() as i64];
    let schema: Schema = [
        ("xs[]".to_string(), ValType::Int),
        ("d".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    // The last divisor is NOT a power of two: it keeps the helper call, which
    // is what proves the assertion below can fail.
    let cases = [(2i64, true), (8, true), (-4, true), (1, true), (3, false)];
    let src = "xs.filter(x, x % d == 0)";
    for (divisor, mask_only) in cases {
        let ds = [divisor];
        let batch = Batch::new(1)
            .column(
                "xs".to_string(),
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Int(&xs))],
                },
            )
            .column("d".to_string(), ColumnRef::Int(&ds));
        let lowered = BatchProgram::compile(src, &schema).expect(src);
        let bound = lowered.bind_per_row(&batch).expect(src);
        reset_persistent_state();
        reset_jit_stats();
        let clean = bound.collect_on(Tier::Clean).unwrap();
        for _ in 0..64 {
            let jit = bound.collect_on(Tier::Jit).unwrap();
            assert_eq!(clean, jit, "{src}: compiled tier diverged from clean");
        }
        let stats = jit_stats();
        assert_eq!(stats.internal_compile_panics, 0, "d={divisor}");
        assert_eq!(
            stats.loops_aborted, 0,
            "d={divisor}: the modulus aborted the trace"
        );
        assert!(
            stats.loops_compiled >= 1,
            "d={divisor}: the filter loop did not compile: {stats:?}"
        );
        let log = majit_metainterp::embed::Census::compiled_opcode_log();
        assert!(
            !log.is_empty(),
            "d={divisor}: compiled but the opcode log is empty"
        );
        let calls: usize = log
            .iter()
            .flatten()
            .filter(|op| format!("{op:?}").starts_with("Call"))
            .count();
        if mask_only {
            assert_eq!(
                calls, 0,
                "d={divisor}: a power-of-two modulus left a residual call in the loop"
            );
        } else {
            assert!(
                calls > 0,
                "d={divisor}: the non-power-of-two control lost its helper call, \
                 so the zero-call assertion above can no longer fail"
            );
        }
    }
}

/// A CONSTANT divisor leaves no residual call on either integer bank.
///
/// `program` is a green argument of the `#[jit_interp]` mainloop, so a divisor
/// carried as an IMMEDIATE word is a constant to the trace optimizer, which
/// expands the division into multiply-and-shift (`optimize_call_int_py_div` /
/// `_py_mod`). The same value in a register is not: the prelude loads it once,
/// outside the row loop, so it reaches the body as an opaque loop-invariant and
/// leaves one `int.udiv`/`int.umod` residual call per element -- which
/// `examples/rca_callcensus.rs` measured on EVERY division in its corpus,
/// power-of-two divisors included. The immediate forms are `OP_DIV_CHK_K` /
/// `OP_MOD_CHK_K` / `OP_UDIV_K` / `OP_UMOD_K`; the register forms stay for a
/// divisor the lowering cannot see, and the last case here is one.
///
/// The census fixture holds ordinary magnitudes. The `|i64::MIN|` dividend and
/// a `uint` at or above 2^63 both read NEGATIVE in the int bank, where the
/// expanded division would answer for the wrong sign, so those keep the helper
/// call by design -- the second fixture runs them for the ANSWER, not for the
/// op census.
#[test]
fn a_constant_divisor_keeps_no_residual_call_in_its_loop() {
    use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
    let _serial = serial();
    let xs: Vec<i64> = (-17..=17).collect();
    let us: Vec<u64> = (0..=34u64).collect();
    let lens = vec![xs.len() as i64];
    let ds = [3i64];
    let schema: Schema = [
        ("xs[]".to_string(), ValType::Int),
        ("us[]".to_string(), ValType::UInt),
        ("d".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let batch = Batch::new(1)
        .column(
            "xs".to_string(),
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&xs))],
            },
        )
        .column(
            "us".to_string(),
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::UInt(&us))],
            },
        )
        .column("d".to_string(), ColumnRef::Int(&ds));
    let cases = [
        ("xs.map(x, x / 2)", true),
        ("xs.map(x, x / 3)", true),
        ("xs.map(x, x / -3)", true),
        ("xs.map(x, x % 3)", true),
        ("xs.map(x, x % -7)", true),
        ("us.map(u, u / 2u)", true),
        ("us.map(u, u / 3u)", true),
        ("us.map(u, u % 3u)", true),
        // The divisor is a COLUMN, so the lowering cannot put it in the stream
        // and the register form runs. Its helper call is what proves the
        // zero-call assertions above can fail.
        ("xs.map(x, x / d)", false),
    ];
    for (src, immediate) in cases {
        let lowered = BatchProgram::compile(src, &schema).expect(src);
        let bound = lowered.bind_per_row(&batch).expect(src);
        reset_persistent_state();
        reset_jit_stats();
        let clean = bound.collect_on(Tier::Clean).unwrap();
        for _ in 0..64 {
            let jit = bound.collect_on(Tier::Jit).unwrap();
            assert_eq!(clean, jit, "{src}: compiled tier diverged from clean");
        }
        let stats = jit_stats();
        assert_eq!(stats.internal_compile_panics, 0, "{src}");
        assert_eq!(
            stats.loops_aborted, 0,
            "{src}: the division aborted the trace"
        );
        assert!(
            stats.loops_compiled >= 1,
            "{src}: the loop did not compile: {stats:?}"
        );
        let log = majit_metainterp::embed::Census::compiled_opcode_log();
        assert!(
            !log.is_empty(),
            "{src}: compiled but the opcode log is empty"
        );
        let calls: usize = log
            .iter()
            .flatten()
            .filter(|op| format!("{op:?}").starts_with("Call"))
            .count();
        if immediate {
            assert_eq!(
                calls, 0,
                "{src}: a constant divisor left a residual call in the loop"
            );
        } else {
            assert!(
                calls > 0,
                "{src}: the column-divisor control lost its helper call, so the \
                 zero-call assertions above can no longer fail"
            );
        }
    }
}

/// The immediate forms answer the two corners their fast path declines exactly
/// as the clean tier does: a `|i64::MIN|` magnitude, a `uint` at or above 2^63,
/// and the `INT_MIN / -1` overflow the tree-walker's `checked_div` reports.
#[test]
fn a_constant_divisor_answers_the_extremes_like_the_clean_tier() {
    use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
    let _serial = serial();
    let xs: Vec<i64> = vec![
        i64::MIN,
        i64::MIN + 1,
        -3,
        -1,
        0,
        1,
        3,
        i64::MAX - 1,
        i64::MAX,
    ];
    let us: Vec<u64> = vec![
        0,
        1,
        3,
        u64::MAX / 3,
        1u64 << 63,
        u64::MAX - 1,
        u64::MAX,
        7,
        8,
    ];
    let lens = vec![xs.len() as i64];
    let schema: Schema = [
        ("xs[]".to_string(), ValType::Int),
        ("us[]".to_string(), ValType::UInt),
    ]
    .into_iter()
    .collect();
    let batch = Batch::new(1)
        .column(
            "xs".to_string(),
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&xs))],
            },
        )
        .column(
            "us".to_string(),
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::UInt(&us))],
            },
        );
    // `/ -1` and `% -1` are the overflow corner: on `i64::MIN` the walker
    // raises, both tiers trap, and the run has no result to compare -- which is
    // itself the thing that has to agree, so the comparison is over the whole
    // `Result`.
    let sources = [
        "xs.map(x, x / 2)",
        "xs.map(x, x / -2)",
        "xs.map(x, x / 3)",
        "xs.map(x, x / 1)",
        "xs.map(x, x / -1)",
        "xs.map(x, x % 2)",
        "xs.map(x, x % 8)",
        "xs.map(x, x % 3)",
        "xs.map(x, x % -1)",
        "xs.map(x, x % 1)",
        "us.map(u, u / 2u)",
        "us.map(u, u / 3u)",
        "us.map(u, u % 8u)",
        "us.map(u, u % 3u)",
        "us.map(u, u % 1u)",
    ];
    for src in sources {
        let lowered = BatchProgram::compile(src, &schema).expect(src);
        let bound = lowered.bind_per_row(&batch).expect(src);
        reset_persistent_state();
        reset_jit_stats();
        let clean = format!("{:?}", bound.collect_on(Tier::Clean));
        for _ in 0..64 {
            let jit = format!("{:?}", bound.collect_on(Tier::Jit));
            assert_eq!(clean, jit, "{src}: compiled tier diverged from clean");
        }
        assert_eq!(jit_stats().internal_compile_panics, 0, "{src}");
    }
}

/// The literal-scan string encoding answers on the compiled tier exactly as it
/// does on the clean one. The data holds distinct non-literal strings — which
/// all share the scan's sentinel id — beside rows equal to each literal, so a
/// sentinel leaking into an equality would change the count.
#[test]
fn a_literal_scan_string_batch_answers_like_the_clean_tier() {
    use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
    let _serial = serial();
    let n = 50_000usize;
    let names: Vec<String> = (0..n)
        .map(|i| match i % 5 {
            0 => "zz".to_string(),
            1 => "ab".to_string(),
            _ => format!("n{i}"),
        })
        .collect();
    let lens = vec![n as i64];
    let schema: Schema = [("items[].name".to_string(), ValType::Str)]
        .into_iter()
        .collect();
    let batch = Batch::new(1).column(
        "items".to_string(),
        ColumnRef::List {
            lens: &lens,
            fields: vec![(Some("name"), ColumnRef::Str(&names))],
        },
    );
    for src in [
        r#"items.exists(i, i.name == "zz")"#,
        r#"items.all(i, i.name != "qq")"#,
    ] {
        let lowered = BatchProgram::compile(src, &schema).expect(src);
        assert!(lowered.lowered().str_ids_literal_only(), "{src}");
        let bound = lowered.bind_per_row(&batch).expect(src);
        reset_persistent_state();
        reset_jit_stats();
        let clean = bound.collect_on(Tier::Clean).unwrap();
        let jit = bound.collect_on(Tier::Jit).unwrap();
        assert_eq!(clean, jit, "{src}: compiled tier diverged from clean");
        let stats = jit_stats();
        assert_eq!(stats.internal_compile_panics, 0, "{src}");
        assert!(
            stats.loops_compiled >= 1,
            "{src}: the string loop did not compile: {stats:?}"
        );
    }
}
