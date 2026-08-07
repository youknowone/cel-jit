//! #128's closing experiment: does the artifact stop iterating internally?
//!
//! #122 established that a compiled cel artifact steps once, permanently, at
//! `trace_eagerness` calls after its loop compiles — call 200 with the default
//! 200 — and that the step is a PER-ROW cost even though the guard that bridges
//! there fails only once per CALL.
//!
//! #128's proposed mechanism: after the bridge, the loop's iteration is carried
//! by the interpreter. Each row becomes one compiled entry, one JUMP/finish
//! exit, and one return through the portal, so the artifact is entered `n` times
//! per call instead of once.
//!
//! The chain behind it is three measurements and one code-reading inference:
//!
//! 1. `guard_failures` excludes two of the three exit kinds —
//!    `should_record_guard_failure = !is_finish && !is_jump_exit`
//!    (`majit-metainterp/src/pyjitpl.rs:2087-2094`).
//! 2. Post-bridge `gfails` FREEZES ⇒ every exit after that is a `finish` or a
//!    JUMP exit. Not fewer — none.
//! 3. #125's integer model: the per-call constant collapses 69 -> 4 (and 4 is
//!    the un-compiled portal's own), while per-back-edge goes 0 -> 23, with
//!    `23 = 12 + 11` ADDITIVELY — one portal pass plus one added construction,
//!    not a duplicated traversal.
//! 4. ⚠ INFERENCE: the only two arms of `back_edge_internal` that return to the
//!    interpreter without recording a guard failure are `is_finish`
//!    (`jitdriver.rs:4339-4347`) and `fail_index == u32::MAX`
//!    (`:4350-4358`, "Normal loop back-edge JUMP, not a guard failure").
//!
//! This probe replaces step 4 with an observation. `PYRE_PORTAL_RCA=1`
//! (`jitdriver.rs:477-480`) already prints one `[portal-rca][compiled-entry]`
//! per compiled entry and one `[portal-rca][compiled-exit]` per exit, carrying
//! `is_finish` and `fail_index`. All this file has to do is drive a known number
//! of calls at a known `n` and mark the call boundaries in the same stream.
//!
//! ## Pre-registered outcomes
//!
//! * **entries/call goes 1 -> ~10, and post-200 exits carry
//!   `fail_index=4294967295` or `is_finish=true`** => #128 confirmed: attaching
//!   the bridge converted an internal back edge into an external one.
//! * **entries/call stays 1** => #128 REFUTED. The 23 allocations/row are
//!   something else and the search redirects.
//!
//! Both branches are informative and neither is the one this probe is "for".
//!
//! ## Observed (2026-08-08, n=10, 260 calls, BOTH backends, byte-identical)
//!
//! | regime | entries/call | exits/call, by kind |
//! |---|---|---|
//! | calls 1-200  | 1.000 | `is_finish=false fail_index=5` x1 |
//! | calls 201-260 | 9.000 | `is_finish=true fail_index=4294967295` x8, then `is_finish=false fail_index=2` x1 |
//!
//! The first branch, exactly. 741 compiled entries under `jit-cranelift` and 741
//! under `jit-dynasm`; the step is at call 201 with nothing in between.
//!
//! Three things the table says that the counters could not:
//!
//! * The 8 added exits per call carry `is_finish=true` AND
//!   `fail_index=u32::MAX` — both of `should_record_guard_failure`'s exclusions
//!   at once. They are not undercounted, they are outside the counter's domain.
//! * `gfails` still reads 1/call after the step (261 in 261 calls), but it is a
//!   DIFFERENT guard: `fail_index=5` before, `fail_index=2` after. An unchanged
//!   rate here means the counted population was replaced, not that nothing moved.
//! * 9, not 10: the 8 finish exits come first and the counted guard failure is
//!   last, so the compiled artifact carries the loop's body and falls out of its
//!   tail — the interpreter re-enters it per row.
//!
//! ## Why n=10 and why a separate file
//!
//! The diagnostic prints ~`n` lines per call once the transition is crossed, so
//! at n=1000 a 250-call run is over a million lines. n=10 is the smallest size
//! at which #122's step is present (2.4x) and it keeps the log readable.
//! `cel/examples/rca88b.rs` carries eight probes and cannot have this variable
//! set on it without drowning them.
//!
//! ## Running it
//!
//! ```text
//! PYRE_PORTAL_RCA=1 cargo run --release -p cel --features jit-cranelift \
//!     --example rca128 2> rca128.log
//! ```
//!
//! then segment `rca128.log` on the `@@@CALL <k>` markers this file writes.

use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// #122's default arm: the loop compiles inside call 1 at n=10, so the first
/// bridge lands at call 200 and the step at call 201.
const THRESHOLD: u32 = 8;
const N: usize = 10;
/// Far enough past 200 to have a settled post-step regime, short enough that
/// the log stays in the tens of thousands of lines.
const CALLS: usize = 260;

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

fn main() {
    // Without the variable this binary emits ZERO `[portal-rca]` lines, and an
    // analysis of that log would read "0 entries per call" — which is the
    // signature of "the artifact is never entered", i.e. a strong and entirely
    // fabricated finding. Refusing to run is the only way "the instrument was
    // off" and "the instrument reported nothing" stay distinguishable.
    assert!(
        std::env::var_os("PYRE_PORTAL_RCA").is_some(),
        "rca128 must run with PYRE_PORTAL_RCA=1; without it the log is empty \
         and an empty log is indistinguishable from `entries/call == 0`"
    );

    let schema: Schema = [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("price + qty * 2", &schema);

    let price: Vec<i64> = (0..N as i64).map(|i| (i * 37) % 200).collect();
    let qty: Vec<i64> = (0..N as i64).map(|i| (i * 11) % 100).collect();
    let columns = vec![Column::Int(&price), Column::Int(&qty)];

    reset_persistent_state();
    reset_jit_stats();

    // The compiled tier must agree with the interpreter, or every line below is
    // a description of a miscompile rather than of an exit path.
    let want = clean_batch_sum_f(&lowered, &columns, N);
    let got = eval_batch_sum_f(&lowered, &columns, N, THRESHOLD);
    assert_eq!(got, want, "jit tier disagrees with clean tier");

    // The markers go to stderr so they interleave with the `[portal-rca]` lines
    // in one stream; two streams would need timestamps to re-order.
    for k in 1..=CALLS {
        eprintln!("@@@CALL {k}");
        black_box(eval_batch_sum_f(&lowered, &columns, N, THRESHOLD));
    }
    eprintln!("@@@CALL {}", CALLS + 1);

    let s = jit_stats();
    // Printed to stderr so the whole record is one file, and stated as a
    // precondition rather than a result: if the run did not cross the
    // transition, the entries/call columns describe the pre-bridge regime only
    // and answer nothing.
    eprintln!(
        "@@@STATS calls={} n={N} loops={} bridges={} gfails={} aborts={}",
        CALLS + 1,
        s.loops_compiled,
        s.bridges_compiled,
        s.guard_failures,
        s.loops_aborted,
    );
    println!(
        "rca128: {} calls at n={N}, loops={} bridges={} gfails={} aborts={}",
        CALLS + 1,
        s.loops_compiled,
        s.bridges_compiled,
        s.guard_failures,
        s.loops_aborted,
    );
    assert!(
        s.bridges_compiled >= 1,
        "no bridge compiled in {} calls — the run never reached the transition \
         this probe exists to measure",
        CALLS + 1
    );
}
