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
//! 2. ⛔ WRONG AS FILED, corrected by this probe: `gfails` was said to freeze at
//!    the first bridge. It freezes after the LAST one; across the first it holds
//!    its RATE while its population is replaced (`fi=5` -> `fi=2`).
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
//! ## Observed (2026-08-08, 700 calls, n = 2/3/5/8/10, BOTH backends IDENTICAL)
//!
//! Regimes, as emitted (`fi` = `fail_index`; `4294967295` = `u32::MAX`):
//!
//! | n | calls | entries/call | exits/call |
//! |---|---|---|---|
//! | 2 | 8-700 | **1** | finish x1 |
//! | 3 | 4-203 | 1 | guard `fi=2` x1 |
//! | 3 | 204-700 | **2** | finish x2 |
//! | 5 | 2-101 | 2 | guard `fi=2` **x2** |
//! | 5 | 102-501 | 2 | guard x1 + finish x1 |
//! | 5 | 502-700 | **4** | finish x4 |
//! | 8 | 1-200 | 1 | guard `fi=5` x1 |
//! | 8 | 201-400 | 7 | guard `fi=2` x1 + finish x6 |
//! | 8 | 401-700 | **7** | finish x7 |
//! | 10 | 1-200 | 1 | guard `fi=5` x1 |
//! | 10 | 201-399 | 9 | guard `fi=2` x1 + finish x8 |
//! | 10 | 400-700 | **9** | finish x9 |
//!
//! The confirming branch, and more than it asked for:
//!
//! * **The settled regime is `entries/call = n-1` EXACTLY at every n**, with every
//!   exit `is_finish=true fail_index=u32::MAX` and not one counted guard failure.
//!   That is the `n-1` axis of #125's allocation model, measured on an unrelated
//!   instrument at five values of `n`.
//! * **`gfails` freezes only after the LAST bridge**, not the first. At n=10 the
//!   first bridge leaves one counted guard (`fi=2`) standing and the second
//!   removes it. Bridges needed = number of distinct counted guards: n=2 needs 0,
//!   n=3 one, n=8/10 two, n=5 three.
//! * **The clock is guard failures, not calls.** n=5 fails TWO per call, so it
//!   reaches `trace_eagerness` in 100 calls and bridges at 102; n=3 compiles at
//!   call 4 and bridges at 204. `bridge@call - compile@call = 200 / gfails-per-call`.
//! * **n=2 is the degenerate case, not an anomaly.** `n-1 = 1`, so "once per call"
//!   and "once per back edge" are the same number; it is in the settled regime
//!   from the moment the loop compiles (call 8), with no bridge, because there is
//!   no internal back edge to convert. Its one exit is already a `finish`, so
//!   `gfails = 0`, so the bridge clock never ticks and it can never bridge.
//!
//! ## The exit-kind decomposition this yields
//!
//! Three constants per backend reproduce **18 of 19** published per-call and
//! per-row figures exactly (`F` = an entry ending in `finish`, `G` = an entry
//! ending in a counted guard failure, `C` = per call):
//!
//! | | F | G | C |
//! |---|---|---|---|
//! | cranelift | 23 | 65 | 4 |
//! | dynasm | 20 | 62 | 4 |
//!
//! `cost/call = F x finishes + G x guards + C`. So n=5's "anomalous" pre-bridge
//! 134/128 is just two guard exits, n=2's 27/24 is one finish exit, and the
//! CL-DYN delta of **3 is per compiled ENTRY** (it is 3 in both `F` and `G`, and
//! 0 in `C`) rather than per back edge or per call.
//!
//! ⛔ The one cell that does not reproduce: n=10 dynasm mid-regime reads 24.580
//! where this predicts 22.600. Every other cell, both backends, is exact.
//!
//! ⛔ It also corrects the reading that "the compiled entry price stops being
//! paid": per exit a guard failure (65) costs ~2.8x a finish (23), so the entry
//! is paid MORE often at a LOWER unit price. The settled regime is cheaper than
//! pre-bridge exactly when `23(n-1) < 65`, i.e. n < 3.8 — which is why n=2 and
//! n=3 improve and n>=5 degrades.
//!
//! ## The negative control (`RCA128_SHAPE=float`)
//!
//! `x * 1.5 + y` holds **1 entry/call with one counted guard failure for all 260
//! calls**, both backends — no burst, no `finish` exits, no regime change. A
//! shape that never bridges never converts its back edge, which is what a
//! bridge-triggered mechanism requires.
//!
//! ⭐ And it says why float never bridges: `loops_aborted` goes **0 -> 1 between
//! call 196 and call 206**, i.e. at the threshold. The bridge trace is attempted
//! and ABORTS. That is one cause for all three of its properties — the threshold
//! fires, no bridge compiles, and the cost does not persist.
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
//!
//! ## Knobs, and why the assertion is one of them
//!
//! | env | default | meaning |
//! |---|---|---|
//! | `RCA128_SHAPE` | `int` | `int` = `price + qty * 2`; `float` = `x * 1.5 + y` |
//! | `RCA128_N` | 10 | rows per call |
//! | `RCA128_CALLS` | 260 | calls after the oracle check |
//! | `RCA128_REQUIRE` | `bridge` | `bridge` = fail unless one compiled; `loop` = fail unless the loop compiled |
//! | `RCA128_THRESHOLD` | 8 | trace threshold; moving it re-partitions n into peeled/flat (#134) |
//!
//! `RCA128_REQUIRE` exists because the probe's controls invert the postcondition.
//! The default run is worthless if it never reached the transition, so it demands
//! a bridge. The `float` control is worthless if the shape never **compiled** —
//! "never bridges" is trivially true of a program that never gets a loop, and
//! reporting that as a negative control would be a fabricated result. So the
//! control demands `loops_compiled >= 1` instead, and the thing it is *for* is
//! then free to be zero.

use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, eval_batch_sum_float, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// #122's default arm: the loop compiles inside call 1 at n=10, so the first
/// bridge lands at call 200 and the step at call 201.
///
/// `RCA128_THRESHOLD` overrides it. #134's law is arithmetic in this number —
/// the optimizer peels unless the back-edge counter trips on a batch boundary,
/// i.e. unless `(n-1)` divides the threshold — so moving it re-partitions n
/// into flat and grown, which is the only way to tell that law apart from any
/// property of the expression.
const THRESHOLD: u32 = 8;

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("{key}={v:?} is not a usize: {e}")),
        Err(_) => default,
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    match std::env::var(key) {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|e| panic!("{key}={v:?} is not a u32: {e}")),
        Err(_) => default,
    }
}

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

    let shape = std::env::var("RCA128_SHAPE").unwrap_or_else(|_| "int".to_string());
    let n = env_usize("RCA128_N", 10);
    let calls = env_usize("RCA128_CALLS", 260);
    let require = std::env::var("RCA128_REQUIRE").unwrap_or_else(|_| "bridge".to_string());
    let threshold = env_u32("RCA128_THRESHOLD", THRESHOLD);

    // Both column sets are built unconditionally so each borrow outlives the
    // closure below; only one is ever read.
    let price: Vec<i64> = (0..n as i64).map(|i| (i * 37) % 200).collect();
    let qty: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 100).collect();
    let xs: Vec<f64> = (0..n).map(|k| (k % 97) as f64 * 0.5).collect();
    let ys: Vec<f64> = (0..n).map(|k| (k % 89) as f64 * 0.25).collect();

    let (src, types): (&str, [(&str, ValType); 2]) = match shape.as_str() {
        "int" => (
            "price + qty * 2",
            [("price", ValType::Int), ("qty", ValType::Int)],
        ),
        // p52's never-bridging case (`cel/examples/rca116.rs:182-183`), the free
        // negative control: it compiles a loop and fails one guard per call like
        // the shapes that step, and compiles no bridge in 6065 calls.
        "float" => (
            "x * 1.5 + y",
            [("x", ValType::Float), ("y", ValType::Float)],
        ),
        other => panic!("RCA128_SHAPE={other:?}; expected `int` or `float`"),
    };
    let schema: Schema = types
        .iter()
        .map(|(name, ty)| ((*name).to_string(), *ty))
        .collect();
    let lowered = lower(src, &schema);
    let columns = match shape.as_str() {
        "int" => vec![Column::Int(&price), Column::Int(&qty)],
        _ => vec![Column::Float(&xs), Column::Float(&ys)],
    };

    reset_persistent_state();
    reset_jit_stats();

    // The compiled tier must agree with an uncompiled oracle, or every line
    // below describes a miscompile rather than an exit path. The int shape has a
    // clean-tier door; the float shape has none, so its oracle is the same door
    // at `threshold = u32::MAX`, which engages majit and never compiles.
    //
    // The comparison is repeated on EVERY call, not just once before the loop.
    // The pre-loop call is call 0: nothing is compiled yet, no guard has failed
    // and no bridge exists, so a single check there is blind to every tier this
    // binary exists to observe — a wrong answer that only appears once the
    // artifact is entered, or once a bridge is attached at `trace_eagerness`,
    // would leave it green.
    let mut call: Box<dyn FnMut()> = match shape.as_str() {
        "int" => {
            let want = clean_batch_sum_f(&lowered, &columns, n);
            let got = eval_batch_sum_f(&lowered, &columns, n, threshold);
            assert_eq!(got, want, "jit tier disagrees with clean tier");
            Box::new(move || {
                assert_eq!(
                    black_box(eval_batch_sum_f(&lowered, &columns, n, threshold)),
                    want,
                    "jit tier disagrees with clean tier"
                );
            })
        }
        _ => {
            let want = eval_batch_sum_float(&lowered, &columns, n, u32::MAX);
            let got = eval_batch_sum_float(&lowered, &columns, n, threshold);
            assert_eq!(got, want, "jit tier disagrees with the never-compiles tier");
            Box::new(move || {
                assert_eq!(
                    black_box(eval_batch_sum_float(&lowered, &columns, n, threshold)),
                    want,
                    "jit tier disagrees with the never-compiles tier"
                );
            })
        }
    };

    // The markers go to stderr so they interleave with the `[portal-rca]` lines
    // in one stream; two streams would need timestamps to re-order.
    for k in 1..=calls {
        eprintln!("@@@CALL {k}");
        call();
    }
    eprintln!("@@@CALL {}", calls + 1);
    drop(call);

    let s = jit_stats();
    // Printed to stderr so the whole record is one file, and stated as a
    // precondition rather than a result: if the run did not satisfy
    // `RCA128_REQUIRE`, the entries/call columns answer a different question
    // than the one asked.
    let summary = format!(
        // `majit=` FIRST: provenance qualifies every field after it, so a
        // truncated or wrapped line still carries it. `env!` not `option_env!`
        // — a missing token must fail the build, not vanish from the line and
        // leave the reading looking unqualified but trustworthy.
        "majit={} shape={shape} calls={} n={n} loops={} bridges={} gfails={} aborts={}",
        env!("CEL_MAJIT_PROVENANCE"),
        calls + 1,
        s.loops_compiled,
        s.bridges_compiled,
        s.guard_failures,
        s.loops_aborted,
    );
    eprintln!("@@@STATS {summary}");
    println!("rca128: {summary}");
    match require.as_str() {
        "bridge" => assert!(
            s.bridges_compiled >= 1,
            "no bridge compiled in {} calls — the run never reached the \
             transition this probe exists to measure",
            calls + 1
        ),
        // A shape that never compiles trivially never bridges, so reporting it
        // as a never-bridging control would be a fabricated negative.
        "loop" => assert!(
            s.loops_compiled >= 1,
            "no loop compiled in {} calls — a control that never compiles \
             cannot witness anything about bridges",
            calls + 1
        ),
        other => panic!("RCA128_REQUIRE={other:?}; expected `bridge` or `loop`"),
    }
}
