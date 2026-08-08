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
//! ## Observed — PRE-#128-FIX (2026-08-08, 700 calls, n = 2/3/5/8/10, BOTH backends IDENTICAL)
//!
//! ⛔ **This census was taken BEFORE majit `7c141d84175`, the fix it motivated.** Every
//! regime below describes the tree #128 was filed against, not the current one:
//! post-fix a back-edge FINISH returns from the portal instead of resuming at
//! `target_pc`, so `entries/call` and the settled onsets both move. The controls
//! section further down already labels this table "the pre-fix 700-call census";
//! the heading did not, so a reader arriving here first had no marker at all.
//! Post-fix onsets are recorded on #150 (203/202/301/200/349/199, non-monotone).
//!
//! ⚠ SHAs in this file are **majit** (the pyre-wasmi repo), not cel-jit, and both
//! were rewritten by the 2026-08-09 rebase of `cel` onto `origin/main`. They are
//! restated here at their post-rebase spellings, each verified by
//! `merge-base --is-ancestor` and identical `patch-id`. A dead SHA still
//! `git show`s off the old branch, so "it resolves" is not a check.
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
//! ## The exit-kind decomposition this yields — ⛔ FITTED TO IMPORTED FIGURES, NOT MEASURED HERE
//!
//! ⛔⛔⛔ **THIS BINARY CANNOT MEASURE A SINGLE NUMBER IN THE TABLE BELOW.** It has
//! no `#[global_allocator]`, and it says of itself that "this probe times
//! nothing" (see the run loop). It measures the **exit stream** — `entries/call`,
//! exit kinds, `is_finish`, `fail_index` — and that is the whole of #128's
//! evidence and is sound. The `F`/`G`/`C` constants are **allocation** counts
//! (`F=23` and `C=4` are #125's allocation model, quoted at the top of this
//! file), imported from other probes' results and fitted here.
//!
//! ⛔⛔ **AND "18 of 19 reproduce exactly" IS SELF-CONSISTENCY, NOT VALIDATION.**
//! The three constants were fitted *to* those same 19 figures, so an exact
//! reproduction is what a 3-parameter fit over 19 mutually-derived points does.
//! It says the inputs are consistent with each other under this model. It says
//! nothing about whether the inputs are right — and the residual **structurally
//! cannot** tell you: rca125s' known stack-capture arming costs **+1 per compiled
//! exit**, and a per-exit inflation lands entirely in `F` and `G` (both per-exit)
//! and not at all in `C` (per call). Inflated inputs would still fit 18 of 19.
//!
//! ⚠ **Provenance is UNRECOVERABLE from this file, which is the defect.** The
//! phrase "published figures" below names no source, no probe and no tree. What
//! is checkable is the vintage: this table was written 2026-08-08 02:53, and the
//! rca125s attribution work that surfaced the +1 is 19:11–19:25 the same day — so
//! **the inflation was not known when these constants were fitted** and nobody
//! could have corrected for it. `F(cranelift)=23` happens to equal the value #141
//! later corrected *to* (rca125s read 24), but `G=65` has no such cross-check, and
//! a table mixing corrected and uncorrected inputs would look exactly this clean.
//! ⇒ **Do not cite `F`/`G` as measured allocation prices.** Re-derive them against
//! a named probe and a named tree, or quote the exit-stream results only.
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
//! ⭐ The CL-DYN delta is the one reading here that **survives the inflation
//! question by construction**: a `+1` applied to every exit on both backends
//! cancels in the difference, so "3 per compiled entry" holds whether or not the
//! inputs were corrected. It is also the claim that does not depend on the fit.
//!
//! ⛔ The one cell that does not reproduce: n=10 dynasm mid-regime reads 24.580
//! where this predicts 22.600. Every other cell, both backends, is exact.
//! ⚠ And "mid-regime" is the tell — a mid-regime cell is a transient average, so
//! this residual may be a window artifact rather than a model failure. See the
//! `RCA128_SETTLED` section: that detector exists because these very windows
//! close inside live transients.
//!
//! ⛔ It also corrects the reading that "the compiled entry price stops being
//! paid": per exit a guard failure (65) costs ~2.8x a finish (23), so the entry
//! is paid MORE often at a LOWER unit price. The settled regime is cheaper than
//! pre-bridge exactly when `23(n-1) < 65`, i.e. n < 3.8 — which is why n=2 and
//! n=3 improve and n>=5 degrades.
//!
//! ## `RCA128_SHAPE=float` — ⛔ NO LONGER THE NEVER-BRIDGING CONTROL
//!
//! ⛔ **The property this control was SELECTED for is gone, and it was retired by
//! a fix rather than by drift.** What it used to say — and what the paragraph
//! here used to assert — was:
//!
//! > `x * 1.5 + y` holds 1 entry/call with one counted guard failure for all 260
//! > calls, both backends; `loops_aborted` goes 0 -> 1 between call 196 and 206,
//! > i.e. the bridge trace is attempted at the threshold and ABORTS; no bridge
//! > compiles in 6065 calls.
//!
//! **Every one of those is now false.** Measured 3/3 (calls=260 twice, 700 once),
//! cranelift: `loops=1 bridges=1 gfails=201 aborts=0` — and at 700 calls that row
//! is **byte-identical to the `int` row**.
//!
//! ⭐ **The cause is known and was deliberate: #133.** `OP_RETURN_F` reached for
//! the inherent `f64::to_bits()`, which `majit-macros` did not recognise as the
//! bitcast intrinsic, so the dispatch arm degraded to an abort stub and the
//! bridge trace aborted on reaching it. Fixed in majit `84155df5133`; float now
//! compiles its bridge like the other shapes, and its `gfails` fell 262 -> 201,
//! landing exactly on the int shape's figure.
//!
//! ⛔⛔ **Do not read this as weakening #128.** Two things keep it separate:
//! * #128's evidence is `entries/call`, the exit-stream census over n = 2/3/5/8/10,
//!   and a three-config backend A/B. **None of it runs through this shape.**
//! * #133 pre-registered the falsifier and it did not fire: with the bitcast
//!   recognition OFF, float still read `bridges=0` *after* `32f3a1b79f7`. So the
//!   abort was proven **independent of #128's fix**, before this shape changed.
//!
//! ⇒ What died is one *corroborating* sentence — "a shape that never bridges
//! never converts its back edge" — which structurally cannot be run against a
//! shape that bridges. ⚠ Anyone citing it is citing #133's pre-fix state.
//!
//! ## What `float` is still good for
//!
//! It remains a valid **null** control for #128's fix: 1 entry/call, flat across
//! 260 calls, unmoved by the bridge. That is a different claim from "never
//! bridges" and it is the one to cite. It is also the shape whose trace is a
//! genuine **peeled loop** (`Label` + `Jump` on `LoopTargetDescr`, verified from a
//! `MAJIT_LOG=1` dump in #133) — so it is the positive case for #137's caution
//! that `loops_compiled` cannot tell a peeled loop from a straight-line trace.
//!
//! ⚠ There is currently **no never-bridging control in this file.** If one is
//! needed, it has to be found and re-verified, not assumed — and see #155.
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
//! | `RCA128_REQUIRE` | `bridge` | `bridge` = fail unless a bridge compiled; `loop` = fail unless *something* compiled |
//! | `RCA128_THRESHOLD` | 8 | trace threshold; moving it re-partitions n into peeled/flat (#134) |
//! | `RCA128_SETTLED` | `require` | `require` = fail unless the window closes in the settled regime; `allow` = label the reading and continue |
//!
//! `RCA128_REQUIRE` exists because the probe's controls invert the postcondition.
//! The default run is worthless if it never reached the transition, so it demands
//! a bridge. The `float` control is worthless if the shape never **compiled** —
//! "never bridges" is trivially true of a program that never gets a loop, and
//! reporting that as a negative control would be a fabricated result. So the
//! control demands `loops_compiled >= 1` instead, and the thing it is *for* is
//! then free to be zero.
//!
//! ⚠ `loop` names the mode, not the artifact's shape. `loops_compiled` counts
//! every compiled artifact, and a straight-line FINISH trace is one of them
//! (#137) — so the check cannot tell a peeled loop from a flat trace, and does
//! not try to. Existence is the whole postcondition. For shape, read `Label` /
//! `Jump` out of a `MAJIT_LOG=1` dump; no counter carries it.
//!
//! ## `RCA128_SETTLED`: the window can close in a regime the run is still leaving
//!
//! `RCA128_CALLS` defaults to 260, and **260 is the wrong window for some `n`**.
//! The regime table above locates the settled onset per `n` — 2 -> call 8,
//! 3 -> 204, 5 -> 502, 8 -> 401, 10 -> 400 — so on the tree that census was taken
//! on, the default closed *inside* a live transient for n = 5, 8 and 10 and
//! printed a mid-regime value with nothing marking it as one.
//!
//! ⛔ **Widening the default is not the fix.** Settling is *not monotone in `n`*
//! (n=5 bridges sooner than n=8 yet settles later — it needs more bridges), so
//! every constant is wrong for some `n`, and a bigger one just relocates the
//! silent failure. ⭐ And locating the onsets would only produce a survey that
//! ages: the post-fix onsets already differ from the pre-fix ones above.
//!
//! So the probe answers it **per run**. The settled regime is exactly "`bridges`
//! stable and `guard_failures` frozen", which is a property this binary can read
//! directly, so it reads it:
//!
//! > **If the window closes while the counters are still moving, say so, or refuse.**
//!
//! ### Why the TAIL, and why "distance from the last change" rather than a fraction
//!
//! ⚠ **n=3 is the design constraint.** It settles at call 204, so a 260-call
//! window *is* settled at the end yet *contains* a transition. Testing the whole
//! run would refuse it — a false refusal on a perfectly good reading. The test
//! therefore has to look at the tail only.
//!
//! ⭐ But "the last X% of the window" is the wrong shape for the tail, because it
//! makes the verdict a cliff: at 260 calls a 20% tail starts at 209 and n=3 passes,
//! a 25% tail starts at 196 and n=3 **false-refuses**. That is a constant tuned to
//! one fixture cell — the very defect this section exists to remove.
//!
//! Measuring **how long the counters have been quiet** removes the cliff, because
//! it is the quantity the controls actually separate on. At the default 260:
//!
//! | n | last counter movement | quiet for | verdict |
//! |---|---|---|---|
//! | 2 | never moves (`gfails` 0, no bridge) | 260 | ✅ settled |
//! | 3 | call ~203 | ~57 | ✅ settled |
//! | 5 | every call (mid regime `102-501`) | **0** | ⛔ refuse |
//! | 8 | every call (mid regime `201-400`) | **0** | ⛔ refuse |
//! | 10 | every call (mid regime `201-399`) | **0** | ⛔ refuse |
//!
//! The passing and refusing cells are separated by **57 against 0**, so the
//! threshold is nowhere near either group and no cell is close to flipping. The
//! choice is `tail = calls / 10` (26 at the default) — **stated here rather than
//! left implicit in a `>=`**, and the table above is the room it has on each side.
//!
//! ### Controls, all collected before the detector was designed
//!
//! * **Negative (must refuse), 5 cells**: the pre-fix 700-call census above, read
//!   at 260 — n=2 pass, n=3 pass, n=5/8/10 refuse.
//! * **Positive (must pass), 9 cells**: `sizes`' settled sweep at
//!   n = 2/3/4/5/8/9/16/32/64, where `bridges` and `gfails` are identical at call
//!   600 and call 700. That puts the last movement at or before 600, so on a
//!   700-call run every cell is quiet for >= 100 against a 70-call tail.
//!
//! ⭐ A diagnostic proved by its own fixture proves nothing; both control sets
//! predate this code.
//!
//! ⚠ **Scope.** This certifies *these two counters* over *the reported window*. A
//! transient in cost/call or allocations/call is entirely compatible with a pass,
//! and so is a bridge landing past the end of the run.

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
    let settled_mode = std::env::var("RCA128_SETTLED").unwrap_or_else(|_| "require".to_string());

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
    //
    // The counters are sampled after every call, so `last_change` is the call at
    // which the artifact last moved rather than a bound inferred from the two
    // endpoints. Sampling every call is safe *here* in a way it is not in
    // `rca88b`'s Probe J: that one keeps `jit_stats()` out of the inner path to
    // keep a timed sequence identical to an uninstrumented one, and this probe
    // times nothing — it already runs a full oracle comparison per call.
    let mut prev = {
        let s = jit_stats();
        (s.bridges_compiled, s.guard_failures)
    };
    // 0 = "never moved", which is the settled-from-the-start case (n=2).
    let mut last_change = 0usize;
    for k in 1..=calls {
        eprintln!("@@@CALL {k}");
        call();
        let s = jit_stats();
        let now = (s.bridges_compiled, s.guard_failures);
        if now != prev {
            last_change = k;
            prev = now;
        }
    }
    eprintln!("@@@CALL {}", calls + 1);
    drop(call);

    // The settled regime is `bridges` stable and `guard_failures` frozen, so
    // "how long have both been quiet" is the whole test. A fraction-of-window
    // tail would be a cliff at n=3 (see the module docs); this is not, because
    // the controls separate 57 against 0 on exactly this quantity.
    let tail = (calls / 10).max(1);
    let quiet_for = calls - last_change;
    let settled = quiet_for >= tail;

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
        // `settled` qualifies every counter on this line the same way `majit=`
        // does: a mid-regime reading is a real plateau, not noise, so it is
        // labelled rather than suppressed.
        "majit={} shape={shape} calls={} n={n} loops={} bridges={} gfails={} aborts={} \
         settled={settled} last_change={last_change} quiet_for={quiet_for} tail={tail}",
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
        // as a never-bridging control would be a fabricated negative. The check
        // is existence, not shape: `loops_compiled` counts every compiled
        // artifact, straight-line FINISH traces included (#137), so a flat cell
        // satisfies it — correctly, because a flat cell did compile.
        "loop" => assert!(
            s.loops_compiled >= 1,
            "nothing compiled in {} calls — a control that never compiles \
             cannot witness anything about bridges",
            calls + 1
        ),
        other => panic!("RCA128_REQUIRE={other:?}; expected `bridge` or `loop`"),
    }

    // Refusing rather than widening `RCA128_CALLS`: settling is not monotone in
    // `n`, so every constant window is wrong for some `n`, and a wrong refusal
    // is auditable where a wrong number is not. `allow` exists because studying
    // a transient is a legitimate use of this probe — the same reason
    // `RCA128_REQUIRE` has a second mode — but it must be asked for, so that a
    // mid-regime reading can never be produced by default and read as settled.
    match settled_mode.as_str() {
        "require" => assert!(
            settled,
            "the {calls}-call window closed while the artifact was still moving: \
             counters last changed at call {last_change}, quiet for only \
             {quiet_for} of a required {tail}. This is a MID regime, not the \
             settled one. Re-run with a larger RCA128_CALLS, or pass \
             RCA128_SETTLED=allow if the transient is what you are measuring."
        ),
        "allow" => {
            if !settled {
                eprintln!(
                    "@@@WARN rca128: NOT SETTLED — counters last changed at call \
                     {last_change}, quiet for {quiet_for} of {tail}; every figure \
                     above describes a mid regime"
                );
            }
        }
        other => panic!("RCA128_SETTLED={other:?}; expected `require` or `allow`"),
    }
}
