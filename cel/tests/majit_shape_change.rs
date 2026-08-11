//! What a driver that already compiled for one inner trip count does when the
//! next batch has a different one.
//!
//! `majit_trace_evidence.rs` censuses ONE data shape per driver on purpose, and
//! `nested_loop_deopts_are_a_warmup_cost_not_a_per_row_cost` compares two batch
//! SIZES of the same shape. Neither covers a driver that has already compiled
//! for a different SHAPE — which is what a long-lived process does, since the
//! driver outlives a call and the program words outlive it with the `LoweredF`
//! that owns them.
//!
//! ## The gap this pins (measured 2026-08-04, majit @ PR 960)
//!
//! Each cell is the settled ns/row as a ratio to the same measured trip count's
//! own cold control, so absolute machine speed cancels. Beside it, the same
//! quantity for PyPy 7.3.22 running an equivalent nested loop, one arm per fresh
//! process:
//!
//! | warm → measured | majit | PyPy 7.3.22 |
//! |---|---|---|
//! | 64 → 64 (diagonal) | 1.12x | 1.03x |
//! | 2 → 2 (diagonal) | 0.92x | 0.97x |
//! | 3 → 64 | 15.03x | **1.02x** |
//! | 2 → 8 | 14.37x | **0.99x** |
//! | 64 → 2 | 11.77x | **1.17x** |
//! | 8 → 2 | 9.69x | **0.92x** |
//! | 2 → 64 | 15.02x | 5.84x |
//! | 8 → 64 | 1.59x | 5.17x |
//!
//! Read it as: the *class* of cost is real upstream — PyPy degrades on `2 → 64`
//! too, so trip-count specialisation plus a bridge is inherent to tracing — but
//! PyPy returns to ~1.0x on five of the seven off-diagonal cells and majit
//! returns to ~1.0x on none. The mechanism is that the outer trace bakes the
//! observed trip count in as a guard (`IntGt(len, k) GuardFalse`); on a shape
//! change the guard's bridge jumps back into the SAME loop. `compiles` is 0 in
//! every degraded batch: no artifact is ever built for the second shape, in
//! either direction. The penalty is symmetric — `64 → 2` is as bad as `2 → 64`.
//!
//! ## What a ratio cannot tell you (measured 2026-08-04)
//!
//! A ratio to a cold control says the two arms differ; it does not say which
//! side moved. Absolute numbers split the 10-15x in two, and only one half is a
//! parity gap:
//!
//! * The degraded PER-ITERATION cost is at parity — majit 5.44 ns/iter against
//!   PyPy's 5.02 on `2 → 64`. The bad state is not worse here than upstream.
//! * The rest is a FIXED per-row cost. Sweeping the measured trip count against
//!   a trip-64 artifact gives 14.8 / 16.6 / 15.5 / 15.7 / 16.9 ns/row at trips
//!   1 / 2 / 4 / 8 / 16 where cold is 2.2 / 2.1 / 3.0 / 3.9 / 6.2 — flat, so it
//!   is ~13 ns paid once per row however little work the row does. Holding the
//!   measured trip at 1 and sweeping the warm trip instead gives 1.8 (cold) then
//!   11.8 / 14.4 / 14.6 / 16.0 / 15.0 / 15.1, so it does not scale with the
//!   artifact either. It is one guard-exit → bridge → loop-re-entry round trip.
//!
//! Every number this test asserts is a CRANELIFT number: `cel`'s `jit` feature
//! selects `majit-metainterp/cranelift` and nothing else, and on that backend a
//! guard exit marshals all 23 live values through the jitframe twice per row
//! where upstream patches the guard's branch straight into a bridge that was
//! register-allocated against the guard's own fail locations
//! (`rpython/jit/backend/aarch64/assembler.py:163,200-202,1054-1060`).
//!
//! ## The backend control (measured 2026-08-04)
//!
//! Running this same test against `majit-metainterp/dynasm` — real aarch64
//! machine code, upstream's register allocator, upstream's patched-branch
//! bridge attachment — says most of the penalty was that edge and not the
//! trace's shape. Same test, same machine, same session; the `fraction` column
//! is this file's own "fraction of the clean VM" and is the honest one, since
//! the two backends have different cold baselines:
//!
//! | warm → measured | cranelift | dynasm | PyPy 7.3.22 |
//! |---|---|---|---|
//! | 64 → 64 (diagonal) | 0.99x | 1.34x | 1.03x |
//! | 2 → 2 (diagonal) | 1.00x | 1.42x | 0.97x |
//! | 3 → 64 | 14.0x | **3.18x** | 1.02x |
//! | 64 → 2 | 12.5x | **2.50x** | 1.17x |
//! | 8 → 2 | 10.8x | **3.25x** | 0.92x |
//! | 2 → 64 | 13.1x | **3.96x** | 5.84x |
//!
//! As a fraction of the untraced VM, the degraded cranelift tier is only
//! 1.4-2.2x faster than no JIT at all (0.445-0.694); the degraded dynasm tier
//! stays 6.5-11.5x faster (0.087-0.155). So roughly three quarters of the
//! penalty was the cross-artifact edge. What survives is real and portable:
//! dynasm still does not return to ~1.0x on the three cells where PyPy does,
//! and only on `2 → 64` — the one cell PyPy also degrades on — is it ahead.
//!
//! Two caveats on that column. The dynasm cells are single runs on a shared
//! machine, so the two diagonals reading 1.34x/1.42x is noise: the same
//! comparison under `examples/poison`'s min-of-rounds matrix puts every
//! diagonal at 1.0x and the off-diagonals at 2.1-4.1x, which is the range to
//! trust. And cel on dynasm was not a sound backend when this was written —
//! 5 of 220 `cel` unit tests miscompiled there (a ternary sum, and
//! float/list-valued results reading back integer bit patterns), so the dynasm
//! column was trustworthy only because [`measure_cell`] asserts every timed
//! batch against the clean VM and that predicate answered correctly.
//!
//! ⚠ That soundness caveat is RETIRED as of 2026-08-11: the dynasm unit suite
//! reads 250 passed / 0 failed at pyre `9970be67cb2`. It is left standing
//! rather than deleted because the *reason* the column was usable — every
//! timed batch is checked against the clean VM — is still the reason, and a
//! reader who remembers "dynasm miscompiles" needs to know it was retired by
//! fixes rather than by someone lowering the bar.
//!
//! ⚠ The first of the two edits below is RETIRED — `cel/Cargo.toml` now carries
//! `jit-dynasm` and `jit-cranelift` backend selectors, so the column reproduces
//! with `cargo test --locked -p cel --features jit-dynasm` and no manifest
//! surgery. A bare `--features jit` is a hard error rather than a silent
//! no-JIT build, so the flag cannot be forgotten. The SECOND edit still
//! stands, and is the one to check before trusting any number here: the
//! `[patch]` lives in `cel-jit/.cargo/config.toml`, which is UNTRACKED. A
//! clean clone therefore resolves the pinned `majit-metainterp` git rev, not
//! this worktree's majit — so every figure in this file describes live majit
//! only for someone who has that untracked file.
//!
//! (Historical, describing the state before the selectors existed:)
//! Reproducing the column needs two edits that are deliberately NOT committed:
//! flip `cel/Cargo.toml`'s `majit-metainterp/cranelift` to
//! `majit-metainterp/dynasm`, and `[patch]` the `majit-*` crates at a checkout
//! carrying pyre "majit: run the GC rewrite pass whether or not a collector is
//! installed" — `0f78ca9fb5c`, an ancestor of `origin/main`, so unlike most
//! cross-repo citations here that sha is permanent and safe to use directly.
//! (It read `35c51a079a1` until 2026-08-11; that tree was rewritten away.)
//! The pinned revision skips the pass that
//! lowers `RAW_LOAD_I` and then panics in the dynasm register allocator.
//!
//! ## What this test asserts, and what it deliberately does NOT
//!
//! It does **not** pin the 10-15x. Encoding today's gap as the expectation would
//! turn a defect into a baseline. It asserts the two lines that bound it:
//!
//! 0. **Something compiled at all.** Every cell asserts `loops_compiled >= 1`
//!    and `internal_compile_panics == 0` out of [`jit_stats`]. This is the half
//!    of claim 1 that is not a speed question, and a counter answers it the
//!    same way on an idle box and a box at load 60. It says nothing about the
//!    SECOND shape: `loops_compiled` is 0 for a degraded off-diagonal batch by
//!    design, and asserting that would turn the defect into a baseline.
//! 1. **The healthy path stays healthy.** Cold and diagonal runs must stay at or
//!    under [`DIAGONAL_CEILING`] of the clean VM. They measure 0.001-0.064x over
//!    40 runs, so this catches compiled code that got dramatically slower
//!    without ceasing to exist.
//! 2. **The tier never becomes worse than no JIT at all.** Every off-diagonal
//!    cell must stay under [`OFF_DIAGONAL_CEILING`] of the clean VM. The worst
//!    reading over the same 40 runs is 0.567x, so the gap has ~1.8x of room
//!    before this fires — it catches the gap WIDENING, which is the regression
//!    this file exists to prevent, without asserting that the gap is
//!    acceptable. It is not; the target is the PyPy column above.
//!
//! Both budgets are ratios against `clean_batch_sum_f` — the same lowered
//! program over the same columns with no tracing machinery.
//!
//! ## Why the ratio is estimated the way it is (measured 2026-08-07)
//!
//! This file used to claim the ratio was "measured in the same process at the
//! same moment, so a loaded machine scales both sides and cancels instead of
//! flaking". It was not, and it did flake. The clean side took a **min of 3**
//! batches; the compiled side ran 3 and kept the **last**, discarding the other
//! two. A single descheduled batch landing on that last iteration went into the
//! numerator with nothing to damp it, and the two sides were not even measured
//! in the same window — the clean batches all ran before the warm-up did.
//!
//! Priced by failure rate, not by a green. Both binaries were built back to
//! back from one snapshot of the `cel` library, then alternated run for run so
//! load drift hit both equally, 40 runs each at load 50-56:
//!
//! | estimator | pass | fail |
//! |---|---|---|
//! | last-of-3 over min-of-3 (the one this replaces) | 27 | **13** |
//! | min of per-round ratios, 9 rounds | **40** | 0 |
//!
//! The worst single reading under the old estimator in that corpus was
//! `warm=Some(2) measured=64` at **3.29x** its ceiling — 2340 ns/row on a cell
//! whose worst of 40 under the committed estimator is 0.556x. A deterministic
//! source change cannot produce 13-of-40, so every one of those reds was
//! misattributable to whatever diff happened to be in the tree.
//!
//! Two knobs were measured against their alternatives the same way, alternated
//! run for run against a same-snapshot build, and **neither difference was
//! visible in pass counts**:
//!
//! * Ratio of the two independent minima vs. min of the per-round ratios: 30
//!   runs each at load 68-76, both 30/30. The estimator is a min of ratios on
//!   the structural argument below, not on a failure-rate difference.
//! * [`ROUNDS`] 5 vs. 9: 30 runs each at load 34-38, both 30/30, per-cell worst
//!   readings within 0.07 of each other. 9 is kept for tail margin at a cost of
//!   ~0.35 s per run; this corpus does not show it earning that.
//!
//! So [`measure_cell`] times both sides in the same round, divides, and takes
//! the min over [`ROUNDS`], after [`SETTLE_BATCHES`] untimed batches. The first
//! batch after a shape change legitimately pays to bridge; a warm-up cost is
//! not the defect. Margins at the worst of those 40 runs are 1.56-2.09x per
//! cell.
//!
//! Neither ceiling was widened to get there — the whole change is to the
//! estimator. If a future red is real, it will be real at the same thresholds
//! this file has always used.
//!
//! ## ⛔ That last sentence is REFUTED, and the estimator is load-seeking
//!
//! Measured 2026-08-11 at pyre `9970be67cb2`, 8 cells x 2 backends x 3 runs,
//! every cell read off the unconditional per-cell `eprintln!` rather than off
//! a failure message — a gate that prints only when it fails cannot be
//! compared against the arm that passes. `min` of per-round ratios is minimised by the
//! round whose DENOMINATOR was largest — that is, by the round where the clean
//! VM was most starved of CPU. So the estimator does not merely tolerate load,
//! it **selects for it**, and the gate gets more permissive as the box gets
//! busier.
//!
//! Prediction registered before looking: within each cell, the run with the
//! largest `clean` has the smallest `fraction`. **15 of 16 cells confirm.** The
//! 16th is not a counter-example — its `clean` spread is 1.0x (35.4-36.1
//! ns/row), so there was no denominator variation for the min to seek.
//!
//! The consequence is a gate whose verdict is not a function of the code:
//!
//! | dynasm cell | run 1 | run 2 | run 3 |
//! |---|---|---|---|
//! | `warm=None measured=64` clean | 3324.8 | 473.6 | 891.1 |
//! | fraction | 0.031 **pass** | 0.226 FAIL | 0.113 FAIL |
//!
//! Three of the four diagonal cells flip verdict across three runs. The
//! numerator over those same runs moves 1.06x; `clean` moves up to 30x
//! (`warm=Some(2) measured=2`, 30.1-918.1). **The ratio inherits the
//! denominator's noise, and the min-selection amplifies it in one direction.**
//!
//! ⛔ Why the 40/0 table above could not have caught this: every one of those
//! 40 runs was cranelift, which clears [`DIAGONAL_CEILING`] with 2-8x of room
//! — a figure that is itself an artefact of the biased estimator, and reads
//! 1.02x once it is fixed. See the measured table further down.
//! An estimator change that makes a comfortable pass more comfortable is
//! indistinguishable from one that blinds the gate **unless the corpus
//! contains a subject near the threshold**. dynasm sits at 0.02-0.23 against a
//! 0.10 ceiling and did not exist in that A/B. A pass-count A/B measures an
//! estimator's STABILITY; only a near-threshold subject measures its
//! SENSITIVITY.
//!
//! ⚠ What survives all of this: the load-independent reading is the NUMERATOR,
//! and it is stable. Settled ns/row, median of 3, diagonal cells: cranelift
//! 35.4 / 35.8 / 1.8 / 1.9, dynasm 104.0 / 102.6 / 6.6 / 4.0 — dynasm is
//! 2.1-3.7x slower on the matched shape, and that is a real gap, not a
//! flake. Do NOT re-bless or widen a ceiling to make it green.
//!
//! ## ⭐ The two backends are not ranked — the gap INVERTS
//!
//! Same run, medians of 3, `dy/cl` of settled ns/row:
//!
//! | cell | ceiling | cranelift | dynasm | dy/cl |
//! |---|---|---|---|---|
//! | `warm=None measured=64` | 0.10 | 35.4 | 104.0 | 2.94x |
//! | `warm=Some(64) measured=64` | 0.10 | 35.8 | 102.6 | 2.87x |
//! | `warm=Some(2) measured=64` | 1.00 | 418.9 | 71.4 | **0.17x** |
//! | `warm=Some(3) measured=64` | 1.00 | 434.0 | 59.3 | **0.14x** |
//!
//! cranelift wins the settled matched shape by ~3x; dynasm wins the
//! shape-change path by 2.7-7.1x. "dynasm is slower" is false as a general
//! statement, and the table under "The backend control" above reads the gap as
//! one-directional because it predates these cells being measured together.
//!
//! Only the diagonal has a tight ceiling, so the single tight number can only
//! ever indict dynasm; [`OFF_DIAGONAL_CEILING`] is a deliberate
//! widening-detector at 1.00, which cranelift clears — the worst off-diagonal
//! reading recorded for it anywhere in this file is 0.731x, under the median
//! estimator with 9 spinners running; the quiet worst is 0.697x. (Both figures
//! were 0.567x while this file used the min estimator, for the reason the
//! section above gives.) No
//! fraction range is quoted for those cells on purpose: per the section above
//! the fraction is the unstable half of the measurement, and citing one would
//! be the error this file now warns about. Compare the settled ns/row column
//! instead. Neither constant can see the backend the other one misses. Sizing
//! them is #120 and is deliberately NOT done here — this section records the
//! measurement, and changing a threshold is a separate decision from
//! discovering that it grades two populations.
//!
//! ⚠ Denominator for everything in these two sections: one host, macOS arm64,
//! load 21-29, 8 cells x 2 backends x 3 runs. The within-host backend
//! comparison is sound — same run, same binary shape, stable numerator. The
//! absolute ns/row figures are NOT portable and must not be quoted as such.
//!
//! Re-pricing this is `n` runs of the built test binary counting passes, not
//! one green: the old estimator produced greens routinely, which is exactly
//! what made a single red uninformative. Build both arms back to back and
//! alternate them — this box shares a worktree with other sessions, and a
//! `cel` source edit between the two builds puts a library difference inside
//! what looks like an estimator A/B.
//!
//! ## ✅ The estimator was replaced, and here is what it measured
//!
//! [`measure_cell`] now reports a ratio of per-round medians and refuses to
//! grade a cell whose clean side was too dispersed. Measured over 7 runs per
//! backend on one host, alternating the two prebuilt binaries so no rebuild
//! sits inside the A/B, plus a 3-run-per-backend arm with 9 spinners on 18
//! cores:
//!
//! | | old (min of ratios) | new (ratio of medians) |
//! |---|---|---|
//! | diagonal cells holding one verdict | 1 of 4 (dynasm, 3 runs) | **16 of 16** (both backends, 7 runs) |
//! | verdict flips observed | pass↔FAIL | pass↔UNMEASURABLE only |
//! | cranelift diagonal headroom | "2-8x" | **1.02x at worst** (0.098 vs 0.10) |
//!
//! ⭐ The third row is the one to read twice. The old estimator's headroom was
//! never real — a `min` over ratios is optimistically biased by construction,
//! so the 2-8x was the bias, not margin. Reading a threshold's safety off a
//! biased estimator overstates it in exactly the direction that hides a
//! regression.
//!
//! ### ⛔ A pre-registered prediction of mine was REFUTED
//!
//! I predicted the median would make the fraction load-invariant. It does not:
//! in 8 of 16 backend-cells the clean median ROSE under load while the fraction
//! FELL. The cause is physical, not statistical — the clean VM does ~16x the
//! work per row of the compiled tier, so contention costs it more, and the two
//! sides simply do not degrade at the same rate. **A ratio of any two
//! differently-load-sensitive timings is load-dependent, and no reduction over
//! rounds can fix that.**
//!
//! What the median did fix is the magnitude and the direction's reach: the
//! worst fall is now 0.57x where the old estimator's was 7.3x, and across 24
//! graded diagonal readings under load no cell crossed a ceiling — cranelift
//! read 0 FAIL / 8 pass / 4 refused, dynasm 9 FAIL / 0 pass / 3 refused. The
//! residual bias is permissive, so it is still the dangerous direction; it is
//! now small enough that [`CLEAN_SPREAD_CEILING`] removes the windows where it
//! is largest before it can reach a verdict.
//!
//! ### Calibrating [`CLEAN_SPREAD_CEILING`]
//!
//! `q3/q1` of the clean side. Every refusal count is taken against the SHIPPED
//! ceiling, so the arms are comparable with each other; a count measured against
//! some other candidate belongs in a different table.
//!
//! | arm | load avg (1-min) | p50 | p90 | p98 | max | refused |
//! |---|---|---|---|---|---|---|
//! | least contended measured | 11.56 → 11.54 | 1.09 | 1.22 | 1.28 | 1.31 | 0 of 112 (0%) |
//! | contended | 41.72 | 1.11 | 1.27 | 1.41 | 2.55 | 4 of 112 (4%) |
//! | + 9 spinners on 18 cores | 39.4-41.3 | 1.27 | 2.19 | 2.68 | 2.68 | 22 of 48 (46%) |
//!
//! 0% → 4% → 46% is the mechanism working: it stands down entirely on the
//! quietest box anyone has measured, engages weakly on a contended one, and
//! refuses most of a deliberately saturated one. Refused cells on the loaded arm
//! carry a median spread of 1.85 against the graded cells' 1.09.
//!
//! ⚠ It has still never prevented a wrong answer on this corpus. On the least
//! contended arm it fired 0 times in 112 cells and every verdict was reached
//! without it; on the contended arm none of its firings changed a verdict. It is
//! insurance whose premium is measured and whose payout is not.
//!
//! ⚠ Refusal also correlates with cell DURATION (the `m=64` cells run ~11ms per
//! round against `m=2`'s ~0.6ms), so a count of refusals is not a reading of box
//! load.
//!
//! ## ⛔ The rule that sized the first ceiling was self-defeating
//!
//! 1.50 was set at the refusal rate of a single arm — "refuse the worst ~2%" —
//! and that rule cannot work, for a reason that has nothing to do with which box
//! it is derived on: **a quantile of the observed population always refuses that
//! quantile's share, by construction.** Re-derive it on a quiet box and it
//! tightens; re-derive it on a burning one and it loosens; either way the gate
//! refuses 2% and has learned nothing about whether the box was fit to measure
//! on. A bound whose job is to detect "too noisy to grade" cannot be defined
//! relative to the noise.
//!
//! ⇒ The ceiling is an ABSOLUTE dispersion bound: above it, the middle of the
//! clean sample is too unsettled for a median ratio to carry meaning.
//! Measurement cannot supply that number. What measurement supplies is an UPPER
//! BOUND on it — you cannot set it below what good hardware actually achieves
//! without refusing everything — so each arm ratchets the constant DOWN and no
//! arm ever fixes it. The shipped value is the current rung, **tightened
//! toward** a
//! correct bound, never *set to* one.
//!
//! ⚠ Arm provenance, stated rather than adjectived: measured at loadavg
//! 11.56 → 11.54 (1-minute, at the start and end of a 10-second sweep; the 5-
//! and 15-minute averages moved 14.15 → 14.06 and 20.52 → 20.41, and one
//! competing `rustc` was running at both ends) on a shared box carrying
//! unrelated load. Start and end agree, so the arm is one population and not two
//! averaged together. **The dedicated-CI-runner population remains unmeasured**,
//! and it is quieter than anything here, so the shipped value is still an upper
//! bound on what
//! is correct for it.
//!
//! ⛔ The predecessor of this section called its baseline arm "quiet" and never
//! measured it. It ran at 41.72 — printed in every run header — and the constant
//! derived from it was mis-sized permissive for exactly as long as the adjective
//! went unchecked. A control named for a property nobody measured is a second
//! treatment arm with an optimistic name, which is why every arm above is
//! labelled with a reading instead of a word.
//!
//! ⭐ Why the third value is structural rather than a safety valve: this gate
//! serves TWO populations — a comparatively dedicated CI runner, and a
//! contended developer box whose reference measurement differs from it by an
//! order of magnitude. No single ceiling is honest in both. The ceiling belongs
//! to the quiet population; the spread bound decides when the gate is entitled
//! to use it. If that refuses most local runs, that is the correct answer — "not
//! measurable here, CI will judge it" is strictly better than a verdict that
//! inverted because someone else was compiling.
//!
//! ⚠ Refusal is correlated with cell DURATION, not purely with box load. The
//! `measured=64` cells run ~11 ms of clean work per round against the
//! `measured=2` cells' ~0.6 ms, so they have more opportunity to be descheduled
//! and they dominate the refused set. That is defensible — a longer round is
//! genuinely more disturbed — but it means an UNMEASURABLE count is not a
//! reading of how busy the box was.
//!
//! ### Numbers for the ceiling decision, which is deliberately NOT taken here
//!
//! The pooled per-backend readings live in [`DIAGONAL_CEILING`]'s own doc,
//! beside the constant they are evidence about, and deliberately in one place
//! only — a measured table copied to two locations is a pair of independent
//! assertions that nothing forces to agree.
//!
//! [`DIAGONAL_CEILING`] was sized on cranelift-only data under the old
//! estimator, and both of those facts push it the same way. Changing it is a
//! separate decision from fixing the instrument that feeds it, and bundling
//! them would make neither attributable.
//!
//! ## ⭐⭐⭐ A margin read through a biased estimator is not a margin
//!
//! This is the finding to carry off this file, and it is bigger than the
//! estimator repair that produced it. The doc here asserted for a long time
//! that cranelift cleared its ceiling "with 2-8x of room", and that sentence is
//! why nobody questioned `0.10`. Replacing the `min` with a ratio of medians —
//! **changing nothing whatsoever about cranelift** — put its worst diagonal
//! reading at 0.098 against a 0.10 ceiling.
//!
//! > **The margin was manufactured by the instrument**, and manufactured in
//! > exactly the direction that hides a regression. Before citing headroom as
//! > safety, ask what reduction produced it: a `min`, a best-of-N, or a
//! > hand-picked quiet run all report optimism as slack.
//!
//! The corollary for anyone repairing an estimator here later: **the repair is
//! never threshold-neutral.** Every constant sized against the old reducer is
//! calibrated to a quantity that selected for load, so re-sizing is not
//! optional follow-up work — it is the other half of the change, and it needs
//! its own measurement rather than an assumption that the old margin survived.
//!
//! ## Reproducing the sweep
//!
//! ⛔ Recorded because it is the perishable part. Sizing any ceiling here needs
//! a **Linux x86_64** run (cel-jit's workflow is `ubuntu-latest`; every number
//! in this file is macOS arm64), and whoever takes it may have none of the
//! context above. It is two prebuilt binaries and a shell loop:
//!
//! ```text
//! # 1. Build BOTH backends first and lift the binaries out, so that no rebuild
//! #    ever sits inside the A/B. --release is required: the tier only reaches
//! #    its compiled trace in release. Never bare `--features jit` -- it names
//! #    no backend and majit-metainterp turns that into a compile_error!.
//! for BK in cranelift dynasm; do
//!   cargo test --locked --release -p cel --features "jit-$BK" \
//!       --test majit_shape_change --no-run --message-format=json > "build-$BK.json"
//!   # Take the path from cargo's OWN record. target/release/deps holds stale
//!   # binaries from earlier feature sets, so a name glob silently mixes builds.
//!   EXE=$(python3 -c 'import json,sys
//! for l in open(sys.argv[1]):
//!     try: m=json.loads(l)
//!     except Exception: continue
//!     if m.get("reason")=="compiler-artifact" and m.get("executable") \
//!        and m.get("target",{}).get("name")=="majit_shape_change": print(m["executable"])
//! ' "build-$BK.json" | tail -1)
//!   cp "$EXE" "shape-$BK"
//! done
//!
//! # 2. Alternate them. 7 runs per backend settled every cell here.
//! for R in 1 2 3 4 5 6 7; do
//!   for BK in cranelift dynasm; do
//!     uptime                                    # RECORD THE LOAD. See below.
//!     "./shape-$BK" --nocapture --test-threads=1
//!   done
//! done
//! ```
//!
//! ⛔ **Record the load average and check it before calling the arm quiet.** The
//! calibration above was taken at load 41.7 on a box whose team routinely runs
//! seven concurrent builds, and it was written up as "quiet" until the number
//! was read back. A control arm named for a property nobody measured is an
//! assertion, not a control.
//!
//! ⚠ Two hosts is the minimum useful comparison and this file has one. The
//! backend gap in particular is a codegen property: dynasm emits machine code
//! directly, so nothing measured on arm64 predicts x86_64.

#![cfg(feature = "jit")]

use std::time::{Duration, Instant};

use cel::majit::bytecode::float_bank::{
    jit_stats, reset_jit_stats, reset_persistent_state, JitStats,
};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// Serializes against the other majit test binaries, which reset and read the
/// same process-global evidence counters and share the thread-local drivers.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

const ROWS: usize = 20_000;
const WARM_ROWS: usize = 4_000;
const THRESHOLD: u32 = 8;

/// Timed rounds per cell; the reported fraction is the ratio of their medians.
///
/// A median needs a MAJORITY of rounds to land in windows the box is not
/// contending for — where the old min needed only one. That is a strictly
/// harder requirement, and it is the point: a reading that survives it is a
/// reading about the code, and one that does not is refused by
/// [`CLEAN_SPREAD_CEILING`] rather than reported.
///
/// ⛔ The prior 5-vs-9 result recorded here (30/30 each at load 34-38, per-cell
/// worst readings within 0.07) was measured under the min estimator and does NOT
/// transfer: it graded how far the best round could reach, and nine rounds
/// bought a wider search for that best round. Under a median the same count buys
/// tolerance of up to four contended rounds instead. 9 is kept — it is the more
/// robust of the two under the new reduction, at ~0.35 s per run — but no
/// measurement in this file now compares it against 5. The whole test runs in
/// about a second.
const ROUNDS: usize = 9;
/// The median and the `q3`/`q1` pair below are both nearest-rank on an odd
/// sample, so every statistic this file reports is a value that was actually
/// observed rather than an average of two that were not.
const _: () = assert!(ROUNDS % 2 == 1 && ROUNDS >= 5);
/// Compiled batches run before timing starts. The first batch after a shape
/// change legitimately pays to bridge, and a warm-up cost is not the defect
/// this file exists to catch.
const SETTLE_BATCHES: usize = 2;

/// ⛔ THIS NUMBER WAS SIZED AGAINST A REDUCER SINCE SHOWN TO BE LOAD-SEEKING,
/// AND IT HAS NOT BEEN RE-SIZED. Do not read it as a value anything has
/// validated.
///
/// Its original justification was "cold and diagonal cells measure 0.001-0.064x
/// of the clean VM over 40 runs at load 50-56". Both that range and this
/// constant come from the `min`-of-ratios estimator the module doc refutes, so
/// the range is a lower bound on what the same cells read now.
///
/// Corrected-estimator readings, pooled across load conditions, graded cells
/// only:
///
/// | backend | n | per-cell medians | worst | vs 0.10 |
/// |---|---|---|---|---|
/// | cranelift | 36 | 0.056 / 0.058 / 0.066 / 0.073 | **0.098** | clears by **1.02x** |
/// | dynasm | 37 | 0.123 / 0.125 / 0.182 / 0.190 | 0.222 | fails |
///
/// ⚠ Read the cranelift row as an operational warning, not a pass. 0.098
/// against 0.10 is not "comfortable headroom" — it is *at* the threshold, one
/// slow day from a red that would present as a code regression. The file used
/// to say cranelift had 2-8x of room; that figure was the estimator's bias, and
/// correcting the instrument is what revealed the true position. Nothing about
/// cranelift changed.
///
/// ⛔⛔ AND THIS GATE HAS EFFECTIVELY NEVER RUN ANYWHERE, so `0.10` is not a
/// number that has been surviving CI:
///
/// * `origin/majit` does not exist — this branch has never been pushed, so
///   cel-jit's own workflow has never executed on it and there is no historical
///   green or red to compare a future verdict against;
/// * `-p cel` in the *parent* pyre-wasmi workspace resolves to
///   `majit/examples/cel`, a DIFFERENT crate. cel-jit has zero tracked files in
///   pyre-wasmi, so `pyre-ci.yml` never builds this test at all.
///
/// ⇒ PRECONDITION ON ANY CHANGE TO THIS VALUE: a Linux x86_64 sweep, because
/// cel-jit's workflow runs `ubuntu-latest` and every number above is macOS
/// arm64. dynasm is an *assembler* backend, so the 2.1-3.7x backend gap is an
/// arm64 codegen property with no reason to transfer. The module doc's
/// "Reproducing the sweep" section has the commands.
const DIAGONAL_CEILING: f64 = 0.10;
/// Off-diagonal cells measure 0.007-0.567x of the clean VM over the same 40
/// runs. See the module doc for why this is a widening-detector and not an
/// endorsement. The same caveat as [`DIAGONAL_CEILING`] applies to the range.
const OFF_DIAGONAL_CEILING: f64 = 1.00;

/// The clean side's `q3/q1` across [`ROUNDS`], above which a cell is graded
/// [`Verdict::Unmeasurable`] instead of pass or fail.
///
/// This is a bound on the DENOMINATOR's own dispersion, not on the ratio. The
/// median resists a minority of contended rounds; what it cannot survive is the
/// box being contended for most of the window, and `q3/q1` is the statistic that
/// separates those two — it asks whether the middle of the clean sample is
/// settled, where a `max/min` would be decided by the single worst round and
/// would refuse to grade a sample the median handles comfortably.
///
/// ⚠ This is an UPPER BOUND that has been tightened, not a value that has been
/// determined. It is the tightest 0.05-granular number that leaves the least
/// contended arm anyone has measured here entirely graded — that arm's worst
/// cell spread is 1.31 over 112 readings. A quieter box would justify a lower
/// one, and the CI runner is quieter than any box these numbers came from. See
/// the module doc for why a quantile of the observed population is the wrong
/// rule, and for the arm's measured load.
///
/// ⭐ Tightening is the safe direction and that is structural, not a judgement:
/// too tight and cells go [`Verdict::Unmeasurable`], which the run reports and
/// which fails loudly if it swallows *every* cell, since a graded count of zero
/// is an assertion failure rather than a pass. Too loose and a cell is graded
/// off a denominator that does not mean anything, which is silent. Only one of
/// those two errors announces itself.
///
/// ⚠ Read the following as a consequence, not as the sizing rule — the constant
/// is derived from the spread distribution alone, because sizing a
/// noise-detector against the verdicts it protects is how a threshold gets tuned
/// until it gives the wanted answer. The consequence: on that arm the shipped
/// ceiling refuses one cell of 112 and it is a `pass`, while every FAIL sits at
/// or below 1.30, so no demonstrated true positive is converted into a refusal.
/// A ceiling at the arm's strict p98 of 1.28 *would* have refused one FAIL.
const CLEAN_SPREAD_CEILING: f64 = 1.35;

/// The worst clean `q3/q1` observed on the arm [`CLEAN_SPREAD_CEILING`] was
/// derived from, kept as a separate constant so the ceiling cannot drift below
/// its own justification silently.
///
/// Lowering the ceiling past this is not a tuning decision, it is a claim that
/// a different arm was measured — so it has to move together with the module
/// doc's calibration table, and this trips if it does not.
const LEAST_CONTENDED_ARM_WORST_SPREAD: f64 = 1.31;
const _: () = assert!(
    CLEAN_SPREAD_CEILING > LEAST_CONTENDED_ARM_WORST_SPREAD,
    "CLEAN_SPREAD_CEILING is at or below the worst spread of the arm it was \
     sized on, so it would refuse cells that arm graded. Re-measure and update \
     the calibration table before lowering it."
);

struct ListColumns {
    lens: Vec<i64>,
    offsets: Vec<i64>,
    elems: Vec<i64>,
}

impl ListColumns {
    /// Every element SATISFIES `i.price > 10`, so `all` never short-circuits and
    /// the inner loop's real trip count is `len` rather than "index of the first
    /// failing element", which would be data rather than shape.
    fn build(rows: usize, len: i64) -> Self {
        let lens = vec![len; rows];
        let mut offsets = Vec::with_capacity(rows);
        let mut total = 0i64;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        let elems = (0..total.max(1)).map(|k| 11 + (k * 7) % 40).collect();
        Self {
            lens,
            offsets,
            elems,
        }
    }

    fn columns(&self) -> [Column<'_>; 3] {
        [
            Column::Int(&self.lens),
            Column::Int(&self.offsets),
            Column::Int(&self.elems),
        ]
    }
}

fn lowered() -> LoweredF {
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let program = Program::compile("items.all(i, i.price > 10)").expect("parse");
    lower_typed(program.expression(), &schema).expect("lower")
}

fn ns_per_row(d: Duration) -> f64 {
    d.as_secs_f64() * 1e9 / ROWS as f64
}

/// What one cell measured, and how much the reading can be trusted.
struct Cell {
    /// Median compiled ns/row over [`ROUNDS`] divided by median clean ns/row.
    fraction: f64,
    /// Median compiled ns/row — the load-independent half of the reading, and
    /// the one to quote when comparing backends or hosts.
    jit_ns: f64,
    /// Median clean ns/row: the denominator of `fraction`.
    clean_ns: f64,
    /// `q3/q1` of the clean side's per-round ns/row. See
    /// [`CLEAN_SPREAD_CEILING`].
    clean_spread: f64,
    stats: JitStats,
}

/// Nearest-rank median of a sample sorted by [`f64::total_cmp`].
///
/// [`ROUNDS`] is odd (asserted beside it), so this is the true middle and never
/// an interpolation between two neighbours.
fn median_of_sorted(sorted: &[f64]) -> f64 {
    sorted[sorted.len() / 2]
}

/// One cell's fraction-of-the-clean-VM, the ns/row behind it, the clean side's
/// dispersion, and the compile counters for the run that produced them.
///
/// ## Why the fraction is a ratio of per-round medians
///
/// Three estimators have stood here. The history is kept because each was
/// replaced for a different reason, and two of the three defects are invisible
/// in a pass count:
///
/// | # | reduction | why it was replaced |
/// |---|---|---|
/// | 1 | clean: min of 3 batches; compiled: last of 3 | the two sides were never in the same window — every clean batch ran before the warm-up. 13 failures in 40 runs at load 50-56, reading up to 3.29x its ceiling |
/// | 2 | **min** of the per-round ratios | the min is achieved in the round with the LARGEST DENOMINATOR, so it selects for load and the gate loosens as the box gets busier. Three of four diagonal cells flipped verdict across three runs |
/// | 3 | ratio of the per-round **medians** | current |
///
/// The round is still the unit, and for the reason estimator 2 got right: the
/// quantity that cancels machine load is a ratio of two timings taken in the
/// *same* window, so the clean VM and the compiled tier are timed back to back
/// and each round yields one pair. What changed is the reduction over those
/// pairs. Taking the extreme of a noisy sample does not find the quiet round; it
/// finds the round whose noise happened to fall on the favourable side, and for
/// a ratio that is the round where the *denominator* was worst. A median asks
/// the opposite question — what does this cell usually cost — and no single
/// round, however contended, can move it.
///
/// ⚠ Medians of numerator and denominator, not a median of the per-round
/// ratios. The two differ, and this is the weaker of the pair: it pairs the
/// typical compiled round with the typical clean round even when those are
/// different rounds. It is chosen because it makes `jit_ns` and `clean_ns`
/// reportable as themselves — a median of ratios has no numerator to print, and
/// the numerator is the half of this measurement that survived the estimator
/// defect above and the half that is comparable across hosts.
///
/// ## What it still cannot do
///
/// A median hides intermittent regressions by construction: one slow round in
/// nine does not move it, and this file does not claim to catch them. Its
/// subject is settled steady-state cost. It also cannot rescue a window in which
/// the box was contended for most of the rounds — for that the cell reports
/// [`Verdict::Unmeasurable`] rather than a number, which is the whole reason
/// `clean_spread` is computed and returned alongside the fraction.
///
/// The clean VM is a plain-`match` interpreter with no tracing machinery
/// (`bytecode.rs:1437`), so interleaving it between compiled batches reads the
/// driver's state without disturbing it.
///
/// Every batch on both sides is checked against the oracle answer, so no timing
/// here is ever taken off a miscompile.
fn measure_cell(lowered: &LoweredF, warm: Option<i64>, measured: i64) -> Cell {
    let mc = ListColumns::build(ROWS, measured);

    reset_persistent_state();
    reset_jit_stats();
    if let Some(w) = warm {
        let wc = ListColumns::build(WARM_ROWS, w);
        let r = eval_batch_sum_f(lowered, &wc.columns(), WARM_ROWS, THRESHOLD);
        assert!(r.is_some(), "warm-up batch at trip {w} declined");
    }

    let oracle = clean_batch_sum_f(lowered, &mc.columns(), ROWS);
    for i in 0..SETTLE_BATCHES {
        let got = eval_batch_sum_f(lowered, &mc.columns(), ROWS, THRESHOLD);
        assert_eq!(
            got, oracle,
            "warm={warm:?} measured={measured} settle batch {i}: answer diverged from the clean VM"
        );
    }
    // Read the counters once the tier has settled and before any timing, so the
    // structural assertion is about the same state the timings describe.
    let stats = jit_stats();

    let mut jit_samples = Vec::with_capacity(ROUNDS);
    let mut clean_samples = Vec::with_capacity(ROUNDS);
    for i in 0..ROUNDS {
        let t0 = Instant::now();
        let got_clean = clean_batch_sum_f(lowered, &mc.columns(), ROWS);
        let clean_ns = ns_per_row(t0.elapsed());

        let t1 = Instant::now();
        let got_jit = eval_batch_sum_f(lowered, &mc.columns(), ROWS, THRESHOLD);
        let jit_ns = ns_per_row(t1.elapsed());

        assert_eq!(
            got_clean, oracle,
            "warm={warm:?} measured={measured} round {i}: the clean VM disagreed with itself"
        );
        assert_eq!(
            got_jit, oracle,
            "warm={warm:?} measured={measured} round {i}: answer diverged from the clean VM"
        );

        jit_samples.push(jit_ns);
        clean_samples.push(clean_ns);
    }

    jit_samples.sort_by(f64::total_cmp);
    clean_samples.sort_by(f64::total_cmp);

    // Middle 50%-ish of the clean sample: with ROUNDS=9 this is v[2] and v[6],
    // the middle five. A `max/min` here would be a report on the single worst
    // round, which is exactly the reading the median was adopted to stop
    // deciding the verdict.
    let k = ROUNDS / 4;
    let (q1, q3) = (clean_samples[k], clean_samples[ROUNDS - 1 - k]);
    // A non-positive q1 cannot happen for a batch of 20_000 rows on any clock
    // this test can run on, but if the timer ever did return 0 the quotient
    // must not become a small number that reads as a settled box.
    let clean_spread = if q1 > 0.0 { q3 / q1 } else { f64::INFINITY };

    let jit_ns = median_of_sorted(&jit_samples);
    let clean_ns = median_of_sorted(&clean_samples);
    Cell {
        fraction: jit_ns / clean_ns,
        jit_ns,
        clean_ns,
        clean_spread,
        stats,
    }
}

/// What a cell's timings support saying about it.
///
/// The third value is the point: a two-valued gate must call a window it could
/// not measure either a pass or a failure, and both are false statements about
/// the code. See [`CLEAN_SPREAD_CEILING`].
enum Verdict {
    Pass,
    Fail,
    Unmeasurable,
}

impl Verdict {
    /// The printed label is DERIVED from the value the branch below reads, so
    /// the two cannot drift apart.
    fn label(&self) -> &'static str {
        match self {
            Verdict::Pass => "pass",
            Verdict::Fail => "FAIL",
            Verdict::Unmeasurable => "UNMEASURABLE",
        }
    }
}

#[test]
fn a_trip_count_change_keeps_the_tier_compiled_and_never_worse_than_no_jit() {
    let _serial = serial();
    let lowered = lowered();

    // Both directions, plus cold and diagonal controls. `None` warm is cold.
    let cases: [(Option<i64>, i64); 8] = [
        (None, 64),
        (Some(64), 64),
        (None, 2),
        (Some(2), 2),
        (Some(2), 64),
        (Some(3), 64),
        (Some(64), 2),
        (Some(8), 2),
    ];

    let mut failures = Vec::new();
    let mut graded = 0usize;
    let mut unmeasurable = 0usize;
    for (warm, measured) in cases {
        let Cell {
            fraction,
            jit_ns,
            clean_ns,
            clean_spread,
            stats,
        } = measure_cell(&lowered, warm, measured);
        let healthy = warm.is_none() || warm == Some(measured);
        let ceiling = if healthy {
            DIAGONAL_CEILING
        } else {
            OFF_DIAGONAL_CEILING
        };
        let verdict = if clean_spread > CLEAN_SPREAD_CEILING {
            Verdict::Unmeasurable
        } else if fraction > ceiling {
            Verdict::Fail
        } else {
            Verdict::Pass
        };
        // Printed for every cell whatever the verdict, and printed with its
        // spread: a gate that prints only when it fails cannot be compared
        // against the arm that passes, which is how the previous estimator's
        // defect stayed invisible through 40 green runs.
        eprintln!(
            "[shape-change] warm={warm:?} measured={measured} settled={jit_ns:.1} \
             clean={clean_ns:.1} spread={clean_spread:.2} fraction={fraction:.3} \
             ceiling={ceiling} verdict={} loops_compiled={} bridges={} panics={}",
            verdict.label(),
            stats.loops_compiled,
            stats.bridges_compiled,
            stats.internal_compile_panics
        );

        // Load-independent floor. "The tier is compiled at all" is a statement
        // about compile counts, not wall-clock, and it is the half of this
        // file's claim 1 that a timing ratio should never have been carrying:
        // if compilation broke outright, this fires identically on a quiet box
        // and a box at load 35. Nothing here asserts anything about the SECOND
        // shape — `loops_compiled` is 0 for a degraded off-diagonal batch by
        // design, and pinning that would turn the defect into a baseline.
        if stats.loops_compiled == 0 {
            failures.push(format!(
                "warm={warm:?} measured={measured}: nothing compiled — the tier never \
                 built an artefact, so the timings below describe the interpreter"
            ));
        }
        if stats.internal_compile_panics != 0 {
            failures.push(format!(
                "warm={warm:?} measured={measured}: {} trace(s) dropped by a panic inside \
                 compilation",
                stats.internal_compile_panics
            ));
        }

        match verdict {
            Verdict::Fail => {
                graded += 1;
                failures.push(format!(
                    "warm={warm:?} measured={measured}: {jit_ns:.1} ns/row is {fraction:.2}x the \
                     clean VM's {clean_ns:.1} (ceiling {ceiling}, clean spread {clean_spread:.2})"
                ));
            }
            Verdict::Pass => graded += 1,
            Verdict::Unmeasurable => unmeasurable += 1,
        }
    }

    // The denominator for every timing verdict above. Without it a run in which
    // the box was too busy to measure anything is indistinguishable from a run
    // in which the tier was fast in all eight cells.
    eprintln!("[shape-change] timing cells graded={graded} unmeasurable={unmeasurable}");

    // A run that graded nothing is not a passing run, and it is not a
    // regression either — it is an absent measurement, and it fails under its
    // own name so that it can never be read as evidence about the tier.
    assert_ne!(
        graded, 0,
        "no timing cell was measurable: the clean VM's per-round spread exceeded \
         {CLEAN_SPREAD_CEILING} in all {unmeasurable} cells, so this run says nothing \
         about the compiled tier's cost. Re-run on a quieter box; do NOT read this as a pass"
    );
    assert!(
        failures.is_empty(),
        "the tier is not staying ahead of the untraced VM it exists to beat \
         ({graded} cells graded, {unmeasurable} unmeasurable):\n  {}",
        failures.join("\n  ")
    );
}
