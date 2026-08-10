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
//! trust. And cel on dynasm is not yet a sound backend — 5 of 220 `cel` unit
//! tests miscompile there (a ternary sum, and float/list-valued results
//! reading back integer bit patterns), so the dynasm column is trustworthy
//! only because [`measure_cell`] asserts every timed batch against the clean VM and
//! this particular predicate answers correctly.
//!
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
//! Re-pricing this is `n` runs of the built test binary counting passes, not
//! one green: the old estimator produced greens routinely, which is exactly
//! what made a single red uninformative. Build both arms back to back and
//! alternate them — this box shares a worktree with other sessions, and a
//! `cel` source edit between the two builds puts a library difference inside
//! what looks like an estimator A/B.

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

/// Timed rounds per cell; the reported fraction is the min of their ratios.
///
/// Taking a min over `n` rounds only helps if at least one round lands in a
/// window the box is not contending for, so the count is set by the slowest
/// cell, not the average one — `warm=Some(2) measured=64`, ~10 ms of compiled
/// work against ~18 ms of clean work per round.
///
/// 5 and 9 were measured against each other, alternated run for run at load
/// 34-38: **both passed 30 of 30**, with per-cell worst readings within 0.07.
/// So this corpus does not show 9 buying anything over 5. It is kept because
/// the cost is ~0.35 s per run and the extra rounds can only widen the window
/// the min is taken over — but that is a headroom argument, not a measurement,
/// and lowering it to 5 would not contradict anything measured here. The whole
/// test runs in about a second.
const ROUNDS: usize = 9;
/// Compiled batches run before timing starts. The first batch after a shape
/// change legitimately pays to bridge, and a warm-up cost is not the defect
/// this file exists to catch.
const SETTLE_BATCHES: usize = 2;

/// Cold and diagonal cells measure 0.001-0.064x of the clean VM over 40 runs at
/// load 50-56.
const DIAGONAL_CEILING: f64 = 0.10;
/// Off-diagonal cells measure 0.007-0.567x of the clean VM over the same 40
/// runs. See the module doc for why this is a widening-detector and not an
/// endorsement.
const OFF_DIAGONAL_CEILING: f64 = 1.00;

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

/// One cell's fraction-of-the-clean-VM, the ns/row behind it, and the compile
/// counters for the run that produced them.
///
/// ## Why the fraction is the min of per-round ratios
///
/// The quantity that cancels machine load is the ratio of two timings taken in
/// the *same* window, so the round is the unit: time the clean VM and the
/// compiled tier back to back, divide, and take the min of that ratio over
/// [`ROUNDS`]. A round whose window was not uniform — a spike landing on one
/// half and not the other — inflates that round's ratio and the min discards
/// it. This is what the module doc always claimed the design did.
///
/// It did not. The clean side took a **min of 3** batches while the compiled
/// side ran 3 and kept the **last**, so one descheduled batch landing on the
/// final iteration went into the numerator undefended, and the two sides were
/// not in the same window at all — every clean batch ran before the warm-up
/// did. Over 40 runs at load 50-56 that estimator failed 13 times, reading up
/// to 3.29x its ceiling on a cell whose worst here is 0.556x.
///
/// A ratio of the two independent minima is the other candidate, and it was
/// measured: 30 runs each at load 68-76, both it and this one passed 30/30. So
/// the choice between them is not a failure-rate result — it is that a ratio of
/// minima pairs the best compiled round with the best clean round even when
/// those are different rounds under different load, where a per-round ratio has
/// its numerator and denominator in one window by construction.
///
/// A min over ratios cannot hide a real regression: if the compiled tier is
/// genuinely slower, every round's ratio rises and so does their minimum. What
/// it does give up is intermittent regressions — one bad round in nine — which
/// this file does not claim to catch; its subject is settled steady-state cost.
///
/// The clean VM is a plain-`match` interpreter with no tracing machinery
/// (`bytecode.rs:1437`), so interleaving it between compiled batches reads the
/// driver's state without disturbing it.
///
/// Every batch on both sides is checked against the oracle answer, so no timing
/// here is ever taken off a miscompile.
fn measure_cell(lowered: &LoweredF, warm: Option<i64>, measured: i64) -> (f64, f64, f64, JitStats) {
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

    let mut best_fraction = f64::INFINITY;
    let mut jit_at_best = 0.0;
    let mut clean_at_best = 0.0;
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

        // Report the ns/row from the round that produced the reported ratio, so
        // the printed numerator and denominator are the pair it came from
        // rather than two figures from different windows.
        let fraction = jit_ns / clean_ns;
        if fraction < best_fraction {
            best_fraction = fraction;
            jit_at_best = jit_ns;
            clean_at_best = clean_ns;
        }
    }
    (best_fraction, jit_at_best, clean_at_best, stats)
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
    for (warm, measured) in cases {
        let (fraction, jit, clean_ns, stats) = measure_cell(&lowered, warm, measured);
        let healthy = warm.is_none() || warm == Some(measured);
        let ceiling = if healthy {
            DIAGONAL_CEILING
        } else {
            OFF_DIAGONAL_CEILING
        };
        eprintln!(
            "[shape-change] warm={warm:?} measured={measured} settled={jit:.1} \
             clean={clean_ns:.1} fraction={fraction:.3} ceiling={ceiling} \
             loops_compiled={} bridges={} panics={}",
            stats.loops_compiled, stats.bridges_compiled, stats.internal_compile_panics
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

        if fraction > ceiling {
            failures.push(format!(
                "warm={warm:?} measured={measured}: {jit:.1} ns/row is {fraction:.2}x the \
                 clean VM's {clean_ns:.1} (ceiling {ceiling})"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the tier is not staying ahead of the untraced VM it exists to beat:\n  {}",
        failures.join("\n  ")
    );
}
