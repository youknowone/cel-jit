//! What a driver that already compiled for one inner trip count does when the
//! next batch has a different one.
//!
//! `majit_trace_evidence.rs` censuses ONE data shape per driver on purpose, and
//! `nested_loop_deopts_are_a_warmup_cost_not_a_per_row_cost` compares two batch
//! SIZES of the same shape. Neither covers a driver that has already compiled
//! for a different SHAPE — which is what a long-lived process does, since the
//! driver and the interned program outlive a call.
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
//! Every number in this file is a CRANELIFT number: `cel`'s `jit` feature
//! selects `majit-metainterp/cranelift` and nothing else, and on that backend a
//! guard exit marshals all 23 live values through the jitframe twice per row
//! where upstream patches the guard's branch straight into a bridge that was
//! register-allocated against the guard's own fail locations
//! (`rpython/jit/backend/aarch64/assembler.py:163,200-202,1054-1060`). The
//! backend control is not available yet — cel's trace panics in the dynasm
//! register allocator — so how much of the gap is portable is still open.
//!
//! ## What this test asserts, and what it deliberately does NOT
//!
//! It does **not** pin the 10-15x. Encoding today's gap as the expectation would
//! turn a defect into a baseline. It asserts the two lines that bound it:
//!
//! 1. **The healthy path stays healthy.** Cold and diagonal runs must stay at or
//!    under [`DIAGONAL_CEILING`] of the clean VM. They measure ~0.05x today, so
//!    this catches a regression that breaks compilation outright.
//! 2. **The tier never becomes worse than no JIT at all.** Every off-diagonal
//!    cell must stay under [`OFF_DIAGONAL_CEILING`] of the clean VM. The worst
//!    cell measures ~0.72x today, so the gap has ~1.4x of room before this
//!    fires — it catches the gap WIDENING, which is the regression this file
//!    exists to prevent, without asserting that the gap is acceptable. It is
//!    not; the target is the PyPy column above.
//!
//! Both budgets are ratios against `clean_batch_sum_f` — the same lowered
//! program over the same columns with no tracing machinery — measured in the
//! same process at the same moment, so a loaded machine scales both sides and
//! cancels instead of flaking.
//!
//! Only the SETTLED batch is read. The first batch after a shape change
//! legitimately pays to bridge; a warm-up cost is not the defect.

#![cfg(feature = "jit")]

use std::time::{Duration, Instant};

use cel::majit::bytecode::float_bank::reset_persistent_state;
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

/// Cold and diagonal cells measure ~0.05x of the clean VM.
const DIAGONAL_CEILING: f64 = 0.10;
/// Off-diagonal cells measure up to ~0.72x of the clean VM. See the module doc
/// for why this is a widening-detector and not an endorsement.
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

/// The clean VM's ns/row for this trip count, and the answer every compiled run
/// is checked against — so no timing below is ever taken off a miscompile.
fn clean(lowered: &LoweredF, trip: i64) -> (f64, Option<i64>) {
    let mc = ListColumns::build(ROWS, trip);
    let mut best = Duration::MAX;
    let mut answer = None;
    for _ in 0..3 {
        let t0 = Instant::now();
        answer = clean_batch_sum_f(lowered, &mc.columns(), ROWS);
        best = best.min(t0.elapsed());
    }
    (ns_per_row(best), answer)
}

/// Settled compiled ns/row for `measured`, on a driver warmed at `warm` (cold
/// when `warm` is `None`).
fn settled(lowered: &LoweredF, warm: Option<i64>, measured: i64, oracle: Option<i64>) -> f64 {
    let mc = ListColumns::build(ROWS, measured);
    reset_persistent_state();
    if let Some(w) = warm {
        let wc = ListColumns::build(WARM_ROWS, w);
        let r = eval_batch_sum_f(lowered, &wc.columns(), WARM_ROWS, THRESHOLD);
        assert!(r.is_some(), "warm-up batch at trip {w} declined");
    }
    let mut last = Duration::ZERO;
    for i in 0..3 {
        let t0 = Instant::now();
        let got = eval_batch_sum_f(lowered, &mc.columns(), ROWS, THRESHOLD);
        last = t0.elapsed();
        assert_eq!(
            got,
            oracle,
            "warm={warm:?} measured={measured} batch {}: answer diverged from the clean VM",
            i + 1
        );
    }
    ns_per_row(last)
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
        let (clean_ns, oracle) = clean(&lowered, measured);
        let jit = settled(&lowered, warm, measured, oracle);
        let fraction = jit / clean_ns;
        let healthy = warm.is_none() || warm == Some(measured);
        let ceiling = if healthy {
            DIAGONAL_CEILING
        } else {
            OFF_DIAGONAL_CEILING
        };
        eprintln!(
            "[shape-change] warm={warm:?} measured={measured} settled={jit:.1} \
             clean={clean_ns:.1} fraction={fraction:.3} ceiling={ceiling}"
        );
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
