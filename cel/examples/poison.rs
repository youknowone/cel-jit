//! Does a SHORT first batch leave the driver worse at a LONG batch than if the
//! long batch had been first?
//!
//! Everything else about the two arms is identical — same lowered program, same
//! green key, same driver shape, same row count, same element values. The only
//! difference is what ran BEFORE the measured long batch:
//!
//! * `cold-long`   — reset, then the long batch. Nothing else ever traced.
//! * `short-first` — reset, a short batch (inner trip count `SHORT`), then the
//!   SAME long batch.
//!
//! If the tier's compiled artifacts were shape-independent the two would land on
//! the same ns/row. A gap means the topology built for the short trip count is
//! still what the long batch runs, and the long batch is not able to replace it.
//!
//! The long batch is repeated three times inside each arm. That separates "the
//! first long batch pays to re-trace" (round 2 recovers) from "the artifact is
//! stuck" (rounds 2 and 3 stay slow) — the discriminator a single timing cannot
//! make. Per-round deopt/compile/abort counts are printed beside each timing so
//! a recovery attempt that happened but did not help is visible.
//!
//! Rounds are min-of-N per cell: interference can only make a round slower.
//!
//! Run: `cargo run --release --example poison --features jit`
//! Optional args: `<rows> <rounds>`

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cel::majit::bytecode::float_bank::{
    reset_persistent_state, COMPILES, GUARD_FAILS, TRACE_ABORTS,
};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// Inner trip count of the batch that runs FIRST in the poisoned arm.
const SHORT: i64 = 2;
/// Inner trip count of the measured batch.
const LONG: i64 = 64;
/// Rows in the short warm-up batch — enough to pass the tracing threshold and
/// compile, which is the whole point of it.
const SHORT_ROWS: usize = 4_000;

const THRESHOLD: u32 = 8;

struct ListColumns {
    lens: Vec<i64>,
    offsets: Vec<i64>,
    elems: Vec<i64>,
}

impl ListColumns {
    fn build(rows: usize, len: i64) -> Self {
        let lens: Vec<i64> = vec![len; rows];
        let mut offsets = Vec::with_capacity(rows);
        let mut total = 0i64;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        // Every element must SATISFY `i.price > 10`. `all` short-circuits on the
        // first element that fails, so a column with failing elements makes the
        // inner loop's real trip count "index of the first failure" — data, not
        // `len`. The whole probe is about trip counts, so the data is chosen to
        // make `len` the trip count exactly: every row runs to the end and
        // returns true.
        let elems: Vec<i64> = (0..total.max(1)).map(|k| 11 + (k * 7) % 40).collect();
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

/// One timed batch. Returns `(elapsed, compiles, deopts, aborts, result)`.
///
/// The counters are zeroed immediately before the run, so they describe THIS
/// batch and not the arm's history.
fn run_batch(
    lowered: &LoweredF,
    cols: &ListColumns,
    rows: usize,
) -> (Duration, usize, usize, usize, Option<i64>) {
    let columns = cols.columns();
    COMPILES.store(0, Ordering::Relaxed);
    GUARD_FAILS.store(0, Ordering::Relaxed);
    TRACE_ABORTS.store(0, Ordering::Relaxed);
    let t0 = Instant::now();
    let result = eval_batch_sum_f(lowered, &columns, rows, THRESHOLD);
    let elapsed = t0.elapsed();
    black_box(result);
    (
        elapsed,
        COMPILES.load(Ordering::Relaxed),
        GUARD_FAILS.load(Ordering::Relaxed),
        TRACE_ABORTS.load(Ordering::Relaxed),
        result,
    )
}

/// Per-round record for one arm: three long batches back to back.
struct ArmRound {
    long: [(Duration, usize, usize, usize); 3],
}

fn run_arm(
    lowered: &LoweredF,
    short: Option<&ListColumns>,
    long: &ListColumns,
    rows: usize,
    oracle: Option<i64>,
) -> ArmRound {
    reset_persistent_state();
    if let Some(s) = short {
        let (_, _, _, _, r) = run_batch(lowered, s, SHORT_ROWS);
        // The warm-up's own answer is checked too: a poisoned measurement is
        // worthless if the warm-up itself miscompiled.
        assert!(r.is_some(), "short warm-up batch declined");
    }
    let mut long_rounds = [(Duration::ZERO, 0usize, 0usize, 0usize); 3];
    for slot in long_rounds.iter_mut() {
        let (d, c, g, a, r) = run_batch(lowered, long, rows);
        assert_eq!(r, oracle, "long batch answer diverged from the clean VM");
        *slot = (d, c, g, a);
    }
    ArmRound { long: long_rounds }
}

fn ns_per_row(d: Duration, rows: usize) -> f64 {
    d.as_secs_f64() * 1e9 / rows as f64
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(20_000);
    let rounds: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(7);

    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let src = "items.all(i, i.price > 10)";
    let program = Program::compile(src).expect("parse");
    let lowered = lower_typed(program.expression(), &schema).expect("lower");

    // Single-pair mode: `<rows> <rounds> <warm> <measured>` runs exactly one
    // (warm-up trip, measured trip) arm and prints every batch. This is the mode
    // to run under `MAJIT_LOG=1` / `MAJIT_BRIDGE_DEBUG=1` — the matrix below
    // interleaves nine warm-up shapes and its log is unreadable.
    let pair = (args.next(), args.next());
    if let (Some(w), Some(m)) = pair {
        let warm: i64 = w.parse().expect("warm trip");
        let measured: i64 = m.parse().expect("measured trip");
        let warm_cols = (warm >= 0).then(|| ListColumns::build(SHORT_ROWS, warm));
        let mc = ListColumns::build(rows, measured);
        let oracle = clean_batch_sum_f(&lowered, &mc.columns(), rows);
        println!("single pair: warm={warm} measured={measured} rows={rows} oracle={oracle:?}");
        reset_persistent_state();
        if let Some(wc) = warm_cols.as_ref() {
            let (d, c, g, a, _) = run_batch(&lowered, wc, SHORT_ROWS);
            println!(
                "  warm-up      {:>10.1} ns/row compiles={c} deopts={g} aborts={a}",
                ns_per_row(d, SHORT_ROWS)
            );
        }
        for i in 0..3 {
            let (d, c, g, a, r) = run_batch(&lowered, &mc, rows);
            assert_eq!(r, oracle, "answer diverged");
            println!(
                "  measured #{}  {:>10.1} ns/row compiles={c} deopts={g} aborts={a}",
                i + 1,
                ns_per_row(d, rows)
            );
        }
        return;
    }

    let short_cols = ListColumns::build(SHORT_ROWS, SHORT);
    let long_warm_cols = ListColumns::build(SHORT_ROWS, LONG);
    let long_cols = ListColumns::build(rows, LONG);

    // Oracle: the plain-`match` VM over the same program and columns. Every
    // timed batch is asserted against it, so no number below is taken off a
    // miscompile.
    let oracle = {
        let columns = long_cols.columns();
        clean_batch_sum_f(&lowered, &columns, rows)
    };

    // Two reference tiers over the SAME program, columns and row count, so a
    // slow arm can be told apart from "this arm is simply not running compiled
    // code": `clean` is the plain-`match` VM with no tracing machinery at all,
    // and `interp` is the identical mainloop instrumented but never compiling
    // (`threshold = u32::MAX`). A poisoned arm that lands on `interp` never
    // entered a compiled loop; one that is slower than `interp` is running
    // compiled code that is worse than no JIT.
    let clean_ns = {
        let columns = long_cols.columns();
        let mut best = Duration::MAX;
        for _ in 0..rounds {
            let t0 = Instant::now();
            black_box(clean_batch_sum_f(&lowered, &columns, rows));
            best = best.min(t0.elapsed());
        }
        ns_per_row(best, rows)
    };
    let interp_ns = {
        let columns = long_cols.columns();
        let mut best = Duration::MAX;
        for _ in 0..rounds {
            reset_persistent_state();
            let t0 = Instant::now();
            black_box(eval_batch_sum_f(&lowered, &columns, rows, u32::MAX));
            best = best.min(t0.elapsed());
        }
        ns_per_row(best, rows)
    };

    println!("cel poison probe — `{src}`");
    println!("  rows={rows} rounds={rounds} short_trip={SHORT} long_trip={LONG} short_rows={SHORT_ROWS} threshold={THRESHOLD}");
    println!("  oracle={oracle:?}");
    println!(
        "  reference: clean VM {clean_ns:.1} ns/row, uncompiled mainloop {interp_ns:.1} ns/row"
    );
    println!();

    // Best (minimum) per (arm, long-round-index) across the outer rounds, with
    // the counters from whichever outer round produced that minimum.
    let mut best: [[Option<(Duration, usize, usize, usize)>; 3]; 3] = Default::default();
    for _ in 0..rounds {
        // Interleaved: a machine that gets busier over the run drifts every arm
        // together instead of penalising whichever ran last.
        //
        // `long-first` is the control for "a warm-up batch per se is the
        // problem": same driver reuse, same row count, same everything as
        // `short-first` — only the warm-up's inner trip count differs.
        for (arm, short) in [
            (0usize, None),
            (1usize, Some(&short_cols)),
            (2usize, Some(&long_warm_cols)),
        ] {
            let r = run_arm(&lowered, short, &long_cols, rows, oracle);
            for (i, cell) in r.long.iter().enumerate() {
                let slot = &mut best[arm][i];
                if slot.map_or(true, |b| cell.0 < b.0) {
                    *slot = Some(*cell);
                }
            }
        }
    }

    println!(
        "{:<14}{:>10}{:>12}{:>10}{:>10}{:>9}",
        "arm", "long#", "ns/row", "compiles", "deopts", "aborts"
    );
    let names = ["cold-long", "short-first", "long-first"];
    for arm in 0..3 {
        for i in 0..3 {
            let (d, c, g, a) = best[arm][i].expect("every cell measured");
            println!(
                "{:<14}{:>10}{:>12.1}{:>10}{:>10}{:>9}",
                if i == 0 { names[arm] } else { "" },
                i + 1,
                ns_per_row(d, rows),
                c,
                g,
                a
            );
        }
    }

    // Compare SETTLED against SETTLED. The cold arm's first long batch pays for
    // tracing and compiling, so a ratio taken against it would credit the
    // poisoned arm for work the cold arm did once and never repeats.
    let settled = |arm: usize| ns_per_row(best[arm][2].unwrap().0, rows);
    println!();
    println!("  settled ns/row (3rd long batch of each arm):");
    for (arm, name) in names.iter().enumerate() {
        println!(
            "    {name:<12} {:>8.1}   {:>6.2}x cold-long   {:>6.2}x clean VM",
            settled(arm),
            settled(arm) / settled(0),
            settled(arm) / clean_ns
        );
    }

    // Matrix over (warm-up trip count) x (measured trip count). `none` is the
    // cold control for a column; the diagonal (warm == measured) says whether an
    // artifact is bad in general or only away from the shape it was built for.
    //
    // A single (2 -> 64) point cannot distinguish "short poisons long" from a
    // narrow band, and the two are different defects. The `x` figure in each
    // cell is against that column's own cold control, so columns with different
    // absolute per-row work stay comparable.
    let warms: [i64; 9] = [-1, 0, 1, 2, 3, 4, 8, 16, 64];
    let measures: [i64; 5] = [2, 3, 4, 8, 64];
    println!();
    println!(
        "  (warm-up trip) x (measured trip), settled ns/row, `x` vs that column's cold control:"
    );
    print!("{:>10}", "warm\\meas");
    for m in measures {
        print!("{:>18}", m);
    }
    println!();

    let measured_cols: Vec<ListColumns> = measures
        .iter()
        .map(|&m| ListColumns::build(rows, m))
        .collect();
    let oracles: Vec<Option<i64>> = measured_cols
        .iter()
        .map(|c| clean_batch_sum_f(&lowered, &c.columns(), rows))
        .collect();

    let mut cold_of_col = [0f64; 5];
    for (row_idx, warm) in warms.iter().enumerate() {
        let warm_cols = (*warm >= 0).then(|| ListColumns::build(SHORT_ROWS, *warm));
        let label = if *warm < 0 {
            "none".to_string()
        } else {
            warm.to_string()
        };
        print!("{label:>10}");
        for (col, mc) in measured_cols.iter().enumerate() {
            let mut cell: Option<(Duration, usize, usize, usize)> = None;
            for _ in 0..rounds {
                let r = run_arm(&lowered, warm_cols.as_ref(), mc, rows, oracles[col]);
                let last = r.long[2];
                if cell.map_or(true, |b| last.0 < b.0) {
                    cell = Some(last);
                }
            }
            let (d, _, g, _) = cell.expect("measured");
            let ns = ns_per_row(d, rows);
            if row_idx == 0 {
                cold_of_col[col] = ns;
            }
            print!("{ns:>10.1}{:>5.1}x{g:>3}", ns / cold_of_col[col]);
        }
        println!();
    }
}
