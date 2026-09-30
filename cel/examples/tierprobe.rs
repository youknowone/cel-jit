//! Where the compiled tier starts to beat the plain interpreter, per shape, in
//! body words.
//!
//! [`cel::majit::batch::Tier::Auto`] has to decide that without measuring, and
//! deciding it by argument is how a routing rule becomes a rule about the cases
//! someone happened to look at. This probe measures it instead: it times the
//! SAME bound batch on `Tier::Clean` and on `Tier::Jit`, sweeping the batch
//! until the two cross, and reports the crossing in body words — the count
//! `BoundBatch::body_words` returns.
//!
//! Two shapes are swept, because the route has to serve both:
//!
//! * a straight-line program over a rising number of ROWS, where the body's
//!   words come from re-running a small body per row, and
//! * a comprehension over a rising number of ELEMENTS at one row, where they
//!   come from one row's inner loop.
//!
//! If the two shapes cross at similar word counts, one word threshold is the
//! right shape of rule. If they cross at very different ones, it is not — and
//! that is what this probe found, four crossings spread over 359..523 words,
//! which is why the route is now the two-term estimate on
//! [`cel::majit::batch::JIT_ENTRY_PS`] and `routeprobe` is what measures its
//! constants. This file stays as the independent check on that rule: the
//! `route` column is what the live rule decides at each point, beside the tier
//! that actually won it.
//!
//! RELEASE ONLY, under the same profile the scoreboard uses:
//!
//! ```text
//! cargo run --profile bench --package cel --features jit-cranelift \
//!     --example tierprobe
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_persistent_state};
use cel::majit::lower::{Schema, ValType};

/// One timed batch must last at least this long, so the clock's own resolution
/// is not what the measurement is against.
const MIN_BATCH: Duration = Duration::from_millis(20);
/// Timed batches per point; the fastest is reported, since other work on the
/// box can only ever make one slower.
const ROUNDS: usize = 7;

/// Time one call, growing an iteration count until a timed batch lasts at least
/// [`MIN_BATCH`], then reporting the fastest of [`ROUNDS`] such batches.
fn per_call<T>(mut run: impl FnMut() -> T) -> f64 {
    fn timed<T>(iters: usize, run: &mut impl FnMut() -> T) -> Duration {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(run());
        }
        start.elapsed()
    }

    let mut iters = 1usize;
    loop {
        let elapsed = timed(iters, &mut run);
        if elapsed >= MIN_BATCH {
            let mut best = elapsed;
            for _ in 1..ROUNDS {
                best = best.min(timed(iters, &mut run));
            }
            return best.as_secs_f64() * 1e9 / iters as f64;
        }
        iters = (iters * 2).max(1);
    }
}

/// Calls in the probe that stands behind the `jit ns` column.
const PROBE_CALLS: usize = 200;

/// Run the batch loop until the trace threshold is crossed and the compiled
/// artifact is entered, then report the share of a fixed probe's calls that
/// ENTERED compiled code.
///
/// The share is the whole point of running the probe. A `Tier::Jit` call that
/// did not enter compiled code is the tracing interpreter's, which is slower
/// than either tier this file is comparing — so an ungated sweep would place
/// the crossing wherever the warm-up happened to fall short, and the constant
/// derived from it would be about the harness rather than about the machine.
fn warm(bound: &BoundBatch<'_, '_>) -> f64 {
    for _ in 0..512 {
        black_box(bound.collect_on(Tier::Jit).expect("warm run"));
    }
    let before = jit_stats().compiled_entries;
    for _ in 0..PROBE_CALLS {
        black_box(bound.collect_on(Tier::Jit).expect("probe run"));
    }
    (jit_stats().compiled_entries - before) as f64 / PROBE_CALLS as f64
}

/// One sweep point: both tiers on one bound batch, plus the word count the
/// route would compare.
struct Point {
    n: usize,
    words: usize,
    clean: f64,
    jit: f64,
    /// Calls per call that entered compiled code, over the probe.
    entries: f64,
    /// What the live route picks for this batch — the rule under test, not a
    /// re-derivation of it.
    route: Tier,
}

impl Point {
    /// Whether `jit` is evidenced as the compiled tier's number rather than the
    /// tracing interpreter's.
    fn entered(&self) -> bool {
        self.entries >= 1.0
    }
}

fn measure(bound: &BoundBatch<'_, '_>, n: usize) -> Point {
    let entries = warm(bound);
    let clean = per_call(|| black_box(bound.collect_on(Tier::Clean).expect("clean run")));
    let jit = per_call(|| black_box(bound.collect_on(Tier::Jit).expect("jit run")));
    Point {
        n,
        words: bound.body_words(),
        clean,
        jit,
        entries,
        route: bound.route(Tier::Auto),
    }
}

/// The word count where `clean` and `jit` cross, by linear interpolation
/// between the last point the interpreter won and the first the compiled tier
/// did. `None` when the sweep never changed hands.
fn crossing(points: &[Point]) -> Option<f64> {
    // A point whose `jit` cell is not the compiled tier cannot locate the
    // crossing between the two tiers, in either direction.
    let points: Vec<&Point> = points.iter().filter(|p| p.entered()).collect();
    let k = points.iter().position(|p| p.jit < p.clean)?;
    let hi = points[k];
    let Some(lo) = k.checked_sub(1).map(|i| points[i]) else {
        // The compiled tier already won at the smallest point swept, so the
        // crossing is at or below it and this sweep cannot locate it.
        return Some(hi.words as f64);
    };
    let (d_lo, d_hi) = (lo.clean - lo.jit, hi.clean - hi.jit);
    let t = -d_lo / (d_hi - d_lo);
    Some(lo.words as f64 + t * (hi.words - lo.words) as f64)
}

fn report(title: &str, sweep: &str, points: &[Point]) {
    println!("\n{title}");
    println!("  sweeping {sweep}");
    println!(
        "  {:>8} {:>10} {:>12} {:>12} {:>11} {:>11} {:>7} {:>4}",
        "n", "words", "clean ns", "jit ns", "enter/call", "winner", "route", "ok"
    );
    for p in points {
        let winner = match (p.entered(), p.jit < p.clean) {
            (false, _) => "not entered",
            (true, true) => "jit",
            (true, false) => "clean",
        };
        let route = match p.route {
            Tier::Jit => "jit",
            Tier::Clean => "clean",
            other => panic!("the route resolved Auto to {other:?}"),
        };
        println!(
            "  {:>8} {:>10} {:>12.1} {:>12.1} {:>11.2} {:>11} {:>7} {:>4}",
            p.n,
            p.words,
            p.clean,
            p.jit,
            p.entries,
            winner,
            route,
            if !p.entered() {
                "-"
            } else if winner == route {
                "y"
            } else {
                "NO"
            }
        );
    }
    match crossing(points) {
        Some(w) => println!("  crossing at ~{w:.0} body words"),
        None => println!("  no crossing within the sweep: the interpreter won every point"),
    }
}

/// A straight-line program over `rows` rows of one int column.
fn straight(source: &str, rows: &[usize]) -> Vec<Point> {
    let mut schema = Schema::new();
    schema.insert("x".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    rows.iter()
        .map(|&n| {
            let col: Vec<i64> = (0..n as i64).collect();
            let batch = Batch::new(n).column("x", ColumnRef::Int(&col));
            // A fresh driver per point, the protocol both gated benches bind
            // under. Without it every point after the first is timed against a
            // pool this sweep itself grew, and the compiled tier's per-call
            // cost — the only cost the crossing is about — rises with it.
            reset_persistent_state();
            let bound = program.bind_per_row(&batch).expect("binds");
            measure(&bound, n)
        })
        .collect()
}

/// A comprehension over one row whose list carries `elems` elements.
fn comprehension(source: &str, elems: &[usize]) -> Vec<Point> {
    let mut schema = Schema::new();
    schema.insert("list[]".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    elems
        .iter()
        .map(|&n| {
            let lens = vec![n as i64];
            let flat: Vec<i64> = (0..n as i64).collect();
            let batch = Batch::new(1).column(
                "list",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Int(&flat))],
                },
            );
            reset_persistent_state();
            let bound = program.bind_per_row(&batch).expect("binds");
            measure(&bound, n)
        })
        .collect()
}

fn main() {
    println!("where Tier::Auto should hand over, in body words");
    println!(
        "`route` is what `Tier::Auto` picks today; `ok` is whether that is the tier that won."
    );
    println!(
        "best of {ROUNDS} batches of >= {} ms per point, both tiers on ONE bound batch.",
        MIN_BATCH.as_millis()
    );

    let rows = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    report(
        "straight-line, small body: `x * 2 + 1`",
        "batch HEIGHT",
        &straight("x * 2 + 1", &rows),
    );
    report(
        "straight-line, larger body: `x > 10 ? x * 2 + 1 : x * 3 - 1`",
        "batch HEIGHT",
        &straight("x > 10 ? x * 2 + 1 : x * 3 - 1", &rows),
    );

    let elems = [1usize, 2, 4, 8, 16, 32, 64, 128, 256];
    report(
        "comprehension at one row: `list.map(x, x * 2)`",
        "ELEMENT count",
        &comprehension("list.map(x, x * 2)", &elems),
    );
    report(
        "comprehension at one row: `list.filter(x, x % 2 == 0)`",
        "ELEMENT count",
        &comprehension("list.filter(x, x % 2 == 0)", &elems),
    );
}
