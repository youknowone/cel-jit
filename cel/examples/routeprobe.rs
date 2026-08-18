//! The constants [`cel::majit::batch::Tier::Auto`] routes by, measured as a
//! family of crossings rather than read off one.
//!
//! `tierprobe` asks "where do the two tiers cross for THIS shape", in body
//! words, and it found that the answer moves. A single word threshold has to be
//! some average of those, and the band between the outermost two is a band of
//! runs that get the tier they do not want.
//!
//! The crossing moves because a word is not the only thing either tier charges
//! for. Both charge per ITERATION as well — per row of the batch, per element
//! of a comprehension's list — and the interpreter's per-iteration cost is the
//! larger of the two, so each iteration hands the compiled tier a saving that
//! has nothing to do with how many words the iteration is. A body of few words
//! per iteration therefore breaks even at FEWER total words than a dense one:
//! measured here, 535 words for a 25-word row body against 1049 for a 59-word
//! one. That is the term a threshold in words has nowhere to put.
//!
//! # What is fitted
//!
//! ```text
//!   saving = GAIN_PER_WORD * body_words + GAIN_PER_ITERATION * (rows + elems)
//! ```
//!
//! against a fixed `JIT_ENTRY_PS`. At a shape's own crossing count `n*` the two
//! are equal, so for a straight-line shape swept by height
//!
//! ```text
//!   1 / n*  =  (GAIN_PER_WORD / ENTRY) * row_words  +  GAIN_PER_ITERATION / ENTRY
//! ```
//!
//! — a straight line through the shapes whose slope and intercept are the two
//! rates divided by the entry. An element sweep gives the same line in
//! `elem_words`, offset by the one row it always carries.
//!
//! `n*` is read by interpolating between the two swept points that BRACKET the
//! crossing, so every number entering the regression was measured next to the
//! decision it is about. Fitting whole lines and extrapolating their intercepts
//! back to zero was tried first and rejected: over these shapes the entry it
//! produced ranged across zero, because a line fitted through 256 rows says very
//! little about the cost of arriving.
//!
//! Only the RATIOS are pinned by that regression, which is all the decision
//! needs — multiply all three constants by any positive number and every route
//! is unchanged. The absolute scale comes from the entry, measured on its own as
//! `jit_fix - clean_fix` with both intercepts taken from the smallest points
//! swept, where the fixed cost is most of what there is to see.
//!
//! Row words and element words are fitted SEPARATELY as well as pooled, and the
//! separate report is the evidence for pooling them: across runs the two
//! estimates of each rate overlap, so two pairs of constants would be fitting
//! noise. Keep an eye on that pair — if they ever separate, the pooled rule is
//! the thing to revisit.
//!
//! # Grading
//!
//! The last section replays every swept point against four rules and counts the
//! points where each names a tier that is not the one that won:
//!
//! * the body-word threshold this replaced,
//! * the BEST word threshold there is, chosen by searching every candidate
//!   against these same points — the control that says whether the second term
//!   earns its place or whether a better-set scalar would have done,
//! * the constants just fitted, and
//! * the LIVE rule in `batch.rs`, as `BoundBatch::route` answered it at bind,
//!   which is what makes this file a check on the shipped constants and not only
//!   the thing that produced them.
//!
//! RELEASE ONLY:
//!
//! ```text
//! cargo run --release --package cel --features jit-dynasm --example routeprobe
//! ```
//!
//! Both backends are worth running: the constants describe compiled code, and
//! `jit-cranelift` compiles it differently.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cel::majit::batch::{
    Batch, BatchProgram, BoundBatch, ColumnRef, Tier, GAIN_PER_ITERATION_PS, GAIN_PER_WORD_PS,
    JIT_ENTRY_PS,
};
use cel::majit::bytecode::float_bank::{reset_persistent_state, COMPILED_ENTRIES};
use cel::majit::lower::{Schema, ValType};
use cel::Value;

/// One timed batch must last at least this long, so the clock's own resolution
/// is not what the measurement is against.
const MIN_BATCH: Duration = Duration::from_millis(20);
/// Timed batches per point; the fastest is reported, since other work on the
/// box can only ever make one slower.
const ROUNDS: usize = 9;
/// Calls in the probe that decides whether a point's `jit` cell is the compiled
/// tier's number or the tracing interpreter's.
const PROBE_CALLS: usize = 200;
/// Points at or below this count carry the fixed cost as most of their time, so
/// they are the ones the intercepts are read from.
const SMALL: usize = 4;
/// Points at or above this count are all slope, so they are the ones the
/// per-unit rates are read from.
const LARGE: usize = 32;
/// The body-word threshold this rule replaced, kept here and nowhere else,
/// because the grading needs the thing being compared to.
const LEGACY_AUTO_JIT_WORDS: usize = 460;

fn timed(iters: usize, run: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        run();
    }
    start.elapsed()
}

/// Time TWO calls against each other, INTERLEAVED: one iteration count is
/// calibrated so a batch of either lasts at least [`MIN_BATCH`], then the two
/// take turns for [`ROUNDS`] batches each and the fastest of each is reported.
///
/// Interleaved rather than one arm and then the other, because the difference
/// between the two is the whole measurement and a busy box does not hold still
/// for the length of a sweep. Timing all of `clean` and then all of `jit` puts
/// a load excursion entirely inside one arm; alternating puts it in both. On a
/// box at load 150 this was worth 30-40% of the crossing, which was more than
/// the difference between any two shapes.
fn per_call_pair(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let mut iters = 1usize;
    while timed(iters, &mut a).max(timed(iters, &mut b)) < MIN_BATCH {
        iters = (iters * 2).max(1);
    }
    let (mut best_a, mut best_b) = (Duration::MAX, Duration::MAX);
    for _ in 0..ROUNDS {
        best_a = best_a.min(timed(iters, &mut a));
        best_b = best_b.min(timed(iters, &mut b));
    }
    let ns = |d: Duration| d.as_secs_f64() * 1e9 / iters as f64;
    (ns(best_a), ns(best_b))
}

/// Run the batch loop until the trace threshold is crossed, then report the
/// share of a fixed probe's calls that ENTERED compiled code.
///
/// A `Tier::Jit` call that did not enter compiled code is the tracing
/// interpreter's, which is slower than either tier this file compares, so a
/// point fitted through it would describe the harness rather than the machine.
/// Every point states its share and everything downstream drops the ones below
/// one.
fn warm(bound: &BoundBatch<'_, '_>, out: &mut Vec<Value>) -> f64 {
    for _ in 0..512 {
        bound.collect_into_on(Tier::Jit, out).expect("warm run");
        black_box(out.as_slice());
    }
    let before = COMPILED_ENTRIES.load(Ordering::Relaxed);
    for _ in 0..PROBE_CALLS {
        bound.collect_into_on(Tier::Jit, out).expect("probe run");
        black_box(out.as_slice());
    }
    (COMPILED_ENTRIES.load(Ordering::Relaxed) - before) as f64 / PROBE_CALLS as f64
}

/// One sweep point of one shape: both tiers on ONE bound batch.
struct Point {
    /// Rows for a height sweep, elements for an element sweep.
    n: usize,
    words: usize,
    clean: f64,
    jit: f64,
    entries: f64,
    /// What the LIVE route picked for this batch — the rule under test, asked
    /// of the batch itself rather than re-derived here.
    route: Tier,
}

impl Point {
    fn entered(&self) -> bool {
        self.entries >= 1.0
    }
    fn jit_won(&self) -> bool {
        self.jit < self.clean
    }
}

/// Time both tiers on one bound batch through the door the scoreboard measures:
/// `collect_into_on` with a buffer the caller keeps.
///
/// Building each row's `Value` is the same code on both tiers, so it lands in
/// both columns equally and cancels out of every difference this file takes.
/// The door is chosen for what it does to the NOISE — one allocation per call
/// is the largest thing a `collect_on` sweep would add that the machine under
/// measurement is not.
fn measure(bound: &BoundBatch<'_, '_>, n: usize) -> Point {
    let mut out: Vec<Value> = Vec::new();
    let entries = warm(bound, &mut out);
    // Two buffers, not one: the arms alternate, and a shared buffer would hand
    // each arm the other's allocation state.
    let mut out_jit: Vec<Value> = out.clone();
    let (clean, jit) = {
        let run = |tier, buf: &mut Vec<Value>| {
            bound.collect_into_on(tier, buf).expect("run");
            black_box(buf.as_slice());
        };
        per_call_pair(
            || run(Tier::Clean, &mut out),
            || run(Tier::Jit, &mut out_jit),
        )
    };
    Point {
        n,
        words: bound.body_words(),
        clean,
        jit,
        entries,
        route: bound.route(Tier::Auto),
    }
}

/// Ordinary least squares of `y` against `x`, as `(intercept, slope)`.
fn line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
    (my - slope * mx, slope)
}

/// How much of the spread in `ys` a fitted line accounts for. Printed rather
/// than gated on: it is what says whether "the saving is proportional to the
/// words" is a description of these shapes or a hope about them.
fn r_squared(xs: &[f64], ys: &[f64], fit: (f64, f64)) -> f64 {
    let my = ys.iter().sum::<f64>() / ys.len() as f64;
    let ss_tot: f64 = ys.iter().map(|y| (y - my) * (y - my)).sum();
    let ss_res: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| {
            let d = y - (fit.0 + fit.1 * x);
            d * d
        })
        .sum();
    if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Which axis a shape is swept along, and therefore which of the two per-unit
/// rates its crossing constrains.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    /// Batch HEIGHT, at no list column: the crossing is a row count.
    Rows,
    /// ELEMENT count at ONE row: the crossing is an element count.
    Elems,
}

/// One shape, swept, with everything the two regressions take from it.
struct Shape {
    label: &'static str,
    axis: Axis,
    row_words: usize,
    elem_words: usize,
    /// The count where the two tiers change hands, by interpolation between the
    /// bracketing swept points. `None` when the sweep never changed hands.
    crossing: Option<f64>,
    /// Intercepts from the points at or below [`SMALL`], slopes from the points
    /// at or above [`LARGE`].
    clean_fix: f64,
    jit_fix: f64,
    clean_slope: f64,
    jit_slope: f64,
    points: Vec<Point>,
}

impl Shape {
    /// The word count whose rate this shape's crossing constrains.
    fn unit_words(&self) -> usize {
        match self.axis {
            Axis::Rows => self.row_words,
            Axis::Elems => self.elem_words,
        }
    }
    /// What a call pays to reach compiled code, from this shape alone.
    fn entry(&self) -> f64 {
        self.jit_fix - self.clean_fix
    }
    /// The crossing restated in body words, for comparison with `tierprobe`'s
    /// number and with the threshold this replaced.
    fn crossing_words(&self) -> Option<f64> {
        let n = self.crossing?;
        Some(match self.axis {
            Axis::Rows => n * self.row_words as f64,
            // An element sweep always runs at exactly one row, whose words are
            // there whatever the element count.
            Axis::Elems => n * self.elem_words as f64 + self.row_words as f64,
        })
    }
}

/// Where `clean` and `jit` change hands, by linear interpolation between the
/// last point the interpreter won and the first the compiled tier did.
///
/// Local by construction: the two points bracket the crossing, so nothing far
/// from the decision is asked about it. Points whose `jit` cell was not
/// evidenced as compiled code are dropped first, in both directions.
fn crossing(points: &[Point]) -> Option<f64> {
    let ps: Vec<&Point> = points.iter().filter(|p| p.entered()).collect();
    let k = ps.iter().position(|p| p.jit_won())?;
    let hi = ps[k];
    // The compiled tier already won at the smallest point swept: the crossing
    // is at or below it, and this sweep cannot say where.
    let lo = k.checked_sub(1).map(|i| ps[i])?;
    let (d_lo, d_hi) = (lo.clean - lo.jit, hi.clean - hi.jit);
    let t = -d_lo / (d_hi - d_lo);
    Some(lo.n as f64 + t * (hi.n - lo.n) as f64)
}

fn shape(
    label: &'static str,
    axis: Axis,
    row_words: usize,
    elem_words: usize,
    points: Vec<Point>,
) -> Shape {
    let region = |keep: &dyn Fn(usize) -> bool| {
        let used: Vec<&Point> = points.iter().filter(|p| p.entered() && keep(p.n)).collect();
        let xs: Vec<f64> = used.iter().map(|p| p.n as f64).collect();
        let cs: Vec<f64> = used.iter().map(|p| p.clean).collect();
        let js: Vec<f64> = used.iter().map(|p| p.jit).collect();
        (line(&xs, &cs), line(&xs, &js))
    };
    // Two regions, because one line over the whole sweep is asked for two
    // different things and is good at only one of them: fitted through 256 rows
    // it reports the per-row rate well and an intercept extrapolated back
    // across two orders of magnitude.
    let (small_clean, small_jit) = region(&|n| n <= SMALL);
    let (large_clean, large_jit) = region(&|n| n >= LARGE);
    Shape {
        label,
        axis,
        row_words,
        elem_words,
        crossing: crossing(&points),
        clean_fix: small_clean.0,
        jit_fix: small_jit.0,
        clean_slope: large_clean.1,
        jit_slope: large_jit.1,
        points,
    }
}

/// A straight-line program swept by batch HEIGHT. Two int columns are always
/// declared and always bound, so a shape may use one or both without changing
/// the harness.
fn sweep_rows(label: &'static str, source: &'static str, ns: &[usize]) -> Shape {
    let mut schema = Schema::new();
    schema.insert("x".to_string(), ValType::Int);
    schema.insert("y".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    let lowered = program.lowered();
    let (row_words, elem_words) = (lowered.row_words, lowered.elem_words);
    let points = ns
        .iter()
        .map(|&n| {
            let x: Vec<i64> = (0..n as i64).collect();
            let y: Vec<i64> = (0..n as i64).map(|v| v + 3).collect();
            let batch = Batch::new(n)
                .column("x", ColumnRef::Int(&x))
                .column("y", ColumnRef::Int(&y));
            // A fresh driver per point, the protocol both gated benches bind
            // under. Without it every point after the first is timed against a
            // pool this sweep itself grew, and the compiled tier's per-call
            // cost — the thing the entry names — rises with it.
            reset_persistent_state();
            let bound = program.bind_per_row(&batch).expect("binds");
            measure(&bound, n)
        })
        .collect();
    shape(label, Axis::Rows, row_words, elem_words, points)
}

/// A comprehension at ONE row whose list carries a rising number of elements.
fn sweep_elems(label: &'static str, source: &'static str, ns: &[usize]) -> Shape {
    let mut schema = Schema::new();
    schema.insert("list[]".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    let lowered = program.lowered();
    let (row_words, elem_words) = (lowered.row_words, lowered.elem_words);
    let points = ns
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
        .collect();
    shape(label, Axis::Elems, row_words, elem_words, points)
}

fn report(shapes: &[Shape]) {
    println!(
        "\n  {:<50} {:>5} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "shape", "words", "clean fix", "jit fix", "clean/u", "jit/u", "cross n", "cross W"
    );
    for s in shapes {
        println!(
            "  {:<50} {:>5} {:>9.1} {:>9.1} {:>9.3} {:>9.3} {:>8} {:>8}",
            s.label,
            s.unit_words(),
            s.clean_fix,
            s.jit_fix,
            s.clean_slope,
            s.jit_slope,
            match s.crossing {
                Some(n) => format!("{n:.1}"),
                None => "-".to_string(),
            },
            match s.crossing_words() {
                Some(w) => format!("{w:.0}"),
                None => "-".to_string(),
            },
        );
    }
}

/// The two-term rule's own arithmetic, in the picosecond unit `batch.rs` states
/// it in, so what is printed is what would be pasted there.
#[derive(Clone, Copy)]
struct Rule {
    gain_per_word_ps: f64,
    gain_per_iteration_ps: f64,
    entry_ps: f64,
}

impl Rule {
    /// Build from the normalized `(slope, intercept)` of a crossing regression
    /// and an entry in ns. The regression only ever pins the two rates as
    /// SHARES of the entry; the entry is what turns them into times.
    fn from_shares(slope: f64, intercept: f64, entry_ns: f64) -> Rule {
        Rule {
            gain_per_word_ps: slope * entry_ns * 1000.0,
            gain_per_iteration_ps: intercept * entry_ns * 1000.0,
            entry_ps: entry_ns * 1000.0,
        }
    }
    fn saving_ps(&self, words: f64, iterations: f64) -> f64 {
        words * self.gain_per_word_ps + iterations * self.gain_per_iteration_ps
    }
    fn picks_jit(&self, words: f64, iterations: f64) -> bool {
        self.saving_ps(words, iterations) >= self.entry_ps
    }
    /// The count of a shape's swept axis at which this rule changes its mind —
    /// what it PREDICTS the crossing to be, for comparison with the measured
    /// one.
    fn predicted_crossing(&self, s: &Shape) -> Option<f64> {
        let (fixed, per_unit) = match s.axis {
            Axis::Rows => (0.0, self.saving_ps(s.row_words as f64, 1.0)),
            // An element sweep always runs at exactly one row, whose words and
            // whose iteration are there whatever the element count.
            Axis::Elems => (
                self.saving_ps(s.row_words as f64, 1.0),
                self.saving_ps(s.elem_words as f64, 1.0),
            ),
        };
        (per_unit > 0.0).then(|| (self.entry_ps - fixed) / per_unit)
    }
}

/// The body words and the iteration count a swept point stands for — the two
/// numbers a bind hands the rule.
fn extent(s: &Shape, p: &Point) -> (f64, f64) {
    let iterations = match s.axis {
        Axis::Rows => p.n as f64,
        Axis::Elems => 1.0 + p.n as f64,
    };
    (p.words as f64, iterations)
}

/// Regress `1 / crossing` against the shape's per-unit word count, over the
/// shapes on `axes`. The line's slope and intercept are the per-word and
/// per-iteration rates, each as a SHARE of one entry — which is all the
/// decision depends on, and the only part of it a crossing can measure.
///
/// `so_far` is the rule the offsets are taken against: an element sweep runs at
/// exactly one row, and that row has already spent part of the entry before the
/// first element is reached. Passing the previous iterate and running this twice
/// resolves it; the row-only fit needs no offset and converges immediately.
fn rates(shapes: &[Shape], axes: &[Axis], so_far: Rule) -> (f64, f64, usize, f64) {
    let group: Vec<&Shape> = shapes
        .iter()
        .filter(|s| axes.contains(&s.axis) && s.crossing.is_some())
        .collect();
    let xs: Vec<f64> = group.iter().map(|s| s.unit_words() as f64).collect();
    let ys: Vec<f64> = group
        .iter()
        .map(|s| {
            let spent = match s.axis {
                Axis::Rows => 0.0,
                Axis::Elems => so_far.saving_ps(s.row_words as f64, 1.0) / so_far.entry_ps,
            };
            (1.0 - spent) / s.crossing.expect("filtered")
        })
        .collect();
    let fit = line(&xs, &ys);
    (fit.1, fit.0, group.len(), r_squared(&xs, &ys, fit))
}

/// The word threshold that gets the FEWEST of these points wrong — the control
/// the two-term rule has to beat to have earned its second term.
///
/// Searched over every distinct body-word count the sweep produced, so it is the
/// best such rule there is against this evidence, not a plausible one.
fn best_threshold(shapes: &[Shape]) -> (usize, usize) {
    let mut candidates: Vec<usize> = shapes
        .iter()
        .flat_map(|s| s.points.iter().filter(|p| p.entered()).map(|p| p.words))
        .collect();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
        .iter()
        .map(|&w| (wrong(shapes, |p_words, _, _| p_words >= w as f64), w))
        .min()
        .expect("the sweep produced points")
}

/// How many entered points a decision function names the losing tier on.
fn wrong(shapes: &[Shape], picks_jit: impl Fn(f64, f64, &Shape) -> bool) -> usize {
    let mut bad = 0;
    for s in shapes {
        for p in s.points.iter().filter(|p| p.entered()) {
            let (words, iterations) = extent(s, p);
            bad += usize::from(picks_jit(words, iterations, s) != p.jit_won());
        }
    }
    bad
}

fn main() {
    println!("routing constants, from a family of measured crossings");
    println!(
        "best of {ROUNDS} batches of >= {} ms per point; both tiers on ONE bound batch;\n\
         a fresh driver pool per point; intercepts from n <= {SMALL}, slopes from n >= {LARGE}.",
        MIN_BATCH.as_millis()
    );
    println!(
        "batch.rs today: {GAIN_PER_WORD_PS} ps/word + {GAIN_PER_ITERATION_PS} ps/iteration \
         against an entry of {JIT_ENTRY_PS} ps"
    );

    // Dense low, sparse high: the crossings sit at a few tens of units and the
    // interpolation wants its bracket tight, while the slopes want a long arm.
    let ns = [1usize, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 128, 256];
    let mut shapes = vec![
        sweep_rows("rows: x + 1", "x + 1", &ns),
        sweep_rows("rows: x * 2 + 1", "x * 2 + 1", &ns),
        sweep_rows("rows: x + y", "x + y", &ns),
        sweep_rows(
            "rows: x > 3 && y < 90 || x == y",
            "x > 3 && y < 90 || x == y",
            &ns,
        ),
        sweep_rows(
            "rows: x > 10 ? x * 2 + 1 : x * 3 - 1",
            "x > 10 ? x * 2 + 1 : x * 3 - 1",
            &ns,
        ),
        sweep_rows(
            "rows: ((x + 1) * (y - 2)) / ((x + 3) - (y * 4))",
            "((x + 1) * (y - 2)) / ((x + 3) - (y * 4))",
            &ns,
        ),
        sweep_rows(
            "rows: (x + y) * (x - y) + (x + 1) * (y + 1) + x * y",
            "(x + y) * (x - y) + (x + 1) * (y + 1) + x * y",
            &ns,
        ),
        sweep_rows(
            "rows: x*2 + x*3 + y*4 + y*5 + x*6 + y*7",
            "x*2 + x*3 + y*4 + y*5 + x*6 + y*7",
            &ns,
        ),
        sweep_elems("elems: list.all(e, e > 0)", "list.all(e, e > 0)", &ns),
        sweep_elems(
            "elems: list.exists(e, e == 3)",
            "list.exists(e, e == 3)",
            &ns,
        ),
        sweep_elems("elems: list.map(e, e + 1)", "list.map(e, e + 1)", &ns),
        sweep_elems("elems: list.map(e, e * 2)", "list.map(e, e * 2)", &ns),
        sweep_elems(
            "elems: list.map(e, e * 2 + 1)",
            "list.map(e, e * 2 + 1)",
            &ns,
        ),
        sweep_elems(
            "elems: list.filter(e, e % 2 == 0)",
            "list.filter(e, e % 2 == 0)",
            &ns,
        ),
        sweep_elems(
            "elems: list.filter(e, e > 5 && e < 100)",
            "list.filter(e, e > 5 && e < 100)",
            &ns,
        ),
        sweep_elems(
            "elems: list.map(e, (e + 1) * (e + 2) + e * 3)",
            "list.map(e, (e + 1) * (e + 2) + e * 3)",
            &ns,
        ),
    ];
    shapes.sort_by_key(|s| (s.axis == Axis::Elems, s.unit_words()));

    report(&shapes);

    let entries: Vec<f64> = shapes.iter().map(Shape::entry).collect();
    let entry = median(entries.clone());
    let (lo, hi) = (
        entries.iter().cloned().fold(f64::MAX, f64::min),
        entries.iter().cloned().fold(f64::MIN, f64::max),
    );

    // Rows first and with no offset, then pooled twice: an element sweep runs
    // at one row, so its crossing already spends part of the entry on that row
    // and cannot be placed until the rates are approximately known.
    let seed = Rule::from_shares(0.0, 0.0, entry);
    let (a_row, b_row, n_row, r2_row) = rates(&shapes, &[Axis::Rows], seed);
    let mut rule = Rule::from_shares(a_row, b_row, entry);
    let mut pooled = (0.0, 0.0, 0, 0.0);
    for _ in 0..2 {
        pooled = rates(&shapes, &[Axis::Rows, Axis::Elems], rule);
        rule = Rule::from_shares(pooled.0, pooled.1, entry);
    }
    let (a_elem, b_elem, n_elem, r2_elem) = rates(&shapes, &[Axis::Elems], rule);
    let (a_all, b_all, n_all, r2_all) = pooled;

    println!("\nrates, regressed over the crossings (as a share of one entry)");
    for (what, a, b, n, r2) in [
        ("row words only ", a_row, b_row, n_row, r2_row),
        ("elem words only", a_elem, b_elem, n_elem, r2_elem),
        ("POOLED         ", a_all, b_all, n_all, r2_all),
    ] {
        println!("  {what}: 1/n* = {a:.6} * words + {b:.6}   ({n} shapes, R2 {r2:.3})");
    }
    println!(
        "  the first two are the evidence for pooling: two pairs of constants are worth\n  \
         shipping only where those two lines are apart by more than they move between runs."
    );
    println!(
        "  entry:    median {entry:.1} ns over {} shapes, spread {lo:.1} .. {hi:.1}",
        entries.len()
    );

    println!("\n  as picoseconds, for batch.rs:");
    for (name, ps) in [
        ("GAIN_PER_WORD_PS", rule.gain_per_word_ps),
        ("GAIN_PER_ITERATION_PS", rule.gain_per_iteration_ps),
        ("JIT_ENTRY_PS", rule.entry_ps),
    ] {
        println!("    {name:<22} = {ps:.0}");
    }

    println!(
        "\n  crossing in the swept unit: {:>10} {:>10}",
        "measured", "fitted"
    );
    for s in &shapes {
        println!(
            "    {:<50} {:>10} {:>10}",
            s.label,
            match s.crossing {
                Some(n) => format!("{n:.1}"),
                None => "-".to_string(),
            },
            match rule.predicted_crossing(s) {
                Some(n) => format!("{n:.1}"),
                None => "never".to_string(),
            },
        );
    }

    println!("\ngrading four rules against the tier that actually won each point");
    let entered = shapes
        .iter()
        .flat_map(|s| s.points.iter().filter(|p| p.entered()))
        .count();
    let old_bad = wrong(&shapes, |w, _, _| w >= LEGACY_AUTO_JIT_WORDS as f64);
    let (best_bad, best_w) = best_threshold(&shapes);
    let new_bad = wrong(&shapes, |w, i, _| rule.picks_jit(w, i));
    let mut live_bad = 0;
    let mut disagreements = Vec::new();
    for s in &shapes {
        for p in s.points.iter().filter(|p| p.entered()) {
            let old = p.words >= LEGACY_AUTO_JIT_WORDS;
            let live = p.route == Tier::Jit;
            live_bad += usize::from(live != p.jit_won());
            if old != live {
                let name = |b| if b { "jit" } else { "clean" };
                disagreements.push(format!(
                    "  {:<50} n={:<5} W={:<6} won {:<5} words {:<5} live {:<5} {}",
                    s.label,
                    p.n,
                    p.words,
                    name(p.jit_won()),
                    name(old),
                    name(live),
                    match (old == p.jit_won(), live == p.jit_won()) {
                        (false, true) => "FIXED",
                        (true, false) => "BROKEN",
                        _ => "both wrong",
                    }
                ));
            }
        }
    }
    println!("  points that entered compiled code:    {entered}");
    println!("  words threshold {LEGACY_AUTO_JIT_WORDS} (what this replaced): {old_bad} wrong");
    println!("  BEST word threshold there is ({best_w} words): {best_bad} wrong");
    println!("  two-term, constants fitted above:     {new_bad} wrong");
    println!("  two-term, constants LIVE in batch.rs: {live_bad} wrong");
    if !disagreements.is_empty() {
        println!("\n  where the words threshold and the live rule disagree:");
        for l in &disagreements {
            println!("{l}");
        }
    }
}
