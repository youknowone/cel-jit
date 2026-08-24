//! Wall-clock A/B for the NESTED comprehension shape — `items.all(i, ...)` over
//! a runtime-length list column, the workload the bridge/virtualizable fixes
//! were about.
//!
//! `cel/tests/majit_trace_evidence.rs` pins the deopt census for this shape, but
//! a deopt count is not a speedup: the census can be satisfied while the
//! compiled tier is still slower than the plain interpreter. This example
//! measures the thing the census cannot.
//!
//! Three tiers over the SAME lowered program and the SAME columns:
//!
//! * `clean`  — `clean_batch_sum_f`, the plain-`match` Rust VM with no tracing
//!   machinery in the loop. This is the honest "no JIT at all" baseline.
//! * `interp` — the `#[jit_interp]` mainloop with `threshold = u32::MAX`, i.e.
//!   the same loop carrying the meta-tracer's instrumentation but never
//!   compiling. The gap `clean → interp` is what the tier costs when it never
//!   pays off.
//! * `jit`    — the same mainloop at `threshold = 8`. The driver outlives a run
//!   and the program words outlive it with the owner that built them, so only
//!   the first run to reach the merge point pays for tracing and compiling; the
//!   `cmp` column says which one did.
//!
//! A single batch size still cannot separate "the compiled code is slow" from
//! "the batch was too short to pay for compiling", so each shape is swept over a
//! geometric ladder of row counts and the totals are fitted. Read the fitted
//! intercept as a compile cost only for a point whose `cmp` is non-zero; where
//! `cmp` is 0 the loop was already compiled and the intercept is measuring
//! per-call setup alone:
//!
//! ```text
//! jit_total(n) ≈ compile_cost + steady_ns_per_row * n
//! ```
//!
//! by ordinary least squares over the whole ladder. `steady` is what the
//! compiled trace actually costs per row; `compile` is the fixed price of
//! getting there; `break-even` is where the JIT total overtakes the clean VM
//! total. A fourth tier, cel's tree walker, is reported per shape as a floor
//! for the columnar pipeline itself. It is reached as `Value::resolve_value`,
//! not through `Program::execute`: that door is the bytecode VM whenever the
//! `vm` feature is on, and `vm` is a DEFAULT feature that
//! `required-features = ["jit"]` does not turn off, so the public door would
//! make the floor a VM measurement under a tree-walker label.
//!
//! Two things keep this readable on a shared machine: rounds are interleaved
//! (clean, interp, jit, clean, interp, jit, …) rather than run in blocks, so a
//! box that gets busier over the run drifts all tiers together instead of
//! penalising whichever ran last; and each tier reports the MINIMUM of its
//! rounds, since interference can only make a round slower. RELEASE ONLY.
//!
//! Run: `cargo run --release --example majit_nested_bench --features jit-cranelift`
//! (or `jit-dynasm`; `jit` alone names no backend and does not link)
//! Optional args: `<max_rows> <rounds>`.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

/// Reference tier: how many distinct activations to build, and how many
/// `Value::resolve_value` calls to time over them.
const TREE_POOL: usize = 512;
const TREE_EVALS: usize = 20_000;

/// The list column of `items.all(i, i.price > 10)`: a per-row element count and
/// the flattened element buffer. The `offset(items)` column the lowering also
/// reads is derived at bind time from the counts.
struct ListColumns {
    lens: Vec<i64>,
    elems: Vec<i64>,
}

impl ListColumns {
    fn build(rows: usize, len_of: fn(usize) -> i64) -> Self {
        let lens: Vec<i64> = (0..rows).map(len_of).collect();
        let total: i64 = lens.iter().sum();
        // `.max(1)` keeps the buffer non-empty at length 0, where nothing reads
        // it but a column still needs a base address.
        let elems: Vec<i64> = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
        Self { lens, elems }
    }

    /// The first `rows` rows as a batch. The element buffer is shared whole —
    /// the derived offsets restrict which part of it a prefix reads.
    fn prefix(&self, rows: usize) -> Batch<'_> {
        Batch::new(rows).column(
            "items",
            ColumnRef::List {
                lens: &self.lens[..rows],
                fields: vec![(Some("price"), ColumnRef::Int(&self.elems))],
            },
        )
    }

    /// Mean elements per row — the list LENGTH, not the loop's trip count.
    fn mean_trip(&self) -> f64 {
        self.lens.iter().sum::<i64>() as f64 / self.lens.len() as f64
    }

    /// Elements the comprehension actually inspects, over the first `rows`.
    ///
    /// `all()` stops at the first element that fails the predicate, so the list
    /// length is NOT the trip count, and where that first failure lands is most
    /// of what separates these shapes. Dividing ns/row by the length would
    /// charge the loop for elements it never read, which is why the per-element
    /// column reports against this instead.
    fn examined(&self, rows: usize) -> u64 {
        let mut total = 0;
        let mut off = 0usize;
        for r in 0..rows {
            let len = self.lens[r] as usize;
            let mut seen = 0;
            for k in 0..len {
                seen += 1;
                if self.elems[off + k] <= 10 {
                    break;
                }
            }
            total += seen;
            off += len;
        }
        total
    }
}

/// The MINIMUM over rounds, not the median. Interference from other work on the
/// box can only ever make a round slower, so the fastest round is the one that
/// ran with the least of it and is the robust estimator of the tier's own cost.
/// A median is pulled around by how loaded the machine happened to be, which on
/// a shared box swings the baseline several-fold and makes the ratio unreadable.
fn best(values: Vec<Duration>) -> Duration {
    values.into_iter().min().expect("at least one round")
}

fn ns_per_row(d: Duration, rows: usize) -> f64 {
    d.as_secs_f64() * 1e9 / rows as f64
}

/// One swept data shape: a label and the per-row element count it produces.
type Shape = (&'static str, fn(usize) -> i64);

fn nested_schema() -> Schema {
    [("items[].price".to_string(), ValType::Int)]
        .into_iter()
        .collect()
}

fn lower(src: &str, schema: &Schema) -> BatchProgram {
    BatchProgram::compile(src, schema).unwrap_or_else(|e| panic!("lower `{src}`: {e}"))
}

struct Point {
    rows: usize,
    clean: Duration,
    interp: Duration,
    jit: Duration,
    compiles: usize,
    deopts: usize,
    aborts: usize,
}

fn measure_at(
    lowered: &BatchProgram,
    data: &ListColumns,
    rows: usize,
    rounds: usize,
    label: &str,
) -> Point {
    let batch = data.prefix(rows);
    let bound = lowered.bind(&batch).expect("bind list columns");

    // Oracle first: never report a timing taken off a miscompile.
    let expected = bound.sum_on(Tier::Clean).ok();
    reset_jit_stats();
    let compiled = bound.sum_on(Tier::Jit).ok();
    let compiles = jit_stats().loops_compiled;
    let deopts = jit_stats().guard_failures;
    let aborts = jit_stats().loops_aborted;
    assert_eq!(
        expected, compiled,
        "{label} @{rows}: compiled tier diverged from the oracle tier"
    );
    assert_eq!(
        expected,
        bound.sum_on(Tier::Interpreter).ok(),
        "{label} @{rows}: majit interpreter tier diverged from the oracle tier"
    );

    let mut clean_times = Vec::with_capacity(rounds);
    let mut interp_times = Vec::with_capacity(rounds);
    let mut jit_times = Vec::with_capacity(rounds);
    // Every timed run is UNWRAPPED. A refused batch returns early, so `.ok()`
    // here would not report a failure as slow — it would report it as fast, and
    // the ladder would read as if the tier had got better.
    let run = |tier| {
        bound
            .sum_on(tier)
            .unwrap_or_else(|e| panic!("{label} @{rows}: {tier:?}: {e}"))
    };
    for _ in 0..rounds {
        let start = Instant::now();
        black_box(run(Tier::Clean));
        clean_times.push(start.elapsed());

        let start = Instant::now();
        black_box(run(Tier::Interpreter));
        interp_times.push(start.elapsed());

        let start = Instant::now();
        black_box(run(Tier::Jit));
        jit_times.push(start.elapsed());
    }

    Point {
        rows,
        clean: best(clean_times),
        interp: best(interp_times),
        jit: best(jit_times),
        compiles,
        deopts,
        aborts,
    }
}

/// The tree-walker over the SAME expression and the same per-row data, as a
/// sanity floor for the whole columnar pipeline: a JIT that beats the bytecode
/// VM would still be no use if the bytecode VM were itself slower than the
/// evaluator cel actually ships.
///
/// This is a REFERENCE, not a ratio to quote. It cycles a small pool of
/// activations, so its data stays in cache while the columnar tiers stream
/// hundreds of thousands of distinct rows out of memory — the comparison is
/// biased in the tree-walker's favour, which is the safe direction for a floor.
/// Activation construction is outside the timed region, as in `majit_ab`.
fn tree_walker_ns_per_eval(
    program: &Program,
    data: &ListColumns,
    rounds: usize,
    pool: usize,
) -> (f64, usize) {
    let mut base = 0usize;
    // One shared root, a child scope per row. `Context::default` constructs
    // `Env::stdlib()` on every call, so building one per row would hold `pool`
    // copies of the standard library alive for the whole timed region.
    let root = Context::default();
    let contexts: Vec<Context<'_>> = (0..pool)
        .map(|r| {
            let len = data.lens[r] as usize;
            let items: Vec<Value> = (0..len)
                .map(|k| {
                    Value::from(HashMap::from([(
                        "price".to_string(),
                        Value::Int(data.elems[base + k]),
                    )]))
                })
                .collect();
            base += len;
            let mut context = root.new_inner_scope();
            context.add_variable_from_value("items", items);
            context
        })
        .collect();

    // Same predicate, same rows: the tree-walker's `true` count over the pool
    // must equal the columnar tiers' sum over those rows.
    let matches = contexts
        .iter()
        .filter(|c| {
            match Value::resolve_value(program.expression(), c).expect("tree-walk failed") {
                Value::Bool(v) => v,
                other => panic!("expected a boolean result, got {other:?}"),
            }
        })
        .count();

    let mut times = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let start = Instant::now();
        let mut observed = 0usize;
        for i in 0..TREE_EVALS {
            let context = black_box(&contexts[i % pool]);
            let v = Value::resolve_value(black_box(program).expression(), context)
                .expect("tree-walk failed inside the timed region");
            observed ^= matches!(v, Value::Bool(true)) as usize;
        }
        black_box(observed);
        times.push(start.elapsed());
    }
    (best(times).as_secs_f64() * 1e9 / TREE_EVALS as f64, matches)
}

/// Ordinary least squares of `total = intercept + slope * rows` over the ladder.
/// Returns `(intercept_ns, slope_ns_per_row)`.
fn fit(points: &[Point], pick: impl Fn(&Point) -> Duration) -> (f64, f64) {
    let n = points.len() as f64;
    let xs: Vec<f64> = points.iter().map(|p| p.rows as f64).collect();
    let ys: Vec<f64> = points.iter().map(|p| pick(p).as_secs_f64() * 1e9).collect();
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let num: f64 = xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let den: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let slope = num / den;
    (mean_y - slope * mean_x, slope)
}

fn ladder(max_rows: usize) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut n = 10_000usize;
    while n < max_rows {
        sizes.push(n);
        n *= 4;
    }
    sizes.push(max_rows);
    sizes
}

fn main() {
    let mut args = std::env::args().skip(1);
    let max_rows: usize = args
        .next()
        .map(|a| a.parse().expect("max_rows must be a number"))
        .unwrap_or(640_000);
    let rounds: usize = args
        .next()
        .map(|a| a.parse().expect("rounds must be a number"))
        .unwrap_or(5);
    // Optional substring filter over the shape labels. The `MAJIT_STATS`
    // counters are process-global and cumulative, so attributing a
    // Tracing/Backend split to one shape means running only that shape.
    let only = args.next();

    const SRC: &str = "items.all(i, i.price > 10)";
    let schema = nested_schema();
    let lowered = lower(SRC, &schema);
    let program = Program::compile(SRC).expect("compile the reference program");

    let cases: [Shape; 5] = [
        ("constant 8", |_| 8),
        ("alternating 8/9", |r| if r % 2 == 0 { 8 } else { 9 }),
        ("cycle 4..12", |r| 4 + (r % 9) as i64),
        ("spread 0..32", |r| ((r * 2654435761) % 32) as i64),
        ("constant 64", |_| 64),
    ];
    let sizes = ladder(max_rows);

    println!("items.all(i, i.price > 10) — {rounds} interleaved rounds per point");
    println!(
        "the driver and the program words outlive a run, so trace + compile are inside the \
         timed region only until one of them has paid for the loop; the `cmp` column says which"
    );

    // Whether the compiled tier was ever reached. Tracked across the whole run
    // rather than per shape or per point: the program is the same allocation
    // everywhere here — only the data and the seeded base registers change — so
    // the first point to trace pays for every later one, and `cmp 0` on a
    // shape is the normal case, not a miss.
    let mut swept = 0usize;
    let mut compiled_somewhere = false;

    for (label, len_of) in cases {
        if let Some(filter) = &only {
            if !label.contains(filter.as_str()) {
                continue;
            }
        }
        let data = ListColumns::build(max_rows, len_of);
        let pool_rows = TREE_POOL.min(max_rows);
        let (tree_ns, tree_matches) = tree_walker_ns_per_eval(&program, &data, rounds, pool_rows);
        // The reference tier and the columnar tiers must agree on the same rows.
        let pool_batch = data.prefix(pool_rows);
        assert_eq!(
            lowered
                .bind(&pool_batch)
                .expect("bind list columns")
                .sum_on(Tier::Clean)
                .ok(),
            Some(Value::Int(tree_matches as i64)),
            "{label}: the tree-walker and the columnar lowering disagree"
        );
        println!();
        println!(
            "[{label}]  mean list length {:.2}, mean elements examined {:.2};  \
             tree-walker reference \
             {tree_ns:.2} ns/eval (cached Value::resolve_value, pool of {pool_rows})",
            data.mean_trip(),
            data.examined(max_rows) as f64 / max_rows as f64,
        );
        println!(
            "{:>10} {:>11} {:>12} {:>11} {:>9} {:>10} {:>10} {:>5} {:>8} {:>5}",
            "rows",
            "clean",
            "interp",
            "jit",
            "jit/elem",
            "jit/clean",
            "jit/interp",
            "cmp",
            "deopts",
            "abrt"
        );
        let mut points = Vec::new();
        for &n in &sizes {
            let p = measure_at(&lowered, &data, n, rounds, label);
            println!(
                "{:>10} {:>8.2} ns {:>9.2} ns {:>8.2} ns {:>6.2} ns {:>9.2}x {:>9.2}x {:>5} \
                 {:>8} {:>5}",
                p.rows,
                ns_per_row(p.clean, p.rows),
                ns_per_row(p.interp, p.rows),
                ns_per_row(p.jit, p.rows),
                p.jit.as_secs_f64() * 1e9 / data.examined(p.rows) as f64,
                p.clean.as_secs_f64() / p.jit.as_secs_f64(),
                p.interp.as_secs_f64() / p.jit.as_secs_f64(),
                p.compiles,
                p.deopts,
                p.aborts,
            );
            points.push(p);
        }

        swept += points.len();
        compiled_somewhere |= points.iter().any(|p| p.compiles >= 1);

        // Least squares over the whole ladder rather than a two-point slope: one
        // noisy point cannot then invert the fit into a negative per-row cost.
        let (compile, steady_jit) = fit(&points, |p| p.jit);
        let (_, steady_clean) = fit(&points, |p| p.clean);
        let first_win = points.iter().find(|p| p.jit < p.clean).map(|p| p.rows);
        // A negative intercept or a negative slope means the totals do not
        // separate into a fixed price plus a per-row one, so report that rather
        // than a negative compile time or a negative ns/row. It happens at both
        // ends: when a per-row deopt leaves no fixed cost to find, and when the
        // per-row cost is so small that noise on the short sizes dominates.
        if compile < 0.0 || steady_jit <= 0.0 {
            println!(
                "  fit degenerate (intercept {:.2} ms, slope {steady_jit:.2} ns/row): \
                 the ladder does not separate a fixed cost from a per-row one here",
                compile / 1e6
            );
        } else {
            println!(
                "  fit: steady jit {steady_jit:.2} ns/row vs clean {steady_clean:.2} ns/row \
                 = {:.2}x steady;  trace+compile {:.2} ms",
                steady_clean / steady_jit,
                compile / 1e6,
            );
        }
        let break_even = if steady_clean > steady_jit && steady_jit > 0.0 && compile > 0.0 {
            Some(compile / (steady_clean - steady_jit))
        } else {
            None
        };
        match (first_win, break_even) {
            (Some(n), Some(b)) => {
                println!("  JIT total first wins at {n} swept rows; fitted break-even {b:.0} rows")
            }
            (Some(n), None) => println!("  JIT total first wins at {n} swept rows"),
            (None, _) => println!("  no swept size where the JIT total wins"),
        }
        let largest = points.last().expect("the ladder has at least one point");
        println!(
            "  floor check: tree-walker {tree_ns:.2} ns/eval vs clean VM {:.2} ns/row \
             vs jit {:.2} ns/row @{} rows",
            ns_per_row(largest.clean, largest.rows),
            ns_per_row(largest.jit, largest.rows),
            largest.rows,
        );
    }

    // Without this, a run where nothing ever compiled would still print a `jit`
    // column — the tracing interpreter's, which is the SLOWEST tier here, under
    // the compiled tier's heading.
    assert!(
        swept == 0 || compiled_somewhere,
        "no swept size compiled the loop, so every `jit` column is the interpreter"
    );
}
