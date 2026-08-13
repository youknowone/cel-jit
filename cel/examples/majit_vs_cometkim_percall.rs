//! cometkim's benchmark set (cel-jit PR #233, `benches/comparison.rs`) in
//! cometkim's OWN REGIME, so a number here can be read beside one of his.
//!
//! `majit_vs_cometkim` asks his expressions of the batch machine over 50,000
//! varying rows and says, at the top of the file, that this is NOT comparable to
//! his figures: his unit is `b.iter(|| compiled.execute(&ctx))` — one
//! evaluation, one FIXED activation, a program compiled once outside the timer.
//! This file measures that unit. Same expressions, same activations built the
//! same way (his literal values, added by name to a root `Context`), one call
//! timed.
//!
//! What each column is, and what it is not:
//!
//! * **stock** — `Value::resolve_value(program.expression(), &ctx)`, the tree
//!   walker, the same evaluator his `interpreted` column measures. His ran
//!   against upstream `cel` 0.11.6; this one runs against ours, so the two are
//!   one measurement of two versions of one evaluator. The walker is called
//!   DIRECTLY, not through `Program::execute`, because that door is the bytecode
//!   VM whenever the `vm` feature is on — a DEFAULT feature, which
//!   `required-features = ["jit"]` does not turn off. Going through it would
//!   silently make this column a different evaluator from the one his figures
//!   were taken on, which is the entire basis of the comparison.
//! * **majit** — the compiled tier through a ONE-ROW batch: `bind_per_row` once,
//!   `collect_on(Tier::Jit)` per call. That returns the row's `Value`, which is
//!   what `execute` returns, so it is the same contract.
//! * **raw** — the same call through `collect_raw_on`, where a columnar consumer
//!   takes the machine's own buffers and no `Value` is built. Not comparable to
//!   `stock`, which necessarily produces one.
//! * **bind** — one `bind_per_row` of the same activation, timed on its own.
//!
//! ⚠️ The deviation that matters, stated plainly: his compiled function reads
//! every variable out of the `Context` BY NAME on every call (`rt_get_variable`
//! into a `BTreeMap`), and ours does not — a majit activation is resolved to
//! slots and encoded into columns once, at `bind`. Both hold the activation
//! fixed outside the timer, exactly as he does, but the work left inside it is
//! not the same work. `bind` is printed so a reader can put that cost back and
//! bound the advantage: it is the whole per-activation encoding, which is MORE
//! than name resolution, so `majit + bind` is a pessimistic upper bound on what
//! a fair single-shot majit call would cost.
//!
//! ⚠️ `variable_access/resolver` is his one case whose stock side reads through a
//! `VariableResolver` rather than the context map; that is reproduced here,
//! while the majit side reads the same value from a one-element column.
//!
//! The two activations — his hand-built context and the batch's columns — are
//! written out separately and then GATED against each other through the
//! library's own `RowReader`, so a column and the variable it stands for cannot
//! drift apart silently.
//!
//! RELEASE ONLY, and to match his build (`[profile.bench]`: lto, one codegen
//! unit) run it under the same profile:
//!
//! ```text
//! cargo run --profile bench --package cel --features jit-cranelift \
//!     --example majit_vs_cometkim_percall
//! ```
//!
//! To reproduce HIS two columns on the same machine, check out
//! `cel-rust/cel-rust` at the PR-233 head and run
//! `cargo bench -p cel-jit --bench comparison`; that measures upstream 0.11.6's
//! tree-walker and his Cranelift AOT backend, and nothing in this file is
//! derived from it.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cel::context::VariableResolver;
use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, RawOutput, RowReader, Tier};
use cel::majit::bytecode::float_bank::{
    reset_persistent_state, COMPILES, GUARD_FAILS, TRACE_ABORTS,
};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

/// One timed batch must last at least this long, so the clock's own resolution
/// is not what a 7 ns call is being measured against.
const MIN_BATCH: Duration = Duration::from_millis(20);
/// Timed batches per measurement. The MINIMUM is reported: other work on the box
/// can only ever make a batch slower, so the fastest one ran with the least
/// interference. A median moves with how loaded the machine happened to be.
const ROUNDS: usize = 7;

/// His `benchmark_variable_access` resolver, verbatim.
struct Resolver;

impl VariableResolver for Resolver {
    fn resolve(&self, expr: &str) -> Option<Value> {
        const V: Value = Value::Bool(false);
        const NOT_V: Value = Value::Bool(true);
        match expr {
            "fruit" => Some(NOT_V),
            "carrot" => Some(NOT_V),
            "orange" => Some(NOT_V),
            "banana" => Some(V),
            _ => None,
        }
    }
}

static RESOLVER: Resolver = Resolver;

/// One input column of the single activation, owned, in the layout the batch
/// reads: a [`ColumnRef`] over one row.
enum Col {
    Int(Vec<i64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
    /// A list of `int`. One row, so `lens` is a single element count and `elems`
    /// is that row's elements.
    IntList {
        lens: Vec<i64>,
        elems: Vec<i64>,
    },
}

impl Col {
    fn column_ref(&self) -> ColumnRef<'_> {
        match self {
            Col::Int(c) => ColumnRef::Int(c),
            Col::Bool(c) => ColumnRef::Bool(c),
            Col::Str(c) => ColumnRef::Str(c),
            Col::IntList { lens, elems } => ColumnRef::List {
                lens,
                fields: vec![(None, ColumnRef::Int(elems))],
            },
        }
    }
}

/// One expression from his suite, with his activation.
struct Case {
    label: String,
    src: String,
    /// Every path the expression reads. The lowering declines an undeclared
    /// path, so this is the case's input type declaration, not a convenience.
    schema: Vec<(String, ValType)>,
    cols: Vec<(String, Col)>,
    /// His activation: the variables he adds by name, and the functions he
    /// registers, on the ROOT context the timed call evaluates against. A
    /// registered function is an opaque Rust closure, so the lowering — which
    /// holds only the schema — declines the expression and the walker answers
    /// it; without this the walker could not either.
    stock: Box<dyn Fn(&mut Context<'static>)>,
    /// Read the variable through [`RESOLVER`] instead of the context map, which
    /// is what his `variable_access/resolver` measures.
    stock_resolver: bool,
    /// `(ladder, n)` for a size-ladder member, `None` otherwise.
    ///
    /// Carried as data rather than parsed back out of `label`: the decomposition
    /// below is the number the P5 gate is stated in, and deriving its input by
    /// splitting a display string would make a renamed case silently drop out of
    /// the fit instead of failing.
    ladder: Option<(&'static str, i64)>,
}

impl Case {
    fn new(label: &str, src: &str) -> Case {
        Case {
            label: label.to_string(),
            src: src.to_string(),
            schema: Vec::new(),
            cols: Vec::new(),
            stock: Box::new(|_| {}),
            stock_resolver: false,
            ladder: None,
        }
    }

    fn in_ladder(mut self, ladder: &'static str, n: i64) -> Case {
        self.ladder = Some((ladder, n));
        self
    }

    fn col(mut self, name: &str, ty: ValType, path_suffix: &str, col: Col) -> Case {
        self.schema.push((format!("{name}{path_suffix}"), ty));
        self.cols.push((name.to_string(), col));
        self
    }

    fn int(self, name: &str, v: i64) -> Case {
        self.col(name, ValType::Int, "", Col::Int(vec![v]))
    }

    fn bool(self, name: &str, v: bool) -> Case {
        self.col(name, ValType::Bool, "", Col::Bool(vec![v]))
    }

    fn text(self, name: &str, v: &str) -> Case {
        self.col(name, ValType::Str, "", Col::Str(vec![v.to_string()]))
    }

    fn int_list(self, name: &str, elems: Vec<i64>) -> Case {
        let lens = vec![elems.len() as i64];
        self.col(name, ValType::Int, "[]", Col::IntList { lens, elems })
    }

    fn stock(mut self, f: impl Fn(&mut Context<'static>) + 'static) -> Case {
        self.stock = Box::new(f);
        self
    }

    fn via_resolver(mut self) -> Case {
        self.stock_resolver = true;
        self
    }
}

/// A `HashMap` literal the way his benchmark writes one.
fn map_of(pairs: Vec<(&'static str, Value)>) -> Value {
    Value::from(pairs.into_iter().collect::<HashMap<&str, Value>>())
}

/// His 18 benchmark expressions with his contexts. Where he benchmarks a size
/// ladder the whole ladder is here, because that is where two evaluators
/// separate.
fn cases() -> Vec<Case> {
    let mut cases = vec![
        Case::new("simple_arithmetic", "1 + 2 * 3 - 4 / 2"),
        Case::new("comparison", "10 > 5 && 3 < 7 || 1 == 1"),
        Case::new("conditional", "x > 10 ? x * 2 : x + 5")
            .int("x", 15)
            .stock(|ctx| ctx.add_variable_from_value("x", 15i64)),
        Case::new(
            "nested_expression",
            "((a + b) * (c - d)) / ((e + f) - (g * h))",
        )
        .int("a", 10)
        .int("b", 20)
        .int("c", 30)
        .int("d", 5)
        .int("e", 15)
        .int("f", 25)
        .int("g", 2)
        .int("h", 3)
        .stock(|ctx| {
            for (name, v) in [
                ("a", 10i64),
                ("b", 20),
                ("c", 30),
                ("d", 5),
                ("e", 15),
                ("f", 25),
                ("g", 2),
                ("h", 3),
            ] {
                ctx.add_variable_from_value(name, v);
            }
        }),
        Case::new("variable_access/hashmap", "apple")
            .bool("apple", true)
            .stock(|ctx| ctx.add_variable_from_value("apple", true)),
        Case::new("variable_access/resolver", "banana")
            .bool("banana", false)
            .via_resolver(),
        Case::new("member_access", "obj.nested.value + obj.other")
            .int("obj.nested.value", 42)
            .int("obj.other", 10)
            .stock(|ctx| {
                let obj = map_of(vec![
                    ("nested", map_of(vec![("value", Value::Int(42))])),
                    ("other", Value::Int(10)),
                ]);
                ctx.add_variable_from_value("obj", obj);
            }),
        Case::new("list_indexing", "list[0] + list[5] + list[9]")
            .int_list("list", (1..=10).collect())
            .stock(|ctx| ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())),
        Case::new(
            "list_filter",
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(x, x > 5)",
        ),
        Case::new("list_map", "[1, 2, 3, 4, 5].map(x, x * 2)"),
        Case::new("all_comprehension", "[1, 2, 3, 4, 5].all(x, x > 0)"),
        Case::new("exists_comprehension", "[1, 2, 3, 4, 5].exists(x, x == 3)"),
    ];

    let ladder = |ladder: &'static str, n: i64, src: &str, name: &'static str, elems: Vec<i64>| {
        let stock = elems.clone();
        Case::new(&format!("{ladder}/{n}"), src)
            .int_list(name, elems)
            .stock(move |ctx| ctx.add_variable_from_value(name, stock.clone()))
            .in_ladder(ladder, n)
    };
    for size in [1i64, 10, 100, 1_000, 10_000] {
        cases.push(ladder(
            "map_list_scaling",
            size,
            "list.map(x, x * 2)",
            "list",
            (0..size).collect(),
        ));
    }
    for size in [1i64, 10, 100, 1_000, 10_000] {
        cases.push(ladder(
            "filter_list_scaling",
            size,
            "list.filter(x, x % 2 == 0)",
            "list",
            (0..size).collect(),
        ));
    }
    for size in [10i64, 50, 100, 500] {
        cases.push(ladder(
            "comprehension_scaling",
            size,
            "items.filter(x, x % 2 == 0).map(x, x * 2)",
            "items",
            (1..=size).collect(),
        ));
    }

    cases.push(Case::new(
        "string_operations",
        r#""hello world".startsWith("hello") && "hello world".endsWith("world") && "hello world".contains("o w")"#,
    ));
    cases.push(
        Case::new("custom_function", "add(x, y) + multiply(a, b)")
            .int("x", 10)
            .int("y", 20)
            .int("a", 5)
            .int("b", 3)
            .stock(|ctx| {
                for (name, v) in [("x", 10i64), ("y", 20), ("a", 5), ("b", 3)] {
                    ctx.add_variable_from_value(name, v);
                }
                ctx.add_function("add", |a: i64, b: i64| a + b);
                ctx.add_function("multiply", |a: i64, b: i64| a * b);
            }),
    );
    cases.push(
        Case::new(
            "real_world_policy",
            r#"user.age >= 18 &&
               user.role in ["admin", "moderator"] &&
               request.method == "POST" &&
               request.path.startsWith("/api/") &&
               size(request.body) < 1000000"#,
        )
        .int("user.age", 25)
        .text("user.role", "admin")
        .text("request.method", "POST")
        .text("request.path", "/api/users")
        .text("request.body", "{}")
        .stock(|ctx| {
            ctx.add_variable_from_value(
                "user",
                map_of(vec![
                    ("age", Value::Int(25)),
                    ("role", Value::from("admin")),
                ]),
            );
            ctx.add_variable_from_value(
                "request",
                map_of(vec![
                    ("method", Value::from("POST")),
                    ("path", Value::from("/api/users")),
                    ("body", Value::from("{}")),
                ]),
            );
        }),
    );
    cases
}

/// Time ONE call. Grows an iteration count until a timed batch lasts at least
/// [`MIN_BATCH`], then reports the fastest of [`ROUNDS`] such batches.
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
            break;
        }
        // Aim straight at the target instead of doubling: a call that costs
        // milliseconds would otherwise spend most of the calibration finding
        // that out, and one that costs nanoseconds would spend 20 doublings.
        let want = MIN_BATCH.as_secs_f64() / elapsed.as_secs_f64().max(1e-9);
        let grow = (want.ceil() as usize).clamp(2, 1 << 12);
        iters = iters.saturating_mul(grow);
    }

    (0..ROUNDS)
        .map(|_| timed(iters, &mut run).as_nanos() as f64 / iters as f64)
        .fold(f64::INFINITY, f64::min)
}

/// A columnar consumer. It reads the buffers the run wrote and never builds a
/// `Value`; the sum is only so the run cannot be optimized away.
fn consume_raw(out: RawOutput<'_>) -> i64 {
    let add = |a: i64, &b: &i64| a.wrapping_add(b);
    match out {
        RawOutput::Scalar { values, .. } => values.iter().fold(0, add),
        RawOutput::List { lens, fields, .. } => {
            let n: usize = lens.iter().map(|&c| c.max(0) as usize).sum();
            fields
                .iter()
                .map(|(_, _, buf)| buf[..n].iter().fold(0i64, add))
                .fold(0, i64::wrapping_add)
        }
    }
}

/// What the compiled tier did, or why it never ran.
struct Compiled {
    /// The plain Rust bytecode VM over the same lowered program, with no tracing
    /// machinery at all. It is the FLOOR the compiled tier has to beat, and the
    /// control that says whether a slow `majit` cell is the cost of the machine
    /// or the cost of the tracer failing to get out of the way.
    clean: f64,
    majit: f64,
    raw: f64,
    bind: f64,
    compiles: usize,
    /// Traces started and thrown away, per call, once warm. A loop that never
    /// compiles is either aborting — counted here — or never reaching its merge
    /// point hot enough to be traced; only this tells the two apart.
    aborts: f64,
    guard_fails: f64,
}

struct Row {
    label: String,
    /// `(ladder, n)`, copied from the case so the decomposition below has its
    /// input as data.
    ladder: Option<(&'static str, i64)>,
    stock: f64,
    /// `Err` when the expression does not lower: the tree-walker answers it —
    /// through the library's own fallback, not a hand-written one — and there is
    /// no compiled tier to put beside it.
    compiled: Result<Compiled, String>,
}

fn run_case(case: &Case) -> Row {
    let program = Program::compile(&case.src)
        .unwrap_or_else(|e| panic!("{}: parse error: {e:?}", case.label));
    let schema: Schema = case.schema.iter().cloned().collect();

    let mut batch = Batch::new(1);
    for (name, col) in &case.cols {
        batch = batch.column(name.clone(), col.column_ref());
    }

    // His activation, built his way and held fixed outside the timer: a root
    // context with the variables added by name.
    let mut activation = Context::default();
    (case.stock)(&mut activation);
    if case.stock_resolver {
        activation.set_variable_resolver(&RESOLVER);
    }
    let expected = Value::resolve_value(program.expression(), &activation)
        .unwrap_or_else(|e| panic!("{}: stock execute: {e:?}", case.label));

    // Drift gate: the same walker, over a child scope the LIBRARY's `RowReader`
    // filled from the very columns the compiled tier reads, must reach the same
    // answer. Without it a column and the variable it is supposed to stand for
    // can disagree and every ratio below silently compares two workloads.
    let mirrored = RowReader::new(&batch).scope(&activation, 0);
    assert_eq!(
        Value::resolve_value(program.expression(), &mirrored)
            .ok()
            .as_ref(),
        Some(&expected),
        "{}: the batch columns and the hand-built activation disagree",
        case.label
    );

    // Nothing timed in this file may DISCARD a `Result`. A refused evaluation
    // returns early, so a swallowed error is not a slow number, it is a fast
    // one — which is exactly what cometkim's own benchmark reports: his
    // `b.iter(|| black_box(compiled.execute(&ctx)))` never unwraps, and his
    // backend answers `items.filter(..).map(..)` with
    // `UndeclaredReference("@result")`, so all four `comprehension_scaling`
    // rows time a failure and print it as a two-fold speedup.
    let stock = per_call(|| {
        Value::resolve_value(program.expression(), black_box(&activation))
            .unwrap_or_else(|e| panic!("{}: stock execute: {e:?}", case.label))
    });

    let lowered = match BatchProgram::from_program(&program, &schema) {
        Ok(bp) => bp,
        Err(e) => {
            return Row {
                label: case.label.clone(),
                ladder: case.ladder,
                stock,
                compiled: Err(format!("declines: {e}")),
            }
        }
    };
    // A fresh driver, so `compiles` counts this case's loops and not a loop an
    // earlier case left compiled at the same program address.
    reset_persistent_state();
    let bound = match lowered.bind_per_row(&batch) {
        Ok(b) => b,
        Err(e) => {
            return Row {
                label: case.label.clone(),
                ladder: case.ladder,
                stock,
                compiled: Err(format!("cannot bind: {e}")),
            }
        }
    };

    // Miscompile gate: all three tiers, and the tree-walker, agree on the one
    // row. `collect` is the per-row door, so this compares the VALUE his
    // `execute` returns, not a batch reduction of it.
    COMPILES.store(0, Ordering::Relaxed);
    for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
        let got = bound
            .collect_on(tier)
            .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label));
        assert_eq!(
            got.as_slice(),
            std::slice::from_ref(&expected),
            "{}: {tier:?} vs stock",
            case.label
        );
    }
    // One row per call means the batch loop crosses its header once per call, so
    // the trace threshold is reached across CALLS. Warm until it is, before
    // anything is timed: otherwise the `majit` column of a case that never
    // compiled would be the tracing interpreter's number under the compiled
    // tier's heading.
    warm(&bound);
    let compiles = COMPILES.load(Ordering::Relaxed);
    // What the driver is still doing per call once it is as warm as it will get.
    const SETTLED: usize = 1_000;
    let (a0, g0) = (
        TRACE_ABORTS.load(Ordering::Relaxed),
        GUARD_FAILS.load(Ordering::Relaxed),
    );
    for _ in 0..SETTLED {
        black_box(bound.collect_on(Tier::Jit).expect("settled run"));
    }
    let aborts = (TRACE_ABORTS.load(Ordering::Relaxed) - a0) as f64 / SETTLED as f64;
    let guard_fails = (GUARD_FAILS.load(Ordering::Relaxed) - g0) as f64 / SETTLED as f64;

    let collect = |tier| {
        bound
            .collect_on(tier)
            .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label))
    };
    let clean = per_call(|| collect(Tier::Clean));
    let majit = per_call(|| collect(Tier::Jit));
    let raw = per_call(|| {
        bound
            .collect_raw_on(Tier::Jit, consume_raw)
            .unwrap_or_else(|e| panic!("{}: raw: {e}", case.label))
    });
    let bind = per_call(|| {
        lowered
            .bind_per_row(&batch)
            .unwrap_or_else(|e| panic!("{}: rebind: {e}", case.label))
    });

    Row {
        label: case.label.clone(),
        ladder: case.ladder,
        stock,
        compiled: Ok(Compiled {
            clean,
            majit,
            raw,
            bind,
            compiles,
            aborts,
            guard_fails,
        }),
    }
}

fn warm(bound: &BoundBatch<'_, '_>) {
    for _ in 0..256 {
        black_box(bound.collect_on(Tier::Jit).expect("warm run"));
    }
}

/// A cost model `fixed + per_elem * n`, fitted through two points.
struct Fit {
    fixed: f64,
    per_elem: f64,
}

impl Fit {
    /// Through `(n_lo, t_lo)` and `(n_hi, t_hi)`.
    ///
    /// Two points, not a least-squares line over all of them, because that is
    /// what task #88's table was computed with and reproducing its METHOD is the
    /// point — a different estimator would make the two figures incomparable for
    /// a reason that has nothing to do with the machine. The fit therefore
    /// passes through its endpoints by construction, so it is not evidence of
    /// linearity; `worst_mid_err` below is what tests that.
    fn two_point(lo: (f64, f64), hi: (f64, f64)) -> Fit {
        let per_elem = (hi.1 - lo.1) / (hi.0 - lo.0);
        Fit {
            fixed: lo.1 - lo.0 * per_elem,
            per_elem,
        }
    }

    fn at(&self, n: f64) -> f64 {
        self.fixed + self.per_elem * n
    }
}

/// One ladder's decomposition.
struct Decomposition {
    ladder: &'static str,
    /// The two `n` the fit was taken through, and how many compiled points the
    /// ladder had in total.
    n_lo: i64,
    n_hi: i64,
    points: usize,
    /// Ladder members whose loop never compiled, and which are therefore NOT in
    /// the fit: their `majit` cell is the tracing interpreter, so including one
    /// would fit a different machine.
    excluded: usize,
    majit: Fit,
    clean: Fit,
    /// Worst relative error of the majit fit at a point it did NOT pass through,
    /// or `None` when the ladder has only the two endpoints.
    worst_mid_err: Option<f64>,
    /// The largest `gfails/call` over the compiled members. Printed beside the
    /// fit because a fixed cost and a per-call guard failure are the same
    /// finding read two ways, and #88's own tripwire is that a compile count
    /// must never be reported without it.
    max_gfails: f64,
}

/// Task #88's two-point decomposition over the size ladders, computed here
/// rather than by hand off the table above.
///
/// The gate this epic is under — "a compiled cel artifact's fixed per-call cost
/// under ~1 µs on both backends" — is stated in the `majit fixed` column, and
/// that column has until now been arithmetic somebody did in a notebook. A
/// number that decides a phase should be produced by the program that measures
/// it.
fn decompose(rows: &[Row]) -> Vec<Decomposition> {
    let mut ladders: Vec<&'static str> = Vec::new();
    for r in rows {
        if let Some((name, _)) = r.ladder {
            if !ladders.contains(&name) {
                ladders.push(name);
            }
        }
    }

    let mut out = Vec::new();
    for name in ladders {
        let mut pts: Vec<(f64, &Compiled)> = Vec::new();
        let mut excluded = 0usize;
        for r in rows {
            let Some((ladder, n)) = r.ladder else {
                continue;
            };
            if ladder != name {
                continue;
            }
            match &r.compiled {
                Ok(c) if c.compiles > 0 => pts.push((n as f64, c)),
                _ => excluded += 1,
            }
        }
        if pts.len() < 2 {
            continue;
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (lo, hi) = (&pts[0], &pts[pts.len() - 1]);

        let majit = Fit::two_point((lo.0, lo.1.majit), (hi.0, hi.1.majit));
        let clean = Fit::two_point((lo.0, lo.1.clean), (hi.0, hi.1.clean));

        let worst_mid_err = pts[1..pts.len() - 1]
            .iter()
            .map(|(n, c)| ((majit.at(*n) - c.majit) / c.majit).abs())
            .fold(None::<f64>, |acc, e| Some(acc.map_or(e, |a: f64| a.max(e))));

        out.push(Decomposition {
            ladder: name,
            n_lo: lo.0 as i64,
            n_hi: hi.0 as i64,
            points: pts.len(),
            excluded,
            max_gfails: pts.iter().map(|(_, c)| c.guard_fails).fold(0.0, f64::max),
            majit,
            clean,
            worst_mid_err,
        });
    }
    out
}

/// `n` where the compiled tier's cost model crosses the clean VM's.
///
/// `None` when the compiled tier is not cheaper per element, in which case it
/// never catches up and there is no crossing to report — a state the table has
/// to be able to print, because #88 measured a NEGATIVE per-element cost once
/// (`cranelift/map`, −0.32 ns) and a sign flip there is a real outcome.
fn break_even(d: &Decomposition) -> Option<f64> {
    let gain = d.clean.per_elem - d.majit.per_elem;
    (gain > 0.0).then(|| (d.majit.fixed - d.clean.fixed) / gain)
}

fn print_decomposition(rows: &[Row]) {
    let table = decompose(rows);
    if table.is_empty() {
        return;
    }
    println!(
        "\ntask #88's two-point decomposition, computed here rather than by hand.\n\
         `majit fixed` is the column the STOP-AT-P5 gate is stated in: its re-entry\n\
         criterion is a compiled artifact's fixed per-call cost under ~1 us."
    );
    println!(
        "\n{:<22} {:>7} {:>13} {:>12} {:>11} {:>12} {:>11} {:>11} {:>9} {:>12}",
        "ladder",
        "points",
        "fit through n",
        "majit fixed",
        "majit/elem",
        "clean fixed",
        "clean/elem",
        "break-even",
        "mid err",
        "gfails/call"
    );
    for d in &table {
        let be = match break_even(d) {
            Some(n) => format!("{n:>11.0}"),
            // The compiled tier is not cheaper per element on this ladder, so it
            // never overtakes. Printed rather than left blank.
            None => format!("{:>11}", "never"),
        };
        let mid = match d.worst_mid_err {
            Some(e) => format!("{:>8.1}%", e * 100.0),
            None => format!("{:>9}", "-"),
        };
        println!(
            "{:<22} {:>7} {:>6}..{:<6} {:>12.1} {:>11.3} {:>12.1} {:>11.3} {be} {mid} {:>12.2}",
            d.ladder,
            d.points,
            d.n_lo,
            d.n_hi,
            d.majit.fixed,
            d.majit.per_elem,
            d.clean.fixed,
            d.clean.per_elem,
            d.max_gfails,
        );
    }
    let excluded: usize = table.iter().map(|d| d.excluded).sum();
    println!(
        "\nRead it with three cautions.\n\
         * The fit passes through its two endpoints BY CONSTRUCTION, so it cannot\n\
           disagree with them. `mid err` is the whole test of the model: it is the\n\
           worst relative error at a ladder point the fit did not touch, and a\n\
           `-` means the ladder had no such point and the row is unchecked.\n\
         * {excluded} ladder member(s) are excluded because their loop never\n\
           compiled. Their `majit` cell is the tracing interpreter, and fitting it\n\
           would decompose a different machine.\n\
         * `gfails/call` belongs beside `majit fixed`, not in a separate table:\n\
           #88 found 1.00 guard failure per call on every compiled case and named\n\
           it the prime suspect for the fixed cost it measured. A fixed cost read\n\
           without it is half a finding."
    );
}

fn main() {
    println!("cometkim's benchmark expressions in his own regime (cel-jit PR #233)");
    println!(
        "one expression, one FIXED activation, ONE evaluation timed; \
         best of {ROUNDS} batches of >= {} ms.\n",
        MIN_BATCH.as_millis()
    );
    println!(
        "{:<28} {:>11} {:>11} {:>11} {:>12} {:>10} {:>10} {:>9} {:>12} {:>12}",
        "case",
        "stock ns",
        "clean ns",
        "majit ns",
        "majit/stock",
        "raw ns",
        "bind ns",
        "compiles",
        "aborts/call",
        "gfails/call"
    );

    let cases = cases();
    let mut declined = Vec::new();
    let mut lowered = 0usize;
    let mut never_compiled = Vec::new();
    // Retained, not just printed: the decomposition below needs every ladder
    // member's cells at once, and re-running a case to get them back would time
    // a second, differently-warmed process state.
    let mut rows = Vec::with_capacity(cases.len());
    for case in &cases {
        let r = run_case(case);
        match &r.compiled {
            Ok(c) => {
                lowered += 1;
                if c.compiles == 0 {
                    never_compiled.push(r.label.clone());
                }
                println!(
                    "{:<28} {:>11.1} {:>11.1} {:>11.1} {:>11.2}x {:>10.1} {:>10.1} {:>9} {:>12.2} {:>12.2}",
                    r.label,
                    r.stock,
                    c.clean,
                    c.majit,
                    r.stock / c.majit,
                    c.raw,
                    c.bind,
                    c.compiles,
                    c.aborts,
                    c.guard_fails
                );
            }
            Err(why) => {
                // Still measured and still ANSWERED — by the tree-walker,
                // through the library's fallback. A row missing from the table
                // would read as an expression this crate cannot evaluate.
                println!(
                    "{:<28} {:>11.1} {:>11} {:>11} {:>12} {:>10} {:>10} {:>9} {:>12} {:>12}",
                    r.label, r.stock, "-", "walker", "-", "-", "-", "-", "-", "-"
                );
                declined.push((r.label.clone(), why.clone()));
            }
        }
        rows.push(r);
    }

    println!(
        "\ncoverage: {lowered}/{} lower to the compiled tier; {}/{} are answered",
        cases.len(),
        cases.len(),
        cases.len()
    );
    for (label, why) in &declined {
        println!("  {label:<28} {why}");
    }
    if !never_compiled.is_empty() {
        println!(
            "\n⚠️ the batch loop never compiled for {} of {lowered} lowered cases: {}",
            never_compiled.len(),
            never_compiled.join(", ")
        );
        println!(
            "   their `majit ns` is the TRACING INTERPRETER under the compiled tier's heading.\n\
                `aborts/call` says which of the two reasons applies: a nonzero count is a loop\n\
                the tracer keeps trying and throwing away, a zero one is a loop that never gets\n\
                hot — at one row per call the batch loop has no back edge to be hot on."
        );
    }
    println!(
        "\n`clean ns` is the same lowered bytecode on the plain Rust VM, with no tracing\n\
         machinery at all. Where `majit` is far above it the cost is the tracer, not the\n\
         machine, and the compiled tier is not what answered the call."
    );
    println!(
        "\n`majit ns` holds the activation fixed exactly as his `execute(&ctx)` does, but a\n\
         majit activation was resolved to slots and encoded into columns at `bind`, while his\n\
         compiled code looks every variable up in the context BY NAME on every call. `bind ns`\n\
         is that encoding timed on its own — it is MORE than name resolution, so adding it back\n\
         is a pessimistic bound on the difference, not an estimate of it."
    );
    print_decomposition(&rows);
}
