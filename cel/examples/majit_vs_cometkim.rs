//! cometkim's benchmark set (cel-jit PR #233, `benches/comparison.rs`) used as
//! the yardstick for this JIT: every expression his suite measures, asked of
//! the typed lowering, and — where it lowers — evaluated on all three tiers
//! over the same data.
//!
//! Two things are reported, and they answer different questions:
//!
//! * **Coverage.** How many of his 18 benchmark expressions this JIT lowers at
//!   all, and the reason each remaining one declines. An external expression
//!   set nobody here chose is a harder coverage test than a census of what we
//!   already support. A declining case still gets a row: it is still ANSWERED,
//!   by the tree-walker through the library's own fallback, and dropping it
//!   from the table would read as an expression this crate cannot evaluate.
//! * **Throughput.** For the ones that lower: the stock tree-walker per row,
//!   the plain Rust bytecode VM, and the compiled trace, over one batch of
//!   identical data. Only `majit / clean VM` isolates compilation; the ratio to
//!   stock also contains the data-model change (slot resolution, no boxing).
//!   `stock` calls `Value::resolve_value` DIRECTLY rather than
//!   `Program::execute`, because that door is the bytecode VM whenever the `vm`
//!   feature is on — a DEFAULT feature, and one `required-features = ["jit"]`
//!   does not turn off — which would leave `stock` naming a walker and running
//!   a VM, and would put a VM on both sides of the `stock / clean VM` ratio.
//!
//! ⚠️ This is NOT a comparison against cometkim's own numbers. His regime is one
//! `CompiledProgram::execute(&ctx)` per criterion iteration over a FIXED
//! context — per-call latency with a persistent AOT function. This one is batch
//! throughput over varying rows. Putting his ns/call beside a ns/row would
//! invite a ratio that isolates nothing, so his numbers are not carried here;
//! his benchmark contributes its EXPRESSIONS, which is what a yardstick is for.
//!
//! ⚠️ The `stock` column dropped by up to 2.2x when the activations moved to the
//! library's `RowReader` (`list_indexing` ~210 -> ~92 ns/row, `real_world_policy`
//! ~630 -> ~350). That is a HARNESS defect fixed, not an evaluator change: the
//! old builder called `Context::default()` per row, and `Context::default`
//! constructs `Env::stdlib()` every call, so 50,000 copies of the standard
//! library were built and held alive at once. One shared root with a child
//! scope per row measures the same evaluator without that footprint. Every
//! ratio to stock in this table is therefore SMALLER than it used to print, and
//! the older, larger figures should not be quoted.
//!
//! Row-vs-element note: `map_list_scaling` and `filter_list_scaling` are the two
//! cases whose cost scales with a list's length, and this file measures them at
//! one length. `./bench.sh majit_nested_bench` is the ladder for that shape.
//!
//! RELEASE ONLY. Run: `./bench.sh majit_vs_cometkim`.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, RawOutput, RowReader, Tier};
use cel::majit::bytecode::float_bank::{reset_persistent_state, COMPILES};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

const ROWS: usize = 50_000;
/// Elements per list row. cometkim's `list_indexing` context is a 10-element
/// list and reads `[9]`, so a list column here is at least that long.
const LIST_LEN: i64 = 10;
const ROUNDS: usize = 5;

const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

/// One input column, owned, in the layout both consumers need: the batch reads
/// it as a [`ColumnRef`], the tree-walker oracle reads row `r` out of it as a
/// [`Value`].
enum Col {
    Int(Vec<i64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
    /// A list of `int`, the same number of elements in every row, flattened the
    /// way the machine reads one: `lens[r]` elements for row `r`, all rows end
    /// to end.
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

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    *state
}

/// `n` ints in `[0, span)`, offset by `bias`. Kept small so the expressions'
/// arithmetic — and the batch's own accumulator — cannot overflow, which the
/// tree-walker raises on and the batch answers `None` to.
fn ints(state: &mut u64, n: usize, span: u64, bias: i64) -> Vec<i64> {
    (0..n)
        .map(|_| (next_u64(state) % span) as i64 + bias)
        .collect()
}

fn bools(state: &mut u64, n: usize) -> Vec<bool> {
    (0..n).map(|_| next_u64(state) & 1 != 0).collect()
}

fn strings(state: &mut u64, n: usize, choices: &[&str]) -> Vec<String> {
    (0..n)
        .map(|_| choices[next_u64(state) as usize % choices.len()].to_string())
        .collect()
}

fn int_list(state: &mut u64, n: usize, span: u64) -> Col {
    Col::IntList {
        lens: vec![LIST_LEN; n],
        elems: ints(state, n * LIST_LEN as usize, span, 0),
    }
}

type Build = fn(usize) -> Vec<(&'static str, Col)>;

fn no_columns(_n: usize) -> Vec<(&'static str, Col)> {
    Vec::new()
}

fn one_int_x(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x2545_F491_4F6C_DD1D;
    vec![("x", Col::Int(ints(&mut s, n, 1 << 20, -(1 << 19))))]
}

/// `((a + b) * (c - d)) / ((e + f) - (g * h))`, ranged so the divisor is
/// provably nonzero on every row: `e + f` in [512, 1022] and `g * h` in
/// [0, 225], so the divisor is in [287, 1022]. A zero divisor is not a slow
/// case, it is a DIVERGENCE — the walker raises and the batch refuses.
fn nested_operands(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x9E37_79B9_7F4A_7C15;
    vec![
        ("a", Col::Int(ints(&mut s, n, 256, 0))),
        ("b", Col::Int(ints(&mut s, n, 256, 0))),
        ("c", Col::Int(ints(&mut s, n, 256, 0))),
        ("d", Col::Int(ints(&mut s, n, 256, 0))),
        ("e", Col::Int(ints(&mut s, n, 256, 256))),
        ("f", Col::Int(ints(&mut s, n, 256, 256))),
        ("g", Col::Int(ints(&mut s, n, 16, 0))),
        ("h", Col::Int(ints(&mut s, n, 16, 0))),
    ]
}

fn one_bool_apple(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0xD1B5_4A32_D192_ED03;
    vec![("apple", Col::Bool(bools(&mut s, n)))]
}

fn one_bool_banana(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0xA076_1D64_78BD_642F;
    vec![("banana", Col::Bool(bools(&mut s, n)))]
}

fn member_fields(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0xE703_7ED1_A0B4_28DB;
    vec![
        ("obj.nested.value", Col::Int(ints(&mut s, n, 1 << 20, 0))),
        ("obj.other", Col::Int(ints(&mut s, n, 1 << 20, 0))),
    ]
}

fn one_int_list(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x8EBC_6AF0_9C88_C6E3;
    vec![("list", int_list(&mut s, n, 1 << 16))]
}

fn one_int_list_items(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x5891_1E52_23A9_9DF7;
    vec![("items", int_list(&mut s, n, 1 << 16))]
}

fn custom_fn_operands(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x2B99_2DDF_A232_49D6;
    vec![
        ("x", Col::Int(ints(&mut s, n, 1 << 16, 0))),
        ("y", Col::Int(ints(&mut s, n, 1 << 16, 0))),
        ("a", Col::Int(ints(&mut s, n, 1 << 16, 0))),
        ("b", Col::Int(ints(&mut s, n, 1 << 16, 0))),
    ]
}

/// cometkim's policy context, varied per row so no branch of the `&&` chain is
/// constant across the batch.
fn policy_request(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x1D8E_4E27_C47D_124F;
    vec![
        ("user.age", Col::Int(ints(&mut s, n, 60, 5))),
        (
            "user.role",
            Col::Str(strings(&mut s, n, &["admin", "moderator", "guest"])),
        ),
        (
            "request.method",
            Col::Str(strings(&mut s, n, &["POST", "GET", "DELETE"])),
        ),
        (
            "request.path",
            Col::Str(strings(
                &mut s,
                n,
                &["/api/users", "/admin/audit", "/healthz"],
            )),
        ),
        (
            "request.body",
            Col::Str(strings(&mut s, n, &["{}", "{\"a\":1}", "{\"name\":\"x\"}"])),
        ),
    ]
}

/// One expression from cometkim's suite: his source, the type of every path it
/// reads, and the data to read.
struct Case {
    label: &'static str,
    src: &'static str,
    /// Every path the expression reads. The lowering declines an undeclared
    /// path, so this is the case's input type declaration, not a convenience.
    schema: &'static [(&'static str, ValType)],
    build: Build,
    /// The functions this case's expression calls, registered on the context
    /// the tree-walker gets. A registered function is an opaque Rust closure,
    /// so the lowering — which holds only the schema — declines the expression
    /// and the walker answers it; without this the walker could not either.
    register: fn(&mut Context<'static>),
}

/// The cases that call nothing beyond the standard library.
fn no_functions(_: &mut Context<'static>) {}

const CASES: &[Case] = &[
    Case {
        label: "simple_arithmetic",
        src: "1 + 2 * 3 - 4 / 2",
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "comparison",
        src: "10 > 5 && 3 < 7 || 1 == 1",
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "conditional",
        src: "x > 10 ? x * 2 : x + 5",
        schema: &[("x", ValType::Int)],
        build: one_int_x,
        register: no_functions,
    },
    Case {
        label: "nested_expression",
        src: "((a + b) * (c - d)) / ((e + f) - (g * h))",
        schema: &[
            ("a", ValType::Int),
            ("b", ValType::Int),
            ("c", ValType::Int),
            ("d", ValType::Int),
            ("e", ValType::Int),
            ("f", ValType::Int),
            ("g", ValType::Int),
            ("h", ValType::Int),
        ],
        build: nested_operands,
        register: no_functions,
    },
    Case {
        label: "variable_access/hashmap",
        src: "apple",
        schema: &[("apple", ValType::Bool)],
        build: one_bool_apple,
        register: no_functions,
    },
    // cometkim benchmarks the same expression shape through a
    // `VariableResolver` instead of the context's map. A resolver is a way of
    // supplying an activation, not a feature of the expression, so on this side
    // it is the same one-column read.
    Case {
        label: "variable_access/resolver",
        src: "banana",
        schema: &[("banana", ValType::Bool)],
        build: one_bool_banana,
        register: no_functions,
    },
    Case {
        label: "member_access",
        src: "obj.nested.value + obj.other",
        schema: &[
            ("obj.nested.value", ValType::Int),
            ("obj.other", ValType::Int),
        ],
        build: member_fields,
        register: no_functions,
    },
    Case {
        label: "list_indexing",
        src: "list[0] + list[5] + list[9]",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
        register: no_functions,
    },
    Case {
        label: "list_filter",
        src: "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(x, x > 5)",
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "list_map",
        src: "[1, 2, 3, 4, 5].map(x, x * 2)",
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "all_comprehension",
        src: "[1, 2, 3, 4, 5].all(x, x > 0)",
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "exists_comprehension",
        src: "[1, 2, 3, 4, 5].exists(x, x == 3)",
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "map_list_scaling",
        src: "list.map(x, x * 2)",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
        register: no_functions,
    },
    Case {
        label: "filter_list_scaling",
        src: "list.filter(x, x % 2 == 0)",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
        register: no_functions,
    },
    Case {
        label: "comprehension_scaling",
        src: "items.filter(x, x % 2 == 0).map(x, x * 2)",
        schema: &[("items[]", ValType::Int)],
        build: one_int_list_items,
        register: no_functions,
    },
    Case {
        label: "string_operations",
        src: r#""hello world".startsWith("hello") && "hello world".endsWith("world") && "hello world".contains("o w")"#,
        schema: &[],
        build: no_columns,
        register: no_functions,
    },
    Case {
        label: "custom_function",
        src: "add(x, y) + multiply(a, b)",
        schema: &[
            ("x", ValType::Int),
            ("y", ValType::Int),
            ("a", ValType::Int),
            ("b", ValType::Int),
        ],
        build: custom_fn_operands,
        register: |ctx| {
            ctx.add_function("add", |a: i64, b: i64| a + b);
            ctx.add_function("multiply", |a: i64, b: i64| a * b);
        },
    },
    Case {
        label: "real_world_policy",
        src: r#"user.age >= 18 &&
                user.role in ["admin", "moderator"] &&
                request.method == "POST" &&
                request.path.startsWith("/api/") &&
                size(request.body) < 1000000"#,
        schema: &[
            ("user.age", ValType::Int),
            ("user.role", ValType::Str),
            ("request.method", ValType::Str),
            ("request.path", ValType::Str),
            ("request.body", ValType::Str),
        ],
        build: policy_request,
        register: no_functions,
    },
];

/// The MINIMUM over rounds. Other work on the box can only ever make a round
/// slower, so the fastest round is the one that ran with the least interference
/// and is the robust estimator of a tier's own cost; a median moves with how
/// loaded the machine happened to be, which is what makes a ratio unreadable.
fn best(v: Vec<f64>) -> f64 {
    v.into_iter().fold(f64::INFINITY, f64::min)
}

fn time_ns_per_row<T>(mut run: impl FnMut() -> T) -> f64 {
    let start = Instant::now();
    black_box(run());
    start.elapsed().as_nanos() as f64 / ROWS as f64
}

/// The tree-walker's answer for the whole batch, in the two shapes the batch
/// can return: a running total, or one value per row.
enum Oracle {
    Sum(Value),
    PerRow(Vec<Value>),
}

fn walker_sum(values: &[Value], label: &str) -> Value {
    let mut total = 0i64;
    for v in values {
        total += match v {
            Value::Bool(b) => *b as i64,
            Value::Int(v) => *v,
            Value::UInt(v) => *v as i64,
            other => panic!("{label}: a sum has nothing to do with {other:?}"),
        };
    }
    Value::Int(total)
}

/// What the compiled tiers did, or why they did not run.
struct Batched {
    clean: f64,
    jit: f64,
    /// The same two tiers read through `collect_raw`, for a per-row case only.
    /// A different CONTRACT, not a faster path to the same answer: the consumer
    /// takes the machine's own columns instead of a `Value` per row.
    raw: Option<(f64, f64)>,
    compiles: usize,
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

struct Row {
    label: &'static str,
    stock: f64,
    /// `Err` when the expression does not lower: the tree-walker answers it —
    /// through the library's own fallback, not a hand-written one — and there
    /// is no compiled tier to put beside a clean one.
    batch: Result<Batched, String>,
}

fn run_case(case: &Case) -> Row {
    let program =
        Program::compile(case.src).unwrap_or_else(|e| panic!("{}: parse error: {e:?}", case.label));
    let schema: Schema = case
        .schema
        .iter()
        .map(|(p, t)| (p.to_string(), *t))
        .collect();

    let cols = (case.build)(ROWS);
    let mut batch = Batch::new(ROWS);
    for (name, col) in &cols {
        batch = batch.column(*name, col.column_ref());
    }

    // The stock panel evaluates prebuilt activations: cometkim's benchmark also
    // holds its context fixed outside the timed region, and building one per row
    // inside it would time context construction, not evaluation.
    //
    // They come from the LIBRARY's `RowReader`, over the same `Batch` the
    // compiled tiers read, so the oracle cannot drift from what the machine is
    // fed — which is the whole reason the reconstruction moved out of here.
    let mut base = Context::default();
    (case.register)(&mut base);
    let reader = RowReader::new(&batch);
    let contexts: Vec<Context> = (0..ROWS).map(|r| reader.scope(&base, r)).collect();
    let walked: Vec<Value> = contexts
        .iter()
        .map(|ctx| {
            Value::resolve_value(program.expression(), ctx)
                .unwrap_or_else(|e| panic!("{}: stock execute: {e:?}", case.label))
        })
        .collect();

    let lowered = BatchProgram::from_program(&program, &schema);
    // A fresh driver, so `compiles` counts this case's loops and not a loop an
    // earlier case left compiled at the same program address.
    reset_persistent_state();
    let bound = match &lowered {
        Ok(bp) => {
            let per_row = bp.lowered().list_output.is_some();
            let b = if per_row {
                bp.bind_per_row(&batch)
            } else {
                bp.bind(&batch)
            };
            match b {
                Ok(b) => Ok((b, per_row)),
                Err(e) => Err(format!("cannot bind: {e}")),
            }
        }
        Err(e) => Err(format!("declines: {e}")),
    };
    let bound = match bound {
        Ok(b) => Some(b),
        Err(why) => {
            return Row {
                label: case.label,
                stock: best(time_stock(&program, &contexts, case.label)),
                batch: Err(why),
            }
        }
    };
    let (bound, per_row) = bound.unwrap();

    let expected = if per_row {
        Oracle::PerRow(walked.clone())
    } else {
        Oracle::Sum(walker_sum(&walked, case.label))
    };

    // Miscompile gate: all three tiers, and the tree-walker, agree.
    COMPILES.store(0, Ordering::Relaxed);
    for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
        let got = if per_row {
            Oracle::PerRow(
                bound
                    .collect_on(tier)
                    .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label)),
            )
        } else {
            Oracle::Sum(
                bound
                    .sum_on(tier)
                    .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label)),
            )
        };
        match (&expected, &got) {
            (Oracle::Sum(a), Oracle::Sum(b)) => {
                assert_eq!(a, b, "{}: {tier:?} vs stock", case.label)
            }
            (Oracle::PerRow(a), Oracle::PerRow(b)) => {
                assert_eq!(a, b, "{}: {tier:?} vs stock", case.label)
            }
            _ => unreachable!("the reduction is chosen once"),
        }
    }
    let compiles = COMPILES.load(Ordering::Relaxed);
    // Without this the `majit` column of a case that never compiled would be the
    // tracing interpreter's number under the compiled tier's heading. The driver
    // was reset just above, so every case has to compile on its own.
    assert!(
        compiles >= 1,
        "{}: the hot batch loop never compiled",
        case.label
    );

    let mut stock = Vec::with_capacity(ROUNDS);
    let mut clean = Vec::with_capacity(ROUNDS);
    let mut jit = Vec::with_capacity(ROUNDS);
    let mut raw_clean = Vec::with_capacity(ROUNDS);
    let mut raw_jit = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        stock.push(time_one_stock(&program, &contexts, case.label));
        // Nothing timed here may DISCARD a `Result`. A refused run returns
        // immediately, so a swallowed error is not a slow number, it is a fast
        // one — the same failure mode that puts a two-fold speedup on
        // cometkim's own `comprehension_scaling` rows, where the compiled side
        // returns `UndeclaredReference("@result")` and his `b.iter` never
        // unwraps it.
        let run = |tier| {
            if per_row {
                bound.collect_on(tier).map(|_| ())
            } else {
                bound.sum_on(tier).map(|_| ())
            }
            .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label))
        };
        clean.push(time_ns_per_row(|| run(Tier::Clean)));
        jit.push(time_ns_per_row(|| run(Tier::Jit)));
        // Only a per-row case boxes anything per row: a summed case already
        // returns one `Value` for the whole batch, so there is nothing for the
        // raw door to take away.
        if per_row {
            let raw = |tier| {
                bound
                    .collect_raw_on(tier, consume_raw)
                    .unwrap_or_else(|e| panic!("{}: {tier:?} raw: {e}", case.label))
            };
            raw_clean.push(time_ns_per_row(|| raw(Tier::Clean)));
            raw_jit.push(time_ns_per_row(|| raw(Tier::Jit)));
        }
    }

    Row {
        label: case.label,
        stock: best(stock),
        batch: Ok(Batched {
            clean: best(clean),
            jit: best(jit),
            raw: per_row.then(|| (best(raw_clean), best(raw_jit))),
            compiles,
        }),
    }
}

/// One timed pass of the tree-walker over every prebuilt activation.
///
/// The row is UNWRAPPED, not tested for `is_ok`: a raise leaves the walker
/// early, so tolerating one here would report a batch that failed as a batch
/// that was fast.
fn time_one_stock(program: &Program, contexts: &[Context], label: &str) -> f64 {
    time_ns_per_row(|| {
        let mut sink = 0usize;
        for ctx in contexts {
            let v = black_box(Value::resolve_value(program.expression(), black_box(ctx)))
                .unwrap_or_else(|e| panic!("{label}: stock execute: {e:?}"));
            sink ^= matches!(v, Value::Bool(true)) as usize;
        }
        sink
    })
}

fn time_stock(program: &Program, contexts: &[Context], label: &str) -> Vec<f64> {
    (0..ROUNDS)
        .map(|_| time_one_stock(program, contexts, label))
        .collect()
}

fn main() {
    println!("cometkim's benchmark expressions (cel-jit PR #233 benches/comparison.rs)");
    println!("{ROWS} rows per case; best of {ROUNDS}; every tier gated against the tree-walker.\n");
    println!(
        "{:<24} {:>12} {:>12} {:>12} {:>10} {:>13} {:>9}",
        "case",
        "stock ns/row",
        "clean ns/row",
        "majit ns/row",
        "majit/clean",
        "raw majit/clean",
        "compiles"
    );

    let mut declined = Vec::new();
    let mut lowered = 0;
    for case in CASES {
        let r = run_case(case);
        match &r.batch {
            Ok(b) => {
                lowered += 1;
                let raw = match b.raw {
                    Some((c, j)) => format!("{:.2}x rc={c:.2} rj={j:.2}", c / j),
                    None => "-".to_string(),
                };
                println!(
                    "{:<24} {:>12.1} {:>12.2} {:>12.2} {:>9.2}x {:>13} {:>9}",
                    r.label,
                    r.stock,
                    b.clean,
                    b.jit,
                    b.clean / b.jit,
                    raw,
                    b.compiles
                );
            }
            Err(why) => {
                // Still measured and still ANSWERED — by the tree-walker,
                // through the library's fallback. A row missing from the table
                // would read as an expression this crate cannot evaluate.
                println!(
                    "{:<24} {:>12.1} {:>12} {:>12} {:>10} {:>13} {:>9}",
                    r.label, r.stock, "-", "walker", "-", "-", "-"
                );
                declined.push((case.label, why.clone()));
            }
        }
    }

    println!(
        "\ncoverage: {lowered}/{} of cometkim's expressions lower to the compiled tier; \
         {}/{} are answered",
        CASES.len(),
        CASES.len(),
        CASES.len()
    );
    for (label, why) in &declined {
        println!("  {label:<24} {why}");
    }
    println!(
        "\nOnly majit/clean isolates compilation: it is the same lowered bytecode over the\n\
         same columns, run by the plain Rust VM and by the compiled trace. The ratio to\n\
         stock also contains the data-model change (slot resolution, no boxed values), and\n\
         stock is a per-activation evaluator being asked to do a batch, which is not the\n\
         workload it is built for."
    );
    println!(
        "\nThe last five cases return a LIST per row, so `collect` builds a `Vec<Value>`\n\
         and an `Arc` for every row — on the compiled tier that box costs several times\n\
         the evaluation it wraps, and both tiers pay it, which is what holds majit/clean\n\
         near 2-3x there. `raw majit/clean` is the same two tiers read through\n\
         `collect_raw`, where a columnar consumer takes the machine's own buffers and\n\
         nothing is boxed. It is not comparable to the stock column, which necessarily\n\
         produces values."
    );
}
