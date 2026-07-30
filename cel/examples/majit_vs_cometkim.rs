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
//!   already support.
//! * **Throughput.** For the ones that lower: stock `Program::execute` per row,
//!   the plain Rust bytecode VM, and the compiled trace, over one batch of
//!   identical data. Only `majit / clean VM` isolates compilation; the ratio to
//!   stock also contains the data-model change (slot resolution, no boxing).
//!
//! ⚠️ This is NOT a comparison against cometkim's own numbers. His regime is one
//! `CompiledProgram::execute(&ctx)` per criterion iteration over a FIXED
//! context — per-call latency with a persistent AOT function. This one is batch
//! throughput over varying rows. Putting his ns/call beside a ns/row would
//! invite a ratio that isolates nothing, so his numbers are not carried here;
//! his benchmark contributes its EXPRESSIONS, which is what a yardstick is for.
//!
//! Row-vs-element note: `map_list_scaling` and `filter_list_scaling` are the two
//! cases whose cost scales with a list's length, and this file measures them at
//! one length. `./bench.sh majit_nested_bench` is the ladder for that shape.
//!
//! RELEASE ONLY. Run: `./bench.sh majit_vs_cometkim`.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchError, BatchProgram, ColumnRef, Tier};
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

    /// Row `r` as the tree-walker sees it.
    fn value(&self, r: usize) -> Value {
        match self {
            Col::Int(c) => Value::Int(c[r]),
            Col::Bool(c) => Value::Bool(c[r]),
            Col::Str(c) => Value::from(c[r].as_str()),
            Col::IntList { lens, elems } => {
                let len = lens[r] as usize;
                let off = r * len;
                Value::from(elems[off..off + len].to_vec())
            }
        }
    }
}

/// A dotted path tree, so `obj.nested.value` reaches the walker as the nested
/// maps cometkim's context builds by hand.
#[derive(Default)]
struct Node {
    leaf: Option<Value>,
    kids: HashMap<String, Node>,
}

impl Node {
    fn insert(&mut self, path: &str, value: Value) {
        match path.split_once('.') {
            None => self.kids.entry(path.to_string()).or_default().leaf = Some(value),
            Some((head, rest)) => self
                .kids
                .entry(head.to_string())
                .or_default()
                .insert(rest, value),
        }
    }

    fn value(&self) -> Value {
        match &self.leaf {
            Some(v) => v.clone(),
            None => Value::from(
                self.kids
                    .iter()
                    .map(|(k, n)| (k.clone(), n.value()))
                    .collect::<HashMap<String, Value>>(),
            ),
        }
    }
}

/// The activation for row `r`: every column's row value, nested under its path.
fn row_context(cols: &[(&'static str, Col)], r: usize) -> Context<'static> {
    let mut root = Node::default();
    for (name, col) in cols {
        root.insert(name, col.value(r));
    }
    let mut ctx = Context::default();
    for (name, node) in &root.kids {
        ctx.add_variable_from_value(name.clone(), node.value());
    }
    ctx
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
}

const CASES: &[Case] = &[
    Case {
        label: "simple_arithmetic",
        src: "1 + 2 * 3 - 4 / 2",
        schema: &[],
        build: no_columns,
    },
    Case {
        label: "comparison",
        src: "10 > 5 && 3 < 7 || 1 == 1",
        schema: &[],
        build: no_columns,
    },
    Case {
        label: "conditional",
        src: "x > 10 ? x * 2 : x + 5",
        schema: &[("x", ValType::Int)],
        build: one_int_x,
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
    },
    Case {
        label: "variable_access/hashmap",
        src: "apple",
        schema: &[("apple", ValType::Bool)],
        build: one_bool_apple,
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
    },
    Case {
        label: "member_access",
        src: "obj.nested.value + obj.other",
        schema: &[
            ("obj.nested.value", ValType::Int),
            ("obj.other", ValType::Int),
        ],
        build: member_fields,
    },
    Case {
        label: "list_indexing",
        src: "list[0] + list[5] + list[9]",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
    },
    Case {
        label: "list_filter",
        src: "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(x, x > 5)",
        schema: &[],
        build: no_columns,
    },
    Case {
        label: "list_map",
        src: "[1, 2, 3, 4, 5].map(x, x * 2)",
        schema: &[],
        build: no_columns,
    },
    Case {
        label: "all_comprehension",
        src: "[1, 2, 3, 4, 5].all(x, x > 0)",
        schema: &[],
        build: no_columns,
    },
    Case {
        label: "exists_comprehension",
        src: "[1, 2, 3, 4, 5].exists(x, x == 3)",
        schema: &[],
        build: no_columns,
    },
    Case {
        label: "map_list_scaling",
        src: "list.map(x, x * 2)",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
    },
    Case {
        label: "filter_list_scaling",
        src: "list.filter(x, x % 2 == 0)",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
    },
    Case {
        label: "comprehension_scaling",
        src: "items.filter(x, x % 2 == 0).map(x, x * 2)",
        schema: &[("items[]", ValType::Int)],
        build: one_int_list_items,
    },
    Case {
        label: "string_operations",
        src: r#""hello world".startsWith("hello") && "hello world".endsWith("world") && "hello world".contains("o w")"#,
        schema: &[],
        build: no_columns,
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
    },
];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
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

struct Row {
    label: &'static str,
    stock: f64,
    clean: f64,
    jit: f64,
    compiles: usize,
}

fn run_case(case: &Case) -> Result<Row, String> {
    let program = Program::compile(case.src).map_err(|e| format!("parse error: {e:?}"))?;
    let schema: Schema = case
        .schema
        .iter()
        .map(|(p, t)| (p.to_string(), *t))
        .collect();
    let lowered =
        BatchProgram::from_program(&program, &schema).map_err(|e| format!("declines: {e}"))?;

    let cols = (case.build)(ROWS);
    let mut batch = Batch::new(ROWS);
    for (name, col) in &cols {
        batch = batch.column(*name, col.column_ref());
    }

    // The stock panel evaluates prebuilt activations: cometkim's benchmark also
    // holds its context fixed outside the timed region, and building one per row
    // inside it would time context construction, not evaluation.
    let contexts: Vec<Context> = (0..ROWS).map(|r| row_context(&cols, r)).collect();
    let walked: Vec<Value> = contexts
        .iter()
        .map(|ctx| {
            program
                .execute(ctx)
                .unwrap_or_else(|e| panic!("{}: stock execute: {e:?}", case.label))
        })
        .collect();

    let per_row = lowered.lowered().list_output.is_some();
    let expected = if per_row {
        Oracle::PerRow(walked.clone())
    } else {
        Oracle::Sum(walker_sum(&walked, case.label))
    };

    // A fresh driver, so `compiles` counts this case's loops and not a loop an
    // earlier case left compiled at the same program address.
    reset_persistent_state();
    let bound = match if per_row {
        lowered.bind_per_row(&batch)
    } else {
        lowered.bind(&batch)
    } {
        Ok(b) => b,
        Err(BatchError::Lower(e)) => return Err(format!("declines: {e}")),
        Err(e) => return Err(format!("cannot bind: {e}")),
    };

    // Miscompile gate: all three tiers, and the tree-walker, agree.
    COMPILES.store(0, Ordering::Relaxed);
    for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
        let got = if per_row {
            bound
                .collect_on(tier)
                .map(Oracle::PerRow)
                .map_err(|e| format!("{}: {tier:?}: {e}", case.label))?
        } else {
            bound
                .sum_on(tier)
                .map(Oracle::Sum)
                .map_err(|e| format!("{}: {tier:?}: {e}", case.label))?
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

    let mut stock = Vec::with_capacity(ROUNDS);
    let mut clean = Vec::with_capacity(ROUNDS);
    let mut jit = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        stock.push(time_ns_per_row(|| {
            let mut sink = 0usize;
            for ctx in &contexts {
                sink ^= black_box(program.execute(black_box(ctx)).is_ok()) as usize;
            }
            sink
        }));
        let run = |tier| {
            if per_row {
                bound.collect_on(tier).map(|_| ())
            } else {
                bound.sum_on(tier).map(|_| ())
            }
        };
        clean.push(time_ns_per_row(|| run(Tier::Clean)));
        jit.push(time_ns_per_row(|| run(Tier::Jit)));
    }

    Ok(Row {
        label: case.label,
        stock: median(stock),
        clean: median(clean),
        jit: median(jit),
        compiles,
    })
}

fn main() {
    println!("cometkim's benchmark expressions (cel-jit PR #233 benches/comparison.rs)");
    println!(
        "{ROWS} rows per case; median of {ROUNDS}; every tier gated against the tree-walker.\n"
    );
    println!(
        "{:<24} {:>12} {:>12} {:>12} {:>10} {:>9}",
        "case", "stock ns/row", "clean ns/row", "majit ns/row", "majit/clean", "compiles"
    );

    let mut declined = Vec::new();
    let mut lowered = 0;
    for case in CASES {
        match run_case(case) {
            Ok(r) => {
                lowered += 1;
                println!(
                    "{:<24} {:>12.1} {:>12.2} {:>12.2} {:>9.2}x {:>9}",
                    r.label,
                    r.stock,
                    r.clean,
                    r.jit,
                    r.clean / r.jit,
                    r.compiles
                );
            }
            Err(why) => declined.push((case.label, why)),
        }
    }

    println!(
        "\ncoverage: {lowered}/{} of cometkim's expressions lower",
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
}
