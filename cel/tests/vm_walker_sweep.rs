//! A generated differential gate: the bytecode VM against the tree walker.
//!
//! `tests/oracle.rs` pins a few hundred hand-written rows to a *frozen answer*.
//! This gate pins nothing: it generates tens of thousands of expressions and
//! asserts only that the two evaluators produce the same thing. That is the
//! precondition for making `Program::execute` the VM -- the switch must not
//! change any answer -- and it is a property the frozen corpus cannot carry,
//! because a corpus can only cover what someone thought to write down.
//!
//! The universe is deliberately **parser-reachable CEL**, because that is the
//! only universe a `Program` can contain. Hand-built ASTs can reach shapes the
//! parser cannot (an `Expr::Unspecified`, an operator at the wrong arity, a
//! comprehension whose `loop_cond` reads the iteration variable before it is
//! bound), and the two evaluators do *not* agree on all of them. Those are out
//! of scope here and would need their own decision before `cel::vm` is offered
//! as a general AST evaluator.
//!
//! Two failures are reported separately because they mean different things:
//!
//! * a **decline** is a coverage hole -- the compiler could not express the
//!   expression at all. `vm::eval` would turn that into an `InternalError`,
//!   which is indistinguishable from a genuine runtime fault, so this gate
//!   compiles explicitly and names the decline for what it is;
//! * a **divergence** is two evaluators disagreeing about a program both can run.
//!
//! Answers are compared by their `Debug` rendering rather than by `PartialEq`,
//! because `Value`'s equality is CEL equality: it crosses numeric types and
//! makes `NaN` unequal to itself. Neither is what "the switch changed nothing"
//! means.

use std::sync::Arc;

use cel::objects::{Key, OptionalValue};
use cel::parser::Parser;
use cel::{Context, ExecutionError, Value};

/// The fewest expressions the generator may compare before the gate refuses.
///
/// The generator yields a little over 23 000 today. The floor exists so that a
/// mistake that empties an operand table -- or a parser change that rejects most
/// of what is generated -- fails loudly instead of passing a sweep of nothing.
/// It is set well below the current count so that ordinary additions to the
/// corpus do not have to move it, and far above zero so that a collapse cannot
/// hide.
const FLOOR: usize = 20_000;

/// Selects how much of the generated universe runs.
///
/// Unset means the whole thing. The variable exists so the sweep can be *cut
/// down* on a slow machine, never so that it can be skipped: an unrecognised
/// value is a hard failure rather than a silent full run, because an
/// environment variable that quietly does nothing is exactly how a gate comes to
/// report success for work it never did.
const SELECTOR: &str = "CEL_VM_SWEEP";

fn ctx() -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("i", Value::Int(42));
    ctx.add_variable_from_value("neg", Value::Int(-7));
    ctx.add_variable_from_value("z", Value::Int(0));
    ctx.add_variable_from_value("u", Value::UInt(7));
    ctx.add_variable_from_value("d", Value::Float(1.5));
    ctx.add_variable_from_value("b", Value::Bool(true));
    ctx.add_variable_from_value("f", Value::Bool(false));
    ctx.add_variable_from_value("s", Value::String(Arc::new("hello".to_string())));
    ctx.add_variable_from_value("by", Value::Bytes(Arc::new(b"hi".to_vec())));
    ctx.add_variable_from_value("nil", Value::Null);
    ctx.add_variable_from_value(
        "xs",
        Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
    );
    ctx.add_variable_from_value("empty", Value::list(Vec::<Value>::new()));
    let mut m = std::collections::HashMap::new();
    m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
    m.insert(Key::String(Arc::new("b".to_string())), Value::Int(2));
    ctx.add_variable_from_value("m", Value::Map(cel::objects::Map::object(Arc::new(m))));
    ctx.add_variable_from_value(
        "opt_some",
        Value::Opaque(Arc::new(OptionalValue::of(Value::Int(9)))),
    );
    ctx.add_variable_from_value("opt_none", Value::Opaque(Arc::new(OptionalValue::none())));
    ctx.add_function("double_it", |v: i64| v * 2);
    ctx.add_function("boom", || -> Result<i64, ExecutionError> {
        Err(ExecutionError::function_error("boom", "deliberate"))
    });
    ctx
}

fn show(r: &Result<Value, ExecutionError>) -> String {
    match r {
        Ok(v) => format!("{v:?}"),
        Err(e) => format!("ERR({e:?})"),
    }
}

/// The operand alphabet the binary and unary grid is built from.
///
/// Every value family the language has, each in a literal form and a bound-
/// variable form, plus the shapes that are not values at all: an undeclared
/// name, a division by zero, a host function that fails. Those last three are
/// the interesting half -- an evaluator's error *propagation* is where the two
/// implementations have the most room to differ.
const OPERANDS: &[&str] = &[
    "1",
    "-3",
    "0",
    "2u",
    "1.5",
    "true",
    "false",
    "null",
    "\"ab\"",
    "b\"hi\"",
    "[1, 2]",
    "[]",
    "{\"a\": 1}",
    "{}",
    "i",
    "neg",
    "z",
    "u",
    "d",
    "b",
    "f",
    "s",
    "by",
    "nil",
    "xs",
    "empty",
    "m",
    "opt_some",
    "opt_none",
    "undeclared_name",
    "(1 / 0)",
    "boom()",
    "double_it(3)",
    "duration(\"1s\")",
    "timestamp(\"2020-01-01T00:00:00Z\")",
    "size(xs)",
    "dyn(1)",
    "xs.all(x, x > 0)",
    "xs.map(x, x * 2)",
    "optional.of(1)",
    "optional.none()",
];

const BINOPS: &[&str] = &[
    "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||", "in",
];

const UNARY: &[&str] = &["-", "!"];

/// Shapes the operand grid cannot reach: macros, indexing, selection, receiver
/// calls, nesting, and the constructs the two evaluators reach through
/// different code (comprehension order, namespace shadowing, struct literals).
const SHAPES: &[&str] = &[
    // comprehension ordering and the accumulator fast path
    "xs.map(x, x + 1)",
    "xs.filter(x, x > 1)",
    "xs.map(x, x > 1, x * 10)",
    "xs.all(x, x > 0)",
    "xs.exists(x, x > 2)",
    "xs.exists_one(x, x == 2)",
    "empty.all(x, x > 0)",
    "empty.exists(x, x > 0)",
    "empty.map(x, x)",
    "xs.all(x, undeclared_name)",
    "xs.map(x, undeclared_name)",
    "xs.map(x, 1 / 0)",
    "xs.filter(x, 1 / 0 == 1)",
    "xs.map(x, xs.map(y, x * y))",
    "xs.all(x, xs.all(y, x <= y))",
    "m.all(k, k == \"a\" || k == \"b\")",
    "m.map(k, k)",
    "undeclared_name.all(x, x > 0)",
    "nil.all(x, x > 0)",
    "i.all(x, x > 0)",
    // namespace vs receiver
    "xs.all(optional, optional.of(1) == optional.of(1))",
    "xs.map(optional, optional)",
    "xs.map(size, size)",
    "xs.map(dyn, dyn)",
    "math.max(1, 2)",
    "optional.of(1)",
    "optional.none()",
    "s.startsWith(\"h\")",
    "s.matches(\"h.*\")",
    "\"abc\".contains(\"b\")",
    "undeclared_ns.thing(1)",
    "i.undefinedMethod(1)",
    // selection / has
    "m.a",
    "m.zzz",
    "has(m.a)",
    "has(m.zzz)",
    "has(i.a)",
    "has(nil.a)",
    "has(xs.a)",
    "nil.a",
    "i.a",
    // indexing
    "xs[0]",
    "xs[9]",
    "xs[-1]",
    "m[\"a\"]",
    "m[\"zzz\"]",
    "m[1]",
    "nil[0]",
    "i[0]",
    "s[0]",
    // ternary / short circuit corner cases
    "true ? 1 : (1/0)",
    "false ? (1/0) : 1",
    "(1/0 == 1) ? 1 : 2",
    "i ? 1 : 2",
    "(1/0 == 1) && false",
    "false && (1/0 == 1)",
    "(1/0 == 1) && true",
    "(1/0 == 1) || true",
    "(1/0 == 1) || false",
    "undeclared_name && false",
    "undeclared_name || true",
    "undeclared_name && undeclared_name2",
    "1 && true",
    "true && 1",
    "1 || false",
    "((1/0 == 1) && true) && false",
    "((undeclared_name && true) && false)",
    "(undeclared_name || false) || true",
    // aggregate literals with errors inside
    "[1, undeclared_name]",
    "{\"a\": undeclared_name}",
    "{undeclared_name: 1}",
    "{[1]: 1}",
    "{1: 1, 1: 2}",
    "[1, 2] + [3]",
    "size([1,2,3])",
    // conversions and stdlib
    "int(\"12\")",
    "uint(3)",
    "double(3)",
    "string(1)",
    "bytes(\"ab\")",
    "int(\"nope\")",
    // `type()` returns an opaque whose *runtime type name* is `type` for every
    // answer, so a comparison key built from that name cannot tell `type(1)`
    // from `type("a")`. This gate compares `Debug`, which carries the whole
    // `Type`, and these rows are here to keep that true.
    "type(1)",
    "type(\"a\")",
    "type(1.5)",
    "type(true)",
    "type(null)",
    "type(xs)",
    "type(m)",
    "type(by)",
    "type(opt_some)",
    "type(type(1))",
    "type(1) == type(2)",
    "type(1) == type(\"a\")",
    "type(undeclared_name)",
    "type(1 / 0)",
    "type(1).zzz",
    "duration(\"1s\") + duration(\"2s\")",
    "timestamp(\"2020-01-01T00:00:00Z\") + duration(\"1s\")",
    "timestamp(\"2020-01-01T00:00:00Z\") - timestamp(\"2019-01-01T00:00:00Z\")",
    "-duration(\"1s\")",
    "duration(\"1s\").getSeconds()",
    "timestamp(\"2020-01-01T00:00:00Z\").getFullYear()",
    // struct literals, whose refusal is feature-shaped
    "cel.MyStruct { }",
    "cel.MyStruct { x: 1 }",
    "cel.MyStruct { x: 1 / 0 }",
    "cel.MyStruct { x: undeclared_name }",
    "cel.MyStruct { x: 1 }.x",
    "has(cel.MyStruct { x: 1 }.x)",
    // host functions
    "double_it(3)",
    "double_it(3, 4)",
    "double_it()",
    "boom()",
    "boom() && false",
    "undeclared_fn(1)",
];

/// Shapes that need `enable_optional_syntax`.
const OPT_SHAPES: &[&str] = &[
    "m[?\"a\"]",
    "m[?\"zzz\"]",
    "opt_some[?0]",
    "opt_none[?0]",
    "opt_none[1/0]",
    "opt_none[?1/0]",
    "m.?a",
    "m.?zzz",
    "opt_some.?a",
    "opt_none.?a",
    "nil.?a",
    "[?opt_some, 1]",
    "[?opt_none, 1]",
    "[?1, 2]",
    "{?\"k\": opt_some}",
    "{?\"k\": opt_none}",
    "{?\"k\": 1}",
    "opt_some.hasValue()",
    "opt_none.hasValue()",
    "opt_some.value()",
    "opt_none.value()",
    "opt_some.orValue(3)",
    "opt_none.orValue(3)",
];

/// How much of the generated universe to compare.
enum Extent {
    /// Every generated expression.
    Full,
    /// The hand-written shapes, plus every `stride`-th cell of the operand
    /// grid. For a machine on which the full sweep is too slow.
    Strided(usize),
}

impl Extent {
    /// Reads [`SELECTOR`], refusing anything it does not recognise.
    fn from_env() -> Extent {
        match std::env::var(SELECTOR) {
            Err(_) => Extent::Full,
            Ok(v) if v == "full" => Extent::Full,
            Ok(v) => match v.strip_prefix("stride:").and_then(|n| n.parse().ok()) {
                Some(n) if n >= 1 => Extent::Strided(n),
                _ => panic!(
                    "{SELECTOR}={v:?} is not a value this gate understands. \
                     Use `full` or `stride:<n>` with n >= 1, or unset it for `full`. \
                     Refusing rather than running a sweep the setting did not ask for."
                ),
            },
        }
    }
}

/// Every expression the gate compares, as `(source, needs_optional_syntax)`.
fn generate(extent: &Extent) -> Vec<(String, bool)> {
    let stride = match extent {
        Extent::Full => 1,
        Extent::Strided(n) => *n,
    };
    let mut sources: Vec<(String, bool)> = Vec::new();
    let mut cell = 0usize;
    for a in OPERANDS {
        for op in BINOPS {
            for b in OPERANDS {
                if cell % stride == 0 {
                    sources.push((format!("({a}) {op} ({b})"), false));
                }
                cell += 1;
            }
        }
        for op in UNARY {
            sources.push((format!("{op}({a})"), false));
        }
        sources.push((format!("({a}) == ({a})"), false));
    }
    for s in SHAPES {
        sources.push((s.to_string(), false));
    }
    for s in OPT_SHAPES {
        sources.push((s.to_string(), true));
    }
    sources
}

#[test]
fn the_vm_answers_what_the_walker_answers() {
    let extent = Extent::from_env();
    let sources = generate(&extent);
    let ctx = ctx();

    let mut compared = 0usize;
    let mut unparsed = 0usize;
    let mut declined: Vec<String> = Vec::new();
    let mut diverged: Vec<String> = Vec::new();

    for (src, optional) in &sources {
        let parser = if *optional {
            Parser::default().enable_optional_syntax(true)
        } else {
            Parser::default()
        };
        // Not every generated string is a program -- the grid crosses operand
        // tables blindly. A rejected parse is the parser's business, not this
        // gate's, and is only counted so that the floor below cannot be met by
        // expressions nothing ever evaluated.
        let Ok(expr) = parser.parse(src) else {
            unparsed += 1;
            continue;
        };
        let code = match cel::vm::compile(&expr) {
            Ok(code) => code,
            Err(e) => {
                declined.push(format!("  {src}\n      compiler: {e}"));
                continue;
            }
        };
        let walker = show(&Value::resolve_value(&expr, &ctx));
        let vm = show(&cel::vm::cel_eval_loop(&code, &ctx));
        compared += 1;
        if walker != vm {
            diverged.push(format!(
                "  {src}\n      walker: {walker}\n      vm:     {vm}"
            ));
        }
    }

    let floor = match extent {
        Extent::Full => FLOOR,
        // A strided run still has to be a sweep. Nothing here scales the floor
        // by the stride, because the point of the floor is that a run this
        // small is not evidence of anything.
        Extent::Strided(_) => FLOOR / 10,
    };
    assert!(
        compared >= floor,
        "the sweep compared {compared} expressions ({unparsed} unparsed, \
         {} declined) out of {} generated, below the floor of {floor}. \
         A sweep this small proves nothing, so it is a failure rather than a pass.",
        declined.len(),
        sources.len(),
    );

    // Every failing row is reported, up to a cap, rather than stopping at the
    // first: which expressions share a failure is most of the diagnosis.
    let report = |rows: &[String], of: usize, kind: &str| {
        let shown = rows.len().min(20);
        format!(
            "{} of {of} expressions {kind}:\n{}{}",
            rows.len(),
            rows[..shown].join("\n"),
            if rows.len() > shown {
                format!("\n  ... and {} more", rows.len() - shown)
            } else {
                String::new()
            }
        )
    };
    assert!(
        declined.is_empty(),
        "{}",
        report(
            &declined,
            compared + declined.len(),
            "the compiler could not express -- a coverage hole, not a runtime fault"
        )
    );
    assert!(
        diverged.is_empty(),
        "{}",
        report(
            &diverged,
            compared,
            "were answered differently by the two evaluators"
        )
    );

    println!("{compared} expressions agree ({unparsed} unparsed)");
}

/// Agreement on a comprehension large enough that the accumulator's growth
/// strategy shows: the two evaluators build the result list by different means
/// and must still hand back the same list.
#[test]
fn a_large_comprehension_agrees() {
    for n in [1000usize, 16_000] {
        let mut ctx = Context::default();
        ctx.add_variable_from_value(
            "src",
            Value::list((0..n as i64).map(Value::Int).collect::<Vec<_>>()),
        );
        for src in [
            "src.map(x, x + 1)",
            "src.filter(x, x % 2 == 0)",
            "src.map(x, x % 3 == 0, x * 10)",
            "src.all(x, x >= 0)",
            "src.exists(x, x == 5)",
        ] {
            let expr = Parser::default().parse(src).expect("parses");
            let code = cel::vm::compile(&expr).expect("compiles");
            let walker = show(&Value::resolve_value(&expr, &ctx));
            let vm = show(&cel::vm::cel_eval_loop(&code, &ctx));
            assert_eq!(walker, vm, "`{src}` at n={n}");
        }
    }
}
