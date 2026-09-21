//! Generated differential: the bytecode VM against the tree walker.
//!
//! The walker (`Value::resolve_value`) is the one implementation of CEL
//! semantics. `Program::execute` — under `--features jit-dynasm`, the portal
//! plus residual interpreter — must answer the same thing, including the
//! error kind. One run prints every disagreement (`program | vm | walker`)
//! so the census is visible before any fix.
//!
//! Deterministic: no RNG, no clock. The generator is a cross product of
//! operators/forms with operand shapes, kept to a few thousand programs.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cel::common::types::TypeValue;
use cel::objects::{Key, OptionalValue};
use cel::parser::Parser;
use cel::{Context, ExecutionError, Program, Value};

/// Named programs whose walker answer is believed not to match the CEL spec.
/// Empty until a row is classified that way; a skip needs a one-line reason.
const KNOWN_WALKER_QUESTIONS: &[(&str, &str)] = &[];

/// A collapse of the generator, or a parser that rejects most of it, must
/// not pass a gate of nothing.
const FLOOR: usize = 1_500;

fn make_ctx() -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("i", Value::Int(42));
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
    ctx.add_variable_from_value(
        "mixed",
        Value::list(vec![
            Value::Int(1),
            Value::Bool(true),
            Value::String(Arc::new("a".to_string())),
            Value::UInt(1),
        ]),
    );
    let mut m = HashMap::new();
    m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
    m.insert(Key::String(Arc::new("b".to_string())), Value::Int(2));
    ctx.add_variable_from_value("m", Value::Map(cel::objects::Map::object(Arc::new(m))));
    let mut cross = HashMap::new();
    cross.insert(Key::Int(1), Value::String(Arc::new("a".to_string())));
    cross.insert(Key::Uint(1), Value::String(Arc::new("b".to_string())));
    ctx.add_variable_from_value(
        "cross",
        Value::Map(cel::objects::Map::object(Arc::new(cross))),
    );
    ctx.add_variable_from_value(
        "opt_some",
        Value::Opaque(Arc::new(OptionalValue::of(Value::Int(9)))),
    );
    ctx.add_variable_from_value("opt_none", Value::Opaque(Arc::new(OptionalValue::none())));
    ctx
}

fn has_interned(v: &Value) -> bool {
    match v {
        Value::Interned(_) => true,
        Value::List(list) => list.iter().any(|e| has_interned(&e)),
        Value::Map(map) => map.iter().any(|(_, e)| has_interned(e.as_ref())),
        Value::Opaque(o) => match o.downcast_ref::<OptionalValue>() {
            Some(opt) => opt.value().is_some_and(has_interned),
            None => false,
        },
        _ => false,
    }
}

fn show(r: &Result<Value, ExecutionError>) -> String {
    match r {
        Ok(v) => format!("OK({})", render(&v.unpack())),
        Err(e) => format!("ERR({})", err_kind(e)),
    }
}

fn err_kind(err: &ExecutionError) -> String {
    #[allow(deprecated)]
    match err {
        ExecutionError::InvalidArgumentCount { expected, actual } => {
            format!("InvalidArgumentCount:{expected}:{actual}")
        }
        ExecutionError::UnsupportedTargetType { .. } => "UnsupportedTargetType".into(),
        ExecutionError::NotSupportedAsMethod { method, .. } => {
            format!("NotSupportedAsMethod:{method}")
        }
        ExecutionError::UnsupportedKeyType(_) => "UnsupportedKeyType".into(),
        ExecutionError::UnexpectedType { got, want } => format!("UnexpectedType:{got}:{want}"),
        ExecutionError::NoSuchKey(k) => format!("NoSuchKey:{k}"),
        ExecutionError::NoSuchOverload => "NoSuchOverload".into(),
        ExecutionError::UndeclaredReference(n) => format!("UndeclaredReference:{n}"),
        ExecutionError::MissingArgumentOrTarget => "MissingArgumentOrTarget".into(),
        ExecutionError::ValuesNotComparable(_, _) => "ValuesNotComparable".into(),
        ExecutionError::UnsupportedUnaryOperator(op, _) => {
            format!("UnsupportedUnaryOperator:{op}")
        }
        ExecutionError::UnsupportedBinaryOperator(op, _, _) => {
            format!("UnsupportedBinaryOperator:{op}")
        }
        ExecutionError::UnsupportedMapIndex(_) => "UnsupportedMapIndex".into(),
        ExecutionError::UnsupportedListIndex(_) => "UnsupportedListIndex".into(),
        ExecutionError::UnsupportedIndex(_, _) => "UnsupportedIndex".into(),
        ExecutionError::UnsupportedFunctionCallIdentifierType(_) => {
            "UnsupportedFunctionCallIdentifierType".into()
        }
        ExecutionError::UnsupportedFieldsConstruction(_) => {
            "UnsupportedFieldsConstruction".into()
        }
        ExecutionError::FunctionError { function, .. } => format!("FunctionError:{function}"),
        ExecutionError::DivisionByZero(_) => "DivisionByZero".into(),
        ExecutionError::RemainderByZero(_) => "RemainderByZero".into(),
        ExecutionError::Overflow(op, _, _) => format!("Overflow:{op}"),
        ExecutionError::IndexOutOfBounds(_) => "IndexOutOfBounds".into(),
        ExecutionError::InternalError(_) => "InternalError".into(),
        other => format!("Other:{other:?}"),
    }
}

fn render(value: &Value) -> String {
    let unpacked = value.unpack();
    match &unpacked {
        Value::Int(i) => format!("int({i})"),
        Value::UInt(u) => format!("uint({u})"),
        Value::Float(f) => {
            if f.is_nan() {
                "double(nan)".into()
            } else if *f == f64::INFINITY {
                "double(inf)".into()
            } else if *f == f64::NEG_INFINITY {
                "double(-inf)".into()
            } else {
                format!("double({f:?})")
            }
        }
        Value::Bool(b) => format!("bool({b})"),
        Value::String(s) => format!("string({s:?})"),
        Value::Bytes(b) => format!("bytes({b:?})"),
        Value::Null => "null".into(),
        Value::List(list) => {
            let items: Vec<String> = list.iter().map(|v| render(&v)).collect();
            format!("list[{}]", items.join(", "))
        }
        Value::Map(map) => {
            let mut entries: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", render_key(k), render(&v)))
                .collect();
            entries.sort();
            format!("map{{{}}}", entries.join(", "))
        }
        Value::Opaque(o) => match o.downcast_ref::<OptionalValue>() {
            Some(opt) => match opt.value() {
                Some(v) => format!("optional({})", render(v)),
                None => "optional.none".into(),
            },
            None => match o.downcast_ref::<TypeValue>() {
                Some(t) => format!("type({})", t.name()),
                None => format!("opaque({})", o.runtime_type_name()),
            },
        },
        #[cfg(feature = "chrono")]
        Value::Duration(d) => format!("duration({d:?})"),
        #[cfg(feature = "chrono")]
        Value::Timestamp(t) => format!("timestamp({t:?})"),
        Value::Interned(_) => render(&unpacked.unpack()),
        #[cfg(feature = "structs")]
        Value::Struct(s) => format!("struct({})", s.name()),
    }
}

fn render_key(key: &Key) -> String {
    match key {
        Key::Int(i) => format!("int({i})"),
        Key::Uint(u) => format!("uint({u})"),
        Key::Bool(b) => format!("bool({b})"),
        Key::String(s) => format!("string({s:?})"),
    }
}

/// Map iteration order is not specified. A comprehension over a map may
/// yield the same keys in HashMap order (walker) or source order (interned
/// VM); those are not different answers.
fn map_comprehension(src: &str) -> bool {
    let s = src.trim();
    let over_map = s.starts_with('{')
        || s.starts_with("m.")
        || s.starts_with("cross.");
    over_map
        && (s.contains(".map(")
            || s.contains(".filter(")
            || s.contains(".all(")
            || s.contains(".exists(")
            || s.contains(".exists_one("))
}

fn list_multiset_eq(a: &Value, b: &Value) -> bool {
    let a = a.unpack();
    let b = b.unpack();
    match (&a, &b) {
        (Value::List(x), Value::List(y)) => {
            let mut xs: Vec<String> = x.iter().map(|v| render(&v)).collect();
            let mut ys: Vec<String> = y.iter().map(|v| render(&v)).collect();
            xs.sort();
            ys.sort();
            xs == ys
        }
        _ => false,
    }
}

fn answers_agree(
    src: &str,
    walker: &Result<Value, ExecutionError>,
    vm: &Result<Value, ExecutionError>,
) -> bool {
    if show(walker) == show(vm) {
        return true;
    }
    match (walker, vm) {
        (Ok(a), Ok(b)) if map_comprehension(src) => list_multiset_eq(a, b),
        _ => false,
    }
}

fn skip_reason(src: &str) -> Option<&'static str> {
    KNOWN_WALKER_QUESTIONS
        .iter()
        .find(|(p, _)| *p == src)
        .map(|(_, r)| *r)
}

/// Operand shapes the operators are applied to.
const LEFT: &[&str] = &[
    "1",
    "1000",
    "i",
    "7u",
    "1.5",
    "b",
    "\"ab\"",
    "s",
    "b\"hi\"",
    "null",
    "xs",
    "[1, 2]",
    "[i]",
    "[]",
    "m",
    "{\"a\": 1}",
    "{\"a\": i}",
    "{\"a\": 1, \"a\": 2}",
    "{1: 2, 1u: 3}",
    "(1 / 0)",
];

const RIGHT: &[&str] = &["1", "i", "7u", "b", "\"a\"", "(1 / 0)"];

const BINOPS: &[&str] = &[
    "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||", "in",
];

const UNARY: &[&str] = &["-", "!"];

/// Ranges whose element type (and error/short-circuit behaviour) changes
/// between iterations.
const CHANGING: &[&str] = &[
    "[1, true, \"a\", 1u]",
    "[1, true]",
    "[true, 1]",
    "[1, false]",
    "[false, 1]",
    "[1, 2]",
    "[2, 1]",
    "mixed",
    "xs",
];

const MACROS: &[&str] = &["map", "filter", "all", "exists", "exists_one"];

/// Bodies applied to the iteration variable. Includes operators, conversions,
/// an error on one iteration and a value on the next, and a short-circuit on
/// one iteration with fall-through on the next.
const BODIES: &[&str] = &[
    "e + 1",
    "e - 1",
    "e * 2",
    "e / 1",
    "e % 2",
    "-e",
    "!e",
    "e == 1",
    "e != true",
    "e < 2",
    "e > 0",
    "e && true",
    "e || false",
    "e && e",
    "e || e",
    "e ? 1 : 2",
    "e in [1, true, \"a\"]",
    "size(e)",
    "type(e)",
    "string(e)",
    "(10 / (e - 1) > 0) && e > 1",
    "e && (e == true)",
    "e || (e == 1)",
];

fn generate() -> Vec<(String, bool)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |src: String, optional: bool| {
        if seen.insert(src.clone()) {
            out.push((src, optional));
        }
    };

    for a in LEFT {
        for op in BINOPS {
            for b in RIGHT {
                push(format!("({a}) {op} ({b})"), false);
            }
        }
        for op in UNARY {
            push(format!("{op}({a})"), false);
        }
        push(format!("type({a})"), false);
        push(format!("string({a})"), false);
        push(format!("int({a})"), false);
        push(format!("uint({a})"), false);
        push(format!("double({a})"), false);
        push(format!("size({a})"), false);
        push(format!("({a}).size()"), false);
    }

    for a in LEFT {
        push(format!("({a}) && false"), false);
        push(format!("false && ({a})"), false);
        push(format!("({a}) || true"), false);
        push(format!("true || ({a})"), false);
        push(format!("b ? ({a}) : 2"), false);
        push(format!("f ? 1 : ({a})"), false);
        push(format!("({a}) ? 1 : 2"), false);
    }

    let containers = [
        "xs",
        "[1, 2, 3]",
        "[i, 2]",
        "m",
        "{\"a\": 1}",
        "{\"a\": i}",
        "{1: 2, 1u: 3}",
        "cross",
        "s",
        "[1, true, \"a\"]",
        "[]",
        "{}",
    ];
    let indices = [
        "0", "1", "-1", "9", "i", "1u", "1.5", "\"a\"", "true", "(1 / 0)", "no_such",
    ];
    for c in containers {
        for ix in indices {
            push(format!("({c})[{ix}]"), false);
        }
    }

    let recvs = [
        "m",
        "{\"a\": 1}",
        "{\"a\": i}",
        "{\"a\": 1, \"a\": 2}",
        "i",
        "null",
        "xs",
        "{\"a\": 1, \"b\": 2}",
        "cross",
    ];
    for r in recvs {
        for f in ["a", "b", "zzz"] {
            push(format!("({r}).{f}"), false);
            push(format!("has(({r}).{f})"), false);
        }
    }

    for recv in ["s", "\"hello\"", "\"ab\"", "\"\""] {
        for meth in ["startsWith", "endsWith", "contains"] {
            for n in ["\"h\"", "\"ab\"", "\"z\"", "\"\""] {
                push(format!("{recv}.{meth}({n})"), false);
            }
        }
    }

    for src in [
        "[1, 2, 3]",
        "[]",
        "[i]",
        "[1, i]",
        "[1, true, \"a\"]",
        "[1, true, \"a\", 1u]",
        "[[1], [2]]",
        "[[], [i]]",
        "[1, [2, 3]]",
        "{\"a\": 1}",
        "{}",
        "{\"a\": i}",
        "{\"a\": 1, \"a\": 2}",
        "{\"a\": 1, \"b\": 2}",
        "{1: \"a\", 1u: \"b\"}",
        "{1u: \"a\", 1: \"b\"}",
        "{i: 1, 42: 2}",
        "{true: 1, false: 2}",
        "{\"a\": 1, \"a\": i}",
        "{1: 2, 1: 3}",
        "[1, 2] + [3]",
        "[1, 2] + xs",
        "xs + [i]",
        "\"a\" + \"b\" + \"c\" + \"d\"",
        "b\"ab\" + b\"cd\"",
        "no_such",
        "m[\"zzz\"]",
        "xs[9]",
        "1 / 0",
        "1 % 0",
        "size(xs)",
        "type(1) == type(2)",
        "type(1) == type(\"a\")",
        "optional.of(1)",
        "optional.none()",
        "1 in {1u: 2}",
        "1u in {1: 2}",
        "1.0 in {1: 2}",
        "1 in {i: 1}",
        "15 in {i: 1}",
        "42 in {i: 1}",
        "\"a\" in {\"a\": 1, \"a\": 2}",
        "{1: \"a\", 1u: \"b\"}[1]",
        "{1: \"a\", 1u: \"b\"}[1u]",
        "{1u: \"a\", 1: \"b\"}[1]",
        "{i: 1, 42: 2}[42]",
        "{i: 1}[42]",
        "{i: 1}[42u]",
        "{i: 1}[42.0]",
        "cross[1]",
        "cross[1u]",
        "1 in cross",
        "1u in cross",
        "[{\"a\": 1}, {\"a\": 2}].map(e, e.a)",
        "[{\"a\": 1}, 1].map(e, e.a)",
        "[m, {\"a\": 9}].map(e, e.a)",
        "[xs, [1]].map(e, e[0])",
        "[xs, 1].map(e, e[0])",
        // 3-arg map is filter-then-transform. Two-variable forms are tried
        // below; this parser binds only one iteration variable.
        "xs.map(e, e > 1, e * 2)",
        "[1, true, 2].map(e, e > 1, e * 2)",
        "mixed.map(e, e == 1, e)",
        "m.map(k, k == \"a\", k)",
        "xs.map(i, v, i)",
        "m.map(k, v, k)",
        "[1, 2, 1, 2].map(e, (10 / (e - 1) > 0) && e > 1)",
        "[1, 2, 1, 3].map(e, (10 / (e - 1) < 0) || e < 3)",
        "[1, true, 1, true].map(e, e && (e == true))",
        "[1, false, 1, false].map(e, e || (e == 1))",
        "[1, 2, 3].filter(e, (10 / (e - 1) > 0) && e > 1)",
        "[1, 2, 3].all(e, ((10 / (e - 1) > 0) && e > 1) || e == 1)",
        "[1, true, false].filter(e, e && (e == true))",
        "[1, true].all(e, (e && (e == true)) || e == 1)",
        "xs.all(e, 1)",
        "xs.exists(e, 1)",
        "empty.all(e, e > 0)",
        "empty.exists(e, e > 0)",
        "empty.map(e, e)",
        "m.map(k, k)",
        "cross.map(k, k)",
        "{\"a\": 1, \"a\": 2}.map(k, k)",
        "{1: \"a\", 1u: \"b\"}.map(k, k)",
        // nested short-circuit / error absorption whose left shape changes
        "[1, true].map(e, (e && true) && (e == true))",
        "[true, 1].map(e, (e && true) && (e == true))",
        "[1, false].map(e, (e || false) || (e == 1))",
        "[false, 1].map(e, (e || false) || (e == 1))",
        "[1, 2].map(e, (10 / (e - 1) > 0) && false)",
        "[2, 1].map(e, (10 / (e - 1) > 0) && false)",
        "[1, 2].map(e, false && (10 / (e - 1) > 0))",
        "[2, 1].map(e, true || (10 / (e - 1) > 0))",
        "[false, 1].map(e, no_such && e)",
        "[true, false].map(e, no_such || e)",
        "[1, true].map(e, e ? (e && true) : (e || false))",
        "[true, 1].map(e, e ? 1 : (1 / 0))",
        "[1, true].map(e, e ? (1 / 0) : 2)",
        // list / map construction in a comprehension body
        "xs.map(e, [e, e])",
        "xs.map(e, {\"k\": e})",
        "xs.map(e, {e: e})",
        "[1, true].map(e, [e, e])",
        "[1, true].map(e, {\"k\": e})",
        "[1, \"a\"].map(e, {e: 1})",
        "mixed.map(e, [e])",
        "mixed.map(e, type(e))",
        // uint / double index, negative, nested
        "xs[1u]",
        "xs[0u]",
        "[1, 2, 3][1u]",
        "[1, 2, 3][1.0]",
        "[1, 2, 3][-1]",
        "[[1, 2], [3]][0][1]",
        "{\"a\": {\"b\": [10, 20]}}[\"a\"][\"b\"][1]",
        "m[\"a\"]",
        "m[\"zzz\"]",
        "cross[1.0]",
        "0u in {0: 1}",
        "0 in {0u: 1}",
        "true in {true: 1}",
        "false in {true: 1}",
        "\"\" in {\"\": 1}",
        "1u in xs",
        "1.0 in xs",
        "true in xs",
        "1 in [1u, 2]",
        "1u in [1, 2]",
        "1.0 in [1, 2]",
        "\"a\" in [1, \"a\"]",
        "1 in \"ab\"",
        "\"a\" in \"ab\"",
        "b\"hi\" in [b\"hi\"]",
        // overflow, unicode, bytes, conversions
        "9223372036854775807 + 1",
        "-9223372036854775808 - 1",
        "9223372036854775807 * 2",
        "18446744073709551615u + 1u",
        "size(\"héllo\")",
        "\"héllo\".startsWith(\"hé\")",
        "\"héllo\".contains(\"é\")",
        "\"héllo\".endsWith(\"o\")",
        "b\"hi\".size()",
        "size(by)",
        "int(\"12\")",
        "int(\"nope\")",
        "uint(-1)",
        "uint(\"7\")",
        "double(\"1.5\")",
        "string(1000)",
        "string(7u)",
        "dyn(1)",
        "dyn(xs)",
        "[1, 2] + [\"a\"]",
        "[1u] + [1]",
        "[] + []",
        "[i] + [i] + []",
        "true ? m.zzz : 1",
        "false ? m.zzz : 1",
        "true ? 1 : (1 / 0)",
        "false ? (1 / 0) : 1",
        "(1 / 0 == 1) ? 1 : 2",
        "(1 / 0 == 1) && false",
        "false && (1 / 0 == 1)",
        "(1 / 0 == 1) || true",
        "no_such && false",
        "no_such || true",
        "1 && true",
        "true && 1",
        "has({\"a\": 1, \"a\": 2}.a)",
        "has(m.zzz)",
        "has(i.a)",
        "has(nil.a)",
        "size({\"a\": 1, \"a\": 2})",
        "size({1: 1, 1u: 2})",
        "{\"a\": 1, \"a\": 2} == {\"a\": 2}",
        "{1: \"a\", 1u: \"b\"} == {1: \"b\"}",
        "{1: \"a\", 1u: \"b\"} == {1: \"a\", 1u: \"b\"}",
        "type([1]) == type(xs)",
        "type({\"a\": 1}) == type(m)",
        "[1, 2].map(e, \"p\" + string(e) + \"s\")",
        "[1, true, \"a\", 1u].map(e, e.contains(\"a\"))",
        "[1, true, \"a\", 1u].map(e, e.startsWith(\"a\"))",
        "[s, \"ab\"].map(e, e.startsWith(\"h\"))",
        "[s, 1].map(e, e.startsWith(\"h\"))",
        "[m, {\"a\": 9}].map(e, has(e.a))",
        "[m, 1].map(e, has(e.a))",
        "[xs, [9]].map(e, 1 in e)",
        "[xs, 1].map(e, 1 in e)",
        "[{\"a\": 1, \"a\": 2}, m].map(e, e.a)",
        "[{1: \"a\", 1u: \"b\"}, cross].map(e, e[1])",
        "[{1: \"a\", 1u: \"b\"}, cross].map(e, 1u in e)",
        "[{1: \"a\", 1u: \"b\"}, {1u: \"b\"}].map(e, e[1u])",
    ] {
        push(src.to_string(), false);
    }

    for range in CHANGING {
        for mac in MACROS {
            for body in BODIES {
                push(format!("{range}.{mac}(e, {body})"), false);
            }
            push(format!("{range}.map(e, e > 0, e)"), false);
            push(format!("{range}.map(e, e && true, e)"), false);
        }
    }

    for src in [
        "m.?a",
        "m.?zzz",
        "m.?b",
        "{\"a\": 1}.?a",
        "{\"a\": 1}.?z",
        "nil.?a",
        "opt_some.?a",
        "opt_none.?a",
        "m[?\"a\"]",
        "m[?\"zzz\"]",
        "xs[?0]",
        "xs[?9]",
        "[1, 2][?1]",
        "[1, 2][?5]",
        "{\"a\": 1}[?\"a\"]",
        "opt_some[?0]",
        "opt_none[?0]",
        "opt_none[?1/0]",
        "[?opt_some, 1]",
        "[?opt_none, 1]",
        "[?1, 2]",
        "{?\"k\": opt_some}",
        "{?\"k\": opt_none}",
        "{?\"k\": 1}",
        "optional.of(i)",
        "optional.ofNonZeroValue(0)",
        "opt_some.hasValue()",
        "opt_none.hasValue()",
        "opt_some.value()",
        "opt_none.value()",
        "opt_some.orValue(3)",
        "opt_none.orValue(3)",
        "[1, true].map(e, optional.of(e))",
    ] {
        push(src.to_string(), true);
    }

    out
}

struct Evaled {
    src: String,
    optional: bool,
}

fn compile_case(src: &str, optional: bool) -> Result<Compiled, String> {
    let parser = if optional {
        Parser::default().enable_optional_syntax(true)
    } else {
        Parser::default()
    };
    let expr = parser
        .parse(src)
        .map_err(|e| format!("parse: {e}"))?;
    if optional {
        let code = cel::vm::compile(&expr).map_err(|e| format!("compile: {e}"))?;
        Ok(Compiled::Optional { expr, code })
    } else {
        let program = Program::compile(src).map_err(|e| format!("program: {e}"))?;
        Ok(Compiled::Plain { expr, program })
    }
}

enum Compiled {
    Plain {
        expr: cel::parser::Expression,
        program: Program,
    },
    Optional {
        expr: cel::parser::Expression,
        code: cel::vm::CelCode,
    },
}

impl Compiled {
    fn walker(&self, ctx: &Context) -> Result<Value, ExecutionError> {
        let expr = match self {
            Compiled::Plain { expr, .. } | Compiled::Optional { expr, .. } => expr,
        };
        Value::resolve_value(expr, ctx)
    }

    fn execute(&self, ctx: &Context) -> Result<Value, ExecutionError> {
        match self {
            Compiled::Plain { program, .. } => program.execute(ctx),
            Compiled::Optional { code, .. } => cel::vm::cel_eval_loop(code, ctx),
        }
    }
}

#[test]
fn the_vm_answers_what_the_walker_answers() {
    let sources = generate();
    let mut compared = 0usize;
    let mut unparsed = 0usize;
    let mut declined = 0usize;
    let mut skipped = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    let mut cases: Vec<Evaled> = Vec::new();

    for (src, optional) in &sources {
        if skip_reason(src).is_some() {
            skipped += 1;
            continue;
        }
        cases.push(Evaled {
            src: src.clone(),
            optional: *optional,
        });
    }

    for case in &cases {
        let compiled = match compile_case(&case.src, case.optional) {
            Ok(c) => c,
            Err(e) if e.starts_with("parse:") => {
                unparsed += 1;
                continue;
            }
            Err(_) => {
                declined += 1;
                continue;
            }
        };

        let ctx = make_ctx();
        let walker = compiled.walker(&ctx);
        if let Ok(v) = &walker {
            assert!(
                !has_interned(v),
                "{}: walker result contains Interned",
                case.src
            );
        }
        let walker_s = show(&walker);
        let mut failed = false;
        for i in 0..3 {
            let vm = compiled.execute(&ctx);
            if let Ok(v) = &vm {
                assert!(
                    !has_interned(v),
                    "{}: vm result contains Interned",
                    case.src
                );
            }
            if !answers_agree(&case.src, &walker, &vm) {
                mismatches.push(format!(
                    "{} | {} | {walker_s}  (execute {i})",
                    case.src,
                    show(&vm)
                ));
                failed = true;
                break;
            }
        }
        if !failed {
            let fresh = make_ctx();
            let walker_fresh = compiled.walker(&fresh);
            let vm = compiled.execute(&fresh);
            if !answers_agree(&case.src, &walker_fresh, &vm) {
                mismatches.push(format!(
                    "{} | {} | {}  (fresh ctx)",
                    case.src,
                    show(&vm),
                    show(&walker_fresh)
                ));
            }
        }
        compared += 1;
    }

    println!(
        "generated={} compared={compared} unparsed={unparsed} declined={declined} skipped={skipped} mismatches={}",
        sources.len(),
        mismatches.len()
    );
    for row in &mismatches {
        println!("{row}");
    }

    assert!(
        compared >= FLOOR,
        "compared {compared} programs (unparsed={unparsed} declined={declined}) below floor {FLOOR}"
    );
    assert!(
        mismatches.is_empty(),
        "{} mismatches (program | vm | walker):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
