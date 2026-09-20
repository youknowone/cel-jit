//! Portal interned helpers must decline (or match the walker) rather than
//! invent an answer the single implementation would refuse.
//!
//! One row per diverging class found in the walker sweep: `in` on a string,
//! `in` on a map with a non-key needle, optional select on a plain miss,
//! optional list/map construction of a bound optional, and `in` of a uint
//! against an interned int list.

use std::sync::Arc;

use cel::objects::{Key, OptionalValue};
use cel::parser::Parser;
use cel::{Context, ExecutionError, Program, Value};

fn show(r: &Result<Value, ExecutionError>) -> String {
    match r {
        Ok(v) => format!("{v:?}"),
        Err(e) => format!("ERR({e:?})"),
    }
}

fn ctx() -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    ctx.add_variable_from_value(
        "xs",
        Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
    );
    let mut m = std::collections::HashMap::new();
    m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
    ctx.add_variable_from_value("m", Value::Map(cel::objects::Map::object(Arc::new(m))));
    ctx.add_variable_from_value(
        "opt_some",
        Value::Opaque(Arc::new(OptionalValue::of(Value::Int(9)))),
    );
    ctx.add_variable_from_value("opt_none", Value::Opaque(Arc::new(OptionalValue::none())));
    ctx
}

fn agree(src: &str, optional: bool) {
    let parser = if optional {
        Parser::default().enable_optional_syntax(true)
    } else {
        Parser::default()
    };
    let expr = parser.parse(src).unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let code = cel::vm::compile(&expr).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    let walker = show(&Value::resolve_value(&expr, &ctx()));
    let vm = show(&cel::vm::cel_eval_loop(&code, &ctx()));
    assert_eq!(walker, vm, "`{src}`");
}

#[test]
fn in_on_a_string_is_no_such_overload() {
    agree("(1) in (\"ab\")", false);
    agree("(\"ab\") in (\"ab\")", false);
}

#[test]
fn in_on_a_map_rejects_a_non_key_needle() {
    agree("(1.5) in ({})", false);
    agree("(null) in (m)", false);
}

#[test]
fn opt_select_on_a_plain_miss_is_no_such_key() {
    agree("m.?z", true);
    agree("m.?zzz", true);
}

#[test]
fn optional_list_and_map_of_a_bound_optional() {
    agree("[?opt_some, 1]", true);
    agree("[?opt_none, 1]", true);
    agree("{?\"k\": opt_some}", true);
    agree("{?\"k\": opt_none}", true);
}

#[test]
fn uint_in_an_int_list_matches_the_walker() {
    agree("(2u) in (xs)", false);
}

/// `And`/`Or` (left is not a slot) stay in the portal on interned bools.
#[test]
fn or_and_agree_on_bool_operands() {
    agree("true || false", false);
    agree("false || true", false);
    agree("true && true", false);
    agree("false && true", false);
    agree("15 == 15 || 15 == 16", false);
    agree("15 > 3 && true", false);
}

/// Non-bool left is not a short-circuit; residual merge matches the walker.
#[test]
fn or_declines_a_non_bool_left() {
    agree("(1) || true", false);
    agree("(1) || false", false);
    agree("true || (1)", false);
    agree("false && (1)", false);
    agree("(1) && false", false);
    agree("true && (1)", false);
}

#[test]
fn uint_in_an_int_list_literal_matches_the_walker() {
    agree("(2u) in ([1, 2])", false);
    agree("(2.0) in ([1, 2])", false);
}

/// Mixed / nested / empty constant lists intern into the code object's pool,
/// including elements that are immortal prebuilts (small int, bool, null).
#[test]
fn mixed_constant_lists_agree() {
    agree("[1, \"a\", 2.0]", false);
    agree("[true, null, 1u, 2]", false);
    agree("[[1], [\"a\"]]", false);
    agree("[]", false);
}

/// An all-constant map is one `LoadConst` from the code object's pool.
/// Run through [`Program::execute`] three times so a portal that rebuilt
/// or mutated the pooled object cannot hide behind a single pass.
///
/// Two-entry maps are compared by equality and field reads rather than
/// `Debug`: `HashMap` iteration order is not insertion order, so a
/// compile-time table and a freshly built one print keys differently.
#[test]
fn constant_maps_agree() {
    agree_repeat("{\"a\": 1}");
    agree_repeat("{}");
    agree_repeat("{\"a\": 1, \"b\": 2} == {\"b\": 2, \"a\": 1}");
    agree_repeat("{\"a\": 1, \"b\": 2}.a");
    agree_repeat("{\"a\": 1, \"b\": 2}.b");
    agree_repeat("{\"a\": 1}.a");
    agree_repeat("has({\"a\": 1}.a)");
}

/// The same pooled constant is read twice in one program; the second
/// load must still see the original, not a mutation of the first use.
#[test]
fn the_same_constant_evaluated_twice_agrees() {
    agree_repeat("{\"a\": 1} == {\"a\": 1}");
    agree_repeat("[1, 2] == [1, 2]");
    agree_repeat("[{\"k\": 1}, {\"k\": 1}]");
}

/// Concatenating or extending a pooled constant must copy, not mutate
/// the object the code object owns.
#[test]
fn concatenating_a_constant_list_agrees() {
    agree_repeat("[1, 2] + [3, 4]");
    agree_repeat("[1, 2] + xs");
    agree_repeat("xs + [1, 2]");
    agree_repeat("[1, 2] + [x]");
}

/// Interned maps must match the walker's `HashMap<Key, Value>`: replace on
/// exact `Key`, lookup exact kind first then the other numeric kind, reject
/// a float index. Compared with the walker (Value equality, not Debug order).
#[test]
fn interned_maps_match_the_walker_on_keys() {
    let mut failed = Vec::new();
    for src in [
        r#"size({"a":1,"a":2})"#,
        r#"{"a":1,"a":2}.map(k,k)"#,
        r#"{"a":1,"a":2} == {"a":2}"#,
        r#"size({"a":x,"a":2})"#,
        r#"{x:1, 15:2}[15]"#,
        r#"{x:"p", 15:"q"}.map(k, k)"#,
        r#"{1:"a", 1u:"b"}[1u]"#,
        r#"{1u:"a", 1:"b"}[1]"#,
        r#"{x:1, 15u:2}[15u]"#,
        r#"{x:1}[15.0]"#,
        r#"{x:1, 15u:2}[15]"#,
        r#"15 in {x:1, 15u:2}"#,
        r#"has({"a":x,"a":2}.a)"#,
        r#"{"a":x,"a":2}"#,
        r#"{x:1}[15u]"#,
        r#"{"a":1, "a":2}.a"#,
        r#"{1:"a", 1u:"b"}[1]"#,
        r#"15u in {x:1}"#,
        r#"15 in {x:1}"#,
        r#"15.0 in {x:1}"#,
        r#"15u in {15u:1}"#,
        r#"1 in {1u:2}"#,
        r#""a" in {"a":1}"#,
        r#"true in {true:1}"#,
        r#"null in {x:1}"#,
    ] {
        if let Some(msg) = agree_repeat_if_compiles(src) {
            failed.push(msg);
        }
    }
    assert!(
        failed.is_empty(),
        "walker/vm disagreements:\n{}",
        failed.join("\n")
    );
}

/// A folded constant map must keep source-order pairs and the same
/// duplicate-key / cross-type numeric lookup as unfolded NewMap/MapInsert.
/// Compared with the walker (Value equality, not HashMap Debug order).
#[test]
fn folded_maps_and_concat_chains_agree_with_the_walker() {
    let mut failed = Vec::new();
    for src in [
        r#"{"a":1, "a":2}"#,
        r#"{1:"a", 1u:"b"}"#,
        r#"{"a":1, "a":2}.a"#,
        r#"{1:"a", 1u:"b"}[1]"#,
        r#"{1u:"a", 1:"b"}[1u]"#,
        r#"{"a": {"b": [1, 2]}}.a.b[1]"#,
        r#"{"a":1}.b"#,
        r#"has({"a":1}.b)"#,
        r#"{true: 1, false: 2}[x > 3]"#,
        r#"{"a":1, "b":2}.map(k, k).size()"#,
        r#"{"a":1, "b":2}.all(k, k in {"a":1, "b":2})"#,
        r#"{"k": x, "a": 1, "a": 2}"#,
        r#"[{"a":1}, {"a":1}][0] == {"a":1}"#,
        r#"{1: 2, 2: 3}[1] + {1: 2}[1]"#,
        r#"{"a": 1.5, "b": b"x", "c": null}"#,
        r#"{} == {}"#,
        r#"size({})"#,
        r#"{"a": []}.a + [x]"#,
        r#""a" + "b" + "c" + "d""#,
        r#"[1,2].map(e, "p" + string(e) + "s")"#,
        r#"b"ab" + b"cd" + b"ef""#,
        r#"[] + []"#,
        r#"[x] + [x] + []"#,
        r#"[1, 2] + ["a"]"#,
        r#"[1u] + [1]"#,
        r#"type([] + [])"#,
    ] {
        if let Some(msg) = agree_repeat_if_compiles(src) {
            failed.push(msg);
        }
    }
    assert!(
        failed.is_empty(),
        "walker/vm disagreements:\n{}",
        failed.join("\n")
    );
}

/// `AndLocal` does not write the logic slot; merge still raises NoSuchOverload
/// when the right side is not a bool (`keep_right_merge`'s never-wrote case).
#[test]
fn all_exists_non_bool_body_matches_the_walker() {
    agree("xs.all(x, 1)", false);
    agree("xs.exists(x, 1)", false);
}

/// An error on the left is absorbed when the right decides.
#[test]
fn or_absorbs_a_left_error_when_the_right_is_true() {
    agree("no_such || true", false);
    agree("no_such && false", false);
}

/// One iteration's residual `And`/`Or` must not leave a logic slot that a
/// later iteration's portal fall-through merge can read.
///
/// Run through [`Program::execute`] three times so a jit-dynasm portal that
/// mixed residual and interned arms across loop iterations cannot hide behind
/// a single `cel_eval_loop` pass.
fn agree_repeat(src: &str) {
    let expr = Parser::default()
        .parse(src)
        .unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    let ctx = ctx();
    let walker = Value::resolve_value(&expr, &ctx);
    for i in 0..3 {
        let vm = program.execute(&ctx);
        assert_eq!(
            show(&walker),
            show(&vm),
            "`{src}` execute {i}"
        );
        match (&walker, &vm) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "`{src}` execute {i} value"),
            _ => {}
        }
    }
}

fn agree_repeat_if_compiles(src: &str) -> Option<String> {
    let Ok(expr) = Parser::default().parse(src) else {
        return None;
    };
    let Ok(program) = Program::compile(src) else {
        return None;
    };
    let ctx = ctx();
    let walker = Value::resolve_value(&expr, &ctx);
    for i in 0..3 {
        let vm = program.execute(&ctx);
        let ok = match (&walker, &vm) {
            (Ok(a), Ok(b)) => a == b,
            _ => show(&walker) == show(&vm),
        };
        if !ok {
            return Some(format!(
                "`{src}` execute {i} walker={} vm={}",
                show(&walker),
                show(&vm)
            ));
        }
    }
    None
}

#[test]
fn and_or_slot_is_not_stale_across_loop_iterations() {
    agree_repeat("[1,2].map(e, (10/(e-1) > 0) && e > 1)");
    agree_repeat("[1,2,1,2].map(e, (10/(e-1) > 0) && e > 1)");
    agree_repeat("[1,2].map(e, (10/(e-1) < 0) || e < 2)");
    agree_repeat("[1,2,1,3].map(e, (10/(e-1) < 0) || e < 3)");
    agree_repeat("[1,2,3].filter(e, (10/(e-1) > 0) && e > 1)");
    agree_repeat("[1,2,3].all(e, ((10/(e-1) > 0) && e > 1) || e == 1)");
}

/// `AndLocal`/`OrLocal` fire when the left operand is already a slot. A
/// comprehension variable can be a non-bool on one iteration (residual write)
/// and an interned bool on the next (portal fall-through, no write).
#[test]
fn and_or_local_slot_is_not_stale_across_loop_iterations() {
    agree_repeat("[1, true].map(e, e && (e == true))");
    agree_repeat("[1, true, 1, true].map(e, e && (e == true))");
    agree_repeat("[1, false].map(e, e || (e == 1))");
    agree_repeat("[1, false, 1, false].map(e, e || (e == 1))");
    agree_repeat("[1, true, false].filter(e, e && (e == true))");
    agree_repeat("[1, true].all(e, (e && (e == true)) || e == 1)");
}
