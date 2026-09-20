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
    let walker = show(&Value::resolve_value(&expr, &ctx));
    for i in 0..3 {
        let vm = show(&program.execute(&ctx));
        assert_eq!(walker, vm, "`{src}` execute {i}");
    }
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
