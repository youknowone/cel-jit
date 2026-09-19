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
use cel::{Context, ExecutionError, Value};

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
