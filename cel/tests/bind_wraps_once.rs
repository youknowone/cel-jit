//! Wrap-at-bind: a bound variable is interned once, then loads are pointer copies.
//!
//! A public list or map stored as-is would be copied onto the thread heap on
//! every evaluation. After bind wraps, `size(list)` on a reused `Context` must
//! not grow the heap the second time any more than evaluating `1` does.

use std::collections::HashMap;
use std::sync::Arc;

use cel::objects::{Key, Map};
use cel::runtime::heap::with_heap;
use cel::{Context, Program, Value};

fn eval_vm(src: &str, ctx: &Context) -> Result<Value, String> {
    let program = Program::compile(src).map_err(|e| format!("compile {src}: {e:?}"))?;
    program.execute(ctx).map_err(|e| format!("vm {src}: {e:?}"))
}

fn eval_walker(src: &str, ctx: &Context) -> Result<Value, String> {
    let program = Program::compile(src).map_err(|e| format!("compile {src}: {e:?}"))?;
    Value::resolve_value(program.expression(), ctx).map_err(|e| format!("walker {src}: {e:?}"))
}

fn show(r: &Result<Value, String>) -> String {
    match r {
        Ok(v) => format!("{v:?}"),
        Err(e) => format!("ERR({e})"),
    }
}

fn agree(bound_src: &str, inline_src: &str, ctx: &Context) {
    let vm = eval_vm(bound_src, ctx);
    let walker = eval_walker(bound_src, ctx);
    let inline_vm = eval_vm(inline_src, ctx);
    let inline_walker = eval_walker(inline_src, ctx);
    assert_eq!(show(&vm), show(&walker), "{bound_src}: vm vs walker");
    assert_eq!(
        show(&vm),
        show(&inline_vm),
        "{bound_src} vs inline {inline_src}: vm"
    );
    assert_eq!(
        show(&walker),
        show(&inline_walker),
        "{bound_src} vs inline {inline_src}: walker"
    );
}

#[test]
fn bound_list_agrees_with_inline_literal() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("v", vec![1i64, 2, 3]);
    agree("size(v)", "size([1, 2, 3])", &ctx);
    agree("v[1]", "[1, 2, 3][1]", &ctx);
    agree("2 in v", "2 in [1, 2, 3]", &ctx);
    agree("v.map(x, x)", "[1, 2, 3].map(x, x)", &ctx);
    agree("v == v", "[1, 2, 3] == [1, 2, 3]", &ctx);
    agree("v + v", "[1, 2, 3] + [1, 2, 3]", &ctx);
}

#[test]
fn bound_map_agrees_with_inline_literal() {
    let mut ctx = Context::default();
    let mut m = HashMap::new();
    m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
    m.insert(Key::String(Arc::new("b".to_string())), Value::Int(2));
    ctx.add_variable_from_value("v", Value::Map(Map::object(Arc::new(m))));
    agree("size(v)", "size({'a': 1, 'b': 2})", &ctx);
    agree("v['a']", "{'a': 1, 'b': 2}['a']", &ctx);
    agree("'a' in v", "'a' in {'a': 1, 'b': 2}", &ctx);
    agree("v == v", "{'a': 1, 'b': 2} == {'a': 1, 'b': 2}", &ctx);
    // Key order of a two-entry HashMap is not the literal's insertion order,
    // so `map` is compared between evaluators only.
    let vm = eval_vm("v.map(x, x)", &ctx);
    let walker = eval_walker("v.map(x, x)", &ctx);
    assert_eq!(show(&vm), show(&walker), "v.map(x, x): vm vs walker");
}

#[test]
fn bound_string_and_int_agree_with_inline_literal() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("s", "hello");
    ctx.add_variable_from_value("n", 7i64);
    agree("size(s)", "size('hello')", &ctx);
    agree("s == s", "'hello' == 'hello'", &ctx);
    agree("s + s", "'hello' + 'hello'", &ctx);
    agree("n == n", "7 == 7", &ctx);
    agree("n + n", "7 + 7", &ctx);
}

#[test]
fn get_variable_compares_equal_and_debugs_like_the_bound_value() {
    let original_list: Value = vec![1i64, 2, 3].into();
    let original_map = {
        let mut m = HashMap::new();
        m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
        Value::Map(Map::object(Arc::new(m)))
    };
    let original_str: Value = "hello".into();
    let original_int: Value = 7i64.into();

    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", original_list.clone());
    ctx.add_variable_from_value("map", original_map.clone());
    ctx.add_variable_from_value("s", original_str.clone());
    ctx.add_variable_from_value("n", original_int.clone());

    for (name, original) in [
        ("list", original_list),
        ("map", original_map),
        ("s", original_str),
        ("n", original_int),
    ] {
        let got = ctx.get_variable(name).expect(name);
        assert_eq!(got, original, "{name}: PartialEq with the bound value");
        assert_eq!(
            format!("{got:?}"),
            format!("{original:?}"),
            "{name}: Debug of get_variable"
        );
    }
}

#[test]
fn second_size_of_bound_list_allocates_like_evaluating_one() {
    let mut ctx = Context::default();
    let list: Value = (0..200i64).collect::<Vec<i64>>().into();
    ctx.add_variable_from_value("list", list);

    let size = Program::compile("size(list)").expect("compiles");
    let one = Program::compile("1").expect("compiles");

    size.execute(&ctx).expect("size warms");
    let after_first = with_heap(|h| h.allocated_bytes());
    size.execute(&ctx).expect("size second");
    let after_second = with_heap(|h| h.allocated_bytes());
    let size_second = after_second - after_first;

    let before_one = with_heap(|h| h.allocated_bytes());
    one.execute(&ctx).expect("one");
    let after_one = with_heap(|h| h.allocated_bytes());
    let one_delta = after_one - before_one;

    assert_eq!(
        size_second, one_delta,
        "second size(list) grew the heap by {size_second} bytes; evaluating 1 grew it by {one_delta}"
    );
}

#[test]
fn bound_comprehensions_agree_with_inline_literal() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", vec![1i64, 2, 3, 4, 5]);
    agree("list.map(v, v * 2)", "[1, 2, 3, 4, 5].map(v, v * 2)", &ctx);
    agree(
        "list.filter(v, v > 2)",
        "[1, 2, 3, 4, 5].filter(v, v > 2)",
        &ctx,
    );
    agree(
        "list.exists(v, v == 5)",
        "[1, 2, 3, 4, 5].exists(v, v == 5)",
        &ctx,
    );
    agree("list.all(v, v > 0)", "[1, 2, 3, 4, 5].all(v, v > 0)", &ctx);
    agree(
        "list.map(a, list.exists(b, b == a))",
        "[1, 2, 3, 4, 5].map(a, [1, 2, 3, 4, 5].exists(b, b == a))",
        &ctx,
    );
}

#[test]
fn walker_comprehension_over_bound_ints_does_not_intern_per_element() {
    let mut ctx = Context::default();
    let list: Value = (0..1000i64).collect::<Vec<i64>>().into();
    ctx.add_variable_from_value("list", list);

    let bound = Program::compile("list.map(v, v * 2)").expect("compiles");
    let one = Program::compile("1").expect("compiles");
    let expr = bound.expression();
    let one_expr = one.expression();

    Value::resolve_value(expr, &ctx).expect("warms");
    let after_first = with_heap(|h| h.allocated_bytes());
    Value::resolve_value(expr, &ctx).expect("second");
    let after_second = with_heap(|h| h.allocated_bytes());
    let map_second = after_second - after_first;

    let before_one = with_heap(|h| h.allocated_bytes());
    Value::resolve_value(one_expr, &ctx).expect("one");
    let after_one = with_heap(|h| h.allocated_bytes());
    let one_delta = after_one - before_one;

    assert_eq!(
        map_second, one_delta,
        "second walker map over a bound 1000-int list grew the heap by {map_second} bytes; evaluating 1 grew it by {one_delta}"
    );
}

#[test]
fn bound_int_list_is_ints_strategy_and_mixes_with_object_literals() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("bound", vec![1i64, 2, 3]);
    agree("bound == [1, 2, 3]", "[1, 2, 3] == [1, 2, 3]", &ctx);
    agree("[1, 2, 3] == bound", "[1, 2, 3] == [1, 2, 3]", &ctx);
    agree("bound + [4]", "[1, 2, 3] + [4]", &ctx);
    agree("bound[0]", "[1, 2, 3][0]", &ctx);
    agree("2 in bound", "2 in [1, 2, 3]", &ctx);
    agree("size(bound)", "size([1, 2, 3])", &ctx);
    agree("bound == bound", "[1, 2, 3] == [1, 2, 3]", &ctx);

    let got = ctx.get_variable("bound").expect("bound");
    assert_eq!(
        got,
        Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
    assert_eq!(
        format!("{got:?}"),
        format!(
            "{:?}",
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        )
    );
    let unpacked = got.unpack();
    assert!(matches!(unpacked, Value::List(_)));
    assert_eq!(
        unpacked,
        Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
    );
}
