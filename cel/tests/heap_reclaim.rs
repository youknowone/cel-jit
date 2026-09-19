//! The evaluation nursery is reset when the outermost evaluation finishes.

use std::collections::HashMap;
use std::sync::Arc;

use cel::objects::{Key, Map};
use cel::runtime::heap::with_heap;
use cel::{Context, Program, Value};

#[test]
fn eval_scope_is_active_during_execute() {
    use cel::runtime::heap::{eval_depth, is_young};
    let mut ctx = Context::default();
    ctx.add_function("depth", || eval_depth() as i64);
    ctx.add_function("young_alloc", || {
        let p = with_heap(|h| h.alloc(7u64));
        i64::from(is_young(p as *const u8))
    });
    let d = eval_vm("depth()", &ctx);
    let y = eval_vm("young_alloc()", &ctx);
    assert_eq!(d, Value::Int(1), "depth during execute, got {d:?}");
    assert_eq!(
        y,
        Value::Int(1),
        "alloc during execute should be young, got {y:?}"
    );
}

fn eval_vm(src: &str, ctx: &Context) -> Value {
    let program = Program::compile(src).expect("compiles");
    program
        .execute(ctx)
        .unwrap_or_else(|e| panic!("vm {src}: {e:?}"))
}

fn eval_walker(src: &str, ctx: &Context) -> Value {
    let program = Program::compile(src).expect("compiles");
    Value::resolve_value(program.expression(), ctx)
        .unwrap_or_else(|e| panic!("walker {src}: {e:?}"))
}

fn agree(src: &str, ctx: &Context) -> Value {
    let vm = eval_vm(src, ctx);
    let walker = eval_walker(src, ctx);
    assert_eq!(
        format!("{vm:?}"),
        format!("{walker:?}"),
        "{src}: vm vs walker"
    );
    vm
}

#[test]
fn ten_thousand_maps_leave_old_space_flat() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", (1..=1000i64).collect::<Vec<i64>>());
    let program = Program::compile("list.map(v, v * 2)").expect("compiles");

    program.execute(&ctx).expect("warms");
    let old_bytes = with_heap(|h| h.old_allocated_bytes());
    let old_segs = with_heap(|h| h.old_segments());
    let water = with_heap(|h| h.nursery_high_water());

    for _ in 0..10_000 {
        program.execute(&ctx).expect("evaluates");
    }

    assert_eq!(
        with_heap(|h| h.old_allocated_bytes()),
        old_bytes,
        "old-space bytes grew across 10000 maps"
    );
    assert_eq!(
        with_heap(|h| h.old_segments()),
        old_segs,
        "old-space segments grew across 10000 maps"
    );
    assert_eq!(
        with_heap(|h| h.nursery_high_water()),
        water,
        "nursery high-water grew after the first map"
    );
}

#[test]
fn a_kept_result_survives_later_evaluations() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", vec![1i64, 2, 3]);
    ctx.add_variable_from_value("n", 7i64);
    let mut m = HashMap::new();
    m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
    ctx.add_variable_from_value("map", Value::Map(Map::object(Arc::new(m))));
    ctx.add_variable_from_value("s", "hello");
    ctx.add_variable_from_value("b", Value::Bytes(Arc::new(vec![1, 2, 3])));

    let cases: &[(&str, Value)] = &[
        (
            "list.map(v, v * 2)",
            Value::list(vec![Value::Int(2), Value::Int(4), Value::Int(6)]),
        ),
        ("{'x': n, 'y': n + 1}", {
            let mut got = HashMap::new();
            got.insert(Key::String(Arc::new("x".to_string())), Value::Int(7));
            got.insert(Key::String(Arc::new("y".to_string())), Value::Int(8));
            Value::Map(Map::object(Arc::new(got)))
        }),
        ("s + s", Value::String(Arc::new("hellohello".to_string()))),
        ("b + b", Value::Bytes(Arc::new(vec![1, 2, 3, 1, 2, 3]))),
        (
            "optional.of(n)",
            eval_vm("optional.of(7)", &Context::default()),
        ),
        ("[{'k': n}]", {
            let mut inner = HashMap::new();
            inner.insert(Key::String(Arc::new("k".to_string())), Value::Int(7));
            Value::list(vec![Value::Map(Map::object(Arc::new(inner)))])
        }),
        (
            "list",
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
        ),
    ];

    for (src, expected) in cases {
        let kept_vm = eval_vm(src, &ctx);
        let kept_walker = eval_walker(src, &ctx);
        for _ in 0..100 {
            let _ = eval_vm("list.map(v, v * 2)", &ctx);
            let _ = eval_walker("list.map(v, v * 2)", &ctx);
        }
        assert_eq!(kept_vm, *expected, "{src}: vm result after reuse");
        assert_eq!(kept_walker, *expected, "{src}: walker result after reuse");
        assert_eq!(kept_vm, kept_walker, "{src}: vm vs walker after reuse");
    }
}

#[test]
fn a_host_reentry_returns_a_valid_inner_result() {
    fn eval_int(source: &str, ctx: &Context) -> i64 {
        match Program::compile(source)
            .expect("compiles")
            .execute(ctx)
            .map(|v| v.unpack())
        {
            Ok(Value::Int(n)) => n,
            other => panic!("{source}: expected an int, got {other:?}"),
        }
    }

    let mut ctx = Context::default();
    ctx.add_function("inner", || -> i64 {
        eval_int("size([1, 2, 3].map(x, x * 2))", &Context::default())
    });
    let program = Program::compile("[1, 2].map(v, v + inner())").expect("compiles");
    let want = Value::from(vec![4i64, 5]);
    assert_eq!(program.execute(&ctx).expect("evaluates"), want);
    let walker = Value::resolve_value(program.expression(), &ctx).expect("walker");
    assert_eq!(walker, want);

    program.execute(&ctx).expect("second");
    let old = with_heap(|h| h.old_allocated_bytes());
    program.execute(&ctx).expect("third");
    assert_eq!(with_heap(|h| h.old_allocated_bytes()), old);
}

#[test]
fn a_panicking_host_resets_the_nursery() {
    let mut ctx = Context::default();
    ctx.add_function("explode", || -> i64 {
        panic!("host function panicking on purpose")
    });
    let exploding = Program::compile("[1, 2, 3].map(v, v * explode())").expect("compiles");
    let ordinary = Program::compile("[1, 2, 3].map(v, v * 2)").expect("compiles");
    ordinary.execute(&ctx).expect("warms");
    let old = with_heap(|h| h.old_allocated_bytes());

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = exploding.execute(&ctx);
    }));
    std::panic::set_hook(previous);
    assert!(caught.is_err(), "the host function was expected to panic");

    assert_eq!(
        ordinary.execute(&ctx).expect("evaluates"),
        Value::from(vec![2i64, 4, 6])
    );
    assert_eq!(with_heap(|h| h.old_allocated_bytes()), old);
}

#[test]
fn vm_and_walker_agree_on_reclaim_cases() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", vec![1i64, 2, 3, 4, 5]);
    agree("list.map(v, v * 2)", &ctx);
    agree("list.filter(v, v > 2)", &ctx);
    agree("list.exists(v, v == 5)", &ctx);
    agree("size(list)", &ctx);
    agree("list[2]", &ctx);
    agree("5 in list", &ctx);
    agree("list == list", &ctx);
}
