//! Bound containers unpack through the public link recorded at wrap.

use std::collections::HashMap;
use std::sync::Arc;

use cel::objects::{Key, Map};
use cel::{Context, Program, Value};

fn eval_vm(src: &str, ctx: &Context) -> Value {
    Program::compile(src)
        .expect("compiles")
        .execute(ctx)
        .unwrap_or_else(|e| panic!("vm {src}: {e:?}"))
}

fn eval_walker(src: &str, ctx: &Context) -> Value {
    let program = Program::compile(src).expect("compiles");
    Value::resolve_value(program.expression(), ctx)
        .unwrap_or_else(|e| panic!("walker {src}: {e:?}"))
}

#[test]
fn bound_list_door_returns_the_same_buffer() {
    let original: Value = (0..1000i64).collect::<Vec<i64>>().into();
    let Value::List(orig) = &original else {
        panic!("expected list");
    };
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", original.clone());

    let got = ctx.get_variable("list").expect("list");
    assert!(
        !matches!(got, Value::Interned(_)),
        "get_variable must not expose Interned for a bound list"
    );
    let Value::List(got_list) = &got else {
        panic!("get_variable: {got:?}");
    };
    assert!(orig.ptr_eq(got_list));

    for (name, eval) in [
        ("vm", eval_vm as fn(&str, &Context) -> Value),
        ("walker", eval_walker),
    ] {
        let door = eval("list", &ctx);
        assert!(
            !matches!(door, Value::Interned(_)),
            "{name} door must not return Interned"
        );
        let Value::List(door_list) = &door else {
            panic!("{name} door: {door:?}");
        };
        assert!(orig.ptr_eq(door_list), "{name} list identity");

        let nested = eval("[list, list]", &ctx);
        let Value::List(shell) = &nested else {
            panic!("{name} nested: {nested:?}");
        };
        assert_eq!(shell.len(), 2);
        for i in 0..2 {
            match shell.get(i).expect("elt") {
                Value::List(elt) => assert!(orig.ptr_eq(&elt), "{name} nested element {i}"),
                other => panic!("{name} nested element {i}: {other:?}"),
            }
        }
    }
}

#[test]
fn bound_map_door_returns_the_same_table() {
    let mut entries = HashMap::new();
    entries.insert(Key::String(Arc::from("k")), Value::Int(1));
    let original = Map::object(Arc::new(entries));
    let mut ctx = Context::default();
    ctx.add_variable_from_value("m", Value::Map(original.clone()));

    let got = ctx.get_variable("m").expect("m");
    assert!(!matches!(got, Value::Interned(_)));
    let Value::Map(got_map) = &got else {
        panic!("{got:?}");
    };
    assert!(original.ptr_eq(got_map));

    let door = eval_vm("m", &ctx);
    let Value::Map(door_map) = &door else {
        panic!("{door:?}");
    };
    assert!(original.ptr_eq(door_map));
}

#[test]
fn bound_string_door_returns_the_same_arc() {
    let original: Arc<str> = Arc::from("x".repeat(1024));
    let mut ctx = Context::default();
    ctx.add_variable_from_value("s", Value::String(original.clone()));

    let got = ctx.get_variable("s").expect("s");
    assert!(!matches!(got, Value::Interned(_)));
    let Value::String(got_s) = &got else {
        panic!("{got:?}");
    };
    assert!(Arc::ptr_eq(&original, got_s));

    let door = eval_vm("s", &ctx);
    let Value::String(door_s) = &door else {
        panic!("{door:?}");
    };
    assert!(Arc::ptr_eq(&original, door_s));
    let walker = eval_walker("s", &ctx);
    let Value::String(walker_s) = &walker else {
        panic!("{walker:?}");
    };
    assert!(Arc::ptr_eq(&original, walker_s));
}

#[test]
fn rebind_replaces_the_name_and_the_old_public_handle_stays_valid() {
    let first: Value = vec![1i64, 2, 3].into();
    let second: Value = vec![4i64, 5].into();
    let Value::List(first_buf) = &first else {
        panic!("first");
    };
    let Value::List(second_buf) = &second else {
        panic!("second");
    };
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", first.clone());
    let held = ctx.get_variable("list").expect("first get");
    ctx.add_variable_from_value("list", second.clone());
    let now = ctx.get_variable("list").expect("second get");
    let Value::List(now_list) = &now else {
        panic!("{now:?}");
    };
    assert!(second_buf.ptr_eq(now_list));
    let Value::List(held_list) = &held else {
        panic!("{held:?}");
    };
    assert!(first_buf.ptr_eq(held_list));
    assert_eq!(eval_vm("list", &ctx), second);
}

fn row_value(i: i64) -> Value {
    let b: &str = if i % 2 == 0 { "k" } else { "x" };
    let mut m = HashMap::new();
    m.insert("a", Value::Int(i));
    m.insert("b", Value::from(b));
    m.insert("c", Value::Int(i % 10));
    Value::from(m)
}

/// A filter that only selects rows must return the input maps themselves.
/// Fails while `finish` rebuilds each survivor through `try_build_map`.
#[test]
fn filtered_row_is_pointer_identical_to_the_input_row() {
    let rows: Vec<Value> = (0..8).map(row_value).collect();
    let originals: Vec<Map> = rows
        .iter()
        .map(|v| match v {
            Value::Map(m) => m.clone(),
            other => panic!("row: {other:?}"),
        })
        .collect();
    let mut ctx = Context::default();
    ctx.add_variable_from_value("rows", Value::list(rows));

    for (name, eval) in [
        ("vm", eval_vm as fn(&str, &Context) -> Value),
        ("walker", eval_walker),
    ] {
        let got = eval(r#"rows.filter(r, r.a > 3 && r.b == "k")"#, &ctx);
        let Value::List(list) = &got else {
            panic!("{name}: {got:?}");
        };
        assert!(!list.is_empty(), "{name} kept no rows");
        for i in 0..list.len() {
            let Value::Map(row) = list.get(i).expect("row") else {
                panic!("{name} elt {i}");
            };
            assert!(
                originals.iter().any(|orig| orig.ptr_eq(&row)),
                "{name} kept row {i} is not pointer-identical to an input row"
            );
        }
    }
}

/// A filter that only selects strings must return the input arcs themselves.
#[test]
fn filtered_string_is_pointer_identical_to_the_input_string() {
    let tags: Vec<Value> = (0..8)
        .map(|i| {
            if i % 3 == 0 {
                Value::from(format!("ab{i}"))
            } else {
                Value::from(format!("zz{i}"))
            }
        })
        .collect();
    let originals: Vec<Arc<str>> = tags
        .iter()
        .map(|v| match v {
            Value::String(s) => Arc::clone(s),
            other => panic!("tag: {other:?}"),
        })
        .collect();
    let mut ctx = Context::default();
    ctx.add_variable_from_value("tags", Value::list(tags));

    for (name, eval) in [
        ("vm", eval_vm as fn(&str, &Context) -> Value),
        ("walker", eval_walker),
    ] {
        let got = eval(r#"tags.filter(t, t.startsWith("ab"))"#, &ctx);
        let Value::List(list) = &got else {
            panic!("{name}: {got:?}");
        };
        assert!(!list.is_empty(), "{name} kept no tags");
        for i in 0..list.len() {
            let Value::String(s) = list.get(i).expect("tag") else {
                panic!("{name} elt {i}");
            };
            assert!(
                originals.iter().any(|orig| Arc::ptr_eq(orig, &s)),
                "{name} kept tag {i} is not pointer-identical to an input string"
            );
        }
    }
}
