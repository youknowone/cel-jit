//! Insertion-ordered public maps (`MapStorage::Entries`).

use std::collections::HashMap;
use std::sync::Arc;

use cel::objects::{Key, Map, MapStorage};
use cel::{Context, Program, Value};

fn entries_map(pairs: Vec<(Key, Value)>) -> Map {
    Map::ordered(pairs.into_boxed_slice())
}

fn object_map(pairs: &[(Key, Value)]) -> Map {
    let mut map = HashMap::with_capacity(pairs.len());
    for (k, v) in pairs {
        map.insert(k.clone(), v.clone());
    }
    Map::object(Arc::new(map))
}

fn assert_storages_agree(pairs: &[(Key, Value)]) {
    let entries = entries_map(pairs.to_vec());
    let object = object_map(pairs);
    assert_eq!(entries.len(), object.len());
    assert_eq!(entries, object);

    for (key, value) in pairs {
        assert_eq!(entries.get(key).as_deref(), Some(value));
        assert_eq!(object.get(key).as_deref(), Some(value));
        assert!(entries.contains_key(key));
        assert!(object.contains_key(key));
        match key {
            Key::Int(i) => {
                if let Ok(u) = u64::try_from(*i) {
                    assert_eq!(entries.get(&Key::Uint(u)).as_deref(), Some(value));
                    assert_eq!(object.get(&Key::Uint(u)).as_deref(), Some(value));
                    assert!(!entries.contains_key(&Key::Uint(u)));
                    assert!(!object.contains_key(&Key::Uint(u)));
                }
            }
            Key::Uint(u) => {
                if let Ok(i) = i64::try_from(*u) {
                    assert_eq!(entries.get(&Key::Int(i)).as_deref(), Some(value));
                    assert_eq!(object.get(&Key::Int(i)).as_deref(), Some(value));
                    assert!(!entries.contains_key(&Key::Int(i)));
                    assert!(!object.contains_key(&Key::Int(i)));
                }
            }
            _ => {}
        }
    }

    assert_eq!(entries.to_hashmap(), object.to_hashmap());
    if pairs.len() <= 8 {
        assert!(matches!(entries.storage(), MapStorage::Entries(_)));
        let order: Vec<Key> = entries.iter().map(|(k, _)| k.clone()).collect();
        let expected: Vec<Key> = pairs.iter().map(|(k, _)| k.clone()).collect();
        assert_eq!(order, expected);
    } else {
        assert!(matches!(entries.storage(), MapStorage::Object(_)));
    }
}

#[test]
fn three_entry_entries_agrees_with_object() {
    let pairs = vec![
        (Key::Int(1), Value::Int(10)),
        (Key::Uint(2), Value::Float(2.5)),
        (Key::String(Arc::new("k".into())), Value::Bool(true)),
    ];
    assert_storages_agree(&pairs);
}

#[test]
fn twenty_entry_entries_agrees_with_object() {
    let pairs: Vec<(Key, Value)> = (1i64..=20)
        .map(|i| (Key::Int(i), Value::Int(i * 10)))
        .collect();
    assert_storages_agree(&pairs);
}

#[cfg(feature = "vm")]
#[test]
fn const_map_literal_is_entries_in_source_order() {
    let ctx = Context::default();
    let program = Program::compile(r#"{"c": 1, "a": 2, "b": 3}"#).expect("compiles");
    let Value::Map(map) = program.execute(&ctx).expect("execute") else {
        panic!("expected map");
    };
    assert!(matches!(map.storage(), MapStorage::Entries(_)));
    let keys: Vec<&str> = map
        .iter()
        .map(|(k, _)| match k {
            Key::String(s) => s.as_str(),
            other => panic!("expected string key, got {other:?}"),
        })
        .collect();
    assert_eq!(keys, ["c", "a", "b"]);
}

#[cfg(feature = "vm")]
#[test]
fn vm_map_result_is_entries_in_source_order() {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 1i64);
    ctx.add_variable_from_value("y", 2i64);
    ctx.add_variable_from_value("z", 3i64);
    let program = Program::compile(r#"{"c": x, "a": y, "b": z}"#).expect("compiles");
    let Value::Map(map) = program.execute(&ctx).expect("execute") else {
        panic!("expected map");
    };
    assert!(matches!(map.storage(), MapStorage::Entries(_)));
    let keys: Vec<&str> = map
        .iter()
        .map(|(k, _)| match k {
            Key::String(s) => s.as_str(),
            other => panic!("expected string key, got {other:?}"),
        })
        .collect();
    assert_eq!(keys, ["c", "a", "b"]);

    let program = Program::compile(r#"[1,2,3].map(e, {"k": e})"#).expect("compiles");
    let Value::List(list) = program.execute(&ctx).expect("execute") else {
        panic!("expected list");
    };
    assert_eq!(list.len(), 3);
    for i in 0..3 {
        let Value::Map(inner) = list.get(i).expect("elt") else {
            panic!("expected map element");
        };
        assert!(matches!(inner.storage(), MapStorage::Entries(_)));
        let pairs: Vec<(Key, Value)> = inner
            .iter()
            .map(|(k, v)| (k.clone(), v.into_owned()))
            .collect();
        assert_eq!(
            pairs,
            vec![(
                Key::String(Arc::new("k".into())),
                Value::Int(i as i64 + 1)
            )]
        );
    }
}

#[test]
fn bound_entries_map_round_trips_as_ptr_eq() {
    let original = entries_map(vec![
        (Key::String(Arc::new("a".into())), Value::Int(1)),
        (Key::String(Arc::new("b".into())), Value::Int(2)),
        (Key::Int(3), Value::Float(1.5)),
    ]);
    let mut ctx = Context::default();
    ctx.add_variable_from_value("m", Value::Map(original.clone()));
    let program = Program::compile("m").expect("compiles");
    let Value::Map(got) = program.execute(&ctx).expect("execute") else {
        panic!("expected map");
    };
    assert!(original.ptr_eq(&got));
}

fn assert_replace_keeps_first_position(input: Vec<(Key, Value)>, unique: &[(Key, Value)]) {
    let map = entries_map(input);
    assert!(matches!(map.storage(), MapStorage::Entries(_)));
    assert_eq!(map.len(), unique.len());
    let got: Vec<(Key, Value)> = map
        .iter()
        .map(|(k, v)| (k.clone(), v.into_owned()))
        .collect();
    assert_eq!(got, unique);
    for (key, value) in unique {
        assert_eq!(map.get(key).as_deref(), Some(value));
        assert!(map.contains_key(key));
    }
}

#[test]
fn new_replaces_duplicate_keys_on_a_scanned_table() {
    let a = Key::String(Arc::new("a".into()));
    let b = Key::String(Arc::new("b".into()));
    let input = vec![
        (a.clone(), Value::Int(1)),
        (b.clone(), Value::Int(2)),
        (a.clone(), Value::Int(3)),
    ];
    assert_replace_keeps_first_position(input, &[(a, Value::Int(3)), (b, Value::Int(2))]);
}

#[test]
fn new_replaces_duplicate_keys_on_an_object_table() {
    let mut input: Vec<(Key, Value)> = (1i64..=18)
        .map(|i| (Key::Int(i), Value::Int(i)))
        .collect();
    input.push((Key::Int(1), Value::Int(100)));
    input.push((Key::Int(10), Value::Int(1000)));
    assert_eq!(input.len(), 20);
    let map = entries_map(input);
    assert!(matches!(map.storage(), MapStorage::Object(_)));
    assert_eq!(map.len(), 18);
    for i in 1i64..=18 {
        let v = match i {
            1 => 100,
            10 => 1000,
            other => other,
        };
        assert_eq!(map.get(&Key::Int(i)).as_deref(), Some(&Value::Int(v)));
    }
}

#[test]
fn twenty_entry_entries_maps_compare_equal_in_any_order() {
    let pairs: Vec<(Key, Value)> = (1i64..=20)
        .map(|i| (Key::Int(i), Value::Int(i * 10)))
        .collect();
    let a = entries_map(pairs.clone());
    let mut reversed = pairs;
    reversed.reverse();
    let b = entries_map(reversed);
    assert_eq!(a, b);
    assert_eq!(a.len(), 20);
    assert_eq!(b.len(), 20);
}

fn twenty_literal() -> String {
    let mut src = String::from("{");
    for i in 0..20 {
        if i > 0 {
            src.push_str(", ");
        }
        src.push_str(&format!("\"k{i}\": x"));
    }
    src.push('}');
    src
}

#[cfg(feature = "vm")]
#[test]
fn twenty_entry_vm_and_walker_are_object_and_equal() {
    let src = twenty_literal();
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 7i64);
    let program = Program::compile(&src).expect("compiles");
    let walker = Value::resolve_value(program.expression(), &ctx).expect("walker");
    let Value::Map(walker_map) = &walker else {
        panic!("expected map");
    };
    assert!(matches!(walker_map.storage(), MapStorage::Object(_)));

    let door = program.execute(&ctx).expect("execute");
    let Value::Map(vm_map) = &door else {
        panic!("expected map");
    };
    assert!(matches!(vm_map.storage(), MapStorage::Object(_)));
    assert_eq!(walker, door);
    let object = object_map(
        &(0..20)
            .map(|i| {
                (
                    Key::String(Arc::new(format!("k{i}"))),
                    Value::Int(7),
                )
            })
            .collect::<Vec<_>>(),
    );
    assert_eq!(*walker_map, object);
    assert_eq!(*vm_map, object);
}

#[test]
fn thousand_entry_object_looks_up_by_hash() {
    let pairs: Vec<(Key, Value)> = (0..1000)
        .map(|i| (Key::Int(i), Value::Int(i)))
        .collect();
    let map = entries_map(pairs);
    assert!(matches!(map.storage(), MapStorage::Object(_)));
    for i in 0..1000 {
        assert_eq!(map.get(&Key::Int(i)).as_deref(), Some(&Value::Int(i)));
    }
}

#[test]
fn bound_object_map_round_trips_as_ptr_eq() {
    let pairs: Vec<(Key, Value)> = (1i64..=20)
        .map(|i| (Key::Int(i), Value::Int(i)))
        .collect();
    let original = entries_map(pairs);
    assert!(matches!(original.storage(), MapStorage::Object(_)));
    let mut ctx = Context::default();
    ctx.add_variable_from_value("m", Value::Map(original.clone()));
    let program = Program::compile("m").expect("compiles");
    let Value::Map(got) = program.execute(&ctx).expect("execute") else {
        panic!("expected map");
    };
    assert!(original.ptr_eq(&got));
}

#[test]
fn walker_map_literal_is_entries_in_source_order() {
    let src = r#"{"a": 1, "b": 2}"#;
    let ctx = Context::default();
    let program = Program::compile(src).expect("compiles");
    let walker = Value::resolve_value(program.expression(), &ctx).expect("walker");
    let Value::Map(map) = &walker else {
        panic!("expected map");
    };
    assert!(matches!(map.storage(), MapStorage::Entries(_)));
    let keys: Vec<&str> = map
        .iter()
        .map(|(k, _)| match k {
            Key::String(s) => s.as_str(),
            other => panic!("expected string key, got {other:?}"),
        })
        .collect();
    assert_eq!(keys, ["a", "b"]);
    let door = program.execute(&ctx).expect("execute");
    assert_eq!(walker, door);
}
