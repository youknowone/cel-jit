//! String-key maps share one mapdict layout.
//!
//! Two public maps with the same key set intern to the same layout pointer
//! whatever order the keys were inserted. Field read, `has`, `size`, `in`,
//! key iteration, equality with an object-strategy map, and the public
//! round trip all read that layout. A portal loop keeps answering when a
//! list mixes two key sets, so a layout guard that fails still hits the
//! right slot.

use std::collections::HashMap;
use std::sync::Arc;

use cel::objects::{Key, KeyRef, Map};
use cel::runtime::binop::{map_contains_key, map_key_refs, w_map_eq};
use cel::runtime::const_pool::ConstPool;
use cel::runtime::convert::{
    intern_leaf, interned_map_contains, interned_map_lookup_string, interned_to_public,
};
use cel::runtime::object::{
    map_len, map_try_insert, new_int, new_map, new_string, string_as_str, CelRef, MapStrategy,
    W_MapObject, MAPDICT_MAX_ENTRIES,
};
use cel::{Context, Program, Value};

fn string_map(pairs: &[(&str, i64)]) -> Value {
    let mut map = HashMap::new();
    for (key, value) in pairs {
        map.insert(*key, Value::Int(*value));
    }
    Value::from(map)
}

fn intern_map(value: &Value) -> CelRef {
    intern_leaf(value).expect("map interns")
}

fn map_leaf<'a>(w: CelRef) -> &'a W_MapObject {
    unsafe { &*w.cast::<W_MapObject>() }
}

fn run(ctx: &Context, src: &str) -> Value {
    Program::compile(src)
        .unwrap_or_else(|err| panic!("parse {src}: {err:?}"))
        .execute(ctx)
        .unwrap_or_else(|err| panic!("execute {src}: {err:?}"))
}

#[test]
fn same_key_set_shares_one_layout() {
    let left = string_map(&[("price", 1), ("name", 2), ("extra", 3)]);
    let right = string_map(&[("extra", 9), ("price", 8), ("name", 7)]);
    let left_w = intern_map(&left);
    let right_w = intern_map(&right);
    let left_leaf = map_leaf(left_w);
    let right_leaf = map_leaf(right_w);
    assert_eq!(left_leaf.strategy, MapStrategy::Mapdict);
    assert_eq!(right_leaf.strategy, MapStrategy::Mapdict);
    assert_ne!(left_leaf.layout, 0);
    assert_eq!(left_leaf.layout, right_leaf.layout);

    let fewer = map_leaf(intern_map(&string_map(&[("price", 1), ("name", 2)])));
    assert_eq!(fewer.strategy, MapStrategy::Mapdict);
    assert_ne!(fewer.layout, left_leaf.layout);

    let empty_a = map_leaf(intern_map(&string_map(&[])));
    let empty_b = map_leaf(intern_map(&string_map(&[])));
    assert_eq!(empty_a.strategy, MapStrategy::Mapdict);
    assert_eq!(empty_a.layout, empty_b.layout);
    assert_ne!(empty_a.layout, 0);
    assert_ne!(empty_a.layout, left_leaf.layout);

    let mut pool = ConstPool::default();
    let pooled = pool.intern(&left);
    assert!(!pooled.is_null(), "const pool interns the map");
    let pooled_leaf = map_leaf(pooled);
    assert_eq!(pooled_leaf.strategy, MapStrategy::Mapdict);
    assert_eq!(pooled_leaf.layout, left_leaf.layout);
}

#[test]
fn mapdict_reads_has_size_keys_eq_and_roundtrip() {
    let value = string_map(&[("price", 4), ("name", 5)]);
    let w = intern_map(&value);
    assert_eq!(map_leaf(w).strategy, MapStrategy::Mapdict);
    assert_eq!(unsafe { map_len(w) }, 2);

    let price = unsafe { interned_map_lookup_string(w, "price") }.expect("price");
    assert_eq!(interned_to_public(price), Value::Int(4));
    assert!(unsafe { interned_map_lookup_string(w, "missing") }.is_none());
    assert!(unsafe { interned_map_contains(w, KeyRef::String("name")) });
    assert!(!unsafe { interned_map_contains(w, KeyRef::String("missing")) });
    assert!(!unsafe { interned_map_contains(w, KeyRef::Int(1)) });

    let name = new_string("name") as CelRef;
    let missing = new_string("missing") as CelRef;
    assert!(unsafe { map_contains_key(w, name) });
    assert!(!unsafe { map_contains_key(w, missing) });

    let mut keys = Vec::new();
    for key in unsafe { map_key_refs(w) } {
        keys.push(
            unsafe { string_as_str(key) }
                .expect("key string")
                .to_owned(),
        );
    }
    keys.sort();
    assert_eq!(keys, vec!["name".to_string(), "price".to_string()]);

    let object = new_map(&[
        (new_string("name") as CelRef, new_int(5) as CelRef),
        (new_string("price") as CelRef, new_int(4) as CelRef),
    ]);
    assert_eq!(unsafe { (*object).strategy }, MapStrategy::Object);
    assert_eq!(unsafe { (*object).layout }, 0);
    assert!(unsafe { w_map_eq(w, object as CelRef) });

    assert_eq!(interned_to_public(w), value);

    let mut ctx = Context::default();
    ctx.add_variable_from_value("m", value);
    assert_eq!(run(&ctx, "m.price"), Value::Int(4));
    assert_eq!(run(&ctx, "m.name"), Value::Int(5));
    assert_eq!(run(&ctx, "has(m.price)"), Value::Bool(true));
    assert_eq!(run(&ctx, "has(m.missing)"), Value::Bool(false));
    assert_eq!(run(&ctx, "size(m)"), Value::Int(2));
    assert_eq!(run(&ctx, "\"name\" in m"), Value::Bool(true));
    assert_eq!(run(&ctx, "\"missing\" in m"), Value::Bool(false));
    assert_eq!(run(&ctx, "m.exists(k, k == \"price\")"), Value::Bool(true));
}

#[test]
fn mapdict_overwrite_stays_and_a_missing_key_is_refused() {
    let w = intern_map(&string_map(&[("price", 1), ("name", 2)]));
    let layout = map_leaf(w).layout;
    assert!(unsafe { map_try_insert(w, new_string("price") as CelRef, new_int(9) as CelRef) });
    assert_eq!(map_leaf(w).strategy, MapStrategy::Mapdict);
    assert_eq!(map_leaf(w).layout, layout);
    let price = unsafe { interned_map_lookup_string(w, "price") }.expect("price");
    assert_eq!(interned_to_public(price), Value::Int(9));

    assert!(!unsafe { map_try_insert(w, new_string("extra") as CelRef, new_int(3) as CelRef) });
    assert!(!unsafe { map_try_insert(w, new_int(1) as CelRef, new_int(3) as CelRef) });
    assert_eq!(map_leaf(w).strategy, MapStrategy::Mapdict);
    assert_eq!(map_leaf(w).layout, layout);
    assert_eq!(unsafe { map_len(w) }, 2);
    assert!(unsafe { interned_map_lookup_string(w, "extra") }.is_none());
    let name = unsafe { interned_map_lookup_string(w, "name") }.expect("name");
    assert_eq!(interned_to_public(name), Value::Int(2));
    let price = unsafe { interned_map_lookup_string(w, "price") }.expect("price");
    assert_eq!(interned_to_public(price), Value::Int(9));
}

#[test]
fn wide_or_non_string_maps_stay_object() {
    let mut wide = HashMap::new();
    for i in 0..=MAPDICT_MAX_ENTRIES {
        wide.insert(format!("k{i:02}"), Value::Int(i as i64));
    }
    let wide_w = intern_map(&Value::from(wide));
    assert_eq!(map_leaf(wide_w).strategy, MapStrategy::Object);
    assert_eq!(map_leaf(wide_w).layout, 0);

    let mut mixed = HashMap::new();
    mixed.insert(Key::Int(1), Value::Int(2));
    mixed.insert(Key::String(Arc::from("a")), Value::Int(3));
    let mixed_w = intern_map(&Value::Map(Map::object(Arc::new(mixed))));
    assert_eq!(map_leaf(mixed_w).strategy, MapStrategy::Object);
    assert_eq!(map_leaf(mixed_w).layout, 0);
}

fn price_name_list(with_extra_on_odd: bool) -> Value {
    let items: Vec<Value> = (0..8i64)
        .map(|i| {
            let mut map = HashMap::new();
            map.insert("name", Value::from(format!("n{i}")));
            map.insert("price", Value::Int(i));
            if with_extra_on_odd && i % 2 == 1 {
                map.insert("extra", Value::Int(1));
            }
            Value::from(map)
        })
        .collect();
    Value::list(items)
}

/// `i.price` / `has(i.price)` / `i.name` stay correct after the portal
/// compiles, including a list whose maps do not share a layout.
///
/// Even elements are `{name, price}` (`price` at index 1). Odd elements of
/// the mixed list are `{extra, name, price}` (`price` at index 2). The
/// trace specialises on the first element; a missing layout guard would
/// read the name string as `i.price` and would miss `has(i.extra)`.
#[test]
fn mapdict_field_ops_answer_after_the_portal_compiles() {
    // SAFETY: stored before this test builds a driver. The knob is read
    // once, when that driver is created.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let prices = Value::list((0..8i64).map(Value::Int).collect::<Vec<_>>());
    let mut uniform = Context::default();
    uniform.add_variable_from_value("items", price_name_list(false));
    let mut mixed = Context::default();
    mixed.add_variable_from_value("items", price_name_list(true));

    let map_price = Program::compile("items.map(i, i.price)").unwrap();
    let exists_name = Program::compile(r#"items.exists(i, i.name == "n3")"#).unwrap();
    let all_price = Program::compile("items.all(i, has(i.price))").unwrap();
    let exists_extra = Program::compile("items.exists(i, has(i.extra))").unwrap();

    for i in 0..200 {
        assert_eq!(
            map_price.execute(&uniform).expect("uniform map"),
            prices,
            "uniform map {i}"
        );
        assert_eq!(
            exists_name.execute(&uniform).expect("uniform exists"),
            Value::Bool(true),
            "uniform exists {i}"
        );
        assert_eq!(
            all_price.execute(&uniform).expect("uniform all"),
            Value::Bool(true),
            "uniform all {i}"
        );
        assert_eq!(
            exists_extra.execute(&uniform).expect("uniform extra"),
            Value::Bool(false),
            "uniform extra {i}"
        );

        assert_eq!(
            map_price.execute(&mixed).expect("mixed map"),
            prices,
            "mixed map {i}"
        );
        assert_eq!(
            exists_name.execute(&mixed).expect("mixed exists"),
            Value::Bool(true),
            "mixed exists {i}"
        );
        assert_eq!(
            all_price.execute(&mixed).expect("mixed all"),
            Value::Bool(true),
            "mixed all {i}"
        );
        assert_eq!(
            exists_extra.execute(&mixed).expect("mixed extra"),
            Value::Bool(true),
            "mixed extra {i}"
        );
    }
}
