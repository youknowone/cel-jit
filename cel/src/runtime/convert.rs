//! Boundary between the public [`crate::Value`] enum and this class family.
//!
//! [`crate::Value`] is the cel drop-in and does not change: callers still
//! construct `Value::Int`, match variants, and bind `This<Arc<String>>`.
//! Evaluators that want a header-first object cross here, and only here.
//!
//! Leftover host opaques cross as [`super::object::W_OpaqueObject`], with the
//! `Arc<dyn Opaque>` parked in this thread's heap table (D12). Record-row
//! maps and column-window lists intern as strategy windows so the schema
//! stays shared.

use std::collections::HashMap;
use std::sync::Arc;

use super::object::{
    new_bool, new_bytes, new_double, new_int, new_list, new_list_ints, new_list_window, new_map,
    new_map_record, new_null, new_opaque, new_optional, new_optional_none, new_string, new_type,
    new_uint, opaque_host_index, w_kind, w_type, CelClass, CelKind, CelRef, ListStrategy,
    MapStrategy, W_BoolObject, W_BytesObject, W_DoubleObject, W_IntColumn, W_IntObject,
    W_ListObject, W_MapObject, W_OptionalObject, W_StringObject, W_TypeObject, W_UIntObject,
    CEL_BOOL_CLASS, CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_INT_CLASS, CEL_LIST_CLASS,
    CEL_MAP_CLASS, CEL_NULL_CLASS, CEL_OPAQUE_CLASS, CEL_OPTIONAL_CLASS, CEL_STRING_CLASS,
    CEL_TYPE_CLASS, CEL_UINT_CLASS,
};
use super::object_array::{bytes_base, items_block_items_base};
use crate::common::types::{
    Kind, Type, TypeValue, BOOL_TYPE, BYTES_TYPE, DOUBLE_TYPE, INT_TYPE, LIST_TYPE, MAP_TYPE,
    NULL_TYPE, OPTIONAL_TYPE, STRING_TYPE, TYPE_TYPE, UINT_TYPE,
};
use crate::objects::{Key, ListRef, ListStorage, Map, MapStorage, Opaque, OptionalValue};
use crate::Value;

#[cfg(feature = "structs")]
use super::object::{new_struct, W_StructObject, CEL_STRUCT_CLASS};
#[cfg(feature = "structs")]
use crate::common::types::CelStruct;

#[cfg(feature = "chrono")]
use super::object::{
    new_duration, new_timestamp, W_DurationObject, W_TimestampObject, CEL_DURATION_CLASS,
    CEL_TIMESTAMP_CLASS,
};
#[cfg(feature = "chrono")]
use crate::common::types::{DURATION_TYPE, TIMESTAMP_TYPE};

/// Why a value could not cross the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertError {
    /// This family has no leaf for the public variant.
    Unsupported(&'static str),
    /// The `CelRef` is not a value this boundary can read back.
    Corrupt(&'static str),
}

/// Move a nursery result to the public owned form. An interned value that
/// already lives in old or immortal memory is returned as it is.
pub fn promote_eval_result(v: Value) -> Value {
    super::heap::with_heap(|h| match v {
        Value::Interned(w) if h.is_young(w as *const u8) => Value::from_interned(w).unpack(),
        other => other,
    })
}

/// Intern `v` onto a class-family leaf. Total: every public variant has
/// a leaf, including leftover opaques and column/record windows.
pub fn intern_leaf(v: &Value) -> Option<CelRef> {
    match v {
        Value::Interned(w) => {
            if super::heap::is_forcing_old() && super::heap::is_young(*w as *const u8) {
                intern_leaf(&Value::from_interned(*w).unpack())
            } else {
                Some(*w)
            }
        }
        Value::Int(i) => Some(new_int(*i) as CelRef),
        Value::UInt(u) => Some(new_uint(*u) as CelRef),
        Value::Float(f) => Some(new_double(*f) as CelRef),
        Value::Bool(b) => Some(new_bool(*b) as CelRef),
        Value::Null => Some(new_null() as CelRef),
        Value::String(s) => Some(new_string(s) as CelRef),
        Value::Bytes(b) => Some(new_bytes(b) as CelRef),
        Value::List(_) => value_to_ref(v).ok(),
        Value::Map(_) => value_to_ref(v).ok(),
        Value::Opaque(opaque) if opaque.downcast_ref::<OptionalValue>().is_some() => {
            value_to_ref(v).ok()
        }
        Value::Opaque(opaque) => opaque
            .downcast_ref::<TypeValue>()
            .and_then(type_class)
            .map(|cls| new_type(cls) as CelRef)
            .or_else(|| Some(intern_host_opaque(opaque))),
        #[cfg(feature = "chrono")]
        Value::Duration(_) | Value::Timestamp(_) => value_to_ref(v).ok(),
        #[cfg(feature = "structs")]
        Value::Struct(_) => value_to_ref(v).ok(),
    }
}

/// Allocate the internal form of `v` on this thread's heap.
pub fn value_to_ref(v: &Value) -> Result<CelRef, ConvertError> {
    match v {
        Value::Interned(w) => {
            intern_leaf(&Value::Interned(*w)).ok_or(ConvertError::Corrupt("interned"))
        }
        Value::Int(i) => Ok(new_int(*i) as CelRef),
        Value::UInt(u) => Ok(new_uint(*u) as CelRef),
        Value::Float(f) => Ok(new_double(*f) as CelRef),
        Value::Bool(b) => Ok(new_bool(*b) as CelRef),
        Value::Null => Ok(new_null() as CelRef),
        Value::String(s) => Ok(new_string(s) as CelRef),
        Value::Bytes(b) => Ok(new_bytes(b) as CelRef),
        Value::List(list) => Ok(intern_list(list)),
        Value::Map(map) => Ok(intern_map(map)?),
        #[cfg(feature = "structs")]
        Value::Struct(s) => {
            let name = new_string(s.name());
            let fields = s.field_values();
            let mut pairs = Vec::with_capacity(fields.len());
            for (fname, fval) in fields.iter() {
                pairs.push((new_string(fname) as CelRef, value_to_ref(fval)?));
            }
            Ok(new_struct(name, &pairs) as CelRef)
        }
        Value::Opaque(opaque) => {
            if let Some(opt) = opaque.downcast_ref::<OptionalValue>() {
                return match opt.value() {
                    None => Ok(new_optional_none() as CelRef),
                    Some(inner) => Ok(new_optional(value_to_ref(inner)?) as CelRef),
                };
            }
            if let Some(tv) = opaque.downcast_ref::<TypeValue>() {
                if let Some(cls) = type_class(tv) {
                    return Ok(new_type(cls) as CelRef);
                }
            }
            Ok(intern_host_opaque(opaque))
        }
        #[cfg(feature = "chrono")]
        Value::Duration(d) => {
            let nanos = d
                .num_nanoseconds()
                .ok_or(ConvertError::Unsupported("duration"))?;
            Ok(new_duration(nanos) as CelRef)
        }
        #[cfg(feature = "chrono")]
        Value::Timestamp(ts) => {
            let nanos = ts
                .timestamp_nanos_opt()
                .ok_or(ConvertError::Unsupported("timestamp"))?;
            Ok(new_timestamp(nanos, i64::from(ts.offset().local_minus_utc())) as CelRef)
        }
    }
}

/// Read `w` back as the public [`Value`].
///
/// # Safety
///
/// `w` must point at a live object this thread's heap still owns.
pub unsafe fn ref_to_value(w: CelRef) -> Result<Value, ConvertError> {
    if w.is_null() {
        return Err(ConvertError::Corrupt("null pointer"));
    }
    let kind = unsafe { w_kind(w) };
    let class = unsafe { w_type(w) };
    match kind {
        CelKind::Int if class == &CEL_INT_CLASS => {
            Ok(Value::Int(unsafe { (*w.cast::<W_IntObject>()).intval }))
        }
        CelKind::UInt if class == &CEL_UINT_CLASS => {
            Ok(Value::UInt(unsafe { (*w.cast::<W_UIntObject>()).uintval }))
        }
        CelKind::Double if class == &CEL_DOUBLE_CLASS => Ok(Value::Float(unsafe {
            (*w.cast::<W_DoubleObject>()).floatval
        })),
        CelKind::Bool if class == &CEL_BOOL_CLASS => Ok(Value::Bool(
            unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0,
        )),
        CelKind::Null if class == &CEL_NULL_CLASS => Ok(Value::Null),
        CelKind::Str if class == &CEL_STRING_CLASS => {
            let leaf = unsafe { &*w.cast::<W_StringObject>() };
            Ok(Value::String(string_from_leaf(leaf)?))
        }
        CelKind::Bytes if class == &CEL_BYTES_CLASS => {
            let leaf = unsafe { &*w.cast::<W_BytesObject>() };
            let n = leaf.length as usize;
            let base = unsafe { bytes_base(leaf.data) };
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("bytes"));
            }
            let bytes = unsafe { std::slice::from_raw_parts(base, n) };
            Ok(Value::Bytes(Arc::new(bytes.to_vec())))
        }
        CelKind::List if class == &CEL_LIST_CLASS => Ok(Value::List(unsafe { list_from_ref(w)? })),
        CelKind::Map if class == &CEL_MAP_CLASS => Ok(Value::Map(unsafe { map_from_ref(w)? })),
        #[cfg(feature = "structs")]
        CelKind::Struct if class == &CEL_STRUCT_CLASS => {
            let leaf = unsafe { &*w.cast::<W_StructObject>() };
            let name = string_from_ref(leaf.name as CelRef)?;
            let n = leaf.length as usize;
            let base = unsafe { items_block_items_base(leaf.fields) };
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("struct"));
            }
            let mut s = CelStruct::new((*name).clone());
            for i in 0..n {
                let fname = unsafe { string_from_ref(*base.add(2 * i))? };
                let fval = unsafe { ref_to_value(*base.add(2 * i + 1))? };
                s.add_field_value((*fname).clone(), fval);
            }
            Ok(Value::Struct(Arc::new(s)))
        }
        CelKind::Optional if class == &CEL_OPTIONAL_CLASS => {
            let inner = unsafe { (*w.cast::<W_OptionalObject>()).w_value };
            let opt = if inner.is_null() {
                OptionalValue::none()
            } else {
                OptionalValue::of(unsafe { ref_to_value(inner)? })
            };
            Ok(Value::Opaque(Arc::new(opt)))
        }
        CelKind::Type if class == &CEL_TYPE_CLASS => {
            let denoted = unsafe { (*w.cast::<W_TypeObject>()).cls };
            let ty = type_from_class(denoted).ok_or(ConvertError::Unsupported("type"))?;
            Ok(Value::Opaque(Arc::new(TypeValue::new(ty))))
        }
        CelKind::Opaque if class == &CEL_OPAQUE_CLASS => {
            let idx = unsafe { opaque_host_index(w) };
            host_opaque(idx)
                .map(Value::Opaque)
                .ok_or(ConvertError::Corrupt("opaque"))
        }
        #[cfg(feature = "chrono")]
        CelKind::Duration if class == &CEL_DURATION_CLASS => {
            let nanos = unsafe { (*w.cast::<W_DurationObject>()).nanos };
            Ok(Value::Duration(chrono::Duration::nanoseconds(nanos)))
        }
        #[cfg(feature = "chrono")]
        CelKind::Timestamp if class == &CEL_TIMESTAMP_CLASS => {
            let leaf = unsafe { &*w.cast::<W_TimestampObject>() };
            let off = chrono::FixedOffset::east_opt(leaf.off_s as i32)
                .ok_or(ConvertError::Corrupt("timestamp"))?;
            let utc = chrono::DateTime::from_timestamp_nanos(leaf.nanos);
            Ok(Value::Timestamp(utc.with_timezone(&off)))
        }
        _ => Err(ConvertError::Unsupported("class")),
    }
}

fn type_class(tv: &TypeValue) -> Option<&'static CelClass> {
    match tv.cel_type().kind() {
        Kind::Int => Some(&CEL_INT_CLASS),
        Kind::UInt => Some(&CEL_UINT_CLASS),
        Kind::Double => Some(&CEL_DOUBLE_CLASS),
        Kind::Boolean => Some(&CEL_BOOL_CLASS),
        Kind::String => Some(&CEL_STRING_CLASS),
        Kind::Bytes => Some(&CEL_BYTES_CLASS),
        Kind::NullType => Some(&CEL_NULL_CLASS),
        Kind::List => Some(&CEL_LIST_CLASS),
        Kind::Map => Some(&CEL_MAP_CLASS),
        Kind::Type => Some(&CEL_TYPE_CLASS),
        Kind::Opaque if tv.name() == "optional_type" => Some(&CEL_OPTIONAL_CLASS),
        #[cfg(feature = "chrono")]
        Kind::Duration => Some(&CEL_DURATION_CLASS),
        #[cfg(feature = "chrono")]
        Kind::Timestamp => Some(&CEL_TIMESTAMP_CLASS),
        _ => None,
    }
}

fn type_from_class(cls: *const CelClass) -> Option<Type> {
    if cls == (&CEL_INT_CLASS as *const CelClass) {
        return Some(INT_TYPE.to_owned());
    }
    if cls == (&CEL_UINT_CLASS as *const CelClass) {
        return Some(UINT_TYPE.to_owned());
    }
    if cls == (&CEL_DOUBLE_CLASS as *const CelClass) {
        return Some(DOUBLE_TYPE.to_owned());
    }
    if cls == (&CEL_BOOL_CLASS as *const CelClass) {
        return Some(BOOL_TYPE.to_owned());
    }
    if cls == (&CEL_STRING_CLASS as *const CelClass) {
        return Some(STRING_TYPE.to_owned());
    }
    if cls == (&CEL_BYTES_CLASS as *const CelClass) {
        return Some(BYTES_TYPE.to_owned());
    }
    if cls == (&CEL_NULL_CLASS as *const CelClass) {
        return Some(NULL_TYPE.to_owned());
    }
    if cls == (&CEL_LIST_CLASS as *const CelClass) {
        return Some(LIST_TYPE.to_owned());
    }
    if cls == (&CEL_MAP_CLASS as *const CelClass) {
        return Some(MAP_TYPE.to_owned());
    }
    if cls == (&CEL_TYPE_CLASS as *const CelClass) {
        return Some(TYPE_TYPE.to_owned());
    }
    if cls == (&CEL_OPTIONAL_CLASS as *const CelClass) {
        return Some(OPTIONAL_TYPE.to_owned());
    }
    if cls == (&CEL_OPAQUE_CLASS as *const CelClass) {
        return Some(crate::common::types::Type::new_opaque_type("opaque"));
    }
    #[cfg(feature = "chrono")]
    if cls == (&CEL_DURATION_CLASS as *const CelClass) {
        return Some(DURATION_TYPE.to_owned());
    }
    #[cfg(feature = "chrono")]
    if cls == (&CEL_TIMESTAMP_CLASS as *const CelClass) {
        return Some(TIMESTAMP_TYPE.to_owned());
    }
    None
}

fn intern_list(list: &ListRef) -> CelRef {
    match list.storage() {
        ListStorage::Ints(values) => {
            let start = list.window_start();
            let end = start + list.len();
            new_list_ints(&values[start..end]) as CelRef
        }
        ListStorage::Object(_)
            if list.window_start() == 0 && list.len() == list.storage().len() =>
        {
            if let Some(ints) = object_list_as_ints(list) {
                return new_list_ints(&ints) as CelRef;
            }
            let mut items = Vec::with_capacity(list.len());
            for elt in list.iter() {
                items.push(value_to_ref(&elt).unwrap_or_else(|_| new_null() as CelRef));
            }
            new_list(&items) as CelRef
        }
        _ => {
            let host = intern_host_any(Box::new(list.clone()));
            new_list_window(host, 0, list.len() as i64) as CelRef
        }
    }
}

/// All-int object lists become an int column. Empty lists and the first
/// non-int stop the scan and keep object storage.
fn object_list_as_ints(list: &ListRef) -> Option<Vec<i64>> {
    if list.is_empty() {
        return None;
    }
    let mut ints = Vec::with_capacity(list.len());
    for elt in list.iter() {
        ints.push(value_as_int(&elt)?);
    }
    Some(ints)
}

fn value_as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Int(i) => Some(*i),
        Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::Int => {
            Some(unsafe { (*w.cast::<W_IntObject>()).intval })
        }
        _ => None,
    }
}

unsafe fn list_from_ref(w: CelRef) -> Result<ListRef, ConvertError> {
    let leaf = &*w.cast::<W_ListObject>();
    match leaf.strategy {
        ListStrategy::Object => {
            let n = leaf.length as usize;
            let base = items_block_items_base(leaf.items);
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("list"));
            }
            let mut items = Vec::with_capacity(n);
            let start = leaf.start as usize;
            for i in 0..n {
                items.push(unsafe { ref_to_value(*base.add(start + i))? });
            }
            Ok(ListRef::from(items))
        }
        ListStrategy::Ints => {
            if leaf.storage.is_null() {
                return Ok(ListRef::from(Vec::new()));
            }
            let col = &*leaf.storage.cast::<W_IntColumn>();
            let start = leaf.start as usize;
            let n = leaf.length as usize;
            if col.data.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("list"));
            }
            let ints = unsafe { std::slice::from_raw_parts(col.data.add(start), n) };
            Ok(ListRef::whole(Arc::new(ListStorage::Ints(ints.to_vec()))))
        }
        ListStrategy::Window => {
            host_list_ref(opaque_host_index(leaf.storage)).ok_or(ConvertError::Corrupt("list"))
        }
    }
}

/// Look up a string field on an interned map, including record-row strategy.
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
pub unsafe fn interned_map_lookup_string(w: CelRef, field: &str) -> Option<CelRef> {
    let leaf = &*w.cast::<W_MapObject>();
    match leaf.strategy {
        MapStrategy::Object => super::object::map_lookup_string(w, field),
        MapStrategy::Record => {
            let map = host_map(opaque_host_index(leaf.storage))?;
            intern_leaf(map.get(&Key::from(field))?.as_ref())
        }
    }
}

/// Item `index` of an interned list, including int-column and window strategies.
///
/// # Safety
///
/// `w` is a live [`W_ListObject`].
pub unsafe fn interned_list_get(w: CelRef, index: i64) -> Option<CelRef> {
    let leaf = &*w.cast::<W_ListObject>();
    if leaf.strategy != ListStrategy::Window {
        return super::object::list_get(w, index);
    }
    if index < 0 || index >= leaf.length {
        return None;
    }
    let list = host_list_ref(opaque_host_index(leaf.storage))?;
    intern_leaf(&list.get(index as usize)?)
}

fn intern_map(map: &Map) -> Result<CelRef, ConvertError> {
    match map.storage() {
        MapStorage::Object(_) => Ok(new_map(&map_pairs(map)?) as CelRef),
        MapStorage::Record { .. } => {
            let host = intern_host_any(Box::new(map.clone()));
            Ok(new_map_record(host, map.len() as i64) as CelRef)
        }
    }
}

unsafe fn map_from_ref(w: CelRef) -> Result<Map, ConvertError> {
    let leaf = &*w.cast::<W_MapObject>();
    match leaf.strategy {
        MapStrategy::Object => {
            let n = leaf.length as usize;
            let base = items_block_items_base(leaf.items);
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("map"));
            }
            let mut entries = HashMap::with_capacity(n);
            for i in 0..n {
                let key = unsafe { ref_to_key(*base.add(2 * i))? };
                let value = unsafe { ref_to_value(*base.add(2 * i + 1))? };
                entries.insert(key, value);
            }
            Ok(Map::object(Arc::new(entries)))
        }
        MapStrategy::Record => {
            host_map(opaque_host_index(leaf.storage)).ok_or(ConvertError::Corrupt("map"))
        }
    }
}

fn host_map(idx: i64) -> Option<Map> {
    super::heap::with_heap(|h| {
        h.with_host(idx, |any| any.downcast_ref::<Map>().cloned())
            .flatten()
    })
}

fn intern_host_any(host: Box<dyn std::any::Any>) -> CelRef {
    let idx = super::heap::with_heap(|h| h.push_host(host));
    new_opaque(new_type(&CEL_OPAQUE_CLASS) as CelRef, idx) as CelRef
}

fn host_list_ref(idx: i64) -> Option<ListRef> {
    super::heap::with_heap(|h| {
        h.with_host(idx, |any| any.downcast_ref::<ListRef>().cloned())
            .flatten()
    })
}

/// Park `host` in this thread's heap table and return a [`W_OpaqueObject`].
fn intern_host_opaque(host: &Arc<dyn Opaque>) -> CelRef {
    let idx = super::heap::with_heap(|h| h.push_host(Box::new(host.clone())));
    new_opaque(new_type(&CEL_OPAQUE_CLASS) as CelRef, idx) as CelRef
}

/// The host parked at `idx`, if that slot still holds an [`Opaque`].
pub(crate) fn host_opaque(idx: i64) -> Option<Arc<dyn Opaque>> {
    super::heap::with_heap(|h| {
        h.with_host(idx, |any| any.downcast_ref::<Arc<dyn Opaque>>().cloned())
            .flatten()
    })
}

/// Structural equality of two host-table slots. Sequential borrows so the
/// table's `RefCell` is not held twice.
pub(crate) fn opaque_hosts_equal(a: i64, b: i64) -> bool {
    match (host_opaque(a), host_opaque(b)) {
        (Some(left), Some(right)) => left.opaque_eq(right.as_ref()),
        _ => false,
    }
}

fn key_to_ref(k: &Key) -> CelRef {
    match k {
        Key::Int(i) => new_int(*i) as CelRef,
        Key::Uint(u) => new_uint(*u) as CelRef,
        Key::Bool(b) => new_bool(*b) as CelRef,
        Key::String(s) => new_string(s) as CelRef,
    }
}

fn map_pairs(map: &Map) -> Result<Vec<(CelRef, CelRef)>, ConvertError> {
    match map.storage() {
        MapStorage::Object(entries) => {
            let mut pairs = Vec::with_capacity(entries.len());
            for (k, v) in entries.iter() {
                pairs.push((key_to_ref(k), value_to_ref(v)?));
            }
            Ok(pairs)
        }
        // Record is a batch encoding, not a second class-family strategy:
        // explode it at this boundary into the same interleaved pairs.
        MapStorage::Record { .. } => {
            let exploded = map.to_hashmap();
            let mut pairs = Vec::with_capacity(exploded.len());
            for (k, v) in exploded.iter() {
                pairs.push((key_to_ref(k), value_to_ref(v)?));
            }
            Ok(pairs)
        }
    }
}

fn string_from_leaf(leaf: &W_StringObject) -> Result<Arc<String>, ConvertError> {
    let n = leaf.byte_len as usize;
    let base = unsafe { bytes_base(leaf.chars) };
    if base.is_null() && n != 0 {
        return Err(ConvertError::Corrupt("string"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(base, n) };
    let s = std::str::from_utf8(bytes).map_err(|_| ConvertError::Corrupt("string"))?;
    Ok(Arc::new(s.to_string()))
}

#[cfg(feature = "structs")]
fn string_from_ref(w: CelRef) -> Result<Arc<String>, ConvertError> {
    if w.is_null() || unsafe { w_type(w) } != &CEL_STRING_CLASS {
        return Err(ConvertError::Corrupt("string"));
    }
    string_from_leaf(unsafe { &*w.cast::<W_StringObject>() })
}

fn ref_to_key(w: CelRef) -> Result<Key, ConvertError> {
    match unsafe { ref_to_value(w) }? {
        Value::Int(i) => Ok(Key::Int(i)),
        Value::UInt(u) => Ok(Key::Uint(u)),
        Value::Bool(b) => Ok(Key::Bool(b)),
        Value::String(s) => Ok(Key::String(s)),
        _ => Err(ConvertError::Corrupt("map key")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(v: Value) -> Value {
        let w = value_to_ref(&v).expect("to_ref");
        unsafe { ref_to_value(w) }.expect("to_value")
    }

    #[test]
    fn public_scalars_roundtrip() {
        for v in [
            Value::Int(-3),
            Value::UInt(9),
            Value::Float(1.5),
            Value::Bool(true),
            Value::Bool(false),
            Value::Null,
            Value::String(Arc::new("hi".into())),
            Value::Bytes(Arc::new(b"xy".to_vec())),
        ] {
            assert_eq!(roundtrip(v.clone()), v, "{v:?}");
        }
    }

    #[test]
    fn intern_leaf_is_the_prebuilt_for_bool_null_and_small_int() {
        assert_eq!(
            intern_leaf(&Value::Bool(true)),
            Some(new_bool(true) as CelRef)
        );
        assert_eq!(intern_leaf(&Value::Null), Some(new_null() as CelRef));
        assert_eq!(intern_leaf(&Value::Int(3)), Some(new_int(3) as CelRef));
        assert!(intern_leaf(&Value::String(Arc::new("x".into()))).is_some());
        let list = Value::List(ListRef::from(vec![Value::Int(1)]));
        assert!(intern_leaf(&list).is_some());
        assert!(intern_leaf(&Value::UInt(3)).is_some());
        assert!(intern_leaf(&Value::Float(1.5)).is_some());
        let object = Value::Map(Map::object(Arc::new(
            [(Key::String(Arc::new("a".into())), Value::Int(1))]
                .into_iter()
                .collect(),
        )));
        assert!(intern_leaf(&object).is_some());
        let schema = Arc::new(crate::objects::RecordSchema::new(
            vec![Key::String(Arc::new("a".into()))],
            vec![crate::objects::ValueColumn::Scalar {
                bank: crate::objects::ScalarBank::Int,
                words: Arc::from([1i64]),
            }],
        ));
        let record = Value::Map(Map::record(schema, 0));
        let interned_record = intern_leaf(&record).expect("record-row intern");
        assert_eq!(unsafe { w_kind(interned_record) }, CelKind::Map);
        assert_eq!(roundtrip(record.clone()), record);
        let ints = Value::List(ListRef::whole(Arc::new(ListStorage::Ints(vec![1, 2, 3]))));
        let interned_ints = intern_leaf(&ints).expect("ints list intern");
        assert_eq!(unsafe { w_kind(interned_ints) }, CelKind::List);
        assert_eq!(roundtrip(ints.clone()), ints);
        assert_eq!(Value::int(3).kind(), CelKind::Int);
        assert_eq!(Value::int(3), Value::Int(3));
        assert_eq!(Value::bool(true), Value::Bool(true));
        assert_eq!(Value::null(), Value::Null);
        let none = Value::Opaque(Arc::new(OptionalValue::none()));
        assert!(intern_leaf(&none).is_some());
        let int_ty = Value::Opaque(Arc::new(TypeValue::new(INT_TYPE.to_owned())));
        let interned_ty = intern_leaf(&int_ty).expect("type intern");
        assert_eq!(unsafe { w_kind(interned_ty) }, CelKind::Type);
        assert_eq!(roundtrip(int_ty.clone()), int_ty);
        let some = Value::Opaque(Arc::new(OptionalValue::of(Value::Int(4))));
        assert!(intern_leaf(&some).is_some());
        let host = Value::Opaque(Arc::new(HostId(7)));
        let interned_host = intern_leaf(&host).expect("leftover opaque intern");
        assert_eq!(unsafe { w_kind(interned_host) }, CelKind::Opaque);
        assert_eq!(roundtrip(host.clone()), host);
        assert_eq!(
            unsafe { crate::runtime::binop::values_equal(interned_host, interned_host) },
            true
        );
        let also = intern_leaf(&Value::Opaque(Arc::new(HostId(7)))).unwrap();
        assert!(unsafe { crate::runtime::binop::values_equal(interned_host, also) });
        let other = intern_leaf(&Value::Opaque(Arc::new(HostId(8)))).unwrap();
        assert!(!unsafe { crate::runtime::binop::values_equal(interned_host, other) });
        #[cfg(feature = "chrono")]
        {
            assert!(intern_leaf(&Value::Duration(chrono::Duration::seconds(1))).is_some());
            assert!(intern_leaf(&Value::Timestamp(
                chrono::DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z").unwrap()
            ))
            .is_some());
        }
    }

    #[test]
    fn interned_string_and_list_add_roundtrip() {
        let hello = intern_leaf(&Value::String(Arc::new("he".into()))).unwrap();
        let lo = intern_leaf(&Value::String(Arc::new("llo".into()))).unwrap();
        let joined = unsafe { crate::runtime::binop::cel_add(hello, lo) };
        assert_eq!(
            unsafe { ref_to_value(joined) }.unwrap(),
            Value::String(Arc::new("hello".into()))
        );
        let a = intern_leaf(&Value::List(ListRef::from(vec![Value::Int(1)]))).unwrap();
        let b = intern_leaf(&Value::List(ListRef::from(vec![Value::Int(2)]))).unwrap();
        let cat = unsafe { crate::runtime::binop::cel_add(a, b) };
        assert_eq!(
            unsafe { ref_to_value(cat) }.unwrap(),
            Value::List(ListRef::from(vec![Value::Int(1), Value::Int(2)]))
        );
        let ua = intern_leaf(&Value::UInt(2)).unwrap();
        let ub = intern_leaf(&Value::UInt(3)).unwrap();
        assert_eq!(
            unsafe { ref_to_value(crate::runtime::binop::cel_add(ua, ub)) }.unwrap(),
            Value::UInt(5)
        );
        let fa = intern_leaf(&Value::Float(1.5)).unwrap();
        let fb = intern_leaf(&Value::Float(2.25)).unwrap();
        assert_eq!(
            unsafe { ref_to_value(crate::runtime::binop::cel_add(fa, fb)) }.unwrap(),
            Value::Float(3.75)
        );
        let ma = intern_leaf(&object_map()).unwrap();
        let mb = intern_leaf(&object_map()).unwrap();
        assert_eq!(
            unsafe { ref_to_value(crate::runtime::binop::cel_equals(ma, mb)) }.unwrap(),
            Value::Bool(true)
        );
        let none = intern_leaf(&Value::Opaque(Arc::new(OptionalValue::none()))).unwrap();
        let also_none = intern_leaf(&Value::Opaque(Arc::new(OptionalValue::none()))).unwrap();
        assert_eq!(
            unsafe { ref_to_value(crate::runtime::binop::cel_equals(none, also_none)) }.unwrap(),
            Value::Bool(true)
        );
        #[cfg(feature = "chrono")]
        {
            let d1 = intern_leaf(&Value::Duration(chrono::Duration::seconds(1))).unwrap();
            let d2 = intern_leaf(&Value::Duration(chrono::Duration::seconds(2))).unwrap();
            assert_eq!(
                unsafe { ref_to_value(crate::runtime::binop::cel_add(d1, d2)) }.unwrap(),
                Value::Duration(chrono::Duration::seconds(3))
            );
        }
    }

    #[derive(Debug, Eq, PartialEq)]
    struct HostId(u64);

    impl crate::objects::Opaque for HostId {
        fn runtime_type_name(&self) -> &str {
            "example.HostId"
        }
    }

    fn object_map() -> Value {
        Value::Map(Map::object(Arc::new(
            [(Key::String(Arc::new("k".into())), Value::Int(1))]
                .into_iter()
                .collect(),
        )))
    }

    #[test]
    fn public_list_and_optional_roundtrip() {
        let list = Value::List(ListRef::from(vec![Value::Int(1), Value::Int(2)]));
        assert_eq!(roundtrip(list.clone()), list);
        let none = Value::Opaque(Arc::new(OptionalValue::none()));
        assert_eq!(roundtrip(none.clone()), none);
        let some = Value::Opaque(Arc::new(OptionalValue::of(Value::Int(4))));
        assert_eq!(roundtrip(some.clone()), some);
    }

    #[test]
    fn public_map_roundtrip() {
        use crate::objects::{RecordSchema, ScalarBank, ValueColumn};

        let empty = Value::Map(Map::object(Arc::new(HashMap::new())));
        assert_eq!(roundtrip(empty.clone()), empty);

        let mut entries = HashMap::new();
        entries.insert(Key::Int(1), Value::String(Arc::new("one".into())));
        entries.insert(Key::Uint(2), Value::Int(2));
        entries.insert(Key::Bool(true), Value::Bool(false));
        entries.insert(Key::String(Arc::new("k".into())), Value::UInt(3));
        let object = Value::Map(Map::object(Arc::new(entries)));
        assert_eq!(roundtrip(object.clone()), object);

        let schema = Arc::new(RecordSchema::new(
            vec![Key::from("a"), Key::Int(7)],
            vec![
                ValueColumn::Scalar {
                    bank: ScalarBank::Int,
                    words: Arc::from([42i64]),
                },
                ValueColumn::Scalar {
                    bank: ScalarBank::Bool,
                    words: Arc::from([1i64]),
                },
            ],
        ));
        let record = Value::Map(Map::record(schema, 0));
        assert_eq!(roundtrip(record.clone()), record);
    }

    #[cfg(feature = "structs")]
    #[test]
    fn public_struct_roundtrip() {
        let mut s = CelStruct::new("cel.Problem".into());
        s.add_field_value("answer".into(), Value::Int(42));
        s.add_field_value("solved".into(), Value::Bool(true));
        let v = Value::Struct(Arc::new(s));
        let back = roundtrip(v.clone());
        match (v, back) {
            (Value::Struct(a), Value::Struct(b)) => assert_eq!(*a, *b),
            _ => panic!("expected struct"),
        }

        let empty = Value::Struct(Arc::new(CelStruct::new("cel.Empty".into())));
        let back = roundtrip(empty.clone());
        match (empty, back) {
            (Value::Struct(a), Value::Struct(b)) => assert_eq!(*a, *b),
            _ => panic!("expected empty struct"),
        }
    }

    #[cfg(feature = "chrono")]
    #[test]
    fn public_temporal_roundtrip() {
        let d = Value::Duration(chrono::Duration::seconds(3));
        assert_eq!(roundtrip(d.clone()), d);
        let ts = Value::Timestamp(
            chrono::DateTime::from_timestamp(1_700_000_000, 0)
                .unwrap()
                .with_timezone(&chrono::FixedOffset::east_opt(3600).unwrap()),
        );
        assert_eq!(roundtrip(ts.clone()), ts);
    }

    #[test]
    fn from_arc_string_still_builds_the_public_value() {
        let v: Value = Arc::new("drop-in".to_string()).into();
        assert_eq!(v, Value::String(Arc::new("drop-in".into())));
        assert_eq!(roundtrip(v.clone()), v);
    }
}
