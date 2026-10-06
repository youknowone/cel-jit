//! Boundary between the public [`crate::Value`] enum and this class family.
//!
//! [`crate::Value`] is the cel drop-in. `Value::String` is an `Arc<str>`
//! (one allocation). Callers still construct `Value::Int`, match variants,
//! and bind `This<Arc<str>>` or `This<Arc<String>>`.
//! Evaluators that want a header-first object cross here, and only here.
//!
//! Leftover host opaques cross as [`super::object::W_OpaqueObject`], with the
//! `Arc<dyn Opaque>` parked in this thread's heap table (D12). Record-row
//! maps and column-window lists intern as strategy windows so the schema
//! stays shared.

use std::borrow::Cow;
use std::collections::HashMap;
use std::mem::MaybeUninit;
use std::sync::Arc;

use super::object::{
    mapdict_get, mapdict_layout_for_names, mapdict_name_at, new_bool, new_bytes, new_double,
    new_host_list, new_host_list_ints, new_host_list_window, new_int, new_map, new_map_mapdict,
    new_map_record, new_null, new_opaque, new_optional, new_optional_none, new_string, new_type,
    new_uint, opaque_host_index, prebuilt_int, prebuilt_type, prepare_mapdict_rows, w_kind, w_type,
    CelClass, CelKind, CelRef, ListStrategy, MapStrategy, W_BoolObject, W_BytesObject,
    W_DoubleObject, W_FloatColumn, W_HostListObject, W_IntColumn, W_IntObject, W_ListObject,
    W_MapObject, W_OptionalObject, W_StringObject, W_TypeObject, W_UIntObject, CEL_BOOL_CLASS,
    CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_HOST_LIST_CLASS, CEL_INT_CLASS, CEL_LIST_CLASS,
    CEL_MAP_CLASS, CEL_NULL_CLASS, CEL_OPAQUE_CLASS, CEL_OPTIONAL_CLASS, CEL_STRING_CLASS,
    CEL_TYPE_CLASS, CEL_UINT_CLASS, MAPDICT_MAX_ENTRIES,
};
use super::object_array::{
    bytes_base, float_words_base, int_words_base, items_block_items_base, items_capacity,
};
use crate::common::types::{
    Kind, Type, TypeValue, BOOL_TYPE, BYTES_TYPE, DOUBLE_TYPE, INT_TYPE, LIST_TYPE, MAP_TYPE,
    NULL_TYPE, OPTIONAL_TYPE, STRING_TYPE, TYPE_TYPE, UINT_TYPE,
};
use crate::objects::{
    map_get_by_key, map_has_exact_key, try_build_map, AsKeyRef, Key, KeyRef, ListRef, ListStorage,
    Map, MapStorage, Opaque, OptionalValue, PackedRecordBuf, RecordSchema, ScalarBank,
    ScalarRowsBuf, ValueColumn, ORDERED_SCAN_LIMIT,
};
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
    match v {
        Value::Interned(w) => interned_to_public(w),
        other => other,
    }
}

/// An interned immediate as a public scalar. No allocation.
#[inline]
pub(crate) fn interned_immediate(w: CelRef) -> Option<Value> {
    match unsafe { w_kind(w) } {
        CelKind::Int => Some(Value::Int(unsafe { (*w.cast::<W_IntObject>()).intval })),
        CelKind::UInt => Some(Value::UInt(unsafe { (*w.cast::<W_UIntObject>()).uintval })),
        CelKind::Double => Some(Value::Float(unsafe {
            (*w.cast::<W_DoubleObject>()).floatval
        })),
        CelKind::Bool => Some(Value::Bool(
            unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0,
        )),
        CelKind::Null => Some(Value::Null),
        _ => None,
    }
}

/// An interned container, string or bytes that still has its bind-time
/// public handle: one Arc clone, no rebuild.
#[inline]
pub(crate) fn interned_linked(w: CelRef) -> Option<Value> {
    unsafe {
        match w_kind(w) {
            CelKind::List => {
                if w_type(w) != &CEL_HOST_LIST_CLASS {
                    return None;
                }
                let leaf = &*w.cast::<W_HostListObject>();
                ListRef::clone_from_public(leaf.public).map(|buf| {
                    Value::List(ListRef::from_linked(
                        buf,
                        leaf.public_start,
                        leaf.public_len,
                    ))
                })
            }
            CelKind::Map => {
                let leaf = &*w.cast::<W_MapObject>();
                linked_public_map(leaf).map(Value::Map)
            }
            CelKind::Str => {
                let leaf = &*w.cast::<W_StringObject>();
                clone_linked_str(leaf.public, leaf.byte_len).map(Value::String)
            }
            CelKind::Bytes => {
                let leaf = &*w.cast::<W_BytesObject>();
                clone_arc(leaf.public as *const Vec<u8>).map(Value::Bytes)
            }
            _ => None,
        }
    }
}

/// Public form of `w`. Immediates and linked handles do not allocate;
/// everything else rebuilds through [`ref_to_value`].
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[inline]
pub fn interned_to_public(w: CelRef) -> Value {
    interned_immediate(w)
        .or_else(|| interned_linked(w))
        .unwrap_or_else(|| match unsafe { ref_to_value(w) } {
            Ok(Value::Interned(_)) => Value::Null,
            Ok(v) => v,
            Err(_) => Value::Null,
        })
}

/// The leaf [`intern_leaf`] returns without allocating: small ints
/// (`intobject.py` `PREBUILTINTFROM` / `PREBUILTINTTO`), bool, and null.
pub(crate) fn intern_prebuilt(v: &Value) -> Option<CelRef> {
    match v {
        Value::Int(i) => prebuilt_int(*i).map(|w| w as CelRef),
        Value::Bool(b) => Some(new_bool(*b) as CelRef),
        Value::Null => Some(new_null() as CelRef),
        _ => None,
    }
}

/// Intern `v` onto a class-family leaf.
///
/// A duration or timestamp whose instant does not fit i64 nanoseconds has no
/// leaf and stays public: `None` here is that case, not an error and not a
/// truncation.
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
            .map(|cls| prebuilt_type(cls) as CelRef)
            .or_else(|| Some(intern_host_opaque(opaque))),
        #[cfg(feature = "chrono")]
        Value::Duration(d) => d.num_nanoseconds().map(|n| new_duration(n) as CelRef),
        #[cfg(feature = "chrono")]
        Value::Timestamp(ts) => ts
            .timestamp_nanos_opt()
            .map(|n| new_timestamp(n, i64::from(ts.offset().local_minus_utc())) as CelRef),
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
// Default inlining leaves this call in the list fill. The hot match has to
// be in the element loop; the rare arms stay in [`ref_to_value_cold`].
#[inline(always)]
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
            Ok(Value::Bytes(bytes_from_leaf(leaf)?))
        }
        CelKind::List if class == &CEL_LIST_CLASS || class == &CEL_HOST_LIST_CLASS => {
            Ok(Value::List(unsafe { list_from_ref(w)? }))
        }
        CelKind::Map if class == &CEL_MAP_CLASS => Ok(Value::Map(unsafe { map_from_ref(w)? })),
        _ => unsafe { ref_to_value_cold(w, kind, class) },
    }
}

/// Struct, optional, type, opaque, timestamp, duration. Out of line so the
/// scalar/string/list/map arms of [`ref_to_value`] stay small enough to inline.
#[cold]
#[inline(never)]
unsafe fn ref_to_value_cold(
    w: CelRef,
    kind: CelKind,
    class: *const CelClass,
) -> Result<Value, ConvertError> {
    match kind {
        #[cfg(feature = "structs")]
        CelKind::Struct if class == &CEL_STRUCT_CLASS => {
            let leaf = unsafe { &*w.cast::<W_StructObject>() };
            let name = string_from_ref(leaf.name as CelRef)?;
            let n = leaf.length as usize;
            let base = unsafe { items_block_items_base(leaf.fields) };
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("struct"));
            }
            let mut s = CelStruct::new(name.as_ref().to_owned());
            for i in 0..n {
                let fname = unsafe { string_from_ref(*base.add(2 * i))? };
                let fval = unsafe { ref_to_value(*base.add(2 * i + 1))? };
                s.add_field_value(fname.as_ref().to_owned(), fval);
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
    if std::ptr::eq(cls, &CEL_INT_CLASS) {
        return Some(INT_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_UINT_CLASS) {
        return Some(UINT_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_DOUBLE_CLASS) {
        return Some(DOUBLE_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_BOOL_CLASS) {
        return Some(BOOL_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_STRING_CLASS) {
        return Some(STRING_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_BYTES_CLASS) {
        return Some(BYTES_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_NULL_CLASS) {
        return Some(NULL_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_LIST_CLASS) || std::ptr::eq(cls, &CEL_HOST_LIST_CLASS) {
        return Some(LIST_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_MAP_CLASS) {
        return Some(MAP_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_TYPE_CLASS) {
        return Some(TYPE_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_OPTIONAL_CLASS) {
        return Some(OPTIONAL_TYPE.to_owned());
    }
    if std::ptr::eq(cls, &CEL_OPAQUE_CLASS) {
        return Some(crate::common::types::Type::new_opaque_type("opaque"));
    }
    #[cfg(feature = "chrono")]
    if std::ptr::eq(cls, &CEL_DURATION_CLASS) {
        return Some(DURATION_TYPE.to_owned());
    }
    #[cfg(feature = "chrono")]
    if std::ptr::eq(cls, &CEL_TIMESTAMP_CLASS) {
        return Some(TIMESTAMP_TYPE.to_owned());
    }
    None
}

fn intern_list(list: &ListRef) -> CelRef {
    if let Some(values) = list.ints_slice() {
        let start = list.window_start();
        let end = start + list.len();
        new_host_list_ints(&values[start..end]) as CelRef
    } else if list.is_whole() && list.object_slice().is_some() {
        if let Some(ints) = object_list_as_ints(list) {
            return new_host_list_ints(&ints) as CelRef;
        }
        let mut items = Vec::with_capacity(list.len());
        for elt in list.iter() {
            items.push(value_to_ref(&elt).unwrap_or_else(|_| new_null() as CelRef));
        }
        new_host_list(&items) as CelRef
    } else {
        let host = intern_host_any(Box::new(list.clone()));
        new_host_list_window(host, 0, list.len() as i64) as CelRef
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

/// Record a non-owning link from an interned leaf back to the public
/// handle it was wrapped from. The caller keeps that handle alive.
pub(crate) fn link_public_handle(w: CelRef, value: &Value) {
    unsafe {
        match value {
            Value::List(list) => {
                if w_type(w) != &CEL_HOST_LIST_CLASS {
                    return;
                }
                let leaf = &mut *w.cast::<W_HostListObject>();
                leaf.public = list.public_ptr();
                leaf.public_start = list.window_start() as u32;
                leaf.public_len = list.len() as u32;
            }
            Value::Map(map) => {
                if w_kind(w) != CelKind::Map {
                    return;
                }
                let leaf = &mut *w.cast::<W_MapObject>();
                if let Some(arc) = map.object_arc() {
                    leaf.public = Arc::as_ptr(arc) as *const ();
                    leaf.public_kind = MAP_PUBLIC_OBJECT;
                    leaf.public_len = 0;
                } else if let Some(arc) = map.entries_arc() {
                    let n = arc.len();
                    if n > u32::MAX as usize {
                        return;
                    }
                    leaf.public = Arc::as_ptr(arc) as *const (Key, Value) as *const ();
                    leaf.public_kind = MAP_PUBLIC_ENTRIES;
                    leaf.public_len = n as u32;
                }
            }
            Value::String(s) => {
                if w_kind(w) != CelKind::Str {
                    return;
                }
                let leaf = &mut *w.cast::<W_StringObject>();
                debug_assert_eq!(leaf.byte_len, s.len() as i64);
                leaf.public = crate::objects::arc_str_thin(s);
            }
            Value::Bytes(b) => {
                if w_kind(w) != CelKind::Bytes {
                    return;
                }
                (*w.cast::<W_BytesObject>()).public = Arc::as_ptr(b) as *const ();
            }
            _ => {}
        }
    }
}

/// [`link_public_handle`] on `w`, then on every interned child that still
/// names a borrowed public value. Bind retains the parent; a window list
/// intern's children on get and links them there.
pub(crate) fn link_public_tree(w: CelRef, value: &Value) {
    if w.is_null() {
        return;
    }
    link_public_handle(w, value);
    unsafe { link_public_children(w, value) };
}

/// # Safety
///
/// `w` is the interned form of `value`, allocated on this thread's heap.
unsafe fn link_public_children(w: CelRef, value: &Value) {
    match value {
        Value::List(list) => {
            if unsafe { w_kind(w) } != CelKind::List {
                return;
            }
            let leaf = unsafe { &*w.cast::<W_ListObject>() };
            match leaf.strategy {
                ListStrategy::Object | ListStrategy::Strs => {}
                ListStrategy::Ints
                | ListStrategy::Floats
                | ListStrategy::Window
                | ListStrategy::Size => return,
            }
            let n = list.len() as i64;
            let mut i = 0i64;
            while i < n {
                if let (Some(elt), Some(child)) = (list.get(i as usize), interned_list_get(w, i)) {
                    link_public_tree(child, &elt);
                }
                i += 1;
            }
        }
        Value::Map(map) => {
            if unsafe { w_kind(w) } != CelKind::Map {
                return;
            }
            for (k, v) in map.iter() {
                // Record rows yield `Cow::Owned`; those values die at the
                // end of the iteration, so a link would dangle.
                let Cow::Borrowed(inner) = v else {
                    continue;
                };
                if let Some(child) = interned_map_get(w, k.as_keyref()) {
                    link_public_tree(child, inner);
                }
            }
        }
        _ => {}
    }
}

/// Increment the strong count of the `Arc` behind `ptr` and return a new
/// handle. `ptr` is `Arc::as_ptr` of a live allocation.
#[inline]
unsafe fn clone_arc<T>(ptr: *const T) -> Option<Arc<T>> {
    if ptr.is_null() {
        return None;
    }
    unsafe {
        Arc::increment_strong_count(ptr);
        Some(Arc::from_raw(ptr))
    }
}

/// Integers of an object-strategy list, or `None` if any element is not an
/// interned int. The public form of a literal `[1, 2, 3]` is this buffer
/// rather than a `Vec<Value>` rebuilt element by element.
#[inline(never)]
unsafe fn interned_object_list_ints(leaf: &W_ListObject) -> Option<ListRef> {
    let n = leaf.length as usize;
    if n == 0 {
        return None;
    }
    let base = items_block_items_base(leaf.items);
    if base.is_null() {
        return None;
    }
    let start = leaf.start as usize;
    // A nested list or map as the first element is the common miss: do not
    // allocate the int buffer before that is known.
    let first = unsafe { *base.add(start) };
    if first.is_null() || unsafe { w_kind(first) } != CelKind::Int {
        return None;
    }
    ListRef::try_fill_ints::<()>(n, |i| {
        let item = unsafe { *base.add(start + i) };
        if item.is_null() || unsafe { w_kind(item) } != CelKind::Int {
            return Err(());
        }
        Ok(Some(unsafe { (*item.cast::<W_IntObject>()).intval }))
    })
    .ok()
}

/// Items-block list → one [`Value`] per element.
///
/// Object lists and string-strategy lists share this path.
unsafe fn values_from_items(leaf: &W_ListObject) -> Result<ListRef, ConvertError> {
    let n = leaf.length as usize;
    let base = items_block_items_base(leaf.items);
    if base.is_null() && n != 0 {
        return Err(ConvertError::Corrupt("list"));
    }
    let start = leaf.start as usize;
    ListRef::try_fill_values(n, |i| {
        Ok(Some(unsafe { ref_to_value(*base.add(start + i))? }))
    })
}

/// Fixed allocations of one owned [`ListStorage::Record`] list: the key
/// vector, the column vector, the schema `Arc`, and the storage `Arc`. Every
/// scalar column shares one more word-buffer `Arc`. Used when the row has
/// more than [`ORDERED_SCAN_LIMIT`] fields.
const RECORD_FIXED_ALLOCS: usize = 4;
/// The shared word buffer behind an owned record list's columns.
const RECORD_WORD_ALLOCS: usize = 1;
/// A packed record list is one block: header, field names, and words.
const PACKED_RECORD_ALLOCS: usize = 1;
/// A shared scalar-row list is one block: header, per-row offsets, and words.
const COLUMN_FIXED_ALLOCS: usize = 1;

/// One row of a list-of-lists, when every element is the same scalar bank.
enum RowSpan {
    Empty,
    Words { bank: ScalarBank, len: usize },
}

/// A list of same-shaped scalar maps finishes as one packed record block.
/// A list of scalar lists finishes as one scalar-row block. Either form is
/// built only when it allocates less than one container per element
/// ([`try_build_map`] / [`ListRef::try_fill_ints`]). Fewer than three rows
/// stay separate: one nested list is two allocations, and the length guard
/// keeps a two-row list on that path.
///
/// # Safety
///
/// `leaf` is a live object-strategy [`W_ListObject`].
unsafe fn try_coalesce_object_list(leaf: &W_ListObject) -> Option<ListRef> {
    if leaf.length < 3 || leaf.start < 0 {
        return None;
    }
    let n = leaf.length as usize;
    let start = leaf.start as usize;
    let base = unsafe { items_block_items_base(leaf.items) };
    if base.is_null() {
        return None;
    }
    let cap = unsafe { items_capacity(leaf.items) };
    if start.checked_add(n).is_none_or(|end| end > cap) {
        return None;
    }
    let first = unsafe { *base.add(start) };
    if first.is_null() {
        return None;
    }
    match unsafe { w_kind(first) } {
        CelKind::Map => unsafe { try_coalesce_record_list(base, start, n) },
        CelKind::List => unsafe { try_coalesce_column_list(base, start, n) },
        _ => None,
    }
}

/// Object-strategy map entries, or `None` when the leaf is not one fresh
/// object map. A non-null `public` is already a shared table; rebuilding it
/// as a record would allocate the schema the link does not.
///
/// # Safety
///
/// `w` is a live reference or null.
unsafe fn object_map_entries(w: CelRef) -> Option<(*mut CelRef, usize)> {
    if w.is_null() || unsafe { w_kind(w) } != CelKind::Map || unsafe { w_type(w) } != &CEL_MAP_CLASS
    {
        return None;
    }
    let leaf = unsafe { &*w.cast::<W_MapObject>() };
    if leaf.strategy != MapStrategy::Object || !leaf.public.is_null() || leaf.length < 0 {
        return None;
    }
    let n = leaf.length as usize;
    if n == 0 {
        return Some((core::ptr::null_mut(), 0));
    }
    let base = unsafe { items_block_items_base(leaf.items) };
    if base.is_null() || n.saturating_mul(2) > unsafe { items_capacity(leaf.items) } {
        return None;
    }
    Some((base, n))
}

unsafe fn map_field(base: *mut CelRef, field: usize, value: bool) -> Option<CelRef> {
    let mut index = field.checked_mul(2)?;
    if value {
        index = index.checked_add(1)?;
    }
    Some(unsafe { *base.add(index) })
}

unsafe fn same_interned_key(a: CelRef, b: CelRef) -> bool {
    if a == b {
        return true;
    }
    if a.is_null() || b.is_null() {
        return false;
    }
    let kind = unsafe { w_kind(a) };
    if kind != unsafe { w_kind(b) } {
        return false;
    }
    unsafe {
        match kind {
            CelKind::Int => (*a.cast::<W_IntObject>()).intval == (*b.cast::<W_IntObject>()).intval,
            CelKind::UInt => {
                (*a.cast::<W_UIntObject>()).uintval == (*b.cast::<W_UIntObject>()).uintval
            }
            CelKind::Bool => {
                (*a.cast::<W_BoolObject>()).boolval == (*b.cast::<W_BoolObject>()).boolval
            }
            CelKind::Str => interned_string_eq(a, b),
            _ => false,
        }
    }
}

unsafe fn interned_string_eq(a: CelRef, b: CelRef) -> bool {
    let left = unsafe { &*a.cast::<W_StringObject>() };
    let right = unsafe { &*b.cast::<W_StringObject>() };
    if left.byte_len != right.byte_len || left.byte_len < 0 {
        return false;
    }
    let n = left.byte_len as usize;
    if n == 0 {
        return true;
    }
    let left_bytes = unsafe { bytes_base(left.chars) };
    let right_bytes = unsafe { bytes_base(right.chars) };
    if left_bytes.is_null() || right_bytes.is_null() {
        return false;
    }
    unsafe {
        std::slice::from_raw_parts(left_bytes, n) == std::slice::from_raw_parts(right_bytes, n)
    }
}

/// Allocations [`string_from_leaf`] makes for this key. `None` is a key that
/// leaf cannot turn into a [`Key`]. A linked `public` arc is a refcount.
/// An unlinked string is one `Arc<str>`.
unsafe fn fresh_key_allocs(w: CelRef) -> Option<usize> {
    if w.is_null() {
        return None;
    }
    unsafe {
        match w_kind(w) {
            CelKind::Int | CelKind::UInt | CelKind::Bool => Some(0),
            CelKind::Str => {
                let leaf = &*w.cast::<W_StringObject>();
                if !leaf.public.is_null() {
                    return Some(0);
                }
                if leaf.byte_len < 0 {
                    return None;
                }
                let n = leaf.byte_len as usize;
                if n == 0 {
                    return Some(1);
                }
                let base = bytes_base(leaf.chars);
                if base.is_null() {
                    return None;
                }
                let bytes = std::slice::from_raw_parts(base, n);
                if std::str::from_utf8(bytes).is_err() {
                    return None;
                }
                Some(1)
            }
            _ => None,
        }
    }
}

unsafe fn scalar_word(w: CelRef) -> Option<(ScalarBank, i64)> {
    if w.is_null() {
        return None;
    }
    let class = unsafe { w_type(w) };
    unsafe {
        match w_kind(w) {
            CelKind::Int if class == &CEL_INT_CLASS => {
                Some((ScalarBank::Int, (*w.cast::<W_IntObject>()).intval))
            }
            CelKind::UInt if class == &CEL_UINT_CLASS => {
                Some((ScalarBank::UInt, (*w.cast::<W_UIntObject>()).uintval as i64))
            }
            CelKind::Bool if class == &CEL_BOOL_CLASS => {
                Some((ScalarBank::Bool, (*w.cast::<W_BoolObject>()).boolval))
            }
            CelKind::Double if class == &CEL_DOUBLE_CLASS => Some((
                ScalarBank::Float,
                (*w.cast::<W_DoubleObject>()).floatval.to_bits() as i64,
            )),
            _ => None,
        }
    }
}

/// # Safety
///
/// `base` is the items pointer of a live object list. `start..start + n`
/// is in range and every slot there is non-null.
unsafe fn try_coalesce_record_list(base: *mut CelRef, start: usize, n: usize) -> Option<ListRef> {
    let (base0, fields) = unsafe { object_map_entries(*base.add(start)) }?;
    // No column means [`RecordSchema::rows`] is 0, so the row count is lost.
    if fields == 0 {
        return None;
    }
    let mut key_allocs = 0usize;
    for field in 0..fields {
        let key = unsafe { map_field(base0, field, false) }?;
        for prev in 0..field {
            let earlier = unsafe { map_field(base0, prev, false) }?;
            if unsafe { same_interned_key(key, earlier) } {
                return None;
            }
        }
        key_allocs = key_allocs.saturating_add(unsafe { fresh_key_allocs(key) }?);
        let (bank0, _) = unsafe { scalar_word(map_field(base0, field, true)?) }?;
        for row in 1..n {
            let (row_base, row_fields) = unsafe { object_map_entries(*base.add(start + row)) }?;
            if row_fields != fields {
                return None;
            }
            let row_key = unsafe { map_field(row_base, field, false) }?;
            if unsafe { !same_interned_key(key, row_key) } {
                return None;
            }
            let (bank, _) = unsafe { scalar_word(map_field(row_base, field, true)?) }?;
            if bank != bank0 {
                return None;
            }
        }
    }
    // [`try_build_map`] is one `Arc` at or below [`ORDERED_SCAN_LIMIT`] and a
    // table plus an `Arc` above it. The outer list is one more buffer.
    let per_row: usize = if fields <= ORDERED_SCAN_LIMIT { 1 } else { 2 };
    let current = per_row.saturating_mul(n).saturating_add(1);
    let structural = if fields <= ORDERED_SCAN_LIMIT {
        PACKED_RECORD_ALLOCS
    } else {
        RECORD_FIXED_ALLOCS.saturating_add(RECORD_WORD_ALLOCS)
    };
    let record_cost = structural.saturating_add(key_allocs);
    if record_cost >= current {
        return None;
    }
    unsafe { build_record_list(base, start, n, base0, fields) }
}

unsafe fn build_record_list(
    base: *mut CelRef,
    start: usize,
    n: usize,
    base0: *mut CelRef,
    fields: usize,
) -> Option<ListRef> {
    let total = n.checked_mul(fields)?;
    if fields > ORDERED_SCAN_LIMIT || u32::try_from(n).is_err() {
        return unsafe { build_owned_record_list(base, start, n, base0, fields, total) };
    }
    unsafe { build_packed_record_list(base, start, n, base0, fields, total) }
}

unsafe fn build_packed_record_list(
    base: *mut CelRef,
    start: usize,
    n: usize,
    base0: *mut CelRef,
    fields: usize,
    total: usize,
) -> Option<ListRef> {
    if n.checked_mul(fields) != Some(total) {
        return None;
    }
    let mut block = PackedRecordBuf::alloc(n, fields)?;
    for field in 0..fields {
        let key = unsafe { map_field(base0, field, false) }?;
        block.push_key(ref_to_key(key).ok()?);
        let (bank, _) = unsafe { scalar_word(map_field(base0, field, true)?) }?;
        block.set_column(field, bank);
        let origin = field * n;
        for row in 0..n {
            let (row_base, row_fields) = unsafe { object_map_entries(*base.add(start + row)) }?;
            if row_fields != fields {
                return None;
            }
            let (row_bank, word) = unsafe { scalar_word(map_field(row_base, field, true)?) }?;
            if row_bank != bank {
                return None;
            }
            block.write_word(origin + row, word);
        }
    }
    Some(block.finish())
}

unsafe fn build_owned_record_list(
    base: *mut CelRef,
    start: usize,
    n: usize,
    base0: *mut CelRef,
    fields: usize,
    total: usize,
) -> Option<ListRef> {
    let mut keys = Vec::with_capacity(fields);
    for field in 0..fields {
        let key = unsafe { map_field(base0, field, false) }?;
        keys.push(ref_to_key(key).ok()?);
    }
    let mut uninit: Arc<[MaybeUninit<i64>]> = Arc::new_uninit_slice(total);
    {
        let slot = Arc::get_mut(&mut uninit).expect("unique");
        for field in 0..fields {
            let (bank, _) = unsafe { scalar_word(map_field(base0, field, true)?) }?;
            let origin = field * n;
            for row in 0..n {
                let (row_base, row_fields) = unsafe { object_map_entries(*base.add(start + row)) }?;
                if row_fields != fields {
                    return None;
                }
                let (row_bank, word) = unsafe { scalar_word(map_field(row_base, field, true)?) }?;
                if row_bank != bank {
                    return None;
                }
                slot[origin + row].write(word);
            }
        }
    }
    let words = unsafe { uninit.assume_init() };
    let mut columns = Vec::with_capacity(fields);
    for field in 0..fields {
        let (bank, _) = unsafe { scalar_word(map_field(base0, field, true)?) }?;
        columns.push(ValueColumn::range(bank, Arc::clone(&words), field * n, n));
    }
    let schema = Arc::new(RecordSchema::new(keys, columns));
    Some(ListRef::whole(Arc::new(ListStorage::Record(schema))))
}

/// # Safety
///
/// `base` is the items pointer of a live object list. `start..start + n`
/// is in range.
unsafe fn try_coalesce_column_list(base: *mut CelRef, start: usize, n: usize) -> Option<ListRef> {
    let mut bank = None;
    let mut total = 0usize;
    // The outer list is one buffer. Each non-empty inner list is one more
    // ([`ListRef::try_fill_ints`] / [`ListRef::try_fill_values`]). An empty
    // inner list is the static empty header and allocates nothing.
    let mut current = 1usize;
    for row in 0..n {
        match unsafe { row_span(*base.add(start + row)) }? {
            RowSpan::Empty => {}
            RowSpan::Words {
                bank: row_bank,
                len,
            } => {
                match bank {
                    None => bank = Some(row_bank),
                    Some(prev) if prev == row_bank => {}
                    Some(_) => return None,
                }
                total = total.checked_add(len)?;
                current = current.saturating_add(1);
            }
        }
    }
    let bank = bank?;
    if total == 0 || COLUMN_FIXED_ALLOCS >= current {
        return None;
    }
    unsafe { build_column_list(base, start, n, bank, total) }
}

unsafe fn build_column_list(
    base: *mut CelRef,
    start: usize,
    n: usize,
    bank: ScalarBank,
    total: usize,
) -> Option<ListRef> {
    if u32::try_from(n).is_err() || u32::try_from(total).is_err() {
        return None;
    }
    let mut block = ScalarRowsBuf::alloc(n, total, bank)?;
    let mut at = 0usize;
    for row in 0..n {
        let len = unsafe { row_len(*base.add(start + row)) };
        let end = at.checked_add(len)?;
        if end > total {
            return None;
        }
        let written =
            unsafe { write_row_words(*base.add(start + row), bank, block.word_slot(at, len)) };
        let written = written?;
        if written != len {
            return None;
        }
        block.set_row(row, at as u32, len as u32);
        at = end;
    }
    if at != total {
        return None;
    }
    Some(block.finish())
}

unsafe fn row_len(w: CelRef) -> usize {
    match unsafe { row_span(w) } {
        Some(RowSpan::Empty) => 0,
        Some(RowSpan::Words { len, .. }) => len,
        None => 0,
    }
}

/// # Safety
///
/// `w` is a live reference or null.
unsafe fn row_span(w: CelRef) -> Option<RowSpan> {
    if w.is_null()
        || unsafe { w_kind(w) } != CelKind::List
        || unsafe { w_type(w) } != &CEL_LIST_CLASS
    {
        return None;
    }
    let leaf = unsafe { &*w.cast::<W_ListObject>() };
    if leaf.length < 0 || leaf.start < 0 {
        return None;
    }
    let n = leaf.length as usize;
    let start = leaf.start as usize;
    match leaf.strategy {
        ListStrategy::Size => (n == 0).then_some(RowSpan::Empty),
        ListStrategy::Ints => unsafe { word_column_span(leaf, start, n, true) },
        ListStrategy::Floats => unsafe { word_column_span(leaf, start, n, false) },
        ListStrategy::Object => unsafe { object_scalar_span(leaf, start, n) },
        ListStrategy::Strs | ListStrategy::Window => None,
    }
}

unsafe fn word_column_span(
    leaf: &W_ListObject,
    start: usize,
    n: usize,
    ints: bool,
) -> Option<RowSpan> {
    if n == 0 {
        return Some(RowSpan::Empty);
    }
    if leaf.storage.is_null() {
        return None;
    }
    let (data_null, col_len, bank) = if ints {
        let col = unsafe { &*leaf.storage.cast::<W_IntColumn>() };
        (col.data.is_null(), col.length, ScalarBank::Int)
    } else {
        let col = unsafe { &*leaf.storage.cast::<W_FloatColumn>() };
        (col.data.is_null(), col.length, ScalarBank::Float)
    };
    if data_null || col_len < 0 {
        return None;
    }
    let end = start.checked_add(n)?;
    if end > col_len as usize {
        return None;
    }
    Some(RowSpan::Words { bank, len: n })
}

unsafe fn object_scalar_span(leaf: &W_ListObject, start: usize, n: usize) -> Option<RowSpan> {
    if n == 0 {
        return Some(RowSpan::Empty);
    }
    let base = unsafe { items_block_items_base(leaf.items) };
    if base.is_null() {
        return None;
    }
    let cap = unsafe { items_capacity(leaf.items) };
    if start.checked_add(n).is_none_or(|end| end > cap) {
        return None;
    }
    let (bank, _) = unsafe { scalar_word(*base.add(start)) }?;
    for i in 1..n {
        let (row_bank, _) = unsafe { scalar_word(*base.add(start + i)) }?;
        if row_bank != bank {
            return None;
        }
    }
    Some(RowSpan::Words { bank, len: n })
}

unsafe fn write_row_words(
    w: CelRef,
    bank: ScalarBank,
    dest: &mut [MaybeUninit<i64>],
) -> Option<usize> {
    match unsafe { row_span(w) }? {
        RowSpan::Empty => Some(0),
        RowSpan::Words {
            bank: row_bank,
            len,
        } => {
            if row_bank != bank || len > dest.len() {
                return None;
            }
            unsafe { copy_row_words(w, &mut dest[..len]) }?;
            Some(len)
        }
    }
}

unsafe fn copy_row_words(w: CelRef, dest: &mut [MaybeUninit<i64>]) -> Option<()> {
    let leaf = unsafe { &*w.cast::<W_ListObject>() };
    let n = dest.len();
    if n == 0 {
        return Some(());
    }
    let start = leaf.start as usize;
    match leaf.strategy {
        ListStrategy::Ints => {
            let col = unsafe { &*leaf.storage.cast::<W_IntColumn>() };
            let src = unsafe { int_words_base(col.data) };
            if src.is_null() {
                return None;
            }
            unsafe {
                std::ptr::copy_nonoverlapping(src.add(start), dest.as_mut_ptr().cast::<i64>(), n);
            }
            Some(())
        }
        ListStrategy::Floats => {
            let col = unsafe { &*leaf.storage.cast::<W_FloatColumn>() };
            let src = unsafe { float_words_base(col.data) };
            if src.is_null() {
                return None;
            }
            for (slot, word) in dest.iter_mut().zip(0..n) {
                let bits = unsafe { (*src.add(start + word)).to_bits() } as i64;
                slot.write(bits);
            }
            Some(())
        }
        ListStrategy::Object => {
            let base = unsafe { items_block_items_base(leaf.items) };
            if base.is_null() {
                return None;
            }
            for (slot, index) in dest.iter_mut().zip(0..n) {
                let (_, word) = unsafe { scalar_word(*base.add(start + index)) }?;
                slot.write(word);
            }
            Some(())
        }
        ListStrategy::Strs | ListStrategy::Window | ListStrategy::Size => None,
    }
}

unsafe fn list_from_ref(w: CelRef) -> Result<ListRef, ConvertError> {
    if unsafe { w_type(w) } == &CEL_HOST_LIST_CLASS {
        let host = &*w.cast::<W_HostListObject>();
        if let Some(buf) = unsafe { ListRef::clone_from_public(host.public) } {
            return Ok(ListRef::from_linked(
                buf,
                host.public_start,
                host.public_len,
            ));
        }
    }
    let leaf = &*w.cast::<W_ListObject>();
    match leaf.strategy {
        ListStrategy::Object => {
            if let Some(ints) = interned_object_list_ints(leaf) {
                return Ok(ints);
            }
            if let Some(coalesced) = unsafe { try_coalesce_object_list(leaf) } {
                return Ok(coalesced);
            }
            values_from_items(leaf)
        }
        ListStrategy::Floats => {
            if leaf.storage.is_null() {
                return Ok(ListRef::from(Vec::new()));
            }
            let col = &*leaf.storage.cast::<super::object::W_FloatColumn>();
            let start = leaf.start as usize;
            let n = leaf.length as usize;
            if n == 0 || col.data.is_null() {
                if n != 0 {
                    return Err(ConvertError::Corrupt("list"));
                }
                return Ok(ListRef::from(Vec::new()));
            }
            let mut bits = Vec::with_capacity(n);
            let mut i = 0;
            while i < n {
                bits.push(unsafe {
                    (*crate::runtime::object_array::float_words_base(col.data).add(start + i))
                        .to_bits() as i64
                });
                i += 1;
            }
            Ok(ListRef::whole(Arc::new(ListStorage::Column(
                ValueColumn::Scalar {
                    bank: ScalarBank::Float,
                    words: Arc::from(bits),
                },
            ))))
        }
        ListStrategy::Strs => values_from_items(leaf),
        ListStrategy::Ints => {
            if leaf.storage.is_null() {
                return Ok(ListRef::from(Vec::new()));
            }
            let col = &*leaf.storage.cast::<W_IntColumn>();
            let start = leaf.start as usize;
            let n = leaf.length as usize;
            if n == 0 || col.data.is_null() {
                if n != 0 {
                    return Err(ConvertError::Corrupt("list"));
                }
                return ListRef::try_fill_ints::<ConvertError>(0, |_| Ok(None));
            }
            let ints = unsafe {
                std::slice::from_raw_parts(
                    crate::runtime::object_array::int_words_base(col.data).add(start),
                    n,
                )
            };
            ListRef::try_fill_ints(n, |i| Ok(Some(ints[i])))
        }
        ListStrategy::Window => {
            host_list_ref(opaque_host_index(leaf.storage)).ok_or(ConvertError::Corrupt("list"))
        }
        ListStrategy::Size => ListRef::try_fill_values(0, |_| Ok(None)),
    }
}

/// An interned leaf as a map [`KeyRef`], or `None` if it cannot be a map key.
///
/// # Safety
///
/// `w` is a live value.
pub unsafe fn interned_as_keyref<'a>(w: CelRef) -> Option<KeyRef<'a>> {
    match w_kind(w) {
        CelKind::Int => Some(KeyRef::Int((*w.cast::<W_IntObject>()).intval)),
        CelKind::UInt => Some(KeyRef::Uint((*w.cast::<W_UIntObject>()).uintval)),
        CelKind::Bool => Some(KeyRef::Bool((*w.cast::<W_BoolObject>()).boolval != 0)),
        CelKind::Str => super::object::string_as_str(w).map(KeyRef::String),
        _ => None,
    }
}

unsafe fn interned_object_get_exact(w: CelRef, needle: KeyRef<'_>) -> Option<CelRef> {
    let leaf = &*w.cast::<W_MapObject>();
    if leaf.strategy != MapStrategy::Object {
        return None;
    }
    let n = leaf.length as usize;
    let base = items_block_items_base(leaf.items);
    if base.is_null() {
        return None;
    }
    let mut i = 0;
    while i < n {
        let k = *base.add(2 * i);
        if interned_key_eq(k, needle) {
            return Some(*base.add(2 * i + 1));
        }
        i += 1;
    }
    None
}

#[inline]
unsafe fn interned_key_eq(w: CelRef, needle: KeyRef<'_>) -> bool {
    match needle {
        KeyRef::Int(i) => w_kind(w) == CelKind::Int && (*w.cast::<W_IntObject>()).intval == i,
        KeyRef::Uint(u) => w_kind(w) == CelKind::UInt && (*w.cast::<W_UIntObject>()).uintval == u,
        KeyRef::Bool(b) => {
            w_kind(w) == CelKind::Bool && ((*w.cast::<W_BoolObject>()).boolval != 0) == b
        }
        KeyRef::String(s) => super::object::string_eq_str(w, s),
    }
}

/// Look up `needle` on an interned map with the walker's exact-then-cross-type
/// rule ([`map_get_by_key`]).
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
pub unsafe fn interned_map_get(w: CelRef, needle: KeyRef<'_>) -> Option<CelRef> {
    let leaf = &*w.cast::<W_MapObject>();
    match leaf.strategy {
        MapStrategy::Object => map_get_by_key(|k| interned_object_get_exact(w, k), needle),
        MapStrategy::Mapdict => match needle {
            KeyRef::String(s) => mapdict_get(leaf, s.as_bytes()),
            _ => None,
        },
        MapStrategy::Record => {
            let map = host_map(opaque_host_index(leaf.storage))?;
            intern_leaf(map.get(&needle)?.as_ref())
        }
    }
}

/// `in` on an interned map: exact `Key` ([`map_has_exact_key`]), the same
/// probe [`crate::objects::Map::contains_key`] uses.
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
#[inline(never)]
pub unsafe fn interned_map_contains(w: CelRef, needle: KeyRef<'_>) -> bool {
    let leaf = &*w.cast::<W_MapObject>();
    match leaf.strategy {
        MapStrategy::Object => {
            map_has_exact_key(|k| interned_object_get_exact(w, k).is_some(), needle)
        }
        MapStrategy::Mapdict => match needle {
            KeyRef::String(s) => mapdict_get(leaf, s.as_bytes()).is_some(),
            _ => false,
        },
        MapStrategy::Record => host_map(opaque_host_index(leaf.storage))
            .map(|m| m.contains_key(&needle))
            .unwrap_or(false),
    }
}

/// Look up a string field on an interned map, including record-row strategy.
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
pub unsafe fn interned_map_lookup_string(w: CelRef, field: &str) -> Option<CelRef> {
    interned_map_get(w, KeyRef::String(field))
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
    let v = list.get(index as usize)?;
    let child = intern_leaf(&v)?;
    link_public_handle(child, &v);
    Some(child)
}

fn intern_map(map: &Map) -> Result<CelRef, ConvertError> {
    match map.storage() {
        MapStorage::Object(_) | MapStorage::Entries(_) => {
            if let Some(w) = intern_string_key_mapdict(map)? {
                return Ok(w);
            }
            Ok(new_map(&map_pairs(map)?) as CelRef)
        }
        MapStorage::Record { .. } => {
            let host = intern_host_any(Box::new(map.clone()));
            Ok(new_map_record(host, map.len() as i64) as CelRef)
        }
    }
}

/// Mapdict when every key is a string and the map is small enough.
///
/// `Ok(None)` means "use the object strategy". A value that fails to intern
/// is `Err` and does not fall through: the caller would intern it again.
fn intern_string_key_mapdict(map: &Map) -> Result<Option<CelRef>, ConvertError> {
    if map.len() > MAPDICT_MAX_ENTRIES {
        return Ok(None);
    }
    // One pass. A `HashMap` does not walk in the same order twice, and the
    // order is per-instance random.
    let mut rows: Vec<(&str, &Value)> = Vec::with_capacity(map.len());
    for (k, v) in map.iter() {
        let Key::String(s) = k else {
            return Ok(None);
        };
        // `Cow::as_ref` would borrow the loop temporary. Object and entries
        // maps yield `Borrowed`; an owned value stays on the object strategy.
        let std::borrow::Cow::Borrowed(value) = v else {
            return Ok(None);
        };
        rows.push((s.as_ref(), value));
    }
    // Attribute order is the byte order of the key, so the same key set
    // shares one layout (`mapdict.py` `_get_new_attr`).
    if !prepare_mapdict_rows(&mut rows) {
        return Ok(None);
    }
    let mut names = Vec::with_capacity(rows.len());
    let mut values = Vec::with_capacity(rows.len());
    for (name, value) in &rows {
        let w = value_to_ref(value)?;
        names.push(name.as_bytes());
        values.push(w);
    }
    let layout = mapdict_layout_for_names(&names);
    Ok(Some(new_map_mapdict(layout, &values) as CelRef))
}

/// `W_MapObject::public` addresses a [`HashMap<Key, Value>`].
const MAP_PUBLIC_OBJECT: u32 = 0;
/// `W_MapObject::public` is the data pointer of an `Arc<[(Key, Value)]>`;
/// [`W_MapObject::public_len`] is the slice length.
const MAP_PUBLIC_ENTRIES: u32 = 1;

unsafe fn linked_public_map(leaf: &W_MapObject) -> Option<Map> {
    if leaf.public.is_null() {
        return None;
    }
    if leaf.public_kind == MAP_PUBLIC_ENTRIES {
        clone_arc_slice(leaf.public as *const (Key, Value), leaf.public_len as usize)
            .map(Map::from_linked_entries)
    } else {
        clone_arc(leaf.public as *const HashMap<Key, Value>).map(Map::object)
    }
}

/// Rebuild an `Arc<[T]>` from the data pointer stored in a leaf and the
/// leaf's length word.
///
/// # Safety
///
/// `data` is `Arc::as_ptr` of a live `Arc<[T]>` of length `len`, and that
/// `Arc` outlives this increment.
unsafe fn clone_arc_slice<T>(data: *const T, len: usize) -> Option<Arc<[T]>> {
    if data.is_null() {
        return None;
    }
    let fat = std::ptr::slice_from_raw_parts(data, len);
    unsafe {
        Arc::increment_strong_count(fat);
        Some(Arc::from_raw(fat))
    }
}

unsafe fn map_from_ref(w: CelRef) -> Result<Map, ConvertError> {
    let leaf = &*w.cast::<W_MapObject>();
    if let Some(map) = linked_public_map(leaf) {
        return Ok(map);
    }
    match leaf.strategy {
        MapStrategy::Object => {
            let n = leaf.length as usize;
            let base = items_block_items_base(leaf.items);
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("map"));
            }
            try_build_map::<_, false>(n, |i| {
                let key = unsafe { ref_to_key(*base.add(2 * i))? };
                let value = unsafe { ref_to_value(*base.add(2 * i + 1))? };
                Ok(Some((key, value)))
            })
        }
        MapStrategy::Mapdict => {
            let n_i = leaf.length;
            if n_i < 0 {
                return Err(ConvertError::Corrupt("map"));
            }
            let n = n_i as usize;
            let base = items_block_items_base(leaf.items);
            if n != 0 && (base.is_null() || unsafe { items_capacity(leaf.items) } < n) {
                return Err(ConvertError::Corrupt("map"));
            }
            try_build_map::<_, false>(n, |i| {
                let name =
                    mapdict_name_at(leaf.layout, i as i64).ok_or(ConvertError::Corrupt("map"))?;
                let text = std::str::from_utf8(name).map_err(|_| ConvertError::Corrupt("map"))?;
                let key = Key::String(Arc::from(text));
                let value = unsafe { ref_to_value(*base.add(i))? };
                Ok(Some((key, value)))
            })
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
        MapStorage::Object(_) | MapStorage::Entries(_) => {
            let mut pairs = Vec::with_capacity(map.len());
            for (k, v) in map.iter() {
                pairs.push((key_to_ref(k), value_to_ref(v.as_ref())?));
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

fn string_from_leaf(leaf: &W_StringObject) -> Result<Arc<str>, ConvertError> {
    if let Some(s) = clone_linked_str(leaf.public, leaf.byte_len) {
        return Ok(s);
    }
    let n = leaf.byte_len as usize;
    let base = unsafe { bytes_base(leaf.chars) };
    if base.is_null() && n != 0 {
        return Err(ConvertError::Corrupt("string"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(base, n) };
    let s = std::str::from_utf8(bytes).map_err(|_| ConvertError::Corrupt("string"))?;
    Ok(Arc::from(s))
}

/// Rebuild the `Arc<str>` a leaf's `public` word names. `len` is the leaf's
/// `byte_len`, which is the fat pointer's metadata.
fn clone_linked_str(data: *const (), len: i64) -> Option<Arc<str>> {
    if data.is_null() || len < 0 {
        return None;
    }
    let fat = unsafe { crate::objects::str_ptr_from_thin(data.cast::<u8>(), len as usize) };
    unsafe {
        Arc::increment_strong_count(fat);
        Some(Arc::from_raw(fat))
    }
}

fn bytes_from_leaf(leaf: &W_BytesObject) -> Result<Arc<Vec<u8>>, ConvertError> {
    if let Some(b) = unsafe { clone_arc(leaf.public as *const Vec<u8>) } {
        return Ok(b);
    }
    let n = leaf.length as usize;
    let base = unsafe { bytes_base(leaf.data) };
    if base.is_null() && n != 0 {
        return Err(ConvertError::Corrupt("bytes"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(base, n) };
    Ok(Arc::new(bytes.to_vec()))
}

#[cfg(feature = "structs")]
fn string_from_ref(w: CelRef) -> Result<Arc<str>, ConvertError> {
    if w.is_null() || unsafe { w_type(w) } != &CEL_STRING_CLASS {
        return Err(ConvertError::Corrupt("string"));
    }
    string_from_leaf(unsafe { &*w.cast::<W_StringObject>() })
}

fn ref_to_key(w: CelRef) -> Result<Key, ConvertError> {
    if w.is_null() {
        return Err(ConvertError::Corrupt("map key"));
    }
    match unsafe { w_kind(w) } {
        CelKind::Int => Ok(Key::Int(unsafe { (*w.cast::<W_IntObject>()).intval })),
        CelKind::UInt => Ok(Key::Uint(unsafe { (*w.cast::<W_UIntObject>()).uintval })),
        CelKind::Bool => Ok(Key::Bool(
            unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0,
        )),
        CelKind::Str => {
            let leaf = unsafe { &*w.cast::<W_StringObject>() };
            Ok(Key::String(string_from_leaf(leaf)?))
        }
        _ => Err(ConvertError::Corrupt("map key")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::MapEntries;
    use crate::runtime::object::{map_try_insert, new_list, new_list_ints, new_map_with_capacity};

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
            Value::String(Arc::from("hi")),
            Value::Bytes(Arc::new(b"xy".to_vec())),
        ] {
            assert_eq!(roundtrip(v.clone()), v, "{v:?}");
        }
    }

    #[test]
    fn interned_object_list_of_ints_matches_public_ints() {
        let w = new_list(&[
            new_int(1) as CelRef,
            new_int(2) as CelRef,
            new_int(3) as CelRef,
        ]);
        let v = interned_to_public(w as CelRef);
        assert_eq!(
            v,
            Value::List(ListRef::whole(Arc::new(ListStorage::Ints(vec![1, 2, 3]))))
        );
    }

    #[test]
    fn nested_object_list_finish_rebuilds_each_inner() {
        let inner = new_list(&[new_int(15) as CelRef]);
        let outer = new_list(&[inner as CelRef]);
        let v = interned_to_public(outer as CelRef);
        assert_eq!(
            v,
            Value::list(vec![Value::List(ListRef::whole(Arc::new(
                ListStorage::Ints(vec![15])
            )))])
        );
        let Value::List(outer) = v else {
            panic!("list");
        };
        let Value::List(inner) = outer.get(0).expect("row") else {
            panic!("inner list");
        };
        assert!(
            inner.is_ints(),
            "one nested list stays an int buffer, not a shared column"
        );
    }

    /// [`link_public_handle`] stores `Arc::as_ptr` and does not keep the
    /// allocation alive. The returned arc has to outlive `interned_to_public`.
    fn linked_string(text: &str) -> (CelRef, Arc<str>) {
        let arc = Arc::from(text);
        let leaf = new_string(text) as CelRef;
        link_public_handle(leaf, &Value::String(Arc::clone(&arc)));
        (leaf, arc)
    }

    fn int_rows(key: CelRef, n: i64) -> CelRef {
        let rows: Vec<CelRef> = (1..=n)
            .map(|i| new_map(&[(key, new_int(i) as CelRef)]) as CelRef)
            .collect();
        new_list(&rows) as CelRef
    }

    fn assert_record_rows(value: &Value, n: usize) {
        let Value::List(list) = value else {
            panic!("list");
        };
        assert!(list.is_record(), "rows share one record schema");
        assert_eq!(list.len(), n);
    }

    fn assert_entry_rows(value: &Value) {
        let Value::List(list) = value else {
            panic!("list");
        };
        let Value::Map(map) = list.get(0).expect("row") else {
            panic!("map");
        };
        assert!(
            matches!(map.storage(), MapStorage::Entries(_)),
            "short or mixed map list stays one table per row: {map:?}"
        );
    }

    #[test]
    fn homogeneous_scalar_maps_finish_as_one_record() {
        let (key, key_arc) = linked_string("k");
        let value = interned_to_public(int_rows(key, 5));
        assert_record_rows(&value, 5);
        let expected: Vec<Value> = (1..=5)
            .map(|i| {
                Value::Map(
                    try_build_map::<(), false>(1, |_| {
                        Ok(Some((Key::String(Arc::clone(&key_arc)), Value::Int(i))))
                    })
                    .expect("map"),
                )
            })
            .collect();
        assert_eq!(value, Value::list(expected));
        let Value::List(list) = &value else {
            panic!("list");
        };
        let Value::Map(first) = list.get(0).expect("row") else {
            panic!("map");
        };
        assert_eq!(
            first.get(&Key::String(Arc::clone(&key_arc))).as_deref(),
            Some(&Value::Int(1))
        );
    }

    #[test]
    fn four_scalar_maps_finish_as_one_record() {
        let (key, key_arc) = linked_string("k");
        let value = interned_to_public(int_rows(key, 4));
        assert_record_rows(&value, 4);
        let Value::List(list) = &value else {
            panic!("list");
        };
        assert_eq!(list.len(), 4);
        let Value::Map(first) = list.get(0).expect("row") else {
            panic!("map");
        };
        assert_eq!(
            first.get(&Key::String(key_arc)).as_deref(),
            Some(&Value::Int(1))
        );
    }

    #[test]
    fn record_rows_keep_insertion_order() {
        let (key_c, arc_c) = linked_string("c");
        let (key_a, arc_a) = linked_string("a");
        let (key_b, arc_b) = linked_string("b");
        let rows: Vec<CelRef> = (0..7)
            .map(|i| {
                new_map(&[
                    (key_c, new_int(i) as CelRef),
                    (key_a, new_int(i + 10) as CelRef),
                    (key_b, new_int(i + 20) as CelRef),
                ]) as CelRef
            })
            .collect();
        let value = interned_to_public(new_list(&rows) as CelRef);
        assert_record_rows(&value, 7);
        let Value::List(list) = &value else {
            panic!("list");
        };
        let Value::Map(map) = list.get(3).expect("row") else {
            panic!("map");
        };
        assert!(matches!(map.storage(), MapStorage::Record { .. }));
        assert_eq!(map.len(), 3);
        let keys: Vec<&str> = map
            .iter()
            .map(|(key, _)| match key {
                Key::String(text) => text.as_ref(),
                other => panic!("string key, got {other:?}"),
            })
            .collect();
        assert_eq!(keys, [arc_c.as_ref(), arc_a.as_ref(), arc_b.as_ref()]);
        assert_eq!(
            map.get(&Key::String(arc_a)).as_deref(),
            Some(&Value::Int(13))
        );
    }

    #[test]
    fn mismatched_map_rows_stay_separate_tables() {
        let (key_k, arc_k) = linked_string("k");
        let (key_z, arc_z) = linked_string("z");
        let mut rows: Vec<CelRef> = (1..=6)
            .map(|i| new_map(&[(key_k, new_int(i) as CelRef)]) as CelRef)
            .collect();
        rows.push(new_map(&[(key_z, new_int(7) as CelRef)]) as CelRef);
        let value = interned_to_public(new_list(&rows) as CelRef);
        assert_entry_rows(&value);
        let Value::List(list) = &value else {
            panic!("list");
        };
        let Value::Map(first) = list.get(0).expect("row") else {
            panic!("map");
        };
        let Value::Map(last) = list.get(6).expect("row") else {
            panic!("map");
        };
        assert_eq!(
            first.get(&Key::String(arc_k)).as_deref(),
            Some(&Value::Int(1))
        );
        assert_eq!(
            last.get(&Key::String(arc_z)).as_deref(),
            Some(&Value::Int(7))
        );
    }

    #[test]
    fn duplicate_keys_and_non_scalar_values_stay_separate_tables() {
        let (key, key_arc) = linked_string("k");
        let dupes: Vec<CelRef> = (0..7)
            .map(|i| {
                new_map(&[(key, new_int(i) as CelRef), (key, new_int(i + 1) as CelRef)]) as CelRef
            })
            .collect();
        let duped = interned_to_public(new_list(&dupes) as CelRef);
        assert_entry_rows(&duped);
        let Value::List(list) = &duped else {
            panic!("list");
        };
        let Value::Map(map) = list.get(0).expect("row") else {
            panic!("map");
        };
        assert!(map.get(&Key::String(Arc::clone(&key_arc))).is_some());

        let nested: Vec<CelRef> = (0..7)
            .map(|_| new_map(&[(key, new_list(&[new_int(1) as CelRef]) as CelRef)]) as CelRef)
            .collect();
        assert_entry_rows(&interned_to_public(new_list(&nested) as CelRef));
    }

    fn assert_shared_windows(value: &Value, lens: &[usize], word: impl Fn(usize, usize) -> Value) {
        let Value::List(outer) = value else {
            panic!("list");
        };
        assert_eq!(outer.len(), lens.len());
        let mut rows = Vec::new();
        for (index, len) in lens.iter().copied().enumerate() {
            let Value::List(row) = outer.get(index).expect("row") else {
                panic!("inner list");
            };
            assert_eq!(row.len(), len);
            for at in 0..len {
                assert_eq!(row.get(at), Some(word(index, at)));
            }
            rows.push(row);
        }
        let first = &rows[0];
        for row in &rows[1..] {
            assert!(
                first.shares_storage_with(row),
                "inner lists are windows of one column"
            );
        }
    }

    #[test]
    fn three_int_lists_share_one_column() {
        let rows: Vec<CelRef> = (0..3)
            .map(|i| new_list(&[new_int(i) as CelRef, new_int(i + 10) as CelRef]) as CelRef)
            .collect();
        let value = interned_to_public(new_list(&rows) as CelRef);
        assert_shared_windows(&value, &[2, 2, 2], |row, at| {
            Value::Int(row as i64 + if at == 0 { 0 } else { 10 })
        });
    }

    #[test]
    fn two_int_lists_stay_separate_buffers() {
        let rows = [
            new_list_ints(&[1, 2]) as CelRef,
            new_list_ints(&[3, 4]) as CelRef,
        ];
        let value = interned_to_public(new_list(&rows) as CelRef);
        let Value::List(outer) = value else {
            panic!("list");
        };
        let Value::List(left) = outer.get(0).expect("row") else {
            panic!("inner");
        };
        let Value::List(right) = outer.get(1).expect("row") else {
            panic!("inner");
        };
        assert!(left.is_ints());
        assert!(right.is_ints());
        assert!(!left.shares_storage_with(&right));
        assert_eq!(left.get(1), Some(Value::Int(2)));
        assert_eq!(right.get(0), Some(Value::Int(3)));
    }

    #[test]
    fn int_columns_of_different_lengths_share_one_buffer() {
        let rows = [
            new_list_ints(&[1, 2]) as CelRef,
            new_list(&[]) as CelRef,
            new_list_ints(&[3, 4]) as CelRef,
            new_list_ints(&[5]) as CelRef,
        ];
        let value = interned_to_public(new_list(&rows) as CelRef);
        assert_shared_windows(&value, &[2, 0, 2, 1], |row, at| {
            let words = [1i64, 2, 3, 4, 5];
            let start = [0usize, 2, 2, 4][row];
            Value::Int(words[start + at])
        });
    }

    #[test]
    fn uint_and_float_rows_share_a_column_and_mixed_banks_do_not() {
        let uints: Vec<CelRef> = (0..3)
            .map(|i| new_list(&[new_uint(i as u64) as CelRef]) as CelRef)
            .collect();
        let value = interned_to_public(new_list(&uints) as CelRef);
        assert_shared_windows(&value, &[1, 1, 1], |row, _| Value::UInt(row as u64));

        let floats: Vec<CelRef> = [1.5f64, 2.5, 3.5]
            .into_iter()
            .map(|n| new_list(&[new_double(n) as CelRef]) as CelRef)
            .collect();
        let value = interned_to_public(new_list(&floats) as CelRef);
        assert_shared_windows(&value, &[1, 1, 1], |row, _| {
            Value::Float([1.5, 2.5, 3.5][row])
        });

        let mixed = [
            new_list_ints(&[1, 2]) as CelRef,
            new_list(&[new_bool(true) as CelRef, new_bool(false) as CelRef]) as CelRef,
            new_list_ints(&[3, 4]) as CelRef,
            new_list_ints(&[5, 6]) as CelRef,
        ];
        let value = interned_to_public(new_list(&mixed) as CelRef);
        let Value::List(outer) = value else {
            panic!("list");
        };
        let Value::List(first) = outer.get(0).expect("row") else {
            panic!("inner");
        };
        let Value::List(third) = outer.get(2).expect("row") else {
            panic!("inner");
        };
        assert!(!first.shares_storage_with(&third));
        let Value::List(bools) = outer.get(1).expect("row") else {
            panic!("inner");
        };
        assert_eq!(bools.get(0), Some(Value::Bool(true)));
    }

    #[test]
    fn intern_leaf_is_the_prebuilt_for_bool_null_and_small_int() {
        let int_ty = crate::common::types::r#type::type_ident("int").unwrap();
        let interned_int_ty = intern_leaf(&int_ty);
        assert_eq!(
            interned_int_ty,
            Some(prebuilt_type(&CEL_INT_CLASS) as CelRef)
        );
        assert_eq!(intern_leaf(&int_ty), interned_int_ty);
        assert_eq!(
            intern_leaf(&Value::Bool(true)),
            Some(new_bool(true) as CelRef)
        );
        assert_eq!(intern_leaf(&Value::Null), Some(new_null() as CelRef));
        assert_eq!(intern_leaf(&Value::Int(3)), Some(new_int(3) as CelRef));
        assert_eq!(
            intern_prebuilt(&Value::Int(3)),
            intern_leaf(&Value::Int(3)),
            "small ints are prebuilts; intern_leaf does not allocate"
        );
        assert_eq!(intern_prebuilt(&Value::Int(1000)), None);
        assert!(intern_leaf(&Value::String(Arc::from("x"))).is_some());
        let list = Value::List(ListRef::from(vec![Value::Int(1)]));
        assert!(intern_leaf(&list).is_some());
        assert!(intern_leaf(&Value::UInt(3)).is_some());
        assert!(intern_leaf(&Value::Float(1.5)).is_some());
        let object = Value::Map(Map::object(Arc::new(
            [(Key::String(Arc::from("a")), Value::Int(1))]
                .into_iter()
                .collect(),
        )));
        assert!(intern_leaf(&object).is_some());
        let schema = Arc::new(crate::objects::RecordSchema::new(
            vec![Key::String(Arc::from("a"))],
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
        assert!(unsafe { crate::runtime::binop::values_equal(interned_host, interned_host) });
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
        let hello = intern_leaf(&Value::String(Arc::from("he"))).unwrap();
        let lo = intern_leaf(&Value::String(Arc::from("llo"))).unwrap();
        let joined = unsafe { crate::runtime::binop::cel_add(hello, lo) };
        assert_eq!(
            unsafe { ref_to_value(joined) }.unwrap(),
            Value::String(Arc::from("hello"))
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
            [(Key::String(Arc::from("k")), Value::Int(1))]
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
        entries.insert(Key::Int(1), Value::String(Arc::from("one")));
        entries.insert(Key::Uint(2), Value::Int(2));
        entries.insert(Key::Bool(true), Value::Bool(false));
        entries.insert(Key::String(Arc::from("k")), Value::UInt(3));
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
        assert_eq!(v, Value::String(Arc::from("drop-in")));
        assert_eq!(roundtrip(v.clone()), v);
    }

    #[test]
    fn a_linked_public_list_unpacks_the_same_buffer() {
        let original = ListRef::from(vec![Value::Int(1), Value::Int(2)]);
        let value = Value::List(original.clone());
        let w = intern_leaf(&value).expect("intern");
        link_public_handle(w, &value);
        assert_eq!(unsafe { w_type(w) }, &CEL_HOST_LIST_CLASS as *const _);
        let back = unsafe { list_from_ref(w) }.expect("unpack");
        assert!(original.ptr_eq(&back));
    }

    #[test]
    fn nested_maps_in_a_linked_list_unpack_the_same_table() {
        let mut entries = HashMap::new();
        entries.insert(Key::String(Arc::from("a")), Value::Int(1));
        let original = Map::object(Arc::new(entries));
        let value = Value::list(vec![Value::Map(original.clone())]);
        let w = intern_leaf(&value).expect("intern");
        link_public_tree(w, &value);
        let child = unsafe { interned_list_get(w, 0) }.expect("elt");
        let back = unsafe { map_from_ref(child) }.expect("unpack");
        assert!(original.ptr_eq(&back));
    }

    #[test]
    fn nested_strings_in_a_linked_list_unpack_the_same_arc() {
        let original: Arc<str> = Arc::from("ab0");
        let value = Value::list(vec![Value::String(original.clone())]);
        let w = intern_leaf(&value).expect("intern");
        link_public_tree(w, &value);
        let child = unsafe { interned_list_get(w, 0) }.expect("elt");
        let back = string_from_leaf(unsafe { &*child.cast::<W_StringObject>() }).expect("unpack");
        assert!(Arc::ptr_eq(&original, &back));
    }

    #[test]
    fn a_vm_born_list_has_no_public_link() {
        let w = new_list(&[new_int(1) as CelRef]) as CelRef;
        assert_eq!(unsafe { w_type(w) }, &CEL_LIST_CLASS as *const _);
        assert_ne!(unsafe { w_type(w) }, &CEL_HOST_LIST_CLASS as *const _);
    }

    #[test]
    fn a_linked_public_map_unpacks_the_same_table() {
        let mut entries = HashMap::new();
        entries.insert(Key::String(Arc::from("a")), Value::Int(1));
        let original = Map::object(Arc::new(entries));
        let value = Value::Map(original.clone());
        let w = intern_leaf(&value).expect("intern");
        link_public_handle(w, &value);
        let back = unsafe { map_from_ref(w) }.expect("unpack");
        assert!(original.ptr_eq(&back));
    }

    #[test]
    fn a_linked_public_entries_map_unpacks_the_same_table() {
        let original = Map::entries(MapEntries::new(
            vec![(Key::String(Arc::from("a")), Value::Int(1))].into_boxed_slice(),
        ));
        let value = Value::Map(original.clone());
        let w = intern_leaf(&value).expect("intern");
        link_public_handle(w, &value);
        let back = unsafe { map_from_ref(w) }.expect("unpack");
        assert!(original.ptr_eq(&back));
    }

    #[test]
    fn linked_entries_map_survives_map_try_insert() {
        // A successful insert clears the public link, so copy-out rebuilds
        // from the interned items and includes the new pair rather than
        // returning the stale linked table.
        let original = Map::ordered(
            vec![
                (Key::String(Arc::from("a")), Value::Int(1)),
                (Key::String(Arc::from("b")), Value::Int(2)),
            ]
            .into_boxed_slice(),
        );
        let value = Value::Map(original.clone());
        let w = new_map_with_capacity(4) as CelRef;
        assert!(unsafe { map_try_insert(w, new_string("a") as CelRef, new_int(1) as CelRef,) });
        assert!(unsafe { map_try_insert(w, new_string("b") as CelRef, new_int(2) as CelRef,) });
        link_public_handle(w, &value);
        let leaf = unsafe { &*w.cast::<W_MapObject>() };
        assert_eq!(leaf.length, 2);
        assert_eq!(leaf.public_kind, MAP_PUBLIC_ENTRIES);
        assert_eq!(leaf.public_len, 2);
        assert!(unsafe { map_try_insert(w, new_string("c") as CelRef, new_int(3) as CelRef,) });
        let leaf = unsafe { &*w.cast::<W_MapObject>() };
        assert_eq!(leaf.length, 3, "insert grew the interned table");
        assert!(leaf.public.is_null(), "mutation cleared the public link");
        assert_eq!(leaf.public_len, 0);
        assert_eq!(leaf.public_kind, 0);
        let back = unsafe { map_from_ref(w) }.expect("unpack");
        assert!(
            !original.ptr_eq(&back),
            "copy-out is a rebuild, not the stale link"
        );
        assert_eq!(back.len(), 3);
        assert_eq!(
            back.get(&Key::String(Arc::from("c"))).as_deref(),
            Some(&Value::Int(3))
        );
    }

    #[test]
    fn map_copy_out_error_drops_a_written_opaque() {
        let spy: Arc<dyn Opaque> = Arc::new(HostId(7));
        let opaque_w = intern_leaf(&Value::Opaque(spy.clone())).expect("opaque intern");
        let before = Arc::strong_count(&spy);
        let w = new_map(&[
            (new_string("k") as CelRef, opaque_w),
            (
                new_list(&[new_int(0) as CelRef]) as CelRef,
                new_int(2) as CelRef,
            ),
        ]);
        let err = unsafe { map_from_ref(w as CelRef) };
        assert!(matches!(err, Err(ConvertError::Corrupt("map key"))));
        assert_eq!(
            Arc::strong_count(&spy),
            before,
            "the first pair must be dropped when a later key fails"
        );
    }

    #[test]
    fn a_linked_public_string_unpacks_the_same_arc() {
        let original: Arc<str> = Arc::from("hello");
        let value = Value::String(original.clone());
        let w = intern_leaf(&value).expect("intern");
        link_public_handle(w, &value);
        let back = string_from_leaf(unsafe { &*w.cast::<W_StringObject>() }).expect("unpack");
        assert!(Arc::ptr_eq(&original, &back));
    }
}
