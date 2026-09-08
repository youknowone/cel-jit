//! Boundary between the public [`crate::Value`] enum and this class family.
//!
//! [`crate::Value`] is the cel drop-in and does not change: callers still
//! construct `Value::Int`, match variants, and bind `This<Arc<String>>`.
//! Evaluators that want a header-first object cross here, and only here.
//!
//! Types this family has no leaf for — `map`, `struct`, and a non-optional
//! `opaque` — stay on the public enum. That is a missing leaf, not a signal
//! to widen [`crate::Value`].

use std::sync::Arc;

use super::object::{
    new_bool, new_bytes, new_double, new_int, new_list, new_null, new_optional, new_optional_none,
    new_string, new_uint, w_kind, w_type, CelKind, CelRef, W_BoolObject, W_BytesObject,
    W_DoubleObject, W_IntObject, W_ListObject, W_OptionalObject, W_StringObject, W_UIntObject,
    CEL_BOOL_CLASS, CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_INT_CLASS, CEL_LIST_CLASS,
    CEL_NULL_CLASS, CEL_OPTIONAL_CLASS, CEL_STRING_CLASS, CEL_UINT_CLASS,
};
use super::object_array::{bytes_base, items_block_items_base};
use crate::objects::{ListRef, OptionalValue};
use crate::Value;

#[cfg(feature = "chrono")]
use super::object::{
    new_duration, new_timestamp, W_DurationObject, W_TimestampObject, CEL_DURATION_CLASS,
    CEL_TIMESTAMP_CLASS,
};

/// Why a value could not cross the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertError {
    /// This family has no leaf for the public variant.
    Unsupported(&'static str),
    /// The `CelRef` is not a value this boundary can read back.
    Corrupt(&'static str),
}

/// Allocate the internal form of `v` on this thread's heap.
pub fn value_to_ref(v: &Value) -> Result<CelRef, ConvertError> {
    match v {
        Value::Int(i) => Ok(new_int(*i) as CelRef),
        Value::UInt(u) => Ok(new_uint(*u) as CelRef),
        Value::Float(f) => Ok(new_double(*f) as CelRef),
        Value::Bool(b) => Ok(new_bool(*b) as CelRef),
        Value::Null => Ok(new_null() as CelRef),
        Value::String(s) => Ok(new_string(s) as CelRef),
        Value::Bytes(b) => Ok(new_bytes(b) as CelRef),
        Value::List(list) => {
            let mut items = Vec::with_capacity(list.len());
            for elt in list.iter() {
                items.push(value_to_ref(&elt)?);
            }
            Ok(new_list(&items) as CelRef)
        }
        Value::Map(_) => Err(ConvertError::Unsupported("map")),
        #[cfg(feature = "structs")]
        Value::Struct(_) => Err(ConvertError::Unsupported("struct")),
        Value::Opaque(opaque) => {
            let Some(opt) = opaque.downcast_ref::<OptionalValue>() else {
                return Err(ConvertError::Unsupported("opaque"));
            };
            match opt.value() {
                None => Ok(new_optional_none() as CelRef),
                Some(inner) => Ok(new_optional(value_to_ref(inner)?) as CelRef),
            }
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
            let n = leaf.byte_len as usize;
            let base = unsafe { bytes_base(leaf.chars) };
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("string"));
            }
            let bytes = unsafe { std::slice::from_raw_parts(base, n) };
            let s = std::str::from_utf8(bytes).map_err(|_| ConvertError::Corrupt("string"))?;
            Ok(Value::String(Arc::new(s.to_string())))
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
        CelKind::List if class == &CEL_LIST_CLASS => {
            let leaf = unsafe { &*w.cast::<W_ListObject>() };
            let n = leaf.length as usize;
            let base = unsafe { items_block_items_base(leaf.items) };
            if base.is_null() && n != 0 {
                return Err(ConvertError::Corrupt("list"));
            }
            let mut items = Vec::with_capacity(n);
            for i in 0..n {
                items.push(unsafe { ref_to_value(*base.add(i))? });
            }
            Ok(Value::List(ListRef::from(items)))
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
    fn public_list_and_optional_roundtrip() {
        let list = Value::List(ListRef::from(vec![Value::Int(1), Value::Int(2)]));
        assert_eq!(roundtrip(list.clone()), list);
        let none = Value::Opaque(Arc::new(OptionalValue::none()));
        assert_eq!(roundtrip(none.clone()), none);
        let some = Value::Opaque(Arc::new(OptionalValue::of(Value::Int(4))));
        assert_eq!(roundtrip(some.clone()), some);
    }

    #[test]
    fn map_stays_on_the_public_enum() {
        use crate::objects::Map;
        let v = Value::Map(Map::object(Arc::new(std::collections::HashMap::new())));
        assert_eq!(value_to_ref(&v), Err(ConvertError::Unsupported("map")));
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
