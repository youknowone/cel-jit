use crate::common::types::{self, OPTIONAL_TYPE};
use crate::objects::{OptionalValue, Value};
use crate::ExecutionError;
use std::sync::Arc;

/// Reads the receiver an overload declared as `optional_type`.
fn expect_optional(value: &Value) -> Result<&OptionalValue, ExecutionError> {
    match value {
        Value::Opaque(o) => o.downcast_ref::<OptionalValue>(),
        _ => None,
    }
    .ok_or_else(|| super::type_error(value, &OPTIONAL_TYPE))
}

/// Whether a value is its type's zero, as `optional.ofNonZeroValue` tests it.
///
/// Deliberately not [`Value::is_zero`], which answers `false` for the epoch
/// timestamp and for a field-less struct where this answers `true`.
fn is_zero_value(value: &Value) -> bool {
    let unpacked = value.unpack();
    let value = &unpacked;
    match value {
        Value::Bool(b) => !b,
        Value::Int(i) => *i == 0,
        Value::UInt(u) => *u == 0,
        Value::Float(f) => *f == 0.0,
        Value::String(s) => s.is_empty(),
        Value::Bytes(b) => b.is_empty(),
        Value::Null => true,
        Value::List(l) => l.is_empty(),
        Value::Map(m) => m.is_empty(),
        #[cfg(feature = "chrono")]
        Value::Duration(d) => d.is_zero(),
        #[cfg(feature = "chrono")]
        Value::Timestamp(ts) => ts.timestamp_nanos_opt().is_some_and(|ns| ns == 0),
        #[cfg(feature = "structs")]
        Value::Struct(s) => s.is_empty(),
        // An optional, and any host handle, declares no zero value.
        Value::Opaque(_) => false,
        Value::Interned(_) => is_zero_value(&value.unpack()),
    }
}

fn optional_none(_args: Vec<Value>) -> Result<Value, ExecutionError> {
    Ok(Value::Opaque(Arc::new(OptionalValue::none())))
}

fn optional_of(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    Ok(Value::Opaque(Arc::new(OptionalValue::of(args.remove(0)))))
}

fn optional_of_non_zero_value(args: Vec<Value>) -> Result<Value, ExecutionError> {
    match is_zero_value(&args[0]) {
        true => optional_none(args),
        false => optional_of(args),
    }
}

fn optional_value(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    expect_optional(&args.remove(0))?
        .inner()
        .cloned()
        .ok_or_else(|| ExecutionError::function_error("value", "optional.none() dereference"))
}

fn optional_has_value(args: Vec<Value>) -> Result<Value, ExecutionError> {
    Ok(Value::Bool(expect_optional(&args[0])?.inner().is_some()))
}

fn optional_or_optional(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let other = args.remove(1);
    let this = args.remove(0);
    match expect_optional(&this)?.inner().is_some() {
        true => Ok(this),
        false => Ok(other),
    }
}

fn optional_or_value(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let other = args.remove(1);
    let this = args.remove(0);
    Ok(expect_optional(&this)?.inner().cloned().unwrap_or(other))
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload("optional.none", "optional_none", vec![], optional_none)
        .expect("Must be unique");
    env.add_overload(
        "optional.of",
        "optional_of",
        vec![types::DYN_TYPE],
        optional_of,
    )
    .expect("Must be unique");
    env.add_overload(
        "optional.ofNonZeroValue",
        "optional_ofNonZeroValue",
        vec![types::DYN_TYPE],
        optional_of_non_zero_value,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "value",
        "optional_value",
        OPTIONAL_TYPE,
        vec![],
        optional_value,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "hasValue",
        "optional_has_value",
        OPTIONAL_TYPE,
        vec![],
        optional_has_value,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "or",
        "optional_or_optional",
        OPTIONAL_TYPE,
        vec![OPTIONAL_TYPE],
        optional_or_optional,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "orValue",
        "optional_or_value",
        OPTIONAL_TYPE,
        vec![types::DYN_TYPE],
        optional_or_value,
    )
    .expect("Must be unique");
}

#[cfg(test)]
mod tests {
    use crate::common::types;
    use crate::objects::Value;

    /// `OPTIONAL_TYPE` is parameterised on `dyn`, so it accepts every value and
    /// the optional overloads are selected on name and arity alone.
    #[test]
    fn is_assignable() {
        assert!(types::OPTIONAL_TYPE.is_assignable(&Value::String("foo".to_owned().into())));
        assert!(types::OPTIONAL_TYPE.is_assignable(&Value::Int(42)));
    }
}
