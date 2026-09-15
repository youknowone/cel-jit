use crate::common::traits::{self};
use crate::objects::Value;
use crate::ExecutionError;
use std::string::String as StdString;
use std::sync::Arc;

/// Reads an argument the overload declared as `string`.
fn expect_string(value: &Value) -> Result<&str, ExecutionError> {
    match value {
        Value::String(s) => Ok(s.as_str()),
        other => Err(super::type_error(other, &super::STRING_TYPE)),
    }
}

fn string_contains(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let target = expect_string(&args[0])?;
    let needle = expect_string(&args[1])?;
    Ok(Value::Bool(target.contains(needle)))
}

fn ends_with_string(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let target = expect_string(&args[0])?;
    let needle = expect_string(&args[1])?;
    Ok(Value::Bool(target.ends_with(needle)))
}

fn starts_with_string(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let target = expect_string(&args[0])?;
    let needle = expect_string(&args[1])?;
    Ok(Value::Bool(target.starts_with(needle)))
}

#[cfg(feature = "regex")]
fn matches(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let this = expect_string(&args[0])?;
    let pattern = expect_string(&args[1])?;
    match crate::runtime::regex_intern::intern_regex(pattern) {
        Ok(re) => Ok(Value::bool(re.is_match(this))),
        Err(message) => Err(ExecutionError::FunctionError {
            function: "matches".to_string(),
            message,
        }),
    }
}

fn string(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let arg = args.remove(0);
    let converted = match &arg {
        Value::String(_) => return Ok(arg),
        Value::Int(i) => i.to_string(),
        Value::UInt(u) => u.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Bytes(b) => StdString::from_utf8_lossy(b).into_owned(),
        #[cfg(feature = "chrono")]
        Value::Timestamp(ts) => ts.to_rfc3339(),
        #[cfg(feature = "chrono")]
        Value::Duration(d) => crate::duration::format_duration(d),
        // Unreachable through the overload table, which declares `string` only
        // over the families above.
        other => {
            return Err(ExecutionError::FunctionError {
                function: "string".to_owned(),
                message: format!("cannot convert {other:?} to string"),
            })
        }
    };
    Ok(Value::String(Arc::new(converted)))
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload(
        "string",
        "string_to_string",
        vec![super::STRING_TYPE],
        string,
    )
    .expect("Must be unique id");
    env.add_overload("string", "int64_to_string", vec![super::INT_TYPE], string)
        .expect("Must be unique id");
    env.add_overload("string", "uint64_to_string", vec![super::UINT_TYPE], string)
        .expect("Must be unique id");
    env.add_overload(
        "string",
        "double_to_string",
        vec![super::DOUBLE_TYPE],
        string,
    )
    .expect("Must be unique id");
    env.add_overload("string", "bytes_to_string", vec![super::BYTES_TYPE], string)
        .expect("Must be unique id");

    #[cfg(feature = "chrono")]
    {
        env.add_overload(
            "string",
            "timestamp_to_string",
            vec![super::TIMESTAMP_TYPE],
            string,
        )
        .expect("Must be unique id");
        env.add_overload(
            "string",
            "duration_to_string",
            vec![super::DURATION_TYPE],
            string,
        )
        .expect("Must be unique id");
    }

    env.add_member_overload(
        "contains",
        "contains_string",
        super::STRING_TYPE,
        vec![super::STRING_TYPE],
        string_contains,
    )
    .expect("Must be unique id");
    env.add_member_overload(
        "endsWith",
        "ends_with_string",
        super::STRING_TYPE,
        vec![super::STRING_TYPE],
        ends_with_string,
    )
    .expect("Must be unique id");
    env.add_overload(
        "size",
        "size_string",
        vec![super::STRING_TYPE],
        traits::adapter::sizer_size,
    )
    .expect("Must be unique id");
    env.add_member_overload(
        "size",
        "string_size",
        super::STRING_TYPE,
        vec![],
        traits::adapter::sizer_size,
    )
    .expect("Must be unique id");
    env.add_member_overload(
        "startsWith",
        "starts_with_string",
        super::STRING_TYPE,
        vec![super::STRING_TYPE],
        starts_with_string,
    )
    .expect("Must be unique id");
    #[cfg(feature = "regex")]
    env.add_member_overload(
        "matches",
        "matches",
        super::STRING_TYPE,
        vec![super::STRING_TYPE],
        matches,
    )
    .expect("Must be unique id");
}
