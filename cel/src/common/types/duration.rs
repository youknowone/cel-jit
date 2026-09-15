use crate::{ExecutionError, Value};

/// Reads the receiver an accessor overload declared as `google.protobuf.Duration`.
fn expect_duration(value: &Value) -> Result<&chrono::Duration, ExecutionError> {
    match value {
        Value::Duration(d) => Ok(d),
        other => Err(super::type_error(other, &super::DURATION_TYPE)),
    }
}

/// Builds an accessor overload that projects one integer field out of a duration.
macro_rules! duration_accessor {
    ($name:ident, $method:ident) => {
        fn $name(args: Vec<Value>) -> Result<Value, ExecutionError> {
            Ok(Value::Int(expect_duration(&args[0])?.$method()))
        }
    };
}

duration_accessor!(millis, num_milliseconds);
duration_accessor!(seconds, num_seconds);
duration_accessor!(minutes, num_minutes);
duration_accessor!(hours, num_hours);

fn duration(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let text = match &args[0] {
        Value::String(s) => s.as_str(),
        other => return Err(super::type_error(other, &super::STRING_TYPE)),
    };
    let (_, parsed) = crate::duration::parse_duration(text)
        .map_err(|e| ExecutionError::function_error("duration", e.to_string()))?;
    Ok(Value::Duration(parsed))
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload(
        "duration",
        "string_to_duration",
        vec![super::STRING_TYPE],
        duration,
    )
    .expect("Must be unique");
    env.add_overload(
        "duration",
        "duration_to_duration",
        vec![super::DURATION_TYPE],
        super::noop,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getHours",
        "duration_to_hours",
        super::DURATION_TYPE,
        Vec::default(),
        hours,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getMinutes",
        "duration_to_minutes",
        super::DURATION_TYPE,
        Vec::default(),
        minutes,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getSeconds",
        "duration_to_seconds",
        super::DURATION_TYPE,
        Vec::default(),
        seconds,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getMilliseconds",
        "duration_to_millis",
        super::DURATION_TYPE,
        Vec::default(),
        millis,
    )
    .expect("Must be unique");
}
