use crate::{ExecutionError, Value};

fn uint(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let arg = args.remove(0);
    match arg {
        Value::UInt(_) => Ok(arg),
        Value::Int(i) => Ok(Value::UInt(i as u64)),
        Value::Float(f) => Ok(Value::UInt(f as u64)),
        Value::String(s) => match s.parse::<u64>() {
            Ok(parsed) => Ok(Value::UInt(parsed)),
            Err(e) => Err(ExecutionError::FunctionError {
                function: "int".to_owned(),
                message: format!("string parse error: {e}"),
            }),
        },
        // Unreachable through the overload table, which declares `uint` only
        // over the four families above.
        other => Err(ExecutionError::FunctionError {
            function: "double".to_owned(),
            message: format!("cannot convert {other:?} to double"),
        }),
    }
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload("uint", "uint64_to_uint64", vec![super::UINT_TYPE], uint)
        .expect("Must be unique id");
    env.add_overload("uint", "int64_to_uint64", vec![super::INT_TYPE], uint)
        .expect("Must be unique id");
    env.add_overload("uint", "double_to_uint64", vec![super::DOUBLE_TYPE], uint)
        .expect("Must be unique id");
    env.add_overload("uint", "string_to_uint64", vec![super::STRING_TYPE], uint)
        .expect("Must be unique id");
}
