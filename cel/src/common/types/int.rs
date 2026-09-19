use crate::objects::Value;
use crate::ExecutionError;

fn int(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let arg = args.remove(0).unpack();
    match arg {
        Value::Int(_) => Ok(arg),
        Value::UInt(u) => Ok(Value::Int(u as i64)),
        Value::Float(f) => Ok(Value::Int(f as i64)),
        Value::String(s) => match s.parse::<i64>() {
            Ok(parsed) => Ok(Value::Int(parsed)),
            Err(e) => Err(ExecutionError::FunctionError {
                function: "int".to_owned(),
                message: format!("string parse error: {e}"),
            }),
        },
        // Unreachable through the overload table, which declares `int` only
        // over the four families above.
        other => Err(ExecutionError::FunctionError {
            function: "double".to_owned(),
            message: format!("cannot convert {other:?} to double"),
        }),
    }
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload("int", "int64_to_int64", vec![super::INT_TYPE], int)
        .expect("Must be unique id");
    env.add_overload("int", "uint64_to_int64", vec![super::UINT_TYPE], int)
        .expect("Must be unique id");
    env.add_overload("int", "double_to_int64", vec![super::DOUBLE_TYPE], int)
        .expect("Must be unique id");
    env.add_overload("int", "string_to_int64", vec![super::STRING_TYPE], int)
        .expect("Must be unique id");
}
