use crate::{ExecutionError, Value};

fn double(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let arg = args.remove(0);
    match arg {
        Value::Float(_) => Ok(arg),
        Value::Int(i) => Ok(Value::Float(i as f64)),
        Value::UInt(u) => Ok(Value::Float(u as f64)),
        Value::String(s) => match s.parse::<f64>() {
            Ok(parsed) => Ok(Value::Float(parsed)),
            Err(e) => Err(ExecutionError::FunctionError {
                function: "double".to_owned(),
                message: format!("string parse error: {e}"),
            }),
        },
        // Unreachable through the overload table, which declares `double` only
        // over the four families above.
        other => Err(ExecutionError::FunctionError {
            function: "double".to_owned(),
            message: format!("cannot convert {other:?} to double"),
        }),
    }
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload(
        "double",
        "double_to_double",
        vec![super::DOUBLE_TYPE],
        double,
    )
    .expect("Must be unique id");
    env.add_overload("double", "int64_to_double", vec![super::INT_TYPE], double)
        .expect("Must be unique id");
    env.add_overload("double", "uint64_to_double", vec![super::UINT_TYPE], double)
        .expect("Must be unique id");
    env.add_overload(
        "double",
        "string_to_double",
        vec![super::STRING_TYPE],
        double,
    )
    .expect("Must be unique id");
}
