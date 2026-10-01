use crate::{ExecutionError, Value};

fn uint(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let arg = args.remove(0).unpack();
    match arg {
        Value::UInt(_) => Ok(arg),
        Value::Int(i) => match u64::try_from(i) {
            Ok(value) => Ok(Value::UInt(value)),
            Err(_) => Err(ExecutionError::FunctionError {
                function: "uint".to_owned(),
                message: "unsigned integer overflow".to_owned(),
            }),
        },
        Value::Float(value) => {
            // [0, maxUint). `u64::MAX as f64` rounds up to 2^64. NaN and the
            // infinities fail the comparison.
            if !(value >= 0.0 && value < (u64::MAX as f64)) {
                return Err(ExecutionError::FunctionError {
                    function: "uint".to_owned(),
                    message: "unsigned integer overflow".to_owned(),
                });
            }
            Ok(Value::UInt(value as u64))
        }
        Value::String(s) => match s.parse::<u64>() {
            Ok(parsed) => Ok(Value::UInt(parsed)),
            Err(e) => Err(ExecutionError::FunctionError {
                function: "uint".to_owned(),
                message: format!("string parse error: {e}"),
            }),
        },
        // Unreachable through the overload table, which declares `uint` only
        // over the four families above.
        other => Err(ExecutionError::FunctionError {
            function: "uint".to_owned(),
            message: format!("cannot convert {other:?} to uint"),
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

#[cfg(test)]
mod tests {
    use crate::{Context, Program};

    #[test]
    fn test_conversion_boundaries() {
        let context = Context::default();

        let program = Program::compile("uint(0.0)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, 0u64.into());

        // The closest integer double below 2^64 is 2^64 - 2^11.
        let program = Program::compile("uint(18446744073709549568.0)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, 18446744073709549568u64.into());

        let program = Program::compile("uint(0)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, 0u64.into());

        let program = Program::compile("uint(9223372036854775807)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, 9223372036854775807u64.into());
    }

    #[test]
    fn test_conversion_errors() {
        let context = Context::default();

        let program = Program::compile("uint(-1)").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("uint(-1.0)").unwrap();
        assert!(program.execute(&context).is_err());

        // 2^64.
        let program = Program::compile("uint(18446744073709551616.0)").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("uint(double('NaN'))").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("uint(double('infinity'))").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("uint(double('-infinity'))").unwrap();
        assert!(program.execute(&context).is_err());
    }
}
