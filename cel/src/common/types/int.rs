use crate::objects::Value;
use crate::ExecutionError;

fn int(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    let arg = args.remove(0).unpack();
    match arg {
        Value::Int(_) => Ok(arg),
        Value::UInt(u) => match i64::try_from(u) {
            Ok(value) => Ok(Value::Int(value)),
            Err(_) => Err(ExecutionError::FunctionError {
                function: "int".to_owned(),
                message: "integer overflow".to_owned(),
            }),
        },
        Value::Float(value) => {
            // (minInt, maxInt), exclusive. `i64::MIN as f64` is exactly -2^63,
            // and `i64::MAX as f64` rounds up to 2^63. NaN and the infinities
            // fail the comparison.
            if !(value > (i64::MIN as f64) && value < (i64::MAX as f64)) {
                return Err(ExecutionError::FunctionError {
                    function: "int".to_owned(),
                    message: "integer overflow".to_owned(),
                });
            }
            Ok(Value::Int(value as i64))
        }
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
            function: "int".to_owned(),
            message: format!("cannot convert {other:?} to int"),
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

#[cfg(test)]
mod tests {
    use crate::{Context, Program};

    #[test]
    fn test_conversion_boundaries() {
        let context = Context::default();

        // The largest double below 2^63 is 2^63 - 2^10.
        let program = Program::compile("int(9223372036854774784.0)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, 9223372036854774784i64.into());

        // The smallest double above -2^63 is -(2^63) + 2^10.
        let program = Program::compile("int(-9223372036854774784.0)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, (-9223372036854774784i64).into());

        // i64::MAX is the largest uint that still fits in an int.
        let program = Program::compile("int(9223372036854775807u)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, 9223372036854775807i64.into());
    }

    #[test]
    fn test_conversion_errors() {
        let context = Context::default();

        // -2^63 is exactly representable as f64 and equals i64::MIN, but the
        // lower bound is exclusive.
        let program = Program::compile("int(-9223372036854775808.0)").unwrap();
        assert!(program.execute(&context).is_err());

        // The f64 literal rounds up to 2^63, outside the accepted range.
        let program = Program::compile("int(9223372036854775807.0)").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("int(double('NaN'))").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("int(double('infinity'))").unwrap();
        assert!(program.execute(&context).is_err());

        let program = Program::compile("int(double('-infinity'))").unwrap();
        assert!(program.execute(&context).is_err());

        // One above the largest uint that fits in an int.
        let program = Program::compile("int(9223372036854775808u)").unwrap();
        assert!(program.execute(&context).is_err());
    }
}
