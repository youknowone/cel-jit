use crate::Value;
use crate::{common::traits, ExecutionError};
use std::sync::Arc;

fn bytes_to_bytes(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    Ok(args.remove(0))
}

fn string_to_bytes(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    match args.remove(0) {
        Value::String(s) => {
            let value = Arc::try_unwrap(s).unwrap_or_else(|s| s.as_str().to_owned());
            Ok(Value::Bytes(Arc::new(value.into_bytes())))
        }
        other => Err(ExecutionError::UnexpectedType {
            got: super::type_name(&other),
            want: "Bytes".to_owned(),
        }),
    }
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload(
        "bytes",
        "string_to_bytes",
        vec![super::STRING_TYPE],
        string_to_bytes,
    )
    .expect("Must be unique id");
    env.add_overload(
        "bytes",
        "bytes_to_bytes",
        vec![super::BYTES_TYPE],
        bytes_to_bytes,
    )
    .expect("Must be unique id");
    env.add_overload(
        "size",
        "size_bytes",
        vec![super::BYTES_TYPE],
        traits::adapter::sizer_size,
    )
    .expect("Must be unique id");
    env.add_member_overload(
        "size",
        "bytes_size",
        super::BYTES_TYPE,
        vec![],
        traits::adapter::sizer_size,
    )
    .expect("Must be unique id");
}
