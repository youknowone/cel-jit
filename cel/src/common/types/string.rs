use crate::common::traits::{self, Adder, Comparer, Sizer, Zeroer};
use crate::common::types::{CelBool, CelBytes, CelDouble, CelInt, CelUInt, Kind, Type};
#[cfg(feature = "chrono")]
use crate::common::types::{CelDuration, CelTimestamp};
use crate::common::value::{Downcast, Val};
use crate::objects::Value;
use crate::ExecutionError;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::Deref;
use std::string::String as StdString;
use std::sync::Arc;

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct String(StdString);

impl String {
    pub fn into_inner(self) -> StdString {
        self.0
    }

    pub fn inner(&self) -> &str {
        &self.0
    }
}

impl Deref for String {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.inner()
    }
}

impl Val for String {
    fn get_type(&self) -> &Type {
        &super::STRING_TYPE
    }

    fn as_adder(&self) -> Option<&dyn Adder> {
        Some(self)
    }

    fn as_comparer(&self) -> Option<&dyn Comparer> {
        Some(self)
    }

    fn as_sizer(&self) -> Option<&dyn Sizer> {
        Some(self)
    }

    fn as_zeroer(&self) -> Option<&dyn Zeroer> {
        Some(self)
    }

    fn equals(&self, other: &dyn Val) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| self.0 == other.0)
    }

    fn clone_as_boxed(&self) -> Box<dyn Val> {
        Box::new(String(self.0.clone()))
    }
}

impl Adder for String {
    fn add<'a>(&'a self, rhs: &dyn Val) -> Result<Cow<'a, dyn Val>, ExecutionError> {
        if let Some(rhs) = rhs.downcast_ref::<Self>() {
            let mut s = StdString::with_capacity(rhs.0.len() + self.0.len());
            s.push_str(&self.0);
            s.push_str(&rhs.0);
            Ok(Cow::<dyn Val>::Owned(Box::new(Self(s))))
        } else {
            Err(ExecutionError::UnsupportedBinaryOperator(
                "add",
                (self as &dyn Val).try_into()?,
                rhs.try_into()?,
            ))
        }
    }
}

impl Comparer for String {
    fn compare(&self, rhs: &dyn Val) -> Result<Ordering, ExecutionError> {
        if let Some(rhs) = rhs.downcast_ref::<Self>() {
            Ok(self.0.cmp(&rhs.0))
        } else {
            Err(ExecutionError::NoSuchOverload)
        }
    }
}

impl Sizer for String {
    fn size(&self) -> CelInt {
        (self.inner().len() as i64).into()
    }
}

impl Zeroer for String {
    fn is_zero_value(&self) -> bool {
        self.inner().is_empty()
    }
}

impl From<StdString> for String {
    fn from(v: StdString) -> Self {
        Self(v)
    }
}

impl From<String> for StdString {
    fn from(v: String) -> Self {
        v.0
    }
}

impl From<&str> for String {
    fn from(value: &str) -> Self {
        Self(StdString::from(value))
    }
}

impl TryFrom<Box<dyn Val>> for StdString {
    type Error = Box<dyn Val>;

    fn try_from(value: Box<dyn Val>) -> Result<Self, Self::Error> {
        super::cast_boxed::<String>(value).map(|s| s.into_inner())
    }
}

impl<'a> TryFrom<&'a dyn Val> for &'a str {
    type Error = &'a dyn Val;
    fn try_from(value: &'a dyn Val) -> Result<Self, Self::Error> {
        if let Some(s) = value.downcast_ref::<String>() {
            return Ok(s.inner());
        }
        Err(value)
    }
}

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
    match regex::Regex::new(pattern) {
        Ok(re) => Ok(Value::Bool(re.is_match(this))),
        Err(err) => Err(ExecutionError::FunctionError {
            function: "matches".to_string(),
            message: format!("'{pattern}' not a valid regex:\n{err}"),
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

#[cfg(test)]
mod tests {
    use super::StdString;
    use super::String;
    use crate::common::value::Val;

    #[test]
    fn test_try_into_string() {
        let str: Box<dyn Val> = Box::new(String::from("cel-rust"));
        assert_eq!(Ok(StdString::from("cel-rust")), str.try_into())
    }

    #[test]
    fn test_try_into_str() {
        let str: Box<dyn Val> = Box::new(String::from("cel-rust"));
        assert_eq!(Ok("cel-rust"), str.as_ref().try_into())
    }
}
