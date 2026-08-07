use crate::common::traits::{Adder, Comparer, Subtractor, Zeroer};
use crate::common::types::{CelDuration, CelInt, CelString, Type};
use crate::common::value::Val;
use crate::{ExecutionError, Value};
use chrono::{DateTime, FixedOffset};
use chrono::{Datelike, Days, Months};
use chrono::{TimeZone, Timelike};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::ops::{Add, Sub};
use std::sync::LazyLock;

#[derive(Clone, Debug, PartialEq)]
pub struct Timestamp(chrono::DateTime<chrono::FixedOffset>);

impl Timestamp {
    pub fn into_inner(self) -> chrono::DateTime<chrono::FixedOffset> {
        self.0
    }

    pub fn inner(&self) -> &chrono::DateTime<chrono::FixedOffset> {
        &self.0
    }
}

impl Val for Timestamp {
    fn get_type(&self) -> &Type {
        &super::TIMESTAMP_TYPE
    }

    fn as_adder(&self) -> Option<&dyn Adder> {
        Some(self)
    }

    fn as_comparer(&self) -> Option<&dyn Comparer> {
        Some(self)
    }

    fn as_subtractor(&self) -> Option<&dyn Subtractor> {
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
        Box::new(Timestamp(self.0))
    }
}

/// Timestamp values are limited to the range of values which can be serialized as a string:
/// `["0001-01-01T00:00:00Z", "9999-12-31T23:59:59.999999999Z"]`. Since the max is a smaller
/// and the min is a larger timestamp than what is possible to represent with
/// [`chrono::DateTime`],
/// we need to perform our own spec-compliant overflow checks.
///
/// <https://github.com/google/cel-spec/blob/master/doc/langdef.md#overflow>
static MAX_TIMESTAMP: LazyLock<chrono::DateTime<chrono::FixedOffset>> = LazyLock::new(|| {
    let naive = chrono::NaiveDate::from_ymd_opt(9999, 12, 31)
        .unwrap()
        .and_hms_nano_opt(23, 59, 59, 999_999_999)
        .unwrap();
    chrono::FixedOffset::east_opt(0)
        .unwrap()
        .from_utc_datetime(&naive)
});

static MIN_TIMESTAMP: LazyLock<chrono::DateTime<chrono::FixedOffset>> = LazyLock::new(|| {
    let naive = chrono::NaiveDate::from_ymd_opt(1, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    chrono::FixedOffset::east_opt(0)
        .unwrap()
        .from_utc_datetime(&naive)
});

impl Adder for Timestamp {
    fn add<'a>(&'a self, rhs: &dyn Val) -> Result<Cow<'a, dyn Val>, ExecutionError> {
        if let Some(rhs) = rhs.downcast_ref::<CelDuration>() {
            let result = self.0.add(*rhs.inner());
            if result > *MAX_TIMESTAMP || result < *MIN_TIMESTAMP {
                return Err(ExecutionError::Overflow(
                    "add",
                    (self as &dyn Val).try_into().unwrap_or(Value::Null),
                    (rhs as &dyn Val).try_into().unwrap_or(Value::Null),
                ));
            }
            Ok(Cow::<dyn Val>::Owned(Box::new(Self(result))))
        } else {
            Err(ExecutionError::UnsupportedBinaryOperator(
                "add",
                (self as &dyn Val).try_into().unwrap_or(Value::Null),
                rhs.try_into().unwrap_or(Value::Null),
            ))
        }
    }
}

impl Comparer for Timestamp {
    fn compare(&self, rhs: &dyn Val) -> Result<Ordering, ExecutionError> {
        if let Some(rhs) = rhs.downcast_ref::<Self>() {
            Ok(self.0.cmp(&rhs.0))
        } else {
            Err(ExecutionError::NoSuchOverload)
        }
    }
}

impl Subtractor for Timestamp {
    fn sub<'a>(&'a self, rhs: &'_ dyn Val) -> Result<Cow<'a, dyn Val>, ExecutionError> {
        if let Some(rhs) = rhs.downcast_ref::<CelDuration>() {
            let result = self.0.sub(*rhs.inner());
            if result > *MAX_TIMESTAMP || result < *MIN_TIMESTAMP {
                return Err(ExecutionError::Overflow(
                    "sub",
                    (self as &dyn Val).try_into().unwrap_or(Value::Null),
                    (rhs as &dyn Val).try_into().unwrap_or(Value::Null),
                ));
            }
            Ok(Cow::<dyn Val>::Owned(Box::new(Self(result))))
        } else if let Some(rhs) = rhs.downcast_ref::<Self>() {
            Ok(Cow::<dyn Val>::Owned(Box::new(CelDuration::from(
                self.0.signed_duration_since(rhs.inner()),
            ))))
        } else {
            Err(ExecutionError::UnsupportedBinaryOperator(
                "sub",
                (self as &dyn Val).try_into().unwrap_or(Value::Null),
                rhs.try_into().unwrap_or(Value::Null),
            ))
        }
    }
}

impl Zeroer for Timestamp {
    fn is_zero_value(&self) -> bool {
        self.0.timestamp_nanos_opt().is_some_and(|ns| ns == 0)
    }
}

impl From<chrono::DateTime<chrono::FixedOffset>> for Timestamp {
    fn from(system_time: chrono::DateTime<chrono::FixedOffset>) -> Self {
        Self(system_time)
    }
}

impl From<Timestamp> for chrono::DateTime<chrono::FixedOffset> {
    fn from(timestamp: Timestamp) -> Self {
        timestamp.0
    }
}

impl TryFrom<Box<dyn Val>> for chrono::DateTime<chrono::FixedOffset> {
    type Error = Box<dyn Val>;

    fn try_from(value: Box<dyn Val>) -> Result<Self, Self::Error> {
        if let Some(ts) = value.downcast_ref::<Timestamp>() {
            return Ok(ts.0);
        }
        Err(value)
    }
}

impl<'a> TryFrom<&'a dyn Val> for &'a chrono::DateTime<chrono::FixedOffset> {
    type Error = &'a dyn Val;

    fn try_from(value: &'a dyn Val) -> Result<Self, Self::Error> {
        if let Some(ts) = value.downcast_ref::<Timestamp>() {
            return Ok(&ts.0);
        }
        Err(value)
    }
}

/// Reads the receiver an accessor overload declared as `google.protobuf.Timestamp`.
fn expect_timestamp(value: &Value) -> Result<&DateTime<FixedOffset>, ExecutionError> {
    match value {
        Value::Timestamp(ts) => Ok(ts),
        other => Err(super::type_error(other, &super::TIMESTAMP_TYPE)),
    }
}

/// Builds an accessor overload that projects one integer field out of a timestamp.
macro_rules! timestamp_accessor {
    ($name:ident, |$ts:ident| $body:expr) => {
        fn $name(args: Vec<Value>) -> Result<Value, ExecutionError> {
            let $ts = expect_timestamp(&args[0])?;
            Ok(Value::Int($body))
        }
    };
}

timestamp_accessor!(millis, |ts| ts.timestamp_subsec_millis() as i64);
timestamp_accessor!(seconds, |ts| ts.second() as i64);
timestamp_accessor!(minutes, |ts| ts.minute() as i64);
timestamp_accessor!(hours, |ts| ts.hour() as i64);
timestamp_accessor!(day_of_week, |ts| ts.weekday().num_days_from_sunday() as i64);
timestamp_accessor!(date, |ts| ts.day() as i64);
timestamp_accessor!(day_of_month, |ts| ts.day0() as i64);
timestamp_accessor!(month, |ts| ts.month0() as i64);
timestamp_accessor!(full_year, |ts| ts.year() as i64);

fn day_of_year(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let ts = expect_timestamp(&args[0])?;
    let year = ts
        .checked_sub_days(Days::new(ts.day0() as u64))
        .unwrap()
        .checked_sub_months(Months::new(ts.month0()))
        .unwrap();
    Ok(Value::Int(ts.signed_duration_since(year).num_days()))
}

fn timestamp(args: Vec<Value>) -> Result<Value, ExecutionError> {
    let text = match &args[0] {
        Value::String(s) => s.as_str(),
        other => return Err(super::type_error(other, &super::STRING_TYPE)),
    };
    let parsed = chrono::DateTime::parse_from_rfc3339(text)
        .map_err(|e| ExecutionError::function_error("timestamp", e.to_string().as_str()))?;
    Ok(Value::Timestamp(parsed))
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload(
        "timestamp",
        "string_to_timestamp",
        vec![super::STRING_TYPE],
        timestamp,
    )
    .expect("Must be unique");
    env.add_overload(
        "timestamp",
        "timestamp_to_timestamp",
        vec![super::TIMESTAMP_TYPE],
        super::noop,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getFullYear",
        "timestamp_to_year",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        full_year,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getMonth",
        "timestamp_to_month",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        month,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getDayOfYear",
        "timestamp_to_day_of_year",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        day_of_year,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getDayOfMonth",
        "timestamp_to_day_of_month",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        day_of_month,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getDate",
        "timestamp_to_day_of_month_1_based",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        date,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getDayOfWeek",
        "timestamp_to_day_of_week",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        day_of_week,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getHours",
        "timestamp_to_hours",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        hours,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getMinutes",
        "timestamp_to_minutes",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        minutes,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getSeconds",
        "timestamp_to_seconds",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        seconds,
    )
    .expect("Must be unique");
    env.add_member_overload(
        "getMilliseconds",
        "timestamp_to_millis",
        super::TIMESTAMP_TYPE,
        Vec::default(),
        millis,
    )
    .expect("Must be unique");
}
