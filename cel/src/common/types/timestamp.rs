use crate::{ExecutionError, Value};
use chrono::Timelike;
use chrono::{DateTime, FixedOffset};
use chrono::{Datelike, Days, Months};

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
