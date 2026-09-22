//! A specialised temporal leaf may only agree with chrono or decline to it.
//!
//! `W_DurationObject` / `W_TimestampObject` store i64 nanoseconds. That is a
//! representation limit of the fast path, not the language domain: chrono's
//! public `Value::Duration` / `Value::Timestamp` arithmetic decides overflow.
//! A sum (or difference, shift, negation) that overflows i64 ns while the
//! public type still holds the result must be the public answer in both
//! evaluators, not `Overflow`.

#![cfg(feature = "chrono")]

use cel::{Context, ExecutionError, Program, Value};

fn show(r: &Result<Value, ExecutionError>) -> String {
    match r {
        Ok(v) => format!("{v:?}"),
        Err(e) => format!("ERR({e:?})"),
    }
}

fn eval_walker(src: &str, ctx: &Context) -> Result<Value, ExecutionError> {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    Value::resolve_value(program.expression(), ctx)
}

fn eval_vm(src: &str, ctx: &Context) -> Result<Value, ExecutionError> {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    program.execute(ctx)
}

fn agree(src: &str, ctx: &Context, expected: Result<Value, ExecutionError>) {
    let walker = eval_walker(src, ctx);
    let vm = eval_vm(src, ctx);
    let want = show(&expected);
    assert_eq!(show(&walker), want, "`{src}` walker");
    assert_eq!(show(&vm), want, "`{src}` vm");
}

fn duration_ns(nanos: i64) -> chrono::Duration {
    chrono::Duration::nanoseconds(nanos)
}

fn timestamp_ns(nanos: i64) -> chrono::DateTime<chrono::FixedOffset> {
    chrono::DateTime::from_timestamp_nanos(nanos).fixed_offset()
}

/// Two durations that each fit i64 nanoseconds, whose sum does not.
/// chrono's TimeDelta still holds the sum.
#[test]
fn duration_sum_past_i64_nanos_agrees_with_chrono() {
    let bound = i64::MAX / 2;
    let one = duration_ns(bound + 10);
    let sum = one.checked_add(&one).expect("TimeDelta holds the sum");
    assert!(
        sum.num_nanoseconds().is_none(),
        "the sum is exactly the case the leaf cannot intern"
    );

    let mut ctx = Context::default();
    ctx.add_variable_from_value("a", Value::Duration(one));
    ctx.add_variable_from_value("b", Value::Duration(one));

    agree("a + b > a", &ctx, Ok(Value::Bool(true)));
    agree("a + b", &ctx, Ok(Value::Duration(sum)));
}

/// A difference whose nanosecond magnitude overflows i64, still held by TimeDelta.
#[test]
fn duration_difference_past_i64_nanos_agrees_with_chrono() {
    let hi = duration_ns(i64::MAX);
    let lo = duration_ns(i64::MIN);
    let diff = hi.checked_sub(&lo).expect("TimeDelta holds the difference");
    assert!(diff.num_nanoseconds().is_none());

    let mut ctx = Context::default();
    ctx.add_variable_from_value("a", Value::Duration(hi));
    ctx.add_variable_from_value("b", Value::Duration(lo));

    agree("a - b", &ctx, Ok(Value::Duration(diff)));
}

/// A timestamp sitting against the leaf's i64-ns ceiling, plus a duration that
/// pushes the instant past that ceiling while staying inside the language range.
#[test]
fn timestamp_plus_duration_past_i64_nanos_agrees_with_chrono() {
    let ts = timestamp_ns(i64::MAX - 50);
    let d = duration_ns(100);
    let later = ts
        .checked_add_signed(d)
        .expect("DateTime holds the shifted instant");
    assert!(later.timestamp_nanos_opt().is_none());

    let mut ctx = Context::default();
    ctx.add_variable_from_value("t", Value::Timestamp(ts));
    ctx.add_variable_from_value("d", Value::Duration(d));

    agree("t + d", &ctx, Ok(Value::Timestamp(later)));
}

/// TimeDelta itself refuses this sum; both evaluators must raise the same error.
#[test]
fn duration_add_that_chrono_rejects_is_overflow_in_both() {
    let big = chrono::Duration::milliseconds(i64::MAX / 2 + 1);
    assert!(
        big.checked_add(&big).is_none(),
        "this pair is a language overflow"
    );
    assert!(cel::runtime::convert::intern_leaf(&Value::Duration(big)).is_none());

    let mut ctx = Context::default();
    ctx.add_variable_from_value("a", Value::Duration(big));
    ctx.add_variable_from_value("b", Value::Duration(big));

    let expected = Err(ExecutionError::Overflow(
        "add",
        Value::Duration(big),
        Value::Duration(big),
    ));
    agree("a + b", &ctx, expected);
}

/// The CEL timestamp range ends at 9999-12-31; both evaluators refuse a step past it.
#[test]
fn timestamp_add_that_chrono_rejects_is_overflow_in_both() {
    let ts = chrono::DateTime::parse_from_rfc3339("9999-12-31T23:59:59Z").unwrap();
    let d = chrono::Duration::seconds(1);
    assert!(cel::runtime::convert::intern_leaf(&Value::Timestamp(ts)).is_none());

    let mut ctx = Context::default();
    ctx.add_variable_from_value("t", Value::Timestamp(ts));
    ctx.add_variable_from_value("d", Value::Duration(d));

    let expected = Err(ExecutionError::Overflow(
        "add",
        Value::Timestamp(ts),
        Value::Duration(d),
    ));
    agree("t + d", &ctx, expected);
}

/// A bound duration/timestamp that does not fit the leaf stays public.
#[test]
fn intern_leaf_leaves_an_oversized_temporal_public() {
    let d = chrono::Duration::milliseconds(i64::MAX);
    assert!(cel::runtime::convert::intern_leaf(&Value::Duration(d)).is_none());

    let ts = chrono::DateTime::parse_from_rfc3339("0001-01-01T00:00:00Z").unwrap();
    assert!(cel::runtime::convert::intern_leaf(&Value::Timestamp(ts)).is_none());

    let mut ctx = Context::default();
    ctx.add_variable_from_value("d", Value::Duration(d));
    ctx.add_variable_from_value("t", Value::Timestamp(ts));
    agree("d == d", &ctx, Ok(Value::Bool(true)));
    agree("t == t", &ctx, Ok(Value::Bool(true)));
}

/// Negating `i64::MIN` nanoseconds overflows the leaf and not TimeDelta.
#[test]
fn duration_negation_past_i64_nanos_agrees_with_chrono() {
    let one = duration_ns(i64::MIN);
    let negated = -one;
    assert!(negated.num_nanoseconds().is_none());

    let mut ctx = Context::default();
    ctx.add_variable_from_value("a", Value::Duration(one));
    agree("-a", &ctx, Ok(Value::Duration(negated)));
}
