//! Arithmetic, dispatched by a narrowing chain on the class word.
//!
//! Each operator is a chain of **pointer-identity tests on the header word**,
//! every arm calling a concrete monomorphic function. This is
//! `descroperation.py`'s `type(w_obj1) is type(w_obj2)` followed by the
//! per-class shortcut, transliterated, and it is the only dispatch shape
//! front-end B lowers to a *direct* call:
//!
//! - `(*w).ob_type` under a raw-pointer deref is recognised as a typed
//!   `FieldRead` with a `__pyre_cast_instance` narrow in front of it, rather
//!   than a classdef-less read that stalls the annotator.
//! - `ta == &CEL_INT_CLASS` is the pointer-identity chain the annotator already
//!   reads for `knowntypedata` narrowing.
//! - each arm's body is a plain path call, so it inlines when the callee is in
//!   the graph closure — no indirect target list, no wrapper family.
//!
//! At trace time exactly one arm is taken and the rest collapse into guards,
//! the same fold an unrolled opcode ladder gets.
//!
//! The shapes that look equivalent and are not are worth naming, because none
//! of them fails loudly: a `CelClass` holding function pointers reaches the
//! frontend as a dynamic call and stops the graph; a `match` on
//! [`CelKind`](super::object::CelKind) reads an integer with no class fact
//! attached, so nothing narrows and the header read never happens.
//!
//! # What is here
//!
//! `+ - * / %` and unary negation over the landed leaves, matching the
//! `Value` operator set arm for arm — `checked_*` with the same overflow,
//! division-by-zero and remainder-by-zero dispositions, so the frozen oracle
//! corpus keeps answering the same way when these become the production path.
//! Equality and ordering are deliberately absent: CEL compares across the
//! numeric types, which is a semantic question the corpus has to settle rather
//! than one to infer here.
//!
//! Errors leave through [`super::error`], never a `Result`.
//!
//! # Why `clippy::ptr_eq` is refused here
//!
//! `clippy` asks for `std::ptr::eq(ta, &CEL_INT_CLASS)` in place of `ta ==
//! &CEL_INT_CLASS`. Taking it would quietly cost the chain its whole purpose.
//! LLBC is extracted from MIR before optimization, so `ptr::eq` survives to the
//! frontend as a *call* to `core::ptr::eq` — an unregistered host path, which is
//! the single largest class of prepass failure measured over cel's closure — and
//! the comparison the annotator reads for its narrowing would no longer be in
//! this graph at all. The bare `==` is also the spelling the `charon-corpus`
//! fixture uses, i.e. the one majit's own lowering assertions are written
//! against, where the premise is stated as two header reads and two `eq`s.
//!
//! The suppression is module-wide rather than per-site because the lint fires
//! unevenly: it flags the hand-written comparisons and skips the ones produced
//! by `same_class_chain!`, so per-site allows would drift as arms move between
//! the two forms.
#![allow(clippy::ptr_eq)]

use super::error::{raise, CelErrCode, ERROR_SENTINEL};
use super::object::{
    new_double, new_duration, new_int, new_timestamp, new_uint, CelClass, CelRef, CEL_DOUBLE_CLASS,
    CEL_DURATION_CLASS, CEL_INT_CLASS, CEL_TIMESTAMP_CLASS, CEL_UINT_CLASS,
};

/// The class word of `w`, as the chains read it.
///
/// # Safety
///
/// `w` must point at a live value.
#[inline]
unsafe fn class_of(w: CelRef) -> *const CelClass {
    (*w).ob_type
}

/// Read a payload field that follows the header.
///
/// The leaves are `#[repr(C)]` with the header first, so a `CelRef` known to
/// be of class `T` casts to `*mut T` without adjustment.
///
/// Expands to a bare dereference, so every use site must already be an unsafe
/// context — an `unsafe` block of its own would be redundant inside the
/// `unsafe fn`s below and would warn.
macro_rules! payload {
    ($w:expr, $leaf:ty, $field:ident) => {
        (*($w as *mut $leaf)).$field
    };
}

use super::object::{
    W_DoubleObject, W_DurationObject, W_IntObject, W_TimestampObject, W_UIntObject,
};

// -- the per-class arms ----------------------------------------------------
//
// One concrete function per (operator, class). They are what the chain's arms
// call, and what the tracer inlines; keeping them separate from the chain is
// what makes each one monomorphic.

macro_rules! checked_int_arm {
    ($name:ident, $leaf:ty, $field:ident, $ctor:ident, $checked:ident, $op:literal) => {
        /// # Safety
        ///
        /// Both operands must be live values of this arm's class.
        pub unsafe fn $name(a: CelRef, b: CelRef) -> CelRef {
            let l = payload!(a, $leaf, $field);
            let r = payload!(b, $leaf, $field);
            match l.$checked(r) {
                Some(v) => $ctor(v) as CelRef,
                None => raise(CelErrCode::Overflow, $op, a, b),
            }
        }
    };
}

checked_int_arm!(w_int_add, W_IntObject, intval, new_int, checked_add, "add");
checked_int_arm!(w_int_sub, W_IntObject, intval, new_int, checked_sub, "sub");
checked_int_arm!(w_int_mul, W_IntObject, intval, new_int, checked_mul, "mul");
checked_int_arm!(
    w_uint_add,
    W_UIntObject,
    uintval,
    new_uint,
    checked_add,
    "add"
);
checked_int_arm!(
    w_uint_sub,
    W_UIntObject,
    uintval,
    new_uint,
    checked_sub,
    "sub"
);
checked_int_arm!(
    w_uint_mul,
    W_UIntObject,
    uintval,
    new_uint,
    checked_mul,
    "mul"
);

macro_rules! float_arm {
    ($name:ident, $apply:expr) => {
        /// # Safety
        ///
        /// Both operands must be live `double` values.
        pub unsafe fn $name(a: CelRef, b: CelRef) -> CelRef {
            let l = payload!(a, W_DoubleObject, floatval);
            let r = payload!(b, W_DoubleObject, floatval);
            #[allow(clippy::redundant_closure_call)]
            let out: f64 = ($apply)(l, r);
            new_double(out) as CelRef
        }
    };
}

// `double` arithmetic does not raise: IEEE-754 answers every case, including
// division by zero, which is why these arms have no error edge at all.
float_arm!(w_double_add, |l: f64, r: f64| l + r);
float_arm!(w_double_sub, |l: f64, r: f64| l - r);
float_arm!(w_double_mul, |l: f64, r: f64| l * r);
float_arm!(w_double_div, |l: f64, r: f64| l / r);

/// `int / int`.
///
/// A zero divisor is `DivisionByZero` naming the dividend, and is tested before
/// the division rather than recovered from it, because `checked_div` cannot
/// distinguish it from `MIN / -1`, which is an overflow.
///
/// # Safety
///
/// Both operands must be live `int` values.
pub unsafe fn w_int_div(a: CelRef, b: CelRef) -> CelRef {
    let l = payload!(a, W_IntObject, intval);
    let r = payload!(b, W_IntObject, intval);
    if r == 0 {
        return raise(CelErrCode::DivisionByZero, "div", a, b);
    }
    match l.checked_div(r) {
        Some(v) => new_int(v) as CelRef,
        None => raise(CelErrCode::Overflow, "div", a, b),
    }
}

/// `uint / uint`.
///
/// Unsigned division has no overflow case, so the only failure is a zero
/// divisor and `checked_div` identifies it exactly.
///
/// # Safety
///
/// Both operands must be live `uint` values.
pub unsafe fn w_uint_div(a: CelRef, b: CelRef) -> CelRef {
    let l = payload!(a, W_UIntObject, uintval);
    let r = payload!(b, W_UIntObject, uintval);
    match l.checked_div(r) {
        Some(v) => new_uint(v) as CelRef,
        None => raise(CelErrCode::DivisionByZero, "div", a, b),
    }
}

/// `int % int`.
///
/// # Safety
///
/// Both operands must be live `int` values.
pub unsafe fn w_int_rem(a: CelRef, b: CelRef) -> CelRef {
    let l = payload!(a, W_IntObject, intval);
    let r = payload!(b, W_IntObject, intval);
    if r == 0 {
        return raise(CelErrCode::RemainderByZero, "rem", a, b);
    }
    match l.checked_rem(r) {
        Some(v) => new_int(v) as CelRef,
        None => raise(CelErrCode::Overflow, "rem", a, b),
    }
}

/// `uint % uint`.
///
/// # Safety
///
/// Both operands must be live `uint` values.
pub unsafe fn w_uint_rem(a: CelRef, b: CelRef) -> CelRef {
    let l = payload!(a, W_UIntObject, uintval);
    let r = payload!(b, W_UIntObject, uintval);
    match l.checked_rem(r) {
        Some(v) => new_uint(v) as CelRef,
        None => raise(CelErrCode::RemainderByZero, "rem", a, b),
    }
}

/// `duration + duration` / `duration - duration`.
macro_rules! duration_arm {
    ($name:ident, $checked:ident, $op:literal) => {
        /// # Safety
        ///
        /// Both operands must be live `duration` values.
        pub unsafe fn $name(a: CelRef, b: CelRef) -> CelRef {
            let l = payload!(a, W_DurationObject, nanos);
            let r = payload!(b, W_DurationObject, nanos);
            match l.$checked(r) {
                Some(v) => new_duration(v) as CelRef,
                None => raise(CelErrCode::Overflow, $op, a, b),
            }
        }
    };
}

duration_arm!(w_duration_add, checked_add, "add");
duration_arm!(w_duration_sub, checked_sub, "sub");

/// `timestamp ± duration`, keeping the timestamp's offset.
///
/// The offset is display state, not part of the instant, so shifting an instant
/// must not reinterpret it in another zone.
///
/// # Safety
///
/// `ts` must be a live `timestamp` and `d` a live `duration`.
unsafe fn timestamp_shift(ts: CelRef, d: CelRef, add: bool, op: &'static str) -> CelRef {
    let nanos = payload!(ts, W_TimestampObject, nanos);
    let off_s = payload!(ts, W_TimestampObject, off_s);
    let delta = payload!(d, W_DurationObject, nanos);
    let shifted = if add {
        nanos.checked_add(delta)
    } else {
        nanos.checked_sub(delta)
    };
    match shifted {
        Some(v) => new_timestamp(v, off_s) as CelRef,
        None => raise(CelErrCode::Overflow, op, ts, d),
    }
}

// -- the chains ------------------------------------------------------------

/// Test both operands against one class.
///
/// Written out rather than folded into a helper taking the arm as a value: an
/// arm reached through a function pointer is the dynamic shape this module
/// exists to avoid.
macro_rules! same_class_chain {
    ($ta:expr, $a:expr, $b:expr, $( $class:expr => $arm:expr ),+ $(,)?) => {
        $(
            if $ta == (&$class as *const CelClass) {
                return $arm($a, $b);
            }
        )+
    };
}

/// CEL `+`.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_add(a: CelRef, b: CelRef) -> CelRef {
    let ta = class_of(a);
    let tb = class_of(b);
    if ta == tb {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_add,
            CEL_UINT_CLASS => w_uint_add,
            CEL_DOUBLE_CLASS => w_double_add,
            CEL_DURATION_CLASS => w_duration_add,
        );
    }
    cel_add_slow(a, b, ta, tb)
}

/// The mixed-type arms of `+`, off the fast chain on purpose.
///
/// # Safety
///
/// As [`cel_add`].
unsafe fn cel_add_slow(a: CelRef, b: CelRef, ta: *const CelClass, tb: *const CelClass) -> CelRef {
    let ts = &CEL_TIMESTAMP_CLASS as *const CelClass;
    let dur = &CEL_DURATION_CLASS as *const CelClass;
    if ta == ts && tb == dur {
        return timestamp_shift(a, b, true, "add");
    }
    if ta == dur && tb == ts {
        return timestamp_shift(b, a, true, "add");
    }
    raise(CelErrCode::UnsupportedBinaryOperator, "add", a, b)
}

/// CEL `-`.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_sub(a: CelRef, b: CelRef) -> CelRef {
    let ta = class_of(a);
    let tb = class_of(b);
    if ta == tb {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_sub,
            CEL_UINT_CLASS => w_uint_sub,
            CEL_DOUBLE_CLASS => w_double_sub,
            CEL_DURATION_CLASS => w_duration_sub,
        );
        // `timestamp - timestamp` is the one same-class operation whose result
        // is a different class, so it does not belong in the chain above.
        if ta == (&CEL_TIMESTAMP_CLASS as *const CelClass) {
            return w_timestamp_diff(a, b);
        }
    }
    cel_sub_slow(a, b, ta, tb)
}

/// `timestamp - timestamp`, yielding a `duration`.
///
/// # Safety
///
/// Both operands must be live `timestamp` values.
pub unsafe fn w_timestamp_diff(a: CelRef, b: CelRef) -> CelRef {
    let l = payload!(a, W_TimestampObject, nanos);
    let r = payload!(b, W_TimestampObject, nanos);
    match l.checked_sub(r) {
        Some(v) => new_duration(v) as CelRef,
        None => raise(CelErrCode::Overflow, "sub", a, b),
    }
}

/// # Safety
///
/// As [`cel_sub`].
unsafe fn cel_sub_slow(a: CelRef, b: CelRef, ta: *const CelClass, tb: *const CelClass) -> CelRef {
    if ta == (&CEL_TIMESTAMP_CLASS as *const CelClass)
        && tb == (&CEL_DURATION_CLASS as *const CelClass)
    {
        return timestamp_shift(a, b, false, "sub");
    }
    raise(CelErrCode::UnsupportedBinaryOperator, "sub", a, b)
}

/// CEL `*`.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_mul(a: CelRef, b: CelRef) -> CelRef {
    let ta = class_of(a);
    if ta == class_of(b) {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_mul,
            CEL_UINT_CLASS => w_uint_mul,
            CEL_DOUBLE_CLASS => w_double_mul,
        );
    }
    raise(CelErrCode::UnsupportedBinaryOperator, "mul", a, b)
}

/// CEL `/`.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_div(a: CelRef, b: CelRef) -> CelRef {
    let ta = class_of(a);
    if ta == class_of(b) {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_div,
            CEL_UINT_CLASS => w_uint_div,
            CEL_DOUBLE_CLASS => w_double_div,
        );
    }
    raise(CelErrCode::UnsupportedBinaryOperator, "div", a, b)
}

/// CEL `%`.
///
/// Defined for the integer types only — `double % double` has no CEL overload.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_rem(a: CelRef, b: CelRef) -> CelRef {
    let ta = class_of(a);
    if ta == class_of(b) {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_rem,
            CEL_UINT_CLASS => w_uint_rem,
        );
    }
    raise(CelErrCode::UnsupportedBinaryOperator, "rem", a, b)
}

/// Unary `-`, and `!` on a bool, which share one entry point.
///
/// `uint` is absent deliberately: negating one has no CEL overload, so it takes
/// the refusal rather than wrapping.
///
/// # Safety
///
/// The operand must be a live value.
pub unsafe fn cel_negate(a: CelRef) -> CelRef {
    use super::object::{new_bool, W_BoolObject, CEL_BOOL_CLASS};
    let ta = class_of(a);
    if ta == (&CEL_INT_CLASS as *const CelClass) {
        let v = payload!(a, W_IntObject, intval);
        return match v.checked_neg() {
            Some(n) => new_int(n) as CelRef,
            None => raise(CelErrCode::Overflow, "negate", a, ERROR_SENTINEL),
        };
    }
    if ta == (&CEL_DOUBLE_CLASS as *const CelClass) {
        return new_double(-payload!(a, W_DoubleObject, floatval)) as CelRef;
    }
    if ta == (&CEL_BOOL_CLASS as *const CelClass) {
        return new_bool(payload!(a, W_BoolObject, boolval) == 0) as CelRef;
    }
    if ta == (&CEL_DURATION_CLASS as *const CelClass) {
        let v = payload!(a, W_DurationObject, nanos);
        return match v.checked_neg() {
            Some(n) => new_duration(n) as CelRef,
            None => raise(CelErrCode::Overflow, "negate", a, ERROR_SENTINEL),
        };
    }
    raise(CelErrCode::NoSuchOverload, "negate", a, ERROR_SENTINEL)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::error::{clear_error, take_error};
    use crate::runtime::object::{new_bool, w_type, W_BoolObject, CEL_BOOL_CLASS};

    /// Every test starts with an empty channel, so a leaked error from an
    /// earlier case cannot be read as this one's.
    fn fresh() {
        clear_error();
    }

    unsafe fn int_of(w: CelRef) -> i64 {
        assert_eq!(w_type(w), &CEL_INT_CLASS as *const CelClass);
        payload!(w, W_IntObject, intval)
    }

    #[test]
    fn same_class_arithmetic_matches_the_value_operators() {
        fresh();
        unsafe {
            assert_eq!(
                int_of(cel_add(new_int(2) as CelRef, new_int(3) as CelRef)),
                5
            );
            assert_eq!(
                int_of(cel_sub(new_int(2) as CelRef, new_int(3) as CelRef)),
                -1
            );
            assert_eq!(
                int_of(cel_mul(new_int(2) as CelRef, new_int(3) as CelRef)),
                6
            );
            assert_eq!(
                int_of(cel_div(new_int(7) as CelRef, new_int(2) as CelRef)),
                3
            );
            assert_eq!(
                int_of(cel_rem(new_int(7) as CelRef, new_int(2) as CelRef)),
                1
            );

            let u = cel_add(new_uint(2) as CelRef, new_uint(3) as CelRef);
            assert_eq!(payload!(u, W_UIntObject, uintval), 5);
            let f = cel_div(new_double(1.0) as CelRef, new_double(4.0) as CelRef);
            assert_eq!(payload!(f, W_DoubleObject, floatval), 0.25);
        }
        assert!(!super::super::error::has_error());
    }

    #[test]
    fn overflow_raises_and_returns_the_sentinel() {
        fresh();
        unsafe {
            let out = cel_add(new_int(i64::MAX) as CelRef, new_int(1) as CelRef);
            assert_eq!(out, ERROR_SENTINEL);
        }
        let err = take_error().expect("a sentinel comes with a raised error");
        assert_eq!(err.code, CelErrCode::Overflow);
        assert_eq!(err.op, "add");
    }

    /// A zero divisor is `DivisionByZero` and not an overflow, and `%` reports
    /// its own code — the distinction the public error type draws.
    #[test]
    fn zero_divisors_report_their_own_codes() {
        fresh();
        unsafe {
            assert_eq!(
                cel_div(new_int(1) as CelRef, new_int(0) as CelRef),
                ERROR_SENTINEL
            );
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::DivisionByZero);

        unsafe {
            assert_eq!(
                cel_rem(new_int(1) as CelRef, new_int(0) as CelRef),
                ERROR_SENTINEL
            );
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::RemainderByZero);

        // Unsigned takes the same disposition through a different route.
        unsafe {
            assert_eq!(
                cel_div(new_uint(1) as CelRef, new_uint(0) as CelRef),
                ERROR_SENTINEL
            );
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::DivisionByZero);
    }

    /// `MIN / -1` overflows rather than dividing by zero: the two failures of
    /// signed division are distinguished, which is why the zero test precedes
    /// `checked_div` instead of reading its `None`.
    #[test]
    fn signed_division_separates_its_two_failures() {
        fresh();
        unsafe {
            assert_eq!(
                cel_div(new_int(i64::MIN) as CelRef, new_int(-1) as CelRef),
                ERROR_SENTINEL
            );
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::Overflow);
    }

    /// Division by zero is not an error for `double`; IEEE-754 answers it.
    #[test]
    fn double_division_by_zero_is_infinity_not_an_error() {
        fresh();
        unsafe {
            let out = cel_div(new_double(1.0) as CelRef, new_double(0.0) as CelRef);
            assert_ne!(out, ERROR_SENTINEL);
            assert!(payload!(out, W_DoubleObject, floatval).is_infinite());
        }
        assert!(!super::super::error::has_error());
    }

    #[test]
    fn mixed_classes_take_the_refusal() {
        fresh();
        unsafe {
            assert_eq!(
                cel_add(new_int(1) as CelRef, new_uint(1) as CelRef),
                ERROR_SENTINEL
            );
        }
        let err = take_error().unwrap();
        assert_eq!(err.code, CelErrCode::UnsupportedBinaryOperator);
        assert_eq!(err.op, "add");
    }

    /// The three mixed-type arms CEL does define over these leaves.
    #[test]
    fn timestamp_and_duration_arithmetic() {
        fresh();
        unsafe {
            let ts = new_timestamp(1_000, -7) as CelRef;
            let d = new_duration(500) as CelRef;

            let later = cel_add(ts, d);
            assert_eq!(payload!(later, W_TimestampObject, nanos), 1_500);
            assert_eq!(
                payload!(later, W_TimestampObject, off_s),
                -7,
                "shifting an instant must not reinterpret its zone",
            );

            // The commuted spelling is the same operation.
            let also = cel_add(d, ts);
            assert_eq!(payload!(also, W_TimestampObject, nanos), 1_500);

            let earlier = cel_sub(ts, d);
            assert_eq!(payload!(earlier, W_TimestampObject, nanos), 500);

            let span = cel_sub(ts, new_timestamp(400, 0) as CelRef);
            assert_eq!(
                w_type(span),
                &CEL_DURATION_CLASS as *const CelClass,
                "timestamp - timestamp is a duration",
            );
            assert_eq!(payload!(span, W_DurationObject, nanos), 600);
        }
        assert!(!super::super::error::has_error());
    }

    #[test]
    fn negation_covers_int_double_bool_and_duration() {
        fresh();
        unsafe {
            assert_eq!(int_of(cel_negate(new_int(3) as CelRef)), -3);
            assert_eq!(
                payload!(
                    cel_negate(new_double(0.5) as CelRef),
                    W_DoubleObject,
                    floatval
                ),
                -0.5
            );
            let not_true = cel_negate(new_bool(true) as CelRef);
            assert_eq!(w_type(not_true), &CEL_BOOL_CLASS as *const CelClass);
            assert_eq!(payload!(not_true, W_BoolObject, boolval), 0);
            assert_eq!(
                payload!(
                    cel_negate(new_duration(9) as CelRef),
                    W_DurationObject,
                    nanos
                ),
                -9
            );

            // `uint` has no negation overload, and overflow is reported apart
            // from that refusal.
            assert_eq!(cel_negate(new_uint(1) as CelRef), ERROR_SENTINEL);
            assert_eq!(take_error().unwrap().code, CelErrCode::NoSuchOverload);
            assert_eq!(cel_negate(new_int(i64::MIN) as CelRef), ERROR_SENTINEL);
            assert_eq!(take_error().unwrap().code, CelErrCode::Overflow);
        }
    }
}
