//! The CEL `optional` operations over the class family.
//!
//! [`super::object::W_OptionalObject`] is the family's only leaf with a managed
//! payload, so these are the first operations that both allocate a
//! pointer-carrying leaf and read one back out. Everything else in
//! [`super::binop`] moves `i64`/`f64` through a header.
//!
//! The oracle is [`crate::functions`]' walker-side pair — `optional_none`,
//! `optional_of`, `optional_of_non_zero_value`, `optional_value`,
//! `optional_has_value`, `optional_or_optional`, `optional_or_value`. Where the
//! two could differ they do not: the receiver-type checks and the order they
//! happen in are copied from there rather than re-derived, including
//! [`cel_optional_or`]'s asymmetry.
//!
//! # What is not here
//!
//! The optional-field and optional-index syntax (`m[?k]`, `[?x]`, `{?k: v}`)
//! reaches optionals through the map and list leaves, which have no class-family
//! spelling yet. These are the seven functions whose operands are all leaves
//! that exist.
//!
//! # Why `clippy::ptr_eq` is refused here
//!
//! Same reason as [`super::binop`], which states it in full: `ptr::eq` survives
//! MIR into LLBC as a call to an unregistered host path, so the comparison the
//! annotator narrows on would no longer be in this graph. The class tests below
//! are the same two-header-reads-and-an-`eq` shape the chains use.
#![allow(clippy::ptr_eq)]

use super::error::{raise, CelErrCode, ERROR_SENTINEL};
use super::object::{
    bytes_len, list_len, map_len, new_bool, new_optional, new_optional_none, payload,
    string_byte_len, w_type, CelClass, CelRef, W_BoolObject, W_DoubleObject, W_DurationObject,
    W_IntObject, W_OptionalObject, W_TimestampObject, W_UIntObject, CEL_BOOL_CLASS,
    CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_DURATION_CLASS, CEL_INT_CLASS, CEL_LIST_CLASS,
    CEL_MAP_CLASS, CEL_NULL_CLASS, CEL_OPTIONAL_CLASS, CEL_STRING_CLASS, CEL_TIMESTAMP_CLASS,
    CEL_UINT_CLASS,
};

/// The `optional` class, as the receiver tests read it.
#[inline]
fn optional_class() -> *const CelClass {
    &CEL_OPTIONAL_CLASS as *const CelClass
}

/// `optional.of(v)`.
///
/// # Safety
///
/// `v` must be a live value, and stays reachable only through the optional this
/// returns.
pub unsafe fn cel_optional_of(v: CelRef) -> CelRef {
    new_optional(v) as CelRef
}

/// `optional.none()`.
///
/// Safe: it reads no operand. A fresh allocation per call, so two nones are
/// never pointer-equal — which is why [`super::binop::w_optional_eq`] compares
/// presence rather than identity.
pub fn cel_optional_none() -> CelRef {
    new_optional_none() as CelRef
}

/// `optional.ofNonZeroValue(v)`.
///
/// # Safety
///
/// As [`cel_optional_of`].
pub unsafe fn cel_optional_of_non_zero_value(v: CelRef) -> CelRef {
    // Two constructors rather than one call with a conditional argument: the
    // none case has to allocate through `new_optional_none`, whose null is a
    // literal at the allocation site. Passing a null *into* `new_optional`
    // would be the same object built by a body the boxing fuse sees
    // differently, which is the distinction `new_optional_none` exists for.
    if is_zero(v) {
        new_optional_none() as CelRef
    } else {
        new_optional(v) as CelRef
    }
}

/// CEL's zero test, as `optional.ofNonZeroValue` reads it.
///
/// Mirrors `common/types/optional.rs` `is_zero_value`, not [`Value::is_zero`]:
/// an empty string/bytes/list/map is zero, and a timestamp at the epoch is
/// zero. A type value has no zero and falls through to false.
///
/// # Safety
///
/// `v` must be a live value.
unsafe fn is_zero(v: CelRef) -> bool {
    let t = w_type(v);
    if t == (&CEL_INT_CLASS as *const CelClass) {
        return payload!(v, W_IntObject, intval) == 0;
    }
    if t == (&CEL_UINT_CLASS as *const CelClass) {
        return payload!(v, W_UIntObject, uintval) == 0;
    }
    if t == (&CEL_DOUBLE_CLASS as *const CelClass) {
        return payload!(v, W_DoubleObject, floatval) == 0.0;
    }
    if t == (&CEL_BOOL_CLASS as *const CelClass) {
        return payload!(v, W_BoolObject, boolval) == 0;
    }
    if t == (&CEL_DURATION_CLASS as *const CelClass) {
        return payload!(v, W_DurationObject, nanos) == 0;
    }
    if t == (&CEL_TIMESTAMP_CLASS as *const CelClass) {
        return payload!(v, W_TimestampObject, nanos) == 0;
    }
    if t == (&CEL_STRING_CLASS as *const CelClass) {
        return string_byte_len(v) == 0;
    }
    if t == (&CEL_BYTES_CLASS as *const CelClass) {
        return bytes_len(v) == 0;
    }
    if t == (&CEL_LIST_CLASS as *const CelClass) {
        return list_len(v) == 0;
    }
    if t == (&CEL_MAP_CLASS as *const CelClass) {
        return map_len(v) == 0;
    }
    #[cfg(feature = "structs")]
    if t == (&super::object::CEL_STRUCT_CLASS as *const CelClass) {
        return (*v.cast::<super::object::W_StructObject>()).length == 0;
    }
    t == (&CEL_NULL_CLASS as *const CelClass)
}

/// `opt.hasValue()`.
///
/// # Safety
///
/// `o` must be a live value; it is checked to be an `optional` here rather than
/// assumed.
pub unsafe fn cel_optional_has_value(o: CelRef) -> CelRef {
    if w_type(o) != optional_class() {
        return raise(CelErrCode::NoSuchOverload, "hasValue", o, ERROR_SENTINEL);
    }
    new_bool(!payload!(o, W_OptionalObject, w_value).is_null()) as CelRef
}

/// `opt.value()`.
///
/// Raises on a none rather than answering `null`: an absent optional and an
/// optional wrapping `null` are different values, and returning `null` would
/// merge them.
///
/// # Safety
///
/// As [`cel_optional_has_value`].
pub unsafe fn cel_optional_value(o: CelRef) -> CelRef {
    if w_type(o) != optional_class() {
        return raise(CelErrCode::NoSuchOverload, "value", o, ERROR_SENTINEL);
    }
    let v = payload!(o, W_OptionalObject, w_value);
    if v.is_null() {
        return raise(CelErrCode::NoneDereference, "value", o, ERROR_SENTINEL);
    }
    v
}

/// `opt.or(other)` — the first present of the two optionals.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_optional_or(o: CelRef, other: CelRef) -> CelRef {
    if w_type(o) != optional_class() {
        return raise(CelErrCode::NoSuchOverload, "or", o, other);
    }
    if !payload!(o, W_OptionalObject, w_value).is_null() {
        return o;
    }
    // `other`'s type is checked only on this arm, and that asymmetry is the
    // oracle's: `optional_or_optional` converts `other` inside its `None`
    // branch, so a present receiver answers without ever inspecting it.
    if w_type(other) != optional_class() {
        return raise(CelErrCode::NoSuchOverload, "or", o, other);
    }
    other
}

/// `opt.orValue(other)` — the wrapped value, or `other` when absent.
///
/// `other` is a plain value here, not an optional, so it carries no type test on
/// either arm.
///
/// # Safety
///
/// As [`cel_optional_or`].
pub unsafe fn cel_optional_or_value(o: CelRef, other: CelRef) -> CelRef {
    if w_type(o) != optional_class() {
        return raise(CelErrCode::NoSuchOverload, "orValue", o, other);
    }
    let v = payload!(o, W_OptionalObject, w_value);
    if v.is_null() {
        return other;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::binop::values_equal;
    use crate::runtime::error::{has_error, take_error};
    use crate::runtime::object::{
        new_double, new_duration, new_int, new_null, new_timestamp, new_uint,
    };

    /// `has_error` is process-visible state, and a raise left in the slot fails
    /// the next test to look at it rather than the one that raised.
    fn assert_clean() {
        assert!(!has_error(), "an earlier case left an error in the slot");
    }

    #[test]
    fn of_wraps_and_none_is_absent() {
        unsafe {
            assert_clean();
            let one = cel_optional_of(new_int(1) as CelRef);
            let none = cel_optional_none();

            assert!(values_equal(
                cel_optional_has_value(one),
                new_bool(true) as CelRef
            ));
            assert!(values_equal(
                cel_optional_has_value(none),
                new_bool(false) as CelRef
            ));
            assert!(values_equal(cel_optional_value(one), new_int(1) as CelRef));
            assert_clean();
        }
    }

    /// Every leaf `Value::is_zero` answers true for, plus the two it does not.
    #[test]
    fn of_non_zero_value_folds_the_zero_of_each_leaf() {
        unsafe {
            assert_clean();
            let zeros: [CelRef; 10] = [
                new_int(0) as CelRef,
                new_uint(0) as CelRef,
                new_double(0.0) as CelRef,
                new_bool(false) as CelRef,
                new_duration(0) as CelRef,
                new_timestamp(0, 0) as CelRef,
                new_null() as CelRef,
                crate::runtime::object::new_string("") as CelRef,
                crate::runtime::object::new_bytes(b"") as CelRef,
                crate::runtime::object::new_list(&[]) as CelRef,
            ];
            for z in zeros {
                let opt = cel_optional_of_non_zero_value(z);
                assert!(
                    values_equal(cel_optional_has_value(opt), new_bool(false) as CelRef),
                    "a zero-valued leaf should fold to none"
                );
            }

            let nonzeros: [CelRef; 5] = [
                new_int(1) as CelRef,
                new_uint(1) as CelRef,
                new_double(0.5) as CelRef,
                new_bool(true) as CelRef,
                new_duration(1) as CelRef,
            ];
            for v in nonzeros {
                let opt = cel_optional_of_non_zero_value(v);
                assert!(
                    values_equal(cel_optional_has_value(opt), new_bool(true) as CelRef),
                    "a non-zero leaf should stay present"
                );
                assert!(values_equal(cel_optional_value(opt), v));
            }
            assert_clean();
        }
    }

    #[test]
    fn value_on_a_none_raises_rather_than_answering_null() {
        unsafe {
            assert_clean();
            let wrapped_null = cel_optional_of(new_null() as CelRef);
            assert!(values_equal(
                cel_optional_value(wrapped_null),
                new_null() as CelRef
            ));
            assert_clean();

            assert_eq!(cel_optional_value(cel_optional_none()), ERROR_SENTINEL);
            let err = take_error().expect("value() on a none raises");
            assert_eq!(err.code, CelErrCode::NoneDereference);
            assert_eq!(err.op, "value");
            assert_clean();
        }
    }

    /// One operation per step, drained before the next.
    ///
    /// A table of pre-computed answers would raise four times before the first
    /// check ran, and the slot holds at most one error — so every case would be
    /// graded against the last one's.
    macro_rules! raises_no_such_overload {
        ($call:expr, $op:literal) => {{
            assert_eq!($call, ERROR_SENTINEL, concat!($op, " should raise"));
            let err = take_error().expect("a raise reaches the slot");
            assert_eq!(err.code, CelErrCode::NoSuchOverload);
            assert_eq!(err.op, $op);
        }};
    }

    #[test]
    fn a_non_optional_receiver_raises_on_every_operation() {
        unsafe {
            assert_clean();
            let not_opt = new_int(1) as CelRef;
            raises_no_such_overload!(cel_optional_has_value(not_opt), "hasValue");
            raises_no_such_overload!(cel_optional_value(not_opt), "value");
            raises_no_such_overload!(cel_optional_or(not_opt, cel_optional_none()), "or");
            raises_no_such_overload!(cel_optional_or_value(not_opt, not_opt), "orValue");
            assert_clean();
        }
    }

    #[test]
    fn or_takes_the_first_present() {
        unsafe {
            assert_clean();
            let one = cel_optional_of(new_int(1) as CelRef);
            let two = cel_optional_of(new_int(2) as CelRef);
            let none = cel_optional_none();

            assert!(values_equal(
                cel_optional_value(cel_optional_or(one, two)),
                new_int(1) as CelRef
            ));
            assert!(values_equal(
                cel_optional_value(cel_optional_or(none, two)),
                new_int(2) as CelRef
            ));
            assert!(values_equal(
                cel_optional_has_value(cel_optional_or(none, cel_optional_none())),
                new_bool(false) as CelRef
            ));
            assert_clean();
        }
    }

    /// The asymmetry copied from `optional_or_optional`: a present receiver
    /// answers without inspecting `other`, so a non-optional `other` is only an
    /// error when the receiver is absent.
    #[test]
    fn or_checks_other_only_when_the_receiver_is_absent() {
        unsafe {
            assert_clean();
            let one = cel_optional_of(new_int(1) as CelRef);
            let not_opt = new_int(2) as CelRef;

            assert!(values_equal(
                cel_optional_value(cel_optional_or(one, not_opt)),
                new_int(1) as CelRef
            ));
            assert_clean();

            assert_eq!(
                cel_optional_or(cel_optional_none(), not_opt),
                ERROR_SENTINEL
            );
            let err = take_error().expect("a none receiver checks `other`");
            assert_eq!(err.code, CelErrCode::NoSuchOverload);
            assert_clean();
        }
    }

    #[test]
    fn or_value_unwraps_or_falls_back() {
        unsafe {
            assert_clean();
            let one = cel_optional_of(new_int(1) as CelRef);
            let fallback = new_int(9) as CelRef;

            assert!(values_equal(
                cel_optional_or_value(one, fallback),
                new_int(1) as CelRef
            ));
            assert!(values_equal(
                cel_optional_or_value(cel_optional_none(), fallback),
                new_int(9) as CelRef
            ));
            // `other` is a plain value, so a non-optional fallback is fine.
            assert!(values_equal(
                cel_optional_or_value(cel_optional_none(), new_bool(true) as CelRef),
                new_bool(true) as CelRef
            ));
            assert_clean();
        }
    }
}
