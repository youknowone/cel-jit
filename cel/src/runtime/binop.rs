//! Arithmetic and comparison, dispatched by a narrowing chain on the class word.
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
//! `+ - * / %`, unary negation, `==`/`!=` and the four orderings over the
//! landed leaves, matching the `Value` operator set arm for arm — `checked_*`
//! with the same overflow, division-by-zero and remainder-by-zero dispositions,
//! and the same cross-type numeric comparisons down to the lossy `i64 as f64`,
//! so the frozen oracle corpus keeps answering the same way when these become
//! the production path. The comparison tests are written against that oracle
//! rather than against a restatement of it: `PartialEq for Value` and
//! `objects::compare_values` are called on the same operand pairs and the two
//! answers are required to agree.
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
    new_bool, new_bytes, new_double, new_duration, new_int, new_list, new_null, new_string,
    new_timestamp, new_uint, CelClass, CelRef, W_BytesObject, W_MapObject, W_StringObject,
    CEL_BOOL_CLASS, CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_DURATION_CLASS, CEL_INT_CLASS,
    CEL_LIST_CLASS, CEL_MAP_CLASS, CEL_NULL_CLASS, CEL_OPAQUE_CLASS, CEL_OPTIONAL_CLASS,
    CEL_STRING_CLASS, CEL_TIMESTAMP_CLASS, CEL_TYPE_CLASS, CEL_UINT_CLASS,
};
use super::object_array::{bytes_base, items_block_items_base};

/// The class word of `w`, as the chains read it.
///
/// # Safety
///
/// `w` must point at a live value.
#[inline]
unsafe fn class_of(w: CelRef) -> *const CelClass {
    (*w).ob_type
}

use super::object::payload;
use crate::Value;

use super::object::{
    W_BoolObject, W_DoubleObject, W_DurationObject, W_IntObject, W_OptionalObject,
    W_TimestampObject, W_TypeObject, W_UIntObject,
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
        #[inline]
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
/// # Safety
///
/// `w` is a live string.
unsafe fn string_bytes(w: CelRef) -> &'static [u8] {
    let leaf = &*w.cast::<W_StringObject>();
    let n = leaf.byte_len as usize;
    let base = bytes_base(leaf.chars);
    if base.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(base, n)
    }
}

/// # Safety
///
/// `w` is a live bytes value.
unsafe fn bytes_payload(w: CelRef) -> &'static [u8] {
    let leaf = &*w.cast::<W_BytesObject>();
    let n = leaf.length as usize;
    let base = bytes_base(leaf.data);
    if base.is_null() {
        &[]
    } else {
        std::slice::from_raw_parts(base, n)
    }
}

/// # Safety
///
/// `w` is a live list.
unsafe fn list_items(w: CelRef) -> Vec<CelRef> {
    let n = super::object::list_len(w);
    let mut out = Vec::with_capacity(n as usize);
    let mut i = 0;
    while i < n {
        if let Some(item) = super::convert::interned_list_get(w, i) {
            out.push(item);
        }
        i += 1;
    }
    out
}

fn cmp_bytes(l: &[u8], r: &[u8]) -> i64 {
    match l.cmp(r) {
        std::cmp::Ordering::Less => CMP_LESS,
        std::cmp::Ordering::Equal => CMP_EQUAL,
        std::cmp::Ordering::Greater => CMP_GREATER,
    }
}

/// # Safety
///
/// Both operands are live strings.
pub unsafe fn w_string_add(a: CelRef, b: CelRef) -> CelRef {
    let left = string_bytes(a);
    let right = string_bytes(b);
    let mut out = String::with_capacity(left.len() + right.len());
    out.push_str(std::str::from_utf8_unchecked(left));
    out.push_str(std::str::from_utf8_unchecked(right));
    new_string(&out) as CelRef
}

/// # Safety
///
/// Both operands are live bytes.
pub unsafe fn w_bytes_add(a: CelRef, b: CelRef) -> CelRef {
    let left = bytes_payload(a);
    let right = bytes_payload(b);
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    new_bytes(&out) as CelRef
}

/// # Safety
///
/// Both operands are live lists.
pub unsafe fn w_list_add(a: CelRef, b: CelRef) -> CelRef {
    let left = list_items(a);
    let right = list_items(b);
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(&left);
    out.extend_from_slice(&right);
    new_list(&out) as CelRef
}

/// # Safety
///
/// Both operands are live strings.
pub unsafe fn w_string_eq(a: CelRef, b: CelRef) -> bool {
    string_bytes(a) == string_bytes(b)
}

/// # Safety
///
/// Both operands are live bytes.
pub unsafe fn w_bytes_eq(a: CelRef, b: CelRef) -> bool {
    bytes_payload(a) == bytes_payload(b)
}

/// # Safety
///
/// `w` is a live map.
unsafe fn map_pairs(w: CelRef) -> Vec<CelRef> {
    let leaf = &*w.cast::<W_MapObject>();
    match leaf.strategy {
        super::object::MapStrategy::Object => {
            let n = (leaf.length as usize).saturating_mul(2);
            let base = items_block_items_base(leaf.items);
            if base.is_null() {
                Vec::new()
            } else {
                std::slice::from_raw_parts(base, n).to_vec()
            }
        }
        super::object::MapStrategy::Record => {
            let Ok(Value::Map(map)) = super::convert::ref_to_value(w) else {
                return Vec::new();
            };
            let mut pairs = Vec::with_capacity(map.len() * 2);
            for (k, v) in map.iter() {
                pairs.push(
                    super::convert::intern_leaf(&Value::from(k.clone()))
                        .unwrap_or(new_null() as CelRef),
                );
                pairs.push(super::convert::intern_leaf(v.as_ref()).unwrap_or(new_null() as CelRef));
            }
            pairs
        }
    }
}

/// # Safety
///
/// Both operands are live maps.
pub unsafe fn w_map_eq(a: CelRef, b: CelRef) -> bool {
    let left = map_pairs(a);
    let right = map_pairs(b);
    if left.len() != right.len() {
        return false;
    }
    let n = left.len() / 2;
    let mut i = 0;
    while i < n {
        let lk = left[2 * i];
        let lv = left[2 * i + 1];
        let mut found = false;
        let mut j = 0;
        while j < n {
            if values_equal(lk, right[2 * j]) && values_equal(lv, right[2 * j + 1]) {
                found = true;
                break;
            }
            j += 1;
        }
        if !found {
            return false;
        }
        i += 1;
    }
    true
}

/// `descr_contains` over a list leaf: `space.eq_w` on each item.
///
/// # Safety
///
/// Both operands are live values; `w` is a list.
pub unsafe fn list_contains(w: CelRef, needle: CelRef) -> bool {
    let items = list_items(w);
    let mut i = 0;
    while i < items.len() {
        if values_equal(items[i], needle) {
            return true;
        }
        i += 1;
    }
    false
}

/// Look up `key` on a map leaf. Keys compare with [`values_equal`].
///
/// # Safety
///
/// `w` is a live map; `key` is a live value.
pub unsafe fn map_lookup(w: CelRef, key: CelRef) -> Option<CelRef> {
    let pairs = map_pairs(w);
    let n = pairs.len() / 2;
    let mut i = 0;
    while i < n {
        if values_equal(pairs[2 * i], key) {
            return Some(pairs[2 * i + 1]);
        }
        i += 1;
    }
    None
}

/// `key in map`.
///
/// # Safety
///
/// As [`map_lookup`].
pub unsafe fn map_contains_key(w: CelRef, key: CelRef) -> bool {
    map_lookup(w, key).is_some()
}

/// The keys of a map leaf, in storage order.
///
/// # Safety
///
/// `w` is a live map.
pub unsafe fn map_key_refs(w: CelRef) -> Vec<CelRef> {
    let pairs = map_pairs(w);
    let n = pairs.len() / 2;
    let mut keys = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        keys.push(pairs[2 * i]);
        i += 1;
    }
    keys
}

/// # Safety
///
/// Both operands are live lists.
pub unsafe fn w_list_eq(a: CelRef, b: CelRef) -> bool {
    let left = list_items(a);
    let right = list_items(b);
    if left.len() != right.len() {
        return false;
    }
    let mut i = 0;
    while i < left.len() {
        if !values_equal(left[i], right[i]) {
            return false;
        }
        i += 1;
    }
    true
}

/// # Safety
///
/// Both operands are live strings.
pub unsafe fn w_string_cmp(a: CelRef, b: CelRef) -> i64 {
    cmp_bytes(string_bytes(a), string_bytes(b))
}

/// # Safety
///
/// Both operands are live bytes.
pub unsafe fn w_bytes_cmp(a: CelRef, b: CelRef) -> i64 {
    cmp_bytes(bytes_payload(a), bytes_payload(b))
}

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
///
/// It says nothing about what an arm returns, which is why the three chains
/// share it: arithmetic answers a [`CelRef`], `==` a `bool`, ordering a
/// `CMP_*` code.
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
#[inline]
pub unsafe fn cel_add(a: CelRef, b: CelRef) -> CelRef {
    let ta = class_of(a);
    let tb = class_of(b);
    if ta == tb {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_add,
            CEL_UINT_CLASS => w_uint_add,
            CEL_DOUBLE_CLASS => w_double_add,
            CEL_DURATION_CLASS => w_duration_add,
            CEL_STRING_CLASS => w_string_add,
            CEL_BYTES_CLASS => w_bytes_add,
            CEL_LIST_CLASS => w_list_add,
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
#[inline]
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
#[inline]
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
#[inline]
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
#[inline]
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

// -- comparison ------------------------------------------------------------
//
// Equality and ordering are two different operations here, not one with a
// projection: `==` is total and never raises, while `<` refuses a pair it has
// no ordering for. `Value` draws the same line — `OpCode::Equals` pushes
// `Value::Bool(lhs == rhs)` unconditionally, `OpCode::Less` goes through
// `objects::compare_values`, which can answer `NoSuchOverload`.

/// [`cel_compare`]'s answer: an ordering, or the refusal.
///
/// Plain `i64` codes rather than `Option<Ordering>`, for the reason
/// [`super::error`] gives for refusing `Result`. `Option<Ordering>` is a
/// data-carrying enum, and front-end B's only general enum-variant lowering is
/// anchored to `Result`, so every construction of one would arrive as an
/// `OpKind::New` plus a discriminant write — a materialized shell per
/// comparison. An integer is a register.
pub const CMP_LESS: i64 = -1;
/// Equal. See [`CMP_LESS`].
pub const CMP_EQUAL: i64 = 0;
/// Greater. See [`CMP_LESS`].
pub const CMP_GREATER: i64 = 1;
/// The operands have no ordering — `partial_cmp`'s `None`, which the four
/// operators turn into [`CelErrCode::NoSuchOverload`].
///
/// Deliberately not `0`: a caller that forgets the test reads it as some
/// ordering, never as "equal".
pub const CMP_INCOMPARABLE: i64 = 2;

/// Three-way compare of two signed words.
///
/// Branchless rather than an `if`/`else if` chain because the chain form on an
/// `Ord` type is what `clippy::comparison_chain` asks to be rewritten as a
/// `match` on `l.cmp(&r)`, which puts a `core::cmp` call in a graph that two
/// compares and a subtract keep out of it.
fn cmp_i64(l: i64, r: i64) -> i64 {
    (l > r) as i64 - (l < r) as i64
}

/// Three-way compare of two unsigned words. As [`cmp_i64`].
fn cmp_u64(l: u64, r: u64) -> i64 {
    (l > r) as i64 - (l < r) as i64
}

/// Three-way compare of two doubles, with `partial_cmp`'s `None` as
/// [`CMP_INCOMPARABLE`].
///
/// Written as a chain and not branchlessly: `NaN` answers false to `<`, `>` and
/// `==` alike, so the subtraction form would report it *equal* to everything.
/// The trailing arm is exactly the case where one operand is `NaN`.
fn cmp_f64(l: f64, r: f64) -> i64 {
    if l < r {
        CMP_LESS
    } else if l > r {
        CMP_GREATER
    } else if l == r {
        CMP_EQUAL
    } else {
        CMP_INCOMPARABLE
    }
}

/// The same comparison with the operands the other way round.
///
/// Not a negation: [`CMP_EQUAL`] and [`CMP_INCOMPARABLE`] are their own
/// reverses, and `-CMP_INCOMPARABLE` is not a code at all.
fn cmp_reverse(code: i64) -> i64 {
    if code == CMP_LESS {
        CMP_GREATER
    } else if code == CMP_GREATER {
        CMP_LESS
    } else {
        code
    }
}

/// Declare the ordering arm of one class, over one payload field.
macro_rules! ordered_arm {
    (
        $(#[$doc:meta])*
        $name:ident, $leaf:ty, $field:ident, $cmp:ident
    ) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// Both operands must be live values of this arm's class.
        pub unsafe fn $name(a: CelRef, b: CelRef) -> i64 {
            $cmp(payload!(a, $leaf, $field), payload!(b, $leaf, $field))
        }
    };
}

ordered_arm!(w_int_cmp, W_IntObject, intval, cmp_i64);
ordered_arm!(w_uint_cmp, W_UIntObject, uintval, cmp_u64);
ordered_arm!(w_double_cmp, W_DoubleObject, floatval, cmp_f64);
ordered_arm!(w_bool_cmp, W_BoolObject, boolval, cmp_i64);
ordered_arm!(w_duration_cmp, W_DurationObject, nanos, cmp_i64);
ordered_arm! {
    /// `timestamp` against `timestamp`, by instant.
    ///
    /// `off_s` is not read, here or in [`w_timestamp_eq`]: it records how the
    /// value was written, not when it happened, and `DateTime`'s own ordering
    /// compares instants across offsets.
    w_timestamp_cmp, W_TimestampObject, nanos, cmp_i64
}

/// Declare the equality arm of one class, over one payload field.
macro_rules! eq_arm {
    (
        $(#[$doc:meta])*
        $name:ident, $leaf:ty, $field:ident
    ) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// Both operands must be live values of this arm's class.
        pub unsafe fn $name(a: CelRef, b: CelRef) -> bool {
            payload!(a, $leaf, $field) == payload!(b, $leaf, $field)
        }
    };
}

eq_arm!(w_int_eq, W_IntObject, intval);
eq_arm!(w_uint_eq, W_UIntObject, uintval);
eq_arm! {
    /// `double` against `double`, IEEE-754: `NaN` equals nothing, itself
    /// included.
    w_double_eq, W_DoubleObject, floatval
}
eq_arm!(w_bool_eq, W_BoolObject, boolval);
eq_arm!(w_duration_eq, W_DurationObject, nanos);
eq_arm! {
    /// `timestamp` against `timestamp`, by instant. See [`w_timestamp_cmp`].
    w_timestamp_eq, W_TimestampObject, nanos
}
eq_arm! {
    /// Two type values are equal when they denote the same class.
    ///
    /// The payload is a `*const CelClass` and the classes are `'static`
    /// prebuilts, so identity of the denoted class *is* pointer identity of
    /// the payload.
    w_type_eq, W_TypeObject, cls
}

/// `null == null`, the only pair this arm is reached with.
///
/// A payloadless leaf has nothing to compare and two `null`s are always equal,
/// so the arm exists to keep `null` in the chain rather than in the
/// cross-class tail, where it would answer `false`.
///
/// # Safety
///
/// Both operands must be live `null` values.
pub unsafe fn w_null_eq(_a: CelRef, _b: CelRef) -> bool {
    true
}

/// `optional == optional`: two absent values are equal, an absent and a present
/// one are not, and two present ones defer to what they hold.
///
/// The absent test is `is_null` on the payload, not a pointer comparison
/// against a canonical none. A none is *any* optional whose `w_value` is null —
/// `new_optional_none` allocates a fresh one per call — so identity would
/// answer `false` for two nones.
///
/// The present case recurses through [`values_equal`] rather than comparing the
/// wrapped pointers, so `optional.of(1) == optional.of(1)` holds for two
/// separately boxed `1`s, and so the numeric cross-class arms stay reachable
/// through a wrapper.
///
/// # Safety
///
/// Both operands must be live `optional` values, and a present payload must be
/// a live value.
pub unsafe fn w_optional_eq(a: CelRef, b: CelRef) -> bool {
    let va = payload!(a, W_OptionalObject, w_value);
    let vb = payload!(b, W_OptionalObject, w_value);
    if va.is_null() || vb.is_null() {
        return va.is_null() && vb.is_null();
    }
    unsafe { values_equal(va, vb) }
}

/// `opaque == opaque` through the heap host table (D12).
///
/// # Safety
///
/// Both operands must be live [`super::object::W_OpaqueObject`] values.
pub unsafe fn w_opaque_eq(a: CelRef, b: CelRef) -> bool {
    super::convert::opaque_hosts_equal(
        super::object::opaque_host_index(a),
        super::object::opaque_host_index(b),
    )
}

// The cross-type numeric helpers. Three, not six: `==` is symmetric, and each
// reversed ordering is the forward one under `cmp_reverse`.
//
// None of them uses `try_into`, which is how `PartialEq`/`PartialOrd for Value`
// spell the narrowing conversion, because `TryFrom` yields a `Result` and a
// `Result` does not lower (see `super::error`). The sign tests below are the
// same predicates: an `i64` fails to be a `u64` exactly when it is negative,
// and a `u64` fails to be an `i64` exactly when it exceeds `i64::MAX` — which
// is what the source's own two comments say its `unwrap_or`s mean.
//
// `as f64` on an integer is lossy above 2^53. That is the pinned behaviour of
// the four int/double arms, reproduced rather than corrected: the oracle corpus
// answers with it.

/// Whether an `int` and a `uint` denote the same number.
fn eq_int_uint(l: i64, r: u64) -> bool {
    l >= 0 && (l as u64) == r
}

/// Whether an `int` and a `double` denote the same number.
fn eq_int_double(l: i64, r: f64) -> bool {
    (l as f64) == r
}

/// Whether a `uint` and a `double` denote the same number.
fn eq_uint_double(l: u64, r: f64) -> bool {
    (l as f64) == r
}

/// An `int` against a `uint`, in that order.
fn cmp_int_uint(l: i64, r: u64) -> i64 {
    if l < 0 {
        CMP_LESS
    } else {
        cmp_u64(l as u64, r)
    }
}

/// An `int` against a `double`, in that order.
fn cmp_int_double(l: i64, r: f64) -> i64 {
    cmp_f64(l as f64, r)
}

/// A `uint` against a `double`, in that order.
fn cmp_uint_double(l: u64, r: f64) -> i64 {
    cmp_f64(l as f64, r)
}

/// CEL `==`.
///
/// Total: no pair of values refuses, so this never touches [`super::error`] and
/// its result is always a live `bool`. Values of different classes are unequal
/// rather than an error — except across `int`, `uint` and `double`, which CEL
/// compares numerically.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_equals(a: CelRef, b: CelRef) -> CelRef {
    new_bool(values_equal(a, b)) as CelRef
}

/// CEL `!=`, the negation of [`cel_equals`].
///
/// # Safety
///
/// As [`cel_equals`].
pub unsafe fn cel_not_equals(a: CelRef, b: CelRef) -> CelRef {
    new_bool(!values_equal(a, b)) as CelRef
}

/// The predicate under [`cel_equals`].
///
/// A `bool` rather than a boxed one so `!=` can negate it without allocating a
/// value to read back.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn values_equal(a: CelRef, b: CelRef) -> bool {
    let ta = class_of(a);
    let tb = class_of(b);
    if ta == tb {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_eq,
            CEL_UINT_CLASS => w_uint_eq,
            CEL_DOUBLE_CLASS => w_double_eq,
            CEL_BOOL_CLASS => w_bool_eq,
            CEL_NULL_CLASS => w_null_eq,
            CEL_DURATION_CLASS => w_duration_eq,
            CEL_TIMESTAMP_CLASS => w_timestamp_eq,
            CEL_TYPE_CLASS => w_type_eq,
            CEL_OPTIONAL_CLASS => w_optional_eq,
            CEL_STRING_CLASS => w_string_eq,
            CEL_BYTES_CLASS => w_bytes_eq,
            CEL_LIST_CLASS => w_list_eq,
            CEL_MAP_CLASS => w_map_eq,
            CEL_OPAQUE_CLASS => w_opaque_eq,
        );
    }
    values_equal_mixed(a, b, ta, tb)
}

/// The cross-class arms of `==`: the six numeric pairs, and `false` for
/// everything else.
///
/// # Safety
///
/// As [`values_equal`], with `ta`/`tb` their classes.
unsafe fn values_equal_mixed(
    a: CelRef,
    b: CelRef,
    ta: *const CelClass,
    tb: *const CelClass,
) -> bool {
    let int = &CEL_INT_CLASS as *const CelClass;
    let uint = &CEL_UINT_CLASS as *const CelClass;
    let double = &CEL_DOUBLE_CLASS as *const CelClass;
    if ta == int {
        if tb == uint {
            return eq_int_uint(
                payload!(a, W_IntObject, intval),
                payload!(b, W_UIntObject, uintval),
            );
        }
        if tb == double {
            return eq_int_double(
                payload!(a, W_IntObject, intval),
                payload!(b, W_DoubleObject, floatval),
            );
        }
    } else if ta == uint {
        if tb == int {
            return eq_int_uint(
                payload!(b, W_IntObject, intval),
                payload!(a, W_UIntObject, uintval),
            );
        }
        if tb == double {
            return eq_uint_double(
                payload!(a, W_UIntObject, uintval),
                payload!(b, W_DoubleObject, floatval),
            );
        }
    } else if ta == double {
        if tb == int {
            return eq_int_double(
                payload!(b, W_IntObject, intval),
                payload!(a, W_DoubleObject, floatval),
            );
        }
        if tb == uint {
            return eq_uint_double(
                payload!(b, W_UIntObject, uintval),
                payload!(a, W_DoubleObject, floatval),
            );
        }
    }
    false
}

/// Three-way compare of two values, as one of the `CMP_*` codes.
///
/// `null` and `type` are absent from the chain on purpose, and the omission is
/// the behaviour rather than a gap in it. `PartialOrd for Value` does order
/// `Null` against `Null`, but `objects::compare_values` refuses on
/// `has_comparer(&lhs)` before it consults the ordering, so `null < null` is
/// `NoSuchOverload`; leaving both classes out reaches that same answer through
/// the one exit below, and every other pair those two classes can form has no
/// ordering either.
///
/// # Safety
///
/// Both operands must be live values.
pub unsafe fn cel_compare(a: CelRef, b: CelRef) -> i64 {
    let ta = class_of(a);
    let tb = class_of(b);
    if ta == tb {
        same_class_chain!(ta, a, b,
            CEL_INT_CLASS => w_int_cmp,
            CEL_UINT_CLASS => w_uint_cmp,
            CEL_DOUBLE_CLASS => w_double_cmp,
            CEL_BOOL_CLASS => w_bool_cmp,
            CEL_DURATION_CLASS => w_duration_cmp,
            CEL_TIMESTAMP_CLASS => w_timestamp_cmp,
            CEL_STRING_CLASS => w_string_cmp,
            CEL_BYTES_CLASS => w_bytes_cmp,
        );
    }
    cel_compare_slow(a, b, ta, tb)
}

/// The cross-class arms of ordering: the six numeric pairs, and
/// [`CMP_INCOMPARABLE`] for everything else.
///
/// The three reversed pairs go through [`cmp_reverse`], which reproduces the
/// source's arms rather than approximating them. `(uint, int)` is the one worth
/// checking: the source converts the `uint` to `i64` and answers `Greater` when
/// it does not fit, and `cmp_reverse(cmp_int_uint(int, uint))` answers the same
/// in each case — a `uint` above `i64::MAX` exceeds every `int`, a negative
/// `int` is below every `uint`, and otherwise both fit in `u64` and are
/// compared there.
///
/// # Safety
///
/// As [`cel_compare`], with `ta`/`tb` their classes.
unsafe fn cel_compare_slow(a: CelRef, b: CelRef, ta: *const CelClass, tb: *const CelClass) -> i64 {
    let int = &CEL_INT_CLASS as *const CelClass;
    let uint = &CEL_UINT_CLASS as *const CelClass;
    let double = &CEL_DOUBLE_CLASS as *const CelClass;
    if ta == int {
        if tb == uint {
            return cmp_int_uint(
                payload!(a, W_IntObject, intval),
                payload!(b, W_UIntObject, uintval),
            );
        }
        if tb == double {
            return cmp_int_double(
                payload!(a, W_IntObject, intval),
                payload!(b, W_DoubleObject, floatval),
            );
        }
    } else if ta == uint {
        if tb == int {
            return cmp_reverse(cmp_int_uint(
                payload!(b, W_IntObject, intval),
                payload!(a, W_UIntObject, uintval),
            ));
        }
        if tb == double {
            return cmp_uint_double(
                payload!(a, W_UIntObject, uintval),
                payload!(b, W_DoubleObject, floatval),
            );
        }
    } else if ta == double {
        if tb == int {
            return cmp_reverse(cmp_int_double(
                payload!(b, W_IntObject, intval),
                payload!(a, W_DoubleObject, floatval),
            ));
        }
        if tb == uint {
            return cmp_reverse(cmp_uint_double(
                payload!(b, W_UIntObject, uintval),
                payload!(a, W_DoubleObject, floatval),
            ));
        }
    }
    CMP_INCOMPARABLE
}

/// Declare one ordering operator over the [`cel_compare`] code.
///
/// The accept predicate is written into each operator rather than taken as the
/// `fn(Ordering) -> bool` `objects::compare_values` receives: a predicate
/// reached through a function pointer is the dynamic shape this module exists
/// to avoid.
///
/// `$op` is diagnostic only. The refusal an ordering produces is
/// `NoSuchOverload`, which carries neither operator nor operands, so no
/// spelling here is observable at the boundary — unlike the arithmetic ops,
/// whose strings are rendered by `UnsupportedBinaryOperator`.
macro_rules! ordering_op {
    (
        $(#[$doc:meta])*
        $name:ident, $op:literal, |$code:ident| $accept:expr
    ) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// Both operands must be live values.
        pub unsafe fn $name(a: CelRef, b: CelRef) -> CelRef {
            let $code = cel_compare(a, b);
            if $code == CMP_INCOMPARABLE {
                return raise(CelErrCode::NoSuchOverload, $op, a, b);
            }
            new_bool($accept) as CelRef
        }
    };
}

ordering_op! {
    /// CEL `<`.
    cel_less, "less", |code| code == CMP_LESS
}

ordering_op! {
    /// CEL `<=`.
    ///
    /// Spelled "not greater", the way `OpCode::LessEquals` accepts
    /// `o != Ordering::Greater`. The two agree because the refusal has already
    /// been taken above, so only the three ordering codes reach the predicate.
    cel_less_equals, "less_equals", |code| code != CMP_GREATER
}

ordering_op! {
    /// CEL `>`.
    cel_greater, "greater", |code| code == CMP_GREATER
}

ordering_op! {
    /// CEL `>=`. Spelled "not less", as [`cel_less_equals`] is spelled.
    cel_greater_equals, "greater_equals", |code| code != CMP_LESS
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;
    use crate::objects::compare_values;
    use crate::runtime::error::{clear_error, take_error};
    use crate::runtime::object::{new_null, new_optional, new_optional_none, new_type, w_type};
    use crate::{ExecutionError, Value};

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

    // -- comparison, against the `Value` oracle ----------------------------
    //
    // The two comparison tests below do not restate the semantics; they call
    // the production ones on the same operands. Whatever `PartialEq for Value`
    // and `objects::compare_values` answer is what the chains have to answer,
    // including the cases that are arguably wrong (`i64 as f64` above 2^53,
    // `null < null` refused while `Null.partial_cmp(&Null)` is `Equal`), which
    // is the point: the corpus is frozen against those answers.

    /// One value of every class the chains dispatch on, with the boundaries
    /// that separate the cross-type arms: the two conversion failures
    /// (`int` negative, `uint` above `i64::MAX`), and `NaN`/infinity.
    fn oracle_values() -> Vec<Value> {
        let mut values = vec![
            Value::Int(-1),
            Value::Int(0),
            Value::Int(1),
            Value::Int(i64::MIN),
            Value::Int(i64::MAX),
            Value::UInt(0),
            Value::UInt(1),
            Value::UInt(i64::MAX as u64),
            Value::UInt(u64::MAX),
            Value::Float(-1.0),
            Value::Float(0.0),
            Value::Float(1.0),
            Value::Float(f64::NAN),
            Value::Float(f64::INFINITY),
            Value::Bool(false),
            Value::Bool(true),
            Value::Null,
        ];
        #[cfg(feature = "chrono")]
        {
            values.push(Value::Duration(chrono::Duration::nanoseconds(5)));
            values.push(Value::Duration(chrono::Duration::nanoseconds(-5)));
            values.push(Value::Timestamp(
                chrono::DateTime::from_timestamp_nanos(7).fixed_offset(),
            ));
        }
        values
    }

    /// The class-family value denoting the same thing as `v`.
    fn boxed(v: &Value) -> CelRef {
        match v {
            Value::Int(i) => new_int(*i) as CelRef,
            Value::UInt(u) => new_uint(*u) as CelRef,
            Value::Float(f) => new_double(*f) as CelRef,
            Value::Bool(b) => new_bool(*b) as CelRef,
            Value::Null => new_null() as CelRef,
            #[cfg(feature = "chrono")]
            Value::Duration(d) => new_duration(d.num_nanoseconds().unwrap()) as CelRef,
            #[cfg(feature = "chrono")]
            Value::Timestamp(t) => new_timestamp(
                t.timestamp_nanos_opt().unwrap(),
                t.offset().local_minus_utc() as i64,
            ) as CelRef,
            other => panic!("no leaf for {other:?}"),
        }
    }

    /// Every ordered pair of [`oracle_values`] agrees with `PartialEq for
    /// Value`, in both `==` and `!=`.
    #[test]
    fn equality_agrees_with_the_value_operator() {
        fresh();
        let values = oracle_values();
        for lhs in &values {
            for rhs in &values {
                let want = lhs == rhs;
                unsafe {
                    let (l, r) = (boxed(lhs), boxed(rhs));
                    let eq = cel_equals(l, r);
                    assert_eq!(w_type(eq), &CEL_BOOL_CLASS as *const CelClass);
                    assert_eq!(
                        payload!(eq, W_BoolObject, boolval) != 0,
                        want,
                        "{lhs:?} == {rhs:?}",
                    );
                    let ne = cel_not_equals(l, r);
                    assert_eq!(
                        payload!(ne, W_BoolObject, boolval) != 0,
                        !want,
                        "{lhs:?} != {rhs:?}",
                    );
                }
            }
        }
        assert!(
            !super::super::error::has_error(),
            "equality is total and must never raise",
        );
    }

    /// How [`ordering_agrees_with_compare_values`] divided, so that the test
    /// cannot pass by never reaching a branch.
    ///
    /// A cross-check agrees trivially if the oracle refuses everything: both
    /// sides would answer "refused" for reasons that have nothing to do with
    /// each other. The counts below are the coverage claim, asserted at the
    /// end of the test rather than described in a comment.
    #[derive(Default)]
    struct Split {
        answered: u32,
        answered_cross_class: u32,
        refused: u32,
    }

    /// The answer `objects::compare_values` gives for one operator, required of
    /// the chain that replaces it.
    ///
    /// `got` is computed by the caller so that it lands in the error slot
    /// immediately before this reads it.
    unsafe fn agrees_with_compare_values(
        lhs: &Value,
        rhs: &Value,
        name: &str,
        accept: fn(Ordering) -> bool,
        got: CelRef,
        split: &mut Split,
    ) {
        match compare_values(lhs, rhs, accept) {
            Ok(Value::Bool(want)) => {
                split.answered += 1;
                if std::mem::discriminant(lhs) != std::mem::discriminant(rhs) {
                    split.answered_cross_class += 1;
                }
                assert_ne!(
                    got, ERROR_SENTINEL,
                    "{name}: {lhs:?} vs {rhs:?} refused, oracle answered {want}",
                );
                assert_eq!(w_type(got), &CEL_BOOL_CLASS as *const CelClass);
                assert_eq!(
                    payload!(got, W_BoolObject, boolval) != 0,
                    want,
                    "{name}: {lhs:?} vs {rhs:?}",
                );
            }
            Err(ExecutionError::NoSuchOverload) => {
                split.refused += 1;
                assert_eq!(
                    got, ERROR_SENTINEL,
                    "{name}: {lhs:?} vs {rhs:?} answered, oracle refused",
                );
                assert_eq!(take_error().unwrap().code, CelErrCode::NoSuchOverload);
            }
            other => panic!("unexpected oracle answer for {name}: {other:?}"),
        }
    }

    /// Every ordered pair of [`oracle_values`], through all four operators,
    /// agrees with `objects::compare_values`.
    #[test]
    fn ordering_agrees_with_compare_values() {
        let values = oracle_values();
        let mut split = Split::default();
        for lhs in &values {
            for rhs in &values {
                unsafe {
                    let (l, r) = (boxed(lhs), boxed(rhs));
                    fresh();
                    agrees_with_compare_values(
                        lhs,
                        rhs,
                        "less",
                        |o| o == Ordering::Less,
                        cel_less(l, r),
                        &mut split,
                    );
                    fresh();
                    agrees_with_compare_values(
                        lhs,
                        rhs,
                        "less_equals",
                        |o| o != Ordering::Greater,
                        cel_less_equals(l, r),
                        &mut split,
                    );
                    fresh();
                    agrees_with_compare_values(
                        lhs,
                        rhs,
                        "greater",
                        |o| o == Ordering::Greater,
                        cel_greater(l, r),
                        &mut split,
                    );
                    fresh();
                    agrees_with_compare_values(
                        lhs,
                        rhs,
                        "greater_equals",
                        |o| o != Ordering::Less,
                        cel_greater_equals(l, r),
                        &mut split,
                    );
                }
            }
        }
        // Measured over the table above: 712 answered (448 of them across two
        // classes) and 888 refused, of 20 x 20 x 4. The floors are a quarter of
        // the total rather than those figures, so adding a value to the table
        // does not have to restate them — but a change that collapses either
        // branch, or drops the cross-class arms out of reach, still fails here.
        let total = (values.len() * values.len() * 4) as u32;
        assert_eq!(
            split.answered + split.refused,
            total,
            "every case classified"
        );
        assert!(split.answered > total / 4, "answered {}", split.answered);
        assert!(split.refused > total / 4, "refused {}", split.refused);
        assert!(
            split.answered_cross_class > total / 8,
            "cross-class answers are the arms this test exists for: {}",
            split.answered_cross_class,
        );
    }

    /// The three codes and the refusal are distinct, and the refusal is not
    /// reachable by negating an ordering.
    #[test]
    fn compare_codes_are_the_four_the_operators_test() {
        fresh();
        unsafe {
            assert_eq!(
                cel_compare(new_int(1) as CelRef, new_int(2) as CelRef),
                CMP_LESS,
            );
            assert_eq!(
                cel_compare(new_int(2) as CelRef, new_int(2) as CelRef),
                CMP_EQUAL,
            );
            assert_eq!(
                cel_compare(new_int(3) as CelRef, new_int(2) as CelRef),
                CMP_GREATER,
            );
            // Two classes with no ordering between them, and one value with no
            // ordering at all.
            assert_eq!(
                cel_compare(new_int(1) as CelRef, new_bool(true) as CelRef),
                CMP_INCOMPARABLE,
            );
            assert_eq!(
                cel_compare(new_null() as CelRef, new_null() as CelRef),
                CMP_INCOMPARABLE,
            );
            assert_eq!(cmp_reverse(CMP_INCOMPARABLE), CMP_INCOMPARABLE);
            assert_eq!(cmp_reverse(CMP_EQUAL), CMP_EQUAL);
            assert_eq!(cmp_reverse(CMP_LESS), CMP_GREATER);
        }
        assert!(
            !super::super::error::has_error(),
            "cel_compare never raises"
        );
    }

    /// `null == null` is true while `null < null` refuses — the asymmetry
    /// `has_comparer` puts in front of `PartialOrd`, which the missing chain
    /// arm reproduces.
    #[test]
    fn null_compares_equal_and_refuses_to_order() {
        fresh();
        unsafe {
            let eq = cel_equals(new_null() as CelRef, new_null() as CelRef);
            assert_eq!(payload!(eq, W_BoolObject, boolval), 1);
            assert_eq!(
                cel_less(new_null() as CelRef, new_null() as CelRef),
                ERROR_SENTINEL,
            );
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::NoSuchOverload);
    }

    /// Type values compare by the class they denote, and carry no ordering.
    #[test]
    fn type_values_compare_by_denoted_class() {
        fresh();
        unsafe {
            let int_ty = new_type(&CEL_INT_CLASS) as CelRef;
            let also_int = new_type(&CEL_INT_CLASS) as CelRef;
            let uint_ty = new_type(&CEL_UINT_CLASS) as CelRef;
            assert_ne!(int_ty, also_int, "two allocations, not one interned value");
            assert!(values_equal(int_ty, also_int));
            assert!(!values_equal(int_ty, uint_ty));
            assert_eq!(cel_less(int_ty, uint_ty), ERROR_SENTINEL);
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::NoSuchOverload);
    }

    /// Timestamps compare by instant, so the offset a value was written with
    /// changes neither equality nor ordering.
    #[cfg(feature = "chrono")]
    #[test]
    fn timestamp_comparison_ignores_the_offset() {
        fresh();
        unsafe {
            let utc = new_timestamp(1_000, 0) as CelRef;
            let plus_nine = new_timestamp(1_000, 9 * 3_600) as CelRef;
            assert!(values_equal(utc, plus_nine));
            assert_eq!(cel_compare(utc, plus_nine), CMP_EQUAL);

            let later = new_timestamp(2_000, 9 * 3_600) as CelRef;
            assert_eq!(cel_compare(utc, later), CMP_LESS);
        }
        assert!(!super::super::error::has_error());
    }

    /// Optionals: absence is a null payload, not an identity.
    ///
    /// Graded against stated expectations rather than `compare_values`, which
    /// cannot reach this case: the walker spells an optional
    /// `Value::Opaque(Arc<OptionalValue>)`, and `has_comparer` refuses an
    /// opaque, so the oracle the other equality tests use answers nothing here.
    #[test]
    fn optionals_compare_by_presence_and_by_payload() {
        fresh();
        unsafe {
            let none = new_optional_none() as CelRef;
            let also_none = new_optional_none() as CelRef;
            assert_ne!(none, also_none, "two allocations, not one interned none");
            assert!(values_equal(none, also_none));

            let one = new_optional(new_int(1) as CelRef) as CelRef;
            let also_one = new_optional(new_int(1) as CelRef) as CelRef;
            let two = new_optional(new_int(2) as CelRef) as CelRef;
            assert!(
                values_equal(one, also_one),
                "equal by payload, not identity"
            );
            assert!(!values_equal(one, two));

            // Absent and present are never equal, in either argument order.
            assert!(!values_equal(none, one));
            assert!(!values_equal(one, none));

            // The payload recursion reaches the cross-class numeric arms, so a
            // wrapper does not make `1 == 1u` stop holding.
            let one_u = new_optional(new_uint(1) as CelRef) as CelRef;
            assert!(values_equal(one, one_u));
        }
        assert!(!super::super::error::has_error());
    }

    /// An optional carries no ordering, like `null` and `type`: it is absent
    /// from `cel_compare`'s chain, so the operators take the refusal.
    #[test]
    fn optionals_refuse_to_order() {
        fresh();
        unsafe {
            let one = new_optional(new_int(1) as CelRef) as CelRef;
            let two = new_optional(new_int(2) as CelRef) as CelRef;
            assert_eq!(cel_less(one, two), ERROR_SENTINEL);
        }
        assert_eq!(take_error().unwrap().code, CelErrCode::NoSuchOverload);
    }
}
