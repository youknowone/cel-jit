//! The header-first value class family.
//!
//! A port of `rclass.py`'s `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
//! hints={'immutable': True, 'shouldntbenull': True, 'typeptr': True})`,
//! spelled as `pyre-object`'s `PyObject`. Every value is a `#[repr(C)]` struct
//! whose **first** field is a [`CelObject`] header, so one raw-pointer type,
//! [`CelRef`], addresses all of them and the class word is always at offset 0.
//!
//! # Three ways to get this wrong, all of them silent
//!
//! The point of this family is that `fuse_boxing_alloc` turns an allocation
//! into a `NewWithVtable`, which the trace optimizer can then delete outright.
//! The fuse declines with a bare `continue` — no error, no warning, just a
//! worse graph — so each of its conditions is pinned by a test below rather
//! than left to be discovered as a disappointing census months later:
//!
//! 1. **The allocation call.** Its path must end `lltype::malloc_typed` and it
//!    must take exactly one argument, the finished value, by value. See
//!    [`super::lltype`].
//! 2. **The `w_class` word.** `resolve_vtable_addr` keeps a single vtable
//!    address to stand in for both header stores, and verifies the
//!    substitution is sound by reading the `w_class` store back through
//!    `get_instantiate_arg_addr` and comparing it to the `ob_type` address. A
//!    header without that store resolves to `None`, which compares unequal,
//!    and the whole cluster declines. See [`super::pyre_object`].
//! 3. **The header offset.** `fuse_boxing_alloc` matches the aggregate's
//!    `ob_header` store by name and the backend reads the class word at a
//!    fixed offset (`set_vtable_offset(Some(0))` emits
//!    `cmp [obj + 0], classptr`). Every leaf therefore const-asserts
//!    `offset_of!(T, ob_header) == 0`.
//!
//! # The header is two words, and that is a correction
//!
//! An earlier design took one word — `CelObject { ob_type }` — reasoning that
//! pyre needs `w_class` for user-defined Python classes while CEL's universe
//! is closed, so the second word carries nothing. The reasoning is sound and
//! the conclusion does not survive contact with `resolve_vtable_addr`, which
//! is condition 2 above: with no `w_class` store the fuse returns 0 and
//! **every** allocation stays residual, in silence. Two words is what the
//! lowering is actually tested against, in `charon-corpus`.
//!
//! Recovering the word is a majit change, not a cel one: teach
//! `resolve_vtable_addr` to read an absent `w_class` store as "base-type
//! instance" rather than as a mismatch. That is sound precisely for a universe
//! without subclassing, which is CEL's. It is worth doing — the word is paid
//! on every value — but it is a separate change with its own test, and
//! nothing here depends on which way it goes: `#[repr(C)]` with `ob_header`
//! first means narrowing the header changes each leaf's *size* and no leaf's
//! *source*, and the offset assertions hold either way.

use core::mem::offset_of;

use super::lltype;
use super::pyre_object::pyobject::get_instantiate;

/// The coarse family a value belongs to.
///
/// A fieldless `#[repr(u8)]` enum, so a read of it is an integer load the
/// annotator can narrow, not a pointer chase. It is a *summary*: dispatch
/// tests the class pointer itself ([`w_type`]), because pointer identity is
/// what gives the annotator its `knowntypedata` narrowing. This exists for the
/// cases that genuinely want the family rather than the type — error messages,
/// and CEL's own `type()` grouping.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CelKind {
    Null = 0,
    Bool = 1,
    Int = 2,
    UInt = 3,
    Double = 4,
    Timestamp = 5,
    Duration = 6,
    Type = 7,
    Optional = 8,
}

/// One value class.
///
/// Held only as a `'static`: the fuse resolves the vtable to a constant
/// address, which requires the class to be a prebuilt whose address is known
/// at compile time.
///
/// The collector's type id and the `subclassrange_{min,max}` pair that
/// `freeze_types()` assigns are deliberately **absent**. They arrive with the
/// registration mechanism that fills them; declaring them now as immutable
/// fields nothing can write would be a shape that has to be undone.
#[repr(C)]
#[derive(Debug)]
pub struct CelClass {
    pub name: &'static str,
    pub kind: CelKind,
}

impl CelClass {
    const fn new(name: &'static str, kind: CelKind) -> Self {
        CelClass { name, kind }
    }
}

/// The object header, first field of every value.
///
/// `w_class` is always `ob_type` for CEL — see the module documentation for
/// why the redundant word is here anyway.
#[repr(C)]
pub struct CelObject {
    pub ob_type: *const CelClass,
    pub w_class: *const CelClass,
}

/// A pointer to any value.
///
/// Every leaf starts with a [`CelObject`], so this addresses all of them and
/// [`w_type`] is valid on any of them.
pub type CelRef = *mut CelObject;

/// The class of `w`.
///
/// The shape matters as much as the result: a deref of a raw pointer under a
/// field access is what the frontend recognises, inserting a
/// `__pyre_cast_instance` narrow so the read lowers to a *typed* `FieldRead`
/// rather than a classdef-less one that stalls the annotator downstream.
///
/// # Safety
///
/// `w` must point at a live value allocated through [`super::lltype`].
#[inline]
pub unsafe fn w_type(w: CelRef) -> *const CelClass {
    unsafe { (*w).ob_type }
}

/// The family of `w`.
///
/// # Safety
///
/// As [`w_type`].
#[inline]
pub unsafe fn w_kind(w: CelRef) -> CelKind {
    unsafe { (*w_type(w)).kind }
}

/// Read a payload field that follows the header.
///
/// The leaves are `#[repr(C)]` with the header first, so a `CelRef` known to
/// be of class `T` casts to `*mut T` without adjustment.
///
/// Expands to a bare dereference, so every use site must already be an unsafe
/// context — an `unsafe` block of its own would be redundant inside the
/// `unsafe fn`s that read a payload and would warn.
macro_rules! payload {
    ($w:expr, $leaf:ty, $field:ident) => {
        (*($w as *mut $leaf)).$field
    };
}

pub(crate) use payload;

/// Declare a fixed-size leaf with one payload field.
///
/// One macro rather than eight hand-written copies, because the constructor
/// body is the shape the boxing fuse matches: written out per leaf it would be
/// eight chances for a spelling to drift out of recognition, and a drifted one
/// reports nothing.
macro_rules! scalar_leaf {
    (
        $(#[$leaf_doc:meta])*
        $leaf:ident { $payload:ident : $pty:ty }
        $(#[$class_doc:meta])*
        $class:ident = ($name:literal, $kind:expr)
        $(#[$ctor_doc:meta])*
        $ctor:ident
        $marker:ident
    ) => {
        $(#[$leaf_doc])*
        #[repr(C)]
        #[allow(non_camel_case_types)]
        pub struct $leaf {
            pub ob_header: CelObject,
            pub $payload: $pty,
        }

        $(#[$class_doc])*
        pub static $class: CelClass = CelClass::new($name, $kind);

        // Condition 3: the class word must be at offset 0.
        const _: () = {
            assert!(offset_of!($leaf, ob_header) == 0);
        };

        /// The payload is written once, at allocation, and never again, so its
        /// reads may fold to a pure getfield. `harvest_immutable_fields_from_llbcs`
        /// collects this marker by its `_immutable_fields_` prefix.
        #[allow(non_upper_case_globals)]
        pub const $marker: &str = stringify!($payload);

        $(#[$ctor_doc])*
        pub fn $ctor(value: $pty) -> *mut $leaf {
            lltype::malloc_typed($leaf {
                ob_header: CelObject {
                    ob_type: &$class,
                    w_class: get_instantiate(&$class),
                },
                $payload: value,
            })
        }
    };
}

scalar_leaf! {
    /// A CEL `int`: a signed 64-bit integer.
    W_IntObject { intval: i64 }
    CEL_INT_CLASS = ("int", CelKind::Int)
    /// Box `value` as a CEL `int`.
    new_int
    _immutable_fields_W_IntObject
}

scalar_leaf! {
    /// A CEL `uint`: an unsigned 64-bit integer, a distinct type from `int`.
    W_UIntObject { uintval: u64 }
    CEL_UINT_CLASS = ("uint", CelKind::UInt)
    /// Box `value` as a CEL `uint`.
    new_uint
    _immutable_fields_W_UIntObject
}

scalar_leaf! {
    /// A CEL `double`.
    W_DoubleObject { floatval: f64 }
    CEL_DOUBLE_CLASS = ("double", CelKind::Double)
    /// Box `value` as a CEL `double`.
    new_double
    _immutable_fields_W_DoubleObject
}

scalar_leaf! {
    /// A CEL `bool`.
    ///
    /// The payload is `i64` rather than `bool` so the field is a full machine
    /// word: the backend's fields are typed `Signed`, and a narrower slot would
    /// need a widening read on every access for no space saved once the header
    /// is present.
    W_BoolObject { boolval: i64 }
    CEL_BOOL_CLASS = ("bool", CelKind::Bool)
    /// Box `value` as a CEL `bool`.
    new_bool_raw
    _immutable_fields_W_BoolObject
}

scalar_leaf! {
    /// A CEL `duration`, as a whole number of nanoseconds.
    ///
    /// Nanoseconds rather than a split seconds/subsecond pair because every
    /// arithmetic operation on a duration would otherwise have to normalise,
    /// and the range at 64 bits is ±292 years.
    W_DurationObject { nanos: i64 }
    CEL_DURATION_CLASS = ("duration", CelKind::Duration)
    /// Box `nanos` as a CEL `duration`.
    new_duration
    _immutable_fields_W_DurationObject
}

scalar_leaf! {
    /// A CEL `optional`, holding the value it wraps or nothing.
    ///
    /// A NULL `w_value` is the none case. That spelling, rather than a
    /// discriminant beside the payload, keeps the leaf one header plus one
    /// word and keeps the none test a null check on a field that is read
    /// anyway.
    ///
    /// The payload is the family's first MANAGED edge — every other leaf holds
    /// a scalar. Nothing traces it yet, because nothing traces any of them:
    /// [`CelClass`] carries no collector type id, and the offset list that
    /// would name this field arrives with the registration mechanism that fills
    /// it.
    W_OptionalObject { w_value: CelRef }
    CEL_OPTIONAL_CLASS = ("optional_type", CelKind::Optional)
    /// Wrap `value` as a present CEL `optional`.
    ///
    /// The none case is [`new_optional_none`], not this function called with a
    /// null: there the null is a literal inside the allocation body rather than
    /// a value arriving as an argument.
    new_optional
    _immutable_fields_W_OptionalObject
}

/// Allocate the absent CEL `optional`.
///
/// Written out rather than delegating to [`new_optional`] so the null it stores
/// is spelled at the allocation site. A fresh allocation per call, for the same
/// reason [`new_null`] is: interning waits for an allocator that can mint
/// immortal objects.
pub fn new_optional_none() -> *mut W_OptionalObject {
    lltype::malloc_typed(W_OptionalObject {
        ob_header: CelObject {
            ob_type: &CEL_OPTIONAL_CLASS,
            w_class: get_instantiate(&CEL_OPTIONAL_CLASS),
        },
        w_value: core::ptr::null_mut(),
    })
}

/// A CEL `null`.
///
/// The one leaf with no payload, so it is written out rather than passed
/// through [`scalar_leaf`]: the macro's shape is a header plus exactly one
/// field, and widening it to cover a payloadless case would make the
/// fuse-critical constructor body conditional.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_NullObject {
    pub ob_header: CelObject,
}

/// The class of [`W_NullObject`].
pub static CEL_NULL_CLASS: CelClass = CelClass::new("null_type", CelKind::Null);

const _: () = {
    assert!(offset_of!(W_NullObject, ob_header) == 0);
};

/// Allocate a CEL `null`.
///
/// A fresh allocation per call for now. `null` is the canonical singleton
/// candidate, but a prebuilt has to carry a real GC header — a plain Rust
/// `static` would put the class word in read-only memory where the backend's
/// header-relative reads do not point — so interning it waits for the
/// allocator that can mint immortal objects.
pub fn new_null() -> *mut W_NullObject {
    lltype::malloc_typed(W_NullObject {
        ob_header: CelObject {
            ob_type: &CEL_NULL_CLASS,
            w_class: get_instantiate(&CEL_NULL_CLASS),
        },
    })
}

/// A CEL `timestamp`.
///
/// Two payload fields — nanoseconds since the epoch and the offset in seconds
/// the value was written with — so it does not go through [`scalar_leaf`]
/// either. CEL compares timestamps by instant and formats them by offset, so
/// dropping the offset would be lossy at the boundary.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_TimestampObject {
    pub ob_header: CelObject,
    pub nanos: i64,
    pub off_s: i64,
}

/// The class of [`W_TimestampObject`].
pub static CEL_TIMESTAMP_CLASS: CelClass = CelClass::new("timestamp", CelKind::Timestamp);

const _: () = {
    assert!(offset_of!(W_TimestampObject, ob_header) == 0);
};

/// Both payload fields are write-once.
#[allow(non_upper_case_globals)]
pub const _immutable_fields_W_TimestampObject: &str = "nanos,off_s";

/// Box an instant as a CEL `timestamp`.
pub fn new_timestamp(nanos: i64, off_s: i64) -> *mut W_TimestampObject {
    lltype::malloc_typed(W_TimestampObject {
        ob_header: CelObject {
            ob_type: &CEL_TIMESTAMP_CLASS,
            w_class: get_instantiate(&CEL_TIMESTAMP_CLASS),
        },
        nanos,
        off_s,
    })
}

/// A CEL type value — what `type(x)` evaluates to.
///
/// `cls` is the class of the type this value *denotes*, while the header's own
/// `ob_type` is [`CEL_TYPE_CLASS`]. The two being different is what makes
/// `type(type(1)) == type(string)` hold, and it is why the two spellings must
/// not be collapsed: `(*type_value).cls` is a class, `(*any_value).ob_type` is
/// a class, and only the type value itself is an allocated object.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_TypeObject {
    pub ob_header: CelObject,
    pub cls: *const CelClass,
}

/// The class of [`W_TypeObject`] — the type of a type.
pub static CEL_TYPE_CLASS: CelClass = CelClass::new("type", CelKind::Type);

const _: () = {
    assert!(offset_of!(W_TypeObject, ob_header) == 0);
};

/// Written once at allocation.
#[allow(non_upper_case_globals)]
pub const _immutable_fields_W_TypeObject: &str = "cls";

/// The type value denoting `cls`.
pub fn new_type(cls: &'static CelClass) -> *mut W_TypeObject {
    lltype::malloc_typed(W_TypeObject {
        ob_header: CelObject {
            ob_type: &CEL_TYPE_CLASS,
            w_class: get_instantiate(&CEL_TYPE_CLASS),
        },
        cls,
    })
}

/// Box `value` as a CEL `bool`.
///
/// Thin wrapper over [`new_bool_raw`] so callers do not spell the `i64`
/// payload convention at every site.
pub fn new_bool(value: bool) -> *mut W_BoolObject {
    new_bool_raw(i64::from(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Condition 3, checked at run time as well as in the const asserts: a
    /// leaf pointer and its header pointer are the same address, which is what
    /// lets one [`CelRef`] address every leaf.
    #[test]
    fn every_leaf_starts_with_its_header() {
        let w = new_int(7);
        assert_eq!(w as usize, w as CelRef as usize);
        let t = new_timestamp(1, 2);
        assert_eq!(t as usize, t as CelRef as usize);
        let n = new_null();
        assert_eq!(n as usize, n as CelRef as usize);
    }

    /// Condition 2's cel-side half: `w_class` and `ob_type` must be the same
    /// address, or `resolve_vtable_addr` declines the fuse for that cluster.
    /// This does not prove the fuse fires — that needs the lowering — but a
    /// constructor that broke the invariant would fail here rather than
    /// showing up as an unexplained zero in a census.
    #[test]
    fn every_constructor_agrees_on_ob_type_and_w_class() {
        fn check(w: CelRef, expected: *const CelClass) {
            unsafe {
                assert_eq!((*w).ob_type, expected);
                assert_eq!((*w).w_class, expected, "w_class must equal ob_type");
            }
        }
        check(new_int(1) as CelRef, &CEL_INT_CLASS);
        check(new_uint(1) as CelRef, &CEL_UINT_CLASS);
        check(new_double(1.0) as CelRef, &CEL_DOUBLE_CLASS);
        check(new_bool(true) as CelRef, &CEL_BOOL_CLASS);
        check(new_null() as CelRef, &CEL_NULL_CLASS);
        check(new_duration(1) as CelRef, &CEL_DURATION_CLASS);
        check(new_timestamp(1, 0) as CelRef, &CEL_TIMESTAMP_CLASS);
        check(new_type(&CEL_INT_CLASS) as CelRef, &CEL_TYPE_CLASS);
    }

    #[test]
    fn payloads_survive_the_round_trip() {
        unsafe {
            assert_eq!((*new_int(-9)).intval, -9);
            assert_eq!((*new_uint(9)).uintval, 9);
            assert_eq!((*new_double(0.5)).floatval, 0.5);
            assert_eq!((*new_bool(true)).boolval, 1);
            assert_eq!((*new_bool(false)).boolval, 0);
            assert_eq!((*new_duration(-3)).nanos, -3);
            let t = &*new_timestamp(11, -5);
            assert_eq!((t.nanos, t.off_s), (11, -5));
        }
    }

    /// The class pointer is the dispatch key, so distinct types must not share
    /// one. Equally, `type(1)` and `type(2)` must share theirs, or a narrowing
    /// chain would never fold.
    #[test]
    fn class_pointers_are_per_type_identities() {
        let classes: [*const CelClass; 8] = [
            &CEL_INT_CLASS,
            &CEL_UINT_CLASS,
            &CEL_DOUBLE_CLASS,
            &CEL_BOOL_CLASS,
            &CEL_NULL_CLASS,
            &CEL_DURATION_CLASS,
            &CEL_TIMESTAMP_CLASS,
            &CEL_TYPE_CLASS,
        ];
        for (i, a) in classes.iter().enumerate() {
            for b in &classes[i + 1..] {
                assert_ne!(a, b, "two classes share one address");
            }
        }
        unsafe {
            assert_eq!(w_type(new_int(1) as CelRef), w_type(new_int(2) as CelRef));
        }
    }

    /// A type value's own class is `type`, while the class it denotes is its
    /// payload — the distinction `type(type(1)) == type(string)` rests on.
    #[test]
    fn a_type_values_own_class_is_type() {
        unsafe {
            let int_type = new_type(&CEL_INT_CLASS);
            assert_eq!((*int_type).cls, &CEL_INT_CLASS as *const CelClass);
            assert_eq!(
                w_type(int_type as CelRef),
                &CEL_TYPE_CLASS as *const CelClass
            );

            let string_stand_in = new_type(&CEL_BOOL_CLASS);
            assert_eq!(
                w_type(int_type as CelRef),
                w_type(string_stand_in as CelRef),
                "every type value shares one class",
            );
        }
    }

    #[test]
    fn kinds_come_off_the_class_word() {
        unsafe {
            assert_eq!(w_kind(new_int(1) as CelRef), CelKind::Int);
            assert_eq!(w_kind(new_uint(1) as CelRef), CelKind::UInt);
            assert_eq!(w_kind(new_null() as CelRef), CelKind::Null);
            assert_eq!(w_kind(new_type(&CEL_INT_CLASS) as CelRef), CelKind::Type);
        }
    }
}
