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
//! 2. **The class word.** `resolve_vtable_addr` keeps a single vtable address
//!    to stand in for every header store the fuse drops, so it has to be sure
//!    no dropped store carried a *different* class. Where the header declares
//!    a `w_class` field it reads that store back through
//!    `get_instantiate_arg_addr` and compares; where the header declares no
//!    such field there is nothing that could disagree, which is the arm
//!    `header_declares_no_class_word` admits and the one every leaf here
//!    takes. Declaring the field and omitting the store still declines, in
//!    silence.
//! 3. **The header offset.** `fuse_boxing_alloc` matches the aggregate's
//!    `ob_header` store by name and the backend reads the class word at a
//!    fixed offset (`set_vtable_offset(Some(0))` emits
//!    `cmp [obj + 0], classptr`). Every leaf therefore const-asserts
//!    `offset_of!(T, ob_header) == 0`.
//!
//! # The header is one word, and the word was recovered upstream
//!
//! An earlier revision of this file carried two — `ob_type` and a `w_class`
//! that was always equal to it — and said so as a correction. It was one at
//! first, reasoning that pyre needs `w_class` for user-defined Python classes
//! while CEL's universe is closed; the reasoning was sound and did not survive
//! contact with `resolve_vtable_addr`, which back then resolved an absent
//! `w_class` store to `None`, compared it unequal to the `ob_type` address,
//! and declined the whole cluster. Every allocation stayed residual, silently
//! — the failure mode this list exists to pin.
//!
//! The word came back on the majit side rather than here, by widening that
//! check rather than by re-spelling anything in cel:
//! `header_declares_no_class_word` reads a header that *declares* no class
//! word as a base-type instance, requiring both that no `w_class` store exists
//! and that the header struct's registered layout has no such field. What that
//! arm gives up is subclassing, which a CEL value universe does not have: the
//! type object of an instance of `T` is `T` itself. What it buys is a word on
//! every value.
//!
//! Gone with the field is the `get_instantiate` call the old check read its
//! argument out of, and with that call the module that carried pyre's name so
//! the three-segment path suffix would match.

use core::mem::{align_of, offset_of};

use super::lltype;
use super::object_array::{self, CelBytesBlock, CelItemsBlock};

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
    Bytes = 9,
    Str = 10,
    List = 11,
    Map = 12,
    #[cfg(feature = "structs")]
    Struct = 13,
    /// A foreign host object. The leaf is [`W_OpaqueObject`]; the host itself
    /// lives in the heap's side table (D12).
    Opaque = 14,
    /// The activation record. Not a CEL value; `type()` never answers this.
    Frame = 15,
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
/// One word, and declaring no class word beyond it is what admits the fuse's
/// base-type arm — see the module documentation.
#[repr(C)]
pub struct CelObject {
    pub ob_type: *const CelClass,
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
    ) => {
        $(#[$leaf_doc])*
        // The payload is written once, at allocation, and never again, so its
        // reads may fold to a pure getfield. The attribute leaves the
        // `_immutable_fields_<Struct>` marker Charon extracts; spelling it by
        // hand here is what the marker's own consumer stopped needing.
        #[cfg_attr(feature = "jit", majit_macros::jit_immutable_fields($payload))]
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

        $(#[$ctor_doc])*
        #[inline]
        pub fn $ctor(value: $pty) -> *mut $leaf {
            lltype::malloc_typed($leaf {
                ob_header: CelObject { ob_type: &$class },
                $payload: value,
            })
        }
    };
}

scalar_leaf! {
    /// A CEL `int`: a signed 64-bit integer.
    W_IntObject { intval: i64 }
    CEL_INT_CLASS = ("int", CelKind::Int)
    /// Allocate a fresh `int`. The interned range goes through [`new_int`].
    new_int_raw
}

scalar_leaf! {
    /// A CEL `uint`: an unsigned 64-bit integer, a distinct type from `int`.
    W_UIntObject { uintval: u64 }
    CEL_UINT_CLASS = ("uint", CelKind::UInt)
    /// Box `value` as a CEL `uint`.
    new_uint
}

scalar_leaf! {
    /// A CEL `double`.
    W_DoubleObject { floatval: f64 }
    CEL_DOUBLE_CLASS = ("double", CelKind::Double)
    /// Box `value` as a CEL `double`.
    new_double
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
}

/// Allocate the absent CEL `optional`.
///
/// Written out rather than delegating to [`new_optional`] so the null it stores
/// is spelled at the allocation site. A fresh allocation per call: the none
/// payload is a `CelRef`, so the immortal path — pointer-free leaves only —
/// cannot hold it.
pub fn new_optional_none() -> *mut W_OptionalObject {
    lltype::malloc_typed(W_OptionalObject {
        ob_header: CelObject {
            ob_type: &CEL_OPTIONAL_CLASS,
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
/// The one process-wide null. A plain Rust `static` would put the class
/// word in read-only memory, where a header-relative `guard_is_object`
/// load does not point; the immortal allocator puts a header word in
/// front of the payload.
pub fn new_null() -> *mut W_NullObject {
    *NULL.get_or_init(|| {
        lltype::malloc_typed_immortal(W_NullObject {
            ob_header: CelObject {
                ob_type: &CEL_NULL_CLASS,
            },
        }) as usize
    }) as *mut W_NullObject
}

/// A CEL `timestamp`.
///
/// Two payload fields — nanoseconds since the epoch and the offset in seconds
/// the value was written with — so it does not go through [`scalar_leaf`]
/// either. CEL compares timestamps by instant and formats them by offset, so
/// dropping the offset would be lossy at the boundary.
#[cfg_attr(feature = "jit", majit_macros::jit_immutable_fields(nanos, off_s))]
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

/// Box an instant as a CEL `timestamp`.
pub fn new_timestamp(nanos: i64, off_s: i64) -> *mut W_TimestampObject {
    lltype::malloc_typed(W_TimestampObject {
        ob_header: CelObject {
            ob_type: &CEL_TIMESTAMP_CLASS,
        },
        nanos,
        off_s,
    })
}

// -- the variable-length leaves ------------------------------------------
//
// Each is a FIXED-size leaf holding a pointer to a payload block allocated by
// [`super::object_array`], which is where the encoding is argued. The short
// version: a varsize tail cannot be passed by value, so it cannot reach the
// fuse, and upstream forbids a GC array inlined in a struct anyway.
//
// This makes them ordinary members of the family as far as the fuse is
// concerned — a header plus scalar and pointer fields, exactly the shape
// `W_OptionalObject` is measured fusing. What they add is a residual call per
// value for the block, which the fuse never sees and the optimizer cannot
// delete.

/// A CEL `bytes`.
#[cfg_attr(feature = "jit", majit_macros::jit_immutable_fields(data, length))]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_BytesObject {
    pub ob_header: CelObject,
    /// The payload, or null for a value whose block has not been built. Its
    /// offset-0 word is the allocated capacity, not this length.
    pub data: *mut CelBytesBlock,
    /// Live length in bytes. Upstream's `("length", Signed)` on the wrapper:
    /// the block's word is a capacity and a shrink must not move it.
    pub length: i64,
    /// Non-owning pointer at the public `Arc<Vec<u8>>` this leaf was wrapped
    /// from, or null if the bytes were allocated by the VM.
    pub public: *const (),
}

pub static CEL_BYTES_CLASS: CelClass = CelClass::new("bytes", CelKind::Bytes);

const _: () = {
    assert!(offset_of!(W_BytesObject, ob_header) == 0);
};

/// Box `bytes` as a CEL `bytes`.
///
/// The block is built BEFORE the allocation call, not inside it: the fuse
/// matches a call taking exactly one argument, the finished value, so anything
/// the value needs has to already exist when that call is made.
pub fn new_bytes(bytes: &[u8]) -> *mut W_BytesObject {
    let data = object_array::new_bytes_block(bytes);
    let length = bytes.len() as i64;
    lltype::malloc_typed(W_BytesObject {
        ob_header: CelObject {
            ob_type: &CEL_BYTES_CLASS,
        },
        data,
        length,
        public: core::ptr::null(),
    })
}

/// A CEL `string`.
///
/// Its own leaf rather than a `bytes` with a different class word, because the
/// two are different CEL types with different operations, and a shared leaf
/// would make the class word the only thing separating them at every use site.
///
/// The payload is UTF-8, so `byte_len` is what indexes the block and is not the
/// character count.
#[cfg_attr(feature = "jit", majit_macros::jit_immutable_fields(chars, byte_len))]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_StringObject {
    pub ob_header: CelObject,
    pub chars: *mut CelBytesBlock,
    pub byte_len: i64,
    /// Non-owning pointer at the public `Arc<String>` this leaf was wrapped
    /// from, or null if the string was allocated by the VM.
    pub public: *const (),
}

pub static CEL_STRING_CLASS: CelClass = CelClass::new("string", CelKind::Str);

const _: () = {
    assert!(offset_of!(W_StringObject, ob_header) == 0);
};

/// Box `s` as a CEL `string`.
pub fn new_string(s: &str) -> *mut W_StringObject {
    let chars = object_array::new_bytes_block(s.as_bytes());
    let byte_len = s.len() as i64;
    lltype::malloc_typed(W_StringObject {
        ob_header: CelObject {
            ob_type: &CEL_STRING_CLASS,
        },
        chars,
        byte_len,
        public: core::ptr::null(),
    })
}

/// How a [`W_ListObject`] holds its elements.
///
/// An inline tag, not a strategy object: a discriminant read plus a
/// narrowing chain, the same guard-then-fold as a typeptr and one fewer
/// object. D4.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListStrategy {
    /// Boxed [`CelRef`]s in [`W_ListObject::items`].
    Object = 0,
    /// Unboxed `i64`s in a [`W_IntColumn`] at [`W_ListObject::storage`].
    Ints = 1,
    /// A column or record window parked in the heap host table. The
    /// schema/buffer stays shared; `start`/`length` are the window.
    Window = 2,
}

/// An unboxed integer column. Not pointer-traced; the payload is raw `i64`s.
#[cfg_attr(feature = "jit", majit_macros::jit_immutable_fields(data, length))]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_IntColumn {
    pub ob_header: CelObject,
    pub data: *mut i64,
    pub length: i64,
}

pub static CEL_INT_COLUMN_CLASS: CelClass = CelClass::new("int_column", CelKind::List);

const _: () = {
    assert!(offset_of!(W_IntColumn, ob_header) == 0);
};

/// Box `values` as an unboxed int column.
pub fn new_int_column(values: &[i64]) -> *mut W_IntColumn {
    let length = values.len() as i64;
    let data = if values.is_empty() {
        core::ptr::null_mut()
    } else {
        let bytes = values.len() * core::mem::size_of::<i64>();
        let ptr = super::heap::with_heap(|h| h.alloc_raw(bytes, align_of::<i64>())) as *mut i64;
        unsafe {
            core::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
        }
        ptr
    };
    lltype::malloc_typed(W_IntColumn {
        ob_header: CelObject {
            ob_type: &CEL_INT_COLUMN_CLASS,
        },
        data,
        length,
    })
}

/// A CEL `list`.
///
/// Strategy tag + storage + window, the `listobject.py` shape. Object
/// lists keep their items block on the leaf (the block is not a
/// class-family value); int columns and host windows live at `storage`.
#[cfg_attr(
    feature = "jit",
    majit_macros::jit_immutable_fields(strategy, storage, items, start, length)
)]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_ListObject {
    pub ob_header: CelObject,
    pub strategy: ListStrategy,
    /// [`W_IntColumn`] or a [`W_OpaqueObject`] holding a window host.
    pub storage: CelRef,
    pub items: *mut CelItemsBlock,
    pub start: i64,
    pub length: i64,
    /// Non-owning pointer at the public list buffer this leaf was wrapped
    /// from, or null if the list was allocated by the VM. The Context that
    /// performed the wrap keeps the owning handle alive.
    pub public: *const (),
    pub public_start: u32,
    pub public_len: u32,
}

pub static CEL_LIST_CLASS: CelClass = CelClass::new("list", CelKind::List);

const _: () = {
    assert!(offset_of!(W_ListObject, ob_header) == 0);
};

/// ⚠ Immutable in the sense the JIT means: written once at allocation, so a
/// read may fold. CEL lists are immutable values, so this is not the bet it
/// Box `values` as a CEL `list`.
///
/// The elements are the family's first MULTIPLE managed edges from one value,
/// and they live in the block rather than in the leaf. Nothing traces them yet,
/// for the same reason nothing traces [`W_OptionalObject`]'s single edge.
/// Live length of a list leaf.
///
/// # Safety
///
/// `w` is a live [`W_ListObject`].
pub unsafe fn list_len(w: CelRef) -> i64 {
    (*w.cast::<W_ListObject>()).length
}

/// Live entry count of a map leaf.
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
pub unsafe fn map_len(w: CelRef) -> i64 {
    (*w.cast::<W_MapObject>()).length
}

/// UTF-8 byte length of a string leaf. Matches `Arc<String>::len`.
///
/// # Safety
///
/// `w` is a live [`W_StringObject`].
pub unsafe fn string_byte_len(w: CelRef) -> i64 {
    (*w.cast::<W_StringObject>()).byte_len
}

/// Borrow the UTF-8 payload of a string leaf.
///
/// # Safety
///
/// `w` is a live [`W_StringObject`] that outlives the returned slice.
pub unsafe fn string_as_str<'a>(w: CelRef) -> Option<&'a str> {
    if w_kind(w) != CelKind::Str {
        return None;
    }
    let leaf = &*w.cast::<W_StringObject>();
    let n = leaf.byte_len as usize;
    let base = crate::runtime::object_array::bytes_base(leaf.chars);
    if base.is_null() {
        return Some("");
    }
    std::str::from_utf8(std::slice::from_raw_parts(base, n)).ok()
}

/// Live length of a bytes leaf.
///
/// # Safety
///
/// `w` is a live [`W_BytesObject`].
pub unsafe fn bytes_len(w: CelRef) -> i64 {
    (*w.cast::<W_BytesObject>()).length
}

/// Item `index` of a list leaf, or null if out of range.
///
/// # Safety
///
/// `w` is a live [`W_ListObject`].
pub unsafe fn list_get(w: CelRef, index: i64) -> Option<CelRef> {
    if index < 0 {
        return None;
    }
    let leaf = &*w.cast::<W_ListObject>();
    if index >= leaf.length {
        return None;
    }
    let at = leaf.start + index;
    match leaf.strategy {
        ListStrategy::Object => {
            let base = crate::runtime::object_array::items_block_items_base(leaf.items);
            if base.is_null() {
                return None;
            }
            Some(*base.add(at as usize))
        }
        ListStrategy::Ints => {
            if leaf.storage.is_null() {
                return None;
            }
            let col = &*leaf.storage.cast::<W_IntColumn>();
            if col.data.is_null() || at < 0 || at >= col.length {
                return None;
            }
            Some(new_int(*col.data.add(at as usize)) as CelRef)
        }
        ListStrategy::Window => None,
    }
}

/// The unboxed integer at `index` of an Ints-strategy list, with no allocation.
///
/// `None` if `w` is not an Ints list or `index` is out of range.
///
/// # Safety
///
/// `w` is a live [`W_ListObject`].
pub unsafe fn list_int_at(w: CelRef, index: i64) -> Option<i64> {
    if index < 0 {
        return None;
    }
    let leaf = &*w.cast::<W_ListObject>();
    if leaf.strategy != ListStrategy::Ints || index >= leaf.length {
        return None;
    }
    if leaf.storage.is_null() {
        return None;
    }
    let col = &*leaf.storage.cast::<W_IntColumn>();
    let at = leaf.start + index;
    if col.data.is_null() || at < 0 || at >= col.length {
        return None;
    }
    Some(*col.data.add(at as usize))
}

/// The int column of an Ints-strategy list, or `None` if `w` is not one.
///
/// # Safety
///
/// `w` is a live value.
pub unsafe fn list_ints_slice<'a>(w: CelRef) -> Option<&'a [i64]> {
    if w_kind(w) != CelKind::List {
        return None;
    }
    let leaf = &*w.cast::<W_ListObject>();
    if leaf.strategy != ListStrategy::Ints || leaf.storage.is_null() {
        return None;
    }
    let col = &*leaf.storage.cast::<W_IntColumn>();
    let start = leaf.start as usize;
    let n = leaf.length as usize;
    if col.data.is_null() || start.saturating_add(n) > col.length as usize {
        return None;
    }
    Some(std::slice::from_raw_parts(col.data.add(start), n))
}

/// Equality of two interned lists when at least one is an Ints column.
///
/// `None` if neither side is Ints; the caller then uses the generic path.
///
/// # Safety
///
/// Both operands are live values.
pub unsafe fn interned_list_eq(a: CelRef, b: CelRef) -> Option<bool> {
    match (list_ints_slice(a), list_ints_slice(b)) {
        (Some(la), Some(lb)) => Some(la == lb),
        (Some(ints), None) if w_kind(b) == CelKind::List => Some(object_list_eq_ints(b, ints)),
        (None, Some(ints)) if w_kind(a) == CelKind::List => Some(object_list_eq_ints(a, ints)),
        _ => None,
    }
}

unsafe fn object_list_eq_ints(w: CelRef, ints: &[i64]) -> bool {
    if list_len(w) as usize != ints.len() {
        return false;
    }
    let mut i = 0i64;
    while i < ints.len() as i64 {
        let Some(item) = list_get(w, i) else {
            return false;
        };
        if w_kind(item) != CelKind::Int {
            return false;
        }
        if (*item.cast::<W_IntObject>()).intval != ints[i as usize] {
            return false;
        }
        i += 1;
    }
    true
}

/// An empty list whose items block has room for `cap` appends.
pub fn new_list_with_capacity(cap: i64) -> *mut W_ListObject {
    super::heap::with_heap(|h| new_list_with_capacity_in(h, cap))
}

/// [`new_list_with_capacity`] on `heap`. Slots past `length` are written
/// by append before anything reads them, so they are not zeroed.
#[inline(never)]
pub fn new_list_with_capacity_in(
    heap: &super::heap::CelHeap,
    cap: i64,
) -> *mut W_ListObject {
    let n = cap.max(0) as usize;
    let items = object_array::new_items_block_with_zeroed_prefix_in(heap, n, 0);
    heap.alloc(W_ListObject {
        ob_header: CelObject {
            ob_type: &CEL_LIST_CLASS,
        },
        strategy: ListStrategy::Object,
        storage: core::ptr::null_mut(),
        items,
        start: 0,
        length: 0,
        public: core::ptr::null(),
        public_start: 0,
        public_len: 0,
    })
}

/// Append `item` to an object-strategy list if the block still has room.
///
/// Used while a comprehension fills a list opened by [`new_list_with_capacity`].
///
/// # Safety
///
/// `w` is a live [`W_ListObject`].
pub unsafe fn list_try_append(w: CelRef, item: CelRef) -> bool {
    if w_kind(w) != CelKind::List {
        return false;
    }
    let leaf = &mut *w.cast::<W_ListObject>();
    if leaf.strategy != ListStrategy::Object {
        return false;
    }
    let cap = crate::runtime::object_array::items_capacity(leaf.items);
    if leaf.length < 0 || (leaf.length as usize) >= cap {
        return false;
    }
    let base = crate::runtime::object_array::items_block_items_base(leaf.items);
    if base.is_null() {
        return false;
    }
    *base.add(leaf.length as usize) = item;
    leaf.length += 1;
    true
}

pub fn new_list(values: &[CelRef]) -> *mut W_ListObject {
    let items = object_array::new_items_block(values);
    let length = values.len() as i64;
    lltype::malloc_typed(W_ListObject {
        ob_header: CelObject {
            ob_type: &CEL_LIST_CLASS,
        },
        strategy: ListStrategy::Object,
        storage: core::ptr::null_mut(),
        items,
        start: 0,
        length,
        public: core::ptr::null(),
        public_start: 0,
        public_len: 0,
    })
}

/// Box unboxed integers as a CEL list.
pub fn new_list_ints(values: &[i64]) -> *mut W_ListObject {
    let storage = new_int_column(values) as CelRef;
    let length = values.len() as i64;
    lltype::malloc_typed(W_ListObject {
        ob_header: CelObject {
            ob_type: &CEL_LIST_CLASS,
        },
        strategy: ListStrategy::Ints,
        storage,
        items: core::ptr::null_mut(),
        start: 0,
        length,
        public: core::ptr::null(),
        public_start: 0,
        public_len: 0,
    })
}

/// Box a host-table window as a CEL list.
pub fn new_list_window(storage: CelRef, start: i64, length: i64) -> *mut W_ListObject {
    lltype::malloc_typed(W_ListObject {
        ob_header: CelObject {
            ob_type: &CEL_LIST_CLASS,
        },
        strategy: ListStrategy::Window,
        storage,
        items: core::ptr::null_mut(),
        start,
        length,
        public: core::ptr::null(),
        public_start: 0,
        public_len: 0,
    })
}

/// Flatten `[k, v]` pairs into one items block. Shared by map and struct so
/// the two leaves cannot drift on how they pack an entry.
fn interleaved_pair_block(pairs: &[(CelRef, CelRef)]) -> *mut CelItemsBlock {
    let n = pairs.len();
    let mut items = Vec::with_capacity(n * 2);
    let mut i = 0;
    while i < n {
        items.push(pairs[i].0);
        items.push(pairs[i].1);
        i += 1;
    }
    object_array::new_items_block(&items)
}

/// How a [`W_MapObject`] holds its entries. D4, same as [`ListStrategy`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapStrategy {
    Object = 0,
    /// One record row. The schema lives in the heap host table at `storage`.
    Record = 1,
}

/// A CEL `map`.
///
/// Entries of an object map live as interleaved `[k0, v0, …]` references.
/// A record row parks its schema in the host table so intern does not
/// explode the window.
#[cfg_attr(
    feature = "jit",
    majit_macros::jit_immutable_fields(strategy, storage, items, length)
)]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_MapObject {
    pub ob_header: CelObject,
    pub strategy: MapStrategy,
    pub storage: CelRef,
    pub items: *mut CelItemsBlock,
    /// Live entry count, not the number of references in [`Self::items`].
    pub length: i64,
    /// Non-owning pointer at the public object-map table this leaf was
    /// wrapped from, or null if the map was allocated by the VM.
    pub public: *const (),
}

pub static CEL_MAP_CLASS: CelClass = CelClass::new("map", CelKind::Map);

const _: () = {
    assert!(offset_of!(W_MapObject, ob_header) == 0);
};

/// Look up a string key on a map leaf.
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
pub unsafe fn map_lookup_string(w: CelRef, field: &str) -> Option<CelRef> {
    lookup_string_pairs(
        (*w.cast::<W_MapObject>()).items,
        (*w.cast::<W_MapObject>()).length,
        field,
    )
}

unsafe fn lookup_string_pairs(
    items: *mut crate::runtime::object_array::CelItemsBlock,
    length: i64,
    field: &str,
) -> Option<CelRef> {
    let n = length as usize;
    let base = crate::runtime::object_array::items_block_items_base(items);
    if base.is_null() {
        return None;
    }
    let mut i = 0;
    while i < n {
        let key = *base.add(2 * i);
        if string_eq_str(key, field) {
            return Some(*base.add(2 * i + 1));
        }
        i += 1;
    }
    None
}

unsafe fn string_eq_str(w: CelRef, field: &str) -> bool {
    if w.is_null() || w_kind(w) != CelKind::Str {
        return false;
    }
    let leaf = &*w.cast::<W_StringObject>();
    let n = leaf.byte_len as usize;
    let base = crate::runtime::object_array::bytes_base(leaf.chars);
    if base.is_null() {
        return field.is_empty();
    }
    std::slice::from_raw_parts(base, n) == field.as_bytes()
}

/// An empty map whose items block has room for `cap` entries.
pub fn new_map_with_capacity(cap: i64) -> *mut W_MapObject {
    super::heap::with_heap(|h| new_map_with_capacity_in(h, cap))
}

/// An empty map on `heap` with room for `cap` entries.
#[inline]
pub fn new_map_with_capacity_in(heap: &super::heap::CelHeap, cap: i64) -> *mut W_MapObject {
    let n = cap.max(0) as usize;
    let items = crate::runtime::object_array::new_items_block_zeroed_in(heap, n.saturating_mul(2));
    heap.alloc(W_MapObject {
        ob_header: CelObject {
            ob_type: &CEL_MAP_CLASS,
        },
        strategy: MapStrategy::Object,
        storage: core::ptr::null_mut(),
        items,
        length: 0,
        public: core::ptr::null(),
    })
}

/// Append `(key, value)` to an object-strategy map if the block still has room.
///
/// # Safety
///
/// `w` is a live [`W_MapObject`].
#[inline]
pub unsafe fn map_try_insert(w: CelRef, key: CelRef, value: CelRef) -> bool {
    if w_kind(w) != CelKind::Map {
        return false;
    }
    let leaf = &mut *w.cast::<W_MapObject>();
    if leaf.strategy != MapStrategy::Object {
        return false;
    }
    let cap = crate::runtime::object_array::items_capacity(leaf.items);
    if leaf.length < 0 {
        return false;
    }
    let used = (leaf.length as usize).saturating_mul(2);
    if used.saturating_add(2) > cap {
        return false;
    }
    let base = crate::runtime::object_array::items_block_items_base(leaf.items);
    if base.is_null() {
        return false;
    }
    *base.add(used) = key;
    *base.add(used + 1) = value;
    leaf.length += 1;
    true
}

/// Box `pairs` as a CEL `map`.
pub fn new_map(pairs: &[(CelRef, CelRef)]) -> *mut W_MapObject {
    let items = interleaved_pair_block(pairs);
    let length = pairs.len() as i64;
    lltype::malloc_typed(W_MapObject {
        ob_header: CelObject {
            ob_type: &CEL_MAP_CLASS,
        },
        strategy: MapStrategy::Object,
        storage: core::ptr::null_mut(),
        items,
        length,
        public: core::ptr::null(),
    })
}

/// Box a host-table record row as a CEL map.
pub fn new_map_record(storage: CelRef, length: i64) -> *mut W_MapObject {
    lltype::malloc_typed(W_MapObject {
        ob_header: CelObject {
            ob_type: &CEL_MAP_CLASS,
        },
        strategy: MapStrategy::Record,
        storage,
        items: core::ptr::null_mut(),
        length,
        public: core::ptr::null(),
    })
}

/// A CEL `struct`.
///
/// §3 of the design gives this leaf a `w_type: CelRef` pointing at a type
/// value. That type value is not a leaf yet: [`W_TypeObject`] only holds a
/// `*const CelClass` and cannot carry a per-instance struct name.
/// [`CEL_STRUCT_CLASS`] is the class word; the instance name is an ordinary
/// payload so convert can rebuild `CelStruct::new(name)`.
///
/// Fields live as interleaved `[name0, value0, …]` references, names as
/// [`W_StringObject`]. [`W_StructObject::length`] is the field count.
///
/// ⚠ The strategy tag is absent for the same reason [`W_ListObject`] omits
/// it: a discriminant is only meaningful once there is a second strategy.
#[cfg(feature = "structs")]
#[cfg_attr(
    feature = "jit",
    majit_macros::jit_immutable_fields(name, fields, length)
)]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_StructObject {
    pub ob_header: CelObject,
    pub name: *mut W_StringObject,
    pub fields: *mut CelItemsBlock,
    /// Live field count, not the number of references in [`Self::fields`].
    pub length: i64,
}

#[cfg(feature = "structs")]
pub static CEL_STRUCT_CLASS: CelClass = CelClass::new("struct", CelKind::Struct);

#[cfg(feature = "structs")]
const _: () = {
    assert!(offset_of!(W_StructObject, ob_header) == 0);
};

/// Look up a field on a struct leaf.
///
/// # Safety
///
/// `w` is a live [`W_StructObject`].
#[cfg(feature = "structs")]
pub unsafe fn struct_lookup_field(w: CelRef, field: &str) -> Option<CelRef> {
    lookup_string_pairs(
        (*w.cast::<W_StructObject>()).fields,
        (*w.cast::<W_StructObject>()).length,
        field,
    )
}

/// Box a named struct with `fields` as name/value pairs.
#[cfg(feature = "structs")]
pub fn new_struct(name: *mut W_StringObject, fields: &[(CelRef, CelRef)]) -> *mut W_StructObject {
    let items = interleaved_pair_block(fields);
    let length = fields.len() as i64;
    lltype::malloc_typed(W_StructObject {
        ob_header: CelObject {
            ob_type: &CEL_STRUCT_CLASS,
        },
        name,
        fields: items,
        length,
    })
}

/// A CEL type value — what `type(x)` evaluates to.
///
/// `cls` is the class of the type this value *denotes*, while the header's own
/// `ob_type` is [`CEL_TYPE_CLASS`]. The two being different is what makes
/// `type(type(1)) == type(string)` hold, and it is why the two spellings must
/// not be collapsed: `(*type_value).cls` is a class, `(*any_value).ob_type` is
/// a class, and only the type value itself is an allocated object.
#[cfg_attr(feature = "jit", majit_macros::jit_immutable_fields(cls))]
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

/// The activation record, `pyframe.py` `PyFrame` virtualizable subset.
///
/// `_virtualizable_ = ['last_instr', 'valuestackdepth', 'locals_stack_w[*]']`
/// (`interp_jit.py`). The JIT driver names this object `virtualizables =
/// ['frame']`. Slots are a fixed [`CelItemsBlock`] — `make_sure_not_resized`.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_CelFrame {
    pub ob_header: CelObject,
    /// `virtualizable.py` token. 0 = not virtualized.
    pub vable_token: usize,
    pub last_instr: i64,
    pub valuestackdepth: i64,
    /// Indexed as `frame.locals_stack_w[i]`, the shape `jtransform`
    /// `getarrayitem_vable_*` matches.
    pub locals_stack_w: VableStack,
    pub n_slots: i64,
}

/// The `locals_cells_stack_w[*]` array: a pointer that indexes as a slice.
///
/// `#[repr(transparent)]` so the field is still one pointer in the frame.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct VableStack {
    pub(crate) block: *mut crate::runtime::object_array::CelItemsBlock,
}

impl VableStack {
    fn from_block(block: *mut crate::runtime::object_array::CelItemsBlock) -> Self {
        VableStack { block }
    }

    pub fn capacity(self) -> usize {
        unsafe { crate::runtime::object_array::items_capacity(self.block) }
    }
}

impl core::ops::Index<i64> for VableStack {
    type Output = CelRef;
    fn index(&self, i: i64) -> &CelRef {
        unsafe {
            &*crate::runtime::object_array::items_block_items_base(self.block).add(i as usize)
        }
    }
}

impl core::ops::IndexMut<i64> for VableStack {
    fn index_mut(&mut self, i: i64) -> &mut CelRef {
        unsafe {
            &mut *crate::runtime::object_array::items_block_items_base(self.block).add(i as usize)
        }
    }
}

pub static CEL_FRAME_CLASS: CelClass = CelClass::new("frame", CelKind::Frame);

const _: () = {
    assert!(offset_of!(W_CelFrame, ob_header) == 0);
};

pub const CELFRAME_VABLE_TOKEN_OFFSET: usize = offset_of!(W_CelFrame, vable_token);
pub const CELFRAME_LAST_INSTR_OFFSET: usize = offset_of!(W_CelFrame, last_instr);
pub const CELFRAME_VALUESTACKDEPTH_OFFSET: usize = offset_of!(W_CelFrame, valuestackdepth);
pub const CELFRAME_LOCALS_STACK_OFFSET: usize = offset_of!(W_CelFrame, locals_stack_w);

/// Allocate a frame whose array is `n_slots + max_stack` and never resized.
pub fn new_cel_frame(n_slots: i64, max_stack: i64) -> *mut W_CelFrame {
    crate::runtime::heap::with_heap(|h| new_cel_frame_in(h, n_slots, max_stack))
}

/// Allocate a frame on `heap`.
///
/// The block is `n_slots + max_stack` cells. Locals (`0..n_slots`) start
/// null so a residual hydrate can scan them; stack cells are written
/// before they are read and are left uninitialised.
#[inline(always)]
pub fn new_cel_frame_in(
    heap: &crate::runtime::heap::CelHeap,
    n_slots: i64,
    max_stack: i64,
) -> *mut W_CelFrame {
    let n_slots_us = n_slots.max(0) as usize;
    let cap = n_slots_us.saturating_add(max_stack.max(0) as usize);
    let items = object_array::new_items_block_with_zeroed_prefix_in(heap, cap, n_slots_us);
    heap.alloc(W_CelFrame {
        ob_header: CelObject {
            ob_type: &CEL_FRAME_CLASS,
        },
        vable_token: 0,
        last_instr: -1,
        valuestackdepth: n_slots,
        locals_stack_w: VableStack::from_block(items),
        n_slots,
    })
}

/// Slot `i` of a live frame.
///
/// # Safety
///
/// `frame` is a live [`W_CelFrame`] and `i` is in range.
pub unsafe fn cel_frame_slot(frame: *mut W_CelFrame, i: i64) -> *mut CelRef {
    crate::runtime::object_array::items_block_items_base((*frame).locals_stack_w.block)
        .add(i as usize)
}

/// `virtualizable.py` `force_virtualizable_if_necessary`.
///
/// The token is 0 until a compiled loop owns the frame, so this is a
/// no-op in the interpreter. `jtransform` rewrites a residual force
/// into a call when the token is live.
///
/// # Safety
///
/// `frame` is a live [`W_CelFrame`].
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
pub unsafe fn force_virtualizable_if_necessary(frame: *mut W_CelFrame) {
    if (*frame).vable_token != 0 {
        // Residual: the compiled loop owns the boxes. The interpreter
        // never sets the token, so this arm is not taken here.
    }
}

/// The type value denoting `cls`.
pub fn new_type(cls: &'static CelClass) -> *mut W_TypeObject {
    lltype::malloc_typed(W_TypeObject {
        ob_header: CelObject {
            ob_type: &CEL_TYPE_CLASS,
        },
        cls,
    })
}

/// A foreign host object.
///
/// `w_type` is a type value ([`W_TypeObject`]); `host_index` is a slot in
/// this thread's heap table. D12: the host lives as long as the heap — no
/// finalizer, no `Drop` on the leaf.
#[cfg_attr(
    feature = "jit",
    majit_macros::jit_immutable_fields(w_type, host_index)
)]
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct W_OpaqueObject {
    pub ob_header: CelObject,
    pub w_type: CelRef,
    pub host_index: i64,
}

/// The class of [`W_OpaqueObject`].
pub static CEL_OPAQUE_CLASS: CelClass = CelClass::new("opaque", CelKind::Opaque);

const _: () = {
    assert!(offset_of!(W_OpaqueObject, ob_header) == 0);
};

/// Box a host-table slot as a CEL opaque.
pub fn new_opaque(w_type: CelRef, host_index: i64) -> *mut W_OpaqueObject {
    lltype::malloc_typed(W_OpaqueObject {
        ob_header: CelObject {
            ob_type: &CEL_OPAQUE_CLASS,
        },
        w_type,
        host_index,
    })
}

/// The host-table index of an opaque leaf.
///
/// # Safety
///
/// `w` is a live [`W_OpaqueObject`].
pub unsafe fn opaque_host_index(w: CelRef) -> i64 {
    (*w.cast::<W_OpaqueObject>()).host_index
}

/// Box `value` as a CEL `bool`.
///
/// Returns one of the two immortal singletons. [`new_bool_raw`] is the
/// fuse-shaped allocation; the interned path does not go through it.
pub fn new_bool(value: bool) -> *mut W_BoolObject {
    let slot = if value { &TRUE } else { &FALSE };
    *slot.get_or_init(|| {
        lltype::malloc_typed_immortal(W_BoolObject {
            ob_header: CelObject {
                ob_type: &CEL_BOOL_CLASS,
            },
            boolval: i64::from(value),
        }) as usize
    }) as *mut W_BoolObject
}

/// Inclusive lower bound of the interned `int` table.
///
/// `intobject.py` `PREBUILTINTFROM`. Values outside the table still go
/// through [`new_int_raw`] so the boxing fuse can see a `malloc_typed`.
pub const PREBUILT_INT_FROM: i64 = -5;

/// Exclusive upper bound of the interned `int` table (`PREBUILTINTTO`).
pub const PREBUILT_INT_TO: i64 = 257;

static TRUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static FALSE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static NULL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
static SMALL_INTS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();

fn small_ints() -> &'static [usize] {
    SMALL_INTS.get_or_init(|| {
        (PREBUILT_INT_FROM..PREBUILT_INT_TO)
            .map(|v| {
                lltype::malloc_typed_immortal(W_IntObject {
                    ob_header: CelObject {
                        ob_type: &CEL_INT_CLASS,
                    },
                    intval: v,
                }) as usize
            })
            .collect()
    })
}

/// Box `value` as a CEL `int`.
///
/// Values in [`PREBUILT_INT_FROM`]..[`PREBUILT_INT_TO`] are immortal
/// singletons. Everything else is a fresh [`new_int_raw`].
#[inline]
pub fn new_int(value: i64) -> *mut W_IntObject {
    if (PREBUILT_INT_FROM..PREBUILT_INT_TO).contains(&value) {
        let idx = (value - PREBUILT_INT_FROM) as usize;
        return small_ints()[idx] as *mut W_IntObject;
    }
    new_int_raw(value)
}

/// Box `value` as a CEL `int` on `heap`.
///
/// The interned range is the same singletons [`new_int`] returns. A miss
/// writes a young leaf through `heap` and does not re-resolve thread-local
/// storage.
#[inline]
pub fn new_int_in(heap: &super::heap::CelHeap, value: i64) -> *mut W_IntObject {
    if (PREBUILT_INT_FROM..PREBUILT_INT_TO).contains(&value) {
        let idx = (value - PREBUILT_INT_FROM) as usize;
        return small_ints()[idx] as *mut W_IntObject;
    }
    heap.alloc(W_IntObject {
        ob_header: CelObject {
            ob_type: &CEL_INT_CLASS,
        },
        intval: value,
    })
}

/// Box `value` as a CEL `uint` on `heap`.
#[inline]
pub fn new_uint_in(heap: &super::heap::CelHeap, value: u64) -> *mut W_UIntObject {
    heap.alloc(W_UIntObject {
        ob_header: CelObject {
            ob_type: &CEL_UINT_CLASS,
        },
        uintval: value,
    })
}

/// Box `value` as a CEL `double` on `heap`.
#[inline]
pub fn new_double_in(heap: &super::heap::CelHeap, value: f64) -> *mut W_DoubleObject {
    heap.alloc(W_DoubleObject {
        ob_header: CelObject {
            ob_type: &CEL_DOUBLE_CLASS,
        },
        floatval: value,
    })
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
        let m = new_map(&[]);
        assert_eq!(m as usize, m as CelRef as usize);
        #[cfg(feature = "structs")]
        {
            let s = new_struct(new_string("T"), &[]);
            assert_eq!(s as usize, s as CelRef as usize);
        }
    }

    /// Condition 2's cel-side half: the one header store a constructor makes
    /// names its own class. The fuse keeps a single vtable address for the
    /// whole cluster, so a constructor stamping some other class would make
    /// that address wrong for the values it mints. This does not prove the
    /// fuse fires — that needs the lowering — but a constructor that broke
    /// the invariant would fail here rather than showing up as an unexplained
    /// zero in a census.
    ///
    /// The header declares nothing else to check. That is the point of the
    /// one-word shape, and `header_declares_no_class_word` is the majit-side
    /// half that reads the absence as a base-type instance.
    #[test]
    fn every_constructor_stamps_its_own_class() {
        fn check(w: CelRef, expected: *const CelClass) {
            unsafe {
                assert_eq!((*w).ob_type, expected);
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
        check(
            new_optional(new_int(1) as CelRef) as CelRef,
            &CEL_OPTIONAL_CLASS,
        );
        check(new_optional_none() as CelRef, &CEL_OPTIONAL_CLASS);
        check(new_bytes(b"x") as CelRef, &CEL_BYTES_CLASS);
        check(new_string("x") as CelRef, &CEL_STRING_CLASS);
        check(new_list(&[]) as CelRef, &CEL_LIST_CLASS);
        check(new_cel_frame(0, 0) as CelRef, &CEL_FRAME_CLASS);
        check(new_map(&[]) as CelRef, &CEL_MAP_CLASS);
        check(
            new_opaque(new_type(&CEL_OPAQUE_CLASS) as CelRef, 0) as CelRef,
            &CEL_OPAQUE_CLASS,
        );
        #[cfg(feature = "structs")]
        check(
            new_struct(new_string("T"), &[]) as CelRef,
            &CEL_STRUCT_CLASS,
        );
    }

    /// The variable-length leaves keep their live length on the leaf and their
    /// payload in the block, and the two must agree at construction. A block's
    /// own word is a CAPACITY, so the pair is the only place the distinction is
    /// observable while every block is still exact-sized.
    #[test]
    fn a_variable_length_leaf_and_its_block_agree_on_length() {
        use crate::runtime::object_array::{
            bytes_base, bytes_capacity, items_block_items_base, items_capacity,
        };
        unsafe {
            let b = new_bytes(b"hello");
            assert_eq!((*b).length, 5);
            assert_eq!(bytes_capacity((*b).data), 5);
            assert_eq!(
                core::slice::from_raw_parts(bytes_base((*b).data), 5),
                b"hello"
            );

            let s = new_string("hi");
            assert_eq!((*s).byte_len, 2);
            assert_eq!(bytes_capacity((*s).chars), 2);

            let elems: Vec<CelRef> = (0..3).map(|i| new_int(i) as CelRef).collect();
            let l = new_list(&elems);
            assert_eq!((*l).length, 3);
            assert_eq!(items_capacity((*l).items), 3);
            for (i, e) in elems.iter().enumerate() {
                assert_eq!(*items_block_items_base((*l).items).add(i), *e);
            }

            let pairs = [
                (new_int(1) as CelRef, new_string("a") as CelRef),
                (new_string("b") as CelRef, new_bool(true) as CelRef),
            ];
            let m = new_map(&pairs);
            assert_eq!((*m).length, 2);
            assert_eq!(items_capacity((*m).items), 4);
            let map_base = items_block_items_base((*m).items);
            for (i, (k, v)) in pairs.iter().enumerate() {
                assert_eq!(*map_base.add(2 * i), *k);
                assert_eq!(*map_base.add(2 * i + 1), *v);
            }

            let empty = new_map(&[]);
            assert_eq!((*empty).length, 0);
            assert_eq!(items_capacity((*empty).items), 0);

            #[cfg(feature = "structs")]
            {
                let fields = [(new_string("x") as CelRef, new_int(1) as CelRef)];
                let s = new_struct(new_string("T"), &fields);
                assert_eq!((*s).length, 1);
                assert_eq!(items_capacity((*s).fields), 2);
                let field_base = items_block_items_base((*s).fields);
                assert_eq!(*field_base.add(0), fields[0].0);
                assert_eq!(*field_base.add(1), fields[0].1);
            }
        }
    }

    /// The virtualizable subset matches `interp_jit.py` `_virtualizable_`:
    /// `last_instr`, `valuestackdepth`, `locals_stack_w[*]`, and a token.
    #[test]
    fn a_cel_frame_is_the_virtualizable_activation_record() {
        unsafe {
            let f = new_cel_frame(2, 3);
            assert_eq!((*f).ob_header.ob_type, &CEL_FRAME_CLASS as *const CelClass);
            assert_eq!(w_kind(f as CelRef), CelKind::Frame);
            assert_eq!((*f).vable_token, 0);
            assert_eq!((*f).last_instr, -1);
            assert_eq!((*f).valuestackdepth, 2);
            assert_eq!((*f).n_slots, 2);
            assert_eq!((*f).locals_stack_w.capacity(), 5);
            let a = new_int(1) as CelRef;
            *cel_frame_slot(f, 0) = a;
            assert_eq!(*cel_frame_slot(f, 0), a);
            force_virtualizable_if_necessary(f);
            assert_eq!((*f).vable_token, 0);
            assert_eq!(
                CELFRAME_VABLE_TOKEN_OFFSET,
                offset_of!(W_CelFrame, vable_token)
            );
            assert_eq!(
                CELFRAME_LAST_INSTR_OFFSET,
                offset_of!(W_CelFrame, last_instr)
            );
            assert_eq!(
                CELFRAME_VALUESTACKDEPTH_OFFSET,
                offset_of!(W_CelFrame, valuestackdepth)
            );
            assert_eq!(
                CELFRAME_LOCALS_STACK_OFFSET,
                offset_of!(W_CelFrame, locals_stack_w)
            );
        }
    }

    /// A `string` is not a `bytes` with a different header: the two are
    /// distinct CEL types, and nothing may rely on the class word alone to tell
    /// them apart at a use site that has already narrowed.
    #[test]
    fn string_and_bytes_are_separate_classes() {
        unsafe {
            let s = new_string("x") as CelRef;
            let b = new_bytes(b"x") as CelRef;
            assert_ne!((*s).ob_type, (*b).ob_type);
            assert_eq!(w_kind(s), CelKind::Str);
            assert_eq!(w_kind(b), CelKind::Bytes);
        }
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

    /// TRUE, FALSE, NULL and the small-int table are one address each, and
    /// that address came from the immortal allocator — not a Rust `static`
    /// whose preceding word is rodata.
    #[test]
    fn prebuilts_are_immortal_singletons() {
        assert_eq!(new_bool(true), new_bool(true));
        assert_eq!(new_bool(false), new_bool(false));
        assert_ne!(new_bool(true), new_bool(false));
        assert_eq!(new_null(), new_null());
        assert_eq!(new_int(0), new_int(0));
        assert_eq!(new_int(-5), new_int(-5));
        assert_eq!(new_int(256), new_int(256));
        assert_ne!(new_int(0), new_int(1));

        let outsides = [new_int(-6), new_int(257), new_int(1000)];
        for w in outsides {
            let value = unsafe { (*w).intval };
            assert_ne!(w, new_int(value), "outside the table is not interned");
            assert!(
                !lltype::is_immortal(w as *const u8),
                "a fresh int must not be registered as immortal"
            );
        }

        let prebuilts: [*const u8; 6] = [
            new_bool(true) as *const u8,
            new_bool(false) as *const u8,
            new_null() as *const u8,
            new_int(-5) as *const u8,
            new_int(0) as *const u8,
            new_int(256) as *const u8,
        ];
        for p in prebuilts {
            assert!(
                lltype::is_immortal(p),
                "prebuilt {p:?} must come from malloc_typed_immortal"
            );
            unsafe {
                assert_ne!(
                    crate::runtime::heap::immortal_header(p),
                    0,
                    "header-relative load at obj-8 must be a real word"
                );
            }
        }
    }

    /// Interned ints do not increment this thread's heap counter.
    #[test]
    fn prebuilts_do_not_count_against_the_thread_heap() {
        let before = crate::runtime::heap::with_heap(|h| h.allocated_objects());
        let _ = new_bool(true);
        let _ = new_bool(false);
        let _ = new_null();
        let _ = new_int(1);
        let _ = new_int(2);
        let after = crate::runtime::heap::with_heap(|h| h.allocated_objects());
        assert_eq!(before, after);
        let _ = new_int(1000);
        let later = crate::runtime::heap::with_heap(|h| h.allocated_objects());
        assert_eq!(later, after + 1);
    }

    /// `new_int_in` writes a miss on the heap it was handed, not a second one.
    #[test]
    fn new_int_in_counts_against_the_given_heap() {
        let heap = crate::runtime::heap::CelHeap::new();
        let before = heap.allocated_objects();
        let w = new_int_in(&heap, 1000);
        assert_eq!(heap.allocated_objects(), before + 1);
        unsafe {
            assert_eq!((*w).intval, 1000);
        }
        let interned = new_int_in(&heap, 1);
        assert_eq!(heap.allocated_objects(), before + 1);
        assert_eq!(interned, new_int(1));
    }
}
