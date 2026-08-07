//! The VM's error channel.
//!
//! A [`Result`], not a sentinel return plus out-of-band state. The sentinel
//! form is what a translator emits for a language without sum types; it is
//! not the design, and porting it here would port a workaround for a gap Rust
//! does not have. The `?` operator *is* the post-call check that lowering
//! inserts by hand.
//!
//! What the error channel does have to avoid is a wide payload.
//! [`ExecutionError`] carries `String`, `Arc<String>` and [`Value`] fields, so
//! constructing one allocates -- on a path that CEL exercises for ordinary
//! control flow, because `&&` and `||` absorb errors rather than propagating
//! them. [`CelErr`] is therefore [`Copy`] and pointer-width-ish, and the
//! detail is reconstructed on the cold path.
//!
//! [`ExecutionError`]: crate::ExecutionError
//! [`Value`]: crate::Value

use super::opcode::OpCode;

/// An index into [`CelCode::names`].
///
/// [`CelCode::names`]: super::CelCode::names
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct NameId(pub u32);

/// An index into the execution's cold error table.
///
/// The table holds a full [`ExecutionError`] for the cases whose detail is
/// not recoverable from the program text: a host function's message, and any
/// error naming a value computed at run time. It is written only when an
/// error is actually raised, so the hot path never touches it.
///
/// [`ExecutionError`]: crate::ExecutionError
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ColdId(pub u32);

/// An evaluation error.
///
/// The variant set and the payload of each variant are derived from what an
/// error's *observable* form actually contains, not from [`ExecutionError`]'s
/// shape: the differential oracle renders errors to a string, and that
/// rendering discards every [`Value`] payload. What survives is the
/// discriminant plus, for a minority of variants, one name, one operator, or
/// two small counts -- all of which are either already in the program's name
/// table or recoverable from the opcode that raised them.
///
/// [`ExecutionError`]: crate::ExecutionError
/// [`Value`]: crate::Value
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CelErr {
    // -- no detail beyond the discriminant --------------------------------
    NoSuchOverload,
    MissingArgumentOrTarget,
    DivisionByZero,
    RemainderByZero,
    IndexOutOfBounds,
    ValuesNotComparable,
    UnsupportedKeyType,
    UnsupportedTargetType,
    UnsupportedIndex,
    InternalError,

    // -- named by the program text ----------------------------------------
    //
    // Every name here is a compile-time constant of the expression, so it is
    // already in the name table and costs an index rather than an `Arc<String>`.
    /// A field or key that the program names literally.
    ///
    /// A *computed* key -- `m[k]` where `k` is a variable -- cannot use this
    /// variant, because the key is not in the name table. That case takes
    /// [`CelErr::Cold`], and the split is exactly the compile-time split
    /// between `GetField` and `Index`.
    NoSuchKey(NameId),
    UndeclaredReference(NameId),
    NotSupportedAsMethod(NameId),
    UnexpectedType {
        got: NameId,
        want: NameId,
    },

    // -- recoverable from the raising instruction --------------------------
    //
    // The operator name in these is `"add"`, `"sub"`, `"negate"` and so on,
    // which is a function of the opcode. Carrying the opcode costs one byte
    // where carrying the string would cost a fat pointer.
    Overflow(OpCode),
    UnsupportedUnaryOperator(OpCode),
    UnsupportedBinaryOperator(OpCode),
    InvalidArgumentCount {
        expected: u16,
        actual: u16,
    },

    /// An error whose detail lives in the execution's cold table.
    ///
    /// Host-function failures land here: their message is supplied by the
    /// host at run time and is part of the public error's `Display`, so it
    /// cannot be reconstructed from the program.
    Cold(ColdId),
}

/// The VM's result type.
pub type CelResult<T> = Result<T, CelErr>;

impl CelErr {
    /// The short operator name an error names, for the operator-carrying
    /// variants.
    ///
    /// Exhaustive over [`OpCode`] at the call site rather than here, so this
    /// stays a lookup and a new operator opcode is a compile error there.
    pub const fn operator(self) -> Option<OpCode> {
        match self {
            CelErr::Overflow(op)
            | CelErr::UnsupportedUnaryOperator(op)
            | CelErr::UnsupportedBinaryOperator(op) => Some(op),
            _ => None,
        }
    }
}

/// The error type must stay small and trivially copyable.
///
/// `&&` and `||` absorb a left-hand error and return success, so an error is
/// ordinary control flow in CEL rather than an exceptional path. A payload
/// that allocated would put a heap operation on that path; a payload that was
/// not `Copy` would make the absorb a move out of a `match`.
const _: () = {
    assert!(core::mem::size_of::<CelErr>() <= 16);
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the two properties the design depends on, so a later variant with
    /// a `String` or a `Box` fails here rather than in a benchmark.
    #[test]
    fn cel_err_is_copy_and_small() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<CelErr>();
        assert!(std::mem::size_of::<CelErr>() <= 16);
    }

    #[test]
    fn operator_is_reported_only_by_the_operator_carrying_variants() {
        assert_eq!(CelErr::Overflow(OpCode::Add).operator(), Some(OpCode::Add));
        assert_eq!(
            CelErr::UnsupportedBinaryOperator(OpCode::Sub).operator(),
            Some(OpCode::Sub)
        );
        assert_eq!(CelErr::NoSuchOverload.operator(), None);
        assert_eq!(CelErr::NoSuchKey(NameId(0)).operator(), None);
    }
}
