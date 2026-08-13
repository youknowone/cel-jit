//! The out-of-band error channel.
//!
//! Nothing reachable from the dispatch loop may return a `Result`. Front-end
//! B's `?`-to-exception-link lowering is anchored to pyre's own error type, so
//! a cel `Result` takes the other path and each `Ok`/`Err` constructor becomes
//! an `OpKind::New` plus a discriminant write — **a materialized two-word shell
//! per VM operation**, with no known class and no exception link. Parameterising
//! that lowering is a large majit change bought for a machine cel does not need.
//!
//! So errors travel the way upstream's do: raised into a side slot and signalled
//! by a sentinel return, the shape `OperationError` on the execution context has
//! rather than a return-value union. An operation that fails calls [`raise`],
//! which stores the error and hands back [`ERROR_SENTINEL`]; the caller tests the
//! returned pointer, not a tag. The `Result` API is reconstructed once, at the
//! `Program::execute` boundary, outside any traced graph.
//!
//! The rule this buys is worth stating as a rule, because it is cheap to break
//! and expensive to retrofit: **no `?` and no `Result` anywhere reachable from
//! the portal.**
//!
//! # Where the slot lives
//!
//! Thread-local for now. It belongs to the heap, and moves there with it — at
//! which point it also joins the root set, because [`CelError`] holds two
//! [`CelRef`]s and an unrooted slot holding managed pointers is exactly the
//! dangling hazard the opaque side table is contracted against. Until the
//! collector exists nothing is freed, so the slot cannot dangle yet; the note is
//! here so the root-set entry is added with the collector rather than discovered
//! after it.

use std::cell::RefCell;

use super::object::CelRef;

/// The value an operation returns when it has raised instead of producing.
///
/// Null is unambiguous because no successful operation ever yields it: a CEL
/// `null` is a real allocated [`super::object::W_NullObject`], not a null
/// pointer.
pub const ERROR_SENTINEL: CelRef = std::ptr::null_mut();

/// What went wrong, in the small vocabulary the dispatch loop can raise.
///
/// A plain `Copy` code plus the operands, rather than a rendered message: the
/// public `ExecutionError` spellings carry `Value`s, which do not exist at this
/// layer yet, so rendering happens at the boundary where they do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelErrCode {
    /// A checked arithmetic operation overflowed.
    Overflow,
    /// The divisor of `/` was zero.
    DivisionByZero,
    /// The divisor of `%` was zero.
    RemainderByZero,
    /// No overload of the operator accepts this pair of types.
    UnsupportedBinaryOperator,
    /// No overload of the operator accepts this type.
    NoSuchOverload,
}

/// A raised error: what happened, in which operator, to which operands.
///
/// `rhs` is [`ERROR_SENTINEL`] for a unary operator, matching the public
/// spelling of `negate`'s overflow, which pairs the operand with a zero.
#[derive(Debug, Clone, Copy)]
pub struct CelError {
    pub code: CelErrCode,
    pub op: &'static str,
    pub lhs: CelRef,
    pub rhs: CelRef,
}

thread_local! {
    /// At most one error is in flight: an operation that raises returns the
    /// sentinel immediately, so its caller either propagates or handles before
    /// another can be raised.
    static RAISED: RefCell<Option<CelError>> = const { RefCell::new(None) };
}

/// Raise `code` for `op` over `lhs`/`rhs` and return the sentinel.
///
/// Returning the sentinel rather than `()` is what lets a failing arm be spelled
/// `return raise(..)`, i.e. as a tail call in the same position as a successful
/// one, so no arm needs a branch on a tag.
pub fn raise(code: CelErrCode, op: &'static str, lhs: CelRef, rhs: CelRef) -> CelRef {
    RAISED.with(|slot| {
        *slot.borrow_mut() = Some(CelError { code, op, lhs, rhs });
    });
    ERROR_SENTINEL
}

/// Take the raised error, clearing the slot.
///
/// The boundary calls this after seeing a sentinel. It returns `None` when
/// nothing was raised, which is a caller bug rather than a state to handle:
/// a sentinel and a raised error are produced together by [`raise`].
pub fn take_error() -> Option<CelError> {
    RAISED.with(|slot| slot.borrow_mut().take())
}

/// Whether an error is currently in flight.
///
/// For assertions and for the boundary's own sanity checks; the dispatch loop
/// tests the returned pointer instead, which costs one compare against a
/// constant rather than a thread-local read.
pub fn has_error() -> bool {
    RAISED.with(|slot| slot.borrow().is_some())
}

/// Drop any raised error.
///
/// CEL's `&&` and `||` absorb errors from one operand when the other decides
/// the result, so the channel needs an explicit discard as well as a take.
pub fn clear_error() {
    RAISED.with(|slot| {
        *slot.borrow_mut() = None;
    });
}
