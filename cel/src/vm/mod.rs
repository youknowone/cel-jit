//! A bytecode VM for CEL.
//!
//! This module is being built in stages and does not execute anything yet.
//! What is here is the shape everything later depends on: the instruction
//! set, the code object, and the error channel.
//!
//! # Why a code object at all
//!
//! The tree walker resolves the same questions on every evaluation that the
//! expression already answers once: whether an identifier is a comprehension
//! variable or a context lookup, whether a member call is really a namespaced
//! function, which indices of a list literal are optional, whether a `Select`
//! is a field read or a `has` test. Each is a compile-time fact reached at run
//! time. A code object is where those answers are written down.
//!
//! # Error absorption in `&&` and `||`
//!
//! `&&` and `||` in CEL absorb errors: `error && false` is `false`, and so is
//! `false && error`. The walker gets this for free by *not* using `?` on the
//! left operand -- it is a recursive evaluator, so capturing a
//! sub-evaluation's error costs nothing (`objects.rs`, the `LOGICAL_AND`
//! arm). A flat instruction stream has no such enclosing scope, so the
//! absorbing merge needs a way to catch an error raised anywhere inside the
//! left operand's instructions.
//!
//! The requirement is what rules the options out: **absorption must be able to
//! turn a left-hand error into a successful `false`.** A write-once error flag
//! cannot express that without a save/clear/restore protocol, which is
//! catching with extra state and an extra invariant to get wrong.
//!
//! So the code object carries a [`Handler`] table mapping an instruction range
//! to the merge that absorbs it -- the same device an exception table is, and
//! `Result`-native: the dispatch loop consults it when a step returns `Err`.
//! Each operator gets a range covering its left operand, a slot to record the
//! outcome in, and a merge instruction that weighs the two sides. The right
//! operand is deliberately outside the range, because the walker reaches it
//! through `?`: `1 && undefined_name` is `UndeclaredReference`, not the
//! overload failure the left operand already recorded.
//!
//! [`Handler`]: code::Handler

pub mod code;
pub mod compile;
pub mod error;
pub mod opcode;

pub use code::{CelCode, Handler};
pub use compile::{compile, CompileError, CompileErrorKind};
pub use error::{CelErr, CelResult, ColdId, NameId};
pub use opcode::{OpCode, OPCODE_COUNT};
