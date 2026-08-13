//! The class-based value universe.
//!
//! **Unstable and not yet reachable from [`crate::Value`].** This is the first
//! slice of the re-lay that replaces the `Value` enum — and the `Arc`s inside
//! it — with a header-first class family allocated through a tracing heap. It
//! is additive: nothing here is wired into the evaluators yet, so the crate
//! ships exactly as before and the slice is revertible by deleting the
//! directory.
//!
//! # Why the enum is being replaced
//!
//! Front-end B's only general enum-variant lowering is anchored to
//! `core::result::Result`, so a `CelValue::Int(x)` arrives at the optimizer as
//! an opaque residual — invisible to OptVirtualize, whose entire surface is
//! the allocation opcodes. An enum also forfeits `known_class`, the same field
//! that carries `_is_virtual`. A class family gets both: the allocation fuses
//! to a `NewWithVtable` the optimizer can delete, and the class word gives
//! dispatch a pointer-identity test the annotator can narrow.
//!
//! The `Arc`s are the other half, and they are measurable: a census of the
//! rtyper's two-phase prepass over cel's own closure reports 88 of 95 graphs
//! falling back to the legacy walker, and the single largest cause —
//! 16 of them — is `sync::Arc::deref`. The value universe's `Arc` is not one
//! obstacle among many; it is the head of the list.
//!
//! # What is in this slice
//!
//! The fixed-size scalar leaves only: `int`, `uint`, `double`, `bool`, `null`,
//! `duration`, `timestamp` and the type value. Those need nothing but the
//! fixed-size boxing fuse, which is landed and tested. Everything with a
//! variable-length payload — strings, bytes, lists, maps, structs — needs
//! varsize allocation lowering, which is not, so it is a later slice rather
//! than a half-written one here.
//!
//! Read [`object`] before adding a leaf: three separate conditions have to
//! hold for an allocation to fuse, and all three fail silently.

pub mod binop;
pub mod error;
pub mod lltype;
pub mod object;
pub mod optional;
pub mod pyre_object;
