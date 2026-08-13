//! The allocation entry point the boxing fuse recognises.
//!
//! `fuse_boxing_alloc` (majit-translate `model.rs`) turns a heap allocation
//! into a `NewWithVtable` the trace optimizer can delete, but only for one
//! exact spelling. Its matcher, `is_malloc_typed`, accepts a call whose path
//! ends in `lltype::malloc_typed` or `lltype::malloc_typed_managed` **and
//! which takes exactly one argument** — the fully built value, passed by
//! value. Hence this module's name is load-bearing: it is the second-to-last
//! path segment the matcher tests.
//!
//! The alternative spelling — allocate first, then store the fields through
//! the returned pointer — matches nothing and produces **zero** fused
//! allocations with no diagnostic at all, because the fuse declines with a
//! bare `continue`. See [`super::object`] for the other two silent-decline
//! conditions and the tripwire tests that pin all three.
//!
//! The body is free to grow, and this one did: it was `Box::into_raw` and now
//! reaches this thread's [`super::heap::CelHeap`]. What the matcher inspects
//! is the *call* — the path's last two segments and the argument count — never
//! the callee's body, and the fuse replaces the call outright when it fires, so
//! the body is only ever lowered on the paths that declined.

use super::heap;

/// Allocate `value` on this thread's heap and return a raw pointer to it.
///
/// # Safety of the returned pointer
///
/// It is valid until the thread's heap is torn down, and no sooner: nothing
/// collects yet, because the root set the design enumerates has no walker. See
/// [`super::heap`] for what that bounds and what it does not.
pub fn malloc_typed<T>(value: T) -> *mut T {
    heap::with_heap(|h| h.alloc(value))
}

/// The managed counterpart, accepted by the same matcher.
///
/// Kept distinct from [`malloc_typed`] because the matcher names both and the
/// heap will eventually route them differently — one to the collector's
/// managed old-gen, one to an immortal region. They are the same call today.
pub fn malloc_typed_managed<T>(value: T) -> *mut T {
    heap::with_heap(|h| h.alloc(value))
}
