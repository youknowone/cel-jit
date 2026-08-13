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
//! Today the body is a plain `Box` leak. The tracing heap that replaces it is
//! a later slice: what matters for lowering is the *shape* of the call, which
//! is fixed now so the class family below it never has to be re-laid.

/// Allocate `value` on the heap and return a raw pointer to it.
///
/// # Safety of the returned pointer
///
/// Nothing frees it. Until the tracing heap lands, every value allocated here
/// leaks; that is deliberate for this slice, because the alternative — a
/// `Drop`-based owner — reintroduces exactly the owning-place structure the
/// class family exists to remove.
pub fn malloc_typed<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}

/// The managed counterpart, accepted by the same matcher.
///
/// Kept distinct from [`malloc_typed`] because the matcher names both and the
/// heap will eventually route them differently — one to the collector's
/// managed old-gen, one to an immortal region. They are the same call today.
pub fn malloc_typed_managed<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}
