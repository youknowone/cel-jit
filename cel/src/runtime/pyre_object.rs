//! The class-instantiation lookup, spelled the way the boxing fuse reads it.
//!
//! `fuse_boxing_alloc` keeps one vtable address to stand in for **both**
//! dropped header stores, since the runtime stamps `ob_type` and `w_class`
//! from it. That substitution is only faithful when the object's `w_class` is
//! the type object of the very type `ob_type` names, so `resolve_vtable_addr`
//! checks it: it reads the `w_class` store, resolves it through
//! `get_instantiate_arg_addr`, and returns 0 — declining the fuse in silence —
//! unless the address equals the one `ob_type` carries.
//!
//! `get_instantiate_arg_addr` matches on a **three-segment path suffix**,
//! `pyre_object::pyobject::get_instantiate`, with exactly one argument. That
//! is why this module carries a name from another project: the suffix is the
//! recognition key, and a differently-named module of identical behaviour
//! would leave every allocation unfused with no error. The same reasoning is
//! written into the `charon-corpus` fixture that pins the lowering.

use super::object::CelClass;

/// The class of instances of `tp`.
///
/// A CEL value universe has no subclassing: the type object of an instance of
/// `T` is `T` itself, so this is the identity. It exists as a call rather than
/// as a direct `&CEL_INT_CLASS` because the fuse's check is written against
/// the call — `get_instantiate_arg_addr` walks the argument of *this* path to
/// recover the address it compares.
///
/// Where a universe does have subclasses the two disagree, and the fuse
/// correctly declines rather than re-synthesising a base type; the identity
/// here is what makes every cel allocation eligible.
pub mod pyobject {
    use super::CelClass;

    pub fn get_instantiate(tp: &'static CelClass) -> *const CelClass {
        tp
    }
}
