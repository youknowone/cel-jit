//! Shared immutable scope maps for [`crate::context::Context`] variable storage.
//!
//! A map is a node in a transition tree (`mapdict.py` `AbstractAttribute`):
//! `back`, `name`, `storageindex`, and a transition cache on the node
//! (`_get_new_attr` `cache_attrs`). Binding a new name transitions to a
//! child (`add_attr`); rebinding an existing name writes storage in place
//! and leaves the map unchanged.
//!
//! Maps are process-global and immortal: one static root terminator
//! (`Terminator`), nodes leaked, caches behind `Mutex`. Compiled traces
//! keep map pointers as constants.

use std::sync::Mutex;

use crate::objects::Value;

/// `find_location` miss: no binding and no type identifier.
pub(crate) const LOC_UNBOUND: i64 = -1;
/// A resolver sits on the remaining chain; the residual `intern_var_ptr` path runs.
pub(crate) const LOC_IMPURE: i64 = -2;
/// The name is a type identifier (`type_ident`); the leaf is a constant.
#[cfg(feature = "jit")]
pub(crate) const LOC_TYPE: i64 = -3;

const WALK_CAP: i64 = 4096;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Empty root (`Terminator`).
    RootTerminator,
    /// Empty child whose `back` is the parent's map.
    ChildTerminator,
    /// Resolver present on this scope (`attrkind` marker).
    Resolver,
    /// A bound name (`PlainAttribute`).
    Attribute,
}

/// One node of a scope map (`mapdict.py` `AbstractAttribute`).
///
/// Immortal: leaked to `'static`, never freed. `back` / `name` /
/// `storageindex` / `length` are written before the address is published.
/// The mutexes are the transition caches: `cache_attrs` is `_get_new_attr`,
/// `child_scope` is the per-parent-map child terminator, `with_resolver`
/// is the resolver-present child.
pub struct ScopeMap {
    kind: Kind,
    back: usize,
    name: &'static str,
    storageindex: i64,
    length: i64,
    cache_attrs: Mutex<Vec<(&'static str, usize)>>,
    child_scope: Mutex<Option<usize>>,
    with_resolver: Mutex<Option<usize>>,
}

const fn new_node(
    kind: Kind,
    back: usize,
    name: &'static str,
    storageindex: i64,
    length: i64,
) -> ScopeMap {
    ScopeMap {
        kind,
        back,
        name,
        storageindex,
        length,
        cache_attrs: Mutex::new(Vec::new()),
        child_scope: Mutex::new(None),
        with_resolver: Mutex::new(None),
    }
}

static ROOT: ScopeMap = new_node(Kind::RootTerminator, 0, "", -1, 0);

fn from_ptr(ptr: usize) -> &'static ScopeMap {
    // SAFETY: every non-zero pointer is either `ROOT` or a node leaked by
    // `add_attr` / `child_terminator` / `ensure_resolver`.
    unsafe { &*(ptr as *const ScopeMap) }
}

impl ScopeMap {
    pub(crate) fn root_terminator() -> &'static ScopeMap {
        &ROOT
    }

    #[cfg(feature = "jit")]
    pub(crate) fn from_bits(bits: i64) -> Option<&'static ScopeMap> {
        if bits == 0 {
            None
        } else {
            Some(from_ptr(bits as usize))
        }
    }

    pub(crate) fn as_bits(&'static self) -> i64 {
        self as *const ScopeMap as usize as i64
    }

    fn back_node(&'static self) -> Option<&'static ScopeMap> {
        if self.back == 0 {
            None
        } else {
            Some(from_ptr(self.back))
        }
    }

    /// `mapdict.py` `_find_map_attr` restricted to this scope: walk `back`
    /// while the node is a `PlainAttribute` (or a resolver marker), stop at
    /// a `Terminator`.
    pub(crate) fn find_in_this_scope(&'static self, name: &str) -> Option<u32> {
        let mut node = self;
        let mut steps = 0i64;
        loop {
            if node.kind == Kind::Attribute && node.name == name && node.storageindex >= 0 {
                return Some(node.storageindex as u32);
            }
            match node.kind {
                Kind::RootTerminator | Kind::ChildTerminator => return None,
                _ => {}
            }
            steps += 1;
            if steps >= WALK_CAP {
                return None;
            }
            node = node.back_node()?;
        }
    }

    fn this_scope_has_resolver(&'static self) -> bool {
        let mut node = self;
        let mut steps = 0i64;
        loop {
            match node.kind {
                Kind::Resolver => return true,
                Kind::RootTerminator | Kind::ChildTerminator => return false,
                _ => {}
            }
            steps += 1;
            if steps >= WALK_CAP {
                return false;
            }
            match node.back_node() {
                Some(back) => node = back,
                None => return false,
            }
        }
    }

    fn parent_scope(&'static self) -> Option<&'static ScopeMap> {
        let mut node = self;
        let mut steps = 0i64;
        loop {
            match node.kind {
                Kind::ChildTerminator => return node.back_node(),
                Kind::RootTerminator => return None,
                _ => {}
            }
            steps += 1;
            if steps >= WALK_CAP {
                return None;
            }
            node = node.back_node()?;
        }
    }

    /// Location of `name` on this map chain.
    ///
    /// Walks each scope the way `lookup_interned` does: resolver first,
    /// then this scope's names (`find_map_attr`), then the parent. A
    /// resolver anywhere the walk would consult is [`LOC_IMPURE`]. A miss
    /// at the root terminator is [`LOC_UNBOUND`]; the caller may promote
    /// that to [`LOC_TYPE`].
    pub(crate) fn find_location(&'static self, name: &str) -> i64 {
        let mut node = self;
        let mut depth = 0u32;
        let mut steps = 0i64;
        loop {
            if node.this_scope_has_resolver() {
                return LOC_IMPURE;
            }
            if let Some(index) = node.find_in_this_scope(name) {
                return ((depth as i64) << 32) | i64::from(index);
            }
            match node.parent_scope() {
                Some(parent) => {
                    depth = depth.saturating_add(1);
                    node = parent;
                }
                None => return LOC_UNBOUND,
            }
            steps += 1;
            if steps >= WALK_CAP {
                return LOC_UNBOUND;
            }
        }
    }

    /// `mapdict.py` `_get_new_attr` / `add_attr`: the child for `name`,
    /// cached on this node.
    pub(crate) fn add_attr(&'static self, name: &str) -> &'static ScopeMap {
        let mut guard = self
            .cache_attrs
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let mut i = 0;
        while i < guard.len() {
            if guard[i].0 == name {
                return from_ptr(guard[i].1);
            }
            i += 1;
        }
        let name_static: &'static str = Box::leak(name.to_string().into_boxed_str());
        let child = Box::leak(Box::new(new_node(
            Kind::Attribute,
            self as *const ScopeMap as usize,
            name_static,
            self.length,
            self.length.saturating_add(1),
        )));
        guard.push((name_static, child as *const ScopeMap as usize));
        child
    }

    /// Per-parent-map child scope terminator, cached on this node.
    ///
    /// The empty root terminator has no names and no resolver, so a child
    /// of it starts on the same node. Binding the same names from that
    /// child and from a fresh root then shares the map.
    pub(crate) fn child_terminator(&'static self) -> &'static ScopeMap {
        if self.kind == Kind::RootTerminator {
            return self;
        }
        let mut slot = self
            .child_scope
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if let Some(ptr) = *slot {
            return from_ptr(ptr);
        }
        let child = Box::leak(Box::new(new_node(
            Kind::ChildTerminator,
            self as *const ScopeMap as usize,
            "",
            -1,
            0,
        )));
        *slot = Some(child as *const ScopeMap as usize);
        child
    }

    /// Transition that marks a resolver present on this scope.
    pub(crate) fn ensure_resolver(&'static self) -> &'static ScopeMap {
        if self.this_scope_has_resolver() {
            return self;
        }
        let mut slot = self
            .with_resolver
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        if let Some(ptr) = *slot {
            return from_ptr(ptr);
        }
        let child = Box::leak(Box::new(new_node(
            Kind::Resolver,
            self as *const ScopeMap as usize,
            "",
            -1,
            self.length,
        )));
        *slot = Some(child as *const ScopeMap as usize);
        child
    }

    /// Bind `name` on this scope: in-place store when the name exists,
    /// otherwise `add_attr` and push.
    pub(crate) fn bind(
        map: &'static ScopeMap,
        storage: &mut Vec<Value>,
        name: &str,
        value: Value,
    ) -> &'static ScopeMap {
        if let Some(idx) = map.find_in_this_scope(name) {
            let idx = idx as usize;
            debug_assert!(idx < storage.len());
            if let Some(slot) = storage.get_mut(idx) {
                *slot = value;
            }
            map
        } else {
            let next = map.add_attr(name);
            debug_assert_eq!(next.storageindex as usize, storage.len());
            storage.push(value);
            next
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_names_in_the_same_order_share_a_map() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut ma = ScopeMap::root_terminator();
        let mut mb = ScopeMap::root_terminator();
        for (name, value) in [("x", 1i64), ("y", 2), ("a", 3), ("b", 4)] {
            ma = ScopeMap::bind(ma, &mut a, name, Value::Int(value));
            mb = ScopeMap::bind(mb, &mut b, name, Value::Int(value + 10));
        }
        assert!(std::ptr::eq(ma, mb));
        assert_eq!(
            a,
            vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]
        );
        assert_eq!(
            b,
            vec![
                Value::Int(11),
                Value::Int(12),
                Value::Int(13),
                Value::Int(14)
            ]
        );
    }

    #[test]
    fn rebind_does_not_change_the_map() {
        let mut storage = Vec::new();
        let map = ScopeMap::bind(
            ScopeMap::root_terminator(),
            &mut storage,
            "x",
            Value::Int(1),
        );
        let after = ScopeMap::bind(map, &mut storage, "x", Value::Int(9));
        assert!(std::ptr::eq(map, after));
        assert_eq!(storage, vec![Value::Int(9)]);
    }

    #[test]
    fn a_child_of_an_empty_root_shares_the_root_map() {
        let parent = ScopeMap::root_terminator();
        assert!(std::ptr::eq(parent.child_terminator(), parent));
        let mut a = Vec::new();
        let mut b = Vec::new();
        let ma = ScopeMap::bind(parent.child_terminator(), &mut a, "x", Value::Int(1));
        let mb = ScopeMap::bind(parent, &mut b, "x", Value::Int(2));
        assert!(std::ptr::eq(ma, mb));
    }

    #[test]
    fn two_children_of_one_parent_share_a_map() {
        let parent = ScopeMap::root_terminator();
        let t = parent.child_terminator();
        assert!(std::ptr::eq(t, parent.child_terminator()));
        let mut a = Vec::new();
        let mut b = Vec::new();
        let ma = ScopeMap::bind(t, &mut a, "x", Value::Int(1));
        let mb = ScopeMap::bind(t, &mut b, "x", Value::Int(2));
        assert!(std::ptr::eq(ma, mb));
        let loc = ma.find_location("x");
        assert!(loc >= 0);
        assert_eq!(loc >> 32, 0);
        assert_eq!(loc as u32, 0);
    }

    #[test]
    fn a_resolver_makes_the_location_impure() {
        let map = ScopeMap::root_terminator().ensure_resolver();
        assert_eq!(map.find_location("x"), LOC_IMPURE);
        let mut storage = Vec::new();
        let map = ScopeMap::bind(map, &mut storage, "x", Value::Int(1));
        assert_eq!(map.find_location("x"), LOC_IMPURE);
    }
}
