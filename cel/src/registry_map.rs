//! Shared immutable registry maps for [`crate::magic::FunctionRegistry`].
//!
//! A map is a node in a transition tree (`mapdict.py` `AbstractAttribute`):
//! `back`, `name`, `storageindex`, and a transition cache on the node
//! (`_get_new_attr` `cache_attrs`). Binding a new name transitions to a
//! child (`add_attr`); rebinding an existing name writes storage in place
//! and leaves the map unchanged.
//!
//! Maps are process-global and immortal: one static root terminator
//! (`Terminator`), nodes leaked, caches behind `Mutex`. Compiled traces
//! keep map pointers as constants. Every [`crate::Context::default`]
//! starts from this terminator, so a root that adds no function has the
//! same map.

use std::sync::Mutex;

const WALK_CAP: i64 = 4096;

/// One node of a registry map (`mapdict.py` `AbstractAttribute`).
///
/// Immortal: leaked to `'static`, never freed. `back` / `name` /
/// `storageindex` / `length` are written before the address is published.
/// `cache_attrs` is `_get_new_attr`.
pub struct RegistryMap {
    back: usize,
    name: &'static str,
    storageindex: i64,
    length: i64,
    cache_attrs: Mutex<Vec<(&'static str, usize)>>,
}

const fn new_node(back: usize, name: &'static str, storageindex: i64, length: i64) -> RegistryMap {
    RegistryMap {
        back,
        name,
        storageindex,
        length,
        cache_attrs: Mutex::new(Vec::new()),
    }
}

static ROOT: RegistryMap = new_node(0, "", -1, 0);

fn from_ptr(ptr: usize) -> &'static RegistryMap {
    // SAFETY: every non-zero pointer is either `ROOT` or a node leaked by
    // `add_attr`.
    unsafe { &*(ptr as *const RegistryMap) }
}

fn qualified_eq(full: &str, prefix: &str, name: &str) -> bool {
    full.len() == prefix.len() + 1 + name.len()
        && full.starts_with(prefix)
        && full.as_bytes().get(prefix.len()) == Some(&b'.')
        && full[prefix.len() + 1..] == *name
}

impl RegistryMap {
    pub(crate) fn root_terminator() -> &'static RegistryMap {
        &ROOT
    }

    #[cfg(feature = "jit")]
    pub(crate) fn from_bits(bits: i64) -> Option<&'static RegistryMap> {
        if bits == 0 {
            None
        } else {
            Some(from_ptr(bits as usize))
        }
    }

    pub(crate) fn as_bits(&'static self) -> i64 {
        self as *const RegistryMap as usize as i64
    }

    pub(crate) fn storageindex(&'static self) -> i64 {
        self.storageindex
    }

    fn back_node(&'static self) -> Option<&'static RegistryMap> {
        if self.back == 0 {
            None
        } else {
            Some(from_ptr(self.back))
        }
    }

    /// `mapdict.py` `find_map_attr`: walk `back` until `name` matches.
    /// `-1` is a miss. The terminator's empty name does not match.
    pub(crate) fn find_map_attr(&'static self, name: &str) -> i64 {
        let mut node = self;
        let mut steps = 0i64;
        loop {
            if node.storageindex >= 0 && node.name == name {
                return node.storageindex;
            }
            steps += 1;
            if steps >= WALK_CAP {
                return -1;
            }
            match node.back_node() {
                Some(back) => node = back,
                None => return -1,
            }
        }
    }

    /// Name stored at `index`, walking from this node.
    pub(crate) fn name_at(&'static self, index: i64) -> Option<&'static str> {
        if index < 0 {
            return None;
        }
        let mut node = self;
        let mut steps = 0i64;
        loop {
            if node.storageindex == index {
                return Some(node.name);
            }
            steps += 1;
            if steps >= WALK_CAP {
                return None;
            }
            match node.back_node() {
                Some(back) => node = back,
                None => return None,
            }
        }
    }

    /// `find_map_attr` for the joined name `prefix.name`, without building
    /// that string.
    pub(crate) fn find_qualified(&'static self, prefix: &str, name: &str) -> i64 {
        let mut node = self;
        let mut steps = 0i64;
        loop {
            if node.storageindex >= 0 && qualified_eq(node.name, prefix, name) {
                return node.storageindex;
            }
            steps += 1;
            if steps >= WALK_CAP {
                return -1;
            }
            match node.back_node() {
                Some(back) => node = back,
                None => return -1,
            }
        }
    }

    /// `mapdict.py` `_get_new_attr` / `add_attr`: the child for `name`,
    /// cached on this node.
    pub(crate) fn add_attr(&'static self, name: &str) -> &'static RegistryMap {
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
            self as *const RegistryMap as usize,
            name_static,
            self.length,
            self.length.saturating_add(1),
        )));
        guard.push((name_static, child as *const RegistryMap as usize));
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_names_in_the_same_order_share_a_map() {
        let mut ma = RegistryMap::root_terminator();
        let mut mb = RegistryMap::root_terminator();
        ma = ma.add_attr("add");
        mb = mb.add_attr("add");
        ma = ma.add_attr("multiply");
        mb = mb.add_attr("multiply");
        assert!(std::ptr::eq(ma, mb));
        assert_eq!(ma.find_map_attr("add"), 0);
        assert_eq!(ma.find_map_attr("multiply"), 1);
        assert_eq!(ma.find_map_attr("missing"), -1);
        let q = ma.add_attr("optional.of");
        assert_eq!(q.find_qualified("optional", "of"), 2);
        assert_eq!(q.find_qualified("optional", "none"), -1);
    }

    #[test]
    fn empty_roots_share_the_terminator() {
        assert!(std::ptr::eq(
            RegistryMap::root_terminator(),
            RegistryMap::root_terminator()
        ));
    }
}
