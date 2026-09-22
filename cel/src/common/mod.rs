pub mod ast;
pub mod decls;
pub mod functions;
pub mod traits;
pub mod types;

use std::collections::BTreeMap;
use std::ops::Bound;

/// Looks up the entry a caller would find under `"{prefix}.{name}"`, without
/// building that string.
///
/// A member call whose target parses as an identifier has to ask whether it is
/// really a namespaced call -- `math.max(x)` rather than a `max` method on a
/// variable named `math`. Both tables it asks are keyed by the joined name, and
/// joining it allocates on every evaluation of a call that is almost never
/// namespaced.
///
/// Keys sharing a prefix are contiguous in a [`BTreeMap`], so the candidates
/// start at the first key at or after `prefix` and end at the first key that no
/// longer begins with it. In the overwhelmingly common case -- no function
/// registered under that prefix at all -- that is the seek plus a single
/// rejected comparison.
pub(crate) fn get_qualified<'a, V>(
    map: &'a BTreeMap<String, V>,
    prefix: &str,
    name: &str,
) -> Option<&'a V> {
    map.range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
        .take_while(|(key, _)| key.starts_with(prefix))
        .find(|(key, _)| {
            key[prefix.len()..]
                .strip_prefix('.')
                .is_some_and(|rest| rest == name)
        })
        .map(|(_, value)| value)
}

#[cfg(test)]
mod tests {
    use super::get_qualified;
    use std::collections::BTreeMap;

    /// The joined-name lookup this replaces, kept as the oracle.
    fn joined<'a, V>(map: &'a BTreeMap<String, V>, prefix: &str, name: &str) -> Option<&'a V> {
        map.get(&format!("{prefix}.{name}"))
    }

    #[test]
    fn agrees_with_the_joined_lookup() {
        let mut map = BTreeMap::new();
        // `size`/`startsWith` share `s`'s prefix without being namespaced under
        // it, and `s.ize` is the shape a naive suffix test would confuse with
        // `size`. `math.maximum` sorts adjacent to `math.max`.
        for key in [
            "size",
            "startsWith",
            "string",
            "s.ize",
            "math.max",
            "math.maximum",
            "math",
            "math.",
            "matha.max",
        ] {
            map.insert(key.to_owned(), key.to_owned());
        }
        for (prefix, name) in [
            ("math", "max"),
            ("math", "maximum"),
            ("math", "min"),
            ("math", ""),
            ("matha", "max"),
            ("s", "ize"),
            ("s", "tartsWith"),
            ("size", "x"),
            ("", "math"),
            ("nothing", "here"),
        ] {
            assert_eq!(
                get_qualified(&map, prefix, name),
                joined(&map, prefix, name),
                "prefix={prefix:?} name={name:?}"
            );
        }
    }
}
