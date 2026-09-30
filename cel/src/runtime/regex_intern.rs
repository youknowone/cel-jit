//! Compiled-regex intern.
//!
//! P8: never compile a regex inside the traced loop. The compile is a
//! residual (`dont_look_inside`); the table is keyed by the pattern
//! string, the `@elidable` intern of a compiled pattern object. Two
//! evaluations of the same pattern share one `Regex`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

static TABLE: Mutex<Option<HashMap<String, Arc<regex::Regex>>>> = Mutex::new(None);

/// The compiled form of `pattern`, interned for the process.
///
/// # Safety of the cache
///
/// The key is the pattern text and the value is determined by it, so
/// identity of the `Arc` is not load-bearing — two tables would still
/// match the same strings.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
pub fn intern_regex(pattern: &str) -> Result<Arc<regex::Regex>, String> {
    let mut guard = TABLE.lock().unwrap_or_else(|p| p.into_inner());
    let table = guard.get_or_insert_with(HashMap::new);
    if let Some(re) = table.get(pattern) {
        return Ok(Arc::clone(re));
    }
    match regex::Regex::new(pattern) {
        Ok(re) => {
            let re = Arc::new(re);
            table.insert(pattern.to_string(), Arc::clone(&re));
            Ok(re)
        }
        Err(err) => Err(format!("'{pattern}' not a valid regex:\n{err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_pattern_is_one_compiled_regex() {
        let a = intern_regex("a.*").expect("compile");
        let b = intern_regex("a.*").expect("compile");
        assert!(
            Arc::ptr_eq(&a, &b),
            "a second compile must reuse the intern"
        );
        assert!(a.is_match("abc"));
        let other = intern_regex("z+").expect("compile");
        assert!(!Arc::ptr_eq(&a, &other));
        assert!(!other.is_match("abc"));
    }

    #[test]
    fn a_bad_pattern_is_not_interned() {
        assert!(intern_regex("(").is_err());
        assert!(intern_regex("(").is_err());
    }
}
