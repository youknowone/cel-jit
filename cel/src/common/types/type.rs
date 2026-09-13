use std::sync::Arc;

use super::{type_of, Type};
use crate::objects::{Opaque, Value};
use crate::ExecutionError;

/// A CEL type used as a value.
///
/// Types are values in CEL — langdef.md, "Every value in CEL has a runtime type
/// which is itself a value" — so `type(x)` has to return one.
///
/// It is carried as an [`Opaque`] rather than a new `Value` variant because a
/// type value is inert: nothing indexes it, adds it, iterates it or converts
/// it. A new variant would oblige every exhaustive `match` on `Value` — the
/// walker, the bytecode VM, the JIT lowering — to answer for a family that has
/// no behaviour beyond equality and its own name.
///
/// ⚠ The **derived** `Debug` is load-bearing, and not for debugging.
/// `Value`'s own `Debug` renders an opaque as `Opaque<{runtime_type_name}>({..})`
/// (`objects.rs:1011`), and this value's `runtime_type_name` is `type` for every
/// type it denotes — so the `{..}` half, which is this derive, is the ONLY thing
/// distinguishing `type(1)` from `type('a')` in that rendering.
/// `tests/vm_walker_sweep.rs` compares evaluators by `Debug` and carries
/// `type()` rows for exactly this reason. Replacing the derive with a
/// hand-written impl that prints only the name would make that gate compare
/// equal strings for every type value and stop discriminating, silently.
#[derive(Debug, Eq, PartialEq)]
pub struct TypeValue(Type);

impl TypeValue {
    /// Wrap a CEL type so it can sit on the public value boundary.
    pub(crate) fn new(denoted: Type) -> Self {
        TypeValue(denoted)
    }

    /// The type this value denotes.
    pub fn cel_type(&self) -> &Type {
        &self.0
    }

    /// The denoted type's name: `int`, `list`, `google.protobuf.Timestamp`.
    ///
    /// This is NOT the value's own runtime type, which is always `type`.
    pub fn name(&self) -> &str {
        self.0.name()
    }
}

impl Opaque for TypeValue {
    /// A type value's own runtime type is `type`, whatever type it denotes.
    ///
    /// That is what closes the recursion the spec asserts with
    /// `type(type(1)) == type(string)`: [`type_of`] reads this name back, so
    /// `type(int)` is `type` and `type(type)` is `type` again.
    ///
    /// It also makes equality name-blind in the right direction. `opaque_eq`
    /// rejects a mismatched `runtime_type_name` before downcasting, so every
    /// pair of type values gets as far as comparing the types they denote,
    /// which is where `type(1) == type(2)` and `type(1) != type('a')` are
    /// decided.
    fn runtime_type_name(&self) -> &str {
        // Spelled literally rather than through `TYPE_TYPE.name()`, which
        // cannot outlive the borrow of a `const`. `name_agrees_with_type_type`
        // holds the two together.
        "type"
    }
}

/// `type(x)` — the type denotation function.
///
/// Declared `type(A) -> type` for any `A`, so the overload takes `DYN_TYPE`.
fn to_type(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    Ok(Value::Opaque(Arc::new(TypeValue(type_of(&args.remove(0))))))
}

pub(crate) fn stdlib(env: &mut crate::Env) {
    env.add_overload("type", "type", vec![super::DYN_TYPE], to_type)
        .expect("Must be unique id");
}

/// Resolves a bare type identifier -- `int`, `string`, `null_type` -- to the
/// type value it denotes.
///
/// CEL keeps the function namespace and the identifier namespace separate:
/// `int(1.9)` is a call on the conversion overload `common/types/int.rs`
/// registers, while bare `int` is an identifier that has to resolve to a value.
/// The two spellings share a name and neither disturbs the other, because
/// nothing resolves a callee through [`Context::get_variable`] -- the call path
/// goes to `get_function` in both evaluators (`objects.rs:1526`,
/// `vm/interp.rs:669`). Without the identifier half, langdef.md's own worked
/// example for this function, `type(type(1)) == type(string)`, cannot be
/// written at all.
///
/// [`Context::get_variable`] consults this only after the whole context chain
/// has missed, so a bound variable named `int` shadows the type. That order is
/// the spec's, quoted at `objects.rs:1541` for the analogous case: a local
/// variable "shadows any identifier named `x` in ancestor scopes or the package
/// namespace".
///
/// The names are exactly those the oracle corpus already pins as `type(x)`
/// answers -- the spec's spellings, per its own note that the null family is
/// `null_type` and not `null` -- minus the two that are not identifiers.
/// `google.protobuf.Duration` and `google.protobuf.Timestamp` parse as an
/// `Expr::Select` over an `Ident`, so they arrive at a different resolution
/// path entirely. `dyn` is absent for a different reason: it is never a
/// `type(x)` answer, and `common/types/dyn.rs:15` already spends the name on
/// the function namespace.
///
/// ⚠ `optional_type` is an extension in CEL proper, not core, and is bound here
/// because `env.rs:89` installs the optional stdlib unconditionally. If that
/// ever becomes feature-gated, this entry has to be gated with it, or the
/// identifier outlives the type it denotes.
pub(crate) fn type_ident(name: &str) -> Option<Value> {
    use super::{
        BOOL_TYPE, BYTES_TYPE, DOUBLE_TYPE, INT_TYPE, LIST_TYPE, MAP_TYPE, NULL_TYPE,
        OPTIONAL_TYPE, STRING_TYPE, TYPE_TYPE, UINT_TYPE,
    };
    // Matched on the literal rather than scanned out of a table of the
    // constants: `Type` has no `Clone` -- only the hand-written `ToOwned` at
    // `mod.rs:60`, because `parameters` is a `Cow` -- so a `&'static [Type]`
    // cannot be promoted. `ident_keys_are_the_types_own_names` holds each arm's
    // key to the name of the type that arm returns, so the two cannot drift.
    let denoted = match name {
        "bool" => BOOL_TYPE,
        "bytes" => BYTES_TYPE,
        "double" => DOUBLE_TYPE,
        "int" => INT_TYPE,
        "list" => LIST_TYPE,
        "map" => MAP_TYPE,
        "null_type" => NULL_TYPE,
        "optional_type" => OPTIONAL_TYPE,
        "string" => STRING_TYPE,
        "type" => TYPE_TYPE,
        "uint" => UINT_TYPE,
        _ => return None,
    };
    Some(Value::Opaque(Arc::new(TypeValue(denoted))))
}

/// The type names a JIT `ValType::Type` index denotes, in index order.
///
/// A type value is inert -- equality and its own name are all it has -- so an
/// index into a FIXED table is a complete encoding of one, and no side table
/// has to travel with a lowered program the way the string ranks do. The names
/// are exactly [`type_ident`]'s keys, which is what lets one index answer both
/// directions: the lowering takes a folded `type(x)` constant to an index by
/// name, and the output takes the index back to a value through `type_ident`.
/// `type_const_names_are_type_idents` holds the two lists together.
///
/// A type OUTSIDE the table has no index and the lowering declines it. That is
/// every message type, `google.protobuf.Timestamp` included: `type_ident` does
/// not bind those either, because they are not spellable as an identifier.
pub(crate) const TYPE_CONST_NAMES: [&str; 11] = [
    "bool",
    "bytes",
    "double",
    "int",
    "list",
    "map",
    "null_type",
    "optional_type",
    "string",
    "type",
    "uint",
];

/// The index [`TYPE_CONST_NAMES`] gives `name`, or `None` for a type it does
/// not carry.
pub(crate) fn type_const_id(name: &str) -> Option<i64> {
    TYPE_CONST_NAMES
        .iter()
        .position(|n| *n == name)
        .map(|i| i as i64)
}

/// The type value an index denotes -- the inverse of [`type_const_id`].
///
/// Panics on an index no [`type_const_id`] produced, which is a lowering that
/// minted an index this table cannot answer for.
pub(crate) fn type_const_value(id: i64) -> Value {
    let name = TYPE_CONST_NAMES
        .get(id as usize)
        .expect("an index type_const_id minted");
    type_ident(name).expect("a name type_ident binds")
}

#[cfg(test)]
mod tests {
    use super::{type_const_id, type_const_value, TypeValue, TYPE_CONST_NAMES};
    use crate::common::types::TYPE_TYPE;
    use crate::objects::Value;
    use crate::{Context, Program};

    fn eval(expr: &str) -> crate::objects::ResolveResult {
        Program::compile(expr).unwrap().execute(&Context::default())
    }

    /// The denoted type's name, so a test can say what `type(x)` answered
    /// without reaching through `Opaque` at every call site.
    fn type_name_of(expr: &str) -> String {
        match eval(expr).unwrap() {
            Value::Opaque(o) => o
                .downcast_ref::<TypeValue>()
                .unwrap_or_else(|| panic!("`{expr}` is not a type value"))
                .name()
                .to_owned(),
            other => panic!("`{expr}` answered {other:?}, not a type value"),
        }
    }

    /// Every arm's key is the name of the type that arm returns, so an
    /// identifier and the type it denotes cannot drift apart -- a mistyped key
    /// makes the lookup miss rather than answer the wrong type. Same reason
    /// `name_agrees_with_type_type` below exists.
    #[test]
    fn ident_keys_are_the_types_own_names() {
        use crate::common::types::{
            BOOL_TYPE, BYTES_TYPE, DOUBLE_TYPE, INT_TYPE, LIST_TYPE, MAP_TYPE, NULL_TYPE,
            OPTIONAL_TYPE, STRING_TYPE, TYPE_TYPE, UINT_TYPE,
        };
        let bound = [
            BOOL_TYPE,
            BYTES_TYPE,
            DOUBLE_TYPE,
            INT_TYPE,
            LIST_TYPE,
            MAP_TYPE,
            NULL_TYPE,
            OPTIONAL_TYPE,
            STRING_TYPE,
            TYPE_TYPE,
            UINT_TYPE,
        ];
        assert_eq!(bound.len(), 11, "a name was added or dropped without a row");
        for t in bound {
            let name = t.name().to_owned();
            let Some(Value::Opaque(o)) = super::type_ident(&name) else {
                panic!("`{name}` does not resolve as an identifier");
            };
            let got = o.downcast_ref::<TypeValue>().expect("a type value");
            assert_eq!(got.cel_type(), &t, "`{name}` denotes the wrong type");
        }
        assert!(super::type_ident("no_such_type").is_none());
    }

    #[test]
    fn name_agrees_with_type_type() {
        // The literal in `runtime_type_name` and the constant cannot drift.
        let value = eval("type(1)").unwrap();
        let Value::Opaque(o) = value else {
            panic!("not opaque");
        };
        assert_eq!(o.runtime_type_name(), TYPE_TYPE.name());
    }

    #[test]
    fn names_every_value_family_the_way_the_spec_spells_it() {
        assert_eq!(type_name_of("type(1)"), "int");
        assert_eq!(type_name_of("type(1u)"), "uint");
        assert_eq!(type_name_of("type(1.5)"), "double");
        assert_eq!(type_name_of("type(true)"), "bool");
        assert_eq!(type_name_of("type('a')"), "string");
        assert_eq!(type_name_of("type(b'a')"), "bytes");
        assert_eq!(type_name_of("type([1])"), "list");
        assert_eq!(type_name_of("type({'a': 1})"), "map");
        // Not `null`: langdef.md names the family `null_type`.
        assert_eq!(type_name_of("type(null)"), "null_type");
    }

    #[cfg(feature = "chrono")]
    #[test]
    fn names_the_temporal_families_by_their_protobuf_names() {
        assert_eq!(
            type_name_of("type(duration('1s'))"),
            "google.protobuf.Duration"
        );
        assert_eq!(
            type_name_of("type(timestamp('2000-01-01T00:00:00Z'))"),
            "google.protobuf.Timestamp"
        );
    }

    #[test]
    fn a_types_own_type_is_type() {
        // langdef.md: "those values (`int`, `string`, etc.) also have a type:
        // the type `type`, which is an expression by itself which in turn also
        // has type `type`".
        assert_eq!(type_name_of("type(type(1))"), "type");
        assert_eq!(type_name_of("type(type(type(1)))"), "type");
    }

    #[test]
    fn type_values_compare_by_the_type_they_denote() {
        assert_eq!(eval("type(1) == type(2)"), Ok(true.into()));
        assert_eq!(eval("type(1) == type('a')"), Ok(false.into()));
        assert_eq!(eval("type([1]) == type(['a', 'b'])"), Ok(true.into()));
        assert_eq!(eval("type(1) != type(1u)"), Ok(true.into()));
        // The recursion the spec's own example asserts.
        assert_eq!(eval("type(type(1)) == type(type('a'))"), Ok(true.into()));
    }

    #[test]
    fn a_type_value_is_not_equal_to_its_name() {
        // `type()` returns a type, not the string spelling one.
        assert_eq!(eval("type(1) == 'int'"), Ok(false.into()));
    }

    /// Every index name is a name `type_ident` binds, and the round trip
    /// `name -> id -> value` lands on a value denoting that same name. Without
    /// this the two lists drift and an index starts denoting a different type
    /// than the lowering meant, which no test of either list alone would see.
    #[test]
    fn type_const_names_are_type_idents() {
        for (i, name) in TYPE_CONST_NAMES.iter().enumerate() {
            assert_eq!(type_const_id(name), Some(i as i64), "id of `{name}`");
            let v = type_const_value(i as i64);
            let Value::Opaque(o) = &v else {
                panic!("`{name}` is not an opaque");
            };
            assert_eq!(
                o.downcast_ref::<TypeValue>().expect("a type value").name(),
                *name
            );
        }
        assert_eq!(type_const_id("google.protobuf.Timestamp"), None);
        assert_eq!(type_const_id("dyn"), None);
    }

    #[test]
    fn optional_is_named_by_its_own_family() {
        assert_eq!(type_name_of("type(optional.of(1))"), "optional_type");
        assert_eq!(type_name_of("type(optional.none())"), "optional_type");
    }
}
