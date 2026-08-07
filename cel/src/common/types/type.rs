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
#[derive(Debug, Eq, PartialEq)]
pub struct TypeValue(Type);

impl TypeValue {
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

#[cfg(test)]
mod tests {
    use super::TypeValue;
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

    #[test]
    fn optional_is_named_by_its_own_family() {
        assert_eq!(type_name_of("type(optional.of(1))"), "optional_type");
        assert_eq!(type_name_of("type(optional.none())"), "optional_type");
    }
}
