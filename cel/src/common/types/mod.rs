use crate::common::traits;
use crate::common::traits::TraitSet;
use crate::ExecutionError;
use std::borrow::Cow;

pub(crate) mod bytes;
pub(crate) mod double;
#[cfg(feature = "chrono")]
pub(crate) mod duration;
pub(crate) mod r#dyn;
pub(crate) mod int;
pub(crate) mod list;
pub(crate) mod map;
pub(crate) mod optional;
pub(crate) mod string;
#[cfg(feature = "structs")]
pub(crate) mod r#struct;
#[cfg(feature = "chrono")]
pub(crate) mod timestamp;
pub(crate) mod uint;

use crate::objects::{OptionalValue, Value};
#[cfg(feature = "structs")]
pub use r#struct::Struct as CelStruct;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Unspecified,
    Error,
    Dyn,
    Any,
    Boolean,
    Bytes,
    Double,
    Duration,
    Int,
    List,
    Map,
    NullType,
    Opaque,
    String,
    Struct,
    Timestamp,
    Type,
    TypeParam,
    UInt,
    Unknown,
}

/// Represents a CEL type.
#[derive(Debug, Eq, PartialEq)]
pub struct Type {
    kind: Kind,
    parameters: Cow<'static, [Cow<'static, Type>]>,
    runtime_type_name: Cow<'static, str>,
    trait_mask: TraitSet,
}

impl ToOwned for Type {
    type Owned = Type;

    fn to_owned(&self) -> Self::Owned {
        Self {
            kind: self.kind,
            parameters: self.parameters.clone(),
            runtime_type_name: self.runtime_type_name.clone(),
            trait_mask: self.trait_mask,
        }
    }
}

impl Type {
    /// Returns true if the given value can be assigned to this type.
    ///
    /// `Kind::Opaque` delegates to its first parameter, so `OPTIONAL_TYPE` —
    /// parameterised on `DYN_TYPE` — accepts every value, and the optional
    /// overloads are therefore selected on name and arity alone.
    pub fn is_assignable(&self, val: &Value) -> bool {
        if self.matches(val) {
            true
        } else {
            match self.kind() {
                Kind::Dyn => true,
                Kind::Opaque => self
                    .parameters
                    .first()
                    .is_some_and(|t| t.is_assignable(val)),
                _ => false,
            }
        }
    }

    /// Whether `self` is exactly the value's own type, without building it.
    ///
    /// Every family reports a shared constant except the two that carry a name:
    /// an opaque handle names its host type, and a struct names itself.
    fn matches(&self, val: &Value) -> bool {
        let constant = match val {
            Value::Bool(_) => &BOOL_TYPE,
            Value::Int(_) => &INT_TYPE,
            Value::UInt(_) => &UINT_TYPE,
            Value::Float(_) => &DOUBLE_TYPE,
            Value::String(_) => &STRING_TYPE,
            Value::Bytes(_) => &BYTES_TYPE,
            Value::Null => &NULL_TYPE,
            Value::List(_) => &LIST_TYPE,
            Value::Map(_) => &MAP_TYPE,
            #[cfg(feature = "chrono")]
            Value::Duration(_) => &DURATION_TYPE,
            #[cfg(feature = "chrono")]
            Value::Timestamp(_) => &TIMESTAMP_TYPE,
            #[cfg(feature = "structs")]
            Value::Struct(s) => return self == s.cel_type(),
            Value::Opaque(o) => {
                return if o.downcast_ref::<OptionalValue>().is_some() {
                    self == &OPTIONAL_TYPE
                } else {
                    // The shape `Type::new_opaque_type` builds for a host value.
                    self.kind == Kind::Opaque
                        && self.parameters.is_empty()
                        && self.trait_mask == 0
                        && self.runtime_type_name == o.runtime_type_name()
                };
            }
        };
        self == constant
    }
}

impl Type {
    /// Returns the kind of the type.
    pub fn kind(&self) -> Kind {
        self.kind
    }
}

pub const ANY_TYPE: Type = Type {
    kind: Kind::Any,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("google.protobuf.Any"),
    trait_mask: traits::FIELD_TESTER_TYPE | traits::INDEXER_TYPE,
};

pub const BOOL_TYPE: Type = Type {
    kind: Kind::Boolean,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("bool"),
    trait_mask: traits::COMPARER_TYPE | traits::NEGATOR_TYPE,
};

pub const BYTES_TYPE: Type = Type {
    kind: Kind::Bytes,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("bytes"),
    trait_mask: traits::ADDER_TYPE | traits::COMPARER_TYPE | traits::SIZER_TYPE,
};

pub const DOUBLE_TYPE: Type = Type {
    kind: Kind::Double,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("double"),
    trait_mask: traits::ADDER_TYPE
        | traits::COMPARER_TYPE
        | traits::DIVIDER_TYPE
        | traits::MULTIPLIER_TYPE
        | traits::NEGATOR_TYPE
        | traits::SUBTRACTOR_TYPE,
};

pub const DURATION_TYPE: Type = Type {
    kind: Kind::Duration,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("google.protobuf.Duration"),
    trait_mask: traits::ADDER_TYPE
        | traits::COMPARER_TYPE
        | traits::NEGATOR_TYPE
        | traits::RECEIVER_TYPE
        | traits::SUBTRACTOR_TYPE,
};

pub const DYN_TYPE: Type = {
    let kind = Kind::Dyn;
    Type {
        kind,
        parameters: Cow::Borrowed(&[]),
        runtime_type_name: Cow::Borrowed("dyn"),
        trait_mask: 0,
    }
};

pub const ERROR_TYPE: Type = Type::simple_type(Kind::Error, "error");

pub const INT_TYPE: Type = Type {
    kind: Kind::Int,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("int"),
    trait_mask: traits::ADDER_TYPE
        | traits::COMPARER_TYPE
        | traits::DIVIDER_TYPE
        | traits::MODDER_TYPE
        | traits::MULTIPLIER_TYPE
        | traits::NEGATOR_TYPE
        | traits::SUBTRACTOR_TYPE,
};

pub const LIST_TYPE: Type = {
    Type {
        kind: Kind::List,
        parameters: Cow::Borrowed(&[Cow::Borrowed(&DYN_TYPE)]),
        runtime_type_name: Cow::Borrowed("list"),
        trait_mask: traits::ADDER_TYPE
            | traits::CONTAINER_TYPE
            | traits::INDEXER_TYPE
            | traits::ITERABLE_TYPE
            | traits::SIZER_TYPE,
    }
};

pub const MAP_TYPE: Type = {
    Type {
        kind: Kind::Map,
        parameters: Cow::Borrowed(&[Cow::Borrowed(&DYN_TYPE), Cow::Borrowed(&DYN_TYPE)]),
        runtime_type_name: Cow::Borrowed("map"),
        trait_mask: traits::CONTAINER_TYPE
            | traits::INDEXER_TYPE
            | traits::ITERABLE_TYPE
            | traits::SIZER_TYPE,
    }
};

pub const NULL_TYPE: Type = {
    let kind = Kind::NullType;
    Type {
        kind,
        parameters: Cow::Borrowed(&[]),
        runtime_type_name: Cow::Borrowed("null_type"),
        trait_mask: 0,
    }
};

pub const OPTIONAL_TYPE: Type = Type {
    kind: Kind::Opaque,
    parameters: Cow::Borrowed(&[Cow::Borrowed(&DYN_TYPE)]),
    runtime_type_name: Cow::Borrowed("optional_type"),
    trait_mask: 0,
};

pub const STRING_TYPE: Type = Type {
    kind: Kind::String,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("string"),
    trait_mask: traits::ADDER_TYPE
        | traits::COMPARER_TYPE
        | traits::MATCHER_TYPE
        | traits::RECEIVER_TYPE
        | traits::SIZER_TYPE,
};

pub const TIMESTAMP_TYPE: Type = Type {
    kind: Kind::Timestamp,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("google.protobuf.Timestamp"),
    trait_mask: traits::ADDER_TYPE
        | traits::COMPARER_TYPE
        | traits::RECEIVER_TYPE
        | traits::SUBTRACTOR_TYPE,
};

pub const TYPE_TYPE: Type = Type::simple_type(Kind::Type, "type");

pub const UINT_TYPE: Type = Type {
    kind: Kind::UInt,
    parameters: Cow::Borrowed(&[]),
    runtime_type_name: Cow::Borrowed("uint"),
    trait_mask: traits::ADDER_TYPE
        | traits::COMPARER_TYPE
        | traits::DIVIDER_TYPE
        | traits::MODDER_TYPE
        | traits::MULTIPLIER_TYPE
        | traits::SUBTRACTOR_TYPE,
};

pub const UNKNOWN_TYPE: Type = Type::simple_type(Kind::Unknown, "unknown");

impl Type {
    /// Creates a new simple type with the given kind and name.
    pub const fn simple_type(kind: Kind, name: &'static str) -> Type {
        Type {
            kind,
            parameters: Cow::Borrowed(&[]),
            runtime_type_name: Cow::Borrowed(name),
            trait_mask: 0,
        }
    }

    /// Creates a new list type with the given element type.
    pub fn new_list_type(param: &'static [Cow<Type>; 1]) -> Type {
        Type {
            kind: Kind::List,
            parameters: Cow::Borrowed(param),
            runtime_type_name: Cow::Borrowed("list"),
            trait_mask: traits::ADDER_TYPE
                | traits::CONTAINER_TYPE
                | traits::INDEXER_TYPE
                | traits::ITERABLE_TYPE
                | traits::SIZER_TYPE,
        }
    }

    /// Creates a new map type with the given key and value types.
    pub fn new_map_type(param: &'static [Cow<Type>; 2]) -> Type {
        Type {
            kind: Kind::Map,
            parameters: Cow::Borrowed(param),
            runtime_type_name: Cow::Borrowed("map"),
            trait_mask: traits::CONTAINER_TYPE
                | traits::INDEXER_TYPE
                | traits::ITERABLE_TYPE
                | traits::SIZER_TYPE,
        }
    }

    /// Creates a new unspecified type with the given name.
    pub const fn new_unspecified_type(name: &'static str) -> Type {
        Type {
            kind: Kind::Unspecified,
            parameters: Cow::Borrowed(&[]),
            runtime_type_name: Cow::Borrowed(name),
            trait_mask: 0,
        }
    }

    /// Creates a new opaque type with the given name.
    pub fn new_opaque_type<S: Into<Cow<'static, str>>>(name: S) -> Type {
        Type {
            kind: Kind::Opaque,
            parameters: Cow::Borrowed(&[]),
            runtime_type_name: name.into(),
            trait_mask: 0,
        }
    }

    /// Creates a new struct type with the given name.
    #[cfg(feature = "structs")]
    pub const fn new_struct_type(name: &'static str) -> Type {
        Type {
            kind: Kind::Struct,
            parameters: Cow::Borrowed(&[]),
            runtime_type_name: Cow::Borrowed(name),
            trait_mask: traits::FIELD_TESTER_TYPE | traits::INDEXER_TYPE,
        }
    }

    /// Creates a new struct type with the given owned name.
    #[cfg(feature = "structs")]
    pub const fn new_struct(name: String) -> Type {
        Type {
            kind: Kind::Struct,
            parameters: Cow::Borrowed(&[]),
            runtime_type_name: Cow::Owned(name),
            trait_mask: traits::FIELD_TESTER_TYPE | traits::INDEXER_TYPE,
        }
    }

    /// Returns the name of the type.
    pub fn name(&self) -> &str {
        &self.runtime_type_name
    }

    /// Returns true if the type has the given trait.
    pub fn has_trait(&self, t: u16) -> bool {
        self.trait_mask & t == t
    }
}

/// The CEL type name of a value, as `UnexpectedType` reports it.
///
/// This is `Type::name` of the value's own type. It is deliberately not
/// [`ValueType`](crate::objects::ValueType)'s `Display`, which spells the same
/// families `float`, `duration` and `null`.
pub(crate) fn type_name(value: &Value) -> String {
    match value {
        Value::Bool(_) => BOOL_TYPE.name().to_owned(),
        Value::Int(_) => INT_TYPE.name().to_owned(),
        Value::UInt(_) => UINT_TYPE.name().to_owned(),
        Value::Float(_) => DOUBLE_TYPE.name().to_owned(),
        Value::String(_) => STRING_TYPE.name().to_owned(),
        Value::Bytes(_) => BYTES_TYPE.name().to_owned(),
        Value::Null => NULL_TYPE.name().to_owned(),
        Value::List(_) => LIST_TYPE.name().to_owned(),
        Value::Map(_) => MAP_TYPE.name().to_owned(),
        #[cfg(feature = "chrono")]
        Value::Duration(_) => DURATION_TYPE.name().to_owned(),
        #[cfg(feature = "chrono")]
        Value::Timestamp(_) => TIMESTAMP_TYPE.name().to_owned(),
        #[cfg(feature = "structs")]
        Value::Struct(s) => s.name().to_owned(),
        Value::Opaque(o) => match o.downcast_ref::<OptionalValue>() {
            Some(_) => OPTIONAL_TYPE.name().to_owned(),
            None => o.runtime_type_name().to_owned(),
        },
    }
}

/// The value's own CEL type.
///
/// Returns it owned because the two named families build theirs per instance;
/// overload matching uses [`Type::is_assignable`], which needs no allocation.
pub(crate) fn type_of(value: &Value) -> Type {
    match value {
        Value::Bool(_) => BOOL_TYPE.to_owned(),
        Value::Int(_) => INT_TYPE.to_owned(),
        Value::UInt(_) => UINT_TYPE.to_owned(),
        Value::Float(_) => DOUBLE_TYPE.to_owned(),
        Value::String(_) => STRING_TYPE.to_owned(),
        Value::Bytes(_) => BYTES_TYPE.to_owned(),
        Value::Null => NULL_TYPE.to_owned(),
        Value::List(_) => LIST_TYPE.to_owned(),
        Value::Map(_) => MAP_TYPE.to_owned(),
        #[cfg(feature = "chrono")]
        Value::Duration(_) => DURATION_TYPE.to_owned(),
        #[cfg(feature = "chrono")]
        Value::Timestamp(_) => TIMESTAMP_TYPE.to_owned(),
        #[cfg(feature = "structs")]
        Value::Struct(s) => s.cel_type().to_owned(),
        Value::Opaque(o) => match o.downcast_ref::<OptionalValue>() {
            Some(_) => OPTIONAL_TYPE.to_owned(),
            None => Type::new_opaque_type(o.runtime_type_name().to_owned()),
        },
    }
}

/// The mismatch an overload reports when its argument is not the declared type.
pub(crate) fn type_error(got: &Value, want: &Type) -> ExecutionError {
    ExecutionError::UnexpectedType {
        got: type_name(got),
        want: want.name().to_owned(),
    }
}

fn noop(mut args: Vec<Value>) -> Result<Value, ExecutionError> {
    Ok(args.remove(0))
}
