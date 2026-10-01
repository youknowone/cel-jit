pub type TraitSet = u16;

/// ADDER_TYPE types provide a '+' operator overload.
pub const ADDER_TYPE: TraitSet = 1;

/// COMPARER_TYPE types support ordering comparisons '<', '<=', '>', '>='.
pub const COMPARER_TYPE: TraitSet = ADDER_TYPE << 1;

/// CONTAINER_TYPE types support 'in' operations.
pub const CONTAINER_TYPE: TraitSet = COMPARER_TYPE << 1;

/// DIVIDER_TYPE types support '/' operations.
pub const DIVIDER_TYPE: TraitSet = CONTAINER_TYPE << 1;

/// FIELD_TESTER_TYPE types support the detection of field value presence.
pub const FIELD_TESTER_TYPE: TraitSet = DIVIDER_TYPE << 1;

/// INDEXER_TYPE types support index access with dynamic values.
pub const INDEXER_TYPE: TraitSet = FIELD_TESTER_TYPE << 1;

/// ITERABLE_TYPE types can be iterated over in comprehensions.
pub const ITERABLE_TYPE: TraitSet = INDEXER_TYPE << 1;

/// ITERATOR_TYPE types support iterator semantics.
pub const ITERATOR_TYPE: TraitSet = ITERABLE_TYPE << 1;

/// MATCHER_TYPE types support pattern matching via 'matches' method.
pub const MATCHER_TYPE: TraitSet = ITERATOR_TYPE << 1;

/// MODDER_TYPE types support modulus operations '%'
pub const MODDER_TYPE: TraitSet = MATCHER_TYPE << 1;

/// MULTIPLIER_TYPE types support '*' operations.
pub const MULTIPLIER_TYPE: TraitSet = MODDER_TYPE << 1;

/// NEGATOR_TYPE types support either negation via '!' or '-'
pub const NEGATOR_TYPE: TraitSet = MULTIPLIER_TYPE << 1;

/// RECEIVER_TYPE types support dynamic dispatch to instance methods.
pub const RECEIVER_TYPE: TraitSet = NEGATOR_TYPE << 1;

/// SIZER_TYPE types support the size() method.
pub const SIZER_TYPE: TraitSet = RECEIVER_TYPE << 1;

/// SUBTRACTOR_TYPE types support '-' operations.
pub const SUBTRACTOR_TYPE: TraitSet = SIZER_TYPE << 1;

/// FOLDABLE_TYPE types support comprehensions v2 macros which iterate over (key, value) pairs.
pub const FOLDABLE_TYPE: TraitSet = SUBTRACTOR_TYPE << 1;

/// Backs every `size` overload: `size(x)` and `x.size()` for the four families
/// whose type carries [`SIZER_TYPE`].
pub(crate) mod adapter {
    use crate::common::types::type_name;
    use crate::objects::Value;
    use crate::ExecutionError;

    pub fn sizer_size(args: Vec<Value>) -> Result<Value, ExecutionError> {
        match crate::objects::value_len(&args[0]) {
            Some(size) => Ok(Value::Int(size)),
            None => Err(ExecutionError::UnexpectedType {
                got: type_name(&args[0]),
                want: "missing trait Sizer".to_owned(),
            }),
        }
    }
}
