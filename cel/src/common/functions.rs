use crate::common::traits::TraitSet;
use crate::objects::Value;
use crate::ExecutionError;

#[allow(dead_code)]
pub struct Overload {
    operator: String,
    operand_trait: TraitSet,
    op: Function,
}

pub type Function = fn(Vec<Value>) -> Result<Value, ExecutionError>;
