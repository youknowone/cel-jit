//! The dispatch loop.
//!
//! [`cel_eval_loop`] is a free function taking the code object and the
//! context, not a method: the portal a tracing JIT splits a graph at is
//! derived from its argument list, and a `&self` receiver is neither a green
//! nor a red. [`Program::execute`] stays a thin wrapper over it.
//!
//! # Parity is the whole specification
//!
//! Every arm here is the corresponding arm of the tree walker
//! (`objects.rs`), reached by opcode instead of by `Expr` variant, and the
//! frozen oracle corpus holds the two to the same answers. Where the walker
//! computes something with a helper, this calls the same helper rather than
//! reimplementing it -- `value_index`, `value_field`, `binary_values` and the
//! rest are shared, so a fix to either evaluator is a fix to both.
//!
//! # The operand stack holds a builder, not only a value
//!
//! A list, map or struct literal is built incrementally: allocate, then one
//! append per element (see [`OpCode::NewList`]). Rebuilding the finished
//! [`Value`] after every element would make an *n*-element literal quadratic,
//! so the in-progress aggregate stays on the stack as an [`Operand`] variant
//! the append can mutate in place, and becomes a [`Value`] at the point
//! something pops it.
//!
//! [`Program::execute`]: crate::Program::execute

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::code::{CelCode, Handler};
use super::error::{CelErr, CelResult, ColdId, NameId};
use super::opcode::OpCode;
use crate::context::Context;
use crate::objects::{
    as_optional, binary_values, compare_values, optional_none, optional_of, value_contains,
    value_field, value_index, value_iter, value_key, value_negate, Key, Map,
};
use crate::{ExecutionError, Value};

use std::cmp::Ordering;

/// Run `code` in `ctx`.
///
/// The `Result` is reconstructed here, at the boundary: inside the loop an
/// error is a [`CelErr`], which is [`Copy`] and carries no allocation, because
/// `&&` and `||` absorb errors and so raise them on ordinary control flow.
pub fn cel_eval_loop(code: &CelCode, ctx: &Context) -> Result<Value, ExecutionError> {
    let mut vm = Vm::new(code, ctx);
    match vm.run() {
        Ok(value) => Ok(value),
        Err(err) => Err(vm.public_error(err)),
    }
}

/// One operand-stack entry.
///
/// Only the aggregate literals need anything but a [`Value`]; see the module
/// documentation.
enum Operand {
    Value(Value),
    List(Vec<Value>),
    /// Boxed because an inline [`HashMap`] is 48 bytes and would set the width
    /// of every entry on the stack, including the [`Operand::Value`] that
    /// almost all of them are. The box costs one allocation per map literal --
    /// paid only where a map literal appears -- and takes the entry from 56
    /// bytes to 32.
    ///
    /// `clippy::box_collection` argues the opposite -- that the map is on the
    /// heap already and the box only adds an allocation. That is the trade
    /// being made here on purpose, and the width it buys is asserted below, so
    /// the lint is off for this variant rather than followed.
    #[allow(clippy::box_collection)]
    Map(Box<HashMap<Key, Value>>),
    /// The `names` index of the message type, and the fields set so far. The
    /// type is checked when the struct is opened, so that a bad type name
    /// fails before the field expressions run, as it does in the walker.
    ///
    /// Without the `structs` feature that same check refuses every struct
    /// literal outright, so `open_struct` -- the only constructor -- never
    /// returns and this variant is genuinely unreachable in that build. The
    /// arms below still have to compile, which is what the attribute is for.
    #[cfg_attr(not(feature = "structs"), allow(dead_code))]
    Struct(NameId, BTreeMap<String, Value>),
}

/// The stack entry must stay narrow, because `Vm::new` sizes the operand stack
/// at `max_stack` entries and almost every one of them holds a bare [`Value`].
///
/// Measured: boxing `Map` alone takes this from 56 to 32. Boxing `Struct` as
/// well changes nothing -- its payload is 32 bytes and the discriminant fits in
/// the padding after `NameId` -- so a later variant wider than [`Value`] costs
/// 8 bytes on every entry and fails here rather than in a benchmark.
const _: () = {
    assert!(core::mem::size_of::<Operand>() == 32);
};

struct Vm<'a> {
    code: &'a CelCode,
    ctx: &'a Context<'a>,
    stack: Vec<Operand>,
    slots: Vec<Value>,
    /// One entry per `&&`/`||`, holding the left operand's outcome: its bool,
    /// or the error the handler absorbed for it.
    logic: Vec<CelResult<bool>>,
    /// Full errors for the cases [`CelErr`] cannot spell.
    ///
    /// Written only when an error is raised, and truncated again when one is
    /// absorbed, so a comprehension whose body errors on every iteration does
    /// not accumulate a table the size of the sequence.
    cold: Vec<ExecutionError>,
}

impl<'a> Vm<'a> {
    fn new(code: &'a CelCode, ctx: &'a Context<'a>) -> Self {
        Vm {
            code,
            ctx,
            stack: Vec::with_capacity(code.max_stack as usize),
            slots: vec![Value::Null; code.n_slots as usize],
            logic: vec![Err(CelErr::InternalError); code.n_logic as usize],
            cold: Vec::new(),
        }
    }

    // -- the error channel ------------------------------------------------

    /// Park a full error and return the [`CelErr`] that stands for it.
    fn park(&mut self, err: ExecutionError) -> CelErr {
        let id = u32::try_from(self.cold.len()).unwrap_or(u32::MAX);
        self.cold.push(err);
        CelErr::Cold(ColdId(id))
    }

    /// Discard a parked error that a merge has decided to throw away.
    ///
    /// Called at the merge, not when the handler catches: the recorded error
    /// is still live between those two points, because the merge may raise it
    /// after all. Sound because the merges nest -- anything the right operand
    /// parked has been resolved by its own merge before this one runs -- so a
    /// discarded `Cold` is always the last entry.
    fn unpark(&mut self, err: CelErr) {
        if let CelErr::Cold(ColdId(id)) = err {
            if id as usize + 1 == self.cold.len() {
                self.cold.pop();
            }
        }
    }

    /// Reconstruct the public error.
    ///
    /// Exhaustive over [`CelErr`], so a variant added without a public
    /// counterpart is a compile error here rather than a silent
    /// `InternalError` at run time.
    #[allow(deprecated)]
    fn public_error(&self, err: CelErr) -> ExecutionError {
        let name = |id: NameId| self.code.name(id).unwrap_or("?").to_string();
        // The operator name the public error carries is a property of the
        // opcode, which is why the compact form stores the opcode.
        let operator = |err: CelErr| match err.operator() {
            Some(OpCode::Add) => "add",
            Some(OpCode::Sub) => "sub",
            Some(OpCode::Mul) => "mul",
            Some(OpCode::Div) => "div",
            Some(OpCode::Mod) => "rem",
            Some(OpCode::Negate) => "negate",
            _ => "?",
        };
        match err {
            CelErr::Cold(ColdId(id)) => self
                .cold
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| ExecutionError::InternalError("cold error lost".to_string())),
            CelErr::NoSuchOverload => ExecutionError::NoSuchOverload,
            CelErr::MissingArgumentOrTarget => ExecutionError::MissingArgumentOrTarget,
            CelErr::DivisionByZero => ExecutionError::DivisionByZero(Value::Null),
            CelErr::RemainderByZero => ExecutionError::RemainderByZero(Value::Null),
            CelErr::IndexOutOfBounds => ExecutionError::IndexOutOfBounds(Value::Null),
            CelErr::ValuesNotComparable => {
                ExecutionError::ValuesNotComparable(Value::Null, Value::Null)
            }
            CelErr::UnsupportedKeyType => ExecutionError::unsupported_key_type(Value::Null),
            CelErr::UnsupportedTargetType => ExecutionError::UnsupportedTargetType {
                target: Value::Null,
            },
            CelErr::UnsupportedIndex => ExecutionError::UnsupportedIndex(Value::Null, Value::Null),
            CelErr::InternalError => ExecutionError::InternalError("internal VM error".to_string()),
            CelErr::NoSuchKey(id) => ExecutionError::NoSuchKey(Arc::new(name(id))),
            CelErr::UndeclaredReference(id) => {
                ExecutionError::UndeclaredReference(Arc::new(name(id)))
            }
            CelErr::NotSupportedAsMethod(id) => ExecutionError::NotSupportedAsMethod {
                method: name(id),
                target: Value::Null,
            },
            CelErr::UnexpectedType { got, want } => ExecutionError::UnexpectedType {
                got: name(got),
                want: name(want),
            },
            CelErr::Overflow(_) => {
                ExecutionError::Overflow(operator(err), Value::Null, Value::Null)
            }
            CelErr::UnsupportedUnaryOperator(_) => {
                ExecutionError::UnsupportedUnaryOperator(operator(err), Value::Null)
            }
            CelErr::UnsupportedBinaryOperator(_) => {
                ExecutionError::UnsupportedBinaryOperator(operator(err), Value::Null, Value::Null)
            }
            CelErr::InvalidArgumentCount { expected, actual } => {
                ExecutionError::InvalidArgumentCount {
                    expected: expected as usize,
                    actual: actual as usize,
                }
            }
        }
    }

    // -- the operand stack --------------------------------------------------

    fn push(&mut self, value: Value) {
        self.stack.push(Operand::Value(value));
    }

    /// Pop one operand, finishing an aggregate that was still being built.
    fn pop(&mut self) -> CelResult<Value> {
        let operand = self.stack.pop().ok_or(CelErr::InternalError)?;
        self.finish(operand)
    }

    fn finish(&mut self, operand: Operand) -> CelResult<Value> {
        match operand {
            Operand::Value(value) => Ok(value),
            Operand::List(items) => Ok(Value::list(items)),
            Operand::Map(entries) => Ok(Value::Map(Map::object(Arc::new(*entries)))),
            Operand::Struct(name, fields) => self.close_struct(name, fields),
        }
    }

    #[cfg(feature = "structs")]
    fn close_struct(&mut self, name: NameId, fields: BTreeMap<String, Value>) -> CelResult<Value> {
        let type_name = self.code.name(name).unwrap_or("?").to_string();
        let built = self
            .ctx
            .env()
            .find_struct(&type_name)
            .ok_or(CelErr::UnexpectedType {
                got: name,
                want: NameId(u32::MAX),
            })?
            .new_struct(fields);
        match built {
            Ok(value) => Ok(Value::Struct(Arc::new(value))),
            Err(err) => Err(self.park(err)),
        }
    }

    /// The refusal a build without the `structs` feature owes a struct
    /// literal. Raised where the literal is *opened*, so nothing inside it
    /// runs first; `close_struct` keeps it only because `finish` is total over
    /// [`Operand`].
    #[cfg(not(feature = "structs"))]
    fn no_structs_feature(&mut self, name: NameId) -> CelErr {
        let type_name = self.code.name(name).unwrap_or("?");
        self.park(ExecutionError::InternalError(format!(
            "Found struct {type_name}, feature not enabled!"
        )))
    }

    #[cfg(not(feature = "structs"))]
    fn close_struct(&mut self, name: NameId, _fields: BTreeMap<String, Value>) -> CelResult<Value> {
        Err(self.no_structs_feature(name))
    }

    /// The `n` topmost operands, in the order they were pushed.
    fn pop_n(&mut self, n: usize) -> CelResult<Vec<Value>> {
        let mut args = Vec::with_capacity(n);
        for _ in 0..n {
            args.push(self.pop()?);
        }
        args.reverse();
        Ok(args)
    }

    fn list_mut(&mut self) -> CelResult<&mut Vec<Value>> {
        match self.stack.last_mut() {
            Some(Operand::List(items)) => Ok(items),
            _ => Err(CelErr::InternalError),
        }
    }

    fn map_mut(&mut self) -> CelResult<&mut HashMap<Key, Value>> {
        match self.stack.last_mut() {
            Some(Operand::Map(entries)) => Ok(&mut **entries),
            _ => Err(CelErr::InternalError),
        }
    }

    fn struct_mut(&mut self) -> CelResult<&mut BTreeMap<String, Value>> {
        match self.stack.last_mut() {
            Some(Operand::Struct(_, fields)) => Ok(fields),
            _ => Err(CelErr::InternalError),
        }
    }

    // -- the loop -----------------------------------------------------------

    fn run(&mut self) -> CelResult<Value> {
        let mut pc = 0u32;
        loop {
            let (op, operands) = self.code.decode(pc).ok_or(CelErr::InternalError)?;
            let operands: [u32; 3] = [
                operands.first().copied().unwrap_or(0),
                operands.get(1).copied().unwrap_or(0),
                operands.get(2).copied().unwrap_or(0),
            ];
            let next = pc + op.width();

            match self.step(op, operands, pc, next) {
                Ok(Step::Next) => pc = next,
                Ok(Step::Jump(target)) => pc = target,
                Ok(Step::Return(value)) => return Ok(value),
                Err(err) => pc = self.unwind(err, pc)?,
            }
        }
    }

    /// Deliver `err` to the handler covering `pc`, or give up.
    ///
    /// Absorption is the whole reason a handler table exists: CEL's `&&` and
    /// `||` are commutative over errors, so an error raised inside the left
    /// operand is *recorded* in that operator's logic slot and the right
    /// operand still runs. Nothing else in the language catches.
    fn unwind(&mut self, err: CelErr, pc: u32) -> CelResult<u32> {
        let Some(&Handler {
            land, logic, depth, ..
        }) = self.code.handler_for(pc)
        else {
            return Err(err);
        };
        *self
            .logic
            .get_mut(logic as usize)
            .ok_or(CelErr::InternalError)? = Err(err);
        self.stack.truncate(depth as usize);
        Ok(land)
    }

    fn step(&mut self, op: OpCode, operands: [u32; 3], pc: u32, next: u32) -> CelResult<Step> {
        let [a, b, c] = operands;
        match op {
            // -- loads --------------------------------------------------
            OpCode::LoadConst => {
                let value = self.code.konst(a).ok_or(CelErr::InternalError)?.clone();
                self.push(value);
            }
            OpCode::LoadVar => {
                let name = self.name(a)?;
                let value = self
                    .ctx
                    .get_variable(name)
                    .ok_or(CelErr::UndeclaredReference(NameId(a)))?;
                self.push(value);
            }
            OpCode::LoadLocal => {
                let value = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                self.push(value);
            }
            OpCode::StoreLocal => {
                let value = self.pop()?;
                *self
                    .slots
                    .get_mut(a as usize)
                    .ok_or(CelErr::InternalError)? = value;
            }

            // -- selection ----------------------------------------------
            OpCode::GetField => {
                let operand = self.pop()?;
                let field = self.name(a)?;
                let value = value_field(&operand, field).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::HasField => {
                let operand = self.pop()?;
                let field = self.name(a)?;
                let value = has_field(&operand, field).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::Index | OpCode::OptIndex => {
                let key = self.pop()?;
                let operand = self.pop()?;
                let value = self.index(operand, key, op == OpCode::OptIndex)?;
                self.push(value);
            }
            OpCode::OptSelect => {
                let operand = self.pop()?;
                let field = Value::String(Arc::new(self.name(a)?.to_string()));
                let value = self.opt_select(operand, field)?;
                self.push(value);
            }

            // -- aggregates ----------------------------------------------
            OpCode::NewList => self.stack.push(Operand::List(Vec::new())),
            OpCode::ListAppend => {
                let value = self.pop()?;
                self.list_mut()?.push(value);
            }
            OpCode::ListAppendOptional => {
                let value = self.pop()?;
                match optional_inner(&value) {
                    OptView::Empty => {}
                    OptView::Present(inner) => self.list_mut()?.push(inner),
                    OptView::Plain => self.list_mut()?.push(value),
                }
            }
            OpCode::NewMap => self.stack.push(Operand::Map(Box::default())),
            OpCode::MapInsert | OpCode::MapInsertOptional => {
                let value = self.pop()?;
                let key = self.pop()?;
                let key = value_key(key).map_err(|e| self.park(e))?;
                if op == OpCode::MapInsert {
                    self.map_mut()?.insert(key, value);
                } else {
                    match optional_inner(&value) {
                        OptView::Empty => {}
                        OptView::Present(inner) => {
                            self.map_mut()?.insert(key, inner);
                        }
                        OptView::Plain => {
                            self.map_mut()?.insert(key, value);
                        }
                    }
                }
            }
            OpCode::NewStruct => {
                self.open_struct(NameId(a))?;
            }
            OpCode::StructSet | OpCode::StructSetOptional => {
                let value = self.pop()?;
                let field = self.name(a)?.to_string();
                if op == OpCode::StructSet {
                    self.struct_mut()?.insert(field, value);
                } else {
                    match optional_inner(&value) {
                        OptView::Empty => {}
                        OptView::Present(inner) => {
                            self.struct_mut()?.insert(field, inner);
                        }
                        OptView::Plain => {
                            self.struct_mut()?.insert(field, value);
                        }
                    }
                }
            }

            // -- binary operators -----------------------------------------
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                let name = match op {
                    OpCode::Add => "add",
                    OpCode::Sub => "sub",
                    OpCode::Mul => "mul",
                    OpCode::Div => "div",
                    _ => "rem",
                };
                let value = binary_values(name, lhs, rhs).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::Equals | OpCode::NotEquals => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                self.push(Value::Bool((lhs == rhs) == (op == OpCode::Equals)));
            }
            OpCode::Less | OpCode::LessEquals | OpCode::Greater | OpCode::GreaterEquals => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                let accept: fn(Ordering) -> bool = match op {
                    OpCode::Less => |o| o == Ordering::Less,
                    OpCode::LessEquals => |o| o != Ordering::Greater,
                    OpCode::Greater => |o| o == Ordering::Greater,
                    _ => |o| o != Ordering::Less,
                };
                let value = compare_values(lhs, rhs, accept).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::In => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                let value = value_contains(&rhs, &lhs).map_err(|e| self.park(e))?;
                self.push(Value::Bool(value));
            }

            // -- unary operators -------------------------------------------
            OpCode::Not => {
                let value = self.pop()?;
                match value {
                    Value::Bool(b) => self.push(Value::Bool(!b)),
                    _ => return Err(CelErr::NoSuchOverload),
                }
            }
            OpCode::Negate => {
                let value = self.pop()?;
                let value = value_negate(value).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::NotStrictlyFalse => {
                let value = self.pop()?;
                self.push(Value::Bool(as_bool(&value).unwrap_or(true)));
            }

            // -- calls -------------------------------------------------------
            OpCode::CallHost => {
                let args = self.pop_n(b as usize)?;
                let value = self.call_global(NameId(a), args)?;
                self.push(value);
            }
            OpCode::CallMethod => {
                let target = self.pop()?;
                let args = self.pop_n(b as usize)?;
                let value = self.call_member(NameId(a), target, args)?;
                self.push(value);
            }
            OpCode::CallQualified => {
                let args = self.pop_n(b as usize)?;
                // A miss leaves the stack as it was, because the receiver
                // path that follows re-pushes the same arguments.
                if let Some(value) = self.call_qualified(NameId(a), args)? {
                    self.push(value);
                    return Ok(Step::Jump(c));
                }
            }

            // -- iteration ----------------------------------------------------
            OpCode::IterElems => {
                let value = self.pop()?;
                let items = value_iter(&value).map_err(|e| self.park(e))?;
                self.push(Value::list(items));
            }
            OpCode::IterKeys => {
                let value = self.pop()?;
                let items = iter_keys(&value).map_err(|e| self.park(e))?;
                self.push(Value::list(items));
            }
            OpCode::IterLen => {
                // Read in place. The length is the only thing wanted out of
                // the slot, and taking it through the stack would clone the
                // whole `Value` -- an atomic refcount pair for a list -- once
                // per element of the loop that emits this.
                let len = match self.slots.get(a as usize).ok_or(CelErr::InternalError)? {
                    Value::List(list) => list.len() as i64,
                    _ => return Err(CelErr::InternalError),
                };
                self.push(Value::Int(len));
            }
            OpCode::IterAt => {
                // Read in place, as `IterLen` does. The borrows end with the
                // block, so the error path below can still park on `self`.
                let element = {
                    let sequence = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                    let index = self.slots.get(b as usize).ok_or(CelErr::InternalError)?;
                    value_index(sequence, index)
                };
                let value = element.map_err(|e| self.park(e))?;
                self.push(value);
            }

            // -- control flow ---------------------------------------------------
            OpCode::Jump => return Ok(Step::Jump(a)),
            OpCode::JumpIfOptNone => {
                let empty = match self.stack.last() {
                    Some(Operand::Value(value)) => {
                        matches!(optional_inner(value), OptView::Empty)
                    }
                    _ => false,
                };
                if empty {
                    return Ok(Step::Jump(a));
                }
            }
            OpCode::JumpIfFalse | OpCode::JumpIfTrue => {
                let value = self.pop()?;
                let taken = as_bool(&value)? == (op == OpCode::JumpIfTrue);
                if taken {
                    return Ok(Step::Jump(a));
                }
            }
            OpCode::And | OpCode::Or => {
                let value = self.pop()?;
                let short = op == OpCode::Or;
                let outcome = as_bool(&value);
                *self
                    .logic
                    .get_mut(a as usize)
                    .ok_or(CelErr::InternalError)? = outcome;
                if outcome == Ok(short) {
                    self.push(Value::Bool(short));
                    return Ok(Step::Jump(b));
                }
            }
            OpCode::AndMerge | OpCode::OrMerge => {
                let right = self.pop()?;
                let left = *self.logic.get(a as usize).ok_or(CelErr::InternalError)?;
                let value = merge(left, &right, op == OpCode::OrMerge)?;
                // The left-hand error survived only to be weighed here, and
                // it has just lost. Nothing can observe it now.
                if let Err(absorbed) = left {
                    self.unpark(absorbed);
                }
                self.push(Value::Bool(value));
            }
            OpCode::Return => return Ok(Step::Return(self.pop()?)),
        }
        let _ = (pc, next);
        Ok(Step::Next)
    }

    fn name(&self, id: u32) -> CelResult<&'a str> {
        self.code.name(NameId(id)).ok_or(CelErr::InternalError)
    }

    // -- the arms that are more than one expression -------------------------

    /// `container[key]`, which also unwraps an optional container.
    ///
    /// The walker reaches `_[_]` and `_[?_]` through one arm and recovers
    /// which it is from the operator's name; here the two are separate opcodes
    /// decided at compile time, and `is_optional` is still a run-time fact
    /// because an optional *container* makes a plain index optional too.
    fn index(&mut self, container: Value, key: Value, mut is_optional: bool) -> CelResult<Value> {
        let container = match optional_inner(&container) {
            OptView::Plain => container,
            OptView::Empty => return Ok(optional_none()),
            OptView::Present(inner) => {
                is_optional = true;
                inner
            }
        };
        let result = value_index(&container, &key);
        if is_optional {
            return Ok(match result {
                Ok(value) => optional_of(value),
                Err(_) => optional_none(),
            });
        }
        result.map_err(|e| self.park(e))
    }

    /// `a?.b`.
    ///
    /// The nested-optional shape on a miss is the walker's, mirrored rather
    /// than corrected: `Optional::map` keeps the outer `Some` and substitutes
    /// `optional.none` for the missing field.
    fn opt_select(&mut self, operand: Value, field: Value) -> CelResult<Value> {
        Ok(match optional_inner(&operand) {
            OptView::Empty => optional_none(),
            OptView::Present(inner) => {
                optional_of(value_index(&inner, &field).unwrap_or_else(|_| optional_none()))
            }
            OptView::Plain => optional_of(value_index(&operand, &field).map_err(|e| self.park(e))?),
        })
    }

    #[cfg(feature = "structs")]
    fn open_struct(&mut self, name: NameId) -> CelResult<()> {
        let type_name = self.code.name(name).unwrap_or("?");
        if self.ctx.env().find_struct(type_name).is_none() {
            // `want` is not a program name, so the detail goes cold.
            let err = ExecutionError::UnexpectedType {
                got: type_name.to_owned(),
                want: "known struct".to_owned(),
            };
            return Err(self.park(err));
        }
        self.stack.push(Operand::Struct(name, BTreeMap::new()));
        Ok(())
    }

    /// Without the feature there is no struct to build, and the walker says so
    /// on reaching the node -- before any field expression runs. Deferring the
    /// refusal to the close would report a field's error instead:
    /// `cel.MyStruct { x: 1 / 0 }` answered `DivisionByZero`.
    #[cfg(not(feature = "structs"))]
    fn open_struct(&mut self, name: NameId) -> CelResult<()> {
        Err(self.no_structs_feature(name))
    }

    fn call_global(&mut self, name: NameId, args: Vec<Value>) -> CelResult<Value> {
        let func_name = self.name(name.0)?;
        if let Some(op) = self.ctx.env().find_overload(func_name, &args) {
            return op(args).map_err(|e| self.park(e));
        }
        let func = self
            .ctx
            .get_function(func_name)
            .ok_or(CelErr::UndeclaredReference(name))?;
        let mut fctx = crate::FunctionContext::new(func_name, None, self.ctx, args);
        (func)(&mut fctx).map_err(|e| self.park(e))
    }

    /// The namespaced probe: `math.max(1, 2)`.
    ///
    /// `None` is a miss, and a miss must leave no trace -- the receiver has
    /// not been evaluated yet, because `optional.of(1)` names no variable
    /// `optional`.
    fn call_qualified(&mut self, joined: NameId, args: Vec<Value>) -> CelResult<Option<Value>> {
        let name = self.name(joined.0)?;
        if let Some(op) = self.ctx.env().find_overload(name, &args) {
            return op(args).map(Some).map_err(|e| self.park(e));
        }
        let Some(func) = self.ctx.get_function(name) else {
            for arg in args {
                self.push(arg);
            }
            return Ok(None);
        };
        let mut fctx = crate::FunctionContext::new(name, None, self.ctx, args);
        (func)(&mut fctx).map(Some).map_err(|e| self.park(e))
    }

    fn call_member(&mut self, name: NameId, target: Value, args: Vec<Value>) -> CelResult<Value> {
        let func_name = self.name(name.0)?;
        let mut with_target = args;
        with_target.insert(0, target);
        if let Some(op) = self.ctx.env().find_member_overload(func_name, &with_target) {
            return op(with_target).map_err(|e| self.park(e));
        }
        let target = with_target.remove(0);
        let func = self
            .ctx
            .get_function(func_name)
            .ok_or(CelErr::UndeclaredReference(name))?;
        let mut fctx = crate::FunctionContext::new(func_name, Some(target), self.ctx, with_target);
        (func)(&mut fctx).map_err(|e| self.park(e))
    }
}

/// What one instruction decided.
enum Step {
    Next,
    Jump(u32),
    Return(Value),
}

/// Whether a value is an optional, and what is in it.
enum OptView {
    /// Not an optional at all.
    Plain,
    /// `optional.none`.
    Empty,
    Present(Value),
}

fn optional_inner(value: &Value) -> OptView {
    match as_optional(value) {
        None => OptView::Plain,
        Some(opt) => match opt.value() {
            None => OptView::Empty,
            Some(inner) => OptView::Present(inner.clone()),
        },
    }
}

/// A value used where CEL wants a bool.
///
/// A non-bool is an overload failure rather than a coercion, which is what
/// makes `1 && false` an *absorbed* error and not a truthiness test.
fn as_bool(value: &Value) -> CelResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        _ => Err(CelErr::NoSuchOverload),
    }
}

/// `has(x.y)`.
fn has_field(operand: &Value, field: &str) -> Result<Value, ExecutionError> {
    match operand {
        Value::Map(map) => Ok(Value::Bool(
            map.contains_key(&crate::objects::KeyRef::String(field)),
        )),
        #[cfg(feature = "structs")]
        Value::Struct(_) => Ok(Value::Bool(value_field(operand, field).is_ok())),
        _ => value_field(operand, field),
    }
}

/// What a two-variable comprehension binds to its first variable: a list's
/// indices, or a map's keys.
fn iter_keys(value: &Value) -> Result<Vec<Value>, ExecutionError> {
    match value {
        Value::List(list) => Ok((0..list.len()).map(|i| Value::Int(i as i64)).collect()),
        _ => value_iter(value),
    }
}

/// Combine a short-circuit operator's two sides.
///
/// `left` is the recorded outcome of the left operand -- its bool, or the
/// error a handler absorbed for it. The asymmetry is CEL's: a *recorded* error
/// is discarded when the right side decides the result, and raised otherwise.
fn merge(left: CelResult<bool>, right: &Value, is_or: bool) -> CelResult<bool> {
    let right = match right {
        Value::Bool(b) => Some(*b),
        _ => None,
    };
    match (left, right) {
        (Ok(left), Some(right)) => Ok(if is_or { left || right } else { left && right }),
        (Err(_), Some(decides)) if decides == is_or => Ok(is_or),
        (left, _) => Err(left.err().unwrap_or(CelErr::NoSuchOverload)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ast::{Expr, IdedExpr};
    use crate::parser::Parser;
    use crate::vm::compile;

    fn parse(source: &str) -> IdedExpr {
        Parser::default()
            .parse(source)
            .unwrap_or_else(|e| panic!("parse {source}: {e:?}"))
    }

    fn run(source: &str, ctx: &Context) -> Result<Value, ExecutionError> {
        let code = compile(&parse(source)).unwrap_or_else(|e| panic!("compile {source}: {e}"));
        cel_eval_loop(&code, ctx)
    }

    /// An error absorbed by `&&` is discarded, not accumulated.
    ///
    /// The corpus cannot see this -- the answer is the same either way -- but a
    /// comprehension absorbs once per element, so a table that only grew would
    /// be a leak proportional to the input.
    #[test]
    fn an_absorbing_loop_does_not_accumulate_parked_errors() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value(
            "xs",
            Value::list((0..64).map(Value::Int).collect::<Vec<_>>()),
        );

        // Two things this expression has to get right to be able to fail.
        // The absorbed error must be one that is actually parked --
        // `UndeclaredReference` is minted in its compact form and never
        // reaches the table -- and the loop must not short-circuit, which
        // `all` does the moment the accumulator turns false.
        let source = "xs.map(x, (1 / 0 == 1) && false)";
        let code = compile(&parse(source)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);
        let result = vm.run().expect("every element absorbs its error");
        assert_eq!(
            result,
            Value::list(vec![Value::Bool(false); 64]),
            "the loop has to actually run 64 times"
        );
        assert!(
            vm.cold.len() <= 1,
            "64 absorbed errors left {} parked",
            vm.cold.len()
        );
    }

    /// The absorbed error is still live between the catch and the merge,
    /// because the merge may raise it after all.
    #[test]
    fn an_absorbed_error_survives_to_be_raised_by_the_merge() {
        let ctx = Context::default();
        assert_eq!(
            run("undefined_name && true", &ctx),
            Err(ExecutionError::UndeclaredReference(Arc::new(
                "undefined_name".to_string()
            )))
        );
    }

    /// Catching truncates the operand stack, so a literal that was half-built
    /// when the error landed does not leave its elements behind.
    #[test]
    fn catching_discards_a_half_built_aggregate() {
        let ctx = Context::default();
        let source = "([1, 2, undefined_name][0] == 1) && false";
        let code = compile(&parse(source)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);
        assert_eq!(vm.run(), Ok(Value::Bool(false)));
        assert!(vm.stack.is_empty(), "the operand stack was left dirty");
    }

    /// The parser never builds a two-variable comprehension and the tree
    /// walker ignores `iter_var2` outright, so the corpus cannot reach this
    /// arm from either side.
    #[test]
    fn a_two_variable_comprehension_binds_the_key_and_the_element() {
        let Expr::Comprehension(base) = parse("xs.all(k, k >= 0)").expr else {
            panic!("the macro expands to a comprehension");
        };
        let mut comp = *base;
        comp.iter_var2 = Some("v".to_string());
        // `all` expands to `@result && <predicate>`, and `@result` is not a
        // name CEL can parse, so the predicate is swapped in place rather than
        // the whole step rewritten. It is true only if the second variable
        // really is the element at the first one's index.
        let Expr::Call(step) = &mut comp.loop_step.expr else {
            panic!("`all` steps through `&&`");
        };
        step.args[1] = parse("v == xs[k]");

        let code = compile(&IdedExpr {
            id: 0,
            expr: Expr::Comprehension(Box::new(comp)),
        })
        .expect("compiles");

        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", Value::list(vec![Value::Int(7), Value::Int(8)]));
        assert_eq!(cel_eval_loop(&code, &ctx), Ok(Value::Bool(true)));
    }
}
