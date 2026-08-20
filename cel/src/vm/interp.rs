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
    /// Behind a pointer because an inline [`HashMap`] is 48 bytes and would
    /// set the width of every entry on the stack, including the
    /// [`Operand::Value`] that almost all of them are; the indirection takes
    /// the entry from 56 bytes to 32.
    ///
    /// The pointer is the [`Arc`] the finished [`Map`] holds, not a [`Box`],
    /// so the table is built where it lands. Both are 8 bytes here, but a
    /// `Box` is a different allocation from the one [`Map::object`] needs:
    /// closing the literal then had to allocate the `Arc`, move the 48-byte
    /// table into it and free the box -- one allocation and one move per map
    /// literal, spent only on handing the table over. `finish` now passes the
    /// same pointer through.
    ///
    /// An in-progress table is never shared -- the operand holds the only
    /// reference until `finish` gives it away -- so `Arc::get_mut` answers
    /// every insert; see [`Vm::map_mut`].
    Map(Arc<HashMap<Key, Value>>),
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
/// Measured: putting `Map` behind a pointer alone takes this from 56 to 32.
/// Doing the same to `Struct` changes nothing -- its payload is 32 bytes and
/// the discriminant fits in the padding after `NameId` -- so a later variant
/// wider than [`Value`] costs 8 bytes on every entry and fails here rather
/// than in a benchmark.
const _: () = {
    assert!(core::mem::size_of::<Operand>() == 32);
};

/// The four growable buffers a run needs, kept across runs so that evaluating
/// a program does not allocate an activation record.
///
/// Sizing them per run is what an evaluation of `1 + 1` used to pay: `stack` is
/// at least one entry for any program that compiles, so a scalar expression
/// cost one malloc and one free, plus a second for `slots` where the program
/// has a comprehension and a third for `logic` where it has `&&` or `||`. The
/// tree walker's floor is zero — its operands are recursion locals — so this
/// was a cost the evaluator paid for being a loop rather than for doing work.
///
/// Only the CAPACITY is worth keeping. Every buffer is emptied when it is
/// returned, so nothing here holds a [`Value`] between runs; what survives is
/// the allocation, sized to the largest program this thread has evaluated.
#[derive(Default)]
struct Scratch {
    stack: Vec<Operand>,
    slots: Vec<Value>,
    logic: Vec<CelResult<bool>>,
    cold: Vec<ExecutionError>,
}

impl Scratch {
    /// Drop everything held, keeping the allocations.
    ///
    /// Called when the buffers go back to the pool rather than when they come
    /// out, so a run that ended in an error — or in a panic, since this is
    /// reached from [`Vm`]'s [`Drop`] — cannot leave a large list rooted until
    /// the next evaluation on this thread happens to overwrite it.
    fn release(&mut self) {
        self.stack.clear();
        self.slots.clear();
        self.logic.clear();
        self.cold.clear();
    }
}

std::thread_local! {
    /// This thread's idle buffers, or `None` while a run holds them.
    ///
    /// Held by VALUE and moved in and out, which is what makes re-entry
    /// correct rather than merely unlikely: a host function registered in the
    /// [`Context`] may call [`crate::Program::execute`] again from inside an
    /// evaluation, and the nested run finds this slot empty and builds its own
    /// buffers. Lending a `&mut` to a shared buffer instead would alias, and no
    /// arrangement of flags recovers from that — the outer run's operand stack
    /// is live across the call.
    ///
    /// Nesting therefore pays one set of allocations, and it is the nesting
    /// caller that pays. On the way out the innermost run to finish is the
    /// first to store, so the outermost run's buffers — the widest, and the
    /// ones a subsequent top-level call wants — are what the slot ends up
    /// holding.
    ///
    /// Reached through `try_with` on both sides, never `with`. `Scratch` has a
    /// destructor, so this slot can already have been torn down by the time a
    /// thread-local of the caller's runs an evaluation of its own from its own
    /// destructor. `with` would panic there, and a panic on the way out of
    /// [`Vm`] would be a panic in a destructor.
    static SCRATCH: std::cell::Cell<Option<Scratch>> = const { std::cell::Cell::new(None) };
}

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
    /// Arguments a missed [`OpCode::CallQualified`] popped, held for the
    /// [`OpCode::CallMethod`] the compiler emitted right after it.
    ///
    /// The pair is one call. The probe has to pop its arguments to ask
    /// `find_overload` about them, and the receiver path then wants the same
    /// values in the same order; re-pushing them for `pop_n` to rebuild is a
    /// second `Vec` per member call on an identifier receiver, which is an
    /// allocation the walker never pays -- it resolves its arguments once and
    /// lends the probe a slice.
    ///
    /// Live only across the `LoadVar` that loads the receiver, and cleared on
    /// both ways out: taken by [`OpCode::CallMethod`], and dropped by
    /// [`Vm::unwind`], which is where that load's error goes when a `&&`/`||`
    /// absorbs it and the method call never runs.
    pending_args: Option<Vec<Value>>,
}

/// Return the buffers to this thread's pool.
///
/// A [`Drop`] impl rather than a call at the end of [`cel_eval_loop`], because
/// the buffers have to come back on EVERY way out — the `Ok`, the `Err`, and a
/// panic from a host function called mid-evaluation. Taking them out and
/// putting them back is also what keeps re-entry sound: the slot is empty for
/// exactly as long as a run holds it.
impl Drop for Vm<'_> {
    fn drop(&mut self) {
        // `mem::take` on a `Vec` leaves a dangling-free empty one and allocates
        // nothing, which is the only way to move fields out of a type that
        // implements `Drop`.
        let mut scratch = Scratch {
            stack: std::mem::take(&mut self.stack),
            slots: std::mem::take(&mut self.slots),
            logic: std::mem::take(&mut self.logic),
            cold: std::mem::take(&mut self.cold),
        };
        scratch.release();
        // Dropped rather than pooled where the slot is already gone; see
        // `SCRATCH`.
        let _ = SCRATCH.try_with(|slot| slot.set(Some(scratch)));
    }
}

impl<'a> Vm<'a> {
    /// Borrow this thread's buffers and size them for `code`.
    ///
    /// Every buffer arrives empty — [`Scratch::release`] is what put it back —
    /// so this establishes the lengths the loop indexes into and asks the
    /// operand stack for the depth the compiler proved it needs. On a thread
    /// that has evaluated anything before, all four of those are already
    /// satisfied and none of them allocates.
    fn new(code: &'a CelCode, ctx: &'a Context<'a>) -> Self {
        let mut scratch = SCRATCH
            .try_with(std::cell::Cell::take)
            .ok()
            .flatten()
            .unwrap_or_default();
        scratch.stack.reserve(code.max_stack as usize);
        scratch.slots.resize(code.n_slots as usize, Value::Null);
        scratch
            .logic
            .resize(code.n_logic as usize, Err(CelErr::InternalError));
        Vm {
            code,
            ctx,
            stack: scratch.stack,
            slots: scratch.slots,
            logic: scratch.logic,
            cold: scratch.cold,
            pending_args: None,
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
            Operand::Map(entries) => Ok(Value::Map(Map::object(entries))),
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
        self.pop_n_spare(n, 0)
    }

    /// [`Vm::pop_n`] with room for the receiver [`Vm::call_member`] prepends.
    ///
    /// `call_member` builds the receiver-first vector that
    /// `Env::find_member_overload` matches on by `insert`ing the target at
    /// index 0. A vector sized to exactly the arity has `len == capacity`, so
    /// that insert reallocates and copies -- once per member call, at every
    /// arity from 1 upward.
    ///
    /// Arity 0 is excluded, and the guard is what makes this an improvement
    /// rather than a trade. `Vec::with_capacity(0)` allocates NOTHING;
    /// `Vec::with_capacity(1)` allocates. So at arity 0 the reservation buys
    /// nothing even on the member path -- unreserved, `insert` grows the empty
    /// vector once; reserved, the reservation IS that one allocation. A wash
    /// there, and a pure loss on the path below.
    ///
    /// Both opcodes that can reach `call_member` pop through here.
    /// `CallMethod` is the obvious one; `CallQualified` is the other, because
    /// a miss hands its vector on rather than re-pushing it, and the compiler
    /// only ever emits the probe ahead of a `CallMethod` of the same arity
    /// (`compile::tests::a_probe_and_its_member_call_agree_on_arity`). The
    /// probe pops BEFORE it knows hit from miss, and a HIT never fills the
    /// spare slot -- so above arity 0, where the vector is allocated either
    /// way, widening it costs nothing, but at arity 0 it turns a call that
    /// allocated no argument vector at all into one that does. Nullary
    /// namespaced overloads are ordinary: `optional.none` is one, and so is
    /// any `ctx.add_function("ns.f", || ..)`.
    ///
    /// The spare slot stays a property of these sites rather than of `pop_n`,
    /// which [`OpCode::CallHost`] also uses and which has no receiver.
    fn pop_n_for_member(&mut self, n: usize) -> CelResult<Vec<Value>> {
        self.pop_n_spare(n, usize::from(n > 0))
    }

    /// The `n` topmost operands, in the order they were pushed, in a vector
    /// with `spare` further slots of capacity.
    fn pop_n_spare(&mut self, n: usize, spare: usize) -> CelResult<Vec<Value>> {
        let mut args = Vec::with_capacity(n + spare);
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

    /// The map literal on top of the stack, open for the next insert.
    ///
    /// `Arc::get_mut` cannot fail here: the operand is the only holder of that
    /// pointer until [`Vm::finish`] hands it to [`Map::object`], and nothing
    /// between [`OpCode::NewMap`] and that point clones it. A `None` would be
    /// the same internal-consistency failure as a non-map on top of the stack,
    /// so it takes the same answer.
    fn map_mut(&mut self) -> CelResult<&mut HashMap<Key, Value>> {
        match self.stack.last_mut() {
            Some(Operand::Map(entries)) => Arc::get_mut(entries).ok_or(CelErr::InternalError),
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
        // Whatever a `CallQualified` miss parked belongs to a `CallMethod`
        // that this error has just decided will not run -- whether the handler
        // below absorbs it and lands in the right operand, or nothing catches
        // and the evaluation ends.
        self.pending_args = None;
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
            OpCode::IncLocal => {
                // In place, so nothing is copied onto the stack and nothing is
                // popped back off it. The slot is a comprehension's counter,
                // written once with a zero and thereafter only here, so a
                // non-integer in it is a malformed stream -- the same answer
                // `IterLen` gives a slot that does not hold a list.
                let slot = self
                    .slots
                    .get_mut(a as usize)
                    .ok_or(CelErr::InternalError)?;
                let Value::Int(counter) = slot else {
                    return Err(CelErr::InternalError);
                };
                // Named for the operator this replaces, so that the public
                // error is the one the four-instruction form raised.
                *counter = counter
                    .checked_add(1)
                    .ok_or(CelErr::Overflow(OpCode::Add))?;
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
            OpCode::NewMap => self.stack.push(Operand::Map(Arc::default())),
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
                // Taken before anything can fail, so the park cannot outlive
                // the instruction that owns it.
                let parked = self.pending_args.take();
                let target = self.pop()?;
                let args = match parked {
                    // A `CallQualified` miss already popped them, and only the
                    // receiver sits above.
                    Some(args) => args,
                    None => self.pop_n_for_member(b as usize)?,
                };
                debug_assert_eq!(args.len(), b as usize, "arity lost across the probe");
                let value = self.call_member(NameId(a), target, args)?;
                self.push(value);
            }
            OpCode::CallQualified => {
                let args = self.pop_n_for_member(b as usize)?;
                // A miss parks the arguments instead of re-pushing them, so
                // the stack the receiver path falls through to holds the
                // receiver alone and `CallMethod` reads the park.
                if let Some(value) = self.call_qualified(NameId(a), args)? {
                    self.push(value);
                    return Ok(Step::Jump(c));
                }
            }

            // -- iteration ----------------------------------------------------
            OpCode::IterElems => {
                let value = self.pop()?;
                match value {
                    // A list already IS the sequence this iterates, so the
                    // popped value is pushed straight back. Materializing it
                    // again bought a `Vec<Value>` buffer and the
                    // `Arc<ListStorage>` [`Value::list`] wraps it in -- two
                    // allocations per comprehension, and nothing else: the two
                    // instructions that read the slot, [`OpCode::IterLen`] and
                    // [`OpCode::IterAt`], are both window-relative and neither
                    // cares which buffer answers them.
                    //
                    // The slot now SHARES the caller's buffer instead of owning
                    // a private snapshot of it, and nothing can change that
                    // buffer while the loop runs. A `Value` has no interior
                    // mutability, so the only writes are the two that rewrite a
                    // list in place -- `ListRef::into_vec` and `ListRef::concat`
                    // -- and both go through `Arc::get_mut`, which cannot answer
                    // for as long as this slot holds a reference of its own. The
                    // accumulator cannot be that other reference either: on the
                    // appending path it is an `Operand::List(Vec<Value>)` that
                    // owns its elements outright, and on the general path it is
                    // whatever `accu_init` produced, built before the loop.
                    Value::List(_) => self.push(value),
                    _ => {
                        let items = value_iter(&value).map_err(|e| self.park(e))?;
                        self.push(Value::list(items));
                    }
                }
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
    /// `None` is a miss, and a miss must leave no *evaluated* trace -- the
    /// receiver has not run yet, because `optional.of(1)` names no variable
    /// `optional`. It does leave `args` in [`Vm::pending_args`], which is
    /// where the `CallMethod` after it takes them from.
    fn call_qualified(&mut self, joined: NameId, args: Vec<Value>) -> CelResult<Option<Value>> {
        let name = self.name(joined.0)?;
        if let Some(op) = self.ctx.env().find_overload(name, &args) {
            return op(args).map(Some).map_err(|e| self.park(e));
        }
        let Some(func) = self.ctx.get_function(name) else {
            self.pending_args = Some(args);
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

    /// A `CallQualified` miss hands its arguments to the `CallMethod` the
    /// compiler emitted after it. When the receiver load in between raises and
    /// a short-circuit absorbs the error, that `CallMethod` never runs, and
    /// the hand-off must not survive into the other operand's own member call.
    ///
    /// `undefined_name` names neither a variable nor the first half of a
    /// function, so the probe misses and `LoadVar` raises. Both right operands
    /// have a LITERAL receiver, which is the arm that pops its own arguments
    /// -- so a surviving hand-off is what they would read instead, and each
    /// answer below flips: `true` becomes a raised error, `false` does too.
    #[test]
    fn an_absorbed_receiver_error_does_not_leave_the_probes_arguments_waiting() {
        let ctx = Context::default();
        for (source, want) in [
            (
                r#"undefined_name.startsWith("zzz") || "hello world".contains("o w")"#,
                true,
            ),
            (
                r#"undefined_name.startsWith("o w") && "hello world".contains("zzz")"#,
                false,
            ),
        ] {
            let code = compile(&parse(source)).expect("compiles");
            let mut vm = Vm::new(&code, &ctx);
            let got = vm
                .run()
                .unwrap_or_else(|e| panic!("{source}: {:?}", vm.public_error(e)));
            assert_eq!(got, Value::Bool(want), "{source}");
            assert!(
                vm.pending_args.is_none(),
                "{source}: the probe's arguments outlived the call that popped them"
            );
        }
    }

    /// The argument vector a member call is given has room for the receiver
    /// [`Vm::call_member`] prepends, so that `insert` never reallocates --
    /// EXCEPT at arity 0, where it must have no room at all.
    ///
    /// A `Vec`'s capacity is not observable through the public door -- the
    /// allocation it saves is, and `tests/allocs_per_eval.rs` is where that is
    /// pinned. This asserts the invariant those rows rest on directly, because
    /// a baseline row drifting up by one is a far colder trail than a name.
    ///
    /// The arity-0 leg is the opposite assertion rather than a relaxed one.
    /// `Vec::with_capacity(0)` allocates nothing, so an empty vector that
    /// leaves here with spare capacity has already spent the allocation that
    /// reserving was supposed to save: on the member path `insert` would have
    /// grown it once anyway, and on [`OpCode::CallQualified`]'s hit path --
    /// which pops through here before it knows hit from miss, and which nullary
    /// overloads like `optional.none` take -- nothing is ever inserted, so the
    /// vector is bought and thrown away. `capacity() > 0` here is one
    /// allocation per nullary namespaced call, which is what
    /// `walker/qualified_call/nullary` counts.
    #[test]
    fn a_member_calls_argument_vector_has_room_for_its_receiver() {
        let ctx = Context::default();
        let code = compile(&parse("1")).expect("compiles");
        for arity in 0..4usize {
            let mut vm = Vm::new(&code, &ctx);
            for i in 0..arity {
                vm.push(Value::Int(i as i64));
            }
            let args = vm.pop_n_for_member(arity).expect("the stack holds them");
            assert_eq!(args.len(), arity);
            if arity == 0 {
                assert_eq!(
                    args.capacity(),
                    0,
                    "arity 0: reserving buys nothing here and costs an allocation"
                );
            } else {
                assert!(
                    args.capacity() > arity,
                    "arity {arity}: prepending the receiver would reallocate"
                );
            }
        }
    }

    /// The table a map literal is built in is the table the finished [`Map`]
    /// holds, not a copy of it.
    ///
    /// [`OpCode::NewMap`] opens the operand in the [`Arc`] [`Map::object`]
    /// will take, so closing the literal hands the pointer over. A `Box`
    /// builder is the same 8 bytes on the operand stack but a different
    /// allocation, so `finish` had to allocate the `Arc` as well and move the
    /// 48-byte table into it -- one allocation per map literal on top of the
    /// one that was going to happen anyway. That is `walker/map_literal` in
    /// `tests/allocs_per_eval.rs`; this is the mechanism under that row, and a
    /// pointer identity is a much colder trail to follow from a baseline that
    /// drifted up by one.
    ///
    /// Driven through [`Vm::step`] rather than by pushing an operand by hand,
    /// so the `NewMap`/`MapInsert` pair the compiler emits is what runs. The
    /// literal is NESTED because that is the only shape that reaches
    /// [`Vm::map_mut`] with an outer map already open -- the one place a
    /// builder could come to be shared, which is what would make
    /// `Arc::get_mut` answer `None`.
    #[test]
    fn a_map_literal_is_built_in_the_arc_it_is_handed_over_in() {
        use crate::objects::MapStorage;

        let ctx = Context::default();
        let code = compile(&parse(r#"{"x": {"y": 3}}"#)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);

        // `Vm::run`'s loop with one line added: each builder is recorded the
        // first time it is seen on top of the stack. Neither table is freed
        // before the answer is read -- the inner one moves into the outer --
        // so no recorded address can be reused by the other.
        let mut opened: Vec<*const HashMap<Key, Value>> = Vec::new();
        let mut pc = 0u32;
        let answer = loop {
            let (op, operands) = vm.code.decode(pc).expect("the program decodes");
            let operands: [u32; 3] = [
                operands.first().copied().unwrap_or(0),
                operands.get(1).copied().unwrap_or(0),
                operands.get(2).copied().unwrap_or(0),
            ];
            let next = pc + op.width();
            let step = vm
                .step(op, operands, pc, next)
                .expect("the literal evaluates");
            if let Some(Operand::Map(entries)) = vm.stack.last() {
                let ptr = Arc::as_ptr(entries);
                if !opened.contains(&ptr) {
                    opened.push(ptr);
                }
            }
            match step {
                Step::Next => pc = next,
                Step::Jump(target) => pc = target,
                Step::Return(value) => break value,
            }
        };

        let Value::Map(outer) = answer else {
            panic!("a map literal evaluates to a map")
        };
        let MapStorage::Object(outer_entries) = outer.storage() else {
            panic!("a map LITERAL is an owned table, not a record row")
        };
        let inner = match outer_entries.values().next() {
            Some(Value::Map(inner)) => inner,
            other => panic!("the outer table holds the inner map, got {other:?}"),
        };
        let MapStorage::Object(inner_entries) = inner.storage() else {
            panic!("a map LITERAL is an owned table, not a record row")
        };

        assert_eq!(
            opened,
            vec![Arc::as_ptr(outer_entries), Arc::as_ptr(inner_entries)],
            "the two tables this program built are not the two it answered with, \
             so closing a map literal copied it instead of handing it over"
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

    /// A comprehension over a list that is neither boxed nor whole.
    ///
    /// [`OpCode::IterElems`] hands a list straight back instead of copying it
    /// out, so the loop reads the caller's [`ListRef`] where it stands: an
    /// unboxed column, seen through a WINDOW that starts past its first element
    /// and stops before its last. Copying it out used to normalise both of
    /// those away before the loop ever saw them, which is exactly why the
    /// corpus -- whose lists are all whole boxed buffers -- cannot reach this.
    #[test]
    fn a_comprehension_reads_a_windowed_unboxed_list_through_the_window() {
        use crate::objects::{ListRef, ListStorage, ScalarBank, ValueColumn};

        let words: Arc<[i64]> = Arc::from([10i64, 20, 30, 40, 50, 60].as_slice());
        let storage = Arc::new(ListStorage::Column(ValueColumn::Scalar {
            bank: ScalarBank::Int,
            words,
        }));
        let windowed = Value::List(ListRef::window(storage, 2, 3));

        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", windowed.clone());

        assert_eq!(
            run("xs.map(x, x)", &ctx),
            Ok(Value::list(vec![
                Value::Int(30),
                Value::Int(40),
                Value::Int(50)
            ]))
        );
        assert_eq!(
            run("xs.all(x, x >= 30 && x <= 50)", &ctx),
            Ok(Value::Bool(true))
        );
        // The body names the iteration source, so the loop and the binding read
        // one buffer at once -- and the binding still answers for itself after.
        assert_eq!(
            run("xs.map(x, xs)", &ctx),
            Ok(Value::list(vec![
                windowed.clone(),
                windowed.clone(),
                windowed.clone()
            ]))
        );
        assert_eq!(
            run("xs.filter(x, x in xs)", &ctx),
            run("xs.map(x, x)", &ctx)
        );
        assert_eq!(run("xs", &ctx), Ok(windowed));
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
