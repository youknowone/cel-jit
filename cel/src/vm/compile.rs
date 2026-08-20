//! The compiler: an expression tree in, a [`CelCode`] out.
//!
//! The match over [`Expr`] is exhaustive and has no `_` arm, so an AST node
//! added later is a compile error here rather than a case that silently
//! reaches a runtime fallback.

use std::collections::HashMap;

use super::code::{CelCode, Handler};
use super::error::NameId;
use super::opcode::OpCode;
use crate::common::ast::{
    operators, CallExpr, ComprehensionExpr, EntryExpr, Expr, IdedExpr, ListExpr, LiteralValue,
    MapExpr, SelectExpr, StructExpr,
};
use crate::Value;

/// Why an expression could not be compiled.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CompileErrorKind {
    /// An `Expr::Unspecified` node. The walker reaches this and panics; here
    /// it is refused before anything runs.
    #[error("expression is unspecified")]
    UnspecifiedExpr,
    /// A struct field appeared among a map literal's entries, or a map entry
    /// among a struct literal's. The walker panics on the first and silently
    /// mishandles the second.
    #[error("{found} entry in a {container} literal")]
    MismatchedEntry {
        container: &'static str,
        found: &'static str,
    },
    /// An operator was applied to the wrong number of arguments. The parser
    /// does not produce these, so reaching one means the AST was built by
    /// hand or by a future macro.
    #[error("operator '{operator}' expects {expected} argument(s), got {actual}")]
    OperatorArity {
        operator: String,
        expected: usize,
        actual: usize,
    },
    /// `_?._` requires a literal field name.
    #[error("optional select requires a literal field name")]
    OptSelectFieldNotLiteral,
    /// More constants, names or slots than the operand encoding can address.
    #[error("program exceeds the {0} limit")]
    TooLarge(&'static str),
    /// The compiler's own stack accounting went negative, which means an
    /// opcode's declared stack effect disagrees with how it is emitted.
    #[error("internal: operand stack underflow while compiling")]
    StackUnderflow,
}

/// A [`CompileErrorKind`] together with the AST node that produced it.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{kind} (node {id})")]
pub struct CompileError {
    pub kind: CompileErrorKind,
    /// The `IdedExpr::id` of the offending node, which `SourceInfo` maps back
    /// to a source offset.
    pub id: u64,
}

/// Compile an expression into a code object.
pub fn compile(expr: &IdedExpr) -> Result<CelCode, CompileError> {
    let mut compiler = Compiler::default();
    compiler.expr(expr)?;
    compiler.emit(OpCode::Return, &[], expr.id)?;
    compiler.finish()
}

#[derive(Default)]
struct Compiler {
    code: Vec<u32>,
    consts: Vec<Value>,
    names: Vec<Box<str>>,
    name_index: HashMap<Box<str>, u32>,
    /// Innermost last. Each scope maps a comprehension variable to its slot.
    scopes: Vec<Vec<(Box<str>, u32)>>,
    /// Next free slot; restored to the enclosing mark when a scope closes, so
    /// sibling comprehensions reuse slots and only nested ones stack.
    next_slot: u32,
    /// High-water mark of `next_slot`, which is the activation record's size.
    n_slots: u32,
    /// Next free logic slot, restored when a short-circuit operator's merge
    /// has read it, so siblings share one slot and only nested ones stack.
    next_logic: u32,
    /// High-water mark of `next_logic`.
    n_logic: u32,
    handlers: Vec<Handler>,
    depth: i64,
    max_stack: i64,
}

impl Compiler {
    fn finish(self) -> Result<CelCode, CompileError> {
        Ok(CelCode {
            code: self.code,
            consts: self.consts,
            names: self.names,
            n_slots: self.n_slots,
            max_stack: u32::try_from(self.max_stack).unwrap_or(u32::MAX),
            n_logic: self.n_logic,
            handlers: self.handlers,
        })
    }

    // -- emission ---------------------------------------------------------

    fn emit(&mut self, op: OpCode, operands: &[u32], id: u64) -> Result<u32, CompileError> {
        debug_assert_eq!(op.operands() as usize, operands.len(), "{op:?} arity");
        let at = self.here();
        self.code.push(op as u32);
        self.code.extend_from_slice(operands);

        let (pops, pushes) = op.stack_effect(operands);
        self.depth -= i64::from(pops);
        if self.depth < 0 {
            return Err(CompileError {
                kind: CompileErrorKind::StackUnderflow,
                id,
            });
        }
        self.depth += i64::from(pushes);
        self.max_stack = self.max_stack.max(self.depth);
        Ok(at)
    }

    fn here(&self) -> u32 {
        self.code.len() as u32
    }

    /// Emit a jump whose target is not known yet, returning the operand's
    /// index so it can be patched once the target is.
    fn emit_forward(&mut self, op: OpCode, id: u64) -> Result<usize, CompileError> {
        let at = self.emit(op, &[u32::MAX], id)?;
        Ok(at as usize + 1)
    }

    fn patch_to_here(&mut self, site: usize) {
        self.code[site] = self.here();
    }

    // -- pools ------------------------------------------------------------

    fn add_const(&mut self, value: Value, id: u64) -> Result<u32, CompileError> {
        let index = u32::try_from(self.consts.len()).map_err(|_| CompileError {
            kind: CompileErrorKind::TooLarge("constant pool"),
            id,
        })?;
        self.consts.push(value);
        Ok(index)
    }

    fn add_name(&mut self, name: &str, id: u64) -> Result<NameId, CompileError> {
        if let Some(&index) = self.name_index.get(name) {
            return Ok(NameId(index));
        }
        let index = u32::try_from(self.names.len()).map_err(|_| CompileError {
            kind: CompileErrorKind::TooLarge("name table"),
            id,
        })?;
        let name: Box<str> = name.into();
        self.names.push(name.clone());
        self.name_index.insert(name, index);
        Ok(NameId(index))
    }

    // -- scopes -----------------------------------------------------------

    fn open_scope(&mut self) -> u32 {
        self.scopes.push(Vec::new());
        self.next_slot
    }

    fn close_scope(&mut self, mark: u32) {
        self.scopes.pop();
        self.next_slot = mark;
    }

    fn declare(&mut self, name: &str, id: u64) -> Result<u32, CompileError> {
        let slot = self.next_slot;
        self.next_slot = slot.checked_add(1).ok_or(CompileError {
            kind: CompileErrorKind::TooLarge("slot count"),
            id,
        })?;
        self.n_slots = self.n_slots.max(self.next_slot);
        if let Some(scope) = self.scopes.last_mut() {
            scope.push((name.into(), slot));
        }
        Ok(slot)
    }

    /// A slot the program needs but the expression does not name -- the
    /// iteration sequence, its index, and the two-variable range.
    fn declare_hidden(&mut self, id: u64) -> Result<u32, CompileError> {
        self.declare("", id)
    }

    fn lookup(&self, name: &str) -> Option<u32> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.iter().rev().find(|(n, _)| &**n == name))
            .map(|(_, slot)| *slot)
    }

    // -- the exhaustive match ---------------------------------------------

    fn expr(&mut self, e: &IdedExpr) -> Result<(), CompileError> {
        match &e.expr {
            Expr::Unspecified => Err(CompileError {
                kind: CompileErrorKind::UnspecifiedExpr,
                id: e.id,
            }),
            Expr::Literal(literal) => self.literal(literal, e.id),
            Expr::Ident(name) => self.ident(name, e.id),
            Expr::Select(select) => self.select(select, e.id),
            Expr::List(list) => self.list(list, e.id),
            Expr::Map(map) => self.map(map, e.id),
            Expr::Struct(structure) => self.structure(structure, e.id),
            Expr::Call(call) => self.call(call, e.id),
            Expr::Comprehension(comprehension) => self.comprehension(comprehension, e.id),
        }
    }

    fn literal(&mut self, literal: &LiteralValue, id: u64) -> Result<(), CompileError> {
        // `to_value` is a refcount bump for the owning variants and a copy for
        // the rest, so the pool holds no more than the AST already did.
        let index = self.add_const(literal.to_value(), id)?;
        self.emit(OpCode::LoadConst, &[index], id)?;
        Ok(())
    }

    /// The compile-time split the walker makes at run time: a comprehension
    /// variable is a slot, anything else is a context lookup.
    fn ident(&mut self, name: &str, id: u64) -> Result<(), CompileError> {
        match self.lookup(name) {
            Some(slot) => self.emit(OpCode::LoadLocal, &[slot], id)?,
            None => {
                let name = self.add_name(name, id)?;
                self.emit(OpCode::LoadVar, &[name.0], id)?
            }
        };
        Ok(())
    }

    /// `SelectExpr::test` is set only by the `has` macro expander, so it is a
    /// compile-time constant and picks the opcode rather than a branch.
    fn select(&mut self, select: &SelectExpr, id: u64) -> Result<(), CompileError> {
        self.expr(&select.operand)?;
        let field = self.add_name(&select.field, id)?;
        let op = if select.test {
            OpCode::HasField
        } else {
            OpCode::GetField
        };
        self.emit(op, &[field.0], id)?;
        Ok(())
    }

    fn list(&mut self, list: &ListExpr, id: u64) -> Result<(), CompileError> {
        self.emit(OpCode::NewList, &[], id)?;
        for (index, element) in list.elements.iter().enumerate() {
            self.expr(element)?;
            let op = if list.optional_indices.contains(&index) {
                OpCode::ListAppendOptional
            } else {
                OpCode::ListAppend
            };
            self.emit(op, &[], element.id)?;
        }
        Ok(())
    }

    fn map(&mut self, map: &MapExpr, id: u64) -> Result<(), CompileError> {
        self.emit(OpCode::NewMap, &[], id)?;
        for entry in &map.entries {
            let EntryExpr::MapEntry(kv) = &entry.expr else {
                return Err(CompileError {
                    kind: CompileErrorKind::MismatchedEntry {
                        container: "map",
                        found: "struct field",
                    },
                    id: entry.id,
                });
            };
            self.expr(&kv.key)?;
            self.expr(&kv.value)?;
            let op = if kv.optional {
                OpCode::MapInsertOptional
            } else {
                OpCode::MapInsert
            };
            self.emit(op, &[], entry.id)?;
        }
        Ok(())
    }

    fn structure(&mut self, structure: &StructExpr, id: u64) -> Result<(), CompileError> {
        let type_name = self.add_name(&structure.type_name, id)?;
        self.emit(OpCode::NewStruct, &[type_name.0], id)?;
        for entry in &structure.entries {
            let EntryExpr::StructField(field) = &entry.expr else {
                return Err(CompileError {
                    kind: CompileErrorKind::MismatchedEntry {
                        container: "struct",
                        found: "map",
                    },
                    id: entry.id,
                });
            };
            self.expr(&field.value)?;
            let name = self.add_name(&field.field, entry.id)?;
            let op = if field.optional {
                OpCode::StructSetOptional
            } else {
                OpCode::StructSet
            };
            self.emit(op, &[name.0], entry.id)?;
        }
        Ok(())
    }
}

/// The operators that map to exactly one opcode over their operands.
///
/// Everything with control flow -- the conditional and the two short-circuit
/// operators -- is handled separately, and everything absent from both is an
/// ordinary function call.
fn simple_operator(name: &str) -> Option<(OpCode, usize)> {
    let entry = match name {
        operators::ADD => (OpCode::Add, 2),
        operators::SUBSTRACT => (OpCode::Sub, 2),
        operators::MULTIPLY => (OpCode::Mul, 2),
        operators::DIVIDE => (OpCode::Div, 2),
        operators::MODULO => (OpCode::Mod, 2),
        operators::EQUALS => (OpCode::Equals, 2),
        operators::NOT_EQUALS => (OpCode::NotEquals, 2),
        operators::LESS => (OpCode::Less, 2),
        operators::LESS_EQUALS => (OpCode::LessEquals, 2),
        operators::GREATER => (OpCode::Greater, 2),
        operators::GREATER_EQUALS => (OpCode::GreaterEquals, 2),
        operators::IN => (OpCode::In, 2),
        operators::LOGICAL_NOT => (OpCode::Not, 1),
        operators::NEGATE => (OpCode::Negate, 1),
        operators::NOT_STRICTLY_FALSE => (OpCode::NotStrictlyFalse, 1),
        _ => return None,
    };
    Some(entry)
}

impl Compiler {
    fn call(&mut self, call: &CallExpr, id: u64) -> Result<(), CompileError> {
        // An operator name can only be an operator: the parser mints these
        // and they are not valid identifiers.
        if call.target.is_none() {
            if let Some((op, arity)) = simple_operator(&call.func_name) {
                self.check_arity(call, arity, id)?;
                for arg in &call.args {
                    self.expr(arg)?;
                }
                self.emit(op, &[], id)?;
                return Ok(());
            }
            match call.func_name.as_str() {
                operators::CONDITIONAL => return self.conditional(call, id),
                operators::LOGICAL_AND => return self.short_circuit(call, OpCode::And, id),
                operators::LOGICAL_OR => return self.short_circuit(call, OpCode::Or, id),
                operators::OPT_SELECT => return self.opt_select(call, id),
                operators::INDEX => return self.index(call, OpCode::Index, id),
                operators::OPT_INDEX => return self.index(call, OpCode::OptIndex, id),
                _ => {}
            }
        }
        self.function_call(call, id)
    }

    fn check_arity(&self, call: &CallExpr, expected: usize, id: u64) -> Result<(), CompileError> {
        if call.args.len() == expected {
            return Ok(());
        }
        Err(CompileError {
            kind: CompileErrorKind::OperatorArity {
                operator: call.func_name.clone(),
                expected,
                actual: call.args.len(),
            },
            id,
        })
    }

    fn conditional(&mut self, call: &CallExpr, id: u64) -> Result<(), CompileError> {
        self.check_arity(call, 3, id)?;
        self.expr(&call.args[0])?;
        let to_else = self.emit_forward(OpCode::JumpIfFalse, id)?;

        // Both arms leave exactly one operand, so the depth after the merge
        // must not count them twice.
        let before_arms = self.depth;
        self.expr(&call.args[1])?;
        let to_end = self.emit_forward(OpCode::Jump, id)?;

        self.patch_to_here(to_else);
        self.depth = before_arms;
        self.expr(&call.args[2])?;
        self.patch_to_here(to_end);
        Ok(())
    }

    /// `a && b` and `a || b`.
    ///
    /// Both operands are evaluated unless the left one decides the result on
    /// its own, and an error in the left operand is *recorded* rather than
    /// raised, because the right operand may still discard it:
    /// `undefined_name && false` is `false`.
    ///
    /// So the left operand runs under a handler covering exactly its own
    /// instructions, landing on the right operand with the error already in
    /// the logic slot. The right operand is deliberately outside that range:
    /// an error there propagates, matching the walker's `?`.
    ///
    /// ```text
    /// [handler start]
    ///   <a>
    /// [handler end]   And   logic, L_short   ; a -> logic; false short-circuits
    /// [handler land]  <b>
    ///                 AndMerge logic          ; combine, or raise
    /// L_short:
    /// ```
    ///
    /// Each operator gets its own logic slot, because a nested `&&` in the
    /// right operand writes its own between this one's write and read.
    fn short_circuit(&mut self, call: &CallExpr, op: OpCode, id: u64) -> Result<(), CompileError> {
        self.check_arity(call, 2, id)?;

        let logic = self.next_logic;
        self.next_logic = logic.checked_add(1).ok_or(CompileError {
            kind: CompileErrorKind::TooLarge("logic slot count"),
            id,
        })?;
        self.n_logic = self.n_logic.max(self.next_logic);

        let depth = u32::try_from(self.depth).unwrap_or(0);
        let start = self.here();
        self.expr(&call.args[0])?;
        let end = self.here();

        let at = self.emit(op, &[logic, u32::MAX], id)?;
        let land = self.here();
        self.handlers.push(Handler {
            start,
            end,
            land,
            logic,
            depth,
        });

        self.expr(&call.args[1])?;
        let merge = match op {
            OpCode::And => OpCode::AndMerge,
            _ => OpCode::OrMerge,
        };
        self.emit(merge, &[logic], id)?;
        self.patch_to_here(at as usize + 2);

        // The slot is dead once the merge has read it, so a sibling operator
        // reuses it rather than growing the record.
        self.next_logic = logic;
        Ok(())
    }

    /// `a[b]` and `a[?b]`.
    ///
    /// Not a plain two-operand operator, because an optional container decides
    /// the whole expression by itself and does so *before* the key runs:
    /// `opt_none[1 / 0]` is `optional.none`, not a division error. The guard
    /// is what puts that order into a stream that would otherwise evaluate
    /// both operands and only then look at either.
    fn index(&mut self, call: &CallExpr, op: OpCode, id: u64) -> Result<(), CompileError> {
        self.check_arity(call, 2, id)?;
        self.expr(&call.args[0])?;
        let none = self.emit_forward(OpCode::JumpIfOptNone, id)?;
        self.expr(&call.args[1])?;
        self.emit(op, &[], id)?;
        self.patch_to_here(none);
        Ok(())
    }

    /// `a?.b`, whose field name the parser always supplies as a string
    /// literal.
    fn opt_select(&mut self, call: &CallExpr, id: u64) -> Result<(), CompileError> {
        self.check_arity(call, 2, id)?;
        let Expr::Literal(LiteralValue::String(field)) = &call.args[1].expr else {
            return Err(CompileError {
                kind: CompileErrorKind::OptSelectFieldNotLiteral,
                id: call.args[1].id,
            });
        };
        self.expr(&call.args[0])?;
        let field = self.add_name(field, id)?;
        self.emit(OpCode::OptSelect, &[field.0], id)?;
        Ok(())
    }

    /// A call that is not an operator.
    ///
    /// The receiver form is the delicate one. `math.max(1, 2)` and
    /// `s.startsWith("h")` parse identically, and which one an expression is
    /// depends on the environment, which the compiler does not have. The
    /// walker resolves it by trying the namespaced function *first* and only
    /// evaluating the receiver if that misses -- an order that matters,
    /// because `optional.of(1)` must not fail on `optional` being an
    /// undeclared variable.
    ///
    /// So the emitted program keeps both paths and lets the run-time lookup
    /// pick, with the joined name already in the name table. That is what
    /// makes the probe free: joining it per evaluation is the allocation the
    /// walker used to pay on every member call with an identifier receiver.
    fn function_call(&mut self, call: &CallExpr, id: u64) -> Result<(), CompileError> {
        let arity = u32::try_from(call.args.len()).map_err(|_| CompileError {
            kind: CompileErrorKind::TooLarge("argument count"),
            id,
        })?;

        let Some(target) = &call.target else {
            let name = self.add_name(&call.func_name, id)?;
            for arg in &call.args {
                self.expr(arg)?;
            }
            self.emit(OpCode::CallHost, &[name.0, arity], id)?;
            return Ok(());
        };

        let name = self.add_name(&call.func_name, id)?;
        let namespace = match &target.expr {
            Expr::Ident(prefix) if self.lookup(prefix).is_none() => {
                Some(self.add_name(&format!("{prefix}.{}", call.func_name), id)?)
            }
            // A receiver that is anything else -- a literal, a call, a
            // comprehension variable -- cannot name a namespace.
            _ => None,
        };

        for arg in &call.args {
            self.expr(arg)?;
        }

        let Some(namespace) = namespace else {
            self.expr(target)?;
            self.emit(OpCode::CallMethod, &[name.0, arity], id)?;
            return Ok(());
        };

        let with_args = self.depth;
        let at = self.emit(OpCode::CallQualified, &[namespace.0, arity, u32::MAX], id)?;
        // The emitted effect is the jumping path's, so the depth now models
        // the merge. The falling-through path still has the arguments.
        let merged = self.depth;
        self.depth = with_args;

        self.expr(target)?;
        self.emit(OpCode::CallMethod, &[name.0, arity], id)?;
        debug_assert_eq!(
            self.depth, merged,
            "the two call paths must meet at one depth"
        );
        self.patch_to_here(at as usize + 3);
        Ok(())
    }
}

/// A comprehension whose step only ever appends one element to its own
/// accumulator -- what `map` and `filter` expand to.
///
/// This is the compile-time twin of the walker's `AccuAppend` (`objects.rs`),
/// and it exists for the same reason: the general lowering rebuilds the
/// accumulator once per element, because `@result + [x]` is a list
/// concatenation. That is a fresh buffer and a full copy per iteration, so an
/// *n*-element `map` costs O(n^2) copying where the shape only ever needed a
/// push. The walker has recognised it since before the VM existed; without the
/// same recognition here the VM is asymptotically slower than the evaluator it
/// replaces, which no amount of dispatch tuning can make up.
///
/// The preconditions are the walker's, restated:
///
/// * the loop condition is the literal `true`, so nothing can break early and
///   observe a partial accumulator (`exists` and `all` do break, and are not
///   this shape);
/// * the step is `@result + [elem]`, optionally wrapped in
///   `guard ? @result + [elem] : @result`, which is `filter`;
/// * the appended literal holds exactly one element and no optional index,
///   because `[?x]` appends zero or one depending on the value.
struct AccuAppend<'a> {
    /// `filter`'s condition, when the step is the conditional form.
    guard: Option<&'a IdedExpr>,
    /// The single element the step appends.
    element: &'a IdedExpr,
}

impl<'a> AccuAppend<'a> {
    fn of(comp: &'a ComprehensionExpr) -> Option<Self> {
        // A second variable is bound from the range per iteration; the parser
        // never emits one, and this shape has never been measured with it.
        if comp.iter_var2.is_some() {
            return None;
        }
        match &comp.loop_cond.expr {
            Expr::Literal(LiteralValue::Boolean(b)) if *b => {}
            _ => return None,
        }
        // The accumulator has to start as a list, and it has to be a list the
        // compiler can see: the emitted program builds it on the operand stack
        // rather than storing it, so `[]` must be a literal and not a value
        // that merely turns out to be a list at run time.
        if !matches!(&comp.accu_init.expr, Expr::List(_)) {
            return None;
        }

        let accu_var = comp.accu_var.as_str();
        let is_accu = |e: &IdedExpr| matches!(&e.expr, Expr::Ident(n) if n == accu_var);

        let (guard, step) = match &comp.loop_step.expr {
            Expr::Call(call)
                if call.func_name == operators::CONDITIONAL
                    && call.args.len() == 3
                    && is_accu(&call.args[2]) =>
            {
                (Some(&call.args[0]), &call.args[1])
            }
            _ => (None, &comp.loop_step),
        };

        let Expr::Call(call) = &step.expr else {
            return None;
        };
        if call.func_name != operators::ADD || call.args.len() != 2 || !is_accu(&call.args[0]) {
            return None;
        }
        match &call.args[1].expr {
            Expr::List(list) if list.elements.len() == 1 && list.optional_indices.is_empty() => {
                Some(AccuAppend {
                    guard,
                    element: &list.elements[0],
                })
            }
            _ => None,
        }
    }
}

impl Compiler {
    /// A comprehension, which is the only construct that emits a back edge.
    ///
    /// `iter_range` and `accu_init` are evaluated outside the new scope --
    /// neither can refer to the variables the comprehension binds -- and
    /// `result` inside it, because it refers to the accumulator.
    fn comprehension(&mut self, comp: &ComprehensionExpr, id: u64) -> Result<(), CompileError> {
        if let Some(append) = AccuAppend::of(comp) {
            return self.appending_comprehension(comp, &append, id);
        }
        let two_variable = comp.iter_var2.is_some();

        self.expr(&comp.iter_range)?;

        let mark = self.open_scope();

        // The second variable is the element at the first variable's key, so
        // the range has to outlive the sequence derived from it.
        let range = if two_variable {
            let slot = self.declare_hidden(id)?;
            self.emit(OpCode::StoreLocal, &[slot], id)?;
            self.emit(OpCode::LoadLocal, &[slot], id)?;
            Some(slot)
        } else {
            None
        };

        let source_op = if two_variable {
            OpCode::IterKeys
        } else {
            OpCode::IterElems
        };
        self.emit(source_op, &[], id)?;
        let source = self.declare_hidden(id)?;
        self.emit(OpCode::StoreLocal, &[source], id)?;

        // `accu_init` is outside the scope in CEL, but the accumulator slot
        // has to exist before the store. Declaring the name after evaluating
        // the initialiser is what keeps `accu_init` from seeing it.
        self.expr(&comp.accu_init)?;
        let accu = self.declare(&comp.accu_var, id)?;
        self.emit(OpCode::StoreLocal, &[accu], id)?;

        let index = self.declare_hidden(id)?;
        let zero = self.add_const(Value::Int(0), id)?;
        self.emit(OpCode::LoadConst, &[zero], id)?;
        self.emit(OpCode::StoreLocal, &[index], id)?;

        let iter_var = self.declare(&comp.iter_var, id)?;
        let iter_var2 = match &comp.iter_var2 {
            Some(name) => Some(self.declare(name, id)?),
            None => None,
        };

        let top = self.here();
        self.emit(OpCode::LoadLocal, &[index], id)?;
        self.emit(OpCode::IterLen, &[source], id)?;
        self.emit(OpCode::Less, &[], id)?;
        let exhausted = self.emit_forward(OpCode::JumpIfFalse, id)?;

        self.emit(OpCode::IterAt, &[source, index], id)?;
        self.emit(OpCode::StoreLocal, &[iter_var], id)?;

        if let (Some(range), Some(iter_var2)) = (range, iter_var2) {
            self.emit(OpCode::LoadLocal, &[range], id)?;
            self.emit(OpCode::LoadLocal, &[iter_var], id)?;
            self.emit(OpCode::Index, &[], id)?;
            self.emit(OpCode::StoreLocal, &[iter_var2], id)?;
        }

        self.expr(&comp.loop_cond)?;
        let broke = self.emit_forward(OpCode::JumpIfFalse, id)?;

        self.expr(&comp.loop_step)?;
        self.emit(OpCode::StoreLocal, &[accu], id)?;

        self.emit(OpCode::IncLocal, &[index], id)?;
        self.emit(OpCode::Jump, &[top], id)?;

        self.patch_to_here(exhausted);
        self.patch_to_here(broke);
        self.expr(&comp.result)?;

        self.close_scope(mark);
        Ok(())
    }

    /// The [`AccuAppend`] shape: the accumulator is a builder on the operand
    /// stack, and each iteration pushes one element into it.
    ///
    /// This is the same device a list *literal* already uses -- `NewList` plus
    /// one `ListAppend` per element, with the in-progress aggregate living on
    /// the stack as an `Operand::List` the append mutates in place. The only
    /// new thing is a loop around the appends, so no opcode is added and the
    /// dispatch loop is untouched.
    ///
    /// ```text
    ///   <iter_range>  IterElems  StoreLocal source
    ///   <accu_init>                       ; a list literal: leaves the builder
    ///   LoadConst 0   StoreLocal index
    /// top:
    ///   LoadLocal index  IterLen source  Less  JumpIfFalse done
    ///   IterAt source index  StoreLocal iter_var
    ///   [<guard> JumpIfFalse skip]
    ///   <element>  ListAppend            ; the push, straight into the builder
    /// skip:
    ///   IncLocal index  Jump top
    /// done:
    ///   StoreLocal accu                  ; finishes the builder into a Value
    ///   <result>
    /// ```
    ///
    /// The accumulator's name is declared only *after* the loop. That is not
    /// tidiness: it is what makes `@result` unreadable from the element and the
    /// guard, which is this shape's precondition. A read compiles to `LoadVar`
    /// and fails as an undeclared reference -- the same answer the walker gives,
    /// which holds its accumulator outside the context on this path.
    fn appending_comprehension(
        &mut self,
        comp: &ComprehensionExpr,
        append: &AccuAppend<'_>,
        id: u64,
    ) -> Result<(), CompileError> {
        self.expr(&comp.iter_range)?;

        let mark = self.open_scope();

        self.emit(OpCode::IterElems, &[], id)?;
        let source = self.declare_hidden(id)?;
        self.emit(OpCode::StoreLocal, &[source], id)?;

        // Leaves the builder on the stack, where it stays for the whole loop.
        self.expr(&comp.accu_init)?;

        let index = self.declare_hidden(id)?;
        let zero = self.add_const(Value::Int(0), id)?;
        self.emit(OpCode::LoadConst, &[zero], id)?;
        self.emit(OpCode::StoreLocal, &[index], id)?;

        let iter_var = self.declare(&comp.iter_var, id)?;

        let top = self.here();
        self.emit(OpCode::LoadLocal, &[index], id)?;
        self.emit(OpCode::IterLen, &[source], id)?;
        self.emit(OpCode::Less, &[], id)?;
        let exhausted = self.emit_forward(OpCode::JumpIfFalse, id)?;

        self.emit(OpCode::IterAt, &[source, index], id)?;
        self.emit(OpCode::StoreLocal, &[iter_var], id)?;

        let skipped = match append.guard {
            Some(guard) => {
                self.expr(guard)?;
                Some(self.emit_forward(OpCode::JumpIfFalse, id)?)
            }
            None => None,
        };

        self.expr(append.element)?;
        self.emit(OpCode::ListAppend, &[], id)?;

        if let Some(skipped) = skipped {
            self.patch_to_here(skipped);
        }

        self.emit(OpCode::IncLocal, &[index], id)?;
        self.emit(OpCode::Jump, &[top], id)?;

        self.patch_to_here(exhausted);
        let accu = self.declare(&comp.accu_var, id)?;
        self.emit(OpCode::StoreLocal, &[accu], id)?;
        self.expr(&comp.result)?;

        self.close_scope(mark);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ast::IdedExpr;
    use crate::parser::Parser;

    fn code_of(source: &str) -> CelCode {
        let expr = Parser::default()
            .parse(source)
            .unwrap_or_else(|e| panic!("parse {source}: {e:?}"));
        compile(&expr).unwrap_or_else(|e| panic!("compile {source}: {e}"))
    }

    fn opcodes(code: &CelCode) -> Vec<OpCode> {
        code.instructions().map(|(_, op, _)| op).collect()
    }

    /// The walker reaches `Expr::Unspecified` and panics. Compiling refuses
    /// it, which is the whole point of the match being exhaustive.
    #[test]
    fn unspecified_is_refused_rather_than_panicking() {
        let err = compile(&IdedExpr {
            id: 7,
            expr: Expr::Unspecified,
        })
        .expect_err("an unspecified expression has no value");
        assert_eq!(err.kind, CompileErrorKind::UnspecifiedExpr);
        assert_eq!(err.id, 7);
    }

    /// `has(m.x)` is a flag on a `Select` node, not a call, so it has to
    /// reach its own opcode rather than a branch inside the field read.
    #[test]
    fn has_is_its_own_opcode_and_a_plain_select_is_not() {
        assert!(opcodes(&code_of("has(m.x)")).contains(&OpCode::HasField));
        let plain = opcodes(&code_of("m.x"));
        assert!(plain.contains(&OpCode::GetField));
        assert!(!plain.contains(&OpCode::HasField));
    }

    /// The comprehension variable becomes a slot; only the free variable
    /// stays a context lookup.
    #[test]
    fn a_comprehension_variable_is_a_slot_and_a_free_variable_is_not() {
        let code = code_of("xs.all(x, x > lim)");
        assert!(opcodes(&code).contains(&OpCode::LoadLocal));

        let loaded_names: Vec<&str> = code
            .instructions()
            .filter(|(_, op, _)| *op == OpCode::LoadVar)
            .map(|(_, _, operands)| code.name(NameId(operands[0])).unwrap())
            .collect();
        assert!(loaded_names.contains(&"xs"), "{loaded_names:?}");
        assert!(loaded_names.contains(&"lim"), "{loaded_names:?}");
        assert!(!loaded_names.contains(&"x"), "{loaded_names:?}");
    }

    /// The activation-record claim: one record, siblings reusing slots and
    /// only nesting occupying distinct ranges.
    #[test]
    fn sibling_comprehensions_share_slots_and_nested_ones_do_not() {
        let single = code_of("xs.all(x, x > 0)").n_slots;
        let siblings = code_of("xs.all(x, x > 0) && ys.all(y, y > 0)").n_slots;
        let nested = code_of("xs.all(x, ys.all(y, y > x))").n_slots;

        assert_eq!(
            siblings, single,
            "two comprehensions that cannot be live at once should reuse the range"
        );
        assert!(
            nested > single,
            "a nested comprehension needs its own range: nested={nested} single={single}"
        );
    }

    /// The order that `optional.of(1)` depends on: the namespaced lookup is
    /// tried before anything evaluates `optional` as a variable.
    #[test]
    fn a_namespaced_call_probes_before_the_receiver_is_evaluated() {
        let code = code_of("optional.of(1)");
        let ops = opcodes(&code);

        let qualified = ops
            .iter()
            .position(|op| *op == OpCode::CallQualified)
            .expect("a receiver that is a bare identifier must probe the joined name");
        let receiver = ops
            .iter()
            .position(|op| *op == OpCode::LoadVar)
            .expect("the fallback path still loads the receiver");
        assert!(
            qualified < receiver,
            "the probe must precede the receiver load: {ops:?}"
        );

        // The joined name is built once, here, rather than per evaluation.
        assert!(
            code.names.iter().any(|n| &**n == "optional.of"),
            "{:?}",
            code.names
        );
    }

    /// Every `CallQualified` is followed by a `CallMethod` of the SAME arity.
    ///
    /// That pairing is what lets `Vm::call_qualified`'s miss hand its popped
    /// arguments straight to `Vm::call_member`, and it is why the probe pops
    /// with the receiver's slot already reserved: on a miss that vector is the
    /// one the receiver gets prepended to, and on a hit the spare slot is not
    /// an allocation. A probe emitted without its member call would make both
    /// claims false, so the pairing is asserted rather than assumed.
    #[test]
    fn a_probe_and_its_member_call_agree_on_arity() {
        for source in [
            "optional.of(1)",
            "s.startsWith(\"h\")",
            "s.noArgs()",
            "s.three(1, 2, 3)",
        ] {
            let code = code_of(source);
            let ops: Vec<_> = code
                .instructions()
                .map(|(_, op, args)| (op, args))
                .collect();
            let probes: Vec<_> = ops
                .iter()
                .filter(|(op, _)| *op == OpCode::CallQualified)
                .collect();
            assert_eq!(probes.len(), 1, "{source}: {ops:?}");
            let members: Vec<_> = ops
                .iter()
                .filter(|(op, _)| *op == OpCode::CallMethod)
                .collect();
            assert_eq!(members.len(), 1, "{source}: {ops:?}");
            // Operand 1 is the arity for both opcodes.
            assert_eq!(probes[0].1[1], members[0].1[1], "{source}: {ops:?}");
        }
    }

    /// A receiver that cannot name a namespace skips the probe entirely.
    #[test]
    fn a_non_identifier_receiver_is_only_a_method_call() {
        let ops = opcodes(&code_of(r#""hello".startsWith("h")"#));
        assert!(ops.contains(&OpCode::CallMethod));
        assert!(!ops.contains(&OpCode::CallQualified));
    }

    /// A comprehension variable shadowing a namespace is a receiver again,
    /// because the slot proves it is a value.
    #[test]
    fn a_receiver_bound_as_a_slot_is_never_a_namespace() {
        let ops = opcodes(&code_of("xs.all(optional, optional.of(1))"));
        assert!(!ops.contains(&OpCode::CallQualified), "{ops:?}");
    }

    #[test]
    fn max_stack_covers_the_deeper_conditional_arm() {
        let shallow = code_of("c ? a : b").max_stack;
        let deep = code_of("c ? [a, b, x, y] : b").max_stack;
        assert!(deep > shallow, "deep={deep} shallow={shallow}");
    }

    /// The parser never produces a two-variable comprehension -- every macro
    /// sets `iter_var2: None` -- so this is the only way to reach the arm.
    /// The second variable is the element at the first variable's key, which
    /// is why the range outlives the sequence derived from it.
    #[test]
    fn a_two_variable_comprehension_binds_both_names() {
        let Expr::Comprehension(base) = code_source("xs.all(k, k > 0)") else {
            panic!("the macro expands to a comprehension");
        };
        let mut comp = *base;
        comp.iter_var2 = Some("v".to_string());
        comp.loop_step = parse_expr("v");

        let code = compile(&IdedExpr {
            id: 0,
            expr: Expr::Comprehension(Box::new(comp)),
        })
        .expect("a two-variable comprehension compiles");

        let ops = opcodes(&code);
        assert!(ops.contains(&OpCode::IterKeys), "{ops:?}");
        assert!(!ops.contains(&OpCode::IterElems), "{ops:?}");
        // `v` resolves to a slot, not to a context lookup.
        let loaded: Vec<&str> = code
            .instructions()
            .filter(|(_, op, _)| *op == OpCode::LoadVar)
            .map(|(_, _, operands)| code.name(NameId(operands[0])).unwrap())
            .collect();
        assert!(!loaded.contains(&"v"), "{loaded:?}");
    }

    /// One variable iterates elements, so the two forms differ exactly where
    /// the opcode documentation says they do.
    #[test]
    fn a_one_variable_comprehension_iterates_elements() {
        let ops = opcodes(&code_of("xs.all(x, x > 0)"));
        assert!(ops.contains(&OpCode::IterElems), "{ops:?}");
        assert!(!ops.contains(&OpCode::IterKeys), "{ops:?}");
    }

    /// The loop reaches its source through slot operands and never puts it on
    /// the operand stack. What that replaced was two `LoadLocal`s of the
    /// source per element, and each of those copied the whole `Value` -- for a
    /// list, a `ListRef` clone whose `Arc` refcount is an atomic increment,
    /// matched by a decrement when the instruction that consumed it dropped
    /// the copy again.
    ///
    /// Stated as a property rather than as an expected instruction sequence:
    /// the sequence is what a later change to the loop is entitled to move,
    /// and the invariant is that the source stays out of the stack traffic.
    #[test]
    fn the_comprehension_loop_never_loads_its_source_onto_the_stack() {
        for source in [
            "xs.map(x, x * 2)",
            "xs.filter(x, x > 0)",
            "xs.all(x, x > 0)",
        ] {
            let code = code_of(source);
            let slot_of = |wanted: OpCode| {
                code.instructions()
                    .filter(|(_, op, _)| *op == wanted)
                    .map(|(_, _, operands)| operands[0])
                    .next()
            };
            let slot = slot_of(OpCode::IterAt)
                .unwrap_or_else(|| panic!("{source} iterates, so it names a source slot"));

            let loaded: Vec<u32> = code
                .instructions()
                .filter(|(_, op, _)| *op == OpCode::LoadLocal)
                .map(|(_, _, operands)| operands[0])
                .collect();
            assert!(
                !loaded.contains(&slot),
                "{source} loads its source slot {slot}:\n{}",
                code.disassemble()
            );
            // Both instructions must agree on which slot holds the sequence,
            // or the bound is being read off something other than what is
            // being indexed.
            assert_eq!(
                slot_of(OpCode::IterLen),
                Some(slot),
                "{source}:\n{}",
                code.disassemble()
            );
        }
    }

    /// The loop counter is advanced in place, without operand-stack traffic.
    ///
    /// The counter's slot is named by `IterAt`'s second operand, the way the
    /// source slot is named by its first, so this asks about the slot the loop
    /// actually indexes with rather than about a position in an emitted
    /// sequence. What it pins is that the counter is written once -- before
    /// the loop, with a zero -- and thereafter advanced by an instruction that
    /// neither pushes nor pops. A second `StoreLocal` of that slot is the
    /// load/add/store form coming back.
    ///
    /// Stated as a property for the reason
    /// `the_comprehension_loop_never_loads_its_source_onto_the_stack` is: the
    /// sequence is what a later change to this loop is entitled to move.
    #[test]
    fn the_comprehension_counter_is_advanced_without_operand_stack_traffic() {
        for source in [
            "xs.map(x, x * 2)",
            "xs.filter(x, x > 0)",
            "xs.all(x, x > 0)",
            "xs.exists(x, x > 0)",
        ] {
            let code = code_of(source);
            let counter = code
                .instructions()
                .find(|(_, op, _)| *op == OpCode::IterAt)
                .map(|(_, _, operands)| operands[1])
                .unwrap_or_else(|| panic!("{source} iterates, so it names a counter slot"));

            let stores = code
                .instructions()
                .filter(|(_, op, operands)| *op == OpCode::StoreLocal && operands[0] == counter)
                .count();
            assert_eq!(
                stores,
                1,
                "{source} writes counter slot {counter} through the operand stack:\n{}",
                code.disassemble()
            );

            let advanced = code
                .instructions()
                .any(|(_, op, operands)| op == OpCode::IncLocal && operands[0] == counter);
            assert!(
                advanced,
                "{source} must advance counter slot {counter} in place:\n{}",
                code.disassemble()
            );
        }
    }

    /// The handler covers the left operand and stops short of the right one,
    /// which is the whole of CEL's asymmetry: an error on the left is absorbed
    /// and an error on the right is raised.
    #[test]
    fn a_short_circuit_handler_covers_the_left_operand_only() {
        let code = code_of("a && b");
        assert_eq!(code.handlers.len(), 1, "{}", code.disassemble());
        let handler = code.handlers[0];

        let and = code
            .instructions()
            .find(|(_, op, _)| *op == OpCode::And)
            .map(|(pc, _, operands)| (pc, operands.to_vec()))
            .expect("`&&` emits an And");
        let merge = code
            .instructions()
            .find(|(_, op, _)| *op == OpCode::AndMerge)
            .map(|(pc, _, _)| pc)
            .expect("`&&` emits an AndMerge");

        assert_eq!(handler.start, 0, "the left operand starts the program");
        assert_eq!(handler.end, and.0, "the handler ends where the left does");
        assert_eq!(handler.land, and.0 + OpCode::And.width());
        assert_eq!(handler.depth, 0);
        assert_eq!(handler.logic, and.1[0], "the And writes the slot it covers");

        assert!(code.handler_for(0).is_some(), "the left operand is covered");
        assert!(
            code.handler_for(handler.land).is_none(),
            "the right operand is not: its errors propagate"
        );
        assert!(code.handler_for(merge).is_none());
        // The short-circuit target is past the merge, so both paths arrive at
        // the same depth.
        assert_eq!(and.1[1], merge + OpCode::AndMerge.width());
    }

    /// Nested operators need distinct slots, because the outer one writes its
    /// slot before the inner one has finished with its own.
    #[test]
    fn nested_short_circuits_do_not_share_a_logic_slot() {
        let code = code_of("(a && b) && c");
        assert_eq!(code.n_logic, 2, "{}", code.disassemble());

        let slots: Vec<u32> = code
            .instructions()
            .filter(|(_, op, _)| *op == OpCode::And)
            .map(|(_, _, operands)| operands[0])
            .collect();
        assert_eq!(slots.len(), 2);
        assert_ne!(slots[0], slots[1], "{}", code.disassemble());

        // The inner operator's own left operand is covered by the inner
        // handler, which is the shorter of the two covering it.
        let inner = code.handler_for(0).expect("instruction 0 is covered");
        assert_eq!(inner.end - inner.start, 2, "{}", code.disassemble());
    }

    /// Two operators that cannot be live at once share a slot, for the same
    /// reason sibling comprehensions share activation-record slots.
    #[test]
    fn sibling_short_circuits_share_a_logic_slot() {
        assert_eq!(code_of("a && b").n_logic, 1);
        assert_eq!(code_of("(a && b) || (c && d)").n_logic, 2);
    }

    /// An error raised by the inner operator's *merge* is still the outer
    /// operator's left operand, so it has to be absorbed by the outer one.
    /// This is the `(1 && true) && false` shape.
    #[test]
    fn an_inner_merge_is_inside_the_outer_handler() {
        let code = code_of("(a && b) && c");
        let inner_merge = code
            .instructions()
            .find(|(_, op, _)| *op == OpCode::AndMerge)
            .map(|(pc, _, _)| pc)
            .expect("the inner `&&` merges first");
        let handler = code
            .handler_for(inner_merge)
            .expect("the inner merge is covered");
        assert_eq!(handler.start, 0, "{}", code.disassemble());
    }

    fn count(code: &CelCode, op: OpCode) -> usize {
        opcodes(code).iter().filter(|o| **o == op).count()
    }

    /// `map` and `filter` push into the accumulator instead of rebuilding it.
    ///
    /// The property is stated as an opcode census rather than a timing, because
    /// the cost being removed is a *copy* and the shape is what decides whether
    /// it happens. `Add` is the discriminator: the general lowering emits one
    /// for the step's `@result + [x]`, and the loop counter advances through
    /// `IncLocal` rather than through an addition, so no `Add` at all is
    /// exactly the claim that the concatenation is gone. The bodies here are
    /// chosen not to contain a `+` of their own, which is what keeps the
    /// census a statement about the accumulator.
    #[test]
    fn map_and_filter_append_rather_than_concatenate() {
        for source in [
            "xs.map(x, x * 2)",
            "xs.filter(x, x > 1)",
            "xs.map(x, x > 1, x * 10)",
        ] {
            let code = code_of(source);
            assert_eq!(
                count(&code, OpCode::Add),
                0,
                "{source} still concatenates:\n{}",
                code.disassemble()
            );
            assert_eq!(
                count(&code, OpCode::ListAppend),
                1,
                "{source} should append exactly once per iteration:\n{}",
                code.disassemble()
            );
            // The accumulator lives on the operand stack, so nothing reads it
            // back per iteration; the one store is the finish after the loop.
            assert_eq!(
                count(&code, OpCode::NewList),
                1,
                "{source} should build one accumulator:\n{}",
                code.disassemble()
            );
        }
    }

    /// The macros that are NOT this shape keep the general lowering, so the
    /// recogniser cannot be quietly widened into something that breaks them.
    ///
    /// `all` and `exists` break early on the accumulator, and `exists_one`
    /// counts into an int; none of the three may reach the append path.
    #[test]
    fn breaking_and_counting_macros_keep_the_general_lowering() {
        for source in [
            "xs.all(x, x > 0)",
            "xs.exists(x, x > 0)",
            "xs.exists_one(x, x > 0)",
        ] {
            let code = code_of(source);
            assert_eq!(
                count(&code, OpCode::ListAppend),
                0,
                "{source} is not an appending comprehension:\n{}",
                code.disassemble()
            );
        }
    }

    /// The accumulator's name is not in scope inside the loop.
    ///
    /// That is the precondition the whole shape rests on -- the accumulator is
    /// a builder on the operand stack, so a per-iteration read could not see
    /// it. `@result` is not parseable, so the only way to state this is to
    /// build the tree; the check is that the name compiles to a context lookup
    /// (`LoadVar`) and not to a slot, which is the walker's behaviour too.
    #[test]
    fn the_accumulator_is_not_readable_from_the_loop_body() {
        let Expr::Comprehension(base) = code_source("xs.map(x, x)") else {
            panic!("the macro expands to a comprehension");
        };
        let mut comp = *base;
        let accu = comp.accu_var.clone();

        // `@result + [@result]`: the element reads the accumulator.
        let Expr::Call(step) = &mut comp.loop_step.expr else {
            panic!("`map` steps through `+`");
        };
        let Expr::List(appended) = &mut step.args[1].expr else {
            panic!("`map` appends a one element literal");
        };
        appended.elements[0] = IdedExpr {
            id: 99,
            expr: Expr::Ident(accu.clone()),
        };

        let code = compile(&IdedExpr {
            id: 0,
            expr: Expr::Comprehension(Box::new(comp)),
        })
        .expect("compiles");

        let looked_up: Vec<&str> = code
            .instructions()
            .filter(|(_, op, _)| *op == OpCode::LoadVar)
            .map(|(_, _, operands)| code.name(NameId(operands[0])).unwrap())
            .collect();
        assert!(
            looked_up.contains(&accu.as_str()),
            "the accumulator must resolve as a free variable inside the loop, \
             not as the slot it only occupies after it: {looked_up:?}\n{}",
            code.disassemble()
        );
    }

    fn parse_expr(source: &str) -> IdedExpr {
        Parser::default().parse(source).expect("parses")
    }

    fn code_source(source: &str) -> Expr {
        parse_expr(source).expr
    }
}
