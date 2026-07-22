//! Lowering from the CEL AST ([`IdedExpr`]) to the flat `i64`-word bytecode of
//! [`super::bytecode`], for the majit-traceable subset.
//!
//! Supported subset (everything else returns [`LowerError::Unsupported`], the
//! signal for the caller to fall back to the stock tree-walking evaluator):
//!   * `Int`/`Boolean` literals,
//!   * scalar variables (`Ident`) and constant field-access chains (`Select`),
//!     each resolved to an input **slot** (a register loaded from the row's
//!     field value) — the compile-time replacement for the runtime BTreeMap /
//!     HashMap lookups the tree-walker performs,
//!   * arithmetic `+ - * / %`, unary `-` (division assumes a nonzero,
//!     non-`INT_MIN`/`-1` divisor domain — the schema/domain shape guard),
//!   * comparisons `>= > <= < == !=`,
//!   * boolean `&& || !` (non-short-circuit, correct for the pure int/bool
//!     domain where operands cannot raise),
//!   * `all` / `exists` / `exists_one` comprehensions over a **literal** list
//!     (green-constant length), unrolled into a straight-line fold. `map` /
//!     `filter` build a list and stay out of the int subset.
//!
//! **Schema assumption**: every slot is assumed to carry an `int`/`bool` value.
//! A CEL expression comparing a slot bound to a `double`/`uint`/`string` at
//! runtime is outside this subset; a real integration guards on the context
//! schema before electing the JIT (the PyPy-style shape guard). The lowering
//! itself is type-blind on slots.

use super::bytecode::*;
use crate::common::ast::operators as ops;
use crate::common::ast::{CallExpr, ComprehensionExpr, Expr, IdedExpr, LiteralValue};
use std::collections::HashMap;

/// Reason a CEL expression could not be lowered to the traceable subset.
#[derive(Debug, Clone)]
pub struct LowerError {
    pub reason: String,
}

impl LowerError {
    fn unsupported(what: impl Into<String>) -> Self {
        LowerError {
            reason: what.into(),
        }
    }
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unsupported for majit lowering: {}", self.reason)
    }
}

/// One input slot: a variable path resolved to a fixed register that the row's
/// field value is loaded into before the body runs.
#[derive(Debug, Clone)]
pub struct SlotInfo {
    /// Dotted variable path, e.g. `account.balance`.
    pub path: String,
    /// Register the value is loaded into.
    pub reg: usize,
}

/// A CEL expression compiled to bytecode. The `body` computes the result into
/// `result_reg` given that every slot register already holds its row value;
/// [`Lowered::program_for`] prepends the per-row slot loads and a `RETURN`.
#[derive(Debug, Clone)]
pub struct Lowered {
    /// Straight-line ops for literals + operators (no slot loads, no return).
    pub body: Vec<i64>,
    /// Register holding the expression's result.
    pub result_reg: usize,
    /// Total registers the program uses.
    pub num_regs: usize,
    /// Input slots in first-encounter order.
    pub slots: Vec<SlotInfo>,
}

impl Lowered {
    /// Build a complete, runnable straight-line program for a single input row:
    /// `LOAD_CONST` each slot value into its register, then the body, then
    /// `RETURN result_reg`. `inputs` is aligned to [`Lowered::slots`].
    pub fn program_for(&self, inputs: &[i64]) -> Vec<i64> {
        assert_eq!(
            inputs.len(),
            self.slots.len(),
            "program_for: input arity {} != slot count {}",
            inputs.len(),
            self.slots.len()
        );
        let mut p = Vec::with_capacity(self.slots.len() * 3 + self.body.len() + 2);
        for (slot, &v) in self.slots.iter().zip(inputs) {
            p.push(OP_LOAD_CONST);
            p.push(v);
            p.push(slot.reg as i64);
        }
        p.extend_from_slice(&self.body);
        p.push(OP_RETURN);
        p.push(self.result_reg as i64);
        p
    }

    /// Build a **columnar batch** program: for each row `i` in `0..n`, load each
    /// slot's value `col_k[i]` from its data column via a red-index `raw_load`
    /// (`OP_COL_LOAD`), run the body, and accumulate the result into a running
    /// sum. `bases[k]` is the base address of slot `k`'s `i64` column buffer,
    /// aligned to [`Lowered::slots`]. Returns `(program, total_regs)`.
    ///
    /// The loop machinery (`i`, `acc`, `n`, `one`, `stride`) and the per-slot
    /// column base pointers occupy registers **above** `num_regs`, so the body's
    /// slot/temp registers are untouched by the loop bookkeeping. Every base is
    /// a loop-invariant loaded once into the register file — the shape a compiled
    /// trace can read real context columns through (a scalar-state-field base
    /// trips `VirtualStatesCantMatch` at loop close; see [`super::bytecode`]).
    ///
    /// The back-edge is a do-while (`OP_JUMP_IF_ABOVE` after the body), so the
    /// body runs at least once; callers must pass `n >= 1`.
    pub fn batch_sum_program(&self, bases: &[i64], n: i64) -> (Vec<i64>, usize) {
        assert_eq!(
            bases.len(),
            self.slots.len(),
            "batch_sum_program: base arity {} != slot count {}",
            bases.len(),
            self.slots.len()
        );
        assert!(n >= 1, "batch_sum_program: n must be >= 1 (do-while back-edge)");
        let m = self.num_regs; // first machinery register
        let (r_i, r_acc, r_n, r_one, r_stride, r_ea) = (m, m + 1, m + 2, m + 3, m + 4, m + 5);
        let r_base0 = m + 6;
        let total_regs = r_base0 + self.slots.len();

        let mut p = Vec::new();
        let load_const = |p: &mut Vec<i64>, imm: i64, dst: usize| {
            p.extend_from_slice(&[OP_LOAD_CONST, imm, dst as i64]);
        };
        load_const(&mut p, 0, r_i);
        load_const(&mut p, 0, r_acc);
        load_const(&mut p, n, r_n);
        load_const(&mut p, 1, r_one);
        load_const(&mut p, 8, r_stride);
        for (k, &base) in bases.iter().enumerate() {
            load_const(&mut p, base, r_base0 + k);
        }

        let body_pc = p.len();
        // ea = i * 8 (byte offset of row i in an i64 column)
        p.extend_from_slice(&[OP_MUL, r_i as i64, r_stride as i64, r_ea as i64]);
        // slot_k = *(base_k + ea)   — the red-index columnar read
        for (k, slot) in self.slots.iter().enumerate() {
            p.extend_from_slice(&[
                OP_COL_LOAD,
                (r_base0 + k) as i64,
                r_ea as i64,
                slot.reg as i64,
            ]);
        }
        p.extend_from_slice(&self.body);
        // acc += result; i += 1; if n > i goto @body
        p.extend_from_slice(&[OP_ADD, r_acc as i64, self.result_reg as i64, r_acc as i64]);
        p.extend_from_slice(&[OP_ADD, r_i as i64, r_one as i64, r_i as i64]);
        p.extend_from_slice(&[OP_JUMP_IF_ABOVE, r_n as i64, r_i as i64, body_pc as i64]);
        p.extend_from_slice(&[OP_RETURN, r_acc as i64]);
        (p, total_regs)
    }
}

struct LowerCtx {
    body: Vec<i64>,
    next_reg: usize,
    slots: Vec<SlotInfo>,
    slot_map: HashMap<String, usize>,
    /// Comprehension-bound names (`iter_var`, `accu_var`) mapped to the register
    /// holding their current value. Checked before slot resolution so a bound
    /// variable is a computed value, not an input slot.
    locals: HashMap<String, usize>,
}

impl LowerCtx {
    fn fresh(&mut self) -> usize {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    fn slot(&mut self, path: String) -> usize {
        if let Some(&r) = self.slot_map.get(&path) {
            return r;
        }
        let r = self.fresh();
        self.slot_map.insert(path.clone(), r);
        self.slots.push(SlotInfo { path, reg: r });
        r
    }
}

/// Lower a CEL expression to bytecode, or report why it is out of subset.
pub fn lower(expr: &IdedExpr) -> Result<Lowered, LowerError> {
    let mut ctx = LowerCtx {
        body: Vec::new(),
        next_reg: 0,
        slots: Vec::new(),
        slot_map: HashMap::new(),
        locals: HashMap::new(),
    };
    let result_reg = compile(&mut ctx, expr)?;
    let num_regs = ctx.next_reg;
    Ok(Lowered {
        body: ctx.body,
        result_reg,
        num_regs,
        slots: ctx.slots,
    })
}

fn compile(ctx: &mut LowerCtx, e: &IdedExpr) -> Result<usize, LowerError> {
    match &e.expr {
        Expr::Literal(lit) => compile_literal(ctx, lit),
        Expr::Ident(name) => {
            if let Some(&r) = ctx.locals.get(name) {
                Ok(r)
            } else {
                Ok(ctx.slot(name.clone()))
            }
        }
        Expr::Select(_) => {
            let path = resolve_path(e)?;
            let root = path.split('.').next().unwrap_or_default();
            if ctx.locals.contains_key(root) {
                return Err(LowerError::unsupported("field access on comprehension variable"));
            }
            Ok(ctx.slot(path))
        }
        Expr::Call(call) => compile_call(ctx, call),
        Expr::Comprehension(comp) => compile_comprehension(ctx, comp),
        Expr::List(_) => Err(LowerError::unsupported("list literal")),
        Expr::Map(_) => Err(LowerError::unsupported("map literal")),
        Expr::Struct(_) => Err(LowerError::unsupported("struct literal")),
        Expr::Unspecified => Err(LowerError::unsupported("unspecified expr")),
    }
}

/// Green-length unroll of a comprehension (`all` / `exists` / `exists_one`).
///
/// Only a literal-list range is lowerable: its length and elements are known at
/// compile time (the PyPy-style green-constant trip count), so the loop unrolls
/// into a straight-line fold. Each element binds `iter_var`; `accu_var` threads
/// the accumulator through `loop_step`. `loop_cond`'s short-circuit is dropped —
/// with no side effects or raising operands the eager fold has the same value.
///
/// `map` / `filter` accumulate a list: their `[]` `accu_init` (or list-valued
/// `loop_step`) lowers to a list literal and bails, so they fall back naturally.
fn compile_comprehension(
    ctx: &mut LowerCtx,
    comp: &ComprehensionExpr,
) -> Result<usize, LowerError> {
    if comp.iter_var2.is_some() {
        return Err(LowerError::unsupported("two-variable comprehension"));
    }
    let elements = match &comp.iter_range.expr {
        Expr::List(list) => list.elements.clone(),
        _ => return Err(LowerError::unsupported("comprehension over non-literal range")),
    };

    // Shadow-safe: save any enclosing bindings, restore them on the way out.
    let prev_iter = ctx.locals.remove(&comp.iter_var);
    let prev_accu = ctx.locals.remove(&comp.accu_var);

    let mut accu = compile(ctx, &comp.accu_init)?;
    for elem in &elements {
        let x_reg = compile(ctx, elem)?;
        ctx.locals.insert(comp.iter_var.clone(), x_reg);
        ctx.locals.insert(comp.accu_var.clone(), accu);
        accu = compile(ctx, &comp.loop_step)?;
    }
    ctx.locals.insert(comp.accu_var.clone(), accu);
    let result = compile(ctx, &comp.result)?;

    ctx.locals.remove(&comp.iter_var);
    ctx.locals.remove(&comp.accu_var);
    if let Some(r) = prev_iter {
        ctx.locals.insert(comp.iter_var.clone(), r);
    }
    if let Some(r) = prev_accu {
        ctx.locals.insert(comp.accu_var.clone(), r);
    }
    Ok(result)
}

fn compile_literal(ctx: &mut LowerCtx, lit: &LiteralValue) -> Result<usize, LowerError> {
    let imm: i64 = match lit {
        LiteralValue::Int(i) => i.into_inner(),
        LiteralValue::Boolean(b) => b.into_inner() as i64,
        LiteralValue::UInt(_) => return Err(LowerError::unsupported("uint literal")),
        LiteralValue::Double(_) => return Err(LowerError::unsupported("double literal")),
        LiteralValue::String(_) => return Err(LowerError::unsupported("string literal")),
        LiteralValue::Bytes(_) => return Err(LowerError::unsupported("bytes literal")),
        LiteralValue::Null => return Err(LowerError::unsupported("null literal")),
    };
    let r = ctx.fresh();
    ctx.body.push(OP_LOAD_CONST);
    ctx.body.push(imm);
    ctx.body.push(r as i64);
    Ok(r)
}

/// Extract a plain integer literal, if that is what `e` is.
fn as_int_literal(e: &IdedExpr) -> Option<i64> {
    match &e.expr {
        Expr::Literal(LiteralValue::Int(i)) => Some(i.into_inner()),
        _ => None,
    }
}

/// Resolve an `Ident` or a constant `Select` chain to a dotted variable path.
fn resolve_path(e: &IdedExpr) -> Result<String, LowerError> {
    match &e.expr {
        Expr::Ident(name) => Ok(name.clone()),
        Expr::Select(sel) if !sel.test => {
            let base = resolve_path(&sel.operand)?;
            Ok(format!("{base}.{}", sel.field))
        }
        Expr::Select(_) => Err(LowerError::unsupported("has() presence test")),
        _ => Err(LowerError::unsupported("non-static field access")),
    }
}

fn compile_call(ctx: &mut LowerCtx, call: &CallExpr) -> Result<usize, LowerError> {
    if call.target.is_some() {
        return Err(LowerError::unsupported(format!(
            "method call `{}`",
            call.func_name
        )));
    }
    let name = call.func_name.as_str();

    // ternary `c ? t : f` — the int/bool subset has no side effects or raising
    // operands, so eager-evaluate both arms and select (matches CEL's value).
    if name == ops::CONDITIONAL {
        if call.args.len() != 3 {
            return Err(LowerError::unsupported("_?_:_ arity"));
        }
        let c = compile(ctx, &call.args[0])?;
        let t = compile(ctx, &call.args[1])?;
        let f = compile(ctx, &call.args[2])?;
        let d = ctx.fresh();
        ctx.body
            .extend_from_slice(&[OP_SELECT, c as i64, t as i64, f as i64, d as i64]);
        return Ok(d);
    }

    // constant-index access `base[k]` (k an int literal) — resolves to a slot
    // just like member access (`base[k]` path). A non-literal index is a
    // data-dependent (red) index, outside the subset.
    if name == ops::INDEX {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported("_[_] arity"));
        }
        let base = resolve_path(&call.args[0])?;
        let idx = as_int_literal(&call.args[1])
            .ok_or_else(|| LowerError::unsupported("non-constant index"))?;
        return Ok(ctx.slot(format!("{base}[{idx}]")));
    }

    // n-ary boolean fold (`a && b && c` may be binary-nested or n-ary).
    if name == ops::LOGICAL_AND || name == ops::LOGICAL_OR {
        if call.args.len() < 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let op = if name == ops::LOGICAL_AND {
            OP_AND
        } else {
            OP_OR
        };
        let mut acc = compile(ctx, &call.args[0])?;
        for arg in &call.args[1..] {
            let b = compile(ctx, arg)?;
            let d = ctx.fresh();
            ctx.body
                .extend_from_slice(&[op, acc as i64, b as i64, d as i64]);
            acc = d;
        }
        return Ok(acc);
    }

    let binop = match name {
        ops::ADD => Some(OP_ADD),
        ops::SUBSTRACT => Some(OP_SUB),
        ops::MULTIPLY => Some(OP_MUL),
        ops::DIVIDE => Some(OP_DIV),
        ops::MODULO => Some(OP_MOD),
        ops::GREATER_EQUALS => Some(OP_GE),
        ops::GREATER => Some(OP_GT),
        ops::LESS_EQUALS => Some(OP_LE),
        ops::LESS => Some(OP_LT),
        ops::EQUALS => Some(OP_EQ),
        ops::NOT_EQUALS => Some(OP_NE),
        _ => None,
    };
    if let Some(op) = binop {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let a = compile(ctx, &call.args[0])?;
        let b = compile(ctx, &call.args[1])?;
        let d = ctx.fresh();
        ctx.body
            .extend_from_slice(&[op, a as i64, b as i64, d as i64]);
        return Ok(d);
    }

    match name {
        ops::LOGICAL_NOT => {
            if call.args.len() != 1 {
                return Err(LowerError::unsupported("!_ arity"));
            }
            let a = compile(ctx, &call.args[0])?;
            let d = ctx.fresh();
            ctx.body.extend_from_slice(&[OP_NOT, a as i64, d as i64]);
            Ok(d)
        }
        ops::NEGATE => {
            if call.args.len() != 1 {
                return Err(LowerError::unsupported("-_ arity"));
            }
            let a = compile(ctx, &call.args[0])?;
            let d = ctx.fresh();
            ctx.body.extend_from_slice(&[OP_NEG, a as i64, d as i64]);
            Ok(d)
        }
        _ => Err(LowerError::unsupported(format!("call `{name}`"))),
    }
}
