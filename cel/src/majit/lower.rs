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
//!   * arithmetic `+ - *`, unary `-`,
//!   * comparisons `>= > <= < == !=`,
//!   * boolean `&& || !` (non-short-circuit, correct for the pure int/bool
//!     domain where operands cannot raise).
//!
//! **Schema assumption**: every slot is assumed to carry an `int`/`bool` value.
//! A CEL expression comparing a slot bound to a `double`/`uint`/`string` at
//! runtime is outside this subset; a real integration guards on the context
//! schema before electing the JIT (the PyPy-style shape guard). The lowering
//! itself is type-blind on slots.

use super::bytecode::*;
use crate::common::ast::operators as ops;
use crate::common::ast::{CallExpr, Expr, IdedExpr, LiteralValue};
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
}

struct LowerCtx {
    body: Vec<i64>,
    next_reg: usize,
    slots: Vec<SlotInfo>,
    slot_map: HashMap<String, usize>,
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
        Expr::Ident(_) | Expr::Select(_) => {
            let path = resolve_path(e)?;
            Ok(ctx.slot(path))
        }
        Expr::Call(call) => compile_call(ctx, call),
        Expr::Comprehension(_) => Err(LowerError::unsupported("comprehension")),
        Expr::List(_) => Err(LowerError::unsupported("list literal")),
        Expr::Map(_) => Err(LowerError::unsupported("map literal")),
        Expr::Struct(_) => Err(LowerError::unsupported("struct literal")),
        Expr::Unspecified => Err(LowerError::unsupported("unspecified expr")),
    }
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
