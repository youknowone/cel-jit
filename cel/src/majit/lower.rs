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

fn as_string_literal(e: &IdedExpr) -> Option<&str> {
    match &e.expr {
        Expr::Literal(LiteralValue::String(s)) => Some(s.inner()),
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

/// The bank a slot / sub-expression lives in for the two-bank machine. `Bool`
/// shares the int bank (`0`/`1`); `Float` lives in the parallel `fregs` bank.
/// This is the *shape* a compiled float trace guards on (the caller declares
/// which context columns are `double` via a [`Schema`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValType {
    Int,
    /// Unsigned 64-bit. Shares the int register file (the raw bit pattern), so
    /// storage, column loads, moves, add/sub/mul and eq/ne reuse the int ops;
    /// only ordering comparisons differ (unsigned `OP_ULT`/`OP_ULE`).
    UInt,
    Float,
    /// A string, carried as an `i64` content hash ([`intern_hash`]) in the int
    /// register file. Only equality is defined: a hash compare (`OP_EQ`/`OP_NE`)
    /// equals a content compare bit-for-bit once the batch builder has verified
    /// the hash is injective over the strings present (a collision bails).
    /// Ordering, arithmetic, and any other string op fall back to the
    /// tree-walker.
    Str,
    /// A `timestamp`, carried as `i64` nanoseconds since the Unix epoch in the
    /// int register file. i64-nanos ordering equals the chronological order the
    /// tree-walker compares, so all six comparisons lower to the signed int ops.
    /// Arithmetic (ts±duration, ts−ts) and any timestamp outside the i64-nanos
    /// range fall back to the tree-walker.
    Timestamp,
    /// A `duration`, carried as `i64` nanoseconds in the int register file. Like
    /// [`ValType::Timestamp`], comparisons lower to the signed int ops; a
    /// timestamp vs duration comparison is NoSuchOverload and bails.
    Duration,
}

/// Stable content hash mapping a string to the `i64` id a [`ValType::Str`]
/// column and a string literal share. FNV-1a: deterministic across processes
/// (unlike a randomly-seeded [`std::hash`]), so a literal hashed at lowering
/// time and a column value hashed at batch-build time agree. Injectivity over
/// the strings actually present is checked by the batch builder, not assumed.
pub fn intern_hash(s: &str) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h as i64
}

/// Declared type of each input path. A path absent from the schema defaults to
/// [`ValType::Int`] (the int/bool domain of [`lower`]). A real integration
/// builds this from the context's variable types and guards on it before
/// electing the float JIT.
pub type Schema = HashMap<String, ValType>;

/// A typed register: a bank plus the index within that bank.
#[derive(Debug, Clone, Copy)]
struct TReg {
    bank: ValType,
    idx: usize,
}

/// One typed input slot for the two-bank machine: a path resolved to a register
/// in its bank, plus which bank ([`ValType`]) it is.
#[derive(Debug, Clone)]
pub struct SlotInfoF {
    /// Dotted variable path, e.g. `account.balance`.
    pub path: String,
    /// The bank this slot's column is read into.
    pub ty: ValType,
    /// Register index within [`SlotInfoF::ty`]'s bank.
    pub reg: usize,
}

/// A CEL expression compiled to two-bank bytecode. The result is an int
/// register (a bool/count/int-sum) or a float register (a float total), tracked
/// by [`LoweredF::result_bank`]. [`LoweredF::batch_sum_program`] prepends the
/// loop-invariant [`LoweredF::prelude`] and the per-row columnar loads, then
/// wraps [`LoweredF::body`] in the sum loop, accumulating into the matching
/// bank.
#[derive(Debug, Clone)]
pub struct LoweredF {
    /// Loop-invariant literal loads (int and `double` constants), hoisted to run
    /// **once** before the loop. Every literal a CEL expression references is a
    /// loop invariant, so it belongs here, not re-evaluated per row (PyPy hoists
    /// invariants out of the trace). Hoisting `double` constants also keeps
    /// `OP_LOAD_CONST_F`'s `f64::from_bits` out of the traced loop body.
    pub prelude: Vec<i64>,
    /// Per-row straight-line ops (no literal loads, no slot loads, no return).
    pub body: Vec<i64>,
    /// Bank the top-level result lives in. An int result accumulates into an int
    /// count/sum (`OP_RETURN`); a float result accumulates into a float total
    /// (`OP_RETURN_F`).
    pub result_bank: ValType,
    /// Register holding the result, within [`LoweredF::result_bank`].
    pub result_reg: usize,
    /// Int-bank register count the body uses.
    pub num_int_regs: usize,
    /// Float-bank register count the body uses.
    pub num_float_regs: usize,
    /// Input slots in first-encounter order.
    pub slots: Vec<SlotInfoF>,
    /// String literals the body compares against, as raw content. The batch
    /// builder hashes these with [`intern_hash`] and includes them in the
    /// injectivity check so a literal that collides with a distinct column
    /// string bails rather than miscompiles.
    pub str_literals: Vec<String>,
}

impl LoweredF {
    /// Build a **columnar batch** program over the two-bank machine: for each
    /// row `i` in `0..n`, load each slot's `col_k[i]` via a red-index `raw_load`
    /// (`OP_COL_LOAD` for int slots, `OP_COL_LOAD_F` for float slots), run the
    /// body, and accumulate the result into a running sum in the result's bank
    /// (int `r_acc` -> `OP_RETURN`, or float `f_acc` -> `OP_RETURN_F`). `bases[k]` is the
    /// base address of slot `k`'s column buffer (an `i64` pointer regardless of
    /// bank), aligned to [`LoweredF::slots`]. Returns
    /// `(program, total_int_regs, total_float_regs)`.
    ///
    /// The loop machinery (`i`, `acc`, `n`, `one`, `stride`, `ea`) and the
    /// per-slot base pointers live in the **int** bank above `num_int_regs`, so
    /// the body's registers are untouched. Every base is a loop-invariant int
    /// register (never a scalar state field, which would trip
    /// `VirtualStatesCantMatch` at loop close). The back-edge is a do-while, so
    /// callers must pass `n >= 1`.
    pub fn batch_sum_program(&self, bases: &[i64], n: i64) -> (Vec<i64>, usize, usize) {
        assert_eq!(
            bases.len(),
            self.slots.len(),
            "batch_sum_program: base arity {} != slot count {}",
            bases.len(),
            self.slots.len()
        );
        assert!(n >= 1, "batch_sum_program: n must be >= 1 (do-while back-edge)");
        let m = self.num_int_regs; // first int machinery register
        let (r_i, r_acc, r_n, r_one, r_stride, r_ea) = (m, m + 1, m + 2, m + 3, m + 4, m + 5);
        let r_base0 = m + 6;
        let total_int_regs = r_base0 + self.slots.len();
        // A float result accumulates into a float register above the body's
        // float bank; an int result uses the int `r_acc` and leaves the float
        // bank at the body's count. `f_acc` is unused when the result is int.
        let (f_acc, total_float_regs) = match self.result_bank {
            ValType::Float => (self.num_float_regs, self.num_float_regs + 1),
            ValType::Int | ValType::UInt | ValType::Str | ValType::Timestamp | ValType::Duration => (0, self.num_float_regs),
        };

        let mut p = Vec::new();
        let load_const = |p: &mut Vec<i64>, imm: i64, dst: usize| {
            p.extend_from_slice(&[OP_LOAD_CONST, imm, dst as i64]);
        };
        load_const(&mut p, 0, r_i);
        // Accumulator init, run once before the merge point. `OP_LOAD_CONST_F`'s
        // `f64::from_bits` must stay out of the traced loop body; here it is in
        // the setup (0.0 has zero bits).
        match self.result_bank {
            ValType::Int | ValType::UInt | ValType::Str | ValType::Timestamp | ValType::Duration => load_const(&mut p, 0, r_acc),
            ValType::Float => p.extend_from_slice(&[OP_LOAD_CONST_F, 0, f_acc as i64]),
        }
        load_const(&mut p, n, r_n);
        load_const(&mut p, 1, r_one);
        load_const(&mut p, 8, r_stride);
        for (k, &base) in bases.iter().enumerate() {
            load_const(&mut p, base, r_base0 + k);
        }
        // Loop-invariant literal loads, run once before the merge point.
        p.extend_from_slice(&self.prelude);

        let body_pc = p.len();
        // ea = i * 8 (byte offset of row i in an 8-byte column)
        p.extend_from_slice(&[OP_MUL, r_i as i64, r_stride as i64, r_ea as i64]);
        // slot_k = *(base_k + ea)   — the red-index columnar read, per bank
        for (k, slot) in self.slots.iter().enumerate() {
            let op = match slot.ty {
                ValType::Int | ValType::UInt | ValType::Str | ValType::Timestamp | ValType::Duration => OP_COL_LOAD,
                ValType::Float => OP_COL_LOAD_F,
            };
            p.extend_from_slice(&[op, (r_base0 + k) as i64, r_ea as i64, slot.reg as i64]);
        }
        p.extend_from_slice(&self.body);
        // acc += result (bank-matched); i += 1; if n > i goto @body; return acc.
        // The float accumulate is a loop-carried dependency, so the compiled
        // trace cannot reassociate it — the running total sums in row order, bit
        // for bit like the interpreter tiers.
        match self.result_bank {
            ValType::Int | ValType::UInt | ValType::Str | ValType::Timestamp | ValType::Duration => {
                p.extend_from_slice(&[OP_ADD, r_acc as i64, self.result_reg as i64, r_acc as i64])
            }
            ValType::Float => {
                p.extend_from_slice(&[OP_FADD, f_acc as i64, self.result_reg as i64, f_acc as i64])
            }
        }
        p.extend_from_slice(&[OP_ADD, r_i as i64, r_one as i64, r_i as i64]);
        p.extend_from_slice(&[OP_JUMP_IF_ABOVE, r_n as i64, r_i as i64, body_pc as i64]);
        match self.result_bank {
            ValType::Int | ValType::UInt | ValType::Str | ValType::Timestamp | ValType::Duration => p.extend_from_slice(&[OP_RETURN, r_acc as i64]),
            ValType::Float => p.extend_from_slice(&[OP_RETURN_F, f_acc as i64]),
        }
        (p, total_int_regs, total_float_regs)
    }
}

struct LowerCtxF<'s> {
    /// Loop-invariant literal loads, emitted here instead of into `body` so the
    /// batch builder can run them once before the loop.
    prelude: Vec<i64>,
    body: Vec<i64>,
    next_int: usize,
    next_float: usize,
    slots: Vec<SlotInfoF>,
    slot_map: HashMap<String, TReg>,
    locals: HashMap<String, TReg>,
    /// String literals referenced by the body, in first-encounter order. The
    /// batch builder folds these into the injectivity check alongside the
    /// [`ValType::Str`] column values.
    str_literals: Vec<String>,
    schema: &'s Schema,
}

impl LowerCtxF<'_> {
    fn fresh(&mut self, bank: ValType) -> TReg {
        let idx = match bank {
            // `Str` ids share the int register file (an `i64` content hash).
            ValType::Int | ValType::UInt | ValType::Str | ValType::Timestamp | ValType::Duration => {
                let r = self.next_int;
                self.next_int += 1;
                r
            }
            ValType::Float => {
                let r = self.next_float;
                self.next_float += 1;
                r
            }
        };
        TReg { bank, idx }
    }

    fn slot(&mut self, path: String) -> TReg {
        if let Some(&r) = self.slot_map.get(&path) {
            return r;
        }
        let ty = self.schema.get(&path).copied().unwrap_or(ValType::Int);
        let r = self.fresh(ty);
        self.slot_map.insert(path.clone(), r);
        self.slots.push(SlotInfoF { path, ty, reg: r.idx });
        r
    }
}

/// Lower a CEL expression to two-bank bytecode under a `schema` declaring which
/// paths are `double`, or report why it is out of subset. Same subset as
/// [`lower`] plus `double` literals/columns, but with per-bank register
/// allocation. A float-valued top-level result accumulates into a float total.
/// Same-bank float ternary lowers to a bit-mask FSELECT. Mixed int/float
/// arithmetic (no int->float cast for arithmetic), float modulo, and a
/// mixed-bank ternary still bail (the caller falls back to the tree-walker).
pub fn lower_typed(expr: &IdedExpr, schema: &Schema) -> Result<LoweredF, LowerError> {
    let mut ctx = LowerCtxF {
        prelude: Vec::new(),
        body: Vec::new(),
        next_int: 0,
        next_float: 0,
        slots: Vec::new(),
        slot_map: HashMap::new(),
        locals: HashMap::new(),
        str_literals: Vec::new(),
        schema,
    };
    let result = compile_t(&mut ctx, expr)?;
    // A string- or temporal-valued top-level result is not sum-reducible (the
    // batch loop accumulates an int count or a float total); such an expression
    // bails to the tree-walker rather than accumulating content hashes / nanos.
    if matches!(
        result.bank,
        ValType::Str | ValType::Timestamp | ValType::Duration
    ) {
        return Err(LowerError::unsupported("string/temporal-valued top-level result"));
    }
    Ok(LoweredF {
        prelude: ctx.prelude,
        body: ctx.body,
        result_bank: result.bank,
        result_reg: result.idx,
        num_int_regs: ctx.next_int,
        num_float_regs: ctx.next_float,
        slots: ctx.slots,
        str_literals: ctx.str_literals,
    })
}

fn compile_t(ctx: &mut LowerCtxF, e: &IdedExpr) -> Result<TReg, LowerError> {
    match &e.expr {
        Expr::Literal(lit) => compile_literal_t(ctx, lit),
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
        Expr::Call(call) => compile_call_t(ctx, call),
        Expr::Comprehension(comp) => compile_comprehension_t(ctx, comp),
        Expr::List(_) => Err(LowerError::unsupported("list literal")),
        Expr::Map(_) => Err(LowerError::unsupported("map literal")),
        Expr::Struct(_) => Err(LowerError::unsupported("struct literal")),
        Expr::Unspecified => Err(LowerError::unsupported("unspecified expr")),
    }
}

fn compile_literal_t(ctx: &mut LowerCtxF, lit: &LiteralValue) -> Result<TReg, LowerError> {
    // Literals are loop invariants: emit their loads into the prelude so the
    // batch builder runs them once, not per row.
    match lit {
        LiteralValue::Int(i) => {
            let r = ctx.fresh(ValType::Int);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST, i.into_inner(), r.idx as i64]);
            Ok(r)
        }
        LiteralValue::Boolean(b) => {
            let r = ctx.fresh(ValType::Int);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST, b.into_inner() as i64, r.idx as i64]);
            Ok(r)
        }
        LiteralValue::Double(f) => {
            let r = ctx.fresh(ValType::Float);
            // The f64 travels as its raw i64 bits; the VM reloads with
            // `f64::from_bits`, so the constant is bit-exact.
            ctx.prelude.extend_from_slice(&[
                OP_LOAD_CONST_F,
                f.into_inner().to_bits() as i64,
                r.idx as i64,
            ]);
            Ok(r)
        }
        LiteralValue::UInt(u) => {
            let r = ctx.fresh(ValType::UInt);
            // The u64 travels as its raw i64 bit pattern in the int register file.
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST, u.into_inner() as i64, r.idx as i64]);
            Ok(r)
        }
        LiteralValue::String(s) => {
            // A string literal is a loop invariant: fold it to its content hash
            // and load that `i64` id once in the prelude. Record the raw content
            // so the batch builder can check the hash against the column strings.
            let r = ctx.fresh(ValType::Str);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST, intern_hash(s.inner()), r.idx as i64]);
            ctx.str_literals.push(s.inner().to_string());
            Ok(r)
        }
        LiteralValue::Bytes(_) => Err(LowerError::unsupported("bytes literal")),
        LiteralValue::Null => Err(LowerError::unsupported("null literal")),
    }
}

/// Emit a `double` constant load into the prelude and return its float reg.
fn emit_float_const(ctx: &mut LowerCtxF, v: f64) -> TReg {
    let r = ctx.fresh(ValType::Float);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST_F, v.to_bits() as i64, r.idx as i64]);
    r
}

/// The stdlib's receiver-only temporal accessors (`common/types/duration.rs`
/// and `common/types/timestamp.rs`). All are registered with
/// `add_member_overload` and take no arguments; the first four have both a
/// `duration` overload (a scaled count) and a `timestamp` one (a clock field).
const TEMPORAL_ACCESSORS: &[&str] = &[
    "getHours",
    "getMinutes",
    "getSeconds",
    "getMilliseconds",
    "getDayOfWeek",
    "getFullYear",
    "getMonth",
    "getDate",
    "getDayOfMonth",
    "getDayOfYear",
];

/// Emit a loop-invariant int constant load into the prelude and return its reg.
fn emit_int_const(ctx: &mut LowerCtxF, v: i64) -> TReg {
    let r = ctx.fresh(ValType::Int);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST, v, r.idx as i64]);
    r
}

/// Emit a three-address int-bank op `dst = a <op> b` into the body.
fn emit_int_bin(ctx: &mut LowerCtxF, op: i64, a: TReg, b: TReg) -> TReg {
    let d = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[op, a.idx as i64, b.idx as i64, d.idx as i64]);
    d
}

/// `dst = a <op> k` for a green constant `k` (its load is hoisted to the prelude).
fn emit_int_bin_k(ctx: &mut LowerCtxF, op: i64, a: TReg, k: i64) -> TReg {
    let kr = emit_int_const(ctx, k);
    emit_int_bin(ctx, op, a, kr)
}

/// Nanoseconds in one day — the scale between an instant and its calendar day.
const NANOS_PER_DAY: i64 = 86_400_000_000_000;

/// Split an i64-nanoseconds instant into whole days since the Unix epoch and the
/// nanoseconds-of-day remainder in `[0, NANOS_PER_DAY)`.
///
/// The day count is FLOORED, so a pre-epoch instant lands on the day below and
/// its remainder stays non-negative — that is what a calendar field means.
/// `OP_DIV`/`OP_MOD` truncate toward zero, so the negative case is corrected
/// with the `r < 0` flag; a comparison already yields 0/1, so the correction is
/// plain arithmetic with no branch (the traced loop lowers no control flow).
fn emit_days_and_nanos_of_day(ctx: &mut LowerCtxF, ts: TReg) -> (TReg, TReg) {
    let q = emit_int_bin_k(ctx, OP_DIV, ts, NANOS_PER_DAY);
    let r = emit_int_bin_k(ctx, OP_MOD, ts, NANOS_PER_DAY);
    let neg = emit_int_bin_k(ctx, OP_LT, r, 0);
    let days = emit_int_bin(ctx, OP_SUB, q, neg);
    let back = emit_int_bin_k(ctx, OP_MUL, neg, NANOS_PER_DAY);
    let nanos_of_day = emit_int_bin(ctx, OP_ADD, r, back);
    (days, nanos_of_day)
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 to
/// `(year, month 1-12, day 1-31)`.
///
/// Every division below has a non-negative dividend, so `OP_DIV`'s truncation
/// is the floor the algorithm calls for: an i64-nanosecond instant only spans
/// ~1678-2262, which keeps `days` inside ±106752 and `z = days + 719468` inside
/// [612716, 826220]. A timestamp outside that range cannot exist in this VM —
/// the column payload is i64 nanos.
fn emit_civil_from_days(ctx: &mut LowerCtxF, days: TReg) -> (TReg, TReg, TReg) {
    let z = emit_int_bin_k(ctx, OP_ADD, days, 719_468);
    let era = emit_int_bin_k(ctx, OP_DIV, z, 146_097);
    let era_days = emit_int_bin_k(ctx, OP_MUL, era, 146_097);
    let doe = emit_int_bin(ctx, OP_SUB, z, era_days);

    // yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365
    let by_1460 = emit_int_bin_k(ctx, OP_DIV, doe, 1_460);
    let by_36524 = emit_int_bin_k(ctx, OP_DIV, doe, 36_524);
    let by_146096 = emit_int_bin_k(ctx, OP_DIV, doe, 146_096);
    let t1 = emit_int_bin(ctx, OP_SUB, doe, by_1460);
    let t2 = emit_int_bin(ctx, OP_ADD, t1, by_36524);
    let t3 = emit_int_bin(ctx, OP_SUB, t2, by_146096);
    let yoe = emit_int_bin_k(ctx, OP_DIV, t3, 365);
    let era400 = emit_int_bin_k(ctx, OP_MUL, era, 400);
    let year_of_era = emit_int_bin(ctx, OP_ADD, yoe, era400);

    // doy = doe - (365*yoe + yoe/4 - yoe/100)   (days since 1 March)
    let y365 = emit_int_bin_k(ctx, OP_MUL, yoe, 365);
    let y4 = emit_int_bin_k(ctx, OP_DIV, yoe, 4);
    let y100 = emit_int_bin_k(ctx, OP_DIV, yoe, 100);
    let s1 = emit_int_bin(ctx, OP_ADD, y365, y4);
    let s2 = emit_int_bin(ctx, OP_SUB, s1, y100);
    let doy = emit_int_bin(ctx, OP_SUB, doe, s2);

    // mp = (5*doy + 2)/153 ; day = doy - (153*mp + 2)/5 + 1
    let d5 = emit_int_bin_k(ctx, OP_MUL, doy, 5);
    let d5p2 = emit_int_bin_k(ctx, OP_ADD, d5, 2);
    let mp = emit_int_bin_k(ctx, OP_DIV, d5p2, 153);
    let m153 = emit_int_bin_k(ctx, OP_MUL, mp, 153);
    let m153p2 = emit_int_bin_k(ctx, OP_ADD, m153, 2);
    let month_start = emit_int_bin_k(ctx, OP_DIV, m153p2, 5);
    let dm = emit_int_bin(ctx, OP_SUB, doy, month_start);
    let day = emit_int_bin_k(ctx, OP_ADD, dm, 1);

    // month = mp + (mp < 10 ? 3 : -9), written as mp + 3 - 12*(mp >= 10) so the
    // select is arithmetic on a 0/1 comparison rather than a branch.
    let ge10 = emit_int_bin_k(ctx, OP_GE, mp, 10);
    let mp3 = emit_int_bin_k(ctx, OP_ADD, mp, 3);
    let wrap = emit_int_bin_k(ctx, OP_MUL, ge10, 12);
    let month = emit_int_bin(ctx, OP_SUB, mp3, wrap);

    // The era year starts in March, so January and February belong to the next
    // calendar year: year = year_of_era + (month <= 2).
    let le2 = emit_int_bin_k(ctx, OP_LE, month, 2);
    let year = emit_int_bin(ctx, OP_ADD, year_of_era, le2);
    (year, month, day)
}

/// Days since the Unix epoch of 1 January of `year` — Hinnant's
/// `days_from_civil(year, 1, 1)`, specialised: for month 1 the March-based
/// `doy` term `(153*(m+9) + 2)/5 + d - 1` folds to the constant 306 (1 March to
/// the following 1 January). Used to turn an absolute day count into a
/// day-of-year. `year - 1` is positive over the representable range, so
/// truncation is again the floor the algorithm wants.
fn emit_days_of_jan1(ctx: &mut LowerCtxF, year: TReg) -> TReg {
    let y = emit_int_bin_k(ctx, OP_SUB, year, 1);
    let era = emit_int_bin_k(ctx, OP_DIV, y, 400);
    let era400 = emit_int_bin_k(ctx, OP_MUL, era, 400);
    let yoe = emit_int_bin(ctx, OP_SUB, y, era400);
    let y365 = emit_int_bin_k(ctx, OP_MUL, yoe, 365);
    let y4 = emit_int_bin_k(ctx, OP_DIV, yoe, 4);
    let y100 = emit_int_bin_k(ctx, OP_DIV, yoe, 100);
    let s1 = emit_int_bin(ctx, OP_ADD, y365, y4);
    let s2 = emit_int_bin(ctx, OP_SUB, s1, y100);
    let doe = emit_int_bin_k(ctx, OP_ADD, s2, 306);
    let era_days = emit_int_bin_k(ctx, OP_MUL, era, 146_097);
    let abs = emit_int_bin(ctx, OP_ADD, era_days, doe);
    emit_int_bin_k(ctx, OP_SUB, abs, 719_468)
}

/// Widen an int-bank value to a fresh float reg via a per-row `int as f64` cast
/// (`OP_I2F` -> `cast_int_to_float`). Emitted into the body: unlike a literal
/// (folded to a prelude constant), a data-dependent int is cast per row.
fn emit_i2f(ctx: &mut LowerCtxF, src: TReg) -> TReg {
    debug_assert_eq!(src.bank, ValType::Int, "emit_i2f: source must be int-banked");
    let r = ctx.fresh(ValType::Float);
    ctx.body
        .extend_from_slice(&[OP_I2F, src.idx as i64, r.idx as i64]);
    r
}

/// Compile the two operands of a comparison, promoting a bare `int` literal to
/// a `double` constant when its peer is float. This constant-folds the
/// tree-walker's `int as f64` promotion (CEL compares mixed numeric operands by
/// widening the int to `f64`, symmetric for either side), keeping both operands
/// in the float bank without a per-row cast the traced loop cannot lower. A
/// non-literal int vs float stays mixed and the caller bails.
fn compile_cmp_operands(
    ctx: &mut LowerCtxF,
    e0: &IdedExpr,
    e1: &IdedExpr,
) -> Result<(TReg, TReg), LowerError> {
    match (as_int_literal(e0), as_int_literal(e1)) {
        // Both literal or neither literal: compile in source order.
        (Some(_), Some(_)) | (None, None) => Ok((compile_t(ctx, e0)?, compile_t(ctx, e1)?)),
        // One side is an int literal: compile the peer first to learn its bank,
        // then widen the literal to a float constant if the peer is float. A
        // literal contributes no slots, so peer-first preserves slot order.
        (Some(v0), None) => {
            let b = compile_t(ctx, e1)?;
            let a = if b.bank == ValType::Float {
                emit_float_const(ctx, v0 as f64)
            } else {
                compile_t(ctx, e0)?
            };
            Ok((a, b))
        }
        (None, Some(v1)) => {
            let a = compile_t(ctx, e0)?;
            let b = if a.bank == ValType::Float {
                emit_float_const(ctx, v1 as f64)
            } else {
                compile_t(ctx, e1)?
            };
            Ok((a, b))
        }
    }
}

fn compile_call_t(ctx: &mut LowerCtxF, call: &CallExpr) -> Result<TReg, LowerError> {
    // CEL member syntax `x.f(a)` is sugar for `f(x, a)`: the tree-walker resolves
    // it by inserting the target at `args[0]` and looking up a member overload
    // (`objects.rs:1358-1375`). Desugar to that same shape so a member call
    // reaches the same arms as its global form. An unhandled name still falls
    // through to the `call `{name}`` bail at the end, so this only widens what
    // lowers — it never changes which function a lowered expression runs.
    //
    // The one shape the tree-walker resolves differently is a bare-`Ident`
    // target, which it first tries as the QUALIFIED global `prefix.f`
    // (`objects.rs:1348-1356`, e.g. `optional.none()`). Such a target names a
    // namespace, not a value: it resolves to an untyped (`Int`) slot here and so
    // fails the bank checks of every arm below, bailing to the tree-walker
    // instead of miscompiling. Matching a function by bare name is the same
    // assumption `timestamp`/`duration` already make (see the module header's
    // schema/shape-guard note).
    if let Some(target) = &call.target {
        // Receiver-only stdlib accessors are registered with
        // `add_member_overload` (`common/types/duration.rs:191-222`,
        // `common/types/timestamp.rs:278-357`), so they exist ONLY in receiver
        // form: the global spelling `getHours(d)` is an `UndeclaredReference`
        // error in the tree-walker. Match them here, before the desugar, so the
        // global spelling keeps bailing instead of answering where the walker
        // raises.
        //
        // The receiver's bank picks the meaning. On a `duration`, `getHours` /
        // `getMinutes` / `getSeconds` / `getMilliseconds` are
        // `chrono::Duration::num_*`: the whole number of units, TRUNCATED toward
        // zero (`num_seconds` adds one back when secs is negative and nanos
        // positive, so `-1.5s` is `-1`, not `-2`). A duration is already carried
        // as i64 nanoseconds in the int file, so each is one truncating divide
        // by a green constant — and `OP_DIV` is exactly toward-zero
        // (bytecode.rs:205-221 divides the magnitudes and reapplies the sign),
        // so this is bit-exact with the tree-walker and needs no new opcode. The
        // divisor is a loop invariant, so its load goes in the prelude. The six
        // calendar names have no `duration` overload and bail there.
        //
        // On a `timestamp` the answer is a calendar field instead: split the
        // instant into a FLOORED day count plus nanoseconds-of-day, read the
        // clock fields off the remainder and the date fields off the day count
        // via civil-from-days. All of it is int-file arithmetic on green
        // constants, so the whole conversion stays inside the traced loop.
        if TEMPORAL_ACCESSORS.contains(&call.func_name.as_str()) {
            if !call.args.is_empty() {
                return Err(LowerError::unsupported(format!(
                    "`{}` arity",
                    call.func_name
                )));
            }
            // A timestamp accessor reads a CALENDAR field, so it depends on the
            // instant's UTC offset — which this VM does not carry: a column is
            // i64 nanoseconds and the oracle rebuilds it as
            // `DateTime::from_timestamp_nanos(n).fixed_offset()`, i.e. always
            // +00:00. A folded `timestamp("...+09:00")` literal, by contrast,
            // keeps its offset in the tree-walker while the fold here drops it,
            // so its accessors would disagree. Restricting the receiver to a
            // column path keeps the UTC assumption sound; anything else bails.
            let receiver_is_column = match &target.expr {
                Expr::Ident(n) => !ctx.locals.contains_key(n),
                Expr::Select(_) => true,
                _ => false,
            };
            let a = compile_t(ctx, target)?;
            match a.bank {
                ValType::Duration => {
                    let nanos_per_unit = match call.func_name.as_str() {
                        "getHours" => 3_600_000_000_000i64,
                        "getMinutes" => 60_000_000_000i64,
                        "getSeconds" => 1_000_000_000i64,
                        "getMilliseconds" => 1_000_000i64,
                        // The calendar accessors have no `duration` overload.
                        _ => {
                            return Err(LowerError::unsupported(format!(
                                "`{}` on a duration receiver",
                                call.func_name
                            )))
                        }
                    };
                    return Ok(emit_int_bin_k(ctx, OP_DIV, a, nanos_per_unit));
                }
                ValType::Timestamp => {
                    if !receiver_is_column {
                        return Err(LowerError::unsupported(format!(
                            "`{}` on a non-column timestamp (UTC offset not carried)",
                            call.func_name
                        )));
                    }
                    let (days, nanos_of_day) = emit_days_and_nanos_of_day(ctx, a);
                    return Ok(match call.func_name.as_str() {
                        "getHours" => emit_int_bin_k(ctx, OP_DIV, nanos_of_day, 3_600_000_000_000),
                        "getMinutes" => {
                            let m = emit_int_bin_k(ctx, OP_DIV, nanos_of_day, 60_000_000_000);
                            emit_int_bin_k(ctx, OP_MOD, m, 60)
                        }
                        "getSeconds" => {
                            let s = emit_int_bin_k(ctx, OP_DIV, nanos_of_day, 1_000_000_000);
                            emit_int_bin_k(ctx, OP_MOD, s, 60)
                        }
                        "getMilliseconds" => {
                            let ms = emit_int_bin_k(ctx, OP_DIV, nanos_of_day, 1_000_000);
                            emit_int_bin_k(ctx, OP_MOD, ms, 1_000)
                        }
                        // `weekday().num_days_from_sunday()`: 1970-01-01 was a
                        // Thursday (4 days from Sunday), and `days` can be
                        // negative, so the remainder is floored back into [0, 7).
                        "getDayOfWeek" => {
                            let shifted = emit_int_bin_k(ctx, OP_ADD, days, 4);
                            let rem = emit_int_bin_k(ctx, OP_MOD, shifted, 7);
                            let neg = emit_int_bin_k(ctx, OP_LT, rem, 0);
                            let back = emit_int_bin_k(ctx, OP_MUL, neg, 7);
                            emit_int_bin(ctx, OP_ADD, rem, back)
                        }
                        "getFullYear" => emit_civil_from_days(ctx, days).0,
                        // `month0()` / `day0()` are 0-based; `day()` is 1-based.
                        "getMonth" => {
                            let (_, month, _) = emit_civil_from_days(ctx, days);
                            emit_int_bin_k(ctx, OP_SUB, month, 1)
                        }
                        "getDate" => emit_civil_from_days(ctx, days).2,
                        "getDayOfMonth" => {
                            let (_, _, day) = emit_civil_from_days(ctx, days);
                            emit_int_bin_k(ctx, OP_SUB, day, 1)
                        }
                        // 0-based: the walker subtracts month0 and day0 to reach
                        // 1 January of the same year and takes the day span.
                        "getDayOfYear" => {
                            let (year, _, _) = emit_civil_from_days(ctx, days);
                            let jan1 = emit_days_of_jan1(ctx, year);
                            emit_int_bin(ctx, OP_SUB, days, jan1)
                        }
                        _ => unreachable!("name is in TEMPORAL_ACCESSORS"),
                    });
                }
                _ => {
                    return Err(LowerError::unsupported(format!(
                        "`{}` on a non-temporal receiver",
                        call.func_name
                    )))
                }
            }
        }
        let mut args = Vec::with_capacity(call.args.len() + 1);
        args.push((**target).clone());
        args.extend(call.args.iter().cloned());
        return compile_call_t(
            ctx,
            &CallExpr {
                func_name: call.func_name.clone(),
                target: None,
                args,
            },
        );
    }
    let name = call.func_name.as_str();

    // `timestamp("...")` / `duration("...")` over a string literal are green
    // constants: parse the literal the same way the tree-walker does and fold it
    // to an i64-nanoseconds constant in the prelude. A non-literal argument, a
    // parse error, or a value outside the i64-nanos range bails to the
    // tree-walker (which owns the error / wider-range case).
    if name == "timestamp" && call.args.len() == 1 {
        let s = as_string_literal(&call.args[0])
            .ok_or_else(|| LowerError::unsupported("timestamp() non-literal argument"))?;
        let dt = chrono::DateTime::parse_from_rfc3339(s)
            .map_err(|_| LowerError::unsupported("timestamp() literal parse"))?;
        let nanos = dt
            .timestamp_nanos_opt()
            .ok_or_else(|| LowerError::unsupported("timestamp outside i64-nanos range"))?;
        let r = ctx.fresh(ValType::Timestamp);
        ctx.prelude
            .extend_from_slice(&[OP_LOAD_CONST, nanos, r.idx as i64]);
        return Ok(r);
    }
    if name == "duration" && call.args.len() == 1 {
        let s = as_string_literal(&call.args[0])
            .ok_or_else(|| LowerError::unsupported("duration() non-literal argument"))?;
        let (_, dur) = crate::duration::parse_duration(s)
            .map_err(|_| LowerError::unsupported("duration() literal parse"))?;
        let nanos = dur
            .num_nanoseconds()
            .ok_or_else(|| LowerError::unsupported("duration outside i64-nanos range"))?;
        let r = ctx.fresh(ValType::Duration);
        ctx.prelude
            .extend_from_slice(&[OP_LOAD_CONST, nanos, r.idx as i64]);
        return Ok(r);
    }

    // `x in [e0, e1, ...]` over a LITERAL list unrolls to a constant membership
    // set `(x == e0) || (x == e1) || ...`. Reuses the per-bank equality op
    // (`OP_EQ` for the int-file banks incl. string ids and temporal nanos,
    // `OP_FEQ` for floats). Every element must share `x`'s bank (a heterogeneous
    // element is `false` in the tree-walker; bailing lets it own that); a
    // non-literal container bails.
    if name == ops::IN {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported("@in arity"));
        }
        let elements = match &call.args[1].expr {
            Expr::List(list) => &list.elements,
            _ => return Err(LowerError::unsupported("@in non-literal container")),
        };
        let x = compile_t(ctx, &call.args[0])?;
        if elements.is_empty() {
            // `x in []` is always false (the operand is still evaluated above).
            let d = ctx.fresh(ValType::Int);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST, 0, d.idx as i64]);
            return Ok(d);
        }
        let eq_op = if x.bank == ValType::Float { OP_FEQ } else { OP_EQ };
        let mut acc: Option<TReg> = None;
        for e in elements {
            let ev = compile_t(ctx, e)?;
            if ev.bank != x.bank {
                return Err(LowerError::unsupported("@in heterogeneous element"));
            }
            let t = ctx.fresh(ValType::Int);
            ctx.body
                .extend_from_slice(&[eq_op, x.idx as i64, ev.idx as i64, t.idx as i64]);
            acc = Some(match acc {
                None => t,
                Some(prev) => {
                    let o = ctx.fresh(ValType::Int);
                    ctx.body
                        .extend_from_slice(&[OP_OR, prev.idx as i64, t.idx as i64, o.idx as i64]);
                    o
                }
            });
        }
        return Ok(acc.expect("non-empty element list"));
    }

    // ternary `c ? t : f` — branchless blend on an int condition. Int arms use
    // an arithmetic SELECT; float arms use a bit-mask FSELECT (bit-exact, no
    // reassociation). Mixed-bank arms bail: the tree-walker yields int-or-float
    // per row, which no single result bank can carry.
    if name == ops::CONDITIONAL {
        if call.args.len() != 3 {
            return Err(LowerError::unsupported("_?_:_ arity"));
        }
        let c = compile_t(ctx, &call.args[0])?;
        if c.bank != ValType::Int {
            return Err(LowerError::unsupported("ternary condition must be int/bool"));
        }
        let t = compile_t(ctx, &call.args[1])?;
        let f = compile_t(ctx, &call.args[2])?;
        let (op, bank) = match (t.bank, f.bank) {
            (ValType::Int, ValType::Int) => (OP_SELECT, ValType::Int),
            (ValType::Float, ValType::Float) => (OP_FSELECT, ValType::Float),
            _ => return Err(LowerError::unsupported("mixed-bank ternary arms")),
        };
        let d = ctx.fresh(bank);
        ctx.body.extend_from_slice(&[
            op,
            c.idx as i64,
            t.idx as i64,
            f.idx as i64,
            d.idx as i64,
        ]);
        return Ok(d);
    }

    // constant-index access `base[k]` resolves to a typed slot.
    if name == ops::INDEX {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported("_[_] arity"));
        }
        let base = resolve_path(&call.args[0])?;
        let idx = as_int_literal(&call.args[1])
            .ok_or_else(|| LowerError::unsupported("non-constant index"))?;
        return Ok(ctx.slot(format!("{base}[{idx}]")));
    }

    // n-ary boolean fold — int operands, int result.
    if name == ops::LOGICAL_AND || name == ops::LOGICAL_OR {
        if call.args.len() < 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let op = if name == ops::LOGICAL_AND {
            OP_AND
        } else {
            OP_OR
        };
        let mut acc = compile_t(ctx, &call.args[0])?;
        if acc.bank != ValType::Int {
            return Err(LowerError::unsupported("boolean operand must be int/bool"));
        }
        for arg in &call.args[1..] {
            let b = compile_t(ctx, arg)?;
            if b.bank != ValType::Int {
                return Err(LowerError::unsupported("boolean operand must be int/bool"));
            }
            let d = ctx.fresh(ValType::Int);
            ctx.body
                .extend_from_slice(&[op, acc.idx as i64, b.idx as i64, d.idx as i64]);
            acc = d;
        }
        return Ok(acc);
    }

    // comparisons — same-bank operands, int `0`/`1` result.
    let cmp = match name {
        ops::GREATER_EQUALS => Some((OP_GE, OP_FGE)),
        ops::GREATER => Some((OP_GT, OP_FGT)),
        ops::LESS_EQUALS => Some((OP_LE, OP_FLE)),
        ops::LESS => Some((OP_LT, OP_FLT)),
        ops::EQUALS => Some((OP_EQ, OP_FEQ)),
        ops::NOT_EQUALS => Some((OP_NE, OP_FNE)),
        _ => None,
    };
    if let Some((iop, fop)) = cmp {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let (mut a, mut b) = compile_cmp_operands(ctx, &call.args[0], &call.args[1])?;
        // Strings support only equality here: a content-hash compare (OP_EQ/OP_NE
        // over the int-file ids) equals a string compare once the batch builder
        // has verified the hash is injective. Ordering needs sorted ids, so `<`
        // etc. bail; a string mixed with a non-string is a CEL type error.
        if a.bank == ValType::Str || b.bank == ValType::Str {
            if a.bank != ValType::Str || b.bank != ValType::Str {
                return Err(LowerError::unsupported("mixed string/non-string comparison"));
            }
            let op = match name {
                ops::EQUALS => OP_EQ,
                ops::NOT_EQUALS => OP_NE,
                _ => return Err(LowerError::unsupported("string ordering comparison")),
            };
            let d = ctx.fresh(ValType::Int);
            ctx.body
                .extend_from_slice(&[op, a.idx as i64, b.idx as i64, d.idx as i64]);
            return Ok(d);
        }
        // Timestamp / Duration compare as i64 nanoseconds: the signed int order
        // equals the chronological / magnitude order the tree-walker uses, so all
        // six comparisons use the signed int op. Both operands must be the SAME
        // temporal type (timestamp vs duration is NoSuchOverload; temporal vs a
        // non-temporal operand is a type error) — otherwise bail.
        if matches!(a.bank, ValType::Timestamp | ValType::Duration)
            || matches!(b.bank, ValType::Timestamp | ValType::Duration)
        {
            if a.bank != b.bank {
                return Err(LowerError::unsupported("mixed temporal comparison"));
            }
            let d = ctx.fresh(ValType::Int);
            ctx.body
                .extend_from_slice(&[iop, a.idx as i64, b.idx as i64, d.idx as i64]);
            return Ok(d);
        }
        // Two uint operands compare unsigned: `<`/`<=` map to OP_ULT/OP_ULE and
        // `>`/`>=` reuse them by swapping operands; eq/ne are bit-identical to
        // the signed ops. (A uint peer is never an int literal, so
        // `compile_cmp_operands` performs no float promotion here.)
        if a.bank == ValType::UInt && b.bank == ValType::UInt {
            let (op, lhs, rhs) = match name {
                ops::LESS => (OP_ULT, a, b),
                ops::LESS_EQUALS => (OP_ULE, a, b),
                ops::GREATER => (OP_ULT, b, a),
                ops::GREATER_EQUALS => (OP_ULE, b, a),
                ops::EQUALS => (OP_EQ, a, b),
                ops::NOT_EQUALS => (OP_NE, a, b),
                _ => unreachable!("cmp is one of the six comparison ops"),
            };
            let d = ctx.fresh(ValType::Int);
            ctx.body
                .extend_from_slice(&[op, lhs.idx as i64, rhs.idx as i64, d.idx as i64]);
            return Ok(d);
        }
        // A mixed int/float comparison widens the int side to float (`int as
        // f64`, the tree-walker's promotion). `compile_cmp_operands` already
        // folded a literal int to a float constant; a data-dependent int is
        // widened per row here via `cast_int_to_float`. Any uint mixed with a
        // different bank is a CEL type error and bails.
        let op = match (a.bank, b.bank) {
            (ValType::Int, ValType::Int) => iop,
            (ValType::Float, ValType::Float) => fop,
            (ValType::Int, ValType::Float) => {
                a = emit_i2f(ctx, a);
                fop
            }
            (ValType::Float, ValType::Int) => {
                b = emit_i2f(ctx, b);
                fop
            }
            _ => return Err(LowerError::unsupported("mixed-bank comparison")),
        };
        let d = ctx.fresh(ValType::Int);
        ctx.body
            .extend_from_slice(&[op, a.idx as i64, b.idx as i64, d.idx as i64]);
        return Ok(d);
    }

    // arithmetic — same-bank operands, same-bank result. No float modulo.
    let arith = match name {
        ops::ADD => Some((OP_ADD, Some(OP_FADD))),
        ops::SUBSTRACT => Some((OP_SUB, Some(OP_FSUB))),
        ops::MULTIPLY => Some((OP_MUL, Some(OP_FMUL))),
        ops::DIVIDE => Some((OP_DIV, Some(OP_FDIV))),
        ops::MODULO => Some((OP_MOD, None)),
        _ => None,
    };
    if let Some((iop, fop)) = arith {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let a = compile_t(ctx, &call.args[0])?;
        let b = compile_t(ctx, &call.args[1])?;
        match (a.bank, b.bank) {
            (ValType::Int, ValType::Int) => {
                let d = ctx.fresh(ValType::Int);
                ctx.body
                    .extend_from_slice(&[iop, a.idx as i64, b.idx as i64, d.idx as i64]);
                Ok(d)
            }
            (ValType::UInt, ValType::UInt) => {
                // add/sub/mul are bit-identical to the signed ops (mod 2^64);
                // division/modulo need unsigned opcodes the trace IR lacks.
                if name == ops::DIVIDE || name == ops::MODULO {
                    return Err(LowerError::unsupported("uint division/modulo"));
                }
                let d = ctx.fresh(ValType::UInt);
                ctx.body
                    .extend_from_slice(&[iop, a.idx as i64, b.idx as i64, d.idx as i64]);
                Ok(d)
            }
            (ValType::Float, ValType::Float) => {
                let fop = fop.ok_or_else(|| LowerError::unsupported("float modulo"))?;
                let d = ctx.fresh(ValType::Float);
                ctx.body
                    .extend_from_slice(&[fop, a.idx as i64, b.idx as i64, d.idx as i64]);
                Ok(d)
            }
            _ => Err(LowerError::unsupported("mixed int/float arithmetic")),
        }
    } else {
        match name {
            ops::LOGICAL_NOT => {
                if call.args.len() != 1 {
                    return Err(LowerError::unsupported("!_ arity"));
                }
                let a = compile_t(ctx, &call.args[0])?;
                if a.bank != ValType::Int {
                    return Err(LowerError::unsupported("! operand must be int/bool"));
                }
                let d = ctx.fresh(ValType::Int);
                ctx.body
                    .extend_from_slice(&[OP_NOT, a.idx as i64, d.idx as i64]);
                Ok(d)
            }
            ops::NEGATE => {
                if call.args.len() != 1 {
                    return Err(LowerError::unsupported("-_ arity"));
                }
                let a = compile_t(ctx, &call.args[0])?;
                // Negate is defined for int and double only. `-uint` / `-string`
                // are NoSuchOverload in the tree-walker, so they bail (and a uint
                // or string operand lives in the int bank — OP_FNEG would wrongly
                // read the float register file).
                let op = match a.bank {
                    ValType::Int => OP_NEG,
                    ValType::Float => OP_FNEG,
                    ValType::UInt => {
                        return Err(LowerError::unsupported("unary negate on uint"))
                    }
                    ValType::Str => {
                        return Err(LowerError::unsupported("unary negate on string"))
                    }
                    ValType::Timestamp | ValType::Duration => {
                        return Err(LowerError::unsupported("unary negate on temporal"))
                    }
                };
                let d = ctx.fresh(a.bank);
                ctx.body
                    .extend_from_slice(&[op, a.idx as i64, d.idx as i64]);
                Ok(d)
            }
            _ => Err(LowerError::unsupported(format!("call `{name}`"))),
        }
    }
}

/// Green-length comprehension unroll for the typed path (mirrors
/// [`compile_comprehension`], threading typed accumulator registers).
fn compile_comprehension_t(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
) -> Result<TReg, LowerError> {
    if comp.iter_var2.is_some() {
        return Err(LowerError::unsupported("two-variable comprehension"));
    }
    let elements = match &comp.iter_range.expr {
        Expr::List(list) => list.elements.clone(),
        _ => return Err(LowerError::unsupported("comprehension over non-literal range")),
    };

    let prev_iter = ctx.locals.remove(&comp.iter_var);
    let prev_accu = ctx.locals.remove(&comp.accu_var);

    let mut accu = compile_t(ctx, &comp.accu_init)?;
    for elem in &elements {
        let x_reg = compile_t(ctx, elem)?;
        ctx.locals.insert(comp.iter_var.clone(), x_reg);
        ctx.locals.insert(comp.accu_var.clone(), accu);
        accu = compile_t(ctx, &comp.loop_step)?;
    }
    ctx.locals.insert(comp.accu_var.clone(), accu);
    let result = compile_t(ctx, &comp.result)?;

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
