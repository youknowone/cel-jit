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
//!   * boolean `&& || !` over `bool` operands (non-short-circuit, correct in
//!     this domain because the operands cannot raise),
//!   * `all` / `exists` / `exists_one` comprehensions over a **literal** list
//!     (green-constant length), unrolled into a straight-line fold, or over a
//!     **runtime-length** list column ([`declares_list`]), which gets a real
//!     inner loop instead. `map` / `filter` build a list and stay out of the
//!     int subset.
//!
//! **Schema**: the caller declares every path an expression reads, with its
//! [`ValType`], in a [`Schema`]; an undeclared path is a decline. The declared
//! bank is what decides which operators the path is legal under — `!x` needs a
//! `bool`, `x + 1` needs a numeric bank — and it is the same declaration the
//! caller builds its columns from, so lowering and data agree by construction.
//! A real integration derives it from the context's variable types before
//! electing the JIT (the PyPy-style shape guard).

use super::bytecode::*;
use crate::common::ast::operators as ops;
use crate::common::ast::{CallExpr, ComprehensionExpr, EntryExpr, Expr, IdedExpr, LiteralValue};
use crate::{Context, Value};
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

/// Extract a plain integer literal, if that is what `e` is.
fn as_int_literal(e: &IdedExpr) -> Option<i64> {
    match &e.expr {
        Expr::Literal(LiteralValue::Int(i)) => Some(i.into_inner()),
        _ => None,
    }
}

fn as_bool_literal(e: &IdedExpr) -> Option<bool> {
    match &e.expr {
        Expr::Literal(LiteralValue::Boolean(b)) => Some(b.into_inner()),
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
    /// A `bool`, carried as `0`/`1` in the int register file, so storage, column
    /// loads and moves reuse the int ops. It is a type of its own and not a
    /// spelling of `int`, because CEL says so: `1 && 2`, `!1` and `1 ? x : y`
    /// are all `NoSuchOverload`, `true + true` is an unsupported operator, and
    /// `1 == true` is `false` rather than a bit compare. Ordering IS defined
    /// (`false < true`), so the six comparisons lower to the signed int ops.
    Bool,
    /// Unsigned 64-bit. Shares the int register file (the raw bit pattern), so
    /// storage, column loads, moves, add/sub/mul and eq/ne reuse the int ops;
    /// only ordering comparisons differ (unsigned `OP_ULT`/`OP_ULE`).
    UInt,
    Float,
    /// A string, carried in the int register file as its **rank** among the
    /// batch's distinct strings (`bytecode::StrDict`). The ranking is
    /// order-preserving, so the signed int comparisons are content comparisons
    /// bit-for-bit — equality and ordering alike — and injective by
    /// construction, so no id compare can confuse two distinct strings.
    /// Arithmetic (concatenation) and anything else needing the characters
    /// themselves falls back to the tree-walker.
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

/// Declared type of each input path. Every path an expression reads must appear
/// here: an absent path is a DECLINE, because the bank is what decides which
/// operators the path is legal under, and the caller builds its columns from the
/// same declaration. A real integration builds this from the context's variable
/// types and guards on it before electing the float JIT.
pub type Schema = HashMap<String, ValType>;

/// A typed register: a bank plus the index within that bank.
#[derive(Debug, Clone, Copy)]
struct TReg {
    bank: ValType,
    idx: usize,
}

/// Where in the batch program a slot's column is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    /// A per-ROW column: read at `row * 8` in the outer loop's prologue, which
    /// [`LoweredF::batch_sum_shape`] emits.
    Row,
    /// A flattened ELEMENT column of a list: read at `(offset(list) + j) * 8`
    /// INSIDE a comprehension's inner loop. Only the lowering knows where that
    /// loop is, so the lowering emits the load and the batch builder just parks
    /// the column base in `base_reg` before the row loop starts.
    Element {
        /// Int register holding the column's base address, loop-invariant.
        base_reg: usize,
    },
}

/// One typed input slot for the two-bank machine: a path resolved to a register
/// in its bank, plus which bank ([`ValType`]) it is and where it is read
/// ([`SlotKind`]).
#[derive(Debug, Clone)]
pub struct SlotInfoF {
    /// Dotted variable path, e.g. `account.balance`.
    pub path: String,
    /// The bank this slot's column is read into.
    pub ty: ValType,
    /// Register index within [`SlotInfoF::ty`]'s bank.
    pub reg: usize,
    /// Row column vs flattened list-element column.
    pub kind: SlotKind,
}

/// Slot path of the DERIVED length column for the string column at `path` — the
/// key [`LoweredF::slots`] carries for a `size(<string>)` and the one the batch
/// builder materializes against. Spelled like the call so a lowering dump reads
/// back as the expression that asked for it; it can never collide with a real
/// CEL path, which is a dotted identifier chain.
pub fn size_slot_path(path: &str) -> String {
    format!("size({path})")
}

/// The string path a [`size_slot_path`] key was derived from, or `None` if the
/// key is an ordinary column. For a list column the same key carries the
/// per-row ELEMENT COUNT (see [`elem_slot_path`]).
pub fn size_slot_source(slot_path: &str) -> Option<&str> {
    slot_path.strip_prefix("size(")?.strip_suffix(')')
}

/// Slot path of the DERIVED string column for the column at `path` — the
/// per-row `string(...)` conversion, which needs characters the machine does
/// not carry and so is materialized by the batch builder, exactly as
/// [`size_slot_path`]'s length column is.
pub fn string_slot_path(path: &str) -> String {
    format!("string({path})")
}

/// The column path a [`string_slot_path`] key was derived from.
pub fn string_slot_source(slot_path: &str) -> Option<&str> {
    slot_path.strip_prefix("string(")?.strip_suffix(')')
}

/// Slot path of the `k`-th DERIVED concatenation column, whose operands are
/// [`LoweredF::concats`]`[k]`.
///
/// The operands are held in a side table rather than spelled into the path
/// because one of them can be a LITERAL, and a literal may contain any
/// character at all — including whatever separator a spelled-out path would
/// need to be split on.
pub fn concat_slot_path(k: usize) -> String {
    format!("concat#{k}")
}

/// The [`LoweredF::concats`] index a [`concat_slot_path`] key names.
pub fn concat_slot_index(slot_path: &str) -> Option<usize> {
    slot_path.strip_prefix("concat#")?.parse().ok()
}

/// One operand of a derived concatenation column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcatSide {
    /// A `string` column, by schema path.
    Column(String),
    /// A literal, the same on every row.
    Literal(String),
    /// Another derived concatenation, by its [`LoweredF::concats`] index.
    /// Always a LOWER index than the spec holding it, since an operand is
    /// lowered before the operation over it — so materializing the table in
    /// order resolves every reference.
    Derived(usize),
}

/// The two operands of a derived concatenation column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcatSpec {
    /// Left operand.
    pub left: ConcatSide,
    /// Right operand.
    pub right: ConcatSide,
}

/// Slot path of the DERIVED per-row START INDEX of the list at `path` into its
/// flattened element columns — Arrow's offsets buffer, and the other half of
/// the `(offset, size)` pair that locates one row's elements.
pub fn offset_slot_path(path: &str) -> String {
    format!("offset({path})")
}

/// The list path an [`offset_slot_path`] key was derived from, or `None` if the
/// key is an ordinary column.
pub fn offset_slot_source(slot_path: &str) -> Option<&str> {
    slot_path.strip_prefix("offset(")?.strip_suffix(')')
}

/// Slot path of a flattened ELEMENT column of the list at `list`: the elements
/// themselves for a list of scalars (`field == None`), or one struct field's
/// values for a list of records (`field == Some(f)`).
///
/// Arrow's layout: every row's elements laid end to end in ONE buffer per
/// field, addressed by `offset(list)[row] + j`. A list is therefore not a value
/// on this machine — it is a `(size, offset)` pair of row columns plus one
/// element column per field it reads.
pub fn elem_slot_path(list: &str, field: Option<&str>) -> String {
    match field {
        None => format!("{list}[]"),
        Some(f) => format!("{list}[].{f}"),
    }
}

/// Split an [`elem_slot_path`] key back into `(list, field)`, or `None` if the
/// key is not an element column.
pub fn elem_slot_source(slot_path: &str) -> Option<(&str, Option<&str>)> {
    let (list, rest) = slot_path.split_once("[]")?;
    match rest {
        "" => Some((list, None)),
        _ => Some((list, Some(rest.strip_prefix('.')?))),
    }
}

/// True if `schema` declares `path` as a LIST, i.e. it carries at least one
/// flattened element column ([`elem_slot_path`]). Declaring the elements is
/// what makes a list iterable here; the list path itself never names a
/// register, so there is no list "bank".
pub fn declares_list(schema: &Schema, path: &str) -> bool {
    let prefix = format!("{path}[]");
    schema
        .keys()
        .any(|k| k.as_str() == prefix || k.starts_with(&format!("{prefix}.")))
}

/// Int register reserved for the overflow trap flag of the two-bank machine
/// (see [`OP_ADD_OVF`]). Fixed at 0 and allocated before any body register, so
/// the overflow-checked arithmetic ops can name it while the body is still
/// being emitted — the flag's *address* is only chosen by the batch driver at
/// run time, which is too late for an immediate operand. The batch builder
/// zeroes it in the setup and publishes it with [`OP_TRAP_STORE`] after the
/// loop.
pub const OVF_FLAG_REG: usize = 0;

/// One **broadcast scalar** the body reads, and the int register it arrives in.
///
/// Broadcast means one value for every row of the batch — but not for every
/// batch, which is why it is a seeded register and not an immediate. The whole
/// point of [`BatchSeed`] is that anything data-dependent reaches the program
/// through a register, so one set of words serves every batch and the JIT's
/// green key stays put.
#[derive(Debug, Clone)]
pub struct ScalarSeed {
    /// What the batch builder must resolve to fill the register.
    pub kind: SeedKind,
    /// Int register it arrives in.
    pub reg: usize,
}

/// What a [`ScalarSeed`] asks the batch builder for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedKind {
    /// The id of this string among the batch's distinct strings.
    StrId(String),
    /// Base address of a `0`/`1` table indexed by string id, holding this
    /// predicate's answer for each of the batch's distinct strings.
    ///
    /// Ids are dense `0..k`, so a pure single-string predicate can be answered
    /// once per DISTINCT string at bind and then read per row with a load. This
    /// is dictionary encoding's other half — predicate pushdown — and it needs
    /// no new opcode: the read is `OP_MUL` (id by 8) then `OP_COL_LOAD`, the
    /// same `*(base + ea)` a column read already is.
    StrPredicate(StrPredicate),
}

/// A pure predicate over one string, with its argument fixed by the expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrPredicate {
    /// `s.startsWith(arg)`.
    StartsWith(String),
    /// `s.endsWith(arg)`.
    EndsWith(String),
    /// `s.contains(arg)`.
    Contains(String),
    /// `s.matches(arg)`, an RE2-flavoured regex match.
    #[cfg(feature = "regex")]
    Matches(String),
}

impl StrPredicate {
    /// Answer this predicate for each of `strings`, in id order.
    ///
    /// The operations are the ones `common/types/string.rs` applies per row, so
    /// answering them per distinct string instead cannot change any answer —
    /// only how many times it is computed.
    pub fn table(&self, strings: &[&str]) -> Vec<i64> {
        match self {
            StrPredicate::StartsWith(p) => {
                strings.iter().map(|s| s.starts_with(p) as i64).collect()
            }
            StrPredicate::EndsWith(p) => strings.iter().map(|s| s.ends_with(p) as i64).collect(),
            StrPredicate::Contains(p) => strings
                .iter()
                .map(|s| s.contains(p.as_str()) as i64)
                .collect(),
            #[cfg(feature = "regex")]
            StrPredicate::Matches(p) => {
                // Compiled ONCE for the whole table rather than per row, and
                // valid because the lowering refused the call otherwise.
                let re = regex::Regex::new(p).expect("regex validated when the call was lowered");
                strings.iter().map(|s| re.is_match(s) as i64).collect()
            }
        }
    }
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
    /// Broadcast scalars the body reads, in first-encounter order. The batch
    /// builder resolves each against the batch and seeds its register
    /// ([`BatchSeed::scalar_regs`]).
    pub scalar_seeds: Vec<ScalarSeed>,
    /// Set when the top-level result is a LIST: the row's own value is then its
    /// element COUNT (in `result_reg`), and the elements themselves were stored
    /// through this description as the loop ran.
    pub list_output: Option<ListOutput>,
    /// Operands of the derived `concat#k` columns, indexed by `k`. Each entry's
    /// [`ConcatSide::Derived`] references are all lower indices, so
    /// materializing the table in order resolves them.
    pub concats: Vec<ConcatSpec>,
    /// `Some(b)` when the body does temporal arithmetic: every value in a
    /// `timestamp` or `duration` column must satisfy `|v| <= b` for this
    /// program to answer what the tree-walker answers.
    ///
    /// Both banks are i64 nanoseconds, but the walker computes in chrono,
    /// whose `{secs: i64, nanos: i32}` range is far wider — two durations that
    /// each fit i64 nanoseconds can sum to one that does not, and chrono
    /// answers it. So a machine add would trap where the walker succeeds, and a
    /// trap means "the walker raised", which would be a wrong answer rather
    /// than a missing one.
    ///
    /// Narrowing the DOMAIN removes the disagreement instead of papering over
    /// it: inside `|v| <= i64::MAX / (ops + 1)` no intermediate can leave i64
    /// nanoseconds, so the two agree exactly, and a batch outside it is refused
    /// at bind — the existing "evaluate this with `Program::execute`" channel.
    /// With the usual single operation the bound is ±146 years around the
    /// epoch.
    pub temporal_bound: Option<i64>,
    /// Positions **within [`LoweredF::body`]** of jump target words, which the
    /// lowering writes body-relative because it cannot know where the body
    /// lands. [`LoweredF::batch_sum_shape`] relocates each to an
    /// absolute program address once it does.
    pub jump_fixups: Vec<usize>,
}

/// A batch program's words plus the register banks they run on.
///
/// The words carry only the expression's **shape** — no column address, no row
/// count, no trap-word address. Those are data, and reach the program through
/// [`BatchShape::seed`] instead. One built program therefore serves every batch
/// of that shape, which is what lets the JIT's green key (the program pointer
/// and pc, `trace_ctx.rs` `green_key_raw`) stay put from batch to batch rather
/// than re-keying and recompiling.
///
/// This is the upstream arrangement: `rsre_core.py:384-385` keeps the regex
/// PATTERN green and the subject string and its positions red, so one compiled
/// loop matches every subject; `micronumpy/loop.py:88-89` keeps the arrays,
/// base storage included, red while the greens are the computation's shape.
/// What a batch loop does with each row's result.
///
/// The loop's shape is the same either way — the same red-index column reads,
/// the same body. Only the last instruction of the iteration differs, and with
/// it what the run produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchReduce {
    /// Accumulate into a running total carried across iterations, and return
    /// it. The only answer a `bool` predicate has (a count of matching rows),
    /// and the only one that costs no memory.
    Sum,
    /// Store each row's result to `out[i]` and return nothing but the row
    /// count. The loop carries only `i`, so a result of ANY bank has somewhere
    /// to go — including the string ranks and nanosecond counts a sum cannot
    /// consume.
    PerRow,
}

pub struct BatchShape {
    /// The program words.
    pub code: Vec<i64>,
    /// Float-bank register count the program runs on. The int-bank count is
    /// [`BatchSeed`]'s, since the only thing that needs it is building the bank.
    pub num_float_regs: usize,
    /// Which int registers the caller fills in per batch.
    pub seed: BatchSeed,
}

/// A LIST-valued per-row result: the shape of the ragged output the loop writes.
///
/// The same shape a list COLUMN arrives in — a per-row element count plus one
/// flat buffer per field — because it is the same thing, produced rather than
/// consumed. The count rides the ordinary per-row output; the elements go to
/// these buffers, at a cursor that runs across the whole batch.
#[derive(Debug, Clone)]
pub struct ListOutput {
    /// The list the elements are drawn from, whose flattened element count
    /// bounds how many this can ever write. `None` for a list built from
    /// nothing the schema declares, which does not arise today.
    pub source: String,
    /// One entry per output field, in the order the buffers are seeded.
    /// `None` names the elements themselves (a list of scalars).
    pub fields: Vec<(Option<String>, ValType)>,
    /// Register holding each field buffer's base address, aligned to `fields`.
    pub base_regs: Vec<usize>,
}

/// Which int registers a [`BatchShape`]'s words expect to find already filled
/// in when the mainloop starts.
///
/// These are plain reds. They must **not** be promoted: promotion inserts a
/// `guard_value`, and a guard that fails on every batch generates a bridge per
/// batch (`rlib/jit.py`, `promote`) — per-batch recompilation under another
/// name.
pub struct BatchSeed {
    /// Register holding the row count.
    r_n: usize,
    /// Register holding the overflow trap word's address.
    r_trap: usize,
    /// Register holding the output buffer's base address, under
    /// [`BatchReduce::PerRow`]. Data like any column base, so it is seeded and
    /// never an immediate.
    r_out: Option<usize>,
    /// Register holding each ELEMENT buffer's base address, for a list-valued
    /// result. Aligned to [`ListOutput::fields`].
    list_out_regs: Vec<usize>,
    /// Register holding each column's base address, in [`LoweredF::slots`]
    /// order. A ROW column's base lives in the machinery bank; a list ELEMENT
    /// column's lives in the register its lowering reserved.
    base_regs: Vec<usize>,
    /// Register holding each **broadcast scalar** — one value that is the same
    /// for every row of the batch but not the same for every batch — in
    /// [`LoweredF::str_literals`] order. A column base is the batch's address;
    /// these are the batch's values. Same reason for being red: an immediate
    /// would put batch data in the words the green key is taken over.
    scalar_regs: Vec<usize>,
    /// Length of the int register bank these indices address.
    num_int_regs: usize,
}

impl BatchSeed {
    /// How many broadcast scalars this shape expects, for callers that build
    /// the vector themselves.
    pub fn num_scalars(&self) -> usize {
        self.scalar_regs.len()
    }

    /// Build one batch's initial int register bank: the row count, the trap
    /// word's address, each column's base address and each broadcast scalar in
    /// its own register, every other register zero.
    ///
    /// The back-edge is a do-while, so callers must pass `n >= 1`.
    pub fn regs(&self, bases: &[i64], scalars: &[i64], n: i64, trap_addr: i64) -> Vec<i64> {
        self.regs_out(bases, scalars, n, trap_addr, 0)
    }

    /// [`BatchSeed::regs`] plus the output buffer's base address, which a
    /// [`BatchReduce::PerRow`] shape stores each row's result through. Ignored
    /// by a [`BatchReduce::Sum`] shape, which has no output register.
    pub fn regs_out(
        &self,
        bases: &[i64],
        scalars: &[i64],
        n: i64,
        trap_addr: i64,
        out_addr: i64,
    ) -> Vec<i64> {
        self.regs_list(bases, scalars, n, trap_addr, out_addr, &[])
    }

    /// [`BatchSeed::regs_out`] plus one base per ELEMENT buffer, for a
    /// list-valued result.
    pub fn regs_list(
        &self,
        bases: &[i64],
        scalars: &[i64],
        n: i64,
        trap_addr: i64,
        out_addr: i64,
        list_out: &[i64],
    ) -> Vec<i64> {
        assert_eq!(
            bases.len(),
            self.base_regs.len(),
            "batch seed: base arity {} != slot count {}",
            bases.len(),
            self.base_regs.len()
        );
        assert_eq!(
            scalars.len(),
            self.scalar_regs.len(),
            "batch seed: scalar arity {} != {} broadcast scalars",
            scalars.len(),
            self.scalar_regs.len()
        );
        assert!(n >= 1, "batch seed: n must be >= 1 (do-while back-edge)");
        let mut regs = vec![0i64; self.num_int_regs];
        regs[self.r_n] = n;
        regs[self.r_trap] = trap_addr;
        for (&base, &reg) in bases.iter().zip(&self.base_regs) {
            regs[reg] = base;
        }
        for (&v, &reg) in scalars.iter().zip(&self.scalar_regs) {
            regs[reg] = v;
        }
        if let Some(reg) = self.r_out {
            assert!(
                out_addr != 0,
                "batch seed: PerRow shape needs an output buffer"
            );
            regs[reg] = out_addr;
        }
        assert_eq!(
            list_out.len(),
            self.list_out_regs.len(),
            "batch seed: element-buffer arity {} != {} output fields",
            list_out.len(),
            self.list_out_regs.len()
        );
        for (&addr, &reg) in list_out.iter().zip(&self.list_out_regs) {
            regs[reg] = addr;
        }
        regs
    }
}

impl LoweredF {
    /// Build a **columnar batch** program over the two-bank machine: for each
    /// row `i` in `0..n`, load each slot's `col_k[i]` via a red-index `raw_load`
    /// (`OP_COL_LOAD` for int slots, `OP_COL_LOAD_F` for float slots), run the
    /// body, and accumulate the result into a running sum in the result's bank
    /// (int `r_acc` -> `OP_RETURN`, or float `f_acc` -> `OP_RETURN_F`). `bases[k]` is the
    /// base address of slot `k`'s column buffer (an `i64` pointer regardless of
    /// bank), aligned to [`LoweredF::slots`]. Returns the [`BatchShape`] and the
    /// initial int register bank to run it on, since `bases` and `n` are data
    /// and reach the program through registers rather than as immediates.
    ///
    /// The loop machinery (`i`, `acc`, `n`, `one`, `stride`, `ea`) and the
    /// per-slot base pointers live in the **int** bank above `num_int_regs`, so
    /// the body's registers are untouched. The back-edge is a do-while, so
    /// callers must pass `n >= 1`.
    /// [`LoweredF::batch_sum_shape`] without an overflow trap word:
    /// the program still guards its `int` arithmetic, but nothing publishes the
    /// flag, so the caller **cannot tell** an overflowed row from a good one.
    /// Only for harnesses whose data is bounded by construction; the evaluator
    /// path ([`super::bytecode::eval_batch_sum_f`]) always passes a trap word.
    /// Whether the batch loop's running total can consume this result.
    ///
    /// The loop accumulates an int count/sum or a float total, so a
    /// string-valued or temporal-valued result has nothing to accumulate into:
    /// summing string RANKS or nanosecond counts would be an answer the
    /// tree-walker never gives. This is the REDUCTION's limit and says nothing
    /// about whether the expression lowered — it did, or `lower_typed` would
    /// have said so.
    pub fn sum_reducible(&self) -> Result<(), LowerError> {
        if self.list_output.is_some() {
            return Err(LowerError::unsupported(
                "list-valued result: the batch loop reduces by sum",
            ));
        }
        match self.result_bank {
            ValType::Int | ValType::Bool | ValType::UInt | ValType::Float => Ok(()),
            b => Err(LowerError::unsupported(format!(
                "{b:?}-valued result: the batch loop reduces by sum"
            ))),
        }
    }

    /// The first temporal value in `columns` outside
    /// [`LoweredF::temporal_bound`], as `(slot index, value)`.
    ///
    /// Data-dependent, so it is a property of the BATCH: the same expression
    /// over another batch may be fine. Costs one pass over the temporal columns
    /// and only when the expression does temporal arithmetic at all.
    pub fn temporal_out_of_domain(
        &self,
        columns: &[super::bytecode::Column],
    ) -> Option<(usize, i64)> {
        let bound = self.temporal_bound?;
        for (k, (col, slot)) in columns.iter().zip(&self.slots).enumerate() {
            if !matches!(slot.ty, ValType::Timestamp | ValType::Duration) {
                continue;
            }
            let super::bytecode::Column::Int(values) = col else {
                continue;
            };
            if let Some(&v) = values.iter().find(|v| v.abs() > bound) {
                return Some((k, v));
            }
        }
        None
    }

    /// Takes column bases already computed, so it cannot rank a batch's
    /// strings; an expression carrying a string literal has no id to seed here
    /// and must go through [`super::bytecode::prepare_batch`], which is handed
    /// the strings themselves.
    pub fn batch_sum_program(&self, bases: &[i64], n: i64) -> (BatchShape, Vec<i64>) {
        assert!(
            self.scalar_seeds.is_empty(),
            "batch_sum_program takes bases, not strings: {:?} needs a ranked batch",
            self.scalar_seeds[0].kind
        );
        let shape = self.batch_sum_shape(false);
        let regs = shape.seed.regs(bases, &[], n, 0);
        (shape, regs)
    }

    /// Build the batch program's words and the layout of the registers its
    /// caller seeds.
    ///
    /// `with_trap` emits the epilogue's overflow-flag store (see
    /// [`OP_TRAP_STORE`]). Whether that store is there at all is shape; the
    /// address it writes to is data and rides in a seeded register, so the
    /// evaluator path passes `true` here and the trap word's address to
    /// [`BatchSeed::regs`].
    pub fn batch_sum_shape(&self, with_trap: bool) -> BatchShape {
        self.batch_shape(with_trap, BatchReduce::Sum)
    }

    /// [`LoweredF::batch_sum_shape`] for a chosen reduction.
    ///
    /// [`BatchReduce::PerRow`] has no `sum_reducible` precondition: a store
    /// takes a result of any bank, which is the whole reason the reduction is a
    /// choice.
    pub fn batch_shape(&self, with_trap: bool, reduce: BatchReduce) -> BatchShape {
        // The accumulate below would fold a result the sum cannot consume into
        // the int total — adding string RANKS, nanoseconds, or a collected
        // list's element COUNT. Every public door asks `sum_reducible` first;
        // this catches a harness that did not.
        if reduce == BatchReduce::Sum {
            self.sum_reducible()
                .expect("a summing shape on a result the loop's sum cannot consume");
        }
        let m = self.num_int_regs; // first int machinery register
        let (r_i, r_acc, r_n, r_one, r_stride, r_ea) = (m, m + 1, m + 2, m + 3, m + 4, m + 5);
        let r_trap = m + 6;
        // The output base is a machinery register too, present only where the
        // reduction stores through it.
        let r_out = match reduce {
            BatchReduce::Sum => None,
            BatchReduce::PerRow => Some(m + 7),
        };
        let r_base0 = m + 7 + usize::from(r_out.is_some());
        let total_int_regs = r_base0 + self.slots.len();
        // A float result accumulates into a float register above the body's
        // float bank; an int result uses the int `r_acc` and leaves the float
        // bank at the body's count. `f_acc` is unused when the result is int.
        let (f_acc, total_float_regs) = match (reduce, self.result_bank) {
            (BatchReduce::Sum, ValType::Float) => (self.num_float_regs, self.num_float_regs + 1),
            _ => (0, self.num_float_regs),
        };

        let mut p = Vec::new();
        let load_const = |p: &mut Vec<i64>, imm: i64, dst: usize| {
            p.extend_from_slice(&[OP_LOAD_CONST, imm, dst as i64]);
        };
        load_const(&mut p, 0, r_i);
        // Accumulator init, run once before the merge point. `OP_LOAD_CONST_F`'s
        // `f64::from_bits` must stay out of the traced loop body; here it is in
        // the setup (0.0 has zero bits).
        match (reduce, self.result_bank) {
            (BatchReduce::Sum, ValType::Float) => {
                p.extend_from_slice(&[OP_LOAD_CONST_F, 0, f_acc as i64])
            }
            // A `PerRow` loop carries no accumulator, but zeroing `r_acc` costs
            // one setup instruction and leaves the bank in one known state.
            _ => load_const(&mut p, 0, r_acc),
        }
        load_const(&mut p, 1, r_one);
        load_const(&mut p, 8, r_stride);
        // Overflow trap: the flag starts clear. Where it is published is data,
        // so the address arrives in a seeded register rather than as an
        // immediate, and the epilogue stores through it once the loop is done.
        load_const(&mut p, 0, OVF_FLAG_REG);
        // The row count and every column base are data as well, and reach the
        // program the same way, so the words emitted below are identical for
        // every batch of this shape. Each base is still a loop-invariant int
        // register (never a scalar state field, which would trip
        // `VirtualStatesCantMatch` at loop close): a ROW column's lives in the
        // machinery bank, and a list ELEMENT column's in the register the
        // lowering reserved for it, because the load that reads it was emitted
        // inside the body's inner loop.
        let base_regs: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .map(|(k, slot)| match slot.kind {
                SlotKind::Row => r_base0 + k,
                SlotKind::Element { base_reg } => base_reg,
            })
            .collect();
        // Loop-invariant literal loads, run once before the merge point.
        p.extend_from_slice(&self.prelude);

        let body_pc = p.len();
        // ea = i * 8 (byte offset of row i in an 8-byte column)
        p.extend_from_slice(&[OP_MUL, r_i as i64, r_stride as i64, r_ea as i64]);
        // slot_k = *(base_k + ea)   — the red-index columnar read, per bank.
        // Element columns are skipped: their index is the inner loop's, not the
        // row's, so the lowering already emitted their loads inside the body.
        for (k, slot) in self.slots.iter().enumerate() {
            if slot.kind != SlotKind::Row {
                continue;
            }
            let op = match slot.ty {
                ValType::Int
                | ValType::Bool
                | ValType::UInt
                | ValType::Str
                | ValType::Timestamp
                | ValType::Duration => OP_COL_LOAD,
                ValType::Float => OP_COL_LOAD_F,
            };
            p.extend_from_slice(&[op, (r_base0 + k) as i64, r_ea as i64, slot.reg as i64]);
        }
        let body_at = p.len();
        p.extend_from_slice(&self.body);
        // Relocate the body's jump targets, which were emitted relative to
        // `body[0]`, to absolute program addresses.
        for &f in &self.jump_fixups {
            p[body_at + f] += body_at as i64;
        }
        // acc += result (bank-matched); i += 1; if n > i goto @body; return acc.
        // The float accumulate is a loop-carried dependency, so the compiled
        // trace cannot reassociate it — the running total sums in row order, bit
        // for bit like the interpreter tiers.
        let tail: [i64; 4] = match (reduce, self.result_bank) {
            (BatchReduce::Sum, ValType::Float) => {
                [OP_FADD, f_acc as i64, self.result_reg as i64, f_acc as i64]
            }
            (BatchReduce::Sum, _) => [OP_ADD, r_acc as i64, self.result_reg as i64, r_acc as i64],
            // `out[i] = result`, at the same `i * 8` the row's columns were read
            // at. The float form stores the bit pattern, so one `i64` buffer
            // serves every bank.
            (BatchReduce::PerRow, ValType::Float) => [
                OP_COL_STORE_F,
                r_out.expect("PerRow shape allocates an output register") as i64,
                r_ea as i64,
                self.result_reg as i64,
            ],
            (BatchReduce::PerRow, _) => [
                OP_COL_STORE,
                r_out.expect("PerRow shape allocates an output register") as i64,
                r_ea as i64,
                self.result_reg as i64,
            ],
        };
        p.extend_from_slice(&tail);
        p.extend_from_slice(&[OP_ADD, r_i as i64, r_one as i64, r_i as i64]);
        p.extend_from_slice(&[OP_JUMP_IF_ABOVE, r_n as i64, r_i as i64, body_pc as i64]);
        // Publish the overflow flag. Outside the loop, so it costs the traced
        // body nothing and runs once when the back-edge guard finally exits.
        if with_trap {
            p.extend_from_slice(&[OP_TRAP_STORE, r_trap as i64, OVF_FLAG_REG as i64]);
        }
        // A `PerRow` loop's answer is in the output buffer; what it returns is
        // the row count it wrote, which the caller already knows and can check.
        match (reduce, self.result_bank) {
            (BatchReduce::Sum, ValType::Float) => p.extend_from_slice(&[OP_RETURN_F, f_acc as i64]),
            (BatchReduce::Sum, _) => p.extend_from_slice(&[OP_RETURN, r_acc as i64]),
            (BatchReduce::PerRow, _) => p.extend_from_slice(&[OP_RETURN, r_i as i64]),
        }
        BatchShape {
            code: p,
            num_float_regs: total_float_regs,
            seed: BatchSeed {
                r_n,
                r_trap,
                r_out,
                // Only a per-row run writes elements; a sum never reaches them.
                list_out_regs: match reduce {
                    BatchReduce::PerRow => self
                        .list_output
                        .as_ref()
                        .map(|o| o.base_regs.clone())
                        .unwrap_or_default(),
                    BatchReduce::Sum => Vec::new(),
                },
                base_regs,
                scalar_regs: self.scalar_seeds.iter().map(|s| s.reg).collect(),
                num_int_regs: total_int_regs,
            },
        }
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
    /// Broadcast scalars referenced by the body, in first-encounter order.
    scalar_seeds: Vec<ScalarSeed>,
    /// How many temporal `+`/`-` the body emitted. Bounds every intermediate:
    /// `m` of them combine at most `m + 1` operands.
    temporal_ops: usize,
    /// Folded `timestamp(...)` / `duration(...)` constants, which are operands
    /// too and so must satisfy the same bound the columns do.
    temporal_consts: Vec<i64>,
    /// Derived concatenation columns, in the order the batch builder must
    /// materialize them.
    concats: Vec<ConcatSpec>,
    /// Element slots created by the runtime-list comprehension currently being
    /// lowered, keyed by slot path. Scoped to that comprehension on purpose:
    /// two comprehensions over the same list get their OWN registers, since
    /// each reloads the column at its own inner index. Sharing one register
    /// would let the second loop read the first loop's last element.
    elem_map: HashMap<(String, usize), TReg>,
    /// The runtime-list comprehensions currently being lowered, outermost
    /// first. A STACK, not a slot: two comprehensions over two INDEPENDENT
    /// row-level lists nest fine, because each list's `(offset, size)` pair is
    /// a row column read once per row and the inner loop's index is its own.
    /// What does not nest is a list reached THROUGH an element, which would
    /// need per-element offsets a flat row column cannot express — and that
    /// declines on its own, since such a path is not one the schema declares.
    list_loop: Vec<ListLoop>,
    /// Positions within `body` holding a body-relative jump target.
    jump_fixups: Vec<usize>,
    /// Set when the top-level expression is collected as a list.
    list_output: Option<ListOutput>,
    schema: &'s Schema,
}

/// The runtime-list comprehension being lowered — what an iteration variable
/// resolves against.
struct ListLoop {
    /// Iteration variable name, e.g. `i` in `items.all(i, i.price > 10)`.
    iter_var: String,
    /// Schema path of the list, e.g. `items`.
    list: String,
    /// Int register holding the inner loop's byte offset `(offset + j) * 8`.
    ea_reg: usize,
}

impl LowerCtxF<'_> {
    fn fresh(&mut self, bank: ValType) -> TReg {
        let idx = match bank {
            // `Str` ids share the int register file (an `i64` rank).
            ValType::Int
            | ValType::Bool
            | ValType::UInt
            | ValType::Str
            | ValType::Timestamp
            | ValType::Duration => {
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

    /// Resolve a row slot whose type the schema must declare. An UNDECLARED path
    /// is a decline, not a guess: the bank decides which ops the path is legal
    /// under (`!x` needs `bool`, `x + 1` needs a numeric bank), so defaulting it
    /// would silently pick a meaning the caller never stated — and the caller's
    /// column, built from the same declaration, would then be read in the wrong
    /// bank.
    fn slot(&mut self, path: String) -> Result<TReg, LowerError> {
        let ty = self
            .schema
            .get(&path)
            .copied()
            .ok_or_else(|| LowerError::unsupported(format!("undeclared path `{path}`")))?;
        Ok(self.slot_typed(path, ty))
    }

    /// Resolve a slot whose type the LOWERING knows rather than the schema: the
    /// derived `size(...)` / `offset(...)` columns, which are counts.
    fn slot_typed(&mut self, path: String, ty: ValType) -> TReg {
        if let Some(&r) = self.slot_map.get(&path) {
            return r;
        }
        let r = self.fresh(ty);
        self.slot_map.insert(path.clone(), r);
        self.slots.push(SlotInfoF {
            path,
            ty,
            reg: r.idx,
            kind: SlotKind::Row,
        });
        r
    }

    /// Resolve a flattened ELEMENT slot of the list loop being lowered,
    /// emitting its columnar load **at the first reference** — which is inside
    /// the inner loop, where `ea_reg` holds `(offset + j) * 8`. The inner loop
    /// body is straight-line (no runtime-list comprehension may nest inside
    /// one), so the first reference dominates every later one. The cache is
    /// keyed on the ADDRESS too: `items[0].price + items[1].price` reads the
    /// same element column at two addresses and must load twice.
    fn elem_slot(&mut self, path: String, ea_reg: usize) -> Result<TReg, LowerError> {
        if let Some(&r) = self.elem_map.get(&(path.clone(), ea_reg)) {
            return Ok(r);
        }
        let ty =
            self.schema.get(&path).copied().ok_or_else(|| {
                LowerError::unsupported(format!("undeclared element path `{path}`"))
            })?;
        let r = self.fresh(ty);
        let base_reg = self.fresh(ValType::Int).idx;
        let op = match ty {
            ValType::Float => OP_COL_LOAD_F,
            _ => OP_COL_LOAD,
        };
        self.body
            .extend_from_slice(&[op, base_reg as i64, ea_reg as i64, r.idx as i64]);
        self.elem_map.insert((path.clone(), ea_reg), r);
        self.slots.push(SlotInfoF {
            path,
            ty,
            reg: r.idx,
            kind: SlotKind::Element { base_reg },
        });
        Ok(r)
    }

    /// Resolve `name` (optionally `.field`) against the list loop being
    /// lowered: `Some(reg)` when `name` is its iteration variable, `None` when
    /// it is not.
    fn iter_var_slot(
        &mut self,
        name: &str,
        field: Option<&str>,
    ) -> Result<Option<TReg>, LowerError> {
        // Innermost first, so an inner comprehension's variable shadows an
        // outer one of the same name.
        let Some(l) = self.list_loop.iter().rev().find(|l| l.iter_var == name) else {
            return Ok(None);
        };
        let (list, ea_reg) = (l.list.clone(), l.ea_reg);
        self.elem_slot(elem_slot_path(&list, field), ea_reg)
            .map(Some)
    }

    /// Emit `if regs[a] > regs[b] goto <patched later>` and return the body
    /// index of its target word, for [`LowerCtxF::patch_jump`].
    fn emit_jump_if_above(&mut self, a: TReg, b: TReg) -> usize {
        let at = self.body.len() + 3;
        self.body
            .extend_from_slice(&[OP_JUMP_IF_ABOVE, a.idx as i64, b.idx as i64, 0]);
        self.jump_fixups.push(at);
        at
    }

    /// Point a jump emitted by [`LowerCtxF::emit_jump_if_above`] at the current
    /// end of the body (a forward branch).
    fn patch_jump(&mut self, at: usize) {
        self.body[at] = self.body.len() as i64;
    }

    /// Emit a backward jump `if regs[a] > regs[b] goto tgt` — a loop back-edge,
    /// which is where the mainloop's `can_enter_jit` sits.
    fn emit_back_edge(&mut self, a: TReg, b: TReg, tgt: usize) {
        let at = self.body.len() + 3;
        self.body
            .extend_from_slice(&[OP_JUMP_IF_ABOVE, a.idx as i64, b.idx as i64, tgt as i64]);
        self.jump_fixups.push(at);
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
        // Register 0 is reserved for the overflow trap flag; body allocation
        // starts above it.
        next_int: OVF_FLAG_REG + 1,
        next_float: 0,
        slots: Vec::new(),
        slot_map: HashMap::new(),
        locals: HashMap::new(),
        scalar_seeds: Vec::new(),
        temporal_ops: 0,
        temporal_consts: Vec::new(),
        concats: Vec::new(),
        elem_map: HashMap::new(),
        list_loop: Vec::new(),
        jump_fixups: Vec::new(),
        list_output: None,
        schema,
    };
    // A list-valued TOP-LEVEL result is collected rather than declined: the
    // elements stream to their own buffers and the row's value becomes the
    // count. Only at the top level — a comprehension nested inside another
    // expression has no output stream to write to, and its list is a value the
    // machine still does not have.
    let result = match collect_list_result(&mut ctx, expr) {
        Some(r) => r?,
        None => compile_t(&mut ctx, expr)?,
    };
    // Note what is NOT checked here: whether the batch loop's sum can consume
    // the result. That is [`LoweredF::sum_reducible`]'s question, and it is a
    // different one — `b ? s : s` lowers to a select over string ids perfectly
    // well; there is just no sum of strings for the loop to accumulate. Keeping
    // them apart is what lets a decline say which of the two refused.
    let temporal_bound = match ctx.temporal_ops {
        0 => None,
        m => {
            let bound = i64::MAX / (m as i64 + 1);
            // A folded literal is an operand like any other, and its magnitude
            // is known now rather than at bind, so it is checked now.
            if let Some(&big) = ctx.temporal_consts.iter().find(|v| v.abs() > bound) {
                return Err(LowerError::unsupported(format!(
                    "temporal literal {big}ns is outside the ±{bound}ns arithmetic domain"
                )));
            }
            Some(bound)
        }
    };
    Ok(LoweredF {
        prelude: ctx.prelude,
        body: ctx.body,
        result_bank: result.bank,
        result_reg: result.idx,
        num_int_regs: ctx.next_int,
        num_float_regs: ctx.next_float,
        slots: ctx.slots,
        scalar_seeds: ctx.scalar_seeds,
        concats: ctx.concats,
        list_output: ctx.list_output,
        temporal_bound,
        jump_fixups: ctx.jump_fixups,
    })
}

fn compile_t(ctx: &mut LowerCtxF, e: &IdedExpr) -> Result<TReg, LowerError> {
    // A subexpression that reads no variable has the same value on every row of
    // every batch, so the tree-walker can answer it once, here. This is the
    // general form of the folds already scattered through this file (a literal
    // `duration(...)`, an unrolled `x in [..]`), and it is what lets a literal
    // aggregate — `size([1, 2, 3])`, `[1, 2, 3][0]` — reach a register the
    // machine has no list or map to build.
    //
    // Only a SUCCESSFUL evaluation folds. A constant that raises is left to the
    // normal path: `false && (1 / 0 > 0)` never evaluates its right operand in
    // the tree-walker, and folding eagerly would decline an expression the
    // walker answers.
    if let Some(r) = fold_constant(ctx, e) {
        return r;
    }
    match &e.expr {
        Expr::Literal(lit) => compile_literal_t(ctx, lit),
        Expr::Ident(name) => {
            if let Some(&r) = ctx.locals.get(name) {
                return Ok(r);
            }
            // A runtime-list iteration variable resolves to that list's element
            // column, read at the inner loop's index.
            if let Some(r) = ctx.iter_var_slot(name, None)? {
                return Ok(r);
            }
            if declares_list(ctx.schema, name) {
                return Err(LowerError::unsupported("list-valued expression"));
            }
            ctx.slot(name.clone())
        }
        Expr::Select(sel) => {
            // `list[k].field`: a constant index into the row's list, then one
            // of its element columns. `resolve_path` stops at the index, so
            // recognise the shape before asking it.
            if let (false, Expr::Call(inner)) = (sel.test, &sel.operand.expr) {
                if inner.func_name == ops::INDEX && inner.args.len() == 2 {
                    if let (Ok(base), Some(k)) =
                        (resolve_path(&inner.args[0]), as_int_literal(&inner.args[1]))
                    {
                        if declares_list(ctx.schema, &base) {
                            return lower_const_index(ctx, &base, Some(&sel.field), k);
                        }
                    }
                }
            }
            let path = resolve_path(e)?;
            let (root, field) = path.split_once('.').unwrap_or((path.as_str(), ""));
            if let Some(r) = ctx.iter_var_slot(root, Some(field))? {
                return Ok(r);
            }
            if ctx.locals.contains_key(root) {
                return Err(LowerError::unsupported(
                    "field access on comprehension variable",
                ));
            }
            if declares_list(ctx.schema, &path) {
                return Err(LowerError::unsupported("list-valued expression"));
            }
            ctx.slot(path)
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
            let r = ctx.fresh(ValType::Bool);
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
            // A string literal is a loop invariant, but unlike an int or a
            // `double` it is not a program CONSTANT: its `i64` id is whatever
            // the batch builder's interning assigns it, which is data. So it
            // gets a register and no words at all — the register arrives
            // already holding the id, the same way a column base does. The raw
            // content is recorded so the builder can resolve it and check it
            // against the column strings.
            let r = ctx.fresh(ValType::Str);
            ctx.scalar_seeds.push(ScalarSeed {
                kind: SeedKind::StrId(s.inner().to_string()),
                reg: r.idx,
            });
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

/// Narrow a float-bank value to a fresh int reg via a per-row `f64 as i64` cast
/// (`OP_F2I` -> `cast_float_to_int`), the inverse of [`emit_i2f`].
fn emit_f2i(ctx: &mut LowerCtxF, src: TReg) -> TReg {
    debug_assert_eq!(
        src.bank,
        ValType::Float,
        "emit_f2i: source must be float-banked"
    );
    let r = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_F2I, src.idx as i64, r.idx as i64]);
    r
}

/// CEL's comparison type classes.
///
/// Equality is heterogeneous ACROSS classes and answers rather than raising:
/// `1 == "ab"` is `false`, `1 != "ab"` is `true`, and so is every other pairing
/// of two different classes. Ordering across classes stays `NoSuchOverload`.
/// `int`, `uint` and `double` are ONE class — they compare numerically, so
/// `1 == 1u` is `true` and `1u < 2.0` is `true`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CmpClass {
    Numeric,
    Bool,
    Str,
    Timestamp,
    Duration,
}

fn cmp_class(bank: ValType) -> CmpClass {
    match bank {
        ValType::Int | ValType::UInt | ValType::Float => CmpClass::Numeric,
        ValType::Bool => CmpClass::Bool,
        ValType::Str => CmpClass::Str,
        ValType::Timestamp => CmpClass::Timestamp,
        ValType::Duration => CmpClass::Duration,
    }
}

/// The bank `name` produces from temporal operands, or `None` if this is not a
/// temporal arithmetic overload.
///
/// These are exactly the overloads the evaluator reaches, which dispatch on the
/// LEFT operand through `Adder`/`Subtractor` (`common/types/duration.rs`,
/// `common/types/timestamp.rs`) — not the wider set `Value::add` in `objects.rs`
/// appears to offer, which that path does not use. In particular
/// **`duration + timestamp` is unsupported** even though `timestamp + duration`
/// is, there is no `duration - timestamp`, and `*` `/` `%` have no temporal
/// overload at all.
///
/// The walker's `timestamp ± duration` additionally rejects a result outside
/// the cel-spec range (year 1 to 9999). That check cannot fire here: i64
/// nanoseconds only spans 1677 to 2262, so every representable result is inside
/// it.
fn temporal_arith_result(name: &str, a: ValType, b: ValType) -> Option<ValType> {
    use ValType::{Duration, Timestamp};
    match (name, a, b) {
        (ops::ADD, Duration, Duration) => Some(Duration),
        (ops::ADD, Timestamp, Duration) => Some(Timestamp),
        (ops::SUBSTRACT, Duration, Duration) => Some(Duration),
        (ops::SUBSTRACT, Timestamp, Duration) => Some(Timestamp),
        (ops::SUBSTRACT, Timestamp, Timestamp) => Some(Duration),
        _ => None,
    }
}

/// Match a member call against the four pure string predicates, resolving its
/// argument. `None` is "not one of these names"; `Some(Err(_))` is one of them
/// that this lowering cannot take — a non-literal argument (there would be no
/// single table to build) or, for `matches`, a regex the walker itself would
/// reject.
fn str_predicate(name: &str, args: &[IdedExpr]) -> Option<Result<StrPredicate, LowerError>> {
    let is_predicate = matches!(name, "startsWith" | "endsWith" | "contains" | "matches");
    if !is_predicate {
        return None;
    }
    Some((|| {
        if args.len() != 1 {
            return Err(LowerError::unsupported(format!("`{name}` arity")));
        }
        let arg = as_string_literal(&args[0]).ok_or_else(|| {
            // A per-row argument would need a table per row, which is the work
            // the table exists to avoid.
            LowerError::unsupported(format!("`{name}` argument must be a string literal"))
        })?;
        Ok(match name {
            "startsWith" => StrPredicate::StartsWith(arg.to_string()),
            "endsWith" => StrPredicate::EndsWith(arg.to_string()),
            "contains" => StrPredicate::Contains(arg.to_string()),
            #[cfg(feature = "regex")]
            "matches" => {
                // An invalid regex is an ERROR in the tree-walker, not `false`.
                // Declining leaves the caller on `Program::execute`, which
                // raises it; answering would swallow it.
                regex::Regex::new(arg).map_err(|e| {
                    LowerError::unsupported(format!("`{arg}` is not a valid regex: {e}"))
                })?;
                StrPredicate::Matches(arg.to_string())
            }
            #[cfg(not(feature = "regex"))]
            "matches" => return Err(LowerError::unsupported("`matches` needs the regex feature")),
            _ => unreachable!("name matched one of the four predicates"),
        })
    })())
}

/// Read `pred`'s answer for `s` out of the bind-time table: `ea = id * 8`, then
/// the same `*(base + ea)` load a column read is.
fn emit_str_predicate(ctx: &mut LowerCtxF, pred: StrPredicate, s: TReg) -> TReg {
    let table = ctx.fresh(ValType::Int);
    ctx.scalar_seeds.push(ScalarSeed {
        kind: SeedKind::StrPredicate(pred),
        reg: table.idx,
    });
    // `8` is the element stride, a property of the table and not of the batch,
    // so unlike the table's address it is a genuine immediate.
    let stride = emit_int_const(ctx, 8);
    let ea = emit_bin(ctx, OP_MUL, s, stride, ValType::Int);
    let d = ctx.fresh(ValType::Bool);
    ctx.body
        .extend_from_slice(&[OP_COL_LOAD, table.idx as i64, ea.idx as i64, d.idx as i64]);
    d
}

impl LowerCtxF<'_> {
    /// Classify one operand of a concatenation, or refuse it.
    fn concat_side(&mut self, e: &IdedExpr, r: TReg) -> Result<ConcatSide, LowerError> {
        if let Some(lit) = as_string_literal(e) {
            return Ok(ConcatSide::Literal(lit.to_string()));
        }
        // A register that already IS a derived concatenation, so chains fold
        // into one table rather than refusing.
        if let Some(slot) = self.slots.iter().find(|s| s.reg == r.idx) {
            if let Some(k) = concat_slot_index(&slot.path) {
                return Ok(ConcatSide::Derived(k));
            }
            if slot.ty == ValType::Str {
                return Ok(ConcatSide::Column(slot.path.clone()));
            }
        }
        Err(LowerError::unsupported(
            "string concatenation of an operand that is neither a column nor a literal",
        ))
    }

    /// The slot path a register arrived in, or `None` if it is not a slot
    /// (a literal's seeded register, or a computed one).
    fn slot_path_of(&self, r: TReg) -> Option<String> {
        self.slots
            .iter()
            .find(|s| s.reg == r.idx && s.ty == r.bank)
            .map(|s| s.path.clone())
    }

    /// The slot for `spec`, reusing one already recorded so a repeated
    /// sub-expression materializes a single column.
    fn concat_slot(&mut self, spec: ConcatSpec) -> TReg {
        let k = match self.concats.iter().position(|s| *s == spec) {
            Some(k) => k,
            None => {
                self.concats.push(spec);
                self.concats.len() - 1
            }
        };
        self.slot_typed(concat_slot_path(k), ValType::Str)
    }
}

/// `size(e)` where `e` is not a declared column but may still be a DERIVED
/// string one — a `string(x)` conversion or a concatenation. Those carry
/// characters too, so their byte length is the same kind of derived column,
/// keyed on the slot the string arrived in.
fn size_of_derived_string(ctx: &mut LowerCtxF, e: &IdedExpr) -> Result<TReg, LowerError> {
    let a = compile_t(ctx, e)?;
    let path = (a.bank == ValType::Str)
        .then(|| ctx.slot_path_of(a))
        .flatten()
        .ok_or_else(|| LowerError::unsupported("size() of a non-column argument"))?;
    Ok(ctx.slot_typed(size_slot_path(&path), ValType::Int))
}

/// A string the lowering knows without looking at the row. Like a written
/// literal it gets a seeded register rather than an immediate, because its id
/// is still the batch's to assign.
fn emit_str_const(ctx: &mut LowerCtxF, text: String) -> TReg {
    let r = ctx.fresh(ValType::Str);
    ctx.scalar_seeds.push(ScalarSeed {
        kind: SeedKind::StrId(text),
        reg: r.idx,
    });
    r
}

/// A `bool` the lowering knows without looking at the row, hoisted to the
/// prelude like any other constant.
fn emit_bool_const(ctx: &mut LowerCtxF, v: bool) -> TReg {
    let r = ctx.fresh(ValType::Bool);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST, v as i64, r.idx as i64]);
    r
}

/// A zero in the int file, for the sign tests a mixed int/uint comparison needs.
fn emit_zero_const(ctx: &mut LowerCtxF) -> TReg {
    let r = ctx.fresh(ValType::Int);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST, 0, r.idx as i64]);
    r
}

/// Emit `[op, a, b, dst]` into the body and hand back `dst`.
fn emit_bin(ctx: &mut LowerCtxF, op: i64, a: TReg, b: TReg, bank: ValType) -> TReg {
    let d = ctx.fresh(bank);
    ctx.body
        .extend_from_slice(&[op, a.idx as i64, b.idx as i64, d.idx as i64]);
    d
}

/// Lower one of the six comparisons over one `int` and one `uint` operand.
///
/// Neither machine compare answers this on its own: the signed one misreads a
/// uint above `i64::MAX`, the unsigned one misreads a negative int as huge. The
/// int's sign decides between them — every negative int is below every uint —
/// so each comparison is one sign test combined with the unsigned compare of
/// the two bit patterns. Six shapes, two combiners: `(i >= 0) & u_cmp` where a
/// negative int settles the answer as false, `(i < 0) | u_cmp` where it settles
/// it as true.
///
/// `int_literal` is the int operand's value when the source spelled it as one.
/// `u > 0` and `u < 100` are how a policy usually meets a uint, and there the
/// sign is known while lowering: the guard folds away and the comparison is the
/// single unsigned op it would have been in one bank, three ops down to one.
fn lower_int_uint_cmp(
    ctx: &mut LowerCtxF,
    name: &str,
    a: TReg,
    b: TReg,
    int_literal: Option<i64>,
) -> Result<TReg, LowerError> {
    let int_on_left = a.bank == ValType::Int;
    let (i, u) = if int_on_left { (a, b) } else { (b, a) };
    // `and` reads "a negative int makes this false", `or` "…makes this true".
    // The two orderings that a negative int settles as TRUE — `int < uint` and
    // its mirror `uint > int` — share one shape; so do the two it settles as
    // false. Equality is symmetric, so only the combiner differs there.
    let (and, lhs, rhs, uop) = match (name, int_on_left) {
        (ops::EQUALS, _) => (true, i, u, OP_EQ),
        (ops::NOT_EQUALS, _) => (false, i, u, OP_NE),
        (ops::LESS, true) | (ops::GREATER, false) => (false, i, u, OP_ULT),
        (ops::LESS_EQUALS, true) | (ops::GREATER_EQUALS, false) => (false, i, u, OP_ULE),
        (ops::GREATER, true) | (ops::LESS, false) => (true, u, i, OP_ULT),
        (ops::GREATER_EQUALS, true) | (ops::LESS_EQUALS, false) => (true, u, i, OP_ULE),
        _ => unreachable!("caller matched one of the six comparison ops"),
    };
    match int_literal {
        // The sign settles the whole comparison on its own.
        Some(v) if v < 0 => Ok(emit_bool_const(ctx, !and)),
        // Known non-negative: the unsigned compare is the answer.
        Some(_) => Ok(emit_bin(ctx, uop, lhs, rhs, ValType::Bool)),
        None => {
            let zero = emit_zero_const(ctx);
            let sign = emit_bin(ctx, if and { OP_GE } else { OP_LT }, i, zero, ValType::Bool);
            let cmp = emit_bin(ctx, uop, lhs, rhs, ValType::Bool);
            Ok(emit_bin(
                ctx,
                if and { OP_AND } else { OP_OR },
                sign,
                cmp,
                ValType::Bool,
            ))
        }
    }
}

/// Widen an int-bank value to a fresh float reg via a per-row `int as f64` cast
/// (`OP_I2F` -> `cast_int_to_float`). Emitted into the body: unlike a literal
/// (folded to a prelude constant), a data-dependent int is cast per row.
fn emit_i2f(ctx: &mut LowerCtxF, src: TReg) -> TReg {
    debug_assert_eq!(
        src.bank,
        ValType::Int,
        "emit_i2f: source must be int-banked"
    );
    let r = ctx.fresh(ValType::Float);
    ctx.body
        .extend_from_slice(&[OP_I2F, src.idx as i64, r.idx as i64]);
    r
}

/// Widen a uint-bank value to a fresh float reg (`OP_U2F`). Same shape as
/// [`emit_i2f`], through `u64` rather than `i64` so a uint above `i64::MAX`
/// does not widen to a negative double.
fn emit_u2f(ctx: &mut LowerCtxF, src: TReg) -> TReg {
    debug_assert_eq!(
        src.bank,
        ValType::UInt,
        "emit_u2f: source must be uint-banked"
    );
    let r = ctx.fresh(ValType::Float);
    ctx.body
        .extend_from_slice(&[OP_U2F, src.idx as i64, r.idx as i64]);
    r
}

/// Narrow a float-bank value to a fresh uint reg (`OP_F2U`). The unsigned twin
/// of [`emit_f2i`]; it saturates at different bounds, so `uint(-1.5)` is `0u`
/// where `int(-1.5)` is `-1`.
fn emit_f2u(ctx: &mut LowerCtxF, src: TReg) -> TReg {
    debug_assert_eq!(
        src.bank,
        ValType::Float,
        "emit_f2u: source must be float-banked"
    );
    let r = ctx.fresh(ValType::UInt);
    ctx.body
        .extend_from_slice(&[OP_F2U, src.idx as i64, r.idx as i64]);
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
    // Member syntax `x.f(a)` is NOT sugar for `f(x, a)` in this tree-walker: the
    // two spellings look in DISJOINT namespaces. A global call resolves through
    // `find_overload` (`objects.rs:1331`), which sees only `add_overload`
    // registrations; a member call resolves through `find_member_overload`
    // (`objects.rs:1364`), which sees only `add_member_overload` ones. Of the
    // whole stdlib exactly one name (`size`) is registered both ways, and it is
    // out of subset anyway.
    //
    // So a member call is handled here in RECEIVER form or not at all — rewriting
    // it to the global form would make the JIT answer `"...".timestamp()` or
    // `x.double()`, which the walker rejects as an `UndeclaredReference`.
    if let Some(target) = &call.target {
        // Receiver-only stdlib accessors are registered with
        // `add_member_overload` (`common/types/duration.rs:191-222`,
        // `common/types/timestamp.rs:278-357`), so they exist ONLY in receiver
        // form: the global spelling `getHours(d)` is an `UndeclaredReference`
        // error in the tree-walker, and is left to fall through to the `call
        // `{name}`` bail below.
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
        // The pure single-string predicates. Each answers a question about the
        // CHARACTERS, which the machine does not carry — but the answer depends
        // only on WHICH string, and ids are a dense `0..k` over the batch's
        // distinct strings. So the answer is computed once per distinct string
        // at bind and read per row out of a table indexed by the id.
        if let Some(pred) = str_predicate(&call.func_name, &call.args) {
            let pred = pred?;
            let a = compile_t(ctx, target)?;
            if a.bank != ValType::Str {
                return Err(LowerError::unsupported(format!(
                    "`{}` on a non-string receiver",
                    call.func_name
                )));
            }
            return Ok(emit_str_predicate(ctx, pred, a));
        }
        // `size` is the ONE stdlib name registered in both namespaces
        // (`string.rs:292` + `:299`, and likewise for list/map/bytes), so for it
        // alone `x.size()` and `size(x)` really are the same function and the
        // rewrite to global form is sound.
        if call.func_name == "size" && call.args.is_empty() {
            return compile_call_t(
                ctx,
                &CallExpr {
                    func_name: call.func_name.clone(),
                    target: None,
                    args: vec![(**target).clone()],
                },
            );
        }
        return Err(LowerError::unsupported(format!(
            "member call `{}`",
            call.func_name
        )));
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
        ctx.temporal_consts.push(nanos);
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
        ctx.temporal_consts.push(nanos);
        return Ok(r);
    }

    // `size(x)`. The tree-walker's `String::size` is `str::len()` — the UTF-8
    // BYTE length (`common/types/string.rs:93`) — and `DefaultList::size` is
    // `Vec::len` (`list.rs:188`).
    //
    // A LITERAL list has a green length, so it folds to a prelude constant. A
    // string column's length cannot be computed in the loop at all (the machine
    // carries a string as a 64-bit id and has no bytes to count), so
    // it is read from a DERIVED column under the synthetic slot path
    // `size(<path>)`, which the batch builder materializes from the same strings
    // it interns. That is the columnar move — a length is column metadata, the
    // way an Arrow offsets buffer makes it O(1) — not a computation the trace
    // skipped.
    if name == "size" && call.args.len() == 1 {
        if let Expr::List(list) = &call.args[0].expr {
            return Ok(emit_int_const(ctx, list.elements.len() as i64));
        }
        // A string literal has its bytes right here, so its length is green
        // too. `str::len` is the same BYTE count the walker reports.
        if let Some(lit) = as_string_literal(&call.args[0]) {
            return Ok(emit_int_const(ctx, lit.len() as i64));
        }
        // `size(list.map(..))` / `size(list.filter(..))`: the accumulator is a
        // list this machine has no value for, but its LENGTH is an int, and a
        // length is all `size` asks of it.
        if let Expr::Comprehension(comp) = &call.args[0].expr {
            return compile_comprehension_len(ctx, comp);
        }
        let path = match &call.args[0].expr {
            Expr::Ident(n) if !ctx.locals.contains_key(n) => n.clone(),
            Expr::Select(_) => resolve_path(&call.args[0])?,
            // Anything else may still be a derived string.
            _ => return size_of_derived_string(ctx, &call.args[0]),
        };
        // A `string` column carries its byte length as a derived column, and a
        // LIST column carries its per-row element count as the same derived
        // column — the length half of the `(offset, size)` pair that locates a
        // row's elements. A map/bytes column has no representation on this
        // machine at all.
        if ctx.schema.get(&path).copied() != Some(ValType::Str) && !declares_list(ctx.schema, &path)
        {
            return Err(LowerError::unsupported(
                "size() of a non-string, non-list column",
            ));
        }
        return Ok(ctx.slot_typed(size_slot_path(&path), ValType::Int));
    }

    // `string(x)`. The conversions are plain Rust formatting
    // (`common/types/string.rs`): `Int`/`UInt`/`Double` go through
    // `to_string`, a `Timestamp` through `to_rfc3339`, a `Duration` through
    // `format_duration`. All of them produce CHARACTERS, which the machine does
    // not carry — a string is a rank — so the answer arrives the same way
    // `size(x)` does, as a column the batch builder materializes.
    //
    // Note `string(bool)` is NOT among them: the walker's match has no `Bool`
    // arm and raises a `FunctionError`, so declining here is what agrees.
    if name == "string" && call.args.len() == 1 {
        // A literal's conversion is itself a constant, so it folds here — the
        // same `to_string` the walker would reach, run once instead of per row.
        // `Boolean` is left out for the same reason as a bool column.
        if let Expr::Literal(lit) = &call.args[0].expr {
            let folded = match lit {
                LiteralValue::String(_) => return compile_t(ctx, &call.args[0]),
                LiteralValue::Int(i) => i.into_inner().to_string(),
                LiteralValue::UInt(u) => u.into_inner().to_string(),
                LiteralValue::Double(f) => f.into_inner().to_string(),
                LiteralValue::Boolean(_) | LiteralValue::Bytes(_) | LiteralValue::Null => {
                    return Err(LowerError::unsupported("string() of this literal"))
                }
            };
            return Ok(emit_str_const(ctx, folded));
        }
        // Resolve the argument to a column path BEFORE compiling it, so a
        // decline emits nothing: `compile_t` allocates slots and appends ops.
        let path = match &call.args[0].expr {
            Expr::Ident(n) if !ctx.locals.contains_key(n) => n.clone(),
            Expr::Select(_) => resolve_path(&call.args[0])?,
            _ => return Err(LowerError::unsupported("string() of a non-column argument")),
        };
        return match ctx.schema.get(&path).copied() {
            // On a string the conversion is the identity, needing no
            // characters and no derived column: the rank already IS the answer.
            Some(ValType::Str) => compile_t(ctx, &call.args[0]),
            Some(
                ValType::Int
                | ValType::UInt
                | ValType::Float
                | ValType::Timestamp
                | ValType::Duration,
            ) => Ok(ctx.slot_typed(string_slot_path(&path), ValType::Str)),
            _ => Err(LowerError::unsupported(
                "string() of a bool, list or undeclared column",
            )),
        };
    }

    // Numeric type conversions. `double`/`int`/`uint` are global (non-member)
    // overloads whose `Kind`-dispatched bodies (`common/types/double.rs:199`,
    // `int.rs:229`, `uint.rs:250`) are TOTAL for the numeric arguments — a plain
    // Rust `as` cast, no error — so each lowers to a pure register move, a bank
    // relabel, or one widening cast. The string arguments parse and the rest are
    // a `FunctionError`; both bail.
    //
    // The signed/unsigned pairs are FREE: `int(uint)` is `u64 as i64` and
    // `uint(int)` is `i64 as u64`, i.e. raw reinterpretations, and the int
    // register file already carries a uint as its raw 64-bit pattern. Only the
    // bank label on the result changes, so no instruction is emitted at all.
    if matches!(name, "double" | "int" | "uint") && call.args.len() == 1 {
        let a = compile_t(ctx, &call.args[0])?;
        return match (name, a.bank) {
            // Identity conversions.
            ("double", ValType::Float) | ("int", ValType::Int) | ("uint", ValType::UInt) => Ok(a),
            // Widening: `i64 as f64` per row, the same `cast_int_to_float` the
            // mixed int/float comparisons already use.
            ("double", ValType::Int) => Ok(emit_i2f(ctx, a)),
            // Reinterpretations within the int register file.
            ("int", ValType::UInt) => Ok(TReg {
                bank: ValType::Int,
                idx: a.idx,
            }),
            ("uint", ValType::Int) => Ok(TReg {
                bank: ValType::UInt,
                idx: a.idx,
            }),
            // `u64 as f64`, which differs from `i64 as f64` above 2^63:
            // `double(18446744073709551615u)` is `1.8446744073709552e19`, not
            // `-1.0`.
            ("double", ValType::UInt) => Ok(emit_u2f(ctx, a)),
            // Both narrowings truncate toward zero and SATURATE, with NaN
            // mapping to `0` — total, exactly what the walker's plain `as` cast
            // does, so neither needs a guard. They saturate at different bounds,
            // which is why `uint` cannot reuse the signed op: `uint(-1.5)` is
            // `0u` where `int(-1.5)` is `-1`.
            ("int", ValType::Float) => Ok(emit_f2i(ctx, a)),
            ("uint", ValType::Float) => Ok(emit_f2u(ctx, a)),
            _ => Err(LowerError::unsupported(format!(
                "{name}() of a non-numeric argument"
            ))),
        };
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
            // A DECLARED list column has a red length, so there is nothing to
            // unroll against — the membership test becomes an inner loop, the
            // same one `exists` gets, which is what `in` means over a list.
            _ => match resolve_path(&call.args[1]) {
                Ok(path) if declares_list(ctx.schema, &path) => {
                    return lower_runtime_in(ctx, &call.args[0], &path)
                }
                _ => return Err(LowerError::unsupported("@in non-literal container")),
            },
        };
        let x = compile_t(ctx, &call.args[0])?;
        if elements.is_empty() {
            // `x in []` is always false (the operand is still evaluated above).
            let d = ctx.fresh(ValType::Bool);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST, 0, d.idx as i64]);
            return Ok(d);
        }
        let eq_op = if x.bank == ValType::Float {
            OP_FEQ
        } else {
            OP_EQ
        };
        let mut acc: Option<TReg> = None;
        for e in elements {
            let ev = compile_t(ctx, e)?;
            if ev.bank != x.bank {
                return Err(LowerError::unsupported("@in heterogeneous element"));
            }
            let t = ctx.fresh(ValType::Bool);
            ctx.body
                .extend_from_slice(&[eq_op, x.idx as i64, ev.idx as i64, t.idx as i64]);
            acc = Some(match acc {
                None => t,
                Some(prev) => {
                    let o = ctx.fresh(ValType::Bool);
                    ctx.body.extend_from_slice(&[
                        OP_OR,
                        prev.idx as i64,
                        t.idx as i64,
                        o.idx as i64,
                    ]);
                    o
                }
            });
        }
        return Ok(acc.expect("non-empty element list"));
    }

    // ternary `c ? t : f` — branchless blend on a bool condition. Int arms use
    // an arithmetic SELECT; float arms use a bit-mask FSELECT (bit-exact, no
    // reassociation). Mixed-bank arms bail: the tree-walker yields int-or-float
    // per row, which no single result bank can carry.
    if name == ops::CONDITIONAL {
        if call.args.len() != 3 {
            return Err(LowerError::unsupported("_?_:_ arity"));
        }
        let c = compile_t(ctx, &call.args[0])?;
        // `1 ? x : y` is NoSuchOverload in the tree-walker: the condition is
        // bool, not "anything in the int bank".
        if c.bank != ValType::Bool {
            return Err(LowerError::unsupported("ternary condition must be bool"));
        }
        let t = compile_t(ctx, &call.args[1])?;
        let f = compile_t(ctx, &call.args[2])?;
        let (op, bank) = match (t.bank, f.bank) {
            (ValType::Float, ValType::Float) => (OP_FSELECT, ValType::Float),
            // Every other bank rides the int file, so the arithmetic select
            // hands back the chosen arm's word unchanged — a `0`/`1` bool, a
            // uint's bit pattern, a string's id, a temporal's nanoseconds. Only
            // arms of the SAME bank blend: the tree-walker yields one type or
            // the other per row, which no single result bank can carry.
            (x, y) if x == y => (OP_SELECT, x),
            _ => return Err(LowerError::unsupported("mixed-bank ternary arms")),
        };
        let d = ctx.fresh(bank);
        ctx.body
            .extend_from_slice(&[op, c.idx as i64, t.idx as i64, f.idx as i64, d.idx as i64]);
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
        // A caller may flatten one index into a column of its own, in which
        // case `base[k]` names that column and reads like any other row value.
        let path = format!("{base}[{idx}]");
        if let Some(&ty) = ctx.schema.get(&path) {
            return Ok(ctx.slot_typed(path, ty));
        }
        // Otherwise it indexes into the row's LIST, which is a bounds-checked
        // read of the flattened element column rather than a row column: the
        // element at `k` sits at a different place in every row.
        if declares_list(ctx.schema, &base) {
            return lower_const_index(ctx, &base, None, idx);
        }
        return Err(LowerError::unsupported(format!("undeclared path `{path}`")));
    }

    // n-ary boolean fold — bool operands, bool result. `OP_AND`/`OP_OR` are
    // bitwise, which is only the logical answer on 0/1, so an int operand must
    // bail rather than quietly compute `1 & 2`: `1 && 2` is NoSuchOverload.
    if name == ops::LOGICAL_AND || name == ops::LOGICAL_OR {
        if call.args.len() < 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let op = if name == ops::LOGICAL_AND {
            OP_AND
        } else {
            OP_OR
        };
        // CEL's logical operators are commutative and absorb the other operand
        // whole — its value, its errors and its type. `1 || true` is `true` and
        // `1 && false` is `false`, though `1` alone is NoSuchOverload under
        // either. So an absorbing literal answers before anything else is
        // compiled; without one, `1 || false` stays the type error it is.
        let absorbing = name == ops::LOGICAL_OR;
        if call
            .args
            .iter()
            .any(|a| as_bool_literal(a) == Some(absorbing))
        {
            return Ok(emit_bool_const(ctx, absorbing));
        }
        let mut acc = compile_t(ctx, &call.args[0])?;
        if acc.bank != ValType::Bool {
            return Err(LowerError::unsupported("boolean operand must be bool"));
        }
        for arg in &call.args[1..] {
            let b = compile_t(ctx, arg)?;
            if b.bank != ValType::Bool {
                return Err(LowerError::unsupported("boolean operand must be bool"));
            }
            let d = ctx.fresh(ValType::Bool);
            ctx.body
                .extend_from_slice(&[op, acc.idx as i64, b.idx as i64, d.idx as i64]);
            acc = d;
        }
        return Ok(acc);
    }

    // comparisons — same-bank operands, `bool` result.
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
        // Two LIST columns compare elementwise. Recognized before the operands
        // are compiled, because a list operand has no register to compile into.
        if matches!(name, ops::EQUALS | ops::NOT_EQUALS) {
            if let (Ok(l), Ok(r)) = (resolve_path(&call.args[0]), resolve_path(&call.args[1])) {
                if declares_list(ctx.schema, &l) && declares_list(ctx.schema, &r) {
                    let eq = lower_list_equality(ctx, &l, &r)?;
                    return Ok(match name {
                        ops::EQUALS => eq,
                        _ => {
                            let d = ctx.fresh(ValType::Bool);
                            ctx.body
                                .extend_from_slice(&[OP_NOT, eq.idx as i64, d.idx as i64]);
                            d
                        }
                    });
                }
            }
        }
        let (mut a, mut b) = compile_cmp_operands(ctx, &call.args[0], &call.args[1])?;
        // Operands of two DIFFERENT type classes are never equal and always
        // unequal, whatever the row holds, so `==`/`!=` fold to a constant. The
        // operands still compiled above, so a trap either side raises reaches
        // the row the way the tree-walker's does; only the comparison itself is
        // constant. Ordering across classes is NoSuchOverload and bails.
        let (class_a, class_b) = (cmp_class(a.bank), cmp_class(b.bank));
        if class_a != class_b {
            return match name {
                ops::EQUALS => Ok(emit_bool_const(ctx, false)),
                ops::NOT_EQUALS => Ok(emit_bool_const(ctx, true)),
                _ => Err(LowerError::unsupported(format!(
                    "ordering across type classes ({class_a:?} vs {class_b:?})"
                ))),
            };
        }
        // String, timestamp and duration all compare as signed ints, for the
        // same reason: their `i64` encoding is order-preserving. A timestamp and
        // a duration are i64 nanoseconds, whose signed order is the
        // chronological / magnitude order; a string is its RANK among the
        // batch's distinct strings (`bytecode::StrDict`), whose signed order is
        // lexicographic order. So all six comparisons are the signed int op.
        // The class check above already kept the three from mixing.
        if matches!(
            a.bank,
            ValType::Str | ValType::Timestamp | ValType::Duration
        ) {
            return Ok(emit_bin(ctx, iop, a, b, ValType::Bool));
        }
        // One int and one uint operand: compare NUMERICALLY, which is neither
        // the signed nor the unsigned machine compare. The int side's sign
        // decides which — a negative int is below every uint — so each
        // comparison is a sign test guarding the unsigned compare of the two bit
        // patterns. `1u < -1` is false and `-1 < 1u` is true, the way the
        // tree-walker answers them.
        if (a.bank == ValType::Int && b.bank == ValType::UInt)
            || (a.bank == ValType::UInt && b.bank == ValType::Int)
        {
            let int_literal = if a.bank == ValType::Int {
                as_int_literal(&call.args[0])
            } else {
                as_int_literal(&call.args[1])
            };
            return lower_int_uint_cmp(ctx, name, a, b, int_literal);
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
            let d = ctx.fresh(ValType::Bool);
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
            // Ordering IS defined on bool (`false < true`, `common/types/bool.rs`
            // derives `Ord`), and `0`/`1` in the int file sorts the same way, so
            // all six use the signed int op. A bool mixed with any other bank
            // falls to the bail below: ordering is NoSuchOverload, and
            // `1 == true` answers `false` rather than comparing bits — neither is
            // something the int ops would produce.
            (ValType::Bool, ValType::Bool) => iop,
            (ValType::Float, ValType::Float) => fop,
            (ValType::Int, ValType::Float) => {
                a = emit_i2f(ctx, a);
                fop
            }
            (ValType::Float, ValType::Int) => {
                b = emit_i2f(ctx, b);
                fop
            }
            // A uint against a double widens the same way, through `u64`.
            (ValType::UInt, ValType::Float) => {
                a = emit_u2f(ctx, a);
                fop
            }
            (ValType::Float, ValType::UInt) => {
                b = emit_u2f(ctx, b);
                fop
            }
            _ => return Err(LowerError::unsupported("mixed-bank comparison")),
        };
        let d = ctx.fresh(ValType::Bool);
        ctx.body
            .extend_from_slice(&[op, a.idx as i64, b.idx as i64, d.idx as i64]);
        return Ok(d);
    }

    // arithmetic — same-bank operands, same-bank result. No float modulo.
    //
    // `+ - *` on `int` are OVERFLOW-CHECKED in the tree-walker
    // (`common/types/int.rs:79-187` uses `checked_add`/`checked_sub`/
    // `checked_mul` and raises `ExecutionError::Overflow`), so they lower to the
    // fused `OP_*_OVF` form — `Int*Ovf` + `GuardNoOverflow` in the trace, the
    // PyPy `int_add_ovf` shape — not to the plain wrapping ops. Plain `OP_ADD`/
    // `OP_MUL` stay reserved for the batch machinery's own counters and offsets
    // and for the calendar helpers, whose operands are bounded by construction.
    let arith = match name {
        ops::ADD => Some((OP_ADD_OVF, OP_UADD_OVF, Some(OP_FADD))),
        ops::SUBSTRACT => Some((OP_SUB_OVF, OP_USUB_OVF, Some(OP_FSUB))),
        ops::MULTIPLY => Some((OP_MUL_OVF, OP_UMUL_OVF, Some(OP_FMUL))),
        ops::DIVIDE => Some((OP_DIV_CHK, OP_UDIV, Some(OP_FDIV))),
        ops::MODULO => Some((OP_MOD_CHK, OP_UMOD, None)),
        _ => None,
    };
    if let Some((iop, uop, fop)) = arith {
        if call.args.len() != 2 {
            return Err(LowerError::unsupported(format!("{name} arity")));
        }
        let a = compile_t(ctx, &call.args[0])?;
        let b = compile_t(ctx, &call.args[1])?;
        // Every int-bank arithmetic opcode reachable from a user expression is a
        // TRAPPING (5-word) form carrying `OVF_FLAG_REG`: each of these five
        // operators is partial in the tree-walker on BOTH integer banks, so the
        // JIT either answers what the walker answers or records that it cannot
        // answer at all. The signed and unsigned peers differ only in which
        // bound they check — the values are bit-identical two's complement.
        let emit_trapping = |ctx: &mut LowerCtxF, op: i64, bank: ValType| {
            let d = ctx.fresh(bank);
            ctx.body.extend_from_slice(&[
                op,
                a.idx as i64,
                b.idx as i64,
                d.idx as i64,
                OVF_FLAG_REG as i64,
            ]);
            d
        };
        // `string + string` is concatenation, which produces CHARACTERS the
        // machine does not carry — a string is a rank. So, like `string(x)`,
        // the answer arrives as a column the batch builder materializes, and
        // ranking it alongside every other string is what gives the result an
        // id to compare. Two literals fold outright.
        if name == ops::ADD && a.bank == ValType::Str && b.bank == ValType::Str {
            if let (Some(x), Some(y)) = (
                as_string_literal(&call.args[0]),
                as_string_literal(&call.args[1]),
            ) {
                return Ok(emit_str_const(ctx, format!("{x}{y}")));
            }
            let left = ctx.concat_side(&call.args[0], a)?;
            let right = ctx.concat_side(&call.args[1], b)?;
            return Ok(ctx.concat_slot(ConcatSpec { left, right }));
        }
        // Temporal arithmetic. Both banks are i64 nanoseconds, so the operation
        // itself is the same int add/sub — but the tree-walker's is chrono's,
        // whose range (`{secs: i64, nanos: i32}`) is far WIDER than i64
        // nanoseconds, so a machine add can overflow where the walker answers.
        // What makes the two agree is the bound recorded here and enforced on
        // the batch's columns: see [`LoweredF::temporal_bound`].
        if let Some(result_bank) = temporal_arith_result(name, a.bank, b.bank) {
            // One more `+`/`-` can combine at most one more leaf, so counting
            // the OPERATIONS bounds every intermediate: a tree of `m` of them
            // sums at most `m + 1` operands, each at most `temporal_bound`.
            // `iop` is already this operator's trapping int op, and the trap is
            // defence in depth — the bound is what makes it unreachable, so a
            // trap here would mean the bound was not enforced.
            ctx.temporal_ops += 1;
            return Ok(emit_trapping(ctx, iop, result_bank));
        }
        match (a.bank, b.bank) {
            (ValType::Int, ValType::Int) => Ok(emit_trapping(ctx, iop, ValType::Int)),
            (ValType::UInt, ValType::UInt) => Ok(emit_trapping(ctx, uop, ValType::UInt)),
            (ValType::Float, ValType::Float) => {
                let fop = fop.ok_or_else(|| LowerError::unsupported("float modulo"))?;
                let d = ctx.fresh(ValType::Float);
                ctx.body
                    .extend_from_slice(&[fop, a.idx as i64, b.idx as i64, d.idx as i64]);
                Ok(d)
            }
            // Not only the int/float mix: `string + string` is concatenation and
            // `timestamp + duration` is calendar arithmetic, both of which CEL
            // defines and both of which land here. Name the banks so the census
            // reports the operand types it actually declined rather than
            // filing every one of them under a numeric mismatch.
            _ => Err(LowerError::unsupported(format!(
                "arithmetic on {:?} and {:?}",
                a.bank, b.bank
            ))),
        }
    } else {
        match name {
            ops::LOGICAL_NOT => {
                if call.args.len() != 1 {
                    return Err(LowerError::unsupported("!_ arity"));
                }
                let a = compile_t(ctx, &call.args[0])?;
                // `!1` is NoSuchOverload; `OP_NOT` is only the logical negation
                // on 0/1.
                if a.bank != ValType::Bool {
                    return Err(LowerError::unsupported("! operand must be bool"));
                }
                let d = ctx.fresh(ValType::Bool);
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
                // `-bool` has no CEL overload, but this tier's contract is
                // bit-exact agreement with the evaluator it accelerates, and
                // that evaluator answers `-b` as `!b`
                // (`common/types/bool.rs`, `Bool::negate`). Declining would
                // make the JIT tier and the interpreter tier disagree on the
                // same expression, which is the one thing a JIT may not do.
                if a.bank == ValType::Bool {
                    let d = ctx.fresh(ValType::Bool);
                    ctx.body
                        .extend_from_slice(&[OP_NOT, a.idx as i64, d.idx as i64]);
                    return Ok(d);
                }
                let op = match a.bank {
                    ValType::Int => OP_NEG,
                    ValType::Float => OP_FNEG,
                    ValType::Bool => unreachable!("handled above"),
                    ValType::UInt => return Err(LowerError::unsupported("unary negate on uint")),
                    ValType::Str => return Err(LowerError::unsupported("unary negate on string")),
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
        // A declared list column has a RED length, so there is nothing to
        // unroll against: it gets a real inner loop instead.
        _ => match resolve_path(&comp.iter_range) {
            Ok(path) if declares_list(ctx.schema, &path) => {
                return compile_list_comprehension_t(ctx, comp, &path)
            }
            _ => {
                return Err(LowerError::unsupported(
                    "comprehension over non-literal range",
                ))
            }
        },
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

/// Whether `e` reads no variable, so its value is the same on every row.
///
/// A comprehension's own iteration and accumulator variables are bound WITHIN
/// the subtree, so a closed comprehension over a literal list stays closed.
fn is_constant(e: &IdedExpr, bound: &mut Vec<String>) -> bool {
    let closed = |x: &IdedExpr, b: &mut Vec<String>| is_constant(x, b);
    match &e.expr {
        Expr::Literal(_) => true,
        Expr::Ident(n) => bound.iter().any(|b| b == n),
        Expr::Select(sel) => closed(&sel.operand, bound),
        Expr::List(l) => l.elements.iter().all(|x| is_constant(x, bound)),
        Expr::Map(m) => m.entries.iter().all(|e| match &e.expr {
            EntryExpr::MapEntry(kv) => is_constant(&kv.key, bound) && is_constant(&kv.value, bound),
            // A struct field names a message type this machine has no schema
            // for, so a struct is never folded.
            EntryExpr::StructField(_) => false,
        }),
        Expr::Call(c) => {
            c.target.as_deref().is_none_or(|t| is_constant(t, bound))
                && c.args.iter().all(|a| is_constant(a, bound))
        }
        Expr::Comprehension(comp) => {
            if !is_constant(&comp.iter_range, bound) || !is_constant(&comp.accu_init, bound) {
                return false;
            }
            let depth = bound.len();
            bound.push(comp.iter_var.clone());
            if let Some(v) = &comp.iter_var2 {
                bound.push(v.clone());
            }
            bound.push(comp.accu_var.clone());
            let inner = is_constant(&comp.loop_step, bound) && is_constant(&comp.result, bound);
            bound.truncate(depth);
            inner
        }
        // A struct literal names a message type this machine has no schema for,
        // and an unspecified expression has no value at all.
        Expr::Struct(_) | Expr::Unspecified => false,
    }
}

/// Evaluate `e` with the tree-walker and emit its value as a constant, if it
/// reads no variable and the value lands in a bank the machine has.
///
/// `None` means "not a constant, carry on"; `Some(Err(..))` means it IS a
/// constant but not one this machine can hold — a list, a map, `null` — which
/// is a decline, since re-lowering it would only reach the same place.
fn fold_constant(ctx: &mut LowerCtxF, e: &IdedExpr) -> Option<Result<TReg, LowerError>> {
    // A bare literal is already the cheap path, and a bare identifier is never
    // constant; skipping both keeps the walk off the hot shapes.
    if matches!(e.expr, Expr::Literal(_) | Expr::Ident(_)) {
        return None;
    }
    if !is_constant(e, &mut Vec::new()) {
        return None;
    }
    let value = Value::resolve(e, &Context::default()).ok()?;
    Some(emit_constant_value(ctx, &value))
}

/// A tree-walker [`Value`] as a register holding it, for the banks the machine
/// has. The encodings are the ones a COLUMN of that type arrives in, so a
/// folded constant and a column value compare as themselves.
fn emit_constant_value(ctx: &mut LowerCtxF, v: &Value) -> Result<TReg, LowerError> {
    let (bank, word) = match v {
        Value::Int(i) => (ValType::Int, *i),
        Value::UInt(u) => (ValType::UInt, *u as i64),
        Value::Bool(b) => (ValType::Bool, *b as i64),
        Value::Float(f) => {
            let r = ctx.fresh(ValType::Float);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST_F, f.to_bits() as i64, r.idx as i64]);
            return Ok(r);
        }
        Value::String(s) => return Ok(emit_str_const(ctx, s.to_string())),
        Value::Timestamp(t) => (
            ValType::Timestamp,
            t.timestamp_nanos_opt()
                .ok_or_else(|| LowerError::unsupported("folded timestamp outside i64-nanos"))?,
        ),
        Value::Duration(d) => (
            ValType::Duration,
            d.num_nanoseconds()
                .ok_or_else(|| LowerError::unsupported("folded duration outside i64-nanos"))?,
        ),
        other => {
            return Err(LowerError::unsupported(format!(
                "constant of type `{}`",
                other.type_of()
            )))
        }
    };
    let r = ctx.fresh(bank);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST, word, r.idx as i64]);
    Ok(r)
}

/// `a == b` over two DECLARED list columns: equal lengths, and every element
/// equal at the same index.
///
/// One loop, not two. Both spans are walked at the SAME index `j`, each off its
/// own row-level `offset`, so this is the membership loop with a second column
/// read rather than a new loop shape.
///
/// The trip count is `same_len ? len(a) : 0`, computed branchlessly. That is
/// not an optimization but a safety requirement: walking `len(a)` elements when
/// `b` is shorter would read off the end of `b`'s span, which no result can
/// undo.
///
/// Element equality is per FIELD, over the field set the schema declares. Two
/// lists whose element fields differ decline: the tree-walker compares the
/// elements as maps and answers `false`, and declining leaves that answer to it
/// rather than guessing at a correspondence.
fn lower_list_equality(ctx: &mut LowerCtxF, a: &str, b: &str) -> Result<TReg, LowerError> {
    if !ctx.list_loop.is_empty() {
        return Err(LowerError::unsupported(
            "list equality inside a comprehension",
        ));
    }
    // The element fields each list declares, in one order for both.
    let fields = |list: &str| -> Vec<(Option<String>, ValType)> {
        let mut v: Vec<(Option<String>, ValType)> = ctx
            .schema
            .iter()
            .filter_map(|(k, t)| {
                let (l, f) = elem_slot_source(k)?;
                (l == list).then(|| (f.map(str::to_string), *t))
            })
            .collect();
        v.sort_by(|x, y| x.0.cmp(&y.0));
        v
    };
    let (fa, fb) = (fields(a), fields(b));
    if fa != fb {
        return Err(LowerError::unsupported(
            "list equality over unlike elements",
        ));
    }

    let len_a = ctx.slot_typed(size_slot_path(a), ValType::Int);
    let len_b = ctx.slot_typed(size_slot_path(b), ValType::Int);
    let off_a = ctx.slot_typed(offset_slot_path(a), ValType::Int);
    let off_b = ctx.slot_typed(offset_slot_path(b), ValType::Int);
    let one = emit_int_const(ctx, 1);
    let stride = emit_int_const(ctx, 8);

    // Unequal lengths settle it, and also make the trip count zero so the loop
    // never reads past the shorter span.
    let same_len = emit_bin(ctx, OP_EQ, len_a, len_b, ValType::Bool);
    let zero = emit_int_const(ctx, 0);
    let n = ctx.fresh(ValType::Int);
    ctx.body.extend_from_slice(&[
        OP_SELECT,
        same_len.idx as i64,
        len_a.idx as i64,
        zero.idx as i64,
        n.idx as i64,
    ]);

    // The accumulator is loop-carried, so it lives in a fixed register.
    let eq = ctx.fresh(ValType::Bool);
    emit_mov(ctx, same_len, eq);
    let j = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_LOAD_CONST, 0, j.idx as i64]);
    // Zero-trip guard: two empty lists are equal, and the back-edge is a
    // do-while.
    let zero_trip = ctx.emit_jump_if_above(one, n);

    let inner = ctx.body.len();
    let ia = emit_int_bin(ctx, OP_ADD, off_a, j);
    let ea_a = emit_int_bin(ctx, OP_MUL, ia, stride);
    let ib = emit_int_bin(ctx, OP_ADD, off_b, j);
    let ea_b = emit_int_bin(ctx, OP_MUL, ib, stride);
    for (field, ty) in &fa {
        let f = field.as_deref();
        let va = ctx.elem_slot(elem_slot_path(a, f), ea_a.idx)?;
        let vb = ctx.elem_slot(elem_slot_path(b, f), ea_b.idx)?;
        let op = if *ty == ValType::Float { OP_FEQ } else { OP_EQ };
        let same = emit_bin(ctx, op, va, vb, ValType::Bool);
        ctx.body
            .extend_from_slice(&[OP_AND, eq.idx as i64, same.idx as i64, eq.idx as i64]);
    }
    ctx.elem_map
        .retain(|(_, reg), _| *reg != ea_a.idx && *reg != ea_b.idx);
    ctx.body
        .extend_from_slice(&[OP_ADD, j.idx as i64, one.idx as i64, j.idx as i64]);
    ctx.emit_back_edge(n, j, inner);
    ctx.patch_jump(zero_trip);
    Ok(eq)
}

/// What a runtime-list comprehension's accumulator holds.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AccuMode {
    /// The accumulator's own value, in whatever bank `accu_init` lands in.
    /// What `all`, `exists` and `exists_one` need.
    Value,
    /// The LENGTH of the list the accumulator would have been, as an int. What
    /// `map` and `filter` build, and the only thing `size` asks of it.
    Length,
    /// The length AND the elements: each appended element is stored to the
    /// output buffers at `cursor`, which then advances. The accumulator still
    /// carries the count, so the row's value is its list's length and the
    /// elements are found by it.
    ///
    /// The cursor runs across the WHOLE batch, not the row — it is initialized
    /// in the prelude, which runs once before the outer loop, and never reset.
    /// That is what makes the output flat, exactly like the input's element
    /// buffer.
    Collect { cursor: usize },
}

/// Lower a top-level `list.map(..)` / `list.filter(..)` by COLLECTING it: the
/// elements stream to their own flat buffers and the row's value is the count.
///
/// `None` means "not that shape, lower it the ordinary way". `Some(Err(..))`
/// means it is that shape but something about it declines.
fn collect_list_result(ctx: &mut LowerCtxF, expr: &IdedExpr) -> Option<Result<TReg, LowerError>> {
    let Expr::Comprehension(comp) = &expr.expr else {
        return None;
    };
    // The same `map`/`filter` signature `compile_comprehension_len` checks: an
    // empty list in, the accumulator straight back out.
    if !matches!(&comp.accu_init.expr, Expr::List(l) if l.elements.is_empty()) {
        return None;
    }
    if !matches!(&comp.result.expr, Expr::Ident(n) if *n == comp.accu_var) {
        return None;
    }
    let path = resolve_path(&comp.iter_range).ok()?;
    if !declares_list(ctx.schema, &path) {
        return None;
    }
    Some(collect_list_comprehension(ctx, comp, &path))
}

fn collect_list_comprehension(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    path: &str,
) -> Result<TReg, LowerError> {
    if comp.iter_var2.is_some() {
        return Err(LowerError::unsupported("two-variable comprehension"));
    }
    // What the output's elements look like. `filter` hands back the element
    // itself, so the output fields are the source's; `map` computes a value,
    // so there is one unnamed field whose bank the body decides.
    let appends_element = matches!(
        &comp.loop_step.expr,
        Expr::Call(c)
            if c.args.len() == 3
                && matches!(&c.args[1].expr, Expr::Call(a)
                    if a.args.len() == 2
                        && matches!(&a.args[1].expr, Expr::List(l)
                            if l.elements.len() == 1
                                && matches!(&l.elements[0].expr, Expr::Ident(n)
                                    if *n == comp.iter_var)))
    );
    let fields: Vec<(Option<String>, ValType)> = if appends_element {
        let mut v: Vec<(Option<String>, ValType)> = ctx
            .schema
            .iter()
            .filter_map(|(k, t)| {
                let (l, f) = elem_slot_source(k)?;
                (l == path).then(|| (f.map(str::to_string), *t))
            })
            .collect();
        v.sort_by(|x, y| x.0.cmp(&y.0));
        v
    } else {
        // A `map` body's bank is not known until it is compiled, and it has to
        // be known to seed the buffer. Compiling the body twice would emit it
        // twice, so the bank is taken from a THROWAWAY lowering of the same
        // expression against the same schema — same body, same answer, and its
        // ops are discarded with it.
        let elem = map_body_bank(ctx, comp, path)?;
        vec![(None, elem)]
    };
    if fields.is_empty() {
        return Err(LowerError::unsupported(
            "collected list has no element column",
        ));
    }

    // Buffer bases are batch data, so they ride seeded registers like any
    // column base. The cursor is machinery: initialized once in the prelude,
    // which runs before the outer loop, and carried across every row.
    let base_regs: Vec<usize> = fields.iter().map(|_| ctx.fresh(ValType::Int).idx).collect();
    let cursor = ctx.fresh(ValType::Int);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST, 0, cursor.idx as i64]);
    ctx.list_output = Some(ListOutput {
        source: path.to_string(),
        fields,
        base_regs,
    });
    compile_list_comprehension_mode(ctx, comp, path, AccuMode::Collect { cursor: cursor.idx })
}

/// The bank a `map` body produces, from a throwaway lowering of the same
/// comprehension in `Length` mode — which compiles the body and discards it.
fn map_body_bank(
    ctx: &LowerCtxF,
    comp: &ComprehensionExpr,
    path: &str,
) -> Result<ValType, LowerError> {
    let mut probe = LowerCtxF {
        prelude: Vec::new(),
        body: Vec::new(),
        next_int: OVF_FLAG_REG + 1,
        next_float: 0,
        slots: Vec::new(),
        slot_map: HashMap::new(),
        locals: HashMap::new(),
        scalar_seeds: Vec::new(),
        temporal_ops: 0,
        temporal_consts: Vec::new(),
        concats: Vec::new(),
        elem_map: HashMap::new(),
        list_loop: Vec::new(),
        jump_fixups: Vec::new(),
        list_output: None,
        schema: ctx.schema,
    };
    // The appended expression, lowered on its own with the iteration variable
    // bound — which is what `Length` mode does to it for its errors.
    let Expr::Call(add) = &comp.loop_step.expr else {
        return Err(LowerError::unsupported("collected step is not an append"));
    };
    let Some(Expr::List(l)) = add.args.get(1).map(|a| &a.expr) else {
        return Err(LowerError::unsupported("collected step is not an append"));
    };
    let e = l
        .elements
        .first()
        .ok_or_else(|| LowerError::unsupported("collected step appends nothing"))?;
    let len = probe.slot_typed(size_slot_path(path), ValType::Int);
    let off = probe.slot_typed(offset_slot_path(path), ValType::Int);
    let _ = (len, off);
    let ea = probe.fresh(ValType::Int);
    probe.list_loop.push(ListLoop {
        iter_var: comp.iter_var.clone(),
        list: path.to_string(),
        ea_reg: ea.idx,
    });
    Ok(compile_t(&mut probe, e)?.bank)
}

/// `size(list.map(..))` / `size(list.filter(..))`: the same inner loop, with the
/// accumulator carrying the length of the list rather than the list.
///
/// A list is not a value on this machine, so `map` and `filter` had no
/// accumulator and declined outright — even where the only thing asked of the
/// list was how long it is, which is an int like any other.
///
/// ⚠️Reachable ONLY from `size`. An int length is not a substitute for the list
/// itself; handing this register to any other consumer would answer a list
/// question with a number.
fn compile_comprehension_len(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
) -> Result<TReg, LowerError> {
    // `map`/`filter` start from `[]` and hand the accumulator straight back, so
    // anything else is a comprehension whose length this does not know.
    match &comp.accu_init.expr {
        Expr::List(l) if l.elements.is_empty() => {}
        _ => {
            return Err(LowerError::unsupported(
                "size() of a non-list comprehension",
            ))
        }
    }
    match &comp.result.expr {
        Expr::Ident(n) if *n == comp.accu_var => {}
        _ => return Err(LowerError::unsupported("size() of a mapped comprehension")),
    }
    if comp.iter_var2.is_some() {
        return Err(LowerError::unsupported("two-variable comprehension"));
    }
    let path = resolve_path(&comp.iter_range)?;
    if !declares_list(ctx.schema, &path) {
        return Err(LowerError::unsupported("size() over a non-list range"));
    }
    compile_list_comprehension_mode(ctx, comp, &path, AccuMode::Length)
}

/// One iteration's contribution to a list accumulator's LENGTH.
///
/// The two shapes the macros desugar to, and no others:
///
/// * `map`    — `@result + [e]`, which appends exactly one element.
/// * `filter` — `c ? (@result + [e]) : @result`, which appends one or none.
///
/// `e` is still compiled and its register discarded. Dropping it would answer
/// where the tree-walker raises: `items.map(i, 1 / i.price)` over a zero price
/// is an error, not a length.
///
/// The one element NOT compiled is a bare `iter_var`, which is what `filter`
/// appends. Reading a bound variable cannot raise, so there is no error to
/// preserve — and a record list has no whole-element column to read it from,
/// only the fields the schema declares.
fn compile_len_step(
    ctx: &mut LowerCtxF,
    step: &IdedExpr,
    comp: &ComprehensionExpr,
    accu: TReg,
    mode: AccuMode,
) -> Result<TReg, LowerError> {
    let accu_var = comp.accu_var.as_str();
    let is_accu = |e: &IdedExpr| matches!(&e.expr, Expr::Ident(n) if n == accu_var);
    let is_iter = |e: &IdedExpr| matches!(&e.expr, Expr::Ident(n) if *n == comp.iter_var);
    let Expr::Call(call) = &step.expr else {
        return Err(LowerError::unsupported("size() of an opaque comprehension"));
    };
    match call.func_name.as_str() {
        // `@result + [e]`: one element, unconditionally.
        ops::ADD if call.args.len() == 2 && is_accu(&call.args[0]) => {
            let Expr::List(l) = &call.args[1].expr else {
                return Err(LowerError::unsupported("size() of a non-append step"));
            };
            for e in &l.elements {
                match mode {
                    // Collecting: the element's VALUE is the point, so it is
                    // compiled either way and written out.
                    AccuMode::Collect { cursor } => emit_element_store(ctx, e, comp, cursor)?,
                    // Counting: the element is compiled only for the errors it
                    // can raise, and a bare iteration variable raises none.
                    _ if is_iter(e) => {}
                    _ => {
                        compile_t(ctx, e)?;
                    }
                }
            }
            let delta = emit_int_const(ctx, l.elements.len() as i64);
            Ok(emit_int_bin(ctx, OP_ADD, accu, delta))
        }
        // `c ? (@result + [e]) : @result`: one element where `c` holds.
        ops::CONDITIONAL if call.args.len() == 3 && is_accu(&call.args[2]) => {
            let c = compile_t(ctx, &call.args[0])?;
            if c.bank != ValType::Bool {
                return Err(LowerError::unsupported("filter predicate must be bool"));
            }
            // Collecting under a predicate: the store runs for every element,
            // and the cursor advances only where the predicate holds — so a
            // rejected element writes to the slot the next accepted one will
            // overwrite. That is what keeps the body straight-line, with no
            // branch around the store.
            let zero = emit_int_const(ctx, 0);
            let taken = compile_len_step(ctx, &call.args[1], comp, zero, mode)?;
            let none = emit_int_const(ctx, 0);
            let delta = ctx.fresh(ValType::Int);
            ctx.body.extend_from_slice(&[
                OP_SELECT,
                c.idx as i64,
                taken.idx as i64,
                none.idx as i64,
                delta.idx as i64,
            ]);
            if let AccuMode::Collect { cursor } = mode {
                // Undo the unconditional advance the store made, where the
                // predicate rejected.
                let back = ctx.fresh(ValType::Int);
                ctx.body.extend_from_slice(&[
                    OP_SELECT,
                    c.idx as i64,
                    zero.idx as i64,
                    taken.idx as i64,
                    back.idx as i64,
                ]);
                ctx.body.extend_from_slice(&[
                    OP_SUB,
                    cursor as i64,
                    back.idx as i64,
                    cursor as i64,
                ]);
            }
            Ok(emit_int_bin(ctx, OP_ADD, accu, delta))
        }
        _ => Err(LowerError::unsupported("size() of an opaque comprehension")),
    }
}

/// Write one appended element to the ragged output and advance the cursor.
///
/// A record element (`filter`'s bare iteration variable over a record list)
/// writes one buffer per declared field; a scalar element writes one.
fn emit_element_store(
    ctx: &mut LowerCtxF,
    e: &IdedExpr,
    comp: &ComprehensionExpr,
    cursor: usize,
) -> Result<(), LowerError> {
    let out = ctx
        .list_output
        .clone()
        .expect("collect mode allocates the output description");
    let stride = emit_int_const(ctx, 8);
    let cur = TReg {
        bank: ValType::Int,
        idx: cursor,
    };
    let ea = emit_int_bin(ctx, OP_MUL, cur, stride);
    let is_iter = matches!(&e.expr, Expr::Ident(n) if *n == comp.iter_var);
    for (k, (field, ty)) in out.fields.iter().enumerate() {
        // `filter` appends the element itself, so each output field is that
        // element's own field. `map` appends a computed value, which is the
        // single unnamed field.
        let v = if is_iter {
            ctx.iter_var_slot(&comp.iter_var, field.as_deref())?
                .ok_or_else(|| LowerError::unsupported("collected element is not an element"))?
        } else {
            compile_t(ctx, e)?
        };
        if v.bank != *ty {
            return Err(LowerError::unsupported("collected element bank"));
        }
        let op = if *ty == ValType::Float {
            OP_COL_STORE_F
        } else {
            OP_COL_STORE
        };
        ctx.body
            .extend_from_slice(&[op, out.base_regs[k] as i64, ea.idx as i64, v.idx as i64]);
    }
    let one = emit_int_const(ctx, 1);
    ctx.body
        .extend_from_slice(&[OP_ADD, cursor as i64, one.idx as i64, cursor as i64]);
    Ok(())
}

/// `x in list` over a DECLARED list column: an inner loop over the row's
/// elements, OR-ing `elem == x` into a bool accumulator.
///
/// CEL defines membership over a list as exactly `list.exists(e, e == x)`, so
/// this is the same loop `compile_list_comprehension_t` builds for `exists` —
/// written out here because `@in` arrives as a call rather than as a
/// comprehension and has no iteration variable to bind.
///
/// The loop runs to completion rather than leaving on the first match. A trace
/// wants one straight-line body with one back-edge; an early exit would be a
/// second guard on a data-dependent condition, and the elements are already in
/// cache.
fn lower_runtime_in(
    ctx: &mut LowerCtxF,
    needle: &IdedExpr,
    list: &str,
) -> Result<TReg, LowerError> {
    // Same one-level rule the comprehension has, for the same reason: a list of
    // lists would need per-element offsets.
    if !ctx.list_loop.is_empty() {
        return Err(LowerError::unsupported("nested runtime-list membership"));
    }
    let elem_path = elem_slot_path(list, None);
    let ty =
        ctx.schema.get(&elem_path).copied().ok_or_else(|| {
            LowerError::unsupported(format!("undeclared element path `{elem_path}`"))
        })?;
    // The needle is loop-invariant, so it is compiled before the loop opens.
    let x = compile_t(ctx, needle)?;
    if x.bank != ty {
        // A list whose elements cannot equal the needle's bank is `false` in the
        // tree-walker rather than an error; declining lets the walker own it.
        return Err(LowerError::unsupported("@in element bank"));
    }

    let len = ctx.slot_typed(size_slot_path(list), ValType::Int);
    let off = ctx.slot_typed(offset_slot_path(list), ValType::Int);
    let one = emit_int_const(ctx, 1);
    let stride = emit_int_const(ctx, 8);

    // The accumulator is loop-carried, so it lives in a fixed register.
    let found = ctx.fresh(ValType::Bool);
    ctx.body
        .extend_from_slice(&[OP_LOAD_CONST, 0, found.idx as i64]);
    let j = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_LOAD_CONST, 0, j.idx as i64]);
    // Zero-trip guard: `x in []` is false, and the back-edge is a do-while.
    let zero_trip = ctx.emit_jump_if_above(one, len);

    let inner = ctx.body.len();
    let idx = emit_int_bin(ctx, OP_ADD, off, j);
    let ea = emit_int_bin(ctx, OP_MUL, idx, stride);
    let v = ctx.elem_slot(elem_path, ea.idx)?;
    ctx.elem_map.clear();
    let eq_op = if ty == ValType::Float { OP_FEQ } else { OP_EQ };
    let hit = ctx.fresh(ValType::Bool);
    ctx.body
        .extend_from_slice(&[eq_op, v.idx as i64, x.idx as i64, hit.idx as i64]);
    ctx.body
        .extend_from_slice(&[OP_OR, found.idx as i64, hit.idx as i64, found.idx as i64]);
    ctx.body
        .extend_from_slice(&[OP_ADD, j.idx as i64, one.idx as i64, j.idx as i64]);
    ctx.emit_back_edge(len, j, inner);
    ctx.patch_jump(zero_trip);
    Ok(found)
}

/// `list[k]` / `list[k].field` for a green index `k`: read the row's list
/// element straight out of the flattened element column.
///
/// The address is the comprehension's, with the induction variable pinned to
/// `k`: `(offset(list) + k) * 8`. What the loop gets from its trip count this
/// has to check for itself — `items[1]` on a one-element row is an
/// out-of-range error in the tree-walker, so the row must refuse rather than
/// answer. `k >= len` is OR-ed into the trap flag, and the SAME condition
/// jumps over the load: reading past a row's span would run off the element
/// buffer entirely on the last row, which no trap flag can undo.
fn lower_const_index(
    ctx: &mut LowerCtxF,
    list: &str,
    field: Option<&str>,
    k: i64,
) -> Result<TReg, LowerError> {
    let elem_path = elem_slot_path(list, field);
    let ty =
        ctx.schema.get(&elem_path).copied().ok_or_else(|| {
            LowerError::unsupported(format!("undeclared element path `{elem_path}`"))
        })?;
    // A negative index is an error on every row, whatever the data holds.
    if k < 0 {
        return Err(LowerError::unsupported("negative constant index"));
    }
    let len = ctx.slot_typed(size_slot_path(list), ValType::Int);
    let off = ctx.slot_typed(offset_slot_path(list), ValType::Int);
    let kr = emit_int_const(ctx, k);

    // The result register is written on the in-range path only, so give it a
    // defined value first: the row traps either way, but a register the loop
    // reads must not depend on what a previous row left behind.
    let out = ctx.fresh(ty);
    match ty {
        ValType::Float => {
            let zero = ctx.fresh(ValType::Float);
            ctx.prelude
                .extend_from_slice(&[OP_LOAD_CONST_F, 0, zero.idx as i64]);
            emit_mov(ctx, zero, out);
        }
        _ => ctx
            .body
            .extend_from_slice(&[OP_LOAD_CONST, 0, out.idx as i64]),
    }

    let trap = TReg {
        bank: ValType::Int,
        idx: OVF_FLAG_REG,
    };
    let oob = emit_bin(ctx, OP_GE, kr, len, ValType::Bool);
    ctx.body
        .extend_from_slice(&[OP_OR, trap.idx as i64, oob.idx as i64, trap.idx as i64]);

    // `k + 1 > len` is `k >= len` — the same test, in the form the machine's
    // one forward jump takes.
    let kp1 = emit_int_const(ctx, k + 1);
    let skip = ctx.emit_jump_if_above(kp1, len);
    let idx = emit_int_bin(ctx, OP_ADD, off, kr);
    let stride = emit_int_const(ctx, 8);
    let ea = emit_int_bin(ctx, OP_MUL, idx, stride);
    let v = ctx.elem_slot(elem_path, ea.idx)?;
    emit_mov(ctx, v, out);
    ctx.patch_jump(skip);
    Ok(out)
}

/// Emit a bank-matched register move `dst = src`.
fn emit_mov(ctx: &mut LowerCtxF, src: TReg, dst: TReg) {
    let op = match dst.bank {
        ValType::Float => OP_FMOV,
        _ => OP_MOV,
    };
    ctx.body
        .extend_from_slice(&[op, src.idx as i64, dst.idx as i64]);
}

/// RED-length comprehension over a runtime list: a real inner loop, not an
/// unroll. The element count is a per-row column value, so there is no green
/// trip count to unroll against — and unrolling is not what upstream does
/// either. PyPy only unrolls a loop whose size is `jit.isconstant`
/// (`rlib/jit.py`'s `loop_unrolling_heuristic`); a data-dependent length is
/// simply *traced as a loop*. The inner back-edge is its own `can_enter_jit`
/// point, so the pc-green mainloop gives the element loop its own trace
/// identity, separate from the row loop's.
///
/// The list is stored the columnar (Arrow) way: one flattened element column
/// per field read, laid end to end across rows, plus two derived per-row
/// columns — `size(list)` (element count) and `offset(list)` (start index).
/// Locating a row's elements is then arithmetic, not a pointer chase.
///
/// Emitted shape, with `L` the list and `j` the element index:
///
/// ```text
///     accu = <accu_init>; j = 0
///     if 1 > size(L) goto after          ; zero-trip guard ([].all(..) is true)
///   inner:
///     ea = (offset(L) + j) * 8
///     <element loads at ea, on first reference>
///     accu = <loop_step>; j = j + 1
///     if size(L) > j goto inner          ; back-edge -> can_enter_jit
///   after:
///     <result>
/// ```
///
/// As in the literal unroll, `loop_cond`'s short-circuit is dropped: every
/// element is evaluated. Where the walker would stop early and the eager fold
/// traps instead, the batch answers `None` and the walker owns the row.
fn compile_list_comprehension_t(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    list: &str,
) -> Result<TReg, LowerError> {
    compile_list_comprehension_mode(ctx, comp, list, AccuMode::Value)
}

fn compile_list_comprehension_mode(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    list: &str,
    mode: AccuMode,
) -> Result<TReg, LowerError> {
    let len = ctx.slot_typed(size_slot_path(list), ValType::Int);
    let off = ctx.slot_typed(offset_slot_path(list), ValType::Int);
    let one = emit_int_const(ctx, 1);
    let stride = emit_int_const(ctx, 8);

    let prev_iter = ctx.locals.remove(&comp.iter_var);
    let prev_accu = ctx.locals.remove(&comp.accu_var);

    // The accumulator is loop-carried, so it lives in a FIXED register the step
    // writes back to — `loop_step` lands in a different register each time it
    // is compiled, and here it is compiled once and executed many times.
    let init = match mode {
        AccuMode::Value => compile_t(ctx, &comp.accu_init)?,
        // The list the step would have built starts empty, so its length starts
        // at zero. `accu_init` is not compiled at all: it is the `[]` the
        // machine has no value for.
        AccuMode::Length | AccuMode::Collect { .. } => emit_int_const(ctx, 0),
    };
    let accu = ctx.fresh(init.bank);
    emit_mov(ctx, init, accu);
    let j = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_LOAD_CONST, 0, j.idx as i64]);
    // Zero-trip guard: an empty list must yield `accu_init`, and the back-edge
    // below is a do-while.
    let zero_trip = ctx.emit_jump_if_above(one, len);

    let inner = ctx.body.len();
    let idx = emit_int_bin(ctx, OP_ADD, off, j);
    let ea = emit_int_bin(ctx, OP_MUL, idx, stride);
    ctx.list_loop.push(ListLoop {
        iter_var: comp.iter_var.clone(),
        list: list.to_string(),
        ea_reg: ea.idx,
    });
    ctx.locals.insert(comp.accu_var.clone(), accu);
    let step = match mode {
        AccuMode::Value => compile_t(ctx, &comp.loop_step),
        AccuMode::Length | AccuMode::Collect { .. } => {
            compile_len_step(ctx, &comp.loop_step, comp, accu, mode)
        }
    };
    ctx.list_loop.pop();
    // Drop only THIS loop's element registers. Each comprehension gets its own
    // `ea` register, so keying the retirement on it leaves an enclosing loop's
    // elements — still live below — exactly where they were.
    ctx.elem_map.retain(|(_, reg), _| *reg != ea.idx);
    let step = step?;
    if step.bank != accu.bank {
        return Err(LowerError::unsupported(
            "comprehension accumulator changes bank",
        ));
    }
    emit_mov(ctx, step, accu);
    ctx.body
        .extend_from_slice(&[OP_ADD, j.idx as i64, one.idx as i64, j.idx as i64]);
    ctx.emit_back_edge(len, j, inner);
    ctx.patch_jump(zero_trip);

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
