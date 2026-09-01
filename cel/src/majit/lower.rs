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
use crate::common::types::{type_const_id, TypeValue};
use crate::{Context, Value};
use std::collections::{BTreeSet, HashMap};

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
        Expr::Literal(LiteralValue::Int(i)) => Some(*i),
        _ => None,
    }
}

fn as_bool_literal(e: &IdedExpr) -> Option<bool> {
    match &e.expr {
        Expr::Literal(LiteralValue::Boolean(b)) => Some(*b),
        _ => None,
    }
}

/// Whether `e` is the operand that absorbs a `&&` / `||` whole.
///
/// A bare `true` / `false` is the usual spelling, but absorption is a property
/// of the VALUE: `1 < 2 || x` absorbs exactly as `true || x` does. The
/// resolution is [`fold_constant`]'s — `Context::default()`, no host functions
/// — so the two places agree about which expressions have a constant value, and
/// an operand that raises rather than resolves absorbs nothing.
fn absorbs(e: &IdedExpr, absorbing: bool) -> bool {
    if let Some(b) = as_bool_literal(e) {
        return b == absorbing;
    }
    if !is_constant(e, &mut Vec::new()) {
        return false;
    }
    matches!(
        Value::resolve(e, &Context::default()),
        Ok(Value::Bool(b)) if b == absorbing
    )
}

fn as_string_literal(e: &IdedExpr) -> Option<&str> {
    match &e.expr {
        Expr::Literal(LiteralValue::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Whether `name` is spellable as a CEL identifier, so that `base[name]` and
/// `base.name` denote the same thing. A key that is not — `"a.b"`, `"1"`,
/// `""` — would build a path some other expression could also spell, and the
/// schema would answer for that other one.
fn is_cel_ident(name: &str) -> bool {
    let mut cs = name.chars();
    cs.next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// A CEL `type` used as a value, carried in the int register file as the
    /// index [`type_const_id`] gives its name.
    ///
    /// Types are values in CEL, so `type(x)` has to return one, and the checker
    /// FOLDS such a call to a `Value::Opaque` holding a
    /// [`TypeValue`](crate::common::types::TypeValue) -- a compile-time
    /// constant. So this bank never carries a column and never computes: the
    /// index is minted while lowering and the only operations are `==` and
    /// `!=`, which are index comparisons because the table names each type once.
    ///
    /// Unlike [`ValType::Str`], whose rank is over the batch's own distinct
    /// strings, this index is a position in a fixed table and so denotes the
    /// same type in every program. It has to: the batch decodes a stored word
    /// back to a value with `type_const_value`, outside any lowering. A type
    /// the table does not name has no index, and the lowering declines it.
    Type,
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

/// Words one per-row column load costs in the row loop's prologue: the opcode
/// and its three operands, which [`LoweredF::batch_sum_shape`] emits per
/// [`SlotKind::Row`] slot.
const ROW_SLOT_WORDS: usize = 4;

/// Words the row loop spends on a row besides its body and its column loads:
/// the accumulate or the output store, the induction step, the element-address
/// step, and the back edge — four four-word instructions.
///
/// An UPPER bound, and it has to be: one [`LoweredF`] serves every reduction
/// and both loop shapes, while which of the two induction variables a shape
/// actually steps is decided per shape in [`LoweredF::batch_sum_shape`]. A
/// shape that steps only one spends four words fewer than this says. Making
/// the estimate exact would mean making it depend on the reduction, which is
/// not known here — and it feeds [`LoweredF::body_words_for`], hence the route
/// the batch takes, so it is a contract as much as a count.
const ROW_LOOP_WORDS: usize = 16;

/// Words an element loop's preamble reserves for the one op whose identity is
/// not known until its body has been compiled — the element index's
/// initialiser, or the byte limit its close needs instead. Both spellings are
/// one four-word instruction, which is what lets a single reservation serve
/// either; see [`compile_list_comprehension_mode`].
const PREAMBLE_SLOT_WORDS: usize = 4;

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
    /// The user functions the body calls in scalar form, held so the entry
    /// words the prelude loads for them stay valid for as long as this
    /// lowering does. Nothing reads the list; owning it is its job.
    pub host_fns: Vec<std::sync::Arc<crate::magic::ScalarFn>>,
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
    /// Set when the body ORDERS two strings (`<`, `<=`, `>`, `>=`).
    ///
    /// A string is its rank among the batch's distinct strings, so an ordering
    /// is only the signed int compare because the ranks are assigned in
    /// lexicographic order — which costs a sort over the distinct set at every
    /// bind. Equality does not read that order at all: any injective
    /// assignment answers `==` and `!=` identically.
    ///
    /// Recorded where the ordering is EMITTED, so it cannot disagree with the
    /// body it describes. It is the body's half of the question and not the
    /// answer — ask [`LoweredF::needs_ordered_str_ids`], which also accounts
    /// for a result the caller reads ids back out of.
    pub orders_strings: bool,
    /// Whether some string use needs ids injective over every distinct string.
    /// `false` alone is not a license — ask
    /// [`LoweredF::str_ids_literal_only`], which also accounts for ordering
    /// and a string result.
    pub str_dict_required: bool,
    /// Positions **within [`LoweredF::body`]** of jump target words, which the
    /// lowering writes body-relative because it cannot know where the body
    /// lands. [`LoweredF::batch_sum_shape`] relocates each to an
    /// absolute program address once it does.
    pub jump_fixups: Vec<usize>,
    /// Program words that run once per ITERATED ELEMENT — those enclosed by a
    /// comprehension's or a runtime `in`'s back edge. Zero for a program whose
    /// row body is straight-line, which is the whole of the traceable subset
    /// apart from those two constructs.
    pub elem_words: usize,
    /// Program words that run once per ROW: the rest of the body, plus the row
    /// loop's own — the prologue's column loads and the bookkeeping
    /// [`LoweredF::batch_sum_shape`] wraps every body in.
    ///
    /// The loop's share is not a detail that rounds away. An expression that
    /// only READS a column lowers to an empty body, because the read is the
    /// prologue's; counting the body alone would say such a program costs
    /// nothing per row however tall the batch, which is exactly the case where
    /// the per-row cost is all there is.
    pub row_words: usize,
    /// The batch shapes this program can build, built at most once each and
    /// owned here, indexed by [`shape_slot`].
    ///
    /// This is what makes the words' **address** stable: the `#[jit_interp]`
    /// green key is the program pointer plus pc, so a program rebuilt per bind
    /// gets a new key per bind and loses the compiled loop every time. Owning
    /// them for the life of the `LoweredF` — whose lifetime is the host's,
    /// because the host holds the `BatchProgram` — makes the address stable by
    /// construction rather than by a side table that must never free one entry.
    ///
    /// `OnceLock` rather than a lock or a `RefCell` so `LoweredF` stays `Sync`:
    /// a shape is immutable once built, the key has four values, and racing
    /// initializers agree because only one wins and the loser's words are
    /// dropped before anyone can key on them.
    shapes: [std::sync::OnceLock<BatchShape>; 4],
    /// The one-row shape ([`LoweredF::single_row_shape`]), memoized like the
    /// four above and for the same reason.
    single_row: std::sync::OnceLock<Option<BatchShape>>,
}

/// Index of the memo slot for one `(with_trap, reduce)` pair.
///
/// Total over the product rather than over the pairs seen in practice: three of
/// the four are live today (`batch_sum_program` builds `(false, Sum)`), and a
/// table sized to the observed set would break the moment the fourth is asked
/// for.
const fn shape_slot(with_trap: bool, reduce: BatchReduce) -> usize {
    let r = match reduce {
        BatchReduce::Sum => 0,
        BatchReduce::PerRow => 1,
    };
    (with_trap as usize) * 2 + r
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

#[derive(Debug, Clone)]
pub struct BatchShape {
    /// The program words, owned by the [`LoweredF`] that built them.
    ///
    /// An `Arc` rather than a `Vec` because the **address** is load-bearing, not
    /// just the contents: the `#[jit_interp]` green key is the program pointer
    /// plus pc (`trace_ctx.rs` `green_key_raw`), so a caller that needs an owned
    /// handle takes a refcount bump and keeps the identity the compiled loop was
    /// filed under. `Arc` and not `Rc` because [`LoweredF`] is `Sync` and a
    /// memoized shape is shared, not per-thread.
    pub code: std::sync::Arc<[i64]>,
    /// Float-bank register count the program runs on. The int-bank count is
    /// [`BatchSeed`]'s, since the only thing that needs it is building the bank.
    pub num_float_regs: usize,
    /// Which int registers the caller fills in per batch.
    pub seed: BatchSeed,
    /// [`check_code`](super::bytecode::check_code) over `code` and the two
    /// bank widths the seed fills, taken once here so the clean tier can run
    /// the words without bounds checks on every register read.
    pub check: super::bytecode::CodeCheck,
}

/// A LIST-valued per-row result: the shape of the ragged output the loop writes.
///
/// The same shape a list COLUMN arrives in — a per-row element count plus one
/// flat buffer per field — because it is the same thing, produced rather than
/// consumed. The count rides the ordinary per-row output; the elements go to
/// these buffers, at a cursor that runs across the whole batch.
/// Where a collected list's elements are drawn from — which is what bounds how
/// many the batch can ever write, and so how large the output buffers are.
#[derive(Debug, Clone)]
pub enum ListSource {
    /// A declared list column. Its flattened element count across the whole
    /// batch is an exact bound: `map` writes exactly that many, `filter` fewer.
    Column(String),
    /// A literal list, whose length is green. Every row can append at most that
    /// many, so the batch can append at most `rows` times it.
    Literal(usize),
}

#[derive(Debug, Clone)]
pub struct ListOutput {
    /// What bounds how many elements this can write.
    pub source: ListSource,
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
#[derive(Debug, Clone)]
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
    /// Whether a RESULT of this program is a string: the scalar result itself,
    /// or any field of a collected list. Such a result leaves the machine as an
    /// id, so the caller is handed the id table to read it back with.
    pub fn has_string_result(&self) -> bool {
        self.result_bank == ValType::Str
            || self
                .list_output
                .as_ref()
                .is_some_and(|o| o.fields.iter().any(|(_, t)| *t == ValType::Str))
    }

    /// Whether this program's batch must assign string ids in LEXICOGRAPHIC
    /// order, rather than in whatever order the strings arrive.
    ///
    /// Two things read the order, and skipping the sort for one while the other
    /// needs it is a silent wrong answer, so they are answered here together
    /// rather than at each use:
    ///
    /// * an ordering in the BODY ([`LoweredF::orders_strings`]), which lowered
    ///   to a signed compare of the two ids; and
    /// * a string RESULT, because `RawOutput::Scalar` hands the caller the ids
    ///   themselves alongside the table, and promises that comparing them
    ///   compares the strings.
    pub fn needs_ordered_str_ids(&self) -> bool {
        self.orders_strings || self.has_string_result()
    }

    /// Whether the batch may encode string ids by LITERAL SCAN — each row
    /// compared against the expression's literals, a match taking that
    /// literal's id and every other string one shared sentinel — instead of
    /// ranking every distinct string. Sound exactly when every id the program
    /// consumes is an equality with a literal on one side: distinct
    /// non-literal strings then share the sentinel without ever being asked
    /// to differ.
    pub fn str_ids_literal_only(&self) -> bool {
        !self.str_dict_required && !self.needs_ordered_str_ids()
    }

    /// Whether the row body iterates elements: a comprehension, a chain of
    /// them, or a runtime `in` over a list column. False for the straight-line
    /// shapes — arithmetic, comparison, a ternary, a member read, a constant
    /// index, a string predicate.
    pub fn iterates_elements(&self) -> bool {
        self.elem_words > 0
    }

    /// How many body words a run over `rows` rows carrying `elems` flattened
    /// list elements executes, straight-line words and per-element words added
    /// up. A size, not a time: it is the same count on every tier, which is
    /// what makes it usable to CHOOSE one.
    pub fn body_words_for(&self, rows: usize, elems: usize) -> usize {
        rows.saturating_mul(self.row_words)
            .saturating_add(elems.saturating_mul(self.elem_words))
    }

    /// Set when this program is a PROJECTION: an expression that names one row
    /// column and does nothing to it, so a per-row run stores that column's
    /// element and nothing else.
    ///
    /// Such an expression lowers to an empty body — the read it consists of is
    /// the prologue's column load — and leaves its result in the very register
    /// that load writes. What the row loop then does is copy the input column
    /// into the output buffer, one interpreted row at a time. A tier holding
    /// the column can produce the same buffer by copying it, so this says when
    /// that is the same answer: no invariant to hoist, no body, one row slot,
    /// and the result IS that slot, in its own bank.
    ///
    /// The slot count is pinned at one rather than derived: a second slot's
    /// column would be loaded by the prologue and dropped, and admitting such a
    /// program would leave a consumer copying the FIRST column as the answer to
    /// an expression whose result is a different one.
    pub fn is_row_projection(&self) -> bool {
        self.prelude.is_empty()
            && self.body.is_empty()
            && self.list_output.is_none()
            && self.slots.len() == 1
            && self.slots[0].kind == SlotKind::Row
            && self.slots[0].ty == self.result_bank
            && self.slots[0].reg == self.result_reg
    }

    /// The word every row of this program answers with, when the program is a
    /// constant: no slot, no body, no list output, and a prelude that is one
    /// load of that word into the result register.
    ///
    /// The sibling of [`Self::is_row_projection`] with the column replaced by
    /// a literal. The row loop such a program stands in for stores the same
    /// word for every row, so the batch builder fills the output with it once
    /// and the clean tier answers without running. A folded expression lands
    /// here — `const_reg` puts the one literal the fold left in the prelude and
    /// the body has nothing left to do.
    ///
    /// A string constant is not one: its register is seeded with a batch-ranked
    /// id through `scalar_seeds`, not loaded by the prelude, and the answer
    /// would need the distinct table the loop builds.
    pub fn constant_result(&self) -> Option<i64> {
        let [op, word, reg] = self.prelude.as_slice() else {
            return None;
        };
        let load = if self.result_bank == ValType::Float {
            OP_LOAD_CONST_F
        } else {
            OP_LOAD_CONST
        };
        (*op == load
            && *reg as usize == self.result_reg
            && self.body.is_empty()
            && self.list_output.is_none()
            && self.slots.is_empty()
            && self.scalar_seeds.is_empty())
        .then_some(*word)
    }

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
        // Cloned rather than borrowed so this door keeps its by-value signature.
        // The clone shares the words by refcount, so the ADDRESS the green key
        // is taken over is the same one the memo holds.
        (shape.clone(), regs)
    }

    /// Build the batch program's words and the layout of the registers its
    /// caller seeds.
    ///
    /// `with_trap` emits the epilogue's overflow-flag store (see
    /// [`OP_TRAP_STORE`]). Whether that store is there at all is shape; the
    /// address it writes to is data and rides in a seeded register, so the
    /// evaluator path passes `true` here and the trap word's address to
    /// [`BatchSeed::regs`].
    pub fn batch_sum_shape(&self, with_trap: bool) -> &BatchShape {
        self.batch_shape(with_trap, BatchReduce::Sum)
    }

    /// [`LoweredF::batch_sum_shape`] for a chosen reduction.
    ///
    /// [`BatchReduce::PerRow`] has no `sum_reducible` precondition: a store
    /// takes a result of any bank, which is the whole reason the reduction is a
    /// choice.
    ///
    /// Built at most once per `(with_trap, reduce)` and owned by `self`
    /// thereafter, so every batch of this program runs words at the **same
    /// address** and the compiled loop the green key names survives from batch
    /// to batch. See [`LoweredF::shapes`].
    pub fn batch_shape(&self, with_trap: bool, reduce: BatchReduce) -> &BatchShape {
        self.shapes[shape_slot(with_trap, reduce)]
            .get_or_init(|| self.build_batch_shape(with_trap, reduce, false))
    }

    /// The per-row program for a batch of exactly ONE row, with the trap
    /// published: the same prelude, column reads, body and output store as
    /// [`LoweredF::batch_shape`]`(true, PerRow)`, without the loop around
    /// them — no induction variables to step, no back-edge to test, no
    /// accumulator to clear. Same seed protocol, same registers by role, so a
    /// caller seeds it exactly as it seeds the loop shape.
    ///
    /// The clean tier runs it when the batch it was bound to has one row; the
    /// tracing tiers keep the loop, which is the shape their compiled code and
    /// entry door are keyed on. `None` for a program the straight-line form
    /// cannot express: a list-valued result, whose elements stream through a
    /// cursor the loop owns, or an element slot, whose reads live inside an
    /// inner loop the body carries.
    pub fn single_row_shape(&self) -> Option<&BatchShape> {
        self.single_row
            .get_or_init(|| {
                let expressible = self.list_output.is_none()
                    && self.slots.iter().all(|s| s.kind == SlotKind::Row);
                expressible.then(|| self.build_batch_shape(true, BatchReduce::PerRow, true))
            })
            .as_ref()
    }

    /// [`LoweredF::batch_shape`]'s builder. Split out so the memo above holds
    /// the only call: a second caller would mint a second address for one
    /// program, which is the defect the memo exists to prevent.
    fn build_batch_shape(
        &self,
        with_trap: bool,
        reduce: BatchReduce,
        single_row: bool,
    ) -> BatchShape {
        debug_assert!(
            !single_row || (with_trap && reduce == BatchReduce::PerRow),
            "the one-row shape is a per-row program with its trap published"
        );
        // The accumulate below would fold a result the sum cannot consume into
        // the int total — adding string RANKS, nanoseconds, or a collected
        // list's element COUNT. Every public door asks `sum_reducible` first;
        // this catches a harness that did not.
        if reduce == BatchReduce::Sum {
            self.sum_reducible()
                .expect("a summing shape on a result the loop's sum cannot consume");
        }
        // The loop's two literals — the row step and the byte stride — are NOT
        // registers: they ride in the word stream as immediates (see
        // `OP_ADD_IMM`), which is what keeps them constants inside the trace.
        let m = self.num_int_regs; // first int machinery register
        let (r_i, r_acc, r_n, r_ea) = (m, m + 1, m + 2, m + 3);
        let r_trap = m + 4;
        // The output base is a machinery register too, present only where the
        // reduction stores through it.
        let r_out = match reduce {
            BatchReduce::Sum => None,
            BatchReduce::PerRow => Some(m + 5),
        };
        let r_base0 = m + 5 + usize::from(r_out.is_some());
        let total_int_regs = r_base0 + self.slots.len();
        // A float result accumulates into a float register above the body's
        // float bank; an int result uses the int `r_acc` and leaves the float
        // bank at the body's count. `f_acc` is unused when the result is int.
        let (f_acc, total_float_regs) = match (reduce, self.result_bank) {
            (BatchReduce::Sum, ValType::Float) => (self.num_float_regs, self.num_float_regs + 1),
            _ => (0, self.num_float_regs),
        };

        // Whether the loop needs the byte-offset induction variable at all.
        let needs_ea = reduce == BatchReduce::PerRow
            || self
                .slots
                .iter()
                .any(|slot| slot.kind == SlotKind::Row && slot.ty != ValType::Bool);

        // Whether the row COUNTER is stepped at all. The back edge closes on
        // either induction variable, so the counter earns its step only where
        // something else reads its value: a `bool` column addresses by it, a
        // `PerRow` run returns it as the count it wrote, and a shape with no
        // byte offset has nothing else to close on. Where none of the three
        // holds the offset carries the loop alone, and the step is dead work a
        // row. micronumpy spells the same rule as a flag its iterator carries —
        // `iterators.py:154-155` steps the index under `if self.track_index:`,
        // and `loop.py:24` clears it on the operand whose position nobody asks
        // for.
        let track_index = single_row
            || !needs_ea
            || reduce == BatchReduce::PerRow
            || self
                .slots
                .iter()
                .any(|slot| slot.kind == SlotKind::Row && slot.ty == ValType::Bool);

        let mut p = Vec::new();
        let load_const = |p: &mut Vec<i64>, imm: i64, dst: usize| {
            p.extend_from_slice(&[OP_LOAD_CONST, imm, dst as i64]);
        };
        if track_index {
            load_const(&mut p, 0, r_i);
        } else {
            // A loop that does not step the counter still has to end, so the
            // counter's register carries the byte limit the close compares
            // against. Reusing it rather than reserving one keeps the bank the
            // same width, which is what the seed and `check_code` are sized on.
            p.extend_from_slice(&[OP_MUL_IMM, r_n as i64, 8, r_i as i64]);
        }
        // The byte offset is a SECOND induction variable, stepped by its own
        // stride, rather than `i * 8` recomputed each row. Both forms cost the
        // same one instruction, but only this one is analyzable: the optimizer
        // strength-reduces `int_mul(i, 8)` to `int_lshift(i, 3)`
        // (`autogenintrules.py:387-393 mul_pow2_const`) and
        // `dependency.py:896-948` builds an `IndexVar` for INT_ADD/SUB/MUL
        // only — never for a shift. An offset advanced by a constant keeps the
        // memory references linear in one base var, which is how micronumpy's
        // iterators walk an array and the reason its loops vectorize.
        //
        // A shape with no eight-byte row access — a constant expression, or one
        // reading only `bool` columns, which address by the row counter — never
        // reads it, so it does not carry it.
        if needs_ea {
            load_const(&mut p, 0, r_ea);
        }
        // Accumulator init, run once before the merge point. `OP_LOAD_CONST_F`'s
        // `f64::from_bits` must stay out of the traced loop body; here it is in
        // the setup (0.0 has zero bits).
        match (reduce, self.result_bank) {
            (BatchReduce::Sum, ValType::Float) => {
                p.extend_from_slice(&[OP_LOAD_CONST_F, 0, f_acc as i64])
            }
            // A `PerRow` loop carries no accumulator, but zeroing `r_acc` costs
            // one setup instruction and leaves the bank in one known state.
            // The one-row form has no loop to keep a state for, and skips it.
            _ if single_row => {}
            _ => load_const(&mut p, 0, r_acc),
        }
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
        // slot_k = *(base_k + ea)   — the red-index columnar read, per bank.
        // Element columns are skipped: their index is the inner loop's, not the
        // row's, so the lowering already emitted their loads inside the body.
        for (k, slot) in self.slots.iter().enumerate() {
            if slot.kind != SlotKind::Row {
                continue;
            }
            // A `bool` column is the caller's own `&[bool]` — one byte per row,
            // so its effective address is the ROW COUNTER itself and the `* 8`
            // above does not apply to it.
            let (op, addr_reg) = match slot.ty {
                ValType::Int
                | ValType::UInt
                | ValType::Str
                | ValType::Timestamp
                | ValType::Duration
                // No `ColumnRef` builds one, so a `Type` slot cannot bind; the
                // arm is here because the bank IS an int-file word and an
                // `unreachable!` would be a claim about the caller's schema.
                | ValType::Type => (OP_COL_LOAD, r_ea),
                ValType::Bool => (OP_COL_LOAD_B, r_i),
                ValType::Float => (OP_COL_LOAD_F, r_ea),
            };
            p.extend_from_slice(&[op, (r_base0 + k) as i64, addr_reg as i64, slot.reg as i64]);
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
        // The one row was read and its result stored at offset zero; there is
        // no next row to step to and no back-edge to take.
        if !single_row {
            if track_index {
                p.extend_from_slice(&[OP_ADD_IMM, r_i as i64, 1, r_i as i64]);
            }
            if needs_ea {
                p.extend_from_slice(&[OP_ADD_IMM, r_ea as i64, 8, r_ea as i64]);
            }
            // Whichever variable the loop stepped is the one it closes on, and
            // the other register holds that variable's limit.
            let (limit, iv) = if track_index { (r_n, r_i) } else { (r_i, r_ea) };
            p.extend_from_slice(&[OP_JUMP_IF_ABOVE, limit as i64, iv as i64, body_pc as i64]);
        }
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
            // The one-row form never stepped `r_i`; the count it wrote is the
            // seeded row count, which is what the loop's `r_i` ends at.
            (BatchReduce::PerRow, _) if single_row => p.extend_from_slice(&[OP_RETURN, r_n as i64]),
            (BatchReduce::PerRow, _) => p.extend_from_slice(&[OP_RETURN, r_i as i64]),
        }
        // Only a per-row run writes elements; a sum never reaches them.
        let list_out_regs: Vec<usize> = match reduce {
            BatchReduce::PerRow => self
                .list_output
                .as_ref()
                .map(|o| o.base_regs.clone())
                .unwrap_or_default(),
            BatchReduce::Sum => Vec::new(),
        };
        let scalar_regs: Vec<usize> = self.scalar_seeds.iter().map(|s| s.reg).collect();

        // Narrow both files now that the program is whole. Every index below is
        // the packer's answer, not the lowering's: the words are rewritten in
        // place, so the seed has to be rewritten with them or it would fill
        // registers the program no longer reads.
        let seeded: Vec<usize> = [r_n, r_trap]
            .into_iter()
            .chain(r_out)
            .chain(base_regs.iter().copied())
            .chain(scalar_regs.iter().copied())
            .chain(list_out_regs.iter().copied())
            .collect();
        let ints = pack_file(&mut p, RegFile::Ints, total_int_regs, &seeded);
        // No float register is seeded: every value the caller supplies — a
        // count, an address, a broadcast scalar — travels in the int file.
        let floats = pack_file(&mut p, RegFile::Floats, total_float_regs, &[]);

        let code: std::sync::Arc<[i64]> = p.into();
        let check =
            super::bytecode::check_code(&code, ints.width, floats.width).unwrap_or_else(|why| {
                panic!("the lowering built a program its interpreter cannot run: {why}")
            });
        BatchShape {
            code,
            num_float_regs: floats.width,
            seed: BatchSeed {
                r_n: ints.of(r_n),
                r_trap: ints.of(r_trap),
                r_out: r_out.map(|r| ints.of(r)),
                list_out_regs: list_out_regs.iter().map(|&r| ints.of(r)).collect(),
                base_regs: base_regs.iter().map(|&r| ints.of(r)).collect(),
                scalar_regs: scalar_regs.iter().map(|&r| ints.of(r)).collect(),
                num_int_regs: ints.width,
            },
            check,
        }
    }
}

/// The two register files [`pack_file`] narrows, one call each.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RegFile {
    Ints,
    Floats,
}

/// What one [`pack_file`] run decided: where each old register went, and how
/// wide the file ended up.
struct Packing {
    /// `old -> new`, indexed by the old register.
    map: Vec<usize>,
    /// Registers the packed program addresses — one past the highest it names.
    width: usize,
}

impl Packing {
    fn of(&self, reg: usize) -> usize {
        self.map[reg]
    }
}

/// One decoded instruction: where it starts, and what the words after it mean.
struct Decoded {
    at: usize,
    ops: &'static [Operand],
}

/// Split a program into instructions. Widths come from [`OPERANDS`], so an
/// opcode this walk does not anticipate is still stepped over correctly.
fn decode(p: &[i64]) -> Vec<Decoded> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < p.len() {
        let ops = OPERANDS[p[at] as usize];
        out.push(Decoded { at, ops });
        at += 1 + ops.len();
    }
    out
}

/// Narrow one register file to the width the program's own liveness allows,
/// rewriting every operand that names it.
///
/// A tree-walking lowering hands each sub-expression a private register, so a
/// program's file is as wide as the number of values it ever names rather than
/// the number it holds AT ONCE — and once a comprehension body runs a chain of
/// short-lived temporaries the two differ by a lot. The width is not merely a
/// memory cost. It is the number of words a compiled trace's entry reloads and
/// the number of live values its register allocator is handed, so a file wider
/// than the program needs turns into spill traffic in the innermost loop, where
/// it is paid per element.
///
/// `seeded` names the registers the CALLER fills in before the program starts —
/// the row count, column bases, broadcast scalars. Those keep a word of their
/// own, as does anything live before the first instruction runs: the seeding
/// writes them all at once, and two sharing a word would leave one of the two
/// values overwritten before a single instruction had run.
fn pack_file(p: &mut [i64], file: RegFile, width: usize, seeded: &[usize]) -> Packing {
    let code = decode(p);
    let n = code.len();
    let index_at: HashMap<usize, usize> = code.iter().enumerate().map(|(k, d)| (d.at, k)).collect();

    // Per-instruction operand roles in this file, and where control can go
    // next. A trap operand counts as both a read and a write: only the trapping
    // path stores through it, so its old value survives every other path.
    let mut reads: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut writes: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, d) in code.iter().enumerate() {
        for (j, role) in d.ops.iter().enumerate() {
            let w = p[d.at + 1 + j] as usize;
            match (file, role) {
                (_, Operand::Target) => succ[k].push(index_at[&w]),
                (RegFile::Ints, Operand::Int) | (RegFile::Floats, Operand::Float) => {
                    reads[k].push(w)
                }
                (RegFile::Ints, Operand::IntOut) | (RegFile::Floats, Operand::FloatOut) => {
                    writes[k].push(w)
                }
                (RegFile::Ints, Operand::IntTrap) => {
                    reads[k].push(w);
                    writes[k].push(w);
                }
                _ => {}
            }
        }
        let returns = p[d.at] == OP_RETURN || p[d.at] == OP_RETURN_F;
        if !returns && k + 1 < n {
            succ[k].push(k + 1);
        }
    }

    // Which registers each instruction may still need. Backwards to a fixpoint,
    // because the program's loops make one pass insufficient.
    let mut live_in = vec![vec![false; width]; n];
    let mut live_out = vec![vec![false; width]; n];
    let mut settled = false;
    while !settled {
        settled = true;
        for k in (0..n).rev() {
            let mut out = vec![false; width];
            for &s in &succ[k] {
                for (r, &live) in live_in[s].iter().enumerate() {
                    out[r] |= live;
                }
            }
            let mut inn = out.clone();
            for &w in &writes[k] {
                inn[w] = false;
            }
            for &r in &reads[k] {
                inn[r] = true;
            }
            if out != live_out[k] || inn != live_in[k] {
                live_out[k] = out;
                live_in[k] = inn;
                settled = false;
            }
        }
    }

    let mut clash = vec![vec![false; width]; width];
    fn mark(clash: &mut [Vec<bool>], a: usize, b: usize) {
        if a != b {
            clash[a][b] = true;
            clash[b][a] = true;
        }
    }
    for k in 0..n {
        // Everything still live once the instruction finishes, plus what it
        // defines: two values that coexist here cannot share a word.
        let mut together: Vec<usize> = (0..width).filter(|&r| live_out[k][r]).collect();
        together.extend(writes[k].iter().copied().filter(|&w| !live_out[k][w]));
        for i in 0..together.len() {
            for j in (i + 1)..together.len() {
                mark(&mut clash, together[i], together[j]);
            }
        }
        // A destination must not land on a source's word either. The read
        // happens first for most of these ops, but not all: a checked add
        // publishes its trap flag and only then recomputes the wrapped sum from
        // its operands, so a flag sharing an operand's word would read back the
        // 1 it just wrote.
        for &w in &writes[k] {
            for &r in &reads[k] {
                mark(&mut clash, w, r);
            }
        }
    }

    // Seeded registers, and anything live before the first instruction, take a
    // word each. The rest are colored greedily against what is already placed.
    let mut color = vec![usize::MAX; width];
    let mut next = 0;
    let entry: Vec<usize> = seeded
        .iter()
        .copied()
        .chain((0..width).filter(|&r| n > 0 && live_in[0][r]))
        .collect();
    for r in entry {
        if color[r] == usize::MAX {
            color[r] = next;
            next += 1;
        }
    }
    let named: Vec<bool> = (0..width)
        .map(|r| (0..n).any(|k| reads[k].contains(&r) || writes[k].contains(&r)))
        .collect();
    for r in 0..width {
        if color[r] != usize::MAX || !named[r] {
            continue;
        }
        let taken: Vec<usize> = (0..width)
            .filter(|&q| clash[r][q] && color[q] != usize::MAX)
            .map(|q| color[q])
            .collect();
        let mut c = 0;
        while taken.contains(&c) {
            c += 1;
        }
        color[r] = c;
    }
    // A register the program never names needs no word of its own; parking them
    // all on the first one keeps the map total without widening the file.
    for c in color.iter_mut() {
        if *c == usize::MAX {
            *c = 0;
        }
    }

    for d in &code {
        for (j, role) in d.ops.iter().enumerate() {
            let names = matches!(
                (file, role),
                (
                    RegFile::Ints,
                    Operand::Int | Operand::IntOut | Operand::IntTrap
                ) | (RegFile::Floats, Operand::Float | Operand::FloatOut)
            );
            if names {
                p[d.at + 1 + j] = color[p[d.at + 1 + j] as usize] as i64;
            }
        }
    }
    let width = color.iter().copied().max().map_or(0, |c| c + 1);
    Packing { map: color, width }
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
    /// Set when the body ORDERS two strings, which is the only thing that reads
    /// the dictionary's order rather than only its injectivity.
    orders_strings: bool,
    /// Set when some string USE needs the dictionary's injectivity over every
    /// distinct string -- an equality with no literal side, a membership whose
    /// needle is not a literal, a predicate table -- rather than only "equal
    /// to this literal or not".
    str_dict_required: bool,
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
    /// Element index registers some emitted op ADDRESSED with, rather than
    /// every one a loop minted. Only [`OP_COL_LOAD_B`] reads a column at the
    /// index -- a `bool` column is the caller's own `&[bool]`, one byte an
    /// element -- so a loop no byte column was read inside does not have to
    /// advance its index at all, and the back edge counts the byte offset that
    /// every other load already needs. Recorded by register rather than by
    /// loop, because a fused chain link shares its base loop's index and would
    /// otherwise set the flag on an entry the close never sees.
    elem_idx_addressed: BTreeSet<usize>,
    /// Body length right after an op that MINTED its destination register and
    /// wrote it as that op's last word. `None` where the last thing emitted was
    /// anything else.
    ///
    /// The length is the guard, and it is why this is safe to consult. Anything
    /// appended since -- by a helper that does not record here, or by one of the
    /// many inline `extend_from_slice` calls -- moves it, and the record is then
    /// refused rather than trusted. So an emitter that never learned about this
    /// field costs a missed rewrite and can never cause a wrong one. Minted is
    /// the other half: a register `fresh` returned and one op wrote is in no
    /// slot, element, local or constant map, so retargeting it cannot rename
    /// something another read still expects to find.
    last_fresh_write: Option<usize>,
    /// Registers holding a loop-invariant constant, keyed by the bank and the
    /// word loaded into it. See [`LowerCtxF::const_reg`].
    const_pool: HashMap<(ValType, i64), TReg>,
    /// Positions within `body` holding a body-relative jump target.
    jump_fixups: Vec<usize>,
    /// Body words enclosed by a back edge, accumulated as each element loop is
    /// closed. See [`LoweredF::elem_words`].
    elem_words: usize,
    /// Set when the top-level expression is collected as a list.
    list_output: Option<ListOutput>,
    schema: &'s Schema,
    /// Where a call to a name no arm above knows is looked up, when the caller
    /// gave one. Only a function with a scalar form ([`crate::magic::ScalarFn`])
    /// is lowered, and only when no stdlib overload is declared under the name,
    /// since the tree-walker tries those first.
    functions: Option<&'s Context<'s>>,
    /// See [`LoweredF::host_fns`].
    host_fns: Vec<std::sync::Arc<crate::magic::ScalarFn>>,
}

/// The runtime-list comprehension being lowered — what an iteration variable
/// resolves against.
struct ListLoop {
    /// Iteration variable name, e.g. `i` in `items.all(i, i.price > 10)`.
    iter_var: String,
    /// Schema path of the list, e.g. `items`.
    list: String,
    /// Int register holding the inner loop's byte offset — the element index
    /// scaled by the word stride.
    ea_reg: usize,
    /// Int register holding the element INDEX — the same address for a column
    /// whose stride is one byte. It advances alongside `ea_reg` rather than
    /// being derived from it, so a byte-column read costs no scaling op of its
    /// own.
    idx_reg: usize,
}

impl LowerCtxF<'_> {
    /// Whether `r` is a register seeded with a string LITERAL's id.
    fn is_str_literal_seed(&self, r: &TReg) -> bool {
        r.bank == ValType::Str
            && self
                .scalar_seeds
                .iter()
                .any(|s| s.reg == r.idx && matches!(s.kind, SeedKind::StrId(_)))
    }

    fn fresh(&mut self, bank: ValType) -> TReg {
        let idx = match bank {
            // `Str` ids share the int register file (an `i64` rank).
            ValType::Int
            | ValType::Bool
            | ValType::UInt
            | ValType::Str
            | ValType::Type
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

    /// The register holding a loop-invariant constant, minted once per
    /// `(bank, word)` and shared by every later reference to that value.
    ///
    /// The load runs in the prelude, before the row loop, and nothing writes
    /// the register afterwards — the only ops that name it name it as an
    /// operand — so one register can serve every occurrence. That is what keeps
    /// a program's register file proportional to the DISTINCT constants it
    /// mentions rather than to how often it mentions them, and register
    /// pressure is what decides whether the backend spills inside a
    /// comprehension's inner loop: a constant is live across the whole body, so
    /// a duplicate costs a live range spanning every instruction, not just a
    /// word of bank.
    ///
    /// Keyed on the bank as well as the word because the bank is what operands
    /// are type-checked against: an `int` 1 and a `bool` true are the same
    /// machine word and not the same operand. A register the lowering will
    /// WRITE — a cursor, a loop counter, an accumulator seeded with 0 — is not
    /// a constant and must keep taking a private register from [`Self::fresh`].
    fn const_reg(&mut self, bank: ValType, word: i64) -> TReg {
        if let Some(&r) = self.const_pool.get(&(bank, word)) {
            return r;
        }
        let r = self.fresh(bank);
        let op = match bank {
            ValType::Float => OP_LOAD_CONST_F,
            _ => OP_LOAD_CONST,
        };
        self.prelude.extend_from_slice(&[op, word, r.idx as i64]);
        self.const_pool.insert((bank, word), r);
        r
    }

    /// The word `r` holds on every row, when `r` is one the prelude loads: the
    /// reverse of [`Self::const_reg`]. A register the body writes never comes
    /// from the pool, so a hit here is a value the lowering can compute once,
    /// now, instead of the row loop computing it on every row — which is what
    /// an unrolled literal comprehension leaves behind: its iteration variable
    /// is a pool register, so `x * 2` over `[1, 2, 3]` is three constant
    /// products.
    fn const_of(&self, r: TReg) -> Option<i64> {
        self.const_pool
            .iter()
            .find_map(|(&(bank, word), p)| (bank == r.bank && p.idx == r.idx).then_some(word))
    }

    /// Note that the op just appended minted its destination and wrote it last.
    /// See [`LowerCtxF::last_fresh_write`] for why only such an op may say so.
    fn note_fresh_write(&mut self) {
        self.last_fresh_write = Some(self.body.len());
    }

    /// Make the op that just wrote `src` write `dst` instead, and answer whether
    /// it did. `false` means the caller still owes the move it was going to skip.
    ///
    /// Declines unless that op is still the last thing in the body, so a caller
    /// may ask at any point and get a wrong answer at none. The banks must agree
    /// because the destination decides which file the op writes.
    fn retarget_last_write(&mut self, src: TReg, dst: TReg) -> bool {
        if src.bank != dst.bank {
            return false;
        }
        let Some(len) = self.last_fresh_write else {
            return false;
        };
        if len != self.body.len() || self.body[len - 1] != src.idx as i64 {
            return false;
        }
        self.body[len - 1] = dst.idx as i64;
        self.last_fresh_write = None;
        true
    }

    /// Resolve a row slot whose type the schema must declare. An UNDECLARED path
    /// is a decline, not a guess: the bank decides which ops the path is legal
    /// under (`!x` needs `bool`, `x + 1` needs a numeric bank), so defaulting it
    /// would silently pick a meaning the caller never stated — and the caller's
    /// column, built from the same declaration, would then be read in the wrong
    /// bank.
    fn slot(&mut self, path: String) -> Result<TReg, LowerError> {
        let Some(ty) = self.schema.get(&path).copied() else {
            // CEL binds the type names as identifiers, so `int` is a value and
            // not an undeclared column. The schema is consulted FIRST, which is
            // the tree-walker's own order (`context.rs:167` reaches
            // `type_ident` only after the variables), so a column may still be
            // named `int`.
            let id = type_const_id(&path)
                .ok_or_else(|| LowerError::unsupported(format!("undeclared path `{path}`")))?;
            return Ok(self.const_reg(ValType::Type, id));
        };
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
        let ty =
            self.schema.get(&path).copied().ok_or_else(|| {
                LowerError::unsupported(format!("undeclared element path `{path}`"))
            })?;
        self.elem_slot_typed(path, ty, ea_reg)
    }

    /// The element INDEX register belonging to the inner loop whose byte offset
    /// is `ea_reg` — the address a one-byte element column is read at.
    ///
    /// Looked up rather than threaded through every chain function: the loop
    /// that owns `ea_reg` is on the stack whenever one of its elements is being
    /// read, and a later chain link that aliases the base loop shares both
    /// registers.
    ///
    /// `None` where no loop owns `ea_reg`. It is not interchangeable with
    /// `ea_reg`: the two differ by the stride, so a byte column read at the word
    /// address lands eight times too far into the caller's slice, which is an
    /// unchecked read past its end rather than an error.
    fn elem_idx_reg(&self, ea_reg: usize) -> Option<usize> {
        self.list_loop
            .iter()
            .rev()
            .find(|l| l.ea_reg == ea_reg)
            .map(|l| l.idx_reg)
    }

    /// Park an element column's base for a load the CALLER emits — the fused
    /// constant-index read — and return the base register. `value` is the
    /// register that load writes, recorded on the slot like
    /// [`LowerCtxF::elem_slot_typed`] records its own.
    fn elem_base(&mut self, path: String, ty: ValType, value: TReg) -> usize {
        let base_reg = self.fresh(ValType::Int).idx;
        self.slots.push(SlotInfoF {
            path,
            ty,
            reg: value.idx,
            kind: SlotKind::Element { base_reg },
        });
        base_reg
    }

    fn elem_slot_typed(
        &mut self,
        path: String,
        ty: ValType,
        ea_reg: usize,
    ) -> Result<TReg, LowerError> {
        if let Some(&r) = self.elem_map.get(&(path.clone(), ea_reg)) {
            return Ok(r);
        }
        let r = self.fresh(ty);
        let base_reg = self.fresh(ValType::Int).idx;
        // A `bool` element column is the caller's own `&[bool]`, one byte per
        // element, so it is read at the element INDEX and not at `index * 8`.
        // Only a loop that owns `ea_reg` carries that index, so a byte column
        // reached from outside one has no address here and declines to the
        // tree-walker; answering from `ea_reg` would read past the slice.
        let (op, addr_reg) = match ty {
            ValType::Float => (OP_COL_LOAD_F, ea_reg),
            ValType::Bool => {
                let idx = self
                    .elem_idx_reg(ea_reg)
                    .ok_or_else(|| LowerError::unsupported("bool element outside its list loop"))?;
                // This load is the reason the index exists; say so, so the
                // loop's close knows it has to advance it.
                self.elem_idx_addressed.insert(idx);
                (OP_COL_LOAD_B, idx)
            }
            _ => (OP_COL_LOAD, ea_reg),
        };
        self.body
            .extend_from_slice(&[op, base_reg as i64, addr_reg as i64, r.idx as i64]);
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
        // Everything from the loop's entry to the back edge inclusive runs once
        // per element rather than once per row. Loops that nest count their
        // inner span on both levels, which is the direction that overstates the
        // work rather than the one that hides it.
        self.elem_words += self.body.len() - tgt;
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
    lower_typed_in(expr, schema, None)
}

/// [`lower_typed`], with `functions` supplying the user functions a call may
/// resolve to. A call to a name that is not a lowered builtin looks the name up
/// there; with `None`, such a call is a decline.
pub fn lower_typed_in(
    expr: &IdedExpr,
    schema: &Schema,
    functions: Option<&Context<'_>>,
) -> Result<LoweredF, LowerError> {
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
        orders_strings: false,
        str_dict_required: false,
        concats: Vec::new(),
        elem_map: HashMap::new(),
        list_loop: Vec::new(),
        elem_idx_addressed: BTreeSet::new(),
        last_fresh_write: None,
        const_pool: HashMap::new(),
        jump_fixups: Vec::new(),
        elem_words: 0,
        list_output: None,
        schema,
        functions,
        host_fns: Vec::new(),
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
    let elem_words = ctx.elem_words.min(ctx.body.len());
    let row_slots = ctx.slots.iter().filter(|s| s.kind == SlotKind::Row).count();
    let row_words = ctx.body.len() - elem_words + row_slots * ROW_SLOT_WORDS + ROW_LOOP_WORDS;
    Ok(LoweredF {
        elem_words,
        row_words,
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
        host_fns: ctx.host_fns,
        temporal_bound,
        orders_strings: ctx.orders_strings,
        str_dict_required: ctx.str_dict_required,
        jump_fixups: ctx.jump_fixups,
        shapes: std::array::from_fn(|_| std::sync::OnceLock::new()),
        single_row: std::sync::OnceLock::new(),
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
                    if let Ok(base) = resolve_path(&inner.args[0]) {
                        if declares_list(ctx.schema, &base) {
                            return match as_int_literal(&inner.args[1]) {
                                Some(k) => lower_const_index(ctx, &base, Some(&sel.field), k),
                                None => {
                                    lower_var_index(ctx, &base, Some(&sel.field), &inner.args[1])
                                }
                            };
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
        LiteralValue::Int(i) => Ok(ctx.const_reg(ValType::Int, *i)),
        LiteralValue::Boolean(b) => Ok(ctx.const_reg(ValType::Bool, *b as i64)),
        // The f64 travels as its raw i64 bits; the VM reloads with
        // `f64::from_bits`, so the constant is bit-exact. Two literals share a
        // register when their BIT PATTERNS agree, which is the identity the
        // reload restores — `0.0` and `-0.0` are two constants, and two `NaN`s
        // with different payloads are two constants.
        LiteralValue::Double(f) => Ok(ctx.const_reg(ValType::Float, f.to_bits() as i64)),
        // The u64 travels as its raw i64 bit pattern in the int register file.
        LiteralValue::UInt(u) => Ok(ctx.const_reg(ValType::UInt, *u as i64)),
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
                kind: SeedKind::StrId(s.as_str().to_owned()),
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
    ctx.const_reg(ValType::Float, v.to_bits() as i64)
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
    ctx.const_reg(ValType::Int, v)
}

/// Emit a three-address int-bank op `dst = a <op> b` into the body.
fn emit_int_bin(ctx: &mut LowerCtxF, op: i64, a: TReg, b: TReg) -> TReg {
    // Two pool constants fold to a third; these are the machine's own wrapping
    // ops, so the fold wraps the same way the row loop would have.
    if let (Some(x), Some(y)) = (ctx.const_of(a), ctx.const_of(b)) {
        let folded = match op {
            OP_ADD => Some(x.wrapping_add(y)),
            OP_SUB => Some(x.wrapping_sub(y)),
            OP_MUL => Some(x.wrapping_mul(y)),
            _ => None,
        };
        if let Some(v) = folded {
            return emit_int_const(ctx, v);
        }
    }
    let d = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[op, a.idx as i64, b.idx as i64, d.idx as i64]);
    ctx.note_fresh_write();
    d
}

/// `dst = 1 - c`, for a register holding a predicate.
///
/// `OP_NOT` carries its `1` in the instruction stream, where a trace reads it as
/// a constant; a pooled `1` lives in a register, which is red. A predicate the
/// lowering already folded takes its complement with it instead, and costs no
/// instruction at all.
fn emit_complement(ctx: &mut LowerCtxF, c: TReg) -> TReg {
    if ctx.const_of(c).is_some() {
        let one = emit_int_const(ctx, 1);
        return emit_int_bin(ctx, OP_SUB, one, c);
    }
    let d = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_NOT, c.idx as i64, d.idx as i64]);
    ctx.note_fresh_write();
    d
}

/// `dst = a <op> k` for a green constant `k`.
///
/// `OP_MUL`, `OP_ADD`, `OP_DIV` and `OP_MOD` have immediate forms, and those
/// are the ones to reach for: the immediate rides in the word stream, which is
/// green, so the traced op is `int_mul(reg, ConstInt(k))`. Hoisting the
/// constant into a prelude register instead puts the store outside the merge
/// point, and the in-loop read is then an opaque input argument —
/// `dependency.py:896-948` builds no `IndexVar` for it. Division is where that
/// costs the most: only a CONSTANT divisor is expanded into
/// multiply-and-shift, so through a register every divide stayed a residual
/// `int.udiv` call. Any other op still hoists.
fn emit_int_bin_k(ctx: &mut LowerCtxF, op: i64, a: TReg, k: i64) -> TReg {
    let imm_op = match op {
        OP_MUL => Some(OP_MUL_IMM),
        OP_ADD => Some(OP_ADD_IMM),
        OP_DIV => Some(OP_DIV_K),
        OP_MOD => Some(OP_MOD_K),
        _ => None,
    };
    match imm_op {
        Some(imm_op) => {
            let d = ctx.fresh(ValType::Int);
            ctx.body
                .extend_from_slice(&[imm_op, a.idx as i64, k, d.idx as i64]);
            ctx.note_fresh_write();
            d
        }
        None => {
            let kr = emit_int_const(ctx, k);
            emit_int_bin(ctx, op, a, kr)
        }
    }
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
    // `ts == q * NANOS_PER_DAY + r` holds exactly for a truncating quotient, so
    // the remainder is a multiply and a subtract rather than a second division
    // -- and a second magnitude round trip, which is what a signed `%` by a
    // constant costs. `q * NANOS_PER_DAY` cannot overflow: its magnitude is at
    // most `|ts|`.
    let qd = emit_int_bin_k(ctx, OP_MUL, q, NANOS_PER_DAY);
    let r = emit_int_bin(ctx, OP_SUB, ts, qd);
    let neg = emit_int_bin_k(ctx, OP_LT, r, 0);
    let days = emit_int_bin(ctx, OP_SUB, q, neg);
    let back = emit_int_bin_k(ctx, OP_MUL, neg, NANOS_PER_DAY);
    let nanos_of_day = emit_int_bin(ctx, OP_ADD, r, back);
    (days, nanos_of_day)
}

/// [`emit_int_bin_k`] for a division whose DIVIDEND the lowering knows cannot
/// be negative -- see [`OP_DIVN_K`] for what that buys and what it owes.
fn emit_nonneg_div_k(ctx: &mut LowerCtxF, a: TReg, k: i64) -> TReg {
    let d = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_DIVN_K, a.idx as i64, k, d.idx as i64]);
    d
}

/// The remainder twin of [`emit_nonneg_div_k`].
fn emit_nonneg_mod_k(ctx: &mut LowerCtxF, a: TReg, k: i64) -> TReg {
    let d = ctx.fresh(ValType::Int);
    ctx.body
        .extend_from_slice(&[OP_MODN_K, a.idx as i64, k, d.idx as i64]);
    d
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 to
/// `(year, month 0-11, day 0-30)`.
///
/// The month and day come back 0-BASED because that is the base most of the
/// readers want: `getMonth` and `getDayOfMonth` are 0-based and would each
/// subtract back the one this had just added, while `getDate` is the single
/// 1-based reader and adds it itself. The pair does not cancel on its own —
/// `OP_SUB` has no immediate form, so the constant it subtracts is hoisted into
/// a prelude register and `int_sub(int_add(x, C), C)` never presents the
/// optimizer with two constants to fold. It survives into the compiled trace,
/// which is why the base is chosen here rather than left to a rewrite rule.
///
/// Every division below has a non-negative dividend, so truncation is the floor
/// the algorithm calls for and [`OP_DIVN_K`] is the instruction that carries
/// that claim: an i64-nanosecond instant only spans
/// ~1678-2262, which keeps `days` inside ±106752 and `z = days + 719468` inside
/// [612716, 826220]. A timestamp outside that range cannot exist in this VM —
/// the column payload is i64 nanos.
fn emit_civil_from_days(ctx: &mut LowerCtxF, days: TReg) -> (TReg, TReg, TReg) {
    let z = emit_int_bin_k(ctx, OP_ADD, days, 719_468);
    let era = emit_nonneg_div_k(ctx, z, 146_097);
    let era_days = emit_int_bin_k(ctx, OP_MUL, era, 146_097);
    let doe = emit_int_bin(ctx, OP_SUB, z, era_days);

    // yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365
    let by_1460 = emit_nonneg_div_k(ctx, doe, 1_460);
    let by_36524 = emit_nonneg_div_k(ctx, doe, 36_524);
    let by_146096 = emit_nonneg_div_k(ctx, doe, 146_096);
    let t1 = emit_int_bin(ctx, OP_SUB, doe, by_1460);
    let t2 = emit_int_bin(ctx, OP_ADD, t1, by_36524);
    let t3 = emit_int_bin(ctx, OP_SUB, t2, by_146096);
    let yoe = emit_nonneg_div_k(ctx, t3, 365);
    let era400 = emit_int_bin_k(ctx, OP_MUL, era, 400);
    let year_of_era = emit_int_bin(ctx, OP_ADD, yoe, era400);

    // doy = doe - (365*yoe + yoe/4 - yoe/100)   (days since 1 March)
    let y365 = emit_int_bin_k(ctx, OP_MUL, yoe, 365);
    let y4 = emit_nonneg_div_k(ctx, yoe, 4);
    let y100 = emit_nonneg_div_k(ctx, yoe, 100);
    let s1 = emit_int_bin(ctx, OP_ADD, y365, y4);
    let s2 = emit_int_bin(ctx, OP_SUB, s1, y100);
    let doy = emit_int_bin(ctx, OP_SUB, doe, s2);

    // mp = (5*doy + 2)/153 ; day0 = doy - (153*mp + 2)/5
    let d5 = emit_int_bin_k(ctx, OP_MUL, doy, 5);
    let d5p2 = emit_int_bin_k(ctx, OP_ADD, d5, 2);
    let mp = emit_nonneg_div_k(ctx, d5p2, 153);
    let m153 = emit_int_bin_k(ctx, OP_MUL, mp, 153);
    let m153p2 = emit_int_bin_k(ctx, OP_ADD, m153, 2);
    let month_start = emit_nonneg_div_k(ctx, m153p2, 5);
    let day0 = emit_int_bin(ctx, OP_SUB, doy, month_start);

    // month0 = mp + (mp < 10 ? 2 : -10), written as mp + 2 - 12*(mp >= 10) so
    // the select is arithmetic on a 0/1 comparison rather than a branch.
    let ge10 = emit_int_bin_k(ctx, OP_GE, mp, 10);
    let mp2 = emit_int_bin_k(ctx, OP_ADD, mp, 2);
    let wrap = emit_int_bin_k(ctx, OP_MUL, ge10, 12);
    let month0 = emit_int_bin(ctx, OP_SUB, mp2, wrap);

    // The era year starts in March, so January and February belong to the next
    // calendar year: year = year_of_era + (month0 <= 1).
    let le1 = emit_int_bin_k(ctx, OP_LE, month0, 1);
    let year = emit_int_bin(ctx, OP_ADD, year_of_era, le1);
    (year, month0, day0)
}

/// Days since the Unix epoch of 1 January of `year` — Hinnant's
/// `days_from_civil(year, 1, 1)`, specialised: for month 1 the March-based
/// `doy` term `(153*(m+9) + 2)/5 + d - 1` folds to the constant 306 (1 March to
/// the following 1 January). Used to turn an absolute day count into a
/// day-of-year. `year - 1` is positive over the representable range, so
/// truncation is again the floor the algorithm wants.
fn emit_days_of_jan1(ctx: &mut LowerCtxF, year: TReg) -> TReg {
    let y = emit_int_bin_k(ctx, OP_SUB, year, 1);
    let era = emit_nonneg_div_k(ctx, y, 400);
    let era400 = emit_int_bin_k(ctx, OP_MUL, era, 400);
    let yoe = emit_int_bin(ctx, OP_SUB, y, era400);
    let y365 = emit_int_bin_k(ctx, OP_MUL, yoe, 365);
    let y4 = emit_nonneg_div_k(ctx, yoe, 4);
    let y100 = emit_nonneg_div_k(ctx, yoe, 100);
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
    Type,
}

fn cmp_class(bank: ValType) -> CmpClass {
    match bank {
        ValType::Int | ValType::UInt | ValType::Float => CmpClass::Numeric,
        ValType::Bool => CmpClass::Bool,
        ValType::Str => CmpClass::Str,
        ValType::Timestamp => CmpClass::Timestamp,
        ValType::Duration => CmpClass::Duration,
        ValType::Type => CmpClass::Type,
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
        // `+` on a timestamp and a duration commutes, and the tree-walker
        // answers both orders (`objects.rs` `Duration`/`Timestamp` add arms).
        // Only this order was missing here, so `d + t` was declined while
        // `t + d` lowered.
        (ops::ADD, Duration, Timestamp) => Some(Timestamp),
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
    // The table is indexed by the string's id, one answer per DISTINCT string,
    // which the literal-scan encoding's shared sentinel cannot carry.
    ctx.str_dict_required = true;
    let table = ctx.fresh(ValType::Int);
    ctx.scalar_seeds.push(ScalarSeed {
        kind: SeedKind::StrPredicate(pred),
        reg: table.idx,
    });
    // `8` is the element stride, a property of the table and not of the batch,
    // so unlike the table's address it is a genuine immediate.
    let ea = emit_int_bin_k(ctx, OP_MUL, s, 8);
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
        // Filtered on the bank as well, like `slot_path_of` below: the int and
        // float files number independently, so a register index alone names a
        // slot in either of them.
        if let Some(slot) = self
            .slots
            .iter()
            .find(|s| s.reg == r.idx && s.ty == r.bank)
        {
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
    ctx.const_reg(ValType::Bool, v as i64)
}

/// A zero in the int file, for the sign tests a mixed int/uint comparison needs.
fn emit_zero_const(ctx: &mut LowerCtxF) -> TReg {
    ctx.const_reg(ValType::Int, 0)
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
        // by a constant — and `OP_DIV_K` is exactly toward-zero (it divides the
        // magnitudes and reapplies the sign, as `OP_DIV` does), so this is
        // bit-exact with the tree-walker. The divisor rides in the word stream,
        // which is what lets the optimizer expand it; the same constant in a
        // prelude register left a residual call per row. The six calendar names
        // have no `duration` overload and bail there.
        //
        // On a `timestamp` the answer is a calendar field instead: split the
        // instant into a FLOORED day count plus nanoseconds-of-day, read the
        // clock fields off the remainder and the date fields off the day count
        // via civil-from-days. All of it is int-file arithmetic on immediates,
        // so the whole conversion stays inside the traced loop with no call in
        // it -- `t.getDayOfMonth()` is eleven divisions, and through registers
        // it was eleven residual calls per row.
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
                        "getHours" => emit_nonneg_div_k(ctx, nanos_of_day, 3_600_000_000_000),
                        "getMinutes" => {
                            let m = emit_nonneg_div_k(ctx, nanos_of_day, 60_000_000_000);
                            emit_nonneg_mod_k(ctx, m, 60)
                        }
                        "getSeconds" => {
                            let s = emit_nonneg_div_k(ctx, nanos_of_day, 1_000_000_000);
                            emit_nonneg_mod_k(ctx, s, 60)
                        }
                        "getMilliseconds" => {
                            let ms = emit_nonneg_div_k(ctx, nanos_of_day, 1_000_000);
                            emit_nonneg_mod_k(ctx, ms, 1_000)
                        }
                        // `weekday().num_days_from_sunday()`: 1970-01-01 was a
                        // Thursday (4 days from Sunday). `days` can be negative,
                        // so it is first shifted past zero by a MULTIPLE OF 7 --
                        // `106_757` is `7 * 15_251` and exceeds the `106_752`
                        // days an i64-nanosecond instant can reach -- which
                        // leaves the residue unchanged and the remainder already
                        // in `[0, 7)`. The alternative is a signed remainder plus
                        // a floor correction: a magnitude round trip and three
                        // more ops to undo it.
                        "getDayOfWeek" => {
                            let shifted = emit_int_bin_k(ctx, OP_ADD, days, 4 + 106_757);
                            emit_nonneg_mod_k(ctx, shifted, 7)
                        }
                        "getFullYear" => emit_civil_from_days(ctx, days).0,
                        // `month0()` / `day0()` are 0-based, which is the base
                        // the helper answers in; `day()` is the one that adds.
                        "getMonth" => emit_civil_from_days(ctx, days).1,
                        "getDate" => {
                            let (_, _, day0) = emit_civil_from_days(ctx, days);
                            emit_int_bin_k(ctx, OP_ADD, day0, 1)
                        }
                        "getDayOfMonth" => emit_civil_from_days(ctx, days).2,
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

    // `type(x)` the checker could not fold, because its argument is not a
    // literal. The answer is still decided while lowering: the argument's BANK
    // is its CEL type, so the call is a green type constant and the argument's
    // ops are the only thing left of it.
    //
    // Reading the bank is sound because a bank is only assigned to a value the
    // machine actually carries: a list-valued expression has no scalar register
    // and never reaches the bank read -- the arm just below answers the one
    // shape that names a list, and any other list-valued argument declines in
    // `compile_t` -- so no `type(xs)` can be answered `int` from its element
    // bank. `timestamp` and `duration` decline because their type names --
    // `google.protobuf.Timestamp` and `.Duration` -- are message types, which
    // the index table does not carry.
    if name == "type" && call.args.len() == 1 {
        // A declared list's type is `list` whatever it holds, so this one needs
        // no register and does not have to be a value the machine carries. The
        // name is resolved the way `compile_t` resolves it, because a local and
        // an iteration variable both shadow the schema: in
        // `xs.map(xs, type(xs))` the inner `xs` is an element, not the list.
        if let Expr::Ident(v) = &call.args[0].expr {
            if !ctx.locals.contains_key(v)
                && !ctx.list_loop.iter().any(|l| l.iter_var == *v)
                && declares_list(ctx.schema, v)
            {
                let id = type_const_id("list").expect("a bank name the index table carries");
                return Ok(ctx.const_reg(ValType::Type, id));
            }
        }
        let a = compile_t(ctx, &call.args[0])?;
        let denoted = match a.bank {
            ValType::Int => "int",
            ValType::UInt => "uint",
            ValType::Float => "double",
            ValType::Bool => "bool",
            ValType::Str => "string",
            ValType::Type => "type",
            ValType::Timestamp | ValType::Duration => {
                return Err(LowerError::unsupported("type() of a message type"))
            }
        };
        let id = type_const_id(denoted).expect("a bank name the index table carries");
        return Ok(ctx.const_reg(ValType::Type, id));
    }

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
        ctx.temporal_consts.push(nanos);
        return Ok(ctx.const_reg(ValType::Timestamp, nanos));
    }
    if name == "duration" && call.args.len() == 1 {
        let s = as_string_literal(&call.args[0])
            .ok_or_else(|| LowerError::unsupported("duration() non-literal argument"))?;
        let (_, dur) = crate::duration::parse_duration(s)
            .map_err(|_| LowerError::unsupported("duration() literal parse"))?;
        let nanos = dur
            .num_nanoseconds()
            .ok_or_else(|| LowerError::unsupported("duration outside i64-nanos range"))?;
        ctx.temporal_consts.push(nanos);
        return Ok(ctx.const_reg(ValType::Duration, nanos));
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
        // A runtime-list ELEMENT's length, which no row column carries.
        if let Some(r) = size_of_list_element(ctx, &call.args[0])? {
            return Ok(r);
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
                LiteralValue::Int(i) => i.to_string(),
                LiteralValue::UInt(u) => u.to_string(),
                LiteralValue::Double(f) => f.to_string(),
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
            return Ok(emit_bool_const(ctx, false));
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
            if x.bank == ValType::Str
                && !ctx.is_str_literal_seed(&x)
                && !ctx.is_str_literal_seed(&ev)
            {
                ctx.str_dict_required = true;
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
        let Some(idx) = as_int_literal(&call.args[1]) else {
            // A RUNTIME index reads the same flattened element column at an
            // address the row supplies. Only a list has that form: a caller's
            // flattened `base[k]` column names one FIXED index and cannot
            // answer a varying one.
            if declares_list(ctx.schema, &base) {
                return lower_var_index(ctx, &base, None, &call.args[1]);
            }
            // A literal string key is map lookup, which `base.key` spells the
            // same way and the schema declares under that name. An undeclared
            // key is not a hole here: the walker raises NoSuchKey for it, and
            // declining is what lets it.
            if let Some(key) = as_string_literal(&call.args[1]).filter(|k| is_cel_ident(k)) {
                return ctx.slot(format!("{base}.{key}"));
            }
            return Err(LowerError::unsupported("non-constant index"));
        };
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
        // either. So an absorbing operand answers before anything else is
        // compiled; without one, `1 || false` stays the type error it is.
        let absorbing = name == ops::LOGICAL_OR;
        if call.args.iter().any(|a| absorbs(a, absorbing)) {
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
            ctx.note_fresh_write();
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
        // `a % K == 0` for a power-of-two `K` is `(a & (|K| - 1)) == 0`.
        // Divisibility does not depend on the dividend's sign, so the magnitude
        // round trip `OP_MOD_CHK_K` needs in order to ANSWER `a % K` is dead
        // when the only consumer is a zero test.
        if matches!(name, ops::EQUALS | ops::NOT_EQUALS) {
            if let Some(r) = lower_divisibility(ctx, iop, call)? {
                return Ok(r);
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
        // A type value carries equality and nothing else -- `type(1) < type(2)`
        // is NoSuchOverload -- and it is a constant by construction, because the
        // only thing that mints one is a `type(x)` the checker already folded.
        // Its index comes from one injective table, so equality is an index
        // compare and both sides being constant it answers here.
        if a.bank == ValType::Type {
            if !matches!(name, ops::EQUALS | ops::NOT_EQUALS) {
                return Err(LowerError::unsupported("ordering on type values"));
            }
            return Ok(match (ctx.const_of(a), ctx.const_of(b)) {
                (Some(x), Some(y)) => emit_bool_const(ctx, (x == y) == (name == ops::EQUALS)),
                _ => emit_bin(ctx, iop, a, b, ValType::Bool),
            });
        }
        // Two constants of one numeric or bool bank compare now, once.
        if let (Some(x), Some(y)) = (ctx.const_of(a), ctx.const_of(b)) {
            if let Some(word) = fold_cmp(name, a.bank, b.bank, x, y) {
                return Ok(ctx.const_reg(ValType::Bool, word));
            }
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
            // Recorded here rather than by re-walking the AST: this is the one
            // site that turns a string ordering into an int compare, so the
            // flag and the op it describes are emitted together.
            if a.bank == ValType::Str && !matches!(name, ops::EQUALS | ops::NOT_EQUALS) {
                ctx.orders_strings = true;
            }
            if a.bank == ValType::Str
                && !ctx.is_str_literal_seed(&a)
                && !ctx.is_str_literal_seed(&b)
            {
                // Neither side is a literal: the ids must separate every
                // distinct string, not only the literals.
                ctx.str_dict_required = true;
            }
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
        // Both operands known while lowering — an unrolled literal
        // comprehension's `x * 2`, a literal appended by a step — fold to one
        // constant. Only where the row loop would have ANSWERED: an overflow or
        // a zero divisor stays an op, so the row traps the way the tree-walker
        // raises.
        if let (Some(x), Some(y)) = (ctx.const_of(a), ctx.const_of(b)) {
            if let Some(word) = fold_arith(name, a.bank, b.bank, x, y) {
                return Ok(ctx.const_reg(a.bank, word));
            }
        }
        // A constant divisor rides in the instruction stream, not in a register.
        // `program` is a green argument, so the immediate is a CONSTANT the
        // trace optimizer expands into multiply-and-shift, while the same value
        // in a prelude-loaded register reaches the row loop as an opaque
        // loop-invariant and leaves the `int.udiv`/`int.umod` residual call in
        // the body. A zero divisor keeps the register form, whose guard traps it
        // the way the tree-walker raises.
        let const_divisor = ctx.const_of(b).filter(|k| *k != 0);
        let kop = if a.bank == b.bank {
            match (name, a.bank) {
                (ops::DIVIDE, ValType::Int) => Some(OP_DIV_CHK_K),
                (ops::MODULO, ValType::Int) => Some(OP_MOD_CHK_K),
                (ops::DIVIDE, ValType::UInt) => Some(OP_UDIV_K),
                (ops::MODULO, ValType::UInt) => Some(OP_UMOD_K),
                _ => None,
            }
        } else {
            None
        };
        if let (Some(k), Some(kop)) = (const_divisor, kop) {
            let d = ctx.fresh(a.bank);
            ctx.body
                .extend_from_slice(&[kop, a.idx as i64, k, d.idx as i64]);
            // Only the signed pair can still trap: `INT_MIN / -1` and
            // `INT_MIN % -1` are what the tree-walker's `checked_div`/
            // `checked_rem` report as `Overflow`. Every `uint` pair with a
            // nonzero divisor has a representable answer, so those two carry no
            // trap word.
            if a.bank == ValType::Int {
                ctx.body.push(OVF_FLAG_REG as i64);
            }
            return Ok(d);
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
                    ValType::Type => return Err(LowerError::unsupported("unary negate on type")),
                    // `-duration` is CEL and the walker answers it
                    // (`objects.rs` `Value::Duration(d) => Value::Duration(-d)`);
                    // `-timestamp` has no overload, so the walker raises and
                    // there is no answer to agree with.
                    //
                    // The bank is i64 nanoseconds, so the negate is the int
                    // one. Counting the operation is what keeps it exact:
                    // `temporal_bound` is `i64::MAX / (ops + 1)`, and every
                    // temporal column is checked against it before the batch
                    // runs, so `|v| <= bound < i64::MAX` and `-v` cannot
                    // overflow. Left uncounted, an expression whose ONLY
                    // temporal operation is this one would carry no bound at
                    // all, and `-i64::MIN` would wrap where chrono's
                    // `{secs, nanos}` does not.
                    ValType::Duration => {
                        ctx.temporal_ops += 1;
                        OP_NEG
                    }
                    ValType::Timestamp => {
                        return Err(LowerError::unsupported("unary negate on timestamp"))
                    }
                };
                let d = ctx.fresh(a.bank);
                ctx.body
                    .extend_from_slice(&[op, a.idx as i64, d.idx as i64]);
                Ok(d)
            }
            _ => compile_host_call_t(ctx, name, call),
        }
    }
}

/// A call to a user function, in its scalar form.
///
/// The function comes from the [`Context`] the lowering was given, and only
/// when it has a [`ScalarFn`] form and the stdlib declares no overload under
/// the name — the tree-walker consults `Env::find_overload` before the
/// context's functions (`objects.rs`, the `call.target == None` arm), so a
/// user function under a stdlib name may or may not be the one it calls, and
/// this tier does not guess. Each argument must already be in the closure's
/// bank: `i64` arguments read the int bank, and only [`ValType::Int`] is one
/// (`bool`, `uint` and the temporal types share the bank but are not `i64` to
/// the walker's `FromValue`). The closure's entry word is a loop invariant, so
/// its load goes in the prelude with the literals.
fn compile_host_call_t(
    ctx: &mut LowerCtxF,
    name: &str,
    call: &CallExpr,
) -> Result<TReg, LowerError> {
    use crate::magic::ScalarFn;
    let decline = || LowerError::unsupported(format!("call `{name}`"));
    let Some(functions) = ctx.functions else {
        return Err(decline());
    };
    if functions.env().declares_function(name) {
        return Err(decline());
    }
    let Some(scalar) = functions
        .get_function(name)
        .and_then(|f| f.scalar().cloned())
    else {
        return Err(decline());
    };
    let (arity, bank, op) = match &*scalar {
        ScalarFn::Int1(_) => (1, ValType::Int, OP_HOST_CALL1_I),
        ScalarFn::Int2(_) => (2, ValType::Int, OP_HOST_CALL2_I),
        ScalarFn::Float1(_) => (1, ValType::Float, OP_HOST_CALL1_F),
        ScalarFn::Float2(_) => (2, ValType::Float, OP_HOST_CALL2_F),
    };
    if call.args.len() != arity {
        return Err(LowerError::unsupported(format!("call `{name}` arity")));
    }
    let mut args = Vec::with_capacity(arity);
    for arg in &call.args {
        let r = compile_t(ctx, arg)?;
        if r.bank != bank {
            return Err(LowerError::unsupported(format!(
                "call `{name}` on {:?}",
                r.bank
            )));
        }
        args.push(r.idx as i64);
    }
    let f = ctx.const_reg(ValType::Int, scalar.entry_word());
    ctx.host_fns.push(scalar);
    let d = ctx.fresh(bank);
    ctx.body.push(op);
    ctx.body.push(f.idx as i64);
    ctx.body.extend_from_slice(&args);
    ctx.body.push(d.idx as i64);
    Ok(d)
}

/// `size()` of a runtime-list ELEMENT — the loop variable itself, or one of its
/// declared fields. `None` when the argument is not one.
///
/// A row's `size(<string>)` is a derived per-row column; this is the same thing
/// one level down. The element lengths are a derived column over the FLATTENED
/// element buffer, read at the address the element itself is read at, so an
/// element's length costs the same load its value does — and the loop body
/// still computes no lengths.
fn size_of_list_element(ctx: &mut LowerCtxF, e: &IdedExpr) -> Result<Option<TReg>, LowerError> {
    let (name, field) = match &e.expr {
        Expr::Ident(n) => (n.as_str(), None),
        Expr::Select(sel) if !sel.test => match &sel.operand.expr {
            Expr::Ident(n) => (n.as_str(), Some(sel.field.as_str())),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };
    // A local shadows the loop variable, and the unroll binds its own that way.
    if ctx.locals.contains_key(name) {
        return Ok(None);
    }
    // Innermost first, as everywhere an iteration variable is resolved.
    let Some(l) = ctx.list_loop.iter().rev().find(|l| l.iter_var == name) else {
        return Ok(None);
    };
    let (list, ea_reg) = (l.list.clone(), l.ea_reg);
    let elem = elem_slot_path(&list, field);
    // Only a string element has a length here. A list of lists needs per-element
    // offsets, which a flat element column cannot express, and a numeric element
    // has no `size` overload at all.
    if ctx.schema.get(&elem).copied() != Some(ValType::Str) {
        return Err(LowerError::unsupported("size() of a non-string element"));
    }
    Ok(Some(ctx.elem_slot_typed(
        size_slot_path(&elem),
        ValType::Int,
        ea_reg,
    )?))
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

    compile_literal_comprehension_mode(ctx, comp, &elements, AccuMode::Value)
}

/// The green-length unroll, in a chosen accumulator mode.
///
/// Straight-line, so unlike [`compile_list_comprehension_mode`] the accumulator
/// needs no fixed register: each iteration's step simply becomes the next one's
/// accumulator.
fn compile_literal_comprehension_mode(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    elements: &[IdedExpr],
    mode: AccuMode,
) -> Result<TReg, LowerError> {
    let prev_iter = ctx.locals.remove(&comp.iter_var);
    let prev_accu = ctx.locals.remove(&comp.accu_var);

    let mut accu = match mode {
        AccuMode::Value => compile_t(ctx, &comp.accu_init)?,
        // The list the step would have built starts empty, so its length starts
        // at zero. `accu_init` is not compiled at all: it is the `[]` the
        // machine has no value for.
        AccuMode::Length | AccuMode::Collect { .. } => emit_int_const(ctx, 0),
    };
    for elem in elements {
        let x_reg = compile_t(ctx, elem)?;
        ctx.locals.insert(comp.iter_var.clone(), x_reg);
        ctx.locals.insert(comp.accu_var.clone(), accu);
        accu = match mode {
            AccuMode::Value => compile_t(ctx, &comp.loop_step)?,
            AccuMode::Length | AccuMode::Collect { .. } => {
                compile_len_step(ctx, &comp.loop_step, comp, accu, mode)?
            }
        };
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
        Value::Float(f) => return Ok(emit_float_const(ctx, *f)),
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
        // `type(x)` is folded by the checker, so a type value reaches the
        // lowering as a constant opaque and never as a call. It carries no
        // payload beyond which type it denotes, so its index into
        // `TYPE_CONST_NAMES` IS the value.
        Value::Opaque(o) => {
            let tv = o
                .downcast_ref::<TypeValue>()
                .ok_or_else(|| LowerError::unsupported("constant of type `opaque`"))?;
            let id = type_const_id(tv.name())
                .ok_or_else(|| LowerError::unsupported(format!("type constant `{}`", tv.name())))?;
            (ValType::Type, id)
        }
        other => {
            return Err(LowerError::unsupported(format!(
                "constant of type `{}`",
                other.type_of()
            )))
        }
    };
    Ok(ctx.const_reg(bank, word))
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
    // Zero-trip guard: two empty lists are equal, and the back-edge is a
    // do-while.
    let zero_trip = ctx.emit_jump_if_above(one, n);

    // Nothing here reads an element index, only the two byte offsets, so each
    // list carries its own offset across the back edge and the first one is
    // also the counter. Deriving them instead costs two adds and two multiplies
    // an element, against the one add the counter would have cost anyway.
    let ea_a = emit_int_bin_k(ctx, OP_MUL, off_a, 8);
    let ea_b = emit_int_bin_k(ctx, OP_MUL, off_b, 8);
    let last_a = emit_int_bin(ctx, OP_ADD, off_a, n);
    let limit = emit_int_bin_k(ctx, OP_MUL, last_a, 8);

    let inner = ctx.body.len();
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
    ctx.body.extend_from_slice(&[
        OP_ADD_IMM,
        ea_a.idx as i64,
        8,
        ea_a.idx as i64,
        OP_ADD_IMM,
        ea_b.idx as i64,
        8,
        ea_b.idx as i64,
    ]);
    ctx.emit_back_edge(limit, ea_a, inner);
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

/// One `map`/`filter` level of a comprehension chain.
///
/// `items.filter(x, p).map(x, f)` is two comprehensions, the outer one
/// iterating the inner one's RESULT — a list whose length is data-dependent,
/// which this machine has no value for. But CEL has no way to name that
/// intermediate twice: a comprehension result is consumed by exactly one
/// chained call, so the whole chain is one pass over the base list with each
/// link's predicate ANDed and each link's value fed to the next. That is what
/// gets compiled, and the intermediate list never exists.
struct Link<'a> {
    /// The link's own iteration variable, bound to what the link before it
    /// produced.
    iter_var: &'a str,
    /// A `filter` predicate, or `None` for a `map`.
    cond: Option<&'a IdedExpr>,
    /// The appended expression — a bare `iter_var` for a `filter`, which is
    /// what makes a `filter` pass its element through unchanged.
    elem: &'a IdedExpr,
}

fn is_ident(e: &IdedExpr, name: &str) -> bool {
    matches!(&e.expr, Expr::Ident(n) if n == name)
}

/// One chain level, or `None` if this comprehension is not one of the two
/// shapes the `map`/`filter` macros desugar to.
fn parse_link(comp: &ComprehensionExpr) -> Option<Link<'_>> {
    if comp.iter_var2.is_some() {
        return None;
    }
    // `map`/`filter` start from `[]` and hand the accumulator straight back, so
    // anything else is a comprehension whose elements these are not.
    if !matches!(&comp.accu_init.expr, Expr::List(l) if l.elements.is_empty()) {
        return None;
    }
    if !is_ident(&comp.result, &comp.accu_var) {
        return None;
    }
    // `c ? (@result + [e]) : @result` for `filter`, `@result + [e]` for `map`.
    let (cond, append) = match &comp.loop_step.expr {
        Expr::Call(c)
            if c.func_name == ops::CONDITIONAL
                && c.args.len() == 3
                && is_ident(&c.args[2], &comp.accu_var) =>
        {
            (Some(&c.args[0]), &c.args[1])
        }
        _ => (None, &comp.loop_step),
    };
    let Expr::Call(add) = &append.expr else {
        return None;
    };
    if add.func_name != ops::ADD || add.args.len() != 2 || !is_ident(&add.args[0], &comp.accu_var) {
        return None;
    }
    let Expr::List(l) = &add.args[1].expr else {
        return None;
    };
    let [elem] = &l.elements[..] else {
        return None;
    };
    Some(Link {
        iter_var: &comp.iter_var,
        cond,
        elem,
    })
}

/// Flatten a `map`/`filter` chain into its links, innermost first, with the
/// schema path of the declared list they all read from.
///
/// A single `items.map(..)` is a chain of one, so this is the only route into a
/// collected runtime-list comprehension — chained or not.
fn chain_links<'a>(
    schema: &Schema,
    comp: &'a ComprehensionExpr,
) -> Option<(String, Vec<Link<'a>>)> {
    let mut links = Vec::new();
    let mut cur = comp;
    let path = loop {
        links.push(parse_link(cur)?);
        match &cur.iter_range.expr {
            Expr::Comprehension(inner) => cur = inner,
            _ => break resolve_path(&cur.iter_range).ok()?,
        }
    };
    if !declares_list(schema, &path) {
        return None;
    }
    links.reverse();
    Some((path, links))
}

/// Compile one fused iteration of a chain over the base loop's element.
///
/// Returns the ANDed predicate of every `filter` link — `None` when the chain
/// is all `map`s — and the value the last link appends, which is `None` when
/// every link passed the element straight through.
///
/// Each link's variable is bound to what the link before it produced: an ALIAS
/// for the loop's element while the value is still the element itself, and an
/// ordinary local once a `map` has computed one.
///
/// As everywhere else in this loop body, the fold is eager: a later link runs
/// even for an element an earlier predicate rejected. Where that raises and the
/// tree-walker would not have looked, the row traps and the walker owns it.
fn compile_chain(
    ctx: &mut LowerCtxF,
    links: &[Link],
    list: &str,
    ea_reg: usize,
) -> Result<(Option<TReg>, Option<TReg>), LowerError> {
    let depth = ctx.list_loop.len();
    let saved: Vec<(&str, Option<TReg>)> = links
        .iter()
        .map(|l| (l.iter_var, ctx.locals.remove(l.iter_var)))
        .collect();
    let r = compile_chain_body(ctx, links, list, ea_reg);
    ctx.list_loop.truncate(depth);
    // Reversed, so a name two links share is restored to what it held before
    // the FIRST of them took it.
    for (name, prev) in saved.into_iter().rev() {
        match prev {
            Some(v) => ctx.locals.insert(name.to_string(), v),
            None => ctx.locals.remove(name),
        };
    }
    r
}

fn compile_chain_body(
    ctx: &mut LowerCtxF,
    links: &[Link],
    list: &str,
    ea_reg: usize,
) -> Result<(Option<TReg>, Option<TReg>), LowerError> {
    let mut cond: Option<TReg> = None;
    let mut value: Option<TReg> = None;
    for (k, link) in links.iter().enumerate() {
        // The base loop already registered the first link's variable.
        if k > 0 {
            match value {
                None => {
                    // A later link iterates the SAME elements as the base loop,
                    // so it shares that loop's index register; there is no
                    // second one to derive.
                    let idx_reg = ctx.elem_idx_reg(ea_reg).ok_or_else(|| {
                        LowerError::unsupported("chain link outside its base loop")
                    })?;
                    ctx.list_loop.push(ListLoop {
                        iter_var: link.iter_var.to_string(),
                        list: list.to_string(),
                        ea_reg,
                        idx_reg,
                    })
                }
                Some(v) => {
                    ctx.locals.insert(link.iter_var.to_string(), v);
                }
            }
        }
        if let Some(c) = link.cond {
            let c = compile_t(ctx, c)?;
            if c.bank != ValType::Bool {
                return Err(LowerError::unsupported("filter predicate must be bool"));
            }
            cond = Some(match cond {
                None => c,
                Some(prev) => {
                    let r = ctx.fresh(ValType::Bool);
                    ctx.body.extend_from_slice(&[
                        OP_AND,
                        prev.idx as i64,
                        c.idx as i64,
                        r.idx as i64,
                    ]);
                    r
                }
            });
        }
        if !is_ident(link.elem, link.iter_var) {
            value = Some(compile_t(ctx, link.elem)?);
        }
    }
    Ok((cond, value))
}

/// One fused iteration of a collected chain: the element stored, and the
/// accumulator advanced by one where every predicate holds.
fn emit_chain_collect(
    ctx: &mut LowerCtxF,
    links: &[Link],
    list: &str,
    ea_reg: usize,
    cursor: usize,
) -> Result<(), LowerError> {
    let (cond, value) = compile_chain(ctx, links, list, ea_reg)?;
    let appended = match value {
        Some(v) => Appended::Value(v),
        // Every link passed the element through, so what is stored is the
        // element itself — under the name the BASE loop registered.
        None => Appended::Element(links[0].iter_var),
    };
    // Collecting under a predicate: the store runs for every element, and the
    // cursor advances only where the predicate holds — so a rejected element
    // writes to the slot the next accepted one will overwrite. That is what
    // keeps the body straight-line, with no branch around the store.
    //
    // The advance IS the predicate. `OP_SELECT c, t, f` computes `f + c * (t - f)`,
    // so selecting between the constants 1 and 0 is `0 + c * 1` — the condition
    // register itself, for any word it holds. Handing it to the store as the
    // advance rather than advancing by one and subtracting the complement is
    // what keeps the append one instruction: the two ops that dance cancel.
    emit_element_store(
        ctx,
        appended,
        cursor,
        match cond {
            Some(c) => Advance::By(c),
            None => Advance::One,
        },
    )
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
    // A LITERAL range has a green trip count, so it unrolls; a declared column
    // has a red one and gets an inner loop. Either way the elements go to the
    // ragged output, because neither can be a value on this machine.
    if let Expr::List(l) = &comp.iter_range.expr {
        return Some(collect_literal_comprehension(ctx, comp, &l.elements));
    }
    let (path, links) = chain_links(ctx.schema, comp)?;
    Some(collect_list_comprehension(ctx, comp, &path, &links))
}

/// [`collect_list_comprehension`] for a literal range: the same ragged output,
/// filled by the unroll rather than by an inner loop.
///
/// A literal list has no schema entry to take field names from, so the output
/// is always a list of scalars — one unnamed field, whose bank the appended
/// expression decides.
fn collect_literal_comprehension(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    elements: &[IdedExpr],
) -> Result<TReg, LowerError> {
    if comp.iter_var2.is_some() {
        return Err(LowerError::unsupported("two-variable comprehension"));
    }
    let elem = literal_append_bank(ctx, comp, elements)?;
    let base_regs = vec![ctx.fresh(ValType::Int).idx];
    let cursor = ctx.fresh(ValType::Int);
    ctx.prelude
        .extend_from_slice(&[OP_LOAD_CONST, 0, cursor.idx as i64]);
    ctx.list_output = Some(ListOutput {
        source: ListSource::Literal(elements.len()),
        fields: vec![(None, elem)],
        base_regs,
    });
    compile_literal_comprehension_mode(
        ctx,
        comp,
        elements,
        AccuMode::Collect { cursor: cursor.idx },
    )
}

/// The expression a `map`/`filter` step appends, or `None` if the step is
/// neither of the two shapes the macros desugar to (`@result + [e]`, and
/// `c ? (@result + [e]) : @result`).
fn appended_element(step: &IdedExpr) -> Option<&IdedExpr> {
    let Expr::Call(call) = &step.expr else {
        return None;
    };
    match call.func_name.as_str() {
        ops::ADD if call.args.len() == 2 => match &call.args[1].expr {
            Expr::List(l) => l.elements.first(),
            _ => None,
        },
        ops::CONDITIONAL if call.args.len() == 3 => appended_element(&call.args[1]),
        _ => None,
    }
}

/// The bank a literal-list comprehension appends, from a THROWAWAY lowering of
/// the appended expression with the iteration variable bound to the list's
/// first element — same reason as [`map_body_bank`]: the buffer has to be
/// described before the body is compiled, and compiling the body twice would
/// emit it twice.
///
/// The first element stands for all of them. Where they disagree, the store
/// itself rejects the mismatch: [`emit_element_store`] compares every element's
/// bank against this one.
fn literal_append_bank(
    ctx: &LowerCtxF,
    comp: &ComprehensionExpr,
    elements: &[IdedExpr],
) -> Result<ValType, LowerError> {
    let e = appended_element(&comp.loop_step)
        .ok_or_else(|| LowerError::unsupported("collected step is not an append"))?;
    let first = elements
        .first()
        .ok_or_else(|| LowerError::unsupported("collected list has no element column"))?;
    let mut probe = probe_ctx(ctx);
    let x = compile_t(&mut probe, first)?;
    probe.locals.insert(comp.iter_var.clone(), x);
    Ok(compile_t(&mut probe, e)?.bank)
}

fn collect_list_comprehension(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    path: &str,
    links: &[Link],
) -> Result<TReg, LowerError> {
    // What the output's elements look like. A chain that ends by computing a
    // value has one unnamed field whose bank the chain decides; one that hands
    // the element back — what `filter` does — has the source's own fields.
    let fields: Vec<(Option<String>, ValType)> = match chain_value_bank(ctx, links, path)? {
        Some(elem) => vec![(None, elem)],
        None => source_fields(ctx.schema, path),
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
        source: ListSource::Column(path.to_string()),
        fields,
        base_regs,
    });
    compile_list_comprehension_mode(
        ctx,
        comp,
        path,
        AccuMode::Collect { cursor: cursor.idx },
        links,
    )
}

/// The declared element fields of a list column, in name order — what a chain
/// of pure `filter`s hands back unchanged.
fn source_fields(schema: &Schema, path: &str) -> Vec<(Option<String>, ValType)> {
    let mut v: Vec<(Option<String>, ValType)> = schema
        .iter()
        .filter_map(|(k, t)| {
            let (l, f) = elem_slot_source(k)?;
            (l == path).then(|| (f.map(str::to_string), *t))
        })
        .collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    v
}

/// An empty lowering over the same schema, for asking what bank an expression
/// lands in without emitting it into the real body. Its registers, slots and
/// ops are all discarded with it; only the answer is kept.
fn probe_ctx<'a>(ctx: &LowerCtxF<'a>) -> LowerCtxF<'a> {
    LowerCtxF {
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
        orders_strings: false,
        str_dict_required: false,
        concats: Vec::new(),
        elem_map: HashMap::new(),
        list_loop: Vec::new(),
        elem_idx_addressed: BTreeSet::new(),
        last_fresh_write: None,
        const_pool: HashMap::new(),
        jump_fixups: Vec::new(),
        elem_words: 0,
        list_output: None,
        schema: ctx.schema,
        functions: ctx.functions,
        host_fns: Vec::new(),
    }
}

/// The bank a chain's last link appends, or `None` when every link passes the
/// element through.
///
/// The output buffers have to be described before the body is compiled, and
/// compiling the body twice would emit it twice — so the answer comes from a
/// THROWAWAY lowering of the same links against the same schema. Same chain,
/// same answer, and its registers, slots and ops are discarded with it.
fn chain_value_bank(
    ctx: &LowerCtxF,
    links: &[Link],
    path: &str,
) -> Result<Option<ValType>, LowerError> {
    let mut probe = probe_ctx(ctx);
    probe.slot_typed(size_slot_path(path), ValType::Int);
    probe.slot_typed(offset_slot_path(path), ValType::Int);
    let ea = probe.fresh(ValType::Int);
    let idx = probe.fresh(ValType::Int);
    probe.list_loop.push(ListLoop {
        iter_var: links[0].iter_var.to_string(),
        list: path.to_string(),
        ea_reg: ea.idx,
        idx_reg: idx.idx,
    });
    Ok(compile_chain(&mut probe, links, path, ea.idx)?
        .1
        .map(|v| v.bank))
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
    compile_list_comprehension_mode(ctx, comp, &path, AccuMode::Length, &[])
}

/// One iteration's contribution to a list accumulator's LENGTH.
///
/// In `Collect` mode this is the LITERAL unroll's step — a runtime list's is
/// fused from its whole chain by [`emit_chain_collect`] instead.
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
                    // compiled either way and written out. Over a literal range
                    // the iteration variable is the ordinary local the unroll
                    // bound, so it compiles like any other expression.
                    AccuMode::Collect { cursor } => {
                        let v = compile_t(ctx, e)?;
                        emit_element_store(ctx, Appended::Value(v), cursor, Advance::One)?
                    }
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
            // `OP_SELECT c, t, f` is `f + c * (t - f)`, so against the constant
            // zero the advance is `c * taken` and the rewind is `taken * (1 - c)`.
            // A branch that appends exactly one element makes those `c` and `!c`,
            // which the condition register already holds. A branch that appends
            // some other count — `c ? (@result + [a, b]) : @result`, or a nested
            // conditional whose own count is not a constant — makes them neither,
            // and there the select is the operation rather than a spelling of it.
            let one_element = ctx.const_of(taken) == Some(1);
            let delta = if one_element {
                c
            } else {
                let none = emit_int_const(ctx, 0);
                let d = ctx.fresh(ValType::Int);
                ctx.body.extend_from_slice(&[
                    OP_SELECT,
                    c.idx as i64,
                    taken.idx as i64,
                    none.idx as i64,
                    d.idx as i64,
                ]);
                d
            };
            if let AccuMode::Collect { cursor } = mode {
                // Undo the unconditional advance the store made, where the
                // predicate rejected.
                let back = if one_element {
                    emit_complement(ctx, c)
                } else {
                    let b = ctx.fresh(ValType::Int);
                    ctx.body.extend_from_slice(&[
                        OP_SELECT,
                        c.idx as i64,
                        zero.idx as i64,
                        taken.idx as i64,
                        b.idx as i64,
                    ]);
                    b
                };
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

/// How far the cursor moves once one collected element has been stored.
enum Advance {
    /// Past the element just written, which is kept.
    One,
    /// By what a bool register holds. A rejected element leaves the cursor
    /// where it was, so the next accepted store overwrites the slot.
    By(TReg),
}

/// What one collected iteration appends.
enum Appended<'a> {
    /// The loop's element itself — what `filter` hands back — named by the
    /// iteration variable it is bound to. Over a record list that is one
    /// element load per declared field, which is why it stays a NAME here
    /// rather than a register.
    Element(&'a str),
    /// A value the body computed, which is the single unnamed output field.
    Value(TReg),
}

/// Write one appended element to the ragged output and advance the cursor.
///
/// The advance is a parameter rather than a fixed step because a predicated
/// append is the same store: only how far the cursor travels afterwards
/// differs, and every spelling below carries that in the advance it already
/// had to emit.
fn emit_element_store(
    ctx: &mut LowerCtxF,
    appended: Appended,
    cursor: usize,
    advance: Advance,
) -> Result<(), LowerError> {
    let out = ctx
        .list_output
        .clone()
        .expect("collect mode allocates the output description");
    let cur = TReg {
        bank: ValType::Int,
        idx: cursor,
    };
    let mut values = Vec::with_capacity(out.fields.len());
    for (field, ty) in &out.fields {
        let v = match appended {
            Appended::Element(name) => {
                ctx.iter_var_slot(name, field.as_deref())?.ok_or_else(|| {
                    LowerError::unsupported("collected element is not a loop element")
                })?
            }
            Appended::Value(v) => v,
        };
        if v.bank != *ty {
            return Err(LowerError::unsupported("collected element bank"));
        }
        values.push(v);
    }
    // One output column appends in one instruction: the store and the advance
    // fused, the byte scale inside. Several columns share one scaled address
    // and advance once after the last store.
    if let [v] = values[..] {
        let (op, pred) = match (v.bank == ValType::Float, advance) {
            (false, Advance::One) => (OP_COL_PUSH, None),
            (true, Advance::One) => (OP_COL_PUSH_F, None),
            (false, Advance::By(p)) => (OP_COL_PUSH_IF, Some(p)),
            (true, Advance::By(p)) => (OP_COL_PUSH_IF_F, Some(p)),
        };
        ctx.body
            .extend_from_slice(&[op, out.base_regs[0] as i64, cursor as i64, v.idx as i64]);
        ctx.body.extend(pred.map(|p: TReg| p.idx as i64));
        ctx.body.push(cursor as i64);
        return Ok(());
    }
    let ea = emit_int_bin_k(ctx, OP_MUL, cur, 8);
    for (k, v) in values.into_iter().enumerate() {
        let op = if v.bank == ValType::Float {
            OP_COL_STORE_F
        } else {
            OP_COL_STORE
        };
        ctx.body
            .extend_from_slice(&[op, out.base_regs[k] as i64, ea.idx as i64, v.idx as i64]);
    }
    ctx.body.extend_from_slice(&match advance {
        Advance::One => [OP_ADD_IMM, cursor as i64, 1, cursor as i64],
        Advance::By(p) => [OP_ADD, cursor as i64, p.idx as i64, cursor as i64],
    });
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
    if ty == ValType::Str && !ctx.is_str_literal_seed(&x) {
        // The needle is not a literal: element ids must separate every
        // distinct string.
        ctx.str_dict_required = true;
    }

    let len = ctx.slot_typed(size_slot_path(list), ValType::Int);
    let off = ctx.slot_typed(offset_slot_path(list), ValType::Int);
    let one = emit_int_const(ctx, 1);

    // The accumulator is loop-carried, so it lives in a fixed register.
    let found = ctx.fresh(ValType::Bool);
    ctx.body
        .extend_from_slice(&[OP_LOAD_CONST, 0, found.idx as i64]);
    // Zero-trip guard: `x in []` is false, and the back-edge is a do-while.
    let zero_trip = ctx.emit_jump_if_above(one, len);

    // Only the byte offset is read here, so the loop carries it across the back
    // edge and counts with it. Deriving it instead costs an add and a multiply
    // an element, against the one add the counter would have cost anyway.
    let ea = emit_int_bin_k(ctx, OP_MUL, off, 8);
    let last = emit_int_bin(ctx, OP_ADD, off, len);
    let limit = emit_int_bin_k(ctx, OP_MUL, last, 8);

    let inner = ctx.body.len();
    let v = ctx.elem_slot(elem_path, ea.idx)?;
    ctx.elem_map.clear();
    let eq_op = if ty == ValType::Float { OP_FEQ } else { OP_EQ };
    let hit = ctx.fresh(ValType::Bool);
    ctx.body
        .extend_from_slice(&[eq_op, v.idx as i64, x.idx as i64, hit.idx as i64]);
    ctx.body
        .extend_from_slice(&[OP_OR, found.idx as i64, hit.idx as i64, found.idx as i64]);
    ctx.body
        .extend_from_slice(&[OP_ADD_IMM, ea.idx as i64, 8, ea.idx as i64]);
    ctx.emit_back_edge(limit, ea, inner);
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

    // An int or double element reads in one fused instruction, `OP_INDEX_K`,
    // which carries the range check, the trap, the default and the load. A
    // `bool` element column is one byte per element and has no fused form, so
    // it keeps the expanded shape below.
    if matches!(ty, ValType::Int | ValType::Float) {
        let out = ctx.fresh(ty);
        let base_reg = ctx.elem_base(elem_path, ty, out);
        let op = match ty {
            ValType::Float => OP_INDEX_K_F,
            _ => OP_INDEX_K,
        };
        ctx.body.extend_from_slice(&[
            op,
            off.idx as i64,
            len.idx as i64,
            k,
            base_reg as i64,
            out.idx as i64,
            OVF_FLAG_REG as i64,
        ]);
        return Ok(out);
    }

    let kr = emit_int_const(ctx, k);

    // The result register is written on the in-range path only, so give it a
    // defined value first: the row traps either way, but a register the loop
    // reads must not depend on what a previous row left behind.
    let out = ctx.fresh(ty);
    ctx.body
        .extend_from_slice(&[OP_LOAD_CONST, 0, out.idx as i64]);

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
    let ea = emit_int_bin_k(ctx, OP_MUL, idx, 8);
    let v = ctx.elem_slot(elem_path, ea.idx)?;
    emit_mov(ctx, v, out);
    ctx.patch_jump(skip);
    Ok(out)
}

/// [`lower_const_index`] for an index the ROW supplies: `list[i]` where `i` is
/// any int- or uint-bank expression rather than a literal.
///
/// One fused instruction, like the constant form: `OP_INDEX_R` carries the
/// range check, the trap, the default and the load. The expanded shape the
/// constant form falls back to for a `bool` element cannot be reused here --
/// it skips the load with a forward jump whose target the const index fixes --
/// so a non-numeric element still declines and takes the tree-walker.
fn lower_var_index(
    ctx: &mut LowerCtxF,
    list: &str,
    field: Option<&str>,
    index: &IdedExpr,
) -> Result<TReg, LowerError> {
    let elem_path = elem_slot_path(list, field);
    let ty =
        ctx.schema.get(&elem_path).copied().ok_or_else(|| {
            LowerError::unsupported(format!("undeclared element path `{elem_path}`"))
        })?;
    if !matches!(ty, ValType::Int | ValType::Float) {
        return Err(LowerError::unsupported(
            "runtime index of a non-numeric element",
        ));
    }
    // cel indexes with `int` or `uint`; anything else is NoSuchOverload in the
    // tree-walker, so it is not an answer this can give either.
    let k = compile_t(ctx, index)?;
    if !matches!(k.bank, ValType::Int | ValType::UInt) {
        return Err(LowerError::unsupported("index must be int or uint"));
    }
    let len = ctx.slot_typed(size_slot_path(list), ValType::Int);
    let off = ctx.slot_typed(offset_slot_path(list), ValType::Int);
    let out = ctx.fresh(ty);
    let base_reg = ctx.elem_base(elem_path, ty, out);
    let op = match ty {
        ValType::Float => OP_INDEX_R_F,
        _ => OP_INDEX_R,
    };
    ctx.body.extend_from_slice(&[
        op,
        off.idx as i64,
        len.idx as i64,
        k.idx as i64,
        base_reg as i64,
        out.idx as i64,
        OVF_FLAG_REG as i64,
    ]);
    Ok(out)
}

/// The mask `|k| - 1` when `k` divides exactly the values `k` masks -- i.e.
/// when `|k|` is a power of two of at least 2 -- and `None` otherwise.
///
/// `|k| == 1` is excluded, and that is the whole subtlety: `1` IS a power of
/// two, but `a % -1` overflows on `i64::MIN` and the tree-walker raises there,
/// where a mask would answer `true`. Every other power of two makes `a % k`
/// total, so the rewritten form has no trap to reproduce.
///
/// `i64::MIN` is admitted: `unsigned_abs` gives `2^63` exactly, and its mask
/// `2^63 - 1` is `i64::MAX`.
fn divisibility_mask(k: i64) -> Option<i64> {
    let m = k.unsigned_abs();
    (m >= 2 && m.is_power_of_two()).then(|| (m - 1) as i64)
}

/// `a % K == 0` -- either operand order -- as `(a & (|K| - 1)) == 0`, or `None`
/// where the shape is not that and the ordinary comparison path answers.
///
/// Recognised on the AST rather than on the emitted ops: `OP_MOD_CHK_K` has
/// already reapplied the dividend's sign by then, and the sign is exactly what
/// a zero test does not read. The dividend is compiled first because its BANK
/// decides whether the rewrite applies at all, and compiling it is the only way
/// to learn that; a pool constant is handed back to the ordinary path, which
/// folds the whole comparison instead of masking it.
fn lower_divisibility(
    ctx: &mut LowerCtxF,
    iop: i64,
    call: &CallExpr,
) -> Result<Option<TReg>, LowerError> {
    for (m, z) in [(0usize, 1usize), (1, 0)] {
        let (m, z) = (&call.args[m], &call.args[z]);
        if as_int_literal(z) != Some(0) {
            continue;
        }
        let Expr::Call(inner) = &m.expr else {
            continue;
        };
        if inner.func_name != ops::MODULO || inner.args.len() != 2 {
            continue;
        }
        let Some(mask) = as_int_literal(&inner.args[1]).and_then(divisibility_mask) else {
            continue;
        };
        let a = compile_t(ctx, &inner.args[0])?;
        if a.bank != ValType::Int || ctx.const_of(a).is_some() {
            return Ok(None);
        }
        let masked = emit_int_bin_k(ctx, OP_AND, a, mask);
        let zero = emit_int_const(ctx, 0);
        return Ok(Some(emit_bin(ctx, iop, masked, zero, ValType::Bool)));
    }
    Ok(None)
}

/// `x <name> y` for two same-bank constants, as the word the bank stores, or
/// `None` where the op would trap (or the banks are not one numeric bank).
fn fold_arith(name: &str, bank_a: ValType, bank_b: ValType, x: i64, y: i64) -> Option<i64> {
    if bank_a != bank_b {
        return None;
    }
    match bank_a {
        ValType::Int => match name {
            ops::ADD => x.checked_add(y),
            ops::SUBSTRACT => x.checked_sub(y),
            ops::MULTIPLY => x.checked_mul(y),
            ops::DIVIDE => x.checked_div(y),
            ops::MODULO => x.checked_rem(y),
            _ => None,
        },
        ValType::UInt => {
            let (x, y) = (x as u64, y as u64);
            match name {
                ops::ADD => x.checked_add(y),
                ops::SUBSTRACT => x.checked_sub(y),
                ops::MULTIPLY => x.checked_mul(y),
                ops::DIVIDE => x.checked_div(y),
                ops::MODULO => x.checked_rem(y),
                _ => None,
            }
            .map(|v| v as i64)
        }
        ValType::Float => {
            let (x, y) = (f64::from_bits(x as u64), f64::from_bits(y as u64));
            match name {
                ops::ADD => Some(x + y),
                ops::SUBSTRACT => Some(x - y),
                ops::MULTIPLY => Some(x * y),
                ops::DIVIDE => Some(x / y),
                _ => None,
            }
            .map(|v| v.to_bits() as i64)
        }
        _ => None,
    }
}

/// `x <name> y` for two same-bank constants of a numeric or bool bank, as the
/// bool word, or `None` for a bank whose word is not its value (a string is a
/// rank the batch assigns).
fn fold_cmp(name: &str, bank_a: ValType, bank_b: ValType, x: i64, y: i64) -> Option<i64> {
    use std::cmp::Ordering;
    if bank_a != bank_b {
        return None;
    }
    let ord = match bank_a {
        ValType::Int | ValType::Bool => x.cmp(&y),
        ValType::UInt => (x as u64).cmp(&(y as u64)),
        ValType::Float => f64::from_bits(x as u64).partial_cmp(&f64::from_bits(y as u64))?,
        _ => return None,
    };
    let v = match name {
        ops::GREATER_EQUALS => ord != Ordering::Less,
        ops::GREATER => ord == Ordering::Greater,
        ops::LESS_EQUALS => ord != Ordering::Greater,
        ops::LESS => ord == Ordering::Less,
        ops::EQUALS => ord == Ordering::Equal,
        ops::NOT_EQUALS => ord != Ordering::Equal,
        _ => return None,
    };
    Some(v as i64)
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
/// Emitted shape, with `L` the list and `i` the element index:
///
/// ```text
///     accu = <accu_init>
///     if 1 > size(L) goto after          ; zero-trip guard ([].all(..) is true)
///     ea = offset(L) * 8; last = offset(L) + size(L)
///     limit = last * 8                   ; or `i = offset(L)`, per the close
///   inner:
///     <element loads at ea, on first reference>
///     accu = <loop_step>; ea = ea + 8
///     if limit > ea goto inner           ; back-edge -> can_enter_jit
///   after:
///     <result>
/// ```
///
/// A body that reads a `bool` element column reads it at the element INDEX, so
/// that loop advances `i` as well and closes on `last > i` instead. Every other
/// loop has no `i` at all: the byte offset is the whole induction variable, and
/// an index nothing addresses with is four words an element that buy nothing.
///
/// The two closes want different preamble ops and exactly one each, so the
/// preamble reserves [`PREAMBLE_SLOT_WORDS`] and the close writes whichever it
/// took. A loop therefore carries no op its own close does not read.
///
/// As in the literal unroll, `loop_cond`'s short-circuit is dropped: every
/// element is evaluated. Where the walker would stop early and the eager fold
/// traps instead, the batch answers `None` and the walker owns the row.
fn compile_list_comprehension_t(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    list: &str,
) -> Result<TReg, LowerError> {
    compile_list_comprehension_mode(ctx, comp, list, AccuMode::Value, &[])
}

/// `chain` carries the fused `map`/`filter` links in `Collect` mode, where it
/// is what the body compiles instead of `comp.loop_step`. The other modes read
/// the step straight off `comp` and pass an empty chain.
fn compile_list_comprehension_mode(
    ctx: &mut LowerCtxF,
    comp: &ComprehensionExpr,
    list: &str,
    mode: AccuMode,
    chain: &[Link],
) -> Result<TReg, LowerError> {
    let len = ctx.slot_typed(size_slot_path(list), ValType::Int);
    let off = ctx.slot_typed(offset_slot_path(list), ValType::Int);
    let one = emit_int_const(ctx, 1);

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
    // A collected comprehension's accumulator is written ONCE, after the loop,
    // as the distance its cursor travelled -- so it takes no starting value
    // here. The other modes carry theirs across the back edge and do.
    let cursor_start = match mode {
        AccuMode::Collect { cursor } => {
            let start = ctx.fresh(ValType::Int);
            emit_mov(
                ctx,
                TReg {
                    bank: ValType::Int,
                    idx: cursor,
                },
                start,
            );
            Some(start)
        }
        _ => {
            emit_mov(ctx, init, accu);
            None
        }
    };
    // Zero-trip guard: an empty list must yield `accu_init`, and the back-edge
    // below is a do-while.
    let zero_trip = ctx.emit_jump_if_above(one, len);

    // Whichever addresses the body needs advance by a constant, so they are
    // carried across the back edge rather than derived inside the loop from a
    // counter: deriving the byte offset from an index costs a multiply an
    // element on top of the counter's own add.
    let ea = emit_int_bin_k(ctx, OP_MUL, off, 8);
    let last = emit_int_bin(ctx, OP_ADD, off, len);
    // The two closes need ONE preamble op each, so the words are reserved here
    // and written at the close, by which time the body has said whether it read
    // a byte column. Reserving is what keeps the choice free -- emitting both
    // ops charges every loop for the one its close does not read. It is also
    // why the choice is not made by inserting the op later: every jump the body
    // records is an absolute position into `body`, and `inner` below is one of
    // them, so nothing may shift once the body has begun.
    let idx = ctx.fresh(ValType::Int);
    let limit_ea = ctx.fresh(ValType::Int);
    let preamble_slot = ctx.body.len();
    ctx.body.extend_from_slice(&[0; PREAMBLE_SLOT_WORDS]);

    let inner = ctx.body.len();
    // The loop's element is bound to the FIRST link's variable: the chain runs
    // innermost-first, and the outer comprehension's own variable belongs to
    // its last link.
    ctx.list_loop.push(ListLoop {
        iter_var: chain
            .first()
            .map_or(comp.iter_var.as_str(), |l| l.iter_var)
            .to_string(),
        list: list.to_string(),
        ea_reg: ea.idx,
        idx_reg: idx.idx,
    });
    ctx.locals.insert(comp.accu_var.clone(), accu);
    let step = match mode {
        AccuMode::Value => compile_t(ctx, &comp.loop_step).map(Some),
        AccuMode::Length => compile_len_step(ctx, &comp.loop_step, comp, accu, mode).map(Some),
        // Collecting keeps no running total: the cursor already counts what the
        // store put in the buffer, so the accumulator is read off it once the
        // loop is over rather than carried an element at a time.
        AccuMode::Collect { cursor } => {
            emit_chain_collect(ctx, chain, list, ea.idx, cursor).map(|()| None)
        }
    };
    ctx.list_loop.pop();
    // Drop only THIS loop's element registers. Each comprehension gets its own
    // `ea` register, so keying the retirement on it leaves an enclosing loop's
    // elements — still live below — exactly where they were.
    ctx.elem_map.retain(|(_, reg), _| *reg != ea.idx);
    if let Some(step) = step? {
        if step.bank != accu.bank {
            return Err(LowerError::unsupported(
                "comprehension accumulator changes bank",
            ));
        }
        // The accumulator is where the step's value has to land, and the op that
        // produced it can write there itself. A move an element buys nothing
        // that naming the destination once does not.
        if !ctx.retarget_last_write(step, accu) {
            emit_mov(ctx, step, accu);
        }
    }
    // The index is carried only for a byte column, which reads at the element
    // index rather than at its word address. Where the body read none, the byte
    // offset is the whole induction variable and the index does not advance --
    // and the limit it closes on is the one the reserved preamble word becomes.
    let preamble_op = if ctx.elem_idx_addressed.contains(&idx.idx) {
        ctx.body.extend_from_slice(&[
            OP_ADD_IMM,
            idx.idx as i64,
            1,
            idx.idx as i64,
            OP_ADD_IMM,
            ea.idx as i64,
            8,
            ea.idx as i64,
        ]);
        ctx.emit_back_edge(last, idx, inner);
        // `idx = offset(L)`, spelled as an add of zero because the reserved
        // words are a fixed width: a three-word move would leave a word the
        // interpreter reaches and decodes as an opcode.
        [OP_ADD_IMM, off.idx as i64, 0, idx.idx as i64]
    } else {
        ctx.body
            .extend_from_slice(&[OP_ADD_IMM, ea.idx as i64, 8, ea.idx as i64]);
        ctx.emit_back_edge(limit_ea, ea, inner);
        [OP_MUL_IMM, last.idx as i64, 8, limit_ea.idx as i64]
    };
    ctx.body[preamble_slot..preamble_slot + PREAMBLE_SLOT_WORDS].copy_from_slice(&preamble_op);
    ctx.patch_jump(zero_trip);
    // Both paths reach here, so the empty list answers zero without the guard
    // having had to seed anything: the cursor did not move.
    if let (AccuMode::Collect { cursor }, Some(start)) = (mode, cursor_start) {
        ctx.body.extend_from_slice(&[
            OP_SUB,
            cursor as i64,
            start.idx as i64,
            accu.idx as i64,
        ]);
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
