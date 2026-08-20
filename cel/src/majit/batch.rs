//! Public columnar-batch API for the majit tier.
//!
//! [`super::lower`] turns a CEL expression into two-bank bytecode and
//! [`super::bytecode`] runs it, but both speak the machine's language: `i64`
//! register banks, base pointers, string ids, Arrow offset buffers. Every
//! caller that wanted the batch tier had to reimplement the same encoding, in
//! the same order the lowering happened to allocate slots in — which is how a
//! `bool` column came to be declared as an `int` and `!frozen` came to lower as
//! a bitwise complement.
//!
//! This module is that encoding, written once, in CEL's types:
//!
//! ```
//! use cel::majit::batch::{Batch, BatchProgram, ColumnRef};
//! use cel::majit::lower::{Schema, ValType};
//!
//! let schema: Schema = [
//!     ("balance".to_string(), ValType::Int),
//!     ("amount".to_string(), ValType::Int),
//!     ("frozen".to_string(), ValType::Bool),
//! ]
//! .into_iter()
//! .collect();
//! let program = BatchProgram::compile("balance >= amount && !frozen", &schema)?;
//!
//! let (balance, amount, frozen) = (vec![10i64], vec![5i64], vec![false]);
//! let batch = Batch::new(1)
//!     .column("balance", ColumnRef::Int(&balance))
//!     .column("amount", ColumnRef::Int(&amount))
//!     .column("frozen", ColumnRef::Bool(&frozen));
//! let matching_rows = program.bind(&batch)?.sum()?;
//! assert_eq!(matching_rows, cel::Value::Int(1));
//! # Ok::<(), cel::majit::batch::BatchError>(())
//! ```
//!
//! **What it computes.** The machine evaluates the expression per row. What it
//! does with each row's result is the caller's choice, made at bind:
//!
//! * [`BatchProgram::bind`] reduces with a running total, so the answer is
//!   `sum over rows of expr(row)` — for a boolean predicate, the number of
//!   matching rows. The reduction is the DRIVER's, not a CEL `+`: it wraps at
//!   64 bits and sums floats left to right in row order (bit-exact with the
//!   same rows summed through the tree-walker, since float addition does not
//!   reassociate). A total is only meaningful for a numeric or boolean result,
//!   so this door refuses the rest.
//! * [`BatchProgram::bind_per_row`] stores each row's result and
//!   [`BoundBatch::collect`] reads them back as [`Value`]s. A store takes a
//!   result of any type, so this is the compiled path for the `string`-,
//!   `timestamp`- and `duration`-valued expressions a sum has nothing to do
//!   with — and the one that keeps each row's own answer rather than a figure
//!   they all collapse into. [`BoundBatch::collect_into`] is the same read into
//!   a buffer the caller owns, for one that evaluates repeatedly and would
//!   rather keep its output vector than allocate one per run.
//!
//! Both compile the same loop over the same red-index column reads; only the
//! last instruction of an iteration differs.
//!
//! **What it refuses.** Everything outside the traceable subset declines at
//! [`BatchProgram::compile`], and a batch whose data would make the tree-walker
//! raise (an `int` overflow, a division by zero) refuses at the run.
//! Both are the signal to evaluate that expression with
//! [`crate::Program::execute`], which owns the error.

use std::collections::HashMap;
use std::sync::Arc;

use super::bytecode::{float_bank, prepare_batch_reduce, BatchRun, Column};
use super::lower::{
    concat_slot_index, concat_slot_path, elem_slot_source, lower_typed, offset_slot_source,
    size_slot_source, string_slot_source, BatchReduce, ConcatSide, LoweredF, Schema, SlotKind,
    ValType,
};
use crate::objects::{Key, ListRef, ListStorage, RecordSchema, ScalarBank, StrBank, ValueColumn};
use crate::{Context, Program, Value};

/// Trace threshold [`Tier::Jit`] runs at: the batch loop compiles after this
/// many iterations. Matches the threshold the benchmarks and tests use.
pub const DEFAULT_JIT_THRESHOLD: u32 = 8;

/// Whether the compiled tier is Cranelift's rather than dynasm's — the one
/// thing the three routing constants below have to be told, because the two
/// backends do not compile the batch loop into the same code and do not cross
/// the interpreter in the same place.
///
/// One set of numbers for both was tried and rejected by measurement:
/// `routeprobe`'s sixteen shapes cross at a median of ~750 body words under
/// dynasm and ~535 under Cranelift, and grading dynasm's constants against
/// Cranelift's own sweep named the losing tier on 13 of 240 points where the
/// 460-word threshold they replaced named it on 9. Per-backend, each names it
/// on 5 to 7.
///
/// Both features on at once is not a configuration to measure from — the
/// manifest says so where they are declared, and `majit-metainterp` picks —
/// so Cranelift wins the tie here for the same reason.
const CRANELIFT: bool = cfg!(feature = "jit-cranelift");

/// Picoseconds the compiled tier saves for each body WORD the plain
/// interpreter would have dispatched. See [`JIT_ENTRY_PS`] for the rule the
/// three constants feed and for how all three were measured.
pub const GAIN_PER_WORD_PS: i64 = if CRANELIFT { 143 } else { 137 };

/// Picoseconds the compiled tier saves per ITERATION — per row of the batch,
/// and per element a comprehension's inner loop visits — beyond what that
/// iteration's words account for.
///
/// This term is why one word threshold could not be right. Both tiers charge
/// per iteration as well as per word, and the interpreter's per-iteration cost
/// is the larger, so every iteration hands the compiled tier a saving that has
/// nothing to do with how long the iteration is. A body of few words per
/// iteration therefore breaks even at FEWER total words than a dense one, and a
/// rule stated in words alone has nowhere to put that: `routeprobe` measures a
/// 25-word row body crossing at 535 body words and a 59-word one at 1049, on
/// the same box in the same run.
pub const GAIN_PER_ITERATION_PS: i64 = if CRANELIFT { 2_888 } else { 2_622 };

/// Picoseconds a call must expect to SAVE before [`Tier::Auto`] hands it to the
/// compiled tier — the fixed cost of getting there: the driver lookup, the
/// program-table insert, the state republish, the per-call buffers. The plain
/// interpreter pays none of it and instead spends per word and per iteration,
/// so the two tiers cross where the run's accumulated saving reaches this
/// number.
///
/// # The rule
///
/// [`compiled_saving_ps`] estimates a run's saving from the two counts a bind
/// already has:
///
/// ```text
///   saving = body_words * GAIN_PER_WORD_PS
///          + (rows + elems) * GAIN_PER_ITERATION_PS
/// ```
///
/// and [`BoundBatch::route`] compares it against this constant. Two terms
/// rather than one count of words, because the crossing is not at a fixed word
/// count: it moves with the shape, and `tierprobe` measured it moving over
/// 607..860 words across four shapes in one run. The per-iteration term is what
/// carries that movement.
///
/// # How the three were measured
///
/// By `routeprobe`, over sixteen shapes — eight straight-line bodies swept by
/// batch height, eight comprehensions swept by element count, 25 to 79 words
/// per iteration. For each it finds the count at which the two tiers change
/// hands, by interpolating between the two swept points that BRACKET the
/// crossing, and regresses `1 / crossing` against the shape's words per
/// iteration. That line's slope and intercept are these two rates divided by
/// the entry, so the regression pins them as SHARES of it; the entry itself is
/// the median over the shapes of `jit_fix - clean_fix`, both intercepts read
/// from the smallest batches swept. Points where the probe could not evidence
/// that the `Tier::Jit` cell entered compiled code are dropped before any of
/// it, since such a point times the tracing interpreter rather than either tier.
///
/// Row words and element words were fitted separately as well as pooled, and
/// pooled is what shipped: over four runs the two estimates of each rate
/// overlapped, so a rule with one pair of constants per axis would have been
/// fitting the noise between them. `routeprobe` still prints both fits, and
/// their separating is the signal to revisit this.
///
/// The shipped numbers are the MEDIAN of several such runs, taken on a box
/// under other load, which is what the spread is for. Four runs under dynasm:
/// per-word share 0.00073..0.00107 of an entry, per-iteration share
/// 0.0074..0.0378, entry 134..150 ns. Three under Cranelift: 0.00119..0.00172,
/// 0.0021..0.0424, 106..110 ns. The rule is a ranking of two tiers rather than
/// a prediction of either, so it tolerates that spread; what would not tolerate
/// it is reading any one of the three as a cost.
///
/// # What invalidates them
///
/// They are times, so anything that changes what a tier costs: a cheaper or
/// dearer path into compiled code (the entry), a change to the interpreter's
/// dispatch or to what the compiled loop emits per iteration (the two rates).
/// They are also per-PROFILE, which [`CRANELIFT`] does not cover and nothing
/// here does: these were measured under `--release`, the build the scoreboard is
/// read in. Do not assume they carry across profiles. The constant they replaced
/// was 460 body words, measured under `--profile bench` with `jit-cranelift`,
/// and re-running its own probe (`tierprobe`) under `--release` with
/// `jit-dynasm` puts the same four crossings at 607..860 rather than the
/// 359..523 it was set from. What does NOT invalidate them is the machine being
/// uniformly faster or slower, since the decision depends only on the ratios
/// among the three.
pub const JIT_ENTRY_PS: i64 = if CRANELIFT { 107_700 } else { 142_550 };

/// Picoseconds the compiled tier is expected to save on a run of `lowered` over
/// `rows` rows carrying `elems` flattened list elements — the left-hand side of
/// the rule documented on [`JIT_ENTRY_PS`].
///
/// A time, unlike [`LoweredF::body_words_for`]'s count, and therefore only ever
/// an estimate: it is a model of two tiers fitted over sixteen shapes, not a
/// measurement of this one. That is all a route needs, and it is why the answer
/// is spent on a comparison rather than reported.
///
/// Saturating throughout because `rows` and `elems` are the caller's numbers: a
/// batch that claims more of either than a saving can be counted in should route
/// to the compiled tier, which is what a saturated positive does.
pub fn compiled_saving_ps(lowered: &LoweredF, rows: usize, elems: usize) -> i64 {
    let count = |n: usize| i64::try_from(n).unwrap_or(i64::MAX);
    count(lowered.body_words_for(rows, elems))
        .saturating_mul(GAIN_PER_WORD_PS)
        .saturating_add(
            count(rows)
                .saturating_add(count(elems))
                .saturating_mul(GAIN_PER_ITERATION_PS),
        )
}

/// Why a batch could not be answered. Every variant means the same thing to a
/// caller — evaluate this expression with [`crate::Program::execute`] instead —
/// but they are distinguished because they say different things about the
/// expression: [`BatchError::Lower`] is permanent for this expression, the rest
/// depend on the data.
#[derive(Debug, Clone)]
pub enum BatchError {
    /// The source is not a valid CEL expression.
    Parse(String),
    /// The expression is outside the traceable subset.
    Lower(super::lower::LowerError),
    /// The expression reads a column the batch does not carry.
    MissingColumn(String),
    /// A column's declared type is not the one the expression's schema declared.
    ColumnType {
        /// The column's name.
        name: String,
        /// What the schema said it was.
        declared: ValType,
    },
    /// A column's length disagrees with the batch's row count.
    RowCount {
        /// The column's name.
        name: String,
        /// Its length.
        len: usize,
        /// The batch's row count.
        rows: usize,
    },
    /// A `timestamp` or `duration` column carries a value outside the range in
    /// which this expression's arithmetic agrees with the tree-walker's, whose
    /// chrono range is wider than i64 nanoseconds. Data-dependent: another
    /// batch of the same expression may be fine.
    TemporalOutOfDomain {
        /// The column's name.
        name: String,
        /// The offending value, in nanoseconds.
        value: i64,
        /// The largest magnitude this expression's arithmetic can take.
        bound: i64,
    },
    /// A row's arithmetic trapped — an `int` overflow or a division by zero,
    /// where the tree-walker raises. No sum is the right answer.
    Trapped,
    /// The tree-walker could not evaluate a row either, on the fallback path.
    /// A real CEL error, not a limit of the batch model — the expression has no
    /// value for this row however it is run.
    Row {
        /// Which row.
        row: usize,
        /// What the walker said.
        message: String,
    },
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::Parse(e) => write!(f, "parse error: {e}"),
            BatchError::Lower(e) => write!(f, "{e}"),
            BatchError::MissingColumn(n) => write!(f, "batch has no column `{n}`"),
            BatchError::ColumnType { name, declared } => {
                write!(f, "column `{name}` is not the declared {declared:?}")
            }
            BatchError::RowCount { name, len, rows } => {
                write!(f, "column `{name}` has {len} rows, batch has {rows}")
            }
            BatchError::TemporalOutOfDomain { name, value, bound } => write!(
                f,
                "column `{name}` holds {value}ns, outside the ±{bound}ns range \
                 this expression's temporal arithmetic is exact in"
            ),
            BatchError::Trapped => write!(f, "a row trapped (overflow or division by zero)"),
            BatchError::Row { row, message } => write!(f, "row {row}: {message}"),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<super::lower::LowerError> for BatchError {
    fn from(e: super::lower::LowerError) -> Self {
        BatchError::Lower(e)
    }
}

/// One input column, borrowed, in CEL's types rather than the machine's. The
/// variants are exactly the [`ValType`]s a slot can have, plus [`ColumnRef::List`]
/// for a list column a comprehension iterates.
///
/// `Timestamp` and `Duration` carry `i64` nanoseconds — the representation the
/// machine compares on, and the one a columnar store already holds them in.
pub enum ColumnRef<'a> {
    /// An `int` column.
    Int(&'a [i64]),
    /// A `bool` column.
    Bool(&'a [bool]),
    /// A `uint` column.
    UInt(&'a [u64]),
    /// A `double` column.
    Float(&'a [f64]),
    /// A `string` column. Encoded to order-preserving `i64` ranks over the
    /// batch's distinct strings, so equality AND ordering are content
    /// comparisons (see [`ValType::Str`]).
    Str(&'a [String]),
    /// A `timestamp` column, as nanoseconds since the Unix epoch.
    Timestamp(&'a [i64]),
    /// A `duration` column, as nanoseconds.
    Duration(&'a [i64]),
    /// A list column, laid out the way Arrow lays one out: a per-row element
    /// count plus one flattened buffer per field, every buffer as long as the
    /// batch's total element count. `None` names the elements themselves (a
    /// list of scalars, schema path `list[]`); `Some(f)` names one record field
    /// (`list[].f`).
    List {
        /// Per-row element count.
        lens: &'a [i64],
        /// The flattened element buffers.
        fields: Vec<(Option<&'a str>, ColumnRef<'a>)>,
    },
}

impl ColumnRef<'_> {
    /// The [`ValType`] a schema must declare for this column, or `None` for a
    /// list (whose ELEMENTS are what the schema declares, under `list[]`).
    fn val_type(&self) -> Option<ValType> {
        Some(match self {
            ColumnRef::Int(_) => ValType::Int,
            ColumnRef::Bool(_) => ValType::Bool,
            ColumnRef::UInt(_) => ValType::UInt,
            ColumnRef::Float(_) => ValType::Float,
            ColumnRef::Str(_) => ValType::Str,
            ColumnRef::Timestamp(_) => ValType::Timestamp,
            ColumnRef::Duration(_) => ValType::Duration,
            ColumnRef::List { .. } => return None,
        })
    }
}

/// A batch of rows: named columns plus the row count they share.
///
/// The row count is the caller's to state rather than read off a column,
/// because a lowering need not have a row column to read it from — a list's
/// flattened element buffer is as long as the total element count, and an
/// expression over literals has no column at all.
#[derive(Default)]
pub struct Batch<'a> {
    columns: HashMap<String, ColumnRef<'a>>,
    rows: usize,
}

impl<'a> Batch<'a> {
    /// An empty batch of `rows` rows.
    pub fn new(rows: usize) -> Self {
        Batch {
            columns: HashMap::new(),
            rows,
        }
    }

    /// Add a column, consuming and returning the batch so calls chain.
    pub fn column(mut self, name: impl Into<String>, col: ColumnRef<'a>) -> Self {
        self.columns.insert(name.into(), col);
        self
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }
}

/// Which tier evaluates a bound batch. All three compute the same answer; they
/// differ in what runs the loop, which is what the benchmarks and the
/// cross-tier tests need to select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tier {
    /// Let the bound batch choose, by how much work its run has to do — see
    /// [`BoundBatch::route`]. This is what the no-suffix doors
    /// ([`BoundBatch::sum`], [`BoundBatch::collect`],
    /// [`BoundBatch::collect_raw`]) ask for; every `_on` door still names a
    /// tier outright and gets exactly that one.
    #[default]
    Auto,
    /// The meta-tracing tier: traces the batch loop and compiles it.
    Jit,
    /// The meta-tracing interpreter with compilation disabled — the tracing
    /// machinery runs but never produces machine code.
    Interpreter,
    /// A plain `match` interpreter over the same bytecode, with no tracing
    /// machinery at all. The oracle a majit answer is checked against.
    Clean,
}

/// A CEL expression lowered to the traceable subset, ready to be bound to data.
///
/// Compiling is per EXPRESSION and binding is per BATCH: the lowering, and the
/// compiled loop the driver holds for it, are reused across every batch of the
/// same shape. That reuse is the whole point — see `CONVERGENCE.md`.
pub struct BatchProgram {
    lowered: LoweredF,
}

impl BatchProgram {
    /// Lower `source` against `schema`, which must declare every path the
    /// expression reads. Returns [`BatchError::Lower`] if the expression is
    /// outside the traceable subset.
    pub fn compile(source: &str, schema: &Schema) -> Result<Self, BatchError> {
        let program = Program::compile(source).map_err(|e| BatchError::Parse(e.to_string()))?;
        Self::from_program(&program, schema)
    }

    /// [`BatchProgram::compile`] for an already-parsed program, so a caller that
    /// keeps a [`Program`] for the tree-walker does not parse twice.
    pub fn from_program(program: &Program, schema: &Schema) -> Result<Self, BatchError> {
        // Whether the expression LOWERS is asked here. Whether a given
        // REDUCTION can consume its result is a separate question, asked at
        // bind: `sum` refuses a string or a timestamp, `bind_per_row` takes any
        // bank, and one program can be bound either way.
        let lowered = lower_typed(program.expression(), schema)?;
        Ok(BatchProgram { lowered })
    }

    /// The lowering, for callers that drive the machine directly.
    pub fn lowered(&self) -> &LoweredF {
        &self.lowered
    }

    /// The type the sum will have, decided by the expression, not the data.
    pub fn result_type(&self) -> ValType {
        self.lowered.result_bank
    }

    /// Encode `batch` into the machine's columns, in slot order, materializing
    /// the buffers the schema does not declare: a `size(...)` length column, a
    /// list's `offset(...)` prefix sums. String columns pass through as
    /// strings; `prepare_batch` ranks them, since the ids span the batch.
    ///
    /// This is the per-batch work. Keep the [`BoundBatch`] and call
    /// [`BoundBatch::sum`] on it rather than rebinding to run again.
    pub fn bind<'a, 'b>(&'b self, batch: &'a Batch<'a>) -> Result<BoundBatch<'a, 'b>, BatchError> {
        // Only a sum has a result the loop cannot consume, so only this door
        // asks.
        self.lowered.sum_reducible()?;
        self.bind_reduce(batch, BatchReduce::Sum)
    }

    /// [`BatchProgram::bind`] for a per-row result: the loop stores `expr(row)`
    /// to `out[i]` instead of accumulating, and [`BoundBatch::collect`] reads
    /// them back as CEL values.
    ///
    /// Takes a result of ANY type, since a store does not have to add: this is
    /// the accelerated path for the `string`-, `timestamp`- and
    /// `duration`-valued expressions a sum has nothing to do with.
    pub fn bind_per_row<'a, 'b>(
        &'b self,
        batch: &'a Batch<'a>,
    ) -> Result<BoundBatch<'a, 'b>, BatchError> {
        self.bind_reduce(batch, BatchReduce::PerRow)
    }

    fn bind_reduce<'a, 'b>(
        &'b self,
        batch: &'a Batch<'a>,
        reduce: BatchReduce,
    ) -> Result<BoundBatch<'a, 'b>, BatchError> {
        // Derived buffers are owned by the BoundBatch; `plan` records, per slot,
        // either a borrowed column or an index into them. Building the plan
        // first keeps `derived` from reallocating under a borrow.
        let mut derived: Vec<DerivedColumn> = Vec::new();
        let mut plan: Vec<Plan<'a>> = Vec::new();
        for slot in &self.lowered.slots {
            plan.push(self.plan_slot(batch, slot.path.as_str(), slot.ty, &mut derived)?);
        }

        // Row columns must be as long as the batch says. An ELEMENT column is
        // as long as the flattened element count instead, so it is exempt.
        for (slot, p) in self.lowered.slots.iter().zip(&plan) {
            if slot.kind != SlotKind::Row {
                continue;
            }
            let len = match p {
                Plan::Borrowed(c) => c.len(),
                Plan::Derived(k) => derived[*k].len(),
            };
            if len != batch.rows {
                return Err(BatchError::RowCount {
                    name: slot.path.clone(),
                    len,
                    rows: batch.rows,
                });
            }
        }

        // Build the batch program once, from the caller's buffers and the ones
        // the encoding materialized.
        let columns: Vec<Column<'a>> = plan
            .iter()
            .map(|p| match p {
                Plan::Borrowed(c) => *c,
                // SAFETY: `derived` moves into the `BoundBatch` returned below,
                // which owns the program these pointers are baked into, so the
                // boxed buffers outlive every run made through it.
                Plan::Derived(k) => unsafe { derived[*k].column() },
            })
            .collect();
        // Temporal arithmetic agrees with chrono only inside the domain the
        // lowering recorded; outside it this batch has no answer, though
        // another batch of the same expression may.
        if let Some((k, value)) = self.lowered.temporal_out_of_domain(&columns) {
            return Err(BatchError::TemporalOutOfDomain {
                name: self.lowered.slots[k].path.clone(),
                value,
                bound: self
                    .lowered
                    .temporal_bound
                    .expect("a bound found the value"),
            });
        }
        let run = prepare_batch_reduce(
            &self.lowered,
            &columns,
            batch.rows,
            "BatchProgram::bind",
            reduce,
        );
        // An ELEMENT column is as long as the batch's flattened element count,
        // which is what a comprehension's inner loop iterates over. The longest
        // one bounds every loop in the body.
        let elems = self
            .lowered
            .slots
            .iter()
            .zip(&columns)
            .filter(|(slot, _)| slot.kind != SlotKind::Row)
            .map(|(_, c)| c.len())
            .max()
            .unwrap_or(0);
        Ok(BoundBatch {
            program: self,
            reduce,
            body_words: self.lowered.body_words_for(batch.rows, elems),
            compiled_saving_ps: compiled_saving_ps(&self.lowered, batch.rows, elems),
            projected: reduce == BatchReduce::PerRow && self.lowered.is_row_projection(),
            run: std::cell::RefCell::new(run),
            _derived: derived,
        })
    }

    /// The strings a string-valued SLOT PATH stands for: a declared `string`
    /// column, a `string(x)` conversion of some other column, or a `concat#k`.
    ///
    /// One resolver for all three is what lets them nest — `size(a + string(i))`
    /// is a length column over a concatenation over a conversion. Recursion
    /// terminates because a `concat#k` only ever references a LOWER index.
    fn strings_for(
        &self,
        batch: &Batch,
        path: &str,
        rows: usize,
    ) -> Result<Vec<String>, BatchError> {
        let wrong_type = |name: &str| BatchError::ColumnType {
            name: name.to_string(),
            declared: ValType::Str,
        };
        if let Some(k) = concat_slot_index(path) {
            let spec = &self.lowered.concats[k];
            let side = |s: &ConcatSide| -> Result<Vec<String>, BatchError> {
                Ok(match s {
                    ConcatSide::Literal(text) => vec![text.clone(); rows],
                    ConcatSide::Derived(j) => {
                        self.strings_for(batch, &concat_slot_path(*j), rows)?
                    }
                    ConcatSide::Column(p) => self.strings_for(batch, p, rows)?,
                })
            };
            let (l, r) = (side(&spec.left)?, side(&spec.right)?);
            return Ok(l.into_iter().zip(r).map(|(a, b)| a + &b).collect());
        }
        if let Some(src) = string_slot_source(path) {
            return column_to_strings(lookup(batch, src)?).ok_or_else(|| wrong_type(src));
        }
        match lookup(batch, path)? {
            ColumnRef::Str(c) if c.len() == rows => Ok(c.to_vec()),
            ColumnRef::Str(c) => Err(BatchError::RowCount {
                name: path.to_string(),
                len: c.len(),
                rows,
            }),
            _ => Err(wrong_type(path)),
        }
    }

    /// Resolve one slot path to the column that feeds it.
    fn plan_slot<'a>(
        &self,
        batch: &'a Batch<'a>,
        path: &str,
        ty: ValType,
        derived: &mut Vec<DerivedColumn>,
    ) -> Result<Plan<'a>, BatchError> {
        // `size(x)`: the element count of a list, or the byte length of a
        // string — which may itself be a derived one.
        if let Some(src) = size_slot_source(path) {
            if let Ok(ColumnRef::List { lens, .. }) = lookup(batch, src) {
                derived.push(DerivedColumn::int(lens.to_vec()));
                return Ok(Plan::Derived(derived.len() - 1));
            }
            // A list ELEMENT's byte length: one entry per flattened element, so
            // it is read at the same address the element itself is.
            if let Some((list, field)) = elem_slot_source(src) {
                let ColumnRef::List { fields, .. } = lookup(batch, list)? else {
                    return Err(BatchError::MissingColumn(list.to_string()));
                };
                let Some(ColumnRef::Str(c)) =
                    fields.iter().find(|(f, _)| *f == field).map(|(_, c)| c)
                else {
                    return Err(BatchError::ColumnType {
                        name: src.to_string(),
                        declared: ValType::Str,
                    });
                };
                derived.push(DerivedColumn::int(
                    c.iter().map(|s| s.len() as i64).collect(),
                ));
                return Ok(Plan::Derived(derived.len() - 1));
            }
            let buf = self
                .strings_for(batch, src, batch.rows)?
                .iter()
                .map(|s| s.len() as i64)
                .collect();
            derived.push(DerivedColumn::int(buf));
            return Ok(Plan::Derived(derived.len() - 1));
        }
        // The two string-producing derived columns, which need the characters.
        if string_slot_source(path).is_some() || concat_slot_index(path).is_some() {
            let buf = self.strings_for(batch, path, batch.rows)?;
            derived.push(DerivedColumn::str(buf));
            return Ok(Plan::Derived(derived.len() - 1));
        }
        // `offset(x)`: exclusive prefix sums of a list's element counts.
        if let Some(src) = offset_slot_source(path) {
            let ColumnRef::List { lens, .. } = lookup(batch, src)? else {
                return Err(BatchError::MissingColumn(src.to_string()));
            };
            let mut acc = 0i64;
            let buf = lens
                .iter()
                .map(|&l| {
                    let o = acc;
                    acc += l;
                    o
                })
                .collect();
            derived.push(DerivedColumn::int(buf));
            return Ok(Plan::Derived(derived.len() - 1));
        }
        // `list[]` / `list[].field`: one of a list column's flattened buffers.
        if let Some((list, field)) = elem_slot_source(path) {
            let ColumnRef::List { fields, .. } = lookup(batch, list)? else {
                return Err(BatchError::MissingColumn(list.to_string()));
            };
            let col = fields
                .iter()
                .find(|(f, _)| *f == field)
                .map(|(_, c)| c)
                .ok_or_else(|| BatchError::MissingColumn(path.to_string()))?;
            return encode(col, ty, path);
        }
        encode(lookup(batch, path)?, ty, path)
    }
}

/// `string(col)` per row, or `None` for a column CEL's `string` has no overload
/// for.
///
/// Each arm is the conversion `common/types/string.rs` applies — plain Rust
/// formatting for the numerics, RFC 3339 for a timestamp, CEL's own duration
/// spelling for a duration — so doing it per row here rather than per row there
/// cannot change an answer. A `bool` column is absent on purpose: the walker's
/// match has no `Bool` arm and raises.
fn column_to_strings(col: &ColumnRef) -> Option<Vec<String>> {
    Some(match col {
        ColumnRef::Int(c) => c.iter().map(|v| v.to_string()).collect(),
        ColumnRef::UInt(c) => c.iter().map(|v| v.to_string()).collect(),
        ColumnRef::Float(c) => c.iter().map(|v| v.to_string()).collect(),
        ColumnRef::Str(c) => c.to_vec(),
        ColumnRef::Timestamp(c) => c
            .iter()
            .map(|&n| {
                chrono::DateTime::from_timestamp_nanos(n)
                    .fixed_offset()
                    .to_rfc3339()
            })
            .collect(),
        ColumnRef::Duration(c) => c
            .iter()
            .map(|&n| crate::duration::format_duration(&chrono::Duration::nanoseconds(n)))
            .collect(),
        ColumnRef::Bool(_) | ColumnRef::List { .. } => return None,
    })
}

/// A column the encoding had to materialize because the caller's data is not
/// already in the machine's representation. Boxed, so the buffer's address is
/// fixed the moment it is built and moving the owning `Vec` cannot move it.
enum DerivedColumn {
    Int(Box<[i64]>),
    /// A `string(...)` conversion column. Held as strings, not ids: the ids are
    /// ranks over the whole batch, so `prepare_batch` assigns them alongside
    /// the caller's own string columns.
    Str(Box<[String]>),
}

impl DerivedColumn {
    fn int(v: Vec<i64>) -> Self {
        DerivedColumn::Int(v.into_boxed_slice())
    }

    fn str(v: Vec<String>) -> Self {
        DerivedColumn::Str(v.into_boxed_slice())
    }

    fn len(&self) -> usize {
        match self {
            DerivedColumn::Int(c) => c.len(),
            DerivedColumn::Str(c) => c.len(),
        }
    }

    /// The buffer as a [`Column`] valid for `'a`.
    ///
    /// # Safety
    /// The caller must keep this `DerivedColumn` alive, and unmoved-from, for
    /// all of `'a`. [`BatchProgram::bind`] does: the whole `Vec<DerivedColumn>`
    /// moves into the same [`BoundBatch`] that holds the program built from
    /// these pointers, and a boxed slice's buffer does not move with it.
    unsafe fn column<'a>(&self) -> Column<'a> {
        match self {
            DerivedColumn::Int(c) => Column::Int(unsafe { &*(&**c as *const [i64]) }),
            DerivedColumn::Str(c) => Column::Str(unsafe { &*(&**c as *const [String]) }),
        }
    }
}

/// Where one slot's data comes from: straight out of the caller's buffer, or
/// out of a buffer the encoding built.
enum Plan<'a> {
    Borrowed(Column<'a>),
    Derived(usize),
}

/// Hand one caller column to the bank its slot reads, checking that the column
/// is the type the schema declared.
///
/// Every declared type is now readable where the caller keeps it, so this only
/// borrows — it materializes nothing and can fail only on a type mismatch.
fn encode<'a>(col: &'a ColumnRef<'a>, ty: ValType, path: &str) -> Result<Plan<'a>, BatchError> {
    if col.val_type() != Some(ty) {
        return Err(BatchError::ColumnType {
            name: path.to_string(),
            declared: ty,
        });
    }
    // `int`, `timestamp` and `duration` are already `i64` in the machine's
    // representation and are read straight out of the caller's buffer; so is
    // `uint`, whose raw bit pattern is what the int file carries. `bool` is read
    // where the caller keeps it, ONE BYTE per row, by a load whose descr says
    // one. Only `string` is not readable in place.
    match col {
        ColumnRef::Int(c) | ColumnRef::Timestamp(c) | ColumnRef::Duration(c) => {
            Ok(Plan::Borrowed(Column::Int(c)))
        }
        ColumnRef::Float(c) => Ok(Plan::Borrowed(Column::Float(c))),
        ColumnRef::UInt(c) => {
            // SAFETY: `u64` and `i64` have the same size and alignment and every
            // bit pattern is valid for both, and the int register file carries a
            // `uint` as exactly that bit pattern (see `ValType::UInt`), so the
            // column is reinterpreted rather than copied.
            let bits = unsafe { core::slice::from_raw_parts(c.as_ptr().cast::<i64>(), c.len()) };
            Ok(Plan::Borrowed(Column::Int(bits)))
        }
        ColumnRef::Bool(c) => Ok(Plan::Borrowed(Column::Bool(c))),
        // Strings go to `prepare_batch` as strings: the ids are ranks over the
        // whole batch, which one column cannot compute on its own.
        ColumnRef::Str(c) => Ok(Plan::Borrowed(Column::Str(c))),
        ColumnRef::List { .. } => Err(BatchError::MissingColumn(path.to_string())),
    }
}

fn lookup<'a>(batch: &'a Batch<'a>, name: &str) -> Result<&'a ColumnRef<'a>, BatchError> {
    batch
        .columns
        .get(name)
        .ok_or_else(|| BatchError::MissingColumn(name.to_string()))
}

/// A [`BatchProgram`] bound to one batch of data: run it as many times as you
/// like, on whichever [`Tier`].
///
/// Not `Send`: the trace driver is thread-local (`float_bank::DRIVERS`), so a
/// bound batch belongs to the thread that bound it. The program words are no
/// longer part of that reason — the [`super::lower::LoweredF`] owns them and
/// hands out an `Arc` — but the conclusion is unchanged, because `DRIVERS`
/// alone establishes it.
pub struct BoundBatch<'a, 'b> {
    program: &'b BatchProgram,
    /// Which reduction the program was prepared with. A run answers through the
    /// matching door only: a `Sum` batch has no output buffer to collect from,
    /// and a `PerRow` one returns its row count rather than a total.
    reduce: BatchReduce,
    /// The prepared program. `RefCell` because running writes the trap word,
    /// while `sum` takes `&self` so a caller can hold the batch across runs.
    run: std::cell::RefCell<BatchRun<'a>>,
    /// Body words one run over this batch executes, from
    /// [`LoweredF::body_words_for`] with the batch's own row and element
    /// counts. Counted at bind, where both are known.
    ///
    /// A SIZE, and no longer what the route compares — [`Tier::Auto`] asks
    /// [`compiled_saving_ps`] instead, which spends the same two counts through
    /// two rates rather than one. Kept because it is what a caller reading the
    /// scoreboard's `words` column is reading, and because the size is the one
    /// number here that is the same on every tier.
    body_words: usize,
    /// Picoseconds the compiled tier is expected to save on one run over this
    /// batch, from [`compiled_saving_ps`] with the batch's own row and element
    /// counts. Evaluated at bind, where both are known, so [`Tier::Auto`] costs
    /// one comparison per call rather than two multiplies and a walk.
    compiled_saving_ps: i64,
    /// Whether this run is a projection the clean tier answers as a column
    /// copy. Decided at bind, where the reduction and the program are both
    /// known, so [`BoundBatch::route`] stays a comparison.
    projected: bool,
    /// The columns the encoding materialized. Never read again — the program's
    /// base pointers address their buffers — but they must outlive the runs.
    _derived: Vec<DerivedColumn>,
}

impl BoundBatch<'_, '_> {
    /// Body words one run over this batch executes — a size, the same count on
    /// every tier, which is what makes it comparable across them.
    pub fn body_words(&self) -> usize {
        self.body_words
    }

    /// Picoseconds the compiled tier is expected to save on one run over this
    /// batch — what [`BoundBatch::route`] compares against [`JIT_ENTRY_PS`].
    pub fn compiled_saving_ps(&self) -> i64 {
        self.compiled_saving_ps
    }

    /// Which tier [`Tier::Auto`] resolves to for this batch: [`Tier::Jit`] once
    /// the run is expected to save at least [`JIT_ENTRY_PS`] by being compiled,
    /// and [`Tier::Clean`] below that. Any other tier is returned unchanged.
    ///
    /// The compiled tier buys a cheaper body and charges a fixed cost to reach
    /// it, so which one wins is a question about how much the body is worth. A
    /// batch answers it with two numbers it already has — how many rows, and how
    /// many list elements those rows carry — spent at bind through the rates on
    /// [`JIT_ENTRY_PS`], which leaves this a comparison.
    pub fn route(&self, tier: Tier) -> Tier {
        match tier {
            // A projection is the one shape the estimate cannot speak for: its
            // clean tier copies the column rather than running the loop it was
            // fitted on, so more rows buy the compiled tier nothing to be
            // cheaper at. Measured on a 10 000-row `x`, the clean tier answers
            // in 643 ns and the compiled one in 10 965 ns — the estimate below
            // would route this to the slower tier by a factor of seventeen, and
            // it grows with the batch.
            Tier::Auto if self.projected => Tier::Clean,
            Tier::Auto if self.compiled_saving_ps >= JIT_ENTRY_PS => Tier::Jit,
            Tier::Auto => Tier::Clean,
            explicit => explicit,
        }
    }

    /// Evaluate every row and return the running total, on the tier
    /// [`BoundBatch::route`] picks.
    pub fn sum(&self) -> Result<Value, BatchError> {
        self.sum_on(Tier::Auto)
    }

    /// [`BoundBatch::sum`] on a chosen tier.
    pub fn sum_on(&self, tier: Tier) -> Result<Value, BatchError> {
        let tier = self.route(tier);
        self.sum_with(tier, threshold_for(tier))
    }

    /// [`BoundBatch::sum_on`] with an explicit trace threshold, for a caller
    /// measuring where the compiled tier starts to pay for itself.
    pub fn sum_with(&self, tier: Tier, threshold: u32) -> Result<Value, BatchError> {
        let tier = self.route(tier);
        let lowered = &self.program.lowered;
        assert_eq!(
            self.reduce,
            BatchReduce::Sum,
            "sum on a batch bound per-row: use `collect`"
        );
        let raw = self.execute(tier, threshold).ok_or(BatchError::Trapped)?;
        // The accumulator's bank is the expression's result bank: an int-bank
        // result sums in the int accumulator (a `uint` as its raw bit pattern, a
        // `bool` predicate as a count of matching rows), a float result in the
        // float accumulator, whose bits come back here.
        Ok(match lowered.result_bank {
            ValType::Float => Value::Float(f64::from_bits(raw as u64)),
            ValType::UInt => Value::UInt(raw as u64),
            _ => Value::Int(raw),
        })
    }

    /// Evaluate every row and return each row's own value, on [`Tier::Jit`].
    ///
    /// Requires the batch to have been bound with
    /// [`BatchProgram::bind_per_row`].
    pub fn collect(&self) -> Result<Vec<Value>, BatchError> {
        self.collect_on(Tier::Auto)
    }

    /// [`BoundBatch::collect`] on a chosen tier.
    pub fn collect_on(&self, tier: Tier) -> Result<Vec<Value>, BatchError> {
        let tier = self.route(tier);
        self.collect_with(tier, threshold_for(tier))
    }

    /// [`BoundBatch::collect_on`] with an explicit trace threshold.
    pub fn collect_with(&self, tier: Tier, threshold: u32) -> Result<Vec<Value>, BatchError> {
        self.collect_raw_with(tier, threshold, |out| out.to_values())
    }

    /// [`BoundBatch::collect`] writing into a buffer the CALLER owns, so a
    /// caller evaluating the same batch repeatedly can hand the same buffer
    /// back and stop allocating an output vector per call.
    ///
    /// Contract: on success `out` holds exactly what the matching
    /// [`BoundBatch::collect`] would have returned — the buffer is CLEARED
    /// first, so anything it carried from an earlier call is dropped and never
    /// mixed into this call's rows, and its capacity is what carries over. On
    /// an error `out` is left untouched: the run has to produce a result before
    /// there is anything to decode into it.
    ///
    /// Both doors box every row through [`RawOutput::extend_values`], so the
    /// values they produce cannot drift; the only thing that differs is who
    /// owns the vector they land in.
    pub fn collect_into(&self, out: &mut Vec<Value>) -> Result<(), BatchError> {
        self.collect_into_on(Tier::Auto, out)
    }

    /// [`BoundBatch::collect_into`] on a chosen tier.
    pub fn collect_into_on(&self, tier: Tier, out: &mut Vec<Value>) -> Result<(), BatchError> {
        let tier = self.route(tier);
        self.collect_into_with(tier, threshold_for(tier), out)
    }

    /// [`BoundBatch::collect_into_on`] with an explicit trace threshold.
    pub fn collect_into_with(
        &self,
        tier: Tier,
        threshold: u32,
        out: &mut Vec<Value>,
    ) -> Result<(), BatchError> {
        self.collect_raw_with(tier, threshold, |raw| {
            out.clear();
            raw.extend_values(out);
        })
    }

    /// Evaluate every row and hand the results to `f` in the machine's OWN
    /// columnar encoding, on [`Tier::Jit`] — without boxing a row into a
    /// [`Value`].
    ///
    /// This is the door for a consumer that wants columns back. Boxing is not
    /// free and it is not small: measured on `x * 2 + 1` over 50k rows the
    /// compiled loop evaluates a row in ~1.0ns and building its `Value` costs
    /// ~2.4ns more; on `nums.map(y, y * 2)` over 10-element lists the loop
    /// costs ~8.8ns per row and the `Vec<Value>` + `Arc` per row costs ~59ns.
    /// A caller that is going to read an `i64` back out of the `Value` anyway
    /// pays that entirely for nothing.
    ///
    /// [`BoundBatch::collect`] is this function with [`RawOutput::to_values`]
    /// as `f`, so the two doors decode through the same code and cannot drift.
    ///
    /// The results are borrowed from the run's own buffers, which the next run
    /// overwrites — hence the callback rather than a returned slice.
    pub fn collect_raw<T>(&self, f: impl FnOnce(RawOutput<'_>) -> T) -> Result<T, BatchError> {
        self.collect_raw_on(Tier::Auto, f)
    }

    /// [`BoundBatch::collect_raw`] on a chosen tier.
    pub fn collect_raw_on<T>(
        &self,
        tier: Tier,
        f: impl FnOnce(RawOutput<'_>) -> T,
    ) -> Result<T, BatchError> {
        let tier = self.route(tier);
        self.collect_raw_with(tier, threshold_for(tier), f)
    }

    /// [`BoundBatch::collect_raw_on`] with an explicit trace threshold.
    pub fn collect_raw_with<T>(
        &self,
        tier: Tier,
        threshold: u32,
        f: impl FnOnce(RawOutput<'_>) -> T,
    ) -> Result<T, BatchError> {
        let tier = self.route(tier);
        assert_eq!(
            self.reduce,
            BatchReduce::PerRow,
            "collect on a batch bound to sum: use `bind_per_row`"
        );
        let lowered = &self.program.lowered;
        let mut run = self.run.borrow_mut();
        // A projection's output is its input column, so the clean tier writes
        // it as a copy instead of interpreting the loop that spells the copy
        // out. The tracing tiers still run the program: what they are measured
        // and counted on is what the program costs them.
        //
        // The bind-time flag is asked first so that a program which is NOT a
        // projection — every other case on this door — pays one bool test and
        // not a call that would answer the same thing.
        if !(tier == Tier::Clean && self.projected && run.project()) {
            run.run(|code, regs, nf, banks| dispatch(tier, threshold, code, regs, nf, banks))
                .ok_or(BatchError::Trapped)?;
        }
        // A LIST-valued result stored each row's element COUNT rather than a
        // value, and the elements themselves went to their own flat buffers at
        // a cursor running across the batch — the same Arrow layout an input
        // list column arrives in.
        let out = match &lowered.list_output {
            Some(out) => RawOutput::List {
                lens: run.output(),
                fields: out
                    .fields
                    .iter()
                    .zip(run.list_output())
                    .map(|((name, ty), buf)| (name.as_deref(), *ty, buf))
                    .collect(),
                distinct: run.distinct(),
            },
            None => RawOutput::Scalar {
                ty: lowered.result_bank,
                values: run.output(),
                distinct: run.distinct(),
            },
        };
        Ok(f(out))
    }

    fn execute(&self, tier: Tier, threshold: u32) -> Option<i64> {
        self.run
            .borrow_mut()
            .run(|code, regs, nf, banks| dispatch(tier, threshold, code, regs, nf, banks))
    }
}

fn threshold_for(tier: Tier) -> u32 {
    match tier {
        Tier::Jit => DEFAULT_JIT_THRESHOLD,
        // `Auto` never reaches here: every door resolves it through
        // `BoundBatch::route` first, which is where the batch's own size is.
        Tier::Auto | Tier::Interpreter | Tier::Clean => u32::MAX,
    }
}

fn dispatch(
    tier: Tier,
    threshold: u32,
    code: &std::sync::Arc<[i64]>,
    regs: &[i64],
    nf: usize,
    banks: &mut float_bank::Banks,
) -> i64 {
    match tier {
        Tier::Auto | Tier::Clean => float_bank::clean_interp_seeded_f_in(code, regs, nf, banks),
        // No bank hand-off here: these enter through the traced portal, which
        // builds its own. See the field's doc on `BatchRun`.
        Tier::Interpreter | Tier::Jit => {
            float_bank::run_jit_persistent_f(code, regs, nf, threshold)
        }
    }
}

/// Reads a batch's columns back as the values the tree-walker takes.
///
/// This is the other half of the batch model's contract. Every [`BatchError`]
/// means "evaluate this with [`Program::execute`] instead", and without this a
/// caller has to build that path themselves — including the two parts that are
/// easy to get wrong:
///
/// * A DOTTED column name is a NESTED MAP to the walker, not a variable with a
///   dot in its name. `obj.nested.value` must arrive as `obj` → `nested` →
///   `value`, or the walker raises `NoSuchKey` on an expression the batch
///   answers.
/// * A list column's row starts at the sum of every earlier row's element
///   count. Walked per row that is quadratic, so the offsets are computed once
///   here.
pub struct RowReader<'a, 'b> {
    batch: &'b Batch<'a>,
    /// Exclusive prefix sums of each list column's per-row element counts.
    offsets: HashMap<&'b str, Vec<i64>>,
}

/// One level of the variable tree a dotted column name expands to.
#[derive(Default)]
struct Node<'b> {
    leaf: Option<Value>,
    kids: HashMap<&'b str, Node<'b>>,
}

impl<'b> Node<'b> {
    fn insert(&mut self, path: &'b str, v: Value) {
        match path.split_once('.') {
            Some((head, rest)) => self.kids.entry(head).or_default().insert(rest, v),
            None => self.kids.entry(path).or_default().leaf = Some(v),
        }
    }

    /// A leaf is its own value; anything else is the map of its children.
    fn value(self) -> Value {
        if let Some(v) = self.leaf {
            return v;
        }
        Value::Map(crate::objects::Map::object(std::sync::Arc::new(
            self.kids
                .into_iter()
                .map(|(k, n)| (Key::String(std::sync::Arc::new(k.to_string())), n.value()))
                .collect(),
        )))
    }
}

impl<'a, 'b> RowReader<'a, 'b> {
    pub fn new(batch: &'b Batch<'a>) -> Self {
        let offsets = batch
            .columns
            .iter()
            .filter_map(|(name, col)| {
                let ColumnRef::List { lens, .. } = col else {
                    return None;
                };
                let mut acc = 0i64;
                let offs = lens
                    .iter()
                    .map(|&l| {
                        let o = acc;
                        acc += l;
                        o
                    })
                    .collect();
                Some((name.as_str(), offs))
            })
            .collect();
        Self { batch, offsets }
    }

    /// A CHILD scope of `base` holding `row`'s variables.
    ///
    /// A child rather than a fresh context so the caller's own functions and
    /// variables stay visible: an expression the batch declines for calling a
    /// registered function is exactly the one that needs them.
    pub fn scope<'p>(&self, base: &'p Context<'p>, row: usize) -> Context<'p> {
        let mut tree = Node::default();
        for (name, col) in &self.batch.columns {
            tree.insert(name, self.value(name, col, row));
        }
        let mut ctx = base.new_inner_scope();
        for (name, node) in tree.kids {
            ctx.add_variable_from_value(name, node.value());
        }
        ctx
    }

    /// One column's value at `row` — a scalar, or a list's own elements.
    fn value(&self, name: &str, col: &ColumnRef<'a>, row: usize) -> Value {
        let ColumnRef::List { lens, fields } = col else {
            return cell(col, row);
        };
        let start = self.offsets[name][row] as usize;
        let elems: Vec<Value> = (0..lens[row] as usize)
            .map(|j| match fields.as_slice() {
                // An unnamed field names the elements themselves, so a row's
                // element IS the scalar rather than a one-entry map.
                [(None, c)] => cell(c, start + j),
                _ => Value::Map(crate::objects::Map::object(std::sync::Arc::new(
                    fields
                        .iter()
                        .filter_map(|(f, c)| {
                            let f = (*f)?;
                            Some((
                                Key::String(std::sync::Arc::new(f.to_string())),
                                cell(c, start + j),
                            ))
                        })
                        .collect(),
                ))),
            })
            .collect();
        Value::list(elems)
    }
}

/// One scalar cell, as the tree-walker sees it. A temporal column is
/// nanoseconds, which is the representation the batch declared it in.
fn cell(col: &ColumnRef, k: usize) -> Value {
    match col {
        ColumnRef::Int(c) => Value::Int(c[k]),
        ColumnRef::Bool(c) => Value::Bool(c[k]),
        ColumnRef::UInt(c) => Value::UInt(c[k]),
        ColumnRef::Float(c) => Value::Float(c[k]),
        ColumnRef::Str(c) => Value::String(std::sync::Arc::new(c[k].clone())),
        ColumnRef::Timestamp(c) => {
            Value::Timestamp(chrono::DateTime::from_timestamp_nanos(c[k]).fixed_offset())
        }
        ColumnRef::Duration(c) => Value::Duration(chrono::Duration::nanoseconds(c[k])),
        // A list of lists is not a shape the schema can declare, so a list
        // column is never a field of another one.
        ColumnRef::List { .. } => Value::Null,
    }
}

/// Which evaluator answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Answered {
    /// The lowered bytecode, on the tier that was asked for.
    Batch,
    /// The tree-walker, row by row: the expression did not lower, or a row
    /// trapped.
    Walker,
}

/// Every row's value for `program` over `batch` — from the batch tiers where
/// the expression lowers and no row traps, and from the tree-walker otherwise.
///
/// This is the batch model's whole contract in one call. An expression outside
/// the subset and a trapped row are not outcomes a caller can act on
/// differently — both mean "the walker owns this" — so they are taken here
/// rather than handed back, and [`Answered`] says which path ran.
///
/// What is still handed back is a real error: a column the expression reads and
/// the batch does not carry, a column whose type or length disagrees, or a row
/// the WALKER could not evaluate either.
///
/// For a sum, use [`BatchProgram`] directly and fall back with [`RowReader`];
/// folding per-row values here would hide which of the two ran.
pub fn eval_per_row(
    program: &Program,
    schema: &Schema,
    batch: &Batch,
    base: &Context,
) -> Result<(Vec<Value>, Answered), BatchError> {
    eval_per_row_on(program, schema, batch, base, Tier::Jit)
}

/// [`eval_per_row`] on a chosen tier — for a harness comparing them, or a
/// caller that wants the plain VM.
pub fn eval_per_row_on(
    program: &Program,
    schema: &Schema,
    batch: &Batch,
    base: &Context,
    tier: Tier,
) -> Result<(Vec<Value>, Answered), BatchError> {
    let batched = BatchProgram::from_program(program, schema)
        .and_then(|bp| bp.bind_per_row(batch)?.collect_on(tier));
    match batched {
        Ok(v) => Ok((v, Answered::Batch)),
        // Only the two data-independent-of-the-caller outcomes fall back. A
        // missing or mistyped column is the caller's own description of the
        // batch being wrong, and the walker would fail on it too.
        Err(
            BatchError::Lower(_) | BatchError::Trapped | BatchError::TemporalOutOfDomain { .. },
        ) => {
            let reader = RowReader::new(batch);
            let values = (0..batch.rows())
                .map(|row| {
                    program
                        .execute(&reader.scope(base, row))
                        .map_err(|e| BatchError::Row {
                            row,
                            message: e.to_string(),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((values, Answered::Walker))
        }
        Err(e) => Err(e),
    }
}

/// One stored row result, read back as the CEL value it stands for.
///
/// `distinct` is the batch's strings in rank order and is only consulted for a
/// `string` result, where the stored `i64` is a rank rather than the value.
///
/// The index is always in range, by construction rather than by check: the
/// machine has no op that COMPUTES a string id. Every string a row can produce
/// is either a column value, a broadcast literal, or one of those two selected
/// between — and the ranking is taken over exactly that set (see
/// `prepare_batch`), the derived `string(x)` and `concat#k` columns included.
/// One run's per-row results in the two-bank machine's own columnar encoding,
/// as [`BoundBatch::collect_raw`] hands them over.
///
/// Every buffer is an `i64` column in its [`ValType`]'s own encoding, which is
/// the same encoding an INPUT column of that type is read in: a `float` is its
/// 64-bit pattern (`f64::from_bits(v as u64)`), a `uint` its raw bit pattern
/// (`v as u64`), a `bool` is `0`/`1`, a `timestamp` is nanoseconds since the
/// epoch, a `duration` is nanoseconds, and a `string` is a RANK into
/// `distinct`. [`RawOutput::to_values`] is the inverse.
pub enum RawOutput<'r> {
    /// One value per row.
    Scalar {
        ty: ValType,
        values: &'r [i64],
        distinct: &'r [String],
    },
    /// A LIST per row: row `i` owns the `lens[i]` elements that follow every
    /// earlier row's, so a row's elements start at the exclusive prefix sum of
    /// `lens`. `None` names the elements themselves (a list of scalars),
    /// `Some(f)` one record field.
    List {
        lens: &'r [i64],
        fields: Vec<(Option<&'r str>, ValType, &'r [i64])>,
        distinct: &'r [String],
    },
}

impl RawOutput<'_> {
    /// How many rows the run produced.
    pub fn rows(&self) -> usize {
        match self {
            RawOutput::Scalar { values, .. } => values.len(),
            RawOutput::List { lens, .. } => lens.len(),
        }
    }

    /// Box every row into the [`Value`] the tree-walker returns. This is what
    /// [`BoundBatch::collect`] does with a raw output, and the cost the raw
    /// door exists to let a columnar consumer skip.
    pub fn to_values(&self) -> Vec<Value> {
        let mut out = Vec::with_capacity(self.rows());
        self.extend_values(&mut out);
        out
    }

    /// [`RawOutput::to_values`] appending into a buffer the caller owns, which
    /// is what [`BoundBatch::collect_into_on`] hands a reused one to. Boxing a
    /// row costs the same either way; what the caller keeps is the output
    /// vector's own allocation, which is not part of evaluating a row.
    ///
    /// `to_values` is this function into a fresh vector, so a row is decoded by
    /// one piece of code whichever door asked for it.
    ///
    /// `#[inline]` because a caller in another crate is the case this is for:
    /// a non-generic `pub fn` without the attribute publishes no MIR to inline
    /// from downstream, so such a call is opaque however small the arm it
    /// takes. Measured from another crate on a one-row `bool` output, the call
    /// alone — not the work — was 2.0 ns of a 3.9 ns decode. A caller who
    /// arrives through [`BoundBatch::collect_into_on`] is already inside this
    /// crate by then and never paid it; `cel/examples/cleanfixprobe.rs` is
    /// where the two are told apart.
    #[inline]
    pub fn extend_values(&self, out: &mut Vec<Value>) {
        match self {
            RawOutput::Scalar {
                ty,
                values,
                distinct,
            } => {
                // The bank is a property of the OUTPUT, not of a row, so it is
                // decided once here and each arm is then a loop over one known
                // constructor — the same reason the record shape below is
                // decided outside its element loop. Matching per row cost a
                // measured 0.7 ns per row, which a one-row output pays in full.
                //
                // The string table is built inside the one arm that reads it,
                // so an output in any other bank does not pay for it at all.
                // See [`intern`] for why building it unconditionally was not
                // free.
                match *ty {
                    ValType::Int => out.extend(values.iter().map(|&v| Value::Int(v))),
                    ValType::UInt => out.extend(values.iter().map(|&v| Value::UInt(v as u64))),
                    ValType::Bool => out.extend(values.iter().map(|&v| Value::Bool(v != 0))),
                    ValType::Float => out.extend(
                        values
                            .iter()
                            .map(|&v| Value::Float(f64::from_bits(v as u64))),
                    ),
                    ValType::Str => {
                        let interned = intern(distinct);
                        out.extend(
                            values
                                .iter()
                                .map(|&v| Value::String(interned[v as usize].clone())),
                        );
                    }
                    ValType::Timestamp => out.extend(values.iter().map(|&v| {
                        Value::Timestamp(chrono::DateTime::from_timestamp_nanos(v).fixed_offset())
                    })),
                    ValType::Duration => out.extend(
                        values
                            .iter()
                            .map(|&v| Value::Duration(chrono::Duration::nanoseconds(v))),
                    ),
                }
            }
            RawOutput::List {
                lens,
                fields,
                distinct,
            } => {
                let mut at = 0usize;
                out.reserve(lens.len());
                // The distinct strings are interned once for the whole output,
                // and only when a field will read them — `column_of` consults
                // the table in its `Str` arm alone. It is NOT free when the
                // output has none: see [`intern`].
                let interned = fields
                    .iter()
                    .any(|(_, ty, _)| matches!(*ty, ValType::Str))
                    .then(|| intern(distinct));
                let interned = interned.as_ref();
                // The field shape is a property of the output, not of a row or
                // an element, so it is decided once here. Matching it inside
                // the element loop re-dispatched it per element.
                match fields.as_slice() {
                    // A list of scalars: the element IS the value, and every
                    // bank has an unboxed column, so the batch's elements
                    // become ONE shared column and a row is a WINDOW onto it.
                    // A row then costs a reference count and no per-element
                    // write, where boxing cost two allocations plus a 24-byte
                    // `Value` per element.
                    [(None, ty, buf)] => {
                        let column = Arc::new(ListStorage::Column(column_of(*ty, buf, interned)));
                        for &count in *lens {
                            // `ListRef::window` carries the bound check: the
                            // boxed arm sliced the buffer and so failed on a
                            // row length the run never produced, and an
                            // out-of-range window would otherwise surface as a
                            // short list.
                            let len = count.max(0) as usize;
                            out.push(Value::List(ListRef::window(Arc::clone(&column), at, len)));
                            at += len;
                        }
                    }
                    // A list of records. Every element carries the same field
                    // names over the same column buffers, so the names and the
                    // columns become ONE schema for the whole output and an
                    // element is an index into it. Rebuilding a `HashMap` per
                    // element cost an `Arc` and a table each, and the field
                    // names on top of that.
                    fields => {
                        let keys: Vec<Key> = fields
                            .iter()
                            .map(|(name, _, _)| {
                                Key::String(Arc::new(name.unwrap_or_default().to_string()))
                            })
                            .collect();
                        let columns = fields
                            .iter()
                            .map(|(_, ty, buf)| column_of(*ty, buf, interned))
                            .collect();
                        let schema = Arc::new(ListStorage::Record(Arc::new(RecordSchema::new(
                            keys, columns,
                        ))));
                        for &count in *lens {
                            let len = count.max(0) as usize;
                            out.push(Value::List(ListRef::window(Arc::clone(&schema), at, len)));
                            at += len;
                        }
                    }
                }
            }
        }
    }
}

/// The column a record field becomes: the batch's own words when the bank has
/// an unboxed representation, and otherwise the boxed values decoded ONCE for
/// the whole output rather than once per element.
/// The column a bank becomes. Every [`ValType`] has an unboxed form, so this
/// is total and a new bank cannot quietly fall back to boxing.
/// `interned` is `None` when no field of the output is a `Str`, which is the
/// only arm that reads it — its caller decides that once for the whole output
/// rather than building a table every arm but one ignores.
fn column_of(bank: ValType, words: &[i64], interned: Option<&Arc<[Arc<String>]>>) -> ValueColumn {
    let bank = match bank {
        ValType::Int => ScalarBank::Int,
        ValType::UInt => ScalarBank::UInt,
        ValType::Bool => ScalarBank::Bool,
        ValType::Float => ScalarBank::Float,
        ValType::Timestamp => ScalarBank::Timestamp,
        ValType::Duration => ScalarBank::Duration,
        ValType::Str => {
            // Reachable only through the predicate that built the table, so a
            // `None` here is that predicate and this arm having disagreed.
            let interned = interned.expect("a Str field is what makes the intern table needed");
            return ValueColumn::Str(Arc::new(StrBank::new(
                Arc::from(words),
                Arc::clone(interned),
            )));
        }
    };
    ValueColumn::Scalar {
        bank,
        words: Arc::from(words),
    }
}

/// The batch's distinct strings, interned once per output. A `Value::String`
/// is then a reference count rather than a fresh `String` and `Arc` per value,
/// which is what the rank encoding exists to make possible.
///
/// Both callers gate this on a `Str` actually being present, because it is NOT
/// free on an empty `distinct`: collecting into an `Arc<[_]>` allocates the
/// 16-byte `ArcInner` header whatever the length, and `prepare_batch_reduce`
/// leaves `distinct` empty for every result that is not a string. Ungated it
/// was one of three heap allocations on every `collect_on`, spent on output
/// nothing reads — measured at ~12 ns of a ~46 ns fixed per-call cost. A
/// comment here previously asserted the opposite, that an empty `Arc<[_]>`
/// does not allocate; a counting `#[global_allocator]` says it does.
fn intern(distinct: &[String]) -> Arc<[Arc<String>]> {
    distinct.iter().map(|s| Arc::new(s.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema(pairs: &[(&str, ValType)]) -> Schema {
        pairs.iter().map(|(p, t)| (p.to_string(), *t)).collect()
    }

    #[test]
    fn bool_column_is_declared_and_encoded_as_bool() {
        let s = schema(&[
            ("balance", ValType::Int),
            ("amount", ValType::Int),
            ("frozen", ValType::Bool),
        ]);
        let program = BatchProgram::compile("balance >= amount && !frozen", &s).unwrap();
        let balance = vec![10i64, 3, 7, 1];
        let amount = vec![5i64, 5, 7, 9];
        let frozen = vec![false, false, true, false];
        let batch = Batch::new(4)
            .column("balance", ColumnRef::Int(&balance))
            .column("amount", ColumnRef::Int(&amount))
            .column("frozen", ColumnRef::Bool(&frozen));
        let bound = program.bind(&batch).unwrap();
        // Row 0 matches; row 1 fails the compare; row 2 is frozen; row 3 fails.
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            assert_eq!(bound.sum_on(tier).unwrap(), Value::Int(1), "{tier:?}");
        }
    }

    /// The route is about which tier runs, never about what comes back. Two
    /// shapes on both sides of the crossing, each answered through the default
    /// door and through all three tiers by name.
    #[test]
    fn the_auto_route_answers_what_every_named_tier_answers() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x * 2 + 1", &s).unwrap();
        for rows in [1usize, 4, 1_000] {
            let x: Vec<i64> = (0..rows as i64).collect();
            let batch = Batch::new(rows).column("x", ColumnRef::Int(&x));
            let bound = program.bind_per_row(&batch).unwrap();
            let want: Vec<Value> = x.iter().map(|v| Value::Int(v * 2 + 1)).collect();
            assert_eq!(bound.collect().unwrap(), want, "auto at {rows} rows");
            for tier in [Tier::Auto, Tier::Clean, Tier::Interpreter, Tier::Jit] {
                assert_eq!(bound.collect_on(tier).unwrap(), want, "{tier:?} at {rows}");
            }
        }
    }

    /// The caller-buffer door owes exactly what the vector-returning one
    /// returns. Both decode through `RawOutput::extend_values`, and this is
    /// what keeps that true: every bank an output can carry — including a list
    /// result, whose arm builds a shared column rather than a value per row —
    /// asked on every tier, through both doors.
    #[test]
    fn collect_into_answers_what_collect_answers() {
        let ints = [7i64, -1, 0, 5];
        let floats = [1.5f64, -0.25, 0.0, 8.0];
        let bools = [true, false, false, true];
        let strs: Vec<String> = ["pear", "fig", "apple", "fig"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let elems = [1i64, 2, 3, 4, 5, 6];
        let lens = [2i64, 3, 1];
        let scalars: Vec<(&str, &str, Schema, ColumnRef)> = vec![
            (
                "x * 2 + 1",
                "x",
                schema(&[("x", ValType::Int)]),
                ColumnRef::Int(&ints),
            ),
            (
                "f * 2.0",
                "f",
                schema(&[("f", ValType::Float)]),
                ColumnRef::Float(&floats),
            ),
            (
                "!b",
                "b",
                schema(&[("b", ValType::Bool)]),
                ColumnRef::Bool(&bools),
            ),
            (
                "s",
                "s",
                schema(&[("s", ValType::Str)]),
                ColumnRef::Str(&strs),
            ),
        ];
        // One buffer across every case, so a case also inherits whatever the
        // case before it left behind.
        let mut buf = Vec::new();
        for (source, name, s, col) in scalars {
            let rows = match col {
                ColumnRef::Str(c) => c.len(),
                _ => ints.len(),
            };
            let batch = Batch::new(rows).column(name, col);
            let program = BatchProgram::compile(source, &s).unwrap();
            let bound = program.bind_per_row(&batch).unwrap();
            for tier in [Tier::Auto, Tier::Clean, Tier::Interpreter, Tier::Jit] {
                let want = bound.collect_on(tier).unwrap();
                bound.collect_into_on(tier, &mut buf).unwrap();
                assert_eq!(buf, want, "{source}: {tier:?}");
            }
        }

        let s = schema(&[("list[]", ValType::Int)]);
        let batch = Batch::new(lens.len()).column(
            "list",
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&elems))],
            },
        );
        let program = BatchProgram::compile("list.map(x, x * 2)", &s).unwrap();
        let bound = program.bind_per_row(&batch).unwrap();
        for tier in [Tier::Auto, Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let want = bound.collect_on(tier).unwrap();
            bound.collect_into_on(tier, &mut buf).unwrap();
            assert_eq!(buf, want, "list output: {tier:?}");
        }
    }

    /// A reused buffer holds the LATEST run and nothing else. The door clears
    /// before it decodes, so a shorter batch after a longer one cannot leave
    /// the longer one's tail behind — which is the failure a buffer that was
    /// merely overwritten in place would produce, and the one a caller reusing
    /// a buffer across differently sized batches would hit first.
    ///
    /// Run with a `string` result as well as an `int` one, because clearing a
    /// string row drops a reference count where clearing an int row drops
    /// nothing.
    #[test]
    fn a_reused_buffer_holds_exactly_the_latest_run() {
        let long: Vec<i64> = (0..64).collect();
        let short = [9i64, 4];
        let words: Vec<String> = ["fig", "pear", "fig"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let int_program =
            BatchProgram::compile("x * 2 + 1", &schema(&[("x", ValType::Int)])).unwrap();
        let str_program = BatchProgram::compile("s", &schema(&[("s", ValType::Str)])).unwrap();

        for tier in [Tier::Auto, Tier::Clean, Tier::Interpreter, Tier::Jit] {
            // Junk from before the first call, which the first call owes the
            // caller nothing of.
            let mut buf = vec![Value::Int(-777)];
            let run = |xs: &[i64], buf: &mut Vec<Value>| {
                let batch = Batch::new(xs.len()).column("x", ColumnRef::Int(xs));
                int_program
                    .bind_per_row(&batch)
                    .unwrap()
                    .collect_into_on(tier, buf)
                    .unwrap();
            };
            let want =
                |xs: &[i64]| -> Vec<Value> { xs.iter().map(|v| Value::Int(v * 2 + 1)).collect() };

            run(&long, &mut buf);
            assert_eq!(buf, want(&long), "{tier:?}: first run");
            // Shrinking, then growing back into the capacity the first run
            // left: neither direction may show the other run's rows.
            run(&short, &mut buf);
            assert_eq!(buf, want(&short), "{tier:?}: shorter second run");
            run(&long, &mut buf);
            assert_eq!(buf, want(&long), "{tier:?}: back to the taller batch");

            let batch = Batch::new(words.len()).column("s", ColumnRef::Str(&words));
            let bound = str_program.bind_per_row(&batch).unwrap();
            bound.collect_into_on(tier, &mut buf).unwrap();
            assert_eq!(buf, bound.collect_on(tier).unwrap(), "{tier:?}: strings");
        }
    }

    /// An expression that is a bare variable is a PROJECTION, and the clean
    /// tier answers one by copying the column instead of interpreting the loop
    /// that copies it a row at a time. What that shortcut owes is the answer
    /// the loop gives — in every bank an output can carry, and at a height
    /// where the copy is the whole run as well as one where it is not.
    ///
    /// The tiers are the oracle here rather than a hand-written expectation:
    /// they still run the program, so a copy that encodes a row differently
    /// than the loop stores it disagrees with three witnesses at once.
    #[test]
    fn a_projection_answers_what_the_loop_it_replaces_answers() {
        let strs: Vec<String> = ["pear", "fig", "apple", "fig"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ints = [7i64, -1, 0, i64::MAX];
        let floats = [1.5f64, -0.25, 0.0, f64::INFINITY];
        let bools = [true, false, false, true];
        let cases = [
            ("i", ValType::Int),
            ("f", ValType::Float),
            ("b", ValType::Bool),
            ("s", ValType::Str),
            ("t", ValType::Timestamp),
        ];
        for (name, ty) in cases {
            let program = BatchProgram::compile(name, &schema(&[(name, ty)])).unwrap();
            assert!(
                program.lowered().is_row_projection(),
                "`{name}` is a bare variable, so it is a projection"
            );
            for rows in [1usize, 4] {
                let col = match ty {
                    ValType::Int => ColumnRef::Int(&ints[..rows]),
                    ValType::Float => ColumnRef::Float(&floats[..rows]),
                    ValType::Bool => ColumnRef::Bool(&bools[..rows]),
                    ValType::Str => ColumnRef::Str(&strs[..rows]),
                    ValType::Timestamp => ColumnRef::Timestamp(&ints[..rows]),
                    other => unreachable!("no projection case declares {other:?}"),
                };
                let batch = Batch::new(rows).column(name, col);
                let bound = program.bind_per_row(&batch).unwrap();
                let want = bound.collect_on(Tier::Interpreter).unwrap();
                assert_eq!(want.len(), rows, "{name} at {rows} rows");
                for tier in [Tier::Auto, Tier::Clean, Tier::Jit] {
                    assert_eq!(bound.collect_on(tier).unwrap(), want, "{name}: {tier:?}");
                }
            }
        }
    }

    /// A projection stays on the clean tier however tall the batch gets, where
    /// the body-word rule would hand a tall one to the compiled tier.
    ///
    /// The rule is about a crossing, and a projection has none: the clean tier
    /// copies the column, so the work it does per row does not grow into what
    /// compiling is worth paying for. Pinned as a route rather than a time for
    /// the reason the word-count route is — the decision is the contract.
    #[test]
    fn a_projection_stays_clean_however_tall_the_batch() {
        let s = schema(&[("x", ValType::Int)]);
        let projection = BatchProgram::compile("x", &s).unwrap();
        let computed = BatchProgram::compile("x * 2 + 1", &s).unwrap();
        let tall: Vec<i64> = (0..10_000).collect();
        let batch = Batch::new(tall.len()).column("x", ColumnRef::Int(&tall));

        let bound = projection.bind_per_row(&batch).unwrap();
        assert!(
            bound.compiled_saving_ps() >= JIT_ENTRY_PS,
            "the batch is tall enough for the estimate to fire"
        );
        assert_eq!(bound.route(Tier::Auto), Tier::Clean);

        // The same height, one operation away from a projection, still crosses.
        let bound = computed.bind_per_row(&batch).unwrap();
        assert_eq!(bound.route(Tier::Auto), Tier::Jit);

        // The shortcut is per-row only, so a summing bind keeps the word rule.
        let bound = projection.bind(&batch).unwrap();
        assert_eq!(bound.route(Tier::Auto), Tier::Jit);
    }

    /// The other side of the predicate: an expression that DOES something to
    /// the column it reads is not a projection, so the clean tier runs its
    /// program. Pinned because the shortcut is only sound where the loop it
    /// stands in for would have stored the column unchanged.
    #[test]
    fn an_expression_with_a_body_is_not_a_projection() {
        let s = schema(&[("x", ValType::Int), ("y", ValType::Int)]);
        for src in ["x * 2", "x + y", "x == 1", "-x"] {
            assert!(
                !BatchProgram::compile(src, &s)
                    .unwrap()
                    .lowered()
                    .is_row_projection(),
                "`{src}` has a body"
            );
        }
    }

    /// What the route is FOR: a one-row straight-line expression saves too
    /// little by being compiled to pay for reaching compiled code and stays on
    /// the interpreter, while the same expression over a tall batch, and a
    /// comprehension over a long list, save more than [`JIT_ENTRY_PS`] and are
    /// handed to the compiled tier.
    ///
    /// Pinned as a routing DECISION rather than as a time, because a time is
    /// what the machine happens to cost today and the decision is the contract.
    #[test]
    fn the_route_follows_what_a_run_saves_by_being_compiled() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x * 2 + 1", &s).unwrap();
        let lowered = program.lowered();
        assert!(
            !lowered.iterates_elements(),
            "no comprehension in `x * 2 + 1`"
        );

        let one = vec![7i64];
        let batch = Batch::new(1).column("x", ColumnRef::Int(&one));
        let bound = program.bind_per_row(&batch).unwrap();
        assert!(bound.compiled_saving_ps() < JIT_ENTRY_PS);
        assert_eq!(bound.route(Tier::Auto), Tier::Clean);

        let tall: Vec<i64> = (0..1_000).collect();
        let batch = Batch::new(tall.len()).column("x", ColumnRef::Int(&tall));
        let bound = program.bind_per_row(&batch).unwrap();
        assert!(bound.compiled_saving_ps() >= JIT_ENTRY_PS);
        assert_eq!(bound.route(Tier::Auto), Tier::Jit);

        // One row, but the row's own list is what the body iterates.
        let s = schema(&[("list", ValType::Int), ("list[]", ValType::Int)]);
        let program = BatchProgram::compile("list.map(e, e * 2)", &s).unwrap();
        assert!(program.lowered().iterates_elements());
        let lens = vec![256i64];
        let elems: Vec<i64> = (0..256).collect();
        let batch = Batch::new(1).column(
            "list",
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&elems))],
            },
        );
        let bound = program.bind_per_row(&batch).unwrap();
        assert!(bound.compiled_saving_ps() >= JIT_ENTRY_PS);
        assert_eq!(bound.route(Tier::Auto), Tier::Jit);
    }

    /// The smallest count at which `route_at` hands the run to the compiled
    /// tier — a shape's crossing, as the live constants place it.
    ///
    /// Scanned rather than computed, so the tests that use it pin what the rule
    /// DOES and re-derive nothing about how it is spelled.
    fn crossing(mut route_at: impl FnMut(usize) -> Tier, limit: usize) -> usize {
        (1..=limit)
            .find(|&n| route_at(n) == Tier::Jit)
            .expect("the shape crosses within the limit")
    }

    /// The inversion a threshold in body words cannot express, and the reason
    /// the route is no longer one: a run of NO MORE words is handed to the
    /// compiled tier while a larger one stays on the interpreter.
    ///
    /// What separates them is how those words are spread. `x + 1` spends 25
    /// words on a row and the six-product body spends 79, and each row costs the
    /// interpreter something beyond its words that the compiled tier does not
    /// pay — so the thin body's rows are worth relatively more to compile, and
    /// it breaks even after fewer of its own words have gone by.
    ///
    /// No word count can name this pair the right way round: one at or below the
    /// thin run's size compiles both, one above the dense run's size compiles
    /// neither, and there is nothing in between. That is the whole claim, and it
    /// is made against whatever the constants happen to be — each shape's
    /// crossing is scanned for, not assumed, so this stays true across backends
    /// and survives a re-measurement that moves the level.
    #[test]
    fn a_thin_row_body_is_compiled_where_a_larger_dense_one_is_not() {
        let s = schema(&[("x", ValType::Int), ("y", ValType::Int)]);
        let thin = BatchProgram::compile("x + 1", &s).unwrap();
        let dense = BatchProgram::compile("x*2 + x*3 + y*4 + y*5 + x*6 + y*7", &s).unwrap();
        assert!(
            thin.lowered().row_words < dense.lowered().row_words,
            "the shapes are named for their words per row"
        );

        let at = |program: &BatchProgram, rows: usize| {
            let col: Vec<i64> = (0..rows as i64).collect();
            let batch = Batch::new(rows)
                .column("x", ColumnRef::Int(&col))
                .column("y", ColumnRef::Int(&col));
            let bound = program.bind_per_row(&batch).unwrap();
            (bound.route(Tier::Auto), bound.body_words())
        };

        // Each shape at its own crossing, and the dense one one row below its.
        let thin_rows = crossing(|n| at(&thin, n).0, 512);
        let dense_rows = crossing(|n| at(&dense, n).0, 512);
        assert!(dense_rows > 1, "the dense shape crosses above one row");
        let (thin_route, thin_words) = at(&thin, thin_rows);
        let (dense_route, dense_words) = at(&dense, dense_rows - 1);

        assert_eq!(thin_route, Tier::Jit, "{thin_rows} thin rows are compiled");
        assert_eq!(
            dense_route,
            Tier::Clean,
            "{} dense rows are not",
            dense_rows - 1
        );
        assert!(
            thin_words <= dense_words,
            "and the compiled run is the smaller: {thin_words} words compiled, \
             {dense_words} not"
        );
    }

    /// The same inversion on the element loop: two comprehensions, and the one
    /// whose inner body is thinner is compiled at no more total words than the
    /// denser one is left on the interpreter with.
    ///
    /// Pinned separately from the row loop because the two counts reach the rule
    /// by different routes — one is `rows`, the other is the batch's flattened
    /// element count — and a rule that lost the element half would still pass
    /// the test above.
    #[test]
    fn a_thin_element_body_is_compiled_where_a_larger_dense_one_is_not() {
        let s = schema(&[("list", ValType::Int), ("list[]", ValType::Int)]);
        let thin = BatchProgram::compile("list.map(e, e + 1)", &s).unwrap();
        let dense = BatchProgram::compile("list.map(e, (e + 1) * (e + 2) + e * 3)", &s).unwrap();
        assert!(
            thin.lowered().elem_words < dense.lowered().elem_words,
            "the shapes are named for their words per element"
        );

        let at = |program: &BatchProgram, elems: usize| {
            let lens = vec![elems as i64];
            let flat: Vec<i64> = (0..elems as i64).collect();
            let batch = Batch::new(1).column(
                "list",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Int(&flat))],
                },
            );
            let bound = program.bind_per_row(&batch).unwrap();
            (bound.route(Tier::Auto), bound.body_words())
        };

        let thin_elems = crossing(|n| at(&thin, n).0, 512);
        let dense_elems = crossing(|n| at(&dense, n).0, 512);
        assert!(dense_elems > 1, "the dense shape crosses above one element");
        let (thin_route, thin_words) = at(&thin, thin_elems);
        let (dense_route, dense_words) = at(&dense, dense_elems - 1);

        assert_eq!(thin_route, Tier::Jit);
        assert_eq!(dense_route, Tier::Clean);
        assert!(
            thin_words <= dense_words,
            "the compiled run is the smaller: {thin_words} words compiled, \
             {dense_words} not"
        );
    }

    /// Naming a tier still gets that tier. The route is the DEFAULT door's
    /// choice, and every cross-tier test in this file — and every oracle a
    /// majit answer is graded against — depends on the `_on` doors meaning
    /// exactly what they say.
    #[test]
    fn naming_a_tier_overrides_the_route() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x * 2 + 1", &s).unwrap();
        let one = vec![7i64];
        let batch = Batch::new(1).column("x", ColumnRef::Int(&one));
        let bound = program.bind_per_row(&batch).unwrap();
        assert_eq!(bound.route(Tier::Auto), Tier::Clean);
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            assert_eq!(bound.route(tier), tier);
        }
    }

    #[test]
    fn a_bool_column_declared_int_is_rejected_rather_than_answered() {
        // The whole point of `ValType::Bool`: this expression cannot lower at
        // all under an `int` declaration, so there is no way to reach the
        // bitwise `&&` that used to answer here.
        let s = schema(&[("frozen", ValType::Int)]);
        assert!(matches!(
            BatchProgram::compile("!frozen", &s),
            Err(BatchError::Lower(_))
        ));
    }

    #[test]
    fn a_column_of_the_wrong_type_is_an_error_not_a_reinterpretation() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x > 0", &s).unwrap();
        let wrong = vec![1.5f64];
        let batch = Batch::new(1).column("x", ColumnRef::Float(&wrong));
        assert!(matches!(
            program.bind(&batch),
            Err(BatchError::ColumnType { .. })
        ));
    }

    #[test]
    fn a_missing_column_is_an_error() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x > 0", &s).unwrap();
        let batch = Batch::new(1);
        assert!(matches!(
            program.bind(&batch),
            Err(BatchError::MissingColumn(_))
        ));
    }

    #[test]
    fn a_short_column_is_an_error() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x > 0", &s).unwrap();
        let x = vec![1i64, 2];
        let batch = Batch::new(4).column("x", ColumnRef::Int(&x));
        assert!(matches!(
            program.bind(&batch),
            Err(BatchError::RowCount { .. })
        ));
    }

    #[test]
    fn overflow_refuses_rather_than_wrapping() {
        let s = schema(&[("x", ValType::Int)]);
        let program = BatchProgram::compile("x + x", &s).unwrap();
        let x = vec![i64::MAX];
        let batch = Batch::new(1).column("x", ColumnRef::Int(&x));
        let bound = program.bind(&batch).unwrap();
        assert!(matches!(
            bound.sum_on(Tier::Clean),
            Err(BatchError::Trapped)
        ));
    }

    #[test]
    fn string_equality_goes_through_ranked_ids() {
        let s = schema(&[("name", ValType::Str)]);
        let program = BatchProgram::compile("name == \"ab\"", &s).unwrap();
        let name: Vec<String> = ["ab", "cd", "ab", "ab"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let batch = Batch::new(4).column("name", ColumnRef::Str(&name));
        assert_eq!(
            program.bind(&batch).unwrap().sum_on(Tier::Clean).unwrap(),
            Value::Int(3)
        );
    }

    /// The raw door hands back the machine's OWN columns, in the encoding
    /// [`RawOutput`] documents — not values, and not a copy. Asserted against
    /// the expected column contents directly rather than against `collect`,
    /// which decodes through the same `to_values` and so could not catch a
    /// wrong encoding or a wrong layout on its own.
    #[test]
    fn raw_output_is_the_machines_own_column() {
        let s = schema(&[
            ("x", ValType::Int),
            ("f", ValType::Float),
            ("name", ValType::Str),
        ]);
        let x = vec![3i64, -1, 7, 0];
        let f = vec![0.5f64, 1.25, -0.75, 2.0];
        let name: Vec<String> = ["cd", "ab", "cd", "ef"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let batch = || {
            Batch::new(4)
                .column("x", ColumnRef::Int(&x))
                .column("f", ColumnRef::Float(&f))
                .column("name", ColumnRef::Str(&name))
        };
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            // An int result is the plain column.
            let b = batch();
            let p = BatchProgram::compile("x * 2 + 1", &s).unwrap();
            let bound = p.bind_per_row(&b).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    assert_eq!(out.rows(), 4);
                    let RawOutput::Scalar { ty, values, .. } = out else {
                        panic!("scalar expression gave a list output")
                    };
                    assert_eq!(ty, ValType::Int);
                    assert_eq!(values, [7, -1, 15, 1]);
                })
                .unwrap();

            // A bool result is `0`/`1` in the int file, not a byte column.
            let b = batch();
            let p = BatchProgram::compile("x > 0", &s).unwrap();
            let bound = p.bind_per_row(&b).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    let RawOutput::Scalar { ty, values, .. } = out else {
                        panic!("scalar expression gave a list output")
                    };
                    assert_eq!(ty, ValType::Bool);
                    assert_eq!(values, [1, 0, 1, 0]);
                })
                .unwrap();

            // A float result is its 64-BIT PATTERN, so the caller reads it back
            // with `from_bits` and gets the bit-exact value.
            let b = batch();
            let p = BatchProgram::compile("f * 2.0", &s).unwrap();
            let bound = p.bind_per_row(&b).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    let RawOutput::Scalar { ty, values, .. } = out else {
                        panic!("scalar expression gave a list output")
                    };
                    assert_eq!(ty, ValType::Float);
                    let got: Vec<f64> = values.iter().map(|&v| f64::from_bits(v as u64)).collect();
                    assert_eq!(got, [1.0, 2.5, -1.5, 4.0]);
                })
                .unwrap();

            // A string result is a RANK into `distinct`, which is ordered, so
            // the rank compares the way the string does.
            let b = batch();
            let p = BatchProgram::compile("name", &s).unwrap();
            let bound = p.bind_per_row(&b).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    let RawOutput::Scalar {
                        ty,
                        values,
                        distinct,
                    } = out
                    else {
                        panic!("scalar expression gave a list output")
                    };
                    assert_eq!(ty, ValType::Str);
                    let got: Vec<&str> = values.iter().map(|&v| &*distinct[v as usize]).collect();
                    assert_eq!(got, ["cd", "ab", "cd", "ef"]);
                    assert!(distinct.windows(2).all(|w| w[0] < w[1]), "{distinct:?}");
                })
                .unwrap();
        }
    }

    /// A list-valued result comes back in the SAME Arrow layout an input list
    /// column arrives in: per-row element counts plus one buffer per field
    /// packed across the batch. Nothing per row is allocated.
    #[test]
    fn raw_output_of_a_list_result_is_arrow_shaped() {
        let s = schema(&[("nums[]", ValType::Int)]);
        let lens = vec![3i64, 0, 2];
        let flat = vec![1i64, -2, 3, 4, -5];
        let batch = || {
            Batch::new(3).column(
                "nums",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Int(&flat))],
                },
            )
        };
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            // `map` keeps every element, so the counts are the source's.
            let b = batch();
            let p = BatchProgram::compile("nums.map(y, y * 2)", &s).unwrap();
            let bound = p.bind_per_row(&b).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    assert_eq!(out.rows(), 3);
                    let RawOutput::List { lens, fields, .. } = out else {
                        panic!("list expression gave a scalar output")
                    };
                    assert_eq!(lens, [3, 0, 2]);
                    assert_eq!(fields.len(), 1);
                    let (name, ty, buf) = fields[0];
                    assert_eq!(name, None, "a list of scalars has no field name");
                    assert_eq!(ty, ValType::Int);
                    assert_eq!(&buf[..5], [2, -4, 6, 8, -10]);
                })
                .unwrap();

            // `filter` keeps only what the predicate admits, so a row's count
            // shrinks and the next row's elements move up behind it.
            let b = batch();
            let p = BatchProgram::compile("nums.filter(y, y > 0)", &s).unwrap();
            let bound = p.bind_per_row(&b).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    let RawOutput::List { lens, fields, .. } = out else {
                        panic!("list expression gave a scalar output")
                    };
                    assert_eq!(lens, [2, 0, 1]);
                    assert_eq!(&fields[0].2[..3], [1, 3, 4]);
                })
                .unwrap();
        }
    }

    /// The COLLECT door on the same shape, which is the arm the raw test above
    /// bypasses. Every assertion here crosses the two map strategies: `expect`
    /// is built the boxed way and the output is a record view over the batch's
    /// own columns, so an equality that passes says the unboxed row reads back
    /// as the entries it stands for.
    #[test]
    /// Every row of one output must be a window onto ONE buffer — that is what
    /// makes a row cost no allocation. Measured at 1.000 allocs/row before and
    /// 0.001 after (`cargo run --release --example allocs`), but an allocation
    /// count is not something the test suite can assert, so the sharing itself
    /// is.
    fn every_row_of_a_list_output_windows_one_shared_buffer() {
        let elems = [1i64, 2, 3, 4, 5, 6];
        let lens = vec![2i64, 3, 1];
        // The scalar-element arm and the record arm build different storages,
        // so both are checked.
        for (source, s, field) in [
            (
                "list.map(x, x * 2)",
                schema(&[("list[]", ValType::Int)]),
                None,
            ),
            (
                "list.filter(i, i.price > 0)",
                schema(&[("list[].price", ValType::Int)]),
                Some("price"),
            ),
        ] {
            let program = BatchProgram::compile(source, &s).unwrap();
            let batch = Batch::new(lens.len()).column(
                "list",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(field, ColumnRef::Int(&elems))],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(Tier::Jit)
                .unwrap();
            assert_eq!(rows.len(), 3, "{source}");
            let Value::List(first) = &rows[0] else {
                panic!("{source}: not a list");
            };
            for (i, row) in rows.iter().enumerate() {
                let Value::List(row) = row else {
                    panic!("{source}: row {i} is not a list");
                };
                assert!(
                    first.shares_storage_with(row),
                    "{source}: row {i} has its own buffer"
                );
            }
        }
    }

    #[test]
    fn a_record_list_collected_as_values_reads_back_every_field() {
        use crate::objects::KeyRef;

        fn record(fields: &[(&str, i64)]) -> Value {
            Value::Map(crate::objects::Map::object(Arc::new(
                fields
                    .iter()
                    .map(|(k, v)| (Key::String(Arc::new(k.to_string())), Value::Int(*v)))
                    .collect(),
            )))
        }

        let s = schema(&[
            ("items[].price", ValType::Int),
            ("items[].qty", ValType::Int),
        ]);
        // The third row is empty, so a row that admits nothing still has to
        // land as an empty list rather than run off the columns.
        let lens = vec![2i64, 1, 0];
        let price = vec![10i64, 3, 7];
        let qty = vec![1i64, 5, 2];
        let expect = vec![
            Value::list(vec![record(&[("price", 10), ("qty", 1)])]),
            Value::list(vec![record(&[("price", 7), ("qty", 2)])]),
            Value::list(Vec::<Value>::new()),
        ];

        // `filter` hands the element back rather than computing one, so the
        // output is a list of records and `to_values` takes the record arm.
        let program = BatchProgram::compile("items.filter(i, i.price > 5)", &s).unwrap();
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(lens.len()).column(
                "items",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![
                        (Some("price"), ColumnRef::Int(&price)),
                        (Some("qty"), ColumnRef::Int(&qty)),
                    ],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            assert_eq!(rows, expect, "{tier:?}");

            // ... and the same entries through the map accessors, so a strategy
            // that only satisfies the equality is not enough.
            let Value::List(items) = &rows[0] else {
                panic!("{tier:?}: row 0 is not a list")
            };
            assert_eq!(items.len(), 1);
            let Some(Value::Map(row)) = items.get(0) else {
                panic!("{tier:?}: the element is not a map")
            };
            assert_eq!(row.len(), 2);
            assert!(row.contains_key(&KeyRef::String("price")));
            assert!(!row.contains_key(&KeyRef::String("absent")));
            assert_eq!(
                *row.get(&KeyRef::String("price")).unwrap(),
                Value::Int(10),
                "{tier:?}"
            );
            assert_eq!(
                *row.get(&KeyRef::String("qty")).unwrap(),
                Value::Int(1),
                "{tier:?}"
            );
            assert!(row.get(&KeyRef::String("absent")).is_none());
            let mut entries: Vec<(Key, Value)> = row
                .iter()
                .map(|(k, v)| (k.clone(), v.into_owned()))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(
                entries,
                vec![
                    (Key::String(Arc::new("price".to_string())), Value::Int(10)),
                    (Key::String(Arc::new("qty".to_string())), Value::Int(1)),
                ],
                "{tier:?}"
            );
        }
    }

    /// The temporal banks. They were the last two with no unboxed column, and
    /// nothing else covers a timestamp or duration ELEMENT list — the existing
    /// temporal tests all use a scalar column.
    #[test]
    fn a_temporal_element_list_decodes_from_the_shared_column() {
        let lens = vec![2i64, 1];
        let nanos = vec![1_000_000_000i64, 2_500_000_000, 7_000_000_000];

        let s = schema(&[("at[]", ValType::Timestamp)]);
        let program =
            BatchProgram::compile("at.filter(t, t > timestamp(\"1970-01-01T00:00:01Z\"))", &s)
                .expect("timestamp element list is in the traceable subset");
        let stamp =
            |n: i64| Value::Timestamp(chrono::DateTime::from_timestamp_nanos(n).fixed_offset());
        let expect = vec![
            Value::list(vec![stamp(2_500_000_000)]),
            Value::list(vec![stamp(7_000_000_000)]),
        ];
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(lens.len()).column(
                "at",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Timestamp(&nanos))],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            assert_eq!(rows, expect, "{tier:?}");
        }

        let s = schema(&[("took[]", ValType::Duration)]);
        let program = BatchProgram::compile("took.filter(d, d > duration(\"1s\"))", &s)
            .expect("duration element list is in the traceable subset");
        let span = |n: i64| Value::Duration(chrono::Duration::nanoseconds(n));
        let expect = vec![
            Value::list(vec![span(2_500_000_000)]),
            Value::list(vec![span(7_000_000_000)]),
        ];
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(lens.len()).column(
                "took",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Duration(&nanos))],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            assert_eq!(rows, expect, "{tier:?}");
        }
    }

    /// A string element list. The elements are ranks into the output's
    /// interned table, so two equal strings must come back as the SAME `Arc` —
    /// which is the property the rank encoding is there to preserve and the
    /// one an equality check alone would not catch.
    #[test]
    fn a_string_list_collected_as_values_shares_one_arc_per_distinct_string() {
        let s = schema(&[("tags[]", ValType::Str)]);
        let lens = vec![3i64, 1];
        let tags: Vec<String> = ["a", "b", "a", "c"].iter().map(|t| t.to_string()).collect();
        let text = |t: &str| Value::String(Arc::new(t.to_string()));
        let expect = vec![
            Value::list(vec![text("a"), text("a")]),
            Value::list(vec![text("c")]),
        ];

        let program = BatchProgram::compile("tags.filter(t, t != \"b\")", &s).unwrap();
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(lens.len()).column(
                "tags",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Str(&tags))],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            assert_eq!(rows, expect, "{tier:?}");

            let Value::List(items) = &rows[0] else {
                panic!("{tier:?}: row 0 is not a list")
            };
            let (Some(Value::String(first)), Some(Value::String(second))) =
                (items.get(0), items.get(1))
            else {
                panic!("{tier:?}: the elements are not strings")
            };
            assert!(
                Arc::ptr_eq(&first, &second),
                "{tier:?}: two equal strings are not the same interned Arc"
            );
        }
    }

    /// A SCALAR string result, which is the arm that decides whether the
    /// interned table gets built at all.
    ///
    /// `to_values` builds the table only when the output will read it, and for
    /// a scalar output that decision is `ty == ValType::Str` alone. If the gate
    /// and `decode`'s `Str` arm ever disagree, `decode` indexes an empty table
    /// — so this pins the one shape where skipping the build would be wrong,
    /// alongside the two list tests that pin the other arm. Distinct strings
    /// per row, so a row reading the wrong rank cannot pass by coincidence.
    #[test]
    fn a_scalar_string_result_still_reads_its_interned_table() {
        let s = schema(&[("name", ValType::Str)]);
        let names: Vec<String> = ["ada", "grace", "ada"]
            .iter()
            .map(|t| t.to_string())
            .collect();
        let expect: Vec<Value> = ["ada!", "grace!", "ada!"]
            .iter()
            .map(|t| Value::String(Arc::new(t.to_string())))
            .collect();

        let program = BatchProgram::compile("name + \"!\"", &s).unwrap();
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(names.len()).column("name", ColumnRef::Str(&names));
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            assert_eq!(rows, expect, "{tier:?}");
        }
    }

    /// A record whose field is a string: the column is ranks into the same
    /// interned table, read back through the map accessors.
    #[test]
    fn a_record_lists_string_field_reads_back_from_the_interned_table() {
        use crate::objects::KeyRef;

        let s = schema(&[
            ("items[].name", ValType::Str),
            ("items[].price", ValType::Int),
        ]);
        let lens = vec![2i64, 1];
        let name: Vec<String> = ["x", "y", "x"].iter().map(|t| t.to_string()).collect();
        let price = vec![10i64, 3, 7];

        let program = BatchProgram::compile("items.filter(i, i.price > 5)", &s).unwrap();
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(lens.len()).column(
                "items",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![
                        (Some("name"), ColumnRef::Str(&name)),
                        (Some("price"), ColumnRef::Int(&price)),
                    ],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();

            // Rows 0 and 1 each admit one element, and both name the same
            // string, so the two reads must land on one interned `Arc`.
            let mut names = Vec::new();
            for (row, want_price) in rows.iter().zip([10i64, 7]) {
                let Value::List(items) = row else {
                    panic!("{tier:?}: not a list")
                };
                let Some(Value::Map(record)) = items.get(0) else {
                    panic!("{tier:?}: the element is not a map")
                };
                assert_eq!(
                    *record.get(&KeyRef::String("price")).unwrap(),
                    Value::Int(want_price),
                    "{tier:?}"
                );
                let Value::String(got) = record.get(&KeyRef::String("name")).unwrap().into_owned()
                else {
                    panic!("{tier:?}: the name field is not a string")
                };
                assert_eq!(*got, "x", "{tier:?}");
                names.push(got);
            }
            assert!(
                Arc::ptr_eq(&names[0], &names[1]),
                "{tier:?}: the same string in two rows is not the same interned Arc"
            );
        }
    }

    /// A list of RECORDS gets one buffer per field, named and ordered the way
    /// the schema declares them — the caller reads a record column-wise instead
    /// of taking a `Value::Map` per element.
    #[test]
    fn raw_output_of_a_record_list_is_one_buffer_per_field() {
        let s = schema(&[
            ("items[].price", ValType::Int),
            ("items[].qty", ValType::Int),
        ]);
        let lens = vec![2i64, 1];
        let price = vec![10i64, 3, 7];
        let qty = vec![1i64, 5, 2];
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(2).column(
                "items",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![
                        (Some("price"), ColumnRef::Int(&price)),
                        (Some("qty"), ColumnRef::Int(&qty)),
                    ],
                },
            );
            let p = BatchProgram::compile("items.filter(i, i.price > 5)", &s).unwrap();
            let bound = p.bind_per_row(&batch).unwrap();
            bound
                .collect_raw_on(tier, |out| {
                    let RawOutput::List { lens, fields, .. } = out else {
                        panic!("list expression gave a scalar output")
                    };
                    assert_eq!(lens, [1, 1]);
                    let names: Vec<Option<&str>> = fields.iter().map(|f| f.0).collect();
                    assert_eq!(names, [Some("price"), Some("qty")]);
                    assert_eq!(&fields[0].2[..2], [10, 7]);
                    assert_eq!(&fields[1].2[..2], [1, 2]);
                })
                .unwrap();
        }
    }

    /// A `bool` column is READ WHERE THE CALLER KEEPS IT — one byte per row —
    /// so its slot's load must use the row index and not `row * 8`.
    ///
    /// A `bool` is `0`/`1`, which is exactly the value a wrongly-plumbed load
    /// is most likely to return by accident, so this pins the ADDRESSING rather
    /// than the value: the pattern below alternates in a way that a load at
    /// `row * 8` reproduces only for row 0. Every tier is checked, because the
    /// clean interpreter, the tracing interpreter and the compiled trace each
    /// compute the address by their own route.
    #[test]
    fn a_bool_column_is_read_a_byte_at_a_time() {
        let s = schema(&[("b", ValType::Bool), ("x", ValType::Int)]);
        // 24 rows: more than one 8-byte stride, and `true` in positions that a
        // stride-8 read would land on `false` for.
        let b: Vec<bool> = (0..24).map(|i| i % 3 == 0).collect();
        let x: Vec<i64> = (0..24).collect();
        let expect: Vec<Value> = b
            .iter()
            .zip(&x)
            .map(|(&f, &v)| Value::Int(if f { v * 2 } else { -v }))
            .collect();
        let program = BatchProgram::compile("b ? x * 2 : -x", &s).unwrap();
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(24)
                .column("b", ColumnRef::Bool(&b))
                .column("x", ColumnRef::Int(&x));
            let got = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            assert_eq!(got, expect, "{tier:?}");
        }
    }

    /// The same one-byte read for a `bool` LIST FIELD, whose address is the
    /// inner loop's element index rather than the row's.
    ///
    /// The element path computes `(offset + j) * 8` for a word column and stops
    /// at `offset + j` for a byte one, and both live behind the same `ea_reg`,
    /// so a byte field read at the word address is the mistake this pins. The
    /// `active` pattern is chosen so a stride-8 read agrees only on element 0.
    #[test]
    fn a_bool_list_field_is_read_a_byte_at_a_time() {
        let s = schema(&[
            ("items[].price", ValType::Int),
            ("items[].active", ValType::Bool),
        ]);
        let lens = vec![5i64, 0, 7, 3];
        let total = lens.iter().sum::<i64>() as usize;
        let price: Vec<i64> = (1..=total as i64).collect();
        let active: Vec<bool> = (0..total).map(|k| k % 3 != 1).collect();
        // The walker's answer, computed here from the same flat buffers.
        let mut at = 0usize;
        let expect: Vec<Value> = lens
            .iter()
            .map(|&n| {
                let end = at + n as usize;
                let total: i64 = (at..end).filter(|&k| active[k]).map(|k| price[k]).sum();
                at = end;
                Value::Int(total)
            })
            .collect();
        let program =
            BatchProgram::compile("items.filter(i, i.active).map(i, i.price)", &s).unwrap();
        for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
            let batch = Batch::new(lens.len()).column(
                "items",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![
                        (Some("active"), ColumnRef::Bool(&active)),
                        (Some("price"), ColumnRef::Int(&price)),
                    ],
                },
            );
            let rows = program
                .bind_per_row(&batch)
                .unwrap()
                .collect_on(tier)
                .unwrap();
            // Sum each row's collected prices: the per-element identity is what
            // the addressing decides, and the sum is wrong the moment one
            // element is admitted or dropped by the wrong `active` byte.
            let got: Vec<Value> = rows
                .iter()
                .map(|r| match r {
                    Value::List(items) => Value::Int(
                        items
                            .iter()
                            .map(|v| match v {
                                Value::Int(i) => i,
                                other => panic!("{other:?}"),
                            })
                            .sum(),
                    ),
                    other => panic!("{other:?}"),
                })
                .collect();
            assert_eq!(got, expect, "{tier:?}");
        }
    }

    /// A `bool` slot takes a `bool` column and nothing else.
    ///
    /// The load reads one byte at the row index, so an `i64` column of `0`/`1`
    /// under a `bool` slot would be read at an eighth of its stride and answer
    /// with whatever byte sat there — a wrong answer, not an error. The bind
    /// refuses it instead.
    #[test]
    fn an_int_column_cannot_back_a_bool_slot() {
        let s = schema(&[("b", ValType::Bool)]);
        let program = BatchProgram::compile("b", &s).unwrap();
        let as_ints = vec![1i64, 0, 1, 0];
        let batch = Batch::new(4).column("b", ColumnRef::Int(&as_ints));
        assert!(matches!(
            program.bind(&batch),
            Err(BatchError::ColumnType { .. })
        ));
    }

    #[test]
    fn float_result_comes_back_as_a_float() {
        let s = schema(&[("f", ValType::Float)]);
        let program = BatchProgram::compile("f * 2.0", &s).unwrap();
        assert_eq!(program.result_type(), ValType::Float);
        let f = vec![0.5f64, 1.25, -0.75];
        let batch = Batch::new(3).column("f", ColumnRef::Float(&f));
        assert_eq!(
            program.bind(&batch).unwrap().sum_on(Tier::Clean).unwrap(),
            Value::Float(2.0)
        );
    }

    #[test]
    fn a_list_column_feeds_a_comprehension() {
        let s = schema(&[("limit", ValType::Int), ("items[]", ValType::Int)]);
        let program = BatchProgram::compile("items.all(i, i > limit)", &s).unwrap();
        // Rows: [3,4] > 2 yes; [1] > 2 no; [] > 5 vacuously yes.
        let limit = vec![2i64, 2, 5];
        let lens = vec![2i64, 1, 0];
        let elems = vec![3i64, 4, 1];
        let batch = Batch::new(3)
            .column("limit", ColumnRef::Int(&limit))
            .column(
                "items",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Int(&elems))],
                },
            );
        assert_eq!(
            program.bind(&batch).unwrap().sum_on(Tier::Clean).unwrap(),
            Value::Int(2)
        );
    }

    /// The fallback contract: an expression the batch cannot answer is still
    /// answered, by the tree-walker, over the SAME columns — including the two
    /// reconstructions a caller would have to get right by hand.
    #[test]
    fn the_walker_answers_what_the_batch_refuses() {
        let s = schema(&[
            ("x", ValType::Int),
            ("obj.nested.value", ValType::Int),
            ("tags[]", ValType::Str),
        ]);
        let x = vec![1i64, 2, 3, 4];
        let nested = vec![10i64, 20, 30, 40];
        let lens = vec![2i64, 0, 1, 3];
        let tags: Vec<String> = ["a", "bb", "ccc", "d", "ee", "f"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let batch = Batch::new(4)
            .column("x", ColumnRef::Int(&x))
            .column("obj.nested.value", ColumnRef::Int(&nested))
            .column(
                "tags",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Str(&tags))],
                },
            );
        let mut base = Context::default();
        base.add_function("triple", |v: i64| v * 3);

        // In subset: the batch answers, and the dotted name is a column.
        let p = Program::compile("x + obj.nested.value").unwrap();
        let (v, who) = eval_per_row(&p, &s, &batch, &base).unwrap();
        assert_eq!(who, Answered::Batch);
        assert_eq!(v[2], Value::Int(33));

        // Out of subset (a registered function the lowering cannot see into):
        // the walker answers, and it needs `obj` as a NESTED MAP and the
        // caller's own function.
        let p = Program::compile("triple(obj.nested.value) + x").unwrap();
        let (v, who) = eval_per_row(&p, &s, &batch, &base).unwrap();
        assert_eq!(who, Answered::Walker);
        assert_eq!(
            v,
            vec![
                Value::Int(31),
                Value::Int(62),
                Value::Int(93),
                Value::Int(124)
            ]
        );

        // A list column, rebuilt per row from the flattened buffer at the right
        // offset — row 1 is empty and row 3 starts at element 3.
        let p = Program::compile("triple(x) > 0 ? tags : tags").unwrap();
        let (v, who) = eval_per_row(&p, &s, &batch, &base).unwrap();
        assert_eq!(who, Answered::Walker);
        let row = |k: usize| match &v[k] {
            Value::List(l) => l
                .iter()
                .map(|e| match e {
                    Value::String(s) => s.to_string(),
                    other => panic!("not a string: {other:?}"),
                })
                .collect::<Vec<_>>(),
            other => panic!("not a list: {other:?}"),
        };
        assert_eq!(row(0), ["a", "bb"]);
        assert!(row(1).is_empty());
        assert_eq!(row(2), ["ccc"]);
        assert_eq!(row(3), ["d", "ee", "f"]);
    }

    /// A trapped batch falls back rather than surfacing the trap, and the
    /// walker's own error surfaces as a row error rather than as a fallback.
    #[test]
    fn a_trap_falls_back_and_a_real_error_surfaces() {
        let s = schema(&[("a", ValType::Int), ("b", ValType::Int)]);
        let a = vec![10i64, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        let base = Context::default();

        // Division by zero on one row: the batch traps, the walker answers the
        // rows it can -- and raises on the one it cannot, which is a row error.
        let mut b = vec![2i64; 10];
        b[7] = 0;
        let batch = Batch::new(10)
            .column("a", ColumnRef::Int(&a))
            .column("b", ColumnRef::Int(&b));
        let p = Program::compile("a / b").unwrap();
        assert!(matches!(
            eval_per_row(&p, &s, &batch, &base),
            Err(BatchError::Row { row: 7, .. })
        ));

        // The same expression with no zero divisor is answered by the batch.
        let b = vec![2i64; 10];
        let batch = Batch::new(10)
            .column("a", ColumnRef::Int(&a))
            .column("b", ColumnRef::Int(&b));
        let (v, who) = eval_per_row(&p, &s, &batch, &base).unwrap();
        assert_eq!(who, Answered::Batch);
        assert_eq!(v[0], Value::Int(5));
    }

    /// A column the expression reads and the batch does not carry is the
    /// caller's own mistake, not something the walker can rescue.
    #[test]
    fn a_missing_column_is_not_fallen_back_on() {
        let s = schema(&[("a", ValType::Int)]);
        let batch = Batch::new(2);
        let p = Program::compile("a + 1").unwrap();
        assert!(matches!(
            eval_per_row(&p, &s, &batch, &Context::default()),
            Err(BatchError::MissingColumn(_))
        ));
    }
}
