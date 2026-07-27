//! Public columnar-batch API for the majit tier.
//!
//! [`super::lower`] turns a CEL expression into two-bank bytecode and
//! [`super::bytecode`] runs it, but both speak the machine's language: `i64`
//! register banks, base pointers, content-hash ids, Arrow offset buffers. Every
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
//! **What it computes.** The machine evaluates the expression per row and
//! reduces with a running total, so the answer is `sum over rows of expr(row)` —
//! for a boolean predicate, the number of matching rows. There is no per-row
//! output: the reduction is what the compiled trace's loop-carried accumulator
//! is. The reduction is the DRIVER's, not a CEL `+`: it wraps at 64 bits and
//! sums floats left to right in row order (bit-exact with the same rows summed
//! through the tree-walker, since float addition does not reassociate).
//!
//! **What it refuses.** Everything outside the traceable subset declines at
//! [`BatchProgram::compile`], and a batch whose data would make the tree-walker
//! raise (an `int` overflow, a division by zero) refuses at [`BoundBatch::sum`].
//! Both are the signal to evaluate that expression with
//! [`crate::Program::execute`], which owns the error.

use std::collections::HashMap;

use super::bytecode::{float_bank, prepare_batch, BatchRun, Column};
use super::lower::{
    elem_slot_source, intern_hash, lower_typed, offset_slot_source, size_slot_source, LoweredF,
    Schema, SlotKind, ValType,
};
use crate::{Program, Value};

/// Trace threshold [`Tier::Jit`] runs at: the batch loop compiles after this
/// many iterations. Matches the threshold the benchmarks and tests use.
pub const DEFAULT_JIT_THRESHOLD: u32 = 8;

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
    /// Two distinct strings in this batch share a content hash, so an id compare
    /// would not equal a content compare. Data-dependent: another batch of the
    /// same expression may be fine.
    HashCollision(String, String),
    /// A row's arithmetic trapped — an `int` overflow or a division by zero,
    /// where the tree-walker raises. No sum is the right answer.
    Trapped,
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
            BatchError::HashCollision(a, b) => {
                write!(f, "content-hash collision between `{a}` and `{b}`")
            }
            BatchError::Trapped => write!(f, "a row trapped (overflow or division by zero)"),
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
    /// A `string` column. Encoded to `i64` content hashes, which is why only
    /// equality is defined on it (see [`ValType::Str`]).
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
    /// The meta-tracing tier: traces the batch loop and compiles it.
    #[default]
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
        Ok(BatchProgram {
            lowered: lower_typed(program.expression(), schema)?,
        })
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
    /// the buffers the schema does not declare: a string column's content-hash
    /// ids, a `size(...)` length column, a list's `offset(...)` prefix sums.
    ///
    /// This is the per-batch work. Keep the [`BoundBatch`] and call
    /// [`BoundBatch::sum`] on it rather than rebinding to run again.
    pub fn bind<'a, 'b>(&'b self, batch: &'a Batch<'a>) -> Result<BoundBatch<'a, 'b>, BatchError> {
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

        // A string id compare equals a content compare only while the hash is
        // injective over the strings PRESENT — the column values and the
        // expression's own literals. A collision is data-dependent, so it is an
        // error on this batch, not on the expression.
        self.check_hash_injectivity(batch)?;

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
        let run = prepare_batch(&self.lowered, &columns, batch.rows, "BatchProgram::bind");
        Ok(BoundBatch {
            program: self,
            run: std::cell::RefCell::new(run),
            _derived: derived,
        })
    }

    /// Resolve one slot path to the column that feeds it.
    fn plan_slot<'a>(
        &self,
        batch: &'a Batch<'a>,
        path: &str,
        ty: ValType,
        derived: &mut Vec<DerivedColumn>,
    ) -> Result<Plan<'a>, BatchError> {
        // `size(x)`: the byte length of a string, or the element count of a list.
        if let Some(src) = size_slot_source(path) {
            let col = lookup(batch, src)?;
            let buf = match col {
                ColumnRef::Str(c) => c.iter().map(|s| s.len() as i64).collect(),
                ColumnRef::List { lens, .. } => lens.to_vec(),
                _ => {
                    return Err(BatchError::ColumnType {
                        name: src.to_string(),
                        declared: ValType::Str,
                    })
                }
            };
            derived.push(DerivedColumn::int(buf));
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
            return encode(col, ty, path, derived);
        }
        encode(lookup(batch, path)?, ty, path, derived)
    }

    /// FNV-1a is injective over every string set these batches have carried, but
    /// it is a 64-bit hash, so "is" has to be checked rather than assumed.
    fn check_hash_injectivity<'s>(&'s self, batch: &'s Batch) -> Result<(), BatchError> {
        fn note<'s>(seen: &mut HashMap<i64, &'s str>, s: &'s str) -> Result<(), BatchError> {
            match seen.insert(intern_hash(s), s) {
                Some(prev) if prev != s => {
                    Err(BatchError::HashCollision(prev.to_string(), s.to_string()))
                }
                _ => Ok(()),
            }
        }
        fn strings<'s>(c: &'s ColumnRef) -> &'s [String] {
            match c {
                ColumnRef::Str(c) => c,
                _ => &[],
            }
        }
        let mut seen: HashMap<i64, &str> = HashMap::new();
        for lit in &self.lowered.str_literals {
            note(&mut seen, lit)?;
        }
        for col in batch.columns.values() {
            for s in strings(col) {
                note(&mut seen, s)?;
            }
            if let ColumnRef::List { fields, .. } = col {
                for (_, f) in fields {
                    for s in strings(f) {
                        note(&mut seen, s)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// A column the encoding had to materialize because the caller's data is not
/// already in the machine's representation. Boxed, so the buffer's address is
/// fixed the moment it is built and moving the owning `Vec` cannot move it.
enum DerivedColumn {
    Int(Box<[i64]>),
}

impl DerivedColumn {
    fn int(v: Vec<i64>) -> Self {
        DerivedColumn::Int(v.into_boxed_slice())
    }

    fn len(&self) -> usize {
        match self {
            DerivedColumn::Int(c) => c.len(),
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
        }
    }
}

/// Where one slot's data comes from: straight out of the caller's buffer, or
/// out of a buffer the encoding built.
enum Plan<'a> {
    Borrowed(Column<'a>),
    Derived(usize),
}

/// Encode one caller column into the bank its slot reads, checking that the
/// column is the type the schema declared.
fn encode<'a>(
    col: &'a ColumnRef<'a>,
    ty: ValType,
    path: &str,
    derived: &mut Vec<DerivedColumn>,
) -> Result<Plan<'a>, BatchError> {
    if col.val_type() != Some(ty) {
        return Err(BatchError::ColumnType {
            name: path.to_string(),
            declared: ty,
        });
    }
    // `int`, `timestamp` and `duration` are already `i64` in the machine's
    // representation and are read straight out of the caller's buffer; so is
    // `uint`, whose raw bit pattern is what the int file carries. `bool` and
    // `string` are not, so they get a materialized column.
    let buf: Vec<i64> = match col {
        ColumnRef::Int(c) | ColumnRef::Timestamp(c) | ColumnRef::Duration(c) => {
            return Ok(Plan::Borrowed(Column::Int(c)))
        }
        ColumnRef::Float(c) => return Ok(Plan::Borrowed(Column::Float(c))),
        ColumnRef::UInt(c) => {
            // SAFETY: `u64` and `i64` have the same size and alignment and every
            // bit pattern is valid for both, and the int register file carries a
            // `uint` as exactly that bit pattern (see `ValType::UInt`), so the
            // column is reinterpreted rather than copied.
            let bits = unsafe { core::slice::from_raw_parts(c.as_ptr().cast::<i64>(), c.len()) };
            return Ok(Plan::Borrowed(Column::Int(bits)));
        }
        ColumnRef::Bool(c) => c.iter().map(|&b| b as i64).collect(),
        ColumnRef::Str(c) => c.iter().map(|s| intern_hash(s)).collect(),
        ColumnRef::List { .. } => return Err(BatchError::MissingColumn(path.to_string())),
    };
    derived.push(DerivedColumn::int(buf));
    Ok(Plan::Derived(derived.len() - 1))
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
/// Not `Send`: the trace driver and the interned programs are thread-local, so
/// a bound batch belongs to the thread that bound it.
pub struct BoundBatch<'a, 'b> {
    program: &'b BatchProgram,
    /// The prepared program. `RefCell` because running writes the trap word,
    /// while `sum` takes `&self` so a caller can hold the batch across runs.
    run: std::cell::RefCell<BatchRun<'a>>,
    /// The columns the encoding materialized. Never read again — the program's
    /// base pointers address their buffers — but they must outlive the runs.
    _derived: Vec<DerivedColumn>,
}

impl BoundBatch<'_, '_> {
    /// Evaluate every row and return the running total, on [`Tier::Jit`].
    pub fn sum(&self) -> Result<Value, BatchError> {
        self.sum_on(Tier::Jit)
    }

    /// [`BoundBatch::sum`] on a chosen tier.
    pub fn sum_on(&self, tier: Tier) -> Result<Value, BatchError> {
        let threshold = match tier {
            Tier::Jit => DEFAULT_JIT_THRESHOLD,
            Tier::Interpreter | Tier::Clean => u32::MAX,
        };
        self.sum_with(tier, threshold)
    }

    /// [`BoundBatch::sum_on`] with an explicit trace threshold, for a caller
    /// measuring where the compiled tier starts to pay for itself.
    pub fn sum_with(&self, tier: Tier, threshold: u32) -> Result<Value, BatchError> {
        let lowered = &self.program.lowered;
        let raw = self.run.borrow_mut().run(|code, regs, nf| match tier {
            Tier::Clean => float_bank::clean_interp_seeded_f(code, regs, nf),
            Tier::Interpreter | Tier::Jit => {
                float_bank::run_jit_persistent_f(code, regs, nf, threshold)
            }
        });
        let raw = raw.ok_or(BatchError::Trapped)?;
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
    fn string_equality_goes_through_content_hashes() {
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
}
