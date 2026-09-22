//! Task #59: the per-bind allocation census for the columnar `encode` path.
//!
//! Task #58 priced `bind` at ~100 ns FIXED per bind — invariant in the row
//! count across three runs — and named the mechanism: a small, constant number
//! of heap allocations, with no single named stage over 12% of the whole. The
//! lever is therefore the allocation COUNT, not any one stage, and this file is
//! the instrument that grades it.
//!
//! An allocation COUNT is a property of the code, not of the machine, so it is
//! the same number on a quiet box and on one at load average 70. That is why it
//! is the primary gate for #59 and why nothing here reports a time.
//!
//! Four doors are counted per shape, because `bind` is two halves and both the
//! whole and each half are separately reachable from the public API:
//!
//!     bind                  resolve + encode, summing
//!     bind_per_row          resolve + encode, storing per row
//!     resolve               the first half alone (one map lookup per path)
//!     bind_per_row_resolved the second half alone — the ENCODE this task cuts
//!
//! Each is measured on its own `reset()`/`read()` window with the counters read
//! BEFORE the `BoundBatch` is dropped, so a freed allocation still counts as
//! work the bind did. The doors are all warmed once first:
//! `LoweredF::batch_shape` memoizes its `BatchShape` in a `OnceLock`, so the
//! FIRST bind of a program builds the word stream and no later one does — an
//! unwarmed first row would report that one-off as if it were per-bind.
//!
//! RELEASE ONLY. Run: `./bench.sh bindallocs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cel::majit::batch::{Batch, BatchProgram, ColumnRef};
use cel::majit::lower::{Schema, ValType};

/// Counts every allocation the process makes. `dealloc` is deliberately not
/// counted: the question is how much work the bind DOES, and a transient buffer
/// that is freed before the bind returns was still allocated.
struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Swept so that a count which SCALES separates from one that is fixed. #58's
/// claim is invariance, and only a sweep can refute it.
const SIZES: &[usize] = &[1, 10, 100];

fn reset() {
    ALLOCS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
}

fn read() -> (usize, usize) {
    (
        ALLOCS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

fn header() {
    println!(
        "{:<30} {:>5} {:>5} {:>5} {:>6} {:>7} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "shape", "slot", "seed", "rows", "bind", "B", "per_row", "B", "resolv", "B", "encode", "B"
    );
}

/// One shape at one row count, through all four doors.
///
/// The `Batch` and the program are built by the caller and are OUTSIDE every
/// window: constructing the column map allocates, and that is the caller's cost
/// rather than the bind's.
fn census(shape: &str, source: &str, program: &BatchProgram, batch: &Batch<'_>, rows: usize) {
    // Warm every door once. Only the first bind of a program pays for
    // `batch_shape`'s word stream; measuring an unwarmed door would report that
    // one-off as a per-bind cost.
    let sum_ok = program.bind(batch).is_ok();
    program.bind_per_row(batch).expect("binds per row");
    let resolved = program.resolve(batch).expect("resolves");
    program
        .bind_per_row_resolved(&resolved)
        .expect("encodes over a resolution");

    // A sum over a result the loop cannot accumulate — a list, a string — has
    // no `bind` door at all, and the row says so rather than reporting a zero
    // that would read as "free".
    let sum = if sum_ok {
        reset();
        let bound = program.bind(batch).expect("binds");
        let counts = read();
        drop(bound);
        Some(counts)
    } else {
        None
    };

    reset();
    let bound = program.bind_per_row(batch).expect("binds per row");
    let per_row = read();
    drop(bound);

    reset();
    let taken = program.resolve(batch).expect("resolves");
    let resolve = read();
    drop(taken);

    reset();
    let bound = program
        .bind_per_row_resolved(&resolved)
        .expect("encodes over a resolution");
    let encode = read();
    drop(bound);

    let (sa, sb) = sum.unwrap_or((usize::MAX, usize::MAX));
    let cell = |v: usize| {
        if v == usize::MAX {
            "n/a".to_string()
        } else {
            v.to_string()
        }
    };
    // The slot and seed counts are what every transient buffer in the encode
    // path is sized by, so they are reported beside the counts rather than
    // inferred from them: they are what decides whether an inline buffer of a
    // given width covers a shape or spills.
    println!(
        "{:<30} {:>5} {:>5} {:>5} {:>6} {:>7} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}",
        if rows == SIZES[0] { source } else { shape },
        program.lowered().slots.len(),
        program.lowered().scalar_seeds.len(),
        rows,
        cell(sa),
        cell(sb),
        per_row.0,
        per_row.1,
        resolve.0,
        resolve.1,
        encode.0,
        encode.1,
    );
}

fn main() {
    println!("per-bind allocation census for the columnar encode path (task #59)");
    println!("counts are ONE bind through each door; `B` is bytes requested\n");
    header();

    scalar_int();
    two_int();
    scalar_float();
    predicate();
    string_eq();
    string_result();
    list_size();
    list_map();
    concat();
    wide();

    println!(
        "\n`bind` = resolve + encode summing; `per_row` = resolve + encode storing.\n\
         A count that does not move with `rows` is a FIXED per-bind cost, which\n\
         is what #58 measured and what #59 cuts."
    );
}

/// The shape #58 split by amplification: one int column, no list, no string —
/// the least there is to encode, and still ~100 ns.
fn scalar_int() {
    let schema: Schema = [("x".to_string(), ValType::Int)].into_iter().collect();
    let program = BatchProgram::compile("x * 2 + 1", &schema).expect("lowers");
    for &rows in SIZES {
        let vals: Vec<i64> = (0..rows as i64).collect();
        let batch = Batch::new(rows).column("x", ColumnRef::Int(&vals));
        census("(int scalar)", "x * 2 + 1", &program, &batch, rows);
    }
}

/// Two slots rather than one, so a per-SLOT term separates from a per-bind one.
fn two_int() {
    let schema: Schema = [
        ("a".to_string(), ValType::Int),
        ("b".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let program = BatchProgram::compile("a * 2 + b", &schema).expect("lowers");
    for &rows in SIZES {
        let a: Vec<i64> = (0..rows as i64).collect();
        let b: Vec<i64> = (0..rows as i64).map(|k| k % 7).collect();
        let batch = Batch::new(rows)
            .column("a", ColumnRef::Int(&a))
            .column("b", ColumnRef::Int(&b));
        census("(int, 2 slots)", "a * 2 + b", &program, &batch, rows);
    }
}

/// The float bank, whose result accumulates into a float register rather than
/// the int one — a different shape, the same buffers.
fn scalar_float() {
    let schema: Schema = [("x".to_string(), ValType::Float)].into_iter().collect();
    let program = BatchProgram::compile("x * 2.0 + 1.0", &schema).expect("lowers");
    for &rows in SIZES {
        let vals: Vec<f64> = (0..rows).map(|k| k as f64).collect();
        let batch = Batch::new(rows).column("x", ColumnRef::Float(&vals));
        census("(float scalar)", "x * 2.0 + 1.0", &program, &batch, rows);
    }
}

/// A boolean result, which a sum counts rather than adds.
fn predicate() {
    let schema: Schema = [("x".to_string(), ValType::Int)].into_iter().collect();
    let program = BatchProgram::compile("x > 5", &schema).expect("lowers");
    for &rows in SIZES {
        let vals: Vec<i64> = (0..rows as i64).collect();
        let batch = Batch::new(rows).column("x", ColumnRef::Int(&vals));
        census("(bool predicate)", "x > 5", &program, &batch, rows);
    }
}

/// A string COLUMN, which the encoding ranks: this is the shape whose `str_ids`
/// and dictionary are real work rather than an empty collect.
fn string_eq() {
    let schema: Schema = [("name".to_string(), ValType::Str)].into_iter().collect();
    let program = BatchProgram::compile("name == \"k-3\"", &schema).expect("lowers");
    for &rows in SIZES {
        let names: Vec<String> = (0..rows).map(|k| format!("k-{}", k % 8)).collect();
        let batch = Batch::new(rows).column("name", ColumnRef::Str(&names));
        census("(string column)", "name == \"k-3\"", &program, &batch, rows);
    }
}

/// A string RESULT, which additionally owns one `String` per distinct string in
/// the batch so the ids can be decoded — a count that is expected to scale with
/// the batch's distinct strings and not with its rows.
fn string_result() {
    let schema: Schema = [("name".to_string(), ValType::Str)].into_iter().collect();
    let program = BatchProgram::compile("name", &schema).expect("lowers");
    for &rows in SIZES {
        let names: Vec<String> = (0..rows).map(|k| format!("k-{}", k % 8)).collect();
        let batch = Batch::new(rows).column("name", ColumnRef::Str(&names));
        census("(string result)", "name", &program, &batch, rows);
    }
}

/// A list column read only for its length, which the encoding MATERIALIZES: the
/// caller's `lens` buffer is copied into a derived column the bound batch owns.
fn list_size() {
    let schema: Schema = [("nums[]".to_string(), ValType::Int)].into_iter().collect();
    let program = BatchProgram::compile("size(nums)", &schema).expect("lowers");
    for &rows in SIZES {
        let lens: Vec<i64> = vec![4; rows];
        let elems: Vec<i64> = (0..rows as i64 * 4).collect();
        let batch = Batch::new(rows).column(
            "nums",
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&elems))],
            },
        );
        census("(list size)", "size(nums)", &program, &batch, rows);
    }
}

/// A list-valued result, whose per-bind cost includes one output buffer per
/// output field sized by summing the source's `size(..)` column.
fn list_map() {
    let schema: Schema = [("nums[]".to_string(), ValType::Int)].into_iter().collect();
    let program = BatchProgram::compile("nums.map(v, v * 2)", &schema).expect("lowers");
    for &rows in SIZES {
        let lens: Vec<i64> = vec![4; rows];
        let elems: Vec<i64> = (0..rows as i64 * 4).collect();
        let batch = Batch::new(rows).column(
            "nums",
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&elems))],
            },
        );
        census("(list map)", "nums.map(v, v * 2)", &program, &batch, rows);
    }
}

/// A per-row STRING CONCATENATION, the shape `majit_columnar_batch_per_row`
/// runs: the encoding builds the characters, so this is the shape with the most
/// derived work per bind.
fn concat() {
    let schema: Schema = [
        ("role".to_string(), ValType::Str),
        ("region".to_string(), ValType::Str),
    ]
    .into_iter()
    .collect();
    let program = BatchProgram::compile("role + \"@\" + region", &schema).expect("lowers");
    for &rows in SIZES {
        let role: Vec<String> = (0..rows).map(|k| format!("r-{}", k % 4)).collect();
        let region: Vec<String> = (0..rows).map(|k| format!("z-{}", k % 3)).collect();
        let batch = Batch::new(rows)
            .column("role", ColumnRef::Str(&role))
            .column("region", ColumnRef::Str(&region));
        census(
            "(string concat)",
            "role + \"@\" + region",
            &program,
            &batch,
            rows,
        );
    }
}

/// Six slots, wider than any expression the examples carry, so a buffer sized
/// for the common case is exercised on the shape that OVERFLOWS it. A spill
/// must cost what a plain `Vec` costs and no more — never a panic and never a
/// truncated column list.
fn wide() {
    let names = ["a", "b", "c", "d", "e", "f"];
    let schema: Schema = names
        .iter()
        .map(|n| (n.to_string(), ValType::Int))
        .collect();
    let program = BatchProgram::compile("a + b + c + d + e + f", &schema).expect("lowers");
    for &rows in SIZES {
        let cols: Vec<Vec<i64>> = (0..6)
            .map(|j| (0..rows as i64).map(|k| k + j).collect())
            .collect();
        let mut batch = Batch::new(rows);
        for (n, c) in names.iter().zip(&cols) {
            batch = batch.column(*n, ColumnRef::Int(c));
        }
        census("(6 slots)", "a + b + c + d + e + f", &program, &batch, rows);
    }
}
