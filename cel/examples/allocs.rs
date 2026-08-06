//! Allocation census for the two collect doors, on the list-producing shape.
//!
//! Timing on this host is not trustworthy — it is shared, and a run at load
//! average 29 moved the untouched `collect_raw` control arm by 20-36%. An
//! allocation COUNT is immune to that: it is a property of the code, not of the
//! machine, so it is the same number on a quiet box and a loaded one.
//!
//! Both doors execute the identical compiled loop
//! (`BoundBatch::collect_raw_with` runs it either way); the only difference is
//! what happens to the output. `collect_on` calls `RawOutput::to_values`, which
//! builds a `Value::List(Arc<Vec<Value>>)` per row; `collect_raw_on` hands the
//! flat buffers straight to the caller. The DELTA between the two arms is
//! therefore exactly what boxing costs, with the run subtracted out.
//!
//! Two list lengths are swept so the per-row term and the per-element term
//! separate: `Arc::new(vec)` is a fixed pair of allocations per row (the `Arc`
//! control block and the `Vec` buffer), while an element is a 24-byte `Value`
//! written into that buffer and allocates only when it owns something.
//!
//! RELEASE ONLY. Run: `./bench.sh allocs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::lower::{Schema, ValType};
use cel::Value;

/// Counts every allocation the process makes. `dealloc` is deliberately not
/// counted: the question is how much work the boxing DOES, and a freed
/// allocation was still made.
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

const ROWS: usize = 50_000;

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

/// Touch every element, and every field of every element -- the traversal a
/// consumer of `collect_on` does. A strategy that boxes on ACCESS has only
/// MOVED the cost unless this arm is cheap too, so it is measured rather than
/// assumed.
fn walk(rows: &[Value]) -> i64 {
    let mut sum = 0i64;
    for row in rows {
        let Value::List(items) = row else { continue };
        for element in items.iter() {
            match element {
                Value::Int(i) => sum += i,
                Value::Map(m) => {
                    for (_, v) in m.iter() {
                        if let Value::Int(i) = *v {
                            sum += i;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    sum
}

fn main() {
    // The representation width decides what an element costs, so it is part
    // of the census rather than something to assume.
    println!(
        "size_of::<Value>() = {}, size_of::<Map>() = {}, size_of::<ListStorage>() = {}",
        std::mem::size_of::<cel::Value>(),
        std::mem::size_of::<cel::objects::Map>(),
        std::mem::size_of::<cel::objects::ListStorage>(),
    );
    println!("allocation census: collect_on vs collect_raw_on, {ROWS} rows");
    println!("both arms execute the SAME compiled loop; the delta is the boxing\n");
    println!(
        "{:<28} {:>10} {:>10} {:>12} {:>12}",
        "arm", "allocs/row", "bytes/row", "allocs total", "bytes total"
    );

    for list_len in [10i64, 40i64] {
        let lens: Vec<i64> = vec![list_len; ROWS];
        let elems: Vec<i64> = (0..ROWS as i64 * list_len)
            .map(|k| (k * 7) % 1000)
            .collect();

        let schema: Schema = [("list[]".to_string(), ValType::Int)].into_iter().collect();
        let program = BatchProgram::compile("list.map(x, x * 2)", &schema)
            .unwrap_or_else(|e| panic!("lower: {e:?}"));
        let batch = Batch::new(ROWS).column(
            "list",
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&elems))],
            },
        );
        let bound = program.bind_per_row(&batch).expect("bind");

        // Warm the tier first: tracing and compiling allocate, and they are a
        // one-off that would otherwise land in whichever arm ran first.
        let _ = bound.collect_on(Tier::Jit).expect("warmup");
        let _ = bound
            .collect_raw_on(Tier::Jit, |out| out.rows())
            .expect("warmup raw");

        println!("\n-- {list_len} elements per row --");

        reset();
        let boxed = bound.collect_on(Tier::Jit).expect("collect");
        let (a_boxed, b_boxed) = read();
        // Read the counters BEFORE dropping, so the frees are outside the window
        // and cannot be mistaken for work the arm avoided.
        assert_eq!(boxed.len(), ROWS);

        reset();
        let checksum = walk(&boxed);
        let (a_walk, b_walk) = read();
        assert_ne!(checksum, 0);
        drop(boxed);

        reset();
        let rows = bound
            .collect_raw_on(Tier::Jit, |out| out.rows())
            .expect("collect_raw");
        let (a_raw, b_raw) = read();
        assert_eq!(rows, ROWS);

        for (label, a, b) in [
            ("collect_on (boxed)", a_boxed, b_boxed),
            ("+ walk every element", a_walk, b_walk),
            ("collect_raw_on", a_raw, b_raw),
            (
                "delta = the boxing",
                (a_boxed + a_walk).saturating_sub(a_raw),
                (b_boxed + b_walk).saturating_sub(b_raw),
            ),
        ] {
            println!(
                "{:<28} {:>10.3} {:>10.1} {:>12} {:>12}",
                label,
                a as f64 / ROWS as f64,
                b as f64 / ROWS as f64,
                a,
                b
            );
        }
    }

    println!(
        "\nA boxed `Value::List` cannot cost less than 2 allocations per row: the \
         `Arc`\ncontrol block and the element buffer are separate, and each \
         element is a\n24-byte `Value` written into that buffer. An unboxed \
         `ListStorage` strategy\nshares ONE buffer across the batch and a row \
         is a slice of it, so the row\ncosts the single `Arc` and the \
         per-element term drops to the width of the\nbank -- 8 bytes for the \
         int and float banks."
    );

    record_list();
}

/// The other shape `to_values` decodes: a list of RECORDS, which
/// `collect_list_comprehension` produces whenever the chain hands the element
/// back rather than computing one — `filter` on a record list, whose fields are
/// then `source_fields(schema, path)` (`lower.rs:3304-3308`).
///
/// This arm rebuilds a `Value::Map` per element, so unlike the scalar arm it
/// allocates per element, and the count says how much of that is the map itself
/// and how much is the field NAMES.
fn record_list() {
    const LIST_LEN: i64 = 10;
    let lens: Vec<i64> = vec![LIST_LEN; ROWS];
    let n = ROWS as i64 * LIST_LEN;
    let price: Vec<i64> = (0..n).map(|k| (k * 7) % 100).collect();
    let qty: Vec<i64> = (0..n).map(|k| (k * 3) % 50).collect();

    let schema: Schema = [
        ("items[].price".to_string(), ValType::Int),
        ("items[].qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let program = match BatchProgram::compile("items.filter(i, i.price > 10)", &schema) {
        Ok(p) => p,
        Err(e) => {
            println!("\n-- record list -- NOT LOWERABLE: {e:?}");
            return;
        }
    };
    let batch = Batch::new(ROWS).column(
        "items",
        ColumnRef::List {
            lens: &lens,
            fields: vec![
                (Some("price"), ColumnRef::Int(&price)),
                (Some("qty"), ColumnRef::Int(&qty)),
            ],
        },
    );
    let bound = program.bind_per_row(&batch).expect("bind");
    let _ = bound.collect_on(Tier::Jit).expect("warmup");
    let _ = bound
        .collect_raw_on(Tier::Jit, |out| out.rows())
        .expect("warmup raw");

    println!("\n-- record list, {LIST_LEN} elements per row, 2 named fields --");

    reset();
    let boxed = bound.collect_on(Tier::Jit).expect("collect");
    let (a_boxed, b_boxed) = read();
    assert_eq!(boxed.len(), ROWS);

    reset();
    let checksum = walk(&boxed);
    let (a_walk, b_walk) = read();
    assert_ne!(checksum, 0);
    drop(boxed);

    reset();
    let rows = bound
        .collect_raw_on(Tier::Jit, |out| out.rows())
        .expect("collect_raw");
    let (a_raw, b_raw) = read();
    assert_eq!(rows, ROWS);

    for (label, a, b) in [
        ("collect_on (boxed)", a_boxed, b_boxed),
        ("+ walk every element", a_walk, b_walk),
        ("collect_raw_on", a_raw, b_raw),
        (
            "delta = the boxing",
            (a_boxed + a_walk).saturating_sub(a_raw),
            (b_boxed + b_walk).saturating_sub(b_raw),
        ),
    ] {
        println!(
            "{:<28} {:>10.3} {:>10.1} {:>12} {:>12}",
            label,
            a as f64 / ROWS as f64,
            b as f64 / ROWS as f64,
            a,
            b
        );
    }
}
