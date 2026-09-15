//! What a runtime index costs, against the route it used to take.
//!
//! `list[i]` for a non-constant `i` used to decline in `lower_typed`, and a
//! decline is not local: the whole expression falls to the tree-walker. This
//! times the two routes on the same data, interleaved round by round so drift
//! lands on both and the RATIO is the reading.
//!
//! `nums[0]` is the control and it is the point of the table: a CONSTANT index
//! always lowered, so its ratio is what the route costs with the index held
//! fixed. A runtime index reading the same ratio is the statement -- the shape
//! moved onto the route the constant one was already on, and the number is the
//! route's, not the index's.
//!
//! The walker leg builds one activation per row, which is what the row-by-row
//! fallback door does, so most of its time is the activation and not the read.
//!
//! ```text
//! cargo run --release --package cel --no-default-features \
//!   --features regex,chrono,jit-cranelift --example rca_indexcost
//! ```

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::reset_persistent_state;
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

const ROWS: usize = 4096;
const LEN: i64 = 8;

fn main() {
    let lens = vec![LEN; ROWS];
    let nums: Vec<i64> = (0..ROWS as i64 * LEN).map(|i| i % 97).collect();
    let xs: Vec<i64> = (0..ROWS as i64).collect();
    let schema: Schema = [
        ("nums[]".to_string(), ValType::Int),
        ("x".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let batch = Batch::new(ROWS)
        .column(
            "nums".to_string(),
            ColumnRef::List {
                lens: &lens,
                fields: vec![(None, ColumnRef::Int(&nums))],
            },
        )
        .column("x".to_string(), ColumnRef::Int(&xs));

    for src in [
        "nums[x % 8]",
        "nums[x % 4] + nums[x % 8]",
        "nums[x % 8] * 3 + 1",
        "nums[0]",
    ] {
        let program = Program::compile(src).unwrap();
        let lowered = BatchProgram::compile(src, &schema).unwrap();
        let bound = lowered.bind_per_row(&batch).unwrap();
        reset_persistent_state();
        for _ in 0..8 {
            bound.collect_on(Tier::Jit).unwrap();
        }
        // The walker's own route: one activation per row, as the fallback runs.
        let walk = || {
            let mut off = 0usize;
            let mut acc = 0i64;
            for (row, len) in lens.iter().enumerate() {
                let n = *len as usize;
                let list: Vec<Value> = nums[off..off + n].iter().map(|v| Value::Int(*v)).collect();
                off += n;
                let mut ctx = Context::default();
                ctx.add_variable_from_value("nums", Value::list(list));
                ctx.add_variable_from_value("x", xs[row]);
                if let Ok(Value::Int(i)) = program.execute(&ctx) {
                    acc += i;
                }
            }
            acc
        };
        let mut best = [f64::INFINITY; 2];
        for _ in 0..10 {
            let t = std::time::Instant::now();
            for _ in 0..4 {
                bound.collect_on(Tier::Jit).unwrap();
            }
            let ns = t.elapsed().as_nanos() as f64 / (4 * ROWS) as f64;
            best[0] = best[0].min(ns);
            let t = std::time::Instant::now();
            std::hint::black_box(walk());
            let ns = t.elapsed().as_nanos() as f64 / ROWS as f64;
            best[1] = best[1].min(ns);
        }
        println!(
            "  {src:28}  jit {:8.2} ns/row   walker {:9.2} ns/row   {:6.1}x",
            best[0],
            best[1],
            best[1] / best[0]
        );
    }
}
