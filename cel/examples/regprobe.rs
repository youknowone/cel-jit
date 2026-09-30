//! Dump the int/float register widths and the emitted per-row program for the
//! three comprehension ladder shapes, so a register-allocation change can be
//! graded on counters rather than on wall clock.

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::lower::{BatchReduce, Schema, ValType};

/// Run one shape in the per-call regime — a ONE-row batch whose single list
/// column holds `elems` elements — until the tier is compiling, so
/// `MAJIT_DUMP_CLIF=1` dumps that shape's loop and nothing else.
fn run_jit(src: &str, col: &str, elems: usize, calls: usize) {
    let schema: Schema = [(format!("{col}[]"), ValType::Int)].into_iter().collect();
    let program = BatchProgram::compile(src, &schema).expect("ladder shape lowers");
    let values: Vec<i64> = (1..=elems as i64).collect();
    let lens = [elems as i64];
    let batch = Batch::new(1).column(
        col.to_string(),
        ColumnRef::List {
            lens: &lens,
            fields: vec![(None, ColumnRef::Int(&values))],
        },
    );
    let bound = program.bind_per_row(&batch).expect("the column binds");
    for _ in 0..calls {
        bound.collect_on(Tier::Jit).expect("the row evaluates");
    }
}

fn main() {
    if let Some(shape) = std::env::args().nth(1) {
        let (src, col) = match shape.as_str() {
            "map" => ("list.map(x, x * 2)", "list"),
            "filter" => ("list.filter(x, x % 2 == 0)", "list"),
            "chained" => ("items.filter(x, x % 2 == 0).map(x, x * 2)", "items"),
            other => panic!("unknown shape `{other}`"),
        };
        run_jit(src, col, 500, 4096);
        return;
    }
    let shapes = [
        ("map", "list.map(x, x * 2)", "list"),
        ("filter", "list.filter(x, x % 2 == 0)", "list"),
        (
            "chained",
            "items.filter(x, x % 2 == 0).map(x, x * 2)",
            "items",
        ),
    ];
    for (name, src, col) in shapes {
        let schema: Schema = [(format!("{col}[]"), ValType::Int)].into_iter().collect();
        let program = BatchProgram::compile(src, &schema).expect("ladder shape lowers");
        let lowered = program.lowered();
        let shape = lowered.batch_shape(true, BatchReduce::PerRow);
        // The machinery bank the batch builder appends above the body's own:
        // `r_i, r_acc, r_n, r_ea, r_trap, r_out` plus one base per slot.
        let total_int = lowered.num_int_regs + 6 + lowered.slots.len();
        println!("== {name}: {src}");
        println!(
            "body_int={} body_float={} slots={} total_int={} total_float={}",
            lowered.num_int_regs,
            lowered.num_float_regs,
            lowered.slots.len(),
            total_int,
            shape.num_float_regs
        );
        for s in &lowered.slots {
            println!("slot {} reg={} ty={:?}", s.path, s.reg, s.ty);
        }
        println!("code {} words", shape.code.len());
        for (i, w) in shape.code.iter().enumerate() {
            println!("w {i} {w}");
        }
        println!("== end {name}");
    }
}
