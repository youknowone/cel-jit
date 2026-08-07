//! #91 probe: is the nested `all()` deopt count a function of ROWS or of the
//! number of guards that reach `trace_eagerness`?
//!
//! Prints `bridges` next to `guard` for the nested sweep at several row counts.
//! Temporary RCA instrument; delete once #91 is written up.

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

const THRESHOLD: u32 = 8;

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

fn run(label: &str, lowered: &LoweredF, lens: &[i64], rows: usize) {
    let mut offsets = Vec::with_capacity(rows);
    let mut total = 0i64;
    for &l in lens {
        offsets.push(total);
        total += l;
    }
    let elems: Vec<i64> = (0..total.max(1)).map(|k| (k * 7) % 40).collect();
    let columns = [
        Column::Int(lens),
        Column::Int(&offsets),
        Column::Int(&elems),
    ];

    reset_persistent_state();
    reset_jit_stats();
    let clean = clean_batch_sum_f(lowered, &columns, rows);
    let jit = eval_batch_sum_f(lowered, &columns, rows, THRESHOLD);
    assert_eq!(clean, jit, "{label}: miscompile, not a census");
    let s = jit_stats();
    let predicted = 200 * s.bridges_compiled + 1;
    // The budget `majit_trace_evidence.rs` gates on.
    let budget = 200 * (s.bridges_compiled + 2) + 1;
    println!(
        "{label:<28} rows={rows:>7} loops={} bridges={} guard={} abrt={} \
         ops={}->{} result={:?}  E*b+1={predicted} budget={budget} {}",
        s.loops_compiled,
        s.bridges_compiled,
        s.guard_failures,
        s.loops_aborted,
        s.trace_ops_before,
        s.trace_ops_after,
        jit,
        if s.guard_failures <= budget {
            "PASS"
        } else {
            "*** GATE FIRES ***"
        },
    );
}

fn main() {
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let lowered = lower("items.all(i, i.price > 10)", &schema);

    for per_row in [0i64, 1, 2, 3, 4, 5, 8, 9] {
        for rows in [4_000usize, 20_000, 100_000] {
            let lens = vec![per_row; rows];
            run(
                &format!("constant per_row={per_row}"),
                &lowered,
                &lens,
                rows,
            );
        }
    }

    let shapes: [(&str, fn(usize) -> i64); 3] = [
        ("alternating 8/9", |r| if r % 2 == 0 { 8 } else { 9 }),
        ("cycle 4..12", |r| 4 + (r % 9) as i64),
        ("spread 0..32", |r| ((r * 2654435761) % 32) as i64),
    ];
    for (label, len_of) in shapes {
        for rows in [4_000usize, 20_000, 100_000] {
            let lens: Vec<i64> = (0..rows).map(len_of).collect();
            run(label, &lowered, &lens, rows);
        }
    }

    // NON-VACUITY CONTROL for the `warmup_budget` gate in
    // `tests/majit_trace_evidence.rs`.
    //
    // Widening the trip-count spread starves the warmup: a guard has to fail
    // `trace_eagerness` times before its bridge is attached, so once the spread
    // is wide enough that no single exit recurs that often within the batch,
    // nothing bridges and every exit is a raw bail back to the interpreter.
    // That is a per-row bail, produced from the data column alone with no source
    // change — exactly the regression the budget exists to catch.
    //
    // Row counts are capped here because these shapes allocate `rows * mean_len`
    // elements.
    let wide: [(&str, fn(usize) -> i64, usize); 4] = [
        ("spread 0..200", |r| ((r * 2654435761) % 200) as i64, 4_000),
        ("spread 0..200", |r| ((r * 2654435761) % 200) as i64, 20_000),
        (
            "spread 0..2000",
            |r| ((r * 2654435761) % 2000) as i64,
            4_000,
        ),
        (
            "spread 0..2000",
            |r| ((r * 2654435761) % 2000) as i64,
            20_000,
        ),
    ];
    for (label, len_of, rows) in wide {
        let lens: Vec<i64> = (0..rows).map(len_of).collect();
        run(label, &lowered, &lens, rows);
    }
}
