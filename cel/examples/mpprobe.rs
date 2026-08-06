//! Minimal single-shape probe for merge-point / compile diagnostics.
//!
//! One flat row loop, one mainloop call, a row count small enough that
//! `MAJIT_MPTRACE=1` / `MAJIT_LOG=1` output stays readable. Prints the same
//! census row `examples/jitstats.rs` prints, so the two agree by construction.
//!
//! ```text
//! MAJIT_MPTRACE=1 MAJIT_LOG=1 CEL_BACKEND=cranelift ./bench.sh mpprobe 64
//! ```

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, Schema, ValType};
use cel::Program;

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let rounds: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let src = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "a + b * 2".to_string());

    let schema: Schema = [
        ("a".to_string(), ValType::Int),
        ("b".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let program = Program::compile(&src).expect("parse");
    let lowered = lower_typed(program.expression(), &schema).expect("lower");

    let a: Vec<i64> = (0..n as i64).map(|i| (i * 37) % 1000).collect();
    let b: Vec<i64> = (0..n as i64).map(|i| (i * 11) % 1000).collect();
    let columns = [Column::Int(&a), Column::Int(&b)];

    reset_persistent_state();
    reset_jit_stats();
    let clean = clean_batch_sum_f(&lowered, &columns, n);
    for r in 0..rounds {
        let jit = eval_batch_sum_f(&lowered, &columns, n, 8);
        assert_eq!(clean, jit, "round {r}: compiled tier diverged");
        eprintln!("@@@ROUND {r} done");
    }
    let s = jit_stats();
    println!("mpprobe n={n} rounds={rounds} src=`{src}` -> {clean:?}");
    println!("{s}");
}
