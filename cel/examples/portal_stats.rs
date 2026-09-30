//! Census the interpreter portal on a reused `Program`.
//!
//! ```text
//! CEL_PORTAL_THRESHOLD=8 MAJIT_STATS=1 \
//!   cargo run -p cel --release --features jit-dynasm --example portal_stats -- 64 20
//! ```
//!
//! First arg is list length (back-edge heat). Second is execute repeats
//! (function-entry heat). Without a persisted driver both counters reset
//! every call and nothing compiles.

use cel::{Context, Program};

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let repeats: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let xs: Vec<i64> = (0..n as i64).collect();
    let mut ctx = Context::default();
    ctx.add_variable_from_value("xs", xs);
    let program = Program::compile("xs.map(x, x + 1)").expect("compile");
    for _ in 0..repeats {
        let _ = program.execute(&ctx).expect("eval");
    }
    println!("ok n={n} repeats={repeats}");
}
