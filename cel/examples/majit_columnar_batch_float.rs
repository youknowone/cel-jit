//! cell-majit **float** columnar batch evaluator vs the stock tree-walker
//! (issue #357). The `double` analog of `majit_columnar_batch`.
//!
//! Runs the REAL cel path: a CEL `Program` over `double` columns is lowered
//! (`cel::majit::lower::lower_typed`, under a schema declaring the float slots)
//! and evaluated over a batch of rows via `eval_batch_sum_f`, which reads each
//! `f64` context column at the data-dependent (red) row index through a compiled
//! `raw_load` trace — the buffer bases held loop-invariant in the register file,
//! the float values in the parallel `fregs` bank. A float comparison crosses
//! banks: `f64` operands, an int `0`/`1` result the count accumulates.
//!
//! FAIR comparison = hot vs hot. Both sides receive their data already laid out
//! (`f64` columns for the JIT, a live `Context` for the walker) and are measured
//! steady-state, with no per-row setup on either side. The baseline is the
//! tree-walker at its best: ONE reused `Context` whose variables are overwritten
//! per row, then `Program::execute`.
//!
//! Three steady-state measurements on identical data columns:
//!   naive    stock tree-walker, reused Context  — the baseline to beat
//!   JIT-off  majit bytecode interpreter tier    — lowering only, no compilation
//!   JIT-on   majit compiled trace               — the win
//! JIT-off ≈ naive is the point: the speedup comes from COMPILATION. The result
//! is bit-exact (no float tolerance); the 3-way equality gate enforces it.
//! RELEASE ONLY.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::bytecode::float_bank::COMPILES;
use cel::majit::bytecode::{eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, Schema, ValType};
use cel::{Context, Program, Value};

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// Deterministic `f64` column in `[lo, hi)`, exact f64 (`from_bits`) so the
/// tree-walker oracle compares against the very same bits.
fn make_col_f(n: usize, lo: f64, hi: f64, seed: u64) -> Vec<f64> {
    let mut x = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        let mant = x & ((1u64 << 52) - 1);
        let u = f64::from_bits((0x3ffu64 << 52) | mant) - 1.0; // [0, 1)
        v.push(lo + u * (hi - lo));
    }
    v
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ns_per_row<F: FnMut() -> i64>(n: usize, mut f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn main() {
    // Flagship float policy: a float column clears a float threshold AND another
    // stays under a float limit.
    let expr = "price >= 100.0 && qty < 50.0";
    let program = Program::compile(expr).expect("compile");
    let schema: Schema = [
        ("price".to_string(), ValType::Float),
        ("qty".to_string(), ValType::Float),
    ]
    .into_iter()
    .collect();
    let lowered = lower_typed(program.expression(), &schema).expect("lower float policy");
    let slot_paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(slot_paths, ["price", "qty"], "slot order");

    let n: usize = 2_000_000;
    let price = make_col_f(n, 0.0, 200.0, 0x2545_F491_4F6C_DD1D);
    let qty = make_col_f(n, 0.0, 100.0, 0x9E37_79B9_7F4A_7C15);
    let columns: Vec<Column> = vec![Column::Float(&price), Column::Float(&qty)];

    // FAIR baseline: the stock tree-walker at its best — reuse one Context,
    // overwrite the two float variables per row (hot; no per-row Context alloc).
    let naive = || -> i64 {
        let mut acc = 0i64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("price", price[i]);
            ctx.add_variable_from_value("qty", qty[i]);
            acc += match program.execute(&ctx).expect("execute") {
                Value::Bool(b) => b as i64,
                Value::Int(v) => v,
                other => panic!("unexpected {other:?}"),
            };
        }
        acc
    };

    // Correctness gate: naive == JIT-off == JIT-on (bit-exact; same columns).
    COMPILES.store(0, Ordering::Relaxed);
    let base = naive();
    let off = eval_batch_sum_f(&lowered, &columns, u32::MAX);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = eval_batch_sum_f(&lowered, &columns, 8);
    let on_c = COMPILES.load(Ordering::Relaxed);
    assert_eq!(base, off, "naive vs JIT-off divergence");
    assert_eq!(base, on, "naive vs JIT-on divergence -> miscompile");
    assert_eq!(off_c, 0, "JIT-off must never compile");
    assert!(on_c >= 1, "JIT-on must compile the batch loop");
    println!("policy: {expr}");
    println!(
        "n = {n}, matching rows = {on} ({:.1}%)  (all paths agree)  compiles: off={off_c} on={on_c}",
        100.0 * on as f64 / n as f64
    );

    let rounds = 5;
    let (mut on_t, mut off_t, mut naive_t) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        naive_t.push(time_ns_per_row(n, naive));
        off_t.push(time_ns_per_row(n, || eval_batch_sum_f(&lowered, &columns, u32::MAX)));
        on_t.push(time_ns_per_row(n, || eval_batch_sum_f(&lowered, &columns, 8)));
    }
    let (jit, jit_off, nv) = (median(on_t), median(off_t), median(naive_t));
    println!();
    println!("  naive   (cel tree-walk, reused ctx) : {nv:>9.2} ns/row   ← fair baseline (hot)");
    println!("  majit   JIT-off (bytecode interp)   : {jit_off:>9.2} ns/row   ← lowering only, no compile");
    println!(
        "  majit   JIT-on  (compiled trace)    : {jit:>9.2} ns/row   ← {:.0}x faster than naive  {}",
        nv / jit,
        if nv / jit > 1.0 { "✅" } else { "❌" }
    );
    println!();
    println!(
        "  => JIT-off is {:.2}x of naive, so the {:.0}x win is COMPILATION, not lowering.",
        jit_off / nv,
        nv / jit
    );
    black_box((&price, &qty));
}
