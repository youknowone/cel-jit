//! cell-majit columnar batch evaluator vs the stock tree-walker (issue #357).
//!
//! Runs the REAL cel path: a CEL `Program` is lowered (`cel::majit::lower::lower`)
//! and evaluated over a batch of rows via `eval_batch_sum`, which reads each
//! context column at the data-dependent (red) row index through a compiled
//! `raw_load` trace, the buffer bases held loop-invariant in the register file.
//!
//! FAIR comparison = hot vs hot. Both sides receive their data already laid out
//! (i64 columns for the JIT, a live `Context` for the walker) and are measured
//! steady-state, with no per-row setup on either side. The baseline is therefore
//! the tree-walker at its best: ONE reused `Context` whose variables are
//! overwritten per row, then `Program::execute`. We deliberately do NOT compare
//! against a fresh-`Context`-per-row walker — that pays a per-row allocation the
//! JIT never does (cold vs hot), which would flatter the JIT dishonestly.
//!
//! Three steady-state measurements on identical data columns:
//!   naive    stock tree-walker, reused Context  — the baseline to beat
//!   JIT-off  majit bytecode interpreter tier    — lowering only, no compilation
//!   JIT-on   majit compiled trace               — the win
//! JIT-off ≈ naive is the point: the speedup comes from COMPILATION, not from
//! lowering the expression to integer bytecode. RELEASE ONLY (i64 wrap; 3-way
//! equality gate).

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::bytecode::{eval_batch_sum, COMPILES};
use cel::majit::lower::lower;
use cel::{Context, Program, Value};

const LCG_A: i64 = 6364136223846793005;
const LCG_C: i64 = 1442695040888963407;

/// Deterministic column: an LCG mapped into `[lo, hi]`.
fn make_col(n: usize, lo: i64, hi: i64, seed: i64) -> Vec<i64> {
    let span = (hi - lo + 1) as u64;
    let mut x = seed as u64;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        x = x.wrapping_mul(LCG_A as u64).wrapping_add(LCG_C as u64);
        v.push(lo + ((x >> 33) % span) as i64);
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
    // Flagship policy predicate over three int/bool columns.
    let expr = "balance >= amount && !frozen";
    let program = Program::compile(expr).expect("compile");
    let lowered = lower(program.expression()).expect("lower policy to majit subset");
    let slot_paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(slot_paths, ["balance", "amount", "frozen"], "slot order");

    let n: usize = 2_000_000;
    let balance = make_col(n, -1_000_000, 1_000_000, 0x2545F491);
    let amount = make_col(n, -1_000_000, 1_000_000, 0x9E3779B9);
    let frozen = make_col(n, 0, 1, 0x1000_0001);
    let columns: Vec<&[i64]> = vec![&balance, &amount, &frozen];

    // FAIR baseline: the stock tree-walker at its best — reuse one Context,
    // overwrite the three variables per row (hot; no per-row Context alloc).
    let naive = || -> i64 {
        let mut acc = 0i64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("balance", balance[i]);
            ctx.add_variable_from_value("amount", amount[i]);
            ctx.add_variable_from_value("frozen", frozen[i] != 0);
            acc += match program.execute(&ctx).expect("execute") {
                Value::Bool(b) => b as i64,
                Value::Int(v) => v,
                other => panic!("unexpected {other:?}"),
            };
        }
        acc
    };

    // Correctness gate: naive == JIT-off == JIT-on (all read the same columns).
    COMPILES.store(0, Ordering::Relaxed);
    let base = naive();
    let off = eval_batch_sum(&lowered, &columns, u32::MAX);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = eval_batch_sum(&lowered, &columns, 8);
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
        off_t.push(time_ns_per_row(n, || eval_batch_sum(&lowered, &columns, u32::MAX)));
        on_t.push(time_ns_per_row(n, || eval_batch_sum(&lowered, &columns, 8)));
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
    black_box((&balance, &amount, &frozen));
}
