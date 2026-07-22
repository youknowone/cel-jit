//! cell-majit columnar batch evaluator vs the stock tree-walker (issue #357).
//!
//! The de-risk examples (`majit/examples/celcolumn`, `majit_vs_cometkim`)
//! established the perf envelope on hand-written bytecode. This runs the REAL
//! cel path end to end: a CEL `Program` is lowered
//! (`cel::majit::lower::lower`) and evaluated over a batch of rows via
//! `eval_batch_sum`, which reads each context column at the data-dependent
//! (red) row index through a compiled `raw_load` trace — the buffer bases held
//! loop-invariant in the register file.
//!
//! The honest baseline is what a cel user runs TODAY: `Program::execute` once
//! per row against a freshly built `Context` (the tree-walker, with its
//! per-row map inserts + trait-object dispatch + `HashMap` variable lookups).
//! The standing goal is for the JIT batch to beat that naive per-row path.
//!
//! Four measurements on identical data columns:
//!   (a) majit batch JIT-on    — `eval_batch_sum(threshold = small)`
//!   (b) naive cel per-row      — fresh `Context` + `Program::execute` each row
//!   (b2) reuse-ctx cel per-row — one `Context`, overwrite vars + `execute` each row
//!   (c) majit batch JIT-off    — `eval_batch_sum(threshold = u32::MAX)`
//! (b) is what a cel user writes first; (b2) is the same user's obvious
//! optimization (don't rebuild the map). The JIT must beat both. Meaningful
//! ratios = (b)/(a) and (b2)/(a). RELEASE ONLY (i64 wrap; 3-way equality gate).

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

    let fold = |v: Value| -> i64 {
        match v {
            Value::Bool(b) => b as i64,
            Value::Int(v) => v,
            other => panic!("unexpected {other:?}"),
        }
    };

    // Honest baseline: the stock tree-walker, once per row, over a fresh Context.
    let naive = || -> i64 {
        let mut acc = 0i64;
        for i in 0..n {
            let mut ctx = Context::default();
            ctx.add_variable_from_value("balance", balance[i]);
            ctx.add_variable_from_value("amount", amount[i]);
            ctx.add_variable_from_value("frozen", frozen[i] != 0);
            acc += fold(program.execute(&ctx).expect("execute"));
        }
        acc
    };

    // Optimized baseline: reuse one Context, overwrite the three variables per row.
    let naive_reuse = || -> i64 {
        let mut acc = 0i64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("balance", balance[i]);
            ctx.add_variable_from_value("amount", amount[i]);
            ctx.add_variable_from_value("frozen", frozen[i] != 0);
            acc += fold(program.execute(&ctx).expect("execute"));
        }
        acc
    };

    // Correctness gate: all four paths agree (all read the same columns).
    COMPILES.store(0, Ordering::Relaxed);
    let base = naive();
    let base_reuse = naive_reuse();
    let off = eval_batch_sum(&lowered, &columns, u32::MAX);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = eval_batch_sum(&lowered, &columns, 8);
    let on_c = COMPILES.load(Ordering::Relaxed);
    assert_eq!(base, base_reuse, "naive vs reuse-ctx divergence");
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
    let (mut a, mut b, mut b2, mut c) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        b.push(time_ns_per_row(n, naive));
        b2.push(time_ns_per_row(n, naive_reuse));
        c.push(time_ns_per_row(n, || eval_batch_sum(&lowered, &columns, u32::MAX)));
        a.push(time_ns_per_row(n, || eval_batch_sum(&lowered, &columns, 8)));
    }
    let (a, b, b2, c) = (median(a), median(b), median(b2), median(c));
    println!("(a)  majit batch JIT-on    : {a:.3} ns/row");
    println!("(b)  naive cel per-row     : {b:.3} ns/row");
    println!("(b2) reuse-ctx cel per-row : {b2:.3} ns/row");
    println!("(c)  majit batch JIT-off   : {c:.3} ns/row");
    println!();
    println!("ratio (b)/(a)  fresh-ctx naive vs JIT : {:.1}x", b / a);
    println!("ratio (b2)/(a) reuse-ctx naive vs JIT : {:.1}x", b2 / a);
    println!("ratio (c)/(a)  majit interp vs JIT    : {:.1}x", c / a);
    println!(
        "goal (batch JIT beats BOTH naive paths): {}",
        if b / a > 1.0 && b2 / a > 1.0 { "PASS" } else { "FAIL" }
    );
    black_box((&balance, &amount, &frozen));
}
