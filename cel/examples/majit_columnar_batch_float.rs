//! cell-majit **float** columnar batch evaluator vs the stock tree-walker
//! (issue #357). The `double` analog of `majit_columnar_batch`.
//!
//! This is an explicit **cross-model batch experiment**, not the default fair
//! CEL JIT benchmark. Pre-transposed columns plus a fused row loop make the
//! stock/JIT ratio unsuitable as a request-latency or JIT-only claim. Run
//! `./bench.sh` for the fair request/engine/cold suite.
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
//! Four steady-state measurements over one prebuilt batch program:
//!   stock    stock tree-walker, reused Context
//!   clean VM plain Rust bytecode interpreter, no tracing/JIT machinery
//!   JIT-off  majit tracing interpreter, compilation disabled
//!   JIT-on   majit compiled trace
//! This separates the representation/lowering win (`stock / clean VM`) from
//! the compilation win (`clean VM / JIT-on`). The result is bit-exact (no float
//! tolerance); a 4-way equality gate enforces it. RELEASE ONLY.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::COMPILES;
use cel::majit::lower::{Schema, ValType};
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

fn time_ns_per_row<T, F: FnMut() -> T>(n: usize, mut f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

/// The batch API answers in CEL's types; the predicate panel counts matching
/// rows and the aggregate panel totals a float.
fn count(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("unexpected batch result {other:?}"),
    }
}

fn total(v: Value) -> f64 {
    match v {
        Value::Float(f) => f,
        other => panic!("unexpected batch result {other:?}"),
    }
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
    let lowered = BatchProgram::compile(expr, &schema).expect("lower float policy");

    let n: usize = 2_000_000;
    let price = make_col_f(n, 0.0, 200.0, 0x2545_F491_4F6C_DD1D);
    let qty = make_col_f(n, 0.0, 100.0, 0x9E37_79B9_7F4A_7C15);
    let batch = Batch::new(n)
        .column("price", ColumnRef::Float(&price))
        .column("qty", ColumnRef::Float(&qty));
    let bound = lowered.bind(&batch).expect("bind columns");

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

    // Correctness gate: stock == clean VM == JIT-off == JIT-on.
    COMPILES.store(0, Ordering::Relaxed);
    let base = naive();
    let clean = count(bound.sum_on(Tier::Clean).expect("clean tier"));
    let off = count(bound.sum_on(Tier::Interpreter).expect("interp tier"));
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = count(bound.sum_on(Tier::Jit).expect("jit tier"));
    let on_c = COMPILES.load(Ordering::Relaxed);
    assert_eq!(base, clean, "stock vs clean VM divergence");
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
    let (mut on_t, mut off_t, mut clean_t, mut naive_t) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        naive_t.push(time_ns_per_row(n, naive));
        clean_t.push(time_ns_per_row(n, || {
            count(bound.sum_on(Tier::Clean).expect("clean tier"))
        }));
        off_t.push(time_ns_per_row(n, || {
            count(bound.sum_on(Tier::Interpreter).expect("interp tier"))
        }));
        on_t.push(time_ns_per_row(n, || {
            count(bound.sum_on(Tier::Jit).expect("jit tier"))
        }));
    }
    let (jit, jit_off, vm, nv) = (
        median(on_t),
        median(off_t),
        median(clean_t),
        median(naive_t),
    );
    println!();
    println!("  stock   (cel tree-walk, reused ctx) : {nv:>9.2} ns/row");
    println!("  clean VM(lowered bytecode, no JIT)  : {vm:>9.2} ns/row");
    println!("  majit   (tracing interp, JIT off)    : {jit_off:>9.2} ns/row");
    println!("  majit   (compiled trace, JIT on)     : {jit:>9.2} ns/row");
    println!();
    println!(
        "  VM/data-model effect  stock / clean VM : {:>8.2}x",
        nv / vm
    );
    println!(
        "  JIT effect          clean VM / JIT-on  : {:>8.2}x",
        vm / jit
    );
    println!(
        "  majit tier delta     JIT-off / JIT-on  : {:>8.2}x",
        jit_off / jit
    );
    println!(
        "  cross-model batch    stock / JIT-on    : {:>8.2}x",
        nv / jit
    );

    // --- Float AGGREGATE: a float-valued result summed into a float
    // accumulator (OP_RETURN_F). `sum(price * qty)` folds the per-row FMUL into
    // a register and carries the running total across the loop. Same columns,
    // bit-exact (float addition is order-sensitive, so the loop and the oracle
    // both sum left to right).
    let agg_expr = "price * qty";
    let agg_program = Program::compile(agg_expr).expect("compile aggregate");
    let agg_lowered = BatchProgram::compile(agg_expr, &schema).expect("lower float aggregate");
    assert_eq!(
        agg_lowered.result_type(),
        ValType::Float,
        "aggregate must be float-valued"
    );
    let agg_bound = agg_lowered.bind(&batch).expect("bind aggregate columns");

    let naive_agg = || -> f64 {
        let mut acc = 0.0f64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("price", price[i]);
            ctx.add_variable_from_value("qty", qty[i]);
            acc += match agg_program.execute(&ctx).expect("execute") {
                Value::Float(v) => v,
                other => panic!("unexpected {other:?}"),
            };
        }
        acc
    };

    let base_a = naive_agg();
    let clean_a = total(agg_bound.sum_on(Tier::Clean).expect("clean tier"));
    let off_a = total(agg_bound.sum_on(Tier::Interpreter).expect("interp tier"));
    COMPILES.store(0, Ordering::Relaxed);
    let on_a = total(agg_bound.sum_on(Tier::Jit).expect("jit tier"));
    let on_ac = COMPILES.load(Ordering::Relaxed);
    assert_eq!(
        base_a.to_bits(),
        clean_a.to_bits(),
        "aggregate stock vs clean VM"
    );
    assert_eq!(
        base_a.to_bits(),
        off_a.to_bits(),
        "aggregate naive vs JIT-off divergence"
    );
    assert_eq!(
        base_a.to_bits(),
        on_a.to_bits(),
        "aggregate naive vs JIT-on -> miscompile"
    );
    assert!(on_ac >= 1, "aggregate JIT-on must compile the batch loop");

    let (mut on_a_t, mut off_a_t, mut clean_a_t, mut naive_a_t) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        naive_a_t.push(time_ns_per_row(n, || naive_agg().to_bits()));
        clean_a_t.push(time_ns_per_row(n, || {
            total(agg_bound.sum_on(Tier::Clean).expect("clean tier")).to_bits()
        }));
        off_a_t.push(time_ns_per_row(n, || {
            total(agg_bound.sum_on(Tier::Interpreter).expect("interp tier")).to_bits()
        }));
        on_a_t.push(time_ns_per_row(n, || {
            total(agg_bound.sum_on(Tier::Jit).expect("jit tier")).to_bits()
        }));
    }
    let (jit_a, jit_off_a, vm_a, nv_a) = (
        median(on_a_t),
        median(off_a_t),
        median(clean_a_t),
        median(naive_a_t),
    );
    println!();
    println!("aggregate: sum({agg_expr}) = {base_a:.3}  (all paths agree)  compiles on={on_ac}");
    println!("  stock   (cel tree-walk, reused ctx) : {nv_a:>9.2} ns/row");
    println!("  clean VM(lowered bytecode, no JIT)  : {vm_a:>9.2} ns/row");
    println!("  majit   (tracing interp, JIT off)    : {jit_off_a:>9.2} ns/row");
    println!("  majit   (compiled trace, JIT on)     : {jit_a:>9.2} ns/row");
    println!(
        "  VM/data-model effect  stock / clean VM : {:>8.2}x",
        nv_a / vm_a
    );
    println!(
        "  JIT effect          clean VM / JIT-on  : {:>8.2}x",
        vm_a / jit_a
    );
    println!(
        "  majit tier delta     JIT-off / JIT-on  : {:>8.2}x",
        jit_off_a / jit_a
    );
    println!(
        "  cross-model batch    stock / JIT-on    : {:>8.2}x",
        nv_a / jit_a
    );

    black_box((&price, &qty));
}
