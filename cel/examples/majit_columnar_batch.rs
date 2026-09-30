//! cell-majit columnar batch evaluator vs the stock tree-walker (issue #357).
//!
//! This is an explicit **cross-model batch experiment**, not the default fair
//! CEL JIT benchmark. The stock side evaluates one structured activation per
//! call; the majit side consumes pre-transposed primitive columns and fuses the
//! outer row loop. Do not report `stock / JIT-on` as a JIT-only speedup. Run
//! `./bench.sh` for the fair request/engine/cold suite.
//!
//! Runs the REAL cel path through the public batch API
//! (`cel::majit::batch`): a CEL source is lowered against a schema and bound to
//! named columns, and each column is read at the data-dependent (red) row index
//! through a compiled `raw_load` trace, the buffer bases held loop-invariant in
//! the register file. The caller states column TYPES and never sees slot order,
//! register banks or base pointers.
//!
//! FAIR comparison = hot vs hot. Both sides receive their data already laid out
//! (i64 columns for the JIT, a live `Context` for the walker) and are measured
//! steady-state, with no per-row setup on either side. The baseline is therefore
//! the tree-walker at its best: ONE reused `Context` whose variables are
//! overwritten per row, then `Value::resolve_value` — the walker itself, called
//! directly. NOT `Program::execute`: that is the bytecode VM whenever the `vm`
//! feature is on, and `vm` is a DEFAULT feature, so through the public door
//! `stock` would be a second VM and `stock / clean VM` would compare two
//! bytecode interpreters rather than a walker against one. This example carries
//! `required-features = ["jit"]`, which does not imply `--no-default-features`,
//! so the door would be the VM in every ordinary run of it.
//! We deliberately do NOT compare
//! against a fresh-`Context`-per-row walker — that pays a per-row allocation the
//! JIT never does (cold vs hot), which would flatter the JIT dishonestly.
//!
//! Four steady-state measurements over one prebuilt batch program:
//!   stock    stock tree-walker, reused Context
//!   clean VM plain Rust bytecode interpreter, no tracing/JIT machinery
//!   JIT-off  majit tracing interpreter, compilation disabled
//!   JIT-on   majit compiled trace
//! This separates the representation/lowering win (`stock / clean VM`, i.e. the
//! tree-walker against the lowered columnar bytecode) from
//! the compilation win (`clean VM / JIT-on`). RELEASE ONLY (i64 wrap; 4-way
//! equality gate).

use std::hint::black_box;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats};
use cel::majit::lower::{Schema, ValType};
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

/// The batch API answers in CEL's types; this benchmark counts matching rows.
fn count(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => panic!("unexpected batch result {other:?}"),
    }
}

fn main() {
    // Flagship policy predicate over three int/bool columns.
    let expr = "balance >= amount && !frozen";
    let schema: Schema = [
        ("balance".to_string(), ValType::Int),
        ("amount".to_string(), ValType::Int),
        ("frozen".to_string(), ValType::Bool),
    ]
    .into_iter()
    .collect();
    let program = Program::compile(expr).expect("compile");
    let lowered = BatchProgram::compile(expr, &schema).expect("lower policy to majit subset");

    let n: usize = 2_000_000;
    let balance = make_col(n, -1_000_000, 1_000_000, 0x2545F491);
    let amount = make_col(n, -1_000_000, 1_000_000, 0x9E3779B9);
    let frozen: Vec<bool> = make_col(n, 0, 1, 0x1000_0001)
        .iter()
        .map(|&v| v != 0)
        .collect();
    let batch = Batch::new(n)
        .column("balance", ColumnRef::Int(&balance))
        .column("amount", ColumnRef::Int(&amount))
        .column("frozen", ColumnRef::Bool(&frozen));
    let bound = lowered.bind(&batch).expect("bind columns");

    // FAIR baseline: the stock tree-walker at its best — reuse one Context,
    // overwrite the three variables per row (hot; no per-row Context alloc).
    let naive = || -> i64 {
        let mut acc = 0i64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("balance", balance[i]);
            ctx.add_variable_from_value("amount", amount[i]);
            ctx.add_variable_from_value("frozen", frozen[i]);
            acc += match Value::resolve_value(program.expression(), &ctx).expect("execute") {
                Value::Bool(b) => b as i64,
                Value::Int(v) => v,
                other => panic!("unexpected {other:?}"),
            };
        }
        acc
    };

    // Correctness gate: stock == clean VM == JIT-off == JIT-on.
    reset_jit_stats();
    let base = naive();
    let clean = count(bound.sum_on(Tier::Clean).expect("clean tier"));
    let off = count(bound.sum_on(Tier::Interpreter).expect("interp tier"));
    let off_c = jit_stats().loops_compiled;
    reset_jit_stats();
    let on = count(bound.sum_on(Tier::Jit).expect("jit tier"));
    let on_c = jit_stats().loops_compiled;
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
    black_box((&balance, &amount, &frozen));
}
