//! cell-majit **uint** columnar batch evaluator vs the stock tree-walker
//! (issue #357). The unsigned analog of `majit_columnar_batch`.
//!
//! This is an explicit **cross-model batch experiment**, not the default fair
//! CEL JIT benchmark. Its stock/JIT ratio combines columnar specialization,
//! batch fusion, and compilation; it is not a JIT-only claim.
//!
//! Runs the REAL cel path: a CEL `Program` over `uint` columns is lowered
//! (`cel::majit::lower::lower_typed`, under a schema declaring the uint slots)
//! and evaluated over a batch of rows via `eval_batch_sum_f`. uint values share
//! the int register file (their raw 64-bit pattern), so the compiled trace reads
//! each column as `i64` bits via a `raw_load`; ordering comparisons emit the
//! unsigned `OP_ULT`/`OP_ULE` (`>`/`>=` via an operand swap), and the tree-walker
//! orders `Value::UInt` unsigned, so the two agree on the full u64 range.
//!
//! FAIR comparison = hot vs hot. Both sides receive their data already laid out
//! (u64 columns for the JIT, a live `Context` for the walker) and are measured
//! steady-state, with no per-row setup on either side. The baseline is the
//! tree-walker at its best: ONE reused `Context` whose variables are overwritten
//! per row, then `Program::execute`.
//!
//! The columns are full-range u64 (about half the rows have the high bit set),
//! so a signed compare would give a different count — the win is not bought by
//! restricting the data to the non-negative i64 range.
//!
//! The benchmark reports four paths over one prebuilt batch program: stock,
//! clean bytecode VM, majit with compilation disabled, and compiled majit.
//! `stock / clean VM` measures the representation/lowering effect, while
//! `clean VM / JIT-on` measures the compilation effect. RELEASE ONLY (4-way
//! equality gate).

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::bytecode::float_bank::{clean_interp_seeded_f, run_jit_seeded_f, COMPILES};
use cel::majit::bytecode::Column;
use cel::majit::lower::{lower_typed, Schema, ValType};
use cel::{Context, Program, Value};

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// Deterministic full-range u64 column, returned as the i64 bit pattern the int
/// register file carries. `as u64` in the oracle recovers the unsigned value.
fn make_col_u(n: usize, seed: u64) -> Vec<i64> {
    let mut x = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        v.push(x as i64);
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
    // Flagship uint policy: an unsigned column clears a per-row unsigned minimum
    // AND stays under an unsigned ceiling above the signed range (10^19 >
    // i64::MAX), so unsigned ordering is load-bearing.
    let expr = "account >= minimum && account < 10000000000000000000u";
    let program = Program::compile(expr).expect("compile");
    let schema: Schema = [
        ("account".to_string(), ValType::UInt),
        ("minimum".to_string(), ValType::UInt),
    ]
    .into_iter()
    .collect();
    let lowered = lower_typed(program.expression(), &schema).expect("lower uint policy");
    let slot_paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
    assert_eq!(slot_paths, ["account", "minimum"], "slot order");

    let n: usize = 2_000_000;
    let account = make_col_u(n, 0x2545_F491_4F6C_DD1D);
    let minimum = make_col_u(n, 0x9E37_79B9_7F4A_7C15);
    let columns: Vec<Column> = vec![Column::Int(&account), Column::Int(&minimum)];
    let bases: Vec<i64> = columns.iter().map(Column::base).collect();
    let (shape, regs) = lowered.batch_sum_program(&bases, n as i64);
    let (batch, num_float) = (shape.code, shape.num_float_regs);

    // FAIR baseline: the stock tree-walker at its best — reuse one Context,
    // overwrite the two uint variables per row (hot; no per-row Context alloc).
    let naive = || -> i64 {
        let mut acc = 0i64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("account", account[i] as u64);
            ctx.add_variable_from_value("minimum", minimum[i] as u64);
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
    let clean = clean_interp_seeded_f(&batch, &regs, num_float);
    let off = run_jit_seeded_f(&batch, &regs, num_float, u32::MAX);
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = run_jit_seeded_f(&batch, &regs, num_float, 8);
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
            clean_interp_seeded_f(&batch, &regs, num_float)
        }));
        off_t.push(time_ns_per_row(n, || {
            run_jit_seeded_f(&batch, &regs, num_float, u32::MAX)
        }));
        on_t.push(time_ns_per_row(n, || {
            run_jit_seeded_f(&batch, &regs, num_float, 8)
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
    black_box((&account, &minimum));
}
