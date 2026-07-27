//! cell-majit **string-equality** columnar batch evaluator vs the stock
//! tree-walker (issue #357). The string analog of `majit_columnar_batch_uint`.
//!
//! This is an explicit **cross-model batch experiment**, not the default fair
//! CEL JIT benchmark. Hash interning, columnar specialization, batch fusion,
//! and compilation are all present in its stock/JIT ratio.
//!
//! Runs the REAL cel path: a CEL `Program` whose predicate is a string equality
//! (`role == "admin" && region == "us-west-2"`) is lowered
//! (`cel::majit::lower::lower_typed`, under a schema declaring the string slots)
//! and evaluated over a batch of rows via `eval_batch_sum_f`. A string column is
//! interned to an `i64` content-hash column (`cel::majit::lower::intern_hash`,
//! shared with the literal folding), so the compiled trace reads each column as
//! `i64` bits via a `raw_load` and the equality is a single `OP_EQ` — no string
//! comparison in the hot loop. The tree-walker, by contrast, compares the actual
//! `Arc<String>` content per row.
//!
//! The interning is verified injective over every distinct string present
//! (column values + the expression's literals): with an injective hash an id
//! compare equals a content compare bit for bit. A real collision would bail to
//! the tree-walker; the controlled data here is collision-free.
//!
//! FAIR comparison = hot vs hot. Both sides receive their data already laid out
//! (id columns for the JIT, a live `Context` for the walker) and are measured
//! steady-state, with no per-row setup on either side beyond the walker's
//! variable overwrite.
//!
//! The benchmark reports four paths over one prebuilt batch program: stock,
//! clean bytecode VM, majit with compilation disabled, and compiled majit.
//! `stock / clean VM` measures the representation/lowering effect, while
//! `clean VM / JIT-on` measures the compilation effect. RELEASE ONLY (4-way
//! equality gate).

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::COMPILES;
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// Deterministic per-column string data drawn from a small `choices` set.
fn make_col_str(n: usize, seed: u64, choices: &[&str]) -> Vec<String> {
    let mut x = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        v.push(choices[((x >> 33) as usize) % choices.len()].to_string());
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
    // Flagship string policy: two string equalities joined by `&&`.
    let expr = "role == \"admin\" && region == \"us-west-2\"";
    let program = Program::compile(expr).expect("compile");
    let schema: Schema = [
        ("role".to_string(), ValType::Str),
        ("region".to_string(), ValType::Str),
    ]
    .into_iter()
    .collect();
    let lowered = BatchProgram::compile(expr, &schema).expect("lower string policy");

    let n: usize = 2_000_000;
    let roles = ["admin", "user", "guest", "auditor", "root", "service"];
    let regions = ["us-west-2", "us-east-1", "eu-west-1", "ap-south-1"];
    let role = make_col_str(n, 0x2545_F491_4F6C_DD1D, &roles);
    let region = make_col_str(n, 0x9E37_79B9_7F4A_7C15, &regions);

    // Binding interns each string column to an i64 content-hash column and
    // verifies the hash is injective over every distinct string present (column
    // values + the expression's literals), so an id compare equals a content
    // compare bit for bit. A collision would be a `BatchError::HashCollision`.
    let batch = Batch::new(n)
        .column("role", ColumnRef::Str(&role))
        .column("region", ColumnRef::Str(&region));
    let bound = lowered.bind(&batch).expect("intern string columns");

    // FAIR baseline: the stock tree-walker at its best — reuse one Context,
    // overwrite the two string variables per row (hot; no per-row Context
    // alloc). This is where the walker pays the per-row string comparison the
    // compiled id trace does not.
    let naive = || -> i64 {
        let mut acc = 0i64;
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("role", role[i].clone());
            ctx.add_variable_from_value("region", region[i].clone());
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
    black_box((&role, &region));
}
