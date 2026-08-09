//! cel-majit **per-row** columnar batch evaluator vs the stock tree-walker.
//!
//! The other columnar examples all reduce a batch to one number, which is only
//! meaningful when the result is numeric or boolean. This one measures the
//! batch loop's OTHER reduction: instead of accumulating each row's result it
//! stores it, so a `string`-, `timestamp`- or `duration`-valued expression —
//! which a running total has nothing to do with — still runs on the compiled
//! tier, and every row keeps its own answer.
//!
//! The expression here is a string concatenation, `role + "@" + region`, whose
//! result is a string. Both sides do the same work per row and both are
//! measured hot:
//!
//! * The tree-walker allocates and formats a `String` per row.
//! * The batch loop's hot body never touches characters. `bind_per_row`
//!   materializes the concatenation as a derived column once, ranks it with
//!   every other string in the batch, and the loop stores the row's RANK; the
//!   strings come back at `collect`, decoded through that ranking.
//!
//! So the comparison is honest about what moved: the per-row string building is
//! not eliminated, it is hoisted out of the loop into the bind, which is where a
//! columnar engine wants it. What the loop pays per row is one `i64` store.
//!
//! The four paths — stock, clean bytecode VM, majit with compilation disabled,
//! compiled majit — are gated on producing identical values for every row.
//! RELEASE ONLY.
//!
//! `stock` is `Value::resolve_value`, the tree-walker called directly — NOT
//! `Program::execute`, which is the bytecode VM whenever the `vm` feature is
//! on, and `vm` is a DEFAULT feature. `required-features = ["jit"]` does not
//! imply `--no-default-features`, so through the public door the panel labelled
//! `stock tree-walker` would in fact be running the VM.

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

fn time_ns_per_row<F: FnMut() -> Vec<Value>>(n: usize, mut f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn main() {
    let expr = "role + \"@\" + region";
    let program = Program::compile(expr).expect("compile");
    let schema: Schema = [
        ("role".to_string(), ValType::Str),
        ("region".to_string(), ValType::Str),
    ]
    .into_iter()
    .collect();
    let lowered = BatchProgram::compile(expr, &schema).expect("lower string concatenation");
    assert_eq!(
        lowered.result_type(),
        ValType::Str,
        "a string result is the whole point: a sum could not take it"
    );

    let n: usize = 2_000_000;
    let roles = ["admin", "user", "guest", "auditor", "root", "service"];
    let regions = ["us-west-2", "us-east-1", "eu-west-1", "ap-south-1"];
    let role = make_col_str(n, 0x2545_F491_4F6C_DD1D, &roles);
    let region = make_col_str(n, 0x9E37_79B9_7F4A_7C15, &regions);

    let batch = Batch::new(n)
        .column("role", ColumnRef::Str(&role))
        .column("region", ColumnRef::Str(&region));
    // Binding materializes the concatenation column and ranks it together with
    // the input columns and the expression's own literal, so one order covers
    // every string the batch can produce or be asked about.
    let bound = lowered.bind_per_row(&batch).expect("bind per row");

    // FAIR baseline: the stock tree-walker at its best — one reused Context,
    // the two string variables overwritten per row.
    let naive = || -> Vec<Value> {
        let mut out = Vec::with_capacity(n);
        let mut ctx = Context::default();
        for i in 0..n {
            ctx.add_variable_from_value("role", role[i].clone());
            ctx.add_variable_from_value("region", region[i].clone());
            out.push(Value::resolve_value(program.expression(), &ctx).expect("execute"));
        }
        out
    };

    // Correctness gate: stock == clean VM == JIT-off == JIT-on, per row.
    COMPILES.store(0, Ordering::Relaxed);
    let base = naive();
    let clean = bound.collect_on(Tier::Clean).expect("clean tier");
    let off = bound.collect_on(Tier::Interpreter).expect("interp tier");
    let off_c = COMPILES.load(Ordering::Relaxed);
    COMPILES.store(0, Ordering::Relaxed);
    let on = bound.collect_on(Tier::Jit).expect("jit tier");
    let on_c = COMPILES.load(Ordering::Relaxed);
    assert_eq!(base, clean, "stock vs clean VM divergence");
    assert_eq!(base, off, "naive vs JIT-off divergence");
    assert_eq!(base, on, "naive vs JIT-on divergence -> miscompile");
    assert_eq!(off_c, 0, "JIT-off must never compile");
    assert!(on_c >= 1, "JIT-on must compile the batch loop");
    println!(
        "expression: {expr}  (result type {:?})",
        lowered.result_type()
    );
    println!(
        "n = {n}, rows produced = {}  (all paths agree)  compiles: off={off_c} on={on_c}",
        on.len()
    );

    let rounds = 5;
    let (mut on_t, mut off_t, mut clean_t, mut naive_t) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        naive_t.push(time_ns_per_row(n, naive));
        clean_t.push(time_ns_per_row(n, || {
            bound.collect_on(Tier::Clean).expect("clean tier")
        }));
        off_t.push(time_ns_per_row(n, || {
            bound.collect_on(Tier::Interpreter).expect("interp tier")
        }));
        on_t.push(time_ns_per_row(n, || {
            bound.collect_on(Tier::Jit).expect("jit tier")
        }));
    }
    let (naive_ns, clean_ns, off_ns, on_ns) = (
        median(naive_t),
        median(clean_t),
        median(off_t),
        median(on_t),
    );
    println!("stock tree-walker : {naive_ns:8.3} ns/row");
    println!(
        "clean bytecode VM : {clean_ns:8.3} ns/row  ({:.2}x)",
        naive_ns / clean_ns
    );
    println!(
        "majit  (JIT off)  : {off_ns:8.3} ns/row  ({:.2}x)",
        naive_ns / off_ns
    );
    println!(
        "majit  (JIT on)   : {on_ns:8.3} ns/row  ({:.2}x)",
        naive_ns / on_ns
    );
}
