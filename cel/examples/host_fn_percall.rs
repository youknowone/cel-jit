//! Per-call cost of a user function in scalar form: `add(x, y) + multiply(a, b)`
//! through the tree-walker (`Program::execute`) and through the batch machine's
//! default door (`BatchProgram::from_program_in` + one-row `bind_per_row`, then
//! `collect_on(Tier::Auto)` per call), the way `majit_vs_cometkim_percall`
//! measures every other case. Prints nanoseconds per call for each.
use std::hint::black_box;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program};

fn per_call(mut f: impl FnMut()) -> f64 {
    let warm = 20_000;
    let n = 2_000_000;
    for _ in 0..warm {
        f();
    }
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    t.elapsed().as_nanos() as f64 / n as f64
}

fn main() {
    let src = "add(x, y) + multiply(a, b)";
    let program = Program::compile(src).unwrap();

    let mut ctx = Context::default();
    for (name, v) in [("x", 10i64), ("y", 20), ("a", 5), ("b", 3)] {
        ctx.add_variable_from_value(name, v);
    }
    ctx.add_function("add", |a: i64, b: i64| a + b);
    ctx.add_function("multiply", |a: i64, b: i64| a * b);
    let stock = per_call(|| {
        black_box(program.execute(&ctx).unwrap());
    });

    let schema: Schema = ["x", "y", "a", "b"]
        .iter()
        .map(|n| (n.to_string(), ValType::Int))
        .collect();
    let (x, y, a, b) = (vec![10i64], vec![20i64], vec![5i64], vec![3i64]);
    let batch = Batch::new(1)
        .column("x", ColumnRef::Int(&x))
        .column("y", ColumnRef::Int(&y))
        .column("a", ColumnRef::Int(&a))
        .column("b", ColumnRef::Int(&b));
    let batched = BatchProgram::from_program_in(&program, &schema, &ctx).unwrap();
    let bound = batched.bind_per_row(&batch).unwrap();
    let mut out = Vec::new();
    let auto = per_call(|| {
        out.clear();
        bound.collect_into_on(Tier::Auto, &mut out).unwrap();
        black_box(&out);
    });
    let answer = bound.collect_on(Tier::Auto).unwrap();
    assert_eq!(answer, vec![program.execute(&ctx).unwrap()]);

    println!("{src}");
    println!("  stock ns  {stock:8.1}");
    println!("  auto ns   {auto:8.1}");
    println!("  stock/auto {:.2}x", stock / auto);
}
