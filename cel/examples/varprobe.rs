//! Decompose what one clean-tier call costs on a single-variable expression,
//! the shape `variable_access` in the per-call scoreboard routes to.
//!
//! Times the same batch through doors that differ by one layer each, and over a
//! row ladder, so a difference between two rows prices one layer and the ladder
//! slope prices one dispatched instruction.

use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::lower::{BatchReduce, Schema, ValType};

const MIN_BATCH: Duration = Duration::from_millis(20);
const ROUNDS: usize = 9;

fn per_call<T>(mut run: impl FnMut() -> T) -> f64 {
    fn timed<T>(iters: usize, run: &mut impl FnMut() -> T) -> Duration {
        let start = Instant::now();
        for _ in 0..iters {
            black_box(run());
        }
        start.elapsed()
    }
    let mut iters = 1usize;
    loop {
        let elapsed = timed(iters, &mut run);
        if elapsed >= MIN_BATCH {
            break;
        }
        let want = MIN_BATCH.as_secs_f64() / elapsed.as_secs_f64().max(1e-9);
        let grow = (want.ceil() as usize).clamp(2, 1 << 12);
        iters = iters.saturating_mul(grow);
    }
    (0..ROUNDS)
        .map(|_| timed(iters, &mut run).as_nanos() as f64 / iters as f64)
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let schema: Schema = [("apple".to_string(), ValType::Bool)].into_iter().collect();
    let program = BatchProgram::compile("apple", &schema).expect("lowers");
    let lowered = program.lowered();
    let shape = lowered.batch_shape(true, BatchReduce::PerRow);
    println!(
        "body_int={} body_float={} slots={} code_words={}",
        lowered.num_int_regs,
        lowered.num_float_regs,
        lowered.slots.len(),
        shape.code.len()
    );

    // Row ladder: one more row is one more pass of the five-instruction body,
    // so the slope prices a dispatch.
    let vals = [true; 8];
    for n in [1usize, 2, 4, 8] {
        let batch = Batch::new(n).column("apple".to_string(), ColumnRef::Bool(&vals[..n]));
        let bound = program.bind_per_row(&batch).expect("binds");
        let raw = per_call(|| bound.collect_raw_on(Tier::Clean, |o| o.rows()).unwrap());
        let boxed = per_call(|| bound.collect_on(Tier::Clean).unwrap());
        println!("rows={n:<2} raw={raw:6.2} ns  boxed={boxed:6.2} ns");
    }

    let batch = Batch::new(1).column("apple".to_string(), ColumnRef::Bool(&vals[..1]));
    let bound = program.bind_per_row(&batch).expect("binds");
    println!(
        "collect()      {:6.2} ns",
        per_call(|| bound.collect().unwrap())
    );
    println!(
        "route(Auto)    {:6.2} ns",
        per_call(|| bound.route(Tier::Auto))
    );
    println!(
        "vec1           {:6.2} ns",
        per_call(|| vec![cel::Value::Bool(true)])
    );
    println!(
        "vec1+drop_only {:6.2} ns",
        per_call(|| Vec::<cel::Value>::with_capacity(1))
    );
    println!("size_of::<Value>() = {}", std::mem::size_of::<cel::Value>());

    // Where the route sends a TALL projection, and what the two tiers it is
    // choosing between actually cost there. `route` reads body words, which a
    // projection has as many of as any other program — but the clean tier no
    // longer executes them.
    let tall: Vec<i64> = (0..10_000).collect();
    let tschema: Schema = [("x".to_string(), ValType::Int)].into_iter().collect();
    let tprog = BatchProgram::compile("x", &tschema).expect("lowers");
    let tbatch = Batch::new(tall.len()).column("x".to_string(), ColumnRef::Int(&tall));
    let tb = tprog.bind_per_row(&tbatch).expect("binds");
    println!(
        "tall projection: rows={} words={} route={:?}",
        tall.len(),
        tb.body_words(),
        tb.route(Tier::Auto)
    );
    for tier in [Tier::Clean, Tier::Jit] {
        // Warm the compiled tier before timing it.
        for _ in 0..2000 {
            black_box(tb.collect_raw_on(tier, |o| o.rows()).unwrap());
        }
        println!(
            "  {tier:?} raw {:9.1} ns",
            per_call(|| tb.collect_raw_on(tier, |o| o.rows()).unwrap())
        );
    }

    // Control: a constant per-row expression. Same entry, same loop tail, no
    // column read — so its raw door is the floor a projection could reach.
    let empty: Schema = Schema::default();
    if let Ok(konst) = BatchProgram::compile("true", &empty) {
        let kshape = konst.lowered().batch_shape(true, BatchReduce::PerRow);
        println!("const code_words={}", kshape.code.len());
        let kbatch = Batch::new(1);
        if let Ok(kb) = konst.bind_per_row(&kbatch) {
            println!(
                "const raw      {:6.2} ns",
                per_call(|| kb.collect_raw_on(Tier::Clean, |o| o.rows()).unwrap())
            );
            println!(
                "const boxed    {:6.2} ns",
                per_call(|| kb.collect_on(Tier::Clean).unwrap())
            );
        } else {
            println!("const: does not bind");
        }
    } else {
        println!("const: does not lower");
    }
}
