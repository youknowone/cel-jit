//! Sampling-profile target for the columnar ENCODE — `bind_per_row_resolved`,
//! the second half of `bind`, which the percall board prices at ~70-100 ns of
//! FIXED cost per call on a one-row batch.
//!
//! The stage probes (`paired_ab --features __encode-stage-probe`) found no
//! named stage above 12% of that and left more than half of it in an unnamed
//! residual. Amplification cannot name what it did not amplify; a sampling
//! profiler can, so this loops one encode against one resolution taken once,
//! for long enough to be sampled, and prints the loop's own ns/encode beside
//! it. The struct sizes are printed too: a wide struct returned by value is a
//! memmove per call that no stage probe names, and it has happened once
//! already on the JIT entry (a 704-byte `CompileResult`).
//!
//! RELEASE ONLY:
//!   cargo run --release -p cel --features jit-cranelift --example rca_encode \
//!       -- [shape] [secs] [rows]
//!   sample <pid> 5 -file out.txt      # while it runs

use std::hint::black_box;
use std::mem::size_of;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, ResolvedBatch};
use cel::majit::bytecode::BatchRun;
use cel::majit::lower::{LoweredF, Schema, ValType};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shape = args.get(1).map_or("x * 2 + 1", |s| s.as_str());
    let secs: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let rows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    println!(
        "size_of: BatchRun={} BoundBatch={} ResolvedBatch={} LoweredF={}",
        size_of::<BatchRun<'static>>(),
        size_of::<BoundBatch<'static, 'static>>(),
        size_of::<ResolvedBatch<'static>>(),
        size_of::<LoweredF>(),
    );
    // Every int name a scalar shape might spell, so one schema and one batch
    // serve any of them; a program reads only the slots it references.
    let names = ["x", "a", "b", "c", "d", "e", "f"];
    let schema: Schema = names
        .iter()
        .map(|n| (n.to_string(), ValType::Int))
        .collect();
    let program = BatchProgram::compile(shape, &schema).expect("lowers");
    let vals: Vec<i64> = (0..rows as i64).collect();
    let mut batch = Batch::new(rows);
    for n in names {
        batch = batch.column(n, ColumnRef::Int(&vals));
    }
    let resolved = program.resolve(&batch).expect("resolves");
    // The first bind of a program builds its word stream; that one-off is not
    // what is being sampled.
    drop(program.bind_per_row_resolved(&resolved).expect("encodes"));
    println!("pid={} sampling {shape:?} rows={rows} for {secs}s", std::process::id());
    let start = Instant::now();
    let mut n = 0u64;
    while start.elapsed().as_secs_f64() < secs {
        for _ in 0..1000 {
            let b = program
                .bind_per_row_resolved(black_box(&resolved))
                .expect("encodes");
            black_box(&b);
        }
        n += 1000;
    }
    let ns = start.elapsed().as_nanos() as f64 / n as f64;
    println!("{shape}: rows={rows} encodes={n} ns/encode={ns:.1}");
}
