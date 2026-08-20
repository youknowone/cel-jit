//! Attribute the fixed per-call cost of the clean tier's `collect_into_on`
//! door on a single-identifier expression -- the shape the per-call scoreboard
//! reports as `variable_access`.
//!
//! The arms split that call into the half that runs the batch and the half
//! that boxes its rows, and put beside the second one the same decode written
//! by hand — with the bank a constant and with it opaque, and with the row
//! count known and with it opaque. A difference between two of those prices one
//! property of the decode rather than the decode as a whole.
//!
//! The stock walker is timed in the same process, against the same activation
//! the scoreboard builds, so the gap the scoreboard reports is reproduced here
//! rather than carried over as a number.
//!
//! ⚠ `extend_values` is reached here from ANOTHER CRATE, which is the door a
//! columnar consumer of `collect_raw` uses. The scoreboard's `clean` column is
//! not: `collect_into_on` is a non-generic `pub fn`, so a caller crosses into
//! `cel` once and everything past that is inlined inside it. The two are
//! different questions about the same code and only the first is answered here.

use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, RawOutput, Tier};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

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

/// A local copy of the crate's row decoder, so the cost of dispatching on a
/// runtime bank can be told apart from the cost of calling across the crate
/// boundary to do it.
fn decode_local(bank: ValType, v: i64, interned: &[std::sync::Arc<String>]) -> Value {
    match bank {
        ValType::Int => Value::Int(v),
        ValType::UInt => Value::UInt(v as u64),
        ValType::Bool => Value::Bool(v != 0),
        ValType::Float => Value::Float(f64::from_bits(v as u64)),
        ValType::Str => Value::String(interned[v as usize].clone()),
        ValType::Timestamp => {
            Value::Timestamp(chrono::DateTime::from_timestamp_nanos(v).fixed_offset())
        }
        ValType::Duration => Value::Duration(chrono::Duration::nanoseconds(v)),
    }
}

fn main() {
    let vals = [true];
    let schema: Schema = [("apple".to_string(), ValType::Bool)].into_iter().collect();
    let program = Program::compile("apple").expect("parses");
    let lowered = BatchProgram::from_program(&program, &schema).expect("lowers");
    let batch = Batch::new(1).column("apple".to_string(), ColumnRef::Bool(&vals));
    let bound = lowered.bind_per_row(&batch).expect("binds");

    let mut activation = Context::default();
    activation.add_variable_from_value("apple", true);

    let mut out: Vec<Value> = Vec::new();
    bound.collect_into_on(Tier::Clean, &mut out).expect("clean");

    // A raw output identical to the one the door builds, but built OUTSIDE the
    // door, so the decode half can be timed with none of the door around it.
    let values = [1i64];
    let distinct: [String; 0] = [];
    let standalone = || RawOutput::Scalar {
        ty: ValType::Bool,
        values: &values,
        distinct: &distinct,
    };

    let arms: Vec<(&str, f64)> = vec![
        (
            "stock  resolve_value",
            per_call(|| {
                Value::resolve_value(program.expression(), black_box(&activation)).expect("stock")
            }),
        ),
        (
            "clean  collect_into_on",
            per_call(|| {
                bound.collect_into_on(Tier::Clean, &mut out).expect("clean");
                black_box(out.as_slice());
            }),
        ),
        (
            "clean  collect_raw_on(|_| ())",
            per_call(|| bound.collect_raw_on(Tier::Clean, |_| ()).expect("raw")),
        ),
        (
            "clean  collect_raw_on(rows)",
            per_call(|| {
                bound
                    .collect_raw_on(Tier::Clean, |o| o.rows())
                    .expect("raw")
            }),
        ),
        (
            "decode clear+extend_values",
            per_call(|| {
                out.clear();
                standalone().extend_values(&mut out);
                black_box(out.as_slice());
            }),
        ),
        (
            "decode clear+extend(mono)",
            per_call(|| {
                out.clear();
                out.extend(values.iter().map(|&v| Value::Bool(v != 0)));
                black_box(out.as_slice());
            }),
        ),
        (
            "decode clear+extend(local dyn ty)",
            per_call(|| {
                out.clear();
                let ty = black_box(ValType::Bool);
                out.extend(values.iter().map(|&v| decode_local(ty, v, &[])));
                black_box(out.as_slice());
            }),
        ),
        (
            "decode clear+extend(local const ty)",
            per_call(|| {
                out.clear();
                out.extend(values.iter().map(|&v| decode_local(ValType::Bool, v, &[])));
                black_box(out.as_slice());
            }),
        ),
        (
            "decode clear+extend(local dyn ty, opaque len)",
            per_call(|| {
                out.clear();
                let ty = black_box(ValType::Bool);
                let vs: &[i64] = black_box(&values[..]);
                out.extend(vs.iter().map(|&v| decode_local(ty, v, &[])));
                black_box(out.as_slice());
            }),
        ),
        (
            "decode clear+extend(local const ty, opaque len)",
            per_call(|| {
                out.clear();
                let vs: &[i64] = black_box(&values[..]);
                out.extend(vs.iter().map(|&v| decode_local(ValType::Bool, v, &[])));
                black_box(out.as_slice());
            }),
        ),
        (
            "decode clear+push(Bool)",
            per_call(|| {
                out.clear();
                out.push(Value::Bool(true));
                black_box(out.as_slice());
            }),
        ),
        ("route  route(Auto)", per_call(|| bound.route(Tier::Auto))),
        ("floor  black_box(0u64)", per_call(|| 0u64)),
    ];

    for (name, ns) in &arms {
        println!("{name:<30} {ns:7.3} ns");
    }
    let get = |k: &str| arms.iter().find(|(n, _)| *n == k).unwrap().1;
    println!();
    println!(
        "door overhead (raw_unit - floor)      {:7.3} ns",
        get("clean  collect_raw_on(|_| ())") - get("floor  black_box(0u64)")
    );
    println!(
        "decode half   (into - raw_unit)       {:7.3} ns",
        get("clean  collect_into_on") - get("clean  collect_raw_on(|_| ())")
    );
    println!(
        "gap to stock  (into - stock)          {:7.3} ns",
        get("clean  collect_into_on") - get("stock  resolve_value")
    );
    println!("size_of::<Value>() = {}", std::mem::size_of::<Value>());
    println!(
        "needs_drop::<Value>() = {}",
        std::mem::needs_drop::<Value>()
    );
}
