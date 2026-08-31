//! Per-element cost of the division family, interleaved against a control the
//! change does not touch.
//!
//! Every case lives in one process and the cases alternate round by round, so
//! drift lands on all of them and the RATIO against `map` is the reading. The
//! minimum over rounds is reported, the board's own estimator.
//!
//! ```text
//! cargo run --release --package cel --no-default-features \
//!   --features regex,chrono,jit-cranelift --example rca_divcost
//! ```

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::reset_persistent_state;
use cel::majit::lower::{Schema, ValType};

fn main() {
    let n: i64 = std::env::var("RCADC_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let rounds = 30;
    let calls = 400;
    let schema: Schema = [("list[]".to_string(), ValType::Int)].into_iter().collect();
    let elems: Vec<i64> = (0..n).collect();
    let lens = vec![n];
    let batch = Batch::new(1).column(
        "list".to_string(),
        ColumnRef::List {
            lens: &lens,
            fields: vec![(None, ColumnRef::Int(&elems))],
        },
    );
    let cases = [
        ("map", "list.map(x, x * 2)"),
        ("div2", "list.map(x, x / 2)"),
        ("div3", "list.map(x, x / 3)"),
        ("mod3", "list.map(x, x % 3)"),
        ("mod2f", "list.filter(x, x % 2 == 0)"),
        ("mod3f", "list.filter(x, x % 3 == 0)"),
    ];
    let lowered: Vec<_> = cases
        .iter()
        .map(|(_, src)| BatchProgram::compile(src, &schema).expect("lowers"))
        .collect();
    let bound: Vec<_> = lowered
        .iter()
        .map(|l| l.bind_per_row(&batch).expect("binds"))
        .collect();
    reset_persistent_state();
    for b in &bound {
        for _ in 0..64 {
            b.collect_on(Tier::Jit).unwrap();
        }
    }
    let mut best = vec![f64::INFINITY; cases.len()];
    for _ in 0..rounds {
        for (i, b) in bound.iter().enumerate() {
            let t = std::time::Instant::now();
            for _ in 0..calls {
                b.collect_on(Tier::Jit).unwrap();
            }
            let ns = t.elapsed().as_nanos() as f64 / calls as f64;
            if ns < best[i] {
                best[i] = ns;
            }
        }
    }
    println!("== interleaved warm timing, n={n}, min of {rounds} x {calls}");
    for (i, (label, src)) in cases.iter().enumerate() {
        println!(
            "   {label:6} {:10.1} ns/call  {:6.3} ns/elem  {:5.3}x map  ({src})",
            best[i],
            best[i] / n as f64,
            best[i] / best[0],
        );
    }
    temporal(n as usize, rounds, calls / 4);
}

/// The calendar accessors, per row, against an int control the division change
/// does not touch. A calendar field is a chain of divisions by fixed scales, so
/// this is where a per-division residual call is paid the most times.
fn temporal(rows: usize, rounds: usize, calls: usize) {
    let schema: Schema = [
        ("t".to_string(), ValType::Timestamp),
        ("x".to_string(), ValType::Int),
        ("y".to_string(), ValType::Int),
        ("yodd".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let ts: Vec<i64> = (0..rows as i64)
        .map(|i| 1_700_000_000_000_000_000 + i * 1_000_000_007)
        .collect();
    let xs: Vec<i64> = (0..rows as i64).collect();
    // Never zero: a division that traps falls back to the row-by-row walker,
    // and a fallback would be timing the walker rather than the compiled loop.
    let ys: Vec<i64> = (0..rows as i64).map(|i| i % 97 + 1).collect();
    // Never a power of two, so `OP_MOD_CHK`'s mask arm is never the answer.
    // Read against `modvar`, whose column MIXES the two arms: a guard the trace
    // records one side of and the data then contradicts is paid as a bridge
    // entry per row, which a uniform column does not pay.
    let yodds: Vec<i64> = (0..rows as i64).map(|i| (i % 97) * 2 + 3).collect();
    let batch = Batch::new(rows)
        .column("t".to_string(), ColumnRef::Timestamp(&ts))
        .column("x".to_string(), ColumnRef::Int(&xs))
        .column("y".to_string(), ColumnRef::Int(&ys))
        .column("yodd".to_string(), ColumnRef::Int(&yodds));
    let cases = [
        ("ctl", "x + 1"),
        ("hours", "t.getHours()"),
        ("minutes", "t.getMinutes()"),
        ("seconds", "t.getSeconds()"),
        ("dayofmonth", "t.getDayOfMonth()"),
        ("dayofweek", "t.getDayOfWeek()"),
        // The residual-call question, with its two controls: `divk` is the same
        // division with the divisor in the instruction stream, which the
        // optimizer expands, and `mulvar` is the same two columns under an
        // operation that has a native opcode. The gap between `divvar` and
        // those two is what an unsigned divide opcode would be worth.
        ("divvar", "x / y"),
        ("divk", "x / 97"),
        ("mulvar", "x * y"),
        ("modvar", "x % y"),
        ("mododd", "x % yodd"),
        ("divodd", "x / yodd"),
    ];
    let lowered: Vec<_> = cases
        .iter()
        .map(|(_, src)| BatchProgram::compile(src, &schema).expect("lowers"))
        .collect();
    let bound: Vec<_> = lowered
        .iter()
        .map(|l| l.bind_per_row(&batch).expect("binds"))
        .collect();
    reset_persistent_state();
    for b in &bound {
        for _ in 0..64 {
            b.collect_on(Tier::Jit).unwrap();
        }
    }
    let mut best = vec![f64::INFINITY; cases.len()];
    for _ in 0..rounds {
        for (i, b) in bound.iter().enumerate() {
            let t = std::time::Instant::now();
            for _ in 0..calls {
                b.collect_on(Tier::Jit).unwrap();
            }
            let ns = t.elapsed().as_nanos() as f64 / calls as f64;
            if ns < best[i] {
                best[i] = ns;
            }
        }
    }
    println!("== temporal accessors, rows={rows}, min of {rounds} x {calls}");
    for (i, (label, src)) in cases.iter().enumerate() {
        println!(
            "   {label:10} {:10.1} ns/call  {:6.3} ns/row  {:5.2}x ctl  ({src})",
            best[i],
            best[i] / rows as f64,
            best[i] / best[0],
        );
    }
}
