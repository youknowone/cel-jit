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
}
