//! Why `filter_list_scaling/10000` costs ~2.9x `map_list_scaling/10000` per
//! element on the compiled tier when its program is only ~1.46x the words
//! (540034 vs 370034) and neither case fails a guard or takes a bridge.
//!
//! The instrument: `majit_metainterp::embed::Census::compiled_opcode_log`
//! records the opcode list of every compiled loop, so the two cases' traces
//! can be differenced op kind by op kind instead of guessed at.
//!
//! Run it:
//!
//! ```text
//! cargo run --release --package cel --no-default-features \
//!   --features regex,chrono,jit-cranelift --example rca_filtercost
//! ```

use std::collections::BTreeMap;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::lower::{Schema, ValType};
use majit_metainterp::embed::Census;

fn probe(label: &str, src: &str, n: i64) {
    let schema: Schema = [("list[]".to_string(), ValType::Int)].into_iter().collect();
    let lowered = match BatchProgram::compile(src, &schema) {
        Ok(l) => l,
        Err(e) => {
            println!("== {label}: declines: {e}");
            return;
        }
    };
    let elems: Vec<i64> = (0..n).collect();
    let lens = vec![n];
    let batch = Batch::new(1).column(
        "list".to_string(),
        ColumnRef::List {
            lens: &lens,
            fields: vec![(None, ColumnRef::Int(&elems))],
        },
    );
    let bound = lowered.bind_per_row(&batch).expect("binds");
    reset_persistent_state();
    reset_jit_stats();
    for _ in 0..64 {
        bound.collect_on(Tier::Jit).unwrap();
    }
    println!("== {label}: {src}");
    println!("   stats={}", jit_stats());
    for (i, ops) in Census::compiled_opcode_log().into_iter().enumerate() {
        // The last Label starts the steady-state body; everything before it is
        // the preamble the peeled iteration hoists invariants into.
        let body_at = ops
            .iter()
            .rposition(|op| format!("{op:?}") == "Label")
            .map_or(0, |p| p + 1);
        println!(
            "   loop[{i}] ops={} body={}",
            ops.len(),
            ops.len() - body_at
        );
        for (tag, seg) in [("pre", &ops[..body_at]), ("body", &ops[body_at..])] {
            let mut hist: BTreeMap<String, usize> = BTreeMap::new();
            for op in seg {
                *hist.entry(format!("{op:?}")).or_default() += 1;
            }
            let mut items: Vec<_> = hist.into_iter().collect();
            items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for (name, c) in items {
                println!("      {tag:4} {c:5}  {name}");
            }
        }
    }
}

/// Interleaved warm timing: both cases live in one process and alternate
/// round by round, so drift lands on both and the RATIO is the reading.
/// The minimum over rounds is reported, the board's own estimator.
fn time_pair(n: i64, rounds: usize, calls: usize) {
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
        ("filter", "list.filter(x, x % 2 == 0)"),
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
    let mut best = [f64::INFINITY; 2];
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
    for (i, (label, src)) in cases.iter().enumerate() {
        println!(
            "   {label:8} {:10.1} ns/call  {:6.3} ns/elem  ({src})",
            best[i],
            best[i] / n as f64
        );
    }
    println!("   ratio filter/map = {:.3}", best[1] / best[0]);
}

fn main() {
    let n: i64 = std::env::var("RCAFC_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    probe("map", "list.map(x, x * 2)", n);
    probe("filter", "list.filter(x, x % 2 == 0)", n);
    println!("== interleaved warm timing, n={n}");
    time_pair(n, 30, 400);
}
