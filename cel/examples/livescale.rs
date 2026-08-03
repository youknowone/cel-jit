//! Does the fixed per-row shape-change cost scale with the number of live
//! loop-carried values?
//!
//! `majit_shape_change.rs` establishes that a driver holding one trip count's
//! artifact pays a *fixed* ~13 ns per row at any other trip count — flat in the
//! measured trip and flat in the warm trip, so it is one guard-exit → bridge →
//! loop-re-entry round trip and not per-iteration work. This example asks what
//! that round trip is made of.
//!
//! Two hypotheses, and they predict opposite shapes. If the cost is the
//! cranelift backend marshalling every live value through the jitframe — the
//! guard exit stores each fail arg to `jf_frame[slot]`, the bridge and then the
//! loop each read them all back — it is LINEAR in the live-set size. If it is
//! fixed dispatch overhead (two atomic loads, two indirect calls, a shadowstack
//! pop, a frame-depth check) it is FLAT.
//!
//! Every arm runs the same nested-loop shape at measured trip 1; only the
//! predicate's width changes, so only the register count moves.
//!
//! Measured 2026-08-04 (4000 rows, settled batch, min of 7 sequences, ns/row):
//!
//! | arm | int regs | cold | warm-64 | delta |
//! |---|---|---|---|---|
//! | 1 term | 14 | 1.6 | 14.4 | 12.9 |
//! | 2 terms | 17 | 1.7 | 15.9 | 14.2 |
//! | 3 terms | 20 | 1.9 | 18.5 | 16.6 |
//! | 5 terms | 26 | 2.4 | 22.1 | 19.7 |
//!
//! Linear: ~0.57 ns per extra live value on a ~5 ns fixed base. Two round trips
//! at one store plus one load each is four memory ops per value, and 0.57 ns is
//! about four memory ops' worth — so both hypotheses are true and the
//! marshalling is the part that grows. The `cold` column is the control: the
//! wider predicate's own work costs 0.8 ns/row across the whole ladder while
//! the delta grows by 6.8.
//!
//! Upstream has no such term. Once a bridge is attached
//! (`rpython/jit/backend/aarch64/assembler.py:200-202` → `patch_trace`
//! `:1054-1060`) the failing guard's site is overwritten with a direct branch
//! into a bridge whose incoming locations are the guard's own fail locations
//! (`:163` `rebuild_faillocs_from_descr`, `:167` `prepare_bridge`), and
//! `llsupport/jump.py remap_frame_layout` emits only the minimal permutation.
//!
//! Run: `cargo run --release --example livescale --features jit`

use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::bytecode::float_bank::reset_persistent_state;
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

const THRESHOLD: u32 = 8;
const ROWS: usize = 4000;
const WARM_ROWS: usize = 4000;
/// Whole-sequence repetitions; the reported cell is the min.
const ROUNDS: usize = 7;

struct ListColumns {
    lens: Vec<i64>,
    offsets: Vec<i64>,
    elems: Vec<i64>,
}

impl ListColumns {
    fn build(rows: usize, len: i64) -> Self {
        let lens = vec![len; rows];
        let mut offsets = Vec::with_capacity(rows);
        let mut total = 0i64;
        for &l in &lens {
            offsets.push(total);
            total += l;
        }
        let elems = (0..total.max(1)).map(|k| 11 + (k * 7) % 40).collect();
        Self {
            lens,
            offsets,
            elems,
        }
    }

    fn columns(&self) -> [Column<'_>; 3] {
        [
            Column::Int(&self.lens),
            Column::Int(&self.offsets),
            Column::Int(&self.elems),
        ]
    }
}

fn lowered(src: &str) -> LoweredF {
    let schema: Schema = [
        ("size(items)".to_string(), ValType::Int),
        ("offset(items)".to_string(), ValType::Int),
        ("items[].price".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect();
    let program = Program::compile(src).expect("parse");
    lower_typed(program.expression(), &schema).expect("lower")
}

fn ns_per_row(d: Duration) -> f64 {
    d.as_secs_f64() * 1e9 / ROWS as f64
}

/// Settled ns/row at `measured`, on a driver warmed at `warm` (cold if `None`).
fn settled(l: &LoweredF, warm: Option<i64>, measured: i64) -> f64 {
    let mc = ListColumns::build(ROWS, measured);
    reset_persistent_state();
    if let Some(w) = warm {
        let wc = ListColumns::build(WARM_ROWS, w);
        assert!(
            eval_batch_sum_f(l, &wc.columns(), WARM_ROWS, THRESHOLD).is_some(),
            "warm-up declined"
        );
    }
    let oracle = clean_batch_sum_f(l, &mc.columns(), ROWS);
    let mut last = Duration::ZERO;
    for _ in 0..3 {
        let t0 = Instant::now();
        let got = black_box(eval_batch_sum_f(l, &mc.columns(), ROWS, THRESHOLD));
        last = t0.elapsed();
        assert_eq!(got, oracle, "answer diverged");
    }
    ns_per_row(last)
}

fn main() {
    // Widening the predicate widens the live set without changing the loop
    // shape: the measured trip stays 1, so the row's work stays ~constant.
    let arms: [(&str, &str); 4] = [
        ("1 term ", "items.all(i, i.price > 10)"),
        ("2 terms", "items.all(i, i.price > 10 && i.price < 1000)"),
        (
            "3 terms",
            "items.all(i, i.price > 10 && i.price < 1000 && i.price != 42)",
        ),
        (
            "5 terms",
            "items.all(i, i.price > 10 && i.price < 1000 && i.price != 42 \
             && i.price != 43 && i.price != 44)",
        ),
    ];

    println!(
        "{:<8} {:>5} {:>5} {:>9} {:>9} {:>9}",
        "arm", "regs", "freg", "cold", "warm64", "delta"
    );
    for (name, src) in arms {
        let l = lowered(src);
        let regs = l.num_int_regs;
        let fregs = l.num_float_regs;
        // Min of N whole (reset, warm, three batches) sequences. The delta
        // between two arms is only a few ns and this machine is shared, so a
        // single sequence wanders; interference can only make a round slower.
        let cold = (0..ROUNDS)
            .map(|_| settled(&l, None, 1))
            .fold(f64::MAX, f64::min);
        let warm = (0..ROUNDS)
            .map(|_| settled(&l, Some(64), 1))
            .fold(f64::MAX, f64::min);
        println!(
            "{name:<8} {regs:>5} {fregs:>5} {cold:>9.1} {warm:>9.1} {:>9.1}",
            warm - cold
        );
    }
}
