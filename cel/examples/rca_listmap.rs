//! Why `[1, 2, 3, 4, 5].map(x, x * 2)` minted an entry artifact and never
//! entered it — and the census that separates the cause from three things it
//! was confounded with.
//!
//! `majit_percall_steady`'s smoke run reported `compiles=1`,
//! `compiled_entries=0` for `list_map` over 4096 one-row calls, while
//! `all_comprehension` and `exists_comprehension` over the SAME five-element
//! literal entered at call 9. That reads as "literal-list source plus
//! list-valued result", and this file refutes that: `list_filter` is a literal
//! list with a list-valued result and it enters at call 9 too.
//!
//! What the failing program has and the others do not is a word-scan artefact.
//! `float_bank::loop_header_keys` finds loop headers by scanning for
//! `OP_JUMP_IF_ABOVE`'s value (16) word-wise, and reads the word three past it
//! as a jump target. In the unrolled map body an `OP_MUL_OVF` spells
//! `[42, 16, 17, 18, 0]` — register 16 as its left operand, register 0 as its
//! trap — so the scan sees a back edge to position 0. Position 0 is `ENTRY_PC`,
//! the key the function-entry door itself files under, so from the call after
//! the door minted its artifact the door's own `has_compiled_loop` decline
//! fires on it, forever.
//!
//! Run it:
//!
//! ```text
//! cargo run --package cel --features jit-cranelift --example rca_listmap
//! ```
//!
//! Read the `backward-JIA targets` line beside `first_entry`: a target of `0` is
//! the defect, and it is the only thing the failing case has that the four
//! healthy ones do not. `tests/majit_trace_evidence.rs`
//! `a_spurious_back_edge_to_entry_pc_does_not_shut_the_entry_door` is the pinned
//! regression.

use cel::majit::batch::{Batch, BatchProgram, Tier};
use cel::majit::bytecode::float_bank::{
    jit_stats, reset_jit_stats, reset_persistent_state, COMPILED_ENTRIES, COMPILES,
};
use cel::majit::lower::{BatchReduce, Schema, ValType};
use std::sync::atomic::Ordering;

const OP_JUMP_IF_ABOVE: i64 = 16;

fn scan_loop_targets(code: &[i64]) -> Vec<(usize, usize, Vec<i64>)> {
    let mut targets: Vec<(usize, usize, Vec<i64>)> = Vec::new();
    for pc in 0..code.len().saturating_sub(3) {
        if code[pc] != OP_JUMP_IF_ABOVE {
            continue;
        }
        let t = code[pc + 3];
        if t < 0 || t as usize >= pc {
            continue;
        }
        if targets.iter().any(|(tt, _, _)| *tt == t as usize) {
            continue;
        }
        let lo = pc.saturating_sub(3);
        targets.push((t as usize, pc, code[lo..(pc + 4).min(code.len())].to_vec()));
    }
    targets
}

fn probe(label: &str, src: &str, list_col: Option<(&str, Vec<i64>)>) {
    let schema: Schema = list_col
        .iter()
        .map(|(name, _)| (format!("{name}[]"), ValType::Int))
        .collect();
    let lowered = match BatchProgram::compile(src, &schema) {
        Ok(l) => l,
        Err(e) => {
            println!("{label}: declines: {e}");
            return;
        }
    };
    let shape = lowered.lowered().batch_shape(true, BatchReduce::PerRow);
    let code: &[i64] = &shape.code;
    println!("== {label}: {src}");
    println!("   words={} addr={:p}", code.len(), code.as_ptr());
    println!("   list_output={:?}", lowered.lowered().list_output);
    println!(
        "   backward-JIA targets (word scan) = {:?}",
        scan_loop_targets(code)
    );
    println!("   raw = {code:?}");

    // Now run it repeatedly, one row per call, and see what happens.
    let lens: Vec<i64> = list_col.iter().map(|(_, v)| v.len() as i64).collect();
    let mut batch = Batch::new(1);
    if let Some((name, elems)) = &list_col {
        batch = batch.column(
            name.to_string(),
            cel::majit::batch::ColumnRef::List {
                lens: &lens,
                fields: vec![(None, cel::majit::batch::ColumnRef::Int(elems))],
            },
        );
    }
    let bound = match lowered.bind_per_row(&batch) {
        Ok(b) => b,
        Err(e) => {
            println!("   cannot bind: {e}");
            return;
        }
    };
    reset_persistent_state();
    reset_jit_stats();
    let mut first_entry = None;
    let mut first_compile = None;
    let (mut pc, mut pe) = (0usize, 0usize);
    for call in 0..64 {
        bound.collect_on(Tier::Jit).unwrap();
        let c = COMPILES.load(Ordering::Relaxed);
        let e = COMPILED_ENTRIES.load(Ordering::Relaxed);
        if c > pc && first_compile.is_none() {
            first_compile = Some(call + 1);
        }
        if e > pe && first_entry.is_none() {
            first_entry = Some(call + 1);
        }
        pc = c;
        pe = e;
    }
    println!("   first_compile={first_compile:?} first_entry={first_entry:?}");
    println!("   stats={}", jit_stats());
    println!();
}

fn main() {
    probe("list_map", "[1, 2, 3, 4, 5].map(x, x * 2)", None);
    probe("all_comp", "[1, 2, 3, 4, 5].all(x, x > 0)", None);
    probe("exists_comp", "[1, 2, 3, 4, 5].exists(x, x == 3)", None);
    probe(
        "list_filter",
        "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(x, x > 5)",
        None,
    );
    probe("map_col_1", "list.map(x, x * 2)", Some(("list", vec![7])));
    probe(
        "map_col_5",
        "list.map(x, x * 2)",
        Some(("list", vec![1, 2, 3, 4, 5])),
    );

    // The cases that separate the mechanism from the shape it was first seen
    // in. Two of the four refuted a prediction made before running them, and
    // both refutations sharpened the rule.
    //
    // `.all` over a literal does NOT unroll — `all_times2` lowers to the same
    // 36 words as `all_comp`, one rolled loop — so overflow-checked arithmetic
    // in the body buys it no unrolled registers and it is healthy. Unrolling is
    // what the list-valued comprehensions do.
    probe("all_times2", "[1, 2, 3, 4, 5].all(x, x * 2 > 0)", None);
    // Three elements is already enough: the per-element register stride is 6
    // from a base of 4, so element THREE takes first operand 4 + 2*6 = 16 and
    // `map3` fails exactly as the five-element case does. The element count is
    // not what selects the defect; reaching register 16 is.
    probe("map3", "[1, 2, 3].map(x, x * 2)", None);
    // Two elements stop at first operands 4 and 10, short of 16.
    probe("map2", "[1, 2].map(x, x * 2)", None);
    // And the trigger does not follow the source-level shape. This is a filter
    // over a five-element literal doing the same overflow-checked multiply, so
    // "unrolled literal plus checked arithmetic" predicts it fails — it does
    // not. `.filter` allocates on a different stride, and its word 16 lands
    // where the word three past it is 18, not 0.
    //
    // That is the real scope of the defect: what selects it is whether some
    // instruction's operand happens to hold 16 with a 0 three words later, and
    // no property of the expression decides that. `.map` over a literal of
    // three or more elements with checked arithmetic is one family that spells
    // it, not the definition of the class.
    probe(
        "filter_times2",
        "[1, 2, 3, 4, 5].filter(x, x * 2 > 4)",
        None,
    );
}
