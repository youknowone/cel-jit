//! Head-to-head: cell-majit (batch meta-tracing) vs cometkim's cel-jit
//! (per-call AOT cranelift, PR #233) on cometkim's own benchmark expressions.
//!
//! cometkim's `comparison.rs` measures ONE `CompiledProgram::execute(&ctx)` per
//! criterion iteration over a FIXED context — per-call latency. His compiled
//! numbers (median ns/call, this machine, `cargo bench --bench comparison`) are
//! hard-coded below in `COMETKIM_AOT_NS` / `COMETKIM_INTERP_NS`.
//!
//! majit's regime is THROUGHPUT: one traced batch loop over N rows. We report
//! majit's compiled ns/row. The comparison is conservative — cometkim's
//! fixed-context ns/call is a LOWER BOUND on his throughput ns/eval (varying
//! inputs would add per-row context rebuilds he doesn't pay here), so
//! `majit_ns_per_row < cometkim_aot_ns_per_call` ⇒ majit wins throughput.
//!
//! Only the scalar/int/bool subset cometkim's expressions fall into is covered
//! (arithmetic without `/`, comparison, boolean, ternary, member access,
//! constant index). List/string-returning, custom-function, and variable-length
//! comprehension expressions are outside majit's traceable int subset and are
//! reported separately as "out of subset".
//!
//! RELEASE ONLY. Run: `cargo run --release --example majit_vs_cometkim --features majit-jit`.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::bytecode::{clean_interp, run_jit, Code, COMPILES};
use cel::majit::bytecode::{
    OP_ADD, OP_AND, OP_JUMP_IF_ABOVE, OP_LOAD_CONST, OP_MUL, OP_RETURN,
};
use cel::majit::lower::{lower, Lowered};
use cel::Program;

const LCG_A: i64 = 6364136223846793005;
const LCG_C: i64 = 1442695040888963407;
const SEED: i64 = 0x2545F4914F6CDD1D;

const N: i64 = 5_000_000;
const ROUNDS: usize = 5;
const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

/// One benchmark case: a CEL source, which slots are booleans (LCG low bit),
/// and cometkim's measured ns/call for interp + AOT on this machine.
struct Case {
    label: &'static str,
    src: &'static str,
    bool_slots: &'static [&'static str],
    cometkim_interp_ns: f64,
    cometkim_aot_ns: f64,
}

/// Non-bool data slots are masked to 24 bits so that the small arithmetic in
/// these expressions (`a + b`, `x * 2`, sums of three indices) cannot overflow
/// i64 — keeping majit's wrapping i64 bit-identical to cel's checked i64 (which
/// would otherwise error on overflow). Values still vary per row (non-foldable);
/// this only bounds their magnitude, matching cometkim's small fixed inputs.
const DATA_MASK: i64 = 0xFF_FFFF;

fn build_batch(lowered: &Lowered, n: i64, bool_slots: &[&str]) -> Vec<i64> {
    let base = lowered.num_regs;
    let r_x = base as i64;
    let r_a = (base + 1) as i64;
    let r_c_const = (base + 2) as i64;
    let r_one = (base + 3) as i64;
    let r_i = (base + 4) as i64;
    let r_n = (base + 5) as i64;
    let r_acc = (base + 6) as i64;
    let r_mask = (base + 7) as i64;

    let mut p: Vec<i64> = Vec::new();
    p.extend_from_slice(&[OP_LOAD_CONST, SEED, r_x]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_A, r_a]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_C, r_c_const]);
    p.extend_from_slice(&[OP_LOAD_CONST, 1, r_one]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_i]);
    p.extend_from_slice(&[OP_LOAD_CONST, n, r_n]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_acc]);
    p.extend_from_slice(&[OP_LOAD_CONST, DATA_MASK, r_mask]);

    let body_pc = p.len() as i64;

    for slot in &lowered.slots {
        let dst = slot.reg as i64;
        p.extend_from_slice(&[OP_MUL, r_x, r_a, r_x]);
        p.extend_from_slice(&[OP_ADD, r_x, r_c_const, r_x]);
        if bool_slots.contains(&slot.path.as_str()) {
            p.extend_from_slice(&[OP_AND, r_x, r_one, dst]);
        } else {
            p.extend_from_slice(&[OP_AND, r_x, r_mask, dst]);
        }
    }
    p.extend_from_slice(&lowered.body);
    p.extend_from_slice(&[OP_ADD, r_acc, lowered.result_reg as i64, r_acc]);
    p.extend_from_slice(&[OP_ADD, r_i, r_one, r_i]);
    p.extend_from_slice(&[OP_JUMP_IF_ABOVE, r_n, r_i, body_pc]);
    p.extend_from_slice(&[OP_RETURN, r_acc]);
    p
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ns<F: FnMut() -> i64>(n: i64, mut f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn run_case(case: &Case) {
    let program = match Program::compile(case.src) {
        Ok(p) => p,
        Err(e) => {
            println!("{:22} PARSE ERROR: {e:?}", case.label);
            return;
        }
    };
    let lowered = match lower(program.expression()) {
        Ok(l) => l,
        Err(e) => {
            println!(
                "{:22} out of subset ({e})  [cometkim AOT {:.1} ns]",
                case.label, case.cometkim_aot_ns
            );
            return;
        }
    };
    let nregs = lowered.num_regs + 8;
    let batch = build_batch(&lowered, N, case.bool_slots);
    let code: &Code = &batch;

    // self-consistency miscompile gate
    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp(code, nregs);
    let off = run_jit(code, nregs, JIT_OFF);
    COMPILES.store(0, Ordering::Relaxed);
    let on = run_jit(code, nregs, JIT_ON);
    let compiles = COMPILES.load(Ordering::Relaxed);
    assert_eq!(clean, off, "{}: clean vs jit-off", case.label);
    assert_eq!(clean, on, "{}: clean vs jit-on -> miscompile", case.label);

    let mut a = Vec::new();
    for _ in 0..ROUNDS {
        a.push(time_ns(N, || run_jit(code, nregs, JIT_ON)));
    }
    let majit = median(a);

    let vs_aot = case.cometkim_aot_ns / majit;
    let verdict = if vs_aot >= 1.0 { "majit WINS" } else { "majit slower" };
    println!(
        "{:22} majit {:7.2} | cometkim AOT {:7.1} ({:>2} slots, compiles={compiles}) | {:6.1}x  {verdict}",
        case.label, majit, case.cometkim_aot_ns, lowered.slots.len(), vs_aot,
    );
}

fn main() {
    println!(
        "ns/row (majit batch, throughput) vs ns/call (cometkim AOT, per-call fixed-ctx best case)\n\
         majit < cometkim ⇒ majit wins throughput (cometkim's per-call is a lower bound on his ns/eval)\n"
    );

    let cases = [
        Case { label: "comparison(const)", src: "10 > 5 && 3 < 7 || 1 == 1", bool_slots: &[], cometkim_interp_ns: 36.0, cometkim_aot_ns: 8.34 },
        Case { label: "variable_access", src: "x", bool_slots: &[], cometkim_interp_ns: 7.80, cometkim_aot_ns: 14.0 },
        Case { label: "conditional", src: "x > 10 ? x * 2 : x + 5", bool_slots: &[], cometkim_interp_ns: 44.78, cometkim_aot_ns: 22.58 },
        Case { label: "member_access", src: "obj.nested.value + obj.other", bool_slots: &[], cometkim_interp_ns: 131.3, cometkim_aot_ns: 161.9 },
        Case { label: "list_indexing", src: "list[0] + list[5] + list[9]", bool_slots: &[], cometkim_interp_ns: 61.6, cometkim_aot_ns: 74.1 },
        // out of majit's int subset (division / list / string / custom fn / comprehension):
        Case { label: "simple_arithmetic", src: "1 + 2 * 3 - 4 / 2", bool_slots: &[], cometkim_interp_ns: 50.2, cometkim_aot_ns: 8.03 },
        Case { label: "nested_expr(div)", src: "((a + b) * (c - d)) / ((e + f) - (g * h))", bool_slots: &[], cometkim_interp_ns: 215.4, cometkim_aot_ns: 159.9 },
        Case { label: "all_comprehension", src: "[1, 2, 3, 4, 5].all(x, x > 0)", bool_slots: &[], cometkim_interp_ns: 478.2, cometkim_aot_ns: 197.8 },
    ];

    for case in &cases {
        run_case(case);
        let _ = case.cometkim_interp_ns;
    }
}
