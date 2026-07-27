//! Historical cross-regime probe: cell-majit batch throughput and cometkim's
//! per-call AOT cranelift latency (PR #233) on the same expression texts.
//! These numbers are shown side by side for context only. They have different
//! inputs, evaluation units, and measurement sessions, so no speedup ratio or
//! winner is reported. Use `./bench.sh` for the fair benchmark suite.
//!
//! cometkim's `comparison.rs` measures ONE `CompiledProgram::execute(&ctx)` per
//! criterion iteration over a FIXED context — per-call latency. His compiled /
//! interpreted medians below (`cometkim_aot_ns` / `cometkim_interp_ns`) are NOT
//! portable across machines — they were measured on THIS machine by re-running
//! his own `cargo bench --bench comparison` in the same session as this harness
//! (fresh, 2026-07-22). Re-measure both sides together on any other machine
//! before using the absolute numbers; the criterion medians are the moving part.
//!
//! majit's regime is THROUGHPUT: one traced batch loop over N rows. Cometkim's
//! regime is one `execute` call over a fixed context. Comparing their numerical
//! magnitudes does not isolate a JIT effect.
//!
//! Every case in cometkim's `comparison.rs` that falls into majit's int subset
//! is covered: constant fold, variable/member/constant-index access, ternary,
//! and integer arithmetic including `/`. List/string-returning and custom
//! functions stay out of the int subset and report "out of subset".
//!
//! RELEASE ONLY. Run: `cargo run --release --example majit_vs_cometkim --features jit`.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::bytecode::float_bank::{clean_interp_f, run_jit_f, COMPILES};
use cel::majit::bytecode::Code;
use cel::majit::bytecode::{OP_ADD, OP_AND, OP_JUMP_IF_ABOVE, OP_LOAD_CONST, OP_MUL, OP_RETURN};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

const LCG_A: i64 = 6364136223846793005;
const LCG_C: i64 = 1442695040888963407;
const SEED: i64 = 0x2545F4914F6CDD1D;

const N: i64 = 5_000_000;
const ROUNDS: usize = 5;
const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

/// A per-row slot value is `(lcg_word & mask) + bias`. The default masks a slot
/// to 24 bits so the small arithmetic in these expressions cannot overflow i64,
/// keeping majit's wrapping i64 bit-identical to cel's checked i64. Cases with
/// division override the shape per slot so the divisor is provably nonzero (a
/// zero divisor would be a *divergence*: cel errors, so it is out of the JIT's
/// int domain, exactly the schema/domain guard a real integration performs).
const DATA_MASK: i64 = 0xFF_FFFF;

type Shape = fn(usize) -> (i64, i64);

fn default_shape(_i: usize) -> (i64, i64) {
    (DATA_MASK, 0)
}

/// Slot roles for `((a + b) * (c - d)) / ((e + f) - (g * h))` in first-encounter
/// order a,b,c,d,e,f,g,h. Numerator operands stay small; the divisor operands
/// are ranged so `(e+f) - (g*h)` in [287, 1022] > 0 for every row.
fn nested_shape(i: usize) -> (i64, i64) {
    match i {
        0..=3 => (0xFF, 0),   // a,b,c,d in [0,255]     -> |num| <= 130050
        4 | 5 => (0xFF, 256), // e,f     in [256,511]   -> e+f in [512,1022]
        _ => (0x0F, 0),       // g,h     in [0,15]      -> g*h in [0,225]
    }
}

/// One benchmark case: a CEL source, the declared type of every path it reads,
/// its per-slot input shape, and cometkim's measured ns/call for interp + AOT on
/// this machine.
struct Case {
    label: &'static str,
    src: &'static str,
    /// Every path the source reads. The lowering declines an undeclared path, so
    /// this is the case's input type declaration, not a convenience.
    schema: &'static [(&'static str, ValType)],
    shape: Shape,
    cometkim_interp_ns: f64,
    cometkim_aot_ns: f64,
}

fn build_batch(lowered: &LoweredF, n: i64, shape: Shape) -> Vec<i64> {
    let base = lowered.num_int_regs as i64;
    let ns = lowered.slots.len() as i64;
    let r_x = base;
    let r_lcg_a = base + 1;
    let r_lcg_c = base + 2;
    let r_one = base + 3;
    let r_i = base + 4;
    let r_n = base + 5;
    let r_acc = base + 6;
    let mask_reg = |i: i64| base + 7 + i;
    let bias_reg = |i: i64| base + 7 + ns + i;

    let mut p: Vec<i64> = Vec::new();
    p.extend_from_slice(&[OP_LOAD_CONST, SEED, r_x]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_A, r_lcg_a]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_C, r_lcg_c]);
    p.extend_from_slice(&[OP_LOAD_CONST, 1, r_one]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_i]);
    p.extend_from_slice(&[OP_LOAD_CONST, n, r_n]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_acc]);
    for i in 0..ns {
        let (mask, bias) = shape(i as usize);
        p.extend_from_slice(&[OP_LOAD_CONST, mask, mask_reg(i)]);
        p.extend_from_slice(&[OP_LOAD_CONST, bias, bias_reg(i)]);
    }

    // Loop-invariant literal loads, hoisted by the typed lowering.
    p.extend_from_slice(&lowered.prelude);

    let body_pc = p.len() as i64;

    for (i, slot) in lowered.slots.iter().enumerate() {
        let (_, bias) = shape(i);
        let i = i as i64;
        let dst = slot.reg as i64;
        // Advance the LCG once per slot so each slot draws a distinct,
        // non-foldable value from the stream, then mask (+bias) into range.
        p.extend_from_slice(&[OP_MUL, r_x, r_lcg_a, r_x]);
        p.extend_from_slice(&[OP_ADD, r_x, r_lcg_c, r_x]);
        p.extend_from_slice(&[OP_AND, r_x, mask_reg(i), dst]);
        if bias != 0 {
            p.extend_from_slice(&[OP_ADD, dst, bias_reg(i), dst]);
        }
    }
    let body_at = p.len();
    p.extend_from_slice(&lowered.body);
    // Body-relative jump targets -> absolute program addresses.
    for &f in &lowered.jump_fixups {
        p[body_at + f] += body_at as i64;
    }
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
    let schema: Schema = case
        .schema
        .iter()
        .map(|(p, t)| (p.to_string(), *t))
        .collect();
    let lowered = match lower_typed(program.expression(), &schema) {
        Ok(l) => l,
        Err(e) => {
            println!(
                "{:22} out of subset ({e})  [cometkim AOT {:.1} ns]",
                case.label, case.cometkim_aot_ns
            );
            return;
        }
    };
    let nregs = lowered.num_int_regs + 7 + 2 * lowered.slots.len();
    let nfregs = lowered.num_float_regs;
    let batch = build_batch(&lowered, N, case.shape);
    let code: &Code = &batch;

    // self-consistency miscompile gate
    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp_f(code, nregs, nfregs);
    let off = run_jit_f(code, nregs, nfregs, JIT_OFF);
    COMPILES.store(0, Ordering::Relaxed);
    let on = run_jit_f(code, nregs, nfregs, JIT_ON);
    let compiles = COMPILES.load(Ordering::Relaxed);
    assert_eq!(clean, off, "{}: clean vs jit-off", case.label);
    assert_eq!(clean, on, "{}: clean vs jit-on -> miscompile", case.label);

    let mut a = Vec::new();
    for _ in 0..ROUNDS {
        a.push(time_ns(N, || run_jit_f(code, nregs, nfregs, JIT_ON)));
    }
    let majit = median(a);

    println!(
        "{:22} majit batch {:7.2} ns/row | cometkim AOT {:7.1} ns/call ({:>2} slots, compiles={compiles})",
        case.label, majit, case.cometkim_aot_ns, lowered.slots.len(),
    );
}

fn main() {
    println!(
        "NON-COMPARABLE UNITS: majit batch ns/row beside cometkim fixed-context ns/call.\n\
         No ratio or winner is valid; use the default ./bench.sh suite for fair measurements.\n"
    );

    let cases = [
        Case {
            label: "comparison(const)",
            src: "10 > 5 && 3 < 7 || 1 == 1",
            schema: &[],
            shape: default_shape,
            cometkim_interp_ns: 37.33,
            cometkim_aot_ns: 7.84,
        },
        Case {
            label: "variable_access",
            src: "x",
            schema: &[("x", ValType::Int)],
            shape: default_shape,
            cometkim_interp_ns: 7.73,
            cometkim_aot_ns: 13.97,
        },
        Case {
            label: "conditional",
            src: "x > 10 ? x * 2 : x + 5",
            schema: &[("x", ValType::Int)],
            shape: default_shape,
            cometkim_interp_ns: 35.14,
            cometkim_aot_ns: 22.25,
        },
        Case {
            label: "member_access",
            src: "obj.nested.value + obj.other",
            schema: &[
                ("obj.nested.value", ValType::Int),
                ("obj.other", ValType::Int),
            ],
            shape: default_shape,
            cometkim_interp_ns: 133.0,
            cometkim_aot_ns: 164.22,
        },
        Case {
            label: "list_indexing",
            src: "list[0] + list[5] + list[9]",
            schema: &[("list[]", ValType::Int)],
            shape: default_shape,
            cometkim_interp_ns: 61.80,
            cometkim_aot_ns: 74.27,
        },
        Case {
            label: "simple_arithmetic",
            src: "1 + 2 * 3 - 4 / 2",
            schema: &[],
            shape: default_shape,
            cometkim_interp_ns: 46.11,
            cometkim_aot_ns: 7.97,
        },
        Case {
            label: "nested_expr(div)",
            src: "((a + b) * (c - d)) / ((e + f) - (g * h))",
            schema: &[
                ("a", ValType::Int),
                ("b", ValType::Int),
                ("c", ValType::Int),
                ("d", ValType::Int),
                ("e", ValType::Int),
                ("f", ValType::Int),
                ("g", ValType::Int),
                ("h", ValType::Int),
            ],
            shape: nested_shape,
            cometkim_interp_ns: 156.97,
            cometkim_aot_ns: 140.45,
        },
        // green-length unroll: literal-list `all` folds to a constant bool.
        Case {
            label: "all_comprehension",
            src: "[1, 2, 3, 4, 5].all(x, x > 0)",
            schema: &[],
            shape: default_shape,
            cometkim_interp_ns: 512.50,
            cometkim_aot_ns: 197.40,
        },
    ];

    for case in &cases {
        run_case(case);
        let _ = case.cometkim_interp_ns;
    }
}
