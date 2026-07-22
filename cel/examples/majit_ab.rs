//! Honest A/B: majit-JIT'd CEL policy evaluation vs cel-rust's stock
//! tree-walking `Program::execute`, over a batch of N synthetic rows.
//!
//! Policy: `a >= b && !c` (scalar int/int/bool). Each row's field values are a
//! serial LCG recurrence (`x = x*A + C`, Knuth MMIX constants) — distinct,
//! data-dependent, and NOT constant-foldable (a counter-derived value would let
//! the optimizer strength-reduce the whole loop into a bogus speedup). Both
//! sides consume the *identical* LCG stream, so their accumulators must match
//! bit-for-bit (the correctness gate).
//!
//!   * naive: for each row, advance the LCG in native Rust, bind `a`/`b`/`c`
//!     into a `Context`, and call `Program::execute` (BTreeMap lookups + `dyn
//!     Val` dispatch + boxed results — the stock cost of evaluating varying
//!     policy inputs).
//!   * majit: one mainloop call over a bytecode batch program that inlines the
//!     LCG generation and the lowered predicate; the outer row loop is the
//!     tracing merge point, so it compiles to a native loop.
//!   * naive-floor: `execute` over a *fixed* context (no per-row bind) — the
//!     fastest cel-rust can evaluate this policy at all. majit paying its own
//!     data-gen still has to beat this to be an unambiguous win.
//!
//! RELEASE ONLY (i64 wrap; the 3-way accumulator equality gate catches any
//! miscompile). Run: `cargo run --release --example majit_ab --features majit-jit`.

use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Instant;

use cel::majit::bytecode::{clean_interp, run_jit, Code, COMPILES};
use cel::majit::bytecode::{
    OP_ADD, OP_AND, OP_JUMP_IF_ABOVE, OP_LOAD_CONST, OP_MOV, OP_MUL, OP_RETURN,
};
use cel::majit::lower::{lower, Lowered};
use cel::{Context, Program, Value};

const LCG_A: i64 = 6364136223846793005;
const LCG_C: i64 = 1442695040888963407;
const SEED: i64 = 0x2545F4914F6CDD1D;

const POLICY: &str = "a >= b && !c";

/// Advance the LCG and return the next full-range draw. Matches the bytecode
/// `MUL x,A,x ; ADD x,C,x` (release i64 wrap).
#[inline(always)]
fn lcg_next(x: &mut i64) -> i64 {
    *x = x.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    *x
}

/// Build the batch bytecode program: prologue (loop constants) + a per-row body
/// that inlines the LCG field generation and the lowered predicate, accumulates
/// the result, and loops back via a backward `JIA` (the majit merge point).
///
/// Slot generation is policy-specific: `a`/`b` get full-range LCG draws, `c`
/// gets the LCG low bit (a bool). Extra machine registers are allocated above
/// the lowering's register range.
fn build_batch(lowered: &Lowered, n: i64) -> Vec<i64> {
    let base = lowered.num_regs;
    let r_x = (base) as i64;
    let r_a = (base + 1) as i64; // LCG multiplier
    let r_c_const = (base + 2) as i64; // LCG increment
    let r_one = (base + 3) as i64;
    let r_i = (base + 4) as i64;
    let r_n = (base + 5) as i64;
    let r_acc = (base + 6) as i64;
    // total registers = base + 7

    let mut p: Vec<i64> = Vec::new();
    // prologue
    p.extend_from_slice(&[OP_LOAD_CONST, SEED, r_x]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_A, r_a]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_C, r_c_const]);
    p.extend_from_slice(&[OP_LOAD_CONST, 1, r_one]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_i]);
    p.extend_from_slice(&[OP_LOAD_CONST, n, r_n]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_acc]);

    let body_pc = p.len() as i64;

    // per-row: generate each slot's field value from a fresh LCG draw
    for slot in &lowered.slots {
        let dst = slot.reg as i64;
        // draw: x = x*A + C
        p.extend_from_slice(&[OP_MUL, r_x, r_a, r_x]);
        p.extend_from_slice(&[OP_ADD, r_x, r_c_const, r_x]);
        if slot.path == "c" {
            // bool field: low bit of the draw
            p.extend_from_slice(&[OP_AND, r_x, r_one, dst]);
        } else {
            p.extend_from_slice(&[OP_MOV, r_x, dst]);
        }
    }
    // the lowered predicate computes into result_reg
    p.extend_from_slice(&lowered.body);
    // acc += result
    p.extend_from_slice(&[OP_ADD, r_acc, lowered.result_reg as i64, r_acc]);
    // i += 1
    p.extend_from_slice(&[OP_ADD, r_i, r_one, r_i]);
    // if n > i goto body
    p.extend_from_slice(&[OP_JUMP_IF_ABOVE, r_n, r_i, body_pc]);
    // return acc
    p.extend_from_slice(&[OP_RETURN, r_acc]);
    p
}

fn num_regs_total(lowered: &Lowered) -> usize {
    lowered.num_regs + 7
}

/// Naive baseline: advance the LCG in Rust, bind `a`/`b`/`c` per row, execute
/// the stock tree-walker, accumulate the boolean result.
fn naive_batch(program: &Program, n: i64) -> i64 {
    let mut x = SEED;
    let mut acc = 0i64;
    let mut ctx = Context::default();
    for _ in 0..n {
        let a = lcg_next(&mut x);
        let b = lcg_next(&mut x);
        let c = (lcg_next(&mut x) & 1) == 1;
        ctx.add_variable_from_value("a", a);
        ctx.add_variable_from_value("b", b);
        ctx.add_variable_from_value("c", c);
        match program.execute(&ctx).unwrap() {
            Value::Bool(t) => acc += t as i64,
            other => panic!("unexpected result {other:?}"),
        }
    }
    acc
}

/// naive-floor: same `execute`, but over a fixed context bound once. The
/// theoretical floor of cel-rust's per-eval cost (no per-row data feed).
fn naive_floor(program: &Program, n: i64) -> i64 {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("a", 7i64);
    ctx.add_variable_from_value("b", 3i64);
    ctx.add_variable_from_value("c", false);
    let mut acc = 0i64;
    for _ in 0..n {
        match program.execute(&ctx).unwrap() {
            Value::Bool(t) => acc += t as i64,
            other => panic!("unexpected result {other:?}"),
        }
    }
    acc
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

const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

fn main() {
    let n: i64 = 5_000_000;
    let rounds = 5;

    let program = Program::compile(POLICY).expect("compile policy");
    let lowered = lower(program.expression()).expect("policy is lowerable");
    let nregs = num_regs_total(&lowered);
    let batch: Vec<i64> = build_batch(&lowered, n);
    let code: &Code = &batch;

    println!("policy: {POLICY}   (slot-resolved; {} slots)\n", lowered.slots.len());

    // correctness gate: majit-on == majit-off == clean == naive
    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp(code, nregs);
    let off = run_jit(code, nregs, JIT_OFF);
    assert_eq!(COMPILES.load(Ordering::Relaxed), 0, "JIT-off must not compile");
    COMPILES.store(0, Ordering::Relaxed);
    let on = run_jit(code, nregs, JIT_ON);
    let compiles = COMPILES.load(Ordering::Relaxed);
    let naive = naive_batch(&program, n);
    assert_eq!(clean, off, "clean vs JIT-off divergence");
    assert_eq!(clean, on, "clean vs JIT-on divergence -> miscompile");
    assert_eq!(clean, naive, "majit vs stock tree-walker divergence -> miscompile");
    println!(
        "correctness OK: pass-count {on} / {n} ({:.1}%)   compiles(on)={compiles}\n",
        100.0 * on as f64 / n as f64
    );

    let (mut a_on, mut a_off, mut a_clean, mut a_naive, mut a_floor) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rounds {
        a_naive.push(time_ns(n, || naive_batch(&program, n)));
        a_floor.push(time_ns(n, || naive_floor(&program, n)));
        a_clean.push(time_ns(n, || clean_interp(code, nregs)));
        a_off.push(time_ns(n, || run_jit(code, nregs, JIT_OFF)));
        a_on.push(time_ns(n, || run_jit(code, nregs, JIT_ON)));
    }
    let (on, off, clean, naive, floor) = (
        median(a_on),
        median(a_off),
        median(a_clean),
        median(a_naive),
        median(a_floor),
    );

    println!("ns per row (median of {rounds}):");
    println!("  naive  (stock execute + per-row bind) : {naive:8.2}");
    println!("  floor  (stock execute, fixed context) : {floor:8.2}");
    println!("  clean  (bytecode interp, no JIT)      : {clean:8.2}");
    println!("  jit-off(majit interp tier)            : {off:8.2}");
    println!("  jit-on (majit compiled trace)         : {on:8.2}");
    println!();
    println!("majit-on speedup vs:");
    println!("  naive     = {:6.2}x   {}", naive / on, verdict(naive / on));
    println!("  floor     = {:6.2}x   {}", floor / on, verdict(floor / on));
    println!("  clean     = {:6.2}x", clean / on);
}

fn verdict(r: f64) -> &'static str {
    if r >= 1.0 {
        "JIT WINS"
    } else {
        "jit slower"
    }
}
