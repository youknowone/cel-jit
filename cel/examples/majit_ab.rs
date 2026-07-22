//! Honest A/B: majit-JIT'd CEL policy evaluation vs cel-rust's stock
//! tree-walking `Program::execute`, over a batch of N synthetic rows.
//!
//! Two policies, both evaluated over the *identical* LCG stream on each side so
//! their accumulators must match bit-for-bit (the correctness gate):
//!   * SCALAR:  `a >= b && !c`  — top-level int/int/bool variables.
//!   * MAP:     `account.balance >= txn.amount && !account.frozen`  — the real
//!     policy shape (member access). cometkim measured member_access at only
//!     1.1x for an AOT method-JIT because the context HashMap lookups dominate;
//!     this probes whether majit's compile-time slot resolution removes them.
//!
//! Each row's field values are a serial LCG recurrence (`x = x*A + C`, Knuth
//! MMIX constants) — distinct, data-dependent, NOT constant-foldable (a
//! counter-derived value would let the optimizer strength-reduce the loop).
//!
//!   * naive: advance the LCG in native Rust, bind the fields into a `Context`
//!     (for MAP, build the `account`/`txn` maps too), and call the stock
//!     `Program::execute` (BTreeMap/HashMap lookups + `dyn Val` dispatch + boxed
//!     results — the real cost of evaluating varying policy inputs).
//!   * majit: one mainloop over a bytecode batch program that inlines the LCG
//!     generation + the lowered predicate; the outer row loop is the tracing
//!     merge point, so it compiles to a native loop.
//!   * floor: `execute` over a *fixed* context — the fastest cel-rust can
//!     evaluate the policy at all (no per-row data feed). majit paying its own
//!     data-gen still has to beat this for an unambiguous win.
//!
//! RELEASE ONLY (i64 wrap; the accumulator equality gate catches miscompiles).
//! Run: `cargo run --release --example majit_ab --features majit-jit`.

use std::collections::HashMap;
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

const N: i64 = 5_000_000;
const ROUNDS: usize = 5;
const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

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
/// Each slot gets a fresh LCG draw; a slot whose path is in `bool_slots` gets
/// the draw's low bit (a bool). Machine registers are above the lowered range.
fn build_batch(lowered: &Lowered, n: i64, bool_slots: &[&str]) -> Vec<i64> {
    let base = lowered.num_regs;
    let r_x = base as i64;
    let r_a = (base + 1) as i64;
    let r_c_const = (base + 2) as i64;
    let r_one = (base + 3) as i64;
    let r_i = (base + 4) as i64;
    let r_n = (base + 5) as i64;
    let r_acc = (base + 6) as i64;

    let mut p: Vec<i64> = Vec::new();
    p.extend_from_slice(&[OP_LOAD_CONST, SEED, r_x]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_A, r_a]);
    p.extend_from_slice(&[OP_LOAD_CONST, LCG_C, r_c_const]);
    p.extend_from_slice(&[OP_LOAD_CONST, 1, r_one]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_i]);
    p.extend_from_slice(&[OP_LOAD_CONST, n, r_n]);
    p.extend_from_slice(&[OP_LOAD_CONST, 0, r_acc]);

    let body_pc = p.len() as i64;

    for slot in &lowered.slots {
        let dst = slot.reg as i64;
        p.extend_from_slice(&[OP_MUL, r_x, r_a, r_x]);
        p.extend_from_slice(&[OP_ADD, r_x, r_c_const, r_x]);
        if bool_slots.contains(&slot.path.as_str()) {
            p.extend_from_slice(&[OP_AND, r_x, r_one, dst]);
        } else {
            p.extend_from_slice(&[OP_MOV, r_x, dst]);
        }
    }
    p.extend_from_slice(&lowered.body);
    p.extend_from_slice(&[OP_ADD, r_acc, lowered.result_reg as i64, r_acc]);
    p.extend_from_slice(&[OP_ADD, r_i, r_one, r_i]);
    p.extend_from_slice(&[OP_JUMP_IF_ABOVE, r_n, r_i, body_pc]);
    p.extend_from_slice(&[OP_RETURN, r_acc]);
    p
}

fn bool_result(program: &Program, ctx: &Context) -> i64 {
    match program.execute(ctx).unwrap() {
        Value::Bool(t) => t as i64,
        other => panic!("unexpected result {other:?}"),
    }
}

// --- SCALAR policy: `a >= b && !c` ---

fn naive_scalar(program: &Program, n: i64) -> i64 {
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
        acc += bool_result(program, &ctx);
    }
    acc
}

fn floor_scalar(program: &Program, n: i64) -> i64 {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("a", 7i64);
    ctx.add_variable_from_value("b", 3i64);
    ctx.add_variable_from_value("c", false);
    let mut acc = 0i64;
    for _ in 0..n {
        acc += bool_result(program, &ctx);
    }
    acc
}

// --- MAP policy: `account.balance >= txn.amount && !account.frozen` ---

fn account_txn(balance: i64, amount: i64, frozen: bool) -> (HashMap<String, Value>, HashMap<String, Value>) {
    let mut account = HashMap::<String, Value>::new();
    account.insert("balance".to_string(), Value::Int(balance));
    account.insert("frozen".to_string(), Value::Bool(frozen));
    let mut txn = HashMap::<String, Value>::new();
    txn.insert("amount".to_string(), Value::Int(amount));
    (account, txn)
}

fn naive_map(program: &Program, n: i64) -> i64 {
    let mut x = SEED;
    let mut acc = 0i64;
    let mut ctx = Context::default();
    for _ in 0..n {
        let balance = lcg_next(&mut x);
        let amount = lcg_next(&mut x);
        let frozen = (lcg_next(&mut x) & 1) == 1;
        let (account, txn) = account_txn(balance, amount, frozen);
        ctx.add_variable_from_value("account", account);
        ctx.add_variable_from_value("txn", txn);
        acc += bool_result(program, &ctx);
    }
    acc
}

fn floor_map(program: &Program, n: i64) -> i64 {
    let mut ctx = Context::default();
    let (account, txn) = account_txn(7, 3, false);
    ctx.add_variable_from_value("account", account);
    ctx.add_variable_from_value("txn", txn);
    let mut acc = 0i64;
    for _ in 0..n {
        acc += bool_result(program, &ctx);
    }
    acc
}

// --- harness ---

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ns<F: FnMut() -> i64>(n: i64, mut f: F) -> f64 {
    let t = Instant::now();
    black_box(f());
    t.elapsed().as_nanos() as f64 / n as f64
}

fn run_policy(
    label: &str,
    src: &str,
    bool_slots: &[&str],
    naive: fn(&Program, i64) -> i64,
    floor: fn(&Program, i64) -> i64,
) {
    let program = Program::compile(src).expect("compile policy");
    let lowered = lower(program.expression()).expect("policy is lowerable");
    let nregs = lowered.num_regs + 7;
    let batch: Vec<i64> = build_batch(&lowered, N, bool_slots);
    let code: &Code = &batch;

    println!("=== {label}: {src}   ({} slots) ===", lowered.slots.len());

    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp(code, nregs);
    let off = run_jit(code, nregs, JIT_OFF);
    assert_eq!(COMPILES.load(Ordering::Relaxed), 0, "JIT-off must not compile");
    COMPILES.store(0, Ordering::Relaxed);
    let on = run_jit(code, nregs, JIT_ON);
    let compiles = COMPILES.load(Ordering::Relaxed);
    let nv = naive(&program, N);
    assert_eq!(clean, off, "{label}: clean vs JIT-off divergence");
    assert_eq!(clean, on, "{label}: clean vs JIT-on divergence -> miscompile");
    assert_eq!(clean, nv, "{label}: majit vs stock tree-walker divergence -> miscompile");
    println!(
        "correctness OK: pass-count {on} / {N} ({:.1}%)   compiles(on)={compiles}",
        100.0 * on as f64 / N as f64
    );

    let (mut a_on, mut a_off, mut a_clean, mut a_naive, mut a_floor) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ROUNDS {
        a_naive.push(time_ns(N, || naive(&program, N)));
        a_floor.push(time_ns(N, || floor(&program, N)));
        a_clean.push(time_ns(N, || clean_interp(code, nregs)));
        a_off.push(time_ns(N, || run_jit(code, nregs, JIT_OFF)));
        a_on.push(time_ns(N, || run_jit(code, nregs, JIT_ON)));
    }
    let (on_t, off_t, clean_t, naive_t, floor_t) = (
        median(a_on),
        median(a_off),
        median(a_clean),
        median(a_naive),
        median(a_floor),
    );

    println!("ns/row (median of {ROUNDS}):");
    println!("  naive  (stock execute + per-row bind) : {naive_t:8.2}");
    println!("  floor  (stock execute, fixed context) : {floor_t:8.2}");
    println!("  clean  (bytecode interp, no JIT)      : {clean_t:8.2}");
    println!("  jit-off(majit interp tier)            : {off_t:8.2}");
    println!("  jit-on (majit compiled trace)         : {on_t:8.2}");
    println!(
        "majit-on vs:  naive={:.2}x [{}]   floor={:.2}x [{}]   clean={:.2}x\n",
        naive_t / on_t,
        verdict(naive_t / on_t),
        floor_t / on_t,
        verdict(floor_t / on_t),
        clean_t / on_t,
    );
}

fn verdict(r: f64) -> &'static str {
    if r >= 1.0 {
        "JIT WINS"
    } else {
        "jit slower"
    }
}

fn main() {
    run_policy(
        "SCALAR",
        "a >= b && !c",
        &["c"],
        naive_scalar,
        floor_scalar,
    );
    run_policy(
        "MAP",
        "account.balance >= txn.amount && !account.frozen",
        &["account.frozen"],
        naive_map,
        floor_map,
    );
}
