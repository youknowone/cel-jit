//! Fair cel-jit benchmark suite.
//!
//! CEL is normally compiled once and evaluated once per request/resource with
//! a structured activation.  The primary panel therefore measures cached
//! `Program::execute(&Context)` over varying, prebuilt request contexts.  Input
//! construction is outside the timer.  The current majit prototype cannot be
//! placed in that panel: it exposes a columnar batch mainloop, not an equivalent
//! single-activation API.  Reporting its batch ns/row as request latency would
//! be a regime mismatch, so the JIT cell is explicitly N/A until an API with the
//! following shape exists:
//!
//! ```text
//! execute_jit(program, activation) -> Result<Value, ExecutionError>
//! ```
//!
//! Two secondary panels measure what is fair today:
//!
//! * **engine-only throughput** compares the plain Rust bytecode interpreter
//!   and compiled majit over the exact same lowered program and columns.
//! * **cold/break-even** creates a fresh JIT driver for each run and reports
//!   total time at several batch sizes.  This includes tracing and compilation,
//!   but excludes CEL parsing/lowering and column construction for both tiers.
//!
//! Both secondary panels drive `cel::majit::bytecode::float_bank` directly
//! rather than the public `cel::majit::batch` API, because both measure the
//! COLD path: a fresh driver per run, so tracing and compilation are inside the
//! timed region. The batch API keeps the driver warm across runs by design (see
//! `CONVERGENCE.md`), which is the steady state the other examples report and
//! the opposite of what this file is for.
//!
//! No ratio crosses the request/columnar boundary.  RELEASE ONLY.
//! Run: `./bench.sh` or
//! `cargo run --release --example majit_ab --features jit-cranelift`
//! (or `jit-dynasm`; `jit` alone names no backend and does not link).

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use cel::majit::bytecode::float_bank::{
    clean_interp_seeded_f, run_jit_persistent_f, run_jit_seeded_f, COMPILES, GUARD_FAILS,
};
use cel::majit::lower::{lower_typed, Schema, ValType};
use cel::{Context, Program, Value};

const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;
const REQUEST_SAMPLES: usize = 1_024;
const REQUEST_EVALS: usize = 200_000;
const ENGINE_ROWS: usize = 2_000_000;
const ROUNDS: usize = 5;
const COLD_ROUNDS: usize = 7;
const JIT_ON: u32 = 8;
const JIT_OFF: u32 = u32::MAX;

struct RequestSample<'a> {
    context: Context<'a>,
    expected: bool,
}

#[inline]
fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    *state
}

fn bool_result(program: &Program, context: &Context<'_>) -> bool {
    match program.execute(context).expect("CEL evaluation failed") {
        Value::Bool(value) => value,
        other => panic!("expected boolean result, got {other:?}"),
    }
}

fn auth_samples<'a>(root: &'a Context<'a>) -> Vec<RequestSample<'a>> {
    let mut state = 0x2545_F491_4F6C_DD1D;
    (0..REQUEST_SAMPLES)
        .map(|_| {
            let balance = (next_u64(&mut state) % 20_001) as i64 - 10_000;
            let withdrawal = (next_u64(&mut state) % 20_001) as i64 - 10_000;
            let frozen = next_u64(&mut state) & 1 != 0;
            let expected = balance >= withdrawal && !frozen;

            let account = HashMap::from([
                ("balance".to_string(), Value::Int(balance)),
                ("frozen".to_string(), Value::Bool(frozen)),
            ]);
            let transaction = HashMap::from([("withdrawal".to_string(), Value::Int(withdrawal))]);
            let mut context = root.new_inner_scope();
            context.add_variable_from_value("account", account);
            context.add_variable_from_value("transaction", transaction);
            RequestSample { context, expected }
        })
        .collect()
}

fn routing_samples<'a>(root: &'a Context<'a>) -> Vec<RequestSample<'a>> {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let methods = ["GET", "POST", "PUT", "DELETE"];
    let paths = ["/admin/users", "/api/items", "/admin/audit", "/healthz"];
    (0..REQUEST_SAMPLES)
        .map(|_| {
            let method = methods[(next_u64(&mut state) as usize) & 3];
            let path = paths[(next_u64(&mut state) as usize) & 3];
            let expected = method == "GET" && path.starts_with("/admin/");
            let request = HashMap::from([
                ("method".to_string(), Value::from(method)),
                ("path".to_string(), Value::from(path)),
            ]);
            let mut context = root.new_inner_scope();
            context.add_variable_from_value("request", request);
            RequestSample { context, expected }
        })
        .collect()
}

fn validation_samples<'a>(root: &'a Context<'a>) -> Vec<RequestSample<'a>> {
    let mut state = 0xD1B5_4A32_D192_ED03;
    (0..REQUEST_SAMPLES)
        .map(|_| {
            let replicas = (next_u64(&mut state) % 14) as i64;
            let long_tag = next_u64(&mut state) & 7 == 0;
            let tags = if long_tag {
                vec!["production", "tag-that-is-over-sixteen"]
            } else {
                vec!["production", "api"]
            };
            let expected = (1..=10).contains(&replicas) && !long_tag;
            let object = HashMap::from([
                ("replicas".to_string(), Value::Int(replicas)),
                ("tags".to_string(), Value::from(tags)),
            ]);
            let mut context = root.new_inner_scope();
            context.add_variable_from_value("object", object);
            RequestSample { context, expected }
        })
        .collect()
}

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn request_latency(label: &str, expression: &str, schema: &Schema, samples: &[RequestSample<'_>]) {
    let program = Program::compile(expression).expect("compile request expression");
    for sample in samples {
        assert_eq!(
            bool_result(&program, &sample.context),
            sample.expected,
            "{label}: oracle mismatch"
        );
    }

    // Warm the stock evaluator and its instruction/data caches. Parsing and
    // activation construction are deliberately outside the measured region.
    for sample in samples {
        black_box(bool_result(&program, &sample.context));
    }

    let mut timings = Vec::with_capacity(ROUNDS);
    let mut observed = 0usize;
    for _ in 0..ROUNDS {
        let start = Instant::now();
        for i in 0..REQUEST_EVALS {
            observed ^= black_box(bool_result(
                black_box(&program),
                black_box(&samples[i & (REQUEST_SAMPLES - 1)].context),
            )) as usize;
        }
        timings.push(start.elapsed().as_nanos() as f64 / REQUEST_EVALS as f64);
    }
    black_box(observed);

    let stock_ns = median_f64(timings);
    // Carry the decline REASON, not just the verdict: a schema this harness
    // declared wrong reads exactly like a structural limit of the subset if all
    // that is printed is "no".
    let subset = match lower_typed(program.expression(), schema) {
        Ok(_) => "lowerable".to_string(),
        Err(e) => format!("not lowerable: {e}"),
    };
    println!("{label}: {expression}");
    println!("  stock cached Program::execute : {stock_ns:>9.2} ns/eval");
    println!("  majit single-activation JIT   :       N/A  (no door; {subset})");
}

fn make_engine_columns(n: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut state = 0x2545_F491_4F6C_DD1D;
    let mut balance = Vec::with_capacity(n);
    let mut amount = Vec::with_capacity(n);
    let mut frozen = Vec::with_capacity(n);
    for _ in 0..n {
        balance.push((next_u64(&mut state) % 2_000_001) as i64 - 1_000_000);
        amount.push((next_u64(&mut state) % 2_000_001) as i64 - 1_000_000);
        frozen.push((next_u64(&mut state) & 1) as i64);
    }
    (balance, amount, frozen)
}

/// The words come back as an `Arc` the caller owns, not as a `Vec`: the
/// `LoweredF` built inside is dropped on return, and the JIT's green key is the
/// program POINTER, so whoever wants one key across several calls has to hold
/// that allocation. `warm_break_even` does; `cold_break_even` deliberately does
/// not.
fn engine_program(
    n: usize,
    balance: &[i64],
    amount: &[i64],
    frozen: &[i64],
) -> (std::sync::Arc<[i64]>, Vec<i64>, usize) {
    let program =
        Program::compile("balance >= amount && !frozen").expect("compile engine expression");
    // `frozen` is a `bool` column — stored as `0`/`1` in the int file like the
    // other two, but declared as what it is, which is what lets `!frozen` lower.
    let schema: Schema = [
        ("balance".to_string(), ValType::Int),
        ("amount".to_string(), ValType::Int),
        ("frozen".to_string(), ValType::Bool),
    ]
    .into_iter()
    .collect();
    let lowered = lower_typed(program.expression(), &schema).expect("engine expression must lower");
    let bases = [
        balance.as_ptr() as i64,
        amount.as_ptr() as i64,
        frozen.as_ptr() as i64,
    ];
    let (shape, regs) = lowered.batch_sum_program(&bases, n as i64);
    (shape.code, regs, shape.num_float_regs)
}

fn time_ns_per_row<F: FnMut() -> i64>(n: usize, mut run: F) -> f64 {
    let start = Instant::now();
    black_box(run());
    start.elapsed().as_nanos() as f64 / n as f64
}

fn engine_only() {
    let (balance, amount, frozen) = make_engine_columns(ENGINE_ROWS);
    let (code, regs, nf) = engine_program(ENGINE_ROWS, &balance, &amount, &frozen);

    COMPILES.store(0, Ordering::Relaxed);
    let clean = clean_interp_seeded_f(&code, &regs, nf);
    let off = run_jit_seeded_f(&code, &regs, nf, JIT_OFF);
    assert_eq!(COMPILES.load(Ordering::Relaxed), 0);
    COMPILES.store(0, Ordering::Relaxed);
    let compiled = run_jit_seeded_f(&code, &regs, nf, JIT_ON);
    let compiles = COMPILES.load(Ordering::Relaxed);
    assert_eq!(clean, off, "clean VM vs majit interpreter divergence");
    assert_eq!(clean, compiled, "clean VM vs compiled trace divergence");
    assert!(compiles >= 1, "hot engine loop did not compile");

    let mut clean_times = Vec::with_capacity(ROUNDS);
    let mut jit_times = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        clean_times.push(time_ns_per_row(ENGINE_ROWS, || {
            clean_interp_seeded_f(&code, &regs, nf)
        }));
        jit_times.push(time_ns_per_row(ENGINE_ROWS, || {
            run_jit_seeded_f(&code, &regs, nf, JIT_ON)
        }));
    }
    let clean_ns = median_f64(clean_times);
    let jit_ns = median_f64(jit_times);
    println!("same lowered bytecode + same columns, {ENGINE_ROWS} rows:");
    println!("  clean Rust bytecode VM : {clean_ns:>9.2} ns/row");
    println!("  compiled majit trace   : {jit_ns:>9.2} ns/row");
    println!("  JIT-only throughput    : {:>9.2}x", clean_ns / jit_ns);
    println!("  timing scope           : one fresh trace/compile included per batch");
    println!("  correctness            : result={compiled}, compiles={compiles}");
}

/// The same sweep with the driver the batch API actually runs on: the trace is
/// compiled once and every later call reuses it, so what is left is the FIXED
/// cost of entering and leaving compiled code.
///
/// It is what decides whether a small batch should reach the compiled tier at
/// all, and the reason there is no single-activation door: at one row the fixed
/// cost is the whole evaluation.
fn warm_break_even() {
    const SIZES: &[usize] = &[1, 2, 8, 16, 64, 256, 1_024, 4_096, 16_384];
    let max_n = *SIZES.last().unwrap();
    let (balance, amount, frozen) = make_engine_columns(max_n);

    println!("persistent driver, compile amortized (the tier the batch API runs):");
    println!("      rows    clean total      JIT total    JIT/clean  guard fails/call");
    let mut first_win = None;
    let mut program: Option<std::sync::Arc<[i64]>> = None;
    for &n in SIZES {
        let (words, regs, nf) = engine_program(n, &balance[..n], &amount[..n], &frozen[..n]);
        // The words are the same for every size (the row count rides a seeded
        // register), so ONE allocation is reused across the sweep. The JIT's
        // green key is the program POINTER, so rebuilding the words per size
        // would mint a new key at every point; holding one owner instead is
        // exactly what a `LoweredF` gives `BatchProgram` between batches.
        let code = match &program {
            Some(p) => {
                assert_eq!(&p[..], &words[..], "n={n}: the program words changed");
                std::sync::Arc::clone(p)
            }
            None => {
                program = Some(std::sync::Arc::clone(&words));
                words
            }
        };
        let expected = clean_interp_seeded_f(&code, &regs, nf);
        // Warm: enough calls that the loop is compiled and stays compiled, so
        // no call in the timed region pays for a trace.
        for _ in 0..64 {
            assert_eq!(run_jit_persistent_f(&code, &regs, nf, JIT_ON), expected);
        }

        let mut clean_times = Vec::with_capacity(COLD_ROUNDS);
        let mut jit_times = Vec::with_capacity(COLD_ROUNDS);
        let g0 = GUARD_FAILS.load(Ordering::Relaxed);
        let mut calls = 0usize;
        for _ in 0..COLD_ROUNDS {
            let start = Instant::now();
            black_box(clean_interp_seeded_f(&code, &regs, nf));
            clean_times.push(start.elapsed());

            let start = Instant::now();
            black_box(run_jit_persistent_f(&code, &regs, nf, JIT_ON));
            jit_times.push(start.elapsed());
            calls += 1;
        }
        let fails = (GUARD_FAILS.load(Ordering::Relaxed) - g0) as f64 / calls as f64;
        let clean = median_duration(clean_times);
        let jit = median_duration(jit_times);
        if jit <= clean && first_win.is_none() {
            first_win = Some(n);
        }
        println!(
            "{n:>10}  {:>10.3} us  {:>12.3} us  {:>10.2}x  {fails:>16.2}",
            clean.as_secs_f64() * 1e6,
            jit.as_secs_f64() * 1e6,
            jit.as_secs_f64() / clean.as_secs_f64(),
        );
    }
    match first_win {
        Some(n) => println!("  first measured JIT win over clean VM: {n} rows"),
        None => println!("  no measured JIT win over clean VM in this size sweep"),
    }
}

fn cold_break_even() {
    const SIZES: &[usize] = &[1, 8, 16, 64, 256, 1_024, 4_096, 16_384, 65_536, 262_144];
    let max_n = *SIZES.last().unwrap();
    let (balance, amount, frozen) = make_engine_columns(max_n);

    println!("fresh driver each run; parse/lower/column build excluded:");
    println!("      rows    clean total      JIT total    JIT/clean  compiles");
    let mut first_win = None;
    for &n in SIZES {
        let (code, regs, nf) = engine_program(n, &balance[..n], &amount[..n], &frozen[..n]);
        let expected = clean_interp_seeded_f(&code, &regs, nf);
        COMPILES.store(0, Ordering::Relaxed);
        assert_eq!(run_jit_seeded_f(&code, &regs, nf, JIT_ON), expected);
        let compiles = COMPILES.load(Ordering::Relaxed);

        let mut clean_times = Vec::with_capacity(COLD_ROUNDS);
        let mut jit_times = Vec::with_capacity(COLD_ROUNDS);
        for _ in 0..COLD_ROUNDS {
            let start = Instant::now();
            black_box(clean_interp_seeded_f(&code, &regs, nf));
            clean_times.push(start.elapsed());

            COMPILES.store(0, Ordering::Relaxed);
            let start = Instant::now();
            black_box(run_jit_seeded_f(&code, &regs, nf, JIT_ON));
            jit_times.push(start.elapsed());
        }
        let clean = median_duration(clean_times);
        let jit = median_duration(jit_times);
        if jit <= clean && first_win.is_none() {
            first_win = Some(n);
        }
        println!(
            "{n:>10}  {:>10.3} us  {:>12.3} us  {:>10.2}x  {compiles:>8}",
            clean.as_secs_f64() * 1e6,
            jit.as_secs_f64() * 1e6,
            jit.as_secs_f64() / clean.as_secs_f64(),
        );
    }
    match first_win {
        Some(n) => println!("  first measured JIT win over clean VM: {n} rows"),
        None => println!("  no measured JIT win over clean VM in this size sweep"),
    }
}

fn main() {
    println!("=== 1. REAL CEL REQUEST LATENCY (primary; compile once/evaluate many) ===");
    // One shared root, a child scope per sample. `Context::default` constructs
    // `Env::stdlib()` on every call, so one per sample would hold
    // `REQUEST_SAMPLES` copies of the standard library alive while they are
    // cycled through the timed region.
    let root = Context::default();
    request_latency(
        "authorization",
        "account.balance >= transaction.withdrawal && !account.frozen",
        &[
            ("account.balance", ValType::Int),
            ("transaction.withdrawal", ValType::Int),
            ("account.frozen", ValType::Bool),
        ]
        .into_iter()
        .map(|(p, t)| (p.to_string(), t))
        .collect(),
        &auth_samples(&root),
    );
    request_latency(
        "HTTP routing",
        r#"request.method == "GET" && request.path.startsWith("/admin/")"#,
        &[
            ("request.method", ValType::Str),
            ("request.path", ValType::Str),
        ]
        .into_iter()
        .map(|(p, t)| (p.to_string(), t))
        .collect(),
        &routing_samples(&root),
    );
    request_latency(
        "resource validation",
        "object.tags.all(t, t.size() <= 16) && object.replicas >= 1 && object.replicas <= 10",
        &[
            ("object.tags[]", ValType::Str),
            ("object.replicas", ValType::Int),
        ]
        .into_iter()
        .map(|(p, t)| (p.to_string(), t))
        .collect(),
        &validation_samples(&root),
    );

    println!();
    println!("=== 2. JIT-ONLY ENGINE THROUGHPUT (secondary; not request latency) ===");
    engine_only();

    println!();
    println!("=== 3. COLD / BREAK-EVEN AGAINST THE SAME CLEAN VM ===");
    cold_break_even();

    println!();
    println!("=== 3b. WARM / FIXED PER-CALL COST OF THE COMPILED TIER ===");
    warm_break_even();

    println!();
    println!("No stock/JIT ratio is reported: current majit has no equivalent");
    println!("single-activation API and its columnar batch path is a different workload.");
}
