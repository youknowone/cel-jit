//! Where does COMPILING pay, and where does the compiled door cost more than it
//! saves?
//!
//! The two boards beside this one each answer half of that and neither answers
//! it head-on. `majit_vs_cometkim` runs one batch of 50 000 rows, where the
//! compiled tier always wins; `majit_vs_cometkim_percall` runs one row, where it
//! always loses. Both are real regimes and the interesting quantity is the
//! BOUNDARY between them, which neither prints: the batch size at which the same
//! lowered program stops being cheaper to interpret and starts being cheaper to
//! compile.
//!
//! Three panels, each answering a question that was asked of the earlier boards
//! and could not be answered from them.
//!
//! **A — Crossover.** One expression, one bound activation, swept over batch
//! sizes, with `Tier::Clean` and `Tier::Jit` timed on the SAME bound batch at
//! each size. The crossover row is the answer: at and above it, compiling wins.
//!
//! **B — Accumulation.** "A request builds a fresh activation and evaluates it
//! once — but the requests keep coming. Doesn't it eventually become hot?" It
//! does: the function-entry door counts CALLS, not rows, so repeated one-row
//! calls warm and then enter compiled code. This panel shows the warm-up as it
//! happens — the call index at which the artifact is minted, the call index at
//! which entries begin, and the per-call cost before and after — beside the
//! interpreter that needs no warm-up at all. A door that fires and still loses
//! is a different finding from a door that never fires, and the earlier boards
//! could not tell them apart.
//!
//! **C — Coverage.** What the lowering accepts, asked twice: once through
//! `BatchProgram::from_program`, which is what both boards call, and once
//! through `from_program_in`, which is handed the `Context` carrying the user's
//! functions. An expression that declines in the first and lowers in the second
//! was never a capability gap; it was a harness that did not pass the functions.
//!
//! Every timed closure UNWRAPS. Every tier's answer is compared against the tree
//! walker's before anything is timed, so a divergence is a failure and not a
//! fast number.
//!
//! RELEASE ONLY:
//!
//! ```text
//! cargo run --profile bench -p cel --features jit-cranelift --example jit_regime
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, RowReader, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

/// Batch sizes panel A sweeps. Dense at the bottom because that is where the
/// crossover is: the compiled door's fixed cost is ~85 ns and a row of compiled
/// work is under a nanosecond, so the interesting region is single-digit to
/// low-hundreds of rows, not the decades above it.
const LADDER: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192];

/// Calls panel B makes at one row before it stops looking for a change.
const ACCUMULATE_CALLS: usize = 20_000;

/// Calls used to settle a tier before timing it: the compiled tier has to warm
/// up through the same door a caller would, and a tier timed during its own
/// warm-up is neither the interpreter's number nor the compiled one's.
const SETTLE: usize = 4_000;

/// Calls the counter probe makes, after settling and before timing. Its only
/// job is to give the counters a denominator: a raw count over the timed loop
/// reports however many calls the timer happened to make, which is not a rate.
const PROBE_CALLS: usize = 1_000;

// -- timing ---------------------------------------------------------------

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}
extern "C" {
    fn clock_gettime(clk: u32, ts: *mut Timespec) -> i32;
}
/// `CLOCK_THREAD_CPUTIME_ID` on Darwin. Thread CPU rather than wall clock, so a
/// loaded box costs repeatability and not truth.
const CLOCK_THREAD_CPUTIME_ID: u32 = 16;

fn cpu_now() -> Duration {
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: writes through the pointer and does nothing else; the pointer is
    // to a live local of exactly the type it expects.
    let rc = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime failed");
    Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
}

/// Best batch of at least 20 ms of thread CPU within `secs` wall seconds, in
/// ns per call. Best rather than mean: descheduling and cache damage only ever
/// add time, so the minimum is the least-disturbed sample.
fn time<F: FnMut()>(secs: f64, mut one: F) -> f64 {
    let start = Instant::now();
    let mut k = 1u64;
    loop {
        let t0 = cpu_now();
        for _ in 0..k {
            one();
        }
        if cpu_now() - t0 >= Duration::from_millis(20) {
            break;
        }
        k *= 2;
    }
    let mut best = f64::INFINITY;
    let mut n = 0;
    while n == 0 || start.elapsed().as_secs_f64() < secs {
        let t0 = cpu_now();
        for _ in 0..k {
            one();
        }
        best = best.min((cpu_now() - t0).as_nanos() as f64 / k as f64);
        n += 1;
    }
    best
}

// -- data -----------------------------------------------------------------

const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;
/// Elements in each row's list, for the cases whose input is a list column.
const LIST_LEN: i64 = 10;

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    *state
}

fn ints(state: &mut u64, n: usize, span: u64, bias: i64) -> Vec<i64> {
    (0..n)
        .map(|_| (next_u64(state) % span) as i64 + bias)
        .collect()
}

fn strings(state: &mut u64, n: usize, choices: &[&str]) -> Vec<String> {
    (0..n)
        .map(|_| choices[next_u64(state) as usize % choices.len()].to_string())
        .collect()
}

/// One input column, owned, in the layout both consumers need: the batch reads
/// it as a [`ColumnRef`] and the tree-walker oracle reads row `r` out of the
/// same object through [`RowReader`].
enum Col {
    Int(Vec<i64>),
    Str(Vec<String>),
    IntList { lens: Vec<i64>, elems: Vec<i64> },
}

impl Col {
    fn column_ref(&self) -> ColumnRef<'_> {
        match self {
            Col::Int(c) => ColumnRef::Int(c),
            Col::Str(c) => ColumnRef::Str(c),
            Col::IntList { lens, elems } => ColumnRef::List {
                lens,
                fields: vec![(None, ColumnRef::Int(elems))],
            },
        }
    }
}

fn int_list(state: &mut u64, n: usize, span: u64) -> Col {
    Col::IntList {
        lens: vec![LIST_LEN; n],
        elems: ints(state, n * LIST_LEN as usize, span, 0),
    }
}

// -- cases ----------------------------------------------------------------

struct Case {
    label: &'static str,
    src: &'static str,
    schema: &'static [(&'static str, ValType)],
    build: fn(usize) -> Vec<(&'static str, Col)>,
    /// User functions the expression calls, registered on the context that both
    /// the walker and `from_program_in` are given.
    register: fn(&mut Context<'static>),
    /// What panel A sweeps: `false` skips it (panel C still reports the case).
    sweep: bool,
}

fn no_functions(_ctx: &mut Context<'static>) {}

fn scalar_x(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x2545_F491_4F6C_DD1D;
    vec![("x", Col::Int(ints(&mut s, n, 64, 0)))]
}

fn nested_operands(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x9E37_79B9_7F4A_7C15;
    vec![
        ("a", Col::Int(ints(&mut s, n, 256, 0))),
        ("b", Col::Int(ints(&mut s, n, 256, 0))),
        ("c", Col::Int(ints(&mut s, n, 256, 0))),
        ("d", Col::Int(ints(&mut s, n, 256, 0))),
        ("e", Col::Int(ints(&mut s, n, 256, 256))),
        ("f", Col::Int(ints(&mut s, n, 256, 256))),
        ("g", Col::Int(ints(&mut s, n, 16, 0))),
        ("h", Col::Int(ints(&mut s, n, 16, 0))),
    ]
}

/// A policy shaped like a deployed one: every conjunct reads an input, so no
/// branch of the `&&` chain is constant and nothing folds at lower time.
fn policy_request(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x1D8E_4E27_C47D_124F;
    vec![
        ("user.age", Col::Int(ints(&mut s, n, 60, 5))),
        (
            "user.role",
            Col::Str(strings(&mut s, n, &["admin", "moderator", "viewer"])),
        ),
        (
            "request.method",
            Col::Str(strings(&mut s, n, &["POST", "GET"])),
        ),
        (
            "request.path",
            Col::Str(strings(&mut s, n, &["/api/users", "/health"])),
        ),
    ]
}

fn one_int_list(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x8EBC_6AF0_9C88_C6E3;
    vec![("list", int_list(&mut s, n, 1 << 16))]
}

fn custom_fn_operands(n: usize) -> Vec<(&'static str, Col)> {
    let mut s = 0x2B99_2DDF_A232_49D6;
    vec![
        ("x", Col::Int(ints(&mut s, n, 1 << 16, 0))),
        ("y", Col::Int(ints(&mut s, n, 1 << 16, 0))),
        ("a", Col::Int(ints(&mut s, n, 1 << 16, 0))),
        ("b", Col::Int(ints(&mut s, n, 1 << 16, 0))),
    ]
}

fn register_add_multiply(ctx: &mut Context<'static>) {
    ctx.add_function("add", |a: i64, b: i64| a + b);
    ctx.add_function("multiply", |a: i64, b: i64| a * b);
}

const CASES: &[Case] = &[
    Case {
        label: "conditional",
        src: "x > 10 ? x * 2 : x + 5",
        schema: &[("x", ValType::Int)],
        build: scalar_x,
        register: no_functions,
        sweep: true,
    },
    Case {
        label: "nested_arithmetic",
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
        build: nested_operands,
        register: no_functions,
        sweep: true,
    },
    Case {
        label: "policy",
        src: "user.age >= 18 && user.role in [\"admin\", \"moderator\"] \
              && request.method == \"POST\" && request.path.startsWith(\"/api/\")",
        schema: &[
            ("user.age", ValType::Int),
            ("user.role", ValType::Str),
            ("request.method", ValType::Str),
            ("request.path", ValType::Str),
        ],
        build: policy_request,
        register: no_functions,
        sweep: true,
    },
    Case {
        label: "list_exists",
        src: "list.exists(e, e > 100)",
        schema: &[("list[]", ValType::Int)],
        build: one_int_list,
        register: no_functions,
        sweep: true,
    },
    Case {
        label: "custom_function",
        src: "add(x, y) + multiply(a, b)",
        schema: &[
            ("x", ValType::Int),
            ("y", ValType::Int),
            ("a", ValType::Int),
            ("b", ValType::Int),
        ],
        build: custom_fn_operands,
        register: register_add_multiply,
        sweep: true,
    },
];

// -- harness --------------------------------------------------------------

/// The walker's answer for every row of `batch`, through the library's own
/// `RowReader`, so the oracle cannot drift from what the machine is fed.
fn oracle(program: &Program, base: &Context, batch: &Batch, n: usize) -> Vec<Value> {
    let reader = RowReader::new(batch);
    (0..n)
        .map(|r| {
            let ctx = reader.scope(base, r);
            Value::resolve_value(program.expression(), &ctx).expect("walker answers")
        })
        .collect()
}

/// Build the columns, bind them, and check both tiers against the walker.
///
/// Returns the bound batch and whether it was bound per row, or the reason it
/// could not be built. Held together in one function because a bound batch
/// borrows the batch and the columns, so a caller cannot take the three apart.
struct Prepared {
    cols: Vec<(&'static str, Col)>,
}

fn prepare(case: &Case, n: usize) -> Prepared {
    Prepared {
        cols: (case.build)(n),
    }
}

fn build_batch<'a>(prepared: &'a Prepared, n: usize) -> Batch<'a> {
    let mut batch = Batch::new(n);
    for (name, col) in &prepared.cols {
        batch = batch.column(*name, col.column_ref());
    }
    batch
}

fn schema_of(case: &Case) -> Schema {
    case.schema
        .iter()
        .map(|(p, t)| (p.to_string(), *t))
        .collect()
}

/// Run `bound` on `tier` into a reused buffer, and check it against the oracle
/// once before any timing.
fn check(bound: &BoundBatch, tier: Tier, expected: &[Value], label: &str) {
    let mut out = Vec::new();
    bound
        .collect_into_on(tier, &mut out)
        .unwrap_or_else(|e| panic!("{label}: {tier:?} refused: {e}"));
    assert_eq!(out.len(), expected.len(), "{label}: {tier:?} row count");
    assert_eq!(out, expected, "{label}: {tier:?} disagrees with the walker");
}

fn panel_a() {
    println!("== A. Crossover: the same bound program, interpreted and compiled, per batch size");
    println!(
        "   `clean` is the batch machine's interpreter over lowered code; `jit` is the compiled"
    );
    println!("   trace. ns/call is the whole call; ns/row divides by the batch size.\n");
    for case in CASES {
        let program = Program::compile(case.src).expect("compiles");
        let schema = schema_of(case);
        let mut base = Context::default();
        (case.register)(&mut base);
        let lowered = match BatchProgram::from_program_in(&program, &schema, &base) {
            Ok(bp) => bp,
            Err(e) => {
                println!("{}: declines to lower: {e}\n", case.label);
                continue;
            }
        };
        if !case.sweep {
            continue;
        }
        println!(
            "{}  `{}`",
            case.label,
            case.src.split_whitespace().collect::<Vec<_>>().join(" ")
        );
        println!(
            "   {:>6} {:>12} {:>12} {:>9} | {:>10} {:>10} | {:>10} {:>10}",
            "rows",
            "clean ns",
            "jit ns",
            "jit/clean",
            "clean/row",
            "jit/row",
            "enter/call",
            "gfail/call"
        );
        let mut crossover = None;
        for &n in LADDER {
            let prepared = prepare(case, n);
            let batch = build_batch(&prepared, n);
            let expected = oracle(&program, &base, &batch, n);
            // Always per-row: this file compares the tiers on the VALUES a
            // caller gets back, one per row, which is the contract
            // `Program::execute` has. `bind` binds for a reduction instead, and
            // `collect_into_on` refuses one.
            reset_persistent_state();
            let bound = lowered
                .bind_per_row(&batch)
                .unwrap_or_else(|e| panic!("{}: bind at {n} rows: {e}", case.label));
            check(&bound, Tier::Clean, &expected, case.label);
            check(&bound, Tier::Jit, &expected, case.label);

            let mut out = Vec::new();
            for _ in 0..SETTLE {
                bound
                    .collect_into_on(Tier::Jit, &mut out)
                    .expect("jit answers");
            }
            // Counters over a window of KNOWN length, so the column is a rate
            // and not a total: a raw count over the timed loop reports however
            // many calls the timer happened to make, which says nothing.
            reset_jit_stats();
            for _ in 0..PROBE_CALLS {
                bound
                    .collect_into_on(Tier::Jit, &mut out)
                    .expect("jit answers");
            }
            let stats = jit_stats();
            let entered = stats.compiled_entries as f64 / PROBE_CALLS as f64;
            let gfails = stats.guard_failures as f64 / PROBE_CALLS as f64;
            let jit = time(0.25, || {
                bound
                    .collect_into_on(Tier::Jit, black_box(&mut out))
                    .expect("jit answers");
            });
            let clean = time(0.25, || {
                bound
                    .collect_into_on(Tier::Clean, black_box(&mut out))
                    .expect("clean answers");
            });
            if crossover.is_none() && jit < clean {
                crossover = Some(n);
            }
            println!(
                "   {n:>6} {clean:>12.1} {jit:>12.1} {:>8.2}x | {:>10.2} {:>10.2} | {entered:>10.2} {gfails:>10.2}",
                jit / clean,
                clean / n as f64,
                jit / n as f64,
            );
        }
        match crossover {
            Some(n) => println!("   -> compiling pays from {n} rows up\n"),
            None => println!("   -> compiling never pays at any size on this ladder\n"),
        }
    }
}

fn panel_b() {
    println!("== B. Accumulation: one row per call, many calls, no warm-up given");
    println!("   The question this answers: a server builds a fresh activation per request and");
    println!("   evaluates it once, but the requests keep coming -- does the JIT ever fire?");
    println!("   `entries` counts calls that ENTERED compiled code, so a non-zero column is the");
    println!("   door firing. The window costs are wall-clock over the window, not best-of.\n");
    println!(
        "   {:>22} {:>10} {:>12} {:>10} {:>10}",
        "case", "calls", "window ns/call", "compiles", "entries"
    );
    for case in CASES {
        let program = Program::compile(case.src).expect("compiles");
        let schema = schema_of(case);
        let mut base = Context::default();
        (case.register)(&mut base);
        let Ok(lowered) = BatchProgram::from_program_in(&program, &schema, &base) else {
            continue;
        };
        let prepared = prepare(case, 1);
        let batch = build_batch(&prepared, 1);
        let expected = oracle(&program, &base, &batch, 1);
        reset_persistent_state();
        reset_jit_stats();
        let bound = lowered.bind_per_row(&batch).expect("binds at one row");
        check(&bound, Tier::Jit, &expected, case.label);

        let mut out = Vec::new();
        let mut done = 0usize;
        for &upto in &[1usize, 10, 100, 1_000, 5_000, ACCUMULATE_CALLS] {
            // The window is the calls this rung ADDS, so each row prices the
            // calls made since the row above it rather than an average that
            // keeps the cold first call in it forever.
            let in_window = upto - done;
            let t0 = cpu_now();
            while done < upto {
                bound
                    .collect_into_on(Tier::Jit, &mut out)
                    .expect("jit answers");
                done += 1;
            }
            let per_call = (cpu_now() - t0).as_nanos() as f64 / in_window as f64;
            let stats = jit_stats();
            println!(
                "   {:>22} {:>10} {:>12.1} {:>10} {:>10}",
                if in_window == upto { case.label } else { "" },
                upto,
                per_call,
                stats.loops_compiled,
                stats.compiled_entries
            );
        }
        // The interpreter needs no warm-up; print it as the thing the door has
        // to beat, measured the same way the crossover panel measures it.
        let clean = time(0.25, || {
            bound
                .collect_into_on(Tier::Clean, black_box(&mut out))
                .expect("clean answers");
        });
        let jit = time(0.25, || {
            bound
                .collect_into_on(Tier::Jit, black_box(&mut out))
                .expect("jit answers");
        });
        println!(
            "   {:>22} {:>10} {:>12.1} settled: clean {clean:.1} ns, jit {jit:.1} ns ({:.2}x)",
            "",
            "settled",
            jit,
            jit / clean
        );
    }
    println!();
}

fn panel_c() {
    println!("== C. Coverage: what lowers, asked with and without the caller's functions");
    println!("   `from_program` is what the two existing boards call. `from_program_in` is handed");
    println!("   the Context that carries the user's functions.\n");
    println!(
        "   {:>22} {:>16} {:>16}   reason it declines",
        "case", "from_program", "from_program_in"
    );
    for case in CASES {
        let program = Program::compile(case.src).expect("compiles");
        let schema = schema_of(case);
        let mut base = Context::default();
        (case.register)(&mut base);
        let bare = BatchProgram::from_program(&program, &schema);
        let with_ctx = BatchProgram::from_program_in(&program, &schema, &base);
        let why = match (&bare, &with_ctx) {
            (Err(e), Ok(_)) => format!("bare: {e}"),
            (Err(_), Err(e)) => format!("both: {e}"),
            _ => String::new(),
        };
        println!(
            "   {:>22} {:>16} {:>16}   {why}",
            case.label,
            if bare.is_ok() { "lowers" } else { "declines" },
            if with_ctx.is_ok() {
                "lowers"
            } else {
                "declines"
            },
        );
    }
    println!();
}

fn main() {
    println!("cel batch machine: interpreter vs compiled trace, by regime");
    println!("thread-CPU timing, best batch of >= 20 ms; every tier checked against the walker\n");
    panel_c();
    panel_a();
    panel_b();
}
