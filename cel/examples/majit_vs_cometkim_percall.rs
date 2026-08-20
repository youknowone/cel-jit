//! cometkim's benchmark set (cel-jit PR #233, `benches/comparison.rs`) in
//! cometkim's OWN REGIME, so a number here can be read beside one of his.
//!
//! `majit_vs_cometkim` asks his expressions of the batch machine over 50,000
//! varying rows and says, at the top of the file, that this is NOT comparable to
//! his figures: his unit is `b.iter(|| compiled.execute(&ctx))` — one
//! evaluation, one FIXED activation, a program compiled once outside the timer.
//! This file measures that unit. Same expressions, same activations built the
//! same way (his literal values, added by name to a root `Context`), one call
//! timed.
//!
//! What each column is, and what it is not:
//!
//! * **stock** — `Value::resolve_value(program.expression(), &ctx)`, the tree
//!   walker, the same evaluator his `interpreted` column measures. His ran
//!   against upstream `cel` 0.11.6; this one runs against ours, so the two are
//!   one measurement of two versions of one evaluator. The walker is called
//!   DIRECTLY, not through `Program::execute`, because that door is the bytecode
//!   VM whenever the `vm` feature is on — a DEFAULT feature, which
//!   `required-features = ["jit"]` does not turn off. Going through it would
//!   silently make this column a different evaluator from the one his figures
//!   were taken on, which is the entire basis of the comparison.
//! * **exec** — `Program::execute`, the door a consumer of this library
//!   actually writes. With the `vm` feature — a DEFAULT feature — that door is
//!   `cel::vm::cel_eval_loop`, the bytecode VM, so `exec` and `stock` are one
//!   activation through TWO evaluators rather than one evaluator timed twice.
//!   It answers what no `stock/..` ratio can: whether the tiers below beat the
//!   evaluator a caller who names nothing already has. `jit` does not imply
//!   `vm`, so which of the two this build's `execute` reaches is PRINTED with
//!   the legend rather than assumed.
//! * **majit** — the compiled tier through a ONE-ROW batch: `bind_per_row` once,
//!   `collect_into_on(Tier::Jit, &mut out)` per call. That builds the row's
//!   `Value`, which is what `execute` returns, so it is the same contract.
//!   Published only where the calls actually ENTERED compiled code —
//!   `enter/call` is the gate, and a case that did not reads `not entered`
//!   rather than a number that would be the tracing interpreter's.
//!
//!   ⚠ Why the buffer door and not `collect_on`, which returns a fresh
//!   `Vec<Value>`: `stock` returns ONE `Value` and allocates no container to
//!   carry it, so a `Vec` per call is ~11 ns this side pays for the batch
//!   API's shape and not for evaluating the expression. `collect_into_on`
//!   removes exactly that and nothing else — every row's `Value` is still
//!   built, by the same code, and the buffer belongs to the caller, who in any
//!   real per-call loop owns one already. `clean`, `majit` and `auto` all go
//!   through it and share ONE buffer, so the tiers stay comparable with each
//!   other; `stock` and cometkim's columns are untouched.
//! * **raw** — the same call through `collect_raw_on`, where a columnar consumer
//!   takes the machine's own buffers and no `Value` is built. Not comparable to
//!   `stock`, which necessarily produces one.
//! * **bind** — one `bind_per_row` of the same activation, timed on its own.
//!
//! ⚠️ The deviation that matters, stated plainly: his compiled function reads
//! every variable out of the `Context` BY NAME on every call (`rt_get_variable`
//! into a `BTreeMap`), and ours does not — a majit activation is resolved to
//! slots and encoded into columns once, at `bind`. Both hold the activation
//! fixed outside the timer, exactly as he does, but the work left inside it is
//! not the same work. `bind` is printed so a reader can put that cost back and
//! bound the advantage: it is the whole per-activation encoding, which is MORE
//! than name resolution, so `majit + bind` is a pessimistic upper bound on what
//! a fair single-shot majit call would cost.
//!
//! ⚠️ `variable_access/resolver` is his one case whose stock side reads through a
//! `VariableResolver` rather than the context map; that is reproduced here,
//! while the majit side reads the same value from a one-element column.
//!
//! The two activations — his hand-built context and the batch's columns — are
//! written out separately and then GATED against each other through the
//! library's own `RowReader`, so a column and the variable it stands for cannot
//! drift apart silently.
//!
//! RELEASE ONLY, and to match his build (`[profile.bench]`: lto, one codegen
//! unit) run it under the same profile:
//!
//! ```text
//! cargo run --profile bench --package cel --features jit-cranelift \
//!     --example majit_vs_cometkim_percall
//! ```
//!
//! To reproduce HIS two columns on the same machine, check out
//! `cel-rust/cel-rust` at the PR-233 head and run
//! `cargo bench -p cel-jit --bench comparison`; that measures upstream 0.11.6's
//! tree-walker and his Cranelift AOT backend, and nothing in this file is
//! derived from it.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::atomic::Ordering;
use std::time::Duration;

use cel::context::VariableResolver;
use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, RawOutput, RowReader, Tier};
use cel::majit::bytecode::float_bank::{
    jit_stats, reset_persistent_state, COMPILED_ENTRIES, COMPILES, GUARD_FAILS, TRACE_ABORTS,
};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

/// One timed batch must burn at least this much user CPU, so the clock's own
/// resolution is not what a 7 ns call is being measured against.
const MIN_BATCH: Duration = Duration::from_millis(20);
/// Timed batches per measurement. The MINIMUM is reported.
///
/// The metric is user CPU (see `per_call_counted`), which already refuses to
/// charge a batch for the time it spent descheduled — so the old reason for the
/// minimum, that a preempted batch reads slow, no longer applies. It survives
/// for a second one: a co-tenant still costs cycles this thread genuinely
/// executes. Cache lines it evicts we re-fetch, TLB entries it shoots down we
/// re-walk, pages it forces out we fault back in, and every one of those is our
/// own instruction stream and lands on our own clock. Contention is therefore
/// still one-sided — it can only add — and the fastest batch is the one that
/// ran with the least of it.
///
/// Thirty draws rather than seven because a minimum over a one-sided
/// distribution can only improve with more of them, and on a box carrying a
/// load average in the hundreds seven may contain no lightly-disturbed batch at
/// all. The cost is bounded and known: `ROUNDS * MIN_BATCH` of CPU per figure.
///
/// `check.py` reaches the opposite conclusion for its interpreter-startup
/// estimate — "The estimator is the MEDIAN and not the minimum" — and that
/// argument does not carry here. Its quantity is SUBTRACTED from a separately
/// measured bench, so an idle minimum taken away from a loaded run leaves the
/// run's own inflation behind in the difference, and median-against-median is
/// what cancels it. This harness subtracts nothing and reports the per-call
/// figure directly, so there is no second measurement whose load conditions
/// have to be matched, and the minimum is the estimator that is wanted.
const ROUNDS: usize = 30;

/// Which evaluator `Program::execute` — the `exec` column — reaches in THIS
/// build.
///
/// `vm` is a default feature, so ordinarily the answer is the bytecode VM. It
/// is not implied by `jit`, though, and `--no-default-features --features
/// jit-dynasm` builds this example with `execute` routed to the tree walker.
/// That build would print `exec ns` as a second `stock ns` and `exec/auto` as
/// a duplicate of `stock/auto`, which is the kind of silent degradation every
/// other gate in this file exists to refuse — so the answer is printed with the
/// legend instead of being inferred from the column's heading.
const EXEC_EVALUATOR: &str = if cfg!(feature = "vm") {
    "cel::vm::cel_eval_loop, the bytecode VM"
} else {
    "the tree walker: `vm` is OFF here, so `exec` is `stock` measured twice"
};

/// His `benchmark_variable_access` resolver, verbatim.
struct Resolver;

impl VariableResolver for Resolver {
    fn resolve(&self, expr: &str) -> Option<Value> {
        const V: Value = Value::Bool(false);
        const NOT_V: Value = Value::Bool(true);
        match expr {
            "fruit" => Some(NOT_V),
            "carrot" => Some(NOT_V),
            "orange" => Some(NOT_V),
            "banana" => Some(V),
            _ => None,
        }
    }
}

static RESOLVER: Resolver = Resolver;

/// One input column of the single activation, owned, in the layout the batch
/// reads: a [`ColumnRef`] over one row.
enum Col {
    Int(Vec<i64>),
    Bool(Vec<bool>),
    Str(Vec<String>),
    /// A list of `int`. One row, so `lens` is a single element count and `elems`
    /// is that row's elements.
    IntList {
        lens: Vec<i64>,
        elems: Vec<i64>,
    },
}

impl Col {
    /// The same one-row activation, `k` times.
    ///
    /// Every replicated row is BIT-IDENTICAL to the row the other columns
    /// measure, so a batch built from these asks the compiled tier exactly the
    /// question his `execute(&ctx)` asks — `k` times over — rather than a
    /// related one over synthetic data. That is what lets a per-row cost taken
    /// here be read against his per-call number at all.
    ///
    /// `IntList` repeats the length column and concatenates the elements; the
    /// bind derives each row's `offset(..)` from `lens` itself, so no offset
    /// column is built here.
    fn replicate(&self, k: usize) -> Col {
        match self {
            Col::Int(c) => Col::Int(c.repeat(k)),
            Col::Bool(c) => Col::Bool(c.repeat(k)),
            // `[T]::repeat` needs `T: Copy`, which `String` is not.
            Col::Str(c) => Col::Str(c.iter().cycle().take(c.len() * k).cloned().collect()),
            Col::IntList { lens, elems } => Col::IntList {
                lens: lens.repeat(k),
                elems: elems.repeat(k),
            },
        }
    }

    fn column_ref(&self) -> ColumnRef<'_> {
        match self {
            Col::Int(c) => ColumnRef::Int(c),
            Col::Bool(c) => ColumnRef::Bool(c),
            Col::Str(c) => ColumnRef::Str(c),
            Col::IntList { lens, elems } => ColumnRef::List {
                lens,
                fields: vec![(None, ColumnRef::Int(elems))],
            },
        }
    }
}

/// One expression from his suite, with his activation.
struct Case {
    label: String,
    src: String,
    /// Every path the expression reads. The lowering declines an undeclared
    /// path, so this is the case's input type declaration, not a convenience.
    schema: Vec<(String, ValType)>,
    cols: Vec<(String, Col)>,
    /// His activation: the variables he adds by name, and the functions he
    /// registers, on the ROOT context the timed call evaluates against. A
    /// registered function is an opaque Rust closure, so the lowering — which
    /// holds only the schema — declines the expression and the walker answers
    /// it; without this the walker could not either.
    stock: Box<dyn Fn(&mut Context<'static>)>,
    /// Read the variable through [`RESOLVER`] instead of the context map, which
    /// is what his `variable_access/resolver` measures.
    stock_resolver: bool,
    /// `(ladder, n)` for a size-ladder member, `None` otherwise.
    ///
    /// Carried as data rather than parsed back out of `label`: the decomposition
    /// below is the number the P5 gate is stated in, and deriving its input by
    /// splitting a display string would make a renamed case silently drop out of
    /// the fit instead of failing.
    ladder: Option<(&'static str, i64)>,
}

impl Case {
    fn new(label: &str, src: &str) -> Case {
        Case {
            label: label.to_string(),
            src: src.to_string(),
            schema: Vec::new(),
            cols: Vec::new(),
            stock: Box::new(|_| {}),
            stock_resolver: false,
            ladder: None,
        }
    }

    fn in_ladder(mut self, ladder: &'static str, n: i64) -> Case {
        self.ladder = Some((ladder, n));
        self
    }

    fn col(mut self, name: &str, ty: ValType, path_suffix: &str, col: Col) -> Case {
        self.schema.push((format!("{name}{path_suffix}"), ty));
        self.cols.push((name.to_string(), col));
        self
    }

    fn int(self, name: &str, v: i64) -> Case {
        self.col(name, ValType::Int, "", Col::Int(vec![v]))
    }

    fn bool(self, name: &str, v: bool) -> Case {
        self.col(name, ValType::Bool, "", Col::Bool(vec![v]))
    }

    fn text(self, name: &str, v: &str) -> Case {
        self.col(name, ValType::Str, "", Col::Str(vec![v.to_string()]))
    }

    fn int_list(self, name: &str, elems: Vec<i64>) -> Case {
        let lens = vec![elems.len() as i64];
        self.col(name, ValType::Int, "[]", Col::IntList { lens, elems })
    }

    fn stock(mut self, f: impl Fn(&mut Context<'static>) + 'static) -> Case {
        self.stock = Box::new(f);
        self
    }

    fn via_resolver(mut self) -> Case {
        self.stock_resolver = true;
        self
    }
}

/// A `HashMap` literal the way his benchmark writes one.
fn map_of(pairs: Vec<(&'static str, Value)>) -> Value {
    Value::from(pairs.into_iter().collect::<HashMap<&str, Value>>())
}

/// His 18 benchmark expressions with his contexts. Where he benchmarks a size
/// ladder the whole ladder is here, because that is where two evaluators
/// separate.
fn cases() -> Vec<Case> {
    let mut cases = vec![
        Case::new("simple_arithmetic", "1 + 2 * 3 - 4 / 2"),
        Case::new("comparison", "10 > 5 && 3 < 7 || 1 == 1"),
        Case::new("conditional", "x > 10 ? x * 2 : x + 5")
            .int("x", 15)
            .stock(|ctx| ctx.add_variable_from_value("x", 15i64)),
        Case::new(
            "nested_expression",
            "((a + b) * (c - d)) / ((e + f) - (g * h))",
        )
        .int("a", 10)
        .int("b", 20)
        .int("c", 30)
        .int("d", 5)
        .int("e", 15)
        .int("f", 25)
        .int("g", 2)
        .int("h", 3)
        .stock(|ctx| {
            for (name, v) in [
                ("a", 10i64),
                ("b", 20),
                ("c", 30),
                ("d", 5),
                ("e", 15),
                ("f", 25),
                ("g", 2),
                ("h", 3),
            ] {
                ctx.add_variable_from_value(name, v);
            }
        }),
        Case::new("variable_access/hashmap", "apple")
            .bool("apple", true)
            .stock(|ctx| ctx.add_variable_from_value("apple", true)),
        Case::new("variable_access/resolver", "banana")
            .bool("banana", false)
            .via_resolver(),
        Case::new("member_access", "obj.nested.value + obj.other")
            .int("obj.nested.value", 42)
            .int("obj.other", 10)
            .stock(|ctx| {
                let obj = map_of(vec![
                    ("nested", map_of(vec![("value", Value::Int(42))])),
                    ("other", Value::Int(10)),
                ]);
                ctx.add_variable_from_value("obj", obj);
            }),
        Case::new("list_indexing", "list[0] + list[5] + list[9]")
            .int_list("list", (1..=10).collect())
            .stock(|ctx| ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())),
        Case::new(
            "list_filter",
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(x, x > 5)",
        ),
        Case::new("list_map", "[1, 2, 3, 4, 5].map(x, x * 2)"),
        Case::new("all_comprehension", "[1, 2, 3, 4, 5].all(x, x > 0)"),
        Case::new("exists_comprehension", "[1, 2, 3, 4, 5].exists(x, x == 3)"),
    ];

    let ladder = |ladder: &'static str, n: i64, src: &str, name: &'static str, elems: Vec<i64>| {
        let stock = elems.clone();
        Case::new(&format!("{ladder}/{n}"), src)
            .int_list(name, elems)
            .stock(move |ctx| ctx.add_variable_from_value(name, stock.clone()))
            .in_ladder(ladder, n)
    };
    for size in [1i64, 10, 100, 1_000, 10_000] {
        cases.push(ladder(
            "map_list_scaling",
            size,
            "list.map(x, x * 2)",
            "list",
            (0..size).collect(),
        ));
    }
    for size in [1i64, 10, 100, 1_000, 10_000] {
        cases.push(ladder(
            "filter_list_scaling",
            size,
            "list.filter(x, x % 2 == 0)",
            "list",
            (0..size).collect(),
        ));
    }
    for size in [10i64, 50, 100, 500] {
        cases.push(ladder(
            "comprehension_scaling",
            size,
            "items.filter(x, x % 2 == 0).map(x, x * 2)",
            "items",
            (1..=size).collect(),
        ));
    }

    cases.push(Case::new(
        "string_operations",
        r#""hello world".startsWith("hello") && "hello world".endsWith("world") && "hello world".contains("o w")"#,
    ));
    cases.push(
        Case::new("custom_function", "add(x, y) + multiply(a, b)")
            .int("x", 10)
            .int("y", 20)
            .int("a", 5)
            .int("b", 3)
            .stock(|ctx| {
                for (name, v) in [("x", 10i64), ("y", 20), ("a", 5), ("b", 3)] {
                    ctx.add_variable_from_value(name, v);
                }
                ctx.add_function("add", |a: i64, b: i64| a + b);
                ctx.add_function("multiply", |a: i64, b: i64| a * b);
            }),
    );
    cases.push(
        Case::new(
            "real_world_policy",
            r#"user.age >= 18 &&
               user.role in ["admin", "moderator"] &&
               request.method == "POST" &&
               request.path.startsWith("/api/") &&
               size(request.body) < 1000000"#,
        )
        .int("user.age", 25)
        .text("user.role", "admin")
        .text("request.method", "POST")
        .text("request.path", "/api/users")
        .text("request.body", "{}")
        .stock(|ctx| {
            ctx.add_variable_from_value(
                "user",
                map_of(vec![
                    ("age", Value::Int(25)),
                    ("role", Value::from("admin")),
                ]),
            );
            ctx.add_variable_from_value(
                "request",
                map_of(vec![
                    ("method", Value::from("POST")),
                    ("path", Value::from("/api/users")),
                    ("body", Value::from("{}")),
                ]),
            );
        }),
    );
    cases
}

/// Time ONE call. Grows an iteration count until a timed batch costs at least
/// [`MIN_BATCH`] of user CPU, then reports the cheapest of [`ROUNDS`] such
/// batches.
fn per_call<T>(run: impl FnMut() -> T) -> f64 {
    per_call_counted(run).0
}

/// [`per_call`], also reporting how many times it invoked the closure —
/// calibration batches included.
///
/// The count is what turns an entry check from `> 0` into `>= calls`. A window
/// in which one call in ten thousand entered compiled code and the rest fell
/// back to the tracing interpreter passes the first and fails the second, and
/// the first is exactly the check a silently degraded column survives.
fn per_call_counted<T>(mut run: impl FnMut() -> T) -> (f64, usize) {
    /// User CPU burned by THIS thread so far.
    ///
    /// A wall clock does not measure this program on a shared box; it measures
    /// this program plus whatever else wanted the CPU. A batch that is
    /// descheduled for 300 ms reports 300 ms it never spent, and nothing in the
    /// figure distinguishes that from a call that genuinely got slower — which
    /// is the whole failure mode an A/B here has to survive. Charging only the
    /// cycles this thread actually ran makes a co-tenant's *preemption* cost the
    /// measurement nothing, and leaves only its cache and TLB damage, which
    /// [`ROUNDS`]'s minimum is there to shed.
    ///
    /// `CLOCK_THREAD_CPUTIME_ID` and not `getrusage(RUSAGE_SELF)`, which reports
    /// the same kind of quantity, on two grounds. It is per-THREAD: the batches
    /// run on one thread, and a process-wide clock would fold in any other
    /// thread's CPU, which is a property of what the example happens to spawn
    /// today rather than of what is being timed. And it is finer — measured on
    /// this machine, back-to-back reads advance by as little as 41 ns and never
    /// repeat a value, where `ru_utime` is a microsecond field that repeated on
    /// 916 of 1000 such reads. Neither resolution is a threat to a 20 ms batch;
    /// the point is that the narrower instrument costs nothing to prefer.
    fn cpu_now() -> Duration {
        // SAFETY: the call writes through the pointer and does nothing else,
        // and the pointer is to a live local of exactly the type it expects.
        let mut ts = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
        // Checked, because the two ways this can fail quietly are both worse
        // than a panic: a clock stuck at 0 leaves the calibration loop below
        // growing `iters` forever hunting a batch that never gets long enough,
        // and one that returns stale values publishes a per-call figure that
        // looks ordinary and is invented.
        assert_eq!(
            rc,
            0,
            "clock_gettime(CLOCK_THREAD_CPUTIME_ID): {}",
            std::io::Error::last_os_error()
        );
        Duration::new(ts.tv_sec as u64, ts.tv_nsec as u32)
    }

    fn timed<T>(iters: usize, run: &mut impl FnMut() -> T) -> Duration {
        let start = cpu_now();
        for _ in 0..iters {
            black_box(run());
        }
        // Thread CPU time never runs backwards, so this cannot underflow.
        cpu_now() - start
    }

    let mut calls = 0usize;
    let mut iters = 1usize;
    loop {
        let elapsed = timed(iters, &mut run);
        calls += iters;
        if elapsed >= MIN_BATCH {
            break;
        }
        // Aim straight at the target instead of doubling: a call that costs
        // milliseconds would otherwise spend most of the calibration finding
        // that out, and one that costs nanoseconds would spend 20 doublings.
        let want = MIN_BATCH.as_secs_f64() / elapsed.as_secs_f64().max(1e-9);
        let grow = (want.ceil() as usize).clamp(2, 1 << 12);
        iters = iters.saturating_mul(grow);
    }

    let best = (0..ROUNDS)
        .map(|_| {
            let d = timed(iters, &mut run);
            calls += iters;
            d.as_nanos() as f64 / iters as f64
        })
        .fold(f64::INFINITY, f64::min);
    (best, calls)
}

/// A columnar consumer. It reads the buffers the run wrote and never builds a
/// `Value`; the sum is only so the run cannot be optimized away.
fn consume_raw(out: RawOutput<'_>) -> i64 {
    let add = |a: i64, &b: &i64| a.wrapping_add(b);
    match out {
        RawOutput::Scalar { values, .. } => values.iter().fold(0, add),
        RawOutput::List { lens, fields, .. } => {
            let n: usize = lens.iter().map(|&c| c.max(0) as usize).sum();
            fields
                .iter()
                .map(|(_, _, buf)| buf[..n].iter().fold(0i64, add))
                .fold(0, i64::wrapping_add)
        }
    }
}

/// What the compiled tier did, or why it never ran.
struct Compiled {
    /// The plain Rust bytecode VM over the same lowered program, with no tracing
    /// machinery at all. It is the FLOOR the compiled tier has to beat, and the
    /// control that says whether a slow `majit` cell is the cost of the machine
    /// or the cost of the tracer failing to get out of the way.
    clean: f64,
    majit: f64,
    /// The DEFAULT route — `Tier::Auto`, which lets the bound batch pick
    /// between the two columns to its left. This is what a caller who names no
    /// tier gets, and the only column here that is about the library's own
    /// choice rather than about a tier.
    auto: f64,
    /// Which tier `Tier::Auto` resolved to, the body-word count of the run, and
    /// the saving the route actually decided on — `compiled_saving_ps` in
    /// nanoseconds, against an entry of `JIT_ENTRY_PS`. A `clean` route beside a
    /// `majit` cell slower than the `clean` one is the route working; the
    /// reverse would be the route mis-set.
    route: Tier,
    words: usize,
    saving: f64,
    raw: f64,
    bind: f64,
    compiles: usize,
    /// Traces started and thrown away, per call, once warm. A loop that never
    /// compiles is either aborting — counted here — or never reaching its merge
    /// point hot enough to be traced; only this tells the two apart.
    aborts: f64,
    guard_fails: f64,
    /// Bridges compiled per call over the same warm window as `aborts` and
    /// `guard_fails`.
    ///
    /// ⚠ Unlike those two this is NOT a callback tally. `jit_stats` builds
    /// `bridges_compiled` by summing the LIVE drivers' own counts and adding the
    /// ones already absorbed from retired drivers, so it is a population read at
    /// two instants and differenced. Nothing in the window retires a driver
    /// without absorbing its count, so the difference is the window's compiles.
    bridges: f64,
    /// Calls that ENTERED compiled code, per call, over the same warm window.
    ///
    /// This is what decides whether the `majit` cell beside it is the compiled
    /// tier or the tracing interpreter, and it is a fact rather than an
    /// inference: `compiles` says an artifact was minted, which is true of
    /// cases that never run a byte of it.
    ///
    /// A one-row call used to read `0.00` here whenever the row loop was the
    /// only loop: the loop is bottom-tested, so a one-row batch takes zero back
    /// edges and never reaches the instruction that consults the compiled loop.
    /// The function-entry door in `float_bank::try_function_entry_jit_f` counts
    /// CALLS instead of rows, so such a case can now enter — which is what makes
    /// gating the `majit` cell on this statistic worth doing rather than merely
    /// correct.
    entries: f64,
    /// Whether the settled window entered compiled code on EVERY one of its
    /// calls, from the integer counter delta rather than from `entries` above.
    ///
    /// The gate in front of the `majit ns` cell. It is measured over the settled
    /// window and not over the timed one, which is the strongest evidence this
    /// harness collects: the timed loop is `per_call`, which cannot afford a
    /// counter read per iteration. A window that enters on all 1 000 settled
    /// calls and then stops entering inside the timed loop would still publish a
    /// number here — for the per-call-evidenced version of this measurement see
    /// `majit_percall_steady`.
    entered_every_settled_call: bool,
    /// The compiled tier's marginal cost of one more activation, from a
    /// replicated-batch slope. `None` when the two probe batches did not both
    /// run in compiled code, which is a refusal to print a number rather than a
    /// failure of the case.
    jit_row: Option<f64>,
    /// `t(K_LO) - K_LO * jit_row`: everything a call pays that is not
    /// per-activation — the pool lookup, the state republish, the entry and
    /// exit, and the one row that always runs interpreted before the first back
    /// edge. It is therefore an UPPER bound on call overhead, not the overhead.
    jit_fix: Option<f64>,
    /// The SAME slope on the tier that cannot compile: the traced portal with
    /// its trace threshold at `u32::MAX`. This is the per-row cost BEFORE the
    /// JIT has produced compiled code; `jit_row` beside it is the same row AFTER
    /// compiling, so the pair is what the compiled tier actually earned.
    ///
    /// It used to be a second thing as well — what a one-row call pays for ever,
    /// since such a call takes no back edge and so never reached the compiled
    /// loop. The function-entry door counts CALLS rather than rows, so that no
    /// longer follows: `entries` says per case whether the one-row calls beside
    /// this column entered compiled code, and where it says they did, this
    /// column is the pre-compile cost only.
    interp_row: Option<f64>,
    interp_fix: Option<f64>,
    /// The same slope with no tracing machinery in the picture at all. It is
    /// the floor under both of the above; `interp_row - clean_row` is what
    /// arming the JIT costs a row that never gets compiled code out of it.
    clean_row: Option<f64>,
    /// `clean_fix` is the one fixed cost with NO driver behind it —
    /// `Tier::Clean` dispatches straight to the plain interpreter and never
    /// touches the pooled-driver path. So `jit_fix - clean_fix` is that path's
    /// per-call cost, measured rather than itemised.
    clean_fix: Option<f64>,
}

struct Row {
    label: String,
    /// `(ladder, n)`, copied from the case so the decomposition below has its
    /// input as data.
    ladder: Option<(&'static str, i64)>,
    stock: f64,
    /// One whole `Program::execute` on the same fixed activation: the door a
    /// consumer of this library writes, and the only column here that is about
    /// what such a consumer gets rather than about a tier this file selects.
    ///
    /// It lives on `Row` and not on `Compiled` because it is answerable whether
    /// or not the expression lowers — `Program::execute` has no batch tier
    /// behind it — so a declined case still carries this cell.
    exec: f64,
    /// `Err` when the expression does not lower: the tree-walker answers it —
    /// through the library's own fallback, not a hand-written one — and there is
    /// no compiled tier to put beside it.
    compiled: Result<Compiled, String>,
}

fn run_case(case: &Case) -> Row {
    let program = Program::compile(&case.src)
        .unwrap_or_else(|e| panic!("{}: parse error: {e:?}", case.label));
    let schema: Schema = case.schema.iter().cloned().collect();

    let mut batch = Batch::new(1);
    for (name, col) in &case.cols {
        batch = batch.column(name.clone(), col.column_ref());
    }

    // His activation, built his way and held fixed outside the timer: a root
    // context with the variables added by name.
    let mut activation = Context::default();
    (case.stock)(&mut activation);
    if case.stock_resolver {
        activation.set_variable_resolver(&RESOLVER);
    }
    let expected = Value::resolve_value(program.expression(), &activation)
        .unwrap_or_else(|e| panic!("{}: stock execute: {e:?}", case.label));

    // Drift gate: the same walker, over a child scope the LIBRARY's `RowReader`
    // filled from the very columns the compiled tier reads, must reach the same
    // answer. Without it a column and the variable it is supposed to stand for
    // can disagree and every ratio below silently compares two workloads.
    let mirrored = RowReader::new(&batch).scope(&activation, 0);
    assert_eq!(
        Value::resolve_value(program.expression(), &mirrored)
            .ok()
            .as_ref(),
        Some(&expected),
        "{}: the batch columns and the hand-built activation disagree",
        case.label
    );

    // Nothing timed in this file may DISCARD a `Result`. A refused evaluation
    // returns early, so a swallowed error is not a slow number, it is a fast
    // one — which is exactly what cometkim's own benchmark reports: his
    // `b.iter(|| black_box(compiled.execute(&ctx)))` never unwraps, and his
    // backend answers `items.filter(..).map(..)` with
    // `UndeclaredReference("@result")`, so all four `comprehension_scaling`
    // rows time a failure and print it as a two-fold speedup.
    let stock = per_call(|| {
        Value::resolve_value(program.expression(), black_box(&activation))
            .unwrap_or_else(|e| panic!("{}: stock execute: {e:?}", case.label))
    });

    // The second evaluator answers the same thing. `expected` is the walker's
    // answer and `exec` below times `Program::execute`, which under the default
    // `vm` feature is a DIFFERENT evaluator over the same activation; a column
    // timing a different answer would be timing a different workload.
    assert_eq!(
        program.execute(&activation).ok().as_ref(),
        Some(&expected),
        "{}: Program::execute and the tree walker disagree",
        case.label
    );

    // The default consumer's door, in the same regime as every other column:
    // one expression, one fixed activation held outside the timer, one
    // evaluation timed. Nothing about it depends on the batch machine, so it is
    // measured before the lowering can decline.
    let exec = per_call(|| {
        program
            .execute(black_box(&activation))
            .unwrap_or_else(|e| panic!("{}: execute: {e:?}", case.label))
    });

    let lowered = match BatchProgram::from_program(&program, &schema) {
        Ok(bp) => bp,
        Err(e) => {
            return Row {
                label: case.label.clone(),
                ladder: case.ladder,
                stock,
                exec,
                compiled: Err(format!("declines: {e}")),
            }
        }
    };
    // A fresh driver, so `compiles` counts this case's loops and not a loop an
    // earlier case left compiled at the same program address.
    reset_persistent_state();
    let bound = match lowered.bind_per_row(&batch) {
        Ok(b) => b,
        Err(e) => {
            return Row {
                label: case.label.clone(),
                ladder: case.ladder,
                stock,
                exec,
                compiled: Err(format!("cannot bind: {e}")),
            }
        }
    };

    // Miscompile gate: all three tiers, and the tree-walker, agree on the one
    // row. `collect` is the per-row door, so this compares the VALUE his
    // `execute` returns, not a batch reduction of it.
    //
    // The buffer door is gated here beside it because it is the one the table
    // TIMES: a gate that checked only `collect_on` would leave the measured
    // door unchecked, and a measured door that produced nothing would read as
    // a very fast one.
    COMPILES.store(0, Ordering::Relaxed);
    let mut gate = Vec::new();
    for tier in [Tier::Clean, Tier::Interpreter, Tier::Jit] {
        let got = bound
            .collect_on(tier)
            .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label));
        assert_eq!(
            got.as_slice(),
            std::slice::from_ref(&expected),
            "{}: {tier:?} vs stock",
            case.label
        );
        bound
            .collect_into_on(tier, &mut gate)
            .unwrap_or_else(|e| panic!("{}: {tier:?} into buffer: {e}", case.label));
        assert_eq!(
            gate, got,
            "{}: {tier:?} buffer door vs collect_on",
            case.label
        );
    }
    // One row per call means the batch loop crosses its header once per call, so
    // the trace threshold is reached across CALLS. Warm until it is, before
    // anything is timed: otherwise the `majit` column of a case that never
    // compiled would be the tracing interpreter's number under the compiled
    // tier's heading.
    warm(&bound);
    let compiles = COMPILES.load(Ordering::Relaxed);
    // What the driver is still doing per call once it is as warm as it will get.
    const SETTLED: usize = 1_000;
    let (a0, g0, b0, e0) = (
        TRACE_ABORTS.load(Ordering::Relaxed),
        GUARD_FAILS.load(Ordering::Relaxed),
        jit_stats().bridges_compiled,
        COMPILED_ENTRIES.load(Ordering::Relaxed),
    );
    for _ in 0..SETTLED {
        black_box(bound.collect_on(Tier::Jit).expect("settled run"));
    }
    let aborts = (TRACE_ABORTS.load(Ordering::Relaxed) - a0) as f64 / SETTLED as f64;
    let guard_fails = (GUARD_FAILS.load(Ordering::Relaxed) - g0) as f64 / SETTLED as f64;
    // The fact the `majit` column's heading has always asserted and never
    // checked. `compiles` above cannot answer it: it counts artifacts minted,
    // and a loop that is minted and never entered leaves every other counter
    // here plausible while the interpreter produces the answers.
    let entries_delta = COMPILED_ENTRIES.load(Ordering::Relaxed) - e0;
    let entries = entries_delta as f64 / SETTLED as f64;
    let entered_every_settled_call = entries_delta >= SETTLED;
    // `saturating_sub` where the two above subtract plainly, because the two
    // above are monotonic counters and this one is a population: only
    // `reset_persistent_state` can drop a live driver's count without absorbing
    // it, and it is not called inside this window, but a negative difference is
    // a wrong number where a clamped zero is a visibly uninformative one.
    let bridges = jit_stats().bridges_compiled.saturating_sub(b0) as f64 / SETTLED as f64;

    // ONE output buffer for all three of our timed columns, reused call after
    // call. See this file's header on why that is the fair door: the buffer is
    // the caller's, and evaluating a row is what is being compared.
    let mut out: Vec<Value> = Vec::new();
    let collect = |tier, out: &mut Vec<Value>| {
        bound
            .collect_into_on(tier, out)
            .unwrap_or_else(|e| panic!("{}: {tier:?}: {e}", case.label));
        // The rows are the result, and nothing downstream reads them, so they
        // are held against elimination here rather than by a return value:
        // the door hands them back through the buffer.
        black_box(out.as_slice());
    };
    let clean = per_call(|| collect(Tier::Clean, &mut out));
    let majit = per_call(|| collect(Tier::Jit, &mut out));
    let auto = per_call(|| collect(Tier::Auto, &mut out));
    let raw = per_call(|| {
        bound
            .collect_raw_on(Tier::Jit, consume_raw)
            .unwrap_or_else(|e| panic!("{}: raw: {e}", case.label))
    });
    let bind = per_call(|| {
        lowered
            .bind_per_row(&batch)
            .unwrap_or_else(|e| panic!("{}: rebind: {e}", case.label))
    });

    // Three legs of one measurement, so they are taken together rather than
    // wherever each is first needed. Order is free — see `row_cost` on why the
    // three tiers cannot contaminate one another — so it runs floor-first,
    // which is also the order the table reads in.
    let (clean_row, clean_fix) = row_cost(case, &lowered, Tier::Clean);
    let (interp_row, interp_fix) = row_cost(case, &lowered, Tier::Interpreter);
    let (jit_row, jit_fix) = row_cost(case, &lowered, Tier::Jit);

    Row {
        label: case.label.clone(),
        ladder: case.ladder,
        stock,
        exec,
        compiled: Ok(Compiled {
            clean,
            auto,
            route: bound.route(Tier::Auto),
            words: bound.body_words(),
            saving: bound.compiled_saving_ps() as f64 / 1000.0,
            majit,
            raw,
            bind,
            compiles,
            aborts,
            guard_fails,
            bridges,
            entries,
            entered_every_settled_call,
            jit_row,
            jit_fix,
            interp_row,
            interp_fix,
            clean_row,
            clean_fix,
        }),
    }
}

/// `tier`'s marginal cost of ONE more activation, and the fixed cost of the
/// call that carries it.
///
/// Run on all three tiers this is the pre-compile / post-compile split: the
/// per-row cost BEFORE the JIT has given the loop compiled code
/// (`Tier::Interpreter`) and AFTER (`Tier::Jit`), in one unit, with
/// `Tier::Clean` — no tracing machinery at all — as the floor under both.
/// Without that split the compiled tier had no number of its own, because the
/// head-to-head column and the compiled tier could be two different machines
/// under one heading.
///
/// Why a slope and not a timing of one activation: the row loop is
/// bottom-tested, so an `n`-row batch takes `n - 1` back edges and a ONE-row
/// batch takes none, and until the function-entry door existed nothing a
/// one-row call executed ever consulted the compiled loop — no amount of warming
/// made such a call run compiled code. That is no longer unconditional: the door
/// in `float_bank::try_function_entry_jit_f` counts CALLS, and
/// `repeated_one_row_calls_reach_the_compiled_tier` in
/// `tests/majit_trace_evidence.rs` is a cold one-row workload that reaches
/// compiled code through it. What still holds is the boundary the door draws —
/// it declines for a program whose own loop is already compiled, pinned by
/// `a_compiled_row_loop_shuts_the_entry_door` — so a one-row call after a big
/// batch is still the pre-compile tier. The slope keeps the two apart without
/// depending on which of those a case is: `enter/call` reports the entry as a
/// fact, and this function's own gates require it over every probe call.
///
/// So: replicate the SAME activation `k` times, bit for bit, and difference two
/// batch sizes. What survives the subtraction is the per-activation cost of
/// `tier`; what it removes is everything a call pays once.
///
/// The three legs do not contaminate each other and do not have to be ordered.
/// The driver pool is keyed on `(regs, fregs, THRESHOLD)`, so the
/// `u32::MAX`-threshold driver `Tier::Interpreter` runs on is a different pool
/// entry from `Tier::Jit`'s and never sees a loop `Tier::Jit` compiled;
/// `Tier::Clean` takes no driver at all. The entry gate below checks that
/// rather than assuming it.
///
/// ⚠ The result is NOT the same unit as cometkim's `compiled` column. His
/// number is one whole `CompiledProgram::execute(&ctx)` including all per-call
/// overhead. `jit/row` differences that away deliberately, which is why
/// `jit fix` is returned beside it — but their SUM models a call this engine
/// cannot currently make, and must be read as a model.
///
/// Returns `(None, None)` rather than a number whenever the evidence for
/// "this ran on `tier` and on nothing else" is not complete.
fn row_cost(case: &Case, lowered: &BatchProgram, tier: Tier) -> (Option<f64>, Option<f64>) {
    // Keep the replicated work bounded: a ladder member already carries
    // thousands of elements per row, and replicating THAT a thousand times
    // would measure the machine's memory system rather than its loop.
    let per_row_elems = case
        .cols
        .iter()
        .map(|(_, c)| match c {
            Col::IntList { elems, .. } => elems.len().max(1),
            _ => 1,
        })
        .max()
        .unwrap_or(1);
    const ELEM_BUDGET: usize = 1 << 18;
    let k_hi = (ELEM_BUDGET / per_row_elems).clamp(32, 1024);
    let k_lo = (k_hi / 8).max(2);

    let build = |k: usize| -> (Vec<(String, Col)>, usize) {
        (
            case.cols
                .iter()
                .map(|(n, c)| (n.clone(), c.replicate(k)))
                .collect(),
            k,
        )
    };
    let (cols_lo, _) = build(k_lo);
    let (cols_hi, _) = build(k_hi);
    let mut batch_lo = Batch::new(k_lo);
    for (name, col) in &cols_lo {
        batch_lo = batch_lo.column(name.clone(), col.column_ref());
    }
    let mut batch_hi = Batch::new(k_hi);
    for (name, col) in &cols_hi {
        batch_hi = batch_hi.column(name.clone(), col.column_ref());
    }
    let (blo, bhi) = match (
        lowered.bind_per_row(&batch_lo),
        lowered.bind_per_row(&batch_hi),
    ) {
        (Ok(lo), Ok(hi)) => (lo, hi),
        _ => return (None, None),
    };

    let run_hi = || {
        bhi.collect_raw_on(tier, consume_raw)
            .unwrap_or_else(|e| panic!("{}: {tier:?} slope hi: {e}", case.label))
    };
    let run_lo = || {
        blo.collect_raw_on(tier, consume_raw)
            .unwrap_or_else(|e| panic!("{}: {tier:?} slope lo: {e}", case.label))
    };

    // `k_hi - 1` back edges in the first call alone, so the threshold is crossed
    // long before decay or bucket eviction could reach the counter.
    for _ in 0..32 {
        black_box(run_hi());
        black_box(run_lo());
    }

    // `Tier::Jit` is the only tier whose number is ABOUT compiled code, so it is
    // the only one required to enter it. For the other two, entering is the
    // failure: a `Tier::Interpreter` slope that ran compiled code is a
    // `Tier::Jit` slope with the wrong heading, which is the very confusion this
    // split exists to end. Both directions are checked against the same counter.
    let must_enter = matches!(tier, Tier::Jit);
    let entry_ok = |seen: usize, calls: usize| {
        if must_enter {
            seen >= calls
        } else {
            seen == 0
        }
    };

    // GATE 1 — entry, over a FIXED-count window. `per_call_counted` calibrates
    // by invoking its closure an unbounded number of times, so a probe that ran
    // through it could not state its own denominator.
    const PROBE: usize = 200;
    let e0 = COMPILED_ENTRIES.load(Ordering::Relaxed);
    for _ in 0..PROBE {
        black_box(run_hi());
    }
    if !entry_ok(COMPILED_ENTRIES.load(Ordering::Relaxed) - e0, PROBE) {
        return (None, None);
    }

    let (c0, ab0, en0) = (
        COMPILES.load(Ordering::Relaxed),
        TRACE_ABORTS.load(Ordering::Relaxed),
        COMPILED_ENTRIES.load(Ordering::Relaxed),
    );
    let (t_hi, calls_hi) = per_call_counted(run_hi);
    let (t_lo, calls_lo) = per_call_counted(run_lo);

    // GATE 2 — every timed call entered, not merely one of them. Under
    // `must_enter == false` this is the opposite claim about the same counter:
    // not one of them did.
    let entered = COMPILED_ENTRIES.load(Ordering::Relaxed) - en0;
    // GATE 3 — steady state: the slope must not be measuring trace/compile
    // churn, and the bigger batch must actually cost more.
    let churned =
        COMPILES.load(Ordering::Relaxed) != c0 || TRACE_ABORTS.load(Ordering::Relaxed) != ab0;
    if !entry_ok(entered, calls_hi + calls_lo) || churned || !(t_hi > t_lo) {
        return (None, None);
    }

    let slope = (t_hi - t_lo) / (k_hi - k_lo) as f64;
    if !(slope > 0.0) {
        return (None, None);
    }
    (Some(slope), Some(t_lo - k_lo as f64 * slope))
}

fn warm(bound: &BoundBatch<'_, '_>) {
    for _ in 0..256 {
        black_box(bound.collect_on(Tier::Jit).expect("warm run"));
    }
}

/// A cost model `fixed + per_elem * n`.
struct Fit {
    fixed: f64,
    per_elem: f64,
}

impl Fit {
    /// Through `(n_lo, t_lo)` and `(n_hi, t_hi)`.
    ///
    /// Two points, not a least-squares line over all of them, because that is
    /// what task #88's table was computed with and reproducing its METHOD is the
    /// point — a different estimator would make the two figures incomparable for
    /// a reason that has nothing to do with the machine. The fit therefore
    /// passes through its endpoints by construction, so it is not evidence of
    /// linearity; `worst_mid_err` below is what tests that.
    fn two_point(lo: (f64, f64), hi: (f64, f64)) -> Fit {
        let per_elem = (hi.1 - lo.1) / (hi.0 - lo.0);
        Fit {
            fixed: lo.1 - lo.0 * per_elem,
            per_elem,
        }
    }

    /// The same model over EVERY point, by least squares.
    ///
    /// It does not replace [`Fit::two_point`] and is not offered as the better
    /// number. It answers the one question the two-point fit structurally
    /// cannot: that fit passes through its endpoints by construction, so its
    /// `fixed` is an extrapolation to `n = 0` with no residual of its own, and
    /// the only thing that can disagree with it is a point it did not touch. A
    /// least-squares line touches nothing, so every point is a residual, and
    /// `fixed` is answerable at ladders with no interior point to spare.
    ///
    /// Reported as a second estimate, never as an interval: at four or five
    /// points of one sample each there is nothing to put a confidence interval
    /// on, and printing one would claim a precision this harness cannot reach.
    fn least_squares(pts: &[(f64, f64)]) -> Fit {
        let n = pts.len() as f64;
        let mean_x = pts.iter().map(|&(x, _)| x).sum::<f64>() / n;
        let mean_y = pts.iter().map(|&(_, y)| y).sum::<f64>() / n;
        let sxx = pts
            .iter()
            .map(|&(x, _)| (x - mean_x) * (x - mean_x))
            .sum::<f64>();
        let sxy = pts
            .iter()
            .map(|&(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
        let per_elem = sxy / sxx;
        Fit {
            fixed: mean_y - per_elem * mean_x,
            per_elem,
        }
    }

    fn at(&self, n: f64) -> f64 {
        self.fixed + self.per_elem * n
    }
}

/// One ladder's decomposition.
struct Decomposition {
    ladder: &'static str,
    /// The two `n` the TWO-POINT fit was taken through, and how many compiled
    /// points the ladder had in total. The least-squares fit uses all of them.
    n_lo: i64,
    n_hi: i64,
    points: usize,
    /// Ladder members whose loop never compiled, and which are therefore NOT in
    /// the fit: their `majit` cell is the tracing interpreter, so including one
    /// would fit a different machine.
    excluded: usize,
    /// The RAW `majit ns` at `n_lo` — nothing fitted, the measurement itself.
    ///
    /// Printed beside `majit fixed` because it is what that intercept sits on:
    /// `fixed = t_lo - per_elem * n_lo`, so the correction is `t_lo - fixed` and
    /// a reader can size it without being told. Where it is a few percent the
    /// intercept is a measured point lightly adjusted, and its precision is the
    /// measurement's rather than the model's.
    ///
    /// ⚠ `n_lo` is the lowest rung that COMPILED, not the ladder's lowest rung —
    /// `map_list_scaling` and `filter_list_scaling` both start at n=1, and a run
    /// where that rung never compiles anchors the intercept at n=10 instead.
    /// That is why the abscissa is printed next to the ordinate: the anchor can
    /// move between runs, and it moves silently otherwise.
    t_lo: f64,
    majit: Fit,
    clean: Fit,
    /// The `majit` model again, fitted by least squares over every included
    /// point. A SECOND estimate of `majit fixed`, printed beside the first
    /// rather than in place of it — see [`Fit::least_squares`] for why both.
    majit_ls: Fit,
    /// Worst relative error of the TWO-POINT majit fit at a point it did NOT
    /// pass through, or `None` when the ladder has only the two endpoints.
    worst_mid_err: Option<f64>,
    /// Worst relative residual of [`Decomposition::majit_ls`] over EVERY
    /// included point, since that fit passes through none of them.
    ///
    /// `None` below three points, where least squares is just the line through
    /// the two and its residual is zero for a reason that says nothing about
    /// the model. Same rule as `worst_mid_err`, reached from the other side.
    worst_ls_err: Option<f64>,
    /// The largest `gfails/call` over the compiled members. Printed beside the
    /// fit because a fixed cost and a per-call guard failure are the same
    /// finding read two ways, and #88's own tripwire is that a compile count
    /// must never be reported without it.
    max_gfails: f64,
    /// The largest `bridges/call` over the compiled members.
    ///
    /// It belongs beside `majit fixed` for the same reason `max_gfails` does.
    /// `majit fixed` is a per-call cost, and the compiled artifacts the driver
    /// deals with per call are the most direct decomposition of it: a fixed cost
    /// that scales with the artifact count is an artifact-count cost, one that
    /// does not is a per-artifact cost, and those two want opposite repairs.
    ///
    /// ⚠ Read what this counts. It is artifacts COMPILED in the warm window, not
    /// artifacts ENTERED — a settled trace tree enters its bridges on every call
    /// and compiles none of them, so a zero here says the population stopped
    /// GROWING, not that the call enters nothing. The nonzero reading is the
    /// strong one: a warm call that is still compiling has compilation itself
    /// inside the fixed cost, which no per-entry repair can reach.
    max_bridges: f64,
}

/// Task #88's decomposition over the size ladders, computed here rather than by
/// hand off the table above, by two estimators: #88's own two-point fit and a
/// least-squares fit over every included point.
///
/// The gate this epic is under — "a compiled cel artifact's fixed per-call cost
/// under ~1 µs on both backends" — is stated in the `majit fixed` column, and
/// that column has until now been arithmetic somebody did in a notebook. A
/// number that decides a phase should be produced by the program that measures
/// it.
fn decompose(rows: &[Row]) -> Vec<Decomposition> {
    let mut ladders: Vec<&'static str> = Vec::new();
    for r in rows {
        if let Some((name, _)) = r.ladder {
            if !ladders.contains(&name) {
                ladders.push(name);
            }
        }
    }

    let mut out = Vec::new();
    for name in ladders {
        let mut pts: Vec<(f64, &Compiled)> = Vec::new();
        let mut excluded = 0usize;
        for r in rows {
            let Some((ladder, n)) = r.ladder else {
                continue;
            };
            if ladder != name {
                continue;
            }
            match &r.compiled {
                Ok(c) if c.compiles > 0 => pts.push((n as f64, c)),
                _ => excluded += 1,
            }
        }
        if pts.len() < 2 {
            continue;
        }
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (lo, hi) = (&pts[0], &pts[pts.len() - 1]);

        let majit = Fit::two_point((lo.0, lo.1.majit), (hi.0, hi.1.majit));
        let clean = Fit::two_point((lo.0, lo.1.clean), (hi.0, hi.1.clean));

        let worst_mid_err = pts[1..pts.len() - 1]
            .iter()
            .map(|(n, c)| ((majit.at(*n) - c.majit) / c.majit).abs())
            .fold(None::<f64>, |acc, e| Some(acc.map_or(e, |a: f64| a.max(e))));

        let majit_pts: Vec<(f64, f64)> = pts.iter().map(|(n, c)| (*n, c.majit)).collect();
        let majit_ls = Fit::least_squares(&majit_pts);
        // Every point, where `worst_mid_err` takes the interior ones: this fit
        // passes through none of them, so there is no endpoint to skip.
        let worst_ls_err = (majit_pts.len() > 2).then(|| {
            majit_pts
                .iter()
                .map(|&(n, t)| ((majit_ls.at(n) - t) / t).abs())
                .fold(0.0, f64::max)
        });

        out.push(Decomposition {
            ladder: name,
            n_lo: lo.0 as i64,
            n_hi: hi.0 as i64,
            points: pts.len(),
            excluded,
            max_gfails: pts.iter().map(|(_, c)| c.guard_fails).fold(0.0, f64::max),
            max_bridges: pts.iter().map(|(_, c)| c.bridges).fold(0.0, f64::max),
            t_lo: lo.1.majit,
            majit,
            clean,
            majit_ls,
            worst_mid_err,
            worst_ls_err,
        });
    }
    out
}

/// `n` where the compiled tier's cost model crosses the clean VM's.
///
/// `None` when the compiled tier is not cheaper per element, in which case it
/// never catches up and there is no crossing to report — a state the table has
/// to be able to print, because #88 measured a NEGATIVE per-element cost once
/// (`cranelift/map`, −0.32 ns) and a sign flip there is a real outcome.
fn break_even(d: &Decomposition) -> Option<f64> {
    let gain = d.clean.per_elem - d.majit.per_elem;
    (gain > 0.0).then(|| (d.majit.fixed - d.clean.fixed) / gain)
}

/// One worst-relative-error cell, or `-` where the ladder cannot supply one.
///
/// Shared by `mid err` and `ls err` so the two are formatted identically: they
/// are only readable against each other if they are printed the same way.
fn err_cell(e: Option<f64>) -> String {
    match e {
        Some(e) => format!("{:>8.1}%", e * 100.0),
        None => format!("{:>9}", "-"),
    }
}

fn print_decomposition(rows: &[Row]) {
    let table = decompose(rows);
    if table.is_empty() {
        return;
    }
    println!(
        "\ntask #88's decomposition, computed here rather than by hand. `majit fixed`\n\
         is the column the STOP-AT-P5 gate is stated in: its re-entry criterion is a\n\
         compiled artifact's fixed per-call cost under ~1 us. It is reported by TWO\n\
         estimators, in the second block, because one of them cannot disagree with\n\
         its own inputs and so cannot report an error against them."
    );
    // Two blocks rather than one 165-column line. The split is by SUBJECT, not
    // by what happened to fit: the per-element slopes and the clean-VM control
    // are one reading, and `majit fixed` under two estimators is the other. No
    // column was dropped or narrowed to make them fit.
    println!("\nthe per-element cost, and the clean VM it is measured against:");
    println!(
        "\n{:<22} {:>7} {:>13} {:>11} {:>11} {:>12} {:>11} {:>11}",
        "ladder",
        "points",
        "fit through n",
        "majit/elem",
        "ls/elem",
        "clean fixed",
        "clean/elem",
        "break-even"
    );
    for d in &table {
        let be = match break_even(d) {
            Some(n) => format!("{n:>11.0}"),
            // The compiled tier is not cheaper per element on this ladder, so it
            // never overtakes. Printed rather than left blank.
            None => format!("{:>11}", "never"),
        };
        println!(
            "{:<22} {:>7} {:>6}..{:<6} {:>11.3} {:>11.3} {:>12.1} {:>11.3} {be}",
            d.ladder,
            d.points,
            d.n_lo,
            d.n_hi,
            d.majit.per_elem,
            d.majit_ls.per_elem,
            d.clean.fixed,
            d.clean.per_elem,
        );
    }
    println!("\n`majit fixed`, the number the gate is stated in, by both estimators:");
    // `n_lo` and `t(n_lo)` lead the block rather than sitting at its end: they
    // are the point the intercept is anchored on, so they are read BEFORE it.
    println!(
        "\n{:<22} {:>7} {:>6} {:>10} {:>12} {:>9} {:>12} {:>9} {:>12} {:>13}",
        "ladder",
        "points",
        "n_lo",
        "t(n_lo)",
        "majit fixed",
        "mid err",
        "ls fixed",
        "ls err",
        "gfails/call",
        "bridges/call"
    );
    for d in &table {
        println!(
            "{:<22} {:>7} {:>6} {:>10.1} {:>12.1} {} {:>12.1} {} {:>12.2} {:>13.2}",
            d.ladder,
            d.points,
            d.n_lo,
            d.t_lo,
            d.majit.fixed,
            err_cell(d.worst_mid_err),
            d.majit_ls.fixed,
            err_cell(d.worst_ls_err),
            d.max_gfails,
            d.max_bridges,
        );
    }
    let excluded: usize = table.iter().map(|d| d.excluded).sum();
    println!(
        "\nRead it with six cautions.\n\
         * The two-point fit passes through its two endpoints BY CONSTRUCTION, so\n\
           it cannot disagree with them: its `majit fixed` is an extrapolation to\n\
           n=0 carrying no residual of its own. `mid err` is the whole test of that\n\
           model — the worst relative error at a ladder point the fit did not\n\
           touch — and a `-` means the ladder had no such point and the row is\n\
           entirely unchecked.\n\
         * `ls fixed` is the same model fitted by LEAST SQUARES over every included\n\
           point. It is the second opinion, not the better number, and it does not\n\
           replace the two-point figure: reproducing #88's METHOD is what makes a\n\
           number here comparable to one taken then. Unlike the first it touches no\n\
           point, so every point is a residual — `ls err` is the worst of them. A\n\
           `-` means fewer than three points, where least squares is just the line\n\
           through the two and its zero residual would say nothing. No interval is\n\
           printed for either: four or five points at one sample each cannot carry\n\
           one, and printing one would claim a precision this harness has not got.\n\
         * `majit fixed` is `t(n_lo)` MINUS the slope's contribution at n_lo, and\n\
           both are printed so that correction can be sized rather than assumed.\n\
           Where it is a few percent of `t(n_lo)` the intercept inherits the\n\
           precision of a DIRECTLY MEASURED point, not the model's: `mid err` and\n\
           `ls err` describe the model BETWEEN the endpoints, so they bound the\n\
           per-element term and any extrapolation past the ladder's reach, and a\n\
           large one does not make the intercept uncertain by the same fraction.\n\
           The reading that does put the intercept in doubt is the opposite one —\n\
           `per_elem * n_lo` a LARGE fraction of `t(n_lo)`, where `majit fixed` is\n\
           mostly the subtraction of a modelled quantity from a measured one.\n\
         * {excluded} ladder member(s) are excluded because their loop never\n\
           compiled. Their `majit` cell is the tracing interpreter, and fitting it\n\
           would decompose a different machine.\n\
         * `gfails/call` belongs beside `majit fixed`, not in a separate table:\n\
           #88 found 1.00 guard failure per call on every compiled case and named\n\
           it the prime suspect for the fixed cost it measured. A fixed cost read\n\
           without it is half a finding.\n\
         * `bridges/call` counts artifacts COMPILED in the warm window, not\n\
           artifacts entered. A zero says the artifact population stopped\n\
           growing, NOT that the call enters no bridge; a nonzero one says a warm\n\
           call is still compiling, which puts compilation inside `majit fixed`."
    );
}

/// The pre-compile / post-compile split, printed on its own.
///
/// Separate from the table above because it is a DIFFERENT UNIT: every column
/// there is one whole call, every column here is one activation with the call's
/// fixed cost differenced away. Putting a per-call and a per-row number side by
/// side under adjacent headings is how the two came to be read as one machine.
///
/// A `-` is a refusal, not a zero: `row_cost` returns nothing unless its entry
/// counter agreed with the tier it was asked for, in both directions.
fn print_row_cost_table(rows: &[Row]) {
    println!(
        "\nper-row cost BY TIER — one replicated-batch slope, run on three tiers.\n\
         `interp` is the traced portal with its trace threshold at `u32::MAX`, so it can\n\
         never compile: that column is the per-row cost BEFORE compiling and `jit` is the\n\
         same row AFTER. `clean` is the plain VM, the floor under both."
    );
    println!(
        "{:<28} {:>10} {:>11} {:>9} {:>12} {:>10} {:>11} {:>9} {:>9}",
        "case",
        "clean/row",
        "interp/row",
        "jit/row",
        "jit earns",
        "clean fix",
        "interp fix",
        "jit fix",
        "drv fix",
    );
    let opt = |v: Option<f64>| match v {
        Some(x) => format!("{x:.1}"),
        None => "-".to_string(),
    };
    for r in rows {
        let c = match &r.compiled {
            Ok(c) => c,
            Err(_) => continue,
        };
        let earns = match (c.interp_row, c.jit_row) {
            (Some(i), Some(j)) if j > 0.0 => format!("{:.2}x", i / j),
            _ => "-".to_string(),
        };
        let drv = match (c.jit_fix, c.clean_fix) {
            (Some(j), Some(cl)) => format!("{:.1}", j - cl),
            _ => "-".to_string(),
        };
        println!(
            "{:<28} {:>10} {:>11} {:>9} {:>12} {:>10} {:>11} {:>9} {:>9}",
            r.label,
            opt(c.clean_row),
            opt(c.interp_row),
            opt(c.jit_row),
            earns,
            opt(c.clean_fix),
            opt(c.interp_fix),
            opt(c.jit_fix),
            drv,
        );
    }
    println!(
        "\n`jit earns` is `interp/row / jit/row` — what compiling bought for ONE activation,\n\
         with the call's fixed cost differenced out of both sides. It is the only ratio in\n\
         this file that compares two of OUR tiers on one unit, and so the only one that\n\
         says whether the compiler is doing its job independently of how the call is\n\
         entered.\n\
         \n\
         `drv fix` is `jit fix - clean fix`. `Tier::Clean` dispatches straight to the plain\n\
         interpreter and takes no driver at all, so this difference is the pooled-driver\n\
         path's own per-call cost — the pool lookup, the program-table insert, the state\n\
         republish, the per-call state buffers — measured rather than itemised.\n\
         \n\
         ⚠ `interp fix` and `jit fix` both still CONTAIN one interpreted row: the batch\n\
         loop is bottom-tested, so row 0 runs before the first back edge on every tier.\n\
         That row's cost scales with the expression, which is why these intercepts differ\n\
         per case while `drv fix` is the part that does not.\n\
         \n\
         ⚠ `jit earns` is not a claim about a single call. It compares two per-ROW slopes,\n\
         and a call in cometkim's regime carries one row: whether that call is on the\n\
         `interp/row` side or the `jit/row` side is what `enter/call` reports per case, and\n\
         it is not the same answer for every case any more. A case whose `enter/call` is 0\n\
         pays `interp/row`; one that enters through the function-entry door does not."
    );
}

fn main() {
    println!("cometkim's benchmark expressions in his own regime (cel-jit PR #233)");
    println!(
        "one expression, one FIXED activation, ONE evaluation timed; \
         best of {ROUNDS} batches of >= {} CPU-ms.\n\
         Every ns figure below is USER CPU on the measuring thread, NOT wall clock:\n\
         time spent descheduled by other work on the box is not charged to it.\n\
         \n\
         THREE evaluators are timed, and the table names each of them: `stock` is the\n\
         tree walker called DIRECTLY, `clean`/`majit`/`auto`/`raw` are the batch machine\n\
         through a `BoundBatch`, and `exec` is `Program::execute` — the door a caller who\n\
         names nothing goes through. In THIS build `execute` reaches\n\
         {EXEC_EVALUATOR}.\n",
        MIN_BATCH.as_millis()
    );
    println!(
        "{:<28} {:>11} {:>11} {:>11} {:>11} {:>10} {:>7} {:>9} {:>9} {:>11} {:>11} {:>11} {:>11} {:>11} {:>12} {:>10} {:>10} {:>9} {:>12} {:>12} {:>13}",
        "case",
        "stock ns",
        "exec ns",
        "clean ns",
        "majit ns",
        "auto ns",
        "route",
        "words",
        "save ns",
        "stock/auto",
        "exec/auto",
        "enter/call",
        "jit/row ns",
        "jit fix ns",
        "stock/majit",
        "raw ns",
        "bind ns",
        "compiles",
        "aborts/call",
        "gfails/call",
        "bridges/call"
    );

    let cases = cases();
    let mut declined = Vec::new();
    let mut lowered = 0usize;
    let mut never_compiled = Vec::new();
    // Retained, not just printed: the decomposition below needs every ladder
    // member's cells at once, and re-running a case to get them back would time
    // a second, differently-warmed process state.
    let mut rows = Vec::with_capacity(cases.len());
    for case in &cases {
        let r = run_case(case);
        match &r.compiled {
            Ok(c) => {
                lowered += 1;
                if c.compiles == 0 {
                    never_compiled.push(r.label.clone());
                }
                let opt = |v: Option<f64>| match v {
                    Some(x) => format!("{x:.1}"),
                    None => "-".to_string(),
                };
                // The timed `majit` number is published only where the settled
                // window entered compiled code on every call. Where it did not,
                // the number is the tracing interpreter's under the compiled
                // tier's heading, and so is the ratio derived from it — both are
                // annotated rather than printed, and `enter/call` beside them
                // says how far short the evidence fell.
                let (majit_cell, ratio_cell) = if c.entered_every_settled_call {
                    (
                        format!("{:.1}", c.majit),
                        format!("{:.2}x", r.stock / c.majit),
                    )
                } else {
                    ("not entered".to_string(), "-".to_string())
                };
                println!(
                    "{:<28} {:>11.1} {:>11.1} {:>11.1} {:>11} {:>10.1} {:>7} {:>9} {:>9.0} {:>11} {:>11} {:>11.2} {:>11} {:>11} {:>12} {:>10.1} {:>10.1} {:>9} {:>12.2} {:>12.2} {:>13.2}",
                    r.label,
                    r.stock,
                    r.exec,
                    c.clean,
                    majit_cell,
                    c.auto,
                    match c.route {
                        Tier::Clean => "clean",
                        Tier::Jit => "jit",
                        other => panic!("auto resolved to {other:?}"),
                    },
                    c.words,
                    c.saving,
                    format!("{:.2}x", r.stock / c.auto),
                    format!("{:.2}x", r.exec / c.auto),
                    c.entries,
                    opt(c.jit_row),
                    opt(c.jit_fix),
                    ratio_cell,
                    c.raw,
                    c.bind,
                    c.compiles,
                    c.aborts,
                    c.guard_fails,
                    c.bridges
                );
            }
            Err(why) => {
                // Still measured and still ANSWERED — by the tree-walker,
                // through the library's fallback. A row missing from the table
                // would read as an expression this crate cannot evaluate.
                println!(
                    "{:<28} {:>11.1} {:>11.1} {:>11} {:>11} {:>10} {:>7} {:>9} {:>9} {:>11} {:>11} {:>11} {:>11} {:>11} {:>12} {:>10} {:>10} {:>9} {:>12} {:>12} {:>13}",
                    r.label,
                    r.stock,
                    r.exec,
                    "-",
                    "walker",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-"
                );
                declined.push((r.label.clone(), why.clone()));
            }
        }
        rows.push(r);
    }

    println!(
        "\ncoverage: {lowered}/{} lower to the compiled tier; {}/{} are answered",
        cases.len(),
        cases.len(),
        cases.len()
    );
    for (label, why) in &declined {
        println!("  {label:<28} {why}");
    }
    if !never_compiled.is_empty() {
        println!(
            "\n⚠️ the batch loop never compiled for {} of {lowered} lowered cases: {}",
            never_compiled.len(),
            never_compiled.join(", ")
        );
        println!(
            "   their `majit ns` is the TRACING INTERPRETER under the compiled tier's heading,\n\
                and is printed as `not entered` for exactly that reason. `aborts/call` says\n\
                which of two reasons applies: a nonzero count is a loop the tracer keeps trying\n\
                and throwing away, a zero one is a loop that never gets hot — at one row per\n\
                call the batch loop has no back edge to be hot on, and the function-entry door\n\
                is the only counter that can warm such a case."
        );
    }
    print_row_cost_table(&rows);
    println!(
        "\n`clean ns` is the same lowered bytecode on the plain Rust VM, with no tracing\n\
         machinery at all. Where `majit` is far above it the cost is the tracer, not the\n\
         machine, and the compiled tier is not what answered the call."
    );
    println!(
        "\n`auto ns` is the DEFAULT door, `collect()`, which names no tier: the bound batch\n\
         picks between `clean` and `majit` by `save ns` — how much the compiled tier is\n\
         ESTIMATED to save on this run: rows times a per-row rate plus elements times a\n\
         per-element one, each rate set by that loop's word count less a fixed per-unit\n\
         cost a word count cannot see. Below `JIT_ENTRY_PS` the run does not save what\n\
         reaching compiled code costs, so it stays on the plain VM. `words` is the same\n\
         run's size on either tier, printed beside it because the two were once one\n\
         decision. `route` says which side of that each case\n\
         landed, and `auto ns` should track whichever of the two columns to its left the\n\
         route named. `majit ns` beside it is still the compiled tier ASKED FOR outright,\n\
         which is what every tier-explicit test and every column of this table below\n\
         measures — the route changes the default, not the `_on` doors.\n\
         \n\
         ⚠ `stock/auto` measures our route against the TREE WALKER, `stock/majit` a\n\
         caller who names `Tier::Jit`, and `exec/auto` our route against\n\
         `Program::execute`. All three are spelled as the division actually performed, so\n\
         each cell checks against the two ns columns it comes from: ABOVE 1.00x we beat\n\
         the evaluator the numerator names, BELOW it that evaluator beat us.\n\
         \n\
         The walker is NOT what a default caller gets, and `exec ns` is the column that\n\
         says so in numbers. `vm` is a default feature, so in this build `Program::execute`\n\
         reaches {EXEC_EVALUATOR},\n\
         while the head of this file says why the walker is nonetheless the right baseline\n\
         for comparing against HIS published figures. So read the two families apart:\n\
         every `stock/..` ratio is against the walker and against nothing else, and\n\
         `exec/auto` is the one that answers whether we beat the evaluator a default\n\
         consumer actually runs.\n\
         \n\
         ⚠ `exec ns` is one whole `Program::execute` and the batch columns beside it are\n\
         not the same setup. `execute` builds its VM state — operand stack, locals, logic\n\
         slots — on every call and looks every variable up in the `Context` BY NAME; the\n\
         batch columns hoist both out, the activation having been resolved to slots and\n\
         encoded into columns at `bind` (timed on its own as `bind ns`) and the run state\n\
         belonging to the `BoundBatch` and reused call after call. `exec/auto` is\n\
         therefore a DOOR-to-door ratio — what a caller gains by moving to the batch API,\n\
         setup and all — and not two evaluators compared on equal footing.\n\
         \n\
         Where the two disagree the route is\n\
         doing something: at one row a straight-line expression has a body of tens of\n\
         words, and no amount of compiling it pays back the entry."
    );
    println!(
        "\n`majit ns` holds the activation fixed exactly as his `execute(&ctx)` does, but a\n\
         majit activation was resolved to slots and encoded into columns at `bind`, while his\n\
         compiled code looks every variable up in the context BY NAME on every call. `bind ns`\n\
         is that encoding timed on its own — it is MORE than name resolution, so adding it back\n\
         is a pessimistic bound on the difference, not an estimate of it."
    );
    println!(
        "\n`enter/call` is calls that ENTERED compiled code, counted at the point the\n\
         compiled body is about to run — not artifacts minted. It now GATES the `majit ns`\n\
         cell beside it rather than merely standing next to it: a case that did not enter on\n\
         every call of the settled window prints `not entered` there and no `stock/majit`\n\
         ratio, because the number would be the tracing interpreter's under the compiled\n\
         tier's heading. A `0.00` with `compiles` at 1 is a loop that was compiled and never\n\
         run. Two ways in can produce a positive number: the row BODY contains a loop that\n\
         gets hot on its own, or the function-entry door — which counts CALLS, not rows —\n\
         warmed on the repeated one-row calls this file makes. The door declines for a\n\
         program whose own loop is already compiled, so which of the two applies is a\n\
         per-case fact.\n\
         \n\
         ⚠ The window this is measured over is the 1 000-call settled window, not the timed\n\
         loop: the timed loop cannot afford a counter read per call. A case that entered on\n\
         all 1 000 settled calls and then stopped would still publish a number. For the\n\
         per-call-evidenced form of the same measurement see `majit_percall_steady`."
    );
    println!(
        "\n`jit/row ns` is the compiled tier's MARGINAL cost of one more activation,\n\
         measured by replicating this case's single activation bit-for-bit into two batch\n\
         sizes and differencing them. Three counter gates stand in front of it and it\n\
         prints `-` if any fails: every call in a fixed 200-call probe entered compiled\n\
         code, every TIMED call entered it, and no compile or trace-abort happened inside\n\
         the timed windows. A number that merely looks fast does not get printed."
    );
    println!(
        "\n⚠ `jit/row ns` is NOT the unit cometkim's `compiled` column is in. His is one\n\
         whole `execute(&ctx)` including every per-call cost; this differences those away\n\
         on purpose. `jit fix ns` is published so they can be added back — but their SUM\n\
         models a call this engine cannot currently make at one activation, and quoting\n\
         `jit/row` against his number without it flatters this side.\n\
         ⚠ A NEGATIVE `jit fix ns` means the two-point model is refuted for that row: cost\n\
         is superlinear in the replication count there, so extrapolating to zero rows\n\
         undershoots. Its `jit/row` is still a measured difference quotient between the\n\
         two sizes, but it is not a constant marginal cost, and neither cell should be\n\
         read as a per-activation figure.\n\
         ⚠ Every replicated row is identical, so the compiled loop gets perfect branch\n\
         prediction and a hot cache. His regime has the same property — he re-evaluates\n\
         one fixed activation — so the comparison is symmetric, but neither side's number\n\
         is what varying data would cost."
    );
    print_decomposition(&rows);
}
