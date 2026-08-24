//! CEL's own regime: compile the expression ONCE, then evaluate it per record,
//! one activation per call, with DIFFERENT bindings every call.
//!
//! ⚠️ What the clock covers is an ALREADY-ENCODED ACTIVATION, not a deployment
//! end to end. The timed loop calls pre-built `BoundBatch` values: the pool's
//! rows are materialized into column storage, wrapped in `Batch`es and bound
//! once, before the clock starts. A deployment handed a fresh record per call
//! pays all three per call, and none of the three is inside `steady ns`.
//! `bind ns` is the last of them, timed separately over already-built `Batch`
//! values, so it prices rebinding and not record materialization or batch
//! construction. `steady + bind` is therefore a bound on the ENCODED-activation
//! path, not on a real per-record cost — the record's own encoding is measured
//! nowhere here. Read every number below as "what one call costs once its
//! arguments are already in columnar form".
//!
//! That is what a policy engine does — K8s admission, Envoy authz, IAM
//! conditions — and it is not the shape any benchmark in this crate measured
//! until now. `majit_vs_cometkim_percall` times one FIXED activation, which is
//! cometkim's unit and the right one for reading a number beside his; the batch
//! examples time thousands of rows per call. Between them sits the workload the
//! library is actually deployed in, and its two costs are the ones a deployment
//! pays: what one call costs once the tier is warm, and how many calls it takes
//! to get there.
//!
//! Neither cost could be measured here before the function-entry door existed.
//! The row loop's `can_enter_jit!` is a back edge on a bottom-tested loop, so an
//! `n`-row batch takes `n - 1` of them and a ONE-row batch takes none: warmup
//! counted there counts ROWS, and a program evaluated a million times at one row
//! per call never warmed at all. The door in `float_bank::try_function_entry_jit_f`
//! counts CALLS instead, and `repeated_one_row_calls_reach_the_compiled_tier` in
//! `tests/majit_trace_evidence.rs` is the pinned witness that a cold one-row
//! workload now reaches compiled code through it.
//!
//! The five phases, per case:
//!
//! * **prepare** — parse and lower, timed. A one-time cost, paid before any
//!   record is seen, and reported so it can be put back into a deployment's
//!   arithmetic rather than assumed negligible.
//! * **warm** — repeated one-row calls until the tier is entering compiled code
//!   on every call and nothing is still compiling. Reported as a call COUNT and
//!   a wall time, and as an explicit `>cap` marker where steady state was never
//!   reached. A run that cannot warm prints no per-call number at all.
//! * **steady** — timed one-row calls on `Tier::Jit`, cycling a pool of distinct
//!   input rows.
//! * **reference** — the same loop on `Tier::Clean`, the plain Rust bytecode VM
//!   with no tracing machinery anywhere in it. That is what one call costs
//!   without a JIT, and it is what the steady-state number has to beat.
//! * **derived** — speedup, and the break-even record count at which the warmup
//!   has paid for itself.
//!
//! ⚠️ The bindings VARY, and that is load-bearing rather than decorative. Every
//! other benchmark in this crate re-evaluates one activation, so a compiled
//! artifact that had baked the first call's values in as constants would answer
//! every later call correctly by accident and read as very fast. Here call `i`
//! runs input row `i % pool`, the pool holds distinct rows, and every answer is
//! checked against the plain interpreter — so an artifact that specialised on
//! its first call answers wrong and the case fails loudly. The `inputs` and
//! `answers` columns say how much variation the pool actually achieved: a case
//! with no columns to vary reads `1`/`1` there, and its oracle check, while
//! still true, cannot detect constant-baking at all.
//!
//! ⚠️ `bind` is OUTSIDE the timed call, and it is kept as its own column rather
//! than folded in. The pool's rows are encoded into columns once, up front, and
//! the steady-state loop only runs them. A deployment receiving a fresh record
//! per call pays that encoding per call, so `bind ns` is measured and printed
//! separately and the tables below never add the two for the reader. What their
//! sum bounds is stated at the top of this header: the encoded-activation path,
//! since `bind` starts from a `Batch` that already exists.
//!
//! RELEASE ONLY, under the `[profile.bench]` settings the other benchmarks use:
//!
//! ```text
//! cargo run --profile bench --package cel --features jit-cranelift \
//!     --example majit_percall_steady
//! ```
//!
//! Counts are settable so a smoke run can prove the gates fire without spending
//! a measurement's worth of time:
//! `-- --steady 50 --rounds 2`. Every count the run used is echoed in the header
//! and in the machine-readable block, because a table whose sample size is not
//! stated cannot be diffed against another run of the same table.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Instant;

use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, RowReader, Tier};
use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::lower::{Schema, ValType};
use cel::{Context, Program, Value};

/// How the run was parameterised. Echoed in full, because every number below is
/// a number OF something and a table that does not say what cannot be compared
/// with the same table from another run.
struct Config {
    /// Distinct input rows the steady-state loop cycles through.
    pool: usize,
    /// Timed one-row calls per round.
    steady: usize,
    /// Rounds per measurement. The FASTEST is reported: other work on the box
    /// can only make a round slower, so the fastest one ran with the least
    /// interference.
    rounds: usize,
    /// Consecutive calls that must each enter compiled code, with nothing
    /// compiling or aborting between them, before the tier is called warm.
    window: usize,
    /// Warm-up calls after which a case is declared never to have reached steady
    /// state. Far above the entry door's own threshold, so a case that stops
    /// here stopped for a reason other than not being given enough calls.
    cap: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            pool: 16,
            steady: 5_000,
            rounds: 7,
            window: 64,
            cap: 4_096,
        }
    }
}

impl Config {
    /// `--pool N --steady N --rounds N --window N --cap N`, in any order.
    ///
    /// An unrecognised flag is fatal rather than ignored: a misspelled `--steady`
    /// that silently kept the default would report the default's sample size in
    /// the header and be indistinguishable from a run that meant it.
    fn from_args() -> Config {
        let mut cfg = Config::default();
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut at = 0usize;
        while at < args.len() {
            let flag = &args[at];
            // Which count the flag names, decided before the value is looked at,
            // so an unrecognised flag is reported as one rather than as a
            // missing or unparseable count.
            let count = match flag.as_str() {
                "--pool" => &mut cfg.pool,
                "--steady" => &mut cfg.steady,
                "--rounds" => &mut cfg.rounds,
                "--window" => &mut cfg.window,
                "--cap" => &mut cfg.cap,
                other => panic!("unknown flag `{other}`"),
            };
            let raw = args
                .get(at + 1)
                .unwrap_or_else(|| panic!("{flag} wants a count after it"));
            *count = raw
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("{flag} {raw}: {e}"));
            at += 2;
        }
        assert!(cfg.pool >= 1, "the input pool needs at least one row");
        assert!(cfg.steady >= 1 && cfg.rounds >= 1, "nothing would be timed");
        cfg
    }
}

/// Samples the one-time prepare cost is taken over. The FASTEST is kept, for the
/// same reason the timed rounds keep theirs.
const PREPARE_SAMPLES: usize = 5;

/// One input column of one activation, owned, in the layout the batch reads.
///
/// `PartialEq` is derived so the pool can count how many DISTINCT rows it
/// actually built. That count is the whole strength of the anti-constant-folding
/// check below, and a harness that asserted variation instead of counting it
/// would keep asserting it after a case lost its last varying column.
#[derive(PartialEq)]
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
    /// Variant `v` of this column: the same shape and the same length, different
    /// values.
    ///
    /// Length is deliberately held fixed. `list_indexing` reads `list[9]`, so a
    /// pool that varied list lengths would turn a benchmark into an out-of-range
    /// error on some variants and measure the error path; and the trip count of a
    /// comprehension is data the compiled loop reads, not something baked into
    /// the words, so varying values already varies everything the artifact could
    /// have specialised on.
    ///
    /// The integer offset sweeps through zero and changes SIGN across the pool
    /// rather than merely growing. `conditional` is `x > 10 ? x * 2 : x + 5` at
    /// x=15: a pool that only added would keep every variant on one arm of the
    /// branch, which is the specialisation this pool exists to break.
    fn vary(&self, v: usize) -> Col {
        let delta = 3 * v as i64 - 24;
        let flip = v % 2 == 1;
        match self {
            Col::Int(c) => Col::Int(c.iter().map(|x| x.wrapping_add(delta)).collect()),
            Col::Bool(c) => Col::Bool(c.iter().map(|x| x ^ flip).collect()),
            Col::Str(c) => Col::Str(c.iter().map(|s| vary_str(s, v)).collect()),
            Col::IntList { lens, elems } => Col::IntList {
                lens: lens.clone(),
                elems: elems.iter().map(|x| x.wrapping_add(delta)).collect(),
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

/// Variant `v` of a string, varying LENGTH and CONTENT both.
///
/// Both matter and for different reasons: content is what a `==` or a
/// `startsWith` guard would have specialised on, and length is what `size(..)`
/// reads. One variant is the original, so the pool still contains the row the
/// corpus was written around.
fn vary_str(s: &str, v: usize) -> String {
    match v % 4 {
        0 => s.to_string(),
        1 => s.to_uppercase(),
        2 => format!("{s}-{v}"),
        _ => format!("{}{s}", v % 10),
    }
}

/// One expression with its activation, from `majit_vs_cometkim_percall`'s corpus.
struct Case {
    label: String,
    src: String,
    /// Every path the expression reads. The lowering declines an undeclared
    /// path, so this is the case's input type declaration, not a convenience.
    schema: Vec<(String, ValType)>,
    cols: Vec<(String, Col)>,
    /// The variables and functions a tree-walker needs on its ROOT context. The
    /// walker is used here only as a SECOND oracle beside `Tier::Clean`, over a
    /// scope the library's own `RowReader` fills from the very columns the
    /// compiled tier reads — so a column and the variable it stands for cannot
    /// drift apart silently. A registered function is an opaque Rust closure the
    /// lowering cannot see, so a case that has one declines and never reaches
    /// the walker at all.
    stock: Box<dyn Fn(&mut Context<'static>)>,
}

impl Case {
    fn new(label: &str, src: &str) -> Case {
        Case {
            label: label.to_string(),
            src: src.to_string(),
            schema: Vec::new(),
            cols: Vec::new(),
            stock: Box::new(|_| {}),
        }
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
}

/// A `HashMap` literal the way the corpus writes one.
fn map_of(pairs: Vec<(&'static str, Value)>) -> Value {
    Value::from(pairs.into_iter().collect::<HashMap<&str, Value>>())
}

/// The corpus, expression for expression as `majit_vs_cometkim_percall` builds
/// it, so a case here and a case there are the same question asked in two
/// regimes.
///
/// Two things that file carries are dropped, both because they have no meaning
/// in this regime rather than to make anything easier. `variable_access/resolver`
/// keeps its expression but loses the `VariableResolver` route: that route is a
/// property of the tree-walker's variable lookup, and nothing on the batch side
/// resolves a name per call at all. The size-ladder membership is dropped
/// because the ladders here are not fitted — every rung is one case, evaluated
/// at one row.
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
        Case::new("variable_access/resolver", "banana").bool("banana", false),
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

/// The four counters a steady-state window is judged by, read at one instant.
///
/// `bridges` is unlike the other three: `jit_stats` builds it by summing the LIVE
/// drivers' own counts plus those already absorbed from retired ones, so it is a
/// population read at two instants and differenced, not a monotonic tally.
/// Nothing in a window here retires a driver, so the difference is still the
/// window's compiles, but the subtraction below saturates for it and does not
/// for the others.
#[derive(Clone, Copy)]
struct Counters {
    compiles: usize,
    aborts: usize,
    entries: usize,
    guard_fails: usize,
    bridges: usize,
}

fn counters() -> Counters {
    Counters {
        compiles: jit_stats().loops_compiled,
        aborts: jit_stats().loops_aborted,
        entries: jit_stats().compiled_entries,
        guard_fails: jit_stats().guard_failures,
        bridges: jit_stats().bridges_compiled,
    }
}

impl Counters {
    /// Whether NOTHING compiled and no trace aborted between `earlier` and
    /// `self` — the condition "steady state" names.
    ///
    /// All three of root loops, BRIDGES and aborts, because a window with a
    /// bridge compiled inside it is a window with compilation latency in it,
    /// exactly as a root-loop compile would be. Only root loops and aborts were
    /// checked before, so a case that guard-failed and bridged its way through
    /// the timed loop certified as steady and published a number that included
    /// the backend's work.
    ///
    /// `bridges` is compared for EQUALITY rather than for growth, unlike the two
    /// tallies beside it: it is a population read twice (see [`Counters`]), so a
    /// decrease is not evidence that nothing compiled, and refusing the window
    /// is the direction that cannot publish a compile as steady state.
    fn settled_since(&self, earlier: &Counters) -> bool {
        self.compiles == earlier.compiles
            && self.aborts == earlier.aborts
            && self.bridges == earlier.bridges
    }
}

/// What the warm-up phase found out.
struct Warm {
    /// Calls made. When `reached` is false this is the cap, and the number is a
    /// floor on the answer rather than the answer.
    calls: usize,
    /// Wall time of every one of those calls.
    ///
    /// ⚠ It is an UPPER bound on the warm-up's own cost: the loop reads five
    /// counters and takes a branch after each call, and that is inside the clock.
    /// The break-even figure derived from it is conservative in the same
    /// direction.
    nanos: f64,
    /// The call on which compiled code was first entered, 1-based.
    ///
    /// This is what says WHICH of the two doors a case came in through, and it
    /// says it as data rather than as a claim about the expression's shape. The
    /// entry door counts calls, so a case whose only loop is the row loop first
    /// enters at around the trace threshold — call 8 or 9. A case whose row BODY
    /// contains its own loop over a list longer than the threshold crosses that
    /// loop's back edge enough times inside the FIRST call, so it enters at call
    /// 1 or 2 — and the entry door then declines for it permanently, by design:
    /// a program whose own loop is compiled already has a way in, and a second
    /// door in front of it takes one away rather than adding one.
    first_entry: Option<usize>,
    reached: bool,
}

/// Repeated one-row calls until the tier enters compiled code on `cfg.window`
/// consecutive calls with nothing compiling or aborting between them — root
/// loops and bridges alike, per [`Counters::settled_since`].
///
/// The window is what makes this steady state rather than first contact. A
/// single entry says an artifact ran once; it is compatible with a case that
/// enters, deopts, retraces and compiles again forever, which would put
/// compilation itself inside every number measured afterwards.
fn warm_to_steady(pool: &[BoundBatch<'_, '_>], cfg: &Config, label: &str) -> Warm {
    let mut prev = counters();
    let mut streak = 0usize;
    let mut first_entry = None;
    let start = Instant::now();
    for call in 0..cfg.cap {
        black_box(
            pool[call % pool.len()]
                .collect_on(Tier::Jit)
                .unwrap_or_else(|e| panic!("{label}: warm call {call}: {e}")),
        );
        let now = counters();
        let entered = now.entries > prev.entries;
        if entered && first_entry.is_none() {
            first_entry = Some(call + 1);
        }
        let settled = now.settled_since(&prev);
        streak = if entered && settled { streak + 1 } else { 0 };
        prev = now;
        if streak >= cfg.window {
            return Warm {
                calls: call + 1,
                nanos: start.elapsed().as_nanos() as f64,
                first_entry,
                reached: true,
            };
        }
    }
    Warm {
        calls: cfg.cap,
        nanos: start.elapsed().as_nanos() as f64,
        first_entry,
        reached: false,
    }
}

/// `cfg.rounds` rounds of `cfg.steady` calls each, in ns/call, and the total
/// number of calls made.
///
/// The fastest round is reported and the count is returned rather than assumed:
/// the count is what turns the entry check below from `> 0` into `>= calls`, and
/// a window in which one call in ten thousand entered compiled code and the rest
/// fell back to the interpreter passes the first and fails the second.
///
/// `call` receives a monotonically increasing index, not an index modulo the
/// pool, so the phase of the input cycle carries across rounds instead of
/// restarting on the same row every time.
fn steady_ns(cfg: &Config, mut call: impl FnMut(usize)) -> (f64, usize) {
    let mut best = f64::INFINITY;
    let mut cursor = 0usize;
    for _ in 0..cfg.rounds {
        let start = Instant::now();
        for _ in 0..cfg.steady {
            call(cursor);
            cursor += 1;
        }
        best = best.min(start.elapsed().as_nanos() as f64 / cfg.steady as f64);
    }
    (best, cursor)
}

/// A case that produced numbers.
struct Measured {
    warm: Warm,
    /// Steady-state ns/call on the compiled tier, or `None` when the evidence
    /// that this window WAS the compiled tier is incomplete. A `-` in the table
    /// is that refusal, never a zero and never a plausible-looking guess.
    jit: Option<f64>,
    /// The same loop on `Tier::Clean`. `None` when that loop entered compiled
    /// code, which would mean the two columns are one tier under two headings.
    clean: Option<f64>,
    /// One `bind_per_row` of a pool row, timed on its own. NOT part of `jit`:
    /// see the module header on why it is excluded and what including it would
    /// bound.
    bind: f64,
    /// Parse and lower, best of [`PREPARE_SAMPLES`].
    prepare: f64,
    entries_per_call: f64,
    /// Smallest and largest compiled-entry delta over any ONE call of the
    /// untimed oracle replay, which makes the same `steady` calls in the same
    /// order as the timed loop.
    ///
    /// This is the per-call evidence behind "every call enters"; `entries_per_call`
    /// beside it is a mean and cannot distinguish a window where every call
    /// entered once from one where half entered twice and half not at all. A
    /// minimum of 0 refuses the `steady ns` cell. A flat expression — one whose
    /// row body holds no loop of its own — reads `1..1`; a maximum above 1 is a
    /// row body entering an inner loop's artifact more than once per call.
    ///
    /// `None` when the run made no replay calls.
    replay_entries: Option<(usize, usize)>,
    gfails_per_call: f64,
    aborts_per_call: f64,
    bridges_per_call: f64,
    compiles: usize,
    /// Distinct input rows the pool actually holds, and distinct answers they
    /// produced. Both are `1` for an expression with no columns, and that pair
    /// is the honest statement that its oracle check cannot detect an artifact
    /// which baked its first call's values in.
    inputs: usize,
    answers: usize,
    timed_calls: usize,
}

struct Row {
    label: String,
    /// `Err` when there is nothing to time: the lowering declined the
    /// expression, the columns would not bind, or a pool row had no answer on
    /// the interpreter tier. Carries the reason, which is printed rather than
    /// summarised.
    outcome: Result<Measured, String>,
}

fn run_case(case: &Case, cfg: &Config) -> Row {
    let schema: Schema = case.schema.iter().cloned().collect();

    // PHASE 0 — the one-time cost, before any record is seen.
    let mut prepare = f64::INFINITY;
    let mut prepared = None;
    for _ in 0..PREPARE_SAMPLES {
        let start = Instant::now();
        let parsed = Program::compile(&case.src)
            .unwrap_or_else(|e| panic!("{}: parse error: {e:?}", case.label));
        let lowered = BatchProgram::from_program(&parsed, &schema);
        prepare = prepare.min(start.elapsed().as_nanos() as f64);
        prepared = Some((parsed, lowered));
    }
    let (program, lowered) = prepared.expect("PREPARE_SAMPLES is at least one");
    let refuse = |why: String| Row {
        label: case.label.clone(),
        outcome: Err(why),
    };
    let lowered = match lowered {
        Ok(l) => l,
        Err(e) => return refuse(format!("declines: {e}")),
    };

    // The pool of distinct input rows, encoded once. Three layers, each
    // borrowing the one before it: owned column data, the batches that view it,
    // the bound programs that read those.
    let pool: Vec<Vec<(String, Col)>> = (0..cfg.pool)
        .map(|v| {
            case.cols
                .iter()
                .map(|(name, col)| (name.clone(), col.vary(v)))
                .collect()
        })
        .collect();
    let inputs = distinct_count(&pool);
    let batches: Vec<Batch<'_>> = pool
        .iter()
        .map(|cols| {
            cols.iter().fold(Batch::new(1), |batch, (name, col)| {
                batch.column(name.clone(), col.column_ref())
            })
        })
        .collect();
    let bounds: Vec<BoundBatch<'_, '_>> = match batches
        .iter()
        .map(|b| lowered.bind_per_row(b))
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(b) => b,
        Err(e) => return refuse(format!("cannot bind: {e}")),
    };

    // The pool's answers, from the tier with no tracing machinery in it, plus a
    // SECOND independent oracle: the tree-walker over a scope the library's own
    // `RowReader` fills from the very columns the compiled tier reads. Without
    // that second reading a column and the variable it stands for can disagree
    // and every ratio below silently compares two different workloads.
    let mut activation = Context::default();
    (case.stock)(&mut activation);
    let mut answers = Vec::with_capacity(cfg.pool);
    for (v, (batch, bound)) in batches.iter().zip(&bounds).enumerate() {
        let clean = match bound.collect_on(Tier::Clean) {
            Ok(values) => values,
            Err(e) => return refuse(format!("pool row {v} has no answer on Clean: {e}")),
        };
        let mirrored = RowReader::new(batch).scope(&activation, 0);
        assert_eq!(
            Value::resolve_value(program.expression(), &mirrored)
                .ok()
                .as_ref(),
            clean.first(),
            "{}: pool row {v}: the tree-walker over the batch's own columns \
             disagrees with the interpreter tier",
            case.label
        );
        answers.push(clean);
    }
    let answers_distinct = distinct_count(&answers);

    // A fresh driver, so the warm-up below counts THIS case's calls and not an
    // artifact an earlier case left compiled at the same program address.
    reset_persistent_state();
    reset_jit_stats();

    // PHASE 1 — warm the entry door.
    let warm = warm_to_steady(&bounds, cfg, &case.label);

    // PHASE 2a — the oracle pass. Every call of the steady-state sequence, in
    // the same order, answered by the compiled tier and checked against the
    // interpreter. It runs UNTIMED and beside the timed loop rather than inside
    // it: an oracle call in the timed loop would put the interpreter's cost into
    // the compiled tier's number, which is the one thing this file exists to
    // measure.
    //
    // It also carries the PER-CALL entry evidence. The gate below can only
    // compare the window's entry count with its call count, and an aggregate
    // `entries >= calls` is not the claim "every call entered": one call with a
    // nested loop can contribute several entries while another contributes
    // none, and the sum still clears the bound. This loop makes the same N calls
    // in the same order, so differencing the entry counter across EACH of them
    // answers the per-call question — and being untimed, it can pay the counter
    // read the timed loop must not.
    // `None` while no replay call has been made, so a run configured with no
    // replay calls at all refuses the gate below instead of clearing it with an
    // extremum nothing ever wrote to.
    let mut replay_entries: Option<(usize, usize)> = None;
    for i in 0..cfg.steady {
        let v = i % bounds.len();
        let before_call = jit_stats().compiled_entries;
        let got = bounds[v]
            .collect_on(Tier::Jit)
            .unwrap_or_else(|e| panic!("{}: oracle call {i}: {e}", case.label));
        let delta = jit_stats().compiled_entries - before_call;
        replay_entries = Some(match replay_entries {
            Some((lo, hi)) => (lo.min(delta), hi.max(delta)),
            None => (delta, delta),
        });
        assert_eq!(
            got, answers[v],
            "{}: call {i} (pool row {v}) diverged from the interpreter tier. An \
             artifact that specialised on an earlier call's bindings answers \
             that call's question here",
            case.label
        );
    }

    // PHASE 2b — the timed loop.
    let before = counters();
    let (jit_ns, timed_calls) = steady_ns(cfg, |i| {
        black_box(
            bounds[i % bounds.len()]
                .collect_on(Tier::Jit)
                .unwrap_or_else(|e| panic!("{}: steady call {i}: {e}", case.label)),
        );
    });
    let after = counters();

    // PHASE 2c — the same check again, AFTER the timed window. 2a proves the
    // artifact was right when the window opened; only this proves it still is, so
    // an artifact that degrades over the window cannot pass as a fast one.
    for (v, bound) in bounds.iter().enumerate() {
        let got = bound
            .collect_on(Tier::Jit)
            .unwrap_or_else(|e| panic!("{}: post-window row {v}: {e}", case.label));
        assert_eq!(
            got, answers[v],
            "{}: pool row {v} diverged from the interpreter tier AFTER the timed \
             window, having agreed before it",
            case.label
        );
    }

    // The gate. Every timed call must have entered compiled code, nothing may
    // have compiled or aborted inside the window, and the warm-up must have
    // reached steady state in the first place. Any one of those missing and the
    // cell prints `-`.
    //
    // "Every timed call entered" is carried by the untimed replay's per-call
    // minimum, not by the aggregate: `entered >= timed_calls` alone is cleared
    // by a window in which some calls entered twice and others not at all. The
    // aggregate is kept beside it because the replay's evidence is about the
    // same sequence rather than about these calls, and a window that entered
    // fewer times in total than the sequence it replays is a window that changed
    // behaviour between the two.
    let entered = after.entries - before.entries;
    let entered_every_replay_call = replay_entries.is_some_and(|(lo, _)| lo >= 1);
    let jit = (warm.reached
        && entered_every_replay_call
        && entered >= timed_calls
        && after.settled_since(&before))
    .then_some(jit_ns);

    // PHASE 3 — the reference. The same rows, the same cycle, on the plain VM.
    // `Tier::Clean` dispatches straight to the interpreter and takes no driver at
    // all, so the entry counter must not move; if it does, the two columns are
    // one tier printed twice.
    let clean_before = counters();
    let (clean_ns, _) = steady_ns(cfg, |i| {
        black_box(
            bounds[i % bounds.len()]
                .collect_on(Tier::Clean)
                .unwrap_or_else(|e| panic!("{}: clean call {i}: {e}", case.label)),
        );
    });
    let clean = (counters().entries == clean_before.entries).then_some(clean_ns);

    // The per-activation encoding a deployment pays and the timed loop does not.
    let (bind, _) = steady_ns(cfg, |i| {
        black_box(
            lowered
                .bind_per_row(&batches[i % batches.len()])
                .unwrap_or_else(|e| panic!("{}: rebind: {e}", case.label)),
        );
    });

    let per_call = |delta: usize| delta as f64 / timed_calls as f64;
    Row {
        label: case.label.clone(),
        outcome: Ok(Measured {
            warm,
            jit,
            clean,
            bind,
            prepare,
            entries_per_call: per_call(entered),
            replay_entries,
            gfails_per_call: per_call(after.guard_fails - before.guard_fails),
            aborts_per_call: per_call(after.aborts - before.aborts),
            // Saturating where the three above subtract plainly: this one is a
            // population read twice, not a monotonic counter, and a clamped zero
            // is visibly uninformative where a negative number would be wrong.
            bridges_per_call: per_call(after.bridges.saturating_sub(before.bridges)),
            compiles: after.compiles,
            inputs,
            answers: answers_distinct,
            timed_calls,
        }),
    }
}

/// How many of `items` are distinct.
///
/// Quadratic, over a pool of sixteen. A `HashSet` would want `Hash` on `Value`
/// and on `Col`, and the pool is small enough that the comparison is not worth
/// a trait bound the corpus would then have to satisfy.
fn distinct_count<T: PartialEq>(items: &[T]) -> usize {
    let mut seen: Vec<&T> = Vec::new();
    for item in items {
        if !seen.iter().any(|s| *s == item) {
            seen.push(item);
        }
    }
    seen.len()
}

/// `warm_ns / (clean_per_call - jit_per_call)`: records after which the warm-up
/// has paid for itself.
///
/// `None` where the compiled tier is not faster, in which case there is no
/// crossing to report and the case has not yet won. That is an outcome the table
/// has to be able to print rather than a hole in it.
fn break_even(m: &Measured) -> Option<f64> {
    let (jit, clean) = (m.jit?, m.clean?);
    (clean > jit).then(|| m.warm.nanos / (clean - jit))
}

/// One optional cell at a fixed precision, or `-` where the gate refused it.
fn cell(v: Option<f64>, prec: usize) -> String {
    match v {
        Some(x) => format!("{x:.prec$}"),
        None => "-".to_string(),
    }
}

/// The warm-up's call count, with an explicit marker where it is a FLOOR rather
/// than an answer.
fn warm_calls(w: &Warm, cfg: &Config) -> String {
    if w.reached {
        w.calls.to_string()
    } else {
        format!(">{}", cfg.cap)
    }
}

fn print_steady_table(rows: &[Row], cfg: &Config) {
    println!(
        "\n{:<28} {:>9} {:>11} {:>10} {:>11} {:>11} {:>9} {:>12}",
        "case",
        "prep us",
        "warm calls",
        "warm ms",
        "steady ns",
        "clean ns",
        "speedup",
        "break-even"
    );
    for row in rows {
        let m = match &row.outcome {
            Ok(m) => m,
            Err(why) => {
                println!(
                    "{:<28} {:>9} {:>11} {:>10} {:>11} {:>11} {:>9} {:>12}",
                    row.label, "-", "-", "-", "-", "-", "-", why
                );
                continue;
            }
        };
        let speedup = match (m.jit, m.clean) {
            (Some(j), Some(c)) if j > 0.0 => format!("{:.2}x", c / j),
            _ => "-".to_string(),
        };
        println!(
            "{:<28} {:>9.1} {:>11} {:>10.1} {:>11} {:>11} {:>9} {:>12}",
            row.label,
            m.prepare / 1_000.0,
            warm_calls(&m.warm, cfg),
            m.warm.nanos / 1e6,
            cell(m.jit, 1),
            cell(m.clean, 1),
            speedup,
            cell(break_even(m), 0),
        );
    }
    println!(
        "\n* `prep us` is one parse plus one lower, the fastest of {PREPARE_SAMPLES}. It is paid\n  \
           once per expression, before any record is seen. It does NOT include building the\n  \
           program's word buffer: that is memoised behind the first `bind`, so the first\n  \
           call of the warm-up pays it and `warm ms` contains it.\n\
         * `warm calls` is one-row calls until the tier entered compiled code on {} consecutive\n  \
           calls with nothing compiling or aborting between them — no root loop AND no bridge.\n  \
           A `>{}` is a case that never got there in the cap, and its `steady ns` is `-` rather\n  \
           than a number taken from a tier that was still warming.\n\
         * `steady ns` is one whole one-row call on the compiled tier — the fastest of {} rounds\n  \
           of {} calls, cycling {} distinct input rows. `bind` is NOT in it; see the evidence\n  \
           table's `bind ns` and the header.\n\
         * `clean ns` is the same call on `Tier::Clean`, the plain Rust bytecode VM with no\n  \
           tracing machinery at all. It is what one evaluation costs with no JIT, and the\n  \
           number `steady ns` has to beat to mean anything.\n\
         * `break-even` is `warm ms / (clean - steady)`, in RECORDS: evaluate the expression\n  \
           this many times and the warm-up has paid for itself. Printed only where the compiled\n  \
           tier is actually faster; a `-` there is a case that does not yet win, and reading it\n  \
           as a small break-even is exactly backwards. It is an UPPER bound twice over — the\n  \
           warm-up's clock includes this harness's own counter reads, and the whole warm-up is\n  \
           charged rather than its excess over what the interpreter would have cost for the\n  \
           same calls.",
        cfg.window,
        cfg.cap,
        cfg.rounds,
        cfg.steady,
        cfg.pool,
    );
}

fn print_evidence_table(rows: &[Row]) {
    println!(
        "\nthe evidence that the numbers above are about compiled code, and about more\n\
         than one input row:\n"
    );
    println!(
        "{:<28} {:>11} {:>12} {:>11} {:>9} {:>12} {:>12} {:>13} {:>7} {:>8} {:>10}",
        "case",
        "enter/call",
        "per-call min",
        "1st entry",
        "compiles",
        "gfails/call",
        "aborts/call",
        "bridges/call",
        "inputs",
        "answers",
        "bind ns",
    );
    for row in rows {
        let m = match &row.outcome {
            Ok(m) => m,
            Err(_) => continue,
        };
        let first = match m.warm.first_entry {
            Some(c) => c.to_string(),
            None => "never".to_string(),
        };
        let replay = match m.replay_entries {
            Some((lo, hi)) => format!("{lo}..{hi}"),
            None => "-".to_string(),
        };
        println!(
            "{:<28} {:>11.2} {:>12} {:>11} {:>9} {:>12.2} {:>12.2} {:>13.2} {:>7} {:>8} {:>10.1}",
            row.label,
            m.entries_per_call,
            replay,
            first,
            m.compiles,
            m.gfails_per_call,
            m.aborts_per_call,
            m.bridges_per_call,
            m.inputs,
            m.answers,
            m.bind,
        );
    }
    println!(
        "\n* `enter/call` counts calls that ENTERED compiled code, at the point the compiled\n  \
           body is about to run — not artifacts minted. It is a MEAN over the timed window,\n  \
           so on its own it cannot tell a window where every call entered once from one where\n  \
           half entered twice and half not at all.\n\
         * `per-call min..max` is the compiled-entry delta of a SINGLE call, smallest and\n  \
           largest, measured over the untimed oracle replay — the same calls in the same order\n  \
           as the timed loop, with the counter read the timed loop cannot afford. This is the\n  \
           evidence behind \"every call enters\": a minimum of 0 refuses the `steady ns` cell\n  \
           beside it. A flat expression reads `1..1`; a maximum above 1 is a row body entering\n  \
           an inner loop's artifact more than once per call, which is exactly the case the\n  \
           aggregate would have hidden.\n\
         * `1st entry` is the call on which compiled code first ran, and it says which of the\n  \
           two doors the case came in through. Around {} — the trace threshold — is the\n  \
           function-entry door, which counts CALLS. A 1 or a 2 is a row BODY whose own loop\n  \
           got hot inside the first call; the entry door then declines for that program\n  \
           permanently and by design, because a program whose loop is compiled already has a\n  \
           way in and a second door in front of it takes one away.\n\
         * `inputs` and `answers` are the distinct input rows the pool holds and the distinct\n  \
           results they produced. A `1`/`1` is an expression with no columns to vary: its\n  \
           per-call oracle check is still true and still run, but it cannot detect an artifact\n  \
           that baked its first call's values in as constants, because there is nothing for\n  \
           such an artifact to get wrong. Read those cases' speedups with that in mind.\n\
         * `gfails/call` at or near 1.00 is a guard failing on every call. With bindings that\n  \
           vary that is not automatically a defect — a value-dependent branch has to exit\n  \
           somewhere — but it is where a disappointing `steady ns` is explained, and a fixed\n  \
           cost reported without it is half a finding.\n\
         * `bridges/call` counts artifacts COMPILED inside the timed window, not artifacts\n  \
           entered. A zero says the population stopped growing, NOT that the call enters no\n  \
           bridge. A nonzero one says a warm call is still compiling, which puts compilation\n  \
           itself inside `steady ns` — and now refuses that cell rather than only annotating\n  \
           it, on the same footing as a root-loop compile.\n\
         * `bind ns` is one `bind_per_row` of a pool row, timed on its own, over a `Batch` that\n  \
           ALREADY EXISTS. The steady-state loop does not pay it — the pool is encoded up front\n  \
           — while a deployment handed a fresh record per call does. `steady + bind` bounds the\n  \
           already-encoded activation path and nothing wider: materializing a record into\n  \
           column storage and building the `Batch` over it are outside both numbers, so this is\n  \
           not a deployment end-to-end figure.",
        cel::majit::batch::DEFAULT_JIT_THRESHOLD,
    );
}

/// One line per case, in a stable key=value shape, so two runs of this file can
/// be diffed by a machine rather than read side by side.
///
/// Every key is present on every line whatever happened, with `-` where a gate
/// refused a number: a diff over lines whose FIELD SET changes with the outcome
/// reports the schema change and hides the number change underneath it.
fn print_machine_readable(rows: &[Row], cfg: &Config) {
    println!(
        "\n#steady-config pool={} steady={} rounds={} window={} cap={}",
        cfg.pool, cfg.steady, cfg.rounds, cfg.window, cfg.cap
    );
    for row in rows {
        let m = match &row.outcome {
            Ok(m) => m,
            Err(why) => {
                println!(
                    "#steady case={} status=declined reason={:?} jit_ns=- clean_ns=- speedup=- \
                     breakeven=- warm_calls=- warm_reached=- warm_ns=- prepare_ns=- bind_ns=- \
                     entries=- replay_entries_min=- replay_entries_max=- gfails=- aborts=- \
                     bridges=- compiles=- inputs=- answers=- calls=-",
                    row.label, why
                );
                continue;
            }
        };
        let status = match m.jit {
            Some(_) => "ok",
            None => "unsteady",
        };
        let speedup = match (m.jit, m.clean) {
            (Some(j), Some(c)) if j > 0.0 => format!("{:.4}", c / j),
            _ => "-".to_string(),
        };
        let (replay_min, replay_max) = match m.replay_entries {
            Some((lo, hi)) => (lo.to_string(), hi.to_string()),
            None => ("-".to_string(), "-".to_string()),
        };
        println!(
            "#steady case={} status={status} jit_ns={} clean_ns={} speedup={speedup} \
             breakeven={} warm_calls={} warm_reached={} warm_ns={:.0} prepare_ns={:.0} \
             bind_ns={:.1} entries={:.4} replay_entries_min={replay_min} \
             replay_entries_max={replay_max} gfails={:.4} aborts={:.4} bridges={:.4} \
             compiles={} inputs={} answers={} calls={}",
            row.label,
            cell(m.jit, 2),
            cell(m.clean, 2),
            cell(break_even(m), 0),
            m.warm.calls,
            u8::from(m.warm.reached),
            m.warm.nanos,
            m.prepare,
            m.bind,
            m.entries_per_call,
            m.gfails_per_call,
            m.aborts_per_call,
            m.bridges_per_call,
            m.compiles,
            m.inputs,
            m.answers,
            m.timed_calls,
        );
    }
}

fn main() {
    let cfg = Config::from_args();
    println!(
        "CEL's steady-state per-call cost: one expression prepared once, then evaluated\n\
         one activation per call with VARYING bindings — the way a policy engine uses it.\n"
    );
    println!(
        "pool={} distinct input rows, {} timed calls per round, best of {} rounds;\n\
         steady state is {} consecutive calls entering compiled code with nothing\n\
         compiling — root loop or bridge — and nothing aborting, given at most {}\n\
         calls to get there.",
        cfg.pool, cfg.steady, cfg.rounds, cfg.window, cfg.cap
    );

    let cases = cases();
    let rows: Vec<Row> = cases.iter().map(|c| run_case(c, &cfg)).collect();

    print_steady_table(&rows, &cfg);
    print_evidence_table(&rows);

    let declined: Vec<&Row> = rows.iter().filter(|r| r.outcome.is_err()).collect();
    let unsteady: Vec<&Row> = rows
        .iter()
        .filter(|r| matches!(&r.outcome, Ok(m) if m.jit.is_none()))
        .collect();
    println!(
        "\ncoverage: {}/{} cases produced a steady-state number.",
        rows.len() - declined.len() - unsteady.len(),
        rows.len()
    );
    for row in &declined {
        let why = row.outcome.as_ref().err().expect("filtered to errors");
        println!("  skipped  {:<28} {why}", row.label);
    }
    for row in &unsteady {
        let m = row.outcome.as_ref().ok().expect("filtered to measured");
        let why = if !m.warm.reached {
            format!(
                "never reached steady state in {} calls (first entry: {})",
                m.warm.calls,
                m.warm
                    .first_entry
                    .map_or("never".to_string(), |c| c.to_string()),
            )
        } else {
            // Which of the three post-warm conditions refused the cell, named
            // rather than summarised: the per-call minimum and the aggregate
            // fail on different populations, and a compile inside the window is
            // a third thing entirely.
            match m.replay_entries {
                None => {
                    "warmed, but no replay call was made to measure per-call entry with".to_string()
                }
                Some((0, hi)) => format!(
                    "warmed, but at least one call of the replay entered no compiled code \
                     (per-call entries 0..{hi})"
                ),
                Some(_) => format!(
                    "warmed, but the timed window entered compiled code on {:.2} of each call \
                     and compiled {:.2} bridges per call",
                    m.entries_per_call, m.bridges_per_call
                ),
            }
        };
        println!("  no number {:<28} {why}", row.label);
    }

    print_machine_readable(&rows, &cfg);
}
