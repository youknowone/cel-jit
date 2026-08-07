//! **Allocations per evaluation** over a fixed corpus — the P0.a instrument of
//! the `cel-unboxed-values` design (task #52), and the primary gate for every
//! phase that changes cel's data model.
//!
//! An allocation COUNT is a property of the code, not of the machine: it does
//! not move with load average the way a timing does. That is the whole reason
//! this is the gate and `cargo bench` is not.
//!
//! ## Why this is not `examples/allocs.rs`
//!
//! `examples/allocs.rs` is an example BINARY wired to the columnar batch API
//! (`Batch`/`BatchProgram`/`ColumnRef`/`Tier`, `ROWS = 50_000`) — the
//! front-end-A tier phase P9 retires. It reports allocations per ROW of a
//! 50,000-row batch. This file reports allocations per EVALUATION of the door a
//! caller actually holds, over the corpus P0.a names, and it is a checked-in
//! test rather than a binary someone has to remember to run.
//!
//! ## Isolation — the two failure modes, and what was done about them
//!
//! 1. **A global allocator counts the harness's own allocations.** The counter
//!    is armed only around the measured call and disarmed before anything is
//!    formatted, pushed, or printed: `Meter::start`/`Meter::stop` read a `Cell`
//!    and an atomic and allocate nothing, and every `String`, `Vec` and
//!    `println!` in this file lives outside an armed window. Two probes prove
//!    it rather than asserting it — `probe/nothing` measures an empty body and
//!    must be 0.000, `probe/one-box-per-eval` measures one `Box::new` and must
//!    be exactly 1.000. Both are in the reported table, not hidden in a
//!    comment.
//!
//! 2. **The default test runner is multi-threaded, so a process-global counter
//!    races.** Two independent defences, because either alone is a silent
//!    failure:
//!    * this target is `harness = false` (see `Cargo.toml`), so there is no
//!      libtest thread pool and no second test running concurrently. This is
//!      the only way to get it: `--test-threads=1` cannot be spelled in
//!      `Cargo.toml`, so a plain `cargo test` would otherwise race.
//!    * the counter that is REPORTED is thread-local, so even an allocation
//!      made by some other thread inside the measured window cannot land in it.
//!      A second, process-global atomic counts every allocation the process
//!      makes; the difference is reported as `off-thread` and is the evidence
//!      that the thread-local isolation is doing (or not doing) something.
//!
//!    The thread-local is `const`-initialised and holds a `Cell<u64>`, which has
//!    no destructor — so it never lazily allocates from inside the allocator and
//!    never registers a TLS destructor. `try_with` keeps a late allocation
//!    during thread teardown from panicking.
//!
//! ## What one "evaluation" is, per group
//!
//! * `walker/*` — one `Program::execute(&ctx)` against a context built ONCE
//!   outside the window. That is cometkim's per-call regime
//!   (`examples/majit_vs_cometkim_percall.rs`), reproduced here so a number
//!   here can be read beside one there.
//! * `bind/*` — one `Context::add_variable_from_value(name, Value::List(..))`
//!   and nothing else. This is task #82's shape.
//! * `comprehension/*` — one bind PLUS one execute, which is what a caller with
//!   changing input actually pays.
//! * `regvm/*` — one call of the typed two-bank register machine
//!   (`src/majit/bytecode.rs` `float_bank`) over a whole `n`-row batch. §12
//!   item 10 of the design says the new class universe pays a header word and a
//!   pointer chase the register bank does not, so this group is what would show
//!   that regression. Requires the `jit` feature (with a backend); without it
//!   the rows are absent and reported as such.
//!
//! ## Running it
//!
//! ```text
//! cargo test -p cel --features jit-cranelift --test allocs_per_eval
//! ```
//!
//! `--features jit-cranelift` (or `jit-dynasm`) is what builds the `regvm/*`
//! group; a bare `cargo test -p cel` runs everything else and says which rows
//! it could not measure. Bare `--features jit` is a hard compile error by
//! design — it names no backend.
//!
//! `CEL_ALLOCS_BLESS=1` rewrites the baseline. `CEL_ALLOCS_GATE=1` turns a
//! drift from the baseline into a failure; it is OFF by default **on purpose**
//! — see the header of `tests/allocs_per_eval.baseline`.
//!
//! Each of those two configurations has its own baseline file
//! (`allocs_per_eval.baseline` and `allocs_per_eval.jit.baseline`) because they
//! measure different corpora. One file would leave whichever configuration it
//! was not blessed under comparing against nothing, which is a gate that cannot
//! fail rather than a gate that passes.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

use cel::context::VariableResolver;
use cel::{Context, Program, Value};

// ---------------------------------------------------------------------------
// the meter
// ---------------------------------------------------------------------------

std::thread_local! {
    /// Allocations made by THIS thread. `const`-initialised and destructor-free
    /// so the allocator can touch it without recursing through lazy TLS setup.
    static LOCAL_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

/// Allocations made by the whole process. Only ever read as
/// `global - local`, which is the count that could NOT have been the subject —
/// i.e. the contamination the thread-local excludes.
static GLOBAL_ALLOCS: AtomicU64 = AtomicU64::new(0);

/// Counts allocations; `dealloc` is deliberately not counted, because the
/// question is how much work the evaluation DOES and a freed allocation was
/// still made.
struct Counting;

#[inline]
fn bump() {
    GLOBAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
    // `try_with`, not `with`: an allocation during this thread's teardown must
    // not turn into a panic inside the allocator.
    let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    // `realloc` is overridden rather than left to the default: the default
    // impl is alloc + copy + dealloc, which would both count correctly AND
    // remove `System.realloc`'s in-place growth. Forwarding keeps the growth
    // behaviour a caller actually sees, and one realloc counts as one
    // allocation event.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// An armed window. Reading a `Cell` and one relaxed atomic, so starting and
/// stopping the meter allocates nothing — which `probe/nothing` then proves
/// rather than assumes.
struct Meter {
    local: u64,
    global: u64,
}

impl Meter {
    #[inline]
    fn start() -> Meter {
        Meter {
            local: LOCAL_ALLOCS.with(Cell::get),
            global: GLOBAL_ALLOCS.load(Ordering::Relaxed),
        }
    }

    /// `(this thread's allocations, allocations made by any OTHER thread)`.
    #[inline]
    fn stop(self) -> (u64, u64) {
        let local = LOCAL_ALLOCS.with(Cell::get) - self.local;
        let total = GLOBAL_ALLOCS.load(Ordering::Relaxed) - self.global;
        (local, total.saturating_sub(local))
    }
}

// ---------------------------------------------------------------------------
// rows
// ---------------------------------------------------------------------------

/// Independent measurement rounds per row. A row whose rounds disagree is not
/// blessable, and saying so is the point of taking more than one.
const ROUNDS: usize = 3;

struct Row {
    label: String,
    /// Evaluations inside one armed window. > 1 so a cost amortized across
    /// calls shows up as a fractional per-eval number instead of hiding in
    /// whichever call happened to pay it.
    iters: u32,
    /// One total per round.
    samples: [u64; ROUNDS],
    /// Allocations made by another thread inside the armed windows. Expected 0;
    /// a nonzero here is exactly the contamination the thread-local excludes.
    off_thread: u64,
    /// Free-form evidence about what the row measured, printed in the note
    /// column. `regvm/jit/*` uses it to say whether the number is a WARM cost
    /// or the cost of the tier re-tracing inside the window — a per-call
    /// allocation figure means two different things in those two cases.
    detail: String,
}

impl Row {
    fn min_per_eval(&self) -> f64 {
        *self.samples.iter().min().unwrap() as f64 / self.iters as f64
    }
    fn max_per_eval(&self) -> f64 {
        *self.samples.iter().max().unwrap() as f64 / self.iters as f64
    }
    fn stable(&self) -> bool {
        self.samples.iter().min() == self.samples.iter().max()
    }
}

/// Measure one row. Everything that allocates — the label, the `Vec` push, the
/// warm-up — happens outside the armed window.
fn bench(out: &mut Vec<Row>, label: String, warmup: u32, iters: u32, mut body: impl FnMut()) {
    // Warm first: lazy statics, the parser's tables, a JIT trace, and a `Vec`
    // reaching its steady-state capacity are all one-offs that would otherwise
    // land entirely in whichever round ran first.
    for _ in 0..warmup {
        body();
    }
    let mut samples = [0u64; ROUNDS];
    let mut off_thread = 0u64;
    for slot in samples.iter_mut() {
        let meter = Meter::start();
        for _ in 0..iters {
            black_box(body());
        }
        let (local, other) = meter.stop();
        *slot = local;
        off_thread += other;
    }
    out.push(Row {
        label,
        iters,
        samples,
        off_thread,
        detail: String::new(),
    });
}

// ---------------------------------------------------------------------------
// group: the two isolation probes
// ---------------------------------------------------------------------------

/// Rows whose `off_thread` count is the POINT, not contamination.
const EXPECTS_OFF_THREAD: &str = "probe/other-thread-allocating";

fn probes(out: &mut Vec<Row>) {
    // If the meter itself allocated, this would not be 0.
    bench(out, "probe/nothing".into(), 4, 64, || {
        black_box(0u64);
    });
    // And if it did not count, this would not be 1.
    bench(out, "probe/one-box-per-eval".into(), 4, 64, || {
        drop(black_box(Box::new(0u64)));
    });

    // The multi-threaded-runner failure mode, demonstrated rather than argued:
    // a second thread allocates flat out for the whole of a measured window
    // whose body allocates nothing. The reported (thread-local) count must
    // still be 0, and `off_thread` must be nonzero — which is what says the
    // process-global counter DID see those allocations and a global-only meter
    // would have charged them to this row.
    static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    STOP.store(false, Ordering::Relaxed);
    let noisy = std::thread::spawn(|| {
        while !STOP.load(Ordering::Relaxed) {
            drop(black_box(Box::new([0u8; 64])));
        }
    });
    // The body WAITS for the other thread to allocate rather than merely
    // hoping it does inside the window. A no-op body would leave the window
    // nanoseconds long, and on a fast host the probe would pass with
    // `off_thread == 0` — i.e. vacuously, having exercised nothing. Spinning on
    // the global counter allocates nothing itself, so the local count is still
    // required to be 0.
    bench(out, EXPECTS_OFF_THREAD.into(), 4, 8, || {
        let before = GLOBAL_ALLOCS.load(Ordering::Relaxed);
        while GLOBAL_ALLOCS.load(Ordering::Relaxed) == before {
            std::hint::spin_loop();
        }
    });
    STOP.store(true, Ordering::Relaxed);
    noisy.join().expect("noisy thread panicked");
}

// ---------------------------------------------------------------------------
// group: the cometkim per-call expression set, on the tree-walker
// ---------------------------------------------------------------------------

/// cometkim's `benchmark_variable_access` resolver, as reproduced in
/// `examples/majit_vs_cometkim_percall.rs`.
struct Resolver;

impl VariableResolver for Resolver {
    fn resolve(&self, expr: &str) -> Option<Value> {
        const V: Value = Value::Bool(false);
        const NOT_V: Value = Value::Bool(true);
        match expr {
            "fruit" | "carrot" | "orange" => Some(NOT_V),
            "banana" => Some(V),
            _ => None,
        }
    }
}

static RESOLVER: Resolver = Resolver;

struct WalkerCase {
    label: String,
    src: String,
    /// The activation cometkim adds by name to a ROOT context, built once and
    /// held fixed outside the timed call — his regime, and ours.
    setup: Box<dyn Fn(&mut Context<'static>)>,
    via_resolver: bool,
    iters: u32,
}

fn walker_case(label: &str, src: &str) -> WalkerCase {
    WalkerCase {
        label: label.to_string(),
        src: src.to_string(),
        setup: Box::new(|_| {}),
        via_resolver: false,
        iters: 8,
    }
}

/// A `HashMap` literal the way his benchmark writes one.
fn map_of(pairs: Vec<(&'static str, Value)>) -> Value {
    Value::from(pairs.into_iter().collect::<HashMap<&str, Value>>())
}

/// His 18 benchmark expressions with his contexts, size ladders included —
/// `examples/majit_vs_cometkim_percall.rs` `cases()`.
fn cometkim_cases() -> Vec<WalkerCase> {
    let mut cases = vec![
        walker_case("simple_arithmetic", "1 + 2 * 3 - 4 / 2"),
        walker_case("comparison", "10 > 5 && 3 < 7 || 1 == 1"),
        WalkerCase {
            setup: Box::new(|ctx| ctx.add_variable_from_value("x", 15i64)),
            ..walker_case("conditional", "x > 10 ? x * 2 : x + 5")
        },
        WalkerCase {
            setup: Box::new(|ctx| {
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
            ..walker_case(
                "nested_expression",
                "((a + b) * (c - d)) / ((e + f) - (g * h))",
            )
        },
        WalkerCase {
            setup: Box::new(|ctx| ctx.add_variable_from_value("apple", true)),
            ..walker_case("variable_access/hashmap", "apple")
        },
        WalkerCase {
            via_resolver: true,
            ..walker_case("variable_access/resolver", "banana")
        },
        WalkerCase {
            setup: Box::new(|ctx| {
                let obj = map_of(vec![
                    ("nested", map_of(vec![("value", Value::Int(42))])),
                    ("other", Value::Int(10)),
                ]);
                ctx.add_variable_from_value("obj", obj);
            }),
            ..walker_case("member_access", "obj.nested.value + obj.other")
        },
        WalkerCase {
            setup: Box::new(|ctx| {
                ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())
            }),
            ..walker_case("list_indexing", "list[0] + list[5] + list[9]")
        },
        walker_case(
            "list_filter",
            "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].filter(x, x > 5)",
        ),
        walker_case("list_map", "[1, 2, 3, 4, 5].map(x, x * 2)"),
        walker_case("all_comprehension", "[1, 2, 3, 4, 5].all(x, x > 0)"),
        walker_case("exists_comprehension", "[1, 2, 3, 4, 5].exists(x, x == 3)"),
    ];

    let ladder = |label: String, src: &str, name: &'static str, elems: Vec<i64>| {
        // A big ladder rung costs milliseconds per evaluation in a dev build,
        // so it gets fewer evaluations per window — the count is exact, the
        // repetition is only there to expose amortization.
        let iters = if elems.len() >= 1_000 { 2 } else { 8 };
        WalkerCase {
            label,
            setup: Box::new(move |ctx| ctx.add_variable_from_value(name, elems.clone())),
            iters,
            ..walker_case("", src)
        }
    };
    for size in [1i64, 10, 100, 1_000, 10_000] {
        cases.push(ladder(
            format!("map_list_scaling/{size}"),
            "list.map(x, x * 2)",
            "list",
            (0..size).collect(),
        ));
    }
    for size in [1i64, 10, 100, 1_000, 10_000] {
        cases.push(ladder(
            format!("filter_list_scaling/{size}"),
            "list.filter(x, x % 2 == 0)",
            "list",
            (0..size).collect(),
        ));
    }
    for size in [10i64, 50, 100, 500] {
        cases.push(ladder(
            format!("comprehension_scaling/{size}"),
            "items.filter(x, x % 2 == 0).map(x, x * 2)",
            "items",
            (1..=size).collect(),
        ));
    }

    cases.push(walker_case(
        "string_operations",
        r#""hello world".startsWith("hello") && "hello world".endsWith("world") && "hello world".contains("o w")"#,
    ));
    // The same three calls as `string_operations`, on a variable rather than on
    // a literal. `string_operations` reaches the member-call path with an
    // `Expr::Member`-free literal target, which is a different arm from the one
    // an ordinary `s.startsWith(..)` takes -- and no other row in this corpus
    // takes that arm, so the whole corpus was blind to its cost (task #98).
    cases.push(WalkerCase {
        setup: Box::new(|ctx| ctx.add_variable_from_value("s", Value::from("hello world"))),
        ..walker_case(
            "member_call_on_variable",
            r#"s.startsWith("hello") && s.endsWith("world") && s.contains("o w")"#,
        )
    });
    cases.push(WalkerCase {
        setup: Box::new(|ctx| {
            for (name, v) in [("x", 10i64), ("y", 20), ("a", 5), ("b", 3)] {
                ctx.add_variable_from_value(name, v);
            }
            ctx.add_function("add", |a: i64, b: i64| a + b);
            ctx.add_function("multiply", |a: i64, b: i64| a * b);
        }),
        ..walker_case("custom_function", "add(x, y) + multiply(a, b)")
    });
    cases.push(WalkerCase {
        setup: Box::new(|ctx| {
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
        ..walker_case(
            "real_world_policy",
            r#"user.age >= 18 &&
               user.role in ["admin", "moderator"] &&
               request.method == "POST" &&
               request.path.startsWith("/api/") &&
               size(request.body) < 1000000"#,
        )
    });
    cases
}

fn walker_group(out: &mut Vec<Row>) {
    for case in cometkim_cases() {
        let program = Program::compile(&case.src)
            .unwrap_or_else(|e| panic!("{}: parse error: {e:?}", case.label));
        let mut ctx: Context<'static> = Context::default();
        (case.setup)(&mut ctx);
        if case.via_resolver {
            ctx.set_variable_resolver(&RESOLVER);
        }
        // Assert the case actually evaluates before measuring it: an
        // allocation count taken off an `Err` return measures the error path.
        program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("{}: execute error: {e:?}", case.label));
        let label = format!("walker/{}", case.label);
        bench(out, label, 4, case.iters, || {
            black_box(program.execute(&ctx)).ok();
        });
    }
}

// ---------------------------------------------------------------------------
// group: binding a Value::List to a Context (task #82)
// ---------------------------------------------------------------------------

fn bind_group(out: &mut Vec<Row>) {
    for n in [1usize, 10, 100, 1_000] {
        let list: Value = (0..n as i64).collect::<Vec<i64>>().into();
        assert!(matches!(list, Value::List(_)));
        let mut ctx: Context<'static> = Context::default();

        // The control row. `Value::List` is a `ListRef` — an `Arc` window — so
        // the clone the bind row includes should cost nothing, and this is what
        // says so rather than a comment claiming it.
        bench(out, format!("bind/clone-only/{n}"), 4, 8, || {
            drop(black_box(list.clone()));
        });
        bench(out, format!("bind/list-to-context/{n}"), 4, 8, || {
            ctx.add_variable_from_value("list", list.clone());
        });
    }
}

// ---------------------------------------------------------------------------
// group: comprehension — bind AND execute, which is what changing input costs
// ---------------------------------------------------------------------------

fn comprehension_group(out: &mut Vec<Row>) {
    let sources = [
        ("map", "list.map(x, x * 2)"),
        ("filter", "list.filter(x, x % 2 == 0)"),
        ("chained", "list.filter(x, x % 2 == 0).map(x, x * 2)"),
        ("all", "list.all(x, x >= 0)"),
        ("exists", "list.exists(x, x == 3)"),
        ("size", "size(list.map(x, x * 2))"),
    ];
    for n in [10usize, 100] {
        let list: Value = (0..n as i64).collect::<Vec<i64>>().into();
        for (name, src) in sources {
            let program = Program::compile(src).unwrap_or_else(|e| panic!("{src}: {e:?}"));
            let mut ctx: Context<'static> = Context::default();
            ctx.add_variable_from_value("list", list.clone());
            program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("{src}: execute error: {e:?}"));
            bench(
                out,
                format!("comprehension/bind+exec/{name}/{n}"),
                4,
                8,
                || {
                    ctx.add_variable_from_value("list", list.clone());
                    black_box(program.execute(&ctx)).ok();
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// group: the register machine's own cases (design §12 item 10)
// ---------------------------------------------------------------------------

#[cfg(feature = "jit")]
fn regvm_group(out: &mut Vec<Row>) {
    use cel::majit::bytecode::float_bank::{
        clean_interp_seeded_f, intern_program, jit_stats, reset_jit_stats, reset_persistent_state,
        run_jit_persistent_f,
    };
    use cel::majit::lower::{lower_typed, Schema, ValType};

    /// A column of the batch, kept alive for the whole measurement so the
    /// base addresses seeded into the register bank stay valid.
    enum Col {
        Int(Vec<i64>),
        Float(Vec<f64>),
    }

    impl Col {
        fn base(&self) -> i64 {
            match self {
                Col::Int(v) => v.as_ptr() as i64,
                Col::Float(v) => v.as_ptr() as i64,
            }
        }
    }

    // Threshold the in-tree tests use. Small enough that an `n`-row loop traces
    // within one call at n >= 8, and within a handful of calls at n == 1.
    const JIT_ON: u32 = 8;

    struct Case {
        label: &'static str,
        src: &'static str,
        schema: &'static [(&'static str, ValType)],
    }

    // Three shapes, chosen so the int bank, the comparison/boolean path and the
    // float bank are each represented. No bool COLUMN: a bool column is a
    // one-byte read and getting its stride wrong would measure a different
    // program than the one named here.
    const CASES: &[Case] = &[
        Case {
            label: "arith",
            src: "a * 2 + b",
            schema: &[("a", ValType::Int), ("b", ValType::Int)],
        },
        Case {
            label: "policy",
            src: "balance >= amount && balance % 2 == 0",
            schema: &[("balance", ValType::Int), ("amount", ValType::Int)],
        },
        Case {
            label: "float",
            src: "x * 1.5 + y",
            schema: &[("x", ValType::Float), ("y", ValType::Float)],
        },
    ];

    for case in CASES {
        let program =
            Program::compile(case.src).unwrap_or_else(|e| panic!("{}: {e:?}", case.label));
        let schema: Schema = case
            .schema
            .iter()
            .map(|(n, t)| (n.to_string(), *t))
            .collect();
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("{}: lower_typed: {e}", case.label));

        for n in [1usize, 1_000] {
            // Build the columns in the lowering's OWN slot order. The schema is
            // a `HashMap`, so an order taken from the literal above would be
            // whatever the hasher produced.
            let cols: Vec<Col> = lowered
                .slots
                .iter()
                .map(|slot| match slot.ty {
                    ValType::Float => Col::Float((0..n).map(|k| (k % 97) as f64 * 0.5).collect()),
                    _ => Col::Int((0..n as i64).map(|k| (k * 7) % 97).collect()),
                })
                .collect();
            let bases: Vec<i64> = cols.iter().map(Col::base).collect();
            let (shape, regs) = lowered.batch_sum_program(&bases, n as i64);
            let nf = shape.num_float_regs;
            // Interned, because the `#[jit_interp]` green key is the program
            // POINTER: a fresh `Vec` per call would never reuse a compiled loop.
            let code = intern_program(shape.code);

            let expected = clean_interp_seeded_f(&code, &regs, nf);
            bench(
                out,
                format!("regvm/clean/{}/n={n}", case.label),
                4,
                8,
                || {
                    black_box(clean_interp_seeded_f(&code, &regs, nf));
                },
            );

            // The JIT arm runs on a driver that outlives the call, so a loop
            // compiled during warm-up is still compiled inside the window.
            reset_persistent_state();
            let got = run_jit_persistent_f(&code, &regs, nf, JIT_ON);
            assert_eq!(
                got, expected,
                "regvm/{}/n={n}: compiled tier diverged from the clean VM",
                case.label
            );
            // A long warm-up on purpose: at n == 1 the inner loop runs once per
            // CALL, so the merge point needs many calls to get hot (task #83).
            //
            // The tier's own counters are read across the whole `bench` call so
            // the number above can be READ AT ALL: an allocation figure taken
            // while the tier is still tracing is the cost of COMPILING, and one
            // taken after it is warm is the cost of ENTERING compiled code.
            // Those are different facts and the table must not conflate them.
            const WARMUP: u32 = 64;
            const ITERS: u32 = 8;
            reset_jit_stats();
            let before = jit_stats();
            let first = out.len();
            bench(
                out,
                format!("regvm/jit/{}/n={n}", case.label),
                WARMUP,
                ITERS,
                || {
                    black_box(run_jit_persistent_f(&code, &regs, nf, JIT_ON));
                },
            );
            let after = jit_stats();
            // The counters span the warm-up AND the measured windows — `bench`
            // owns the warm-up, so there is no seam to reset at. Saying so, and
            // giving the call count, is the difference between a readable
            // number and a misleading one.
            let calls = WARMUP + ITERS * ROUNDS as u32;
            let compiled = after.loops_compiled - before.loops_compiled;
            out[first].detail = format!(
                "over {calls} calls: compiled={compiled} aborted={} guard_fails={} — {}",
                after.loops_aborted - before.loops_aborted,
                after.guard_failures - before.guard_failures,
                if compiled == 0 {
                    "WARM (compiled before this, or never)"
                } else {
                    "NOT WARM: the tier compiled during the measurement"
                }
            );
            reset_persistent_state();
        }
    }
}

#[cfg(not(feature = "jit"))]
fn regvm_group(_out: &mut Vec<Row>) {}

// ---------------------------------------------------------------------------
// baseline
// ---------------------------------------------------------------------------

/// One baseline per row set, so both invocations anyone actually makes are
/// gated: a plain `cargo test -p cel` and a `--features jit-cranelift` run
/// measure different corpora, and a single file could only ever cover one of
/// them. A baseline that no real configuration can compare against is a gate
/// that cannot fail.
fn baseline_path() -> std::path::PathBuf {
    let name = if cfg!(feature = "jit") {
        "tests/allocs_per_eval.jit.baseline"
    } else {
        "tests/allocs_per_eval.baseline"
    };
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(name)
}

/// `label -> allocations per evaluation`, plus the `key: value` header fields.
fn read_baseline() -> (BTreeMap<String, f64>, BTreeMap<String, String>) {
    let mut rows = BTreeMap::new();
    let mut meta = BTreeMap::new();
    let Ok(text) = std::fs::read_to_string(baseline_path()) else {
        return (rows, meta);
    };
    for line in text.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("#!") {
            if let Some((k, v)) = rest.split_once('=') {
                meta.insert(k.trim().to_string(), v.trim().to_string());
            }
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((label, value)) = line.split_once('\t') {
            if let Ok(v) = value.trim().parse::<f64>() {
                rows.insert(label.to_string(), v);
            }
        }
    }
    (rows, meta)
}

fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        "release"
    }
}

fn features() -> String {
    let mut on: Vec<&str> = Vec::new();
    if cfg!(feature = "regex") {
        on.push("regex");
    }
    if cfg!(feature = "chrono") {
        on.push("chrono");
    }
    if cfg!(feature = "json") {
        on.push("json");
    }
    if cfg!(feature = "bytes") {
        on.push("bytes");
    }
    if cfg!(feature = "structs") {
        on.push("structs");
    }
    if cfg!(feature = "jit") {
        on.push("jit");
    }
    on.join(",")
}

fn write_baseline(rows: &[Row]) {
    let mut text = String::new();
    text.push_str(BASELINE_HEADER);
    text.push_str(&format!("#! profile = {}\n", profile()));
    text.push_str(&format!("#! features = {}\n", features()));
    text.push_str(&format!(
        "#! host = {}-{}\n",
        std::env::consts::ARCH,
        std::env::consts::OS
    ));
    text.push_str(&format!("#! rounds = {ROUNDS}\n"));
    text.push('\n');
    for row in rows {
        text.push_str(&format!("{}\t{:.3}\n", row.label, row.min_per_eval()));
    }
    let path = baseline_path();
    std::fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("\nblessed {}", path.display());
}

const BASELINE_HEADER: &str = "\
# cel — allocations per evaluation. Produced by `tests/allocs_per_eval.rs`.
#
# ⚠ THIS FILE IS PER-HOST. It is NOT an absolute contract, and the test does NOT
#   fail when a row differs from it. Run with `CEL_ALLOCS_GATE=1` to make a drift
#   fatal — do that in a re-measurement of ONE change on ONE machine, never as a
#   default CI gate.
#
# There are TWO baselines, one per row set, because there are two invocations
# people actually make and a single file can only cover one of them:
#
#   allocs_per_eval.baseline      no jit backend, 52 rows -- `cargo test -p cel`
#   allocs_per_eval.jit.baseline  a jit backend,  64 rows -- adds `regvm/*`
#
# Bless the one matching your configuration:
#
#   CEL_ALLOCS_BLESS=1 cargo test -p cel --test allocs_per_eval
#   CEL_ALLOCS_BLESS=1 cargo test -p cel --features jit-cranelift \\
#       --test allocs_per_eval
#
# A run whose `features` differ from the pair recorded below is not comparable
# and the report says so. `profile` is recorded but does not invalidate a
# comparison: dev and release were measured identical on every row.
#
# Format: <label> TAB <allocations per evaluation>
";

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

fn main() {
    let mut rows: Vec<Row> = Vec::new();
    probes(&mut rows);
    walker_group(&mut rows);
    bind_group(&mut rows);
    comprehension_group(&mut rows);
    regvm_group(&mut rows);

    let (base, meta) = read_baseline();
    let bless = std::env::var_os("CEL_ALLOCS_BLESS").is_some();
    let gate = std::env::var_os("CEL_ALLOCS_GATE").is_some();

    println!("allocations per evaluation — cel P0.a instrument");
    println!(
        "profile={} features={} host={}-{} rounds={ROUNDS}",
        profile(),
        features(),
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    // Only the feature set invalidates a comparison: it decides which rows exist
    // and which code paths are compiled in. `profile` does not -- dev and release
    // were measured row-for-row identical on all 52 JIT-free rows, which is what
    // an allocation COUNT should do. It is still reported, and a difference gets
    // a note rather than blacking out the `base` and `Δ` columns; suppressing
    // them for a difference that does not affect the numbers is how a baseline
    // ends up comparable to nothing anyone runs.
    let comparable = meta.get("features").map(String::as_str) == Some(&features());
    if !base.is_empty() && !comparable {
        println!(
            "⚠ {} was blessed under features={} — the `base` and `Δ` columns below \n\
             are NOT comparable to this run. Re-bless it in this configuration.",
            baseline_path().file_name().unwrap_or_default().display(),
            meta.get("features").map(String::as_str).unwrap_or("?"),
        );
    }
    if !base.is_empty() && meta.get("profile").map(String::as_str) != Some(profile()) {
        println!(
            "note: baseline blessed under profile={}, this run is {} — allocation \n\
             counts do not vary with the profile, so the columns still compare.",
            meta.get("profile").map(String::as_str).unwrap_or("?"),
            profile(),
        );
    }
    println!();
    println!(
        "{:<44} {:>5} {:>12} {:>12} {:>12} {:>10}  {}",
        "row", "iters", "allocs/eval", "min", "max", "base", "note"
    );

    let mut unstable: Vec<&str> = Vec::new();
    let mut drifted: Vec<&str> = Vec::new();
    let mut contaminated: Vec<&str> = Vec::new();
    for row in &rows {
        let now = row.min_per_eval();
        let (base_col, delta) = match base.get(&row.label) {
            Some(&b) => (format!("{b:.3}"), format!("{:+.3}", now - b)),
            None => ("-".to_string(), "new".to_string()),
        };
        let mut note = String::new();
        if !row.stable() {
            note.push_str("UNSTABLE ");
            unstable.push(&row.label);
        }
        if row.off_thread != 0 {
            note.push_str(&format!("off-thread={} ", row.off_thread));
            if row.label != EXPECTS_OFF_THREAD {
                contaminated.push(&row.label);
            }
        }
        if let Some(&b) = base.get(&row.label) {
            if (now - b).abs() > 1e-9 {
                note.push_str(&delta);
                drifted.push(&row.label);
            }
        } else {
            note.push_str(&delta);
        }
        if !row.detail.is_empty() {
            note.push(' ');
            note.push_str(&row.detail);
        }
        println!(
            "{:<44} {:>5} {:>12.3} {:>12.3} {:>12.3} {:>10}  {}",
            row.label,
            row.iters,
            now,
            row.min_per_eval(),
            row.max_per_eval(),
            base_col,
            note.trim_end()
        );
    }

    let measured: std::collections::BTreeSet<&str> =
        rows.iter().map(|r| r.label.as_str()).collect();
    let missing: Vec<&String> = base
        .keys()
        .filter(|k| !measured.contains(k.as_str()))
        .collect();
    if !missing.is_empty() {
        println!(
            "\nin the baseline but NOT measured in this configuration ({}):",
            missing.len()
        );
        for label in &missing {
            println!("  {label}");
        }
        if !cfg!(feature = "jit") {
            println!("  (build with `--features jit-cranelift` or `jit-dynasm` for `regvm/*`)");
        }
    }

    println!(
        "\nrows={} unstable={} drifted={} off-thread-contaminated={}",
        rows.len(),
        unstable.len(),
        drifted.len(),
        contaminated.len()
    );

    // The two isolation probes are the one thing that IS a hard gate: they are
    // properties of this file, not of the host, and if either moves every
    // number above is meaningless.
    let probe = |label: &str, want: f64| {
        let row = rows
            .iter()
            .find(|r| r.label == label)
            .unwrap_or_else(|| panic!("{label} row is missing"));
        assert!(
            (row.min_per_eval() - want).abs() < 1e-9 && row.stable(),
            "{label}: expected exactly {want} allocations/eval, got {:?} over {} iters — \
             the meter is counting its own harness or is not counting at all",
            row.samples,
            row.iters
        );
    };
    probe("probe/nothing", 0.0);
    probe("probe/one-box-per-eval", 1.0);
    // Same 0.000, but measured while another thread allocated continuously —
    // and the process-global counter must have SEEN those allocations, or the
    // probe proved nothing.
    probe(EXPECTS_OFF_THREAD, 0.0);
    let noisy = rows
        .iter()
        .find(|r| r.label == EXPECTS_OFF_THREAD)
        .expect("probe row is missing");
    assert!(
        noisy.off_thread > 0,
        "{EXPECTS_OFF_THREAD}: the competing thread made no allocation inside the window, \
         so the thread-local isolation was never actually exercised"
    );

    if bless {
        write_baseline(&rows);
        return;
    }
    if gate && !drifted.is_empty() {
        panic!(
            "CEL_ALLOCS_GATE=1 and {} row(s) drifted from the baseline: {:?}",
            drifted.len(),
            drifted
        );
    }
    if gate && !unstable.is_empty() {
        panic!(
            "CEL_ALLOCS_GATE=1 and {} row(s) were not stable across {ROUNDS} rounds: {:?}",
            unstable.len(),
            unstable
        );
    }
}
