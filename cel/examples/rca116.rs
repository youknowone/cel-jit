//! #116 — is the `regvm/jit/*/n=1000` allocation figure COMPILE-time or RUN-time?
//!
//! `tests/allocs_per_eval.rs` reports 66/66/68 allocations per evaluation on
//! cranelift and 63/63/65 on dynasm, against one baseline whose key does not
//! name the backend. Widening the key is only correct if the divergence is
//! compile-side; if it is run-side, the same trace allocating differently on two
//! backends is a finding and keying on backend would hide it forever.
//!
//! The test cannot answer it. Its own comment says why:
//!
//! > "The counters span the warm-up AND the measured windows — `bench` owns the
//! > warm-up, so there is no seam to reset at."
//!
//! and its note column prints `loops_compiled`, `loops_aborted` and
//! `guard_failures` but **not** `bridges_compiled`, so a bridge compiled inside
//! the measured window is compile-time work the note cannot see.
//!
//! This probe builds the seam and reads every counter across it.
//!
//! ## Phase 1 — a window ladder along calls-since-compile
//!
//! Warm 64 calls (the test's `WARMUP`), then meter consecutive windows of 8
//! calls (its `ITERS`), printing allocations per call and the delta of every
//! `JitStats` field for each window. A per-call cost that is really amortized
//! compile work must fall as the window index grows; one that is steady-state
//! run-time must not move at all.
//!
//! It deliberately runs far enough to cross call 200, where #122 measured the
//! first guard bridge. That crossing is this probe's **positive control**: if
//! `bridges_compiled` moves there, the instrument can see compile-side activity,
//! so a zero everywhere else means "none happened" rather than "cannot see it".
//!
//! ## Phase 2 — window-length invariance
//!
//! From a fresh tier each time, warm 64 and meter ONE window of length L for
//! L = 1, 2, 4, 8, 16, 32, 64. A one-off cost divided by L shrinks as L grows;
//! a per-call cost does not. This is the positive form of the same question, and
//! it does not depend on any counter being correct.
//!
//! ⚠ `bridges_compiled` reaches `WarmEnterState::get_stats`, which #117 reports
//! reads chain HEADS only. An undercount can only hide bridges, so a NONZERO
//! reading here is trustworthy and a zero is not conclusive on its own — which
//! is exactly why phase 2 exists.
//!
//! Run both legs; the whole point is the pair:
//!
//! ```text
//! cargo run --release --features jit-cranelift --example rca116
//! cargo run --release --features jit-dynasm    --example rca116
//! ```

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

use cel::majit::bytecode::float_bank::{
    clean_interp_seeded_f, jit_stats, reset_jit_stats, reset_persistent_state,
    run_jit_persistent_f, JitStats,
};
use cel::majit::lower::{lower_typed, Schema, ValType};
use cel::Program;

// ---------------------------------------------------------------------------
// the meter — same semantics as tests/allocs_per_eval.rs, so the numbers here
// are comparable to the baseline rows rather than merely similar to them
// ---------------------------------------------------------------------------

std::thread_local! {
    static LOCAL_ALLOCS: Cell<u64> = const { Cell::new(0) };
}
static GLOBAL_ALLOCS: AtomicU64 = AtomicU64::new(0);

struct Counting;

#[inline]
fn bump() {
    GLOBAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
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
    // Forwarded rather than left to the default alloc+copy+dealloc, so
    // `System`'s in-place growth survives and one realloc counts as one event.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

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

/// Every field of [`JitStats`], as a delta. Printing the whole struct rather
/// than the three fields the test's note column carries is most of the point:
/// `bridges_compiled` is the one that decides this question and it is the one
/// the note omits.
struct Delta {
    loops: i64,
    bridges: i64,
    aborted: i64,
    gfails: i64,
    panics: i64,
}

impl Delta {
    fn between(before: &JitStats, after: &JitStats) -> Delta {
        Delta {
            loops: after.loops_compiled as i64 - before.loops_compiled as i64,
            bridges: after.bridges_compiled as i64 - before.bridges_compiled as i64,
            aborted: after.loops_aborted as i64 - before.loops_aborted as i64,
            gfails: after.guard_failures as i64 - before.guard_failures as i64,
            panics: after.internal_compile_panics as i64 - before.internal_compile_panics as i64,
        }
    }
    /// True when nothing on the COMPILE side moved. Guard failures are excluded
    /// deliberately: a guard failure is the artifact running, not the tier
    /// compiling.
    fn compile_side_quiet(&self) -> bool {
        self.loops == 0 && self.bridges == 0 && self.aborted == 0 && self.panics == 0
    }
    fn render(&self) -> String {
        format!(
            "loops{:+} bridges{:+} aborted{:+} gfails{:+} panics{:+}",
            self.loops, self.bridges, self.aborted, self.gfails, self.panics
        )
    }
}

struct Case {
    label: &'static str,
    src: &'static str,
    schema: &'static [(&'static str, ValType)],
}

/// The three shapes `regvm/*` carries, verbatim from `tests/allocs_per_eval.rs`
/// so this probe measures that table's rows and not a paraphrase of them.
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

const JIT_ON: u32 = 8;
const WARMUP: u32 = 64;
const ITERS: u32 = 8;
/// The one correctness call `arm` makes before the warm-up. Counted, because
/// #122 places the first guard bridge at call 200 exactly and a call index that
/// is off by one is a call index that cannot be compared to that.
const PRIMED: u32 = 1;
/// Far enough past call 200 that the first guard bridge is inside the ladder
/// with room on both sides of it.
const WINDOWS: u32 = 48;
const N: usize = 1000;

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

/// The compiled program plus its live columns. The columns must outlive every
/// call: their base addresses are seeded into the register bank.
struct Fixture {
    /// An `Arc`, not a `Vec`: the `#[jit_interp]` green key is the program
    /// POINTER, so holding the refcount keeps every call filed under the same
    /// identity the compiled loop was compiled for.
    code: std::sync::Arc<[i64]>,
    regs: Vec<i64>,
    nf: usize,
    expected: i64,
    _cols: Vec<Col>,
}

fn build(case: &Case) -> Fixture {
    let program = Program::compile(case.src).unwrap_or_else(|e| panic!("{}: {e:?}", case.label));
    let schema: Schema = case
        .schema
        .iter()
        .map(|(n, t)| (n.to_string(), *t))
        .collect();
    let lowered = lower_typed(program.expression(), &schema)
        .unwrap_or_else(|e| panic!("{}: lower_typed: {e}", case.label));
    // The lowering's OWN slot order. The schema is a `HashMap`, so an order
    // taken from the literal above would be whatever the hasher produced.
    let cols: Vec<Col> = lowered
        .slots
        .iter()
        .map(|slot| match slot.ty {
            ValType::Float => Col::Float((0..N).map(|k| (k % 97) as f64 * 0.5).collect()),
            _ => Col::Int((0..N as i64).map(|k| (k * 7) % 97).collect()),
        })
        .collect();
    let bases: Vec<i64> = cols.iter().map(Col::base).collect();

    // Priced to settle a cross-probe discrepancy, and the answer REFUTED the
    // reason for pricing it. `rca88b`'s Probe L meters `eval_batch_sum_f`, which
    // goes through the batch-assembly door; this probe and the `regvm/jit/*`
    // baseline rows assemble once and then meter `run_jit_persistent_f` alone.
    // The door was the obvious candidate for the 3-allocation gap between the
    // two probes' absolute figures (69/66 there, 66/63 here).
    //
    // It is not: the door costs 9, and 9 on BOTH backends. The probes also run
    // different programs, so their absolute per-call figures were never
    // comparable in the first place. What IS comparable, and what survives, is
    // the cranelift-minus-dynasm difference INSIDE each probe: 3 there, 3 here,
    // on three cases and two harness shapes.
    let meter = Meter::start();
    let (shape, regs) = lowered.batch_sum_program(&bases, N as i64);
    let nf = shape.num_float_regs;
    let code = shape.code.clone();
    let (assembly, _) = meter.stop();
    println!(
        "   batch_sum_program + Arc clone for {}/n={N}: {assembly} allocations",
        case.label
    );

    let expected = clean_interp_seeded_f(&code, &regs, nf);
    Fixture {
        code,
        regs,
        nf,
        expected,
        _cols: cols,
    }
}

impl Fixture {
    #[inline]
    fn call(&self) {
        black_box(run_jit_persistent_f(
            &self.code, &self.regs, self.nf, JIT_ON,
        ));
    }
    /// Fresh tier, one correctness call, `WARMUP` calls, counters zeroed at the
    /// seam. Returns `(after the priming call, at the seam)`.
    ///
    /// The first reading is taken BEFORE `reset_jit_stats`, and it is what
    /// closes two holes a delta cannot:
    ///
    /// * `loops_compiled >= 1` there is the POSITIVE form of "the artifact
    ///   exists". Without it, `loops+0` in every measured window is equally
    ///   consistent with "compiled earlier, running now" and "never compiled at
    ///   all" — the same counter reading from two opposite causes.
    /// * `trace_ops_*` there is the compiled trace's size. Two backends
    ///   allocating differently is only a divergence on ONE trace if the trace
    ///   is in fact one trace, and that has to be measured, not assumed.
    fn arm(&self) -> (JitStats, JitStats) {
        reset_persistent_state();
        let got = run_jit_persistent_f(&self.code, &self.regs, self.nf, JIT_ON);
        assert_eq!(
            got, self.expected,
            "compiled tier diverged from the clean VM"
        );
        let primed = jit_stats();
        reset_jit_stats();
        for _ in 0..WARMUP {
            self.call();
        }
        // THE SEAM the test does not have: everything before this line is
        // warm-up, everything after is measurement, and the counters are read
        // here so the two are attributable separately.
        (primed, jit_stats())
    }
}

fn phase1(case: &Case, fx: &Fixture) {
    println!("\n## phase 1 — window ladder, {}/n={N}", case.label);
    println!(
        "   {PRIMED} priming call + {WARMUP} warm-up calls, then {WINDOWS} windows of {ITERS};\
         \n   call indices are cumulative since reset_persistent_state()"
    );
    let (primed, at_seam) = fx.arm();
    println!(
        "   after the priming call: loops={} bridges={} aborted={} gfails={} panics={} \
         trace_ops {}->{}   (loops>=1 is what says the artifact EXISTS)",
        primed.loops_compiled,
        primed.bridges_compiled,
        primed.loops_aborted,
        primed.guard_failures,
        primed.internal_compile_panics,
        primed.trace_ops_before,
        primed.trace_ops_after
    );
    // Absolute, not a delta: `reset_jit_stats` cannot zero the live drivers'
    // own tallies, only the module's counters, so this is the warm-up's
    // compile-side record plus whatever the priming call left in a driver.
    println!(
        "   at the seam (call {}): loops={} bridges={} aborted={} gfails={} panics={}",
        PRIMED + WARMUP,
        at_seam.loops_compiled,
        at_seam.bridges_compiled,
        at_seam.loops_aborted,
        at_seam.guard_failures,
        at_seam.internal_compile_panics
    );
    let mut prev = at_seam;
    let mut baseline_per_call: Option<f64> = None;
    for w in 0..WINDOWS {
        let first_call = PRIMED + WARMUP + w * ITERS;
        let meter = Meter::start();
        for _ in 0..ITERS {
            fx.call();
        }
        let (local, off_thread) = meter.stop();
        let now = jit_stats();
        let d = Delta::between(&prev, &now);
        prev = now;
        let per_call = local as f64 / ITERS as f64;
        let base = *baseline_per_call.get_or_insert(per_call);
        let mark = if !d.compile_side_quiet() {
            "  <== COMPILE-SIDE ACTIVITY"
        } else if (per_call - base).abs() > f64::EPSILON {
            "  <== allocs/call MOVED with the compile side quiet"
        } else {
            ""
        };
        println!(
            "   calls {first_call:>4}..{:<4} allocs/call {per_call:>9.3}  off_thread {off_thread:>3}  {}{mark}",
            first_call + ITERS,
            d.render()
        );
    }
}

fn phase2(case: &Case, fx: &Fixture) {
    println!(
        "\n## phase 2 — window-length invariance, {}/n={N}",
        case.label
    );
    println!("   a one-off cost divided by L shrinks as L grows; a per-call cost does not");
    for l in [1u32, 2, 4, 8, 16, 32, 64] {
        let (_, before) = fx.arm();
        let meter = Meter::start();
        for _ in 0..l {
            fx.call();
        }
        let (local, off_thread) = meter.stop();
        let d = Delta::between(&before, &jit_stats());
        println!(
            "   L={l:>3}  total {local:>7}  allocs/call {:>9.3}  off_thread {off_thread:>3}  {}{}",
            local as f64 / l as f64,
            d.render(),
            if d.compile_side_quiet() {
                ""
            } else {
                "  <== COMPILE-SIDE ACTIVITY inside the window"
            }
        );
    }
}

fn main() {
    let backend = if cfg!(feature = "jit-cranelift") {
        "cranelift"
    } else if cfg!(feature = "jit-dynasm") {
        "dynasm"
    } else {
        "unknown"
    };
    println!("rca116: backend = {backend}");
    println!(
        "baseline rows under test: regvm/jit/{{arith,policy,float}}/n=1000 \
         = 66/66/68 (cranelift), 63/63/65 (dynasm)"
    );

    for case in CASES {
        let fx = build(case);
        phase1(case, &fx);
        phase2(case, &fx);
        reset_persistent_state();
    }
    // A terminal line, so a run that died partway is distinguishable from one
    // that finished. Cardinality, not exit status.
    println!(
        "\nrca116: DONE — {} cases x (1 ladder of {WINDOWS} windows + 7 lengths)",
        CASES.len()
    );
}
