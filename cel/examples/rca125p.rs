//! #125 — price the guard exit PER EXIT, not per call.
//!
//! ## Why this exists: the two-window route is structurally dead
//!
//! `rca125s` was built to difference a guard-failing call against a quiet one.
//! Its own output refutes that design at both n = 5 and n = 10: **no value of
//! `bridges_compiled` holds both a guard-failing and a quiet call.** That is
//! #91's law — `bridges_compiled = guard_failures / trace_eagerness` — so the
//! bridge count and the guard-failure rate are the SAME AXIS. No window can
//! hold one fixed while the other varies, at any n, with any warmup. The
//! confound lives in the arithmetic that defines the counters, so no amount of
//! instrument care removes it.
//!
//! The escape is to change the **unit of comparison**. A call is a bad unit
//! because its exit mix is tied to the bridge regime. An **exit** is a good one:
//! a single call in the n = 5 transient contains one entry that FINISHES and one
//! that GUARD-FAILS, so both arms occur inside one call, one regime, one
//! artifact.
//!
//! ## The model, and what identifies it
//!
//! ```text
//! allocs(call) = F · n_finish_exits + G · n_guard_exits + C
//! ```
//!
//! `F` and `G` are per-exit costs and `C` is the per-call constant. #125's
//! recorded `GUARD = 42` is the claim `G − F = 42`.
//!
//! This is NOT a two-window difference: `(n_finish, n_guard)` varies call to
//! call across the run, so the three parameters are identified by a fit over
//! hundreds of calls rather than by subtracting two hand-picked windows.
//!
//! ⭐ And the fit can TEST the thing the two-window route could only assume.
//! Adding `bridges_compiled` as a fourth regressor asks whether the bridge count
//! costs anything **beyond** its effect on the exit mix. If #125's model is
//! complete, that coefficient is ~0. The data answers it; I do not have to.
//!
//! ## ⛔⛔ THE INSTRUMENT PERTURBS THE QUANTITY — hence two arms
//!
//! Exit kinds are only observable through `PYRE_PORTAL_RCA=1`, which makes
//! `jitdriver.rs:4364-4375` print per entry — and `format_rca_live_values`
//! builds a `String`, i.e. **the observation channel allocates, on the very path
//! being priced**. Measuring both in one run would count the instrument's own
//! allocations as the program's, which is the mistake `rca125s` already made
//! once with `Backtrace::force_capture` (290 counted against 65 recorded).
//!
//! So:
//!
//! * **Arm A** (`PYRE_PORTAL_RCA` unset) — allocations per call, unperturbed.
//! * **Arm B** (`PYRE_PORTAL_RCA=1`) — exit kinds per call, allocations ignored.
//!
//! joined offline by call index. ⚠ That join assumes the call → exit-kind
//! mapping is the same in both arms. It is an assumption, so it is CHECKED, not
//! asserted: run arm B twice and diff the per-call exit sequences. The probe
//! prints the sequence in a form built for that diff. If the two disagree the
//! join is void and the run must be discarded — a real possibility, since arm B
//! allocates more and allocation can move a compilation threshold.
//!
//! ⭐ Arm B also prints its own allocation counts. They are NOT the measurement,
//! they are the size of the perturbation — printing them keeps "the instrument
//! is expensive" a number instead of an intuition.
//!
//! ## Running it
//!
//! ```text
//! # arm A — the measurement
//! cargo run --release --features jit-cranelift --example rca125p \
//!   > /tmp/a.txt 2>&1
//! # arm B — the exit stream (twice, for the determinism gate)
//! PYRE_PORTAL_RCA=1 cargo run --release --features jit-cranelift \
//!   --example rca125p > /tmp/b1.txt 2>&1
//! PYRE_PORTAL_RCA=1 cargo run --release --features jit-cranelift \
//!   --example rca125p > /tmp/b2.txt 2>&1
//! ```
//!
//! ⛔ `--features jit` alone does NOT work and must not be used: `jit` names no
//! backend on purpose and `majit-metainterp` refuses it (`scripts/
//! feature-matrix.sh:167` asserts that refusal). A backendless build fails with
//! ten unsized-`[TerminalExitLayout]` errors in `pyjitpl.rs` that read exactly
//! like a broken tree.
//!
//! `RCA125P_N`, `RCA125P_THRESHOLD` and `RCA125P_CALLS` override the defaults.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

std::thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

#[inline]
fn bump() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{key}={v:?}: expected an integer"))
        })
        .unwrap_or(default)
}

fn lower(src: &str, schema: &Schema) -> LoweredF {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("parse `{src}`: {e:?}"));
    lower_typed(program.expression(), schema).unwrap_or_else(|e| panic!("lower_typed `{src}`: {e}"))
}

fn flat_schema() -> Schema {
    [
        ("price".to_string(), ValType::Int),
        ("qty".to_string(), ValType::Int),
    ]
    .into_iter()
    .collect()
}

fn flat_columns(rows: usize) -> (Vec<i64>, Vec<i64>) {
    (
        (0..rows as i64).map(|i| (i * 37) % 200).collect(),
        (0..rows as i64).map(|i| (i * 11) % 100).collect(),
    )
}

fn main() {
    let n = env_usize("RCA125P_N", 5);
    // ⚠ 8 because `rca125s` and #125 use 8, and this is the JIT HOTNESS
    // threshold (`JitDriver::new(threshold)`; `u32::MAX` selects the interpreter
    // tier), not a value in the expression. It decides when tracing starts and
    // therefore where every regime boundary in this run lands. A different
    // threshold is a DIFFERENT FIXTURE whose numbers cannot be compared to the
    // measurements this probe exists to settle.
    let threshold = env_usize("RCA125P_THRESHOLD", 8) as u32;
    let calls = env_usize("RCA125P_CALLS", 700);
    let portal = std::env::var_os("PYRE_PORTAL_RCA").is_some();

    let schema = flat_schema();
    // ⛔⛔ THE EXPRESSION IS PART OF THE FIXTURE AND F/G DEPEND ON IT.
    // `rca128` publishes F=23 G=65 C=4 (cranelift) for ITS shape, which is
    // `price + qty * 2` (rca128.rs:175) — two operations. `rca125s` and this
    // probe default to `price * qty`, one operation. Quoting one fixture's
    // `G - F` against the other's is a fixture mismatch, not a correction — so
    // the expression is settable and is printed in the config line.
    let expr = std::env::var("RCA125P_EXPR").unwrap_or_else(|_| "price * qty".to_string());
    let lowered = lower(&expr, &schema);
    let (price, qty) = flat_columns(n);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];

    reset_persistent_state();
    reset_jit_stats();

    // Everything goes to stderr so that in arm B the call markers interleave
    // with majit's own `[portal-rca]` lines in ONE stream. Reconstructing the
    // interleaving from two separate streams would be a guess about buffering.
    eprintln!(
        "[rca125p][config] n={n} threshold={threshold} calls={calls} expr={expr:?} \
         arm={} portal_rca={portal}",
        if portal {
            "B (exit kinds)"
        } else {
            "A (allocations)"
        }
    );

    for k in 1..=calls {
        ALLOCS.with(|c| c.set(0));
        black_box(eval_batch_sum_f(&lowered, &columns, n, threshold));
        let allocs = ALLOCS.with(Cell::get);
        let s = jit_stats();
        // Printed AFTER the call, so this call's `[portal-rca][compiled-exit]`
        // lines are the ones between the previous marker and this one.
        eprintln!(
            "[rca125p][call] k={k} allocs={allocs} bridges={} gf={}",
            s.bridges_compiled, s.guard_failures
        );
    }

    let s = jit_stats();
    eprintln!(
        "[rca125p][done] loops_compiled={} bridges_compiled={} guard_failures={} \
         loops_aborted={}",
        s.loops_compiled, s.bridges_compiled, s.guard_failures, s.loops_aborted
    );
    if portal {
        eprintln!(
            "[rca125p][note] arm B allocation counts include the portal-rca \
             printing itself and are NOT the measurement; they are the size of \
             the perturbation. Use arm A for allocs, arm B for exit kinds."
        );
    }
}
