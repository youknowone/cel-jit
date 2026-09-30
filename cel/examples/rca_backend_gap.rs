//! Per-site attribution of the `regvm/jit/*/n=1000` backend gap.
//!
//! `tests/allocs_per_eval.jit.baseline` records `regvm/jit/*/n=1000` at 65/65/67
//! allocations per call on cranelift and 63/63/65 on dynasm, and its header left
//! the two-allocation difference open when this probe was written. ⛔ Both of
//! those recorded figures are now stale and the difference is gone — see
//! `## Observed` below, which is also what that header now records. Everything
//! that bounded the search is already recorded there and is NOT re-derived here:
//!
//! * the gap is RUN-time — the loop compiles before the window opens, both
//!   backends compile the same trace, and per-call allocations are invariant
//!   over window lengths 1..64 and linear in call count, so ONE warm call
//!   already costs the full 65 (or 63);
//! * the gap is uniform across `arith`/`policy`/`float`, so it is not a
//!   property of one program's shape;
//! * ⚠ the gap was 3 and is now 2, and only ONE leg moved — "majit: return a
//!   back-edge FINISH from the portal instead of resuming at the back edge"
//!   took cranelift 66/66/68 -> 65/65/67 and left dynasm at 63/63/65.
//!
//! So this probe does not measure the gap again. It reproduces ONE call of the
//! exact fixture behind the `regvm/jit/arith/n=1000` row and records the STACK
//! of every allocation that call makes, so the two backends' tables can be
//! differenced site by site.
//!
//! ## Two regimes over one fixture — `RCAGAP_REGIME`
//!
//! The baseline carries two row families over this fixture and they measure
//! different windows, so a per-site table has to say which one it attributes.
//!
//! * `warm` (the default) is the window `regvm/jit/*/n=1000` records: warmed
//!   past the loop compile and stopped SHORT of the first guard bridge. Its
//!   invariant is `bridges_compiled == 0` at the seam, and a bridge landing
//!   inside the warm-up refuses the run.
//! * `steady` is the window `regvm/jit-steady/*/n=1000` records: warmed until
//!   nothing is left to compile and no guard is left to fail. Its warm-up is
//!   SEARCHED, not chosen — calls run in chunks of `RCAGAP_SETTLE_CHUNK` until
//!   a whole chunk moves none of `loops_compiled`, `bridges_compiled`,
//!   `loops_aborted`, `guard_failures`, and the length it took is printed — and
//!   its invariant is that all four stand still across the capture window too,
//!   with at least one compiled entry per call in that window. Any of them
//!   moving refuses the run, exactly as the warm assert refuses a bridge.
//!
//! Everything else is shared and neither regime weakens it: the same fixture,
//! the same door, the same re-entrancy guard, the same `captured == allocs`
//! check, and the same unarmed calls either side of the armed ones.
//!
//! ## The fixture is copied, not approximated
//!
//! `tests/allocs_per_eval.rs:640-760` is the definition: the case list, the
//! `JIT_ON = 8` threshold, the column construction in the LOWERING's slot order
//! (the schema is a `HashMap`, so a literal order is the hasher's), the
//! `batch_sum_program` shape, the `Arc<Code>` clone that keeps the green key's
//! address stable, and `WARMUP = 64` calls before the window. A probe that
//! rebuilt any of those "equivalently" would be measuring a different program
//! and its per-site table would attribute a different number.
//!
//! ⚠ `run_jit_persistent_f` is the door the row measures. `eval_batch_sum_f` —
//! which `rca125p`/`rca125s` use — is a DIFFERENT door with its own per-call
//! bookkeeping, so its per-site table cannot be quoted against this row.
//!
//! ## ⛔⛔ The instrument perturbs the quantity — the guard, and the check
//!
//! `Backtrace::force_capture` allocates, and those allocations arrive straight
//! back in the allocator hook. `rca125s` counted 290 against a recorded 65 the
//! first time. The guard here is `rca125s`'s: `IN_CAPTURE` goes up BEFORE the
//! capture and `bump` returns early while it is up, so an instrument allocation
//! is neither counted nor recorded and `captured == allocs` holds by
//! construction.
//!
//! That makes the two-arm offline join of `rca125p` unnecessary for THIS
//! question, and the check is stronger than a join: the armed call's own
//! metered count sits in the same run and the same regime as the unarmed calls
//! either side of it, so the probe prints the unarmed tail and the armed calls
//! together. If the armed call's total differs from its neighbours', the
//! capture moved the quantity and the table below it is void.
//!
//! ## Running it
//!
//! ```text
//! cargo run --release --no-default-features \
//!   --features regex,chrono,jit-cranelift --example rca_backend_gap
//! cargo run --release --no-default-features \
//!   --features regex,chrono,jit-dynasm --example rca_backend_gap
//! RCAGAP_REGIME=steady cargo run --release --no-default-features \
//!   --features regex,chrono,jit-cranelift --example rca_backend_gap
//! ```
//!
//! ⛔ `--features jit` alone names no backend and `majit-metainterp` refuses it.
//! ⚠ Build with `CARGO_PROFILE_RELEASE_DEBUG=true` or the frames symbolize to
//! addresses. The baseline header records dev and release as identical on every
//! row, so either profile answers the question; release is the one the baseline
//! was taken in.
//!
//! `RCAGAP_REGIME` (`warm`/`steady`), `RCAGAP_CASE` (`arith`/`policy`/`float`),
//! `RCAGAP_N`, `RCAGAP_WARMUP`, `RCAGAP_CAPTURES`, `RCAGAP_TAIL`,
//! `RCAGAP_SETTLE_CHUNK` and `RCAGAP_SETTLE_MAX` override the defaults.
//! `RCAGAP_WARMUP` under `steady` replaces the settle search with a count the
//! caller vouches for; the window invariant still judges where it landed.
//!
//! ## Observed, 2026-08-18, aarch64-macos — THE GAP IS ZERO AND THE BASELINE IS STALE
//!
//! In the `warm` regime, which is the default. The `steady` regime's own table
//! is the section after this one.
//!
//! There were no two allocations to attribute. Both legs cost **52.000** per
//! call on all three cases, against a file that records 65/65/67 (cranelift)
//! and 63/63/65 (dynasm). The gate itself says so — `CEL_ALLOCS_GATE` unset,
//! nothing blessed:
//!
//! ```text
//! regvm/jit/arith/n=1000           52.000  base 65.000  -13.000  (both legs)
//! regvm/jit/policy/n=1000          52.000  base 65.000  -13.000  (both legs)
//! regvm/jit/float/n=1000           52.000  base 67.000  -15.000  (both legs)
//! regvm/jit-steady/arith/n=1000     1.000  base 24.000  -23.000  (both legs)
//! regvm/jit-steady/policy/n=1000    1.000  base 24.000  -23.000  (both legs)
//! regvm/jit-steady/float/n=1000     1.000  base 26.000  -25.000  (both legs)
//! regvm/jit/float/n=1               1.000  base  2.000   -1.000  (both legs)
//! ```
//!
//! and this probe agrees on a different instrument, a different call sequence
//! and no `bench` harness, in BOTH profiles (`dev` and `release` read 52).
//!
//! Those seven rows are the whole drift: 82 printed, 70 compared against the
//! file, 7 moved, all downward and by the same amount on each leg.
//!
//! ⚠ THE ROW THIS PROBE REPRODUCES IS NOT THE ONE THAT MOVED MOST.
//! `regvm/jit-steady/*/n=1000` — which the baseline header designates as the row
//! that measures the artifact's steady state, the `regvm/jit/*` rows being a
//! pre-bridge window — fell 24/24/26 -> 1.000. Its regime notes are unchanged:
//! `bridges_before=1 in_window=0 compiled_in_window=0 guard_fails=0 — STEADY:
//! past every bridge`, so in that regime, with no guard failure and no compile
//! in the window, a call now costs ONE allocation — and `RCAGAP_REGIME=steady`
//! reproduces that window and says below which one. The `regvm/jit/*/n=1000`
//! rows still report `compiled=0 bridges=0 aborted=0 guard_fails=88` over 88
//! calls, one guard failure per call exactly as before, so what fell there is
//! the COST of that call and not the number of them.
//!
//! ⭐ AND THE TWO LEGS' REPORTS ARE BYTE-IDENTICAL, not merely equal on the rows
//! above: `diff` of the two 82-row reports returns nothing. There is no row in
//! that corpus on which the backends disagree.
//!
//! The per-site tables are the finding, not the totals. 45 sites on each leg,
//! summing to 52 on each leg — ⚠ re-measured on 2026-08-18 the same table reads
//! 42 sites, still summing to 52, on both legs. The 42 is not this file's doing:
//! the probe as committed at `7555818`, built and run unmodified on the same
//! tree, prints the same 42 rows and the same total. Sites merged under majit;
//! the quantity did not move. Grouped by innermost majit/cel frame they are
//! **identical, row for row, count for count**, except ONE row — and that row
//! has count 1 on both sides:
//!
//! ```text
//! cranelift  1  size=224  majit_backend_cranelift::compiler::run_compiled_code_inner
//! dynasm     1  size=320  majit_backend::jitframe::alloc_off_gc_jitframe
//! ```
//!
//! i.e. each backend allocates exactly one jitframe per compiled entry and
//! nothing else that the other does not. `stacks passing through:` reports
//! `backend-cranelift=1` / `backend-dynasm=1` against `metainterp=52`, so 51 of
//! the 52 are decided above the backend split and cannot differ by backend at
//! all.
//!
//! ⚠ THE EQUALITY IS ARM-DEPENDENT, exactly as the baseline header warns — but
//! this build's arm is CONFIRMED, not assumed. `run_compiled_code_inner`
//! branches on `cranelift_jitframe_type_id()` at
//! `majit-backend-cranelift/src/compiler.rs:7827`; the `use_gc_alloc == false`
//! arm runs `:7842`-`:7859` and allocates `vec![0i64; HEADER_WORDS + jf_total]`
//! at `:7849`. The captured frame symbolizes to `:7849` — inside that arm — so
//! the frame this build measures is a heap `Vec<i64>` and therefore visible to a
//! global-allocator counter. On a build where the GC type registry answers, that
//! frame is nursery-allocated and invisible, cranelift reads 51 against dynasm's
//! 52, and the one differing row above becomes a differing TOTAL. Establish
//! which arm a re-measurement is on before recording a difference.
//!
//! ⚠ The four line numbers in that paragraph are the ones majit carried when it
//! was written, and majit has moved since: on 2026-08-18 the branch is `:7832`,
//! the arm runs `:7849`-`:7866`, the `vec![0i64; HEADER_WORDS + jf_total]` is
//! `:7856`, and `:7856` is what the frame symbolizes to. The FACT — that this
//! build takes the non-GC arm — is re-confirmed on both regimes; the
//! coordinates are not stable, so re-check by symbol.
//!
//! ⛔⛔ WHICH majit TREE THESE NUMBERS WERE TAKEN ON, because `.cargo/config.toml`
//! patches the majit crates to `../majit` and that is a worktree several agents
//! share. At the time of this reading the enclosing worktree was in DETACHED
//! HEAD on a peer branch with 47 staged files — not this crate's own branch and
//! not a clean upstream/main. The figures are therefore reproductions across
//! trees rather than readings of one: the warm total read 52 over 45 sites on
//! an earlier, different tree and reads 52 over 45 sites here, and the steady
//! total reads 1 here and matched the gate's 1.000 taken on that earlier tree.
//! Agreement across two unrelated trees is evidence the numbers are a property
//! of the code; it is NOT the same as a reading on a clean checkout, and no
//! reading on a clean checkout has been taken. Do not bless a baseline against
//! these until one has been.
//!
//! ## Observed, 2026-08-18, aarch64-macos — THE STEADY REGIME, MEASURED
//!
//! `RCAGAP_REGIME=steady`, `arith`, release, both legs, and the two legs agree
//! on every line of it except the site itself.
//!
//! The warm-up SETTLED after 320 calls — five chunks of 64, the last moving
//! nothing — at `loops_compiled=1 bridges_compiled=1 guard_failures=200
//! loops_aborted=0`. The capture window then moved none of those four and
//! entered compiled code 6 times over its 6 calls, so the window is past every
//! bridge and every call in it ran the artifact. Every call in the run's steady
//! stretch, armed or not, cost ONE allocation: `warm(last 3)=[1, 1, 1]
//! armed=[1, 1, 1] tail=[1, 1, 1]`.
//!
//! ⭐ THAT ONE ALLOCATION IS THE JITFRAME, AND THIS IS THE MEASUREMENT, not the
//! inference from the warm window that preceded it. The whole table is one row:
//!
//! ```text
//! cranelift  1  size=224  majit_backend_cranelift::compiler::run_compiled_code_inner@compiler.rs:7856
//! dynasm     1  size=320  majit_backend::jitframe::alloc_off_gc_jitframe@jitframe.rs:230
//! ```
//!
//! ⭐ Those are the SAME two rows that are the ONLY difference between the two
//! legs' warm tables. The steady table is therefore not an arbitrary subset of
//! the warm one: what survives past every bridge is exactly the row on which the
//! backends disagree, and the 51 the legs share in the warm window are all
//! pre-bridge cost that the steady regime does not pay. `stacks passing
//! through:` reads `backend-cranelift=1 backend=1 metainterp=1 cel=1` (dynasm:
//! `backend-dynasm=1 ...`) — one stack, and it reaches the backend, against a
//! warm window where 51 of 52 stacks stop above the backend split.
//!
//! ⚠ The arm-dependence above is the whole steady total here, not one row of
//! it: on a build whose GC type registry answers, cranelift's jitframe is
//! nursery-allocated and invisible to a global-allocator counter, and this
//! regime would read 0 against dynasm's 1.
//!
//! ⛔ WHICH CHANGE CLOSED THE GAP IS NOT ATTRIBUTED HERE. Separating candidates
//! needs old trees: `.cargo/config.toml` patches the majit crates to `../majit`,
//! so pricing an older majit means building an older enclosing worktree with
//! this checkout inside it. Nothing in this file measures that, and no commit is
//! named as the cause. `allocs_per_eval.jit.baseline`'s header records the
//! window a bisect would start from — later than the `14ea0e7` bless
//! (2026-08-08) that this file's `base` column comes from — together with what
//! the shape of the drop already rules out. ⛔ Read it there rather than
//! assuming a commit range: that window has an enumerable term and an
//! unbounded one, because the majit crates are built out of a worktree several
//! agents write to.

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::hint::black_box;

use cel::majit::bytecode::float_bank::{
    jit_stats, reset_jit_stats, reset_persistent_state, run_jit_persistent_f, JitStats,
};
use cel::majit::lower::{lower_typed, Schema, ValType};
use cel::Program;

std::thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    /// Off except inside a captured call: a backtrace per allocation costs far
    /// more than anything being measured, and every other call here is warmup.
    static SITE_ALL: Cell<bool> = const { Cell::new(false) };
    /// Re-entrancy guard. `Backtrace::force_capture` allocates, and those
    /// allocations arrive straight back in `bump`.
    static IN_CAPTURE: Cell<bool> = const { Cell::new(false) };
    static SITES: RefCell<Vec<(usize, Backtrace)>> = const { RefCell::new(Vec::new()) };
}

struct Counting;

/// Records the stack that asked for `size`.
///
/// `#[cold]` and out of line: every allocation in the process tests the arm
/// even though only a handful of calls in the run ever take this path.
#[cold]
fn record_site(size: usize) {
    if IN_CAPTURE.try_with(|c| c.replace(true)).unwrap_or(true) {
        return;
    }
    // Captured now, symbolized later: `Display` on a `Backtrace` allocates far
    // more than the capture, and none of it belongs on this path.
    let bt = Backtrace::force_capture();
    let _ = SITES.try_with(|s| s.borrow_mut().push((size, bt)));
    let _ = IN_CAPTURE.try_with(|c| c.set(false));
}

#[inline]
fn bump(size: usize) {
    // The instrument's own allocations belong to the INSTRUMENT. Counting them
    // while recording no stack for them makes `captured == allocs`
    // unsatisfiable and fires the integrity check on the probe rather than on
    // the subject.
    if IN_CAPTURE.try_with(Cell::get).unwrap_or(false) {
        return;
    }
    // Counted before the arm is tested, so the metered total and the captured
    // total are the same population by construction.
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
    if SITE_ALL.try_with(Cell::get).unwrap_or(false) {
        record_site(size);
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn env_usize_opt(key: &str) -> Option<usize> {
    std::env::var(key).ok().map(|v| {
        v.parse()
            .unwrap_or_else(|_| panic!("{key}={v:?}: expected an integer"))
    })
}

fn env_usize(key: &str, default: usize) -> usize {
    env_usize_opt(key).unwrap_or(default)
}

/// Which window on the fixture the probe measures.
///
/// The two are the same program, the same door and the same instrument; they
/// differ only in where the capture window sits relative to the tier's
/// compiles, and that placement is what each regime's invariant asserts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Regime {
    /// The pre-bridge window `regvm/jit/*/n=1000` records: warmed past the loop
    /// compile and stopped short of the first guard bridge.
    Warm,
    /// The post-bridge window `regvm/jit-steady/*/n=1000` records: warmed until
    /// nothing is left to compile and no guard is left to fail.
    Steady,
}

impl Regime {
    fn label(self) -> &'static str {
        match self {
            Regime::Warm => "WARM",
            Regime::Steady => "STEADY",
        }
    }
}

/// The counters whose stillness DEFINES the steady regime: a compile, a bridge,
/// an aborted trace and a deopt are the four events that would put the window on
/// the other side of a tier transition. `trace_ops_*` and `compiled_entries` are
/// deliberately absent — they move on every steady call by design, so including
/// them would make "settled" unreachable.
fn regime_counters(s: &JitStats) -> (usize, usize, usize, usize) {
    (
        s.loops_compiled,
        s.bridges_compiled,
        s.loops_aborted,
        s.guard_failures,
    )
}

/// The three cases of `tests/allocs_per_eval.rs:640-657`, verbatim.
struct Case {
    label: &'static str,
    src: &'static str,
    schema: &'static [(&'static str, ValType)],
}

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

/// A column of the batch, kept alive for the whole measurement so the base
/// addresses seeded into the register bank stay valid.
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

/// The innermost majit/cel frames of one capture, as a diffable one-liner.
///
/// ⚠ The filtering is what makes the two backends' tables comparable, so it is
/// spelled out rather than tuned by eye. `alloc::`/`core::`/`std::` frames are
/// dropped even though they mention majit through their generic parameters —
/// `<alloc::vec::Vec<majit_ir::..>>::clone` is the container, not the site, and
/// keeping it spends the frame budget on six spellings of `to_vec`. What is
/// left is the majit/cel code that ASKED, plus each kept frame's own
/// `file:line` where the symbolizer has one.
fn interesting_frames(bt: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;
    for line in bt.lines().map(str::trim) {
        if let Some(loc) = line.strip_prefix("at ") {
            // Attaches to the frame above it, and only if that frame was kept.
            if let Some(frame) = pending.take() {
                let loc = loc.rsplit('/').next().unwrap_or(loc);
                out.push(format!("{frame}@{loc}"));
            }
            continue;
        }
        // `NN: symbol`
        let Some((_, sym)) = line.split_once(": ") else {
            continue;
        };
        if let Some(frame) = pending.take() {
            out.push(frame);
        }
        if out.len() >= 6 {
            break;
        }
        let bare = sym.trim_start_matches('<');
        let is_plumbing = ["alloc::", "core::", "std::"]
            .iter()
            .any(|p| bare.starts_with(p));
        let is_probe = sym.contains("rca_backend_gap")
            || sym.contains("record_site")
            || sym.contains("::bump");
        if is_plumbing || is_probe || !(sym.contains("majit") || sym.contains("cel")) {
            continue;
        }
        pending = Some(sym.replace("::<alloc::alloc::Global>", ""));
    }
    if let Some(frame) = pending {
        out.push(frame);
    }
    out.join(" <- ")
}

fn backend() -> &'static str {
    // Both can be on at once, in which case `majit-metainterp` picks and the
    // configuration is not one to measure from. Say so rather than name one.
    match (
        cfg!(feature = "jit-cranelift"),
        cfg!(feature = "jit-dynasm"),
    ) {
        (true, false) => "cranelift",
        (false, true) => "dynasm",
        (true, true) => "BOTH (not a configuration to measure from)",
        (false, false) => "none declared by cel's features",
    }
}

/// Which crates a capture's FULL stack passes through.
///
/// Computed on the raw backtrace, not on the filtered one-liner: a backend
/// frame six levels out is still a backend frame, and the frame budget that
/// makes the site column readable would hide it. This is the column that
/// discriminates the two legs — a per-call cost that no `majit-backend-*` frame
/// reaches cannot be where the backends differ.
const CRATES: &[(&str, &str)] = &[
    ("backend-cranelift", "majit_backend_cranelift"),
    ("backend-dynasm", "majit_backend_dynasm"),
    ("backend", "majit_backend::"),
    ("metainterp", "majit_metainterp"),
    ("translate", "majit_translate"),
    ("majit-ir", "majit_ir"),
    ("majit-gc", "majit_gc"),
    ("cel", "cel::"),
];

struct Sample {
    call: usize,
    allocs: u64,
    captured: usize,
    /// `(size, stack) -> count`, the call's allocation multiset with sites.
    groups: BTreeMap<(usize, String), usize>,
    /// `crate -> allocations whose stack passes through it`. Sums to more than
    /// `captured`: one stack crosses several crates.
    crates: BTreeMap<&'static str, usize>,
}

fn main() {
    let regime_name = std::env::var("RCAGAP_REGIME").unwrap_or_else(|_| "warm".to_string());
    let regime = match regime_name.as_str() {
        "warm" => Regime::Warm,
        "steady" => Regime::Steady,
        other => panic!("RCAGAP_REGIME={other:?}: expected warm or steady"),
    };
    let case_name = std::env::var("RCAGAP_CASE").unwrap_or_else(|_| "arith".to_string());
    let case = CASES
        .iter()
        .find(|c| c.label == case_name)
        .unwrap_or_else(|| panic!("RCAGAP_CASE={case_name:?}: expected arith, policy or float"));
    let n = env_usize("RCAGAP_N", 1000);
    // `tests/allocs_per_eval.rs:628`. Not a value in the expression: it decides
    // when tracing starts and therefore where every regime boundary lands, so a
    // different threshold is a different fixture.
    const JIT_ON: u32 = 8;
    // `WARMUP` in the test. One priming call precedes it there too, so the
    // first captured call in the warm regime is the test's call 66.
    const WARM_WARMUP: usize = 64;
    // Overrides the regime's own plan in both regimes: in `warm` it is the
    // window's position, in `steady` it replaces the settle search with a
    // number the caller vouches for. The steady invariant still refuses the
    // run if that number lands short of the regime.
    let warmup_override = env_usize_opt("RCAGAP_WARMUP");
    let captures = env_usize("RCAGAP_CAPTURES", 3);
    let tail = env_usize("RCAGAP_TAIL", 3);
    // The steady warm-up is MEASURED, not chosen: calls run in chunks until a
    // whole chunk moves none of the four counters. `RCAGAP_SETTLE_MAX` bounds
    // the search so a fixture that never settles refuses the run instead of
    // spinning.
    let settle_chunk = env_usize("RCAGAP_SETTLE_CHUNK", 64);
    let settle_max = env_usize("RCAGAP_SETTLE_MAX", 8192);
    assert!(
        settle_chunk > 0,
        "RCAGAP_SETTLE_CHUNK=0: a chunk of no calls can never observe a change"
    );

    // `None` is the only way to reach the settle search, so the warm regime —
    // whose window position IS the fixed number — never takes it.
    let fixed_warmup = match regime {
        Regime::Warm => Some(warmup_override.unwrap_or(WARM_WARMUP)),
        Regime::Steady => warmup_override,
    };
    let warmup_plan = match fixed_warmup {
        Some(w) if regime == Regime::Warm => format!("warmup={w}"),
        Some(w) => format!("warmup={w} (fixed)"),
        None => format!("warmup=settle(chunk={settle_chunk},max={settle_max})"),
    };
    println!(
        "rca_backend_gap — per-site attribution of one {} call\n\
         backend={} profile={} case={} src={:?} n={n} threshold={JIT_ON} \
         {warmup_plan} captures={captures} tail={tail}",
        regime.label(),
        backend(),
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
        case.label,
        case.src,
    );

    let program = Program::compile(case.src).unwrap_or_else(|e| panic!("{}: {e:?}", case.label));
    let schema: Schema = case
        .schema
        .iter()
        .map(|(name, ty)| (name.to_string(), *ty))
        .collect();
    let lowered = lower_typed(program.expression(), &schema)
        .unwrap_or_else(|e| panic!("{}: lower_typed: {e}", case.label));

    // In the LOWERING's slot order, as the test builds them.
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
    // The `#[jit_interp]` green key is the program POINTER, so what matters is
    // that this is the same allocation on every call.
    let code = shape.code.clone();

    reset_persistent_state();
    reset_jit_stats();

    // The test's priming call, which is where the correctness assert sits.
    let expected = black_box(run_jit_persistent_f(&code, &regs, nf, JIT_ON));

    // One metered, unarmed call. Both regimes' warm-ups are made of these, and
    // both read the totals back out of `warm_counts` for the join check.
    let metered_call = |counts: &mut Vec<u64>| {
        ALLOCS.with(|c| c.set(0));
        let got = black_box(run_jit_persistent_f(&code, &regs, nf, JIT_ON));
        counts.push(ALLOCS.with(Cell::get));
        assert_eq!(
            got, expected,
            "compiled tier diverged from the priming call"
        );
    };

    let mut warm_counts: Vec<u64> = Vec::new();
    let mut settle_note = String::new();
    let warmup = match fixed_warmup {
        Some(warmup) => {
            for _ in 0..warmup {
                metered_call(&mut warm_counts);
            }
            warmup
        }
        // The steady window is defined by what has ALREADY happened, so its
        // warm-up is a search and not a constant: call in chunks until a whole
        // chunk moves none of the four counters, and report the length it took
        // rather than asserting a number chosen by hand.
        None => loop {
            let before = regime_counters(&jit_stats());
            for _ in 0..settle_chunk {
                metered_call(&mut warm_counts);
            }
            let (compiled, bridges, aborted, guard_fails) = regime_counters(&jit_stats());
            if (compiled, bridges, aborted, guard_fails) == before {
                settle_note = format!(
                    "settled after {} calls: the last {settle_chunk} moved none of \
                     loops_compiled/bridges_compiled/loops_aborted/guard_failures",
                    warm_counts.len()
                );
                break warm_counts.len();
            }
            assert!(
                warm_counts.len() < settle_max,
                "the tier's counters were still moving after {} warm calls \
                 (loops_compiled={compiled} bridges_compiled={bridges} loops_aborted={aborted} \
                 guard_failures={guard_fails}): the steady regime was never reached within \
                 RCAGAP_SETTLE_MAX={settle_max}, so there is no steady call here to attribute",
                warm_counts.len(),
            );
        },
    };

    let s = jit_stats();
    println!(
        "\nafter {} calls (1 prime + {warmup} warm): loops_compiled={} bridges_compiled={} \
         guard_failures={} loops_aborted={}",
        warmup + 1,
        s.loops_compiled,
        s.bridges_compiled,
        s.guard_failures,
        s.loops_aborted,
    );
    match regime {
        // The row this probe attributes is blessed as PRE-BRIDGE. If a bridge
        // has landed by here the window is a different regime and the table
        // below attributes a cost the baseline row does not carry.
        Regime::Warm => assert_eq!(
            s.bridges_compiled, 0,
            "a guard bridge landed inside the warm-up: this is no longer the regime \
             `regvm/jit/{}/n={n}` records",
            case.label
        ),
        // The steady regime's invariant is over the CAPTURE WINDOW and is
        // checked once that window has closed, so what belongs here is only
        // where the warm-up stopped and why. A caller-supplied warm-up leaves
        // no note; the invariant still judges where it landed.
        Regime::Steady => {
            if !settle_note.is_empty() {
                println!("{settle_note}");
            }
        }
    }
    println!(
        "last 8 unarmed warm calls: {:?}",
        &warm_counts[warm_counts.len().saturating_sub(8)..]
    );

    // ── The armed calls ────────────────────────────────────────────────────
    let mut samples: Vec<Sample> = Vec::new();
    for i in 0..captures {
        let call = warmup + 2 + i;
        SITES.with(|s| s.borrow_mut().clear());
        ALLOCS.with(|c| c.set(0));
        SITE_ALL.with(|c| c.set(true));
        let got = black_box(run_jit_persistent_f(&code, &regs, nf, JIT_ON));
        SITE_ALL.with(|c| c.set(false));
        let allocs = ALLOCS.with(Cell::get);
        assert_eq!(got, expected, "compiled tier diverged under capture");

        // Symbolization happens here, outside the allocator, and allocates
        // heavily.
        let captured_raw = SITES.with(|s| std::mem::take(&mut *s.borrow_mut()));
        let mut groups: BTreeMap<(usize, String), usize> = BTreeMap::new();
        let mut crates: BTreeMap<&'static str, usize> = BTreeMap::new();
        for (size, bt) in &captured_raw {
            let full = format!("{bt}");
            for (name, needle) in CRATES {
                if full.contains(needle) {
                    *crates.entry(*name).or_insert(0) += 1;
                }
            }
            *groups
                .entry((*size, interesting_frames(&full)))
                .or_insert(0) += 1;
        }
        samples.push(Sample {
            call,
            allocs,
            captured: captured_raw.len(),
            groups,
            crates,
        });
    }

    // ── The unarmed tail: the same regime, unperturbed ─────────────────────
    let mut tail_counts: Vec<u64> = Vec::new();
    for _ in 0..tail {
        ALLOCS.with(|c| c.set(0));
        black_box(run_jit_persistent_f(&code, &regs, nf, JIT_ON));
        tail_counts.push(ALLOCS.with(Cell::get));
    }
    println!("unarmed tail after the captures: {tail_counts:?}");

    // ── The steady invariant: the window crossed no tier transition ────────
    //
    // `s` was read at the seam, so these deltas span every armed call and the
    // whole unarmed tail — the same span the table below is built from.
    if regime == Regime::Steady {
        let after = jit_stats();
        let window_calls = captures + tail;
        let compiled = after.loops_compiled - s.loops_compiled;
        let bridges = after.bridges_compiled - s.bridges_compiled;
        let aborted = after.loops_aborted - s.loops_aborted;
        let guard_fails = after.guard_failures - s.guard_failures;
        let entered = after.compiled_entries - s.compiled_entries;
        println!(
            "steady window ({captures} armed + {tail} unarmed = {window_calls} calls): \
             bridges_before={} compiled_in_window={compiled} bridges_in_window={bridges} \
             aborted_in_window={aborted} guard_fails={guard_fails} entries_in_window={entered} \
             — {}",
            s.bridges_compiled,
            if s.bridges_compiled == 0 {
                "NO BRIDGE EVER COMPILED — nothing measured here is post-bridge"
            } else {
                "STEADY: past every bridge"
            }
        );
        // The steady counterpart of the warm regime's pre-bridge assert, and
        // just as load-bearing: a compile, a bridge, an aborted trace or a
        // deopt inside the window puts part of a tier transition into the
        // table, which is the one thing this regime exists to exclude.
        assert_eq!(
            (compiled, bridges, aborted, guard_fails),
            (0, 0, 0, 0),
            "(loops_compiled, bridges_compiled, loops_aborted, guard_failures) moved inside the \
             capture window: this is no longer the regime `regvm/jit-steady/{}/n={n}` records",
            case.label
        );
        // Anti-vacuity. Counters also stand still when nothing is running, and
        // a window of calls that never entered compiled code would attribute
        // the interpreter under a steady label.
        assert!(
            entered >= window_calls,
            "{entered} compiled entries over {window_calls} calls in the window: a call did not \
             enter compiled code, so the table below mixes the tiers"
        );
    }

    for sample in &samples {
        println!(
            "\n=== call {} — allocs={} captured={} ===",
            sample.call, sample.allocs, sample.captured
        );
        // ⛔ The one check that makes the table below quotable. If the capture
        // counted or missed anything, the multiset is not the call's.
        assert_eq!(
            sample.allocs as usize, sample.captured,
            "call {}: {} metered against {} captured — the instrument is inside \
             the measurement and this table is void",
            sample.call, sample.allocs, sample.captured
        );
        let crates: Vec<String> = CRATES
            .iter()
            .filter_map(|(name, _)| {
                sample
                    .crates
                    .get(name)
                    .map(|count| format!("{name}={count}"))
            })
            .collect();
        println!("  stacks passing through: {}", crates.join(" "));
        let mut rows: Vec<(&(usize, String), &usize)> = sample.groups.iter().collect();
        rows.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for ((size, stack), count) in rows {
            println!("  {count:4}  size={size:<6} {stack}");
        }
    }

    // The neighbours either side of the armed calls, printed together so the
    // join is checkable at a glance rather than asserted.
    let armed: Vec<u64> = samples.iter().map(|s| s.allocs).collect();
    println!(
        "\njoin check — warm(last 3)={:?} armed={armed:?} tail={tail_counts:?}",
        &warm_counts[warm_counts.len().saturating_sub(3)..]
    );
}
