//! #125 NEXT 3 — WHICH SITE does the guard exit stop allocating?
//!
//! #125 measured the guard-failure exit as a multiset delta of **+45 added, −3
//! removed, net 42**, byte-identical on both backends and by two independent
//! routes. The three removed are `1B x2` and `16B x1`. A transition that
//! *deletes* allocations is a stronger structural clue than one that adds them,
//! so this probe answers the removal half by SITE.
//!
//! ## What it does
//!
//! One `n` and one artifact, two windows that were INTENDED to differ only in
//! guard failures (see the section below — they do not, and the probe says so):
//!
//! * a call taken while `guard_failures` is still stepping **1 per call**, and
//! * a call taken after it has **frozen** (`gf` delta 0),
//!
//! with **every** allocation of each captured call recorded by stack. Grouping
//! by `(size, stack)` and differencing the two gives the guard-exit multiset
//! with a code site attached to every bucket.
//!
//! ## Why capture everything rather than arm the three sizes
//!
//! Arming the allocator on `{1, 16}` would make a zero reading ambiguous: it
//! could mean "the site is gone" (the finding) or "the arming stopped matching"
//! (a broken instrument). Capturing the whole call cannot fail that way — the
//! probe asserts the number of captured stacks equals the independently
//! metered allocation count for the same call, so a miss is visible as a
//! mismatch rather than as a clean zero. Affordable because a call here is a
//! few hundred allocations, not a few million.
//!
//! ## ⛔⛔ AND THE TWO-WINDOW ROUTE CANNOT ISOLATE GUARD — measured, not assumed
//!
//! The probe was built to difference a guard-failing call against a quiet one.
//! Its own trajectory output refutes that design: at **both** n = 5 and n = 10
//! there is no value of `bridges_compiled` holding both a guard-failing and a
//! quiet call. That is not a sampling accident, it is #91's law —
//! `bridges_compiled = guard_failures / trace_eagerness` — so the bridge count
//! and the guard-failure rate are the SAME AXIS: the bridge is compiled *in
//! response to* the failures and its arrival is what ends them. Any delta this
//! probe prints is therefore `guard exit + one bridge`, never guard exit alone.
//!
//! Three predicates were needed before that showed up, each removing a different
//! class of false positive, and every correction moved toward NOT separable:
//!
//! 1. "some call has bridges ≥ 1 and a failure" — tests the wrong thing entirely
//!    (one call, not a pair sharing a `bridges` value).
//! 2. flags per `bridges` value — a bridges value carries "failing" evidence of
//!    exactly **1**, and that 1 is the TRANSITION call, which is precisely the
//!    call that differs in bridges. Counts, not flags.
//! 3. counts — n = 5 then showed 99 failing / 2 quiet at `bridges = 0`, and the
//!    two quiet calls were **calls 1 and 2**: the pre-compilation head, quiet
//!    because no artifact exists (E = 0). Differencing against those measures
//!    "artifact vs no artifact". Hence the `seen_failing` gate.
//!
//! ⭐ A count alone could not catch (3); printing the call INDICES could. When a
//! cell's evidence is 2 out of 101, print which ones.
//!
//! The route that can settle `GUARD` prices **exits**, not calls (rca128's
//! `[portal-rca][compiled-entry]`/`[compiled-exit]` channel): one call in the
//! n = 5 transient contains both a finishing entry and a guard-failing one, so
//! the comparison lives INSIDE a single call and needs no second regime.
//!
//! ## ⛔ The regimes are OBSERVED, not assumed
//!
//! The call indices below are a starting guess. The probe reads the
//! `guard_failures` delta **of the captured call itself** and REFUSES if the
//! two samples do not bracket the transition. A window that does not contain
//! the regime it claims is the error #125, #127, #141 and #150 all made; the
//! detector is cheaper than the retraction.
//!
//! ## Running it
//!
//! Frames symbolize to bare addresses without debug info, and this workspace
//! declares no `[profile.release]`:
//!
//! ```text
//! CARGO_PROFILE_RELEASE_DEBUG=2 \
//!   cargo run --release --features jit-cranelift --example rca125s
//! ```
//!
//! ⛔ **`--features jit` alone does not work and must not be used.** `jit` names
//! no backend on purpose, so `majit-metainterp` refuses it (`scripts/
//! feature-matrix.sh:167` asserts that refusal). The backend-selecting spellings
//! are `jit-cranelift` and `jit-dynasm`. This is not a footnote: a build without
//! a backend fails with ten unrelated-looking `[TerminalExitLayout] cannot be
//! known at compile-time` errors in `pyjitpl.rs`, which read exactly like a
//! broken tree. Cf. #125's own rule that a window must be the one you claim —
//! the same applies to a build configuration.
//!
//! `RCA125S_N`, `RCA125S_THRESHOLD`, `RCA125S_GF_AT` and `RCA125S_SETTLED_AT`
//! override the defaults.

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

std::thread_local! {
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
    /// Off except inside a captured call: a backtrace per allocation costs far
    /// more than anything being measured, and every other call here is warmup.
    static SITE_ALL: Cell<bool> = const { Cell::new(false) };
    /// Re-entrancy guard. `Backtrace::force_capture` allocates, and those
    /// allocations arrive straight back in `bump`; without this the first
    /// capture recurses until the stack ends.
    static IN_CAPTURE: Cell<bool> = const { Cell::new(false) };
    static SITES: RefCell<Vec<(usize, Backtrace)>> = const { RefCell::new(Vec::new()) };
}

struct Counting;

/// Records the stack that asked for `size`.
///
/// Deliberately `#[cold]` and out of line: every allocation in the process
/// tests the arm even though only two calls in the run ever take this path.
#[cold]
fn record_site(size: usize) {
    // The guard goes up BEFORE the capture, because the capture's own
    // allocations re-enter here and would otherwise be recorded as sites of
    // themselves — and would also inflate the count this probe checks against
    // the metered total.
    if IN_CAPTURE.try_with(|c| c.replace(true)).unwrap_or(true) {
        return;
    }
    // Captured now, symbolized later: `Display` on a `Backtrace` allocates a
    // great deal more than the capture, and none of it belongs on this path.
    let bt = Backtrace::force_capture();
    let _ = SITES.try_with(|s| s.borrow_mut().push((size, bt)));
    let _ = IN_CAPTURE.try_with(|c| c.set(false));
}

#[inline]
fn bump(size: usize) {
    // ⛔ `Backtrace::force_capture` allocates heavily, and those allocations
    // arrive straight back here. They belong to the INSTRUMENT, not to the
    // program, so they must not be counted either — counting them while
    // recording no stack for them makes `captured == allocs` unsatisfiable and
    // the integrity check fires on the probe rather than on the subject. (It
    // did: 290 counted against 65 recorded on the first run.)
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

/// Keep only frames inside majit/cel, dropping the probe's own allocator hook.
fn interesting_frames(bt: &str) -> String {
    bt.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("at ") || l.contains("::"))
        .filter(|l| {
            (l.contains("majit") || l.contains("cel"))
                && !l.contains("rca125s")
                && !l.contains("record_site")
                && !l.contains("bump")
        })
        .take(14)
        .collect::<Vec<_>>()
        .join("\n")
}

struct Sample {
    label: &'static str,
    call: usize,
    gf_delta: usize,
    bridges: usize,
    allocs: u64,
    captured: usize,
    /// `(size, stack) -> count`, the call's allocation multiset with sites.
    groups: BTreeMap<(usize, String), usize>,
}

/// Run exactly one call with every allocation's stack recorded.
fn capture_one(
    label: &'static str,
    call: usize,
    lowered: &LoweredF,
    columns: &[Column],
    n: usize,
    threshold: u32,
) -> Sample {
    let gf_before = jit_stats().guard_failures;
    let allocs_before = ALLOCS.with(Cell::get);

    SITES.with(|s| s.borrow_mut().clear());
    SITE_ALL.with(|c| c.set(true));
    black_box(eval_batch_sum_f(lowered, columns, n, threshold));
    SITE_ALL.with(|c| c.set(false));

    let allocs = ALLOCS.with(Cell::get) - allocs_before;
    let stats = jit_stats();
    let captured_raw = SITES.with(|s| std::mem::take(&mut *s.borrow_mut()));

    // Symbolization happens here, outside the allocator, and allocates heavily.
    let mut groups: BTreeMap<(usize, String), usize> = BTreeMap::new();
    for (size, bt) in &captured_raw {
        *groups
            .entry((*size, interesting_frames(&format!("{bt}"))))
            .or_insert(0) += 1;
    }

    Sample {
        label,
        call,
        gf_delta: stats.guard_failures - gf_before,
        bridges: stats.bridges_compiled,
        allocs,
        captured: captured_raw.len(),
        groups,
    }
}

fn main() {
    let n = env_usize("RCA125S_N", 10);
    let threshold = env_usize("RCA125S_THRESHOLD", 8) as u32;
    let gf_at = env_usize("RCA125S_GF_AT", 150);
    let settled_at = env_usize("RCA125S_SETTLED_AT", 500);
    assert!(
        gf_at < settled_at,
        "RCA125S_GF_AT ({gf_at}) must precede RCA125S_SETTLED_AT ({settled_at})"
    );

    println!(
        "rca125s — guard-exit site attribution:  n={n} threshold={threshold} \
         gf_at={gf_at} settled_at={settled_at}"
    );

    let schema = flat_schema();
    let lowered = lower("price * qty", &schema);
    let (price, qty) = flat_columns(n);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];

    reset_persistent_state();
    reset_jit_stats();

    let mut samples: Vec<Sample> = Vec::new();
    // ── Counter trajectory: LOCATE the regimes rather than quote them ──
    // #150 measured that the settle point is not monotone in n and that a
    // task's recorded exposure set can move across its own fix, so any onset
    // taken from an older run is re-derived here rather than trusted.
    let mut prev = (0usize, 0usize);
    let mut bridge_events: Vec<(usize, usize)> = Vec::new();
    let mut last_gf_change = 0usize;
    // For each observed `bridges` value, did we see a call that failed a guard
    // and a call that did not? Both true for some value ⇒ a pair exists that
    // differs in guard failures ALONE. ⛔ "some call has bridges>=1 and gf>0"
    // is NOT that test — it was the first predicate here and it printed a
    // cheerful SEPARABLE for a run in which every bridge transition coincided
    // with the guard-failure freeze.
    // ⚠ COUNTS, not flags: the call at which a bridge appears is itself usually
    // a guard-failing call, so a bridges value can show "failing" evidence of
    // exactly 1 and that 1 is the TRANSITION call. Differencing against it
    // reintroduces the bridge confound the test exists to exclude. A count of 1
    // is a transition artefact; a regime is many.
    let mut by_bridges: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    // ⚠ WHICH calls are quiet matters as much as how many. A quiet call in the
    // PRE-COMPILATION head is quiet because no artifact exists yet (E=0), not
    // because a guard did not fail — differencing against it measures "artifact
    // vs no artifact", which is not this probe's question. Printing the indices
    // is what distinguishes the two; a count alone cannot.
    let mut quiet_calls: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    // Gates the quiet count on an artifact existing. Until the first guard
    // failure the program has not entered compiled code, so every call is
    // trivially "quiet" and none of them is a control.
    let mut seen_failing = false;
    for k in 1..=settled_at {
        if k == gf_at {
            samples.push(capture_one("gf", k, &lowered, &columns, n, threshold));
        } else if k == settled_at {
            samples.push(capture_one("settled", k, &lowered, &columns, n, threshold));
        } else {
            black_box(eval_batch_sum_f(&lowered, &columns, n, threshold));
        }
        let s = jit_stats();
        let now = (s.bridges_compiled, s.guard_failures);
        if now.0 != prev.0 {
            bridge_events.push((k, now.0));
        }
        if now.1 != prev.1 {
            last_gf_change = k;
        }
        let slot = by_bridges.entry(now.0).or_insert((0, 0));
        if now.1 != prev.1 {
            slot.0 += 1;
            seen_failing = true;
        } else if seen_failing {
            // ⛔ Only AFTER the first guard failure. A quiet call before it is
            // quiet because no compiled artifact exists yet, not because a
            // guard did not fail.
            slot.1 += 1;
            quiet_calls.entry(now.0).or_default().push(k);
        }
        prev = now;
    }

    println!(
        "\n{:>9} {:>6} {:>9} {:>8} {:>8} {:>9}",
        "regime", "call", "gf_delta", "bridges", "allocs", "captured"
    );
    for s in &samples {
        println!(
            "{:>9} {:>6} {:>9} {:>8} {:>8} {:>9}",
            s.label, s.call, s.gf_delta, s.bridges, s.allocs, s.captured
        );
    }

    // ── Where the regimes actually are, on THIS binary ──
    println!("\ncounter trajectory (re-derived on this run, not quoted from a task body)");
    println!("  bridge transitions (call, bridges): {bridge_events:?}");
    println!("  last call at which guard_failures moved: {last_gf_change}");
    println!("  bridges -> (guard-failing calls, quiet calls):");
    for (b, (failing, quiet)) in &by_bridges {
        let note = if *failing == 1 {
            "   <== 1 failing call = the TRANSITION call itself, not a regime"
        } else {
            ""
        };
        let empty = Vec::new();
        let qs = quiet_calls.get(b).unwrap_or(&empty);
        let head: Vec<String> = qs.iter().take(4).map(|c| c.to_string()).collect();
        let ell = if qs.len() > 4 { ", …" } else { "" };
        println!(
            "    bridges={b}: failing={failing} quiet={quiet} quiet at calls [{}{}]{note}",
            head.join(", "),
            ell
        );
    }
    match by_bridges.iter().find(|(_, (f, q))| *f > 1 && *q > 1) {
        Some((b, (f, q))) => println!(
            "  ✅ SEPARABLE at n={n}: with bridges={b} held FIXED there are {f} guard-failing and \
             {q} quiet calls — both regimes, neither a lone transition — so a pair differing in \
             guard failures ALONE exists. Capture inside bridges={b} to isolate GUARD."
        ),
        None => println!(
            "  ⛔ NOT SEPARABLE at n={n}: no bridges value has both a guard-failing and a quiet \
             call. Every bridge transition coincides with a change in the guard-failure rate, \
             which is what `bridges_compiled = guard_failures / trace_eagerness` predicts — the \
             bridge is compiled IN RESPONSE to the failures and its arrival is what ends them. \
             So any delta measured here is guard-exit PLUS a bridge, and isolating GUARD needs \
             E measured per regime (Probe P's route), not a two-window difference."
        ),
    }

    // ── Integrity: the captured population IS the metered population ──
    // A capture that silently dropped stacks would make every "removed" bucket
    // below indistinguishable from a site that stopped being recorded.
    for s in &samples {
        assert_eq!(
            s.captured as u64, s.allocs,
            "{}: captured {} stacks but metered {} allocations — the capture \
             dropped some, so no bucket below can be trusted",
            s.label, s.captured, s.allocs
        );
    }

    // ── The windows must actually be the regimes claimed ──
    let (gf_arm, settled_arm) = (&samples[0], &samples[1]);
    assert!(
        gf_arm.gf_delta >= 1,
        "call {} was supposed to sit in the guard-failing regime but its own \
         gf delta is 0 — move RCA125S_GF_AT earlier; a window that does not \
         contain its regime cannot difference it",
        gf_arm.call
    );
    assert_eq!(
        settled_arm.gf_delta, 0,
        "call {} was supposed to sit past the guard-failure freeze but its own \
         gf delta is {} — move RCA125S_SETTLED_AT later",
        settled_arm.call, settled_arm.gf_delta
    );

    // ── Per-size difference ──
    let mut sizes: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
    for ((size, _), c) in &gf_arm.groups {
        sizes.entry(*size).or_insert((0, 0)).0 += c;
    }
    for ((size, _), c) in &settled_arm.groups {
        sizes.entry(*size).or_insert((0, 0)).1 += c;
    }

    println!(
        "\nper-size, one call each ({} gf vs {} settled)",
        gf_arm.label, settled_arm.label
    );
    println!("{:>8} {:>8} {:>9} {:>8}", "size", "gf", "settled", "delta");
    let mut net: i64 = 0;
    for (size, (a, b)) in &sizes {
        let d = *a as i64 - *b as i64;
        net += d;
        let mark = if d == 0 {
            ""
        } else if d < 0 {
            "  <== REMOVED by the guard exit"
        } else {
            "  <== added"
        };
        println!("{size:>8} {a:>8} {b:>9} {d:>8}{mark}");
    }
    println!(
        "{:>8} {:>8} {:>9} {:>8}",
        "net", gf_arm.allocs, settled_arm.allocs, net
    );

    // ── The sites behind every bucket that MOVED ──
    println!("\n=== SITES for every size whose count differs ===");
    for (size, (a, b)) in &sizes {
        if a == b {
            continue;
        }
        println!(
            "\n--- size {size} B:  gf={a}  settled={b}  delta={} ---",
            *a as i64 - *b as i64
        );
        for (arm, s) in [("gf", gf_arm), ("settled", settled_arm)] {
            for ((sz, frames), count) in &s.groups {
                if sz != size {
                    continue;
                }
                println!("\n  [{arm}] x{count}\n{frames}");
            }
        }
    }
}
