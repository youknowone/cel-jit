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
//! One `n` and one artifact, two regimes separated only by guard failures:
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
    for k in 1..=settled_at {
        if k == gf_at {
            samples.push(capture_one("gf", k, &lowered, &columns, n, threshold));
        } else if k == settled_at {
            samples.push(capture_one("settled", k, &lowered, &columns, n, threshold));
        } else {
            black_box(eval_batch_sum_f(&lowered, &columns, n, threshold));
        }
    }

    println!("\n{:>9} {:>6} {:>9} {:>8} {:>8} {:>9}", "regime", "call", "gf_delta", "bridges", "allocs", "captured");
    for s in &samples {
        println!(
            "{:>9} {:>6} {:>9} {:>8} {:>8} {:>9}",
            s.label, s.call, s.gf_delta, s.bridges, s.allocs, s.captured
        );
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

    println!("\nper-size, one call each ({} gf vs {} settled)", gf_arm.label, settled_arm.label);
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
    println!("{:>8} {:>8} {:>9} {:>8}", "net", gf_arm.allocs, settled_arm.allocs, net);

    // ── The sites behind every bucket that MOVED ──
    println!("\n=== SITES for every size whose count differs ===");
    for (size, (a, b)) in &sizes {
        if a == b {
            continue;
        }
        println!("\n--- size {size} B:  gf={a}  settled={b}  delta={} ---", *a as i64 - *b as i64);
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
