//! #156 — the pre-bridge floor step at ~call 141: program state or host state?
//!
//! ## The observation this exists to reproduce
//!
//! Probe J (`rca88b.rs:1403`) timed each call individually at n=10 and its
//! 20-call buckets showed the pre-bridge window splitting in two: the **floor**
//! (`min` over a bucket) moved 2833 → 3666 ns at ~call 141, a +29% step, and
//! never came back — with `bridges_compiled` at 0 across it and
//! `guard_failures` advancing at exactly 1 per call on both sides.
//!
//! A mean can be raised by a load spike. A floor over 20 calls cannot, and a
//! floor that never returns is not a spike at all. So the observation is
//! well-posed. What it is NOT is reproduced: run 2 of the same binary never
//! showed the cheap ~2833 regime at all, so it had no step to find.
//!
//! ## ⛔ Why Probe J could not have reported this itself
//!
//! Probe J's verdict line fires on `mean > 4.0 * baseline` (`:1442`). The step
//! here is **+29% on the floor**. The detector is off by an order of magnitude
//! AND reads the wrong statistic, so the boundary was found by a human reading
//! the table and could never have been found by the probe. That is the specific
//! defect this file repairs: **a detector sized for one step is blind to a
//! smaller one in the same table.**
//!
//! ## The two pre-registered outcomes
//!
//! The question is program state versus host state, and the discriminator is
//! where the step lands across repeated runs of one binary:
//!
//! * the step **pins** near a fixed call index across runs => PROGRAM state.
//!   Something in the JIT changes at a reproducible number of back edges.
//! * the step's index **moves** with the run => HOST state (thermal/DVFS, page
//!   or TLB warmth, allocator layout). #156 already records the coincidence
//!   that points this way: run 1's post-step floor (3666) and run 2's only
//!   floor (3791) agree to 3%, which is what "run 1 started in a cheap state
//!   run 2 never had" looks like.
//! * **no run shows a step** => the observation is a single-run artifact and
//!   #156 closes as not reproducible.
//!
//! All three are informative. Run this binary N times and compare the verdict
//! lines; one process is one sample and cannot answer the question alone.
//!
//! ## ⛔⛔ This probe has NO `#[global_allocator]`, deliberately
//!
//! It measures TIME. A counting allocator taxes every allocation on the path
//! being timed, and the quantity under study is an 833 ns floor move. Adding an
//! allocation counter here would perturb exactly the number this file exists to
//! locate.
//!
//! The allocation half of "widen the sampling" is already answered without it:
//! `rca125p`'s per-call log at n=10, threshold 8 reads **64 allocations per
//! call, flat, across calls 1-200** (then 23 after the bridge), so allocations
//! do not step at ~141. 64 = G + C = 60 + 4 also confirms one guard exit per
//! call, matching the `gfails` rate #156 reports. What was NOT sampled and is
//! sampled here is `loops_compiled`, `loops_aborted` and
//! `internal_compile_panics`.
//!
//! ## Instrument hygiene
//!
//! * `jit_stats()` is read AFTER `elapsed()`, so it is outside every timed
//!   region, and into storage reserved before the loop so the read allocates
//!   nothing.
//! * `RCA156_BUCKET_SAMPLING=1` reverts to Probe J's cadence (counters at
//!   bucket boundaries only). It is a CONTROL: if the step moves when the
//!   sampling changes, the sampling is the cause and nothing else here is
//!   trustworthy.
//! * The fixture defaults to Probe J's own — `price + qty * 2`, n=10,
//!   threshold 8, 460 calls — because an observation must be reproduced before
//!   it is explained.

use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// Window used both for the "before" floor and for the warmup skip.
const WINDOW: usize = 20;
/// A step must lift the floor by at least this fraction to be reported. The
/// observed step is +29%; 10% is deliberately loose so a weaker reproduction
/// still registers rather than being silently rounded away.
const MIN_GAIN: f64 = 0.10;

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

/// One sample of every counter `JitStats` carries, per call.
///
/// ⚠ This is `cel`'s `JitStats` (`cel/src/majit/bytecode.rs:2065`), not
/// `majit`'s (`pyjitpl.rs:1626`). Same name, different struct: cel's carries
/// `usize` panics and two op-count fields majit's does not have.
///
/// ⭐ `trace_ops_after` earns its place: #125 measured that the optimized
/// artifact is 13 ops when `(n−1)` divides the threshold and 25 otherwise, and
/// that two per-entry allocations are sized by it. If the op count moves
/// mid-run, the artifact was recompiled and the question is answered outright.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Marks {
    loops: usize,
    aborts: usize,
    bridges: usize,
    gfails: usize,
    panics: usize,
    ops_before: usize,
    ops_after: usize,
}

fn min_of(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::MAX, f64::min)
}

/// The floor-step detector.
///
/// A step is a boundary `b` such that **no call at or after `b` is as cheap as
/// the floor before it**. That persistence clause is the whole point: it is
/// what separates a regime change from a spike, and it is the property #156
/// identified as making the observation well-posed.
///
/// ⛔ The search is confined to calls strictly before the first bridge. Without
/// that, the detector finds the bridge step at ~201 — a known, larger, already
/// explained event — and reports it as the answer.
///
/// ⛔⛔ It looks for a step in EITHER DIRECTION, and that is not fastidiousness.
/// An up-only version was written first and its positive control REFUSED on 6
/// of 6 runs: post-#128 the bridge makes the artifact ~2.1x CHEAPER, so the one
/// step known to exist on this tree points DOWN. **A direction-signed detector
/// silently fails the control it was supposed to pass**, and had the control
/// been omitted the up-only null would have read as a clean answer.
fn find_floor_step(ns: &[f64], end: usize) -> Option<(usize, f64, f64, f64)> {
    let mut best: Option<(usize, f64, f64, f64)> = None;
    if end < 2 * WINDOW {
        return None;
    }
    for b in WINDOW..end.saturating_sub(WINDOW) {
        let lo_before = min_of(&ns[b - WINDOW..b]);
        if lo_before <= 0.0 {
            continue;
        }
        // Per-window floors after the candidate boundary. Persistence is a
        // statement about EVERY later window, not about the next one: that is
        // what separates a regime change from a spike.
        let after: Vec<f64> = ns[b..end].chunks(WINDOW).map(min_of).collect();
        let worst_up = after.iter().copied().fold(f64::MAX, f64::min);
        let worst_down = after.iter().copied().fold(0.0, f64::max);
        let gain = if worst_up >= lo_before * (1.0 + MIN_GAIN) {
            (worst_up - lo_before) / lo_before
        } else if worst_down <= lo_before * (1.0 - MIN_GAIN) {
            (worst_down - lo_before) / lo_before
        } else {
            continue;
        };
        let level = lo_before * (1.0 + gain);
        if best.is_none_or(|(_, _, _, g)| gain.abs() > g.abs()) {
            best = Some((b, lo_before, level, gain));
        }
    }
    best
}

fn main() {
    let n = env_usize("RCA156_N", 10);
    // Probe J's own threshold (`rca88b.rs:286`). This is the JIT hotness
    // threshold, not a value in the expression; changing it is a different
    // fixture whose call indices cannot be compared to #156's.
    let threshold = env_usize("RCA156_THRESHOLD", 8) as u32;
    let calls = env_usize("RCA156_CALLS", 460);
    let bucket_sampling = std::env::var_os("RCA156_BUCKET_SAMPLING").is_some();
    let dump_rows = std::env::var_os("RCA156_ROWS").is_some();
    // Probe J's shape (`rca128.rs:175`), not `rca125p`'s `price * qty`.
    let expr = std::env::var("RCA156_EXPR").unwrap_or_else(|_| "price + qty * 2".to_string());

    let schema = flat_schema();
    let lowered = lower(&expr, &schema);
    let (price, qty) = flat_columns(n);
    let columns = vec![Column::Int(&price), Column::Int(&qty)];

    println!(
        "[rca156][config] n={n} threshold={threshold} calls={calls} expr={expr:?} \
         bucket_sampling={bucket_sampling} window={WINDOW} min_gain={MIN_GAIN}"
    );

    reset_persistent_state();
    reset_jit_stats();

    let zero = Marks {
        loops: 0,
        aborts: 0,
        bridges: 0,
        gfails: 0,
        panics: 0,
        ops_before: 0,
        ops_after: 0,
    };
    // Reserved before the loop so that recording a sample allocates nothing.
    let mut ns: Vec<f64> = Vec::with_capacity(calls);
    let mut marks: Vec<Marks> = vec![zero; calls];

    let mut last = zero;
    for k in 0..calls {
        let t = std::time::Instant::now();
        black_box(eval_batch_sum_f(&lowered, &columns, n, threshold));
        ns.push(t.elapsed().as_nanos() as f64);
        // Read AFTER `elapsed`: never inside a timed region.
        if !bucket_sampling || (k + 1) % WINDOW == 0 {
            let s = jit_stats();
            last = Marks {
                loops: s.loops_compiled,
                aborts: s.loops_aborted,
                bridges: s.bridges_compiled,
                gfails: s.guard_failures,
                panics: s.internal_compile_panics,
                ops_before: s.trace_ops_before,
                ops_after: s.trace_ops_after,
            };
        }
        marks[k] = last;
    }

    // Every counter transition, with the call it happened at. This is the
    // "widen the sampling" item: `loops_compiled`, `loops_aborted` and
    // `internal_compile_panics` were never sampled by Probe J at all.
    println!("[rca156][transitions] call: loops/aborts/bridges/gfails/panics/trace-ops");
    let mut prev = zero;
    for (k, m) in marks.iter().enumerate() {
        // `gfails` advances every call by construction, so it is reported only
        // when one of the others moves — otherwise it prints 460 rows and
        // buries the events this list exists to surface.
        let structural_move = m.loops != prev.loops
            || m.aborts != prev.aborts
            || m.bridges != prev.bridges
            || m.panics != prev.panics
            || m.ops_before != prev.ops_before
            || m.ops_after != prev.ops_after;
        if structural_move {
            println!(
                "[rca156][transition] call={:<5} loops={} aborts={} bridges={} gfails={} \
                 panics={} ops={}->{}",
                k + 1,
                m.loops,
                m.aborts,
                m.bridges,
                m.gfails,
                m.panics,
                m.ops_before,
                m.ops_after
            );
        }
        prev = *m;
    }

    let prebridge_end = marks
        .iter()
        .position(|m| m.bridges > 0)
        .unwrap_or(marks.len());
    println!(
        "[rca156][window] pre-bridge window is calls 1..{} ({} calls); \
         the floor search is confined to it",
        prebridge_end,
        prebridge_end
    );

    if dump_rows {
        for (k, (t, m)) in ns.iter().zip(marks.iter()).enumerate() {
            eprintln!(
                "[rca156][call] k={:<5} ns={:<10.0} loops={} aborts={} bridges={} gfails={}",
                k + 1,
                t,
                m.loops,
                m.aborts,
                m.bridges,
                m.gfails
            );
        }
    }

    // Bucket table, Probe J's shape, so the two instruments index alike.
    println!(
        "[rca156][buckets] {:>11} {:>11} {:>11} {:>11} {:>6} {:>8}",
        "calls", "mean ns", "min ns", "max ns", "brdg", "gfails"
    );
    for (b, chunk) in ns.chunks(WINDOW).enumerate() {
        let mean = chunk.iter().sum::<f64>() / chunk.len() as f64;
        let lo = min_of(chunk);
        let hi = chunk.iter().copied().fold(0.0, f64::max);
        let m = marks[(b * WINDOW + chunk.len()).min(marks.len()) - 1];
        println!(
            "[rca156][bucket] {:>11} {mean:>11.1} {lo:>11.1} {hi:>11.1} {:>6} {:>8}",
            format!("{}-{}", b * WINDOW + 1, b * WINDOW + chunk.len()),
            m.bridges,
            m.gfails
        );
    }

    // ⛔⛔ POSITIVE CONTROL, and the null result below is void without it.
    // "No step found" and "the detector cannot fire" print the same verdict, so
    // the same detector is run once more over the WHOLE run, where the bridge
    // at ~201 is a known, large, persistent floor step. If this does not fire,
    // nothing else in this output means anything.
    match find_floor_step(&ns, ns.len()) {
        Some((b, lo, hi, gain)) => println!(
            "[rca156][control] detector FIRES on the full run: step at call {} \
             ({lo:.0} -> {hi:.0} ns, +{:.1}%). Expected ~{} (the bridge). \
             => a null pre-bridge result below is a real null.",
            b + 1,
            gain * 100.0,
            prebridge_end + 1
        ),
        None => println!(
            "[rca156][control] ⛔ REFUSING: the detector does not fire even on the \
             full run, where the bridge step is known to exist. It cannot \
             distinguish 'no step' from 'cannot see steps'. Ignore the verdict below."
        ),
    }

    match find_floor_step(&ns, prebridge_end) {
        Some((b, lo_before, rest_floor, gain)) => {
            let m = marks[b];
            println!(
                "[rca156][VERDICT] STEP at call {} — floor {lo_before:.0} -> {rest_floor:.0} ns \
                 (+{:.1}%), and NO call in {}..{} is as cheap as {lo_before:.0}. \
                 At the step: loops={} aborts={} bridges={} gfails={} panics={} ops={}->{}",
                b + 1,
                gain * 100.0,
                b + 1,
                prebridge_end,
                m.loops,
                m.aborts,
                m.bridges,
                m.gfails,
                m.panics,
                m.ops_before,
                m.ops_after
            );
            println!(
                "[rca156][next] one process is one sample. Compare this call index across \
                 runs: PINS => program state, MOVES => host state."
            );
        }
        None => println!(
            "[rca156][VERDICT] NO persistent floor step of >={:.0}% anywhere in the \
             pre-bridge window (calls 1..{}). Pre-bridge floor is {:.0} ns.",
            MIN_GAIN * 100.0,
            prebridge_end,
            if prebridge_end > WINDOW {
                min_of(&ns[WINDOW..prebridge_end])
            } else {
                f64::NAN
            }
        ),
    }
}
