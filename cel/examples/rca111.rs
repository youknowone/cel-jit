//! #111 — re-derive the never-compiles per-row tax on THIS tree.
//!
//! ## What is being re-derived, and why it is not inherited
//!
//! #111's headline is `766 ns + 165.8 ns/row` (cranelift) on majit's
//! never-compiles arm, against the clean interpreter's `~8 ns/row` — a **~20x
//! per-row tax**. That fit comes from Probe B (`rca88b.rs`), measured on the
//! **pre-#128 tree**. #122's headline inverted on this tree, and everything
//! downstream of it must be re-derived rather than inherited.
//!
//! ## ⭐ The mechanism check, stated BEFORE the measurement
//!
//! #128 changed what a back-edge FINISH does **inside compiled code**. The
//! never-compiles arm (`threshold = u32::MAX`) never compiles anything and
//! never enters compiled code, so **#128 has no channel to this number.**
//!
//! * **H1 (mechanism):** the slope is unchanged, ~166 ns/row.
//! * **H0 (a lead from #122's by-product):** a floor sweep at
//!   `threshold = u32::MAX` during the #122 runs read 2042 / 10250 / 82417 ns
//!   at n = 10 / 100 / 1000, which fits **~80 ns/row**, not 166.
//!
//! ⛔ That by-product is a LEAD, NOT A RESULT: it came from a probe with a
//! different estimator, no clean arm, and only three points. This file exists to
//! settle it. **The probe must distinguish 80 from 166** — a 2x separation, easy
//! if and only if the estimator is unbiased. Everything below is about that.
//!
//! ## ⛔⛔ Estimator design — three traps this file is built to avoid
//!
//! **1. `reps` must not be a function of `n`.** Probe B used `reps = 20_000 / n`.
//! #122's method notes record what that cost: *"a normalisation chosen to
//! equalise total work per round silently made the instrument's resolution a
//! function of the variable under study."* Here `REPS` is a **constant**, the
//! same for every arm and every n, so `n` never enters the harness's arithmetic.
//!
//! **2. A slope needs a LADDER, not two points.** Two points through a `min`
//! have previously fabricated a component that a four-point ladder refuted to
//! ~0. The default sweep is five n values and the fit's **worst residual is
//! printed**, so a bad fit cannot be quoted as a slope.
//!
//! **3. Timer overhead is not negligible at the small end.** The clean arm at
//! n=10 is ~166 ns/call and `Instant::now()` costs tens of ns on this host. With
//! `REPS = 8` the overhead is amortised 8x, and it is **measured and printed**
//! rather than assumed away.
//!
//! ## ⛔ Host state (#156) governs every level printed here
//!
//! The pre-bridge floor on this host takes two levels ~1.43x apart, chosen once
//! per process and drifting across a session. Therefore:
//!
//! * all three arms are timed **inside the same round**, back to back, so a
//!   ratio is formed from arms that saw the same host state;
//! * every round's levels are printed, not just an aggregate;
//! * **the ratio and the slope are the results. A level is not a result.**
//!
//! ## Controls
//!
//! * **Non-vacuity (#111's own):** the never-compiles arm must end with
//!   `loops_compiled == 0`. ⚠ The assertion INVERTS with `RCA111_THRESHOLD`, and
//!   both directions are checked: at the default threshold the *compiled* arm
//!   must compile, so `loops_compiled == 0` is the refusal; at
//!   `RCA111_THRESHOLD=4294967295` every arm is a never-compiles arm, so
//!   `loops_compiled == 0` is the PASS and any non-zero value is the refusal.
//!   One threshold cannot test both, so the certification is a second run.
//! * **Positive control on the clean arm:** its slope should land near #111's
//!   ~8 ns/row. ⚠ This landmark is NOT guaranteed intact — #52 ("converge onto
//!   ONE interpreter") is in progress and #102 measured a VM-vs-walker deficit,
//!   so the clean arm is a thing that can legitimately have moved. The clean
//!   slope is therefore **reported as a measurement, not asserted as a check**,
//!   and the portable quantity is the **ratio**, per #111's own rule that levels
//!   are not comparable across probes but differences are.
//!
//! ## No `#[global_allocator]`, deliberately
//!
//! This measures TIME. #111's allocation half (11/hit, 11 of 11 attributed) is
//! already closed by a separate probe and is #128-immune; taxing every
//! allocation here would perturb the quantity under study.

use std::hint::black_box;

use cel::majit::bytecode::float_bank::{jit_stats, reset_jit_stats, reset_persistent_state};
use cel::majit::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
use cel::majit::lower::{lower_typed, LoweredF, Schema, ValType};
use cel::Program;

/// `threshold = u32::MAX`: the back-edge counter can never be reached, so
/// nothing is ever compiled. This is #111's arm.
const NEVER: u32 = u32::MAX;

/// Calls per timed region. **Constant across every arm and every n** — see trap
/// 1 in the header. Its only job is to amortise timer overhead.
const REPS: usize = 8;

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

/// Minimum per-call time over `regions` timed regions of `REPS` calls each,
/// discarding the first `warmup` regions.
///
/// A floor, not a mean: a mean can be lifted by a load spike, a floor cannot.
fn floor_ns<F: FnMut()>(regions: usize, warmup: usize, mut call: F) -> f64 {
    let mut best = f64::MAX;
    for r in 0..regions {
        let t = std::time::Instant::now();
        for _ in 0..REPS {
            call();
        }
        let ns = t.elapsed().as_nanos() as f64 / REPS as f64;
        if r >= warmup && ns < best {
            best = ns;
        }
    }
    best
}

/// Cost of the timing apparatus itself — one `Instant::now()` + `elapsed()` pair,
/// which is what `floor_ns` adds to every region.
///
/// ⛔ The first version of this timed `REPS` iterations of `black_box(0u64)` inside
/// one pair and divided by `REPS`. It printed **`overhead=0.0 ns/call`**, which
/// reads as "the apparatus is free" and is a FALSE ZERO: the region was below the
/// clock's own resolution, so the instrument reported the *floor of its scale* as
/// a measurement. The shape here instead amortises the pair over `K` pairs inside
/// an outer timer, so the quantity is always above the resolution, and it
/// REFUSES rather than returning a zero.
fn timer_pair_ns() -> Option<f64> {
    const K: usize = 10_000;
    let outer = std::time::Instant::now();
    for _ in 0..K {
        let t = std::time::Instant::now();
        black_box(t.elapsed());
    }
    let per_pair = outer.elapsed().as_nanos() as f64 / K as f64;
    (per_pair > 0.0).then_some(per_pair)
}

/// Least-squares fit of `total = intercept + slope * n`.
///
/// Returns `(intercept, slope, worst_abs_residual, worst_rel_residual)`. The
/// residuals are returned rather than kept private because **a slope quoted
/// without its fit quality is the two-point trap wearing more digits.**
fn fit(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let k = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let denom = k * sxx - sx * sx;
    let slope = (k * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / k;
    let mut worst_abs: f64 = 0.0;
    let mut worst_rel: f64 = 0.0;
    for &(x, y) in points {
        let pred = intercept + slope * x;
        worst_abs = worst_abs.max((y - pred).abs());
        if y != 0.0 {
            worst_rel = worst_rel.max(((y - pred) / y).abs());
        }
    }
    (intercept, slope, worst_abs, worst_rel)
}

/// Path, size and mtime of the binary **as it is running**, so the label travels
/// with the numbers instead of living in a header nobody re-reads.
///
/// ⛔ Build artifacts are not durable in this tree: the binary behind a
/// previously published measurement was deleted while five neighbouring probes
/// survived. A sha pins what a binary was built FROM; this pins that the file
/// still existed, at this size, when the numbers were produced. The wrapper
/// computes the sha256 immediately before invoking and passes it in.
fn attest() -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let meta = std::fs::metadata(&exe).expect("metadata of current_exe");
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let sha = std::env::var("RCA111_BIN_SHA").unwrap_or_else(|_| "unset".to_string());
    let short: String = sha.chars().take(12).collect();
    // Short on purpose: this rides on EVERY data line, so the path and length go
    // in the config line once. A label nobody can read gets skipped.
    format!("bin={short}@{mtime}")
}

/// The one-time provenance line: full path and size, printed beside the label
/// that every later line carries.
fn attest_full() -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let meta = std::fs::metadata(&exe).expect("metadata of current_exe");
    format!("len={} path={}", meta.len(), exe.display())
}

fn main() {
    let regions = env_usize("RCA111_REGIONS", 60);
    let warmup = env_usize("RCA111_WARMUP", 20);
    let rounds = env_usize("RCA111_ROUNDS", 5);
    let expr = std::env::var("RCA111_EXPR").unwrap_or_else(|_| "price + qty * 2".to_string());
    let compiled_threshold = env_usize("RCA111_THRESHOLD", 8) as u32;
    // A LADDER, not a pair. Five points so the fit's residual is meaningful.
    let ladder: Vec<usize> = std::env::var("RCA111_LADDER")
        .unwrap_or_else(|_| "10,30,100,300,1000".to_string())
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .expect("RCA111_LADDER: comma-separated integers")
        })
        .collect();
    // A 2-point ladder fits a line EXACTLY, so the worst-residual check below would
    // print 0.0% for any data whatsoever. That is a self-proving diagnostic, and it
    // reads as the strongest possible fit. Refuse rather than print it.
    if ladder.len() < 3 {
        println!(
            "[rca111][config] ⛔ REFUSING: RCA111_LADDER has {} point(s). Fewer than 3 points fit \
             a line exactly, so the worst-residual figure would be 0.0% regardless of the data.",
            ladder.len()
        );
        std::process::exit(2);
    }

    let att = attest();
    println!(
        "[rca111][config] {att} {} expr={expr:?} ladder={ladder:?} reps={REPS} \
         regions={regions} warmup={warmup} rounds={rounds} compiled_threshold={compiled_threshold}",
        attest_full()
    );
    match timer_pair_ns() {
        Some(pair) => println!(
            "[rca111][timer] {att} apparatus={pair:.1} ns/region = {:.2} ns/call at REPS={REPS} \
             — subtract this before reading any level below",
            pair / REPS as f64
        ),
        None => println!(
            "[rca111][timer] {att} ⛔ REFUSING to report an apparatus cost: the measurement came \
             back at or below the clock's resolution, which is the instrument's own floor and \
             NOT a demonstration that timing is free. Treat every level below as carrying an \
             unknown per-region constant."
        ),
    }

    let schema = flat_schema();
    let lowered = lower(&expr, &schema);

    reset_persistent_state();
    reset_jit_stats();

    for round in 1..=rounds {
        // Per-n fits are formed from levels measured in THIS round, so a ratio
        // never spans two host states (#156).
        let mut clean_pts: Vec<(f64, f64)> = Vec::with_capacity(ladder.len());
        let mut never_pts: Vec<(f64, f64)> = Vec::with_capacity(ladder.len());
        let mut comp_pts: Vec<(f64, f64)> = Vec::with_capacity(ladder.len());

        for &n in &ladder {
            let (price, qty) = flat_columns(n);
            let columns = vec![Column::Int(&price), Column::Int(&qty)];

            // ⭐ The three arms are timed back to back, INSIDE the n loop and
            // inside the round, so they share whatever host state this moment
            // has. Ordering the sweep by arm instead would confound arm with
            // time, which is exactly the trap #156 documents.
            let clean = floor_ns(regions, warmup, || {
                black_box(clean_batch_sum_f(&lowered, &columns, n));
            });
            let never = floor_ns(regions, warmup, || {
                black_box(eval_batch_sum_f(&lowered, &columns, n, NEVER));
            });
            let comp = floor_ns(regions, warmup, || {
                black_box(eval_batch_sum_f(&lowered, &columns, n, compiled_threshold));
            });

            clean_pts.push((n as f64, clean));
            never_pts.push((n as f64, never));
            comp_pts.push((n as f64, comp));

            println!(
                "[rca111][level] {att} round={round} n={n:<5} clean={clean:>10.1} \
                 never={never:>10.1} comp={comp:>10.1} never/clean={:>7.2}",
                never / clean
            );
        }

        let (ci, cs, ca, cr) = fit(&clean_pts);
        let (ni, ns_, na, nr) = fit(&never_pts);
        let (pi, ps, pa, pr) = fit(&comp_pts);
        println!(
            "[rca111][fit]   {att} round={round} clean = {ci:.0} ns + {cs:.2} ns/row \
             (worst residual {ca:.0} ns = {:.1}%)",
            cr * 100.0
        );
        println!(
            "[rca111][fit]   {att} round={round} never = {ni:.0} ns + {ns_:.2} ns/row \
             (worst residual {na:.0} ns = {:.1}%)",
            nr * 100.0
        );
        println!(
            "[rca111][fit]   {att} round={round} comp  = {pi:.0} ns + {ps:.2} ns/row \
             (worst residual {pa:.0} ns = {:.1}%)  ⚠ accumulating: this arm crosses \
             its bridge during the sweep",
            pr * 100.0
        );
        println!(
            "[rca111][tax]   {att} round={round} per-row tax = {:.2}x  \
             (never {ns_:.2} ns/row / clean {cs:.2} ns/row)",
            ns_ / cs
        );
    }

    // #111's own non-vacuity check. The never-compiles arm shares this process
    // with the compiled arm, which DOES compile, so the counter cannot be read
    // as a whole-process zero — see the refusal text.
    let s = jit_stats();
    println!(
        "[rca111][counters] {att} loops_compiled={} loops_aborted={} bridges_compiled={} \
         guard_failures={}",
        s.loops_compiled, s.loops_aborted, s.bridges_compiled, s.guard_failures
    );
    // The assertion INVERTS with the threshold. Under RCA111_THRESHOLD=NEVER every arm is a
    // never-compiles arm, so loops_compiled=0 is the PASS — and an earlier revision of this
    // block refused it, i.e. refused the exact certification run its own hint text asks for.
    let verdict = match (compiled_threshold == NEVER, s.loops_compiled) {
        (false, 0) => format!(
            "⛔ REFUSING: loops_compiled=0 for the whole process, but the compiled arm at \
             threshold={compiled_threshold} MUST compile. Nothing was exercised; ignore every \
             number above."
        ),
        (false, k) => format!(
            "compiled arm did compile (loops_compiled={k}), so the process exercised the JIT. \
             ⚠ This does NOT certify the never-compiles arm separately: `run_jit_persistent_f` \
             keys its driver pool on (nregs, nfregs, threshold), so the two arms hold different \
             drivers, and the counters are process-global. To certify the never-compiles arm \
             alone, run with RCA111_THRESHOLD={NEVER} and require loops_compiled=0."
        ),
        (true, 0) => format!(
            "✅ CERTIFICATION MODE (threshold={NEVER}): every arm is a never-compiles arm and \
             the process compiled nothing — loops_compiled/aborted/bridges/guard_failures all 0 \
             is the PASS. ⚠ The `comp` fit line above is degenerate in this mode (it is a second \
             never arm, not a compiled one); the clean/never slopes and the tax remain readable."
        ),
        (true, k) => format!(
            "⛔ REFUSING: threshold={NEVER} yet loops_compiled={k}. Something compiled on a \
             never-compiles threshold, so the arm this probe exists to price is misnamed; \
             ignore every number above."
        ),
    };
    println!("[rca111][control] {att} {verdict}");
}
