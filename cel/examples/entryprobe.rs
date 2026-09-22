//! The ~100 ns a `Tier::Jit` call spends ARRIVING in compiled code, itemised
//! into the named stages `float_bank::run_jit_persistent_f` is made of.
//!
//! `routeprobe` reads that cost whole, as `jit_fix - clean_fix` at the smallest
//! batches it sweeps, and ships it as `batch::JIT_ENTRY_PS`. It is the term
//! that decides twelve of thirteen `Tier::Auto` routes away from the compiled
//! tier, so it is the thing to attack -- and a single number says nothing about
//! where to attack it. This file splits it.
//!
//! # The stages, in call order
//!
//! ```text
//!   A   DRIVERS.with(..)                     one thread-local access
//!   BF  check_out(..) .. check_in(..)        the pool ROUND TRIP
//!   C   reseed_state_f(..)                   two resizes, a memcpy and a fill
//!   D   loop_keys.iter().any(..)             one bucket walk PER loop key
//!   E   back_edge_resolved(..)               frame build, marshalling, jump
//! ```
//!
//! Four of the five are measured. E is a RESIDUAL -- `entry - (A + BF + C + D)`
//! -- and is labelled as one everywhere it appears, because it is the one stage
//! that cannot be repeated: it runs the program.
//!
//! # Inside E
//!
//! E is four fifths of the entry and, being a residual, it also holds the
//! RETURN half of the call. So it is split again, by the same means, inside the
//! warm arm of `JitDriver::back_edge_internal` -- majit's side of the door:
//!
//! ```text
//!   E1  warm-entry gate      token, meta, descriptor, is_compatible
//!   E2  marshal in           sync_before, extract, extend, the clears
//!   E3  THE CALL             frame build, the ABI wrapper, the trace,
//!                            the deadframe decode                 <- RESIDUAL
//!   E4  marshal out          restore_values, sync_after
//! ```
//!
//! Three of the four are measured; E3 is not, and CANNOT be. The amplification
//! trick needs a stage that answers the same thing every time it is run, and
//! E3 runs the program: a second pass would execute the trace again from state
//! the first one already advanced, and would leave through a different exit.
//! There is no honest repeatable form of it, so none is offered -- E3 is what
//! `E - (E1 + E2 + E4)` is, together with the one-shot moves the three measured
//! stages deliberately exclude. `majit_metainterp::BackEdgeStageRepeats` names
//! those on each field; they are `resolve_cell_key`, the single-pass dispatch
//! key TAKE, `take_entry_scratch`'s own `mem::take`, `entry_scratch_out`, and
//! the `drop(result)` on the guard-failure path.
//!
//! The two sides carry SEPARATE barrier arms and each stage is differenced
//! against its own, because the two amplification loops are in different crates
//! and the barrier is the loop's cost, not the door's.
//!
//! # Amplification, not per-stage clocks
//!
//! `Instant::now()` costs 20-25 ns on this box. Five of them inside a 100 ns
//! budget would measure the clock. So each stage is timed by REPETITION
//! instead: the same call, once with every repeat count at zero and once with
//! one stage repeated [`repeat`] extra times, and the difference over that count
//! is that stage. `cleanfixprobe` splits a call into halves the same way.
//!
//! Two things that shape is easy to get wrong, and both are settled in
//! `float_bank::EntryStageRepeats` rather than here:
//!
//! * `check_out` is NOT idempotent -- it takes the driver out of its slot, and
//!   a second one in a row makes the caller BUILD one. So B and F are amplified
//!   together, as a round trip, which is.
//! * Every amplification loop carries a counter and one optimization barrier,
//!   without which the repeated stage is dead code. A fifth arm runs that loop
//!   with NO stage in it, and every stage figure has it subtracted.
//! * A barrier and a reachability counter together still do not prove the stage
//!   ran once per PASS. A stage whose inputs are loop-invariant can be hoisted
//!   out of the amplification loop and run once, which the counter cannot see
//!   -- and which does NOT collapse the arm onto the barrier, because once is
//!   not zero. The count is what tells them apart: see [`REPEAT_K`], and sweep
//!   it.
//!
//! # What the numbers are and are not
//!
//! The whole-entry figure is read exactly as `routeprobe` reads it, from the
//! same two intercepts on the same shape family, so the parts and the whole are
//! about one door. But this binary carries the probe feature, which costs one
//! thread-local read and five zero-trip loop tests per JIT call -- and none per
//! `Tier::Clean` call, so all of it lands in `jit_fix`. The entry printed here
//! is therefore the shipping entry plus that, and the report says so beside the
//! number. Differences between arms are unaffected; both arms pay it.
//!
//! JITFRAME allocation is not an amplified arm once the backend allocates it
//! from the moving GC. Between allocating the real entry frame and entering
//! compiled code there is no root that an extra collecting allocation could
//! update, while a no-collect repetition would retain every scratch frame and
//! eventually measure nursery growth. The whole-entry and single-shot call
//! figures still include the allocation. Process-allocation effects are
//! measured separately by `allocs_per_eval`.
//!
//! Every measured stage is a WARM repeat as well. The passes after the first
//! find the caches the first one filled, so each figure is a LOWER bound on
//! what the call's single execution of that stage costs -- and the residual,
//! being what is left, absorbs the difference along with any stage repetition
//! could not force to happen at all.
//!
//! # The two stages that already had a price
//!
//! `batch::JIT_ENTRY_PS` is the whole this splits, so the entry printed here
//! has to land beside it or the split is about some other door.
//!
//! The yield scan is the other. `float_bank::loop_header_keys` argues in its
//! own doc that its word-wise scan is "the cheaper of two answers this caller
//! cannot tell apart", and the keys that scan admits spuriously have been
//! priced at ~1% -- of the compiled tier's WHOLE per-call cost, which is an
//! order of magnitude larger than the entry. Stage D divides the same
//! nanoseconds by the entry instead, which is the term `BoundBatch::route`
//! compares against, and prints a per-key figure beside the door's own key
//! count. The two denominators are the disagreement; neither reading is wrong
//! about its own.
//!
//! RELEASE ONLY, and the split needs the probe feature:
//!
//! ```text
//! cargo build --release -p cel \
//!     --features jit-cranelift,__entry-stage-probe --example entryprobe
//! target/release/examples/entryprobe
//! ```
//!
//! Without `__entry-stage-probe` the file still builds and still reports the
//! entry whole; it prints which feature the split needs and stops there.

use std::hint::black_box;
use std::time::{Duration, Instant};

use cel::majit::batch::{Batch, BatchProgram, BoundBatch, ColumnRef, Tier, JIT_ENTRY_PS};
use cel::majit::bytecode::float_bank::{jit_stats, reset_persistent_state};
use cel::majit::lower::{Schema, ValType};
use cel::Value;

#[cfg(feature = "__entry-stage-probe")]
use cel::majit::bytecode::float_bank::{
    entry_stage_loop_keys, entry_stage_sub_passes, reset_entry_stage_sub_passes,
    set_entry_stage_repeats, EntryStageRepeats,
};
#[cfg(feature = "__entry-stage-probe")]
use majit_metainterp::{back_edge_stage_passes, set_back_edge_stage_repeats, BackEdgeStageRepeats};
#[cfg(feature = "__entry-stage-probe")]
use majit_metainterp::{
    call_shot_totals, execute_stage_clock_floor_ns, execute_stage_passes, frame_build_passes,
    reset_call_shot_totals, set_execute_stage_repeats, set_frame_build_repeats,
    ExecuteStageRepeats,
};

/// One timed batch must last at least this long, so the clock's own resolution
/// is not what the measurement is against. `routeprobe`'s number, because the
/// entry figure here has to be comparable with its.
const MIN_BATCH: Duration = Duration::from_millis(20);
/// Timed batches per arm; the fastest is reported, since other work on the box
/// can only ever make one slower. Higher than `routeprobe`'s nine: a stage
/// difference is a few tens of nanoseconds inside a call of a few hundred, so
/// it has less room above the noise than a whole tier does.
const ROUNDS: usize = 21;
/// Calls in the probe that decides whether a point's `jit` cell is the compiled
/// tier's number or the tracing interpreter's.
const PROBE_CALLS: usize = 200;
/// The points the two intercepts are read from. `routeprobe` reads them from
/// its sweep's points at or below four, and this is that region on its own:
/// nothing here needs the slopes, so nothing here sweeps for them.
const SWEEP: [usize; 4] = [1, 2, 3, 4];
/// Extra passes per amplified stage, as [`repeat`] reports it.
///
/// Large enough that a one-nanosecond stage moves the call by tens of
/// nanoseconds -- well clear of what the fastest of [`ROUNDS`] batches varies
/// by -- and small enough that the amplified arm stays the same order as the
/// arm it is differenced against, which is what keeps the two comparable under
/// a load excursion.
///
/// SETTABLE at run time with `ENTRYPROBE_REPEAT`, because the count is the only
/// thing that separates a stage running once per pass from one the compiler
/// hoisted out of the amplification loop. Hoisting does not make an arm read
/// zero -- the work still runs ONCE -- so a hoisted arm reads `cost / count`,
/// which is not small and sits just as far above the barrier as a real stage
/// does. It is however the one thing that MOVES: per-pass work reads the same
/// ns/pass at every count, hoisted work HALVES each time the count doubles.
///
/// Sweeping it needs no rebuild and changes no machine code in the arms: the
/// counts reach the amplification loops through a `Cell` inside `cel` itself,
/// so they are already opaque to the optimizer there. One binary, three runs.
/// The same sweep prices any other per-ARMING cost that leaked into a stage --
/// a setup, a cold miss, a first-touch fault -- since all of them decay as
/// `1/count` while the stage itself does not.
#[cfg(feature = "__entry-stage-probe")]
static REPEAT_K: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(32);

/// Extra passes each amplified arm asks for. See [`REPEAT_K`].
#[cfg(feature = "__entry-stage-probe")]
fn repeat() -> u32 {
    REPEAT_K.load(std::sync::atomic::Ordering::Relaxed)
}

/// Entries the single-shot call arm clocks. Higher than the amplified arms
/// need: each entry contributes ONE reading rather than [`repeat`] of them, so
/// the averaging an amplified arm gets for free has to be bought here.
#[cfg(feature = "__entry-stage-probe")]
const CALL_SHOTS: usize = 20_000;

/// The amplified arms, in the order [`Split::raw`] holds them: cel's four
/// stages plus D's three sub-arms and its barrier, then majit's three and its
/// barrier, then the three inside the call and theirs. The three barrier arms
/// are machinery and not stages, and each group's stages are differenced
/// against their own — see [`Split::barrier_for`].
///
/// `D1`/`D2`/`D3` are the parts of `D`, not siblings of it: `D` is the whole
/// yield scan and the three are what it is made of, so they belong to D's sum
/// and not to the entry's. Summing all sixteen would count the scan twice.
const STAGE_LABELS: [&str; 17] = [
    "A  DRIVERS.with",
    "BF check_out+check_in",
    "C  reseed_state_f",
    "D  loop-key yield scan",
    "D1 walk key.resolve",
    "D2 token has_compiled_loop",
    "D4 upgrade Weak+drop",
    "D3 meta get_compiled_meta",
    "(cel barrier)",
    "E1 warm-entry gate",
    "E2 marshal in",
    "E4 marshal out",
    "(majit barrier)",
    "E3a prologue",
    "E3b frame build",
    "E3d deadframe decode",
    "(execute barrier)",
];

/// Where majit's back-edge arms start in [`STAGE_LABELS`].
const MAJIT_FIRST: usize = 9;

/// D's three sub-arms. Parts of `D` rather than entries in the entry's sum —
/// see the note on [`STAGE_LABELS`]. They are cel-side, so their reachability
/// is proved by cel's own counters and not by majit's.
///
/// UNGATED, like every other index constant here, because the report reads
/// these arms from code that the probe feature does not gate. Gating the
/// definition while the uses stay open builds only under
/// `__entry-stage-probe` -- the one configuration that hides it.
const D_SUB: std::ops::Range<usize> = 4..8;
/// D itself, which its sub-arms follow immediately.
const D_IDX: usize = D_SUB.start - 1;

// ⚠ DERIVED, never literals. Every one of these was written as a literal once,
// and inserting arms left them all pointing at the previous occupant -- which
// prints another stage's number under this one's name, or differences a stage
// against a stage. `check_barrier_wiring` asserts the result.
/// The barrier closing each group: the last arm before the next group starts.
const CEL_BARRIER: usize = MAJIT_FIRST - 1;
const MAJIT_BARRIER: usize = EXEC_FIRST - 1;
const EXEC_BARRIER: usize = STAGE_LABELS.len() - 1;
/// The three arms inside E3, in table order.
const E3A: usize = EXEC_FIRST;
const E3B: usize = EXEC_FIRST + 1;
const E3D: usize = EXEC_FIRST + 2;

/// Where the arms INSIDE E3 start. Everything from here down is about
/// `execute_assembler_at_dispatch_key`, and `E3b` is a part of the call rather
/// than a sibling of it — see [`Split::e3_residual`].
const EXEC_FIRST: usize = 13;

/// One arm's repeat counts, on both sides of the door.
///
/// One type and not two so every arm sets BOTH, and the two setter calls are a
/// constant every arm pays rather than a difference between them.
#[cfg(feature = "__entry-stage-probe")]
#[derive(Clone, Copy, Default, Debug)]
struct Repeats {
    cel: EntryStageRepeats,
    majit: BackEdgeStageRepeats,
    /// The stages inside `execute_assembler_at_dispatch_key`. `call_shot` here
    /// is NOT an amplified arm — see [`ExecuteStageRepeats`].
    exec: ExecuteStageRepeats,
    /// Extra frame builds, which live one crate further down because the frame
    /// does. Inside the call, not beside it.
    frame_build: u32,
}

#[cfg(feature = "__entry-stage-probe")]
fn set_repeats(repeats: Repeats) {
    // All four, on every arm, so the setters are a constant every arm pays
    // rather than a difference between them.
    set_entry_stage_repeats(repeats.cel);
    set_back_edge_stage_repeats(repeats.majit);
    set_execute_stage_repeats(repeats.exec);
    set_frame_build_repeats(repeats.frame_build);
}

/// Which repeat count each arm raises. Index-parallel with [`STAGE_LABELS`].
#[cfg(feature = "__entry-stage-probe")]
const STAGE_ARMS: [fn(&mut Repeats); 17] = [
    |r| r.cel.tls = repeat(),
    |r| r.cel.pool = repeat(),
    |r| r.cel.reseed = repeat(),
    |r| r.cel.loop_keys = repeat(),
    |r| r.cel.loop_walk = repeat(),
    |r| r.cel.loop_token = repeat(),
    |r| r.cel.loop_upgrade = repeat(),
    |r| r.cel.loop_meta = repeat(),
    |r| r.cel.barrier = repeat(),
    |r| r.majit.gate = repeat() as u16,
    |r| r.majit.marshal_in = repeat() as u16,
    |r| r.majit.marshal_out = repeat() as u16,
    |r| r.majit.barrier = repeat() as u16,
    |r| r.exec.prologue = repeat() as u16,
    |r| r.frame_build = repeat(),
    |r| r.exec.decode = repeat() as u16,
    |r| r.exec.barrier = repeat() as u16,
];

fn timed(iters: usize, run: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        run();
    }
    start.elapsed()
}

/// Time TWO calls against each other, INTERLEAVED: one iteration count is
/// calibrated so a batch of either lasts at least [`MIN_BATCH`], then the two
/// take turns for [`ROUNDS`] batches each and the fastest of each is reported.
///
/// Interleaved rather than one arm and then the other, because the difference
/// between the two is the whole measurement and a busy box does not hold still
/// for the length of a sweep. `routeprobe` measured that discipline as worth
/// 30-40% of a crossing at load 150; it is worth more here, where the
/// difference being read is a tenth of a call rather than a whole tier.
fn per_call_pair(mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
    let mut iters = 1usize;
    while timed(iters, &mut a).max(timed(iters, &mut b)) < MIN_BATCH {
        iters = (iters * 2).max(1);
    }
    let (mut best_a, mut best_b) = (Duration::MAX, Duration::MAX);
    for _ in 0..ROUNDS {
        best_a = best_a.min(timed(iters, &mut a));
        best_b = best_b.min(timed(iters, &mut b));
    }
    let ns = |d: Duration| d.as_secs_f64() * 1e9 / iters as f64;
    (ns(best_a), ns(best_b))
}

/// Run the batch loop until the trace threshold is crossed, then report the
/// share of a fixed probe's calls that ENTERED compiled code.
///
/// A `Tier::Jit` call that did not enter is the tracing interpreter's, and it
/// pays no entry at all — so a point measured through one describes the harness
/// rather than the door. Everything downstream drops the points below one.
fn warm(bound: &BoundBatch<'_, '_>, out: &mut Vec<Value>) -> f64 {
    for _ in 0..512 {
        bound.collect_into_on(Tier::Jit, out).expect("warm run");
        black_box(out.as_slice());
    }
    let before = jit_stats().compiled_entries;
    for _ in 0..PROBE_CALLS {
        bound.collect_into_on(Tier::Jit, out).expect("probe run");
        black_box(out.as_slice());
    }
    (jit_stats().compiled_entries - before) as f64 / PROBE_CALLS as f64
}

/// One swept point: `(n, clean, jit, entered share)`, both tiers on ONE bound
/// batch and timed against each other.
fn point(bound: &BoundBatch<'_, '_>, n: usize) -> (usize, f64, f64, f64) {
    let mut out: Vec<Value> = Vec::new();
    let entered = warm(bound, &mut out);
    // Two buffers, not one: the arms alternate, and a shared buffer would hand
    // each arm the other's allocation state.
    let mut out_jit = out.clone();
    let run = |tier, buf: &mut Vec<Value>| {
        bound.collect_into_on(tier, buf).expect("run");
        black_box(buf.as_slice());
    };
    let (clean, jit) = per_call_pair(
        || run(Tier::Clean, &mut out),
        || run(Tier::Jit, &mut out_jit),
    );
    (n, clean, jit, entered)
}

/// The nine amplified arms on one bound batch, as nanoseconds per EXTRA pass,
/// plus the loop-key count the door published while the scan arm ran.
///
/// Nothing is subtracted here: the two barrier arms are returned alongside the
/// seven stages rather than folded into them, so the report can print what the
/// amplification itself cost next to what it was used to measure.
#[cfg(feature = "__entry-stage-probe")]
fn arms(bound: &BoundBatch<'_, '_>) -> ([f64; 17], usize, f64, f64, u64) {
    let mut out: Vec<Value> = Vec::new();
    let entered = warm(bound, &mut out);
    // Every arm computes the same answer, and an amplification that broke that
    // would still return `Ok` -- so each arm is checked against the clean tier's
    // answer before any of them is timed. That is the convention
    // `__elem-attr-probe` binds its arms under: an arm that removed work it should
    // not have is a failure and not a faster number.
    let mut want: Vec<Value> = Vec::new();
    bound
        .collect_into_on(Tier::Clean, &mut want)
        .expect("clean witness");
    for arm in STAGE_ARMS {
        let mut amplified = Repeats::default();
        arm(&mut amplified);
        set_repeats(amplified);
        let mut got: Vec<Value> = Vec::new();
        bound
            .collect_into_on(Tier::Jit, &mut got)
            .expect("witness run");
        assert!(
            got == want,
            "an amplified arm changed the answer: {amplified:?}"
        );
    }
    set_repeats(Repeats::default());

    let mut base_buf = out.clone();
    let mut amp_buf = out;
    let mut raw = [0.0f64; 17];
    for (slot, arm) in raw.iter_mut().zip(STAGE_ARMS) {
        let mut amplified = Repeats::default();
        arm(&mut amplified);
        // The counts are set from INSIDE each arm, so both arms pay one
        // thread-local write and one relaxed store per call and both cancel out
        // of their difference. Setting them once outside would not: the arms
        // interleave in batches, and each batch would then have to restore what
        // the other left.
        let run = |repeats, buf: &mut Vec<Value>| {
            set_repeats(repeats);
            bound.collect_into_on(Tier::Jit, buf).expect("jit run");
            black_box(buf.as_slice());
        };
        let (base, amped) = per_call_pair(
            || run(Repeats::default(), &mut base_buf),
            || run(amplified, &mut amp_buf),
        );
        *slot = (amped - base) / f64::from(repeat());
    }
    set_repeats(Repeats::default());

    // ── the one stage that cannot be amplified ────────────────────────────
    // SINGLE-SHOT, and reported as one. The call runs the trace, so instead of
    // repeating it the arm clocks it once per entry and accumulates. The clock
    // pair's own floor is measured in the same crate and subtracted; what is
    // left is the call plus whatever of that pair's cost the floor under-reads,
    // so this is an UPPER bound on the call rather than a two-sided estimate.
    let (call_ns, call_shots) = {
        let mut shot = Repeats::default();
        shot.exec.call_shot = 1;
        reset_call_shot_totals();
        set_repeats(shot);
        let mut shot_buf: Vec<Value> = Vec::new();
        for _ in 0..CALL_SHOTS {
            bound
                .collect_into_on(Tier::Jit, &mut shot_buf)
                .expect("call-shot run");
            black_box(shot_buf.as_slice());
        }
        set_repeats(Repeats::default());
        let (ns, shots) = call_shot_totals();
        let floor = execute_stage_clock_floor_ns();
        let mean = if shots == 0 {
            f64::NAN
        } else {
            ns as f64 / shots as f64 - floor
        };
        (mean, shots)
    };
    (raw, entry_stage_loop_keys(), entered, call_ns, call_shots)
}

/// Without the feature there are no arms to run, and the file reports the entry
/// whole rather than pretending to split it.
#[cfg(not(feature = "__entry-stage-probe"))]
fn arms(bound: &BoundBatch<'_, '_>) -> ([f64; 17], usize, f64, f64, u64) {
    let mut out: Vec<Value> = Vec::new();
    let entered = warm(bound, &mut out);
    ([f64::NAN; 17], 0, entered, f64::NAN, 0)
}

/// Ordinary least squares of `y` against `x`, as `(intercept, slope)`.
fn line(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
    (my - slope * mx, slope)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// One shape's entry, split.
struct Split {
    label: &'static str,
    /// Share of the probe's calls that entered compiled code, at the point the
    /// arms were run on.
    entered: f64,
    /// The count the arms were run at — the smallest swept point that entered.
    stage_n: usize,
    clean_fix: f64,
    jit_fix: f64,
    /// Per-pass cost of each amplified arm, each side's barrier last within its
    /// own run of the array, and NOT yet subtracted.
    raw: [f64; 17],
    /// Loop keys the door walks per call for this program.
    loop_keys: usize,
    /// The compiled call, SINGLE-SHOT: the mean of one clocked reading per
    /// entry, with the clock pair's own floor already subtracted. NaN when the
    /// arm did not run. Not comparable with the amplified figures beside it —
    /// see the header.
    call_ns: f64,
    /// How many readings that mean is over.
    call_shots: u64,
}

impl Split {
    /// What a call pays to reach compiled code, from this shape alone —
    /// `routeprobe`'s reading, on `routeprobe`'s two intercepts.
    fn entry(&self) -> f64 {
        self.jit_fix - self.clean_fix
    }
    /// The barrier arm a stage is differenced against: its OWN side's. The two
    /// amplification loops are in different crates and a barrier prices the
    /// loop, not the door.
    /// ⚠ DERIVED, never written as literals. Each barrier is the LAST arm of
    /// its group, so its index is one before the next group starts. Spelling
    /// them as constants is what went wrong before: inserting arms moved every
    /// barrier and left `barrier_for` returning the old positions, so each
    /// stage was differenced against another STAGE instead of against its
    /// barrier -- which produces a full table of plausible, confidently
    /// negative numbers that no pass-counter check can detect.
    fn barrier_for(i: usize) -> usize {
        if i < MAJIT_FIRST {
            CEL_BARRIER
        } else if i < EXEC_FIRST {
            MAJIT_BARRIER
        } else {
            EXEC_BARRIER
        }
    }
    /// One stage, with the amplification's own cost taken off it.
    fn stage(&self, i: usize) -> f64 {
        self.raw[i] - self.raw[Self::barrier_for(i)]
    }
    /// A + BF + C + D.
    fn measured(&self) -> f64 {
        (0..4).map(|i| self.stage(i)).sum()
    }
    /// E. A RESIDUAL and never a measurement: it is what the entry has left
    /// after the four, so it absorbs every error in them — including a stage an
    /// amplification could not force to repeat.
    fn residual(&self) -> f64 {
        self.entry() - self.measured()
    }
    /// E1 + E2 + E4 — the measured part of E.
    fn e_measured(&self) -> f64 {
        (MAJIT_FIRST..MAJIT_BARRIER).map(|i| self.stage(i)).sum()
    }
    /// E3a + E3d — the AMPLIFIED stages inside E3. Deliberately excludes E3b,
    /// which is a part of the call and not a sibling of it, and excludes the
    /// call, which is single-shot.
    fn e3_amplified(&self) -> f64 {
        self.stage(E3A) + self.stage(E3D)
    }
    /// What E3 has left once its two amplified stages and its single-shot call
    /// are taken off: the result construction and the drops. A RESIDUAL, and
    /// reported as one.
    ///
    /// NaN-safe by construction only when the call arm ran; a build that did
    /// not run it has no business quoting this.
    fn e3_residual(&self) -> f64 {
        self.e_residual() - self.e3_amplified() - self.call_ns
    }
    /// E3, the call. A RESIDUAL inside a residual, for the reason the header
    /// gives: it runs the program, so nothing can make it repeat.
    fn e_residual(&self) -> f64 {
        self.residual() - self.e_measured()
    }
}

/// Finish a shape from its swept points and its arms.
fn split(
    label: &'static str,
    points: &[(usize, f64, f64, f64)],
    raw: [f64; 17],
    loop_keys: usize,
    stage_n: usize,
    entered: f64,
    call_ns: f64,
    call_shots: u64,
) -> Split {
    let used: Vec<&(usize, f64, f64, f64)> = points.iter().filter(|p| p.3 >= 1.0).collect();
    let xs: Vec<f64> = used.iter().map(|p| p.0 as f64).collect();
    let cs: Vec<f64> = used.iter().map(|p| p.1).collect();
    let js: Vec<f64> = used.iter().map(|p| p.2).collect();
    Split {
        label,
        entered,
        stage_n,
        clean_fix: line(&xs, &cs).0,
        jit_fix: line(&xs, &js).0,
        raw,
        loop_keys,
        call_ns,
        call_shots,
    }
}

/// The smallest swept point that evidenced compiled entry — the one the arms
/// run on, because the entry is a FIXED cost and the shortest call is where it
/// is the largest share of what is being timed.
fn stage_point(points: &[(usize, f64, f64, f64)]) -> Option<usize> {
    points.iter().find(|p| p.3 >= 1.0).map(|p| p.0)
}

/// The `n`-row prefix of two int columns, as the batch a row sweep binds.
///
/// A function and not a closure: what it returns borrows the columns, and a
/// closure would tie that borrow to the closure itself rather than to the
/// locals the columns live in.
fn rows_batch<'a>(x: &'a [i64], y: &'a [i64], n: usize) -> Batch<'a> {
    Batch::new(n)
        .column("x", ColumnRef::Int(&x[..n]))
        .column("y", ColumnRef::Int(&y[..n]))
}

/// One row carrying an `n`-element list, as the batch an element sweep binds.
fn elems_batch<'a>(lens: &'a [i64], flat: &'a [i64], n: usize) -> Batch<'a> {
    Batch::new(1).column(
        "list",
        ColumnRef::List {
            lens: &lens[n..n + 1],
            fields: vec![(None, ColumnRef::Int(&flat[..n]))],
        },
    )
}

/// A straight-line program swept by batch HEIGHT. Two int columns are always
/// declared and always bound, so a shape may use one or both.
fn split_rows(label: &'static str, source: &'static str) -> Split {
    let mut schema = Schema::new();
    schema.insert("x".to_string(), ValType::Int);
    schema.insert("y".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    let top = SWEEP[SWEEP.len() - 1];
    // Built once and SLICED per point, so the sweep and the arm pass bind the
    // same columns rather than two allocations that happen to hold equal values.
    let x: Vec<i64> = (0..top as i64).collect();
    let y: Vec<i64> = (0..top as i64).map(|v| v + 3).collect();
    let bind = |n: usize| rows_batch(&x, &y, n);
    let mut points = Vec::new();
    for &n in &SWEEP {
        let batch = bind(n);
        // A fresh driver pool per point, the protocol both gated benches bind
        // under. Without it every point after the first is timed against a pool
        // this sweep itself grew, and the compiled tier's per-call cost — the
        // thing the entry names — rises with it.
        reset_persistent_state();
        let bound = program.bind_per_row(&batch).expect("binds");
        points.push(point(&bound, n));
    }
    let n = stage_point(&points).unwrap_or(SWEEP[0]);
    let batch = bind(n);
    reset_persistent_state();
    let bound = program.bind_per_row(&batch).expect("binds");
    let (raw, keys, entered, call_ns, call_shots) = arms(&bound);
    split(label, &points, raw, keys, n, entered, call_ns, call_shots)
}

/// A comprehension at ONE row whose list carries a rising number of elements.
fn split_elems(label: &'static str, source: &'static str) -> Split {
    let mut schema = Schema::new();
    schema.insert("list[]".to_string(), ValType::Int);
    let program = BatchProgram::compile(source, &schema).expect("lowers");
    let top = SWEEP[SWEEP.len() - 1];
    let flat: Vec<i64> = (0..top as i64).collect();
    let lens: Vec<i64> = (0..=top as i64).collect();
    let bind = |n: usize| elems_batch(&lens, &flat, n);
    let mut points = Vec::new();
    for &n in &SWEEP {
        let batch = bind(n);
        reset_persistent_state();
        let bound = program.bind_per_row(&batch).expect("binds");
        points.push(point(&bound, n));
    }
    let n = stage_point(&points).unwrap_or(SWEEP[0]);
    let batch = bind(n);
    reset_persistent_state();
    let bound = program.bind_per_row(&batch).expect("binds");
    let (raw, keys, entered, call_ns, call_shots) = arms(&bound);
    split(label, &points, raw, keys, n, entered, call_ns, call_shots)
}

/// Run one call per majit arm and report the pass counters the door published,
/// then stop. NO timing.
///
/// The arms are selected at run time, so a build that carries the feature is
/// not yet evidence that any arm was REACHED — a stage figure differenced out
/// of two arms that ran the same code describes the box and nothing else. This
/// is that evidence, and it takes a second rather than a sweep.
#[cfg(feature = "__entry-stage-probe")]
fn armcheck() {
    let mut schema = Schema::new();
    schema.insert("x".to_string(), ValType::Int);
    schema.insert("y".to_string(), ValType::Int);
    let program = BatchProgram::compile("x + 1", &schema).expect("lowers");
    let top = SWEEP[SWEEP.len() - 1];
    let x: Vec<i64> = (0..top as i64).collect();
    let y: Vec<i64> = (0..top as i64).map(|v| v + 3).collect();
    let batch = rows_batch(&x, &y, SWEEP[0]);
    reset_persistent_state();
    let bound = program.bind_per_row(&batch).expect("binds");
    let mut out: Vec<Value> = Vec::new();
    let entered = warm(&bound, &mut out);
    let mut want: Vec<Value> = Vec::new();
    bound
        .collect_into_on(Tier::Clean, &mut want)
        .expect("clean witness");

    println!(
        "armcheck: `x + 1` at n={}, {entered:.2} of the probe's calls entered compiled code",
        SWEEP[0]
    );
    println!(
        "  per arm on ONE call: back-edge [gate, in, out, barrier], \
         exec [prologue, decode, barrier], frames\n"
    );
    for (i, arm) in STAGE_ARMS.iter().enumerate().skip(MAJIT_FIRST) {
        // Covers both majit groups: the back-edge arms and the four inside E3.
        let mut amplified = Repeats::default();
        arm(&mut amplified);
        let before = back_edge_stage_passes();
        let exec_before = execute_stage_passes();
        let frames_before = frame_build_passes();
        set_repeats(amplified);
        let mut got: Vec<Value> = Vec::new();
        bound
            .collect_into_on(Tier::Jit, &mut got)
            .expect("armcheck run");
        set_repeats(Repeats::default());
        // An amplification that changed the answer is a broken arm, not a
        // faster number — the same convention `arms` binds its witness under.
        assert!(got == want, "arm {} changed the answer", STAGE_LABELS[i]);
        let after = back_edge_stage_passes();
        let delta: Vec<u64> = before.iter().zip(after).map(|(b, a)| a - b).collect();
        let exec_after = execute_stage_passes();
        let exec_delta: Vec<u64> = exec_before
            .iter()
            .zip(exec_after)
            .map(|(b, a)| a - b)
            .collect();
        let frames = frame_build_passes() - frames_before;
        let reached =
            delta.iter().any(|d| *d > 0) || exec_delta.iter().any(|d| *d > 0) || frames > 0;
        println!(
            "  {:<22} back-edge {delta:?} exec {exec_delta:?} frames {frames:<4} {}",
            STAGE_LABELS[i],
            if reached { "REACHED" } else { "NOT REACHED" }
        );
    }

    // D's sub-arms are cel-side, so majit's counters cannot see them and the
    // loop above skips them. Their own counter is what proves them reached.
    // The fifth slot is not a pass count: it is how many loop keys resolved to
    // something other than their raw hash. Zero is what licenses D2/D3/D4
    // asking on the hash instead of paying the walk again.
    println!("\n  per arm on ONE call: cel sub-arms [walk, token, upgrade, meta, chained]\n");
    for i in D_SUB {
        let mut amplified = Repeats::default();
        STAGE_ARMS[i](&mut amplified);
        reset_entry_stage_sub_passes();
        set_repeats(amplified);
        let mut got: Vec<Value> = Vec::new();
        bound
            .collect_into_on(Tier::Jit, &mut got)
            .expect("armcheck run");
        set_repeats(Repeats::default());
        assert!(got == want, "arm {} changed the answer", STAGE_LABELS[i]);
        let sub = entry_stage_sub_passes();
        let reached = sub.iter().any(|d| *d > 0);
        println!(
            "  {:<26} sub {sub:?} {}",
            STAGE_LABELS[i],
            if reached { "REACHED" } else { "NOT REACHED" }
        );
    }
}

#[cfg(not(feature = "__entry-stage-probe"))]
fn armcheck() {
    println!("armcheck needs `__entry-stage-probe`; this binary has no arms to reach.");
}

/// Every arm must be differenced against an arm that is actually a BARRIER.
///
/// The pass counters prove an arm ran; nothing proved the arithmetic pointed
/// anywhere sensible. When arms were inserted, the barriers moved and
/// `barrier_for` kept returning their old indices, so stages were differenced
/// against other STAGES -- and the run printed a full table of confident
/// negative nanoseconds rather than failing. This is the check that catches
/// that class, and it costs one pass over a 17-element array.
fn check_barrier_wiring() {
    for i in 0..STAGE_LABELS.len() {
        let b = Split::barrier_for(i);
        assert!(
            STAGE_LABELS[b].contains("barrier"),
            "arm `{}` is differenced against `{}`, which is not a barrier",
            STAGE_LABELS[i],
            STAGE_LABELS[b]
        );
    }
    // The E3 block prints these three by index under hand-written prose, so a
    // stale index there shows another stage's number under this one's name.
    for (idx, want) in [(E3A, "E3a"), (E3B, "E3b"), (E3D, "E3d")] {
        assert!(
            STAGE_LABELS[idx].starts_with(want),
            "E3 label slot holds `{}`, expected the {want} arm",
            STAGE_LABELS[idx]
        );
    }
    // `Split::measured` sums 0..4 as "the entry's four stages" and
    // `Split::e_measured` sums MAJIT_FIRST..MAJIT_BARRIER as "the measured part
    // of E". Neither range can be derived from anything that moves with an
    // insertion, so both are pinned here by the labels they are meant to cover.
    for (idx, want) in [(0, "A "), (1, "BF"), (2, "C "), (3, "D ")] {
        assert!(
            STAGE_LABELS[idx].starts_with(want),
            "entry stage slot {idx} holds `{}`, which `Split::measured` would sum as `{want}`",
            STAGE_LABELS[idx]
        );
    }
    for (idx, want) in [
        (MAJIT_FIRST, "E1"),
        (MAJIT_FIRST + 1, "E2"),
        (MAJIT_FIRST + 2, "E4"),
    ] {
        assert!(
            STAGE_LABELS[idx].starts_with(want),
            "E stage slot {idx} holds `{}`, which `Split::e_measured` would sum as `{want}`",
            STAGE_LABELS[idx]
        );
    }
}

fn main() {
    check_barrier_wiring();
    #[cfg(feature = "__entry-stage-probe")]
    {
        if let Ok(raw) = std::env::var("ENTRYPROBE_REPEAT") {
            let k: u32 = raw
                .parse()
                .unwrap_or_else(|_| panic!("ENTRYPROBE_REPEAT is not an integer: {raw:?}"));
            // The majit-side arms hold their counts in a `u16`, so the cast in
            // `STAGE_ARMS` has to be lossless or those arms would silently ask
            // for a different count than the cel-side ones.
            assert!(
                k > 0 && k <= u32::from(u16::MAX),
                "ENTRYPROBE_REPEAT out of range: {k}"
            );
            REPEAT_K.store(k, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if std::env::args().nth(1).as_deref() == Some("armcheck") {
        armcheck();
        return;
    }
    println!("the JIT entry, itemised: four measured stages and one residual");
    println!(
        "best of {ROUNDS} interleaved batches of >= {} ms per arm; entry read as \
         jit_fix - clean_fix\nover n in {SWEEP:?}, a fresh driver pool per point; \
         batch.rs ships JIT_ENTRY_PS = {JIT_ENTRY_PS} ps.",
        MIN_BATCH.as_millis()
    );
    #[cfg(not(feature = "__entry-stage-probe"))]
    println!(
        "\n⚠ built WITHOUT `__entry-stage-probe`: the entry is reported whole and every stage\n\
         \x20 reads NaN. Rebuild with --features jit-cranelift,__entry-stage-probe for the split."
    );
    #[cfg(feature = "__entry-stage-probe")]
    {
        let k = repeat();
        println!("each stage repeated {k} extra times per call; delta / {k} is the stage.");
    }

    let splits = vec![
        split_rows("rows: x + 1", "x + 1"),
        split_rows(
            "rows: x > 3 && y < 90 || x == y",
            "x > 3 && y < 90 || x == y",
        ),
        split_rows(
            "rows: x*2 + x*3 + y*4 + y*5 + x*6 + y*7",
            "x*2 + x*3 + y*4 + y*5 + x*6 + y*7",
        ),
        split_elems("elems: list.all(e, e > 0)", "list.all(e, e > 0)"),
        split_elems("elems: list.map(e, e + 1)", "list.map(e, e + 1)"),
        split_elems(
            "elems: list.filter(e, e % 2 == 0)",
            "list.filter(e, e % 2 == 0)",
        ),
    ];

    println!(
        "\n  {:<42} {:>6} {:>4} {:>9} {:>9} {:>8}",
        "shape", "enter", "n", "clean fix", "jit fix", "entry"
    );
    for s in &splits {
        println!(
            "  {:<42} {:>6.2} {:>4} {:>9.1} {:>9.1} {:>8.1}",
            s.label,
            s.entered,
            s.stage_n,
            s.clean_fix,
            s.jit_fix,
            s.entry()
        );
    }

    println!("\n  ns per call, by stage (barrier already subtracted from the four)");
    println!(
        "  {:<42} {:>8} {:>8} {:>8} {:>8} {:>9} {:>10}",
        "shape", "A tls", "BF pool", "C reseed", "D keys", "barrier", "E RESIDUAL"
    );
    for s in &splits {
        println!(
            "  {:<42} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>9.2} {:>10.2}",
            s.label,
            s.stage(0),
            s.stage(1),
            s.stage(2),
            s.stage(D_IDX),
            s.raw[CEL_BARRIER],
            s.residual()
        );
    }

    println!("\n  ns per call, inside E (majit barrier already subtracted from the three)");
    println!(
        "  {:<42} {:>8} {:>8} {:>8} {:>9} {:>11}",
        "shape", "E1 gate", "E2 in", "E4 out", "barrier", "E3 RESIDUAL"
    );
    for s in &splits {
        println!(
            "  {:<42} {:>8.2} {:>8.2} {:>8.2} {:>9.2} {:>11.2}",
            s.label,
            s.stage(MAJIT_FIRST),
            s.stage(MAJIT_FIRST + 1),
            s.stage(MAJIT_FIRST + 2),
            s.raw[MAJIT_BARRIER],
            s.e_residual()
        );
    }

    let entry = median(splits.iter().map(Split::entry).collect());
    let stage_med: Vec<f64> = (0..4)
        .map(|i| median(splits.iter().map(|s| s.stage(i)).collect()))
        .collect();
    let barrier = median(splits.iter().map(|s| s.raw[CEL_BARRIER]).collect());
    let measured: f64 = stage_med.iter().sum();
    let residual = entry - measured;
    let (lo, hi) = splits
        .iter()
        .map(Split::entry)
        .fold((f64::MAX, f64::MIN), |(lo, hi), e| (lo.min(e), hi.max(e)));

    println!("\n  ns per call, inside E3 (execute barrier already subtracted from the two)");
    println!(
        "  {:<42} {:>8} {:>10} {:>10} {:>8} {:>10}",
        "shape", "E3a", "E3c call*", "E3b in c*", "E3d", "E3 rest"
    );
    for s in &splits {
        println!(
            "  {:<42} {:>8.2} {:>10.2} {:>10.2} {:>8.2} {:>10.2}",
            s.label,
            s.stage(E3A),
            s.call_ns,
            s.stage(E3B),
            s.stage(E3D),
            s.e3_residual()
        );
    }
    println!(
        "  * E3c is SINGLE-SHOT, one clocked reading per entry. E3b is AMPLIFIED and is a PART\n  \
         \x20 of E3c, not a sibling — it is never added into E3 rest. Median of the two\n  \
         \x20 amplified stages E3a+E3d: {:.2} ns.",
        median(splits.iter().map(Split::e3_amplified).collect())
    );

    println!(
        "\n  MEDIAN over {} shapes, and each as a share of the entry",
        splits.len()
    );
    println!("    entry (jit_fix - clean_fix)  {entry:8.2} ns   spread {lo:.1} .. {hi:.1}");
    for (i, ns) in stage_med.iter().enumerate() {
        println!(
            "    {:<28} {ns:8.2} ns   {:5.1}%",
            STAGE_LABELS[i],
            100.0 * ns / entry
        );
    }
    println!(
        "    {:<28} {measured:8.2} ns   {:5.1}%   <- A + BF + C + D",
        "measured, summed",
        100.0 * measured / entry
    );
    println!(
        "    {:<28} {residual:8.2} ns   {:5.1}%   <- RESIDUAL, not a measurement",
        "E back_edge_resolved",
        100.0 * residual / entry
    );
    println!(
        "    {:<28} {barrier:8.2} ns   <- what one amplification pass costs by itself",
        "(cel barrier)"
    );

    let e_stage_med: Vec<f64> = (MAJIT_FIRST..MAJIT_BARRIER)
        .map(|i| median(splits.iter().map(|s| s.stage(i)).collect()))
        .collect();
    let e_barrier = median(splits.iter().map(|s| s.raw[MAJIT_BARRIER]).collect());
    let e_measured: f64 = e_stage_med.iter().sum();
    let e_residual = residual - e_measured;
    println!("\n  E SPLIT — the same medians, inside the {residual:.2} ns residual above");
    for (i, ns) in e_stage_med.iter().enumerate() {
        println!(
            "    {:<28} {ns:8.2} ns   {:5.1}% of E   {:5.1}% of entry",
            STAGE_LABELS[MAJIT_FIRST + i],
            100.0 * ns / residual,
            100.0 * ns / entry
        );
    }
    println!(
        "    {:<28} {e_measured:8.2} ns   {:5.1}% of E   <- E1 + E2 + E4",
        "measured, summed",
        100.0 * e_measured / residual
    );
    println!(
        "    {:<28} {e_residual:8.2} ns   {:5.1}% of E   <- RESIDUAL: frame build, the ABI\n    \
         {:<28}                             wrapper, the trace, the deadframe decode",
        "E3 the call",
        100.0 * e_residual / residual,
        ""
    );
    println!(
        "    {:<28} {e_barrier:8.2} ns   <- what one majit amplification pass costs by itself",
        "(majit barrier)"
    );

    // ── inside E3 ─────────────────────────────────────────────────────────
    // ⚠ Two KINDS of number here, and they are labelled because they are not
    // comparable: E3a and E3d are amplified, the call is a single-shot clock
    // reading, and E3b is a PART of the call rather than a sibling of it, so it
    // is printed as "of which" and never added into the sum.
    let e3a = median(splits.iter().map(|s| s.stage(E3A)).collect());
    let e3b = median(splits.iter().map(|s| s.stage(E3B)).collect());
    let e3d = median(splits.iter().map(|s| s.stage(E3D)).collect());
    let e3_barrier = median(splits.iter().map(|s| s.raw[EXEC_BARRIER]).collect());
    let call = median(splits.iter().map(|s| s.call_ns).collect());
    let e3 = e_residual;
    let e3_rest = e3 - e3a - e3d - call;
    println!("\n  E3 SPLIT — inside the {e3:.2} ns residual above");
    println!(
        "    {:<28} {e3a:8.2} ns   {:5.1}% of E3   AMPLIFIED",
        STAGE_LABELS[E3A],
        100.0 * e3a / e3
    );
    println!(
        "    {:<28} {call:8.2} ns   {:5.1}% of E3   SINGLE-SHOT, {} readings/shape",
        "E3c the call",
        100.0 * call / e3,
        splits.first().map_or(0, |s| s.call_shots)
    );
    // `frame_build_passes` exists only with `__entry-stage-probe`. Without
    // that feature the import is configured out, and this example still has
    // to build: the frame-build line is then the not-amplifiable row.
    let frame_build_reached = {
        #[cfg(feature = "__entry-stage-probe")]
        {
            frame_build_passes() != 0
        }
        #[cfg(not(feature = "__entry-stage-probe"))]
        {
            false
        }
    };
    if !frame_build_reached {
        println!(
            "    {:<28} {:>8}      GC-managed JITFRAME: not safely amplifiable",
            "  of which frame build", "n/a"
        );
    } else {
        println!(
            "    {:<28} {e3b:8.2} ns   {:5.1}% of E3   AMPLIFIED, and INSIDE the call above",
            "  of which frame build",
            100.0 * e3b / e3
        );
    }
    println!(
        "    {:<28} {e3d:8.2} ns   {:5.1}% of E3   AMPLIFIED",
        STAGE_LABELS[E3D],
        100.0 * e3d / e3
    );
    println!(
        "    {:<28} {e3_rest:8.2} ns   {:5.1}% of E3   <- RESIDUAL: result construction\n    \
         {:<28}                             and the drops",
        "E3 rest",
        100.0 * e3_rest / e3,
        ""
    );
    println!(
        "    {:<28} {e3_barrier:8.2} ns   <- what one execute-side amplification pass costs",
        "(execute barrier)"
    );
    if call.is_finite() {
        println!(
            "    ⚠ the call is a CLOCKED reading and the two beside it are AMPLIFIED ones.\n    \
             \x20 An amplified figure is a warm-repeat LOWER bound; the clocked one carries the\n    \
             \x20 clock pair's residue and is an UPPER bound. Do not read the three as one column."
        );
    }
    #[cfg(feature = "__entry-stage-probe")]
    println!(
        "    passes reached: exec {:?}, frame builds {}",
        execute_stage_passes(),
        frame_build_passes()
    );
    #[cfg(feature = "__entry-stage-probe")]
    println!(
        "    amplified passes reached, [gate, in, out, barrier]: {:?}",
        back_edge_stage_passes()
    );

    if measured > entry {
        println!(
            "\n  ⚠ THE MODEL IS REFUTED ON THESE NUMBERS: the four measured stages sum to\n    \
             {measured:.2} ns, which is more than the {entry:.2} ns entry they are supposed to be\n    \
             inside. Either a stage's amplification prices something the call does not do\n    \
             once, or the entry read is too small."
        );
    }

    let keys = splits.iter().map(|s| s.loop_keys).max().unwrap_or(0);
    println!("\n  cross-checks");
    println!(
        "    entry here {entry:.1} ns vs JIT_ENTRY_PS {:.1} ns ({:+.1}%). This binary carries\n    \
         the probe feature: one thread-local read and five zero-trip loop tests per JIT\n    \
         call, none of it on the Tier::Clean arm, so all of it is inside jit_fix.",
        JIT_ENTRY_PS as f64 / 1000.0,
        100.0 * (entry - JIT_ENTRY_PS as f64 / 1000.0) / (JIT_ENTRY_PS as f64 / 1000.0),
    );
    for s in &splits {
        if s.loop_keys > 0 {
            println!(
                "    D per key, {:<42} {:6.2} ns over {} key(s)",
                s.label,
                s.stage(D_IDX) / s.loop_keys as f64,
                s.loop_keys
            );
        }
    }
    if keys == 0 {
        println!("    D per key: the door published no loop-key count (probe feature off).");
    }

    // The split D was extended for: which half of the yield scan carries it.
    //
    // Printed as its own block and NOT added into the entry's sum, because
    // these three are what D is made of. `walk + token + meta` against D is a
    // CHECK on the split; the short-circuit named on `EntryStageRepeats`
    // is why it is expected close rather than exact.
    let d = median(splits.iter().map(|s| s.stage(D_IDX)).collect());
    let walk = median(splits.iter().map(|s| s.stage(D_SUB.start)).collect());
    let token = median(splits.iter().map(|s| s.stage(D_SUB.start + 1)).collect());
    let upgrade = median(splits.iter().map(|s| s.stage(D_SUB.start + 2)).collect());
    let meta = median(splits.iter().map(|s| s.stage(D_SUB.start + 3)).collect());
    let pct = |v: f64| if d != 0.0 { 100.0 * v / d } else { f64::NAN };
    println!("\n  D split — medians over the shapes above, PARTS of D and not entries beside it");
    println!("    D  loop-key yield scan       {d:7.2} ns");
    println!(
        "      D1 walk  key.resolve       {walk:7.2} ns  {:5.1}% of D",
        pct(walk)
    );
    println!(
        "      D2 token has_compiled_loop {token:7.2} ns  {:5.1}% of D   celltable lookup, \
         Weak::upgrade, the two flag loads, and the Arc drop",
        pct(token)
    );
    println!(
        "      D4 upgrade Weak+drop       {upgrade:7.2} ns  {:5.1}% of D   the same lookup, the \
         Weak::upgrade and the Arc drop, WITHOUT the flag reads",
        pct(upgrade)
    );
    println!(
        "      D3 meta  get_compiled_meta {meta:7.2} ns  {:5.1}% of D   one compiled_loops \
         hash and probe",
        pct(meta)
    );
    // The four buckets the design choice is pre-registered against. D4 is a
    // PART of D2, not a sibling of it, so the sum below uses D2 and reports
    // the flag loads as the D2 - D4 difference rather than adding D4 in.
    println!(
        "\n      the four buckets:  walk {:7.2}   upgrade+drop {:7.2}   flag loads {:7.2}   \
         compiled_loops {:7.2}",
        walk,
        upgrade,
        token - upgrade,
        meta
    );
    println!(
        "      ⚠ `flag loads` is D2 - D4, so it carries both arms' error; and D4's answer is \
         WEAKER\n        than the door's (an invalidated token still upgrades), so it is a cost \
         probe, never a route."
    );
    let sum = walk + token + meta;
    println!(
        "      check: D1+D2+D3 = {sum:.2} vs D {d:.2} ({:+.1}%)",
        if d != 0.0 {
            100.0 * (sum - d) / d
        } else {
            f64::NAN
        }
    );
    println!(
        "      ⇒ walk {:.0}% vs lookups {:.0}% of D.",
        pct(walk),
        pct(token + meta)
    );
    // Cumulative over every call this process made, which is the right
    // denominator: the claim is about the corpus, not about one shape.
    #[cfg(feature = "__entry-stage-probe")]
    let chained = entry_stage_sub_passes()[4];
    #[cfg(not(feature = "__entry-stage-probe"))]
    let chained = 0u64;
    println!(
        "      chained buckets: {chained}  (resolve(driver) != key.hash).  ZERO is what licenses\n      \
         asking D2/D3/D4 on the raw hash; any other number and those three arms stopped\n      \
         being about the cell the door asks about."
    );
}
