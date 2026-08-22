//! The dispatch loop.
//!
//! [`cel_eval_loop`] is a free function taking the code object and the
//! context, not a method: the portal a tracing JIT splits a graph at is
//! derived from its argument list, and a `&self` receiver is neither a green
//! nor a red. [`Program::execute`] stays a thin wrapper over it.
//!
//! # Parity is the whole specification
//!
//! Every arm here is the corresponding arm of the tree walker
//! (`objects.rs`), reached by opcode instead of by `Expr` variant, and the
//! frozen oracle corpus holds the two to the same answers. Where the walker
//! computes something with a helper, this calls the same helper rather than
//! reimplementing it -- `value_index`, `value_field`, `binary_values` and the
//! rest are shared, so a fix to either evaluator is a fix to both.
//!
//! # The operand stack holds a builder, not only a value
//!
//! A list, map or struct literal is built incrementally: allocate, then one
//! append per element (see [`OpCode::NewList`]). Rebuilding the finished
//! [`Value`] after every element would make an *n*-element literal quadratic,
//! so the in-progress aggregate stays on the stack as an [`Operand`] variant
//! the append can mutate in place, and becomes a [`Value`] at the point
//! something pops it.
//!
//! [`Program::execute`]: crate::Program::execute

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::code::{CelCode, Handler, Insn};
use super::error::{CelErr, CelResult, ColdId, NameId};
use super::opcode::OpCode;
use crate::context::Context;
use crate::objects::{
    as_optional, binary_values, compare_values, optional_none, optional_of, value_contains,
    value_field, value_index, value_iter, value_key, value_negate, Key, Map,
};
use crate::{ExecutionError, Value};

use std::cmp::Ordering;

/// Run `code` in `ctx`.
///
/// The `Result` is reconstructed here, at the boundary: inside the loop an
/// error is a [`CelErr`], which is [`Copy`] and carries no allocation, because
/// `&&` and `||` absorb errors and so raise them on ordinary control flow.
pub fn cel_eval_loop(code: &CelCode, ctx: &Context) -> Result<Value, ExecutionError> {
    let mut vm = Vm::new(code, ctx);
    #[cfg(feature = "drop-arm-probe")]
    {
        vm.probe = PROBE.with(std::cell::Cell::get);
    }
    #[cfg(feature = "elem-attr-probe")]
    {
        // The shape was recognised unconditionally in `Vm::new`, so every arm
        // pays the scan. What the arm decides is only whether the fused path
        // is REACHABLE: an arm that fuses nothing moves the anchor out of the
        // instruction stream's range instead of skipping the test, so the test
        // itself is present and perfectly predicted in every arm.
        vm.fuse = FUSE.with(std::cell::Cell::get);
        vm.anchor = if vm.shape.top == u32::MAX || vm.fuse == FuseArm::None {
            u32::MAX
        } else if vm.fuse == FuseArm::AdvanceOnly {
            vm.shape.after_body
        } else {
            // [`FuseArm::AllButBody`] starts here too and then moves the anchor
            // itself, because its second entry point is only reachable once the
            // body has run.
            vm.shape.top
        };
    }
    match vm.run() {
        Ok(value) => Ok(value),
        Err(err) => Err(vm.public_error(err)),
    }
}

/// Run `code` in `ctx` under an explicit probe policy.
///
/// The probe's only door, and it goes through the ordinary one rather than
/// building a machine of its own: `tests/one_dispatch_loop.rs` pins this file
/// to exactly one `Vm::new` and one `run`, because a second construction site
/// is what a nested interpreter looks like. So the policy is announced in
/// advance instead of passed down, which is what [`PROBE`] is for.
///
/// Saved and restored around the one call, so two arms interleaved in one
/// process never observe each other's setting, and an evaluation a host
/// function starts from inside this one inherits the enclosing policy rather
/// than a stale one. A panic escaping the evaluation leaves the policy set;
/// that is a probe, not a library, and the process is going down anyway.
#[cfg(feature = "drop-arm-probe")]
pub fn cel_eval_loop_with_probe(
    code: &CelCode,
    ctx: &Context,
    probe: ProbePolicy,
) -> Result<Value, ExecutionError> {
    let previous = PROBE.with(|slot| slot.replace(probe));
    let out = cel_eval_loop(code, ctx);
    PROBE.with(|slot| slot.set(previous));
    out
}

/// Which groups of an appending comprehension's per-element instruction block
/// the dispatch loop runs as ONE step.
///
/// A measurement probe, not a feature, and it exists to answer one question:
/// the bytecode VM's cost over the tree walker on `list.map(x, x * k)` is a
/// FLAT per-element excess that does not shrink with the list's length, and
/// nothing named accounts for it. The excess can only be dispatch, operand
/// stack traffic, or work the walker does not do -- so each arm here removes
/// one named group of dispatches and operand-stack round trips from the
/// per-element block and leaves everything else exactly where it was.
///
/// The rule every arm obeys: **an arm may remove a dispatch or an operand-stack
/// round trip; it may not remove work the tree walker also performs.**
/// `binary_values` is called by both evaluators, so every arm below still calls
/// it, with the same operands, at the same point. `compare_values` is the one
/// exception, and it gets its own arm precisely so that its cost is separated
/// rather than folded into a dispatch figure -- the loop guard it serves has no
/// counterpart in the walker at all, which iterates with a Rust iterator.
///
/// The arms are cumulative, [`FuseArm::GuardKeepingCompare`] excepted: each
/// fuses everything the one before it fused, plus one more group. Marginal
/// differences are therefore the per-group figures and the end-to-end
/// difference is their sum, which is an additivity check.
///
/// All four groups have since become single instructions --
/// [`OpCode::IterGuard`], [`OpCode::IterBind`],
/// [`OpCode::MulLocalConstAppend`] and [`OpCode::IterAdvance`] -- so the
/// operand-stack round trips those groups used to carry are gone from the
/// STOCK arm too, and what each of the four arms below now removes is one
/// dispatch and nothing else. The block holds no push and no pop at all: the
/// only operand it touches is the builder the append mutates in place, which
/// was put on the stack before the loop and comes off after it.
#[cfg(feature = "elem-attr-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FuseArm {
    /// What the interpreter does without the probe: `IterGuard ; IterBind ;
    /// MulLocalConstAppend ; IterAdvance`, stepped one at a time.
    ///
    /// Named rather than counted, and every arm below names what it fuses for
    /// the same reason: `recognize_map_loop`'s `WANT` is that list and will
    /// disagree out loud when the lowering moves, where a restated count is
    /// checkable against nothing. One already had outlived its lowering here.
    None,
    /// Fuse `IterGuard index source done` -- one dispatch -- and decide the
    /// guard by calling `compare_values` and `as_bool` on the two
    /// `Value::Int`s, then discarding the answer, as the four instructions
    /// `IterGuard` replaced did.
    ///
    /// A POSITIVE CONTROL rather than a rung, and it stopped being a rung when
    /// the guard became one instruction that compares two `i64`s. This arm
    /// RE-ADDS the helpers, so it can cost MORE than [`FuseArm::None`], and its
    /// distance from [`FuseArm::None`] is not a fusion figure. Its difference
    /// from [`FuseArm::Guard`] is what it claims and all it claims, and that
    /// difference is unchanged: exactly what `compare_values` + `as_bool` + the
    /// discard cost. That is the price of the guard's `i64` comparison, taken
    /// from the other side.
    GuardKeepingCompare,
    /// Fuse `IterGuard index source done` -- one dispatch. There is no stack
    /// traffic left in the group to remove; the instruction has none.
    Guard,
    /// ... plus `IterBind source index var` -- one dispatch. The element is
    /// still read through `ListRef::get` and still written into the slot with
    /// the same `mem::replace` and discard.
    Bind,
    /// ... plus `MulLocalConstAppend var k` -- one dispatch. The pair this
    /// replaced, `MulLocalConst var k ; ListAppend`, was the last group in the
    /// block that put a value on the operand stack and took it off again;
    /// what is left is the operator handing its answer to the builder
    /// directly. `binary_values` is still called, with the same two operands,
    /// and the result still reaches the same builder.
    Body,
    /// ... plus `IterAdvance index top` -- one dispatch. The whole element is
    /// one step, and no dispatch at all is left in it. The counter is still
    /// advanced with the same `checked_add`.
    Advance,
    /// `IterAdvance index top` fused and NOTHING ELSE, anchored at the
    /// instruction after the body rather than at the loop header.
    ///
    /// The order control. Every other arm is cumulative, so each group's
    /// marginal price is taken against an arm that has already had the groups
    /// before it removed -- and a marginal taken in one order need not equal the
    /// same marginal taken in another. This measures the advance group from the
    /// STOCK end, where [`FuseArm::Advance`] measures it from the most-fused
    /// end. Two figures that agree say the split does not depend on the order;
    /// two that disagree bound how much it does.
    AdvanceOnly,
    /// Every group but the BODY: the guard, the bind and the advance all run
    /// inline, and the body operator keeps its dispatch.
    ///
    /// Measured DIRECTLY against [`FuseArm::None`], as one paired difference,
    /// rather than assembled from the ladder's marginals. That is the whole
    /// reason it exists. The cumulative ladder prices each group against an arm
    /// that has already had the earlier ones removed, and two of its own checks
    /// say those marginals cannot then be recombined: the order control
    /// disagrees with the ladder's advance figure in every run taken so far,
    /// and the four marginals sum to MORE than the measured end-to-end
    /// difference. Both failures are arithmetic over separately measured arms,
    /// and this arm has no arithmetic in it.
    ///
    /// It is also the shape of a lowering that stays general. Fusing the body
    /// in as well needs one opcode per body operator, which is the growth
    /// [`OpCode`]'s own design refuses; leaving it dispatched needs none,
    /// because the guard, the bind and the advance are the same three
    /// instructions whatever the body computes.
    ///
    /// TWO ENTRY POINTS, ONE ANCHOR. The body sits between the bind and the
    /// advance, so this cannot be a single straight run: the arm resumes at the
    /// body and has to be re-entered at [`MapLoop::after_body`]. It MOVES
    /// `Vm::anchor` between the two rather than adding a second field, which is
    /// what keeps the dispatch loop at exactly one comparison per instruction
    /// for every arm -- the property that field's doc comment exists to state.
    /// The two stores that costs are work no other arm does, so what this arm
    /// reports is if anything an UNDER-statement of the fusion, never an
    /// over-statement.
    AllButBody,
    /// A POSITIVE CONTROL, and the calibration for the "a per-element `Arc`
    /// clone" hypothesis. It re-adds one of the four atomic refcount operations
    /// per element that giving [`OpCode::IterLen`] and [`OpCode::IterAt`] slot
    /// operands removed, so the difference from [`FuseArm::Advance`] is what one
    /// such clone-and-release costs on this box. Its arm must be checked in the
    /// disassembly for an atomic read-modify-write: an arm meant to ADD work
    /// that the optimizer deleted would report the cost as zero.
    AdvancePlusArcRoundTrip,
}

#[cfg(feature = "elem-attr-probe")]
std::thread_local! {
    /// Which arm the next evaluation on this thread runs under.
    static FUSE: std::cell::Cell<FuseArm> = const { std::cell::Cell::new(FuseArm::None) };
}

/// Run `code` in `ctx` with `arm`'s groups fused.
///
/// The probe's only door, announced in advance rather than passed down, for the
/// reason [`cel_eval_loop_with_probe`]'s documentation gives: `interp.rs` is
/// pinned to exactly one `Vm::new` and one `run`, so a second entry point that
/// built its own machine would look like a nested interpreter.
#[cfg(feature = "elem-attr-probe")]
pub fn cel_eval_loop_with_fuse(
    code: &CelCode,
    ctx: &Context,
    arm: FuseArm,
) -> Result<Value, ExecutionError> {
    let previous = FUSE.with(|slot| slot.replace(arm));
    let out = cel_eval_loop(code, ctx);
    FUSE.with(|slot| slot.set(previous));
    out
}

/// The per-element instruction block of an appending comprehension whose body
/// is one binary operator against a constant, located in the instruction
/// stream.
///
/// Every field is read off the stream rather than assumed, so a program that
/// does not have this exact shape is simply not recognised and every arm runs
/// the stock dispatch loop over it.
#[cfg(feature = "elem-attr-probe")]
#[derive(Clone, Copy, Debug)]
struct MapLoop {
    /// `IterGuard`, the loop header. `u32::MAX` when no block matched, which is
    /// the value that keeps the fused path unreachable.
    top: u32,
    /// Where `IterGuard` sends an exhausted loop.
    done: u32,
    /// Slot holding the sequence.
    source: u32,
    /// Slot holding the loop counter.
    index: u32,
    /// Slot holding the iteration variable.
    var: u32,
    /// Constant-pool index of the body operator's folded right operand.
    konst: u32,
    /// `IterBind`, where an arm that fused only the guard resumes.
    after_guard: u32,
    /// The body operator, where an arm that also fused the bind resumes.
    after_bind: u32,
    /// `IterAdvance`, where an arm that also fused the body resumes.
    after_body: u32,
}

#[cfg(feature = "elem-attr-probe")]
const NO_MAP_LOOP: MapLoop = MapLoop {
    top: u32::MAX,
    done: 0,
    source: 0,
    index: 0,
    var: 0,
    konst: 0,
    after_guard: 0,
    after_bind: 0,
    after_body: 0,
};

/// Find the four-instruction per-element block, if the program has one.
///
/// Run once per evaluation by EVERY arm, so its cost is a constant that cancels
/// out of any difference between two of them.
#[cfg(feature = "elem-attr-probe")]
fn recognize_map_loop(code: &CelCode) -> MapLoop {
    const WANT: [OpCode; 4] = [
        OpCode::IterGuard,
        OpCode::IterBind,
        OpCode::MulLocalConstAppend,
        OpCode::IterAdvance,
    ];
    // A shift register rather than a collected stream: this runs once per
    // EVALUATION, and `tests/vm_scratch_pool.rs` pins the evaluator's heap
    // floor at zero allocations. A probe that allocates to decide where to
    // measure has changed the thing it is measuring.
    let mut window = [(0u32, OpCode::Return, [0u32; 3]); WANT.len()];
    let mut filled = 0usize;
    for (pc, op, operands) in code.instructions() {
        let mut words = [0u32; 3];
        for (slot, word) in words.iter_mut().zip(operands) {
            *slot = *word;
        }
        window.rotate_left(1);
        window[WANT.len() - 1] = (pc, op, words);
        filled += 1;
        if filled < window.len() {
            continue;
        }
        if !window.iter().zip(WANT).all(|(entry, want)| entry.1 == want) {
            continue;
        }
        let index = window[0].2[0];
        let source = window[0].2[1];
        let var = window[1].2[2];
        // Every slot the block names has to be the slot the fused form would
        // read, or the fusion is not of THIS loop. The guard now names both the
        // counter and the sequence itself, so the bind is checked against it
        // rather than the other way round.
        let consistent = window[1].2[0] == source
            && window[1].2[1] == index
            && window[2].2[0] == var
            && window[3].2[0] == index
            && window[3].2[1] == window[0].0;
        if !consistent {
            continue;
        }
        return MapLoop {
            top: window[0].0,
            done: window[0].2[2],
            source,
            index,
            var,
            // The body operator names both its operands, so the constant is
            // its SECOND word, behind the slot checked just above.
            konst: window[2].2[1],
            after_guard: window[1].0,
            after_bind: window[2].0,
            after_body: window[3].0,
        };
    }
    NO_MAP_LOOP
}

/// Whether the probe can fuse `code`'s per-element block.
///
/// The recogniser's own answer, exported so that a harness refusing to measure
/// a program that is not this block asks THE RECOGNISER rather than keeping a
/// second copy of the opcode window. Two copies in two files is a silent
/// failure waiting to happen and not a loud one: where they disagree, the
/// recogniser matches nothing, [`Vm::anchor`] stays out of the instruction
/// stream's range, EVERY arm runs the stock dispatch loop -- and every arm
/// still agrees on the answer, because they all compute the same value. The
/// harness would report a full ladder of zeroes as a measurement.
#[cfg(feature = "elem-attr-probe")]
pub fn map_loop_is_fusable(code: &CelCode) -> bool {
    recognize_map_loop(code).top != u32::MAX
}

/// One operand-stack entry.
///
/// Only the aggregate literals need anything but a [`Value`]; see the module
/// documentation.
enum Operand {
    Value(Value),
    List(Vec<Value>),
    /// Behind a pointer because an inline [`HashMap`] is 48 bytes and would
    /// set the width of every entry on the stack, including the
    /// [`Operand::Value`] that almost all of them are; the indirection takes
    /// the entry from 56 bytes to 32.
    ///
    /// The pointer is the [`Arc`] the finished [`Map`] holds, not a [`Box`],
    /// so the table is built where it lands. Both are 8 bytes here, but a
    /// `Box` is a different allocation from the one [`Map::object`] needs:
    /// closing the literal then had to allocate the `Arc`, move the 48-byte
    /// table into it and free the box -- one allocation and one move per map
    /// literal, spent only on handing the table over. `finish` now passes the
    /// same pointer through.
    ///
    /// An in-progress table is never shared -- the operand holds the only
    /// reference until `finish` gives it away -- so `Arc::get_mut` answers
    /// every insert; see [`Vm::map_mut`].
    Map(Arc<HashMap<Key, Value>>),
    /// The `names` index of the message type, and the fields set so far. The
    /// type is checked when the struct is opened, so that a bad type name
    /// fails before the field expressions run, as it does in the walker.
    ///
    /// Without the `structs` feature that same check refuses every struct
    /// literal outright, so `open_struct` -- the only constructor -- never
    /// returns and this variant is genuinely unreachable in that build. The
    /// arms below still have to compile, which is what the attribute is for.
    #[cfg_attr(not(feature = "structs"), allow(dead_code))]
    Struct(NameId, BTreeMap<String, Value>),
}

/// The stack entry must stay narrow, because `Vm::new` sizes the operand stack
/// at `max_stack` entries and almost every one of them holds a bare [`Value`].
///
/// Measured: putting `Map` behind a pointer alone takes this from 56 to 32.
/// Doing the same to `Struct` changes nothing -- its payload is 32 bytes and
/// the discriminant fits in the padding after `NameId` -- so a later variant
/// wider than [`Value`] costs 8 bytes on every entry and fails here rather
/// than in a benchmark.
const _: () = {
    assert!(core::mem::size_of::<Operand>() == 32);
};

/// Which drop policy [`Vm::discard`] applies to a discarded operand.
///
/// A measurement probe, not a feature. It exists to answer one question: an
/// operand the interpreter throws away is dropped through the out-of-line glue
/// for [`Value`], which on the integer path executes a handful of instructions
/// and branches that do nothing -- is the cost the CALL, or the work inside
/// it? The three policies bracket that. The difference between the first two
/// is the call; the difference between the second two is the work.
#[cfg(feature = "drop-arm-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DropArm {
    /// What the interpreter does without the probe: hand the operand to the
    /// glue, whatever it holds.
    Baseline,
    /// Test the discriminant at the call site and reach the glue only for the
    /// variants that own something. Keeps the work; removes the call on the
    /// trivial path.
    InlineDiscriminant,
    /// THIS ARM LEAKS. Forget the operand: no discriminant test, no call, and
    /// no release of anything it owned.
    ///
    /// Valid ONLY where nothing owning is ever discarded -- an integer body
    /// over a list of integers, and nothing else. On a case that carries a
    /// string, a list, a map, a struct or a record it leaks heap proportional
    /// to the element count and holds reference counts that decide whether an
    /// in-place string append is taken, so it corrupts the very timings it is
    /// there to produce. The caller is responsible for proving the case is
    /// safe; see the leak witness in `examples/paired_ab.rs`.
    ForgetUnsound,
}

/// Which lowering [`OpCode::IterAt`] uses to read one element.
///
/// The second half of the same probe, and here for the same reason: two
/// spellings of one instruction, chosen per run, so both live in one binary
/// and neither can be a different compilation of the other.
#[cfg(feature = "drop-arm-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IterAtArm {
    /// Hand both slots to `value_index`, which decides the container's kind,
    /// then the key's kind, then bounds-checks, then answers in the wide
    /// public error type -- which `Vm::park` has to record on `&mut self`.
    ViaValueIndex,
    /// What the interpreter does without the probe: two variant tests and one
    /// unsigned comparison, answering in [`CelErr`].
    KnownList,
}

/// Everything the probe selects, carried on the [`Vm`] and read at the site.
///
/// One field read and one perfectly-predicted branch per site, present
/// identically in every arm because every arm is the same compiled code taking
/// a different branch. Both cancel exactly out of any difference between two
/// arms -- which is also why no arm's ABSOLUTE figure is what the shipping
/// interpreter costs. Only differences are claims.
#[cfg(feature = "drop-arm-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProbePolicy {
    pub drop_arm: DropArm,
    pub iter_at: IterAtArm,
}

#[cfg(feature = "drop-arm-probe")]
impl ProbePolicy {
    /// What the interpreter does without the probe.
    ///
    /// A `const` as well as a [`Default`], because the thread-local below is
    /// initialised in a `const` block and `Default::default` is not callable
    /// there.
    pub const STOCK: ProbePolicy = ProbePolicy {
        drop_arm: DropArm::Baseline,
        iter_at: IterAtArm::KnownList,
    };
}

#[cfg(feature = "drop-arm-probe")]
impl Default for ProbePolicy {
    /// What the interpreter does without the probe, so that a run that names
    /// only one half leaves the other half alone.
    fn default() -> ProbePolicy {
        ProbePolicy::STOCK
    }
}

#[cfg(feature = "drop-arm-probe")]
std::thread_local! {
    /// The policy the next evaluation on this thread runs under.
    ///
    /// Read once per evaluation -- not once per instruction -- and identically
    /// by every arm, so it is a constant that cancels out of any difference
    /// between two of them.
    static PROBE: std::cell::Cell<ProbePolicy> =
        const { std::cell::Cell::new(ProbePolicy::STOCK) };
}

/// [`DropArm::InlineDiscriminant`]'s policy: test the discriminant here, and
/// reach the out-of-line glue only for the variants that own something.
///
/// The trivial branch is spelled with [`std::mem::forget`] rather than as an
/// empty match arm, and the difference is the whole arm. `match value { .. =>
/// {} }` does not MOVE `value` in an arm whose patterns bind nothing, so the
/// scrutinee is still live at the end of the match and is dropped there --
/// through the same glue, with the same call. Verified on the generated code:
/// the empty-arm spelling compiled to a discriminant test in front of two
/// paths that BOTH called `drop_glue::<Value>`, which is the baseline plus a
/// test rather than an alternative to it.
///
/// Forgetting is what makes the branch a real one, and it is sound for exactly
/// the variants listed: each is plain data, and the generated glue returns
/// immediately for every one of their discriminants. A variant added to
/// [`Value`] that owns anything falls to the `else`, because the list is
/// explicit rather than a wildcard.
#[cfg(feature = "drop-arm-probe")]
#[inline(always)]
fn discard_inline(value: Value) {
    #[cfg(feature = "chrono")]
    let trivial = matches!(
        value,
        Value::Int(_)
            | Value::UInt(_)
            | Value::Float(_)
            | Value::Bool(_)
            | Value::Null
            | Value::Duration(_)
            | Value::Timestamp(_)
    );
    #[cfg(not(feature = "chrono"))]
    let trivial = matches!(
        value,
        Value::Int(_) | Value::UInt(_) | Value::Float(_) | Value::Bool(_) | Value::Null
    );
    if trivial {
        std::mem::forget(value);
    } else {
        drop(value);
    }
}

/// The four growable buffers a run needs, kept across runs so that evaluating
/// a program does not allocate an activation record.
///
/// Sizing them per run is what an evaluation of `1 + 1` used to pay: `stack` is
/// at least one entry for any program that compiles, so a scalar expression
/// cost one malloc and one free, plus a second for `slots` where the program
/// has a comprehension and a third for `logic` where it has `&&` or `||`. The
/// tree walker's floor is zero — its operands are recursion locals — so this
/// was a cost the evaluator paid for being a loop rather than for doing work.
///
/// Only the CAPACITY is worth keeping. Every buffer is emptied when it is
/// returned, so nothing here holds a [`Value`] between runs; what survives is
/// the allocation, sized to the largest program this thread has evaluated.
#[derive(Default)]
struct Scratch {
    stack: Vec<Operand>,
    slots: Vec<Value>,
    logic: Vec<CelResult<bool>>,
    cold: Vec<ExecutionError>,
}

impl Scratch {
    /// Drop everything held, keeping the allocations.
    ///
    /// Called when the buffers go back to the pool rather than when they come
    /// out, so a run that ended in an error — or in a panic, since this is
    /// reached from [`Vm`]'s [`Drop`] — cannot leave a large list rooted until
    /// the next evaluation on this thread happens to overwrite it.
    fn release(&mut self) {
        self.stack.clear();
        self.slots.clear();
        self.logic.clear();
        self.cold.clear();
    }
}

std::thread_local! {
    /// This thread's idle buffers, or `None` while a run holds them.
    ///
    /// Held by VALUE and moved in and out, which is what makes re-entry
    /// correct rather than merely unlikely: a host function registered in the
    /// [`Context`] may call [`crate::Program::execute`] again from inside an
    /// evaluation, and the nested run finds this slot empty and builds its own
    /// buffers. Lending a `&mut` to a shared buffer instead would alias, and no
    /// arrangement of flags recovers from that — the outer run's operand stack
    /// is live across the call.
    ///
    /// Nesting therefore pays one set of allocations, and it is the nesting
    /// caller that pays. On the way out the innermost run to finish is the
    /// first to store, so the outermost run's buffers — the widest, and the
    /// ones a subsequent top-level call wants — are what the slot ends up
    /// holding.
    ///
    /// Reached through `try_with` on both sides, never `with`. `Scratch` has a
    /// destructor, so this slot can already have been torn down by the time a
    /// thread-local of the caller's runs an evaluation of its own from its own
    /// destructor. `with` would panic there, and a panic on the way out of
    /// [`Vm`] would be a panic in a destructor.
    static SCRATCH: std::cell::Cell<Option<Scratch>> = const { std::cell::Cell::new(None) };
}

struct Vm<'a> {
    code: &'a CelCode,
    ctx: &'a Context<'a>,
    stack: Vec<Operand>,
    slots: Vec<Value>,
    /// One entry per `&&`/`||`, holding the left operand's outcome: its bool,
    /// or the error the handler absorbed for it.
    logic: Vec<CelResult<bool>>,
    /// Full errors for the cases [`CelErr`] cannot spell.
    ///
    /// Written only when an error is raised, and truncated again when one is
    /// absorbed, so a comprehension whose body errors on every iteration does
    /// not accumulate a table the size of the sequence.
    cold: Vec<ExecutionError>,
    /// Arguments a missed [`OpCode::CallQualified`] popped, held for the
    /// [`OpCode::CallMethod`] the compiler emitted right after it.
    ///
    /// The pair is one call. The probe has to pop its arguments to ask
    /// `find_overload` about them, and the receiver path then wants the same
    /// values in the same order; re-pushing them for `pop_n` to rebuild is a
    /// second `Vec` per member call on an identifier receiver, which is an
    /// allocation the walker never pays -- it resolves its arguments once and
    /// lends the probe a slice.
    ///
    /// Live only across the `LoadVar` that loads the receiver, and cleared on
    /// both ways out: taken by [`OpCode::CallMethod`], and dropped by
    /// [`Vm::unwind`], which is where that load's error goes when a `&&`/`||`
    /// absorbs it and the method call never runs.
    pending_args: Option<Vec<Value>>,
    /// Which lowering the probe's sites take. Probe only; see [`ProbePolicy`].
    #[cfg(feature = "drop-arm-probe")]
    probe: ProbePolicy,
    /// Which groups of the per-element block run as one step. Probe only; see
    /// [`FuseArm`].
    #[cfg(feature = "elem-attr-probe")]
    fuse: FuseArm,
    /// Where that block is. Probe only; see [`MapLoop`].
    #[cfg(feature = "elem-attr-probe")]
    shape: MapLoop,
    /// The one `pc` at which the fused path is taken, or `u32::MAX` for an arm
    /// that fuses nothing. A field rather than a second test, so that the
    /// dispatch loop's per-instruction cost is identical in every arm however
    /// many anchors the probe grows. Probe only.
    #[cfg(feature = "elem-attr-probe")]
    anchor: u32,
}

/// Return the buffers to this thread's pool.
///
/// A [`Drop`] impl rather than a call at the end of [`cel_eval_loop`], because
/// the buffers have to come back on EVERY way out — the `Ok`, the `Err`, and a
/// panic from a host function called mid-evaluation. Taking them out and
/// putting them back is also what keeps re-entry sound: the slot is empty for
/// exactly as long as a run holds it.
impl Drop for Vm<'_> {
    fn drop(&mut self) {
        // `mem::take` on a `Vec` leaves a dangling-free empty one and allocates
        // nothing, which is the only way to move fields out of a type that
        // implements `Drop`. The operand stack goes through its own method
        // because it is the one buffer whose contents are not all in the `Vec`.
        let mut scratch = Scratch {
            stack: self.take_stack(),
            slots: std::mem::take(&mut self.slots),
            logic: std::mem::take(&mut self.logic),
            cold: std::mem::take(&mut self.cold),
        };
        scratch.release();
        // Dropped rather than pooled where the slot is already gone; see
        // `SCRATCH`.
        let _ = SCRATCH.try_with(|slot| slot.set(Some(scratch)));
    }
}

impl<'a> Vm<'a> {
    /// Borrow this thread's buffers and size them for `code`.
    ///
    /// Every buffer arrives empty — [`Scratch::release`] is what put it back —
    /// so this establishes the lengths the loop indexes into and asks the
    /// operand stack for the depth the compiler proved it needs. On a thread
    /// that has evaluated anything before, all four of those are already
    /// satisfied and none of them allocates.
    fn new(code: &'a CelCode, ctx: &'a Context<'a>) -> Self {
        let mut scratch = SCRATCH
            .try_with(std::cell::Cell::take)
            .ok()
            .flatten()
            .unwrap_or_default();
        scratch.stack.reserve(code.max_stack as usize);
        scratch.slots.resize(code.n_slots as usize, Value::Null);
        scratch
            .logic
            .resize(code.n_logic as usize, Err(CelErr::InternalError));
        Vm {
            code,
            ctx,
            stack: scratch.stack,
            slots: scratch.slots,
            logic: scratch.logic,
            cold: scratch.cold,
            pending_args: None,
            #[cfg(feature = "drop-arm-probe")]
            probe: ProbePolicy::default(),
            #[cfg(feature = "elem-attr-probe")]
            fuse: FuseArm::None,
            // Scanned by every arm, including the one that fuses nothing, so
            // the scan is a constant rather than a term of any difference.
            #[cfg(feature = "elem-attr-probe")]
            shape: recognize_map_loop(code),
            #[cfg(feature = "elem-attr-probe")]
            anchor: u32::MAX,
        }
    }

    // -- the error channel ------------------------------------------------

    /// Park a full error and return the [`CelErr`] that stands for it.
    fn park(&mut self, err: ExecutionError) -> CelErr {
        let id = u32::try_from(self.cold.len()).unwrap_or(u32::MAX);
        self.cold.push(err);
        CelErr::Cold(ColdId(id))
    }

    /// Discard a parked error that a merge has decided to throw away.
    ///
    /// Called at the merge, not when the handler catches: the recorded error
    /// is still live between those two points, because the merge may raise it
    /// after all. Sound because the merges nest -- anything the right operand
    /// parked has been resolved by its own merge before this one runs -- so a
    /// discarded `Cold` is always the last entry.
    fn unpark(&mut self, err: CelErr) {
        if let CelErr::Cold(ColdId(id)) = err {
            if id as usize + 1 == self.cold.len() {
                self.cold.pop();
            }
        }
    }

    /// Reconstruct the public error.
    ///
    /// Exhaustive over [`CelErr`], so a variant added without a public
    /// counterpart is a compile error here rather than a silent
    /// `InternalError` at run time.
    #[allow(deprecated)]
    fn public_error(&self, err: CelErr) -> ExecutionError {
        let name = |id: NameId| self.code.name(id).unwrap_or("?").to_string();
        // The operator name the public error carries is a property of the
        // opcode, which is why the compact form stores the opcode.
        let operator = |err: CelErr| match err.operator() {
            Some(OpCode::Add) => "add",
            Some(OpCode::Sub) => "sub",
            Some(OpCode::Mul) => "mul",
            Some(OpCode::Div) => "div",
            Some(OpCode::Mod) => "rem",
            Some(OpCode::Negate) => "negate",
            _ => "?",
        };
        match err {
            CelErr::Cold(ColdId(id)) => self
                .cold
                .get(id as usize)
                .cloned()
                .unwrap_or_else(|| ExecutionError::InternalError("cold error lost".to_string())),
            CelErr::NoSuchOverload => ExecutionError::NoSuchOverload,
            CelErr::MissingArgumentOrTarget => ExecutionError::MissingArgumentOrTarget,
            CelErr::DivisionByZero => ExecutionError::DivisionByZero(Value::Null),
            CelErr::RemainderByZero => ExecutionError::RemainderByZero(Value::Null),
            CelErr::IndexOutOfBounds => ExecutionError::IndexOutOfBounds(Value::Null),
            CelErr::ValuesNotComparable => {
                ExecutionError::ValuesNotComparable(Value::Null, Value::Null)
            }
            CelErr::UnsupportedKeyType => ExecutionError::unsupported_key_type(Value::Null),
            CelErr::UnsupportedTargetType => ExecutionError::UnsupportedTargetType {
                target: Value::Null,
            },
            CelErr::UnsupportedIndex => ExecutionError::UnsupportedIndex(Value::Null, Value::Null),
            CelErr::InternalError => ExecutionError::InternalError("internal VM error".to_string()),
            CelErr::NoSuchKey(id) => ExecutionError::NoSuchKey(Arc::new(name(id))),
            CelErr::UndeclaredReference(id) => {
                ExecutionError::UndeclaredReference(Arc::new(name(id)))
            }
            CelErr::NotSupportedAsMethod(id) => ExecutionError::NotSupportedAsMethod {
                method: name(id),
                target: Value::Null,
            },
            CelErr::UnexpectedType { got, want } => ExecutionError::UnexpectedType {
                got: name(got),
                want: name(want),
            },
            CelErr::Overflow(_) => {
                ExecutionError::Overflow(operator(err), Value::Null, Value::Null)
            }
            CelErr::UnsupportedUnaryOperator(_) => {
                ExecutionError::UnsupportedUnaryOperator(operator(err), Value::Null)
            }
            CelErr::UnsupportedBinaryOperator(_) => {
                ExecutionError::UnsupportedBinaryOperator(operator(err), Value::Null, Value::Null)
            }
            CelErr::InvalidArgumentCount { expected, actual } => {
                ExecutionError::InvalidArgumentCount {
                    expected: expected as usize,
                    actual: actual as usize,
                }
            }
        }
    }

    // -- the operand stack --------------------------------------------------
    //
    // The seven methods below are the whole surface: no other evaluator code
    // reads `Vm::stack`, so what "the top of the stack" means is answered in
    // one place rather than at each of the sites that asks. `Scratch::release`
    // clears the pool's own buffer, which no `Vm` owns by then.

    fn push_operand(&mut self, operand: Operand) {
        self.stack.push(operand);
    }

    /// Take the topmost operand, or `None` where there is none.
    ///
    /// Distinct from [`Vm::pop`] because an aggregate that is still being
    /// built is an operand and not yet a [`Value`]; only the caller that wants
    /// a value pays for closing it.
    fn pop_operand(&mut self) -> Option<Operand> {
        self.stack.pop()
    }

    /// The topmost operand, left where it is.
    fn top(&self) -> Option<&Operand> {
        self.stack.last()
    }

    /// The topmost operand, left where it is, open for mutation.
    ///
    /// An aggregate still being built is reached through here and mutated in
    /// place. Every such caller runs AFTER the value it is about to store has
    /// been popped, so what this answers is the operand under that one.
    fn top_mut(&mut self) -> Option<&mut Operand> {
        self.stack.last_mut()
    }

    /// How many operands are held.
    fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Drop every operand above `depth`.
    ///
    /// Only the unwind path calls this, with a [`Handler::depth`] the compiler
    /// recorded at the guarded region's first instruction. A region's operands
    /// are the ones it pushed, so raising inside one cannot leave the stack
    /// shallower than it was on the way in.
    fn truncate(&mut self, depth: usize) {
        debug_assert!(
            depth <= self.depth(),
            "unwinding to depth {depth} from {}",
            self.depth()
        );
        self.stack.truncate(depth);
    }

    /// Hand the operand stack to the scratch pool, leaving none behind.
    ///
    /// Reached from [`Vm`]'s [`Drop`], on every way out including a panic, so
    /// what it returns is every operand this run still held.
    fn take_stack(&mut self) -> Vec<Operand> {
        std::mem::take(&mut self.stack)
    }

    fn push(&mut self, value: Value) {
        self.push_operand(Operand::Value(value));
    }

    /// Pop one operand, finishing an aggregate that was still being built.
    fn pop(&mut self) -> CelResult<Value> {
        let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
        self.finish(operand)
    }

    /// Throw a popped operand away.
    ///
    /// Written as a call rather than left to end of scope so the discard has
    /// one name a measurement probe can substitute for. Without the probe this
    /// is the drop the arm performed anyway, at the same point.
    #[cfg(not(feature = "drop-arm-probe"))]
    #[inline(always)]
    fn discard(&self, value: Value) {
        drop(value);
    }

    /// Throw a popped operand away, under whichever policy the probe selected.
    ///
    /// One branch on a field, taken identically by every evaluation of a run,
    /// so its cost is a constant that cancels out of any difference between two
    /// arms. The absolute figure an arm produces is therefore NOT what the
    /// shipping interpreter costs; only the differences are claims.
    #[cfg(feature = "drop-arm-probe")]
    #[inline(always)]
    fn discard(&self, value: Value) {
        match self.probe.drop_arm {
            DropArm::Baseline => drop(value),
            DropArm::InlineDiscriminant => discard_inline(value),
            DropArm::ForgetUnsound => {
                // Compiled out of the release build the measurement uses, so
                // this is a development tripwire and not the guard. The guard
                // is the leak witness the caller runs before any timing.
                debug_assert!(
                    matches!(
                        value,
                        Value::Int(_)
                            | Value::UInt(_)
                            | Value::Float(_)
                            | Value::Bool(_)
                            | Value::Null
                    ),
                    "DropArm::ForgetUnsound leaked an owning Value: this arm is \
                     valid only for integer bodies over integer lists"
                );
                std::mem::forget(value);
            }
        }
    }

    fn finish(&mut self, operand: Operand) -> CelResult<Value> {
        match operand {
            Operand::Value(value) => Ok(value),
            Operand::List(items) => Ok(Value::list(items)),
            Operand::Map(entries) => Ok(Value::Map(Map::object(entries))),
            Operand::Struct(name, fields) => self.close_struct(name, fields),
        }
    }

    #[cfg(feature = "structs")]
    fn close_struct(&mut self, name: NameId, fields: BTreeMap<String, Value>) -> CelResult<Value> {
        let type_name = self.code.name(name).unwrap_or("?").to_string();
        let built = self
            .ctx
            .env()
            .find_struct(&type_name)
            .ok_or(CelErr::UnexpectedType {
                got: name,
                want: NameId(u32::MAX),
            })?
            .new_struct(fields);
        match built {
            Ok(value) => Ok(Value::Struct(Arc::new(value))),
            Err(err) => Err(self.park(err)),
        }
    }

    /// The refusal a build without the `structs` feature owes a struct
    /// literal. Raised where the literal is *opened*, so nothing inside it
    /// runs first; `close_struct` keeps it only because `finish` is total over
    /// [`Operand`].
    #[cfg(not(feature = "structs"))]
    fn no_structs_feature(&mut self, name: NameId) -> CelErr {
        let type_name = self.code.name(name).unwrap_or("?");
        self.park(ExecutionError::InternalError(format!(
            "Found struct {type_name}, feature not enabled!"
        )))
    }

    #[cfg(not(feature = "structs"))]
    fn close_struct(&mut self, name: NameId, _fields: BTreeMap<String, Value>) -> CelResult<Value> {
        Err(self.no_structs_feature(name))
    }

    /// The `n` topmost operands, in the order they were pushed.
    fn pop_n(&mut self, n: usize) -> CelResult<Vec<Value>> {
        self.pop_n_spare(n, 0)
    }

    /// [`Vm::pop_n`] with room for the receiver [`Vm::call_member`] prepends.
    ///
    /// `call_member` builds the receiver-first vector that
    /// `Env::find_member_overload` matches on by `insert`ing the target at
    /// index 0. A vector sized to exactly the arity has `len == capacity`, so
    /// that insert reallocates and copies -- once per member call, at every
    /// arity from 1 upward.
    ///
    /// Arity 0 is excluded, and the guard is what makes this an improvement
    /// rather than a trade. `Vec::with_capacity(0)` allocates NOTHING;
    /// `Vec::with_capacity(1)` allocates. So at arity 0 the reservation buys
    /// nothing even on the member path -- unreserved, `insert` grows the empty
    /// vector once; reserved, the reservation IS that one allocation. A wash
    /// there, and a pure loss on the path below.
    ///
    /// Both opcodes that can reach `call_member` pop through here.
    /// `CallMethod` is the obvious one; `CallQualified` is the other, because
    /// a miss hands its vector on rather than re-pushing it, and the compiler
    /// only ever emits the probe ahead of a `CallMethod` of the same arity
    /// (`compile::tests::a_probe_and_its_member_call_agree_on_arity`). The
    /// probe pops BEFORE it knows hit from miss, and a HIT never fills the
    /// spare slot -- so above arity 0, where the vector is allocated either
    /// way, widening it costs nothing, but at arity 0 it turns a call that
    /// allocated no argument vector at all into one that does. Nullary
    /// namespaced overloads are ordinary: `optional.none` is one, and so is
    /// any `ctx.add_function("ns.f", || ..)`.
    ///
    /// The spare slot stays a property of these sites rather than of `pop_n`,
    /// which [`OpCode::CallHost`] also uses and which has no receiver.
    fn pop_n_for_member(&mut self, n: usize) -> CelResult<Vec<Value>> {
        self.pop_n_spare(n, usize::from(n > 0))
    }

    /// The `n` topmost operands, in the order they were pushed, in a vector
    /// with `spare` further slots of capacity.
    fn pop_n_spare(&mut self, n: usize, spare: usize) -> CelResult<Vec<Value>> {
        let mut args = Vec::with_capacity(n + spare);
        for _ in 0..n {
            args.push(self.pop()?);
        }
        args.reverse();
        Ok(args)
    }

    fn list_mut(&mut self) -> CelResult<&mut Vec<Value>> {
        match self.top_mut() {
            Some(Operand::List(items)) => Ok(items),
            _ => Err(CelErr::InternalError),
        }
    }

    /// The map literal on top of the stack, open for the next insert.
    ///
    /// `Arc::get_mut` cannot fail here: the operand is the only holder of that
    /// pointer until [`Vm::finish`] hands it to [`Map::object`], and nothing
    /// between [`OpCode::NewMap`] and that point clones it. A `None` would be
    /// the same internal-consistency failure as a non-map on top of the stack,
    /// so it takes the same answer.
    fn map_mut(&mut self) -> CelResult<&mut HashMap<Key, Value>> {
        match self.top_mut() {
            Some(Operand::Map(entries)) => Arc::get_mut(entries).ok_or(CelErr::InternalError),
            _ => Err(CelErr::InternalError),
        }
    }

    fn struct_mut(&mut self) -> CelResult<&mut BTreeMap<String, Value>> {
        match self.top_mut() {
            Some(Operand::Struct(_, fields)) => Ok(fields),
            _ => Err(CelErr::InternalError),
        }
    }

    // -- the loop -----------------------------------------------------------

    fn run(&mut self) -> CelResult<Value> {
        let mut pc = 0u32;
        loop {
            // One comparison against a field, executed by every arm on every
            // dispatch. An arm that fuses nothing holds `u32::MAX` here and
            // never takes it; an arm that fuses takes it once per element. The
            // branch is perfectly predicted either way, and it is the probe's
            // own overhead: an arm that removes k dispatches also removes k
            // executions of this test, which inflates that arm's measured
            // saving by k times the cost of one predicted compare.
            #[cfg(feature = "elem-attr-probe")]
            if pc == self.anchor {
                pc = self.fused_element(pc)?;
                continue;
            }
            // `max_stack` is `Compiler::emit`'s sum over `stack_effect`, so
            // this is the declared effect of every opcode checked against
            // what the arms below actually push. A fused opcode whose
            // declared net is short by one drifts past this bound and past
            // nothing else -- every other check in this file is about WHAT is
            // on top, not how many.
            debug_assert!(
                self.depth() <= self.code.max_stack as usize,
                "depth {} past the compiler's {} at pc {pc}",
                self.depth(),
                self.code.max_stack
            );
            // Range is the only thing left to check: in a vector of decoded
            // records a word that is not an opcode, and an instruction the
            // stream stops short of the operands of, are states that cannot be
            // built. What a bad jump target can still be is off the end.
            let Some(&Insn { op, ops }) = self.code.insns.get(pc as usize) else {
                return Err(CelErr::InternalError);
            };
            let next = pc + 1;

            match self.step(op, ops, pc, next) {
                Ok(Step::Next) => pc = next,
                Ok(Step::Jump(target)) => pc = target,
                Ok(Step::Return(value)) => return Ok(value),
                Err(err) => pc = self.unwind(err, pc)?,
            }
        }
    }

    /// Deliver `err` to the handler covering `pc`, or give up.
    ///
    /// Absorption is the whole reason a handler table exists: CEL's `&&` and
    /// `||` are commutative over errors, so an error raised inside the left
    /// operand is *recorded* in that operator's logic slot and the right
    /// operand still runs. Nothing else in the language catches.
    fn unwind(&mut self, err: CelErr, pc: u32) -> CelResult<u32> {
        // Whatever a `CallQualified` miss parked belongs to a `CallMethod`
        // that this error has just decided will not run -- whether the handler
        // below absorbs it and lands in the right operand, or nothing catches
        // and the evaluation ends.
        self.pending_args = None;
        let Some(&Handler {
            land, logic, depth, ..
        }) = self.code.handler_for(pc)
        else {
            return Err(err);
        };
        *self
            .logic
            .get_mut(logic as usize)
            .ok_or(CelErr::InternalError)? = Err(err);
        self.truncate(depth as usize);
        Ok(land)
    }

    /// Run the per-element block from [`MapLoop::top`] as one step, up to the
    /// group the arm stops at, and return the `pc` the dispatch loop resumes
    /// at.
    ///
    /// Every quantity below is computed from the same slot, the same constant
    /// and the same helper the instructions it replaces used. What is gone is
    /// the dispatch of those instructions and whatever operand-stack round
    /// trips they still had -- which is none: each of the four groups is one
    /// instruction that names everything it reads.
    ///
    /// The operand stack is left at the depth the fused instructions would have
    /// left it: every group below is stack-neutral end to end, so the builder
    /// the loop appends into stays exactly where it was.
    #[cfg(feature = "elem-attr-probe")]
    fn fused_element(&mut self, pc: u32) -> CelResult<u32> {
        let shape = self.shape;
        let arm = self.fuse;

        // The order control enters at `after_body`, so it skips straight to the
        // advance group; every other arm enters at the loop header.
        if arm == FuseArm::AdvanceOnly {
            return self.fused_advance(shape);
        }
        // [`FuseArm::AllButBody`]'s SECOND entry, taken once the dispatched body
        // has run. Arming the header again before the advance keeps the next
        // element entering at the top.
        if arm == FuseArm::AllButBody && pc == shape.after_body {
            self.anchor = shape.top;
            return self.fused_advance(shape);
        }

        // -- the guard: `IterGuard index source done`
        let index = match self.slots.get(shape.index as usize) {
            Some(&Value::Int(index)) => index,
            _ => return Err(CelErr::InternalError),
        };
        let len = match self.slots.get(shape.source as usize) {
            Some(Value::List(list)) => list.len() as i64,
            _ => return Err(CelErr::InternalError),
        };
        if arm == FuseArm::AdvancePlusArcRoundTrip {
            // Exactly what `LoadLocal source` used to do to this slot, and
            // nothing more: one increment on the way in, one decrement on the
            // way out, on the count the whole loop shares.
            let Some(Value::List(list)) = self.slots.get(shape.source as usize) else {
                return Err(CelErr::InternalError);
            };
            let duplicate = list.clone();
            drop(std::hint::black_box(duplicate));
        }
        let more = if arm == FuseArm::GuardKeepingCompare {
            // Spelled as `step`'s `Less` arm spells it, function pointer
            // included. `IterGuard` decides with an `i64` comparison, so this
            // arm RE-ADDS the helpers rather than keeping them; see
            // [`FuseArm::GuardKeepingCompare`].
            let accept: fn(Ordering) -> bool = |o| o == Ordering::Less;
            let decided = compare_values(Value::Int(index), Value::Int(len), accept)
                .map_err(|e| self.park(e))?;
            let taken = as_bool(&decided)?;
            self.discard(decided);
            taken
        } else {
            index < len
        };
        if !more {
            return Ok(shape.done);
        }
        if arm == FuseArm::GuardKeepingCompare || arm == FuseArm::Guard {
            return Ok(shape.after_guard);
        }
        let _ = pc;

        // -- the bind: `IterBind source index var`
        let element = {
            let Some(Value::List(sequence)) = self.slots.get(shape.source as usize) else {
                return Err(CelErr::InternalError);
            };
            sequence.get(index as usize)
        };
        let element = element.ok_or(CelErr::IndexOutOfBounds)?;
        let slot = self
            .slots
            .get_mut(shape.var as usize)
            .ok_or(CelErr::InternalError)?;
        let previous = std::mem::replace(slot, element);
        self.discard(previous);
        if arm == FuseArm::Bind {
            return Ok(shape.after_bind);
        }
        // [`FuseArm::AllButBody`] hands the body back to the dispatch loop and
        // re-arms the anchor at the advance, which is where it resumes.
        if arm == FuseArm::AllButBody {
            self.anchor = shape.after_body;
            return Ok(shape.after_bind);
        }

        // -- the body and the append: `MulLocalConstAppend var k`
        let lhs = self
            .slots
            .get(shape.var as usize)
            .ok_or(CelErr::InternalError)?
            .clone();
        let rhs = self
            .code
            .konst(shape.konst)
            .ok_or(CelErr::InternalError)?
            .clone();
        let value = binary_values("mul", lhs, rhs).map_err(|e| self.park(e))?;
        self.list_mut()?.push(value);
        if arm == FuseArm::Body {
            return Ok(shape.after_body);
        }

        // -- the advance: `IterAdvance index top`
        self.fused_advance(shape)
    }

    /// `IterAdvance index top`, run as one step.
    ///
    /// Its own function only because two arms reach it: the cumulative ladder
    /// falls into it, and [`FuseArm::AdvanceOnly`] enters directly at it.
    #[cfg(feature = "elem-attr-probe")]
    #[inline(always)]
    fn fused_advance(&mut self, shape: MapLoop) -> CelResult<u32> {
        let slot = self
            .slots
            .get_mut(shape.index as usize)
            .ok_or(CelErr::InternalError)?;
        let Value::Int(counter) = slot else {
            return Err(CelErr::InternalError);
        };
        *counter = counter
            .checked_add(1)
            .ok_or(CelErr::Overflow(OpCode::Add))?;
        Ok(shape.top)
    }

    fn step(&mut self, op: OpCode, operands: [u32; 3], pc: u32, next: u32) -> CelResult<Step> {
        let [a, b, c] = operands;
        match op {
            // -- loads --------------------------------------------------
            OpCode::LoadConst => {
                let value = self.code.konst(a).ok_or(CelErr::InternalError)?.clone();
                self.push(value);
            }
            OpCode::LoadVar => {
                let name = self.name(a)?;
                let value = self
                    .ctx
                    .get_variable(name)
                    .ok_or(CelErr::UndeclaredReference(NameId(a)))?;
                self.push(value);
            }
            OpCode::LoadLocal => {
                let value = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                self.push(value);
            }
            OpCode::StoreLocal => {
                let value = self.pop()?;
                self.store_slot(a, value)?;
            }
            OpCode::IncLocal => self.advance_counter(a)?,

            // -- selection ----------------------------------------------
            OpCode::GetField => {
                let operand = self.pop()?;
                let field = self.name(a)?;
                let value = value_field(&operand, field).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::HasField => {
                let operand = self.pop()?;
                let field = self.name(a)?;
                let value = has_field(&operand, field).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::Index | OpCode::OptIndex => {
                let key = self.pop()?;
                let operand = self.pop()?;
                let value = self.index(operand, key, op == OpCode::OptIndex)?;
                self.push(value);
            }
            OpCode::GetFieldLocal | OpCode::HasFieldLocal => {
                // Read in place; see `Vm::sequence_len` for why the copy the
                // `LoadLocal` made was the operand-stack round trip and not
                // work of its own. On this path that copy was also an atomic
                // pair on the container's `Arc`, per field read.
                let read = {
                    let operand = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                    let field = self.name(b)?;
                    if op == OpCode::HasFieldLocal {
                        has_field(operand, field)
                    } else {
                        value_field(operand, field)
                    }
                };
                let value = read.map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::OptSelect => {
                let operand = self.pop()?;
                let field = Value::String(Arc::new(self.name(a)?.to_string()));
                let value = self.opt_select(operand, field)?;
                self.push(value);
            }

            // -- aggregates ----------------------------------------------
            OpCode::NewList => self.push_operand(Operand::List(Vec::new())),
            OpCode::ListAppend => {
                let value = self.pop()?;
                self.list_mut()?.push(value);
            }
            OpCode::ListAppendOptional => {
                let value = self.pop()?;
                match optional_inner(&value) {
                    OptView::Empty => {}
                    OptView::Present(inner) => self.list_mut()?.push(inner),
                    OptView::Plain => self.list_mut()?.push(value),
                }
            }
            OpCode::NewMap => self.push_operand(Operand::Map(Arc::default())),
            OpCode::MapInsert | OpCode::MapInsertOptional => {
                let value = self.pop()?;
                let key = self.pop()?;
                let key = value_key(key).map_err(|e| self.park(e))?;
                if op == OpCode::MapInsert {
                    self.map_mut()?.insert(key, value);
                } else {
                    match optional_inner(&value) {
                        OptView::Empty => {}
                        OptView::Present(inner) => {
                            self.map_mut()?.insert(key, inner);
                        }
                        OptView::Plain => {
                            self.map_mut()?.insert(key, value);
                        }
                    }
                }
            }
            OpCode::NewStruct => {
                self.open_struct(NameId(a))?;
            }
            OpCode::StructSet | OpCode::StructSetOptional => {
                let value = self.pop()?;
                let field = self.name(a)?.to_string();
                if op == OpCode::StructSet {
                    self.struct_mut()?.insert(field, value);
                } else {
                    match optional_inner(&value) {
                        OptView::Empty => {}
                        OptView::Present(inner) => {
                            self.struct_mut()?.insert(field, inner);
                        }
                        OptView::Plain => {
                            self.struct_mut()?.insert(field, value);
                        }
                    }
                }
            }

            // -- binary operators -----------------------------------------
            OpCode::Add | OpCode::Sub | OpCode::Mul | OpCode::Div | OpCode::Mod => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                let name = match op {
                    OpCode::Add => "add",
                    OpCode::Sub => "sub",
                    OpCode::Mul => "mul",
                    OpCode::Div => "div",
                    _ => "rem",
                };
                let value = binary_values(name, lhs, rhs).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::Equals | OpCode::NotEquals => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                self.push(Value::Bool((lhs == rhs) == (op == OpCode::Equals)));
            }
            OpCode::Less | OpCode::LessEquals | OpCode::Greater | OpCode::GreaterEquals => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                let accept: fn(Ordering) -> bool = match op {
                    OpCode::Less => |o| o == Ordering::Less,
                    OpCode::LessEquals => |o| o != Ordering::Greater,
                    OpCode::Greater => |o| o == Ordering::Greater,
                    _ => |o| o != Ordering::Less,
                };
                let value = compare_values(lhs, rhs, accept).map_err(|e| self.park(e))?;
                self.push(value);
            }
            // The three groups above with the right operand read out of the
            // constant pool. Each calls the same helper with the same
            // operands in the same order, so the error it raises carries the
            // same operator name the pair's did.
            OpCode::AddConst | OpCode::MulConst | OpCode::ModConst => {
                let lhs = self.pop()?;
                // Still cloned: `binary_values` takes its operands by value,
                // and giving it a by-reference twin for this path alone would
                // be a second implementation of an answer the tree walker
                // shares. What is gone is the dispatch and the round trip.
                let rhs = self.code.konst(a).ok_or(CelErr::InternalError)?.clone();
                let name = match op {
                    OpCode::AddConst => "add",
                    OpCode::MulConst => "mul",
                    _ => "rem",
                };
                let value = binary_values(name, lhs, rhs).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::EqualsConst | OpCode::NotEqualsConst => {
                let lhs = self.pop()?;
                // Read in place. `PartialEq` takes both sides by reference, so
                // this is the one fused operator that also removes the
                // constant's clone-and-release.
                let equal = {
                    let rhs = self.code.konst(a).ok_or(CelErr::InternalError)?;
                    lhs == *rhs
                };
                self.push(Value::Bool(equal == (op == OpCode::EqualsConst)));
            }
            OpCode::LessConst | OpCode::GreaterConst | OpCode::GreaterEqualsConst => {
                let lhs = self.pop()?;
                let rhs = self.code.konst(a).ok_or(CelErr::InternalError)?.clone();
                let accept: fn(Ordering) -> bool = match op {
                    OpCode::LessConst => |o| o == Ordering::Less,
                    OpCode::GreaterConst => |o| o == Ordering::Greater,
                    _ => |o| o != Ordering::Less,
                };
                let value = compare_values(lhs, rhs, accept).map_err(|e| self.park(e))?;
                self.push(value);
            }
            // The same three groups again with the LEFT operand read out of a
            // slot instead of popped, so no operand reaches the stack at all.
            // The helpers, their operand order and their operator names are
            // unchanged, which is what keeps the error identical to the pair's.
            OpCode::AddLocalConst | OpCode::MulLocalConst | OpCode::ModLocalConst => {
                // Both still cloned, for the reason the `AddConst` arm gives:
                // `binary_values` takes its operands by value.
                let lhs = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?.clone();
                let name = match op {
                    OpCode::AddLocalConst => "add",
                    OpCode::MulLocalConst => "mul",
                    _ => "rem",
                };
                let value = binary_values(name, lhs, rhs).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::EqualsLocalConst | OpCode::NotEqualsLocalConst => {
                // BOTH sides read in place: `PartialEq` takes them by
                // reference, so this is the one fused form that copies
                // nothing. `EqualsConst` already read its constant in place,
                // so what the pair still spent and this does not is the slot's
                // clone-and-release -- one atomic pair per evaluation on the
                // `Arc` variants, and a string is what an equality predicate
                // is written against.
                let equal = {
                    let lhs = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                    let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                    lhs == rhs
                };
                self.push(Value::Bool(equal == (op == OpCode::EqualsLocalConst)));
            }
            OpCode::LessLocalConst
            | OpCode::GreaterLocalConst
            | OpCode::GreaterEqualsLocalConst => {
                let lhs = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?.clone();
                let accept: fn(Ordering) -> bool = match op {
                    OpCode::LessLocalConst => |o| o == Ordering::Less,
                    OpCode::GreaterLocalConst => |o| o == Ordering::Greater,
                    _ => |o| o != Ordering::Less,
                };
                let value = compare_values(lhs, rhs, accept).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::In => {
                let rhs = self.pop()?;
                let lhs = self.pop()?;
                let value = value_contains(&rhs, &lhs).map_err(|e| self.park(e))?;
                self.push(Value::Bool(value));
            }

            // -- producing straight into a list ----------------------------
            //
            // Each of these is one of the producers above followed by the
            // `ListAppend` that took its answer off the stack again. The
            // answer is computed exactly as the producer computes it -- same
            // slot, same pool entry, same helper, same operand order, so the
            // error a failure raises is the pair's -- and handed to the
            // builder instead of pushed. `list_mut` reads the builder off the
            // top of the stack without popping it, which is what `ListAppend`
            // does too.
            OpCode::LoadLocalAppend => {
                let value = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                self.list_mut()?.push(value);
            }
            OpCode::GetFieldLocalAppend | OpCode::HasFieldLocalAppend => {
                let read = {
                    let operand = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                    let field = self.name(b)?;
                    if op == OpCode::HasFieldLocalAppend {
                        has_field(operand, field)
                    } else {
                        value_field(operand, field)
                    }
                };
                let value = read.map_err(|e| self.park(e))?;
                self.list_mut()?.push(value);
            }
            OpCode::AddLocalConstAppend
            | OpCode::MulLocalConstAppend
            | OpCode::ModLocalConstAppend => {
                let lhs = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?.clone();
                let name = match op {
                    OpCode::AddLocalConstAppend => "add",
                    OpCode::MulLocalConstAppend => "mul",
                    _ => "rem",
                };
                let value = binary_values(name, lhs, rhs).map_err(|e| self.park(e))?;
                self.list_mut()?.push(value);
            }
            OpCode::EqualsLocalConstAppend | OpCode::NotEqualsLocalConstAppend => {
                let equal = {
                    let lhs = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                    let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                    lhs == rhs
                };
                let value = Value::Bool(equal == (op == OpCode::EqualsLocalConstAppend));
                self.list_mut()?.push(value);
            }
            OpCode::LessLocalConstAppend
            | OpCode::GreaterLocalConstAppend
            | OpCode::GreaterEqualsLocalConstAppend => {
                let lhs = self
                    .slots
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?
                    .clone();
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?.clone();
                let accept: fn(Ordering) -> bool = match op {
                    OpCode::LessLocalConstAppend => |o| o == Ordering::Less,
                    OpCode::GreaterLocalConstAppend => |o| o == Ordering::Greater,
                    _ => |o| o != Ordering::Less,
                };
                let value = compare_values(lhs, rhs, accept).map_err(|e| self.park(e))?;
                self.list_mut()?.push(value);
            }

            // -- unary operators -------------------------------------------
            OpCode::Not => {
                let value = self.pop()?;
                match value {
                    Value::Bool(b) => self.push(Value::Bool(!b)),
                    _ => return Err(CelErr::NoSuchOverload),
                }
            }
            OpCode::Negate => {
                let value = self.pop()?;
                let value = value_negate(value).map_err(|e| self.park(e))?;
                self.push(value);
            }
            OpCode::NotStrictlyFalse => {
                let value = self.pop()?;
                self.push(Value::Bool(as_bool(&value).unwrap_or(true)));
            }

            // -- calls -------------------------------------------------------
            OpCode::CallHost => {
                let args = self.pop_n(b as usize)?;
                let value = self.call_global(NameId(a), args)?;
                self.push(value);
            }
            OpCode::CallMethod => {
                // Taken before anything can fail, so the park cannot outlive
                // the instruction that owns it.
                let parked = self.pending_args.take();
                let target = self.pop()?;
                let args = match parked {
                    // A `CallQualified` miss already popped them, and only the
                    // receiver sits above.
                    Some(args) => args,
                    None => self.pop_n_for_member(b as usize)?,
                };
                debug_assert_eq!(args.len(), b as usize, "arity lost across the probe");
                let value = self.call_member(NameId(a), target, args)?;
                self.push(value);
            }
            OpCode::CallQualified => {
                let args = self.pop_n_for_member(b as usize)?;
                // A miss parks the arguments instead of re-pushing them, so
                // the stack the receiver path falls through to holds the
                // receiver alone and `CallMethod` reads the park.
                if let Some(value) = self.call_qualified(NameId(a), args)? {
                    self.push(value);
                    return Ok(Step::Jump(c));
                }
            }

            // -- iteration ----------------------------------------------------
            OpCode::IterElems => {
                let value = self.pop()?;
                match value {
                    // A list already IS the sequence this iterates, so the
                    // popped value is pushed straight back. Materializing it
                    // again bought a `Vec<Value>` buffer and the
                    // `Arc<ListStorage>` [`Value::list`] wraps it in -- two
                    // allocations per comprehension, and nothing else: the two
                    // instructions that read the slot, [`OpCode::IterLen`] and
                    // [`OpCode::IterAt`], are both window-relative and neither
                    // cares which buffer answers them.
                    //
                    // The slot now SHARES the caller's buffer instead of owning
                    // a private snapshot of it, and nothing can change that
                    // buffer while the loop runs. A `Value` has no interior
                    // mutability, so the only writes are the two that rewrite a
                    // list in place -- `ListRef::into_vec` and `ListRef::concat`
                    // -- and both go through `Arc::get_mut`, which cannot answer
                    // for as long as this slot holds a reference of its own. The
                    // accumulator cannot be that other reference either: on the
                    // appending path it is an `Operand::List(Vec<Value>)` that
                    // owns its elements outright, and on the general path it is
                    // whatever `accu_init` produced, built before the loop.
                    Value::List(_) => self.push(value),
                    _ => {
                        let items = value_iter(&value).map_err(|e| self.park(e))?;
                        self.push(Value::list(items));
                    }
                }
            }
            OpCode::IterKeys => {
                let value = self.pop()?;
                let items = iter_keys(&value).map_err(|e| self.park(e))?;
                self.push(Value::list(items));
            }
            OpCode::IterLen => {
                let len = self.sequence_len(a)?;
                self.push(Value::Int(len));
            }
            OpCode::IterAt => {
                let element = self.element_at(a, b)?;
                self.push(element);
            }

            // -- the fused loop -------------------------------------------
            //
            // Each of the three reads the same slots, calls the same helpers
            // and raises the same errors as the group it replaces. What is
            // gone is the dispatch of the instructions in between, and -- for
            // the two groups that had any -- the operand-stack round trips
            // that carried a value from one of them to the next.
            OpCode::IterGuard => {
                // Decided as an `i64` comparison rather than through
                // `compare_values`, because both sides are integers by
                // construction: the counter is written once before the loop
                // with a zero and thereafter only by `IterAdvance`, and the
                // other side is a list's length. Neither is true of the
                // INSTRUCTION STREAM, so a counter slot holding anything else
                // is refused the way `IncLocal` refuses one, and a source slot
                // holding anything else the way `IterLen` refuses one.
                //
                // Read in the order the four instructions read them, so a
                // stream that is malformed in both slots raises what it raised
                // before.
                let Some(&Value::Int(index)) = self.slots.get(a as usize) else {
                    return Err(CelErr::InternalError);
                };
                let len = self.sequence_len(b)?;
                if index >= len {
                    return Ok(Step::Jump(c));
                }
            }
            OpCode::IterBind => {
                let element = self.element_at(a, b)?;
                self.store_slot(c, element)?;
            }
            OpCode::IterAdvance => {
                self.advance_counter(a)?;
                return Ok(Step::Jump(b));
            }

            // -- the fused accumulator ------------------------------------
            OpCode::AccuLoopCond | OpCode::AccuLoopCondNot => {
                // Read in place; see `Vm::sequence_len` for why the copy the
                // `LoadLocal` made was the operand-stack round trip rather
                // than work of its own.
                let accu = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                let more = if op == OpCode::AccuLoopCondNot {
                    // The negation is inside the `@not_strictly_false`, so a
                    // non-bool fails at the `!` and never reaches the test
                    // that would have answered `true` for it.
                    !as_bool(accu)?
                } else {
                    // Which is what the plain form does answer, and why it
                    // cannot fail where its twin can.
                    as_bool(accu).unwrap_or(true)
                };
                if !more {
                    return Ok(Step::Jump(b));
                }
            }
            OpCode::AndLocal | OpCode::OrLocal => {
                let short = op == OpCode::OrLocal;
                // Read in place, as above: the `LoadLocal` this replaces
                // copied the slot only so that the `And` could pop it again.
                let outcome = {
                    let accu = self.slots.get(a as usize).ok_or(CelErr::InternalError)?;
                    as_bool(accu)
                };
                *self
                    .logic
                    .get_mut(b as usize)
                    .ok_or(CelErr::InternalError)? = outcome;
                if outcome == Ok(short) {
                    self.push(Value::Bool(short));
                    return Ok(Step::Jump(c));
                }
            }

            // -- control flow ---------------------------------------------------
            OpCode::Jump => return Ok(Step::Jump(a)),
            OpCode::JumpIfOptNone => {
                let empty = match self.top() {
                    Some(Operand::Value(value)) => {
                        matches!(optional_inner(value), OptView::Empty)
                    }
                    _ => false,
                };
                if empty {
                    return Ok(Step::Jump(a));
                }
            }
            OpCode::JumpIfFalse | OpCode::JumpIfTrue => {
                let value = self.pop()?;
                let taken = as_bool(&value)? == (op == OpCode::JumpIfTrue);
                self.discard(value);
                if taken {
                    return Ok(Step::Jump(a));
                }
            }
            OpCode::And | OpCode::Or => {
                let value = self.pop()?;
                let short = op == OpCode::Or;
                let outcome = as_bool(&value);
                self.discard(value);
                *self
                    .logic
                    .get_mut(a as usize)
                    .ok_or(CelErr::InternalError)? = outcome;
                if outcome == Ok(short) {
                    self.push(Value::Bool(short));
                    return Ok(Step::Jump(b));
                }
            }
            OpCode::AndMerge | OpCode::OrMerge => {
                let right = self.pop()?;
                let left = *self.logic.get(a as usize).ok_or(CelErr::InternalError)?;
                let value = merge(left, &right, op == OpCode::OrMerge)?;
                self.discard(right);
                // The left-hand error survived only to be weighed here, and
                // it has just lost. Nothing can observe it now.
                if let Err(absorbed) = left {
                    self.unpark(absorbed);
                }
                self.push(Value::Bool(value));
            }
            OpCode::Return => return Ok(Step::Return(self.pop()?)),
        }
        let _ = (pc, next);
        Ok(Step::Next)
    }

    fn name(&self, id: u32) -> CelResult<&'a str> {
        self.code.name(NameId(id)).ok_or(CelErr::InternalError)
    }

    // -- the slot reads and writes the loop is made of ----------------------
    //
    // Each of these is one instruction's whole work and part of a fused one's,
    // written once so the two cannot answer differently. All are inlined
    // unconditionally: a call per element is exactly the cost the fused forms
    // exist to remove.

    /// The length of the list in `slot`.
    ///
    /// Read in place. The length is the only thing wanted out of the slot, and
    /// taking it through the operand stack would clone the whole `Value` -- an
    /// atomic refcount pair for a list -- once per element of the loop.
    #[inline(always)]
    fn sequence_len(&self, slot: u32) -> CelResult<i64> {
        match self.slots.get(slot as usize).ok_or(CelErr::InternalError)? {
            Value::List(list) => Ok(list.len() as i64),
            _ => Err(CelErr::InternalError),
        }
    }

    /// The element of the list in `sequence` at the index in `index`.
    ///
    /// Read in place, as [`Vm::sequence_len`] is, and indexed as a list rather
    /// than through the general indexing path.
    ///
    /// Three facts are true of every program the compiler emits: the sequence
    /// slot holds a list, because [`OpCode::IterElems`] and
    /// [`OpCode::IterKeys`] are its only writers and both push one; the index
    /// slot holds an integer, because it is initialised with a zero and
    /// advanced only by the loop's own counter; and that integer is in range,
    /// because the guard that runs immediately before compared it against the
    /// length. None of the three is true of the INSTRUCTION STREAM, which is
    /// public data anyone can build, so each is still established here -- but
    /// as one variant test, one variant test and one unsigned comparison,
    /// rather than by a helper that matches over every container the language
    /// has, then over every key type, and answers in the wide public error type
    /// on a path that never fails.
    ///
    /// The bound in particular is load-bearing rather than defensive:
    /// `ListStorage::element_at` indexes its buffer directly, so an unchecked
    /// index past the end is a panic for a boxed or columnar list and a record
    /// pointing past its own columns for a record one.
    #[inline(always)]
    fn element_at(&mut self, sequence: u32, index: u32) -> CelResult<Value> {
        // The probe's other half: the lowering this replaced, reachable at run
        // time so that what the replacement bought is a difference measured
        // inside one binary rather than between two builds. `value_index`
        // decides the container's kind, then the key's kind, then
        // bounds-checks, then answers in `ExecutionError` -- which `park` has
        // to record on `&mut self`, per element.
        #[cfg(feature = "drop-arm-probe")]
        if self.probe.iter_at == IterAtArm::ViaValueIndex {
            let element = {
                let sequence = self
                    .slots
                    .get(sequence as usize)
                    .ok_or(CelErr::InternalError)?;
                let index = self
                    .slots
                    .get(index as usize)
                    .ok_or(CelErr::InternalError)?;
                value_index(sequence, index)
            };
            return element.map_err(|e| self.park(e));
        }
        let element = {
            let Some(Value::List(sequence)) = self.slots.get(sequence as usize) else {
                return Err(CelErr::InternalError);
            };
            let Some(&Value::Int(index)) = self.slots.get(index as usize) else {
                return Err(CelErr::InternalError);
            };
            // A negative index wraps to a very large `usize` and fails the
            // bound, which is the answer the general path gives it too.
            sequence.get(index as usize)
        };
        element.ok_or(CelErr::IndexOutOfBounds)
    }

    /// Write `value` into `slot`, dropping what was there.
    #[inline(always)]
    fn store_slot(&mut self, slot: u32, value: Value) -> CelResult<()> {
        let slot = self
            .slots
            .get_mut(slot as usize)
            .ok_or(CelErr::InternalError)?;
        let previous = std::mem::replace(slot, value);
        self.discard(previous);
        Ok(())
    }

    /// Add one to the integer in `slot`, in place.
    ///
    /// Nothing is copied onto the operand stack and nothing is popped back off
    /// it. The slot is a comprehension's counter, written once with a zero and
    /// thereafter only here, so a non-integer in it is a malformed stream --
    /// the same answer [`Vm::sequence_len`] gives a slot that does not hold a
    /// list.
    #[inline(always)]
    fn advance_counter(&mut self, slot: u32) -> CelResult<()> {
        let slot = self
            .slots
            .get_mut(slot as usize)
            .ok_or(CelErr::InternalError)?;
        let Value::Int(counter) = slot else {
            return Err(CelErr::InternalError);
        };
        // Named for the operator the increment replaced, so that the public
        // error is the one the load/add/store form raised.
        *counter = counter
            .checked_add(1)
            .ok_or(CelErr::Overflow(OpCode::Add))?;
        Ok(())
    }

    // -- the arms that are more than one expression -------------------------

    /// `container[key]`, which also unwraps an optional container.
    ///
    /// The walker reaches `_[_]` and `_[?_]` through one arm and recovers
    /// which it is from the operator's name; here the two are separate opcodes
    /// decided at compile time, and `is_optional` is still a run-time fact
    /// because an optional *container* makes a plain index optional too.
    fn index(&mut self, container: Value, key: Value, mut is_optional: bool) -> CelResult<Value> {
        let container = match optional_inner(&container) {
            OptView::Plain => container,
            OptView::Empty => return Ok(optional_none()),
            OptView::Present(inner) => {
                is_optional = true;
                inner
            }
        };
        let result = value_index(&container, &key);
        if is_optional {
            return Ok(match result {
                Ok(value) => optional_of(value),
                Err(_) => optional_none(),
            });
        }
        result.map_err(|e| self.park(e))
    }

    /// `a?.b`.
    ///
    /// The nested-optional shape on a miss is the walker's, mirrored rather
    /// than corrected: `Optional::map` keeps the outer `Some` and substitutes
    /// `optional.none` for the missing field.
    fn opt_select(&mut self, operand: Value, field: Value) -> CelResult<Value> {
        Ok(match optional_inner(&operand) {
            OptView::Empty => optional_none(),
            OptView::Present(inner) => {
                optional_of(value_index(&inner, &field).unwrap_or_else(|_| optional_none()))
            }
            OptView::Plain => optional_of(value_index(&operand, &field).map_err(|e| self.park(e))?),
        })
    }

    #[cfg(feature = "structs")]
    fn open_struct(&mut self, name: NameId) -> CelResult<()> {
        let type_name = self.code.name(name).unwrap_or("?");
        if self.ctx.env().find_struct(type_name).is_none() {
            // `want` is not a program name, so the detail goes cold.
            let err = ExecutionError::UnexpectedType {
                got: type_name.to_owned(),
                want: "known struct".to_owned(),
            };
            return Err(self.park(err));
        }
        self.push_operand(Operand::Struct(name, BTreeMap::new()));
        Ok(())
    }

    /// Without the feature there is no struct to build, and the walker says so
    /// on reaching the node -- before any field expression runs. Deferring the
    /// refusal to the close would report a field's error instead:
    /// `cel.MyStruct { x: 1 / 0 }` answered `DivisionByZero`.
    #[cfg(not(feature = "structs"))]
    fn open_struct(&mut self, name: NameId) -> CelResult<()> {
        Err(self.no_structs_feature(name))
    }

    fn call_global(&mut self, name: NameId, args: Vec<Value>) -> CelResult<Value> {
        let func_name = self.name(name.0)?;
        if let Some(op) = self.ctx.env().find_overload(func_name, &args) {
            return op(args).map_err(|e| self.park(e));
        }
        let func = self
            .ctx
            .get_function(func_name)
            .ok_or(CelErr::UndeclaredReference(name))?;
        let mut fctx = crate::FunctionContext::new(func_name, None, self.ctx, args);
        (func)(&mut fctx).map_err(|e| self.park(e))
    }

    /// The namespaced probe: `math.max(1, 2)`.
    ///
    /// `None` is a miss, and a miss must leave no *evaluated* trace -- the
    /// receiver has not run yet, because `optional.of(1)` names no variable
    /// `optional`. It does leave `args` in [`Vm::pending_args`], which is
    /// where the `CallMethod` after it takes them from.
    fn call_qualified(&mut self, joined: NameId, args: Vec<Value>) -> CelResult<Option<Value>> {
        let name = self.name(joined.0)?;
        if let Some(op) = self.ctx.env().find_overload(name, &args) {
            return op(args).map(Some).map_err(|e| self.park(e));
        }
        let Some(func) = self.ctx.get_function(name) else {
            self.pending_args = Some(args);
            return Ok(None);
        };
        let mut fctx = crate::FunctionContext::new(name, None, self.ctx, args);
        (func)(&mut fctx).map(Some).map_err(|e| self.park(e))
    }

    fn call_member(&mut self, name: NameId, target: Value, args: Vec<Value>) -> CelResult<Value> {
        let func_name = self.name(name.0)?;
        let mut with_target = args;
        with_target.insert(0, target);
        if let Some(op) = self.ctx.env().find_member_overload(func_name, &with_target) {
            return op(with_target).map_err(|e| self.park(e));
        }
        let target = with_target.remove(0);
        let func = self
            .ctx
            .get_function(func_name)
            .ok_or(CelErr::UndeclaredReference(name))?;
        let mut fctx = crate::FunctionContext::new(func_name, Some(target), self.ctx, with_target);
        (func)(&mut fctx).map_err(|e| self.park(e))
    }
}

/// What one instruction decided.
enum Step {
    Next,
    Jump(u32),
    Return(Value),
}

/// Whether a value is an optional, and what is in it.
enum OptView {
    /// Not an optional at all.
    Plain,
    /// `optional.none`.
    Empty,
    Present(Value),
}

fn optional_inner(value: &Value) -> OptView {
    match as_optional(value) {
        None => OptView::Plain,
        Some(opt) => match opt.value() {
            None => OptView::Empty,
            Some(inner) => OptView::Present(inner.clone()),
        },
    }
}

/// A value used where CEL wants a bool.
///
/// A non-bool is an overload failure rather than a coercion, which is what
/// makes `1 && false` an *absorbed* error and not a truthiness test.
fn as_bool(value: &Value) -> CelResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        _ => Err(CelErr::NoSuchOverload),
    }
}

/// `has(x.y)`.
fn has_field(operand: &Value, field: &str) -> Result<Value, ExecutionError> {
    match operand {
        Value::Map(map) => Ok(Value::Bool(
            map.contains_key(&crate::objects::KeyRef::String(field)),
        )),
        #[cfg(feature = "structs")]
        Value::Struct(_) => Ok(Value::Bool(value_field(operand, field).is_ok())),
        _ => value_field(operand, field),
    }
}

/// What a two-variable comprehension binds to its first variable: a list's
/// indices, or a map's keys.
fn iter_keys(value: &Value) -> Result<Vec<Value>, ExecutionError> {
    match value {
        Value::List(list) => Ok((0..list.len()).map(|i| Value::Int(i as i64)).collect()),
        _ => value_iter(value),
    }
}

/// Combine a short-circuit operator's two sides.
///
/// `left` is the recorded outcome of the left operand -- its bool, or the
/// error a handler absorbed for it. The asymmetry is CEL's: a *recorded* error
/// is discarded when the right side decides the result, and raised otherwise.
fn merge(left: CelResult<bool>, right: &Value, is_or: bool) -> CelResult<bool> {
    let right = match right {
        Value::Bool(b) => Some(*b),
        _ => None,
    };
    match (left, right) {
        (Ok(left), Some(right)) => Ok(if is_or { left || right } else { left && right }),
        (Err(_), Some(decides)) if decides == is_or => Ok(is_or),
        (left, _) => Err(left.err().unwrap_or(CelErr::NoSuchOverload)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::ast::{Expr, IdedExpr};
    use crate::parser::Parser;
    use crate::vm::compile;

    fn parse(source: &str) -> IdedExpr {
        Parser::default()
            .parse(source)
            .unwrap_or_else(|e| panic!("parse {source}: {e:?}"))
    }

    fn run(source: &str, ctx: &Context) -> Result<Value, ExecutionError> {
        let code = compile(&parse(source)).unwrap_or_else(|e| panic!("compile {source}: {e}"));
        cel_eval_loop(&code, ctx)
    }

    /// The probe's recogniser finds the per-element block, and every arm
    /// answers what the stock arm answers.
    ///
    /// The instrument is otherwise only exercised by a benchmark, which is not
    /// run by the gate. A recogniser that matched nothing would leave every arm
    /// running the stock dispatch loop and reporting agreement -- a silent pass
    /// rather than a failure -- so the match itself is asserted before the
    /// answers are.
    #[cfg(feature = "elem-attr-probe")]
    #[test]
    fn every_fusion_arm_answers_what_the_stock_arm_answers() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", (0..64i64).collect::<Vec<i64>>());
        let code = compile(&parse("xs.map(x, x * 2)")).expect("compiles");

        let shape = recognize_map_loop(&code);
        assert_ne!(
            shape.top,
            u32::MAX,
            "the per-element block must be recognised:\n{}",
            code.disassemble()
        );

        let stock = cel_eval_loop_with_fuse(&code, &ctx, FuseArm::None).expect("stock evaluates");
        assert_eq!(
            stock,
            Value::list((0..64i64).map(|i| Value::Int(i * 2)).collect::<Vec<_>>()),
            "the loop has to actually run 64 times"
        );
        for arm in [
            FuseArm::GuardKeepingCompare,
            FuseArm::Guard,
            FuseArm::Bind,
            FuseArm::Body,
            FuseArm::Advance,
            FuseArm::AdvanceOnly,
            FuseArm::AllButBody,
            FuseArm::AdvancePlusArcRoundTrip,
        ] {
            let got = cel_eval_loop_with_fuse(&code, &ctx, arm).expect("the arm evaluates");
            assert_eq!(
                got, stock,
                "{arm:?} answered differently from the stock arm"
            );
        }
    }

    /// A fused field read answers what the pair answered, on every operand
    /// kind the read can fail on.
    ///
    /// Held against the tree walker rather than against literals alone,
    /// because a field read's failures are its whole surface: a missing key, a
    /// container that has no fields, and `has`, which answers `false` where
    /// the plain read raises. Each is a different arm of `value_field` and
    /// `has_field`, and the fused instruction reaches all of them through the
    /// same two helpers.
    #[test]
    fn a_fused_field_read_answers_what_the_pair_answered() {
        let mut ctx = Context::default();
        let record = Value::Map(crate::objects::Map::from(
            [("price", Value::Int(7))]
                .into_iter()
                .collect::<HashMap<_, _>>(),
        ));
        ctx.add_variable_from_value("items", Value::list(vec![record]));
        ctx.add_variable_from_value("flats", Value::list(vec![Value::Int(1)]));

        for source in [
            // Present, missing, and a container with no fields at all.
            "items.map(i, i.price)",
            "items.map(i, i.missing)",
            "flats.map(i, i.price)",
            // `has` answers rather than raising, on the same three.
            "items.map(i, has(i.price))",
            "items.map(i, has(i.missing))",
            "flats.map(i, has(i.price))",
        ] {
            let expr = parse(source);
            let got = run(source, &ctx);
            let walked = crate::Value::resolve_value(&expr, &ctx);
            // Compared whole, errors included: `Vm::public_error` rebuilds
            // the public error from a compact form, so a fused arm that
            // parked the wrong thing still fails here rather than merely
            // failing differently.
            assert_eq!(got, walked, "{source}");
        }
    }

    /// A folded literal operand answers what the pair answered, error
    /// identity included.
    ///
    /// The three fused groups fail in three different ways -- `binary_values`
    /// raises `Overflow` and `RemainderByZero` under the operator's own name,
    /// `compare_values` raises `NoSuchOverload` for operands of different
    /// types, and equality raises nothing at all -- and each carries the
    /// operands inside the error. A fused arm that passed the wrong operator
    /// name, or the operands in the wrong order, answers a well-formed error
    /// of the same shape, so the whole value is compared against the walker's.
    ///
    /// Split by which half of the fold each source reaches, and each half
    /// names the arms the other half must NOT reach. A single check for "some
    /// fused arm" would be satisfied by `LoadLocal ; <op>Const` -- the
    /// half-fused shape the slot half exists to exclude -- and would leave the
    /// eight stack-operand arms with no source that runs them at all, because
    /// every left operand below the divider is a slot.
    ///
    /// A slot source comes in two forms, and which one it is depends on what
    /// consumes the answer rather than on the fold: a `map` body IS the
    /// element, so its operator appends, and a `filter` guard's feeds a jump,
    /// so its operator pushes. Both are the same fold and both are here.
    #[test]
    fn a_folded_literal_operand_answers_what_the_pair_answered() {
        const SLOT_AND_CONST: [OpCode; 8] = [
            OpCode::AddLocalConst,
            OpCode::MulLocalConst,
            OpCode::ModLocalConst,
            OpCode::EqualsLocalConst,
            OpCode::NotEqualsLocalConst,
            OpCode::LessLocalConst,
            OpCode::GreaterLocalConst,
            OpCode::GreaterEqualsLocalConst,
        ];
        const SLOT_AND_CONST_APPEND: [OpCode; 8] = [
            OpCode::AddLocalConstAppend,
            OpCode::MulLocalConstAppend,
            OpCode::ModLocalConstAppend,
            OpCode::EqualsLocalConstAppend,
            OpCode::NotEqualsLocalConstAppend,
            OpCode::LessLocalConstAppend,
            OpCode::GreaterLocalConstAppend,
            OpCode::GreaterEqualsLocalConstAppend,
        ];
        /// Both slot forms, built from the two above rather than written out:
        /// a free-variable left operand must reach NEITHER, and a third list
        /// that had to be kept in step with them is a list that would not be.
        const EITHER_SLOT_FORM: [OpCode; 16] = {
            let mut both = [OpCode::Return; 16];
            let mut i = 0;
            while i < SLOT_AND_CONST.len() {
                both[i] = SLOT_AND_CONST[i];
                both[i + SLOT_AND_CONST.len()] = SLOT_AND_CONST_APPEND[i];
                i += 1;
            }
            both
        };
        const STACK_AND_CONST: [OpCode; 8] = [
            OpCode::AddConst,
            OpCode::MulConst,
            OpCode::ModConst,
            OpCode::EqualsConst,
            OpCode::NotEqualsConst,
            OpCode::LessConst,
            OpCode::GreaterConst,
            OpCode::GreaterEqualsConst,
        ];

        let hi = || Value::String(std::sync::Arc::new("hi".to_string()));
        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", vec![i64::MAX]);
        ctx.add_variable_from_value("ss", Value::list(vec![hi()]));
        // Free variables, so a left operand naming one of these is a `LoadVar`
        // and only the constant folds. Bound to the same values the elements
        // carry, so both halves raise the same errors.
        ctx.add_variable_from_value("n", i64::MAX);
        ctx.add_variable_from_value("t", hi());

        for (source, want, forbidden) in [
            // -- a slot left operand, appending: both operands fold, and the
            //    answer is the element ------------------------------------
            // `binary_values`: the operator's name reaches the error.
            (
                "xs.map(x, x * 2)",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            (
                "xs.map(x, x + 1)",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            (
                "xs.map(x, x % 0)",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            (
                "ss.map(s, s % 2)",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            // Equality answers `false` across types rather than raising, and
            // is the one group that reads both operands in place.
            (
                "ss.map(s, s == 'hi')",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            (
                "ss.map(s, s != 'hi')",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            (
                "ss.map(s, s == 3)",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            (
                "xs.map(x, x == 1)",
                &SLOT_AND_CONST_APPEND[..],
                &STACK_AND_CONST[..],
            ),
            // -- a slot left operand, pushing: the same fold in a guard, whose
            //    answer a jump consumes rather than a list ------------------
            // `compare_values`: operands of different types.
            (
                "ss.filter(s, s < 3)",
                &SLOT_AND_CONST[..],
                &STACK_AND_CONST[..],
            ),
            (
                "ss.filter(s, s > 3)",
                &SLOT_AND_CONST[..],
                &STACK_AND_CONST[..],
            ),
            (
                "ss.filter(s, s >= 3)",
                &SLOT_AND_CONST[..],
                &STACK_AND_CONST[..],
            ),
            (
                "xs.filter(x, x == 1)",
                &SLOT_AND_CONST[..],
                &STACK_AND_CONST[..],
            ),
            (
                "xs.filter(x, x != 1)",
                &SLOT_AND_CONST[..],
                &STACK_AND_CONST[..],
            ),
            // -- a free-variable left operand: only the constant folds, so
            //    these are the sources that run the eight arms above ---------
            (
                "xs.map(x, n * 2)",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "xs.map(x, n + 1)",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "xs.map(x, n % 0)",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "ss.filter(s, t < 3)",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "ss.filter(s, t > 3)",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "ss.filter(s, t >= 3)",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "ss.map(s, t == 'hi')",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
            (
                "ss.map(s, t != 'hi')",
                &STACK_AND_CONST[..],
                &EITHER_SLOT_FORM[..],
            ),
        ] {
            let expr = parse(source);
            let code = compile(&expr).unwrap_or_else(|e| panic!("compile {source}: {e}"));
            // Asserted before the answers are: a source that reached neither
            // its own arms, or reached the other half's, holds two runs of
            // instructions this case is not about against each other.
            let ops: Vec<OpCode> = code.instructions().map(|(_, op, _)| op).collect();
            assert!(
                ops.iter().any(|op| want.contains(op)),
                "{source} reaches none of {want:?}:\n{}",
                code.disassemble()
            );
            assert!(
                !ops.iter().any(|op| forbidden.contains(op)),
                "{source} reaches one of {forbidden:?}:\n{}",
                code.disassemble()
            );
            assert_eq!(
                cel_eval_loop(&code, &ctx),
                crate::Value::resolve_value(&expr, &ctx),
                "{source}"
            );
        }
    }

    /// An arithmetic fold still pushes where its answer is not the element.
    ///
    /// A `map` body IS the element, so its operator appends; five of the eight
    /// pushing arms are reached by a `filter` guard in the case above. The
    /// three arithmetic ones cannot be a guard on their own -- a non-bool
    /// condition is an error rather than a fold -- so they are reached through
    /// an enclosing comparison. That comparison's own operator is one of the
    /// forms the case above forbids, which is why these are here and not
    /// there.
    #[test]
    fn an_arithmetic_fold_still_pushes_where_its_answer_is_not_the_element() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", vec![3i64, 4]);

        for (source, want, half_fused) in [
            (
                "xs.all(x, x + 1 > 0)",
                OpCode::AddLocalConst,
                OpCode::AddConst,
            ),
            (
                "xs.all(x, x * 2 > 0)",
                OpCode::MulLocalConst,
                OpCode::MulConst,
            ),
            (
                "xs.all(x, x % 2 > 0)",
                OpCode::ModLocalConst,
                OpCode::ModConst,
            ),
        ] {
            let expr = parse(source);
            let code = compile(&expr).unwrap_or_else(|e| panic!("compile {source}: {e}"));
            let ops: Vec<OpCode> = code.instructions().map(|(_, op, _)| op).collect();
            assert!(
                ops.contains(&want),
                "{source} reaches no {want:?}:\n{}",
                code.disassemble()
            );
            assert!(
                !ops.contains(&half_fused),
                "{source} left its left operand on the stack under {half_fused:?}:\n{}",
                code.disassemble()
            );
            assert_eq!(
                cel_eval_loop(&code, &ctx),
                crate::Value::resolve_value(&expr, &ctx),
                "{source}"
            );
        }
    }

    /// Every producer that appends its own answer agrees with the walker, and
    /// the pair it replaced is gone from the program.
    ///
    /// One source per appending opcode, so an arm that read the wrong slot,
    /// the wrong pool entry or the wrong side of a comparison answers a
    /// well-formed value of the same shape and is caught by the walker rather
    /// than by inspection. The `ListAppend` check is the other half: a source
    /// that quietly stopped fusing would still agree with the walker, and
    /// agreeing is not what this is about.
    #[test]
    fn every_appending_producer_answers_what_its_pair_answered() {
        let hi = || Value::String(Arc::new("hi".to_string()));
        let record = Value::Map(crate::objects::Map::from(
            [("price", Value::Int(7))]
                .into_iter()
                .collect::<HashMap<_, _>>(),
        ));
        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", vec![1i64, 2, 3]);
        ctx.add_variable_from_value("ss", Value::list(vec![hi()]));
        ctx.add_variable_from_value("items", Value::list(vec![record]));

        for (source, want) in [
            ("xs.map(x, x)", OpCode::LoadLocalAppend),
            ("items.map(i, i.price)", OpCode::GetFieldLocalAppend),
            ("items.map(i, has(i.price))", OpCode::HasFieldLocalAppend),
            ("xs.map(x, x + 1)", OpCode::AddLocalConstAppend),
            ("xs.map(x, x * 2)", OpCode::MulLocalConstAppend),
            ("xs.map(x, x % 2)", OpCode::ModLocalConstAppend),
            ("xs.map(x, x == 2)", OpCode::EqualsLocalConstAppend),
            ("xs.map(x, x != 2)", OpCode::NotEqualsLocalConstAppend),
            ("xs.map(x, x < 2)", OpCode::LessLocalConstAppend),
            ("xs.map(x, x > 2)", OpCode::GreaterLocalConstAppend),
            ("xs.map(x, x >= 2)", OpCode::GreaterEqualsLocalConstAppend),
            // The operators reach their failing paths through the same arms:
            // `binary_values` under the operator's own name, and
            // `compare_values` across two types.
            ("xs.map(x, x % 0)", OpCode::ModLocalConstAppend),
            ("ss.map(s, s < 3)", OpCode::LessLocalConstAppend),
            ("ss.map(s, s == 'hi')", OpCode::EqualsLocalConstAppend),
        ] {
            let expr = parse(source);
            let code = compile(&expr).unwrap_or_else(|e| panic!("compile {source}: {e}"));
            let ops: Vec<OpCode> = code.instructions().map(|(_, op, _)| op).collect();
            assert!(
                ops.contains(&want),
                "{source} reaches no {want:?}:\n{}",
                code.disassemble()
            );
            assert!(
                !ops.contains(&OpCode::ListAppend),
                "{source} still routes its element through the operand stack:\n{}",
                code.disassemble()
            );
            assert_eq!(
                cel_eval_loop(&code, &ctx),
                crate::Value::resolve_value(&expr, &ctx),
                "{source}"
            );
        }
    }

    /// An element that is not a slot-naming producer keeps the pair.
    ///
    /// Each of these declines for its own reason, and each reason is one that
    /// makes the fusion WRONG rather than merely missed: an operand that has
    /// to come off the stack, a left operand the fold cannot reorder around, a
    /// name that is not a slot at all. Naming the instruction that must still
    /// carry the value is what makes a case fail for its own reason -- a bare
    /// "a `ListAppend` is present" would be satisfied by any of the others
    /// declining in its place.
    ///
    /// Held against the walker as well, because an unfused program that
    /// stopped agreeing is the same defect as a fused one that did.
    #[test]
    fn an_element_that_is_not_a_slot_producer_keeps_the_pair() {
        let record = Value::Map(crate::objects::Map::from(
            [("price", Value::Int(7))]
                .into_iter()
                .collect::<HashMap<_, _>>(),
        ));
        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", vec![1i64, 2, 3]);
        ctx.add_variable_from_value("lim", Value::Int(4));
        ctx.add_variable_from_value("items", Value::list(vec![record]));

        for (source, kept) in [
            // No folded twin, so the constant stays under its own load and
            // the operator pops two.
            ("xs.map(x, x - 1)", OpCode::Sub),
            // The literal is on the LEFT, which the fold cannot reorder
            // around: `1 - x` is not `x - 1`, so neither half folds.
            ("xs.map(x, 1 * x)", OpCode::Mul),
            // A free variable is a context lookup, not a slot; the constant
            // still folds and the left operand still reaches the stack.
            ("xs.map(x, lim * 2)", OpCode::MulConst),
            // The whole element is a context lookup.
            ("xs.map(x, lim)", OpCode::LoadVar),
            // A chain: only the innermost read names a slot, and the outer one
            // pops what it pushed.
            ("items.map(i, i.price.cents)", OpCode::GetField),
            // A call's arguments come off the stack, so its answer does too.
            ("xs.map(x, [x].size())", OpCode::CallMethod),
        ] {
            let expr = parse(source);
            let code = compile(&expr).unwrap_or_else(|e| panic!("compile {source}: {e}"));
            let ops: Vec<OpCode> = code.instructions().map(|(_, op, _)| op).collect();
            assert!(
                ops.contains(&kept),
                "{source} no longer reaches {kept:?}, so it declines for some \
                 other reason than the one this case is about:\n{}",
                code.disassemble()
            );
            assert!(
                ops.contains(&OpCode::ListAppend),
                "{source} fused an element whose value does not come from a \
                 named slot:\n{}",
                code.disassemble()
            );
            assert_eq!(
                cel_eval_loop(&code, &ctx),
                crate::Value::resolve_value(&expr, &ctx),
                "{source}"
            );
        }
    }

    /// `list_mut` finds the builder under the operand that was popped.
    ///
    /// The three `*_mut` accessors read the top of the stack, and every one of
    /// them is called AFTER the value it is about to store has been taken off
    /// it -- so what they answer is the operand one below the one the
    /// instruction was handed. Driven by hand because a program only reaches
    /// that state in the middle of an instruction, where nothing can look.
    #[test]
    fn the_mut_accessors_answer_for_the_operand_under_the_popped_one() {
        let code = CelCode {
            max_stack: 8,
            ..CelCode::default()
        };
        let ctx = Context::default();
        let mut vm = Vm::new(&code, &ctx);

        assert_eq!(vm.depth(), 0);
        assert!(vm.top().is_none());
        assert!(vm.pop_operand().is_none());

        // `NewList` then the element expression: the builder, then the value
        // that is about to be appended to it.
        vm.push_operand(Operand::List(Vec::new()));
        vm.push(Value::Int(2));
        assert_eq!(vm.depth(), 2);
        assert!(matches!(vm.top(), Some(Operand::Value(Value::Int(2)))));

        // `ListAppend`: pop, and only then reach for the builder.
        assert_eq!(vm.pop(), Ok(Value::Int(2)));
        assert_eq!(vm.depth(), 1);
        vm.list_mut()
            .expect("the builder is under the popped value")
            .push(Value::Int(2));

        // Truncating away an operand above it leaves an aggregate that still
        // closes, which is what the unwind path depends on.
        vm.push(Value::Int(3));
        vm.truncate(1);
        assert_eq!(vm.depth(), 1);
        assert_eq!(vm.pop(), Ok(Value::list(vec![Value::Int(2)])));
        assert_eq!(vm.depth(), 0);

        // The handover empties the stack into the buffer the pool releases,
        // so an operand still held at the end is not dropped anywhere else.
        vm.push(Value::Int(4));
        assert_eq!(vm.take_stack().len(), 1);
        assert_eq!(vm.depth(), 0);
    }

    /// A fused `&&`/`||` keeps CEL's asymmetry: the left operand short-circuits
    /// without the right one running, and an error in the right one is raised
    /// rather than absorbed.
    ///
    /// `x` is the iteration variable, so it is a slot and the operator is
    /// emitted fused -- which is the only way to reach `OpCode::AndLocal`'s
    /// short-circuiting path from a compiled program. Inside `all` and
    /// `exists` that path is unreachable, because the loop condition breaks on
    /// exactly the accumulator value that would have taken it.
    ///
    /// Every case is asserted against the tree walker as well as against a
    /// literal, because the asymmetry is the walker's and not this
    /// evaluator's.
    #[test]
    fn a_fused_short_circuit_keeps_the_asymmetry() {
        for (source, want) in [
            // The left operand decides: `undefined_name` never runs.
            ("xs.map(x, x && undefined_name)", Some(false)),
            ("ys.map(x, x || undefined_name)", Some(true)),
            // The left operand does not decide, so the right one runs and
            // raises. The handler covers the left operand alone.
            ("ys.map(x, x && undefined_name)", None),
            ("xs.map(x, x || undefined_name)", None),
        ] {
            let mut ctx = Context::default();
            ctx.add_variable_from_value("xs", Value::list(vec![Value::Bool(false)]));
            ctx.add_variable_from_value("ys", Value::list(vec![Value::Bool(true)]));

            let expr = parse(source);
            let got = run(source, &ctx);
            let walked = crate::Value::resolve_value(&expr, &ctx);
            assert_eq!(
                got.is_ok(),
                walked.is_ok(),
                "{source}: VM {got:?} vs walker {walked:?}"
            );
            match want {
                Some(decided) => assert_eq!(
                    got.expect("the left operand decides"),
                    Value::list(vec![Value::Bool(decided)]),
                    "{source}"
                ),
                None => assert!(got.is_err(), "{source}: {got:?}"),
            }
        }
    }

    /// The two loop conditions the macros carry, including the one that can
    /// fail.
    ///
    /// `exists` reads its accumulator NEGATED, and the negation is inside the
    /// `@not_strictly_false`, so a non-bool accumulator is an overload failure
    /// there rather than the `true` the test answers for it in `all`. Only a
    /// hand-built comprehension can put a non-bool in that slot.
    ///
    /// The step is replaced by the accumulator itself, because `all`'s and
    /// `exists`'s own step reads the accumulator too and would record the same
    /// overload failure in its logic slot -- which its merge then raises. That
    /// is the same answer for a different reason, and it would let this test
    /// pass with the condition doing nothing at all.
    #[test]
    fn the_negated_loop_condition_raises_where_the_plain_one_answers() {
        use crate::common::ast::{ComprehensionExpr, LiteralValue};

        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", Value::list(vec![Value::Int(1)]));

        for (source, want) in [
            ("xs.all(x, true)", Ok(Value::Int(7))),
            ("xs.exists(x, true)", Err(ExecutionError::NoSuchOverload)),
        ] {
            let Expr::Comprehension(base) = parse(source).expr else {
                panic!("the macro expands to a comprehension");
            };
            let mut comp: ComprehensionExpr = *base;
            let accu = comp.accu_var.clone();
            comp.accu_init = IdedExpr {
                id: 98,
                expr: Expr::Literal(LiteralValue::Int(7.into())),
            };
            comp.loop_step = IdedExpr {
                id: 99,
                expr: Expr::Ident(accu),
            };
            let code = compile(&IdedExpr {
                id: 0,
                expr: Expr::Comprehension(Box::new(comp)),
            })
            .expect("compiles");

            assert_eq!(
                cel_eval_loop(&code, &ctx),
                want,
                "{source} with an integer accumulator:\n{}",
                code.disassemble()
            );
        }
    }

    /// An error absorbed by `&&` is discarded, not accumulated.
    ///
    /// The corpus cannot see this -- the answer is the same either way -- but a
    /// comprehension absorbs once per element, so a table that only grew would
    /// be a leak proportional to the input.
    #[test]
    fn an_absorbing_loop_does_not_accumulate_parked_errors() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value(
            "xs",
            Value::list((0..64).map(Value::Int).collect::<Vec<_>>()),
        );

        // Two things this expression has to get right to be able to fail.
        // The absorbed error must be one that is actually parked --
        // `UndeclaredReference` is minted in its compact form and never
        // reaches the table -- and the loop must not short-circuit, which
        // `all` does the moment the accumulator turns false.
        let source = "xs.map(x, (1 / 0 == 1) && false)";
        let code = compile(&parse(source)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);
        let result = vm.run().expect("every element absorbs its error");
        assert_eq!(
            result,
            Value::list(vec![Value::Bool(false); 64]),
            "the loop has to actually run 64 times"
        );
        assert!(
            vm.cold.len() <= 1,
            "64 absorbed errors left {} parked",
            vm.cold.len()
        );
    }

    /// A `CallQualified` miss hands its arguments to the `CallMethod` the
    /// compiler emitted after it. When the receiver load in between raises and
    /// a short-circuit absorbs the error, that `CallMethod` never runs, and
    /// the hand-off must not survive into the other operand's own member call.
    ///
    /// `undefined_name` names neither a variable nor the first half of a
    /// function, so the probe misses and `LoadVar` raises. Both right operands
    /// have a LITERAL receiver, which is the arm that pops its own arguments
    /// -- so a surviving hand-off is what they would read instead, and each
    /// answer below flips: `true` becomes a raised error, `false` does too.
    #[test]
    fn an_absorbed_receiver_error_does_not_leave_the_probes_arguments_waiting() {
        let ctx = Context::default();
        for (source, want) in [
            (
                r#"undefined_name.startsWith("zzz") || "hello world".contains("o w")"#,
                true,
            ),
            (
                r#"undefined_name.startsWith("o w") && "hello world".contains("zzz")"#,
                false,
            ),
        ] {
            let code = compile(&parse(source)).expect("compiles");
            let mut vm = Vm::new(&code, &ctx);
            let got = vm
                .run()
                .unwrap_or_else(|e| panic!("{source}: {:?}", vm.public_error(e)));
            assert_eq!(got, Value::Bool(want), "{source}");
            assert!(
                vm.pending_args.is_none(),
                "{source}: the probe's arguments outlived the call that popped them"
            );
        }
    }

    /// The argument vector a member call is given has room for the receiver
    /// [`Vm::call_member`] prepends, so that `insert` never reallocates --
    /// EXCEPT at arity 0, where it must have no room at all.
    ///
    /// A `Vec`'s capacity is not observable through the public door -- the
    /// allocation it saves is, and `tests/allocs_per_eval.rs` is where that is
    /// pinned. This asserts the invariant those rows rest on directly, because
    /// a baseline row drifting up by one is a far colder trail than a name.
    ///
    /// The arity-0 leg is the opposite assertion rather than a relaxed one.
    /// `Vec::with_capacity(0)` allocates nothing, so an empty vector that
    /// leaves here with spare capacity has already spent the allocation that
    /// reserving was supposed to save: on the member path `insert` would have
    /// grown it once anyway, and on [`OpCode::CallQualified`]'s hit path --
    /// which pops through here before it knows hit from miss, and which nullary
    /// overloads like `optional.none` take -- nothing is ever inserted, so the
    /// vector is bought and thrown away. `capacity() > 0` here is one
    /// allocation per nullary namespaced call, which is what
    /// `walker/qualified_call/nullary` counts.
    #[test]
    fn a_member_calls_argument_vector_has_room_for_its_receiver() {
        let ctx = Context::default();
        let code = compile(&parse("1")).expect("compiles");
        for arity in 0..4usize {
            let mut vm = Vm::new(&code, &ctx);
            for i in 0..arity {
                vm.push(Value::Int(i as i64));
            }
            let args = vm.pop_n_for_member(arity).expect("the stack holds them");
            assert_eq!(args.len(), arity);
            if arity == 0 {
                assert_eq!(
                    args.capacity(),
                    0,
                    "arity 0: reserving buys nothing here and costs an allocation"
                );
            } else {
                assert!(
                    args.capacity() > arity,
                    "arity {arity}: prepending the receiver would reallocate"
                );
            }
        }
    }

    /// The table a map literal is built in is the table the finished [`Map`]
    /// holds, not a copy of it.
    ///
    /// [`OpCode::NewMap`] opens the operand in the [`Arc`] [`Map::object`]
    /// will take, so closing the literal hands the pointer over. A `Box`
    /// builder is the same 8 bytes on the operand stack but a different
    /// allocation, so `finish` had to allocate the `Arc` as well and move the
    /// 48-byte table into it -- one allocation per map literal on top of the
    /// one that was going to happen anyway. That is `walker/map_literal` in
    /// `tests/allocs_per_eval.rs`; this is the mechanism under that row, and a
    /// pointer identity is a much colder trail to follow from a baseline that
    /// drifted up by one.
    ///
    /// Driven through [`Vm::step`] rather than by pushing an operand by hand,
    /// so the `NewMap`/`MapInsert` pair the compiler emits is what runs. The
    /// literal is NESTED because that is the only shape that reaches
    /// [`Vm::map_mut`] with an outer map already open -- the one place a
    /// builder could come to be shared, which is what would make
    /// `Arc::get_mut` answer `None`.
    #[test]
    fn a_map_literal_is_built_in_the_arc_it_is_handed_over_in() {
        use crate::objects::MapStorage;

        let ctx = Context::default();
        let code = compile(&parse(r#"{"x": {"y": 3}}"#)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);

        // `Vm::run`'s loop with one line added: each builder is recorded the
        // first time it is seen on top of the stack. Neither table is freed
        // before the answer is read -- the inner one moves into the outer --
        // so no recorded address can be reused by the other.
        let mut opened: Vec<*const HashMap<Key, Value>> = Vec::new();
        let mut pc = 0u32;
        let answer = loop {
            let &Insn { op, ops } = vm.code.insns.get(pc as usize).expect("`pc` is in range");
            let next = pc + 1;
            let step = vm.step(op, ops, pc, next).expect("the literal evaluates");
            if let Some(Operand::Map(entries)) = vm.stack.last() {
                let ptr = Arc::as_ptr(entries);
                if !opened.contains(&ptr) {
                    opened.push(ptr);
                }
            }
            match step {
                Step::Next => pc = next,
                Step::Jump(target) => pc = target,
                Step::Return(value) => break value,
            }
        };

        let Value::Map(outer) = answer else {
            panic!("a map literal evaluates to a map")
        };
        let MapStorage::Object(outer_entries) = outer.storage() else {
            panic!("a map LITERAL is an owned table, not a record row")
        };
        let inner = match outer_entries.values().next() {
            Some(Value::Map(inner)) => inner,
            other => panic!("the outer table holds the inner map, got {other:?}"),
        };
        let MapStorage::Object(inner_entries) = inner.storage() else {
            panic!("a map LITERAL is an owned table, not a record row")
        };

        assert_eq!(
            opened,
            vec![Arc::as_ptr(outer_entries), Arc::as_ptr(inner_entries)],
            "the two tables this program built are not the two it answered with, \
             so closing a map literal copied it instead of handing it over"
        );
    }

    /// The absorbed error is still live between the catch and the merge,
    /// because the merge may raise it after all.
    #[test]
    fn an_absorbed_error_survives_to_be_raised_by_the_merge() {
        let ctx = Context::default();
        assert_eq!(
            run("undefined_name && true", &ctx),
            Err(ExecutionError::UndeclaredReference(Arc::new(
                "undefined_name".to_string()
            )))
        );
    }

    /// Catching truncates the operand stack, so a literal that was half-built
    /// when the error landed does not leave its elements behind.
    #[test]
    fn catching_discards_a_half_built_aggregate() {
        let ctx = Context::default();
        let source = "([1, 2, undefined_name][0] == 1) && false";
        let code = compile(&parse(source)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);
        assert_eq!(vm.run(), Ok(Value::Bool(false)));
        assert!(vm.stack.is_empty(), "the operand stack was left dirty");
    }

    /// A comprehension over a list that is neither boxed nor whole.
    ///
    /// [`OpCode::IterElems`] hands a list straight back instead of copying it
    /// out, so the loop reads the caller's [`ListRef`] where it stands: an
    /// unboxed column, seen through a WINDOW that starts past its first element
    /// and stops before its last. Copying it out used to normalise both of
    /// those away before the loop ever saw them, which is exactly why the
    /// corpus -- whose lists are all whole boxed buffers -- cannot reach this.
    #[test]
    fn a_comprehension_reads_a_windowed_unboxed_list_through_the_window() {
        use crate::objects::{ListRef, ListStorage, ScalarBank, ValueColumn};

        let words: Arc<[i64]> = Arc::from([10i64, 20, 30, 40, 50, 60].as_slice());
        let storage = Arc::new(ListStorage::Column(ValueColumn::Scalar {
            bank: ScalarBank::Int,
            words,
        }));
        let windowed = Value::List(ListRef::window(storage, 2, 3));

        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", windowed.clone());

        assert_eq!(
            run("xs.map(x, x)", &ctx),
            Ok(Value::list(vec![
                Value::Int(30),
                Value::Int(40),
                Value::Int(50)
            ]))
        );
        assert_eq!(
            run("xs.all(x, x >= 30 && x <= 50)", &ctx),
            Ok(Value::Bool(true))
        );
        // The body names the iteration source, so the loop and the binding read
        // one buffer at once -- and the binding still answers for itself after.
        assert_eq!(
            run("xs.map(x, xs)", &ctx),
            Ok(Value::list(vec![
                windowed.clone(),
                windowed.clone(),
                windowed.clone()
            ]))
        );
        assert_eq!(
            run("xs.filter(x, x in xs)", &ctx),
            run("xs.map(x, x)", &ctx)
        );
        assert_eq!(run("xs", &ctx), Ok(windowed));
    }

    /// The parser never builds a two-variable comprehension and the tree
    /// walker ignores `iter_var2` outright, so the corpus cannot reach this
    /// arm from either side.
    #[test]
    fn a_two_variable_comprehension_binds_the_key_and_the_element() {
        let Expr::Comprehension(base) = parse("xs.all(k, k >= 0)").expr else {
            panic!("the macro expands to a comprehension");
        };
        let mut comp = *base;
        comp.iter_var2 = Some("v".to_string());
        // `all` expands to `@result && <predicate>`, and `@result` is not a
        // name CEL can parse, so the predicate is swapped in place rather than
        // the whole step rewritten. It is true only if the second variable
        // really is the element at the first one's index.
        let Expr::Call(step) = &mut comp.loop_step.expr else {
            panic!("`all` steps through `&&`");
        };
        step.args[1] = parse("v == xs[k]");

        let code = compile(&IdedExpr {
            id: 0,
            expr: Expr::Comprehension(Box::new(comp)),
        })
        .expect("compiles");

        let mut ctx = Context::default();
        ctx.add_variable_from_value("xs", Value::list(vec![Value::Int(7), Value::Int(8)]));
        assert_eq!(cel_eval_loop(&code, &ctx), Ok(Value::Bool(true)));
    }

    /// `IterAt` trusts nothing in its two slots, because an instruction stream
    /// is public data and the buffer underneath it is indexed directly.
    ///
    /// A compiled program cannot reach any of these: the sequence slot is
    /// written only by `IterElems`/`IterKeys`, the index slot only by the
    /// loop's counter, and the guard above the instruction has already
    /// compared the two. That is exactly why they need a hand-built stream to
    /// state.
    #[test]
    fn iter_at_refuses_a_slot_the_compiler_could_not_have_written() {
        let ctx = Context::default();
        let insn = |op, ops| Insn { op, ops };
        let program = |consts: Vec<Value>| CelCode {
            insns: vec![
                insn(OpCode::LoadConst, [0, 0, 0]),
                insn(OpCode::StoreLocal, [0, 0, 0]),
                insn(OpCode::LoadConst, [1, 0, 0]),
                insn(OpCode::StoreLocal, [1, 0, 0]),
                insn(OpCode::IterAt, [0, 1, 0]),
                insn(OpCode::Return, [0, 0, 0]),
            ],
            consts,
            n_slots: 2,
            max_stack: 1,
            ..CelCode::default()
        };

        // A sequence slot that is not a list.
        assert!(cel_eval_loop(&program(vec![Value::Int(7), Value::Int(0)]), &ctx).is_err());
        // An index slot that is not an integer.
        assert!(cel_eval_loop(
            &program(vec![Value::list(vec![Value::Int(1)]), Value::Bool(true)]),
            &ctx
        )
        .is_err());
        // An index past the end, and a negative one.
        for out_of_range in [Value::Int(1), Value::Int(-1)] {
            assert!(cel_eval_loop(
                &program(vec![Value::list(vec![Value::Int(1)]), out_of_range]),
                &ctx
            )
            .is_err());
        }
        // ... and the in-range case still answers.
        assert_eq!(
            cel_eval_loop(
                &program(vec![Value::list(vec![Value::Int(9)]), Value::Int(0)]),
                &ctx
            ),
            Ok(Value::Int(9))
        );
    }

    /// `IterBind` makes the same three checks, on the instruction a compiled
    /// comprehension actually runs.
    ///
    /// The lowering emits no `IterAt`, so leaving those refusals pinned only on
    /// it would pin them where nothing reaches them. Both instructions read
    /// through `Vm::element_at`, and this is what says so from the outside.
    #[test]
    fn iter_bind_refuses_a_slot_the_compiler_could_not_have_written() {
        let ctx = Context::default();
        let insn = |op, ops| Insn { op, ops };
        let program = |consts: Vec<Value>| CelCode {
            insns: vec![
                insn(OpCode::LoadConst, [0, 0, 0]),
                insn(OpCode::StoreLocal, [0, 0, 0]),
                insn(OpCode::LoadConst, [1, 0, 0]),
                insn(OpCode::StoreLocal, [1, 0, 0]),
                insn(OpCode::IterBind, [0, 1, 2]),
                insn(OpCode::LoadLocal, [2, 0, 0]),
                insn(OpCode::Return, [0, 0, 0]),
            ],
            consts,
            n_slots: 3,
            max_stack: 1,
            ..CelCode::default()
        };

        // A sequence slot that is not a list.
        assert!(cel_eval_loop(&program(vec![Value::Int(7), Value::Int(0)]), &ctx).is_err());
        // An index slot that is not an integer.
        assert!(cel_eval_loop(
            &program(vec![Value::list(vec![Value::Int(1)]), Value::Bool(true)]),
            &ctx
        )
        .is_err());
        // An index past the end, and a negative one.
        for out_of_range in [Value::Int(1), Value::Int(-1)] {
            assert!(cel_eval_loop(
                &program(vec![Value::list(vec![Value::Int(1)]), out_of_range]),
                &ctx
            )
            .is_err());
        }
        // ... and the in-range case binds the element into the slot.
        assert_eq!(
            cel_eval_loop(
                &program(vec![Value::list(vec![Value::Int(9)]), Value::Int(0)]),
                &ctx
            ),
            Ok(Value::Int(9))
        );
    }

    /// `IterGuard` refuses the two slots it reads and computes the third
    /// question rather than checking it.
    ///
    /// The bound is not a check the guard makes -- it is the answer the guard
    /// exists to produce -- so it is asked for that answer in both directions
    /// instead. What it does check is the counter's type and the sequence's,
    /// which is what licenses deciding on two `i64`s.
    #[test]
    fn iter_guard_refuses_its_slots_and_decides_the_bound() {
        let ctx = Context::default();
        let insn = |op, ops| Insn { op, ops };
        // `IterGuard counter sequence 7`: falls through to the `true` at 5,
        // and jumps to the `false` at 7 once the counter reaches the length.
        let program = |consts: Vec<Value>| CelCode {
            insns: vec![
                insn(OpCode::LoadConst, [0, 0, 0]),
                insn(OpCode::StoreLocal, [0, 0, 0]),
                insn(OpCode::LoadConst, [1, 0, 0]),
                insn(OpCode::StoreLocal, [1, 0, 0]),
                insn(OpCode::IterGuard, [1, 0, 7]),
                insn(OpCode::LoadConst, [2, 0, 0]),
                insn(OpCode::Return, [0, 0, 0]),
                insn(OpCode::LoadConst, [3, 0, 0]),
                insn(OpCode::Return, [0, 0, 0]),
            ],
            consts,
            n_slots: 2,
            max_stack: 1,
            ..CelCode::default()
        };
        let one = || Value::list(vec![Value::Int(1)]);
        let guard = |sequence: Value, counter: Value| {
            cel_eval_loop(
                &program(vec![
                    sequence,
                    counter,
                    Value::Bool(true),
                    Value::Bool(false),
                ]),
                &ctx,
            )
        };

        // A sequence slot that is not a list, and a counter that is not an
        // integer: the two the `i64` comparison is licensed by.
        assert!(guard(Value::Int(7), Value::Int(0)).is_err());
        assert!(guard(one(), Value::Bool(true)).is_err());

        // The bound itself, both ways.
        assert_eq!(guard(one(), Value::Int(0)), Ok(Value::Bool(true)));
        assert_eq!(guard(one(), Value::Int(1)), Ok(Value::Bool(false)));
    }
}
