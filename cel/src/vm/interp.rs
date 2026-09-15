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
    as_optional, binary_values_ref, compare_values, optional_none, optional_of, value_contains,
    value_field, value_index, value_iter, value_key, value_negate, Key, ListStorage, Map,
};
use crate::runtime::binop::{
    cel_add, cel_div, cel_equals, cel_greater, cel_greater_equals, cel_less, cel_less_equals,
    cel_mul, cel_negate, cel_not_equals, cel_rem, cel_sub, list_contains, map_contains_key,
    map_key_refs, map_lookup,
};
use crate::runtime::convert::{intern_leaf, interned_list_get, ref_to_value};
use crate::runtime::error::{take_error, CelErrCode, ERROR_SENTINEL};
use crate::runtime::object::{
    cel_frame_slot, force_virtualizable_if_necessary, list_len, map_len, new_bytes, new_cel_frame,
    new_double, new_int, new_list, new_null, new_optional, new_optional_none, new_string, new_type,
    new_uint, opaque_host_index, string_as_str, string_byte_len, w_kind, w_type, CelKind, CelRef,
    W_BoolObject, W_CelFrame, W_DoubleObject, W_IntObject, W_UIntObject, CEL_OPAQUE_CLASS,
    CEL_TYPE_CLASS,
};
use crate::runtime::optional::{
    cel_optional_has_value, cel_optional_none, cel_optional_of, cel_optional_of_non_zero_value,
    cel_optional_or, cel_optional_or_value, cel_optional_value,
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
    #[cfg(feature = "__drop-arm-probe")]
    {
        vm.probe = PROBE.with(std::cell::Cell::get);
    }
    #[cfg(feature = "__elem-attr-probe")]
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
    #[cfg(feature = "jit")]
    {
        crate::vm::portal::eval_through_portal(&mut vm, code)
    }
    #[cfg(not(feature = "jit"))]
    {
        match vm.run() {
            Ok(value) => Ok(value),
            Err(err) => Err(vm.public_error(err)),
        }
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
#[cfg(feature = "__drop-arm-probe")]
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
#[cfg(feature = "__elem-attr-probe")]
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

#[cfg(feature = "__elem-attr-probe")]
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
#[cfg(feature = "__elem-attr-probe")]
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
#[cfg(feature = "__elem-attr-probe")]
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

#[cfg(feature = "__elem-attr-probe")]
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
#[cfg(feature = "__elem-attr-probe")]
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
#[cfg(feature = "__elem-attr-probe")]
pub fn map_loop_is_fusable(code: &CelCode) -> bool {
    recognize_map_loop(code).top != u32::MAX
}

/// One operand-stack entry.
///
/// Only the aggregate literals need anything but a [`Value`]; see the module
/// documentation.
#[derive(Clone)]
enum Operand {
    Value(Value),
    /// A class-family leaf (`TRUE`/`FALSE`/`NULL`/small-int, or a heap
    /// `int`). Arithmetic stays on [`crate::runtime::binop`] without
    /// cloning a [`Value`].
    Interned(crate::runtime::object::CelRef),
    /// A list being built, before its first element decides a strategy: the
    /// capacity that first append reserves. `EmptyListStrategy`.
    EmptyList(usize),
    /// A list being built whose every element so far is an integer, as a
    /// buffer of words: a quarter of the writes of a boxed buffer, nothing to
    /// drop element by element when the list goes away, and closed as a
    /// [`ListStorage::Ints`], which every consumer of a list reads through
    /// the same accessors as a boxed one. `IntegerListStrategy`. The first
    /// element that is not an integer boxes what was collected so far, once,
    /// into [`Operand::List`].
    Ints(Vec<i64>),
    /// A list being built, boxed. `ObjectListStrategy`.
    List(Vec<Value>),
    /// A list being built of interned leaves. `ObjectListStrategy` holding
    /// `W_Root` pointers, the same shape `listobject.py` keeps after
    /// `switch_to_object_strategy` when every item is already a boxed object.
    Refs(Vec<CelRef>),
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
    /// A map being built of interned keys and values. Closed as `new_map`.
    MapRefs(Vec<(CelRef, CelRef)>),
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

impl Operand {
    /// What an entry holds before anything is stored in it and after its
    /// operand is popped: the frame's `None`.
    const NULL: Operand = Operand::Value(Value::Null);
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
#[cfg(feature = "__drop-arm-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DropArm {
    /// Hand the operand to the glue, whatever it holds. What the interpreter
    /// did without the probe until the probe priced it; see
    /// [`discard_inline`].
    Baseline,
    /// Test the discriminant at the call site and reach the glue only for the
    /// variants that own something. Keeps the work; removes the call on the
    /// trivial path. What the interpreter does without the probe.
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
#[cfg(feature = "__drop-arm-probe")]
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
#[cfg(feature = "__drop-arm-probe")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ProbePolicy {
    pub drop_arm: DropArm,
    pub iter_at: IterAtArm,
}

#[cfg(feature = "__drop-arm-probe")]
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

#[cfg(feature = "__drop-arm-probe")]
impl Default for ProbePolicy {
    /// What the interpreter does without the probe, so that a run that names
    /// only one half leaves the other half alone.
    fn default() -> ProbePolicy {
        ProbePolicy::STOCK
    }
}

#[cfg(feature = "__drop-arm-probe")]
std::thread_local! {
    /// The policy the next evaluation on this thread runs under.
    ///
    /// Read once per evaluation -- not once per instruction -- and identically
    /// by every arm, so it is a constant that cancels out of any difference
    /// between two of them.
    static PROBE: std::cell::Cell<ProbePolicy> =
        const { std::cell::Cell::new(ProbePolicy::STOCK) };
}

/// Test the discriminant here, and reach the out-of-line glue only for the
/// variants that own something.
///
/// What the shipping `discard` does, and [`DropArm::InlineDiscriminant`]'s
/// policy under the probe. The probe priced one discard site at 1.1-1.3 ns,
/// of which the call was 0.97-1.05 ns and the discriminant work inside it
/// 0.13-0.24 ns: the glue for an integer is a handful of instructions that
/// do nothing, reached through a call that costs more than they do.
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
    /// The activation record: `| locals | stack |`, one array, laid out as
    /// `PyFrame.locals_cells_stack_w` is. See [`Vm::frame`].
    frame: Vec<Operand>,
    /// One entry per `&&`/`||`, holding the left operand's outcome: its bool,
    /// or the error the handler absorbed for it.
    logic: Vec<CelResult<bool>>,
    /// Full errors for the cases [`CelErr`] cannot spell.
    ///
    /// Written only when an error is raised, and truncated again when one is
    /// absorbed, so a comprehension whose body errors on every iteration does
    /// not accumulate a table the size of the sequence.
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
        self.frame.clear();
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
    ///
    /// Boxed, so that what moves in and out is one pointer. Measured before
    /// the box: the hand-back copied four `Vec` headers onto the stack, then
    /// into the cell, and dropped the cell's previous contents, and that was
    /// a third of what a one-instruction program cost end to end.
    static SCRATCH: std::cell::Cell<Option<Box<Scratch>>> = const { std::cell::Cell::new(None) };
}

pub(crate) struct Vm<'a> {
    code: &'a CelCode,
    ctx: &'a Context<'a>,
    /// The activation record, `| locals | stack |` in ONE array, the layout
    /// `PyFrame.__init__` gives `locals_cells_stack_w`. A local is read at
    /// its slot index; the operand stack is the tail from `stack_base` up,
    /// and the vector's length is the absolute index of its next free entry,
    /// `PyFrame.valuestackdepth`.
    ///
    /// The length is the depth rather than a field beside a presized array
    /// because of what a `Vec` already keeps: everything below its length is
    /// initialized and everything above it is not, which is exactly the
    /// frame's invariant that no dead operand is held past its instruction.
    /// `popvalue` restores that by writing `None`; a pop here moves the entry
    /// out and shortens the length, so nothing is written back. Measured the
    /// other way -- a presized array, a depth field, a null stored on every
    /// pop and dropped on every push -- and it cost 2 ns per instruction.
    ///
    /// Sized once per run: the locals, then capacity for `max_stack` more,
    /// the compiler's proof of how deep the stack goes.
    ///
    /// Taken out of the pool's box by `Vm::new` and put back by [`Drop`]:
    /// the one buffer every instruction touches is a field of the record
    /// the loop already holds, not a load away through the box.
    frame: Vec<Operand>,
    /// The pool's box, holding the two buffers a run needs less often, and
    /// the empty twin of `frame` until [`Drop`] returns it. `Vm::new` moves
    /// one pointer in and [`Drop`] moves one pointer back out. Measured the
    /// other way -- take all three buffers out, swap them back -- and the
    /// moves plus the drops of the emptied twins were a third of the fixed
    /// price of evaluating `1`.
    scratch: std::mem::ManuallyDrop<Box<Scratch>>,
    /// Where the operand stack begins in `frame`: `n_slots`.
    stack_base: usize,
    /// `PyFrame` virtualizable: `last_instr`, `valuestackdepth`,
    /// `locals_stack_w[*]`. Interned slots are written through here.
    pub(crate) cel_frame: *mut W_CelFrame,
    /// Result parked by the JIT portal when `dispatch_one` returns.
    pub(crate) portal_ret: Option<CelResult<Value>>,
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
    #[cfg(feature = "__drop-arm-probe")]
    probe: ProbePolicy,
    /// Which groups of the per-element block run as one step. Probe only; see
    /// [`FuseArm`].
    #[cfg(feature = "__elem-attr-probe")]
    fuse: FuseArm,
    /// Where that block is. Probe only; see [`MapLoop`].
    #[cfg(feature = "__elem-attr-probe")]
    shape: MapLoop,
    /// The one `pc` at which the fused path is taken, or `u32::MAX` for an arm
    /// that fuses nothing. A field rather than a second test, so that the
    /// dispatch loop's per-instruction cost is identical in every arm however
    /// many anchors the probe grows. Probe only.
    #[cfg(feature = "__elem-attr-probe")]
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
        // SAFETY: taken exactly once, here, and nothing reads the field
        // afterwards: this is the drop. `ManuallyDrop` is what lets the box
        // leave a type that implements `Drop` without a placeholder that
        // would itself have to be built or dropped.
        let mut scratch = unsafe { std::mem::ManuallyDrop::take(&mut self.scratch) };
        scratch.frame = std::mem::take(&mut self.frame);
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
    /// so this establishes the lengths the loop indexes into: the locals and
    /// the operand stack the compiler proved it needs, as one array. On a
    /// thread that has evaluated anything before, the capacity is already
    /// there and none of this allocates.
    fn new(code: &'a CelCode, ctx: &'a Context<'a>) -> Self {
        let mut scratch = SCRATCH
            .try_with(std::cell::Cell::take)
            .ok()
            .flatten()
            .unwrap_or_default();
        let stack_base = code.n_slots as usize;
        let mut frame = std::mem::take(&mut scratch.frame);
        // Each sizing call is out of line and most programs need neither:
        // a scalar expression has no locals and no `&&`/`||`.
        if stack_base > 0 {
            frame.resize_with(stack_base, || Operand::NULL);
        }
        frame.reserve(code.max_stack as usize);
        if code.n_logic > 0 {
            scratch
                .logic
                .resize(code.n_logic as usize, Err(CelErr::InternalError));
        }
        let cel_frame = new_cel_frame(code.n_slots as i64, code.max_stack as i64);
        Vm {
            code,
            ctx,
            frame,
            scratch: std::mem::ManuallyDrop::new(scratch),
            stack_base,
            cel_frame,
            portal_ret: None,
            pending_args: None,
            #[cfg(feature = "__drop-arm-probe")]
            probe: ProbePolicy::default(),
            #[cfg(feature = "__elem-attr-probe")]
            fuse: FuseArm::None,
            // Scanned by every arm, including the one that fuses nothing, so
            // the scan is a constant rather than a term of any difference.
            #[cfg(feature = "__elem-attr-probe")]
            shape: recognize_map_loop(code),
            #[cfg(feature = "__elem-attr-probe")]
            anchor: u32::MAX,
        }
    }

    // -- the error channel ------------------------------------------------

    /// Park a full error and return the [`CelErr`] that stands for it.
    ///
    /// Out of line and cold: every fallible arm of the dispatch loop ends in
    /// a `map_err` onto this, and inlined it put a `Vec` push, its growth
    /// call and an `ExecutionError` drop at each of those sites, in the one
    /// function whose frame every instruction pays for.
    #[cold]
    #[inline(never)]
    fn park(&mut self, err: ExecutionError) -> CelErr {
        let id = u32::try_from(self.scratch.cold.len()).unwrap_or(u32::MAX);
        self.scratch.cold.push(err);
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
            if id as usize + 1 == self.scratch.cold.len() {
                self.scratch.cold.pop();
            }
        }
    }

    /// Reconstruct the public error.
    ///
    /// Exhaustive over [`CelErr`], so a variant added without a public
    /// counterpart is a compile error here rather than a silent
    /// `InternalError` at run time.
    pub(crate) fn sync_pop_push_interned(&mut self, n_pop: usize, w: CelRef) {
        let mut i = 0;
        while i < n_pop {
            let _ = self.pop_operand();
            i += 1;
        }
        self.push_operand(Operand::Interned(w));
    }

    pub(crate) fn sync_push_interned(&mut self, w: CelRef) {
        self.push_operand(Operand::Interned(w));
    }

    pub(crate) fn sync_store_interned(&mut self, slot: u32, w: CelRef) {
        let _ = self.pop_operand();
        let _ = self.store_operand(slot, Operand::Interned(w));
    }

    pub(crate) fn park_return(&mut self, w: CelRef) {
        self.portal_ret = Some(Ok(crate::Value::from_interned(w)));
    }

    pub(crate) fn public_error(&self, err: CelErr) -> ExecutionError {
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
                .scratch
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

    fn vable_cell(operand: &Operand) -> CelRef {
        match operand {
            Operand::Interned(w) => *w,
            Operand::Value(v) => intern_leaf(v).unwrap_or(core::ptr::null_mut()),
            _ => core::ptr::null_mut(),
        }
    }

    fn write_vable_cell(&mut self, index: usize, w: CelRef) {
        unsafe {
            let cap = (*self.cel_frame).locals_stack_w.capacity();
            if index < cap {
                *cel_frame_slot(self.cel_frame, index as i64) = w;
            }
            (*self.cel_frame).valuestackdepth = self.frame.len() as i64;
        }
    }

    #[inline(always)]
    fn push_operand(&mut self, operand: Operand) {
        let index = self.frame.len();
        let w = Self::vable_cell(&operand);
        self.frame.push(operand);
        self.write_vable_cell(index, w);
    }

    /// Take the topmost operand, or `None` where there is none.
    ///
    /// Distinct from [`Vm::pop`] because an aggregate that is still being
    /// built is an operand and not yet a [`Value`]; only the caller that wants
    /// a value pays for closing it.
    #[inline(always)]
    fn pop_operand(&mut self) -> Option<Operand> {
        if self.frame.len() == self.stack_base {
            return None;
        }
        let popped = self.frame.pop();
        unsafe {
            (*self.cel_frame).valuestackdepth = self.frame.len() as i64;
        }
        popped
    }

    /// The topmost operand, left where it is.
    fn top(&self) -> Option<&Operand> {
        (self.frame.len() > self.stack_base).then(|| &self.frame[self.frame.len() - 1])
    }

    /// The topmost operand, left where it is, open for mutation.
    ///
    /// An aggregate still being built is reached through here and mutated in
    /// place. Every such caller runs AFTER the value it is about to store has
    /// been popped, so what this answers is the operand under that one.
    fn top_mut(&mut self) -> Option<&mut Operand> {
        if self.frame.len() > self.stack_base {
            self.frame.last_mut()
        } else {
            None
        }
    }

    /// How many operands are held.
    fn depth(&self) -> usize {
        self.frame.len() - self.stack_base
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
        self.frame.truncate(self.stack_base + depth);
        unsafe {
            (*self.cel_frame).valuestackdepth = self.frame.len() as i64;
        }
    }

    /// The local in `slot`, or `None` for an index past the locals or an
    /// entry that is not a value -- both states an instruction stream cannot
    /// reach and the arms report as internal errors.
    #[inline(always)]
    #[allow(dead_code)]
    fn local(&self, slot: u32) -> Option<&Value> {
        match self.frame[..self.stack_base].get(slot as usize) {
            Some(Operand::Value(value)) => Some(value),
            _ => None,
        }
    }

    fn local_operand(&self, slot: u32) -> Option<&Operand> {
        self.frame[..self.stack_base].get(slot as usize)
    }

    fn local_operand_mut(&mut self, slot: u32) -> Option<&mut Operand> {
        self.frame[..self.stack_base].get_mut(slot as usize)
    }

    /// The slot as a public [`Value`], converting an interned leaf.
    fn local_as_value(&self, slot: u32) -> Option<Value> {
        match self.local_operand(slot)? {
            Operand::Value(value) => Some(value.clone()),
            Operand::Interned(w) => unsafe { ref_to_value(*w) }.ok(),
            _ => None,
        }
    }

    fn local_leaf(&self, slot: u32) -> Option<CelRef> {
        Self::leaf_of(self.local_operand(slot)?)
    }

    fn local_int(&self, slot: u32) -> Option<i64> {
        match self.local_operand(slot)? {
            Operand::Value(Value::Int(n)) => Some(*n),
            Operand::Interned(w) if unsafe { w_kind(*w) } == CelKind::Int => {
                Some(unsafe { (*(*w as *mut W_IntObject)).intval })
            }
            _ => None,
        }
    }

    #[inline(always)]
    fn push(&mut self, value: Value) {
        if let Some(w) = intern_leaf(&value) {
            self.push_operand(Operand::Interned(w));
        } else {
            self.push_operand(Operand::Value(value));
        }
    }

    fn leaf_of(operand: &Operand) -> Option<CelRef> {
        match operand {
            Operand::Interned(w) => Some(*w),
            Operand::Value(v) => intern_leaf(v),
            _ => None,
        }
    }

    fn push_interned(&mut self, w: CelRef, op: OpCode) -> CelResult<()> {
        if w == ERROR_SENTINEL {
            return Err(self.raised_as_cel_err(op));
        }
        self.push_operand(Operand::Interned(w));
        Ok(())
    }

    fn raised_as_cel_err(&mut self, op: OpCode) -> CelErr {
        let Some(err) = take_error() else {
            return CelErr::InternalError;
        };
        let lhs = unsafe { ref_to_value(err.lhs) }.unwrap_or(Value::Null);
        let rhs = if err.rhs == ERROR_SENTINEL {
            Value::Int(0)
        } else {
            unsafe { ref_to_value(err.rhs) }.unwrap_or(Value::Null)
        };
        let exec = match err.code {
            CelErrCode::Overflow => ExecutionError::Overflow(err.op, lhs, rhs),
            CelErrCode::DivisionByZero => ExecutionError::DivisionByZero(lhs),
            CelErrCode::RemainderByZero => ExecutionError::RemainderByZero(lhs),
            CelErrCode::UnsupportedBinaryOperator => {
                if crate::objects::mismatch_is_no_such_overload(err.op, &lhs) {
                    ExecutionError::NoSuchOverload
                } else {
                    ExecutionError::UnsupportedBinaryOperator(err.op, lhs, rhs)
                }
            }
            CelErrCode::NoSuchOverload => ExecutionError::NoSuchOverload,
            CelErrCode::NoneDereference => {
                ExecutionError::function_error(err.op, "optional.none() dereference")
            }
        };
        let _ = op;
        self.park(exec)
    }

    /// `true` if `w` is a map/struct and the field op was handled.
    fn push_interned_field(&mut self, w: CelRef, field: &str, has: bool) -> CelResult<bool> {
        match unsafe { w_kind(w) } {
            CelKind::Map => {
                let found =
                    unsafe { crate::runtime::convert::interned_map_lookup_string(w, field) };
                if has {
                    self.push(Value::Bool(found.is_some()));
                    return Ok(true);
                }
                match found {
                    Some(v) => {
                        self.push_operand(Operand::Interned(v));
                        Ok(true)
                    }
                    None => Err(self.park(ExecutionError::NoSuchKey(std::sync::Arc::new(
                        field.to_string(),
                    )))),
                }
            }
            #[cfg(feature = "structs")]
            CelKind::Struct => {
                let found = unsafe { crate::runtime::object::struct_lookup_field(w, field) };
                if has {
                    self.push(Value::Bool(found.is_some()));
                    return Ok(true);
                }
                match found {
                    Some(v) => {
                        self.push_operand(Operand::Interned(v));
                        Ok(true)
                    }
                    None => Err(self.park(ExecutionError::NoSuchKey(std::sync::Arc::new(
                        field.to_string(),
                    )))),
                }
            }
            _ => Ok(false),
        }
    }

    fn try_interned_index(
        &mut self,
        operand: &Operand,
        key: &Operand,
        mut is_optional: bool,
    ) -> CelResult<bool> {
        let Some(mut w) = Self::leaf_of(operand) else {
            return Ok(false);
        };
        if unsafe { w_kind(w) } == CelKind::Optional {
            let inner = unsafe { (*w.cast::<crate::runtime::object::W_OptionalObject>()).w_value };
            if inner.is_null() {
                self.push_operand(Operand::Interned(new_optional_none() as CelRef));
                return Ok(true);
            }
            w = inner;
            is_optional = true;
        }
        let item = match unsafe { w_kind(w) } {
            CelKind::List => {
                let Some(index) = interned_int(key) else {
                    return Ok(false);
                };
                match unsafe { interned_list_get(w, index) } {
                    Some(item) => item,
                    None if is_optional => {
                        self.push_operand(Operand::Interned(new_optional_none() as CelRef));
                        return Ok(true);
                    }
                    None => {
                        return Err(self.park(ExecutionError::IndexOutOfBounds(Value::Int(index))))
                    }
                }
            }
            CelKind::Map => {
                let Some(k) = Self::leaf_of(key) else {
                    return Ok(false);
                };
                match unsafe { map_lookup(w, k) } {
                    Some(item) => item,
                    None if is_optional => {
                        self.push_operand(Operand::Interned(new_optional_none() as CelRef));
                        return Ok(true);
                    }
                    None => return Ok(false),
                }
            }
            #[cfg(feature = "structs")]
            CelKind::Struct => {
                let field = match key {
                    Operand::Value(Value::String(s)) => Some(s.as_str()),
                    Operand::Interned(k) => unsafe { crate::runtime::object::string_as_str(*k) },
                    _ => None,
                };
                let Some(field) = field else {
                    return Ok(false);
                };
                match unsafe { crate::runtime::object::struct_lookup_field(w, field) } {
                    Some(item) => item,
                    None if is_optional => {
                        self.push_operand(Operand::Interned(new_optional_none() as CelRef));
                        return Ok(true);
                    }
                    None => return Ok(false),
                }
            }
            _ => return Ok(false),
        };
        if is_optional {
            self.push_operand(Operand::Interned(new_optional(item) as CelRef));
        } else {
            self.push_operand(Operand::Interned(item));
        }
        Ok(true)
    }

    fn apply_cel_binop(
        &mut self,
        op: OpCode,
        name: &'static str,
        lhs: CelRef,
        rhs: CelRef,
    ) -> CelResult<()> {
        let w = unsafe {
            match name {
                "add" => cel_add(lhs, rhs),
                "sub" => cel_sub(lhs, rhs),
                "mul" => cel_mul(lhs, rhs),
                "div" => cel_div(lhs, rhs),
                "rem" => cel_rem(lhs, rhs),
                _ => return Err(CelErr::InternalError),
            }
        };
        self.push_interned(w, op)
    }

    /// Pop one operand, finishing an aggregate that was still being built.
    #[inline(always)]
    fn pop(&mut self) -> CelResult<Value> {
        let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
        self.finish(operand)
    }

    /// Throw a popped operand away.
    ///
    /// Written as a call rather than left to end of scope so the discard has
    /// one name a measurement probe can substitute for. Without the probe this
    /// is the drop the arm performed anyway, at the same point.
    #[cfg(not(feature = "__drop-arm-probe"))]
    #[inline(always)]
    fn discard(&self, value: Value) {
        discard_inline(value);
    }

    /// Throw a popped operand away, under whichever policy the probe selected.
    ///
    /// One branch on a field, taken identically by every evaluation of a run,
    /// so its cost is a constant that cancels out of any difference between two
    /// arms. The absolute figure an arm produces is therefore NOT what the
    /// shipping interpreter costs; only the differences are claims.
    #[cfg(feature = "__drop-arm-probe")]
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

    #[inline(always)]
    fn finish(&mut self, operand: Operand) -> CelResult<Value> {
        match operand {
            Operand::Value(value) => Ok(value),
            Operand::Interned(w) => Ok(Value::from_interned(w)),
            Operand::EmptyList(_) => Ok(Value::from_interned(
                crate::runtime::object::new_list(&[]) as CelRef,
            )),
            Operand::Ints(words) => {
                let items: Vec<CelRef> = words.iter().map(|&n| new_int(n) as CelRef).collect();
                Ok(Value::from_interned(new_list(&items) as CelRef))
            }
            Operand::List(items) => Ok(Value::list(items)),
            Operand::Refs(items) => Ok(Value::from_interned(new_list(&items) as CelRef)),
            Operand::Map(entries) => Ok(Value::Map(Map::object(entries))),
            Operand::MapRefs(pairs) => Ok(Value::from_interned(crate::runtime::object::new_map(
                &pairs,
            ) as CelRef)),
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

    /// Append `value` to the list being built on top of the stack.
    ///
    /// The strategy switch lives here and nowhere else: an integer joins an
    /// integer buffer, anything else turns that buffer into a boxed one. A
    /// switch boxes once and never switches back.
    #[inline(always)]
    fn append_to_list(&mut self, value: Value) -> CelResult<()> {
        self.append_operand(Operand::Value(value))
    }

    /// Append without forcing an interned item through [`Value`].
    ///
    /// Integers stay on [`Operand::Ints`]. Other interned leaves stay
    /// `W_Root` pointers on [`Operand::Refs`], the object-strategy list.
    fn append_operand(&mut self, operand: Operand) -> CelResult<()> {
        match operand {
            Operand::Interned(w) if unsafe { w_kind(w) } == CelKind::Int => {
                let word = unsafe { (*w.cast::<W_IntObject>()).intval };
                self.append_int(word)
            }
            Operand::Interned(w) => self.append_ref(w),
            Operand::Value(Value::Int(word)) => self.append_int(word),
            Operand::Value(value) => {
                if let Some(w) = intern_leaf(&value) {
                    self.append_operand(Operand::Interned(w))
                } else {
                    self.append_boxed(value)
                }
            }
            other => {
                let value = self.finish(other)?;
                self.append_to_list(value)
            }
        }
    }

    fn append_int(&mut self, word: i64) -> CelResult<()> {
        let top = self.top_mut().ok_or(CelErr::InternalError)?;
        match top {
            Operand::Ints(words) => words.push(word),
            Operand::EmptyList(hint) => {
                let mut words = Vec::with_capacity((*hint).max(1));
                words.push(word);
                *top = Operand::Ints(words);
            }
            Operand::Refs(items) => items.push(new_int(word) as CelRef),
            Operand::List(items) => items.push(Value::Int(word)),
            _ => return Err(CelErr::InternalError),
        }
        Ok(())
    }

    fn append_ref(&mut self, w: CelRef) -> CelResult<()> {
        let top = self.top_mut().ok_or(CelErr::InternalError)?;
        match top {
            Operand::Refs(items) => items.push(w),
            Operand::EmptyList(hint) => {
                let mut items = Vec::with_capacity((*hint).max(1));
                items.push(w);
                *top = Operand::Refs(items);
            }
            Operand::Ints(words) => {
                let mut items = Vec::with_capacity(words.capacity().max(words.len() + 1));
                items.extend(words.iter().map(|&n| new_int(n) as CelRef));
                items.push(w);
                *top = Operand::Refs(items);
            }
            Operand::List(items) => {
                items.push(unsafe { ref_to_value(w) }.map_err(|_| CelErr::InternalError)?);
            }
            _ => return Err(CelErr::InternalError),
        }
        Ok(())
    }

    fn append_boxed(&mut self, value: Value) -> CelResult<()> {
        let top = self.top_mut().ok_or(CelErr::InternalError)?;
        match top {
            Operand::List(items) => items.push(value),
            Operand::Ints(words) => {
                let mut items = Vec::with_capacity(words.capacity().max(words.len() + 1));
                items.extend(words.iter().map(|&w| Value::Int(w)));
                items.push(value);
                *top = Operand::List(items);
            }
            Operand::Refs(items) => {
                let refs = std::mem::take(items);
                let mut boxed = Vec::with_capacity(refs.capacity().max(refs.len() + 1));
                for w in refs {
                    boxed.push(unsafe { ref_to_value(w) }.map_err(|_| CelErr::InternalError)?);
                }
                boxed.push(value);
                *top = Operand::List(boxed);
            }
            Operand::EmptyList(hint) => {
                let mut items = Vec::with_capacity((*hint).max(1));
                items.push(value);
                *top = Operand::List(items);
            }
            _ => return Err(CelErr::InternalError),
        }
        Ok(())
    }

    /// The map literal on top of the stack, open for the next insert.
    ///
    /// `Arc::get_mut` cannot fail here: the operand is the only holder of that
    /// pointer until [`Vm::finish`] hands it to [`Map::object`], and nothing
    /// between [`OpCode::NewMap`] and that point clones it. A `None` would be
    /// the same internal-consistency failure as a non-map on top of the stack,
    /// so it takes the same answer.
    fn map_mut(&mut self) -> CelResult<&mut HashMap<Key, Value>> {
        self.ensure_boxed_map()?;
        match self.top_mut() {
            Some(Operand::Map(entries)) => Arc::get_mut(entries).ok_or(CelErr::InternalError),
            _ => Err(CelErr::InternalError),
        }
    }

    fn ensure_boxed_map(&mut self) -> CelResult<()> {
        let top = self.top_mut().ok_or(CelErr::InternalError)?;
        let Operand::MapRefs(pairs) = top else {
            return Ok(());
        };
        let pairs = std::mem::take(pairs);
        let mut entries = HashMap::with_capacity(pairs.len());
        for (k, v) in pairs {
            let key = unsafe { ref_to_value(k) }.map_err(|_| CelErr::InternalError)?;
            let value = unsafe { ref_to_value(v) }.map_err(|_| CelErr::InternalError)?;
            entries.insert(value_key(key).map_err(|e| self.park(e))?, value);
        }
        *self.top_mut().ok_or(CelErr::InternalError)? = Operand::Map(Arc::new(entries));
        Ok(())
    }

    fn close_builder_operand(&mut self, operand: Operand) -> CelResult<Operand> {
        match operand {
            Operand::MapRefs(pairs) => Ok(Operand::Interned(
                crate::runtime::object::new_map(&pairs) as CelRef,
            )),
            Operand::Refs(items) => Ok(Operand::Interned(new_list(&items) as CelRef)),
            Operand::Ints(words) => {
                let refs: Vec<CelRef> = words.iter().map(|&n| new_int(n) as CelRef).collect();
                Ok(Operand::Interned(new_list(&refs) as CelRef))
            }
            Operand::EmptyList(_) => Ok(Operand::Interned(new_list(&[]) as CelRef)),
            Operand::List(items) => {
                let value = Value::list(items);
                Ok(intern_leaf(&value)
                    .map(Operand::Interned)
                    .unwrap_or(Operand::Value(value)))
            }
            Operand::Map(entries) => {
                let value = Value::Map(Map::object(entries));
                Ok(intern_leaf(&value)
                    .map(Operand::Interned)
                    .unwrap_or(Operand::Value(value)))
            }
            other => Ok(other),
        }
    }

    fn insert_map_operand(
        &mut self,
        key: Operand,
        value: Operand,
        optional: bool,
    ) -> CelResult<()> {
        let key = self.close_builder_operand(key)?;
        let value = self.close_builder_operand(value)?;
        let value = if optional {
            match &value {
                Operand::Interned(w) if interned_optional_is_none(*w) => return Ok(()),
                Operand::Interned(w) if unsafe { w_kind(*w) } == CelKind::Optional => {
                    let inner =
                        unsafe { (*w.cast::<crate::runtime::object::W_OptionalObject>()).w_value };
                    if inner.is_null() {
                        return Ok(());
                    }
                    Operand::Interned(inner)
                }
                Operand::Value(v) => match optional_inner(v) {
                    OptView::Empty => return Ok(()),
                    OptView::Present(inner) => Operand::Value(inner),
                    OptView::Plain => value,
                },
                _ => value,
            }
        } else {
            value
        };
        if let (Some(k), Some(v)) = (Self::leaf_of(&key), Self::leaf_of(&value)) {
            if interned_is_map_key(k) {
                if let Some(Operand::MapRefs(pairs)) = self.top_mut() {
                    pairs.push((k, v));
                    return Ok(());
                }
            }
        }
        let key = self.finish(key)?;
        let value = self.finish(value)?;
        let key = value_key(key).map_err(|e| self.park(e))?;
        self.map_mut()?.insert(key, value);
        Ok(())
    }

    fn struct_mut(&mut self) -> CelResult<&mut BTreeMap<String, Value>> {
        match self.top_mut() {
            Some(Operand::Struct(_, fields)) => Ok(fields),
            _ => Err(CelErr::InternalError),
        }
    }

    // -- the loop -----------------------------------------------------------

    pub(crate) fn run(&mut self) -> CelResult<Value> {
        unsafe {
            force_virtualizable_if_necessary(self.cel_frame);
        }
        let mut pc = 0u32;
        loop {
            match self.dispatch_one(pc)? {
                Step::Next => pc += 1,
                Step::Jump(target) => pc = target,
                Step::Return(value) => return Ok(value),
            }
        }
    }

    /// One instruction, for both the interpreter loop and the JIT portal.
    pub(crate) fn dispatch_one(&mut self, pc: u32) -> CelResult<Step> {
        #[cfg(feature = "__elem-attr-probe")]
        if pc == self.anchor {
            let next = self.fused_element(pc)?;
            return Ok(Step::Jump(next));
        }
        debug_assert!(
            self.depth() <= self.code.max_stack as usize,
            "depth {} past the compiler's {} at pc {pc}",
            self.depth(),
            self.code.max_stack
        );
        let Some(&Insn { op, ops }) = self.code.insns.get(pc as usize) else {
            return Err(CelErr::InternalError);
        };
        unsafe {
            (*self.cel_frame).last_instr = i64::from(pc);
        }
        let next = pc + 1;
        match self.step(op, ops, pc, next) {
            Ok(step) => Ok(step),
            Err(err) => Ok(Step::Jump(self.unwind(err, pc)?)),
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
            .scratch
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
    #[cfg(feature = "__elem-attr-probe")]
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
        let index = match self.local_int(shape.index) {
            Some(index) => index,
            _ => return Err(CelErr::InternalError),
        };
        let len = match self.sequence_len(shape.source) {
            Ok(len) => len,
            _ => return Err(CelErr::InternalError),
        };
        if arm == FuseArm::AdvancePlusArcRoundTrip {
            // Exactly what `LoadLocal source` used to do to this slot, and
            // nothing more: one increment on the way in, one decrement on the
            // way out, on the count the whole loop shares.
            let Some(Value::List(list)) = self.local(shape.source) else {
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
            let decided = compare_values(&Value::Int(index), &Value::Int(len), accept)
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
            match self.element_at(shape.source, shape.index) {
                Ok(element) => element,
                Err(err) => return Err(err),
            }
        };
        self.store_slot(shape.var, element)?;
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
            .local_as_value(shape.var)
            .ok_or(CelErr::InternalError)?;
        let rhs = self
            .code
            .konst(shape.konst)
            .ok_or(CelErr::InternalError)?
            .clone();
        let value = binary_values_ref("mul", &lhs, &rhs).map_err(|e| self.park(e))?;
        self.append_to_list(value)?;
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
    #[cfg(feature = "__elem-attr-probe")]
    #[inline(always)]
    fn fused_advance(&mut self, shape: MapLoop) -> CelResult<u32> {
        self.advance_counter(shape.index)?;
        Ok(shape.top)
    }

    #[inline(always)]
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
            OpCode::LoadLocal => match self.local_operand(a).ok_or(CelErr::InternalError)? {
                Operand::Interned(w) => self.push_operand(Operand::Interned(*w)),
                Operand::Value(value) => self.push(value.clone()),
                _ => return Err(CelErr::InternalError),
            },
            OpCode::StoreLocal => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                self.store_operand(a, operand)?;
            }
            OpCode::IncLocal => self.advance_counter(a)?,

            // -- selection ----------------------------------------------
            OpCode::GetField => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                let field = self.name(a)?;
                if let Some(w) = Self::leaf_of(&operand) {
                    if self.push_interned_field(w, field, false)? {
                        // already pushed
                    } else {
                        let operand = self.finish(operand)?;
                        let value = value_field(&operand, field).map_err(|e| self.park(e))?;
                        self.push(value);
                    }
                } else {
                    let operand = self.finish(operand)?;
                    let value = value_field(&operand, field).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            OpCode::HasField => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                let field = self.name(a)?;
                if let Some(w) = Self::leaf_of(&operand) {
                    if self.push_interned_field(w, field, true)? {
                        // already pushed
                    } else {
                        let operand = self.finish(operand)?;
                        let value = has_field(&operand, field).map_err(|e| self.park(e))?;
                        self.push(value);
                    }
                } else {
                    let operand = self.finish(operand)?;
                    let value = has_field(&operand, field).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            OpCode::Index | OpCode::OptIndex => {
                let key = self.pop_operand().ok_or(CelErr::InternalError)?;
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                if self.try_interned_index(&operand, &key, op == OpCode::OptIndex)? {
                    // interned item already pushed
                } else {
                    let key = self.finish(key)?;
                    let operand = self.finish(operand)?;
                    let value = self.index(operand, key, op == OpCode::OptIndex)?;
                    self.push(value);
                }
            }
            OpCode::GetFieldLocal | OpCode::HasFieldLocal => {
                let field = self.name(b)?;
                if let Some(w) = self.local_leaf(a) {
                    if self.push_interned_field(w, field, op == OpCode::HasFieldLocal)? {
                        // already pushed
                    } else {
                        let operand = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                        let read = if op == OpCode::HasFieldLocal {
                            has_field(&operand, field)
                        } else {
                            value_field(&operand, field)
                        };
                        let value = read.map_err(|e| self.park(e))?;
                        self.push(value);
                    }
                } else {
                    let operand = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    let read = if op == OpCode::HasFieldLocal {
                        has_field(&operand, field)
                    } else {
                        value_field(&operand, field)
                    };
                    let value = read.map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            // Held out of the dispatch loop's own body; see
            // `Vm::opt_select_arm`.
            OpCode::OptSelect => self.opt_select_arm(a)?,

            // -- aggregates ----------------------------------------------
            OpCode::NewList => self.push_operand(Operand::EmptyList(a as usize)),
            OpCode::NewListFromArg => {
                let len = self.sequence_len(a)?;
                self.push_operand(Operand::EmptyList(len as usize));
            }
            OpCode::ListAppend => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                self.append_operand(operand)?;
            }
            OpCode::ListAppendOptional => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                match &operand {
                    Operand::Interned(w) if interned_optional_is_none(*w) => {}
                    Operand::Interned(w) if unsafe { w_kind(*w) } == CelKind::Optional => {
                        let inner = unsafe {
                            (*w.cast::<crate::runtime::object::W_OptionalObject>()).w_value
                        };
                        if !inner.is_null() {
                            self.append_operand(Operand::Interned(inner))?;
                        }
                    }
                    _ => {
                        let value = self.finish(operand)?;
                        match optional_inner(&value) {
                            OptView::Empty => {}
                            OptView::Present(inner) => self.append_to_list(inner)?,
                            OptView::Plain => self.append_to_list(value)?,
                        }
                    }
                }
            }
            // Held out too, and for the same reason; see `Vm::new_map_arm`.
            OpCode::NewMap => self.new_map_arm(),
            OpCode::MapInsert | OpCode::MapInsertOptional => {
                let value = self.pop_operand().ok_or(CelErr::InternalError)?;
                let key = self.pop_operand().ok_or(CelErr::InternalError)?;
                self.insert_map_operand(key, value, op == OpCode::MapInsertOptional)?;
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
                let rhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                let lhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                let name = match op {
                    OpCode::Add => "add",
                    OpCode::Sub => "sub",
                    OpCode::Mul => "mul",
                    OpCode::Div => "div",
                    _ => "rem",
                };
                if let (Some(a), Some(b)) = (Self::leaf_of(&lhs), Self::leaf_of(&rhs)) {
                    self.apply_cel_binop(op, name, a, b)?;
                } else {
                    let lhs = self.finish(lhs)?;
                    let rhs = self.finish(rhs)?;
                    let value = binary_values_ref(name, &lhs, &rhs).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            OpCode::Equals | OpCode::NotEquals => {
                let rhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                let lhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let (Some(a), Some(b)) = (Self::leaf_of(&lhs), Self::leaf_of(&rhs)) {
                    let w = unsafe {
                        if op == OpCode::Equals {
                            cel_equals(a, b)
                        } else {
                            cel_not_equals(a, b)
                        }
                    };
                    self.push_interned(w, op)?;
                } else {
                    let lhs = self.finish(lhs)?;
                    let rhs = self.finish(rhs)?;
                    self.push(Value::Bool((lhs == rhs) == (op == OpCode::Equals)));
                }
            }
            OpCode::Less | OpCode::LessEquals | OpCode::Greater | OpCode::GreaterEquals => {
                let rhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                let lhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let (Some(a), Some(b)) = (Self::leaf_of(&lhs), Self::leaf_of(&rhs)) {
                    let w = unsafe {
                        match op {
                            OpCode::Less => cel_less(a, b),
                            OpCode::LessEquals => cel_less_equals(a, b),
                            OpCode::Greater => cel_greater(a, b),
                            _ => cel_greater_equals(a, b),
                        }
                    };
                    self.push_interned(w, op)?;
                } else {
                    let lhs = self.finish(lhs)?;
                    let rhs = self.finish(rhs)?;
                    let accept: fn(Ordering) -> bool = match op {
                        OpCode::Less => |o| o == Ordering::Less,
                        OpCode::LessEquals => |o| o != Ordering::Greater,
                        OpCode::Greater => |o| o == Ordering::Greater,
                        _ => |o| o != Ordering::Less,
                    };
                    let value = compare_values(&lhs, &rhs, accept).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            // The three groups above with the right operand read out of the
            // constant pool. Each calls the same helper with the same
            // operands in the same order, so the error it raises carries the
            // same operator name the pair's did.
            OpCode::AddConst | OpCode::MulConst | OpCode::ModConst => {
                let lhs = self.pop_operand().ok_or(CelErr::InternalError)?;
                let rhs = self.code.konst(a).ok_or(CelErr::InternalError)?;
                let name = match op {
                    OpCode::AddConst => "add",
                    OpCode::MulConst => "mul",
                    _ => "rem",
                };
                if let (Some(a_ref), Some(b_ref)) = (Self::leaf_of(&lhs), intern_leaf(rhs)) {
                    self.apply_cel_binop(op, name, a_ref, b_ref)?;
                } else {
                    let lhs = self.finish(lhs)?;
                    let value = binary_values_ref(name, &lhs, rhs).map_err(|e| self.park(e))?;
                    self.push(value);
                }
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
                let rhs = self.code.konst(a).ok_or(CelErr::InternalError)?;
                let accept: fn(Ordering) -> bool = match op {
                    OpCode::LessConst => |o| o == Ordering::Less,
                    OpCode::GreaterConst => |o| o == Ordering::Greater,
                    _ => |o| o != Ordering::Less,
                };
                let value = compare_values(&lhs, rhs, accept).map_err(|e| self.park(e))?;
                self.push(value);
            }
            // The same three groups again with the LEFT operand read out of a
            // slot instead of popped, so no operand reaches the stack at all.
            // The helpers, their operand order and their operator names are
            // unchanged, which is what keeps the error identical to the pair's.
            OpCode::AddLocalConst | OpCode::MulLocalConst | OpCode::ModLocalConst => {
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                let name = match op {
                    OpCode::AddLocalConst => "add",
                    OpCode::MulLocalConst => "mul",
                    _ => "rem",
                };
                if let (Some(a_ref), Some(b_ref)) = (self.local_leaf(a), intern_leaf(rhs)) {
                    self.apply_cel_binop(op, name, a_ref, b_ref)?;
                } else {
                    let lhs = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    let value = binary_values_ref(name, &lhs, rhs).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            OpCode::EqualsLocalConst | OpCode::NotEqualsLocalConst => {
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                if let (Some(a_ref), Some(b_ref)) = (self.local_leaf(a), intern_leaf(rhs)) {
                    let w = unsafe {
                        if op == OpCode::EqualsLocalConst {
                            cel_equals(a_ref, b_ref)
                        } else {
                            cel_not_equals(a_ref, b_ref)
                        }
                    };
                    self.push_interned(w, op)?;
                } else {
                    let lhs = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    self.push(Value::Bool(
                        (lhs == *rhs) == (op == OpCode::EqualsLocalConst),
                    ));
                }
            }
            OpCode::LessLocalConst
            | OpCode::GreaterLocalConst
            | OpCode::GreaterEqualsLocalConst => {
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                if let (Some(a_ref), Some(b_ref)) = (self.local_leaf(a), intern_leaf(rhs)) {
                    let w = unsafe {
                        match op {
                            OpCode::LessLocalConst => cel_less(a_ref, b_ref),
                            OpCode::GreaterLocalConst => cel_greater(a_ref, b_ref),
                            _ => cel_greater_equals(a_ref, b_ref),
                        }
                    };
                    self.push_interned(w, op)?;
                } else {
                    let lhs = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    let accept: fn(Ordering) -> bool = match op {
                        OpCode::LessLocalConst => |o| o == Ordering::Less,
                        OpCode::GreaterLocalConst => |o| o == Ordering::Greater,
                        _ => |o| o != Ordering::Less,
                    };
                    let value = compare_values(&lhs, rhs, accept).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            OpCode::In => {
                let container = self.pop_operand().ok_or(CelErr::InternalError)?;
                let needle = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let (Some(c), Some(n)) = (Self::leaf_of(&container), Self::leaf_of(&needle)) {
                    let found = match unsafe { w_kind(c) } {
                        CelKind::List => Some(unsafe { list_contains(c, n) }),
                        CelKind::Map => Some(unsafe { map_contains_key(c, n) }),
                        _ => None,
                    };
                    if let Some(found) = found {
                        self.push(Value::Bool(found));
                        return Ok(Step::Next);
                    }
                }
                let rhs = self.finish(container)?;
                let lhs = self.finish(needle)?;
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
            // builder instead of pushed. `append_to_list` reads the builder off the
            // top of the stack without popping it, which is what `ListAppend`
            // does too.
            OpCode::LoadLocalAppend => {
                let operand = self.local_operand(a).ok_or(CelErr::InternalError)?.clone();
                self.append_operand(operand)?;
            }
            OpCode::GetFieldLocalAppend | OpCode::HasFieldLocalAppend => {
                let field = self.name(b)?;
                let value = if let Some(w) = self.local_leaf(a) {
                    if self.push_interned_field(w, field, op == OpCode::HasFieldLocalAppend)? {
                        let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                        self.append_operand(operand)?;
                        return Ok(Step::Next);
                    } else {
                        let operand = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                        let read = if op == OpCode::HasFieldLocalAppend {
                            has_field(&operand, field)
                        } else {
                            value_field(&operand, field)
                        };
                        read.map_err(|e| self.park(e))?
                    }
                } else {
                    let operand = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    let read = if op == OpCode::HasFieldLocalAppend {
                        has_field(&operand, field)
                    } else {
                        value_field(&operand, field)
                    };
                    read.map_err(|e| self.park(e))?
                };
                self.append_to_list(value)?;
            }
            OpCode::AddLocalConstAppend
            | OpCode::MulLocalConstAppend
            | OpCode::ModLocalConstAppend => {
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                let name = match op {
                    OpCode::AddLocalConstAppend => "add",
                    OpCode::MulLocalConstAppend => "mul",
                    _ => "rem",
                };
                if let (Some(a_ref), Some(b_ref)) = (self.local_leaf(a), intern_leaf(rhs)) {
                    let w = unsafe {
                        match name {
                            "add" => cel_add(a_ref, b_ref),
                            "mul" => cel_mul(a_ref, b_ref),
                            _ => cel_rem(a_ref, b_ref),
                        }
                    };
                    if w == ERROR_SENTINEL {
                        return Err(self.raised_as_cel_err(op));
                    }
                    self.append_operand(Operand::Interned(w))?;
                } else {
                    let lhs = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    let value = binary_values_ref(name, &lhs, rhs).map_err(|e| self.park(e))?;
                    self.append_to_list(value)?;
                }
            }
            OpCode::EqualsLocalConstAppend | OpCode::NotEqualsLocalConstAppend => {
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                let equal =
                    if let (Some(a_ref), Some(b_ref)) = (self.local_leaf(a), intern_leaf(rhs)) {
                        let w = unsafe { cel_equals(a_ref, b_ref) };
                        unsafe { (*w.cast::<crate::runtime::object::W_BoolObject>()).boolval != 0 }
                    } else {
                        let lhs = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                        lhs == *rhs
                    };
                let value = Value::Bool(equal == (op == OpCode::EqualsLocalConstAppend));
                self.append_to_list(value)?;
            }
            OpCode::LessLocalConstAppend
            | OpCode::GreaterLocalConstAppend
            | OpCode::GreaterEqualsLocalConstAppend => {
                let rhs = self.code.konst(b).ok_or(CelErr::InternalError)?;
                if let (Some(a_ref), Some(b_ref)) = (self.local_leaf(a), intern_leaf(rhs)) {
                    let w = unsafe {
                        match op {
                            OpCode::LessLocalConstAppend => cel_less(a_ref, b_ref),
                            OpCode::GreaterLocalConstAppend => cel_greater(a_ref, b_ref),
                            _ => cel_greater_equals(a_ref, b_ref),
                        }
                    };
                    if w == ERROR_SENTINEL {
                        return Err(self.raised_as_cel_err(op));
                    }
                    self.append_operand(Operand::Interned(w))?;
                } else {
                    let lhs = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    let accept: fn(Ordering) -> bool = match op {
                        OpCode::LessLocalConstAppend => |o| o == Ordering::Less,
                        OpCode::GreaterLocalConstAppend => |o| o == Ordering::Greater,
                        _ => |o| o != Ordering::Less,
                    };
                    let value = compare_values(&lhs, rhs, accept).map_err(|e| self.park(e))?;
                    self.append_to_list(value)?;
                }
            }

            // -- unary operators -------------------------------------------
            OpCode::Not => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let Some(w) = Self::leaf_of(&operand) {
                    if unsafe { w_kind(w) } != CelKind::Bool {
                        return Err(CelErr::NoSuchOverload);
                    }
                    let out = unsafe { cel_negate(w) };
                    self.push_interned(out, op)?;
                } else {
                    match self.finish(operand)? {
                        Value::Bool(b) => self.push(Value::Bool(!b)),
                        _ => return Err(CelErr::NoSuchOverload),
                    }
                }
            }
            OpCode::Negate => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let Some(w) = Self::leaf_of(&operand) {
                    let out = unsafe { cel_negate(w) };
                    self.push_interned(out, op)?;
                } else {
                    let value = self.finish(operand)?;
                    let value = value_negate(value).map_err(|e| self.park(e))?;
                    self.push(value);
                }
            }
            OpCode::NotStrictlyFalse => {
                let value = self.pop()?;
                self.push(Value::Bool(as_bool(&value).unwrap_or(true)));
            }

            // -- calls -------------------------------------------------------
            OpCode::CallHost => {
                if b == 1 && self.try_interned_size_on_top(NameId(a))? {
                    return Ok(Step::Next);
                }
                if self.try_interned_extremum(NameId(a), b as usize)? {
                    return Ok(Step::Next);
                }
                if b == 1 && self.try_interned_unary_host(NameId(a), op)? {
                    return Ok(Step::Next);
                }
                let args = self.pop_n(b as usize)?;
                let value = self.call_global(NameId(a), args)?;
                self.push(value);
            }
            OpCode::CallMethod => {
                // Taken before anything can fail, so the park cannot outlive
                // the instruction that owns it.
                let parked = self.pending_args.take();
                if parked.is_none() && b == 0 && self.try_interned_size_on_top(NameId(a))? {
                    return Ok(Step::Next);
                }
                if parked.is_none() && b == 0 && self.try_interned_unary_host(NameId(a), op)? {
                    return Ok(Step::Next);
                }
                if parked.is_none() && b == 0 && self.try_interned_temporal_accessor(NameId(a))? {
                    return Ok(Step::Next);
                }
                if parked.is_none() && b == 1 && self.try_interned_string_method(NameId(a))? {
                    return Ok(Step::Next);
                }
                if parked.is_none() && b == 1 && self.try_interned_contains_method(NameId(a))? {
                    return Ok(Step::Next);
                }
                if parked.is_none()
                    && self.try_interned_optional_method(NameId(a), b as usize, op)?
                {
                    return Ok(Step::Next);
                }
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
                if let Some(step) = self.try_interned_qualified(NameId(a), b as usize, c, op)? {
                    return Ok(step);
                }
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
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let Some(w) = Self::leaf_of(&operand) {
                    match unsafe { w_kind(w) } {
                        CelKind::List => {
                            self.push_operand(Operand::Interned(w));
                            return Ok(Step::Next);
                        }
                        CelKind::Map => {
                            self.push_operand(Operand::Interned(unsafe { interned_map_keys(w) }));
                            return Ok(Step::Next);
                        }
                        _ => {}
                    }
                }
                let value = self.finish(operand)?;
                match value {
                    // A list already IS the sequence this iterates, so the
                    // popped value is pushed straight back. Materializing it
                    // again bought a `Vec<Value>` buffer and the
                    // `Arc<ListStorage>` [`Value::list`] wraps it in -- two
                    // allocations per comprehension, and nothing else: the two
                    // instructions that read the slot, [`OpCode::IterLen`] and
                    // [`OpCode::IterAt`], are both window-relative and neither
                    // cares which buffer answers them.
                    Value::List(_) => self.push(value),
                    _ => {
                        let items = value_iter(&value).map_err(|e| self.park(e))?;
                        self.push(Value::list(items));
                    }
                }
            }
            OpCode::IterKeys => {
                let operand = self.pop_operand().ok_or(CelErr::InternalError)?;
                if let Some(w) = Self::leaf_of(&operand) {
                    match unsafe { w_kind(w) } {
                        CelKind::List => {
                            self.push_operand(Operand::Interned(unsafe {
                                interned_list_indices(w)
                            }));
                            return Ok(Step::Next);
                        }
                        CelKind::Map => {
                            self.push_operand(Operand::Interned(unsafe { interned_map_keys(w) }));
                            return Ok(Step::Next);
                        }
                        _ => {}
                    }
                }
                let value = self.finish(operand)?;
                let items = iter_keys(&value).map_err(|e| self.park(e))?;
                self.push(Value::list(items));
            }
            OpCode::IterLen => {
                let len = self.sequence_len(a)?;
                self.push_operand(Operand::Interned(new_int(len) as CelRef));
            }
            OpCode::IterAt => {
                let element = self.element_at(a, b)?;
                self.push_operand(element);
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
                let Some(index) = self.local_int(a) else {
                    return Err(CelErr::InternalError);
                };
                let len = self.sequence_len(b)?;
                if index >= len {
                    return Ok(Step::Jump(c));
                }
            }
            OpCode::IterBind => {
                let element = self.element_at(a, b)?;
                self.store_operand(c, element)?;
            }
            OpCode::IterAdvance => {
                self.advance_counter(a)?;
                return Ok(Step::Jump(b));
            }

            // -- the fused accumulator ------------------------------------
            OpCode::AccuLoopCond | OpCode::AccuLoopCondNot => {
                let accu = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                let more = if op == OpCode::AccuLoopCondNot {
                    !as_bool(&accu)?
                } else {
                    as_bool(&accu).unwrap_or(true)
                };
                if !more {
                    return Ok(Step::Jump(b));
                }
            }
            OpCode::AndLocal | OpCode::OrLocal => {
                let short = op == OpCode::OrLocal;
                let outcome = {
                    let accu = self.local_as_value(a).ok_or(CelErr::InternalError)?;
                    as_bool(&accu)
                };
                *self
                    .scratch
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
                    Some(Operand::Interned(w)) => interned_optional_is_none(*w),
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
                    .scratch
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
                let left = *self
                    .scratch
                    .logic
                    .get(a as usize)
                    .ok_or(CelErr::InternalError)?;
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

    // -- the two arms the tracer is not shown -------------------------------
    //
    // [`Vm::step`] is the graph a meta-tracing JIT has to be able to annotate:
    // it is the dispatch loop, so everything the JIT could ever see is reached
    // through it. Two of its arms build an [`Arc`] in `step`'s OWN body, and
    // `Arc::new`/`Arc::default` are callees the tracer's front end has no
    // registry entry for. The annotator stops at the first such callee, so
    // those two arms alone were what made the whole loop fall back to the
    // legacy walker.
    //
    // Moved out and marked opaque, the call is recorded and the body is not
    // walked, which is enough for `step` itself to annotate. The marker is
    // read out of the extracted LLBC rather than off the generated code -- the
    // extractor turns the MIR optimizations off, so no marked body is folded
    // back into this one and the host inliner stays free to inline both.
    //
    // ORDINARY INHERENT METHODS, deliberately: the marker is an associated
    // const the attribute puts in this impl, and a closure or a synthetic body
    // carries neither it nor a path the registry can name.
    //
    // ONLY THESE TWO, equally deliberately: what an opaque callee costs is a
    // residual call in the trace, so quarantining an arm that does not need it
    // buys nothing and hides work the JIT could have optimized. An arm that
    // reaches an `Arc` only THROUGH a helper -- `map_mut`'s `Arc::get_mut`,
    // `Value::list`'s allocation, the refcount pair a slot's `clone` takes --
    // is not one of these: the callee is in that helper's body, so it is that
    // helper's graph that stops on it and not this one's.

    /// [`OpCode::OptSelect`]: the field name, as a [`Value`], then the select.
    ///
    /// The name is what needs the [`Arc`]: `opt_select` indexes with a
    /// [`Value`] because a map key is one, and the operand it is given is
    /// spelled the same way the walker spells it.
    #[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
    fn opt_select_arm(&mut self, a: u32) -> CelResult<()> {
        let operand = self.pop()?;
        let field = Value::String(Arc::new(self.name(a)?.to_string()));
        let value = self.opt_select(operand, field)?;
        self.push(value);
        Ok(())
    }

    /// [`OpCode::NewMap`]: open a map literal.
    ///
    /// The table is built in the [`Arc`] it is handed over in, which is what
    /// [`Operand::Map`] documents; the allocation is the whole arm.
    #[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
    fn new_map_arm(&mut self) {
        self.push_operand(Operand::MapRefs(Vec::new()));
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
        match self.local_operand(slot).ok_or(CelErr::InternalError)? {
            Operand::Value(Value::List(list)) => Ok(list.len() as i64),
            Operand::Interned(w) if unsafe { w_kind(*w) } == CelKind::List => {
                Ok(unsafe { crate::runtime::object::list_len(*w) })
            }
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
    fn element_at(&mut self, sequence: u32, index: u32) -> CelResult<Operand> {
        // The probe's other half: the lowering this replaced, reachable at run
        // time so that what the replacement bought is a difference measured
        // inside one binary rather than between two builds. `value_index`
        // decides the container's kind, then the key's kind, then
        // bounds-checks, then answers in `ExecutionError` -- which `park` has
        // to record on `&mut self`, per element.
        #[cfg(feature = "__drop-arm-probe")]
        if self.probe.iter_at == IterAtArm::ViaValueIndex {
            let element = {
                let sequence = self.local(sequence).ok_or(CelErr::InternalError)?;
                let index = self.local_as_value(index).ok_or(CelErr::InternalError)?;
                value_index(sequence, &index)
            };
            return element.map(Operand::Value).map_err(|e| self.park(e));
        }
        let Some(index) = self.local_int(index) else {
            return Err(CelErr::InternalError);
        };
        match self.local_operand(sequence).ok_or(CelErr::InternalError)? {
            Operand::Value(Value::List(seq)) => seq
                .get(index as usize)
                .map(Operand::Value)
                .ok_or(CelErr::IndexOutOfBounds),
            Operand::Interned(w) if unsafe { w_kind(*w) } == CelKind::List => {
                let item =
                    unsafe { interned_list_get(*w, index) }.ok_or(CelErr::IndexOutOfBounds)?;
                Ok(Operand::Interned(item))
            }
            _ => Err(CelErr::InternalError),
        }
    }

    /// Write `value` into `slot`, dropping what was there.
    #[inline(always)]
    #[allow(dead_code)]
    fn store_slot(&mut self, slot: u32, value: Value) -> CelResult<()> {
        self.store_operand(slot, Operand::Value(value))
    }

    fn store_operand(&mut self, slot: u32, operand: Operand) -> CelResult<()> {
        let stored = match operand {
            Operand::Interned(w) => Operand::Interned(w),
            Operand::Value(v) => intern_leaf(&v)
                .map(Operand::Interned)
                .unwrap_or(Operand::Value(v)),
            builder => {
                let v = self.finish(builder)?;
                intern_leaf(&v)
                    .map(Operand::Interned)
                    .unwrap_or(Operand::Value(v))
            }
        };
        let w = Self::vable_cell(&stored);
        let dest = self.local_operand_mut(slot).ok_or(CelErr::InternalError)?;
        let previous = std::mem::replace(dest, stored);
        self.write_vable_cell(slot as usize, w);
        if let Ok(value) = self.finish(previous) {
            self.discard(value);
        }
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
        let dest = self.local_operand_mut(slot).ok_or(CelErr::InternalError)?;
        match dest {
            Operand::Value(Value::Int(counter)) => {
                *counter = counter
                    .checked_add(1)
                    .ok_or(CelErr::Overflow(OpCode::Add))?;
            }
            Operand::Interned(w) if unsafe { w_kind(*w) } == CelKind::Int => {
                let n = unsafe { (*(*w as *mut W_IntObject)).intval };
                let next = n.checked_add(1).ok_or(CelErr::Overflow(OpCode::Add))?;
                *w = new_int(next) as CelRef;
            }
            _ => return Err(CelErr::InternalError),
        }
        let w = Self::vable_cell(self.local_operand(slot).ok_or(CelErr::InternalError)?);
        self.write_vable_cell(slot as usize, w);
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

    /// `size(x)` / `x.size()` on an interned list, map, or string.
    ///
    /// PyPy's `descr_len` reads the object in place. The host overload
    /// table takes a public [`Value`], which would rebuild the container.
    fn try_interned_size_on_top(&mut self, name: NameId) -> CelResult<bool> {
        if self.name(name.0)? != "size" {
            return Ok(false);
        }
        let Some(operand) = self.top() else {
            return Ok(false);
        };
        let n = match operand {
            Operand::EmptyList(_) => 0,
            Operand::Ints(words) => words.len() as i64,
            Operand::Refs(items) => items.len() as i64,
            Operand::List(items) => items.len() as i64,
            Operand::MapRefs(pairs) => pairs.len() as i64,
            Operand::Map(entries) => entries.len() as i64,
            other => {
                let Some(w) = Self::leaf_of(other) else {
                    return Ok(false);
                };
                let Some(n) = interned_size(w) else {
                    return Ok(false);
                };
                n
            }
        };
        let _ = self.pop_operand();
        self.push_operand(Operand::Interned(new_int(n) as CelRef));
        Ok(true)
    }

    /// `x.contains(y)` when both sides are interned.
    fn try_interned_contains_method(&mut self, name: NameId) -> CelResult<bool> {
        if self.name(name.0)? != "contains" {
            return Ok(false);
        }
        if self.depth() < 2 {
            return Ok(false);
        }
        // Compile emits args then the receiver, so the receiver is on top.
        let container = self.pop_operand().ok_or(CelErr::InternalError)?;
        let needle = self.pop_operand().ok_or(CelErr::InternalError)?;
        if let (Some(c), Some(n)) = (Self::leaf_of(&container), Self::leaf_of(&needle)) {
            let found = match unsafe { w_kind(c) } {
                CelKind::List => Some(unsafe { list_contains(c, n) }),
                CelKind::Map => Some(unsafe { map_contains_key(c, n) }),
                CelKind::Str => match unsafe { crate::runtime::object::string_as_str(n) } {
                    Some(needle) => unsafe { crate::runtime::object::string_as_str(c) }
                        .map(|hay| hay.contains(needle)),
                    None => Some(false),
                },
                _ => None,
            };
            if let Some(found) = found {
                self.push(Value::Bool(found));
                return Ok(true);
            }
        }
        self.push_operand(needle);
        self.push_operand(container);
        Ok(false)
    }

    /// `s.startsWith(p)` / `s.endsWith(p)` / `s.matches(p)` on interned strings.
    ///
    /// `matches` uses the interned compiled pattern (`regex_intern`). The
    /// compile is residual (`dont_look_inside`), not a traced loop body.
    fn try_interned_string_method(&mut self, name: NameId) -> CelResult<bool> {
        let method = self.name(name.0)?;
        if !matches!(method, "startsWith" | "endsWith" | "matches") {
            return Ok(false);
        }
        #[cfg(not(feature = "regex"))]
        if method == "matches" {
            return Ok(false);
        }
        if self.depth() < 2 {
            return Ok(false);
        }
        // Compile emits args then the receiver, so the receiver is on top.
        let receiver = self.pop_operand().ok_or(CelErr::InternalError)?;
        let needle = self.pop_operand().ok_or(CelErr::InternalError)?;
        if let (Some(r), Some(n)) = (Self::leaf_of(&receiver), Self::leaf_of(&needle)) {
            if let (Some(rs), Some(ns)) = (
                unsafe { crate::runtime::object::string_as_str(r) },
                unsafe { crate::runtime::object::string_as_str(n) },
            ) {
                match method {
                    "startsWith" => {
                        self.push(Value::bool(rs.starts_with(ns)));
                        return Ok(true);
                    }
                    "endsWith" => {
                        self.push(Value::bool(rs.ends_with(ns)));
                        return Ok(true);
                    }
                    #[cfg(feature = "regex")]
                    "matches" => match crate::runtime::regex_intern::intern_regex(ns) {
                        Ok(re) => {
                            self.push(Value::bool(re.is_match(rs)));
                            return Ok(true);
                        }
                        Err(message) => {
                            return Err(self.park(ExecutionError::FunctionError {
                                function: "matches".to_string(),
                                message,
                            }));
                        }
                    },
                    _ => {}
                }
            }
        }
        self.push_operand(needle);
        self.push_operand(receiver);
        Ok(false)
    }

    /// `getHours` / `getFullYear` / … on an interned duration or timestamp.
    fn try_interned_temporal_accessor(&mut self, name: NameId) -> CelResult<bool> {
        let Some(operand) = self.top() else {
            return Ok(false);
        };
        let Some(w) = Self::leaf_of(operand) else {
            return Ok(false);
        };
        let Some(n) = interned_temporal_accessor(self.name(name.0)?, w) else {
            return Ok(false);
        };
        let _ = self.pop_operand();
        self.push_operand(Operand::Interned(new_int(n) as CelRef));
        Ok(true)
    }

    /// `max` / `min` over interned scalars or one interned list.
    fn try_interned_extremum(&mut self, name: NameId, n: usize) -> CelResult<bool> {
        let keep_greater = match self.name(name.0)? {
            "max" => true,
            "min" => false,
            _ => return Ok(false),
        };
        if n == 0 || self.depth() < n {
            return Ok(false);
        }
        let args: Vec<Operand> = self.frame[self.frame.len() - n..].to_vec();
        let refs: Option<Vec<CelRef>> = args.iter().map(Self::leaf_of).collect();
        let Some(refs) = refs else {
            return Ok(false);
        };
        let items = if refs.len() == 1 && unsafe { w_kind(refs[0]) } == CelKind::List {
            let w = refs[0];
            let len = unsafe { list_len(w) };
            if len == 0 {
                self.frame.truncate(self.frame.len() - n);
                self.push_operand(Operand::Interned(new_null() as CelRef));
                return Ok(true);
            }
            let mut items = Vec::with_capacity(len as usize);
            let mut i = 0i64;
            while i < len {
                items.push(unsafe { interned_list_get(w, i) }.ok_or(CelErr::InternalError)?);
                i += 1;
            }
            items
        } else {
            refs
        };
        let Some(best) = interned_extremum(&items, keep_greater) else {
            return Ok(false);
        };
        self.frame.truncate(self.frame.len() - n);
        self.push_operand(Operand::Interned(best));
        Ok(true)
    }

    /// `type` / `int` / `uint` / `double` / `string` / `bytes` / `dyn` on one
    /// interned operand, and the chrono constructors when the feature is on.
    ///
    /// Hosted as both [`OpCode::CallHost`] (`int(1.5)`) and
    /// [`OpCode::CallMethod`] (`(1.5).int()`): the receiver sits on top in
    /// both spellings, so the same pop works.
    fn try_interned_unary_host(&mut self, name: NameId, op: OpCode) -> CelResult<bool> {
        let Some(operand) = self.top() else {
            return Ok(false);
        };
        let Some(w) = Self::leaf_of(operand) else {
            return Ok(false);
        };
        match interned_unary_host(self.name(name.0)?, w) {
            Ok(Some(out)) => {
                let _ = self.pop_operand();
                self.push_interned(out, op)?;
                Ok(true)
            }
            Ok(None) => Ok(false),
            Err(e) => Err(self.park(e)),
        }
    }

    /// `optional.of` / `optional.none` / `optional.ofNonZeroValue`, and a
    /// namespaced `math.max` / `math.min` over interned scalars.
    ///
    /// A hit jumps past the [`OpCode::CallMethod`] the compiler emitted for
    /// the receiver fallback.
    fn try_interned_qualified(
        &mut self,
        name: NameId,
        n: usize,
        skip: u32,
        op: OpCode,
    ) -> CelResult<Option<Step>> {
        match self.name(name.0)? {
            "optional.none" if n == 0 => {
                self.push_operand(Operand::Interned(cel_optional_none()));
                Ok(Some(Step::Jump(skip)))
            }
            "optional.of" if n == 1 => self.try_interned_optional_ctor(cel_optional_of, skip, op),
            "optional.ofNonZeroValue" if n == 1 => {
                self.try_interned_optional_ctor(cel_optional_of_non_zero_value, skip, op)
            }
            _ => Ok(None),
        }
    }

    fn try_interned_optional_ctor(
        &mut self,
        ctor: unsafe fn(CelRef) -> CelRef,
        skip: u32,
        op: OpCode,
    ) -> CelResult<Option<Step>> {
        let Some(operand) = self.top() else {
            return Ok(None);
        };
        let Some(w) = Self::leaf_of(operand) else {
            return Ok(None);
        };
        let _ = self.pop_operand();
        self.push_interned(unsafe { ctor(w) }, op)?;
        Ok(Some(Step::Jump(skip)))
    }

    /// `opt.value()` / `hasValue()` / `or` / `orValue` on interned optionals.
    fn try_interned_optional_method(
        &mut self,
        name: NameId,
        n: usize,
        op: OpCode,
    ) -> CelResult<bool> {
        match self.name(name.0)? {
            "value" if n == 0 => self.try_interned_optional_unary(cel_optional_value, op),
            "hasValue" if n == 0 => self.try_interned_optional_unary(cel_optional_has_value, op),
            "or" if n == 1 => self.try_interned_optional_binary(cel_optional_or, op),
            "orValue" if n == 1 => self.try_interned_optional_binary(cel_optional_or_value, op),
            _ => Ok(false),
        }
    }

    fn try_interned_optional_unary(
        &mut self,
        call: unsafe fn(CelRef) -> CelRef,
        op: OpCode,
    ) -> CelResult<bool> {
        let Some(operand) = self.top() else {
            return Ok(false);
        };
        let Some(w) = Self::leaf_of(operand) else {
            return Ok(false);
        };
        if unsafe { w_kind(w) } != CelKind::Optional {
            return Ok(false);
        }
        let _ = self.pop_operand();
        self.push_interned(unsafe { call(w) }, op)?;
        Ok(true)
    }

    fn try_interned_optional_binary(
        &mut self,
        call: unsafe fn(CelRef, CelRef) -> CelRef,
        op: OpCode,
    ) -> CelResult<bool> {
        if self.depth() < 2 {
            return Ok(false);
        }
        let receiver = self.pop_operand().ok_or(CelErr::InternalError)?;
        let other = self.pop_operand().ok_or(CelErr::InternalError)?;
        if let (Some(r), Some(o)) = (Self::leaf_of(&receiver), Self::leaf_of(&other)) {
            if unsafe { w_kind(r) } == CelKind::Optional {
                self.push_interned(unsafe { call(r, o) }, op)?;
                return Ok(true);
            }
        }
        self.push_operand(other);
        self.push_operand(receiver);
        Ok(false)
    }

    fn call_global(&mut self, name: NameId, args: Vec<Value>) -> CelResult<Value> {
        let func_name = self.name(name.0)?;
        let args = unpack_host_args(args);
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
        let args = unpack_host_args(args);
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
        let mut with_target = unpack_host_args(args);
        with_target.insert(0, target.unpack());
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
pub(crate) enum Step {
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

/// Duration / timestamp accessors, matching `common/types/{duration,timestamp}.rs`.
///
/// A name that is not an accessor, or a receiver that is not the matching
/// leaf, returns `None` so the host overload can raise.
fn interned_temporal_accessor(name: &str, w: CelRef) -> Option<i64> {
    match unsafe { w_kind(w) } {
        CelKind::Duration => {
            let nanos = unsafe { (*w.cast::<crate::runtime::object::W_DurationObject>()).nanos };
            match name {
                "getHours" => Some(nanos / 1_000_000_000 / 3600),
                "getMinutes" => Some(nanos / 1_000_000_000 / 60),
                "getSeconds" => Some(nanos / 1_000_000_000),
                "getMilliseconds" => Some(nanos / 1_000_000),
                _ => None,
            }
        }
        #[cfg(feature = "chrono")]
        CelKind::Timestamp => interned_timestamp_accessor(name, w),
        _ => None,
    }
}

#[cfg(feature = "chrono")]
fn interned_timestamp_accessor(name: &str, w: CelRef) -> Option<i64> {
    use chrono::{Datelike, Timelike};
    let leaf = unsafe { &*w.cast::<crate::runtime::object::W_TimestampObject>() };
    let off = chrono::FixedOffset::east_opt(leaf.off_s as i32)?;
    let utc = chrono::DateTime::from_timestamp_nanos(leaf.nanos);
    let ts = utc.with_timezone(&off);
    match name {
        "getMilliseconds" => Some(i64::from(ts.timestamp_subsec_millis())),
        "getSeconds" => Some(i64::from(ts.second())),
        "getMinutes" => Some(i64::from(ts.minute())),
        "getHours" => Some(i64::from(ts.hour())),
        "getDayOfWeek" => Some(i64::from(ts.weekday().num_days_from_sunday())),
        "getDate" => Some(i64::from(ts.day())),
        "getDayOfMonth" => Some(i64::from(ts.day0())),
        "getMonth" => Some(i64::from(ts.month0())),
        "getFullYear" => Some(i64::from(ts.year())),
        "getDayOfYear" => {
            let year = ts
                .checked_sub_days(chrono::Days::new(u64::from(ts.day0())))?
                .checked_sub_months(chrono::Months::new(ts.month0()))?;
            Some(ts.signed_duration_since(year).num_days())
        }
        _ => None,
    }
}

fn interned_type_of(w: CelRef) -> CelRef {
    unsafe {
        if w_kind(w) == CelKind::Opaque {
            if let Some(host) = crate::runtime::convert::host_opaque(opaque_host_index(w)) {
                if host
                    .downcast_ref::<crate::common::types::TypeValue>()
                    .is_some()
                {
                    return new_type(&CEL_TYPE_CLASS) as CelRef;
                }
                let tv = crate::common::types::TypeValue::new(
                    crate::common::types::Type::new_opaque_type(
                        host.runtime_type_name().to_owned(),
                    ),
                );
                return intern_leaf(&Value::Opaque(Arc::new(tv)))
                    .unwrap_or(new_type(&CEL_OPAQUE_CLASS) as CelRef);
            }
        }
        new_type(&*w_type(w)) as CelRef
    }
}

fn interned_unary_host(name: &str, w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match name {
        "dyn" => Ok(Some(w)),
        "type" => Ok(Some(interned_type_of(w))),
        "int" => interned_to_int(w),
        "uint" => interned_to_uint(w),
        "double" => interned_to_double(w),
        "string" => interned_to_string(w),
        "bytes" => interned_to_bytes(w),
        #[cfg(feature = "chrono")]
        "duration" => interned_to_duration(w),
        #[cfg(feature = "chrono")]
        "timestamp" => interned_to_timestamp(w),
        _ => Ok(None),
    }
}

fn interned_to_int(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::Int => Ok(Some(w)),
        CelKind::UInt => Ok(Some(
            new_int(unsafe { (*w.cast::<W_UIntObject>()).uintval } as i64) as CelRef,
        )),
        CelKind::Double => Ok(Some(
            new_int(unsafe { (*w.cast::<W_DoubleObject>()).floatval } as i64) as CelRef,
        )),
        CelKind::Str => match unsafe { string_as_str(w) } {
            Some(s) => match s.parse::<i64>() {
                Ok(i) => Ok(Some(new_int(i) as CelRef)),
                Err(e) => Err(ExecutionError::FunctionError {
                    function: "int".to_owned(),
                    message: format!("string parse error: {e}"),
                }),
            },
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

fn interned_to_uint(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::UInt => Ok(Some(w)),
        CelKind::Int => Ok(Some(
            new_uint(unsafe { (*w.cast::<W_IntObject>()).intval } as u64) as CelRef,
        )),
        CelKind::Double => Ok(Some(
            new_uint(unsafe { (*w.cast::<W_DoubleObject>()).floatval } as u64) as CelRef,
        )),
        CelKind::Str => match unsafe { string_as_str(w) } {
            Some(s) => match s.parse::<u64>() {
                Ok(u) => Ok(Some(new_uint(u) as CelRef)),
                Err(e) => Err(ExecutionError::FunctionError {
                    function: "int".to_owned(),
                    message: format!("string parse error: {e}"),
                }),
            },
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

fn interned_to_double(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::Double => Ok(Some(w)),
        CelKind::Int => Ok(Some(
            new_double(unsafe { (*w.cast::<W_IntObject>()).intval } as f64) as CelRef,
        )),
        CelKind::UInt => Ok(Some(
            new_double(unsafe { (*w.cast::<W_UIntObject>()).uintval } as f64) as CelRef,
        )),
        CelKind::Str => match unsafe { string_as_str(w) } {
            Some(s) => match s.parse::<f64>() {
                Ok(f) => Ok(Some(new_double(f) as CelRef)),
                Err(e) => Err(ExecutionError::FunctionError {
                    function: "double".to_owned(),
                    message: format!("string parse error: {e}"),
                }),
            },
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

fn interned_to_string(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::Str => Ok(Some(w)),
        CelKind::Int => Ok(Some(new_string(
            &unsafe { (*w.cast::<W_IntObject>()).intval }.to_string(),
        ) as CelRef)),
        CelKind::UInt => Ok(Some(new_string(
            &unsafe { (*w.cast::<W_UIntObject>()).uintval }.to_string(),
        ) as CelRef)),
        CelKind::Double => Ok(Some(new_string(
            &unsafe { (*w.cast::<W_DoubleObject>()).floatval }.to_string(),
        ) as CelRef)),
        CelKind::Bytes => {
            let n = unsafe { crate::runtime::object::bytes_len(w) } as usize;
            let leaf = unsafe { &*w.cast::<crate::runtime::object::W_BytesObject>() };
            let base = unsafe { crate::runtime::object_array::bytes_base(leaf.data) };
            if base.is_null() && n != 0 {
                return Ok(None);
            }
            let bytes = unsafe { std::slice::from_raw_parts(base, n) };
            Ok(Some(
                new_string(&String::from_utf8_lossy(bytes).into_owned()) as CelRef,
            ))
        }
        #[cfg(feature = "chrono")]
        CelKind::Timestamp => match unsafe { ref_to_value(w) } {
            Ok(Value::Timestamp(ts)) => Ok(Some(new_string(&ts.to_rfc3339()) as CelRef)),
            _ => Ok(None),
        },
        #[cfg(feature = "chrono")]
        CelKind::Duration => match unsafe { ref_to_value(w) } {
            Ok(Value::Duration(d)) => Ok(Some(
                new_string(&crate::duration::format_duration(&d)) as CelRef
            )),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}

fn interned_to_bytes(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::Bytes => Ok(Some(w)),
        CelKind::Str => match unsafe { string_as_str(w) } {
            Some(s) => Ok(Some(new_bytes(s.as_bytes()) as CelRef)),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

#[cfg(feature = "chrono")]
fn interned_to_duration(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::Duration => Ok(Some(w)),
        CelKind::Str => {
            let Some(s) = (unsafe { string_as_str(w) }) else {
                return Ok(None);
            };
            match crate::duration::parse_duration(s) {
                Ok((_, parsed)) => match parsed.num_nanoseconds() {
                    Some(nanos) => Ok(Some(crate::runtime::object::new_duration(nanos) as CelRef)),
                    None => Ok(None),
                },
                Err(e) => Err(ExecutionError::function_error("duration", e.to_string())),
            }
        }
        _ => Ok(None),
    }
}

#[cfg(feature = "chrono")]
fn interned_to_timestamp(w: CelRef) -> Result<Option<CelRef>, ExecutionError> {
    match unsafe { w_kind(w) } {
        CelKind::Timestamp => Ok(Some(w)),
        CelKind::Str => {
            let Some(s) = (unsafe { string_as_str(w) }) else {
                return Ok(None);
            };
            match chrono::DateTime::parse_from_rfc3339(s) {
                Ok(parsed) => match parsed.timestamp_nanos_opt() {
                    // A date outside the i64-nanos window still has a public
                    // `Value::Timestamp`; the class-family leaf cannot hold it.
                    Some(nanos) => Ok(Some(crate::runtime::object::new_timestamp(
                        nanos,
                        i64::from(parsed.offset().local_minus_utc()),
                    ) as CelRef)),
                    None => Ok(None),
                },
                Err(e) => Err(ExecutionError::function_error("timestamp", e.to_string())),
            }
        }
        _ => Ok(None),
    }
}

fn unpack_host_args(args: Vec<Value>) -> Vec<Value> {
    args.into_iter().map(|v| v.unpack()).collect()
}

fn interned_int(operand: &Operand) -> Option<i64> {
    match operand {
        Operand::Interned(k) if unsafe { w_kind(*k) } == CelKind::Int => {
            Some(unsafe { (*k.cast::<W_IntObject>()).intval })
        }
        Operand::Value(Value::Int(i)) => Some(*i),
        Operand::Value(v) => match intern_leaf(v) {
            Some(k) if unsafe { w_kind(k) } == CelKind::Int => {
                Some(unsafe { (*k.cast::<W_IntObject>()).intval })
            }
            _ => None,
        },
        _ => None,
    }
}

unsafe fn interned_map_keys(w: CelRef) -> CelRef {
    new_list(&map_key_refs(w)) as CelRef
}

unsafe fn interned_list_indices(w: CelRef) -> CelRef {
    let n = list_len(w);
    let mut keys = Vec::with_capacity(n as usize);
    let mut i = 0i64;
    while i < n {
        keys.push(new_int(i) as CelRef);
        i += 1;
    }
    new_list(&keys) as CelRef
}

fn interned_extremum(items: &[CelRef], keep_greater: bool) -> Option<CelRef> {
    let mut best = *items.first()?;
    for &item in &items[1..] {
        let cmp = unsafe {
            if keep_greater {
                cel_greater(best, item)
            } else {
                cel_less(best, item)
            }
        };
        if cmp == ERROR_SENTINEL {
            return None;
        }
        let keep =
            unsafe { w_kind(cmp) == CelKind::Bool && (*cmp.cast::<W_BoolObject>()).boolval != 0 };
        if !keep {
            best = item;
        }
    }
    Some(best)
}

fn interned_is_map_key(w: CelRef) -> bool {
    matches!(
        unsafe { w_kind(w) },
        CelKind::Int | CelKind::UInt | CelKind::Bool | CelKind::Str
    )
}

fn interned_size(w: CelRef) -> Option<i64> {
    match unsafe { w_kind(w) } {
        CelKind::List => Some(unsafe { list_len(w) }),
        CelKind::Map => Some(unsafe { map_len(w) }),
        CelKind::Str => Some(unsafe { string_byte_len(w) }),
        CelKind::Bytes => Some(unsafe { crate::runtime::object::bytes_len(w) }),
        _ => None,
    }
}

fn interned_optional_is_none(w: CelRef) -> bool {
    unsafe {
        w_kind(w) == CelKind::Optional
            && (*w.cast::<crate::runtime::object::W_OptionalObject>())
                .w_value
                .is_null()
    }
}

fn optional_inner(value: &Value) -> OptView {
    let unpacked = value.unpack();
    match as_optional(&unpacked) {
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
    match value.unpack() {
        Value::Bool(b) => Ok(b),
        _ => Err(CelErr::NoSuchOverload),
    }
}

/// `has(x.y)`.
fn has_field(operand: &Value, field: &str) -> Result<Value, ExecutionError> {
    match &operand.unpack() {
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
    let right = match right.unpack() {
        Value::Bool(b) => Some(b),
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
    #[cfg(feature = "__elem-attr-probe")]
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

    /// `append_to_list` finds the builder under the operand that was popped.
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
        vm.push_operand(Operand::EmptyList(0));
        vm.push(Value::Int(2));
        assert_eq!(vm.depth(), 2);
        assert_eq!(vm.pop(), Ok(Value::Int(2)));
        vm.push(Value::Int(2));

        // `ListAppend`: pop, and only then reach for the builder.
        assert_eq!(vm.pop(), Ok(Value::Int(2)));
        assert_eq!(vm.depth(), 1);
        vm.append_to_list(Value::Int(2))
            .expect("the builder is under the popped value");

        // Truncating away an operand above it leaves an aggregate that still
        // closes, which is what the unwind path depends on.
        vm.push(Value::Int(3));
        vm.truncate(1);
        assert_eq!(vm.depth(), 1);
        assert_eq!(vm.pop(), Ok(Value::list(vec![Value::Int(2)])));
        assert_eq!(vm.depth(), 0);

        // An operand still held at the end goes back to the pool with the
        // frame, where `Scratch::release` drops it; nothing else has to.
        vm.push(Value::Int(4));
        assert_eq!(vm.depth(), 1);
    }

    /// Interned stack slots are written through the virtualizable array.
    #[test]
    fn interned_slots_land_on_the_cel_frame() {
        let code = CelCode {
            n_slots: 1,
            max_stack: 2,
            ..CelCode::default()
        };
        let ctx = Context::default();
        let mut vm = Vm::new(&code, &ctx);
        unsafe {
            assert_eq!((*vm.cel_frame).last_instr, -1);
            assert_eq!((*vm.cel_frame).valuestackdepth, 1);
            assert_eq!((*vm.cel_frame).vable_token, 0);
        }
        let w = crate::runtime::object::new_int(7) as CelRef;
        vm.push_operand(Operand::Interned(w));
        unsafe {
            assert_eq!((*vm.cel_frame).valuestackdepth, 2);
            assert_eq!(*cel_frame_slot(vm.cel_frame, 1), w);
        }
        match vm.pop_operand() {
            Some(Operand::Interned(got)) => assert_eq!(got, w),
            _ => panic!("expected interned"),
        }
        unsafe {
            assert_eq!((*vm.cel_frame).valuestackdepth, 1);
        }
    }

    /// `1 + 2` stays on the interned `int` table through `cel_add`.
    #[test]
    fn interned_add_of_small_ints_is_the_prebuilt() {
        let expr = parse("1 + 2");
        let code = compile(&expr).expect("compile");
        let ctx = Context::default();
        let value = cel_eval_loop(&code, &ctx).expect("eval");
        assert_eq!(value, Value::Int(3));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&value),
            Some(crate::runtime::object::new_int(3) as crate::runtime::object::CelRef)
        );
    }

    /// A comprehension counter stored as an interned int stays on the table.
    #[test]
    fn interned_uint_and_float_add_through_the_vm() {
        let ctx = Context::default();
        let uint = {
            let expr = parse("1u + 2u");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(uint, Value::UInt(3));
        assert!(crate::runtime::convert::intern_leaf(&uint).is_some());
        let float = {
            let expr = parse("1.5 + 2.25");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(float, Value::Float(3.75));
        assert!(crate::runtime::convert::intern_leaf(&float).is_some());
    }

    #[test]
    fn interned_object_map_equality_through_the_vm() {
        let expr = parse("{'a': 1} == {'a': 1}");
        let code = compile(&expr).expect("compile");
        let ctx = Context::default();
        let value = cel_eval_loop(&code, &ctx).expect("eval");
        assert_eq!(value, Value::Bool(true));
    }

    /// A comprehension counter stored as an interned int stays on the table.
    #[test]
    fn interned_map_field_and_list_index_through_the_vm() {
        let ctx = Context::default();
        let field = {
            let expr = parse("{'a': 7}.a");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(field, Value::Int(7));
        let missing = {
            let expr = parse("has({'a': 7}.b)");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(missing, Value::Bool(false));
        let item = {
            let expr = parse("[10, 20, 30][1]");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(item, Value::Int(20));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&item),
            Some(crate::runtime::object::new_int(20) as crate::runtime::object::CelRef)
        );
    }

    #[test]
    fn interned_in_size_and_map_index_through_the_vm() {
        let ctx = Context::default();
        let contained = {
            let expr = parse("2 in [1, 2, 3]");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(contained, Value::Bool(true));
        let missing = {
            let expr = parse("9 in [1, 2, 3]");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(missing, Value::Bool(false));
        let sized = {
            let expr = parse("size([1, 2, 3])");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(sized, Value::Int(3));
        let method = {
            let expr = parse("[1, 2, 3].size()");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(method, Value::Int(3));
        let keyed = {
            let expr = parse("{'a': 7}['a']");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(keyed, Value::Int(7));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&keyed),
            Some(crate::runtime::object::new_int(7) as crate::runtime::object::CelRef)
        );
        let sized_bytes = {
            let expr = parse("size(b'foo')");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(sized_bytes, Value::Int(3));
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("'foobar'.startsWith('foo')")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Bool(true)
        );
        let mut max_ctx = Context::default();
        max_ctx.add_function("max", crate::functions::max);
        let maxed = cel_eval_loop(&compile(&parse("max(1, 3, 2)")).expect("compile"), &max_ctx)
            .expect("eval");
        assert_eq!(maxed, Value::Int(3));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&maxed),
            Some(crate::runtime::object::new_int(3) as crate::runtime::object::CelRef)
        );
    }

    /// A comprehension counter stored as an interned int stays on the table.
    #[test]
    fn interned_optional_and_duration_through_the_vm() {
        let ctx = Context::default();
        let opt = {
            let expr = parse("optional.of(3) == optional.of(3)");
            let code = compile(&expr).expect("compile");
            cel_eval_loop(&code, &ctx).expect("eval")
        };
        assert_eq!(opt, Value::Bool(true));
        let unwrapped = cel_eval_loop(
            &compile(&parse("optional.of(3).value()")).expect("compile"),
            &ctx,
        )
        .expect("eval");
        assert_eq!(unwrapped, Value::Int(3));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&unwrapped),
            Some(crate::runtime::object::new_int(3) as crate::runtime::object::CelRef)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("optional.none().hasValue()")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Bool(false)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("optional.ofNonZeroValue(0).hasValue()")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Bool(false)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("optional.none().orValue(9)")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Int(9)
        );
        #[cfg(feature = "chrono")]
        {
            let dur = {
                let expr = parse("duration(\"1s\") + duration(\"2s\")");
                let code = compile(&expr).expect("compile");
                cel_eval_loop(&code, &ctx).expect("eval")
            };
            assert_eq!(dur, Value::Duration(chrono::Duration::seconds(3)));
            assert!(crate::runtime::convert::intern_leaf(&dur).is_some());
        }
    }

    #[test]
    fn interned_type_and_conversions_through_the_vm() {
        let ctx = Context::default();
        let ty = cel_eval_loop(&compile(&parse("type(1)")).expect("compile"), &ctx).expect("eval");
        assert_eq!(
            ty,
            cel_eval_loop(&compile(&parse("type(2)")).expect("compile"), &ctx).expect("eval")
        );
        assert!(crate::runtime::convert::intern_leaf(&ty).is_some());
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("type(type(1)) == type(string)")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Bool(true)
        );
        let as_int =
            cel_eval_loop(&compile(&parse("int(1.9)")).expect("compile"), &ctx).expect("eval");
        assert_eq!(as_int, Value::Int(1));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&as_int),
            Some(crate::runtime::object::new_int(1) as crate::runtime::object::CelRef)
        );
        let as_string =
            cel_eval_loop(&compile(&parse("string(7)")).expect("compile"), &ctx).expect("eval");
        assert_eq!(as_string, Value::String(Arc::new("7".into())));
        assert!(crate::runtime::convert::intern_leaf(&as_string).is_some());
        let as_bytes =
            cel_eval_loop(&compile(&parse("bytes('ab')")).expect("compile"), &ctx).expect("eval");
        assert_eq!(as_bytes, Value::Bytes(Arc::new(b"ab".to_vec())));
    }

    #[test]
    fn interned_temporal_accessors_and_matches_through_the_vm() {
        let ctx = Context::default();
        let hours = cel_eval_loop(
            &compile(&parse("duration('3661s').getHours()")).expect("compile"),
            &ctx,
        )
        .expect("eval");
        assert_eq!(hours, Value::Int(1));
        assert_eq!(
            crate::runtime::convert::intern_leaf(&hours),
            Some(crate::runtime::object::new_int(1) as crate::runtime::object::CelRef)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("duration('3661s').getMinutes()")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Int(61)
        );
        assert_eq!(
            cel_eval_loop(
                &compile(&parse("duration('1s').getMilliseconds()")).expect("compile"),
                &ctx
            )
            .expect("eval"),
            Value::Int(1000)
        );
        #[cfg(feature = "chrono")]
        {
            let ts = "timestamp('2020-01-02T03:04:05.006Z')";
            assert_eq!(
                cel_eval_loop(
                    &compile(&parse(&format!("{ts}.getFullYear()"))).expect("compile"),
                    &ctx
                )
                .expect("eval"),
                Value::Int(2020)
            );
            assert_eq!(
                cel_eval_loop(
                    &compile(&parse(&format!("{ts}.getMonth()"))).expect("compile"),
                    &ctx
                )
                .expect("eval"),
                Value::Int(0)
            );
            assert_eq!(
                cel_eval_loop(
                    &compile(&parse(&format!("{ts}.getDate()"))).expect("compile"),
                    &ctx
                )
                .expect("eval"),
                Value::Int(2)
            );
            assert_eq!(
                cel_eval_loop(
                    &compile(&parse(&format!("{ts}.getHours()"))).expect("compile"),
                    &ctx
                )
                .expect("eval"),
                Value::Int(3)
            );
            assert_eq!(
                cel_eval_loop(
                    &compile(&parse(&format!("{ts}.getDayOfYear()"))).expect("compile"),
                    &ctx
                )
                .expect("eval"),
                Value::Int(1)
            );
        }
        #[cfg(feature = "regex")]
        {
            let matched = cel_eval_loop(
                &compile(&parse("'abc'.matches('a.*')")).expect("compile"),
                &ctx,
            )
            .expect("eval");
            assert_eq!(matched, Value::Bool(true));
            assert!(crate::runtime::convert::intern_leaf(&matched).is_some());
            assert_eq!(
                cel_eval_loop(
                    &compile(&parse("'abc'.matches('z+')")).expect("compile"),
                    &ctx
                )
                .expect("eval"),
                Value::Bool(false)
            );
        }
    }

    /// A comprehension counter stored as an interned int stays on the table.
    #[test]
    fn interned_string_concat_stays_on_the_class_family() {
        let expr = parse("'he' + 'llo'");
        let code = compile(&expr).expect("compile");
        let ctx = Context::default();
        let value = cel_eval_loop(&code, &ctx).expect("eval");
        assert_eq!(value, Value::String(std::sync::Arc::new("hello".into())));
    }

    /// A comprehension counter stored as an interned int stays on the table.
    #[test]
    fn interned_local_increment_stays_on_the_prebuilt() {
        let expr = parse("[0, 1, 2].map(x, x + 1)");
        let code = compile(&expr).expect("compile");
        let ctx = Context::default();
        let value = cel_eval_loop(&code, &ctx).expect("eval");
        assert_eq!(
            value,
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)])
        );
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
            vm.scratch.cold.len() <= 1,
            "64 absorbed errors left {} parked",
            vm.scratch.cold.len()
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
        let ctx = Context::default();
        let code = compile(&parse(r#"{"x": {"y": 3}}"#)).expect("compiles");
        let mut vm = Vm::new(&code, &ctx);

        // `Vm::run`'s loop with one line added: each builder is recorded the
        // first time it is seen on top of the stack. Neither table is freed
        // before the answer is read -- the inner one moves into the outer --
        // so no recorded address can be reused by the other.
        let mut opened_hash = 0usize;
        let mut opened_refs = 0usize;
        let mut pc = 0u32;
        let answer = loop {
            let &Insn { op, ops } = vm.code.insns.get(pc as usize).expect("`pc` is in range");
            let next = pc + 1;
            let step = vm.step(op, ops, pc, next).expect("the literal evaluates");
            match vm.top() {
                Some(Operand::Map(_)) => opened_hash += 1,
                Some(Operand::MapRefs(_)) => opened_refs += 1,
                _ => {}
            }
            match step {
                Step::Next => pc = next,
                Step::Jump(target) => pc = target,
                Step::Return(value) => break value,
            }
        };

        assert_eq!(
            opened_hash, 0,
            "an internable literal must not box a HashMap"
        );
        assert!(opened_refs > 0, "the builder must stay on interned pairs");
        assert_eq!(answer, {
            let inner = Value::Map(crate::objects::Map::object(Arc::new(
                [(
                    crate::objects::Key::String(Arc::new("y".into())),
                    Value::Int(3),
                )]
                .into_iter()
                .collect(),
            )));
            Value::Map(crate::objects::Map::object(Arc::new(
                [(crate::objects::Key::String(Arc::new("x".into())), inner)]
                    .into_iter()
                    .collect(),
            )))
        });
        assert!(
            crate::runtime::convert::intern_leaf(&answer).is_some(),
            "the closed literal must be an interned map"
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
        assert_eq!(vm.depth(), 0, "the operand stack was left dirty");
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
