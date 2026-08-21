//! Acceptance: the VM evaluates through exactly one dispatch loop.
//!
//! `jit_merge_point` is keyed on `(pc, code)`. That key only has meaning if
//! there is a single place where `pc` advances: it is what gives a trace its
//! identity, what the back-edge counter counts, and what a deopt resumes into.
//! A second loop, or a nested re-entry into the interpreter, gives the tracer
//! two interpreters again — which is the exact defect converging onto one VM
//! exists to remove.
//!
//! So this pins structure, not behaviour, and it reads the source text to do
//! it. A structural test has no baseline to disagree with, which means every
//! path that fails to *look* is a silent pass. Each anchor below therefore
//! refuses when it matches nothing, rather than reporting a vacuous zero.
//!
//! `include_str!` is deliberate: the path is resolved at compile time, so a
//! moved or renamed `interp.rs` is a build failure rather than a test that
//! quietly reads nothing.

const INTERP_RS: &str = include_str!("../src/vm/interp.rs");

/// Where the unit-test module starts. Everything above it is the code that
/// actually runs an expression; everything below it is fixtures, which are
/// allowed their own loops and their own `Vm::new`.
const TEST_MODULE_MARKER: &str = "\n#[cfg(test)]\n";

/// The name the dispatch loop lives in. Pinned separately from the loop count
/// so that renaming it is a loud failure instead of a silently different test.
const DISPATCH_FN: &str = "fn run(&mut self) -> CelResult<Value> {";

/// Below this the slice has collapsed to something that could not contain an
/// interpreter, and any count taken from it is meaningless.
const PRODUCTION_FLOOR_BYTES: usize = 5_000;

/// The production half of `interp.rs`, or a panic explaining why the split
/// could not be made.
fn production_source() -> &'static str {
    let occurrences = INTERP_RS.matches(TEST_MODULE_MARKER).count();
    assert_eq!(
        occurrences,
        1,
        "cel/src/vm/interp.rs has {occurrences} `{}` markers, expected exactly 1. \
         This test splits production code from test fixtures on that marker; \
         without exactly one it cannot tell them apart, and every count below \
         would be taken over the wrong text.",
        TEST_MODULE_MARKER.trim()
    );

    let (production, _fixtures) = INTERP_RS
        .split_once(TEST_MODULE_MARKER)
        .expect("the marker was just counted");

    assert!(
        production.len() >= PRODUCTION_FLOOR_BYTES,
        "the production half of cel/src/vm/interp.rs is {} bytes, below the \
         {PRODUCTION_FLOOR_BYTES}-byte floor. Either the interpreter moved or \
         the split landed in the wrong place; either way the assertions below \
         would pass by looking at almost nothing.",
        production.len()
    );

    assert!(
        production.contains(DISPATCH_FN),
        "cel/src/vm/interp.rs no longer declares `{DISPATCH_FN}`. The dispatch \
         loop is what this test exists to pin, so a rename is a failure here \
         even when the code is correct — update this constant deliberately."
    );

    production
}

/// `needle` must appear exactly `expected` times. Written as its own helper so
/// the message always carries the actual count: `0` and `2` are different
/// failures with different causes, and a bare `assert!` would hide which.
#[track_caller]
fn occurs_exactly(production: &str, needle: &str, expected: usize, why: &str) {
    let found = production.matches(needle).count();
    assert_eq!(
        found, expected,
        "cel/src/vm/interp.rs (production half) contains `{needle}` {found} \
         times, expected {expected}. {why}"
    );
}

#[test]
fn the_vm_has_exactly_one_dispatch_loop() {
    let production = production_source();

    occurs_exactly(
        production,
        "loop {",
        1,
        "The interpreter must advance `pc` in one place. A second loop is a \
         second merge-point candidate, and the tracer cannot key `(pc, code)` \
         on both.",
    );

    occurs_exactly(
        production,
        ".insns.get(",
        1,
        "Exactly one site may read an instruction out of the code object. Two \
         readers means two things claim to be the interpreter. The program is \
         pre-decoded, so that read is an index into `insns`; the needle is \
         updated deliberately when the spelling changes, never widened.",
    );
}

#[test]
fn the_interpreter_is_not_re_entered_recursively() {
    let production = production_source();

    // `cel_eval_loop` builds the machine and runs it; nothing inside the
    // machine may build another one. A nested `Vm` would evaluate a
    // subexpression on its own `pc`, which is the recursive tree-walk shape
    // the bytecode VM replaced.
    occurs_exactly(
        production,
        "Vm::new(",
        1,
        "Only the entry point may construct a machine. A second construction \
         site is a nested interpreter.",
    );

    occurs_exactly(
        production,
        ".run()",
        1,
        "The dispatch loop must be entered once per evaluation. A second entry \
         is recursive descent wearing a bytecode costume.",
    );
}
