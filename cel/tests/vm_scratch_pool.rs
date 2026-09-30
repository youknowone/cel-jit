//! The evaluator's activation record must not be a heap allocation.
//!
//! `Program::execute` used to size an operand stack — and, where the program
//! has a comprehension or a short-circuit operator, a slot array and a logic
//! array — on every call. The tree walker's floor is zero, because its operands
//! are recursion locals, so that was a cost the bytecode evaluator paid for
//! being a loop rather than for doing any work.
//!
//! What is pinned here is the PROPERTY, not the mechanism: evaluating a scalar
//! expression allocates nothing, and re-entering the evaluator from a host
//! function still answers correctly. Both hold for the tree walker too, so most
//! of this file is meaningful under either setting of the `vm` feature.
//!
//! The one place the two legs part company is a comprehension, and it is marked
//! where it happens rather than left to a reader to discover: the walker
//! materializes the sequence it iterates, so `xs.all(v, ..)` has a floor of its
//! own there that has nothing to do with an activation record. Those rows are
//! `vm`-only. Everything else is asserted in both.
//!
//! ## Why an absolute assertion and not a differential one
//!
//! `tests/oracle.rs` holds the two evaluators to the same ANSWERS, which is
//! blind to anything they both do — and to a cost neither of them gets wrong.
//! An allocation floor is not an answer, so it needs a number.
//!
//! ## Isolation
//!
//! The counter is thread-local, so the default multi-threaded test runner
//! cannot contaminate a measured window with another test's allocations. It is
//! `const`-initialised and destructor-free, so the allocator can touch it
//! without recursing through lazy TLS setup. `probe_the_meter_itself` is what
//! says the meter reads zero for an empty window and one for a single box,
//! rather than a comment claiming it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;
use std::sync::Arc;

use cel::{Context, ExecutionError, Program, Value};

// ---------------------------------------------------------------------------
// the meter
// ---------------------------------------------------------------------------

std::thread_local! {
    static LOCAL_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    // Forwarded rather than left to the default alloc+copy+dealloc, so an
    // in-place growth stays in place; one realloc is one allocation event.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = LOCAL_ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Allocations this thread makes while running `body`.
///
/// Reading a `Cell` allocates nothing, which is what lets the window be this
/// tight.
fn allocations(mut body: impl FnMut()) -> u64 {
    let before = LOCAL_ALLOCS.with(Cell::get);
    body();
    LOCAL_ALLOCS.with(Cell::get) - before
}

/// Enough calls that every one-off is behind the measured window: the lazily
/// registered thread-local destructor, any lazy static the evaluator reaches,
/// and the pooled buffers arriving at the width this program needs.
const WARMUP: usize = 8;

/// Run `body` once per warm-up call, then report the allocations of one more.
fn allocations_when_warm(mut body: impl FnMut()) -> u64 {
    for _ in 0..WARMUP {
        body();
    }
    allocations(body)
}

#[test]
fn probe_the_meter_itself() {
    assert_eq!(
        allocations(|| {
            black_box(0u64);
        }),
        0,
        "an empty window must read 0"
    );
    assert_eq!(
        allocations(|| drop(black_box(Box::new(0u64)))),
        1,
        "one box must read 1 — a meter that counted nothing would pass every \
         other assertion in this file vacuously"
    );
}

// ---------------------------------------------------------------------------
// the floor
// ---------------------------------------------------------------------------

/// Expressions whose evaluation needs no heap at all.
///
/// The three shapes are the three buffers the evaluator used to size per call:
/// an operand stack, which every program that compiles needs at least one entry
/// of; a logic array, which a program containing `&&` or `||` needs; and a slot
/// array, which a program containing a comprehension needs. Each row names
/// which of them it exercises, so a regression says which buffer came back.
#[test]
fn a_scalar_expression_allocates_nothing() {
    // `(source, expected)`. Every source is closed over constants or over
    // variables bound before the window opens, so nothing here is measuring a
    // context build.
    let mut cases: Vec<(&str, Value)> = vec![
        // operand stack only
        ("1 + 2 * 3 - 4 / 2", Value::Int(5)),
        ("x > 10 ? x * 2 : x + 5", Value::Int(30)),
        ("((a + b) * (c - d)) / 2", Value::Int(375)),
        ("flag", Value::Bool(true)),
        // operand stack + the logic array
        ("10 > 5 && 3 < 7 || 1 == 1", Value::Bool(true)),
        ("flag && x > 1 && a < b", Value::Bool(true)),
    ];
    // Operand stack + the slot array + the logic array — the widest activation
    // record a program can ask for, and the only shape here that is not also
    // free in the tree walker. The walker materializes the sequence it iterates
    // rather than reading through the bound list, so it pays for the ITERATION
    // whatever the activation record costs; asserting zero there would be
    // asserting something about a different mechanism that happens to share a
    // name with this one.
    if cfg!(feature = "vm") {
        cases.extend([
            ("items.all(v, v >= 0)", Value::Bool(true)),
            ("items.exists(v, v == 3)", Value::Bool(true)),
        ]);
    }

    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    ctx.add_variable_from_value("a", 10i64);
    ctx.add_variable_from_value("b", 20i64);
    ctx.add_variable_from_value("c", 30i64);
    ctx.add_variable_from_value("d", 5i64);
    ctx.add_variable_from_value("flag", true);
    ctx.add_variable_from_value("items", (0..8i64).collect::<Vec<i64>>());

    for (source, expected) in &cases {
        let program = Program::compile(source).unwrap_or_else(|e| panic!("{source}: {e:?}"));
        // Assert the answer before measuring: a count taken off an `Err`
        // measures the error path, not the expression.
        assert_eq!(
            program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("{source}: {e:?}")),
            *expected,
            "{source}"
        );
        let allocs = allocations_when_warm(|| {
            black_box(program.execute(&ctx)).ok();
        });
        assert_eq!(
            allocs, 0,
            "`{source}` allocated {allocs} time(s) per evaluation, and its \
             activation record is expected to cost nothing. The operand stack, \
             the comprehension slots and the short-circuit outcomes are held \
             across calls precisely so that a scalar expression has no heap \
             floor; a non-zero here names one of them being sized per call \
             again."
        );
    }
}

/// The floor holds whatever program ran last, in either order.
///
/// The buffers are shared across programs on a thread, so a wide program
/// followed by a narrow one — and a narrow one followed by a wide one — are two
/// different questions. Only the second could grow a buffer inside the window,
/// and it is the one a single-program test cannot ask.
#[test]
fn the_floor_survives_alternating_program_widths() {
    let ctx = Context::default();
    let narrow = Program::compile("1 + 1").expect("compiles");
    let wide = Program::compile("((1 + 2) * (3 - 4)) / ((5 + 6) - (7 * 8))").expect("compiles");

    // Warm both, so neither is the first sighting of its width.
    for _ in 0..WARMUP {
        narrow.execute(&ctx).expect("evaluates");
        wide.execute(&ctx).expect("evaluates");
    }
    let allocs = allocations(|| {
        black_box(narrow.execute(&ctx)).ok();
        black_box(wide.execute(&ctx)).ok();
        black_box(narrow.execute(&ctx)).ok();
    });
    assert_eq!(
        allocs, 0,
        "alternating a 2-deep and a 4-deep program allocated {allocs} time(s). \
         A shared buffer that is trimmed to the last program's width has to \
         grow again for the next one, which is a per-call allocation for any \
         caller holding more than one program."
    );
}

// ---------------------------------------------------------------------------
// re-entry
// ---------------------------------------------------------------------------

/// A host function that evaluates a program of its own, three levels deep.
///
/// This is the shape that makes a shared buffer unsound: the outer evaluation
/// is suspended with its operands live when the inner one starts. Asserted on
/// the ANSWER rather than on any internal state, because a buffer the two
/// evaluations shared would corrupt the answer and nothing else about the run
/// would look wrong.
#[test]
fn a_host_function_may_re_enter_the_evaluator() {
    fn eval_int(source: &str, ctx: &Context) -> i64 {
        match Program::compile(source)
            .unwrap_or_else(|e| panic!("{source}: {e:?}"))
            .execute(ctx)
        {
            Ok(Value::Int(n)) => n,
            other => panic!("{source}: expected an int, got {other:?}"),
        }
    }

    let mut ctx = Context::default();
    ctx.add_function("middle", || -> i64 {
        let mut ctx = Context::default();
        ctx.add_function("innermost", || -> i64 {
            // Deliberately not a constant: a comprehension needs the slot
            // array, so the deepest level uses the same buffers as the two
            // above it.
            eval_int(
                "size([1, 2, 3, 4, 5].filter(y, y > 3)) + 8",
                &Context::default(),
            )
        });
        // A short-circuit as well as a comprehension, so the middle level holds
        // a logic array open across the innermost call.
        eval_int(
            "(innermost() > 0 && innermost() < 100) ? innermost() * 2 : 0",
            &ctx,
        )
    });

    // The outer expression suspends a comprehension — one slot array, one
    // operand stack with a half-built list on it — for the duration of each
    // `middle()` call.
    let program = Program::compile("[1, 2, 3].map(v, v * middle())").expect("compiles");
    assert_eq!(
        program.execute(&ctx).expect("evaluates"),
        Value::from(vec![20i64, 40, 60]),
        "re-entering the evaluator from a host function changed the outer \
         answer, which is what a shared operand stack looks like from outside"
    );

    // And it is repeatable: a nested call must leave the pool in a state the
    // next top-level call can use.
    for _ in 0..4 {
        assert_eq!(
            program.execute(&ctx).expect("evaluates"),
            Value::from(vec![20i64, 40, 60])
        );
    }
    let ordinary = Program::compile("1 + 1").expect("compiles");
    assert_eq!(
        allocations_when_warm(|| {
            black_box(ordinary.execute(&ctx)).ok();
        }),
        0,
        "an ordinary evaluation after a nested one no longer hits the floor, so \
         the nesting did not give the buffers back"
    );
}

// ---------------------------------------------------------------------------
// the ways out that are not `Ok`
// ---------------------------------------------------------------------------

/// A run that ends in an error must leave nothing rooted.
///
/// The list literal is half built when the division fails, so the operand stack
/// still holds three clones of `s` at the moment the evaluation gives up. If
/// the buffers went back to the pool as they were, those clones would outlive
/// the call and stay alive until some later evaluation on this thread happened
/// to overwrite them — a retention hazard proportional to whatever the caller
/// was building, not to anything the caller still holds.
///
/// Measured on a reference count rather than on an allocation count, because
/// what is wrong in that case is a value being KEPT, and an allocation counter
/// cannot see a value that is merely not dropped.
#[test]
fn a_failed_run_drops_what_was_on_the_stack() {
    let s = Arc::new(String::from(
        "a string big enough to be worth not retaining",
    ));
    let mut ctx = Context::default();
    ctx.add_variable_from_value("s", Value::String(Arc::clone(&s)));
    ctx.add_variable_from_value("zero", 0i64);
    let held_by_the_context = Arc::strong_count(&s);

    let program = Program::compile("[s, s, s, 1 / zero]").expect("compiles");
    for _ in 0..4 {
        assert!(
            program.execute(&ctx).is_err(),
            "the case depends on this failing part-way through the list literal"
        );
        assert_eq!(
            Arc::strong_count(&s),
            held_by_the_context,
            "a failed evaluation left the half-built list alive. The operand \
             stack outlives the call by design — that is what removes the \
             per-call allocation — so it has to be emptied on the way out and \
             not merely on the way in."
        );
    }

    // The error itself is still reported in full, which is the thing emptying
    // the buffers could plausibly have broken.
    assert!(matches!(
        program.execute(&ctx),
        Err(ExecutionError::DivisionByZero(_))
    ));
}

/// A host function that panics must not poison the evaluator for this thread.
///
/// The buffers are handed back by a destructor precisely so that unwinding is
/// one of the ways out, and the assertion is that the next evaluation on this
/// thread is both correct and still at the floor.
#[test]
fn a_panicking_host_function_leaves_the_evaluator_usable() {
    let mut ctx = Context::default();
    ctx.add_function("explode", || -> i64 {
        panic!("host function panicking on purpose")
    });
    let exploding = Program::compile("[1, 2, 3].map(v, v * explode())").expect("compiles");

    // The comprehension's floor is whatever building its result list costs, so
    // it is measured on this thread BEFORE the panic rather than asserted as a
    // number. What the panic could break is the fixed part, and a relation
    // between two measurements is what isolates it. `scalar` pins the same
    // thing in absolute terms, because a comprehension that came back one
    // higher AND a result list that got cheaper would cancel here.
    let ordinary = Program::compile("[1, 2, 3].map(v, v * 2)").expect("compiles");
    let scalar = Program::compile("1 + 1").expect("compiles");
    let floor_before = allocations_when_warm(|| {
        black_box(ordinary.execute(&ctx)).ok();
    });

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = exploding.execute(&ctx);
    }));
    std::panic::set_hook(previous);
    assert!(caught.is_err(), "the host function was expected to panic");

    assert_eq!(
        ordinary.execute(&ctx).expect("evaluates"),
        Value::from(vec![2i64, 4, 6]),
        "an evaluation after a panicking one answered wrongly, so the buffers \
         came back holding the abandoned run's operands"
    );
    let floor_after = allocations_when_warm(|| {
        black_box(ordinary.execute(&ctx)).ok();
    });
    assert_eq!(
        floor_after, floor_before,
        "the same comprehension cost {floor_before} allocation(s) before a host \
         function panicked and {floor_after} after, so unwinding did not give \
         the buffers back and the next call sized its own"
    );
    assert_eq!(
        allocations_when_warm(|| {
            black_box(scalar.execute(&ctx)).ok();
        }),
        0,
        "a scalar expression after a panicking evaluation no longer hits its \
         floor of zero"
    );
}
