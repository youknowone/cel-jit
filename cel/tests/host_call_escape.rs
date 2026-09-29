//! A host call forces the portal frame (`residual_dispatch` runs
//! `force_virtualizable_if_necessary`), so tracing through it must abort with
//! `ABORT_ESCAPE` and hand the half-walked jitcode frames to the blackhole.
//! Compiling the call instead leaves it without GUARD_NOT_FORCED, and the
//! runtime force then reads a null `jf_force_descr`.
//!
//! Both tests lower the portal threshold so the host-call program is traced.
//! The variable is process-wide, which is why they live in their own binary.

#![cfg(feature = "jit")]

use std::cell::Cell;
use std::rc::Rc;

use cel::{Context, Program, Value};

fn low_threshold() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
}

#[test]
fn traced_host_call_keeps_its_result() {
    low_threshold();
    let program = Program::compile("f(x) + 1").expect("compiles");
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 41i64);
    ctx.add_function("f", |a: i64| -> i64 { a });
    for i in 0..2000 {
        assert_eq!(program.execute(&ctx).expect("runs"), Value::Int(42), "iteration {i}");
    }
}

/// Host `n()` re-enters the same program; the nested evaluation's `n`
/// returns a constant.
#[test]
fn traced_reentrant_host_call_does_not_force_a_null_descr() {
    low_threshold();
    struct Shared(Rc<Program>, Rc<Cell<bool>>);
    // SAFETY: the test runs on one thread; `add_function` asks for `Send`.
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    let program = Rc::new(Program::compile("n()").expect("compiles"));
    let shared = Shared(Rc::clone(&program), Rc::new(Cell::new(false)));
    let mut ctx = Context::default();
    ctx.add_function("n", move || -> i64 {
        if shared.1.replace(true) {
            return 1;
        }
        let mut inner = Context::default();
        inner.add_function("n", || 1i64);
        let got = shared.0.execute(&inner).expect("nested");
        shared.1.set(false);
        match got {
            Value::Int(n) => n,
            other => panic!("nested: expected int, got {other:?}"),
        }
    });
    for i in 0..400 {
        assert_eq!(program.execute(&ctx).expect("outer"), Value::Int(1), "iteration {i}");
    }
}
