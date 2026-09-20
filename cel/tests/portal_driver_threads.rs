//! Concurrent evaluation of a shared [`Program`].
//!
//! Callers share one `Arc<Program>` across threads. The portal driver the
//! first thread stores on the code object must not be borrowed mutably by
//! a second thread.
//!
//! [`Program`] contains [`cel::Value`] (interned `*mut` leaves in consts),
//! which the type system does not mark `Send`. The newtype below is the
//! sharing the public door already does: one compiled program, many threads.

#![cfg(feature = "jit")]

use std::sync::Arc;
use std::thread;

use cel::{Context, Program, Value};

/// One compiled program shared across threads.
struct SharedProgram(Program);

unsafe impl Send for SharedProgram {}
unsafe impl Sync for SharedProgram {}

fn ctx_with_x_and_list() -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>());
    ctx
}

fn mapped_list() -> Value {
    Value::from((1..=10i64).map(|n| n * 2).collect::<Vec<_>>())
}

/// Host `n()` re-enters the same [`Program`] with a context whose `n`
/// returns a constant, so the nested evaluation does not recurse.
fn reenter_ctx(program: Arc<SharedProgram>) -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_function("n", move || -> i64 {
        std::thread_local! {
            static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        if DEPTH.with(|d| d.get()) >= 1 {
            return 1;
        }
        DEPTH.with(|d| d.set(1));
        let mut inner = Context::default();
        inner.add_function("n", || 1i64);
        let got = program.0.execute(&inner).expect("reenter");
        DEPTH.with(|d| d.set(0));
        match got {
            Value::Int(n) => n,
            other => panic!("reenter: expected int, got {other:?}"),
        }
    });
    ctx
}

#[test]
fn eight_threads_share_one_arc_program() {
    let arith = Arc::new(SharedProgram(
        Program::compile("x * 2 + 1").expect("arith"),
    ));
    let mapped = Arc::new(SharedProgram(
        Program::compile("list.map(i, i * 2)").expect("map"),
    ));
    let reenter = Arc::new(SharedProgram(Program::compile("n()").expect("reenter")));

    let mut joins = Vec::new();
    for _ in 0..8 {
        let arith = Arc::clone(&arith);
        let mapped = Arc::clone(&mapped);
        let reenter = Arc::clone(&reenter);
        joins.push(thread::spawn(move || {
            let ctx = ctx_with_x_and_list();
            let re_ctx = reenter_ctx(Arc::clone(&reenter));
            let want_map = mapped_list();
            for i in 0..20_000 {
                match i % 3 {
                    0 => assert_eq!(arith.0.execute(&ctx).expect("arith"), Value::Int(31)),
                    1 => assert_eq!(mapped.0.execute(&ctx).expect("map"), want_map),
                    _ => assert_eq!(
                        reenter.0.execute(&re_ctx).expect("reenter"),
                        Value::Int(1)
                    ),
                }
            }
        }));
    }
    for j in joins {
        j.join().expect("thread");
    }
}

#[test]
fn created_on_a_evaluated_on_a_then_b_dropped_on_b() {
    let program = SharedProgram(Program::compile("x * 2 + 1").expect("compiles"));
    let ctx = ctx_with_x_and_list();
    assert_eq!(program.0.execute(&ctx).expect("A"), Value::Int(31));
    let join = thread::spawn(move || {
        let ctx = ctx_with_x_and_list();
        assert_eq!(program.0.execute(&ctx).expect("B"), Value::Int(31));
        drop(program);
    });
    join.join().expect("B");
}

#[test]
fn created_on_a_evaluated_on_b_then_a_dropped_on_a() {
    let program = SharedProgram(Program::compile("x * 2 + 1").expect("compiles"));
    let program = thread::spawn(move || {
        let ctx = ctx_with_x_and_list();
        assert_eq!(program.0.execute(&ctx).expect("B"), Value::Int(31));
        program
    })
    .join()
    .expect("B");
    let ctx = ctx_with_x_and_list();
    assert_eq!(program.0.execute(&ctx).expect("A"), Value::Int(31));
    drop(program);
}
