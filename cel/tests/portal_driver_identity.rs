//! JIT driver table is keyed by a minted code id and sweeps dead entries.

#![cfg(feature = "jit")]

use cel::{Context, Program, Value};

#[test]
fn driver_table_is_keyed_by_code_id_and_sweeps_dead_entries() {
    let ctx = Context::default();
    let mut max_len = 0usize;
    for i in 0..1000 {
        let a = Program::compile("1").expect("A");
        assert_eq!(a.execute(&ctx).expect("eval A"), Value::Int(1));
        drop(a);
        let src = format!("{i} + {}", i + 1);
        let b = Program::compile(&src).expect("B");
        let got = b.execute(&ctx).expect("eval B");
        assert_eq!(got, Value::Int(i as i64 + (i as i64 + 1)));
        let n = cel::vm::portal_driver_len();
        max_len = max_len.max(n);
        assert!(
            n <= 8,
            "driver table length {n} after iteration {i} is not bounded"
        );
        drop(b);
    }
    assert!(max_len <= 8, "driver table grew to {max_len}");
}

/// The thread that first evaluates a program stores the driver on the
/// code object, so this thread's table stays empty.
#[test]
fn the_owner_thread_does_not_put_the_driver_in_the_table() {
    let ctx = Context::default();
    let program = Program::compile("1 + 1").expect("compiles");
    assert_eq!(program.execute(&ctx).expect("eval"), Value::Int(2));
    assert_eq!(
        cel::vm::portal_driver_len(),
        0,
        "the owner thread reaches the driver from the program, not the table"
    );
}

/// A second thread evaluating the same program uses the per-thread table
/// rather than aliasing the owner's driver.
#[test]
fn a_second_thread_uses_the_table_for_a_program_it_does_not_own() {
    use std::sync::Arc;
    struct Shared(Program);
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    let ctx = Context::default();
    let program = Arc::new(Shared(Program::compile("1 + 1").expect("compiles")));
    assert_eq!(program.0.execute(&ctx).expect("owner"), Value::Int(2));
    assert_eq!(cel::vm::portal_driver_len(), 0);

    let other = Arc::clone(&program);
    let n = std::thread::spawn(move || {
        let ctx = Context::default();
        assert_eq!(other.0.execute(&ctx).expect("other"), Value::Int(2));
        cel::vm::portal_driver_len()
    })
    .join()
    .expect("other thread");
    assert!(
        n >= 1 && n <= 8,
        "other thread table length {n} is not a live fallback entry"
    );
}
