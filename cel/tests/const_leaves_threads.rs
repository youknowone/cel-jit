//! A Program compiled on one thread is executed on another after that
//! thread's heap is gone.
//!
//! Constants wrapped at compile must not live in the compiling thread's
//! heap: `LoadConst` would then hold pointers into dropped memory. Debug
//! assertions fill and check live leaves; this file is meant to run with
//! debug assertions on.

use std::sync::Arc;
use std::thread;

use cel::{Context, Program, Value};

/// One compiled program shared across threads. [`Program`] contains interned
/// `*mut` leaves; the public door already shares one compiled program across
/// many threads. Leaves are immutable after compile and freed only when the
/// last `CelCode` drops.
struct SharedProgram(Program);

unsafe impl Send for SharedProgram {}
unsafe impl Sync for SharedProgram {}

#[test]
fn compiled_on_a_thread_that_exits_then_executed_on_main() {
    let program = thread::spawn(|| {
        let p = Program::compile(r#"["hello", 2.5, b"xyz"]"#).expect("compiles");
        let ctx = Context::default();
        let got = p.execute(&ctx).expect("compile-thread eval");
        assert_eq!(
            got,
            Value::list(vec![
                Value::String(Arc::new("hello".to_string())),
                Value::Float(2.5),
                Value::Bytes(Arc::new(b"xyz".to_vec())),
            ])
        );
        drop(ctx);
        SharedProgram(p)
    })
    .join()
    .expect("compile thread");

    let ctx = Context::default();
    for _ in 0..1_000 {
        let got = program.0.execute(&ctx).expect("main eval");
        assert_eq!(
            got,
            Value::list(vec![
                Value::String(Arc::new("hello".to_string())),
                Value::Float(2.5),
                Value::Bytes(Arc::new(b"xyz".to_vec())),
            ])
        );
    }
    drop(ctx);
    drop(program);
}
