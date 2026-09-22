//! Each thread binds and evaluates its own [`Context`].
//!
//! `Context` is neither `Send` nor `Sync`: interned leaves are pointers into
//! a bind region attached to the binding thread's heap, and
//! `&dyn VariableResolver` has no `Send` bound. A Context cannot move onto
//! another thread. The supported concurrent pattern is one Context per
//! thread, which this file exercises for both doors.

use std::thread;

use cel::{Context, Program, Value};

#[test]
fn four_threads_each_own_a_context() {
    let mut joins = Vec::new();
    for i in 0..4i64 {
        joins.push(thread::spawn(move || {
            let mut ctx = Context::default();
            ctx.add_variable_from_value("x", i);
            let program = Program::compile("x").expect("compiles");
            assert_eq!(program.execute(&ctx).expect("vm"), Value::Int(i));
            assert_eq!(
                Value::resolve_value(program.expression(), &ctx).expect("walker"),
                Value::Int(i)
            );
            let plus = Program::compile("x + 1").expect("compiles");
            assert_eq!(plus.execute(&ctx).expect("vm +"), Value::Int(i + 1));
            assert_eq!(
                Value::resolve_value(plus.expression(), &ctx).expect("walker +"),
                Value::Int(i + 1)
            );
        }));
    }
    for j in joins {
        j.join().expect("thread");
    }
}
