//! Compiled `Program::execute` of a comprehension whose result list is
//! larger than a nursery segment.
//!
//! `CALL_MALLOC_NURSERY_VARSIZE` for an 8192-int column (64 KiB plus header)
//! misses the nursery bump and the slow path used to return NULL without
//! arming MemoryError, so `propagate_exception_handle_fail` panicked.

#![cfg(feature = "jit")]

use std::collections::HashMap;

use cel::objects::ListStorage;
use cel::{Context, Program, Value};

fn ints_ctx(n: usize) -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value(
        "xs",
        Value::list(ListStorage::Ints((0..n as i64).collect())),
    );
    ctx
}

fn rows_ctx(n: usize) -> Context<'static> {
    let mut ctx = Context::default();
    let rows: Vec<Value> = (0..n as i64)
        .map(|i| {
            let mut m = HashMap::new();
            m.insert("a", Value::Int(i));
            m.insert("c", Value::Int(i % 10));
            Value::from(m)
        })
        .collect();
    ctx.add_variable_from_value("rows", Value::list(rows));
    ctx
}

fn agree(src: &str, ctx: &Context, times: usize) {
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    let walker = Value::resolve_value(program.expression(), ctx)
        .unwrap_or_else(|e| panic!("walker {src}: {e}"));
    for i in 0..times {
        let got = program
            .execute(ctx)
            .unwrap_or_else(|e| panic!("{src} execute {i}: {e}"));
        assert_eq!(got, walker, "{src} execute {i} vs walker");
    }
}

#[cfg(feature = "jit")]
#[test]
fn compiled_large_map_filter_rows_match_the_walker() {
    // SAFETY: stored before this test builds a driver. Both knobs are
    // read once, when that driver is created. Function-entry compilation
    // is what runs `alloc_int_column` in compiled code; the loop door
    // allocates the column in the interpreter first.
    unsafe {
        std::env::set_var("CEL_PORTAL_THRESHOLD", "10");
        std::env::set_var("CEL_PORTAL_FUNCTION_THRESHOLD", "10");
    }

    // majit's default `function_threshold` is 1619. Go past it so the
    // compiled function entry runs `alloc_int_column` for the result
    // list. n=8192 is the first size `jit_batch one map2` panics at.
    for n in [64usize, 16384, 70000] {
        agree("xs.map(x, x * 2)", &ints_ctx(n), 2000);
        agree("xs.filter(x, x % 3 == 0)", &ints_ctx(n), 2000);
        agree("rows.map(r, r.a + r.c)", &rows_ctx(n), 2000);
    }
}
