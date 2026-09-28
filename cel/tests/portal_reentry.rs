//! A compiled portal must agree with the walker, including a comprehension
//! whose body evaluates another comprehension.
//!
//! `list.map` / `list.all` at threshold 100 used to return
//! `InternalError("internal VM error")` once the portal had compiled.

use cel::parser::Parser;
use cel::{Context, ExecutionError, Program, Value};

fn show(r: &Result<Value, ExecutionError>) -> String {
    match r {
        Ok(v) => format!("{v:?}"),
        Err(e) => format!("ERR({e:?})"),
    }
}

fn ctx() -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>());
    ctx
}

fn agree_compiled(src: &str, times: usize) {
    let expr = Parser::default()
        .parse(src)
        .unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let ctx = ctx();
    let walker = Value::resolve_value(&expr, &ctx);
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    for i in 0..times {
        let vm = program.execute(&ctx);
        assert_eq!(show(&walker), show(&vm), "`{src}` execute {i}");
        if let (Ok(a), Ok(b)) = (&walker, &vm) {
            assert_eq!(a, b, "`{src}` execute {i} value");
        }
    }
}

#[test]
fn compiled_entries_match_the_walker() {
    // `portal_threshold` reads this existing knob. 100 is the threshold at
    // which `list.map` / `list.all` used to fail.
    //
    // SAFETY: stored before this test builds a driver. The knob is read
    // once, when that driver is created.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list.map(e, e * 2)", 150);
    agree_compiled("list.all(e, e > 0)", 150);
    agree_compiled("list.filter(e, e > 3)", 150);
    agree_compiled("x * 2 + 1", 150);
    agree_compiled("list[2]", 150);
    agree_compiled("{\"a\": x}", 150);
    // Inner comprehension, evaluated while the outer map frame is live.
    // Each call walks a length-10 outer loop, so the back-edge counter
    // passes the threshold well inside this window.
    agree_compiled("list.map(e, [1, 2].map(i, i + e))", 150);
    // `Sub` has no fused form, so the loop body is the portal match's
    // fallback, which calls `step_hot`.
    agree_compiled("list.map(e, e - 1)", 150);
}
