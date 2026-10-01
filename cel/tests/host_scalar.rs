//! An arity-2 `i64` host closure goes through `ScalarFn::Int2` on
//! `Program::execute`. Answers and errors match the tree walker, and
//! `add_function` replaces the closure the next call sees.

use std::sync::Arc;

use cel::parser::Parser;
use cel::{Context, ExecutionError, Program, Value};

fn show(r: &Result<Value, ExecutionError>) -> String {
    match r {
        Ok(v) => format!("{v:?}"),
        Err(e) => format!("{e:?}"),
    }
}

fn agree(ctx: &Context, src: &str) {
    let expr = Parser::default()
        .parse(src)
        .unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let walker = Value::resolve_value(&expr, ctx);
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    let vm = program.execute(ctx);
    assert_eq!(show(&walker), show(&vm), "`{src}`");
    assert_eq!(walker, vm, "`{src}`");
}

fn bind_xy(ctx: &mut Context) {
    ctx.add_variable_from_value("x", 10i64);
    ctx.add_variable_from_value("y", 20i64);
    ctx.add_variable_from_value("a", 5i64);
    ctx.add_variable_from_value("b", 3i64);
    ctx.add_variable_from_value("s", "no");
}

#[test]
fn int2_host_call_matches_the_walker() {
    let mut ctx = Context::default();
    bind_xy(&mut ctx);
    ctx.add_function("add", |a: i64, b: i64| a + b);
    ctx.add_function("multiply", |a: i64, b: i64| a * b);
    ctx.add_function("twice", |n: i64| n * 2);
    ctx.add_function("fadd", |a: f64, b: f64| a + b);
    // `size` is a unary stdlib name. Two ints miss that overload.
    ctx.add_function("size", |a: i64, b: i64| a + b);

    agree(&ctx, "add(x, y) + multiply(a, b)");
    agree(&ctx, "add(1000, 2000)");
    agree(&ctx, "twice(21)");
    agree(&ctx, "fadd(1.5, 2.5)");
    agree(&ctx, "size(3, 4)");
    agree(&ctx, "add(1, 2, 3)");
}

#[test]
fn int2_host_errors_match_the_walker() {
    let mut ctx = Context::default();
    bind_xy(&mut ctx);
    ctx.add_function("add", |a: i64, b: i64| a.wrapping_add(b));
    ctx.add_function("checked", |a: i64, b: i64| -> Result<i64, ExecutionError> {
        a.checked_add(b)
            .ok_or_else(|| ExecutionError::Overflow("+", Value::Int(a), Value::Int(b)))
    });

    agree(&ctx, "add(s, y)");
    agree(&ctx, "add(1u, 2)");
    agree(&ctx, "missing(x, y)");
    agree(&ctx, "add(9223372036854775807 + 1, 1)");
    agree(&ctx, "checked(1, 2)");
    agree(&ctx, "checked(9223372036854775807, 1)");
    agree(&ctx, "add(9223372036854775807, 1)");
}

#[test]
fn stdlib_overload_wins_over_a_registered_int2() {
    let mut env = cel::Env::stdlib();
    env.add_overload(
        "add",
        "add_int",
        vec![cel::common::types::INT_TYPE, cel::common::types::INT_TYPE],
        |_args| Ok(Value::Int(42)),
    )
    .unwrap();
    let mut ctx = Context::with_env(Arc::new(env));
    bind_xy(&mut ctx);
    ctx.add_function("add", |a: i64, b: i64| a + b);
    agree(&ctx, "add(1, 2)");
    // The miss is cached. A second call still takes the overload.
    agree(&ctx, "add(x, y)");
}

#[test]
fn reregister_takes_effect_on_the_next_call() {
    let mut ctx = Context::default();
    bind_xy(&mut ctx);
    ctx.add_function("add", |a: i64, b: i64| -> Result<i64, ExecutionError> {
        Ok(a + b + 1)
    });
    let program = Program::compile("add(x, y)").unwrap();
    assert_eq!(program.execute(&ctx).unwrap(), Value::Int(31));

    ctx.add_function("add", |a: i64, b: i64| a + b);
    assert_eq!(program.execute(&ctx).unwrap(), Value::Int(30));
    assert_eq!(
        Value::resolve_value(program.expression(), &ctx).unwrap(),
        Value::Int(30)
    );
    // Filling the cache, then bumping the generation from another name.
    assert_eq!(program.execute(&ctx).unwrap(), Value::Int(30));
    ctx.add_function("other", |a: i64, b: i64| a - b);
    assert_eq!(program.execute(&ctx).unwrap(), Value::Int(30));

    ctx.add_function("add", |a: i64, b: i64| a.wrapping_mul(b));
    agree(&ctx, "add(x, y)");
    assert_eq!(program.execute(&ctx).unwrap(), Value::Int(200));

    ctx.add_function("add", |a: i64, b: i64| -> Result<i64, ExecutionError> {
        let _ = (a, b);
        Err(ExecutionError::function_error("add", "replaced"))
    });
    agree(&ctx, "add(x, y)");
}
