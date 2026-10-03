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

/// Past the function-entry door the host entry word is a constant.
/// `add_function` replaces the registry generation, so later calls run
/// the new body. A second context does not reuse the first word.
#[test]
fn compiled_host_call_observes_reregister() {
    let mut ctx = Context::default();
    bind_xy(&mut ctx);
    ctx.add_function("add", |a: i64, b: i64| a + b);
    ctx.add_function("multiply", |a: i64, b: i64| a * b);
    let program = Program::compile("add(x, y) + multiply(a, b)").unwrap();
    for i in 0..400 {
        assert_eq!(program.execute(&ctx).unwrap(), Value::Int(45), "warm {i}");
    }

    ctx.add_function("add", |a: i64, b: i64| a.wrapping_mul(b));
    let replaced = Value::Int(10 * 20 + 5 * 3);
    for i in 0..8 {
        assert_eq!(
            program.execute(&ctx).unwrap(),
            replaced,
            "after re-register {i}"
        );
    }

    let mut other = Context::default();
    bind_xy(&mut other);
    other.add_function("add", |a: i64, b: i64| a.wrapping_sub(b));
    other.add_function("multiply", |a: i64, b: i64| a * b);
    assert_eq!(
        program.execute(&other).unwrap(),
        Value::Int(10 - 20 + 5 * 3)
    );
    assert_eq!(program.execute(&ctx).unwrap(), replaced);
}

/// One root registry serves every fresh child. A fresh root is a new registry.
///
/// Each evaluation binds a different `x`. The child case shares the root
/// registry, so compiled loops and bridges stay bounded when the portal
/// is on.
#[cfg(feature = "vm")]
#[test]
fn compiled_host_call_across_fresh_scopes() {
    let expr = Parser::default()
        .parse("add(x, y) + multiply(a, b)")
        .unwrap();
    let code = cel::vm::compile(&expr).unwrap();
    let mut root = Context::default();
    root.add_function("add", |a: i64, b: i64| a + b);
    root.add_function("multiply", |a: i64, b: i64| a * b);

    const N: i64 = 5000;
    for i in 0..N {
        let x = 10 + (i % 17);
        let mut child = root.new_inner_scope();
        child.add_variable_from_value("x", x);
        child.add_variable_from_value("y", 20i64);
        child.add_variable_from_value("a", 5i64);
        child.add_variable_from_value("b", 3i64);
        let got = cel::vm::cel_eval_loop(&code, &child).unwrap();
        assert_eq!(got, Value::Int(x + 20 + 5 * 3), "child {i}");
    }

    #[cfg(feature = "jit")]
    let child_counts = cel::vm::portal::portal_compile_counts(&code);

    for i in 0..N {
        let x = 10 + (i % 17);
        let mut ctx = Context::default();
        ctx.add_function("add", |a: i64, b: i64| a + b);
        ctx.add_function("multiply", |a: i64, b: i64| a * b);
        ctx.add_variable_from_value("x", x);
        ctx.add_variable_from_value("y", 20i64);
        ctx.add_variable_from_value("a", 5i64);
        ctx.add_variable_from_value("b", 3i64);
        let got = cel::vm::cel_eval_loop(&code, &ctx).unwrap();
        assert_eq!(got, Value::Int(x + 20 + 5 * 3), "root {i}");
    }

    #[cfg(feature = "jit")]
    {
        let (loops, bridges, retraces, guards) = child_counts;
        assert!(
            loops >= 1 && loops + bridges <= 8,
            "child compiled loops+bridges grew with N={N}: loops={loops} bridges={bridges} retraces={retraces} guards={guards}"
        );
    }
}

/// Straight-line code compiles at the function entry once the driver's
/// threshold is crossed. The replacement has to be visible in that code.
#[cfg(feature = "jit")]
#[test]
fn reregister_after_the_portal_compiles() {
    // SAFETY: read when this program's driver is created, which is its
    // first execute below. Same knob `portal_reentry` uses.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };

    let mut ctx = Context::default();
    bind_xy(&mut ctx);
    ctx.add_function("add", |a: i64, b: i64| a + b);
    ctx.add_function("multiply", |a: i64, b: i64| a * b);
    let program = Program::compile("add(x, y) + multiply(a, b)").unwrap();
    for i in 0..200 {
        let vm = program.execute(&ctx);
        assert_eq!(vm.unwrap(), Value::Int(45), "warm {i}");
    }

    ctx.add_function("add", |a: i64, b: i64| a.wrapping_mul(b));
    let walker = Value::resolve_value(program.expression(), &ctx).unwrap();
    assert_eq!(walker, Value::Int(10 * 20 + 5 * 3));
    for i in 0..5 {
        assert_eq!(
            program.execute(&ctx).unwrap(),
            walker,
            "after re-register {i}"
        );
    }

    ctx.add_function(
        "multiply",
        |a: i64, b: i64| -> Result<i64, ExecutionError> {
            let _ = (a, b);
            Err(ExecutionError::function_error("multiply", "replaced"))
        },
    );
    let walker = Value::resolve_value(program.expression(), &ctx);
    let vm = program.execute(&ctx);
    assert_eq!(show(&walker), show(&vm));
    assert!(vm.is_err(), "erased replacement must run");
}
