//! No `Value::Interned` at any depth of a walker or VM result.

use std::collections::HashMap;
use std::sync::Arc;

use cel::objects::{Key, OptionalValue};
use cel::parser::Parser;
use cel::{Context, ExecutionError, Value};

fn has_interned(v: &Value) -> bool {
    match v {
        Value::Interned(_) => true,
        Value::List(list) => list.iter().any(|e| has_interned(&e)),
        Value::Map(map) => map.iter().any(|(_, e)| has_interned(e.as_ref())),
        Value::Opaque(o) => match o.downcast_ref::<OptionalValue>() {
            Some(opt) => opt.value().is_some_and(has_interned),
            None => false,
        },
        _ => false,
    }
}

fn make_ctx() -> Context<'static> {
    let mut ctx = Context::default();
    ctx.add_variable_from_value(
        "xs",
        Value::list(vec![
            Value::String(Arc::new("a".to_string())),
            Value::list(vec![Value::Int(1)]),
        ]),
    );
    ctx.add_variable_from_value("empty", Value::list(Vec::<Value>::new()));
    ctx.add_variable_from_value("s", Value::String(Arc::new("hello".to_string())));
    let mut m = HashMap::new();
    m.insert(
        Key::String(Arc::new("a".to_string())),
        Value::list(vec![Value::Int(1)]),
    );
    m.insert(
        Key::String(Arc::new("b".to_string())),
        Value::String(Arc::new("x".to_string())),
    );
    ctx.add_variable_from_value("m", Value::Map(cel::objects::Map::object(Arc::new(m))));
    ctx.add_function("id", |v: Value| -> Result<Value, ExecutionError> { Ok(v) });
    ctx
}

fn run(src: &str, ctx: &Context) -> (Value, Value) {
    let expr = Parser::default()
        .enable_optional_syntax(true)
        .parse(src)
        .unwrap_or_else(|e| panic!("{src}: parse {e}"));
    let walker =
        Value::resolve(&expr, ctx).unwrap_or_else(|e| panic!("{src}: walker {e:?}"));
    let code = cel::vm::compile(&expr).unwrap_or_else(|e| panic!("{src}: compile {e}"));
    let vm = cel::vm::cel_eval_loop(&code, ctx).unwrap_or_else(|e| panic!("{src}: vm {e:?}"));
    (walker, vm)
}

#[test]
fn no_interned_in_walker_or_vm_results() {
    const EXPRS: &[&str] = &[
        "xs.map(e, e)",
        "xs.filter(e, true)",
        r#"xs.exists_one(e, e == "a")"#,
        "[xs]",
        r#"{"k": xs}"#,
        "optional.of(xs)",
        "xs.map(e, [e])",
        "m.map(k, m[k])",
        "true ? xs : empty",
        "false ? empty : xs",
        "id(xs)",
        "xs.map(e, id(e))",
        "optional.of(s)",
        r#"{"k": s}"#,
        "[s]",
        "xs.map(e, s)",
        "m.map(k, k)",
        r#"xs.map(e, {"k": e})"#,
        "optional.of([xs])",
        r#"xs.filter(e, e != "z")"#,
        "[xs, empty]",
        "id(s)",
        "xs.map(e, optional.of(e))",
        r#"{"a": xs, "b": s}"#,
        "xs.map(e, [e, e])",
    ];
    assert!(EXPRS.len() >= 25, "need ~25 expressions, got {}", EXPRS.len());

    let ctx = make_ctx();
    let mut pairs = Vec::with_capacity(EXPRS.len());
    for src in EXPRS {
        let (walker, vm) = run(src, &ctx);
        assert!(
            !has_interned(&walker),
            "{src}: walker result contains Interned"
        );
        assert!(!has_interned(&vm), "{src}: vm result contains Interned");
        pairs.push((*src, walker, vm));
    }
    drop(ctx);
    for (src, walker, vm) in pairs {
        assert_eq!(walker, vm, "{src}: walker and vm differ after Context drop");
    }
}
