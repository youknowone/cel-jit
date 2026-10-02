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
    agree_compiled_in(src, &ctx(), times);
}

fn agree_compiled_in(src: &str, ctx: &Context, times: usize) {
    let expr = Parser::default()
        .parse(src)
        .unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let walker = Value::resolve_value(&expr, ctx);
    let program = Program::compile(src).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    for i in 0..times {
        let vm = program.execute(ctx);
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
    agree_compiled("list.map(e, e - 1)", 150);
    agree_compiled("list.filter(e, e in [1, 2, 3])", 150);
    agree_compiled("list.map(e, e in [1, 2, 3])", 150);
    agree_compiled("list.map(e, !(e in [1, 2, 3]))", 150);
    agree_compiled("list.map(e, e in [1, 2.0, \"a\"])", 150);
    agree_compiled("list.map(e, e in [])", 150);
    agree_compiled("list.map(e, e in [x])", 150);
    agree_compiled(
        "list.map(e, e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])",
        150,
    );
    agree_compiled("list.map(e, \"a\" in [\"a\", \"b\"])", 150);
    agree_compiled("list.map(e, 2u in [1, 2])", 150);
    agree_compiled("list.map(e, e in [1, 2, 3])", 150);
    agree_compiled("list.map(e, {\"a\": e}.a)", 150);
    agree_compiled("list.map(e, has({\"a\": e}.a))", 150);
    agree_compiled("list.map(e, has({\"a\": e}.b))", 150);
    agree_compiled("list.map(e, list[e % 10])", 150);
    agree_compiled("list.map(e, list[size(list) - 1 - e])", 150);
    agree_compiled("list.map(e, list[e + 1000])", 150);
    agree_compiled("list.map(e, list[e - 1])", 150);
    agree_compiled("list.filter(e, e % 2 == 0).map(e, e * 2)", 150);
    agree_compiled("{\"a\": x, \"b\": x}.map(k, k)", 150);
    agree_compiled("list.all(e, e < x + 1000)", 150);
    agree_compiled("list.map(e, int(e))", 150);
    agree_compiled("list.map(e, double(e))", 150);
    agree_compiled("list.map(e, double(e))[3]", 150);
    agree_compiled("list.map(e, e > 2 ? double(e) : 1)", 150);
    agree_compiled("size(list.map(e, double(e - 500)))", 150);
    agree_compiled("list.map(e, e > 2 ? \"x\" : 1)", 150);
    agree_compiled("list.map(e, string(e))[7]", 150);
    agree_compiled("size(list.map(e, e % 3 == 0 ? \"p\" : \"q\"))", 150);
    agree_compiled("list.map(e, string(e))", 150);
    agree_compiled("list.map(e, string(e - 500))", 150);
    agree_compiled("list.map(e, double(e) / 2.0)", 150);
    agree_compiled("list.map(e, string(e) + \"x\")", 150);
    agree_compiled("list.map(e, double(e)).filter(f, f > 3.5)", 150);
    agree_compiled("list.map(e, e > 3)", 150);
    agree_compiled("list.map(e, e == 3)", 150);
    agree_compiled("list.filter(e, e % 2 == 0)", 150);
    agree_compiled("list.map(e, (e - 500) % 7)", 150);
    agree_compiled("list.map(e, (e - 500) / 7)", 150);
    agree_compiled("list.map(e, e % 7)", 150);
    agree_compiled("list.map(e, -e % 3)", 150);
    agree_compiled("list.map(e, list.size())", 150);
    agree_compiled("list.map(e, {\"k\": e})", 150);
    agree_compiled("list.map(e, {\"k\": e}.k)", 150);
    agree_compiled("list.map(e, [e, e])", 150);
    agree_compiled("list.map(e, [e][0])", 150);
    // A two-word nursery bump would publish `(vm, cap)` as the list and
    // the indexed reads would not be `e` / `e + 1`.
    agree_compiled("list.map(e, [e, e + 1])", 150);
    agree_compiled("list.map(e, [e, e + 1][0])", 150);
    agree_compiled("list.map(e, [e, e + 1][1])", 150);
    agree_compiled("list.map(e, size([e, 1]))", 150);
    agree_compiled("list.map(e, e + 1 == 2 ? \"a\" : \"b\")", 150);
    agree_compiled("list.map(e, e + 1 == 2 ? string(e) : \"b\")", 150);
    agree_compiled("list.map(e, e % 2 == 0 ? double(e) : e)", 150);
    agree_compiled("list.map(e, [double(e), e])", 150);
    agree_compiled("list.map(e, [e, \"s\"])", 150);
    agree_compiled("list.map(e, [e, 1.5])", 150);
    agree_compiled("list.map(e, e * 3 + 1)", 150);
    agree_compiled("list.map(e, string(e)).map(s, s * 2)", 150);
    agree_compiled("list.map(e, {\"a\": e}).filter(m, m.a > 0)", 150);
    agree_compiled("list.map(e, {\"a\": e}).map(m, m.missing)", 150);
    agree_compiled("list.map(e, {\"a\": e}).filter(m, has(m.a))", 150);
    agree_compiled("list.map(e, {\"a\": e}).filter(m, has(m.z))", 150);
    agree_compiled("list.map(e, has(e.a))", 150);
    agree_compiled("list.map(e, {\"a\": e}).map(m, m.a)", 150);
    agree_compiled("list.map(e, {\"a\": e}).map(m, has(m.a))", 150);
    agree_compiled("list.map(e, {\"a\": e}).map(m, has(m.z))", 150);
    agree_compiled("list.map(e, [10, 20].exists(i, v, i == e))", 150);
    agree_compiled("list.map(e, {\"a\": e}.exists(k, v, k == \"a\"))", 150);
}

/// `string(e)` over negatives and zero, string concat, and a list whose
/// elements are not all ints.
#[test]
fn compiled_string_map_covers_sign_zero_and_mixed_kinds() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let mut signed = Context::default();
    signed.add_variable_from_value("x", 15i64);
    signed.add_variable_from_value("list", vec![-2i64, -1, 0, 1, 7]);
    agree_compiled_in("list.map(e, string(e))", &signed, 150);
    agree_compiled_in("list.map(e, string(e) + \"x\")", &signed, 150);

    let mut mixed = Context::default();
    mixed.add_variable_from_value("x", 15i64);
    mixed.add_variable_from_value(
        "list",
        cel::objects::ListRef::from(vec![
            Value::Float(1.5),
            Value::String(std::sync::Arc::from("ab")),
            Value::Int(-3),
        ]),
    );
    agree_compiled_in("list.map(e, string(e))", &mixed, 150);
}

fn agree_optional(src: &str, times: usize) {
    let parser = Parser::default().enable_optional_syntax(true);
    let expr = parser
        .parse(src)
        .unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let ctx = ctx();
    let walker = Value::resolve_value(&expr, &ctx);
    let code = cel::vm::compile(&expr).unwrap_or_else(|e| panic!("compile {src}: {e}"));
    for i in 0..times {
        let vm = cel::vm::cel_eval_loop(&code, &ctx);
        assert_eq!(show(&walker), show(&vm), "`{src}` execute {i}");
        if let (Ok(a), Ok(b)) = (&walker, &vm) {
            assert_eq!(a, b, "`{src}` execute {i} value");
        }
    }
}

#[test]
fn compiled_optional_opcodes_match_the_walker() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_optional("list.map(e, [e, 1][?0])", 150);
    agree_optional("list.map(e, [1][?5])", 150);
    agree_optional("list.map(e, {\"a\": e}).map(m, m.?a)", 150);
    agree_optional("list.map(e, {\"a\": e}).map(m, m.?missing)", 150);
    agree_optional("list.map(e, e.?a)", 150);
    agree_optional("list.map(e, [?optional.of(e)])", 150);
    agree_optional("list.map(e, [?optional.none()])", 150);
    agree_optional("list.map(e, [?e])", 150);
    agree_optional("list.map(e, {?\"k\": optional.of(e)})", 150);
    agree_optional("list.map(e, {?\"k\": optional.none()})", 150);
    agree_optional("list.map(e, {?\"k\": e})", 150);
}

fn agree_compiled_list(src: &str, list: Vec<i64>, times: usize) {
    let expr = Parser::default()
        .parse(src)
        .unwrap_or_else(|e| panic!("parse {src}: {e}"));
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    ctx.add_variable_from_value("list", list);
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
fn compiled_index_miss_matches_the_walker() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list[100]", 150);
    // `1..=10` makes `e - 1` the indexes `0..=9`. A list that contains `0`
    // makes the first index `-1`, and the compiled tier must raise the
    // same error as the walker.
    agree_compiled_list("list.map(e, list[e - 1])", (0..10).collect(), 40);
}

/// A resolver that returns a new int on every `x` lookup.
struct CountingResolver {
    n: std::cell::Cell<i64>,
}

impl cel::context::VariableResolver for CountingResolver {
    fn resolve(&self, name: &str) -> Option<Value> {
        if name != "x" {
            return None;
        }
        let n = self.n.get();
        self.n.set(n + 1);
        Some(Value::Int(n))
    }
}

fn list_ints(v: &Value) -> Vec<i64> {
    let Value::List(list) = v else {
        panic!("not a list: {v:?}");
    };
    list.iter()
        .map(|e| match e {
            Value::Int(n) => n,
            other => panic!("not an int: {other:?}"),
        })
        .collect()
}

/// The compiled loop must keep the residual lookup: each element sees the
/// next resolver value, including after the portal has compiled.
#[test]
fn compiled_loop_sees_each_resolver_value() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let resolver = CountingResolver {
        n: std::cell::Cell::new(0),
    };
    let mut ctx = Context::default();
    ctx.add_variable_from_value("list", vec![1i64, 2, 3]);
    ctx.set_variable_resolver(&resolver);
    let program = Program::compile("list.map(e, x)").unwrap();
    for i in 0..150 {
        let start = resolver.n.get();
        let got = program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("execute {i}: {e:?}"));
        let nums = list_ints(&got);
        assert_eq!(
            nums,
            vec![start, start + 1, start + 2],
            "execute {i} cached a resolver value"
        );
    }
}

/// A compiled loop must take a later decline with the frame the trace
/// still holds in registers.
///
/// `100 / (e - 5)` stays on the traced int arm for `e` in `6..=10` and
/// declines through `slow_pc` at `e == 5` (divisor 0) and for a negative
/// divisor. Before the residual was `may_force`, execute 25 came back
/// `InternalError` instead of the walker's `DivisionByZero`.
#[test]
fn compiled_decline_after_warmup_matches_the_walker() {
    // SAFETY: stored before this test builds a driver. The knob is read
    // once, when that driver is created.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list.map(e, list[e])", 40);
    agree_compiled("list.map(e, 100 / (e - 5))", 40);
}

/// Object-strategy map literal, then a field select, inside `list.map`.
///
/// Threshold 100. The first executions match the walker. Once the loop
/// is compiled, execute 32 used to come back `InternalError`.
#[test]
fn compiled_map_field_select_matches_the_walker() {
    // SAFETY: stored before this test builds a driver. The knob is read
    // once, when that driver is created.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list.map(e, {\"a\": e}.a)", 40);
    agree_compiled("list.map(e, {\"a\": e, \"b\": e}.b)", 40);
    agree_compiled("list.map(e, {\"b\": 1, \"a\": e}.a)", 40);
    agree_compiled("list.map(e, has({\"a\": e}.a))", 40);
    agree_compiled("list.map(e, has({\"a\": e}.missing))", 40);
    agree_compiled("list.map(e, {\"a\": e}.missing)", 40);
}

/// The folded field name is one immortal cell on the code object.
///
/// `field_name_cell` used to call `new_string`, so each call was a fresh
/// nursery pointer and `is_immortal` was false. A trace that folded that
/// pointer kept it after `rewind_nursery`.
#[test]
fn compiled_field_name_is_one_immortal_cell() {
    let expr = Parser::default()
        .parse("{\"a\": x}.a")
        .unwrap_or_else(|e| panic!("parse: {e}"));
    let code = cel::vm::compile(&expr).expect("compile");
    let mut idx = None;
    let mut i = 0u32;
    loop {
        match code.name(cel::vm::NameId(i)) {
            Some("a") => {
                idx = Some(i);
                break;
            }
            None => break,
            _ => i += 1,
        }
    }
    let id = cel::vm::NameId(idx.expect("name a"));
    let first = code.name_cell(id);
    let second = code.name_cell(id);
    assert!(!first.is_null(), "name a has a cell");
    assert_eq!(first, second, "co_names_w is one cell");
    assert!(
        cel::runtime::lltype::is_immortal(first as *const u8),
        "name cell {first:p} is not immortal"
    );
    let chars = unsafe { (*first.cast::<cel::runtime::object::W_StringObject>()).chars };
    assert!(
        cel::runtime::lltype::is_immortal(chars as *const u8),
        "character block {chars:p} is not immortal"
    );
    assert_eq!(
        unsafe { cel::runtime::object::string_as_str(first) },
        Some("a")
    );
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list.map(e, {\"a\": e}.a)", 40);
    assert_eq!(code.name_cell(id), first);
    assert_eq!(
        unsafe { cel::runtime::object::string_as_str(first) },
        Some("a")
    );
}

fn as_int(v: &Value) -> i64 {
    match v {
        Value::Int(n) => *n,
        other => panic!("not an int: {other:?}"),
    }
}

/// Same `Context`, rebind `x` after the loop compiled. The leaf is keyed
/// on the version, so the next evaluation must return the new value.
#[test]
fn compiled_rebind_changes_the_leaf() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let mut ctx = Context::default();
    ctx.add_variable_from_value("x", 15i64);
    let program = Program::compile("x").unwrap();
    for i in 0..150 {
        let got = program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("{i}: {e:?}"));
        assert_eq!(as_int(&got), 15, "warmup {i}");
    }
    ctx.add_variable_from_value("x", 99i64);
    let got = program.execute(&ctx).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(as_int(&got), 99, "rebind kept the traced leaf");
}

/// Drop the context and allocate another. The address may be reused; the
/// result must follow the new binding.
#[test]
fn compiled_fresh_context_is_not_the_old_leaf() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let program = Program::compile("x").unwrap();
    {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("x", 15i64);
        for i in 0..150 {
            let got = program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("{i}: {e:?}"));
            assert_eq!(as_int(&got), 15, "warmup {i}");
        }
    }
    for n in 0..64 {
        let mut ctx = Context::default();
        let want = 1000 + n;
        ctx.add_variable_from_value("x", want);
        let got = program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("{n}: {e:?}"));
        assert_eq!(as_int(&got), want, "fresh context {n}");
    }
}

/// A compiled function-entry run that fails a guard resumes in the bridge
/// walk. When that walk reaches the portal return it publishes the result
/// on the single-pass finish latch and the driver reports `usize::MAX`.
/// The function-entry door drained only the back-edge latch, then skipped
/// the loop and ran the epilogue, which returned null
/// (`InternalError("portal done without a result")`).
#[test]
fn function_entry_guard_resume_returns_the_bridge_finish() {
    // SAFETY: stored before this test builds a driver. The knob is read
    // once, when that driver is created.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list.all(e, e > 0)", 1000);
    agree_compiled("list.exists(e, e == x)", 1000);
    agree_compiled("list.exists(e, e == 5)", 1000);
    agree_compiled("list.all(e, e < x + 1000)", 1000);
}

/// A resolver context must not bake the leaf observed on the first call.
#[test]
fn compiled_resolver_context_stays_live() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let resolver = CountingResolver {
        n: std::cell::Cell::new(0),
    };
    let mut ctx = Context::default();
    ctx.set_variable_resolver(&resolver);
    let program = Program::compile("x").unwrap();
    for i in 0..150 {
        let got = program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("{i}: {e:?}"));
        assert_eq!(as_int(&got), i, "resolver execute {i}");
    }
}

fn record_items() -> Value {
    let items: Vec<Value> = (0..8i64)
        .map(|i| {
            let mut map = std::collections::HashMap::new();
            map.insert("price", Value::Int(i));
            map.insert("name", Value::from(format!("n{i}")));
            Value::from(map)
        })
        .collect();
    Value::list(items)
}

fn filtered_without_n1() -> Value {
    let items: Vec<Value> = (0..8i64)
        .filter(|i| *i != 1)
        .map(|i| {
            let mut map = std::collections::HashMap::new();
            map.insert("price", Value::Int(i));
            map.insert("name", Value::from(format!("n{i}")));
            Value::from(map)
        })
        .collect();
    Value::list(items)
}

/// `i.name == "zz"` / `== "n3"` / `!= "n1"` stay correct after the portal loop
/// compiles. The constant used to be re-interned on every element, and the
/// string compare used to leave the traced loop.
#[test]
fn record_name_equality_answers_after_the_portal_compiles() {
    // SAFETY: stored before this test builds a driver. The knob is read
    // once, when that driver is created.
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    let mut ctx = Context::default();
    ctx.add_variable_from_value("items", record_items());
    let absent = Program::compile(r#"items.exists(i, i.name == "zz")"#).unwrap();
    let present = Program::compile(r#"items.exists(i, i.name == "n3")"#).unwrap();
    let kept = Program::compile(r#"items.filter(i, i.name != "n1")"#).unwrap();
    let want_kept = filtered_without_n1();
    for i in 0..200 {
        assert_eq!(
            absent.execute(&ctx).expect("exists zz"),
            Value::Bool(false),
            "exists zz {i}"
        );
        assert_eq!(
            present.execute(&ctx).expect("exists n3"),
            Value::Bool(true),
            "exists n3 {i}"
        );
        assert_eq!(
            kept.execute(&ctx).expect("filter"),
            want_kept,
            "filter != n1 {i}"
        );
    }
}
