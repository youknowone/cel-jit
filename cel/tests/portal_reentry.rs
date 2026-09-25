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
    agree_compiled("list.map(e, e - 1)", 150);
    agree_compiled("list.filter(e, e in [1, 2, 3])", 150);
    agree_compiled("list.map(e, e in [1, 2, 3])", 150);
    agree_compiled("list.map(e, {\"a\": e}.a)", 150);
    agree_compiled("list.map(e, has({\"a\": e}.a))", 150);
    agree_compiled("list.map(e, has({\"a\": e}.b))", 150);
    agree_compiled("list.map(e, list[e % 10])", 150);
    agree_compiled("list.all(e, e < x + 1000)", 150);
    agree_compiled("list.map(e, int(e))", 150);
    agree_compiled("list.map(e, double(e))", 150);
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
    agree_compiled("list.map(e, list.size())", 150);
    agree_compiled("list.map(e, {\"k\": e})", 150);
    agree_compiled("list.map(e, {\"k\": e}.k)", 150);
    agree_compiled("list.map(e, [e, e])", 150);
    agree_compiled("list.map(e, [e][0])", 150);
    agree_compiled("list.map(e, size([e, 1]))", 150);
    agree_compiled("list.map(e, e + 1 == 2 ? \"a\" : \"b\")", 150);
    agree_compiled("list.map(e, [e, \"s\"])", 150);
    agree_compiled("list.map(e, [e, 1.5])", 150);
}

#[test]
fn compiled_index_miss_matches_the_walker() {
    unsafe { std::env::set_var("CEL_PORTAL_THRESHOLD", "100") };
    agree_compiled("list[100]", 150);
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
