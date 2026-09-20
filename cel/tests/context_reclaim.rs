//! A fresh Context per evaluation must not grow this thread's old space.
//!
//! Bind wraps into interned leaves. Those used to live in the heap's old
//! space, which is never reset. Dropping the Context must release them.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use cel::objects::{Key, Map};
use cel::runtime::heap::with_heap;
use cel::{Context, Program, Value};

fn sample_map() -> Value {
    let mut m = HashMap::new();
    m.insert(Key::String(Arc::new("k".to_string())), Value::Int(1));
    m.insert(Key::String(Arc::new("v".to_string())), Value::Int(2));
    Value::Map(Map::object(Arc::new(m)))
}

fn bind_four(ctx: &mut Context) {
    ctx.add_variable_from_value("n", 7i64);
    ctx.add_variable_from_value("list", (1..=100i64).collect::<Vec<i64>>());
    ctx.add_variable_from_value("m", sample_map());
    ctx.add_variable_from_value("s", "hello");
}

fn rss_bytes() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps");
    let text = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = text.trim().parse().expect("rss kb");
    kb.saturating_mul(1024)
}

fn old_bytes() -> u64 {
    with_heap(|h| h.old_allocated_bytes())
}

fn eval_vm(program: &Program, ctx: &Context) -> Value {
    program
        .execute(ctx)
        .unwrap_or_else(|e| panic!("vm: {e:?}"))
}

fn eval_walker(program: &Program, ctx: &Context) -> Value {
    Value::resolve_value(program.expression(), ctx)
        .unwrap_or_else(|e| panic!("walker: {e:?}"))
}

/// ns per {new Context, four binds, drop}. Printed for the bind-cost gate.
#[test]
fn bind_four_and_drop_cost_ns() {
    // Warm the allocator and intern tables.
    for _ in 0..1_000 {
        let mut ctx = Context::default();
        bind_four(&mut ctx);
    }
    let n = 50_000u32;
    let start = Instant::now();
    for _ in 0..n {
        let mut ctx = Context::default();
        bind_four(&mut ctx);
    }
    let ns = start.elapsed().as_nanos() / u128::from(n);
    println!("bind_ns_per_cycle: {ns}");
}

#[test]
fn two_hundred_thousand_fresh_contexts_leave_old_space_flat() {
    let program = Program::compile("list.map(i, i * 2)").expect("compiles");
    let rss_before = rss_bytes();
    let old_before = old_bytes();
    println!("rss_before_bytes: {rss_before}");
    println!("old_before_bytes: {old_before}");

    let mut old_at_1k = 0u64;
    const N: u32 = 200_000;
    for i in 1..=N {
        let mut ctx = Context::default();
        bind_four(&mut ctx);
        let vm = eval_vm(&program, &ctx);
        let walker = eval_walker(&program, &ctx);
        assert_eq!(vm, walker, "vm vs walker at {i}");
        drop(ctx);
        if i == 1_000 {
            old_at_1k = old_bytes();
            println!("old_bytes_at_1k: {old_at_1k}");
            println!("rss_at_1k_bytes: {}", rss_bytes());
        }
    }

    let old_at_200k = old_bytes();
    let rss_after = rss_bytes();
    println!("old_bytes_at_200k: {old_at_200k}");
    println!("rss_after_bytes: {rss_after}");
    println!(
        "old_growth_1k_to_200k: {}",
        old_at_200k.saturating_sub(old_at_1k)
    );
    println!(
        "rss_growth_bytes: {}",
        rss_after.saturating_sub(rss_before)
    );

    const SLACK: u64 = 256 * 1024;
    assert!(
        old_at_200k <= old_at_1k.saturating_add(SLACK),
        "old-space bytes grew from {old_at_1k} after 1k contexts to {old_at_200k} after 200k"
    );
}

#[test]
fn a_kept_result_survives_dropping_the_context() {
    let program = Program::compile("list.map(i, i * 2)").expect("compiles");
    let kept_vm;
    let kept_walker;
    {
        let mut ctx = Context::default();
        bind_four(&mut ctx);
        kept_vm = eval_vm(&program, &ctx);
        kept_walker = eval_walker(&program, &ctx);
    }
    let want: Value = (1..=100i64).map(|n| n * 2).collect::<Vec<i64>>().into();
    assert_eq!(kept_vm, want, "vm result after Context drop");
    assert_eq!(kept_walker, want, "walker result after Context drop");
}
