//! Allocation count at `Program::execute` for the three finish-heavy rows.
//! Prints mallocs per evaluation so a finish change has a number, not a guess.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "vm")]
use cel::{Context, Program};

std::thread_local! {
    static LOCAL: Cell<u64> = const { Cell::new(0) };
}
static GLOBAL: AtomicU64 = AtomicU64::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        GLOBAL.fetch_add(1, Ordering::Relaxed);
        let _ = LOCAL.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        GLOBAL.fetch_add(1, Ordering::Relaxed);
        let _ = LOCAL.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[cfg(feature = "vm")]
fn count(src: &str, setup: impl FnOnce(&mut Context)) -> u64 {
    let program = Program::compile(src).expect(src);
    let mut ctx = Context::default();
    setup(&mut ctx);
    program.execute(&ctx).expect(src);
    for _ in 0..8 {
        let _ = program.execute(&ctx);
    }
    let t0 = LOCAL.with(Cell::get);
    let n = 32u64;
    for _ in 0..n {
        let _ = program.execute(&ctx);
    }
    (LOCAL.with(Cell::get) - t0) / n
}

#[cfg(feature = "vm")]
#[test]
fn mallocs_per_eval_on_finish_rows() {
    let nested = count("[[x]]", |ctx| ctx.add_variable_from_value("x", 15i64));
    let chain = count(r#""a" + "b" + "c" + "d""#, |_| {});
    let maps = count(r#"list.map(e, {"k": e})"#, |ctx| {
        ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())
    });
    let inner_lists = count("list.map(e, [e, e])", |ctx| {
        ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())
    });
    println!("[[x]] mallocs/eval={nested}");
    println!("string-chain mallocs/eval={chain}");
    println!("list.map(e, {{k:e}}) mallocs/eval={maps}");
    println!("list.map(e, [e,e]) mallocs/eval={inner_lists}");
    assert_eq!(nested, 4, "[[x]] finish mallocs");
    assert_eq!(chain, 2, "string-chain finish mallocs");
    assert_eq!(maps, 22, "map-in-map finish mallocs");
    assert_eq!(inner_lists, 22, "list-of-lists finish mallocs");
}
