//! Allocation count at `Program::execute` for the finish-heavy rows.
//! Prints mallocs per evaluation so a finish change has a number, not a guess.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "vm")]
use cel::{Context, Program, Value};

std::thread_local! {
    static LOCAL: Cell<u64> = const { Cell::new(0) };
}
static GLOBAL: AtomicU64 = AtomicU64::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        GLOBAL.fetch_add(1, Ordering::Relaxed);
        let _ = LOCAL.try_with(|c| c.set(c.get() + 1));
        ptr
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
    // The first evaluations compile a 10-element loop (the back-edge door is
    // 101). Straight-line programs stay under the function door and never
    // compile. The window below is the steady cost after that.
    program.execute(&ctx).expect(src);
    for _ in 0..24 {
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
fn count_walker(src: &str, setup: impl FnOnce(&mut Context)) -> u64 {
    let program = Program::compile(src).expect(src);
    let mut ctx = Context::default();
    setup(&mut ctx);
    Value::resolve_value(program.expression(), &ctx).expect(src);
    for _ in 0..8 {
        let _ = Value::resolve_value(program.expression(), &ctx);
    }
    let t0 = LOCAL.with(Cell::get);
    let n = 32u64;
    for _ in 0..n {
        let _ = Value::resolve_value(program.expression(), &ctx);
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
    let maps3 = count(r#"list.map(e, {"k": e, "v": e, "w": e})"#, |ctx| {
        ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())
    });
    let inner_lists = count("list.map(e, [e, e])", |ctx| {
        ctx.add_variable_from_value("list", (1..=10i64).collect::<Vec<_>>())
    });
    let pair = count("[x, x]", |ctx| ctx.add_variable_from_value("x", 15i64));
    let ints = count("[1, 2, 3]", |_| {});
    let walker_list = count_walker("[x, x]", |ctx| ctx.add_variable_from_value("x", 15i64));
    println!("[[x]] mallocs/eval={nested}");
    println!("string-chain mallocs/eval={chain}");
    println!("list.map(e, {{k:e}}) mallocs/eval={maps}");
    println!("list.map(e, {{k,v,w}}) mallocs/eval={maps3}");
    let walker_one = count_walker(r#"{"a": x}"#, |ctx| ctx.add_variable_from_value("x", 15i64));
    println!("list.map(e, [e,e]) mallocs/eval={inner_lists}");
    println!("[x, x] mallocs/eval={pair}");
    println!("[1, 2, 3] mallocs/eval={ints}");
    println!("walker [x, x] mallocs/eval={walker_list}");
    println!("walker {{a: x}} mallocs/eval={walker_one}");
    assert_eq!(nested, 2, "[[x]] finish mallocs");
    assert_eq!(chain, 2, "string-chain finish mallocs");
    // A compiled eval also allocates resume buffers on top of the
    // interpreter finish cost below. On the pinned majit, cranelift
    // measures 7 and dynasm 8 for a packed record list, and 9 and 10
    // for a scalar-row list.
    #[cfg(not(feature = "jit"))]
    {
        assert_eq!(maps, 1, "1-entry map list finish mallocs");
        assert_eq!(maps3, 1, "3-entry map list finish mallocs");
        assert_eq!(inner_lists, 1, "list-of-lists finish mallocs");
    }
    #[cfg(feature = "jit")]
    {
        for (got, name) in [(maps, "1-entry map list"), (maps3, "3-entry map list")] {
            assert!(
                got == 7 || got == 8,
                "{name} finish mallocs: cranelift 7, dynasm 8, got {got}"
            );
        }
        assert!(
            inner_lists == 9 || inner_lists == 10,
            "list-of-lists finish mallocs: cranelift 9, dynasm 10, got {inner_lists}"
        );
    }
    assert_eq!(pair, 1, "[x, x] finish mallocs");
    assert_eq!(ints, 0, "int-list finish mallocs");
    assert_eq!(walker_list, 1, "walker list literal mallocs");
    assert_eq!(walker_one, 1, "walker {{a: x}} mallocs");
}
