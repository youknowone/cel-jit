//! Counting allocator: 1000 windows over one column Arc allocate 0 times.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use cel::objects::{ListRef, ListStorage, ScalarBank, ValueColumn};

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

#[test]
fn one_thousand_windows_over_one_column_allocate_zero_times() {
    let storage = Arc::new(ListStorage::Column(ValueColumn::Scalar {
        bank: ScalarBank::Int,
        words: Arc::from([1i64, 2, 3].as_slice()),
    }));
    let mut windows = Vec::with_capacity(1000);
    let t0 = LOCAL.with(Cell::get);
    for _ in 0..1000 {
        windows.push(ListRef::window(Arc::clone(&storage), 0, 3));
    }
    let n = LOCAL.with(Cell::get) - t0;
    assert_eq!(n, 0, "1000 windows allocated {n} times");
    assert!(windows.windows(2).all(|w| w[0].shares_storage_with(&w[1])));
}
