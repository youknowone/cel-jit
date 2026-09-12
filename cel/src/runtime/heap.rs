//! The heap the value family is allocated from.
//!
//! [`super::lltype::malloc_typed`] used to be `Box::into_raw(Box::new(v))` and
//! nothing freed it. This module gives those allocations an owner: one
//! `CelHeap` per thread, holding non-moving segments that are released when the
//! heap is torn down.
//!
//! # Non-moving, and not by preference
//!
//! A `CelRef` is a raw pointer held by whatever built the value — the future
//! `Value`, a `Context` variable, a `CelCode` constant, an in-flight result.
//! Nothing walks those places, so nothing could rewrite a pointer that moved.
//! Compaction is available only once a root walker exists; until then the
//! design's stepping stone is pyre's, stated in `gc_interp`: keep values on the
//! non-moving old-gen and put the safepoint where the live references are all
//! reachable through a registered walker.
//!
//! # What this does NOT do
//!
//! **It does not collect.** The root set §7 of the design enumerates — the live
//! frame and its array, `Context` variables, `CelCode.co_consts`, the in-flight
//! result and the eleven `ExecutionError` payloads, and an embedder handle
//! registry — has no walker, and a collector without a root set frees live
//! objects. So an allocation is reclaimed when its heap is dropped and not
//! before.
//!
//! That is a bound rather than a fix, and it is worth being exact about the
//! direction: against today's `Arc`-based `Value` it is a regression in
//! promptness, and against the class family's `Box::into_raw` it is the
//! difference between bounded and unbounded. The family is not reachable from
//! `crate::Value` yet, so only the second comparison describes anything that
//! runs.
//!
//! ⛔ **Two heaps would be two universes.** [`super::registration`]'s
//! `install_cel_gc` turns on `set_new_via_gc`, after which a compiled
//! `NewWithVtable` allocates from `MiniMarkGC`'s nursery while everything the
//! interpreter builds comes from here. A collector walking either one cannot
//! see the objects in the other, and neither side would report the split — it
//! would show up as a freed live value. `install_cel_gc` has no caller today,
//! so the split is not live; it has to be closed *before* it gets one, by
//! making this heap the collector's rather than by adding a second walker.
//!
//! # Values must not need dropping
//!
//! A segment is freed as raw memory and nothing runs a destructor over what was
//! handed out of it, so a value owning anything a `Drop` impl would release
//! would leak it. [`CelHeap::alloc`] rejects such a type at compile time. The
//! check is an associated const on a helper rather than an inline `const`
//! block, because an inline one cannot mention the function's own type
//! parameter.

use core::alloc::Layout;
use core::cell::{Cell, RefCell};
use core::marker::PhantomData;

/// Bytes per segment.
///
/// Segments are the unit of `alloc`/`dealloc`, not of collection, so the size
/// trades header overhead against how much a mostly-empty last segment wastes.
/// One 64 KiB segment holds ~4000 two-word leaves.
const SEGMENT_BYTES: usize = 64 * 1024;

/// Carrier for the compile-time refusal of a type this heap cannot own.
///
/// Reading `OK` is what forces the constant to be evaluated; a `const fn`
/// taking `T` would not be, and the check would pass silently by never running.
struct AssertNoDrop<T>(PhantomData<T>);

impl<T> AssertNoDrop<T> {
    const OK: () = assert!(
        !core::mem::needs_drop::<T>(),
        "CelHeap frees segments as raw memory and runs no destructor, so a \
         value that needs dropping would leak whatever it owns"
    );
}

/// A single non-moving run of memory, handed out by bumping.
struct Segment {
    base: *mut u8,
    /// Bytes handed out so far. Never decreases: a segment is only reclaimed
    /// whole.
    used: usize,
    capacity: usize,
}

impl Segment {
    fn with_capacity(capacity: usize) -> Segment {
        let layout = Layout::from_size_align(capacity, align_of::<u64>())
            .expect("a segment layout is valid by construction");
        // SAFETY: `capacity` is non-zero for every caller below, which is what
        // `alloc` requires of the layout.
        let base = unsafe { std::alloc::alloc(layout) };
        if base.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Segment {
            base,
            used: 0,
            capacity,
        }
    }

    /// The next address at `align`, or `None` if this segment cannot serve it.
    fn bump(&mut self, size: usize, align: usize) -> Option<*mut u8> {
        let start = (self.used + align - 1) & !(align - 1);
        let end = start.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        self.used = end;
        // SAFETY: `end <= capacity`, so `start` is inside the allocation.
        Some(unsafe { self.base.add(start) })
    }
}

impl Drop for Segment {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, align_of::<u64>())
            .expect("the layout that allocated this segment is still valid");
        // SAFETY: `base` came from `alloc` with this exact layout and is freed
        // once, here.
        unsafe { std::alloc::dealloc(self.base, layout) }
    }
}

/// One thread's value heap.
///
/// Not `Send` and not `Sync`, and deliberately without the `unsafe impl` that
/// would say otherwise: pyre's collector claims both, justified by holding the
/// GIL. CEL has no GIL, so the claim does not transfer.
pub struct CelHeap {
    segments: RefCell<Vec<Segment>>,
    /// Objects handed out. The design's P5 verification reads allocations per
    /// eval from this counter and from the JIT's own, so it counts objects
    /// rather than bytes — a byte total cannot be compared against a `New`
    /// count.
    objects: Cell<u64>,
    bytes: Cell<u64>,
}

impl CelHeap {
    pub fn new() -> CelHeap {
        CelHeap {
            segments: RefCell::new(Vec::new()),
            objects: Cell::new(0),
            bytes: Cell::new(0),
        }
    }

    /// Allocate `value` in this heap and return a pointer to it.
    ///
    /// The pointer is valid until the heap is dropped. Nothing shorter is
    /// available: see the module documentation on why there is no collection
    /// yet.
    pub fn alloc<T>(&self, value: T) -> *mut T {
        let () = AssertNoDrop::<T>::OK;
        let ptr = self.alloc_raw(size_of::<T>(), align_of::<T>()) as *mut T;
        // SAFETY: `alloc_raw` returns an address with `T`'s size and alignment
        // that nothing else has been handed.
        unsafe { ptr.write(value) };
        ptr
    }

    /// Reserve `size` bytes at `align`, growing the heap if the open segment
    /// cannot serve it.
    ///
    /// Public because the payload blocks in [`super::object_array`] are sized
    /// at run time and so cannot go through the generic [`alloc`].
    pub fn alloc_raw(&self, size: usize, align: usize) -> *mut u8 {
        let mut segments = self.segments.borrow_mut();
        // Nested rather than a `let` chain: this crate is edition 2021, where
        // chained `let` in an `if` is not available.
        if let Some(open) = segments.last_mut() {
            if let Some(ptr) = open.bump(size, align) {
                self.count(size);
                return ptr;
            }
        }
        // A request larger than the segment size gets a segment of its own
        // rather than growing every future segment to fit it.
        let capacity = SEGMENT_BYTES.max(size + align);
        segments.push(Segment::with_capacity(capacity));
        let ptr = segments
            .last_mut()
            .expect("the segment just pushed is present")
            .bump(size, align)
            .expect("a segment sized for this request serves it");
        self.count(size);
        ptr
    }

    fn count(&self, size: usize) {
        self.objects.set(self.objects.get() + 1);
        self.bytes.set(self.bytes.get() + size as u64);
    }

    /// Objects handed out since this heap was created.
    pub fn allocated_objects(&self) -> u64 {
        self.objects.get()
    }

    /// Bytes handed out since this heap was created, excluding the padding a
    /// request's alignment skipped.
    pub fn allocated_bytes(&self) -> u64 {
        self.bytes.get()
    }

    /// Segments currently held.
    ///
    /// The one observable that distinguishes "reused the open segment" from
    /// "took a new one", which is what the growth tests need.
    pub fn segments(&self) -> usize {
        self.segments.borrow().len()
    }
}

impl Default for CelHeap {
    fn default() -> CelHeap {
        CelHeap::new()
    }
}

thread_local! {
    /// The heap this thread allocates values from.
    ///
    /// A thread local rather than a parameter because the constructors it
    /// serves are the ones the boxing fuse matches, and the matcher requires
    /// them to take exactly one argument — the finished value. A heap
    /// parameter would be a second one.
    static HEAP: CelHeap = CelHeap::new();
}

/// Run `f` against this thread's heap.
pub fn with_heap<R>(f: impl FnOnce(&CelHeap) -> R) -> R {
    HEAP.with(f)
}

/// Bytes preceding an immortal payload. The backend reads
/// `[obj - HEADER_SIZE]` for `guard_is_object`; a plain Rust `static`
/// would put that load in rodata or unmapped memory.
pub const IMMORTAL_HEADER_SIZE: usize = core::mem::size_of::<usize>();

/// Non-zero so a header-relative load is not a null-page read.
const IMMORTAL_MARK: usize = 0xC3_11_07_7A;

/// Payloads handed out by [`alloc_immortal`]. The tripwire that a
/// prebuilt is not a Rust `static` asserts against this list.
static IMMORTAL_PAYLOADS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Allocate `value` for process lifetime, with a header word in front
/// of the payload.
///
/// Pointer-free leaves only: a reference field written at construction
/// has no write barrier, and the immortal flag would then let a major
/// walk it into freed memory. `W_OptionalObject` and anything holding a
/// [`super::object::CelRef`] stay on [`CelHeap::alloc`].
pub fn alloc_immortal<T>(value: T) -> *mut T {
    let () = AssertNoDrop::<T>::OK;
    let align = align_of::<T>().max(align_of::<usize>());
    let header = IMMORTAL_HEADER_SIZE;
    let size = header
        .checked_add(size_of::<T>())
        .expect("immortal layout fits usize");
    let layout = Layout::from_size_align(size, align).expect("immortal layout is valid");
    // SAFETY: `size` is at least the header word.
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: `base` is aligned for `usize` and owned uniquely here.
    unsafe {
        (base as *mut usize).write(IMMORTAL_MARK);
    }
    let payload = unsafe { base.add(header) as *mut T };
    unsafe {
        payload.write(value);
    }
    IMMORTAL_PAYLOADS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(payload as usize);
    payload
}

/// Whether `ptr` is a payload [`alloc_immortal`] handed out.
pub fn is_immortal(ptr: *const u8) -> bool {
    IMMORTAL_PAYLOADS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .any(|&p| p == ptr as usize)
}

/// The header word immediately before an immortal payload.
///
/// # Safety
///
/// `ptr` must have come from [`alloc_immortal`].
pub unsafe fn immortal_header(ptr: *const u8) -> usize {
    unsafe { *(ptr.sub(IMMORTAL_HEADER_SIZE) as *const usize) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two allocations from one heap are distinct addresses, and the second
    /// does not overlap the first.
    #[test]
    fn allocations_do_not_overlap() {
        let heap = CelHeap::new();
        let a = heap.alloc(1u64);
        let b = heap.alloc(2u64);
        assert_ne!(a, b);
        unsafe {
            assert_eq!(*a, 1);
            assert_eq!(*b, 2);
        }
        assert!((a as usize).abs_diff(b as usize) >= size_of::<u64>());
    }

    /// The counter counts objects, not segments — a heap that served a
    /// thousand values out of one segment still reports a thousand.
    #[test]
    fn the_counter_counts_objects_not_segments() {
        let heap = CelHeap::new();
        for i in 0..1000u64 {
            heap.alloc(i);
        }
        assert_eq!(heap.allocated_objects(), 1000);
        assert_eq!(heap.allocated_bytes(), 1000 * size_of::<u64>() as u64);
        assert_eq!(heap.segments(), 1, "1000 words fit in one segment");
    }

    /// A request larger than a whole segment is served rather than refused,
    /// and it does not enlarge the segments that follow it.
    #[test]
    fn an_oversized_request_gets_its_own_segment() {
        let heap = CelHeap::new();
        heap.alloc_raw(8, 8);
        assert_eq!(heap.segments(), 1);
        heap.alloc_raw(SEGMENT_BYTES * 3, 8);
        assert_eq!(heap.segments(), 2, "the oversized request took a segment");
        // The open segment is now the oversized one. It was sized `size +
        // align` so that alignment padding can never push the request past the
        // end, and on a fresh segment no padding is needed — so `align` bytes
        // of it are slack, and a request larger than that is what forces the
        // next segment.
        heap.alloc_raw(8 + align_of::<u64>(), 8);
        assert_eq!(heap.segments(), 3);
    }

    /// Alignment is honoured across the bump, which is the property the leaves
    /// depend on: `align_of` is 8 for every one of them and a misaligned
    /// header would fault on the class-word read.
    #[test]
    fn every_allocation_is_aligned() {
        let heap = CelHeap::new();
        heap.alloc_raw(1, 1);
        for _ in 0..64 {
            let p = heap.alloc_raw(8, 8);
            assert_eq!(p as usize % 8, 0);
        }
    }

    /// The thread-local heap is the same heap across calls, which is what
    /// makes a pointer from one constructor still valid inside the next.
    #[test]
    fn the_thread_local_heap_persists_across_calls() {
        let before = with_heap(|h| h.allocated_objects());
        let p = with_heap(|h| h.alloc(7u64));
        let after = with_heap(|h| h.allocated_objects());
        assert_eq!(after, before + 1);
        unsafe { assert_eq!(*p, 7) };
    }
}
