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
//! **It does not walk objects.** The nursery is reclaimed by resetting a bump
//! pointer when the outermost evaluation on this thread finishes; old space
//! is not reclaimed until the heap is dropped. There is still no root walker,
//! so nothing is copied except the evaluation result at the public door.
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
use core::any::Any;
use core::cell::{Cell, RefCell};
use core::marker::PhantomData;

/// Bytes per segment.
///
/// Segments are the unit of `alloc`/`dealloc`, not of collection, so the size
/// trades header overhead against how much a mostly-empty last segment wastes.
/// One 64 KiB segment holds ~4000 two-word leaves.
const SEGMENT_BYTES: usize = 64 * 1024;

/// Nursery segments kept after a reset. A huge evaluation may have opened
/// more; those extras are released so they do not pin memory for the rest
/// of the thread's life.
const NURSERY_KEEP_SEGMENTS: usize = 2;

/// Fill byte for reclaimed nursery memory in debug builds.
#[cfg(debug_assertions)]
const NURSERY_POISON: u8 = 0xDB;

/// High bit on a [`CelHeap::push_host`] index: the host sits in the
/// evaluation-scoped table and is dropped when the outermost scope resets.
const YOUNG_HOST_BIT: i64 = 1 << 62;

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

    fn contains(&self, p: usize) -> bool {
        let base = self.base as usize;
        p >= base && p < base.saturating_add(self.used)
    }

    #[cfg(debug_assertions)]
    fn poison_from(&mut self, from: usize) {
        if from < self.used {
            unsafe {
                core::ptr::write_bytes(self.base.add(from), NURSERY_POISON, self.used - from);
            }
        }
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

/// One bump space: a run of non-moving segments.
///
/// The open segment's bump lives in [`Cell`]s so a rewind does not borrow
/// the segment vector. Opening a new segment is the only `RefCell` path.
struct Space {
    segments: RefCell<Vec<Segment>>,
    objects: Cell<u64>,
    bytes: Cell<u64>,
    n_segs: Cell<usize>,
    open_used: Cell<usize>,
    open_base: Cell<*mut u8>,
    open_cap: Cell<usize>,
}

impl Space {
    fn new() -> Space {
        Space {
            segments: RefCell::new(Vec::new()),
            objects: Cell::new(0),
            bytes: Cell::new(0),
            n_segs: Cell::new(0),
            open_used: Cell::new(0),
            open_base: Cell::new(core::ptr::null_mut()),
            open_cap: Cell::new(0),
        }
    }

    fn bump(&self, size: usize, align: usize) -> *mut u8 {
        let used = self.open_used.get();
        let start = (used + align - 1) & !(align - 1);
        if let Some(end) = start.checked_add(size) {
            if end <= self.open_cap.get() {
                self.open_used.set(end);
                self.objects.set(self.objects.get() + 1);
                self.bytes.set(self.bytes.get() + size as u64);
                return unsafe { self.open_base.get().add(start) };
            }
        }
        self.bump_grow(size, align)
    }

    fn bump_grow(&self, size: usize, align: usize) -> *mut u8 {
        let mut segments = self.segments.borrow_mut();
        if let Some(open) = segments.last_mut() {
            open.used = self.open_used.get();
        }
        let capacity = SEGMENT_BYTES.max(size + align);
        segments.push(Segment::with_capacity(capacity));
        let open = segments
            .last_mut()
            .expect("the segment just pushed is present");
        let ptr = open
            .bump(size, align)
            .expect("a segment sized for this request serves it");
        self.open_base.set(open.base);
        self.open_cap.set(open.capacity);
        self.open_used.set(open.used);
        self.n_segs.set(segments.len());
        self.objects.set(self.objects.get() + 1);
        self.bytes.set(self.bytes.get() + size as u64);
        ptr
    }

    fn contains(&self, p: usize) -> bool {
        let segs = self.segments.borrow();
        let last = segs.len().saturating_sub(1);
        segs.iter().enumerate().any(|(i, seg)| {
            if i == last {
                let base = seg.base as usize;
                p >= base && p < base.saturating_add(self.open_used.get())
            } else {
                seg.contains(p)
            }
        })
    }

    fn sync_open(&self) {
        let segs = self.segments.borrow();
        match segs.last() {
            Some(open) => {
                self.open_base.set(open.base);
                self.open_cap.set(open.capacity);
                self.open_used.set(open.used);
                self.n_segs.set(segs.len());
            }
            None => {
                self.open_base.set(core::ptr::null_mut());
                self.open_cap.set(0);
                self.open_used.set(0);
                self.n_segs.set(0);
            }
        }
    }
}

/// One thread's value heap.
///
/// Not `Send` and not `Sync`. Two spaces: old lives until the heap is
/// dropped; the nursery is reset when the outermost evaluation on this
/// thread finishes.
pub struct CelHeap {
    old: Space,
    nursery: Space,
    /// Outermost-evaluation nesting. Zero means no evaluation is running.
    depth: Cell<u32>,
    /// Nested `with_old_space` calls force allocation into old.
    force_old: Cell<u32>,
    snap_len: Cell<usize>,
    snap_used: Cell<usize>,
    snap_bytes: Cell<u64>,
    snap_objects: Cell<u64>,
    snap_hosts: Cell<usize>,
    young_host_n: Cell<usize>,
    nursery_high_water: Cell<u64>,
    /// Host objects behind [`super::object::W_OpaqueObject::host_index`].
    ///
    /// A side table, not a `Drop` payload on the leaf — `alloc` refuses types
    /// that need dropping. Old-space hosts live as long as this heap; young
    /// hosts are dropped when the outermost evaluation resets.
    hosts: RefCell<Vec<Box<dyn Any>>>,
    young_hosts: RefCell<Vec<Box<dyn Any>>>,
}

impl CelHeap {
    pub fn new() -> CelHeap {
        CelHeap {
            old: Space::new(),
            nursery: Space::new(),
            depth: Cell::new(0),
            force_old: Cell::new(0),
            snap_len: Cell::new(0),
            snap_used: Cell::new(0),
            snap_bytes: Cell::new(0),
            snap_objects: Cell::new(0),
            snap_hosts: Cell::new(0),
            young_host_n: Cell::new(0),
            nursery_high_water: Cell::new(0),
            hosts: RefCell::new(Vec::new()),
            young_hosts: RefCell::new(Vec::new()),
        }
    }

    fn in_nursery(&self) -> bool {
        self.depth.get() > 0 && self.force_old.get() == 0
    }

    fn enter(&self) -> bool {
        let d = self.depth.get();
        if d == 0 {
            self.snap_len.set(self.nursery.n_segs.get());
            self.snap_used.set(self.nursery.open_used.get());
            self.snap_bytes.set(self.nursery.bytes.get());
            self.snap_objects.set(self.nursery.objects.get());
            self.snap_hosts.set(self.young_host_n.get());
        }
        self.depth.set(d + 1);
        d == 0
    }

    fn leave(&self, outermost: bool) {
        let d = self.depth.get();
        self.depth.set(d.saturating_sub(1));
        if !outermost {
            return;
        }
        if self.nursery.bytes.get() == self.snap_bytes.get()
            && self.young_host_n.get() == self.snap_hosts.get()
        {
            return;
        }
        if self.nursery.n_segs.get() == self.snap_len.get()
            && self.young_host_n.get() == self.snap_hosts.get()
        {
            self.rewind_nursery();
        } else {
            self.reset_nursery();
        }
    }

    /// Rewind the open bump. Same segment count, no young hosts.
    fn rewind_nursery(&self) {
        #[cfg(debug_assertions)]
        {
            let from = self.snap_used.get();
            let to = self.nursery.open_used.get();
            let base = self.nursery.open_base.get();
            if !base.is_null() && to > from {
                unsafe {
                    core::ptr::write_bytes(base.add(from), NURSERY_POISON, to - from);
                }
            }
        }
        self.nursery.open_used.set(self.snap_used.get());
        self.nursery.bytes.set(self.snap_bytes.get());
        self.nursery.objects.set(self.snap_objects.get());
    }

    fn reset_nursery(&self) {
        let snap_len = self.snap_len.get();
        let snap_used = self.snap_used.get();
        let mut segs = self.nursery.segments.borrow_mut();
        for (i, seg) in segs.iter_mut().enumerate() {
            if i < snap_len {
                if i + 1 == snap_len {
                    #[cfg(debug_assertions)]
                    seg.poison_from(snap_used);
                    seg.used = snap_used;
                }
            } else {
                #[cfg(debug_assertions)]
                seg.poison_from(0);
                seg.used = 0;
            }
        }
        let retain = if segs.is_empty() {
            0
        } else {
            snap_len.max(NURSERY_KEEP_SEGMENTS.min(segs.len()))
        };
        segs.truncate(retain);
        self.nursery.bytes.set(self.snap_bytes.get());
        self.nursery.objects.set(self.snap_objects.get());
        self.young_hosts
            .borrow_mut()
            .truncate(self.snap_hosts.get());
        self.young_host_n.set(self.snap_hosts.get());
        drop(segs);
        self.nursery.sync_open();
    }

    /// Allocate `value` in this heap and return a pointer to it.
    ///
    /// During an evaluation the pointer is nursery memory and is invalid
    /// after the outermost scope resets, unless it was allocated through
    /// [`alloc_old`].
    pub fn alloc<T>(&self, value: T) -> *mut T {
        let () = AssertNoDrop::<T>::OK;
        let ptr = self.alloc_raw(size_of::<T>(), align_of::<T>()) as *mut T;
        // SAFETY: `alloc_raw` returns an address with `T`'s size and alignment
        // that nothing else has been handed.
        unsafe { ptr.write(value) };
        ptr
    }

    /// Allocate `value` in old space even if an evaluation is running.
    pub fn alloc_old<T>(&self, value: T) -> *mut T {
        let () = AssertNoDrop::<T>::OK;
        let ptr = self.alloc_old_raw(size_of::<T>(), align_of::<T>()) as *mut T;
        unsafe { ptr.write(value) };
        ptr
    }

    /// Reserve `size` bytes at `align`, growing the heap if the open segment
    /// cannot serve it.
    ///
    /// Public because the payload blocks in [`super::object_array`] are sized
    /// at run time and so cannot go through the generic [`alloc`].
    pub fn alloc_raw(&self, size: usize, align: usize) -> *mut u8 {
        if self.in_nursery() {
            let ptr = self.nursery.bump(size, align);
            let live = self.nursery.bytes.get();
            if live > self.nursery_high_water.get() {
                self.nursery_high_water.set(live);
            }
            ptr
        } else {
            self.old.bump(size, align)
        }
    }

    /// Reserve `size` bytes in old space.
    pub fn alloc_old_raw(&self, size: usize, align: usize) -> *mut u8 {
        self.old.bump(size, align)
    }

    /// Objects handed out that are still live: old space plus the current
    /// nursery bump. A reset drops the nursery contribution.
    pub fn allocated_objects(&self) -> u64 {
        self.old.objects.get() + self.nursery.objects.get()
    }

    /// Bytes handed out that are still live, excluding alignment padding.
    pub fn allocated_bytes(&self) -> u64 {
        self.old.bytes.get() + self.nursery.bytes.get()
    }

    /// Old-space bytes. Unchanged by an evaluation that only uses the nursery.
    pub fn old_allocated_bytes(&self) -> u64 {
        self.old.bytes.get()
    }

    /// Segments currently held in both spaces.
    pub fn segments(&self) -> usize {
        self.old.segments.borrow().len() + self.nursery.segments.borrow().len()
    }

    /// Old-space segments.
    pub fn old_segments(&self) -> usize {
        self.old.segments.borrow().len()
    }

    /// Peak live nursery bytes since this heap was created.
    pub fn nursery_high_water(&self) -> u64 {
        self.nursery_high_water.get()
    }

    /// Park `host` and return its index. During an evaluation the host is
    /// young and is dropped on reset; otherwise it lives as long as the heap.
    pub fn push_host(&self, host: Box<dyn Any>) -> i64 {
        if self.in_nursery() {
            let mut hosts = self.young_hosts.borrow_mut();
            let idx = hosts.len() as i64;
            hosts.push(host);
            self.young_host_n.set(hosts.len());
            idx | YOUNG_HOST_BIT
        } else {
            let mut hosts = self.hosts.borrow_mut();
            let idx = hosts.len() as i64;
            hosts.push(host);
            idx
        }
    }

    /// Borrow host `idx` for the duration of `f`.
    pub fn with_host<R>(&self, idx: i64, f: impl FnOnce(&dyn Any) -> R) -> Option<R> {
        if idx & YOUNG_HOST_BIT != 0 {
            let hosts = self.young_hosts.borrow();
            let slot = hosts.get((idx & !YOUNG_HOST_BIT) as usize)?;
            Some(f(slot.as_ref()))
        } else {
            let hosts = self.hosts.borrow();
            let slot = hosts.get(idx as usize)?;
            Some(f(slot.as_ref()))
        }
    }

    /// How many host objects this heap is holding, both spaces.
    pub fn host_count(&self) -> usize {
        self.hosts.borrow().len() + self.young_hosts.borrow().len()
    }

    /// Whether `ptr` lies in live old or live nursery memory.
    pub fn contains(&self, ptr: *const u8) -> bool {
        let p = ptr as usize;
        self.old.contains(p) || self.nursery.contains(p)
    }

    /// Whether `ptr` is live nursery memory.
    pub fn is_young(&self, ptr: *const u8) -> bool {
        self.nursery.contains(ptr as usize)
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

/// Guard for one evaluation. Reset runs in [`Drop`], including on panic.
///
/// The heap pointer is resolved once at entry. Depth and the nursery bump
/// live in [`Cell`]s on that heap, so leaving does not touch thread-local
/// storage again.
pub struct EvalScope {
    heap: *const CelHeap,
    outermost: bool,
}

impl EvalScope {
    /// The heap this scope opened. Valid until [`Drop`].
    pub fn heap(&self) -> &CelHeap {
        unsafe { &*self.heap }
    }

    /// Whether this guard opened the outermost scope on this thread.
    pub fn is_outermost(&self) -> bool {
        self.outermost
    }

    /// Move a nursery result out if this is the outermost scope.
    pub fn finish(self, v: crate::Value) -> crate::Value {
        if self.outermost {
            promote_on(self.heap(), v)
        } else {
            v
        }
    }
}

impl Drop for EvalScope {
    fn drop(&mut self) {
        unsafe { (*self.heap).leave(self.outermost) };
    }
}

/// Open an evaluation scope on this thread. Nested calls increment depth;
/// only the outermost reset reclaims the nursery.
pub fn enter_eval() -> EvalScope {
    HEAP.with(|h| EvalScope {
        heap: h as *const CelHeap,
        outermost: h.enter(),
    })
}

fn promote_on(heap: &CelHeap, v: crate::Value) -> crate::Value {
    match v {
        crate::Value::Interned(w)
            if (heap.nursery.bytes.get() != heap.snap_bytes.get()
                || heap.young_host_n.get() != heap.snap_hosts.get())
                && heap.is_young(w as *const u8) =>
        {
            crate::Value::from_interned(w).unpack()
        }
        other => other,
    }
}

/// Force allocations (and host parks) into old space for the duration of `f`.
pub fn with_old_space<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = HEAP.try_with(|h| h.force_old.set(h.force_old.get().saturating_sub(1)));
        }
    }
    let _ = HEAP.try_with(|h| h.force_old.set(h.force_old.get() + 1));
    let _g = Guard;
    f()
}

/// Whether this thread is forcing old-space allocation.
pub fn is_forcing_old() -> bool {
    HEAP.with(|h| h.force_old.get() > 0)
}

/// Whether `ptr` is live nursery memory on this thread.
pub fn is_young(ptr: *const u8) -> bool {
    HEAP.with(|h| h.is_young(ptr))
}

/// Nesting depth of [`enter_eval`] on this thread.
pub fn eval_depth() -> u32 {
    HEAP.with(|h| h.depth.get())
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

/// A frame cell: null, an immortal singleton, or a leaf in live old or
/// live nursery memory. A pointer into reclaimed nursery fails.
#[cfg(debug_assertions)]
pub fn assert_frame_cell(w: crate::runtime::object::CelRef) {
    debug_assert!(
        w.is_null() || is_immortal(w as *const u8) || with_heap(|h| h.contains(w as *const u8)),
        "frame cell is not a live leaf"
    );
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

    /// `contains` is true only for addresses this heap handed out.
    #[test]
    fn contains_reports_this_heaps_payloads() {
        let heap = CelHeap::new();
        let a = heap.alloc(1u64) as *const u8;
        assert!(heap.contains(a));
        assert!(!heap.contains(core::ptr::null()));
        let other = CelHeap::new();
        assert!(!other.contains(a));
    }

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

    /// Host objects live as long as the heap and come back by index.
    #[test]
    fn host_slots_round_trip_until_the_heap_drops() {
        let heap = CelHeap::new();
        let a = heap.push_host(Box::new(7u64));
        let b = heap.push_host(Box::new(8u64));
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(heap.host_count(), 2);
        assert_eq!(
            heap.with_host(a, |h| *h.downcast_ref::<u64>().unwrap()),
            Some(7)
        );
        assert_eq!(
            heap.with_host(b, |h| *h.downcast_ref::<u64>().unwrap()),
            Some(8)
        );
        assert!(heap.with_host(2, |_| ()).is_none());
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

    /// An outermost scope reclaims nursery memory; old space is untouched.
    #[test]
    fn outermost_scope_resets_the_nursery() {
        let heap = CelHeap::new();
        let old = heap.alloc(1u64);
        assert!(heap.enter());
        let young = heap.alloc(2u64);
        assert!(heap.contains(young as *const u8));
        assert!(heap.is_young(young as *const u8));
        assert!(!heap.is_young(old as *const u8));
        let bytes_during = heap.allocated_bytes();
        heap.leave(true);
        assert!(heap.contains(old as *const u8));
        assert!(!heap.contains(young as *const u8));
        assert!(heap.allocated_bytes() < bytes_during);
        unsafe { assert_eq!(*old, 1) };
    }

    /// Nested scopes do not reset; only the outermost does.
    #[test]
    fn nested_scope_does_not_reset() {
        let heap = CelHeap::new();
        assert!(heap.enter());
        let a = heap.alloc(1u64);
        assert!(!heap.enter());
        let b = heap.alloc(2u64);
        heap.leave(false);
        assert!(heap.contains(a as *const u8));
        assert!(heap.contains(b as *const u8));
        heap.leave(true);
        assert!(!heap.contains(a as *const u8));
        assert!(!heap.contains(b as *const u8));
    }
}
