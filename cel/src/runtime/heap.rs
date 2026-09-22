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
//! is not reclaimed until the heap is dropped. Objects wrapped at bind live
//! in a [`BindRegion`] owned by the [`crate::Context`] that bound them and
//! are released when that Context is dropped. There is still no root walker,
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
use core::ptr::{null_mut, NonNull};

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

/// Bind regions kept after a Context drops, so the next wrap on this
/// thread reuses the chunk instead of asking the allocator again.
const SPARE_REGIONS: usize = 2;

/// High bit on a [`CelHeap::push_host`] index: the host sits in the
/// evaluation-scoped table and is dropped when the outermost scope resets.
const YOUNG_HOST_BIT: i64 = 1 << 62;

/// Host parked in a [`BindRegion`]. The next 29 bits are the region's id
/// on this heap; the low 32 bits are the slot in that region's table.
const REGION_HOST_BIT: i64 = 1 << 61;

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

    #[inline]
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

    #[cold]
    #[inline(never)]
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

    #[cfg(debug_assertions)]
    fn poison_all(&self) {
        let mut segs = self.segments.borrow_mut();
        if let Some(open) = segs.last_mut() {
            open.used = self.open_used.get();
        }
        for seg in segs.iter_mut() {
            seg.poison_from(0);
        }
    }
}

/// Chunk list a [`crate::Context`] allocates bind-wrapped leaves from.
///
/// Dropping it poisons the chunks in debug builds and releases them in
/// O(chunks). Immortal singletons are not allocated here.
#[doc(hidden)]
pub struct BindRegion {
    space: Space,
    hosts: RefCell<Vec<Box<dyn Any>>>,
    /// The heap this region was attached to. Evaluation against the owning
    /// Context loads this pointer so the scope guard does not read
    /// thread-local storage. Context is neither Send nor Sync, so the load
    /// stays on the attaching thread.
    heap: Cell<*const CelHeap>,
    id: Cell<u32>,
    /// Evaluation frame reused across `Program::execute` calls on the
    /// owning Context. Allocated in this region's chunks, so rewind and
    /// drop reclaim it. Null until the first execute that needs one.
    eval_frame: Cell<*mut u8>,
    eval_frame_cap: Cell<usize>,
}

impl BindRegion {
    pub fn new() -> BindRegion {
        BindRegion {
            space: Space::new(),
            hosts: RefCell::new(Vec::new()),
            heap: Cell::new(core::ptr::null()),
            id: Cell::new(0),
            eval_frame: Cell::new(core::ptr::null_mut()),
            eval_frame_cap: Cell::new(0),
        }
    }

    #[inline]
    fn bump(&self, size: usize, align: usize) -> *mut u8 {
        self.space.bump(size, align)
    }

    fn contains(&self, p: usize) -> bool {
        self.space.contains(p)
    }

    fn push_host(&self, host: Box<dyn Any>) -> i64 {
        let mut hosts = self.hosts.borrow_mut();
        let idx = hosts.len() as i64;
        hosts.push(host);
        REGION_HOST_BIT | ((self.id.get() as i64) << 32) | idx
    }

    fn rewind(&self) {
        #[cfg(debug_assertions)]
        self.space.poison_all();
        let mut segs = self.space.segments.borrow_mut();
        for seg in segs.iter_mut() {
            seg.used = 0;
        }
        drop(segs);
        self.space.bytes.set(0);
        self.space.objects.set(0);
        self.space.sync_open();
        self.space.open_used.set(0);
        self.hosts.borrow_mut().clear();
        self.id.set(0);
        self.heap.set(core::ptr::null());
        self.eval_frame.set(core::ptr::null_mut());
        self.eval_frame_cap.set(0);
    }

    /// A previously allocated evaluation frame whose item cap is at least
    /// `need`, or null.
    #[inline]
    pub(crate) fn take_eval_frame(&self, need: usize) -> *mut u8 {
        let p = self.eval_frame.get();
        if !p.is_null() && self.eval_frame_cap.get() >= need {
            p
        } else {
            core::ptr::null_mut()
        }
    }

    #[inline]
    pub(crate) fn store_eval_frame(&self, ptr: *mut u8, cap: usize) {
        self.eval_frame.set(ptr);
        self.eval_frame_cap.set(cap);
    }
}

impl Drop for BindRegion {
    fn drop(&mut self) {
        let heap = self.heap.get();
        if !heap.is_null() {
            // SAFETY: `heap` is the attaching [`CelHeap`]; recycle detaches
            // first so this arm does not run for a slot that is returned to
            // the spare list. `detach_region` compares addresses only.
            unsafe { (*heap).detach_region(self as *mut BindRegion) };
        }
        #[cfg(debug_assertions)]
        self.space.poison_all();
    }
}

/// Owning slot for a [`BindRegion`]: one raw pointer from `Box::into_raw`.
///
/// The heap's region list holds the same pointer. Access is a shared
/// reference (the region mutates through interior mutability). Dropping
/// the slot returns the region to this thread's heap spare, so the next
/// Context reuses the chunk.
#[doc(hidden)]
pub struct BindRegionSlot {
    inner: Option<NonNull<BindRegion>>,
}

impl BindRegionSlot {
    pub(crate) const fn empty() -> BindRegionSlot {
        BindRegionSlot { inner: None }
    }

    pub(crate) fn get_or_insert(&mut self) -> *mut BindRegion {
        match self.inner {
            Some(p) => p.as_ptr(),
            None => {
                let p = take_bind_region();
                self.inner = Some(p);
                p.as_ptr()
            }
        }
    }

    /// The heap this slot's region was attached to, if wrap-at-bind has run.
    #[inline]
    pub(crate) fn attached_heap(&self) -> Option<&CelHeap> {
        let region = self.get()?;
        let heap = region.heap.get();
        if heap.is_null() {
            None
        } else {
            // SAFETY: `attach_region` stores the attaching heap; the region
            // is detached before that heap is dropped. Context is neither
            // Send nor Sync, so this is only read on the attaching thread.
            Some(unsafe { &*heap })
        }
    }

    #[inline]
    pub(crate) fn get(&self) -> Option<&BindRegion> {
        self.inner.map(|p| {
            // SAFETY: `p` came from [`Box::into_raw`] and is still owned
            // by this slot. The heap's region list, if any, holds the
            // same pointer and only forms shared references.
            unsafe { p.as_ref() }
        })
    }

    #[cfg(test)]
    pub(crate) fn is_none(&self) -> bool {
        self.inner.is_none()
    }
}

impl Drop for BindRegionSlot {
    fn drop(&mut self) {
        if let Some(region) = self.inner.take() {
            recycle_bind_region(region);
        }
    }
}

fn take_bind_region() -> NonNull<BindRegion> {
    HEAP.with(|h| h.take_spare_region()).unwrap_or_else(|| {
        // SAFETY: `Box::into_raw` is never null.
        unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(BindRegion::new()))) }
    })
}

fn recycle_bind_region(region: NonNull<BindRegion>) {
    // SAFETY: `region` is a live `Box::into_raw` pointer still owned by
    // the slot that just released it.
    let heap = unsafe { region.as_ref() }.heap.get();
    if heap.is_null() {
        // SAFETY: never attached; not in a heap list.
        unsafe { free_region(region) };
        return;
    }
    // SAFETY: `heap` is the attaching [`CelHeap`], still live because
    // the region is detached only inside `recycle_region`.
    unsafe { (*heap).recycle_region(region) };
}

/// # Safety
///
/// `region` came from [`Box::into_raw`] and is not stored in a
/// [`CelHeap`] region list or spare list.
unsafe fn free_region(region: NonNull<BindRegion>) {
    // SAFETY: caller: unique remaining owner of this `Box::into_raw`.
    unsafe { drop(Box::from_raw(region.as_ptr())) };
}

/// One thread's value heap.
///
/// Not `Send` and not `Sync`. Two spaces: old lives until the heap is
/// dropped; the nursery is reset when the outermost evaluation on this
/// thread finishes. Bind-wrapped objects live in [`BindRegion`]s attached
/// here so [`contains`] can see them while the owning Context lives.
pub struct CelHeap {
    old: Space,
    nursery: Space,
    /// Outermost-evaluation nesting. Zero means no evaluation is running.
    depth: Cell<u32>,
    /// Nested `with_old_space` / [`with_bind_region`] calls skip the nursery.
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
    /// Region wrap-at-bind is filling. Only consulted on the old-space path.
    bind_region: Cell<*mut BindRegion>,
    /// Live Context regions, so [`contains`] and region-host lookup see them.
    /// Each pointer is the same `Box::into_raw` address the owning slot holds.
    regions: RefCell<Vec<NonNull<BindRegion>>>,
    next_region_id: Cell<u32>,
    /// Detached, rewound regions waiting for the next Context on this thread.
    spare_regions: RefCell<Vec<NonNull<BindRegion>>>,
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
            bind_region: Cell::new(null_mut()),
            regions: RefCell::new(Vec::new()),
            next_region_id: Cell::new(0),
            spare_regions: RefCell::new(Vec::new()),
        }
    }

    #[cold]
    fn attach_region(&self, region: *mut BindRegion) {
        // SAFETY: `region` is a live `Box::into_raw` pointer. The region
        // mutates through interior mutability; this is a shared borrow.
        let r = unsafe { &*region };
        if !r.heap.get().is_null() {
            return;
        }
        r.heap.set(self as *const CelHeap);
        let id = self.next_region_id.get().wrapping_add(1).max(1);
        self.next_region_id.set(id);
        r.id.set(id);
        // SAFETY: `region` is the same non-null `Box::into_raw` pointer
        // the owning slot holds. Heap and owner both access it through
        // shared references until detach.
        self.regions
            .borrow_mut()
            .push(unsafe { NonNull::new_unchecked(region) });
    }

    #[cold]
    fn detach_region(&self, region: *mut BindRegion) {
        self.regions.borrow_mut().retain(|r| r.as_ptr() != region);
        if self.bind_region.get() == region {
            self.bind_region.set(null_mut());
        }
    }

    #[cold]
    fn take_spare_region(&self) -> Option<NonNull<BindRegion>> {
        self.spare_regions.borrow_mut().pop()
    }

    #[cold]
    fn recycle_region(&self, region: NonNull<BindRegion>) {
        self.detach_region(region.as_ptr());
        // SAFETY: detached; the heap list no longer holds this pointer.
        // Rewind uses interior mutability.
        unsafe { region.as_ref() }.rewind();
        let mut spare = self.spare_regions.borrow_mut();
        if spare.len() < SPARE_REGIONS {
            spare.push(region);
        } else {
            // SAFETY: detached, rewound, not in spare; from `Box::into_raw`.
            unsafe { free_region(region) };
        }
    }

    #[inline]
    fn in_nursery(&self) -> bool {
        self.depth.get() > 0 && self.force_old.get() == 0
    }

    #[inline]
    fn enter(&self) -> bool {
        let d = self.depth.get();
        if d == 0 {
            self.install_snap(NurserySnap::read(self));
        }
        self.depth.set(d + 1);
        d == 0
    }

    #[inline]
    fn leave(&self, outermost: bool) {
        self.leave_with(
            outermost,
            NurserySnap {
                used: self.snap_used.get(),
                len: self.snap_len.get(),
                hosts: self.snap_hosts.get(),
            },
        );
    }

    #[inline(always)]
    fn leave_with(&self, outermost: bool, snap: NurserySnap) {
        let d = self.depth.get();
        self.depth.set(d.saturating_sub(1));
        if !outermost {
            return;
        }
        if self.nursery.open_used.get() == snap.used && self.young_host_n.get() == snap.hosts {
            return;
        }
        self.leave_slow(snap);
    }

    #[cold]
    #[inline(never)]
    fn leave_slow(&self, snap: NurserySnap) {
        self.install_snap(snap);
        let live = self.nursery.bytes.get();
        if live > self.nursery_high_water.get() {
            self.nursery_high_water.set(live);
        }
        if self.nursery.n_segs.get() == snap.len && self.young_host_n.get() == snap.hosts {
            self.rewind_nursery();
        } else {
            self.reset_nursery();
        }
    }

    #[inline]
    fn install_snap(&self, snap: NurserySnap) {
        self.snap_used.set(snap.used);
        self.snap_len.set(snap.len);
        self.snap_hosts.set(snap.hosts);
        self.snap_bytes.set(0);
        self.snap_objects.set(0);
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
    #[inline]
    pub fn alloc<T>(&self, value: T) -> *mut T {
        let () = AssertNoDrop::<T>::OK;
        let ptr = self.alloc_raw(size_of::<T>(), align_of::<T>()) as *mut T;
        // SAFETY: `alloc_raw` returns an address with `T`'s size and alignment
        // that nothing else has been handed. The bytes are written here, so
        // the bump does not zero them.
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
    #[inline]
    pub fn alloc_raw(&self, size: usize, align: usize) -> *mut u8 {
        if self.in_nursery() {
            self.nursery.bump(size, align)
        } else {
            self.alloc_raw_slow(size, align)
        }
    }

    #[cold]
    #[inline(never)]
    fn alloc_raw_slow(&self, size: usize, align: usize) -> *mut u8 {
        let region = self.bind_region.get();
        if !region.is_null() {
            return unsafe { (*region).bump(size, align) };
        }
        self.old.bump(size, align)
    }

    /// Reserve `size` bytes in old space.
    #[inline]
    pub fn alloc_old_raw(&self, size: usize, align: usize) -> *mut u8 {
        self.old.bump(size, align)
    }

    /// Objects handed out that are still live: old space plus the current
    /// nursery bump plus attached bind regions. A reset drops the nursery
    /// contribution; dropping a Context drops its region.
    pub fn allocated_objects(&self) -> u64 {
        self.old.objects.get()
            + self.nursery.objects.get()
            + self
                .regions
                .borrow()
                .iter()
                .map(|r| unsafe { r.as_ref().space.objects.get() })
                .sum::<u64>()
    }

    /// Bytes handed out that are still live, excluding alignment padding.
    pub fn allocated_bytes(&self) -> u64 {
        self.old.bytes.get()
            + self.nursery.bytes.get()
            + self
                .regions
                .borrow()
                .iter()
                .map(|r| unsafe { r.as_ref().space.bytes.get() })
                .sum::<u64>()
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
    /// young and is dropped on reset; during wrap it is parked on the active
    /// bind region; otherwise it lives as long as the heap.
    pub fn push_host(&self, host: Box<dyn Any>) -> i64 {
        if self.in_nursery() {
            let mut hosts = self.young_hosts.borrow_mut();
            let idx = hosts.len() as i64;
            hosts.push(host);
            self.young_host_n.set(hosts.len());
            idx | YOUNG_HOST_BIT
        } else if !self.bind_region.get().is_null() {
            unsafe { (*self.bind_region.get()).push_host(host) }
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
        } else if idx & REGION_HOST_BIT != 0 {
            let id = ((idx & !REGION_HOST_BIT) >> 32) as u32;
            let local = (idx as u32) as usize;
            let regions = self.regions.borrow();
            for region in regions.iter() {
                let region = unsafe { region.as_ref() };
                if region.id.get() == id {
                    let hosts = region.hosts.borrow();
                    let slot = hosts.get(local)?;
                    return Some(f(slot.as_ref()));
                }
            }
            None
        } else {
            let hosts = self.hosts.borrow();
            let slot = hosts.get(idx as usize)?;
            Some(f(slot.as_ref()))
        }
    }

    /// How many host objects this heap is holding, both spaces and regions.
    pub fn host_count(&self) -> usize {
        let region_hosts = self
            .regions
            .borrow()
            .iter()
            .map(|r| unsafe { r.as_ref().hosts.borrow().len() })
            .sum::<usize>();
        self.hosts.borrow().len() + self.young_hosts.borrow().len() + region_hosts
    }

    /// Whether `ptr` lies in live old, live nursery, or a live bind region.
    pub fn contains(&self, ptr: *const u8) -> bool {
        let p = ptr as usize;
        self.old.contains(p)
            || self.nursery.contains(p)
            || self
                .regions
                .borrow()
                .iter()
                .any(|r| unsafe { r.as_ref().contains(p) })
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

impl Drop for CelHeap {
    fn drop(&mut self) {
        for region in self.spare_regions.get_mut().drain(..) {
            // SAFETY: spare regions were detached and rewound; each
            // pointer came from `Box::into_raw` and is not in `regions`.
            unsafe { free_region(region) };
        }
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
    /// Cached pointer at [`HEAP`]. Const-initialised so a load is native TLS
    /// plus a null check, not the lazy-init state machine [`HEAP`] uses.
    static HEAP_PTR: Cell<*const CelHeap> = const { Cell::new(core::ptr::null()) };
    /// Nested [`with_old_space`] depth. A `Cell` of its own so a load during
    /// intern does not go through [`HEAP`].
    static FORCE_OLD: Cell<u32> = const { Cell::new(0) };
}

/// This thread's heap, resolved once.
#[inline]
fn heap_ptr() -> *const CelHeap {
    HEAP_PTR.with(|p| {
        let cached = p.get();
        if cached.is_null() {
            let h = HEAP.with(|heap| heap as *const CelHeap);
            p.set(h);
            h
        } else {
            cached
        }
    })
}

/// Run `f` against this thread's heap.
#[inline]
pub fn with_heap<R>(f: impl FnOnce(&CelHeap) -> R) -> R {
    HEAP.with(f)
}

/// Nursery bump captured at outermost entry. Leave compares [`Self::used`]
/// with the live bump; a match means this evaluation allocated nothing young.
/// `len` / `bytes` / `objects` are filled on the moved path from this
/// snapshot plus zeros: an outermost enter always sees a rewound nursery.
#[derive(Clone, Copy)]
struct NurserySnap {
    used: usize,
    len: usize,
    hosts: usize,
}

impl NurserySnap {
    const ZERO: NurserySnap = NurserySnap {
        used: 0,
        len: 0,
        hosts: 0,
    };

    #[inline]
    fn read(heap: &CelHeap) -> NurserySnap {
        NurserySnap {
            used: heap.nursery.open_used.get(),
            len: heap.nursery.n_segs.get(),
            hosts: heap.young_host_n.get(),
        }
    }
}

/// Guard for one evaluation. Leave runs in [`Self::finish`] on the happy
/// path and in [`Drop`] on panic or error.
///
/// The heap is the Context's bind-region heap when one is attached,
/// otherwise this thread's heap. Leave does not read thread-local storage.
/// An evaluation that never allocated young compares one word and returns.
pub struct EvalScope {
    heap: *const CelHeap,
    outermost: bool,
    snap: NurserySnap,
}

impl EvalScope {
    /// The heap this scope opened. Valid until leave.
    #[inline]
    pub fn heap(&self) -> &CelHeap {
        unsafe { &*self.heap }
    }

    /// Whether this guard opened the outermost scope on this thread.
    #[inline]
    pub fn is_outermost(&self) -> bool {
        self.outermost
    }

    /// The public form of `v`. An interned leaf unpacks; a value that is
    /// already public is returned as it is. Nested and outermost both unpack,
    /// so a host re-entry never observes `Value::Interned`.
    ///
    /// Leave is the inlined bump compare; [`Drop`] is not taken on this path.
    #[inline(always)]
    pub fn finish(self, v: crate::Value) -> crate::Value {
        let out = to_public(v);
        unsafe { (*self.heap).leave_with(self.outermost, self.snap) };
        core::mem::forget(self);
        out
    }
}

impl Drop for EvalScope {
    fn drop(&mut self) {
        unsafe { (*self.heap).leave_with(self.outermost, self.snap) };
    }
}

/// Open an evaluation scope on `heap`. Nested calls increment depth;
/// only the outermost leave reclaims the nursery.
#[inline(always)]
pub fn enter_eval_on(heap: &CelHeap) -> EvalScope {
    let d = heap.depth.get();
    let outermost = d == 0;
    let snap = if outermost {
        NurserySnap::read(heap)
    } else {
        NurserySnap::ZERO
    };
    heap.depth.set(d + 1);
    EvalScope {
        heap,
        outermost,
        snap,
    }
}

/// Open an evaluation scope on this thread. Nested calls increment depth;
/// only the outermost reset reclaims the nursery.
#[inline]
pub fn enter_eval() -> EvalScope {
    enter_eval_on(unsafe { &*heap_ptr() })
}

#[cold]
#[inline(never)]
fn enter_eval_unbound() -> EvalScope {
    enter_eval()
}

/// Open an evaluation scope on `ctx`'s attached heap, or this thread's
/// heap if the Context has not bound a region.
///
/// A bound Context's region is attached to the binding thread's heap.
/// Context is neither Send nor Sync, so evaluation stays on that thread
/// and this path does not read thread-local storage.
#[inline]
pub fn enter_eval_for(ctx: &crate::context::Context) -> EvalScope {
    match ctx.eval_heap() {
        Some(heap) => {
            debug_assert!(core::ptr::eq(heap as *const CelHeap, heap_ptr()));
            enter_eval_on(heap)
        }
        None => enter_eval_unbound(),
    }
}

#[inline]
fn to_public(v: crate::Value) -> crate::Value {
    match v {
        crate::Value::Interned(w) => crate::runtime::convert::interned_to_public(w),
        other => other,
    }
}

/// Force allocations (and host parks) into old space for the duration of `f`.
pub fn with_old_space<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            FORCE_OLD.with(|c| c.set(c.get().saturating_sub(1)));
            let _ = HEAP.try_with(|h| h.force_old.set(h.force_old.get().saturating_sub(1)));
        }
    }
    FORCE_OLD.with(|c| c.set(c.get() + 1));
    let _ = HEAP.try_with(|h| h.force_old.set(h.force_old.get() + 1));
    let _g = Guard;
    f()
}

/// Route allocations and host parks into `region` for the duration of `f`.
///
/// The region stays attached to this thread's heap until it is dropped, so
/// [`CelHeap::contains`] keeps recognising its objects after wrap returns.
#[cold]
pub(crate) fn with_bind_region<R>(region: *mut BindRegion, f: impl FnOnce() -> R) -> R {
    struct Guard {
        prev: *mut BindRegion,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            FORCE_OLD.with(|c| c.set(c.get().saturating_sub(1)));
            let _ = HEAP.try_with(|h| {
                h.force_old.set(h.force_old.get().saturating_sub(1));
                h.bind_region.set(self.prev);
            });
        }
    }
    FORCE_OLD.with(|c| c.set(c.get() + 1));
    let prev = HEAP.with(|h| {
        h.force_old.set(h.force_old.get() + 1);
        let prev = h.bind_region.get();
        h.bind_region.set(region);
        h.attach_region(region);
        prev
    });
    let _g = Guard { prev };
    f()
}

/// Whether this thread is forcing old-space allocation.
pub fn is_forcing_old() -> bool {
    FORCE_OLD.with(|c| c.get() > 0)
}

/// Whether `ptr` is live nursery memory on this thread.
pub fn is_young(ptr: *const u8) -> bool {
    unsafe { (*heap_ptr()).is_young(ptr) }
}

/// Nesting depth of [`enter_eval`] on this thread.
pub fn eval_depth() -> u32 {
    unsafe { (*heap_ptr()).depth.get() }
}

/// Bytes preceding an immortal payload. The backend reads
/// `[obj - HEADER_SIZE]` for `guard_is_object`; a plain Rust `static`
/// would put that load in rodata or unmapped memory.
pub const IMMORTAL_HEADER_SIZE: usize = core::mem::size_of::<usize>();

/// Payloads handed out by [`alloc_immortal`]. The tripwire that a
/// prebuilt is not a Rust `static` asserts against this list.
static IMMORTAL_PAYLOADS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Live constant-pool spans owned by [`super::const_pool::ConstPool`].
/// `(payload, payload+len)` so [`is_immortal`] treats them like prebuilts
/// until the pool drops.
static CONST_SPANS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

pub(crate) const IMMORTAL_MARK: usize = 0xC3_11_07_7A;

/// Register a constant-pool payload so [`is_immortal`] accepts it.
pub(crate) fn register_const_span(payload: *mut u8, len: usize) {
    if payload.is_null() || len == 0 {
        return;
    }
    let start = payload as usize;
    CONST_SPANS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push((start, start.saturating_add(len)));
}

/// Forget a constant-pool payload; the caller then frees the block.
pub(crate) fn unregister_const_span(payload: *mut u8) {
    let start = payload as usize;
    CONST_SPANS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .retain(|(s, _)| *s != start);
}

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

/// A frame cell: null, an immortal singleton, or a leaf in live old,
/// live nursery, or a live bind region. A pointer into reclaimed nursery
/// or a dropped Context's region fails.
#[cfg(debug_assertions)]
pub fn assert_frame_cell(w: crate::runtime::object::CelRef) {
    debug_assert!(
        w.is_null() || is_immortal(w as *const u8) || with_heap(|h| h.contains(w as *const u8)),
        "frame cell is not a live leaf"
    );
}

/// Whether `ptr` is a payload [`alloc_immortal`] handed out, or a live
/// constant-pool leaf owned by a [`super::const_pool::ConstPool`].
pub fn is_immortal(ptr: *const u8) -> bool {
    if ptr.is_null() {
        return false;
    }
    let p = ptr as usize;
    if CONST_SPANS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .any(|&(s, e)| p >= s && p < e)
    {
        return true;
    }
    IMMORTAL_PAYLOADS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .any(|&q| q == p)
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

    /// A young fixed-size allocation updates the live counters by one object
    /// and `size_of` bytes, including when many fit in the open segment.
    #[test]
    fn young_fixed_size_counts_stay_exact() {
        let heap = CelHeap::new();
        assert!(heap.enter());
        let before_n = heap.allocated_objects();
        let before_b = heap.allocated_bytes();
        for i in 0..64u64 {
            heap.alloc(i);
        }
        assert_eq!(heap.allocated_objects(), before_n + 64);
        assert_eq!(
            heap.allocated_bytes(),
            before_b + 64 * size_of::<u64>() as u64
        );
        heap.leave(true);
        assert_eq!(heap.allocated_objects(), before_n);
        assert_eq!(heap.allocated_bytes(), before_b);
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

    #[test]
    fn enter_eval_on_tracks_depth_on_the_given_heap() {
        let scope = enter_eval_on(unsafe { &*heap_ptr() });
        assert!(scope.is_outermost());
        assert_eq!(eval_depth(), 1);
        drop(scope);
        assert_eq!(eval_depth(), 0);
    }

    /// An evaluation that never bumps young memory does not reset: leave
    /// compares the saved bump and returns.
    #[test]
    fn no_young_allocation_skips_reset() {
        let heap = CelHeap::new();
        let old = heap.alloc(1u64);
        let bytes = heap.allocated_bytes();
        assert!(heap.enter());
        heap.leave(true);
        assert_eq!(heap.allocated_bytes(), bytes);
        assert!(heap.contains(old as *const u8));
        unsafe { assert_eq!(*old, 1) };
    }

    /// A bind region is live for `contains` until it is dropped, and its
    /// bytes are not old-space bytes.
    #[test]
    fn a_bind_region_is_contained_until_it_drops() {
        let heap = CelHeap::new();
        let mut slot = BindRegionSlot::empty();
        let region = slot.get_or_insert();
        heap.attach_region(region);
        let p = unsafe { &*region }.bump(size_of::<u64>(), align_of::<u64>()) as *mut u64;
        unsafe { p.write(7) };
        assert!(heap.contains(p as *const u8));
        assert!(!heap.is_young(p as *const u8));
        assert_eq!(heap.old_allocated_bytes(), 0);
        assert_eq!(heap.allocated_objects(), 1);
        drop(slot);
        assert!(!heap.contains(p as *const u8));
        assert_eq!(heap.allocated_objects(), 0);
    }
}
