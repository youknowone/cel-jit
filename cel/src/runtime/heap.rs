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
//! pointer when the open segment is half used (`incminimark.py`
//! `collect_and_reserve`), not after every outermost evaluation; old space
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
//! ⛔ **`install_cel_gc` is a second heap.** It installs `MiniMarkGC` and
//! turns on `set_new_via_gc`, after which a compiled `NewWithVtable` allocates
//! from that nursery while the interpreter builds values here. The portal
//! does not call it. The portal's collector is [`CelGc`]: `gc.py`
//! `GcLLDescr_framework` over this same heap. Every value carries the
//! type-id header, `supports_guard_gc_type` is true, and compiled
//! `CALL_MALLOC_NURSERY` bumps [`CelHeap::nursery_free`]. Nothing collects
//! and nothing moves.
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

use super::lltype::CelGcType;

/// Bytes per segment.
///
/// Segments are the unit of `alloc`/`dealloc`, not of collection, so the size
/// trades header overhead against how much a mostly-empty last segment wastes.
/// One 64 KiB segment holds a few thousand headered leaves.
const SEGMENT_BYTES: usize = 64 * 1024;

/// Bytes in front of every cel payload.
///
/// The type id lives in the low bits of this word. `gc.py`
/// `GcLLDescr_framework` reads it at `obj - GC_HEADER_SIZE`. A build
/// without `majit_gc` stores the same `u64`.
pub const GC_HEADER_SIZE: usize = 8;

#[cfg(feature = "jit")]
const _: () = {
    assert!(GC_HEADER_SIZE == majit_gc::header::GcHeader::SIZE);
};

/// `GC_HEADER_SIZE + size`, rounded up to a multiple of 8 so a nursery
/// bump leaves `nursery_free` 8-aligned. The compiled inline bump does
/// not realign.
pub(crate) fn headered_total(size: usize) -> usize {
    try_headered_total(size).expect("allocation fits")
}

fn try_headered_total(size: usize) -> Option<usize> {
    let total = GC_HEADER_SIZE.checked_add(size)?;
    Some(total.checked_add(7)? & !7)
}

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

impl Default for BindRegion {
    fn default() -> Self {
        Self::new()
    }
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
/// dropped; the nursery is reclaimed when the open segment is half used,
/// not when each outermost evaluation finishes. Bind-wrapped objects live
/// in [`BindRegion`]s attached here so [`contains`] can see them while the
/// owning Context lives.
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
    /// `incminimark.py` `nursery_free`. Absolute pointer into the open
    /// nursery segment. Compiled code bakes the address of this cell
    /// (`gc.py get_nursery_free_addr`). The cell stays put for the life of
    /// the heap; `bump_grow`, rewind and `reset_nursery` store the new
    /// pointer into it.
    nursery_free: Cell<*mut u8>,
    /// `incminimark.py` `nursery_top`. One past the open segment.
    nursery_top: Cell<*mut u8>,
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
    /// VM word [`crate::vm::portal`] publishes for a re-entering step.
    /// The heap is already resolved for the call; this is not a second
    /// thread-local.
    pub(crate) portal_vm: Cell<i64>,
    /// Driver [`crate::vm::portal`] is inside, so a `may_force` residual
    /// can reach it without another thread-local.
    pub(crate) active_driver: Cell<usize>,
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
            nursery_free: Cell::new(null_mut()),
            nursery_top: Cell::new(null_mut()),
            hosts: RefCell::new(Vec::new()),
            young_hosts: RefCell::new(Vec::new()),
            bind_region: Cell::new(null_mut()),
            regions: RefCell::new(Vec::new()),
            next_region_id: Cell::new(0),
            spare_regions: RefCell::new(Vec::new()),
            portal_vm: Cell::new(0),
            active_driver: Cell::new(0),
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

    // Tests call these. A non-test build has no caller.
    #[allow(dead_code)]
    #[inline]
    fn enter(&self) -> bool {
        let d = self.depth.get();
        if d == 0 {
            self.install_snap(NurserySnap::read(self));
        }
        self.depth.set(d + 1);
        d == 0
    }

    #[allow(dead_code)]
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
        // Compiled bumps and the nursery fast path advance `nursery_free`
        // and leave `open_used` for [`flush_nursery_bump`]. Compare the
        // live cursor, or an evaluation that only bumped the pointer
        // looks empty and skips the rewind.
        if self.live_nursery_used() == snap.used && self.young_host_n.get() == snap.hosts {
            return;
        }
        // Same open segment, no young hosts: the cursor moved. `segment.used`
        // is refreshed by [`Self::flush_nursery_bump`] before a grow or a
        // reset reads it, so this path does not borrow the segment vector.
        // `incminimark.py` `collect_and_reserve`: the nursery is reclaimed
        // when it fills, not per evaluation. Dead bytes stay until the open
        // segment is half used; `open_used` is left at the last rewind.
        // The inline bump adds a fixed size and does not realign, so a cursor
        // that is not a multiple of 8 is rewound now. `publish_nursery_bounds`
        // then restores `base + open_used`, which stays aligned.
        if self.nursery.n_segs.get() == snap.len && self.young_host_n.get() == snap.hosts {
            let live = self.live_nursery_used();
            if live % align_of::<u64>() == 0 && live <= self.nursery.open_cap.get() / 2 {
                return;
            }
            self.rewind_same_segment(snap);
            return;
        }
        self.leave_slow(snap);
    }

    /// Rewind the open cursor. The segment list is unchanged.
    fn rewind_same_segment(&self, snap: NurserySnap) {
        let live = self.nursery.bytes.get();
        if live > self.nursery_high_water.get() {
            self.nursery_high_water.set(live);
        }
        self.install_snap(snap);
        #[cfg(debug_assertions)]
        self.flush_nursery_bump();
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
        self.publish_nursery_bounds();
    }

    /// Bytes handed out of the open nursery segment.
    ///
    /// [`Self::nursery_free`] is the cursor both sides bump. `open_used`
    /// catches up in [`Self::flush_nursery_bump`], so a read in between
    /// has to use the pointer.
    #[inline(always)]
    fn live_nursery_used(&self) -> usize {
        let base = self.nursery.open_base.get() as usize;
        let free = self.nursery_free.get() as usize;
        if base != 0 && free >= base {
            free - base
        } else {
            self.nursery.open_used.get()
        }
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
        self.flush_nursery_bump();
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
        self.publish_nursery_bounds();
    }

    fn reset_nursery(&self) {
        self.flush_nursery_bump();
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
        self.publish_nursery_bounds();
    }

    /// Allocate `value` in this heap and return a pointer to the payload.
    ///
    /// The type-id word sits at `payload - GC_HEADER_SIZE`. During an
    /// evaluation the pointer is nursery memory and is invalid after the
    /// outermost scope resets, unless it was allocated through [`alloc_old`].
    #[inline]
    pub fn alloc<T: super::lltype::CelGcType>(&self, value: T) -> *mut T {
        let () = AssertNoDrop::<T>::OK;
        let ptr = self.alloc_headered(
            u64::from(<T as CelGcType>::TYPE_ID),
            size_of::<T>(),
            align_of::<T>(),
            false,
        ) as *mut T;
        // SAFETY: `alloc_headered` returns a payload with `T`'s size and
        // alignment that nothing else has been handed. The bytes are written
        // here, so the bump does not zero them.
        unsafe { ptr.write(value) };
        ptr
    }

    /// Allocate `value` in old space even if an evaluation is running.
    pub fn alloc_old<T: super::lltype::CelGcType>(&self, value: T) -> *mut T {
        let () = AssertNoDrop::<T>::OK;
        let ptr = self.alloc_headered(
            u64::from(<T as CelGcType>::TYPE_ID),
            size_of::<T>(),
            align_of::<T>(),
            true,
        ) as *mut T;
        unsafe { ptr.write(value) };
        ptr
    }

    /// Header plus `size` payload bytes. Returns the payload.
    ///
    /// Block allocations ([`super::object_array`]) go through here. The
    /// reservation is a multiple of 8 so [`Self::nursery_free`] stays aligned.
    #[inline]
    pub fn alloc_raw_typed(&self, type_id: u32, size: usize, align: usize) -> *mut u8 {
        self.alloc_headered(u64::from(type_id), size, align, false)
    }

    /// Reserve a header word and `size` payload bytes. `old` forces old space.
    fn alloc_headered(&self, header: u64, size: usize, align: usize, old: bool) -> *mut u8 {
        let align = align.max(GC_HEADER_SIZE);
        let total = headered_total(size);
        let raw = if old {
            self.alloc_old_raw(total, align)
        } else {
            self.alloc_raw(total, align)
        };
        // SAFETY: `raw` is uniquely owned, aligned for `u64`, and `total`
        // covers the header word.
        unsafe { (raw as *mut u64).write(header) };
        unsafe { raw.add(GC_HEADER_SIZE) }
    }

    /// Reserve `size` bytes at `align`, growing the heap if the open segment
    /// cannot serve it.
    ///
    /// Public because the payload blocks in [`super::object_array`] are sized
    /// at run time and so cannot go through the generic [`alloc`].
    #[inline]
    pub fn alloc_raw(&self, size: usize, align: usize) -> *mut u8 {
        if self.in_nursery() {
            self.bump_nursery(size, align)
        } else {
            self.alloc_raw_slow(size, align)
        }
    }

    /// Bump [`Self::nursery_free`] inside the open segment, or open another.
    ///
    /// Compiled code performs the same two-word bump inline. Both sides
    /// read and write these cells, so a segment that filled under one is
    /// full for the other. The fast path does not touch the segment
    /// vector or `open_used`; [`Self::flush_nursery_bump`] copies the
    /// cursor back before a rewind, a reset, or a new segment.
    #[inline(always)]
    fn bump_nursery(&self, size: usize, align: usize) -> *mut u8 {
        let free = self.nursery_free.get() as usize;
        let top = self.nursery_top.get() as usize;
        if free != 0 {
            let start = (free + align - 1) & !(align - 1);
            if let Some(end) = start.checked_add(size) {
                if end <= top {
                    self.nursery_free.set(end as *mut u8);
                    self.nursery.objects.set(self.nursery.objects.get() + 1);
                    self.nursery
                        .bytes
                        .set(self.nursery.bytes.get() + size as u64);
                    return start as *mut u8;
                }
            }
        }
        self.bump_nursery_grow(size, align)
    }

    /// Open a nursery segment. The fast path's cursor lives only in
    /// [`Self::nursery_free`] until this copies it onto the segment.
    #[cold]
    #[inline(never)]
    fn bump_nursery_grow(&self, size: usize, align: usize) -> *mut u8 {
        self.flush_nursery_bump();
        let ptr = self.nursery.bump_grow(size, align);
        self.publish_nursery_bounds();
        ptr
    }

    /// Store the open segment's cursor into [`Self::nursery_free`] /
    /// [`Self::nursery_top`].
    fn publish_nursery_bounds(&self) {
        let base = self.nursery.open_base.get();
        if base.is_null() {
            self.nursery_free.set(null_mut());
            self.nursery_top.set(null_mut());
            return;
        }
        let used = self.nursery.open_used.get();
        let cap = self.nursery.open_cap.get();
        self.nursery_free.set(unsafe { base.add(used) });
        self.nursery_top.set(unsafe { base.add(cap) });
    }

    /// Compiled bumps move [`Self::nursery_free`] and leave the segment's
    /// `used` behind. Copy the cursor back before a rewind or reset reads it.
    #[cold]
    #[inline(never)]
    fn flush_nursery_bump(&self) {
        let base = self.nursery.open_base.get();
        if base.is_null() {
            return;
        }
        let free = self.nursery_free.get() as usize;
        let base_u = base as usize;
        if free < base_u {
            return;
        }
        let used = free - base_u;
        self.nursery.open_used.set(used);
        if let Some(open) = self.nursery.segments.borrow_mut().last_mut() {
            open.used = used;
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
            || self.nursery_contains(p)
            || self
                .regions
                .borrow()
                .iter()
                .any(|r| unsafe { r.as_ref().contains(p) })
    }

    /// Whether `ptr` is live nursery memory.
    ///
    /// The open segment's end is [`Self::nursery_free`], which compiled
    /// code advances without touching the segment vector.
    pub fn is_young(&self, ptr: *const u8) -> bool {
        self.nursery_contains(ptr as usize)
    }

    fn nursery_contains(&self, p: usize) -> bool {
        let segs = self.nursery.segments.borrow();
        let last = segs.len().saturating_sub(1);
        let free = self.nursery_free.get() as usize;
        segs.iter().enumerate().any(|(i, seg)| {
            if i == last {
                let base = seg.base as usize;
                p >= base && p < free
            } else {
                seg.contains(p)
            }
        })
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
pub(crate) fn heap_ptr() -> *const CelHeap {
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

/// Framework GC descr (`gc.py GcLLDescr_framework`) over this thread's
/// [`CelHeap`].
///
/// Every object carries a type-id word at `payload - GcHeader::SIZE`.
/// `supports_guard_gc_type` is true, so a portal loop may unroll.
/// `gc.py get_nursery_free_addr` / `get_nursery_top_addr` name the two
/// cells on the heap. Allocation bumps the nursery and never collects.
/// `JITFRAME` is registered after the cel types (`jitframe.py`
/// `jitframe_allocate`): a type table cannot be installed without that id.
/// The frame itself is a nursery object, reclaimed with the rest of the
/// evaluation's young objects.
#[cfg(feature = "jit")]
pub struct CelGc {
    types: majit_gc::trace::TypeRegistry,
    jitframe_type_id: Option<u32>,
}

#[cfg(feature = "jit")]
impl CelGc {
    /// Register the root, every class and every block, then freeze.
    ///
    /// Ids are the [`super::lltype::CelGcType`] literals.
    /// [`super::registration::register_cel_classes_unfrozen`] checks that.
    pub fn new() -> CelGc {
        let mut gc = CelGc {
            types: majit_gc::trace::TypeRegistry::new(),
            jitframe_type_id: None,
        };
        let _ids = super::registration::register_cel_classes_unfrozen(&mut gc);
        // After the cel types, so [`super::lltype::CelGcType::TYPE_ID`]
        // stays the literal. `check_jitframe_descr` refuses the install
        // when this id is missing.
        #[cfg(not(target_arch = "wasm32"))]
        {
            majit_metainterp::register_active_backend_jitframe_gc_type(&mut gc);
        }
        majit_gc::GcAllocator::freeze_types(&mut gc);
        gc
    }

    /// Payload of `payload` bytes. `header` is the word at `payload - 8`.
    ///
    /// `0` is a zero header: the compiled nursery slow path stores the
    /// type id itself. A nonzero word is `u64::from(type_id)`, which is
    /// what `GcHeader::new` stores.
    fn bump_payload(&mut self, header: u64, payload: usize) -> majit_ir::GcRef {
        let Some(total) = try_headered_total(payload) else {
            return majit_ir::GcRef(0);
        };
        let raw = unsafe { (*heap_ptr()).bump_nursery(total, GC_HEADER_SIZE) };
        unsafe { (raw as *mut u64).write(header) };
        majit_ir::GcRef(unsafe { raw.add(GC_HEADER_SIZE) } as usize)
    }

    fn bump_typed(&mut self, type_id: u32, payload: usize) -> majit_ir::GcRef {
        // Frames are nursery objects (`jitframe.py jitframe_allocate`),
        // reclaimed with the rest of the evaluation's young objects.
        self.bump_payload(u64::from(type_id), payload)
    }
}

#[cfg(feature = "jit")]
impl majit_gc::GcAllocator for CelGc {
    fn alloc_nursery(&mut self, size: usize) -> majit_ir::GcRef {
        self.bump_payload(0, size)
    }

    fn alloc_nursery_no_collect(&mut self, size: usize) -> majit_ir::GcRef {
        self.bump_payload(0, size)
    }

    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> majit_ir::GcRef {
        self.bump_typed(type_id, size)
    }

    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> majit_ir::GcRef {
        self.bump_typed(type_id, size)
    }

    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> majit_ir::GcRef {
        self.bump_typed(type_id, size)
    }

    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> majit_ir::GcRef {
        // Every result is nursery. A young pointer stored into it needs
        // no creation barrier, and this descr never collects.
        unsafe { *needs_write_barrier = false };
        self.try_alloc_nursery_no_collect_typed(type_id, size)
    }

    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        _root: *mut majit_ir::GcRef,
        needs_write_barrier: *mut bool,
    ) -> majit_ir::GcRef {
        unsafe { *needs_write_barrier = false };
        self.alloc_nursery_typed(type_id, size)
    }

    unsafe fn alloc_fast_nursery_collecting_typed_roots(
        &mut self,
        type_id: u32,
        size: usize,
        _roots: *mut majit_ir::GcRef,
        _root_count: usize,
        needs_write_barrier: *mut bool,
    ) -> majit_ir::GcRef {
        unsafe { *needs_write_barrier = false };
        self.alloc_nursery_typed(type_id, size)
    }

    fn alloc_varsize(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> majit_ir::GcRef {
        let Some(bytes) = item_size
            .checked_mul(length)
            .and_then(|n| base_size.checked_add(n))
        else {
            return majit_ir::GcRef(0);
        };
        // Untyped: header word stays 0, and the length word is not written.
        self.bump_payload(0, bytes)
    }

    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> majit_ir::GcRef {
        self.alloc_varsize(base_size, item_size, length)
    }

    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> majit_ir::GcRef {
        let Some(payload) = item_size
            .checked_mul(length)
            .and_then(|n| base_size.checked_add(n))
        else {
            return majit_ir::GcRef(0);
        };
        if (type_id as usize) >= self.types.len() {
            return majit_ir::GcRef(0);
        }
        let info = self.types.get(type_id);
        let registered_varsize = info.item_size != 0;
        let length_offset = info.length_offset;
        let obj = self.bump_typed(type_id, payload);
        // `malloc_varsize` writes the length when the registered shape is
        // varsize. A fixed type leaves its first word to the caller.
        if !obj.is_null() && registered_varsize {
            unsafe { *((obj.0 + length_offset) as *mut usize) = length };
        }
        obj
    }

    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> majit_ir::GcRef {
        self.bump_typed(type_id, size)
    }

    fn alloc_young_nonmoving_typed(&mut self, type_id: u32, size: usize) -> majit_ir::GcRef {
        self.bump_typed(type_id, size)
    }

    fn write_barrier(&mut self, _obj: majit_ir::GcRef) {}

    fn jit_remember_young_pointer(&mut self, _obj: majit_ir::GcRef) {}

    fn jit_remember_young_pointer_from_array(&mut self, _obj: majit_ir::GcRef) {}

    fn remember_young_pointer_from_array2(
        &mut self,
        _obj: majit_ir::GcRef,
        _index: usize,
        _card_page_shift: u32,
    ) {
    }

    fn collect_nursery(&mut self) {}

    fn collect_full(&mut self) {}

    /// Nothing moves, and [`Self::collect_nursery`] / [`Self::collect_full`]
    /// are no-ops, so no walker has to find live jitframes.
    /// `assembler.py` `_call_header_shadowstack` stays off when
    /// `gcrootmap` is missing.
    fn has_gcrootmap(&self) -> bool {
        false
    }

    fn nursery_free(&self) -> *mut u8 {
        unsafe { (*heap_ptr()).nursery_free.get() }
    }

    fn nursery_free_addr(&self) -> usize {
        unsafe { core::ptr::addr_of!((*heap_ptr()).nursery_free) as usize }
    }

    fn nursery_top(&self) -> *const u8 {
        unsafe { (*heap_ptr()).nursery_top.get() }
    }

    fn nursery_top_addr(&self) -> usize {
        unsafe { core::ptr::addr_of!((*heap_ptr()).nursery_top) as usize }
    }

    fn max_nursery_object_size(&self) -> usize {
        SEGMENT_BYTES
    }

    /// `gc.py GcLLDescr_framework.supports_guard_gc_type`.
    fn supports_guard_gc_type(&self) -> bool {
        true
    }

    fn register_type(&mut self, info: majit_gc::TypeInfo) -> u32 {
        self.types.register(info)
    }

    fn freeze_types(&mut self) {
        self.types.freeze_types();
    }

    fn assign_inheritance_ids_now(&mut self) {
        self.types.assign_inheritance_ids_now();
    }

    fn types_frozen(&self) -> bool {
        self.types.is_frozen()
    }

    fn has_type_registry(&self) -> bool {
        true
    }

    fn set_jitframe_type_id(&mut self, id: u32) {
        assert!(
            (id as usize) < self.types.len(),
            "JITFRAME type id {id} is not registered on this collector"
        );
        self.jitframe_type_id = Some(id);
    }

    fn jitframe_type_id(&self) -> Option<u32> {
        self.jitframe_type_id
    }

    fn type_count(&self) -> usize {
        self.types.len()
    }

    fn type_size(&self, type_id: u32) -> Option<usize> {
        if (type_id as usize) < self.types.len() {
            Some(self.types.get(type_id).size)
        } else {
            None
        }
    }

    fn varsize_layout(&self, obj: majit_ir::GcRef) -> Option<majit_gc::GcVarSizeLayout> {
        let type_id = self.get_actual_typeid(obj)?;
        if type_id as usize >= self.types.len() {
            return None;
        }
        let info = self.types.get(type_id);
        (info.item_size != 0).then_some(majit_gc::GcVarSizeLayout {
            base_size: info.size,
            item_size: info.item_size,
            items_have_gc_ptrs: info.items_have_gc_ptrs,
        })
    }

    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        super::registration::type_id_for_classptr(classptr)
    }

    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        let table = self.types.type_info_table();
        (
            table.as_ptr() as usize,
            majit_gc::trace::TypeEntry::SHIFT_BY,
            majit_gc::trace::TypeInfoLayout::SIZE_OF_TI,
        )
    }

    /// `gc.py _setup_guard_is_object` then
    /// `get_translated_info_for_guard_is_object`: the infobits byte that
    /// holds `T_IS_RPYTHON_INSTANCE`.
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        let infobits_offset = majit_gc::trace::TypeInfoLayout::INFOBITS_OFFSET;
        let mask = majit_gc::trace::TypeInfoLayout::T_IS_RPYTHON_INSTANCE.to_le_bytes();
        let mut plus = 0usize;
        while plus < mask.len() && mask[plus] == 0 {
            plus += 1;
        }
        (infobits_offset + plus, mask[plus])
    }

    fn check_is_object(&self, gcref: majit_ir::GcRef) -> bool {
        if gcref.is_null() {
            return false;
        }
        let Some(typeid) = self.get_actual_typeid(gcref) else {
            return false;
        };
        let (base_type_info, shift_by, _sizeof_ti) = self.get_translated_info_for_typeinfo();
        let (infobits_offset, is_object_flag) = self.get_translated_info_for_guard_is_object();
        let typeid = typeid as usize;
        if typeid >= self.types.len() {
            return false;
        }
        let p = base_type_info + (typeid << shift_by) + infobits_offset;
        let byte = unsafe { *(p as *const u8) };
        (byte & is_object_flag) != 0
    }

    fn get_actual_typeid(&self, gcref: majit_ir::GcRef) -> Option<u32> {
        if gcref.is_null() {
            return None;
        }
        let header_addr = gcref.0.wrapping_sub(majit_gc::header::GcHeader::SIZE);
        let header = unsafe { *(header_addr as *const majit_gc::header::GcHeader) };
        Some(header.type_id())
    }

    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        Some(self.types.get(typeid).is_object)
    }

    fn subclassrange_min_offset(&self) -> usize {
        core::mem::offset_of!(super::object::CelClass, subclassrange_min)
    }

    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        let cls = super::registration::class_at_ptr(classptr)?;
        Some((cls.subclassrange_min, cls.subclassrange_max))
    }

    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        super::registration::class_subclass_range(typeid)
    }
}

/// Nursery bump captured at outermost entry. Leave compares [`Self::used`]
/// with the live bump; a match means this evaluation allocated nothing young.
/// `used` is `open_used`, which stays at the last rewind: bumps move
/// `nursery_free` only, and [`CelHeap::flush_nursery_bump`] either feeds a
/// rewind that stores this base back or opens a segment (`n_segs` changes).
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

#[inline(always)]
fn to_public(v: crate::Value) -> crate::Value {
    match v {
        crate::Value::Interned(w) => {
            // `Interned` owns nothing, but the matched value is still live
            // here and its drop is an out-of-line glue call over every variant.
            core::mem::forget(v);
            if let Some(scalar) = crate::runtime::convert::interned_immediate(w) {
                scalar
            } else {
                crate::runtime::convert::interned_to_public(w)
            }
        }
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

/// Bytes preceding an immortal payload. Same word as [`GC_HEADER_SIZE`]:
/// the type id, which `guard_is_object` reads at `obj - GC_HEADER_SIZE`.
/// A plain Rust `static` would put that load in rodata or unmapped memory.
pub const IMMORTAL_HEADER_SIZE: usize = GC_HEADER_SIZE;

/// Payloads handed out by [`alloc_immortal`]. The tripwire that a
/// prebuilt is not a Rust `static` asserts against this list.
static IMMORTAL_PAYLOADS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Live constant-pool spans owned by [`super::const_pool::ConstPool`].
/// `(payload, payload+len)` so [`is_immortal`] treats them like prebuilts
/// until the pool drops.
static CONST_SPANS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

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

/// Allocate `value` for process lifetime, with its type-id header in
/// front of the payload.
///
/// Pointer-free leaves only: a reference field written at construction
/// has no write barrier, and a major walk would then follow it into
/// freed memory. `W_OptionalObject` and anything holding a
/// [`super::object::CelRef`] stay on [`CelHeap::alloc`].
pub fn alloc_immortal<T: CelGcType>(value: T) -> *mut T {
    let () = AssertNoDrop::<T>::OK;
    let align = align_of::<T>().max(GC_HEADER_SIZE);
    let total = headered_total(size_of::<T>());
    let layout = Layout::from_size_align(total, align).expect("immortal layout is valid");
    // SAFETY: `total` covers the header word and `T`.
    let base = unsafe { std::alloc::alloc(layout) };
    if base.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: `base` is aligned for `u64` and owned uniquely here.
    unsafe {
        (base as *mut u64).write(u64::from(<T as CelGcType>::TYPE_ID));
    }
    let payload = unsafe { base.add(GC_HEADER_SIZE) as *mut T };
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
        .contains(&p)
}

/// The header word immediately before an immortal payload.
///
/// # Safety
///
/// `ptr` must have come from [`alloc_immortal`].
pub unsafe fn immortal_header(ptr: *const u8) -> u64 {
    unsafe { *(ptr.sub(GC_HEADER_SIZE) as *const u64) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_int(n: i64) -> crate::runtime::object::W_IntObject {
        crate::runtime::object::W_IntObject {
            ob_header: crate::runtime::object::CelObject {
                ob_type: &crate::runtime::object::CEL_INT_CLASS,
            },
            intval: n,
        }
    }

    /// `contains` is true only for addresses this heap handed out.
    #[test]
    fn contains_reports_this_heaps_payloads() {
        let heap = CelHeap::new();
        let a = heap.alloc(test_int(1)) as *const u8;
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
        let a = heap.alloc(test_int(1));
        let b = heap.alloc(test_int(2));
        assert_ne!(a, b);
        unsafe {
            assert_eq!((*a).intval, 1);
            assert_eq!((*b).intval, 2);
            assert_eq!(
                *((a as *const u8).sub(GC_HEADER_SIZE) as *const u64),
                u64::from(crate::runtime::object::W_IntObject::TYPE_ID)
            );
        }
        assert!(
            (a as usize).abs_diff(b as usize) >= size_of::<crate::runtime::object::W_IntObject>()
        );
    }

    /// The counter counts objects, not segments — a heap that served a
    /// thousand values out of one segment still reports a thousand.
    #[test]
    fn the_counter_counts_objects_not_segments() {
        let heap = CelHeap::new();
        for i in 0..1000 {
            heap.alloc(test_int(i));
        }
        assert_eq!(heap.allocated_objects(), 1000);
        assert_eq!(
            heap.allocated_bytes(),
            1000 * (GC_HEADER_SIZE + size_of::<crate::runtime::object::W_IntObject>()) as u64
        );
        assert_eq!(heap.segments(), 1, "1000 headered ints fit in one segment");
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
        let p = with_heap(|h| h.alloc(test_int(7)));
        let after = with_heap(|h| h.allocated_objects());
        assert_eq!(after, before + 1);
        unsafe { assert_eq!((*p).intval, 7) };
    }

    /// Open one nursery segment and rewind it, so later leaves take the
    /// same-segment path instead of the first-allocation reset.
    fn prime_open_nursery(heap: &CelHeap) {
        assert!(heap.enter());
        let _ = heap.alloc(test_int(0));
        heap.leave(true);
        assert_eq!(heap.nursery.n_segs.get(), 1);
        assert_eq!(heap.nursery.open_used.get(), 0);
        assert_eq!(heap.nursery.bytes.get(), 0);
    }

    /// Drive same-segment evaluations until the open segment is past half,
    /// which is the leave that rewinds to `base_used`.
    fn fill_past_half(heap: &CelHeap, base_used: usize) {
        let half = heap.nursery.open_cap.get() / 2;
        loop {
            assert!(heap.enter());
            let _ = heap.alloc(test_int(1));
            let over = heap.live_nursery_used() > half;
            heap.leave(true);
            assert_eq!(heap.nursery.n_segs.get(), 1);
            if over {
                assert_eq!(heap.nursery.open_used.get(), base_used);
                assert_eq!(heap.nursery.bytes.get(), 0);
                return;
            }
            assert!(heap.live_nursery_used() <= half);
            assert!(heap.nursery.bytes.get() > 0);
        }
    }

    /// An outermost scope reclaims nursery memory once the open segment is
    /// half used; a smaller leave leaves the bump. Old space is untouched.
    #[test]
    fn outermost_scope_resets_the_nursery() {
        let heap = CelHeap::new();
        let old = heap.alloc(test_int(1));
        prime_open_nursery(&heap);
        let base_bytes = heap.allocated_bytes();
        assert!(heap.enter());
        let young = heap.alloc(test_int(2));
        assert!(heap.contains(young as *const u8));
        assert!(heap.is_young(young as *const u8));
        assert!(!heap.is_young(old as *const u8));
        let bytes_during = heap.allocated_bytes();
        heap.leave(true);
        assert!(heap.contains(old as *const u8));
        assert!(heap.contains(young as *const u8));
        assert_eq!(heap.allocated_bytes(), bytes_during);
        assert_eq!(heap.nursery.n_segs.get(), 1);
        fill_past_half(&heap, 0);
        assert!(heap.contains(old as *const u8));
        assert!(!heap.contains(young as *const u8));
        assert_eq!(heap.allocated_bytes(), base_bytes);
        assert!(heap.nursery_high_water() <= heap.nursery.open_cap.get() as u64);
        unsafe { assert_eq!((*old).intval, 1) };
    }

    /// A young fixed-size allocation updates the live counters by one object
    /// and the headered size, including when many fit in the open segment.
    /// Counters drop on the leave that passes half the segment, not before.
    #[test]
    fn young_fixed_size_counts_stay_exact() {
        let heap = CelHeap::new();
        prime_open_nursery(&heap);
        let before_n = heap.allocated_objects();
        let before_b = heap.allocated_bytes();
        assert!(heap.enter());
        for i in 0..64 {
            heap.alloc(test_int(i));
        }
        assert_eq!(heap.nursery_free.get() as usize % 8, 0);
        assert_eq!(heap.allocated_objects(), before_n + 64);
        assert_eq!(
            heap.allocated_bytes(),
            before_b
                + 64 * (GC_HEADER_SIZE + size_of::<crate::runtime::object::W_IntObject>()) as u64
        );
        heap.leave(true);
        assert_eq!(heap.allocated_objects(), before_n + 64);
        assert_eq!(
            heap.allocated_bytes(),
            before_b
                + 64 * (GC_HEADER_SIZE + size_of::<crate::runtime::object::W_IntObject>()) as u64
        );
        assert_eq!(heap.nursery.n_segs.get(), 1);
        fill_past_half(&heap, 0);
        assert_eq!(heap.allocated_objects(), before_n);
        assert_eq!(heap.allocated_bytes(), before_b);
        assert_eq!(heap.nursery.n_segs.get(), 1);
    }

    /// Nested scopes do not reset. The outermost leave rewinds only once the
    /// open segment is half used.
    #[test]
    fn nested_scope_does_not_reset() {
        let heap = CelHeap::new();
        prime_open_nursery(&heap);
        assert!(heap.enter());
        let a = heap.alloc(test_int(1));
        assert!(!heap.enter());
        let b = heap.alloc(test_int(2));
        heap.leave(false);
        assert!(heap.contains(a as *const u8));
        assert!(heap.contains(b as *const u8));
        heap.leave(true);
        assert!(heap.contains(a as *const u8));
        assert!(heap.contains(b as *const u8));
        assert_eq!(heap.nursery.n_segs.get(), 1);
        fill_past_half(&heap, 0);
        assert!(!heap.contains(a as *const u8));
        assert!(!heap.contains(b as *const u8));
        assert_eq!(heap.nursery.n_segs.get(), 1);
    }

    /// A one-byte bump leaves the cursor off an 8-byte boundary. Leave rewinds
    /// it; the inline nursery bump does not realign.
    #[test]
    fn an_unaligned_cursor_rewinds_immediately() {
        let heap = CelHeap::new();
        prime_open_nursery(&heap);
        assert!(heap.enter());
        let _ = heap.alloc_raw(1, 1);
        assert_ne!(heap.live_nursery_used() % align_of::<u64>(), 0);
        heap.leave(true);
        assert_eq!(heap.nursery.open_used.get(), 0);
        assert_eq!(heap.nursery.bytes.get(), 0);
        assert_eq!(heap.live_nursery_used() % align_of::<u64>(), 0);
        assert_eq!(heap.nursery.n_segs.get(), 1);
    }

    /// Small outermost evaluations share one segment. The rewind waits until
    /// the open segment is half used, so the bump never opens a second one.
    #[test]
    fn many_small_evals_stay_inside_one_segment() {
        let heap = CelHeap::new();
        let cap = SEGMENT_BYTES as u64;
        for _ in 0..100_000 {
            let scope = enter_eval_on(&heap);
            let _ = heap.alloc(test_int(1));
            scope.finish(crate::Value::Int(0));
            assert_eq!(heap.nursery.n_segs.get(), 1);
        }
        assert_eq!(heap.nursery.n_segs.get(), 1);
        assert!(heap.nursery_high_water() <= cap);
        assert!(heap.nursery_high_water() <= heap.nursery.open_cap.get() as u64);
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
        let old = heap.alloc(test_int(1));
        let bytes = heap.allocated_bytes();
        assert!(heap.enter());
        heap.leave(true);
        assert_eq!(heap.allocated_bytes(), bytes);
        assert!(heap.contains(old as *const u8));
        unsafe { assert_eq!((*old).intval, 1) };
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

    /// `GcHeader::new` stores the type id in the low bits and nothing else.
    #[cfg(feature = "jit")]
    #[test]
    fn gc_header_word_is_the_type_id() {
        let id = crate::runtime::object::W_IntObject::TYPE_ID;
        let header = majit_gc::header::GcHeader::new(id);
        assert_eq!(header.tid_and_flags, u64::from(id));
        assert_eq!(header.type_id(), id);
        assert_eq!(GC_HEADER_SIZE, majit_gc::header::GcHeader::SIZE);
    }
}
