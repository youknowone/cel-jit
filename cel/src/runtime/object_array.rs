//! Separately allocated payload blocks for the variable-length leaves.
//!
//! A CEL `string`, `bytes`, `list`, `map` or `struct` has a payload whose size is not known at
//! compile time. This module holds that payload; [`super::object`] holds the
//! fixed-size leaf that points at it.
//!
//! # Why the payload is not a tail on the leaf
//!
//! §3 of the design draws `W_BytesObject` and `W_ObjArray` as a header with a
//! varsize payload following it. That encoding is not available, for three
//! independent reasons, and the third is the one that would have been found
//! last:
//!
//! 1. **The fuse cannot see it.** `fuse_boxing_alloc` matches an allocation
//!    call taking exactly one argument — the finished value, by value. A type
//!    whose size depends on `n` cannot be passed by value, so a varsize leaf
//!    would decline the fuse and lose the whole reason the class family exists.
//! 2. **Upstream forbids it.** `lltype.py`'s `Array._note_inlined_into` raises
//!    on inlining a **GC** array into a structure at all, and otherwise only as
//!    the last field. RPython's own varsize list is a `GcStruct` holding
//!    `("length", Signed)` and `("items", Ptr(GcArray(ITEM)))` — a fixed
//!    wrapper and a separate array, which is exactly the shape below.
//! 3. **The codewriter cannot address it.** `OpKind::ArrayRead` / `ArrayWrite`
//!    / `ArrayLen` carry `nolength: bool`, which spells a length offset of
//!    either "absent" or "zero" and nothing else. A leaf carrying the class
//!    word at offset 0 needs its length word somewhere else, and no array op
//!    could then read it. Everything below that layer already generalises —
//!    `get_array_descr` takes a `length_offset: usize`, and `JitFrame`
//!    registers a nonzero one through `varsize_with_custom_trace` — so this is
//!    a today-limitation of one `bool`, not a design invariant. It is still a
//!    wall until that `bool` widens.
//!
//! # The shape
//!
//! One length word at offset 0, the items immediately after it. That is
//! RPython's `GcArray` body as the C backend emits it — `struct { Signed
//! length; ITEM items[]; }` — and pyre's `ItemsBlock` reproduces it.
//!
//! ⚠ The word at offset 0 is the allocated **capacity**, not the live length.
//! The live length lives on the owning leaf, as `("length", Signed)` does
//! upstream. A block is never resized in place; growing allocates a fresh one.
//!
//! # Registered, but nothing is allocated under the registration
//!
//! [`super::registration`] now registers both blocks as varsize types, from
//! [`CEL_ITEMS_BLOCK_TOKEN`] and [`CEL_BYTES_BLOCK_TOKEN`] unchanged — the
//! reference block with its items traced, the byte block as a leaf. So the
//! shape a collector would walk is on record and tested against a real minor
//! collection.
//!
//! The blocks are reserved on this thread's value heap with the leaves.
//! They still do not carry a collector type id on the allocated body; that
//! moves with the leaves.

use super::object::CelRef;

/// The three numbers describing an inline-varsize body.
///
/// One struct rather than three constants, because they are only ever correct
/// together: `get_array_token` upstream returns the triple, and pyre records a
/// live bug from picking them apart — two array type ids registered with a bare
/// length word while the blocks allocated under them had a padded one, which on
/// wasm32 sized every copy four bytes short.
pub struct ArrayToken {
    /// Offset of item 0 from the block's address.
    pub base_size: usize,
    pub item_size: usize,
    /// Offset of the length word from the block's address.
    pub len_offset: usize,
}

/// A block of managed references: `capacity`, then the items.
///
/// `[CelRef; 0]` rather than a slice or a `Vec`: the items are addressed by
/// offset from the block, and a fat pointer or an owning container would put a
/// second header between the length word and item 0.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct CelItemsBlock {
    /// Allocated capacity, the length word an array descr reads at offset 0.
    /// Fixed for the block's lifetime.
    pub capacity: usize,
    items: [CelRef; 0],
}

pub const CEL_ITEMS_BLOCK_ITEMS_OFFSET: usize = core::mem::offset_of!(CelItemsBlock, items);
pub const CEL_ITEMS_BLOCK_LEN_OFFSET: usize = core::mem::offset_of!(CelItemsBlock, capacity);

/// The length word must be at offset 0: `nolength: bool` in the codewriter can
/// spell no other position, so a block whose capacity moved would be
/// unaddressable rather than merely differently addressed.
const _: () = {
    assert!(CEL_ITEMS_BLOCK_LEN_OFFSET == 0);
};

pub const CEL_ITEMS_BLOCK_TOKEN: ArrayToken = ArrayToken {
    base_size: CEL_ITEMS_BLOCK_ITEMS_OFFSET,
    item_size: core::mem::size_of::<CelRef>(),
    len_offset: CEL_ITEMS_BLOCK_LEN_OFFSET,
};

/// A block of bytes: `capacity`, then the bytes.
///
/// Its own type and its own token rather than a generic one over the item type.
/// The item size is the whole difference between the two blocks, and the token
/// exists precisely so that size travels with the offsets that depend on it.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct CelBytesBlock {
    pub capacity: usize,
    items: [u8; 0],
}

pub const CEL_BYTES_BLOCK_ITEMS_OFFSET: usize = core::mem::offset_of!(CelBytesBlock, items);
pub const CEL_BYTES_BLOCK_LEN_OFFSET: usize = core::mem::offset_of!(CelBytesBlock, capacity);

const _: () = {
    assert!(CEL_BYTES_BLOCK_LEN_OFFSET == 0);
};

pub const CEL_BYTES_BLOCK_TOKEN: ArrayToken = ArrayToken {
    base_size: CEL_BYTES_BLOCK_ITEMS_OFFSET,
    item_size: 1,
    len_offset: CEL_BYTES_BLOCK_LEN_OFFSET,
};

/// Allocate a block and write its length word.
///
/// Deliberately NOT in [`super::lltype`] and deliberately not named
/// `malloc_typed`: the fuse matcher keys on that module path and on a
/// single-argument call, so a size-parameterised allocation living there would
/// be a call the matcher inspects and silently declines. Here it cannot be
/// mistaken for one.
///
/// Reserved on this thread's value heap.
///
/// # Safety
///
/// The caller writes `cap` items before anything reads them.
unsafe fn alloc_block_in(
    heap: &super::heap::CelHeap,
    base: usize,
    item_size: usize,
    align: usize,
    cap: usize,
) -> *mut u8 {
    let size = base + item_size * cap;
    let raw = heap.alloc_raw(size, align);
    // The length word first, so a block is never observable without one.
    unsafe { (raw as *mut usize).write(cap) };
    raw
}

unsafe fn alloc_block(base: usize, item_size: usize, align: usize, cap: usize) -> *mut u8 {
    super::heap::with_heap(|h| alloc_block_in(h, base, item_size, align, cap))
}

/// A block holding `values` on `heap`.
pub fn new_items_block_in(heap: &super::heap::CelHeap, values: &[CelRef]) -> *mut CelItemsBlock {
    let block = unsafe {
        alloc_block_in(
            heap,
            CEL_ITEMS_BLOCK_TOKEN.base_size,
            CEL_ITEMS_BLOCK_TOKEN.item_size,
            core::mem::align_of::<CelItemsBlock>(),
            values.len(),
        )
    } as *mut CelItemsBlock;
    // An index loop rather than `values.iter().enumerate()`, which is the
    // spelling RPython's own array copies use.
    //
    // ✅ **The store below lowers. The loop is what keeps this function out of
    // `new_list`'s closure.** Those are separate facts and conflating them cost
    // this comment a wrong answer once already. Seeded at `new_list`, the census
    // emits ONE jitcode and leaves this one a `residual_call_r_r` — which reads
    // like the store failing, and is not. Seeded AT this function instead
    // (`cel_census_pipeline_runtime_new_items_block`) it emits three jitcodes
    // and the vocabulary carries `setarrayitem_gc_r`: one `arraywrite`, one
    // `arrayread`, two `arraylen`, which is exactly this loop body. The
    // reference array store works.
    //
    // What refuses the function is `JitPolicy::look_inside_graph`'s
    // `contains_loop`: a graph carrying a backedge is not admitted without an
    // `unroll_safe` hint, so `find_all_graphs_bfs` never puts it in
    // `candidate_graphs`, `graphs_from` answers `None`, and `guess_call_kind`
    // returns `Residual`. Measured rather than inferred — in the ULLBC this
    // function carries a backedge while `new_bytes_block`, `alloc_block`,
    // `bytes_base` and `new_list` carry none, and those four are exactly the
    // ones that lower. The portal-seeded census exists because a portal graph is
    // admitted WITHOUT that check, which is the only way to see this body at
    // all.
    //
    // ⛔ An earlier revision of this comment claimed the opposite — "the reason
    // is the store, not the loop" — and cited a re-census after rewriting
    // `values.iter().enumerate()` into this index loop as having refuted the
    // loop hypothesis. **Both spellings carry a backedge.** That experiment
    // could only ever compare spellings within the loopy class, so it was
    // vacuous for the question it was cited as settling. A control that cannot
    // produce the outcome it is meant to detect is not evidence for its absence.
    //
    // The store reaches `setarrayitem_gc_r` through the front end's
    // `is_list_items_elem_ptr_add_parts`. It has FIVE conditions and the store
    // below satisfies all five, because four of them are invisible in the diff:
    //
    // 1. The callee is `<*mut T>::add` — hence `.add(i)` and not an
    //    `items[i]`-style index.
    // 2. The receiver points at a managed reference: `*mut CelRef` is
    //    `*mut *mut CelObject`, and `is_object_ref_items_ptr` reads the pointee
    //    class root off the type. It used to compare that root against the
    //    literal `"PyObject"`, which no non-pyre host could ever answer;
    //    majit-translate now derives it.
    // 3. The index is a runtime local, which `i` is.
    // 4. The base traces to a recognised items-base accessor — which is why
    //    [`items_block_items_base`] carries that name and not a shorter one.
    //    See its own doc: the name is the contract, and it is load-bearing at
    //    BOTH ends.
    // 5. ⚠ The `.add` result is dereferenced exactly once and never escapes
    //    (`add_dest_used_only_as_single_deref`). **This is why the store is
    //    `*base.add(i) = ..` and not `base.add(i).write(..)`.** `.write()` is a
    //    method call taking the pointer BY VALUE, so the guard counts it as an
    //    escape — one `other` use is enough to refuse — and the whole route
    //    declines however the other four conditions land. A place assignment is
    //    a deref the guard can see. The two spellings are identical Rust and
    //    are not identical to the front end.
    //
    // Registration is necessary and not sufficient here, and so is every
    // condition above: the front end decides whether this is an array store
    // before any descr or type id is consulted.
    unsafe {
        let base = items_block_items_base(block);
        let mut i = 0;
        while i < values.len() {
            *base.add(i) = values[i];
            i += 1;
        }
    }
    block
}

/// A block holding `values` on this thread's heap.
pub fn new_items_block(values: &[CelRef]) -> *mut CelItemsBlock {
    super::heap::with_heap(|h| new_items_block_in(h, values))
}

/// An items block of `cap` null slots on `heap`.
///
/// The slots are written in place so a caller does not need a side `Vec` of
/// nulls — that `Vec` was one global-allocator call per evaluation.
pub fn new_items_block_zeroed_in(heap: &super::heap::CelHeap, cap: usize) -> *mut CelItemsBlock {
    let block = unsafe {
        alloc_block_in(
            heap,
            CEL_ITEMS_BLOCK_TOKEN.base_size,
            CEL_ITEMS_BLOCK_TOKEN.item_size,
            core::mem::align_of::<CelItemsBlock>(),
            cap,
        )
    } as *mut CelItemsBlock;
    unsafe {
        let base = items_block_items_base(block);
        let mut i = 0;
        while i < cap {
            *base.add(i) = core::ptr::null_mut();
            i += 1;
        }
    }
    block
}

/// An items block of `cap` null slots on this thread's heap.
pub fn new_items_block_zeroed(cap: usize) -> *mut CelItemsBlock {
    super::heap::with_heap(|h| new_items_block_zeroed_in(h, cap))
}

/// A block holding `bytes`.
pub fn new_bytes_block(bytes: &[u8]) -> *mut CelBytesBlock {
    let block = unsafe {
        alloc_block(
            CEL_BYTES_BLOCK_TOKEN.base_size,
            CEL_BYTES_BLOCK_TOKEN.item_size,
            core::mem::align_of::<CelBytesBlock>(),
            bytes.len(),
        )
    } as *mut CelBytesBlock;
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), bytes_base(block), bytes.len());
    }
    block
}

/// Item 0 of a reference block, or null for a null block.
///
/// ⚠ **The NAME is a contract with the front end, not a style choice, and it
/// is load-bearing at two ends that must agree.** `graph_is_items_block_base_accessor`
/// matches this function's own module-qualified name — `object_array::items_block_items_base`,
/// crate prefix ignored — and rewrites the `.add` below to return the block
/// HEADER, because the array descr that reads through it re-adds `base_size`.
/// `is_object_items_block_base_accessor` matches the same name at the other
/// end, where [`new_items_block`]'s element store is lowered.
///
/// If only the first end matched, this would return the header and the element
/// store would stay a raw `.add` striding from the LENGTH WORD — every item off
/// by one, silently. So the name is either right for both or wrong for both;
/// renaming it to something shorter re-opens exactly that gap. An earlier
/// revision of this function was called `items_base`, which matched neither.
///
/// An earlier revision of this doc also claimed the byte-offset add is "what
/// the front-end recognises as an array base". It is not — the recognition is
/// the name, and the shape only has to be the one the rewrite expects.
///
/// # Safety
///
/// `block` is null or points at a live block.
#[inline]
pub unsafe fn items_block_items_base(block: *mut CelItemsBlock) -> *mut CelRef {
    if block.is_null() {
        return core::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(CEL_ITEMS_BLOCK_ITEMS_OFFSET) as *mut CelRef }
}

/// Byte 0 of a byte block, or null for a null block.
///
/// # Safety
///
/// As [`items_block_items_base`].
#[inline]
pub unsafe fn bytes_base(block: *mut CelBytesBlock) -> *mut u8 {
    if block.is_null() {
        return core::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(CEL_BYTES_BLOCK_ITEMS_OFFSET) }
}

/// The capacity word, or 0 for a null block.
///
/// # Safety
///
/// As [`items_block_items_base`].
#[inline]
pub unsafe fn items_capacity(block: *mut CelItemsBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

/// # Safety
///
/// As [`items_block_items_base`].
#[inline]
pub unsafe fn bytes_capacity(block: *mut CelBytesBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::object::{new_int, CelRef};

    /// The token is the registration's input, so its three numbers are asserted
    /// against the layout rather than trusted to stay in step with it.
    #[test]
    fn the_tokens_describe_the_blocks_they_name() {
        assert_eq!(CEL_ITEMS_BLOCK_TOKEN.len_offset, 0);
        assert_eq!(CEL_ITEMS_BLOCK_TOKEN.base_size, size_of::<usize>());
        assert_eq!(CEL_ITEMS_BLOCK_TOKEN.item_size, size_of::<CelRef>());

        assert_eq!(CEL_BYTES_BLOCK_TOKEN.len_offset, 0);
        assert_eq!(CEL_BYTES_BLOCK_TOKEN.base_size, size_of::<usize>());
        assert_eq!(CEL_BYTES_BLOCK_TOKEN.item_size, 1);
    }

    #[test]
    fn an_items_block_round_trips_its_references() {
        unsafe {
            let values: Vec<CelRef> = (0..4).map(|i| new_int(i) as CelRef).collect();
            let block = new_items_block(&values);
            assert_eq!(items_capacity(block), 4);
            let base = items_block_items_base(block);
            for (i, v) in values.iter().enumerate() {
                assert_eq!(*base.add(i), *v);
            }
        }
    }

    #[test]
    fn a_bytes_block_round_trips_its_bytes() {
        unsafe {
            let block = new_bytes_block(b"hello");
            assert_eq!(bytes_capacity(block), 5);
            let base = bytes_base(block);
            assert_eq!(core::slice::from_raw_parts(base, 5), b"hello");
        }
    }

    /// An empty payload is a real block with a zero length word, not a null.
    /// A null would be indistinguishable from "no payload yet" at every reader.
    #[test]
    fn an_empty_block_is_allocated_and_reads_zero() {
        unsafe {
            let items = new_items_block(&[]);
            assert!(!items.is_null());
            assert_eq!(items_capacity(items), 0);

            let bytes = new_bytes_block(b"");
            assert!(!bytes.is_null());
            assert_eq!(bytes_capacity(bytes), 0);
        }
    }

    /// The accessors answer for a null block rather than dereferencing it: a
    /// leaf whose payload has not been built yet is a state the readers see.
    #[test]
    fn the_accessors_are_null_safe() {
        unsafe {
            assert!(items_block_items_base(core::ptr::null_mut()).is_null());
            assert!(bytes_base(core::ptr::null_mut()).is_null());
            assert_eq!(items_capacity(core::ptr::null_mut()), 0);
            assert_eq!(bytes_capacity(core::ptr::null_mut()), 0);
        }
    }
}
