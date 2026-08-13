//! Separately allocated payload blocks for the variable-length leaves.
//!
//! A CEL `string`, `bytes` or `list` has a payload whose size is not known at
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
//! # Nothing traces these yet
//!
//! The items are managed edges and no collector walks them, for the same reason
//! nothing walks `W_OptionalObject`'s: the type ids and the offset lists arrive
//! with the registration mechanism. [`CEL_ITEMS_BLOCK_TOKEN`] is landed now
//! because it is precisely the value that registration will consume unchanged.

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
/// Leaks, like everything else this layer allocates. The tracing heap arrives
/// with the registration mechanism.
///
/// # Safety
///
/// The caller writes `cap` items before anything reads them.
unsafe fn alloc_block(base: usize, item_size: usize, align: usize, cap: usize) -> *mut u8 {
    let size = base + item_size * cap;
    let layout = std::alloc::Layout::from_size_align(size, align)
        .expect("payload block layout is representable");
    let raw = unsafe { std::alloc::alloc(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // The length word first, so a block is never observable without one.
    unsafe { (raw as *mut usize).write(cap) };
    raw
}

/// A block holding `values`.
pub fn new_items_block(values: &[CelRef]) -> *mut CelItemsBlock {
    let block = unsafe {
        alloc_block(
            CEL_ITEMS_BLOCK_TOKEN.base_size,
            CEL_ITEMS_BLOCK_TOKEN.item_size,
            core::mem::align_of::<CelItemsBlock>(),
            values.len(),
        )
    } as *mut CelItemsBlock;
    // An index loop, not `values.iter().enumerate()`. The iterator spelling is
    // what the first census of this function measured, and it cost the whole
    // graph: `new_bytes_block` next door lowered and became a jitcode while this
    // one declined and survived as a residual call, the only difference between
    // them being the adapter chain. RPython has no iterator adapters either —
    // its own array copies are index loops.
    unsafe {
        let base = items_base(block);
        let mut i = 0;
        while i < values.len() {
            base.add(i).write(values[i]);
            i += 1;
        }
    }
    block
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
/// The shape matters: a byte-offset add on the block pointer is what the
/// front-end recognises as an array base, lowering an access through it to an
/// `ArrayRead`/`ArrayWrite` rather than to opaque pointer arithmetic.
///
/// # Safety
///
/// `block` is null or points at a live block.
#[inline]
pub unsafe fn items_base(block: *mut CelItemsBlock) -> *mut CelRef {
    if block.is_null() {
        return core::ptr::null_mut();
    }
    unsafe { (block as *mut u8).add(CEL_ITEMS_BLOCK_ITEMS_OFFSET) as *mut CelRef }
}

/// Byte 0 of a byte block, or null for a null block.
///
/// # Safety
///
/// As [`items_base`].
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
/// As [`items_base`].
#[inline]
pub unsafe fn items_capacity(block: *mut CelItemsBlock) -> usize {
    if block.is_null() {
        return 0;
    }
    unsafe { (*block).capacity }
}

/// # Safety
///
/// As [`items_base`].
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
            let base = items_base(block);
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
            assert!(items_base(core::ptr::null_mut()).is_null());
            assert!(bytes_base(core::ptr::null_mut()).is_null());
            assert_eq!(items_capacity(core::ptr::null_mut()), 0);
            assert_eq!(bytes_capacity(core::ptr::null_mut()), 0);
        }
    }
}
