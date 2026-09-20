//! Memory a [`crate::vm::CelCode`] owns for interned constants.
//!
//! Leaves here are immutable after compile, treated like immortal prebuilts
//! by [`super::heap::is_immortal`], and freed only when the last code object
//! sharing this pool drops.

use super::heap::{register_const_span, unregister_const_span, IMMORTAL_HEADER_SIZE, IMMORTAL_MARK};
use super::object::{
    CelObject, CelRef, W_BytesObject, W_DoubleObject, W_IntObject, W_StringObject, W_UIntObject,
    CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_INT_CLASS, CEL_STRING_CLASS, CEL_UINT_CLASS,
    PREBUILT_INT_FROM, PREBUILT_INT_TO,
};
use super::object_array::{
    bytes_base, CelBytesBlock, CEL_BYTES_BLOCK_ITEMS_OFFSET,
};
use crate::Value;
use core::alloc::Layout;
use core::mem::{align_of, size_of};
use std::sync::Arc;

/// Arena of interned constant leaves. Shared by clones of a [`crate::vm::CelCode`].
pub struct ConstPool {
    blocks: Vec<Block>,
}

struct Block {
    base: *mut u8,
    payload: *mut u8,
    layout: Layout,
}

impl Default for ConstPool {
    fn default() -> Self {
        ConstPool { blocks: Vec::new() }
    }
}

impl Drop for ConstPool {
    fn drop(&mut self) {
        for block in self.blocks.drain(..) {
            unregister_const_span(block.payload);
            unsafe { std::alloc::dealloc(block.base, block.layout) };
        }
    }
}

impl std::fmt::Debug for ConstPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstPool")
            .field("blocks", &self.blocks.len())
            .finish()
    }
}

impl ConstPool {
    fn alloc_raw(&mut self, size: usize, align: usize) -> *mut u8 {
        let align = align.max(align_of::<usize>());
        let header = IMMORTAL_HEADER_SIZE;
        let total = header
            .checked_add(size)
            .expect("const payload fits usize");
        let layout = Layout::from_size_align(total, align).expect("const layout");
        let base = unsafe { std::alloc::alloc(layout) };
        if base.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe {
            (base as *mut usize).write(IMMORTAL_MARK);
        }
        let payload = unsafe { base.add(header) };
        register_const_span(payload, size);
        self.blocks.push(Block {
            base,
            payload,
            layout,
        });
        payload
    }

    fn alloc<T>(&mut self, value: T) -> *mut T {
        let ptr = self.alloc_raw(size_of::<T>(), align_of::<T>()) as *mut T;
        unsafe { ptr.write(value) };
        ptr
    }

    fn alloc_bytes_block(&mut self, bytes: &[u8]) -> *mut CelBytesBlock {
        let cap = bytes.len();
        let size = CEL_BYTES_BLOCK_ITEMS_OFFSET
            .checked_add(cap)
            .expect("bytes block fits");
        let raw = self.alloc_raw(size, align_of::<CelBytesBlock>());
        let block = raw as *mut CelBytesBlock;
        unsafe {
            (*block).capacity = cap;
            let dest = bytes_base(block);
            if cap != 0 {
                core::ptr::copy_nonoverlapping(bytes.as_ptr(), dest, cap);
            }
        }
        block
    }

    /// Intern `v` into this pool. Small ints, bool and null return null so
    /// `LoadConst` clones the public scalar. Never allocates on a thread heap.
    pub fn intern(&mut self, v: &Value) -> CelRef {
        match v {
            Value::Interned(w) => *w,
            // Small ints, bool and null are already immortal prebuilts;
            // LoadConst clones the public scalar instead of pushing a pointer.
            Value::Int(i) if (*i >= PREBUILT_INT_FROM) && (*i < PREBUILT_INT_TO) => {
                core::ptr::null_mut()
            }
            Value::Int(i) => self.alloc(W_IntObject {
                ob_header: CelObject {
                    ob_type: &CEL_INT_CLASS,
                },
                intval: *i,
            }) as CelRef,
            Value::UInt(u) => self.alloc(W_UIntObject {
                ob_header: CelObject {
                    ob_type: &CEL_UINT_CLASS,
                },
                uintval: *u,
            }) as CelRef,
            Value::Float(f) => self.alloc(W_DoubleObject {
                ob_header: CelObject {
                    ob_type: &CEL_DOUBLE_CLASS,
                },
                floatval: *f,
            }) as CelRef,
            Value::Bool(_) | Value::Null => core::ptr::null_mut(),
            Value::String(s) => {
                let chars = self.alloc_bytes_block(s.as_bytes());
                self.alloc(W_StringObject {
                    ob_header: CelObject {
                        ob_type: &CEL_STRING_CLASS,
                    },
                    chars,
                    byte_len: s.len() as i64,
                    public: core::ptr::null(),
                }) as CelRef
            }
            Value::Bytes(b) => {
                let data = self.alloc_bytes_block(b);
                self.alloc(W_BytesObject {
                    ob_header: CelObject {
                        ob_type: &CEL_BYTES_CLASS,
                    },
                    data,
                    length: b.len() as i64,
                    public: core::ptr::null(),
                }) as CelRef
            }
            #[cfg(feature = "chrono")]
            Value::Duration(d) => d
                .num_nanoseconds()
                .map(|n| {
                    self.alloc(super::object::W_DurationObject {
                        ob_header: CelObject {
                            ob_type: &super::object::CEL_DURATION_CLASS,
                        },
                        nanos: n,
                    }) as CelRef
                })
                .unwrap_or(core::ptr::null_mut()),
            #[cfg(feature = "chrono")]
            Value::Timestamp(ts) => ts
                .timestamp_nanos_opt()
                .map(|n| {
                    self.alloc(super::object::W_TimestampObject {
                        ob_header: CelObject {
                            ob_type: &super::object::CEL_TIMESTAMP_CLASS,
                        },
                        nanos: n,
                        off_s: i64::from(ts.offset().local_minus_utc()),
                    }) as CelRef
                })
                .unwrap_or(core::ptr::null_mut()),
            _ => core::ptr::null_mut(),
        }
    }
}

/// Interned leaves plus the pool that owns them.
///
/// Immutable after compile; freed only in `Drop` of the last `Arc`. Clones
/// of a `CelCode` share the pool, so a Program compiled on one thread can
/// be executed on another after that thread's heap is gone.
#[derive(Clone, Debug)]
pub(crate) struct InternedConsts {
    /// Kept so the last `Arc` drop frees the pool.
    #[allow(dead_code)]
    pub pool: Arc<ConstPool>,
    pub leaves: Vec<CelRef>,
}

impl Default for InternedConsts {
    fn default() -> Self {
        InternedConsts {
            pool: Arc::new(ConstPool::default()),
            leaves: Vec::new(),
        }
    }
}

impl PartialEq for InternedConsts {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl InternedConsts {
    pub fn from_pool(pool: ConstPool, leaves: Vec<CelRef>) -> Self {
        InternedConsts {
            pool: Arc::new(pool),
            leaves,
        }
    }
}
