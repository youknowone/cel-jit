//! Memory a [`crate::vm::CelCode`] owns for interned constants.
//!
//! Leaves here are immutable after compile, treated like immortal prebuilts
//! by [`super::heap::is_immortal`], and freed only when the last code object
//! sharing this pool drops.

use super::heap::{register_const_span, unregister_const_span, IMMORTAL_HEADER_SIZE, IMMORTAL_MARK};
use super::object::{
    new_bool, new_null, prebuilt_int, CelObject, CelRef, ListStrategy, MapStrategy, W_BytesObject,
    W_DoubleObject, W_IntColumn, W_IntObject, W_ListObject, W_MapObject, W_StringObject,
    W_UIntObject, CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_INT_CLASS, CEL_INT_COLUMN_CLASS,
    CEL_LIST_CLASS, CEL_MAP_CLASS, CEL_STRING_CLASS, CEL_UINT_CLASS,
};
use super::object_array::{
    bytes_base, items_block_items_base, CelBytesBlock, CelItemsBlock, CEL_BYTES_BLOCK_ITEMS_OFFSET,
    CEL_ITEMS_BLOCK_ITEMS_OFFSET,
};
use crate::objects::{Key, ListRef, ListStorage, Map, MapStorage};
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
            Value::Int(i) if prebuilt_int(*i).is_some() => core::ptr::null_mut(),
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
            Value::List(list) => self.intern_list(list),
            Value::Map(map) => self.intern_map(map),
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

    /// A list element: pool intern, or the immortal prebuilt [`new_int`] /
    /// [`new_bool`] / [`new_null`] already answer. Never a thread-heap alloc.
    pub(crate) fn intern_elem(&mut self, v: &Value) -> CelRef {
        let w = self.intern(v);
        if !w.is_null() {
            return w;
        }
        match v {
            Value::Int(i) => prebuilt_int(*i)
                .map(|p| p as CelRef)
                .unwrap_or(core::ptr::null_mut()),
            Value::Bool(b) => new_bool(*b) as CelRef,
            Value::Null => new_null() as CelRef,
            _ => core::ptr::null_mut(),
        }
    }

    fn intern_list(&mut self, list: &ListRef) -> CelRef {
        match list.storage() {
            ListStorage::Ints(values) => {
                let start = list.window_start();
                self.intern_ints_list(&values[start..start + list.len()])
            }
            ListStorage::Object(items)
                if list.window_start() == 0 && list.len() == items.len() =>
            {
                let mut refs = Vec::with_capacity(items.len());
                for v in items {
                    let w = self.intern_elem(v);
                    if w.is_null() {
                        return core::ptr::null_mut();
                    }
                    refs.push(w);
                }
                self.intern_object_list(&refs)
            }
            _ => core::ptr::null_mut(),
        }
    }

    fn intern_ints_list(&mut self, ints: &[i64]) -> CelRef {
        let n = ints.len();
        let data = if n == 0 {
            core::ptr::null_mut()
        } else {
            let raw = self.alloc_raw(n * size_of::<i64>(), align_of::<i64>()) as *mut i64;
            unsafe { core::ptr::copy_nonoverlapping(ints.as_ptr(), raw, n) };
            raw
        };
        let col = self.alloc(W_IntColumn {
            ob_header: CelObject {
                ob_type: &CEL_INT_COLUMN_CLASS,
            },
            data,
            length: n as i64,
        }) as CelRef;
        self.alloc(W_ListObject {
            ob_header: CelObject {
                ob_type: &CEL_LIST_CLASS,
            },
            strategy: ListStrategy::Ints,
            storage: col,
            items: core::ptr::null_mut(),
            start: 0,
            length: n as i64,
            public: core::ptr::null(),
            public_start: 0,
            public_len: 0,
        }) as CelRef
    }

    /// An empty object map with room for `cap` [`map_try_insert`]s.
    /// Length starts at 0; the pool supplies the items block.
    pub(crate) fn alloc_empty_map(&mut self, cap: usize) -> CelRef {
        let nrefs = cap.saturating_mul(2);
        let size = CEL_ITEMS_BLOCK_ITEMS_OFFSET
            .checked_add(nrefs.saturating_mul(size_of::<CelRef>()))
            .expect("items block fits");
        let block = self.alloc_raw(size, align_of::<CelItemsBlock>()) as *mut CelItemsBlock;
        unsafe {
            (*block).capacity = nrefs;
        }
        self.alloc(W_MapObject {
            ob_header: CelObject {
                ob_type: &CEL_MAP_CLASS,
            },
            strategy: MapStrategy::Object,
            storage: core::ptr::null_mut(),
            items: block,
            length: 0,
            public: core::ptr::null(),
        }) as CelRef
    }

    fn intern_map(&mut self, map: &Map) -> CelRef {
        let MapStorage::Object(entries) = map.storage() else {
            return core::ptr::null_mut();
        };
        let mut pairs = Vec::with_capacity(entries.len());
        for (k, v) in entries.iter() {
            let key = self.intern_key(k);
            let value = self.intern_elem(v);
            if key.is_null() || value.is_null() {
                return core::ptr::null_mut();
            }
            pairs.push((key, value));
        }
        self.intern_object_map(&pairs)
    }

    fn intern_key(&mut self, key: &Key) -> CelRef {
        match key {
            Key::Int(i) => self.intern_elem(&Value::Int(*i)),
            Key::Uint(u) => self.intern(&Value::UInt(*u)),
            Key::Bool(b) => new_bool(*b) as CelRef,
            Key::String(s) => self.intern(&Value::String(s.clone())),
        }
    }

    fn intern_object_map(&mut self, pairs: &[(CelRef, CelRef)]) -> CelRef {
        let n = pairs.len();
        let nrefs = n.saturating_mul(2);
        let size = CEL_ITEMS_BLOCK_ITEMS_OFFSET
            .checked_add(nrefs.saturating_mul(size_of::<CelRef>()))
            .expect("items block fits");
        let block = self.alloc_raw(size, align_of::<CelItemsBlock>()) as *mut CelItemsBlock;
        unsafe {
            (*block).capacity = nrefs;
            if nrefs != 0 {
                let dest = items_block_items_base(block);
                let mut i = 0;
                while i < n {
                    *dest.add(2 * i) = pairs[i].0;
                    *dest.add(2 * i + 1) = pairs[i].1;
                    i += 1;
                }
            }
        }
        self.alloc(W_MapObject {
            ob_header: CelObject {
                ob_type: &CEL_MAP_CLASS,
            },
            strategy: MapStrategy::Object,
            storage: core::ptr::null_mut(),
            items: block,
            length: n as i64,
            public: core::ptr::null(),
        }) as CelRef
    }

    fn intern_object_list(&mut self, items: &[CelRef]) -> CelRef {
        let n = items.len();
        let size = CEL_ITEMS_BLOCK_ITEMS_OFFSET
            .checked_add(n.saturating_mul(size_of::<CelRef>()))
            .expect("items block fits");
        let block = self.alloc_raw(size, align_of::<CelItemsBlock>()) as *mut CelItemsBlock;
        unsafe {
            (*block).capacity = n;
            if n != 0 {
                core::ptr::copy_nonoverlapping(items.as_ptr(), items_block_items_base(block), n);
            }
        }
        self.alloc(W_ListObject {
            ob_header: CelObject {
                ob_type: &CEL_LIST_CLASS,
            },
            strategy: ListStrategy::Object,
            storage: core::ptr::null_mut(),
            items: block,
            start: 0,
            length: n as i64,
            public: core::ptr::null(),
            public_start: 0,
            public_len: 0,
        }) as CelRef
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
