//! Memory a [`crate::vm::CelCode`] owns for interned constants.
//!
//! Leaves here are immutable after compile, treated like immortal prebuilts
//! by [`super::heap::is_immortal`], and freed only when the last code object
//! sharing this pool drops.

use super::heap::{headered_total, register_const_span, unregister_const_span, GC_HEADER_SIZE};
use super::lltype::CelGcType;
use super::object::{
    mapdict_layout_for_names, new_bool, new_null, prebuilt_int, prepare_mapdict_rows, CelObject,
    CelRef, ListStrategy, MapStrategy, W_BytesObject, W_DoubleObject, W_HostListObject,
    W_IntColumn, W_IntObject, W_ListObject, W_MapObject, W_StringObject, W_TupleObject,
    W_UIntObject, CEL_BYTES_CLASS, CEL_DOUBLE_CLASS, CEL_HOST_LIST_CLASS, CEL_INT_CLASS,
    CEL_INT_COLUMN_CLASS, CEL_MAP_CLASS, CEL_STRING_CLASS, CEL_TUPLE_CLASS, CEL_UINT_CLASS,
    MAPDICT_MAX_ENTRIES,
};
use super::object_array::{
    bytes_base, int_words_base, items_block_items_base, CelBytesBlock, CelIntWords, CelItemsBlock,
    CEL_BYTES_BLOCK_ITEMS_OFFSET, CEL_INT_WORDS_ITEMS_OFFSET, CEL_ITEMS_BLOCK_ITEMS_OFFSET,
};
use crate::objects::{Key, ListRef, Map, MapStorage};
use crate::Value;
use core::alloc::Layout;
use core::mem::{align_of, size_of, size_of_val};
use std::sync::Arc;

/// Arena of interned constant leaves. Shared by clones of a [`crate::vm::CelCode`].
#[derive(Default)]
pub struct ConstPool {
    blocks: Vec<Block>,
    /// Keeps the `Arc` named by a string leaf's `public` word alive for the
    /// pool's lifetime. The leaf is immortal; the pointer is written once
    /// here, before the pool is shared.
    strings: Vec<Arc<str>>,
}

struct Block {
    base: *mut u8,
    payload: *mut u8,
    layout: Layout,
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
    fn alloc_raw_typed(&mut self, type_id: u32, size: usize, align: usize) -> *mut u8 {
        let align = align.max(GC_HEADER_SIZE);
        let total = headered_total(size);
        let layout = Layout::from_size_align(total, align).expect("const layout");
        let base = unsafe { std::alloc::alloc(layout) };
        if base.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        unsafe {
            (base as *mut u64).write(u64::from(type_id));
        }
        let payload = unsafe { base.add(GC_HEADER_SIZE) };
        register_const_span(payload, size);
        self.blocks.push(Block {
            base,
            payload,
            layout,
        });
        payload
    }

    fn alloc<T: CelGcType>(&mut self, value: T) -> *mut T {
        let ptr = self.alloc_raw_typed(T::TYPE_ID, size_of::<T>(), align_of::<T>()) as *mut T;
        unsafe { ptr.write(value) };
        ptr
    }

    fn alloc_bytes_block(&mut self, bytes: &[u8]) -> *mut CelBytesBlock {
        let cap = bytes.len();
        let size = CEL_BYTES_BLOCK_ITEMS_OFFSET
            .checked_add(cap)
            .expect("bytes block fits");
        let raw = self.alloc_raw_typed(CelBytesBlock::TYPE_ID, size, align_of::<CelBytesBlock>());
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
                let public = crate::objects::arc_str_thin(s);
                self.strings.push(Arc::clone(s));
                self.alloc(W_StringObject {
                    ob_header: CelObject {
                        ob_type: &CEL_STRING_CLASS,
                    },
                    chars,
                    byte_len: s.len() as i64,
                    public,
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

    /// `W_TupleObject` of `items`. The items are already interned leaves.
    /// Only `In` consumes the result; it is not a CEL value.
    pub(crate) fn intern_tuple(&mut self, items: &[CelRef]) -> CelRef {
        let n = items.len();
        let size = CEL_ITEMS_BLOCK_ITEMS_OFFSET
            .checked_add(n.saturating_mul(size_of::<CelRef>()))
            .expect("tuple items fit");
        let block = self.alloc_raw_typed(CelItemsBlock::TYPE_ID, size, align_of::<CelItemsBlock>())
            as *mut CelItemsBlock;
        unsafe {
            (*block).capacity = n;
            if n != 0 {
                core::ptr::copy_nonoverlapping(items.as_ptr(), items_block_items_base(block), n);
            }
        }
        self.alloc(W_TupleObject {
            ob_header: CelObject {
                ob_type: &CEL_TUPLE_CLASS,
            },
            length: n as i64,
            items: block,
        }) as CelRef
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
        if let Some(values) = list.ints_slice() {
            let start = list.window_start();
            self.intern_ints_list(&values[start..start + list.len()])
        } else if list.is_whole() {
            if let Some(items) = list.object_slice() {
                let mut refs = Vec::with_capacity(items.len());
                for v in items {
                    let w = self.intern_elem(v);
                    if w.is_null() {
                        return core::ptr::null_mut();
                    }
                    refs.push(w);
                }
                self.intern_object_list(&refs)
            } else {
                core::ptr::null_mut()
            }
        } else {
            core::ptr::null_mut()
        }
    }

    fn intern_ints_list(&mut self, ints: &[i64]) -> CelRef {
        let n = ints.len();
        let data = if n == 0 {
            core::ptr::null_mut()
        } else {
            let bytes = CEL_INT_WORDS_ITEMS_OFFSET + size_of_val(ints);
            let block = self.alloc_raw_typed(CelIntWords::TYPE_ID, bytes, align_of::<CelIntWords>())
                as *mut CelIntWords;
            unsafe {
                (*block).capacity = n;
                core::ptr::copy_nonoverlapping(ints.as_ptr(), int_words_base(block), n);
            }
            block
        };
        let col = self.alloc(W_IntColumn {
            ob_header: CelObject {
                ob_type: &CEL_INT_COLUMN_CLASS,
            },
            data,
            length: n as i64,
        }) as CelRef;
        self.alloc(W_HostListObject {
            base: W_ListObject {
                ob_header: CelObject {
                    ob_type: &CEL_HOST_LIST_CLASS,
                },
                strategy: ListStrategy::Ints,
                storage: col,
                items: core::ptr::null_mut(),
                start: 0,
                length: n as i64,
            },
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
        let block = self.alloc_raw_typed(CelItemsBlock::TYPE_ID, size, align_of::<CelItemsBlock>())
            as *mut CelItemsBlock;
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
            public_kind: 0,
            public_len: 0,
            layout: 0,
        }) as CelRef
    }

    fn intern_map(&mut self, map: &Map) -> CelRef {
        match map.storage() {
            MapStorage::Record { .. } => core::ptr::null_mut(),
            MapStorage::Object(_) | MapStorage::Entries(_) => self.intern_public_map(map),
        }
    }

    /// Mapdict when every key is a string and the map is small enough.
    /// A value that fails to intern returns null immediately.
    fn intern_public_map(&mut self, map: &Map) -> CelRef {
        if map.len() > MAPDICT_MAX_ENTRIES {
            return self.intern_object_from_public(map);
        }
        // One pass. A `HashMap` walks in a per-instance random order, and a
        // second pass is not the same order.
        let mut rows: Vec<(&str, CelRef)> = Vec::with_capacity(map.len());
        for (k, v) in map.iter() {
            let Key::String(s) = k else {
                return self.intern_object_from_public(map);
            };
            let value = self.intern_elem(v.as_ref());
            if value.is_null() {
                return core::ptr::null_mut();
            }
            rows.push((s.as_ref(), value));
        }
        // Key-byte order, so the same key set shares one layout
        // (`mapdict.py` `_get_new_attr`).
        if !prepare_mapdict_rows(&mut rows) {
            return self.object_map_from_string_rows(&rows);
        }
        let mut names = Vec::with_capacity(rows.len());
        let mut values = Vec::with_capacity(rows.len());
        for (name, value) in &rows {
            names.push(name.as_bytes());
            values.push(*value);
        }
        let layout = mapdict_layout_for_names(&names);
        self.alloc_mapdict(layout, &values)
    }

    fn intern_object_from_public(&mut self, map: &Map) -> CelRef {
        let mut pairs = Vec::with_capacity(map.len());
        for (k, v) in map.iter() {
            let key = self.intern_key(k);
            let value = self.intern_elem(v.as_ref());
            if key.is_null() || value.is_null() {
                return core::ptr::null_mut();
            }
            pairs.push((key, value));
        }
        self.intern_object_map(&pairs)
    }

    fn object_map_from_string_rows(&mut self, rows: &[(&str, CelRef)]) -> CelRef {
        let mut pairs = Vec::with_capacity(rows.len());
        for (name, value) in rows {
            let key = self.intern(&Value::String(Arc::from(*name)));
            if key.is_null() {
                return core::ptr::null_mut();
            }
            pairs.push((key, *value));
        }
        self.intern_object_map(&pairs)
    }

    fn alloc_mapdict(&mut self, layout: i64, values: &[CelRef]) -> CelRef {
        let n = values.len();
        let size = CEL_ITEMS_BLOCK_ITEMS_OFFSET
            .checked_add(n.saturating_mul(size_of::<CelRef>()))
            .expect("items block fits");
        let block = self.alloc_raw_typed(CelItemsBlock::TYPE_ID, size, align_of::<CelItemsBlock>())
            as *mut CelItemsBlock;
        unsafe {
            (*block).capacity = n;
            if n != 0 {
                let dest = items_block_items_base(block);
                let mut i = 0;
                while i < n {
                    *dest.add(i) = values[i];
                    i += 1;
                }
            }
        }
        self.alloc(W_MapObject {
            ob_header: CelObject {
                ob_type: &CEL_MAP_CLASS,
            },
            strategy: MapStrategy::Mapdict,
            storage: core::ptr::null_mut(),
            items: block,
            length: n as i64,
            public: core::ptr::null(),
            public_kind: 0,
            public_len: 0,
            layout,
        }) as CelRef
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
        let block = self.alloc_raw_typed(CelItemsBlock::TYPE_ID, size, align_of::<CelItemsBlock>())
            as *mut CelItemsBlock;
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
            public_kind: 0,
            public_len: 0,
            layout: 0,
        }) as CelRef
    }

    fn intern_object_list(&mut self, items: &[CelRef]) -> CelRef {
        let n = items.len();
        let size = CEL_ITEMS_BLOCK_ITEMS_OFFSET
            .checked_add(n.saturating_mul(size_of::<CelRef>()))
            .expect("items block fits");
        let block = self.alloc_raw_typed(CelItemsBlock::TYPE_ID, size, align_of::<CelItemsBlock>())
            as *mut CelItemsBlock;
        unsafe {
            (*block).capacity = n;
            if n != 0 {
                core::ptr::copy_nonoverlapping(items.as_ptr(), items_block_items_base(block), n);
            }
        }
        self.alloc(W_HostListObject {
            base: W_ListObject {
                ob_header: CelObject {
                    ob_type: &CEL_HOST_LIST_CLASS,
                },
                strategy: ListStrategy::Object,
                storage: core::ptr::null_mut(),
                items: block,
                start: 0,
                length: n as i64,
            },
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
