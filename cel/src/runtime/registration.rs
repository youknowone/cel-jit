//! Registering the class family with the collector and with the descr cache.
//!
//! [`super::object`] defines the classes; this module is what makes them known
//! to the machinery that has to recognise an instance at run time. Three
//! registrations, none of which the compiler reconciles with the other two:
//!
//! 1. **The collector.** One `TypeInfo` per class, in a fixed order, then
//!    `freeze_types()` — which is the only place `subclassrange_{min,max}` are
//!    assigned. Plus one `register_vtable_for_type` per class, which is how a
//!    class *address* becomes a type id: RPython derives it arithmetically from
//!    the type-info group base, majit keeps an explicit map, and
//!    `get_typeid_from_classptr_if_gcremovetypeptr` is the contract both spell.
//! 2. **The descr cache**, keyed on the same numeric type id. The two id
//!    namespaces are independent — `GcCache::alloc_type_id` mints one,
//!    `MiniMarkGC::register_type` mints the other — and a mismatch is a silent
//!    mis-trace, because the compiled code stamps `descr.type_id()` into the
//!    header and the collector reads that word back to index its own registry.
//!    [`publish_cel_descrs`] therefore takes the ids from step 1 rather than
//!    minting its own, and asserts the round trip.
//! 3. **The driver**: `set_gc_allocator` plus `set_new_via_gc`, or a compiled
//!    `New` allocates through the backend's `malloc` stub and the object never
//!    enters the traced heap. See [`cel_gc`].
//!
//! Besides the twelve fixed-size classes there are the two payload blocks,
//! registered as **varsize** types from the tokens [`super::object_array`]
//! carries — see [`varsize_block`].
//!
//! # What this does not do yet
//!
//! Nothing allocates a cel value through the collector today: [`super::lltype`]
//! leaks, and no evaluator has been moved onto the class family. So this module
//! registers a family nothing has yet asked the collector about. That is the
//! point of landing it separately — the recipe is verifiable on its own, by the
//! tests below, before anything depends on it being right.

use majit_gc::collector::{GcConfig, MiniMarkGC};
use majit_gc::{GcAllocator, TypeInfo};
use majit_ir::descr::{ArrayFlag, SimpleFieldDescrSpec};
use majit_ir::Type;

use super::object_array::{ArrayToken, CEL_BYTES_BLOCK_TOKEN, CEL_ITEMS_BLOCK_TOKEN};

use super::object::{
    CelClass, CelObject, W_BoolObject, W_BytesObject, W_DoubleObject, W_DurationObject,
    W_IntObject, W_ListObject, W_NullObject, W_OptionalObject, W_StringObject, W_TimestampObject,
    W_TypeObject, W_UIntObject, CEL_BOOL_CLASS, CEL_BYTES_CLASS, CEL_DOUBLE_CLASS,
    CEL_DURATION_CLASS, CEL_INT_CLASS, CEL_LIST_CLASS, CEL_NULL_CLASS, CEL_OPTIONAL_CLASS,
    CEL_STRING_CLASS, CEL_TIMESTAMP_CLASS, CEL_TYPE_CLASS, CEL_UINT_CLASS,
};

/// One payload field, as both registrations need to see it.
pub struct FieldLayout {
    /// The Rust field name. It is the descr cache key, so it has to be the
    /// name the translator will hash, not a display string.
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub ty: Type,
    /// `Type::Int` alone does not say whether the field is signed, and the
    /// descr flag distinguishes them (`FLAG_SIGNED` vs `FLAG_UNSIGNED`).
    pub signed: bool,
    pub immutable: bool,
}

/// One class, as both registrations need to see it.
pub struct ClassLayout {
    pub class: &'static CelClass,
    /// The instance size, excluding the collector's own header.
    pub size: usize,
    /// Byte offsets of the instance's **managed** edges — the fields the
    /// collector walks during a minor collection.
    ///
    /// Not simply "every pointer field". `ob_type` points at a `'static`
    /// class, `W_TypeObject::cls` points at a `'static` class, and
    /// the three payload-block pointers address blocks that
    /// [`super::object_array`] allocates outside the traced heap. Listing any
    /// of them would hand the collector an address to trace that no collection
    /// owns.
    pub gc_ptr_offsets: &'static [usize],
    pub fields: &'static [FieldLayout],
}

/// The scalar payload of a leaf built by `scalar_leaf!`, which is always one
/// field immediately after the header.
const fn scalar(name: &'static str, size: usize, ty: Type, signed: bool) -> FieldLayout {
    FieldLayout {
        name,
        offset: size_of::<CelObject>(),
        size,
        ty,
        signed,
        immutable: true,
    }
}

const WORD: usize = size_of::<i64>();
const HEADER: usize = size_of::<CelObject>();

/// Every class, in the order the registration walks them.
///
/// The order is load-bearing twice over: `register_type` returns ids by
/// position, and `freeze_types` assigns `subclassrange_{min,max}` from a
/// preorder over the registration graph. Reordering this table renumbers both,
/// which is fine as long as it happens in one place — and this is that place.
pub static CEL_CLASS_LAYOUTS: &[ClassLayout] = &[
    ClassLayout {
        class: &CEL_NULL_CLASS,
        size: size_of::<W_NullObject>(),
        gc_ptr_offsets: &[],
        fields: &[],
    },
    ClassLayout {
        class: &CEL_BOOL_CLASS,
        size: size_of::<W_BoolObject>(),
        gc_ptr_offsets: &[],
        fields: &[scalar("boolval", WORD, Type::Int, true)],
    },
    ClassLayout {
        class: &CEL_INT_CLASS,
        size: size_of::<W_IntObject>(),
        gc_ptr_offsets: &[],
        fields: &[scalar("intval", WORD, Type::Int, true)],
    },
    ClassLayout {
        class: &CEL_UINT_CLASS,
        size: size_of::<W_UIntObject>(),
        gc_ptr_offsets: &[],
        fields: &[scalar("uintval", WORD, Type::Int, false)],
    },
    ClassLayout {
        class: &CEL_DOUBLE_CLASS,
        size: size_of::<W_DoubleObject>(),
        gc_ptr_offsets: &[],
        fields: &[scalar("floatval", WORD, Type::Float, true)],
    },
    ClassLayout {
        class: &CEL_DURATION_CLASS,
        size: size_of::<W_DurationObject>(),
        gc_ptr_offsets: &[],
        fields: &[scalar("nanos", WORD, Type::Int, true)],
    },
    ClassLayout {
        class: &CEL_TIMESTAMP_CLASS,
        size: size_of::<W_TimestampObject>(),
        gc_ptr_offsets: &[],
        fields: &[
            scalar("nanos", WORD, Type::Int, true),
            FieldLayout {
                name: "offset_seconds",
                offset: HEADER + WORD,
                size: WORD,
                ty: Type::Int,
                signed: true,
                immutable: true,
            },
        ],
    },
    ClassLayout {
        class: &CEL_TYPE_CLASS,
        size: size_of::<W_TypeObject>(),
        // `cls` is a `'static CelClass`, not an instance.
        gc_ptr_offsets: &[],
        fields: &[scalar("cls", WORD, Type::Int, false)],
    },
    ClassLayout {
        class: &CEL_OPTIONAL_CLASS,
        size: size_of::<W_OptionalObject>(),
        // The family's only managed edge today.
        gc_ptr_offsets: &[HEADER],
        fields: &[scalar("w_value", WORD, Type::Ref, false)],
    },
    ClassLayout {
        class: &CEL_BYTES_CLASS,
        size: size_of::<W_BytesObject>(),
        gc_ptr_offsets: &[],
        fields: &[
            scalar("data", WORD, Type::Int, false),
            FieldLayout {
                name: "length",
                offset: HEADER + WORD,
                size: WORD,
                ty: Type::Int,
                signed: true,
                immutable: true,
            },
        ],
    },
    ClassLayout {
        class: &CEL_STRING_CLASS,
        size: size_of::<W_StringObject>(),
        gc_ptr_offsets: &[],
        fields: &[
            scalar("chars", WORD, Type::Int, false),
            FieldLayout {
                name: "byte_len",
                offset: HEADER + WORD,
                size: WORD,
                ty: Type::Int,
                signed: true,
                immutable: true,
            },
        ],
    },
    ClassLayout {
        class: &CEL_LIST_CLASS,
        size: size_of::<W_ListObject>(),
        gc_ptr_offsets: &[],
        fields: &[
            scalar("items", WORD, Type::Int, false),
            FieldLayout {
                name: "length",
                offset: HEADER + WORD,
                size: WORD,
                ty: Type::Int,
                signed: true,
                immutable: true,
            },
        ],
    },
];

/// What one class got registered as.
pub struct RegisteredClass {
    pub class: &'static CelClass,
    pub type_id: u32,
    pub size: usize,
}

impl RegisteredClass {
    /// The class's address, which is what a `NewWithVtable` carries and what
    /// `get_typeid_from_classptr_if_gcremovetypeptr` is keyed on.
    pub fn vtable(&self) -> usize {
        self.class as *const CelClass as usize
    }
}

/// The result of [`register_cel_classes`].
pub struct CelTypeIds {
    /// The base every leaf inherits from, so that a `GuardSubclass` against
    /// "any cel value" has a class to name. It has no instances of its own.
    pub root: u32,
    pub classes: Vec<RegisteredClass>,
    /// The reference payload block, `CelItemsBlock`.
    ///
    /// Not in `classes` and not reachable through [`Self::type_id_of`]: a block
    /// carries its length word where a leaf carries its class word, so no class
    /// address maps to it and `get_typeid_from_classptr_if_gcremovetypeptr`
    /// cannot answer for it. It is named on its own here because the block is
    /// not the value — the leaf that points at it is.
    pub items_block: u32,
    /// The byte payload block, `CelBytesBlock`. As [`Self::items_block`], with
    /// leaf items.
    pub bytes_block: u32,
}

impl CelTypeIds {
    pub fn type_id_of(&self, class: &'static CelClass) -> Option<u32> {
        let addr = class as *const CelClass as usize;
        self.classes
            .iter()
            .find(|r| r.vtable() == addr)
            .map(|r| r.type_id)
    }
}

/// The `TypeInfo` for one payload block, from the block's own token.
///
/// The three numbers come off the [`ArrayToken`] rather than being spelled
/// here, which is the whole reason that struct exists: `encode_type_shape`
/// reads `base_size` as `ofstovar`, `item_size` as `varitemsize` and
/// `len_offset` as `ofstolength`, and picking them apart per call site is
/// exactly how pyre registered two array type ids whose blocks were laid out
/// four bytes apart from what the collector then copied.
///
/// `items_have_gc_ptrs` is the only difference between the two blocks and is
/// not derivable from the token: it says whether the items are managed edges
/// (`T_IS_GCARRAY_OF_GCPTR`, so a minor collection walks every slot) or plain
/// bytes. The token carries the item *size*; nothing in it carries the item's
/// kind.
///
/// No `gc_ptr_offsets`. A block's fixed part is the length word alone, which is
/// an integer — every managed edge a block has is an item.
fn varsize_block(token: &ArrayToken, items_have_gc_ptrs: bool) -> TypeInfo {
    TypeInfo::varsize(
        token.base_size,
        token.item_size,
        token.len_offset,
        items_have_gc_ptrs,
        Vec::new(),
    )
}

/// Register every class and both payload blocks with `gc`, then freeze the
/// registry.
///
/// The blocks come after the classes so that adding them renumbers nothing:
/// `register_type` hands out ids by position, and [`publish_cel_descrs`] zips
/// `CEL_CLASS_LAYOUTS` against `classes` by position too. They take no
/// `register_vtable_for_type` — a block has a length word at offset 0, not a
/// class word, so an address-to-id lookup over blocks would be reading a
/// capacity as a class pointer.
///
/// Frozen on the way out, deliberately. `freeze_types` is what assigns
/// `subclassrange_{min,max}`, and a caller that forgot it would get a family
/// whose `GuardSubclass` ranges are all `0..0` — which does not fail, it just
/// answers "no" to every subclass test. The blocks are unaffected either way:
/// `assign_inheritance_ids` walks only the types that declare a subclass range,
/// and a varsize type declares none.
fn register_cel_classes_unfrozen(gc: &mut MiniMarkGC) -> CelTypeIds {
    let root = gc.register_type(TypeInfo::object(size_of::<CelObject>()));
    let mut classes = Vec::with_capacity(CEL_CLASS_LAYOUTS.len());
    for layout in CEL_CLASS_LAYOUTS {
        let info = if layout.gc_ptr_offsets.is_empty() {
            TypeInfo::object_subclass(layout.size, root)
        } else {
            TypeInfo::object_subclass_with_gc_ptrs(
                layout.size,
                root,
                layout.gc_ptr_offsets.to_vec(),
            )
        };
        let type_id = gc.register_type(info);
        let registered = RegisteredClass {
            class: layout.class,
            type_id,
            size: layout.size,
        };
        GcAllocator::register_vtable_for_type(gc, registered.vtable(), type_id);
        classes.push(registered);
    }
    // The reference block's items ARE managed edges; the byte block's are
    // bytes. Two type ids rather than one for two blocks that agree on
    // `base_size` and `len_offset`, because one id carries one varsize shape
    // and these two disagree on the item size and on whether the items are
    // traced at all.
    let items_block = gc.register_type(varsize_block(&CEL_ITEMS_BLOCK_TOKEN, true));
    let bytes_block = gc.register_type(varsize_block(&CEL_BYTES_BLOCK_TOKEN, false));
    CelTypeIds {
        root,
        classes,
        items_block,
        bytes_block,
    }
}

pub fn register_cel_classes(gc: &mut MiniMarkGC) -> CelTypeIds {
    let ids = register_cel_classes_unfrozen(gc);
    GcAllocator::freeze_types(gc);
    ids
}

/// Publish a `SizeDescr` and its `FieldDescr`s for every class, keyed on the
/// type ids `register_cel_classes` handed out.
///
/// Dual-published under the bare struct name and under the crate-stripped def
/// path, as pyre does: the analyzer hashes the use-site bare identifier today
/// and the qualified form is where it is going, and one `Arc` reachable from
/// both slots is cheaper than discovering later that only one was filled.
pub fn publish_cel_descrs(ids: &CelTypeIds) {
    for (index, (layout, registered)) in CEL_CLASS_LAYOUTS.iter().zip(&ids.classes).enumerate() {
        let simple_name = leaf_struct_name(layout.class);
        let def_path = format!("runtime::object::{simple_name}");
        let specs: Vec<SimpleFieldDescrSpec> = layout
            .fields
            .iter()
            .enumerate()
            .map(|(index_in_parent, field)| SimpleFieldDescrSpec {
                // `Some(false)` DECLARES that this field is not the class
                // word; it does not decline to answer. cel's `CelObject` header
                // is `ob_type` alone, and this code is reading the layout, so it
                // can say so outright. `None` would hand the answer to the
                // fallback that infers it from the display name built three
                // lines below — and that name is `"{simple_name}.{field}"`, so a
                // field ever spelled `w_class` would be inferred TRUE. cel spells
                // its type-carrying fields `w_type` today, which is naming luck
                // and not a property a declaration has to depend on.
                is_class_word: Some(false),
                index: index_in_parent as u32,
                field_key: field.name.to_string(),
                name: format!("{simple_name}.{}", field.name),
                offset: field.offset,
                field_size: field.size,
                field_type: field.ty,
                is_immutable: field.immutable,
                is_quasi_immutable: false,
                flag: array_flag(field.ty, field.signed),
                virtualizable: false,
                index_in_parent,
            })
            .collect();

        let group = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
            index as u32,
            layout.size,
            registered.type_id,
            majit_ir::descr::path_hash(&def_path),
            registered.vtable(),
            true,
            false,
            &specs,
            // No extra GC edge. pyre's header carries a `w_class` that is a
            // managed `PyObject` and so has to be traced; cel's header is one
            // `'static` class pointer and declares no second word. Adding an
            // edge here to look like pyre would hand the collector a static
            // address to trace.
            &[],
        );

        // The whole reason the two registrations are in one function's reach.
        // A `SizeDescr` published under a type id the collector does not know
        // by that number is not a wrong answer anywhere; it is a header word
        // that indexes the wrong row of the type registry, at collection time,
        // long after the store.
        assert_eq!(
            majit_ir::descr::SizeDescr::type_id(&*group.size_descr),
            registered.type_id,
            "descr type id for {} disagrees with the collector's",
            layout.class.name
        );

        for key in [simple_name.to_string(), def_path] {
            majit_ir::descr_registry::register_keyed_size(
                majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(&key)),
                group.size_descr.clone() as majit_ir::DescrRef,
            );
        }
    }
}

/// The leaf struct a class describes.
///
/// A match on the class address rather than a field on [`ClassLayout`],
/// because the answer has to be the Rust struct's identifier — the thing the
/// translator hashes — and reading it off the CEL type name would silently
/// substitute `"int"` for `"W_IntObject"` the first time the two diverged.
fn leaf_struct_name(class: &'static CelClass) -> &'static str {
    let addr = class as *const CelClass as usize;
    for (candidate, name) in CEL_LEAF_STRUCT_NAMES {
        if *candidate as *const CelClass as usize == addr {
            return name;
        }
    }
    unreachable!("class {} has no leaf struct name", class.name)
}

static CEL_LEAF_STRUCT_NAMES: &[(&CelClass, &str)] = &[
    (&CEL_NULL_CLASS, "W_NullObject"),
    (&CEL_BOOL_CLASS, "W_BoolObject"),
    (&CEL_INT_CLASS, "W_IntObject"),
    (&CEL_UINT_CLASS, "W_UIntObject"),
    (&CEL_DOUBLE_CLASS, "W_DoubleObject"),
    (&CEL_DURATION_CLASS, "W_DurationObject"),
    (&CEL_TIMESTAMP_CLASS, "W_TimestampObject"),
    (&CEL_TYPE_CLASS, "W_TypeObject"),
    (&CEL_OPTIONAL_CLASS, "W_OptionalObject"),
    (&CEL_BYTES_CLASS, "W_BytesObject"),
    (&CEL_STRING_CLASS, "W_StringObject"),
    (&CEL_LIST_CLASS, "W_ListObject"),
];

/// `get_type_flag(FIELDTYPE)`.
fn array_flag(ty: Type, signed: bool) -> ArrayFlag {
    match ty {
        Type::Ref => ArrayFlag::Pointer,
        Type::Float => ArrayFlag::Float,
        Type::Int if signed => ArrayFlag::Signed,
        Type::Int => ArrayFlag::Unsigned,
        Type::Void => ArrayFlag::Void,
    }
}

/// A collector with cel's classes registered, ready for
/// `JitDriver::set_gc_allocator`, and the ids it assigned.
///
/// The ids come back separately because `set_gc_allocator` takes the allocator
/// by `Box` and gives nothing back, so a caller that needs to publish descrs
/// has to have them before it hands the collector over.
pub fn cel_gc(config: GcConfig) -> (Box<dyn GcAllocator>, CelTypeIds) {
    let mut gc = MiniMarkGC::with_config(config);
    let ids = register_cel_classes_unfrozen(&mut gc);
    #[cfg(not(target_arch = "wasm32"))]
    majit_metainterp::register_active_backend_jitframe_gc_type(&mut gc);
    GcAllocator::freeze_types(&mut gc);
    (Box::new(gc), ids)
}

thread_local! {
    /// Whether this execution thread has installed its scalar-tier collector.
    ///
    /// This is deliberately execution-context state: the collector owns only
    /// temporary JITFRAMEs, deadframes are thread-confined, and no CEL object
    /// identity or root is duplicated. Once CEL values move onto this heap,
    /// `install_cel_gc` replaces this temporary boundary with their real owner.
    static JITFRAME_GC_INSTALLED: core::cell::Cell<bool> = const {
        core::cell::Cell::new(false)
    };
}

/// Install the smallest collector the scalar compiled tier needs.
///
/// RPython's `gc_ll_descr.malloc_jitframe` allocates every entry frame as a
/// typed GC object. The scalar tier constructs no CEL heap object yet, so its
/// collector needs exactly that one type. Installation is once per execution
/// thread because the selected native backend owns its active allocator there.
#[cfg(not(target_arch = "wasm32"))]
pub fn install_jitframe_gc<S: majit_metainterp::JitState>(
    driver: &mut majit_metainterp::JitDriver<S>,
) {
    JITFRAME_GC_INSTALLED.with(|installed| {
        if installed.get() {
            return;
        }
        let mut gc = MiniMarkGC::new();
        majit_metainterp::register_active_backend_jitframe_gc_type(&mut gc);
        driver.set_gc_allocator(Box::new(gc));
        installed.set(true);
    });
}

#[cfg(target_arch = "wasm32")]
pub fn install_jitframe_gc<S: majit_metainterp::JitState>(
    _driver: &mut majit_metainterp::JitDriver<S>,
) {
}

/// All three registrations against one driver.
///
/// **No caller yet, and that is deliberate.** cel's compiled tier today runs
/// integer and float columns and constructs no cel object, so calling this
/// would change three backend settings that nothing in that tier reads — and
/// every per-call number on record was taken without them. The wiring point is
/// here, whole and type-checked, for the phase that moves an evaluator onto the
/// class family.
///
/// `set_vtable_offset(Some(0))` belongs with the other two rather than beside
/// the class definitions: it is what makes a `GuardClass` compare the word at
/// offset 0 against a class address directly instead of resolving the address
/// to a type id first. Both routes are registered, because they answer
/// different questions — the offset serves `GuardClass`, the vtable→id map
/// serves the allocation rewrite and `GuardIsObject`.
pub fn install_cel_gc<S: majit_metainterp::JitState>(
    driver: &mut majit_metainterp::JitDriver<S>,
    config: GcConfig,
) -> CelTypeIds {
    let (gc, ids) = cel_gc(config);
    publish_cel_descrs(&ids);
    driver.set_gc_allocator(gc);
    driver.set_new_via_gc(true);
    driver.set_vtable_offset(Some(0));
    JITFRAME_GC_INSTALLED.with(|installed| installed.set(true));
    ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::object::CelRef;

    fn fresh_gc() -> (MiniMarkGC, CelTypeIds) {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        let ids = register_cel_classes(&mut gc);
        (gc, ids)
    }

    /// The table has to describe the structs it names, and nothing checks that
    /// but this: `size_of` is written per row, and a leaf that grew a field
    /// would otherwise be registered at its old size and allocated at its new
    /// one.
    #[test]
    fn every_layout_describes_its_leaf() {
        for layout in CEL_CLASS_LAYOUTS {
            assert!(
                layout.size >= size_of::<CelObject>(),
                "{}: an instance is at least a header",
                layout.class.name
            );
            let payload = layout.size - size_of::<CelObject>();
            let claimed: usize = layout.fields.iter().map(|f| f.size).sum();
            assert_eq!(
                claimed, payload,
                "{}: the field table does not cover the payload",
                layout.class.name
            );
            for field in layout.fields {
                assert!(
                    field.offset + field.size <= layout.size,
                    "{}.{} runs past the instance",
                    layout.class.name,
                    field.name
                );
            }
            for offset in layout.gc_ptr_offsets {
                assert!(
                    layout
                        .fields
                        .iter()
                        .any(|f| f.offset == *offset && f.ty == Type::Ref),
                    "{}: a managed edge at {offset} names no Ref field",
                    layout.class.name
                );
            }
        }
    }

    /// Every class gets its own id, and the address of the class resolves back
    /// to it. This is the whole contract the compiled code's `GuardClass`
    /// rests on: it compares the header word against a class ADDRESS, and the
    /// backend turns that address into the id it checks.
    #[test]
    fn a_class_address_resolves_to_its_type_id() {
        let (gc, ids) = fresh_gc();
        let mut seen = std::collections::BTreeSet::new();
        for registered in &ids.classes {
            assert!(
                seen.insert(registered.type_id),
                "{} shares a type id",
                registered.class.name
            );
            assert_eq!(
                GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, registered.vtable()),
                Some(registered.type_id),
                "{} does not resolve from its address",
                registered.class.name
            );
        }
        // An address that is not a registered class answers nothing rather
        // than answering the nearest one.
        assert_eq!(
            GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, 0xCAFE_BABE),
            None
        );
    }

    /// `freeze_types` is what fills these, and an unfrozen registry answers
    /// `0..0` for every class — which reads as a valid empty range rather than
    /// as "not computed".
    #[test]
    fn freezing_assigns_a_nonempty_subclass_range_per_class() {
        let (gc, ids) = fresh_gc();
        for registered in &ids.classes {
            let range = GcAllocator::subclass_range(&gc, registered.vtable());
            let Some((min, max)) = range else {
                panic!("{} has no subclass range", registered.class.name);
            };
            assert!(
                min < max,
                "{}: empty subclass range {min}..{max}",
                registered.class.name
            );
        }
    }

    /// The recipe end to end, with no pyre in the process: allocate through
    /// the collector at a cel type id, stamp the header the way a compiled
    /// `NewWithVtable` would, and read the type back out of the object.
    ///
    /// `alloc_nursery_typed` rather than `super::super::lltype::malloc_typed`,
    /// because the two are different heaps: the leaf constructors leak, and an
    /// object the collector never allocated has no header for
    /// `get_actual_typeid` to read.
    #[test]
    fn an_object_allocated_at_a_cel_type_id_reports_its_class() {
        let (mut gc, ids) = fresh_gc();
        assert!(
            GcAllocator::supports_guard_gc_type(&gc),
            "the guard_gc_type path is what GuardClass lowers to"
        );

        let int_tid = ids.type_id_of(&CEL_INT_CLASS).expect("int is registered");
        let obj = gc.alloc_nursery_typed(int_tid, size_of::<W_IntObject>());
        assert_ne!(obj.0, 0, "the nursery is big enough for one leaf");

        unsafe {
            let w = obj.0 as CelRef;
            (*w).ob_type = &CEL_INT_CLASS;
            crate::runtime::object::payload!(w, W_IntObject, intval) = 7;
        }

        assert!(GcAllocator::check_is_object(&gc, obj));
        assert_eq!(GcAllocator::get_actual_typeid(&gc, obj), Some(int_tid));
        assert_eq!(
            unsafe { crate::runtime::object::w_type(obj.0 as CelRef) },
            &CEL_INT_CLASS as *const CelClass
        );
    }

    /// The two id namespaces are independent, so the assert inside
    /// `publish_cel_descrs` is the only thing keeping them in step. Running it
    /// is the test.
    #[test]
    fn descrs_publish_under_the_collectors_type_ids() {
        let (_gc, ids) = fresh_gc();
        publish_cel_descrs(&ids);
    }

    /// The blocks register after the classes, so every class keeps the id it
    /// had before they existed. That is not cosmetic: `publish_cel_descrs`
    /// pairs `CEL_CLASS_LAYOUTS` with `ids.classes` by position, and a block
    /// registered in the middle would shift every class past it onto a
    /// neighbour's descr.
    #[test]
    fn the_blocks_register_past_every_class() {
        let (_gc, ids) = fresh_gc();
        let last_class = ids
            .classes
            .iter()
            .map(|r| r.type_id)
            .max()
            .expect("the family is not empty");
        assert!(ids.root < last_class);
        assert!(ids.items_block > last_class);
        assert!(ids.bytes_block > ids.items_block);
    }

    /// A block is not a value: nothing maps a class address to its id, and
    /// `type_id_of` must not start answering for one.
    #[test]
    fn no_class_address_resolves_to_a_block() {
        let (gc, ids) = fresh_gc();
        for registered in &ids.classes {
            let resolved =
                GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, registered.vtable());
            assert_ne!(resolved, Some(ids.items_block));
            assert_ne!(resolved, Some(ids.bytes_block));
        }
    }

    /// The registration's whole point, end to end: a block allocated at
    /// [`CelTypeIds::items_block`] has its items walked.
    ///
    /// The collector reads the length off the word at `len_offset` and forwards
    /// `length` slots from `base_size` — which is `CEL_ITEMS_BLOCK_TOKEN`
    /// verbatim — so a leaf reachable only through a block slot survives the
    /// collection and the slot carries its new address. Nothing but a real
    /// collection can check that: a block registered with a wrong shape does
    /// not fail at registration, it walks the wrong words later.
    ///
    /// `alloc_varsize_typed` rather than `super::super::object_array::new_items_block`,
    /// for the reason `an_object_allocated_at_a_cel_type_id_reports_its_class`
    /// does not use `malloc_typed`: the two are different heaps, and
    /// `new_items_block` allocates from `std::alloc`, where the collector owns
    /// no header to read this type id back out of.
    #[test]
    fn a_minor_collection_walks_an_items_blocks_slots() {
        let (mut gc, ids) = fresh_gc();
        let int_tid = ids.type_id_of(&CEL_INT_CLASS).expect("int is registered");

        let leaf = gc.alloc_nursery_typed(int_tid, size_of::<W_IntObject>());
        assert_ne!(leaf.0, 0);
        unsafe {
            let w = leaf.0 as CelRef;
            (*w).ob_type = &CEL_INT_CLASS;
            crate::runtime::object::payload!(w, W_IntObject, intval) = 7;
        }

        let block = gc.alloc_varsize_typed(
            ids.items_block,
            CEL_ITEMS_BLOCK_TOKEN.base_size,
            CEL_ITEMS_BLOCK_TOKEN.item_size,
            1,
        );
        assert_ne!(block.0, 0);
        unsafe {
            // The length word first, exactly as `alloc_block` writes it: the
            // walker reads it, so a block is never observable without one.
            *((block.0 + CEL_ITEMS_BLOCK_TOKEN.len_offset) as *mut usize) = 1;
            *((block.0 + CEL_ITEMS_BLOCK_TOKEN.base_size) as *mut usize) = leaf.0;
        }
        assert_eq!(
            GcAllocator::get_actual_typeid(&gc, block),
            Some(ids.items_block)
        );

        // The block is the only root. The leaf is reachable through its slot
        // and through nothing else, so it survives only if the slot is walked.
        let mut root = block;
        unsafe { GcAllocator::add_root(&mut gc, &mut root) };
        gc.do_collect_nursery();

        assert_ne!(root.0, block.0, "the collection promoted the block");
        let moved_leaf = unsafe { *((root.0 + CEL_ITEMS_BLOCK_TOKEN.base_size) as *const usize) };
        assert_ne!(moved_leaf, 0, "the slot was cleared rather than forwarded");
        assert_ne!(
            moved_leaf, leaf.0,
            "the slot kept the pre-collection address"
        );
        unsafe {
            let w = moved_leaf as CelRef;
            assert_eq!(crate::runtime::object::payload!(w, W_IntObject, intval), 7);
        }
        GcAllocator::remove_root(&mut gc, &mut root);
    }
}
