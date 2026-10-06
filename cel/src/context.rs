use crate::magic::{Function, FunctionRegistry, IntoFunction};
use crate::objects::{Opaque, TryIntoValue, Value};
use crate::parser::Expression;
use crate::{Env, ExecutionError};
use std::sync::Arc;

pub use crate::scope_map::ScopeMap;

/// Context is a collection of variables and functions that can be used
/// by the interpreter to resolve expressions.
///
/// The context can be either a parent context, or a child context. A
/// parent context is created by default and contains all of the built-in
/// functions. A child context can be created by calling `.new_inner_scope()`. The
/// child context has it's own variables (which can be added to), but it
/// will also reference the parent context. This allows for variables to
/// be overridden within the child context while still being able to
/// resolve variables in the child's parents. You can have theoretically
/// have an infinite number of child contexts that reference each-other.
///
/// So why is this important? Well some CEL-macros such as the `.map` macro
/// declare intermediate user-specified identifiers that should only be
/// available within the macro, and should not override variables in the
/// parent context. The `.map` macro can create a child context from the parent, add the
/// intermediate identifier to the child context, and then evaluate the
/// map expression.
///
/// Intermediate variable stored in child context
///               ↓
/// [1, 2, 3].map(x, x * 2) == [2, 4, 6]
///                  ↑
/// Only in scope for the duration of the map expression
///
/// Neither `Send` nor `Sync`. Interned leaves are pointers into a bind
/// region attached to the binding thread's heap, and
/// `&dyn VariableResolver` has no `Send` bound. A Context cannot move
/// onto another thread; concurrent evaluation is one Context per thread.
///
/// ```compile_fail
/// fn needs_send<T: Send>(_: T) {}
/// needs_send(cel::Context::default());
/// ```
///
/// ```compile_fail
/// fn needs_sync<T: Sync>(_: &T) {}
/// needs_sync(&cel::Context::default());
/// ```
pub enum Context<'a> {
    Root {
        functions: FunctionRegistry,
        /// Shared immortal scope map (`mapdict.py` `_get_mapdict_map`).
        map: &'static ScopeMap,
        /// Per-scope values, indexed by `PlainAttribute.storageindex`.
        storage: Vec<Value>,
        resolver: Option<&'a dyn VariableResolver>,
        env: Arc<Env>,
        /// Owning public handles for values wrapped at bind. Interned
        /// leaves hold a non-owning link into these; they stay alive for
        /// the Context's lifetime so a rebind cannot dangle a pointer
        /// still sitting on the operand stack.
        retained: Vec<Value>,
        /// Chunks wrap-at-bind allocated. Empty until the first wrap;
        /// dropped with the Context. A child created for a comprehension
        /// never wraps, so it stays empty.
        region: crate::runtime::heap::BindRegionSlot,
        /// Interned-leaf slots the portal reads (`_mapdict_read_storage`).
        /// Inline so a child can hop `parent` without a residual. The
        /// items block lives in `region`.
        leaves: crate::runtime::object_array::CelLeafStorage,
    },
    Child {
        parent: &'a Context<'a>,
        map: &'static ScopeMap,
        storage: Vec<Value>,
        resolver: Option<&'a dyn VariableResolver>,
        retained: Vec<Value>,
        region: crate::runtime::heap::BindRegionSlot,
        leaves: crate::runtime::object_array::CelLeafStorage,
    },
}

/// Wrap a value as it enters the system: a class-family leaf is stored as
/// [`Value::Interned`], so a later load is a pointer copy. A container,
/// string or bytes wrapped from a public handle records a non-owning link
/// back to that handle; [`retain_public`] keeps the handle alive.
///
/// The interned leaf is allocated from this Context's region, so dropping
/// the Context releases it. Immortal singletons (`intern_prebuilt`) are
/// not allocated here and do not need a bind region.
fn wrap_entry(ctx: &mut Context, value: Value) -> Value {
    if let Some(w) = crate::runtime::convert::intern_prebuilt(&value) {
        return Value::from_interned(w);
    }
    let region = ctx.ensure_region();
    crate::runtime::heap::with_bind_region(region, || {
        match crate::runtime::convert::intern_leaf(&value) {
            Some(w) => {
                crate::runtime::convert::link_public_tree(w, &value);
                retain_public(ctx, value);
                Value::from_interned(w)
            }
            None => value,
        }
    })
}

fn retain_public(ctx: &mut Context, value: Value) {
    if matches!(
        &value,
        Value::List(_) | Value::Map(_) | Value::String(_) | Value::Bytes(_)
    ) {
        ctx.retained_mut().push(value);
    }
}

fn leaf_of(v: &Value) -> Option<crate::runtime::object::CelRef> {
    match v {
        Value::Interned(w) => Some(*w),
        _ => None,
    }
}

/// The public form of a bound value. Interned leaves unpack through the
/// public link when one exists, so a bound list is `Value::List` again.
fn public_form(v: Value) -> Value {
    match v {
        Value::Interned(w) => crate::runtime::convert::interned_to_public(w),
        other => other,
    }
}

impl<'a> Context<'a> {
    pub fn add_variable<S, V>(
        &mut self,
        name: S,
        value: V,
    ) -> Result<(), <V as TryIntoValue>::Error>
    where
        S: AsRef<str>,
        V: TryIntoValue,
    {
        let wrapped = wrap_entry(self, value.try_into_value()?);
        self.bind_value(name, wrapped);
        Ok(())
    }

    pub fn add_variable_from_value<S, V>(&mut self, name: S, value: V)
    where
        S: AsRef<str>,
        V: Into<Value>,
    {
        let wrapped = wrap_entry(self, value.into());
        self.bind_value(name, wrapped);
    }

    /// Binds an application type that CEL treats as an opaque handle.
    ///
    /// Replaces `add_variable_as_val`, which took a boxed trait object. Equality
    /// and the runtime type name go through [`Opaque`]; CEL cannot index,
    /// iterate or size the value, so member access on it is `NoSuchOverload`.
    /// A backing object whose members should resolve on access — a protobuf
    /// message, a database row — is not expressible this way, because
    /// [`Opaque`] carries no accessors. That capability left with the trait
    /// universe and returns with the class-based value family.
    pub fn add_variable_as_opaque<S>(&mut self, name: S, value: Arc<dyn Opaque>)
    where
        S: AsRef<str>,
    {
        let wrapped = wrap_entry(self, Value::Opaque(value));
        self.bind_value(name, wrapped);
    }

    fn bind_value<S>(&mut self, name: S, value: Value)
    where
        S: AsRef<str>,
    {
        let name = name.as_ref();
        let leaf = leaf_of(&value);
        let index = match self {
            Context::Root { map, storage, .. } | Context::Child { map, storage, .. } => {
                let (next, index) = ScopeMap::bind(*map, storage, name, value);
                *map = next;
                index
            }
        };
        self.sync_leaf_slot(index, leaf);
    }

    fn leaves(&self) -> &crate::runtime::object_array::CelLeafStorage {
        match self {
            Context::Root { leaves, .. } | Context::Child { leaves, .. } => leaves,
        }
    }

    fn leaves_mut(&mut self) -> &mut crate::runtime::object_array::CelLeafStorage {
        match self {
            Context::Root { leaves, .. } | Context::Child { leaves, .. } => leaves,
        }
    }

    /// Write `leaf` at `storageindex` on the items block. A missing leaf
    /// stores null so the portal's `slow_pc` path still runs. The block is
    /// allocated in this Context's bind region, sized from the map
    /// (`_mapdict_init_empty`) so later binds do not regrow.
    fn sync_leaf_slot(&mut self, index: usize, leaf: Option<crate::runtime::object::CelRef>) {
        if leaf.is_none() && self.leaves().items.is_null() {
            return;
        }
        let leaf = leaf.unwrap_or(core::ptr::null_mut());
        if crate::runtime::object_array::items_block_store_existing(
            self.leaves().items,
            index,
            leaf,
        ) {
            return;
        }
        let cap = self.map().likely_storage_len().max(index.saturating_add(1));
        let region = self.ensure_region();
        crate::runtime::heap::with_bind_region(region, || {
            let storage = self.leaves_mut();
            if storage.items.is_null() {
                storage.items = crate::runtime::object_array::new_items_block_zeroed(cap);
                let _ = crate::runtime::object_array::items_block_store_existing(
                    storage.items,
                    index,
                    leaf,
                );
            } else {
                storage.items =
                    crate::runtime::object_array::items_block_store(storage.items, index, leaf);
            }
        });
    }

    fn map(&self) -> &'static ScopeMap {
        match self {
            Context::Root { map, .. } | Context::Child { map, .. } => map,
        }
    }

    fn storage(&self) -> &[Value] {
        match self {
            Context::Root { storage, .. } | Context::Child { storage, .. } => storage,
        }
    }

    fn resolver(&self) -> Option<&dyn VariableResolver> {
        match self {
            Context::Root { resolver, .. } | Context::Child { resolver, .. } => *resolver,
        }
    }

    /// Scope map pointer read once per portal entry (`_get_mapdict_map`).
    pub(crate) fn portal_map(&self) -> i64 {
        self.map().as_bits()
    }

    /// Interned-leaf storage the portal reads (`_mapdict_read_storage`).
    ///
    /// The pointer is this Context's inline
    /// [`crate::runtime::object_array::CelLeafStorage`]. It stays valid for
    /// the borrow of `self` used by one evaluation.
    #[cfg(feature = "jit")]
    pub(crate) fn portal_leaf_storage(&self) -> *mut crate::runtime::object_array::CelLeafStorage {
        self.leaves() as *const crate::runtime::object_array::CelLeafStorage
            as *mut crate::runtime::object_array::CelLeafStorage
    }

    fn retained_mut(&mut self) -> &mut Vec<Value> {
        match self {
            Context::Root { retained, .. } => retained,
            Context::Child { retained, .. } => retained,
        }
    }

    fn ensure_region(&mut self) -> *mut crate::runtime::heap::BindRegion {
        let slot = match self {
            Context::Root { region, .. } | Context::Child { region, .. } => region,
        };
        slot.get_or_insert()
    }

    /// The heap wrap-at-bind attached this Context's region to, if any.
    ///
    /// Child scopes do not walk the parent: an empty child region means
    /// this evaluation has not bound, and the thread-local heap is the
    /// right fallback. A bound Root's pointer is the binding thread's
    /// heap; Context is neither Send nor Sync, so evaluation against it
    /// stays on that thread.
    #[inline]
    pub(crate) fn eval_heap(&self) -> Option<&crate::runtime::heap::CelHeap> {
        match self {
            Context::Root { region, .. } | Context::Child { region, .. } => region.attached_heap(),
        }
    }

    /// This Context's bind region, if wrap-at-bind has created one.
    #[inline]
    pub(crate) fn eval_region(&self) -> Option<&crate::runtime::heap::BindRegion> {
        match self {
            Context::Root { region, .. } | Context::Child { region, .. } => region.get(),
        }
    }

    /// Store `value` as given. A comprehension rebinding is an internal move:
    /// an interned element stays a pointer. An unboxed scalar stays unboxed
    /// and [`leaf_of`] declines, so the block slot is null and `slow_pc` runs.
    pub(crate) fn rebind<S>(&mut self, name: S, value: Value)
    where
        S: AsRef<str>,
    {
        self.bind_value(name, value);
    }

    pub fn set_variable_resolver(&mut self, r: &'a dyn VariableResolver) {
        match self {
            Context::Root { resolver, map, .. } | Context::Child { resolver, map, .. } => {
                *resolver = Some(r);
                *map = map.ensure_resolver();
            }
        }
    }

    /// Reads a bound variable.
    ///
    /// A hit is a [`Value`] clone, which for the compound variants is a
    /// refcount bump. It used to convert a boxed trait object on every read,
    /// and that conversion deep-copied a bound list or map.
    ///
    /// A miss on the whole chain falls back to the type identifiers
    /// ([`crate::common::types::r#type::type_ident`]): `int`, `string`,
    /// `null_type` and the rest are values in CEL, not only the names of
    /// conversion functions. The fallback is **last** on purpose -- a bound
    /// variable named `int` shadows the type, which is the order the spec
    /// states for the analogous case and `objects.rs:1541` quotes: a local
    /// variable "shadows any identifier named `x` in ancestor scopes or the
    /// package namespace". Consulting it first would make the type names
    /// unshadowable.
    ///
    /// It costs a string match only where the answer was previously
    /// `UndeclaredReference`, because every bound name is found before the
    /// chain bottoms out.
    ///
    /// A bound container, string or bytes is returned in public form
    /// (`Value::List` / `Map` / `String` / `Bytes`), never as
    /// [`Value::Interned`].
    pub fn get_variable<S>(&self, name: S) -> Option<Value>
    where
        S: AsRef<str>,
    {
        self.lookup_raw(name.as_ref()).map(public_form)
    }

    /// The interned leaf stored under `name`, without cloning the public
    /// [`Value`]. A miss, or a binding with no leaf, is `None`.
    pub(crate) fn lookup_interned(&self, name: &str) -> Option<crate::runtime::object::CelRef> {
        if self.lookup_is_pure() {
            return self.lookup_interned_pure(name);
        }
        if let Some(v) = self.resolver().and_then(|r| r.resolve(name)) {
            // Resolver answers are public Values, not stored Interned
            // slots. Intern so the residual intern_var_ptr load is a
            // leaf and OP_LOAD_VAR does not take slow_pc (a second
            // resolve). Block slots stay Interned-only via leaf_of.
            return crate::runtime::convert::intern_leaf(&v);
        }
        if let Some(idx) = self.map().find_in_this_scope(name) {
            return self.storage().get(idx as usize).and_then(leaf_of);
        }
        match self {
            Context::Child { parent, .. } => parent.lookup_interned(name),
            Context::Root { .. } => crate::common::types::r#type::type_ident(name)
                .as_ref()
                .and_then(crate::runtime::convert::intern_leaf),
        }
    }

    /// `true` when no [`VariableResolver`] sits on this context or an ancestor.
    ///
    /// The context is borrowed immutably for one evaluation, so the answer
    /// does not change between iterations. Callers treat it as
    /// `effectinfo.py` `EF_ELIDABLE_CANNOT_RAISE` (`pure.py` `OptPure`).
    pub(crate) fn lookup_is_pure(&self) -> bool {
        match self {
            Context::Child { parent, .. } => self.resolver().is_none() && parent.lookup_is_pure(),
            Context::Root { .. } => self.resolver().is_none(),
        }
    }

    /// [`lookup_interned`] without consulting any resolver on the chain.
    ///
    /// Sound only when [`lookup_is_pure`] is true: a resolver can return a
    /// different value on every call.
    pub(crate) fn lookup_interned_pure(
        &self,
        name: &str,
    ) -> Option<crate::runtime::object::CelRef> {
        if let Some(idx) = self.map().find_in_this_scope(name) {
            return self.storage().get(idx as usize).and_then(leaf_of);
        }
        match self {
            Context::Child { parent, .. } => parent.lookup_interned_pure(name),
            Context::Root { .. } => crate::common::types::r#type::type_ident(name)
                .as_ref()
                .and_then(crate::runtime::convert::intern_leaf),
        }
    }

    /// Load a bound identifier for the walker.
    ///
    /// A root interned immediate is handed as the public scalar, which is
    /// the form an evaluation result wants. A child binding is handed as
    /// stored: an interned element stays interned so interned arithmetic
    /// does not unpack it, and an unboxed scalar is copied as that scalar
    /// without going through [`Clone`] on the whole enum.
    #[inline]
    pub(crate) fn load_ident(&self, name: &str) -> Option<Value> {
        fn copy_leaf(v: &Value) -> Value {
            match v {
                Value::Int(i) => Value::Int(*i),
                Value::UInt(u) => Value::UInt(*u),
                Value::Float(f) => Value::Float(*f),
                Value::Bool(b) => Value::Bool(*b),
                Value::Null => Value::Null,
                Value::Interned(w) => Value::Interned(*w),
                other => other.clone(),
            }
        }
        fn from_root(v: &Value) -> Value {
            match v {
                Value::Interned(w) => {
                    if let Some(immediate) = crate::runtime::convert::interned_immediate(*w) {
                        immediate
                    } else {
                        Value::Interned(*w)
                    }
                }
                other => copy_leaf(other),
            }
        }
        if let Some(v) = self.resolver().and_then(|r| r.resolve(name)) {
            return Some(match self {
                Context::Root { .. } => from_root(&v),
                Context::Child { .. } => v,
            });
        }
        if let Some(idx) = self.map().find_in_this_scope(name) {
            let stored = self.storage().get(idx as usize)?;
            return Some(match self {
                Context::Root { .. } => from_root(stored),
                Context::Child { .. } => copy_leaf(stored),
            });
        }
        match self {
            Context::Child { parent, .. } => parent.load_ident(name),
            Context::Root { .. } => crate::common::types::r#type::type_ident(name),
        }
    }

    /// The value stored under `name`, still interned if wrap-at-bind interned
    /// it. Loads inside an evaluation use this so they stay a pointer copy.
    pub(crate) fn lookup_raw(&self, name: &str) -> Option<Value> {
        if let Some(v) = self.resolver().and_then(|r| r.resolve(name)) {
            return Some(v);
        }
        if let Some(idx) = self.map().find_in_this_scope(name) {
            return self.storage().get(idx as usize).cloned();
        }
        match self {
            Context::Child { parent, .. } => parent.lookup_raw(name),
            // The base case of the recursion, so a `Child` reaches this through
            // `parent.lookup_raw` and the type identifiers stay behind every
            // scope at every depth.
            Context::Root { .. } => crate::common::types::r#type::type_ident(name),
        }
    }

    /// Whether `name` is bound by an enclosing comprehension.
    ///
    /// The iteration and accumulator variables live in the `Child` scopes
    /// `new_inner_scope` mints; the `Root`'s variables are the activation the
    /// caller supplied. The distinction is load-bearing at exactly one place --
    /// a call whose receiver is a bare identifier -- because a comprehension
    /// variable shadows the package namespace, so `xs.all(optional,
    /// optional.of(1))` is a member call on the element rather than the
    /// namespaced `optional.of`.
    pub(crate) fn is_comprehension_variable(&self, name: &str) -> bool {
        match self {
            Context::Root { .. } => false,
            Context::Child { parent, .. } => {
                self.map().find_in_this_scope(name).is_some()
                    || parent.is_comprehension_variable(name)
            }
        }
    }

    pub(crate) fn env(&self) -> &Env {
        match self {
            Context::Root { env, .. } => env.as_ref(),
            Context::Child { parent, .. } => parent.env(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn get_function(&self, name: &str) -> Option<&Function> {
        match self {
            Context::Root { functions, .. } => functions.get(name),
            Context::Child { parent, .. } => parent.get_function(name),
        }
    }

    fn root_registry(&self) -> &FunctionRegistry {
        match self {
            Context::Root { functions, .. } => functions,
            Context::Child { parent, .. } => parent.root_registry(),
        }
    }

    /// Registry map pointer (`mapdict.py` `_get_mapdict_map`).
    ///
    /// A child walks to the root. The word is the shared layout, not this
    /// registry's address.
    #[cfg(feature = "jit")]
    pub(crate) fn registry_map_bits(&self) -> i64 {
        self.root_registry().map_bits()
    }

    /// Two-int entry storage the portal reads (`_mapdict_read_storage`).
    ///
    /// A child walks to the root. The pointer is the root registry's
    /// inline [`crate::runtime::object_array::CelInt2Storage`].
    #[cfg(feature = "jit")]
    pub(crate) fn portal_int2_entries(&self) -> *mut crate::runtime::object_array::CelInt2Storage {
        self.root_registry().entries_ptr()
    }

    /// Entry word of a two-int scalar under `name` on the root registry.
    ///
    /// A child walks to the root. See [`FunctionRegistry::int2_entry`].
    pub(crate) fn int2_entry(&self, name: &str) -> Option<i64> {
        self.root_registry().int2_entry(name)
    }

    /// [`Context::get_function`] for a namespaced name, without joining the two
    /// parts into a `String` the lookup would immediately discard.
    pub(crate) fn get_qualified_function(&self, prefix: &str, name: &str) -> Option<&Function> {
        match self {
            Context::Root { functions, .. } => functions.get_qualified(prefix, name),
            Context::Child { parent, .. } => parent.get_qualified_function(prefix, name),
        }
    }

    pub fn add_function<T: 'static, F>(&mut self, name: &str, value: F)
    where
        F: IntoFunction<T> + 'static,
    {
        if let Context::Root { functions, .. } = self {
            functions.add(name, value);
        };
    }

    pub fn resolve(&self, expr: &Expression) -> Result<Value, ExecutionError> {
        Value::resolve(expr, self)
    }

    pub fn resolve_all(&self, exprs: &[Expression]) -> Result<Value, ExecutionError> {
        Value::resolve_all(exprs, self)
    }

    pub fn new_inner_scope(&self) -> Context<'_> {
        let map = self.map().child_terminator();
        // `_mapdict_init_empty` / `_make_storage_mixin_size_n`: size
        // storage from the map the unique cached chain will reach.
        let cap = map.likely_storage_len();
        let mut region = crate::runtime::heap::BindRegionSlot::empty();
        let items = if cap == 0 {
            core::ptr::null_mut()
        } else {
            let r = region.get_or_insert();
            crate::runtime::heap::with_bind_region(r, || {
                crate::runtime::object_array::new_items_block_zeroed(cap)
            })
        };
        Context::Child {
            parent: self,
            map,
            storage: Vec::with_capacity(cap),
            resolver: None,
            retained: Vec::new(),
            region,
            leaves: crate::runtime::object_array::CelLeafStorage {
                parent: self.leaves() as *const crate::runtime::object_array::CelLeafStorage
                    as *mut crate::runtime::object_array::CelLeafStorage,
                items,
            },
        }
    }

    /// Constructs a new empty context with no variables or functions.
    ///
    /// If you're looking for a context that has all the standard methods, functions
    /// and macros already added to the context, use [`Context::default`] instead.
    ///
    /// # Example
    /// ```
    /// use cel::Context;
    /// let mut context = Context::empty();
    /// context.add_function("add", |a: i64, b: i64| a + b);
    /// ```
    pub fn empty() -> Self {
        let env = Arc::new(Env::default());
        Context::Root {
            env: Arc::clone(&env),
            map: ScopeMap::root_terminator(),
            storage: Vec::new(),
            functions: FunctionRegistry::with_env(env),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
            leaves: crate::runtime::object_array::CelLeafStorage::empty(),
        }
    }

    pub fn with_env(env: Arc<Env>) -> Self {
        Context::Root {
            functions: FunctionRegistry::with_env(Arc::clone(&env)),
            env,
            map: ScopeMap::root_terminator(),
            storage: Vec::new(),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
            leaves: crate::runtime::object_array::CelLeafStorage::empty(),
        }
    }
}

impl Default for Context<'_> {
    fn default() -> Self {
        let env = Env::shared_stdlib();
        Context::Root {
            env: Arc::clone(&env),
            map: ScopeMap::root_terminator(),
            storage: Vec::new(),
            functions: FunctionRegistry::with_env(env),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
            leaves: crate::runtime::object_array::CelLeafStorage::empty(),
        }
    }
}

/// VariableResolver implements a custom resolver for variables that is consulted before looking at
/// variables added to the context. This allows dynamic variables, or avoiding HashMap lookup/creation.
///
///
/// # Example
/// ```
/// struct ValueContext {
///     request: cel::Value,
///     response: cel::Value,
/// }
///
/// impl cel::context::VariableResolver for ValueContext {
///     fn resolve(&self, variable: &str) -> Option<cel::Value> {
///         match variable {
///             "request" => Some(self.request.clone()),
///             "response" => Some(self.response.clone()),
///             _ => None,
///         }
///     }
/// }
/// ```
pub trait VariableResolver {
    fn resolve(&self, variable: &str) -> Option<Value>;
}

impl<T: VariableResolver> VariableResolver for Box<T> {
    fn resolve(&self, variable: &str) -> Option<Value> {
        (**self).resolve(variable)
    }
}

impl<T: VariableResolver> VariableResolver for Arc<T> {
    fn resolve(&self, variable: &str) -> Option<Value> {
        (**self).resolve(variable)
    }
}

impl<T: VariableResolver> VariableResolver for &T {
    fn resolve(&self, variable: &str) -> Option<Value> {
        (**self).resolve(variable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::heap::with_heap;

    #[test]
    fn a_child_scope_has_no_region_until_it_wraps() {
        let ctx = Context::default();
        let mut inner = ctx.new_inner_scope();
        inner.rebind("x", Value::Int(1));
        match inner.lookup_raw("x") {
            Some(Value::Int(1)) => {}
            other => panic!("rebind of an unboxed int stays unboxed, got {other:?}"),
        }
        match &inner {
            Context::Child { region, leaves, .. } => {
                if leaves.items.is_null() {
                    assert!(region.is_none(), "no wrap and no preallocated block");
                }
            }
            Context::Root { .. } => panic!("inner scope is a child"),
        }
    }

    #[test]
    fn a_bound_leaf_is_contained_only_while_the_context_lives() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("n", 1000i64);
        let w = ctx.lookup_interned("n").expect("leaf");
        assert_eq!(ctx.lookup_interned_pure("n"), Some(w));
        assert!(
            with_heap(|h| h.contains(w as *const u8)),
            "region object is live while the Context lives"
        );
        assert!(
            !with_heap(|h| h.is_young(w as *const u8)),
            "a bound leaf is not nursery memory"
        );
        drop(ctx);
        assert!(
            !with_heap(|h| h.contains(w as *const u8)),
            "dropping the Context releases the region"
        );
    }

    #[test]
    fn a_bound_context_eval_heap_is_this_threads_heap() {
        let mut ctx = Context::default();
        assert!(ctx.eval_heap().is_none());
        ctx.add_variable_from_value("x", 1i64);
        let heap = ctx.eval_heap().expect("bound");
        with_heap(|h| assert!(core::ptr::eq(heap as *const _, h as *const _)));
        let inner = ctx.new_inner_scope();
        match inner.eval_heap() {
            None => {}
            Some(heap) => {
                with_heap(|h| assert!(core::ptr::eq(heap as *const _, h as *const _)));
                let child_region = inner
                    .eval_region()
                    .map(|r| r as *const crate::runtime::heap::BindRegion);
                let parent_region = ctx
                    .eval_region()
                    .map(|r| r as *const crate::runtime::heap::BindRegion);
                assert_ne!(
                    child_region, parent_region,
                    "a child does not inherit the parent's region"
                );
            }
        }
    }

    #[test]
    fn wrap_entry_interns_a_float_so_storage_is_a_leaf() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("price", 1.5f64);
        match ctx.lookup_raw("price") {
            Some(Value::Interned(w)) => {
                assert!(!w.is_null());
                assert_eq!(
                    unsafe { crate::runtime::object::w_kind(w) },
                    crate::runtime::object::CelKind::Double
                );
            }
            other => panic!("price storage is {other:?}, expected Interned"),
        }
        ctx.add_variable_from_value("n", 7i64);
        assert!(
            matches!(ctx.lookup_raw("n"), Some(Value::Interned(_))),
            "int bind is Interned"
        );
    }

    #[test]
    fn two_fresh_roots_that_bind_the_same_names_share_a_map() {
        let mut a = Context::default();
        a.add_variable_from_value("x", 1i64);
        a.add_variable_from_value("y", 2i64);
        let mut b = Context::default();
        b.add_variable_from_value("x", 9i64);
        b.add_variable_from_value("y", 8i64);
        assert_eq!(a.portal_map(), b.portal_map());
        a.add_variable_from_value("x", 3i64);
        assert_eq!(a.portal_map(), b.portal_map());
        a.add_variable_from_value("z", 4i64);
        assert_ne!(a.portal_map(), b.portal_map());
    }

    #[test]
    fn two_children_of_one_root_that_bind_the_same_names_share_a_map() {
        let root = Context::default();
        let mut a = root.new_inner_scope();
        a.add_variable_from_value("x", 10i64);
        a.add_variable_from_value("y", 20i64);
        let mut b = root.new_inner_scope();
        b.add_variable_from_value("x", 11i64);
        b.add_variable_from_value("y", 21i64);
        assert_eq!(a.portal_map(), b.portal_map());
        let mut c = root.new_inner_scope();
        c.add_variable_from_value("y", 1i64);
        c.add_variable_from_value("x", 2i64);
        assert_ne!(a.portal_map(), c.portal_map());
    }

    #[test]
    fn two_fresh_roots_that_register_the_same_names_share_a_registry_map() {
        let mut a = Context::default();
        a.add_function("add", |x: i64, y: i64| x + y);
        a.add_function("multiply", |x: i64, y: i64| x * y);
        let mut b = Context::default();
        b.add_function("add", |x: i64, y: i64| x - y);
        b.add_function("multiply", |x: i64, y: i64| x * y);
        assert_eq!(a.root_registry().map_bits(), b.root_registry().map_bits());
        let empty = Context::default();
        let also_empty = Context::empty();
        assert_eq!(
            empty.root_registry().map_bits(),
            also_empty.root_registry().map_bits()
        );
        a.add_function("add", |x: i64, y: i64| x.wrapping_mul(y));
        assert_eq!(a.root_registry().map_bits(), b.root_registry().map_bits());
    }

    #[test]
    fn interned_bind_writes_the_leaf_block_at_storageindex() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("x", 10i64);
        ctx.add_variable_from_value("y", 20i64);
        let x = ctx.lookup_interned("x").expect("x");
        let y = ctx.lookup_interned("y").expect("y");
        let items = ctx.leaves().items;
        assert!(!items.is_null());
        unsafe {
            let base = crate::runtime::object_array::items_block_items_base(items);
            assert_eq!(*base.add(0), x);
            assert_eq!(*base.add(1), y);
        }
        ctx.add_variable_from_value("x", 30i64);
        let x2 = ctx.lookup_interned("x").expect("x rebind");
        unsafe {
            let base = crate::runtime::object_array::items_block_items_base(ctx.leaves().items);
            assert_eq!(*base.add(0), x2);
            assert_eq!(*base.add(1), y);
        }
    }

    #[test]
    fn a_child_int2_entries_pointer_is_the_root_registry() {
        let mut root = Context::default();
        root.add_function("add", |x: i64, y: i64| x + y);
        let child = root.new_inner_scope();
        assert!(core::ptr::eq(
            child.root_registry().entries_ptr(),
            root.root_registry().entries_ptr()
        ));
        assert_eq!(
            child.root_registry().map_bits(),
            root.root_registry().map_bits()
        );
        let add = root.int2_entry("add").expect("add");
        unsafe {
            let items = (*child.root_registry().entries_ptr()).items;
            assert!(!items.is_null());
            assert_eq!(*crate::runtime::object_array::int_words_base(items), add);
        }
    }

    #[test]
    fn a_child_leaf_block_parent_is_the_enclosing_storage() {
        let mut root = Context::default();
        root.add_variable_from_value("x", 10i64);
        let child = root.new_inner_scope();
        assert!(core::ptr::eq(
            child.leaves().parent,
            root.leaves() as *const _ as *mut _
        ));
        unsafe {
            let parent_items = (*child.leaves().parent).items;
            let x = root.lookup_interned("x").expect("x");
            assert_eq!(
                *crate::runtime::object_array::items_block_items_base(parent_items),
                x
            );
        }
    }
}
