use crate::magic::{Function, FunctionRegistry, IntoFunction};
use crate::objects::{Opaque, TryIntoValue, Value};
use crate::parser::Expression;
use crate::{Env, ExecutionError};
use std::collections::BTreeMap;
use std::sync::Arc;

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
pub enum Context<'a> {
    Root {
        functions: FunctionRegistry,
        variables: BTreeMap<Box<str>, Value>,
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
    },
    Child {
        parent: &'a Context<'a>,
        variables: BTreeMap<Box<str>, Value>,
        resolver: Option<&'a dyn VariableResolver>,
        retained: Vec<Value>,
        region: crate::runtime::heap::BindRegionSlot,
    },
}

/// Stores `value` under `name`, reusing the key the map already owns.
///
/// Re-binding is the loop case — a comprehension rebinds its iteration
/// variable once per element — and `BTreeMap::insert` takes an owned key, so
/// it allocates a fresh `String` on every pass over a name it already holds.
fn store(variables: &mut BTreeMap<Box<str>, Value>, name: impl AsRef<str>, value: Value) {
    match variables.get_mut(name.as_ref()) {
        Some(slot) => *slot = value,
        None => {
            variables.insert(name.as_ref().into(), value);
        }
    }
}

/// Wrap a value as it enters the system: a class-family leaf is stored as
/// [`Value::Interned`], so a later load is a pointer copy. A container,
/// string or bytes wrapped from a public handle records a non-owning link
/// back to that handle; [`retain_public`] keeps the handle alive.
///
/// The interned leaf is allocated from this Context's region, so dropping
/// the Context releases it. Immortal singletons are not allocated here.
fn wrap_entry(ctx: &mut Context, value: Value) -> Value {
    let region = ctx.ensure_region();
    crate::runtime::heap::with_bind_region(region, || {
        match crate::runtime::convert::intern_leaf(&value) {
            Some(w) => {
                crate::runtime::convert::link_public_handle(w, &value);
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
        let variables = match self {
            Context::Root { variables, .. } => variables,
            Context::Child { variables, .. } => variables,
        };
        store(variables, name, value);
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

    /// Store `value` as given. A comprehension rebinding is an internal move,
    /// not an entry: an interned element stays a pointer, an unboxed scalar
    /// stays unboxed.
    pub(crate) fn rebind<S>(&mut self, name: S, value: Value)
    where
        S: AsRef<str>,
    {
        self.bind_value(name, value);
    }

    pub fn set_variable_resolver(&mut self, r: &'a dyn VariableResolver) {
        match self {
            Context::Root { resolver, .. } => {
                *resolver = Some(r);
            }
            Context::Child { resolver, .. } => {
                *resolver = Some(r);
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
        fn leaf_of(v: &Value) -> Option<crate::runtime::object::CelRef> {
            match v {
                Value::Interned(w) => Some(*w),
                other => crate::runtime::convert::intern_leaf(other),
            }
        }
        let from_resolver =
            |resolver: &Option<&'a dyn VariableResolver>| resolver.and_then(|r| r.resolve(name));
        match self {
            Context::Child {
                variables,
                parent,
                resolver,
                ..
            } => {
                if let Some(v) = from_resolver(resolver) {
                    return leaf_of(&v);
                }
                variables
                    .get(name)
                    .and_then(leaf_of)
                    .or_else(|| parent.lookup_interned(name))
            }
            Context::Root {
                variables,
                resolver,
                ..
            } => {
                if let Some(v) = from_resolver(resolver) {
                    return leaf_of(&v);
                }
                variables.get(name).and_then(leaf_of).or_else(|| {
                    crate::common::types::r#type::type_ident(name)
                        .as_ref()
                        .and_then(leaf_of)
                })
            }
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
                    if let Some(immediate) =
                        crate::runtime::convert::interned_immediate(*w)
                    {
                        immediate
                    } else {
                        Value::Interned(*w)
                    }
                }
                other => copy_leaf(other),
            }
        }
        let from_resolver =
            |resolver: &Option<&'a dyn VariableResolver>| resolver.and_then(|r| r.resolve(name));
        match self {
            Context::Child {
                variables,
                parent,
                resolver,
                ..
            } => from_resolver(resolver)
                .or_else(|| variables.get(name).map(copy_leaf))
                .or_else(|| parent.load_ident(name)),
            Context::Root {
                variables,
                resolver,
                ..
            } => from_resolver(resolver)
                .map(|v| from_root(&v))
                .or_else(|| variables.get(name).map(from_root))
                .or_else(|| crate::common::types::r#type::type_ident(name)),
        }
    }

    /// The value stored under `name`, still interned if wrap-at-bind interned
    /// it. Loads inside an evaluation use this so they stay a pointer copy.
    pub(crate) fn lookup_raw(&self, name: &str) -> Option<Value> {
        let from_resolver =
            |resolver: &Option<&'a dyn VariableResolver>| resolver.and_then(|r| r.resolve(name));
        match self {
            Context::Child {
                variables,
                parent,
                resolver,
                ..
            } => from_resolver(resolver).or_else(|| {
                variables
                    .get(name)
                    .cloned()
                    .or_else(|| parent.lookup_raw(name))
            }),
            // The base case of the recursion, so a `Child` reaches this through
            // `parent.lookup_raw` and the type identifiers stay behind every
            // scope at every depth.
            Context::Root {
                variables,
                resolver,
                ..
            } => from_resolver(resolver)
                .or_else(|| variables.get(name).cloned())
                .or_else(|| crate::common::types::r#type::type_ident(name)),
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
            Context::Child {
                variables, parent, ..
            } => variables.contains_key(name) || parent.is_comprehension_variable(name),
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
        Context::Child {
            parent: self,
            variables: Default::default(),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
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
        Context::Root {
            env: Arc::new(Env::default()),
            variables: Default::default(),
            functions: Default::default(),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
        }
    }

    pub fn with_env(env: Arc<Env>) -> Self {
        Context::Root {
            env,
            variables: Default::default(),
            functions: Default::default(),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
        }
    }
}

impl Default for Context<'_> {
    fn default() -> Self {
        Context::Root {
            env: Env::shared_stdlib(),
            variables: Default::default(),
            functions: Default::default(),
            resolver: None,
            retained: Vec::new(),
            region: crate::runtime::heap::BindRegionSlot::empty(),
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
        match &inner {
            Context::Child { region, .. } => assert!(region.is_none()),
            Context::Root { .. } => panic!("inner scope is a child"),
        }
    }

    #[test]
    fn a_bound_leaf_is_contained_only_while_the_context_lives() {
        let mut ctx = Context::default();
        ctx.add_variable_from_value("n", 1000i64);
        let w = ctx.lookup_interned("n").expect("leaf");
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
}
