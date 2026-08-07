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
    },
    Child {
        parent: &'a Context<'a>,
        variables: BTreeMap<Box<str>, Value>,
        resolver: Option<&'a dyn VariableResolver>,
    },
}

/// Binds `name`, reusing the key the map already owns when the name is bound.
///
/// Re-binding is the loop case — a comprehension rebinds its iteration
/// variable once per element — and `BTreeMap::insert` takes an owned key, so
/// it allocates a fresh `String` on every pass over a name it already holds.
fn bind(variables: &mut BTreeMap<Box<str>, Value>, name: impl AsRef<str>, value: Value) {
    match variables.get_mut(name.as_ref()) {
        Some(slot) => *slot = value,
        None => {
            variables.insert(name.as_ref().into(), value);
        }
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
        self.bind_value(name, value.try_into_value()?);
        Ok(())
    }

    pub fn add_variable_from_value<S, V>(&mut self, name: S, value: V)
    where
        S: AsRef<str>,
        V: Into<Value>,
    {
        self.bind_value(name, value.into());
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
        self.bind_value(name, Value::Opaque(value));
    }

    fn bind_value<S>(&mut self, name: S, value: Value)
    where
        S: AsRef<str>,
    {
        let variables = match self {
            Context::Root { variables, .. } => variables,
            Context::Child { variables, .. } => variables,
        };
        bind(variables, name, value);
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
    pub fn get_variable<S>(&self, name: S) -> Option<Value>
    where
        S: AsRef<str>,
    {
        let name = name.as_ref();
        let from_resolver =
            |resolver: &Option<&'a dyn VariableResolver>| resolver.and_then(|r| r.resolve(name));
        match self {
            Context::Child {
                variables,
                parent,
                resolver,
            } => from_resolver(resolver).or_else(|| {
                variables
                    .get(name)
                    .cloned()
                    .or_else(|| parent.get_variable(name))
            }),
            Context::Root {
                variables,
                resolver,
                ..
            } => from_resolver(resolver).or_else(|| variables.get(name).cloned()),
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

    pub fn add_function<T: 'static, F>(&mut self, name: &str, value: F)
    where
        F: IntoFunction<T> + 'static + Send + Sync,
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
        }
    }

    pub fn with_env(env: Arc<Env>) -> Self {
        Context::Root {
            env,
            variables: Default::default(),
            functions: Default::default(),
            resolver: None,
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
pub trait VariableResolver: Send + Sync {
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
mod test {
    // A helper function that requires T to implement some traits
    fn assert_send<T: Send>() {}

    #[test]
    fn test_context_is_send() {
        // This line will only compile if assertion passes
        assert_send::<super::Context>();
    }
}
