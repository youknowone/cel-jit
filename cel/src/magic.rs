use crate::macros::{impl_conversions, impl_handler};
use crate::objects::{ListRef, Opaque};
use crate::resolvers::{AllArguments, Argument};
use crate::{ExecutionError, FunctionContext, ResolveResult, Value};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

impl_conversions!(
    i64 => Value::Int,
    u64 => Value::UInt,
    f64 => Value::Float,
    Arc<String> => Value::String,
    Arc<Vec<u8>> => Value::Bytes,
    bool => Value::Bool,
    ListRef => Value::List,
    Arc<dyn Opaque> => Value::Opaque
);

#[cfg(feature = "chrono")]
impl_conversions!(
    chrono::Duration => Value::Duration,
    chrono::DateTime<chrono::FixedOffset> => Value::Timestamp,
);

impl From<i32> for Value {
    fn from(value: i32) -> Self {
        Value::Int(value as i64)
    }
}

impl From<u32> for Value {
    fn from(value: u32) -> Self {
        Value::UInt(value as u64)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::Float(value as f64)
    }
}

/// Describes any type that can be converted from a [`Value`] into itself.
/// This is commonly used to convert from [`Value`] into primitive types,
/// e.g. from `Value::Bool(true) -> true`. This trait is auto-implemented
/// for many CEL-primitive types.
trait FromValue {
    fn from_value(value: &Value) -> Result<Self, ExecutionError>
    where
        Self: Sized;
}

impl FromValue for Value {
    fn from_value(value: &Value) -> Result<Self, ExecutionError>
    where
        Self: Sized,
    {
        Ok(value.clone())
    }
}

/// A trait for types that can be converted into a [`ResolveResult`]. Every function that can
/// be registered to the CEL context must return a value that implements this trait.
pub trait IntoResolveResult {
    fn into_resolve_result(self) -> ResolveResult;
}

impl IntoResolveResult for String {
    fn into_resolve_result(self) -> ResolveResult {
        Ok(Value::String(Arc::new(self)))
    }
}

impl IntoResolveResult for Result<Value, ExecutionError> {
    fn into_resolve_result(self) -> ResolveResult {
        self
    }
}

/// Describes any type that can be converted from a [`FunctionContext`] into
/// itself, for example CEL primitives implement this trait to allow them to
/// be used as arguments to functions. This trait is core to the 'magic function
/// parameter' system. Every argument to a function that can be registered to
/// the CEL context must implement this type.
pub(crate) trait FromContext<'a, 'context, 'call> {
    fn from_context(ctx: &'a mut FunctionContext<'context, 'call>) -> Result<Self, ExecutionError>
    where
        Self: Sized;
}

/// A function argument abstraction enabling dynamic method invocation on a
/// target instance or on the first argument if the function is not called
/// as a method.
///
/// This is similar to how methods can be called as functions using the
/// [fully-qualified syntax](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#fully-qualified-syntax-for-disambiguation-calling-methods-with-the-same-name).
///
/// # Using `This`
/// ```
/// # use std::sync::Arc;
/// # use cel::{Program, Context};
/// use cel::extractors::This;
/// # let mut context = Context::default();
/// # context.add_function("startsWith", starts_with);
///
/// /// Notice how `This` refers to the target value when called as a method,
/// /// but the first argument when called as a function.
/// let program1 = "'foobar'.startsWith('foo') == true";
/// let program2 = "startsWith('foobar', 'foo') == true";
/// # let program1 = Program::compile(program1).unwrap();
/// # let program2 = Program::compile(program2).unwrap();
/// # let value = program1.execute(&context).unwrap();
/// # assert_eq!(value, true.into());
/// # let value = program2.execute(&context).unwrap();
/// # assert_eq!(value, true.into());
///
/// fn starts_with(This(this): This<Arc<String>>, prefix: Arc<String>) -> bool {
///     this.starts_with(prefix.as_str())
/// }
/// ```
///
/// # Type of `This`
/// This also accepts a type `T` which determines the specific type
/// that's extracted. Any type that supports [`FromValue`] can be used.
/// In the previous example, the method `startsWith` is only ever called
/// on a string, so we can use `This<Rc<String>>` to extract the string
/// automatically prior to our method actually being called.
///
/// In some cases, you may want access to the raw [`Value`] instead, for
/// example, the `contains` method works for several different types. In these
/// cases, you can use `This<Value>` to extract the raw value.
///
/// ```skip
/// pub fn contains(This(this): This<Value>, arg: Value) -> Result<Value> {
///     Ok(match this {
///         Value::List(v) => v.contains(&arg),
///         ...
///     }
/// }
/// ```
pub struct This<T>(pub T);

impl<'a, 'context, 'call, T> FromContext<'a, 'context, 'call> for This<T>
where
    T: FromValue,
{
    fn from_context(ctx: &'a mut FunctionContext<'context, 'call>) -> Result<Self, ExecutionError>
    where
        Self: Sized,
    {
        if let Some(ref this) = ctx.this {
            Ok(This(T::from_value(this)?))
        } else {
            let arg = arg_value_from_context(ctx)
                .map_err(|_| ExecutionError::missing_argument_or_target())?;
            Ok(This(T::from_value(&arg)?))
        }
    }
}

/// Identifier is an argument extractor that attempts to extract an identifier
/// from an argument's expression.
///
/// It fails if the argument is not available, or if the argument cannot be
/// converted into an expression.
///
/// # Examples
/// Identifiers are useful for functions like `.map` or `.filter` where one
/// of the arguments is the declaration of a variable. In this case, as noted
/// below, the x is an identifier, and we want to be able to parse it
/// automatically.
///
/// ```javascript
/// //        Identifier
/// //            ↓
/// [1, 2, 3].map(x, x * 2) == [2, 4, 6]
/// ```
///
/// The function signature for the Rust implementation of `map` looks like this
///
/// ```skip
/// pub fn map(
///     ftx: &FunctionContext,
///     This(this): This<Value>, // <- [1, 2, 3]
///     ident: Identifier,       // <- x
///     expr: Expression,        // <- x * 2
/// ) -> Result<Value>;
/// ```
#[derive(Clone)]
pub struct Identifier(pub Arc<String>);

impl From<&Identifier> for String {
    fn from(value: &Identifier) -> Self {
        value.0.to_string()
    }
}

impl From<Identifier> for String {
    fn from(value: Identifier) -> Self {
        value.0.as_ref().clone()
    }
}

/// An argument extractor that extracts all the arguments passed to a function, resolves their
/// expressions and returns a vector of [`Value`].
///
/// This is useful for functions that accept a variable number of arguments rather than known
/// arguments and types (for example a `sum` function).
///
/// # Example
/// ```javascript
/// sum(1, 2.0, uint(3)) == 5.0
/// ```
///
/// ```rust
/// # use cel::{Value};
/// use cel::extractors::Arguments;
/// pub fn sum(Arguments(args): Arguments) -> Value {
///     args.iter().fold(0.0, |acc, val| match val {
///         Value::Int(x) => x as f64 + acc,
///         Value::UInt(x) => x as f64 + acc,
///         Value::Float(x) => x + acc,
///         _ => acc,
///     }).into()
/// }
/// ```
#[derive(Clone)]
pub struct Arguments(pub ListRef);

impl<'a> FromContext<'a, '_, '_> for Arguments {
    fn from_context(ctx: &'a mut FunctionContext) -> Result<Self, ExecutionError>
    where
        Self: Sized,
    {
        match ctx.resolve(AllArguments)? {
            Value::List(list) => Ok(Arguments(list.clone())),
            _ => todo!(),
        }
    }
}

impl<'a, 'context, 'call> FromContext<'a, 'context, 'call> for Value {
    fn from_context(ctx: &'a mut FunctionContext<'context, 'call>) -> Result<Self, ExecutionError>
    where
        Self: Sized,
    {
        arg_value_from_context(ctx)
    }
}

/// Returns the next argument specified by the context's `arg_idx` field as after resolving
/// it. Calling this multiple times will increment the `arg_idx` which will return subsequent
/// arguments every time.
///
/// Calling this function when there are no more arguments will result in a panic. Since this
/// function is only ever called within the context of a controlled macro that calls it once
/// for each argument, this should never happen.
fn arg_value_from_context(ctx: &mut FunctionContext) -> Result<Value, ExecutionError> {
    let idx = ctx.arg_idx;
    ctx.arg_idx += 1;
    ctx.resolve(Argument(idx))
}

pub struct WithFunctionContext;

impl_handler!();
impl_handler!(C1);
impl_handler!(C1, C2);
impl_handler!(C1, C2, C3);
impl_handler!(C1, C2, C3, C4);
impl_handler!(C1, C2, C3, C4, C5);
impl_handler!(C1, C2, C3, C4, C5, C6);
impl_handler!(C1, C2, C3, C4, C5, C6, C7);
impl_handler!(C1, C2, C3, C4, C5, C6, C7, C8);
impl_handler!(C1, C2, C3, C4, C5, C6, C7, C8, C9);

// Heavily inspired by https://users.rust-lang.org/t/common-data-type-for-functions-with-different-parameters-e-g-axum-route-handlers/90207/6
// and https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=c6744c27c2358ec1d1196033a0ec11e4

#[derive(Default)]
pub struct FunctionRegistry {
    functions: BTreeMap<String, Function>,
}

impl FunctionRegistry {
    pub(crate) fn add<F, T>(&mut self, name: &str, function: F)
    where
        F: IntoFunction<T> + 'static + Send + Sync,
        T: 'static,
    {
        self.functions
            .insert(name.to_string(), function.into_function());
    }

    #[allow(dead_code)]
    pub(crate) fn get(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    /// [`FunctionRegistry::get`] for a namespaced name, without joining the two
    /// parts into a `String` the lookup would immediately discard.
    pub(crate) fn get_qualified(&self, prefix: &str, name: &str) -> Option<&Function> {
        crate::common::get_qualified(&self.functions, prefix, name)
    }
}

/// A registered function in the form every evaluator calls it: arguments and
/// receiver arrive through the [`FunctionContext`], the answer is a [`Value`].
pub type ErasedFunction = Box<dyn Fn(&mut FunctionContext) -> ResolveResult + Send + Sync>;

/// A registered function.
///
/// Every function has its erased form, and a [`Function`] derefs to it so a
/// caller invokes one as `(func)(&mut ftx)`. A closure whose signature is one
/// of [`ScalarFn`]'s also keeps that signature: the batch machine calls it on
/// its register banks directly, with no [`Value`] built for either side.
pub struct Function {
    erased: ErasedFunction,
    scalar: Option<Arc<ScalarFn>>,
}

impl Function {
    /// A function with only its erased form.
    pub fn erased(erased: ErasedFunction) -> Self {
        Function {
            erased,
            scalar: None,
        }
    }

    /// A function whose typed closure is also offered, boxed as `Any`: it is
    /// kept if it has one of [`ScalarFn`]'s signatures and dropped otherwise.
    pub(crate) fn with_typed(erased: ErasedFunction, typed: Box<dyn Any + Send + Sync>) -> Self {
        Function {
            erased,
            scalar: ScalarFn::from_any(typed).map(Arc::new),
        }
    }

    /// The scalar form, when the closure had one of [`ScalarFn`]'s signatures.
    pub fn scalar(&self) -> Option<&Arc<ScalarFn>> {
        self.scalar.as_ref()
    }
}

impl std::ops::Deref for Function {
    type Target = dyn Fn(&mut FunctionContext) -> ResolveResult + Send + Sync;

    fn deref(&self) -> &Self::Target {
        &*self.erased
    }
}

impl From<ErasedFunction> for Function {
    fn from(erased: ErasedFunction) -> Self {
        Function::erased(erased)
    }
}

/// A user closure over machine scalars, kept in its own signature.
///
/// The set is closed on purpose: each arm is one calling convention the batch
/// machine's interpreters know how to invoke from a register bank. A closure
/// outside it is still a perfectly good [`Function`]; it just has no scalar
/// form, and an expression that calls it is evaluated by the tree-walker.
pub enum ScalarFn {
    /// `fn(i64) -> i64`.
    Int1(Box<dyn Fn(i64) -> i64 + Send + Sync>),
    /// `fn(i64, i64) -> i64`.
    Int2(Box<dyn Fn(i64, i64) -> i64 + Send + Sync>),
    /// `fn(f64) -> f64`.
    Float1(Box<dyn Fn(f64) -> f64 + Send + Sync>),
    /// `fn(f64, f64) -> f64`.
    Float2(Box<dyn Fn(f64, f64) -> f64 + Send + Sync>),
}

impl ScalarFn {
    /// Recover the signature of a typed closure boxed as `Any`.
    ///
    /// A `Box<dyn Fn(A, B) -> R + Send + Sync>` is one concrete `'static` type
    /// per `(A, B, R)`, so downcasting it is an exact test of the signature:
    /// no specialization, no `unsafe`, and a closure with any other signature
    /// fails every arm and is reported as having no scalar form.
    fn from_any(typed: Box<dyn Any + Send + Sync>) -> Option<Self> {
        let typed = match typed.downcast::<Box<dyn Fn(i64, i64) -> i64 + Send + Sync>>() {
            Ok(f) => return Some(ScalarFn::Int2(*f)),
            Err(t) => t,
        };
        let typed = match typed.downcast::<Box<dyn Fn(i64) -> i64 + Send + Sync>>() {
            Ok(f) => return Some(ScalarFn::Int1(*f)),
            Err(t) => t,
        };
        let typed = match typed.downcast::<Box<dyn Fn(f64, f64) -> f64 + Send + Sync>>() {
            Ok(f) => return Some(ScalarFn::Float2(*f)),
            Err(t) => t,
        };
        match typed.downcast::<Box<dyn Fn(f64) -> f64 + Send + Sync>>() {
            Ok(f) => Some(ScalarFn::Float1(*f)),
            Err(_) => None,
        }
    }
}

impl ScalarFn {
    /// The address of this closure's `Box`, as one machine word.
    ///
    /// The batch machine's `host_call_*` helpers (`majit/bytecode.rs`) read
    /// the word back as `*const Box<dyn Fn(..)>` of the arm's exact signature
    /// and call through it; the two are a pair. The address is the `Box` field
    /// inside this `ScalarFn`, so it is valid for as long as the `Arc<ScalarFn>`
    /// that was lowered is held — [`LoweredF::host_fns`] holds it.
    ///
    /// [`LoweredF::host_fns`]: crate::majit::lower::LoweredF::host_fns
    pub fn entry_word(&self) -> i64 {
        (match self {
            ScalarFn::Int1(b) => b as *const _ as usize,
            ScalarFn::Int2(b) => b as *const _ as usize,
            ScalarFn::Float1(b) => b as *const _ as usize,
            ScalarFn::Float2(b) => b as *const _ as usize,
        }) as i64
    }
}

impl std::fmt::Debug for ScalarFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ScalarFn::Int1(_) => "ScalarFn::Int1",
            ScalarFn::Int2(_) => "ScalarFn::Int2",
            ScalarFn::Float1(_) => "ScalarFn::Float1",
            ScalarFn::Float2(_) => "ScalarFn::Float2",
        })
    }
}

pub trait IntoFunction<T> {
    fn into_function(self) -> Function;
}

impl IntoFunction<Function> for Function {
    fn into_function(self) -> Function {
        self
    }
}

impl IntoFunction<ErasedFunction> for ErasedFunction {
    fn into_function(self) -> Function {
        Function::erased(self)
    }
}

#[cfg(test)]
mod scalar_fn_tests {
    use super::*;

    fn scalar<T, F: IntoFunction<T>>(f: F) -> Option<String> {
        f.into_function().scalar().map(|s| format!("{s:?}"))
    }

    #[test]
    fn a_closure_over_machine_scalars_keeps_its_signature() {
        assert_eq!(
            scalar(|a: i64, b: i64| a + b).as_deref(),
            Some("ScalarFn::Int2")
        );
        assert_eq!(scalar(|a: i64| -a).as_deref(), Some("ScalarFn::Int1"));
        assert_eq!(
            scalar(|a: f64, b: f64| a * b).as_deref(),
            Some("ScalarFn::Float2")
        );
        assert_eq!(
            scalar(|a: f64| a.sqrt()).as_deref(),
            Some("ScalarFn::Float1")
        );
    }

    #[test]
    fn any_other_signature_has_no_scalar_form() {
        assert_eq!(
            scalar(|a: i64| -> Result<i64, ExecutionError> { Ok(a) }),
            None
        );
        assert_eq!(scalar(|_: &FunctionContext, a: i64| a), None);
        assert_eq!(scalar(|a: i64, b: f64| a as f64 + b), None);
        assert_eq!(scalar(|s: Arc<String>| s.len() as i64), None);
        assert_eq!(scalar(|| 1i64), None);
        assert_eq!(scalar(|a: i64| a > 0), None);
    }

    #[test]
    fn the_erased_form_still_answers_through_the_context() {
        let mut ctx = crate::Context::default();
        ctx.add_function("add", |a: i64, b: i64| a + b);
        let program = crate::Program::compile("add(2, 3)").unwrap();
        assert_eq!(program.execute(&ctx).unwrap(), Value::Int(5));
    }
}
