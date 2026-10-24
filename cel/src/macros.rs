#[macro_export]
macro_rules! impl_conversions {
    // Capture pairs separated by commas, where each pair is separated by =>
    ($($target_type:ty => $value_variant:path),* $(,)?) => {
        $(
            impl FromValue for $target_type {
                fn from_value(expr: &Value) -> Result<Self, ExecutionError> {
                    let unpacked = expr.unpack();
                    if let $value_variant(v) = unpacked {
                        Ok(v)
                    } else {
                        Err(ExecutionError::UnexpectedType {
                            got: format!("{:?}", expr),
                            want: stringify!($target_type).to_string(),
                        })
                    }
                }
            }

            impl FromValue for Option<$target_type> {
                fn from_value(expr: &Value) -> Result<Self, ExecutionError> {
                    let unpacked = expr.unpack();
                    match unpacked {
                        Value::Null => Ok(None),
                        $value_variant(v) => Ok(Some(v)),
                        _ => Err(ExecutionError::UnexpectedType {
                            got: format!("{:?}", expr),
                            want: stringify!($target_type).to_string(),
                        }),
                    }
                }
            }

            impl From<$target_type> for Value {
                fn from(value: $target_type) -> Self {
                    $value_variant(value)
                }
            }

            impl $crate::magic::IntoResolveResult for $target_type {
                fn into_resolve_result(self) -> ResolveResult {
                    Ok($value_variant(self))
                }
            }

            impl $crate::magic::IntoResolveResult for Result<$target_type, ExecutionError> {
                fn into_resolve_result(self) -> ResolveResult {
                    self.map($value_variant)
                }
            }

            impl<'a, 'context, 'call> FromContext<'a, 'context, 'call> for $target_type {
                fn from_context(ctx: &'a mut FunctionContext<'context, 'call>) -> Result<Self, ExecutionError>
                where
                    Self: Sized,
                {
                    arg_value_from_context(ctx).and_then(|v| FromValue::from_value(&v))
                }
            }
        )*
    }
}

#[macro_export]
macro_rules! impl_handler {
    () => {
        impl<F, R> IntoFunction<()> for F
        where
            F: Fn() -> R + 'static,
            R: IntoResolveResult + 'static,
        {
            fn into_function(self) -> Function {
                $crate::magic::handler0(self)
            }
        }

        impl<F, R> IntoFunction<(WithFunctionContext,)> for F
        where
            F: Fn(&FunctionContext) -> R + 'static,
            R: IntoResolveResult,
        {
            fn into_function(self) -> Function {
                Function::erased(Box::new(move |_ftx| self(_ftx).into_resolve_result()))
            }
        }
    };
    ($c1:ident) => {
        impl<F, $c1, R> IntoFunction<($c1,)> for F
        where
            F: Fn($c1) -> R + 'static,
            $c1: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call> + 'static,
            R: IntoResolveResult + 'static,
        {
            fn into_function(self) -> Function {
                $crate::magic::handler1::<F, $c1, R>(self)
            }
        }

        impl<F, $c1, R> IntoFunction<(WithFunctionContext, $c1)> for F
        where
            F: Fn(&FunctionContext, $c1) -> R + 'static,
            $c1: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call>,
            R: IntoResolveResult,
        {
            fn into_function(self) -> Function {
                Function::erased(Box::new(move |_ftx| {
                    let arg = $c1::from_context(_ftx)?;
                    self(_ftx, arg).into_resolve_result()
                }))
            }
        }
    };
    ($c1:ident, $c2:ident) => {
        impl<F, $c1, $c2, R> IntoFunction<($c1, $c2)> for F
        where
            F: Fn($c1, $c2) -> R + 'static,
            $c1: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call> + 'static,
            $c2: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call> + 'static,
            R: IntoResolveResult + 'static,
        {
            fn into_function(self) -> Function {
                $crate::magic::handler2::<F, $c1, $c2, R>(self)
            }
        }

        impl<F, $c1, $c2, R> IntoFunction<(WithFunctionContext, $c1, $c2)> for F
        where
            F: Fn(&FunctionContext, $c1, $c2) -> R + 'static,
            $c1: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call>,
            $c2: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call>,
            R: IntoResolveResult,
        {
            fn into_function(self) -> Function {
                Function::erased(Box::new(move |_ftx| {
                    let a = $c1::from_context(_ftx)?;
                    let b = $c2::from_context(_ftx)?;
                    self(_ftx, a, b).into_resolve_result()
                }))
            }
        }
    };
    ($($t:ty),+) => {
        pastey::paste! {
            impl<F, $($t,)* R> IntoFunction<($($t,)*)> for F
            where
                F: Fn($($t,)*) -> R + 'static,
                $($t: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call> + 'static,)*
                R: IntoResolveResult + 'static,
            {
                fn into_function(self) -> Function {
                    Function::erased(Box::new(move |_ftx| {
                        $(
                            let [<arg_ $t:lower>] = $t::from_context(_ftx)?;
                        )*
                        self($([<arg_ $t:lower>],)*).into_resolve_result()
                    }))
                }
            }

            impl<F, $($t,)* R> IntoFunction<(WithFunctionContext, $($t,)*)> for F
            where
                F: Fn(&FunctionContext, $($t,)*) -> R + 'static,
                $($t: for<'a, 'context, 'call> $crate::FromContext<'a, 'context, 'call>,)*
                R: IntoResolveResult,
            {
                fn into_function(self) -> Function {
                    Function::erased(Box::new(move |_ftx| {
                        $(
                            let [<arg_ $t:lower>] = $t::from_context(_ftx)?;
                        )*
                        self(_ftx, $([<arg_ $t:lower>],)*).into_resolve_result()
                    }))
                }
            }
        }
    };
}

pub(crate) use impl_conversions;
pub(crate) use impl_handler;
