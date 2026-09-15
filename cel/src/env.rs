use crate::common::{
    decls::FunctionDecl,
    functions::Function,
    types::{self, Type},
};
use crate::objects::Value;
#[cfg(feature = "structs")]
use crate::{common::types::CelStruct, ExecutionError};
use std::{
    collections::{
        btree_map::Entry::{Occupied, Vacant},
        BTreeMap,
    },
    sync::Arc,
};

/// An environment for the CEL execution.
///
/// This is where functions, overloads, and custom structs are defined.
///
/// # Example
///
/// ## Custom Structs
///
/// You can define custom struct types that can be instantiated from CEL expressions.
///
/// ```
/// #[cfg(feature = "structs")]
/// {
/// use cel::{Env, StructDef, Value, common::types};
///
/// let mut env = Env::stdlib();
/// env.add_struct(
///     StructDef::new("cel.MyStruct".to_owned())
///         .add_field("some_field".to_owned(), types::STRING_TYPE)
///         .add_field_with_default("with_default".to_owned(), Value::from("default_value"))
/// );
/// }
/// ```
///
/// ## Function Overloads
///
/// You can add custom function overloads to the environment.
///
/// ```
/// use cel::{Env, Value, common::types};
///
/// let mut env = Env::stdlib();
///
/// // Define a function that takes an integer and returns its square.
/// env.add_overload("square", "int_square", vec![types::INT_TYPE], |args| {
///     match args[0] {
///         Value::Int(v) => Ok(Value::Int(v * v)),
///         _ => unreachable!("the overload declares a single int"),
///     }
/// }).unwrap();
/// ```
#[derive(Default)]
pub struct Env {
    functions: BTreeMap<String, FunctionDecl>,
    #[cfg(feature = "structs")]
    structs: BTreeMap<String, StructDef>,
}

impl Env {
    /// Returns the standard library environment, shared by every caller.
    ///
    /// [`Env`] is immutable once built — a [`Context`](crate::Context) only ever
    /// hands out `&Env` — and [`stdlib`](Self::stdlib) registers several hundred
    /// overloads, so building one per context is the dominant cost of
    /// `Context::default()`.
    pub fn shared_stdlib() -> Arc<Env> {
        thread_local! {
            static SHARED: Arc<Env> = Arc::new(Env::stdlib());
        }
        SHARED.with(Arc::clone)
    }

    /// Returns the standard library environment.
    ///
    /// This environment contains all the standard functions and types as defined by the
    /// CEL specification.
    pub fn stdlib() -> Env {
        let mut env = Env::default();
        types::bytes::stdlib(&mut env);
        types::double::stdlib(&mut env);
        types::r#dyn::stdlib(&mut env);
        types::int::stdlib(&mut env);
        types::list::stdlib(&mut env);
        types::map::stdlib(&mut env);
        types::optional::stdlib(&mut env);
        types::string::stdlib(&mut env);
        types::r#type::stdlib(&mut env);
        types::uint::stdlib(&mut env);

        #[cfg(feature = "chrono")]
        {
            types::duration::stdlib(&mut env);
            types::timestamp::stdlib(&mut env);
        }
        env
    }

    /// Adds a global function overload to the environment.
    ///
    /// The name is the function name (e.g., `_==_`, `size`).
    /// The id is the unique identifier for this overload (e.g., `equals_int64`).
    /// The args are the expected argument types.
    /// The op is the function implementation.
    #[allow(clippy::result_unit_err)]
    pub fn add_overload(
        &mut self,
        name: &str,
        id: &str,
        args: Vec<types::Type>,
        op: Function,
    ) -> Result<(), ()> {
        match self.functions.entry(name.to_owned()) {
            Vacant(vacant_entry) => {
                let mut value = FunctionDecl::new(name);
                value.add_overload(id.to_string(), false, args, op)?;
                vacant_entry.insert(value);
                Ok(())
            }
            Occupied(occupied_entry) => {
                occupied_entry
                    .into_mut()
                    .add_overload(id.to_string(), false, args, op)
            }
        }
    }

    /// Whether any global overload is declared under `name`, for any argument
    /// types. [`Env::find_overload`] is the answer for one call; this is the
    /// answer for a NAME, which is what a caller that has no argument values
    /// yet can ask.
    pub fn declares_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Finds a global function overload that matches the given name and arguments.
    pub fn find_overload(&self, name: &str, args: &[Value]) -> Option<Function> {
        match self.functions.get(name) {
            None => None,
            Some(fn_decl) => fn_decl.find_overload(false, args),
        }
    }

    /// [`Env::find_overload`] for a namespaced name, without joining the two
    /// parts into a `String` the lookup would immediately discard.
    pub(crate) fn find_qualified_overload(
        &self,
        prefix: &str,
        name: &str,
        args: &[Value],
    ) -> Option<Function> {
        crate::common::get_qualified(&self.functions, prefix, name)
            .and_then(|fn_decl| fn_decl.find_overload(false, args))
    }

    /// Adds a member function overload to the environment.
    ///
    /// A member function is one that is called using the receiver syntax (e.g., `x.matches(y)`).
    /// The name is the function name.
    /// The id is the unique identifier for this overload.
    /// The target is the type of the receiver.
    /// The args are the expected argument types (excluding the receiver).
    /// The op is the function implementation.
    #[allow(clippy::result_unit_err)]
    pub fn add_member_overload(
        &mut self,
        name: &str,
        id: &str,
        target: Type,
        args: Vec<types::Type>,
        op: Function,
    ) -> Result<(), ()> {
        let mut args = args;
        args.insert(0, target);
        match self.functions.entry(name.to_owned()) {
            Vacant(vacant_entry) => {
                let mut value = FunctionDecl::new(name);
                value.add_overload(id.to_string(), true, args, op)?;
                vacant_entry.insert(value);
                Ok(())
            }
            Occupied(occupied_entry) => {
                occupied_entry
                    .into_mut()
                    .add_overload(id.to_string(), true, args, op)
            }
        }
    }

    /// Finds a member function overload that matches the given name and arguments.
    pub(crate) fn find_member_overload(&self, name: &str, args: &[Value]) -> Option<Function> {
        match self.functions.get(name) {
            None => None,
            Some(fn_decl) => fn_decl.find_overload(true, args),
        }
    }

    /// Adds a custom struct definition to the environment.
    #[cfg(feature = "structs")]
    pub fn add_struct(&mut self, def: StructDef) {
        self.structs.insert(def.name.clone(), def);
    }

    /// Finds a struct definition by name.
    #[cfg(feature = "structs")]
    pub(crate) fn find_struct(&self, name: &str) -> Option<&StructDef> {
        self.structs.get(name)
    }
}

/// A definition for a custom struct type.
///
/// A struct definition defines the name of the struct, its fields, and any default values
/// for those fields. Struct definitions are added to an [`Env`] to allow them to be
/// instantiated from CEL expressions.
///
/// # Example
///
/// ```
/// use cel::{Env, StructDef, Value, common::types};
///
/// let mut env = Env::stdlib();
/// env.add_struct(
///     StructDef::new("MyStruct".to_owned())
///         .add_field("some_field".to_owned(), types::STRING_TYPE)
///         .add_field_with_default("with_default".to_owned(), Value::from("default_value"))
/// );
/// ```
#[cfg(feature = "structs")]
pub struct StructDef {
    name: String,
    fields: BTreeMap<String, Type>,
    defaults: BTreeMap<String, Value>,
}

#[cfg(feature = "structs")]
impl StructDef {
    /// Creates a new struct definition with the given name.
    ///
    /// The name should be the fully qualified name of the struct as it will be
    /// referenced in CEL expressions (e.g., `cel.MyStruct`).
    pub fn new(name: String) -> Self {
        Self {
            name,
            fields: Default::default(),
            defaults: Default::default(),
        }
    }

    /// Adds a field to the struct definition.
    ///
    /// This method adds a field with the given name and type. When the struct is
    /// instantiated in a CEL expression, this field must be provided unless it
    /// has a default value (see [`add_field_with_default`](Self::add_field_with_default)).
    pub fn add_field(self, field: String, t: Type) -> Self {
        self.insert_field(field, t, None)
    }

    /// Adds a field to the struct definition with a default value.
    ///
    /// This method adds a field with the given name and a default value. The type
    /// of the field is automatically inferred from the default value. When the
    /// struct is instantiated in a CEL expression, this field may be omitted, in
    /// which case the default value will be used.
    pub fn add_field_with_default(self, field: String, default: Value) -> Self {
        self.insert_field(field, types::type_of(&default), Some(default))
    }

    /// Internal method to insert a field into the struct definition.
    fn insert_field(self, field: String, t: Type, default: Option<Value>) -> Self {
        let mut def = self;
        def.fields.insert(field.clone(), t);
        if let Some(default) = default {
            def.defaults.insert(field, default);
        }
        def
    }

    /// Creates a new instance of the struct with the given field values.
    ///
    /// This method is used internally by the CEL execution engine to instantiate
    /// a struct from a CEL expression.
    ///
    /// # Errors
    ///
    /// Missing fields will be populated with their default values if defined.
    /// Returns an error if:
    /// - A field is missing and has no default value.
    /// - A field's type does not match the type in the definition.
    /// - An unknown field name is provided.
    #[cfg(feature = "structs")]
    pub(crate) fn new_struct(
        &self,
        fields: BTreeMap<String, Value>,
    ) -> Result<CelStruct, ExecutionError> {
        let mut s = CelStruct::new(self.name.clone());
        let mut fields = fields;
        for (field, default) in &self.defaults {
            if let Some(value) = fields.remove(field) {
                s.add_field_value(field.clone(), value);
            } else {
                s.add_field_value(field.clone(), default.clone());
            }
        }
        for (field, value) in fields {
            match self.fields.get(&field) {
                Some(t) => {
                    if *t != types::type_of(&value) {
                        return Err(ExecutionError::UnexpectedType {
                            got: types::type_name(&value),
                            want: format!("{} for field {field} in {}", t.name(), self.name),
                        });
                    }
                    s.add_field_value(field, value);
                }
                None => {
                    return Err(ExecutionError::NoSuchKey(std::sync::Arc::new(format!(
                        "field `{field}` on struct `{}`",
                        self.name
                    ))))
                }
            }
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_env_default() {
        let _ = Env::default();
    }

    /// Two `Context::default()`s must share one stdlib environment. Rebuilding
    /// it registers every overload again, which was the whole cost of the call.
    #[test]
    fn shared_stdlib_hands_out_one_environment() {
        assert!(Arc::ptr_eq(&Env::shared_stdlib(), &Env::shared_stdlib()));
        // ... and it is the same environment `stdlib()` builds.
        let own = Env::stdlib();
        let shared = Env::shared_stdlib();
        assert_eq!(
            own.functions.keys().collect::<Vec<_>>(),
            shared.functions.keys().collect::<Vec<_>>()
        );
    }
}
