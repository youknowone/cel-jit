use std::{borrow::Cow, collections::BTreeMap, sync::Arc};

use crate::{
    common::{
        traits::{Indexer, Zeroer},
        types::{CelString, Type},
        value::Val,
    },
    objects::Value,
    ExecutionError,
};

/// A CEL struct value.
///
/// A struct has a type and a set of field values.
#[derive(Debug, Eq, PartialEq)]
pub struct Struct {
    r#type: Type,
    entries: BTreeMap<String, Value>,
}

impl Struct {
    /// Creates a new struct with the given name and no fields.
    pub fn new(name: String) -> Self {
        Self {
            r#type: Type::new_struct(name),
            entries: BTreeMap::default(),
        }
    }

    /// Returns the name of the struct type.
    pub fn name(&self) -> &str {
        self.r#type.name()
    }

    /// Returns the struct's CEL type.
    ///
    /// Unlike every other value family the type is per-instance, carrying the
    /// struct's own name, so overload matching cannot reach it through a
    /// constant.
    pub fn cel_type(&self) -> &Type {
        &self.r#type
    }

    /// Returns the value of the field with the given name, if it exists.
    pub fn field_value(&self, name: &str) -> Option<&Value> {
        self.entries.get(name)
    }

    /// Adds a field value to the struct.
    pub fn add_field_value(&mut self, name: String, value: Value) {
        self.entries.insert(name, value);
    }

    /// Returns a map of all field values in the struct.
    pub fn field_values(&self) -> BTreeMap<String, Value> {
        self.entries.clone()
    }

    /// Whether the struct carries no fields, which is its zero value.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Val for Struct {
    fn get_type(&self) -> &Type {
        &self.r#type
    }

    fn clone_as_boxed(&self) -> Box<dyn Val> {
        Box::new(Self {
            r#type: Type::new_struct(self.name().to_owned()),
            entries: self.entries.clone(),
        })
    }

    fn as_indexer(&self) -> Option<&dyn crate::common::traits::Indexer> {
        Some(self)
    }

    fn into_indexer(self: Box<Self>) -> Option<Box<dyn crate::common::traits::Indexer>> {
        Some(self)
    }

    fn as_zeroer(&self) -> Option<&dyn Zeroer> {
        Some(self)
    }

    fn equals(&self, other: &dyn Val) -> bool {
        other
            .downcast_ref::<Struct>()
            .is_some_and(|other| self == other)
    }
}

impl Indexer for Struct {
    fn get<'a>(&'a self, idx: &dyn Val) -> Result<Cow<'a, dyn Val>, crate::ExecutionError> {
        if let Some(field) = idx.downcast_ref::<CelString>() {
            self.field_value(field.inner())
                .ok_or_else(|| ExecutionError::NoSuchKey(Arc::new(String::from(field.inner()))))
                .and_then(|v| Ok(Cow::<dyn Val>::Owned(v.clone().try_into()?)))
        } else {
            Err(ExecutionError::UnsupportedIndex(
                idx.try_into()?,
                (self as &dyn Val).try_into()?,
            ))
        }
    }

    fn steal(self: Box<Self>, idx: &dyn Val) -> Result<Box<dyn Val>, crate::ExecutionError> {
        self.get(idx).map(Cow::into_owned)
    }
}

impl Zeroer for Struct {
    fn is_zero_value(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::common::types::CelStruct;
    use crate::objects::Value;

    #[test]
    fn equality() {
        let mut s1 = CelStruct::new("foo".to_owned());
        s1.add_field_value("bar".to_owned(), Value::Bool(true));
        let mut s2 = CelStruct::new("foo".to_owned());
        assert_ne!(s1, s2);
        s2.add_field_value("bar".to_owned(), Value::Bool(true));
        assert_eq!(s1, s2);
        s2.add_field_value("bar".to_owned(), Value::Bool(false));
        assert_ne!(s1, s2);
    }
}
