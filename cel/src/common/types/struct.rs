use std::collections::BTreeMap;

use crate::{common::types::Type, objects::Value};

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
