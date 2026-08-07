use crate::common::ast::{operators, CallExpr, ComprehensionExpr, EntryExpr, Expr, LiteralValue};
use crate::common::types::*;
use crate::common::value::Val;
use crate::context::Context;
use crate::ExecutionError::NoSuchOverload;
use crate::{ExecutionError, Expression, FunctionContext};
#[cfg(feature = "chrono")]
use chrono::TimeZone;
use std::any::Any;
use std::borrow::{Borrow, Cow};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::{Infallible, TryFrom, TryInto};
use std::fmt::{Debug, Display, Formatter};
use std::ops;
use std::ops::Deref;
use std::sync::Arc;
#[cfg(feature = "chrono")]
use std::sync::LazyLock;

/// Timestamp values are limited to the range of values which can be serialized as a string:
/// `["0001-01-01T00:00:00Z", "9999-12-31T23:59:59.999999999Z"]`. Since the max is a smaller
/// and the min is a larger timestamp than what is possible to represent with
/// [`chrono::DateTime`], we need to perform our own spec-compliant overflow checks.
///
/// <https://github.com/google/cel-spec/blob/master/doc/langdef.md#overflow>
#[cfg(feature = "chrono")]
static MAX_TIMESTAMP: LazyLock<chrono::DateTime<chrono::FixedOffset>> = LazyLock::new(|| {
    let naive = chrono::NaiveDate::from_ymd_opt(9999, 12, 31)
        .unwrap()
        .and_hms_nano_opt(23, 59, 59, 999_999_999)
        .unwrap();
    chrono::FixedOffset::east_opt(0)
        .unwrap()
        .from_utc_datetime(&naive)
});

#[cfg(feature = "chrono")]
static MIN_TIMESTAMP: LazyLock<chrono::DateTime<chrono::FixedOffset>> = LazyLock::new(|| {
    let naive = chrono::NaiveDate::from_ymd_opt(1, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap();
    chrono::FixedOffset::east_opt(0)
        .unwrap()
        .from_utc_datetime(&naive)
});

/// How a [`Map`] holds its entries -- the map counterpart of [`ListStorage`].
#[derive(Clone)]
pub enum MapStorage {
    /// An owned table of boxed entries.
    Object(Arc<HashMap<Key, Value>>),
    /// One row of a record batch. The field names and the column banks live in
    /// `schema` and are shared with every other row, so this row is an index
    /// into them and a field is boxed only when it is read.
    Record {
        schema: Arc<RecordSchema>,
        index: usize,
    },
}

/// The field names and column banks shared by every row of a record batch. One
/// of these is built per output, so a row costs an index into it rather than a
/// table of its own.
pub struct RecordSchema {
    keys: Vec<Key>,
    columns: Vec<ValueColumn>,
}

/// A string column: order-preserving ranks into the batch's distinct strings,
/// which are interned once so a value costs a reference count. The rank
/// encoding exists so a string column compares as an integer, and rebuilding
/// the `String` per value gave that back.
pub struct StrBank {
    codes: Arc<[i64]>,
    interned: Arc<[Arc<String>]>,
}

impl StrBank {
    pub fn new(codes: Arc<[i64]>, interned: Arc<[Arc<String>]>) -> StrBank {
        StrBank { codes, interned }
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    fn value_at(&self, index: usize) -> Value {
        Value::String(self.interned[self.codes[index] as usize].clone())
    }
}

/// One column of a batch, in the representation the batch already holds it in.
/// It is the element source for both an unnamed element list and a named
/// record field, so every bank has an unboxed form in exactly one place --
/// a per-bank asymmetry here is what let a string column cost 19.5 allocations
/// per row while the int column cost 1.
///
/// Cloning one is a reference count per bank, not a copy of the buffer.
#[derive(Clone)]
pub enum ValueColumn {
    /// Raw words, read through `bank`.
    Scalar { bank: ScalarBank, words: Arc<[i64]> },
    /// Ranks into an interned string table.
    Str(Arc<StrBank>),
}

/// How a [`ValueColumn::Scalar`]'s raw words decode. One variant per bank a batch
/// column can carry, so [`Column`] needs no boxed fallback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScalarBank {
    Int,
    UInt,
    Bool,
    Float,
    /// Nanoseconds since the Unix epoch.
    #[cfg(feature = "chrono")]
    Timestamp,
    /// Nanoseconds.
    #[cfg(feature = "chrono")]
    Duration,
}

impl ValueColumn {
    pub fn value_at(&self, index: usize) -> Value {
        match self {
            ValueColumn::Scalar { bank, words } => {
                let word = words[index];
                match bank {
                    ScalarBank::Int => Value::Int(word),
                    ScalarBank::UInt => Value::UInt(word as u64),
                    ScalarBank::Bool => Value::Bool(word != 0),
                    ScalarBank::Float => Value::Float(f64::from_bits(word as u64)),
                    #[cfg(feature = "chrono")]
                    ScalarBank::Timestamp => Value::Timestamp(
                        chrono::DateTime::from_timestamp_nanos(word).fixed_offset(),
                    ),
                    #[cfg(feature = "chrono")]
                    ScalarBank::Duration => Value::Duration(chrono::Duration::nanoseconds(word)),
                }
            }
            ValueColumn::Str(bank) => bank.value_at(index),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ValueColumn::Scalar { words, .. } => words.len(),
            ValueColumn::Str(bank) => bank.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl RecordSchema {
    /// `keys` names one column of `columns`, and every column holds the same
    /// number of rows -- the two are what makes an index a whole record.
    pub fn new(keys: Vec<Key>, columns: Vec<ValueColumn>) -> RecordSchema {
        assert_eq!(
            keys.len(),
            columns.len(),
            "a record schema names one column per key"
        );
        assert!(
            columns.windows(2).all(|w| w[0].len() == w[1].len()),
            "a record schema's columns must agree on the row count"
        );
        RecordSchema { keys, columns }
    }

    pub fn field_count(&self) -> usize {
        self.keys.len()
    }

    /// How many records the columns hold.
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, ValueColumn::len)
    }

    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// A record carries a handful of fields, so a scan over the shared names
    /// beats hashing and needs no table of its own.
    fn position(&self, key: &(dyn AsKeyRef + '_)) -> Option<usize> {
        let key = key.as_keyref();
        self.keys.iter().position(|k| k.as_keyref() == key)
    }
}

#[derive(Clone)]
pub struct Map {
    storage: MapStorage,
}

impl PartialOrd for Map {
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        None
    }
}

impl Map {
    /// A map over an owned table of boxed entries.
    pub fn object(map: Arc<HashMap<Key, Value>>) -> Map {
        Map {
            storage: MapStorage::Object(map),
        }
    }

    /// Record `index` of `schema`, which costs a reference count and no
    /// allocation at all.
    pub fn record(schema: Arc<RecordSchema>, index: usize) -> Map {
        debug_assert!(index < schema.rows(), "record index is past the columns");
        Map {
            storage: MapStorage::Record { schema, index },
        }
    }

    pub fn storage(&self) -> &MapStorage {
        &self.storage
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            MapStorage::Object(map) => map.len(),
            MapStorage::Record { schema, .. } => schema.field_count(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn contains_key(&self, key: &(dyn AsKeyRef + '_)) -> bool {
        match &self.storage {
            MapStorage::Object(map) => map.contains_key(key),
            MapStorage::Record { schema, .. } => schema.position(key).is_some(),
        }
    }

    /// The value corresponding to the key, boxed on the way out when the
    /// strategy does not already hold it. Implicitly converts between int and
    /// uint keys.
    pub fn get(&self, key: &(dyn AsKeyRef + '_)) -> Option<Cow<'_, Value>> {
        self.get_exact(key).or_else(|| {
            // Also check keys that are cross type comparable.
            let keyref = key.as_keyref();
            match keyref {
                KeyRef::Int(k) => self.get_exact(&KeyRef::Uint(u64::try_from(k).ok()?)),
                KeyRef::Uint(k) => self.get_exact(&KeyRef::Int(i64::try_from(k).ok()?)),
                _ => None,
            }
        })
    }

    fn get_exact(&self, key: &(dyn AsKeyRef + '_)) -> Option<Cow<'_, Value>> {
        match &self.storage {
            MapStorage::Object(map) => map.get(key).map(Cow::Borrowed),
            MapStorage::Record { schema, index } => {
                let field = schema.position(key)?;
                Some(Cow::Owned(schema.columns[field].value_at(*index)))
            }
        }
    }

    /// The entries. Keys are borrowed from whichever strategy owns them, and a
    /// value is boxed only when the strategy does not already hold one.
    pub fn iter(&self) -> MapIter<'_> {
        match &self.storage {
            MapStorage::Object(map) => MapIter::Object(map.iter()),
            MapStorage::Record { schema, index } => MapIter::Record {
                schema,
                index: *index,
                field: 0,
            },
        }
    }

    /// The entries as the owned table the rest of the language expects. Free
    /// for [`MapStorage::Object`], and the point at which a record row pays for
    /// the representation it was avoiding.
    pub fn to_hashmap(&self) -> HashMap<Key, Value> {
        match &self.storage {
            MapStorage::Object(map) => (**map).clone(),
            MapStorage::Record { .. } => self
                .iter()
                .map(|(k, v)| (k.clone(), v.into_owned()))
                .collect(),
        }
    }
}

pub enum MapIter<'a> {
    Object(std::collections::hash_map::Iter<'a, Key, Value>),
    Record {
        schema: &'a RecordSchema,
        index: usize,
        field: usize,
    },
}

impl<'a> Iterator for MapIter<'a> {
    type Item = (&'a Key, Cow<'a, Value>);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            MapIter::Object(it) => it.next().map(|(k, v)| (k, Cow::Borrowed(v))),
            MapIter::Record {
                schema,
                index,
                field,
            } => {
                let at = *field;
                let key = schema.keys.get(at)?;
                *field += 1;
                Some((key, Cow::Owned(schema.columns[at].value_at(*index))))
            }
        }
    }
}

/// Entry-wise, so two maps are equal when they hold equal entries whatever
/// strategy each of them uses.
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        match (&self.storage, &other.storage) {
            (MapStorage::Object(a), MapStorage::Object(b)) => a == b,
            _ => {
                self.len() == other.len()
                    && self
                        .iter()
                        .all(|(k, v)| other.get(k).is_some_and(|o| *o == *v))
            }
        }
    }
}

impl Debug for Map {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Ord, Clone, PartialOrd)]
pub enum Key {
    Int(i64),
    Uint(u64),
    Bool(bool),
    String(Arc<String>),
}

impl From<CelMapKey> for Key {
    fn from(value: CelMapKey) -> Self {
        match value {
            CelMapKey::Bool(b) => b.into_inner().into(),
            CelMapKey::Int(i) => i.into_inner().into(),
            CelMapKey::String(s) => s.into_inner().into(),
            CelMapKey::UInt(u) => u.into_inner().into(),
        }
    }
}

impl From<Key> for CelMapKey {
    fn from(key: Key) -> Self {
        match key {
            Key::Int(i) => CelMapKey::from(i),
            Key::Uint(u) => CelMapKey::from(u),
            Key::Bool(b) => CelMapKey::from(b),
            Key::String(s) => CelMapKey::from(s.as_str()),
        }
    }
}

/// A borrowed version of [`Key`] that avoids allocating for lookups.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum KeyRef<'a> {
    Int(i64),
    Uint(u64),
    Bool(bool),
    String(&'a str),
}

/// Trait for converting to a borrowed [`KeyRef`] for efficient lookups.
pub trait AsKeyRef {
    fn as_keyref(&self) -> KeyRef<'_>;
}

impl AsKeyRef for Key {
    fn as_keyref(&self) -> KeyRef<'_> {
        match self {
            Key::Int(i) => KeyRef::Int(*i),
            Key::Uint(u) => KeyRef::Uint(*u),
            Key::Bool(b) => KeyRef::Bool(*b),
            Key::String(s) => KeyRef::String(s.as_str()),
        }
    }
}

impl<'a> AsKeyRef for KeyRef<'a> {
    fn as_keyref(&self) -> KeyRef<'a> {
        *self
    }
}

/// Trait object implementations for `dyn AsKeyRef` to enable hashing and comparison.
impl<'a> PartialEq for dyn AsKeyRef + 'a {
    fn eq(&self, other: &Self) -> bool {
        self.as_keyref().eq(&other.as_keyref())
    }
}

impl<'a> Eq for dyn AsKeyRef + 'a {}

impl<'a> std::hash::Hash for dyn AsKeyRef + 'a {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_keyref().hash(state)
    }
}

impl<'a> PartialOrd for dyn AsKeyRef + 'a {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for dyn AsKeyRef + 'a {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_keyref().cmp(&other.as_keyref())
    }
}

/// Implement `Borrow<dyn AsKeyRef>` for `Key` to enable efficient lookups.
impl<'a> Borrow<dyn AsKeyRef + 'a> for Key {
    fn borrow(&self) -> &(dyn AsKeyRef + 'a) {
        self
    }
}

/// Implement conversions from primitive types to [`Key`]
impl From<String> for Key {
    fn from(v: String) -> Self {
        Key::String(v.into())
    }
}

impl From<Arc<String>> for Key {
    fn from(v: Arc<String>) -> Self {
        Key::String(v)
    }
}

impl<'a> From<&'a str> for Key {
    fn from(v: &'a str) -> Self {
        Key::String(Arc::new(v.into()))
    }
}

impl From<bool> for Key {
    fn from(v: bool) -> Self {
        Key::Bool(v)
    }
}

impl From<i64> for Key {
    fn from(v: i64) -> Self {
        Key::Int(v)
    }
}

impl From<i32> for Key {
    fn from(v: i32) -> Self {
        Key::Int(v as i64)
    }
}

impl From<u64> for Key {
    fn from(v: u64) -> Self {
        Key::Uint(v)
    }
}

impl From<u32> for Key {
    fn from(v: u32) -> Self {
        Key::Uint(v as u64)
    }
}

impl serde::Serialize for Key {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Key::Int(v) => v.serialize(serializer),
            Key::Uint(v) => v.serialize(serializer),
            Key::Bool(v) => v.serialize(serializer),
            Key::String(v) => v.serialize(serializer),
        }
    }
}

impl Display for Key {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Key::Int(v) => write!(f, "{v}"),
            Key::Uint(v) => write!(f, "{v}"),
            Key::Bool(v) => write!(f, "{v}"),
            Key::String(v) => write!(f, "{v}"),
        }
    }
}

/// Implement conversions from [`Key`] into [`Value`]
impl TryInto<Key> for Value {
    type Error = Value;

    #[inline(always)]
    fn try_into(self) -> Result<Key, Self::Error> {
        match self {
            Value::Int(v) => Ok(Key::Int(v)),
            Value::UInt(v) => Ok(Key::Uint(v)),
            Value::String(v) => Ok(Key::String(v)),
            Value::Bool(v) => Ok(Key::Bool(v)),
            _ => Err(self),
        }
    }
}

/// Implement conversions from [`KeyRef`] into [`Value`]
impl<'a> TryFrom<&'a Value> for KeyRef<'a> {
    type Error = Value;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Int(v) => Ok(KeyRef::Int(*v)),
            Value::UInt(v) => Ok(KeyRef::Uint(*v)),
            Value::String(v) => Ok(KeyRef::String(v.as_str())),
            Value::Bool(v) => Ok(KeyRef::Bool(*v)),
            _ => Err(value.clone()),
        }
    }
}

// Implement conversion from HashMap<K, V> into CelMap
impl<K: Into<Key>, V: Into<Value>> From<HashMap<K, V>> for Map {
    fn from(map: HashMap<K, V>) -> Self {
        let mut new_map = HashMap::with_capacity(map.len());
        for (k, v) in map {
            new_map.insert(k.into(), v.into());
        }
        Map::object(Arc::new(new_map))
    }
}

/// Equality helper for [`Opaque`] values.
///
/// Implementors define how two values of the same runtime type compare for
/// equality when stored as [`Value::Opaque`].
///
/// You normally don't implement this trait manually. It is automatically
/// provided for any `T: Eq + PartialEq + Any + Opaque` (see the blanket impl
/// below). The runtime will first ensure the two values have the same
/// [`Opaque::runtime_type_name`], and only then attempt a downcast and call
/// `Eq::eq`.
pub trait OpaqueEq {
    /// Compare with another [`Opaque`] erased value.
    ///
    /// Implementations should return `false` if `other` does not have the same
    /// runtime type, or if it cannot be downcast to the concrete type of `self`.
    fn opaque_eq(&self, other: &dyn Opaque) -> bool;
}

impl<T> OpaqueEq for T
where
    T: Eq + PartialEq + Any + Opaque,
{
    fn opaque_eq(&self, other: &dyn Opaque) -> bool {
        if self.runtime_type_name() != other.runtime_type_name() {
            return false;
        }
        if let Some(other) = other.downcast_ref::<T>() {
            self.eq(other)
        } else {
            false
        }
    }
}

/// Helper trait to obtain a `&dyn Debug` view.
///
/// This is auto-implemented for any `T: Debug` and is used by the runtime to
/// format [`Opaque`] values without knowing their concrete type.
pub trait AsDebug {
    /// Returns `self` as a `&dyn Debug` trait object.
    fn as_debug(&self) -> &dyn Debug;
}

impl<T> AsDebug for T
where
    T: Debug,
{
    fn as_debug(&self) -> &dyn Debug {
        self
    }
}

/// Trait for user-defined opaque values stored inside [`Value::Opaque`].
///
/// Implement this trait for types that should participate in CEL evaluation as
/// opaque/user-defined values. An opaque value:
/// - must report a stable runtime type name via [`Opaque::runtime_type_name`];
/// - participates in equality via the blanket [`OpaqueEq`] implementation;
/// - can be formatted via [`AsDebug`];
/// - must be thread-safe (`Send + Sync`).
///
/// When the `json` feature is enabled you may optionally provide a JSON
/// representation for diagnostics, logging or interop. Returning `None` keeps the
/// value non-serializable for JSON.
///
/// Example
/// ```rust
/// use std::fmt::{Debug, Formatter, Result as FmtResult};
/// use std::sync::Arc;
/// use cel::objects::{Opaque, Value};
///
/// #[derive(Eq, PartialEq)]
/// struct MyId(u64);
///
/// impl Debug for MyId {
///     fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult { write!(f, "MyId({})", self.0) }
/// }
///
/// impl Opaque for MyId {
///     fn runtime_type_name(&self) -> &str { "example.MyId" }
/// }
///
/// // Values of `MyId` can now be wrapped in `Value::Opaque` and compared.
/// let a = Value::Opaque(Arc::new(MyId(7)));
/// let b = Value::Opaque(Arc::new(MyId(7)));
/// assert_eq!(a, b);
/// ```
pub trait Opaque: Any + OpaqueEq + AsDebug + Send + Sync {
    /// Returns a stable, fully-qualified type name for this value's runtime type.
    ///
    /// This name is used to check type compatibility before attempting downcasts
    /// during equality checks and other operations. It should be stable across
    /// versions and unique within your application or library (e.g., a package
    /// qualified name like `my.pkg.Type`).
    fn runtime_type_name(&self) -> &str;

    /// Optional JSON representation (requires the `json` feature).
    ///
    /// The default implementation returns `None`, indicating that the value
    /// cannot be represented as JSON.
    #[cfg(feature = "json")]
    fn json(&self) -> Option<serde_json::Value> {
        None
    }
}

impl dyn Opaque {
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        let any: &dyn Any = self;
        any.downcast_ref()
    }
}

struct OpaqueVal {
    r#type: Type,
    val: Arc<dyn Opaque>,
}

impl Debug for OpaqueVal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "OpaqueVal<{}>", self.val.runtime_type_name())
    }
}

impl Val for OpaqueVal {
    fn get_type(&self) -> &Type {
        &self.r#type
    }

    fn equals(&self, other: &dyn Val) -> bool {
        if other.get_type() != self.get_type() {
            false
        } else {
            match other.downcast_ref::<OpaqueVal>() {
                None => false,
                Some(other) => self.val.opaque_eq(other.val.deref()),
            }
        }
    }

    fn clone_as_boxed(&self) -> Box<dyn Val> {
        Box::new(Self {
            r#type: Type::new_opaque_type(self.val.runtime_type_name().to_owned()),
            val: self.val.clone(),
        })
    }
}

impl OpaqueVal {
    fn new(val: Arc<dyn Opaque>) -> Self {
        Self {
            r#type: Type::new_opaque_type(val.runtime_type_name().to_owned()),
            val,
        }
    }

    fn clone_inner(&self) -> Arc<dyn Opaque> {
        self.val.clone()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct OptionalValue {
    value: Option<Value>,
}

impl OptionalValue {
    pub fn of(value: Value) -> Self {
        OptionalValue { value: Some(value) }
    }
    pub fn none() -> Self {
        OptionalValue { value: None }
    }
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    pub(crate) fn inner(&self) -> Option<&Value> {
        self.value.as_ref()
    }
}

impl Opaque for OptionalValue {
    fn runtime_type_name(&self) -> &str {
        "optional_type"
    }
}

impl From<OptionalValue> for Option<Value> {
    fn from(value: OptionalValue) -> Self {
        value.value
    }
}

impl<'a> TryFrom<&'a Value> for &'a OptionalValue {
    type Error = ExecutionError;

    fn try_from(value: &'a Value) -> Result<Self, Self::Error> {
        match value {
            Value::Opaque(opaque) if opaque.runtime_type_name() == "optional_type" => opaque
                .downcast_ref::<OptionalValue>()
                .ok_or_else(|| ExecutionError::function_error("optional", "failed to downcast")),
            Value::Opaque(opaque) => Err(ExecutionError::UnexpectedType {
                got: opaque.runtime_type_name().to_string(),
                want: "optional_type".to_string(),
            }),
            v => Err(ExecutionError::UnexpectedType {
                got: v.type_of().to_string(),
                want: "optional_type".to_string(),
            }),
        }
    }
}

pub trait TryIntoValue {
    type Error: std::error::Error + 'static + Send + Sync;
    fn try_into_value(self) -> Result<Value, Self::Error>;
}

impl<T: serde::Serialize> TryIntoValue for T {
    type Error = crate::ser::SerializationError;
    fn try_into_value(self) -> Result<Value, Self::Error> {
        crate::ser::to_value(self)
    }
}
impl TryIntoValue for Value {
    type Error = Infallible;
    fn try_into_value(self) -> Result<Value, Self::Error> {
        Ok(self)
    }
}

/// How a list's elements are stored.
///
/// A list built element by element owns a `Vec<Value>` and is what every
/// caller has always had. A list produced by the batch tier instead names a
/// window into a buffer the whole batch shares: the elements are already laid
/// out contiguously and unboxed, so a row costs one small allocation and no
/// per-element write, and an element is boxed only when it is read.
///
/// This is PyPy's list-strategy arrangement — `objspace/std/listobject.py:1886`
/// `ObjectListStrategy`, `:1939` `IntegerListStrategy`, `:2043`
/// `FloatListStrategy`: a homogeneous list keeps unboxed storage, and boxing
/// happens on access rather than on construction.
///
/// Every consumer goes through the accessors below rather than matching a
/// variant, so adding a strategy does not reopen the call sites.
#[derive(Clone)]
pub enum ListStorage {
    /// Boxed elements, owned by this list.
    Object(Vec<Value>),
    /// An unboxed column. The element is boxed on the way out, which for every
    /// bank but a string is a [`Value`] that owns nothing.
    Column(ValueColumn),
    /// A record column. An element is a [`Map::record`], which costs a
    /// reference count and no allocation.
    Record(Arc<RecordSchema>),
}

impl ListStorage {
    /// How many elements the whole buffer holds — not how many any one list
    /// over it has; that is [`ListRef::len`].
    pub fn len(&self) -> usize {
        match self {
            ListStorage::Object(v) => v.len(),
            ListStorage::Column(c) => c.len(),
            ListStorage::Record(s) => s.rows(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The element at `index` in the whole buffer, boxed on the way out.
    fn element_at(&self, index: usize) -> Value {
        match self {
            ListStorage::Object(v) => v[index].clone(),
            ListStorage::Column(c) => c.value_at(index),
            ListStorage::Record(s) => Value::Map(Map::record(s.clone(), index)),
        }
    }
}

impl Default for ListStorage {
    fn default() -> Self {
        ListStorage::Object(Vec::new())
    }
}

impl From<Vec<Value>> for ListStorage {
    fn from(v: Vec<Value>) -> Self {
        ListStorage::Object(v)
    }
}

impl FromIterator<Value> for ListStorage {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> Self {
        ListStorage::Object(iter.into_iter().collect())
    }
}

/// One list: the window `storage[start .. start + len]`.
///
/// The window is what lets a whole batch of rows share ONE buffer. The rows of
/// a batch are laid out contiguously in the order the run produced them, so a
/// row is a pair of offsets into the buffer and a reference count — no
/// allocation at all, where a list that owned its own storage cost one per row.
///
/// This is PyPy's list-strategy arrangement — `objspace/std/listobject.py:1886`
/// `ObjectListStrategy`, `:1939` `IntegerListStrategy`, `:2043`
/// `FloatListStrategy`: a homogeneous list keeps unboxed storage, and boxing
/// happens on access rather than on construction.
///
/// Every consumer goes through the accessors below rather than matching a
/// storage variant, so adding a strategy does not reopen the call sites.
#[derive(Clone)]
pub struct ListRef {
    storage: Arc<ListStorage>,
    /// 32-bit so a `Value` stays 24 bytes. A buffer is checked against this
    /// bound when the window is built rather than truncated silently.
    start: u32,
    len: u32,
}

impl ListRef {
    /// The window `storage[start .. start + len]`.
    pub fn window(storage: Arc<ListStorage>, start: usize, len: usize) -> ListRef {
        assert!(
            start + len <= storage.len(),
            "list window {start}..{} runs past the {} element buffer",
            start + len,
            storage.len()
        );
        let (Ok(start), Ok(len)) = (u32::try_from(start), u32::try_from(len)) else {
            panic!(
                "a list buffer past {} elements cannot be windowed",
                u32::MAX
            );
        };
        ListRef {
            storage,
            start,
            len,
        }
    }

    /// The whole of `storage` as one list.
    pub fn whole(storage: Arc<ListStorage>) -> ListRef {
        let len = storage.len();
        ListRef::window(storage, 0, len)
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The element at `index`, boxed on the way out. `None` past the end.
    pub fn get(&self, index: usize) -> Option<Value> {
        (index < self.len()).then(|| self.storage.element_at(self.start as usize + index))
    }

    pub fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        (0..self.len()).map(move |i| {
            self.get(i)
                .expect("index below len() is in bounds by construction")
        })
    }

    /// The boxed elements. Free for a whole [`ListStorage::Object`], and the
    /// point at which an unboxed strategy pays for the representation the rest
    /// of the language expects.
    pub fn to_vec(&self) -> Vec<Value> {
        match self.whole_object() {
            Some(v) => v.clone(),
            None => self.iter().collect(),
        }
    }

    /// The elements as an owned `Vec`, moving them when this list is the sole
    /// owner of a boxed buffer it covers entirely.
    pub fn into_vec(mut self) -> Vec<Value> {
        if self.whole_object().is_some() {
            if let Some(ListStorage::Object(v)) = Arc::get_mut(&mut self.storage) {
                return std::mem::take(v);
            }
        }
        self.iter().collect()
    }

    pub fn contains(&self, needle: &Value) -> bool {
        match self.whole_object() {
            // A boxed buffer already holds the `Value`s, so comparing them in
            // place avoids the clone `iter()` owes an unboxed one.
            Some(v) => v.contains(needle),
            None => self.iter().any(|v| &v == needle),
        }
    }

    /// This list's elements followed by `other`'s.
    ///
    /// An unboxed strategy that is asked to grow becomes the boxed one first,
    /// which is what PyPy's strategies do when a list stops being homogeneous
    /// (`listobject.py` `switch_to_object_strategy`). A window onto a buffer it
    /// does not own entirely has to copy out — appending in place would grow
    /// the buffer every other list of the batch is reading.
    pub fn concat(mut self, other: &ListRef) -> ListRef {
        if self.whole_object().is_some() {
            if let Some(ListStorage::Object(items)) = Arc::get_mut(&mut self.storage) {
                items.extend(other.iter());
                let len = items.len();
                self.len = u32::try_from(len).expect("a list past u32::MAX elements");
                return self;
            }
        }
        let mut items = self.to_vec();
        items.extend(other.iter());
        ListRef::from(items)
    }

    /// Whether both lists read the same buffer.
    ///
    /// This is the invariant the batch door depends on: every row of one output
    /// is a window onto one [`ListStorage`], so a row costs no allocation. It
    /// is observable rather than private because a change that quietly went
    /// back to a buffer per row would otherwise pass every test.
    pub fn shares_storage_with(&self, other: &ListRef) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// The boxed buffer, when this list is exactly all of one.
    fn whole_object(&self) -> Option<&Vec<Value>> {
        match &*self.storage {
            ListStorage::Object(v) if self.start == 0 && self.len() == v.len() => Some(v),
            _ => None,
        }
    }
}

impl Default for ListRef {
    fn default() -> Self {
        ListRef::from(Vec::new())
    }
}

impl<T: Into<ListStorage>> From<T> for ListRef {
    fn from(items: T) -> Self {
        ListRef::whole(Arc::new(items.into()))
    }
}

/// Element-wise, so two lists are equal when they hold equal values whatever
/// strategy each of them uses.
impl PartialEq for ListRef {
    fn eq(&self, other: &Self) -> bool {
        match (self.whole_object(), other.whole_object()) {
            (Some(a), Some(b)) => a == b,
            _ => self.len() == other.len() && self.iter().eq(other.iter()),
        }
    }
}

impl std::fmt::Debug for ListRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Value {
    /// A list value over `items`, which may be a `Vec<Value>` or an already
    /// chosen [`ListStorage`] strategy.
    pub fn list(items: impl Into<ListStorage>) -> Value {
        Value::List(ListRef::from(items.into()))
    }
}

#[derive(Clone)]
pub enum Value {
    List(ListRef),
    Map(Map),

    // Atoms
    Int(i64),
    UInt(u64),
    Float(f64),
    String(Arc<String>),
    Bytes(Arc<Vec<u8>>),
    Bool(bool),
    #[cfg(feature = "chrono")]
    Duration(chrono::Duration),
    #[cfg(feature = "chrono")]
    Timestamp(chrono::DateTime<chrono::FixedOffset>),
    Opaque(Arc<dyn Opaque>),
    #[cfg(feature = "structs")]
    Struct(Arc<CelStruct>),
    Null,
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::List(l) => write!(f, "List({:?})", l),
            Value::Map(m) => write!(f, "Map({:?})", m),
            Value::Int(i) => write!(f, "Int({:?})", i),
            Value::UInt(u) => write!(f, "UInt({:?})", u),
            Value::Float(d) => write!(f, "Float({:?})", d),
            Value::String(s) => write!(f, "String({:?})", s),
            Value::Bytes(b) => write!(f, "Bytes({:?})", b),
            Value::Bool(b) => write!(f, "Bool({:?})", b),
            #[cfg(feature = "chrono")]
            Value::Duration(d) => write!(f, "Duration({:?})", d),
            #[cfg(feature = "chrono")]
            Value::Timestamp(t) => write!(f, "Timestamp({:?})", t),
            Value::Opaque(o) => write!(f, "Opaque<{}>({:?})", o.runtime_type_name(), o.as_debug()),
            Value::Null => write!(f, "Null"),
            #[cfg(feature = "structs")]
            Value::Struct(s) => write!(f, "{} {{}}", s.name()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ValueType {
    List,
    Map,
    Int,
    UInt,
    Float,
    String,
    Bytes,
    Bool,
    Duration,
    Timestamp,
    Opaque,
    Null,
    #[cfg(feature = "structs")]
    Struct,
}

impl Display for ValueType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ValueType::List => write!(f, "list"),
            ValueType::Map => write!(f, "map"),
            ValueType::Int => write!(f, "int"),
            ValueType::UInt => write!(f, "uint"),
            ValueType::Float => write!(f, "float"),
            ValueType::String => write!(f, "string"),
            ValueType::Bytes => write!(f, "bytes"),
            ValueType::Bool => write!(f, "bool"),
            ValueType::Opaque => write!(f, "opaque"),
            ValueType::Duration => write!(f, "duration"),
            ValueType::Timestamp => write!(f, "timestamp"),
            ValueType::Null => write!(f, "null"),
            #[cfg(feature = "structs")]
            ValueType::Struct => write!(f, "struct"),
        }
    }
}

impl Value {
    pub fn type_of(&self) -> ValueType {
        match self {
            Value::List(_) => ValueType::List,
            Value::Map(_) => ValueType::Map,
            Value::Int(_) => ValueType::Int,
            Value::UInt(_) => ValueType::UInt,
            Value::Float(_) => ValueType::Float,
            Value::String(_) => ValueType::String,
            Value::Bytes(_) => ValueType::Bytes,
            Value::Bool(_) => ValueType::Bool,
            Value::Opaque(_) => ValueType::Opaque,
            #[cfg(feature = "chrono")]
            Value::Duration(_) => ValueType::Duration,
            #[cfg(feature = "chrono")]
            Value::Timestamp(_) => ValueType::Timestamp,
            Value::Null => ValueType::Null,
            #[cfg(feature = "structs")]
            Value::Struct(_) => ValueType::Struct,
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::List(v) => v.is_empty(),
            Value::Map(v) => v.is_empty(),
            Value::Int(0) => true,
            Value::UInt(0) => true,
            Value::Float(f) => *f == 0.0,
            Value::String(v) => v.is_empty(),
            Value::Bytes(v) => v.is_empty(),
            Value::Bool(false) => true,
            #[cfg(feature = "chrono")]
            Value::Duration(v) => v.is_zero(),
            Value::Null => true,
            _ => false,
        }
    }

    pub fn error_expected_type(&self, expected: ValueType) -> ExecutionError {
        ExecutionError::UnexpectedType {
            got: self.type_of().to_string(),
            want: expected.to_string(),
        }
    }
}

impl From<&Value> for Value {
    fn from(value: &Value) -> Self {
        value.clone()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Map(a), Value::Map(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::UInt(a), Value::UInt(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Bytes(a), Value::Bytes(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Null, Value::Null) => true,
            #[cfg(feature = "chrono")]
            (Value::Duration(a), Value::Duration(b)) => a == b,
            #[cfg(feature = "chrono")]
            (Value::Timestamp(a), Value::Timestamp(b)) => a == b,
            // Allow different numeric types to be compared without explicit casting.
            (Value::Int(a), Value::UInt(b)) => a
                .to_owned()
                .try_into()
                .map(|a: u64| a == *b)
                .unwrap_or(false),
            (Value::Int(a), Value::Float(b)) => (*a as f64) == *b,
            (Value::UInt(a), Value::Int(b)) => a
                .to_owned()
                .try_into()
                .map(|a: i64| a == *b)
                .unwrap_or(false),
            (Value::UInt(a), Value::Float(b)) => (*a as f64) == *b,
            (Value::Float(a), Value::Int(b)) => *a == (*b as f64),
            (Value::Float(a), Value::UInt(b)) => *a == (*b as f64),
            (Value::Opaque(a), Value::Opaque(b)) => a.opaque_eq(b.deref()),
            (_, _) => false,
        }
    }
}

impl Eq for Value {}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => Some(a.cmp(b)),
            (Value::UInt(a), Value::UInt(b)) => Some(a.cmp(b)),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => Some(a.cmp(b)),
            (Value::Bytes(a), Value::Bytes(b)) => Some(a.cmp(b)),
            (Value::Bool(a), Value::Bool(b)) => Some(a.cmp(b)),
            (Value::Null, Value::Null) => Some(Ordering::Equal),
            #[cfg(feature = "chrono")]
            (Value::Duration(a), Value::Duration(b)) => Some(a.cmp(b)),
            #[cfg(feature = "chrono")]
            (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b)),
            // Allow different numeric types to be compared without explicit casting.
            (Value::Int(a), Value::UInt(b)) => Some(
                a.to_owned()
                    .try_into()
                    .map(|a: u64| a.cmp(b))
                    // If the i64 doesn't fit into a u64 it must be less than 0.
                    .unwrap_or(Ordering::Less),
            ),
            (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::UInt(a), Value::Int(b)) => Some(
                a.to_owned()
                    .try_into()
                    .map(|a: i64| a.cmp(b))
                    // If the u64 doesn't fit into a i64 it must be greater than i64::MAX.
                    .unwrap_or(Ordering::Greater),
            ),
            (Value::UInt(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Float(a), Value::UInt(b)) => a.partial_cmp(&(*b as f64)),
            _ => None,
        }
    }
}

impl From<&Key> for Value {
    fn from(value: &Key) -> Self {
        match value {
            Key::Int(v) => Value::Int(*v),
            Key::Uint(v) => Value::UInt(*v),
            Key::Bool(v) => Value::Bool(*v),
            Key::String(v) => Value::String(v.clone()),
        }
    }
}

impl From<Key> for Value {
    fn from(value: Key) -> Self {
        match value {
            Key::Int(v) => Value::Int(v),
            Key::Uint(v) => Value::UInt(v),
            Key::Bool(v) => Value::Bool(v),
            Key::String(v) => Value::String(v),
        }
    }
}

impl From<&Key> for Key {
    fn from(key: &Key) -> Self {
        key.clone()
    }
}

// Convert Vec<T> to Value
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::list(v.into_iter().map(|v| v.into()).collect::<Vec<_>>())
    }
}

// Convert Vec<u8> to Value
impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v.into())
    }
}

#[cfg(feature = "bytes")]
// Convert Bytes to Value
impl From<::bytes::Bytes> for Value {
    fn from(v: ::bytes::Bytes) -> Self {
        Value::Bytes(v.to_vec().into())
    }
}

#[cfg(feature = "bytes")]
// Convert &Bytes to Value
impl From<&::bytes::Bytes> for Value {
    fn from(v: &::bytes::Bytes) -> Self {
        Value::Bytes(v.to_vec().into())
    }
}

// Convert String to Value
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v.into())
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string().into())
    }
}

// Convert Option<T> to Value
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(v) => v.into(),
            None => Value::Null,
        }
    }
}

// Convert HashMap<K, V> to Value
impl<K: Into<Key>, V: Into<Value>> From<HashMap<K, V>> for Value {
    fn from(v: HashMap<K, V>) -> Self {
        Value::Map(v.into())
    }
}

impl From<ExecutionError> for ResolveResult {
    fn from(value: ExecutionError) -> Self {
        Err(value)
    }
}

pub type ResolveResult = Result<Value, ExecutionError>;

impl From<Value> for ResolveResult {
    fn from(value: Value) -> Self {
        Ok(value)
    }
}

impl TryFrom<&dyn Val> for Value {
    type Error = ExecutionError;
    fn try_from(v: &dyn Val) -> Result<Self, Self::Error> {
        match v.get_type().kind() {
            Kind::Boolean => Ok(Value::Bool(*v.downcast_ref::<CelBool>().unwrap().inner())),
            Kind::Int => Ok(Value::Int(*v.downcast_ref::<CelInt>().unwrap().inner())),
            Kind::UInt => Ok(Value::UInt(*v.downcast_ref::<CelUInt>().unwrap().inner())),
            Kind::Double => Ok(Value::Float(
                *v.downcast_ref::<CelDouble>().unwrap().inner(),
            )),
            Kind::String => Ok(Value::String(Arc::new(
                v.downcast_ref::<CelString>().unwrap().inner().to_string(),
            ))),
            Kind::NullType => Ok(Value::Null),
            Kind::Bytes => Ok(Value::Bytes(Arc::new(
                v.downcast_ref::<CelBytes>().unwrap().inner().to_vec(),
            ))),
            #[cfg(feature = "chrono")]
            Kind::Duration => Ok(Value::Duration(
                *v.downcast_ref::<CelDuration>().unwrap().inner(),
            )),
            #[cfg(feature = "chrono")]
            Kind::Timestamp => {
                let ts = v.downcast_ref::<CelTimestamp>().unwrap().inner();
                Ok(Value::Timestamp(*ts))
            }
            Kind::List => {
                let list = v.downcast_ref::<CelList>().unwrap().inner();
                Ok(Value::list(
                    list.iter()
                        .map(|i| i.as_ref().try_into().expect("Not a Value list item"))
                        .collect::<Vec<Value>>(),
                ))
            }
            Kind::Map => {
                let map = v.downcast_ref::<CelMap>().unwrap().inner();
                Ok(Value::Map(Map::object(Arc::new(
                    map.iter()
                        .map(|(k, v)| {
                            (
                                Key::from(k.clone()),
                                Value::try_from(v.as_ref()).expect("Not a Value map value"),
                            )
                        })
                        .collect(),
                ))))
            }
            Kind::Opaque => Ok(Value::Opaque(match v.downcast_ref::<CelOptional>() {
                None => v.downcast_ref::<OpaqueVal>().unwrap().clone_inner(),
                Some(opt) => {
                    let opt: Option<Result<Value, _>> = opt.option().map(|v| v.try_into());
                    match opt {
                        None => Arc::new(OptionalValue::none()),
                        Some(t) => match t {
                            Ok(v) => Arc::new(OptionalValue::of(v)),
                            Err(_) => Arc::new(OptionalValue::none()),
                        },
                    }
                }
            })),
            _ => {
                #[cfg(feature = "structs")]
                {
                    if let Some(v) = v.downcast_ref::<CelStruct>() {
                        use crate::common::value::Downcast;

                        return match v.clone_as_boxed().downcast::<CelStruct>() {
                            Ok(v) => Ok(Value::Struct(Arc::new(*v))),
                            Err(v) => Err(ExecutionError::InternalError(format!(
                                "Not a Struct: `{v:?}`"
                            ))),
                        };
                    }
                }
                if let Some(opaque) = v.downcast_ref::<OpaqueVal>() {
                    Ok(Value::Opaque(opaque.val.clone()))
                } else {
                    Err(ExecutionError::UnexpectedType {
                        got: v.get_type().name().to_string(),
                        want:
                            "(BOOL|INT|UINT|DOUBLE|STRING|NULL|BYTES|TIMESTAMP|DURATION|LIST|MAP)"
                                .to_string(),
                    })
                }
            }
        }
    }
}

impl TryFrom<Value> for Box<dyn Val> {
    type Error = ExecutionError;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Bool(b) => Ok(Box::new(CelBool::from(b))),
            Value::Int(i) => Ok(Box::new(CelInt::from(i))),
            Value::UInt(u) => Ok(Box::new(CelUInt::from(u))),
            Value::Float(f) => Ok(Box::new(CelDouble::from(f))),
            Value::String(s) => Ok(Box::new(CelString::from(s.as_str()))),
            Value::Null => Ok(Box::new(CelNull)),
            Value::Bytes(b) => Ok(Box::new(CelBytes::from(b.as_slice().to_vec()))),
            #[cfg(feature = "chrono")]
            Value::Duration(d) => Ok(Box::new(CelDuration::from(d))),
            #[cfg(feature = "chrono")]
            Value::Timestamp(ts) => Ok(Box::new(CelTimestamp::from(ts))),
            Value::List(l) => {
                let result: Result<Vec<Box<dyn Val>>, ExecutionError> =
                    l.to_vec().into_iter().map(|i| i.try_into()).collect();
                Ok(Box::new(CelList::from(result?)))
            }
            Value::Map(map) => {
                let result: Result<HashMap<CelMapKey, Box<dyn Val>>, ExecutionError> = map
                    .iter()
                    .map(|(k, v)| v.into_owned().try_into().map(|v| (k.clone().into(), v)))
                    .collect();
                Ok(Box::new(CelMap::from(result?)))
            }
            Value::Opaque(o) => {
                let v: Box<dyn Val> = if let Some(value) = o.downcast_ref::<OptionalValue>() {
                    match value.inner() {
                        None => Box::new(CelOptional::none()),
                        Some(v) => Box::new(CelOptional::of(v.clone().try_into()?)),
                    }
                } else {
                    Box::new(OpaqueVal::new(o))
                };
                Ok(v)
            }
            #[cfg(feature = "structs")]
            Value::Struct(s) => Ok(Arc::try_unwrap(s)
                .map(|s| Box::new(s) as Box<dyn Val>)
                .unwrap_or_else(|arc| arc.clone_as_boxed())),
        }
    }
}

/// The append shape the `map` and `filter` macros expand their loop step to.
///
/// `map(x, f(x))` expands to the step `@result + [f(x)]`, and `filter(x, c)`
/// (and `map`'s three-argument form) wraps that in `c ? @result + [..] :
/// @result`. Evaluated as written, each step builds a whole new accumulator to
/// append one element, so a comprehension over n elements copies n^2/2 of them
/// — and a rejected element costs a copy too, because the else-branch resolves
/// the accumulator and the loop then takes it by value.
///
/// Recognising the shape lets the element be pushed onto the accumulator the
/// loop already holds, and lets a rejected element cost nothing.
struct AccuAppend<'a> {
    /// The `filter`'s condition, when the step is the conditional form.
    guard: Option<&'a Expression>,
    /// The single element the step appends.
    element: &'a Expression,
}

impl<'a> AccuAppend<'a> {
    fn of(comprehension: &'a ComprehensionExpr) -> Option<Self> {
        // The accumulator is held outside the context on this path, so nothing
        // evaluated per iteration may read it. The loop condition is the one
        // place a step of this shape still could — `exists` and `all` stop on
        // it — and both macros here emit a constant `true`.
        match &comprehension.loop_cond.expr {
            Expr::Literal(LiteralValue::Boolean(b)) if *b.inner() => {}
            _ => return None,
        }
        let accu_var = comprehension.accu_var.as_str();
        let is_accu = |expr: &Expression| matches!(&expr.expr, Expr::Ident(n) if n == accu_var);

        let (guard, step) = match &comprehension.loop_step.expr {
            Expr::Call(call)
                if call.func_name == operators::CONDITIONAL
                    && call.args.len() == 3
                    && is_accu(&call.args[2]) =>
            {
                (Some(&call.args[0]), &call.args[1])
            }
            _ => (None, &comprehension.loop_step),
        };

        let Expr::Call(call) = &step.expr else {
            return None;
        };
        if call.func_name != operators::ADD || call.args.len() != 2 || !is_accu(&call.args[0]) {
            return None;
        }
        // An optional entry (`[?x]`) appends zero or one element depending on
        // the value, which is not this shape.
        match &call.args[1].expr {
            Expr::List(list) if list.elements.len() == 1 && list.optional_indices.is_empty() => {
                Some(AccuAppend {
                    guard,
                    element: &list.elements[0],
                })
            }
            _ => None,
        }
    }
}

impl Value {
    pub fn resolve_all(expr: &[Expression], ctx: &Context) -> ResolveResult {
        let mut res = Vec::with_capacity(expr.len());
        for expr in expr {
            res.push(Value::resolve(expr, ctx)?);
        }
        Ok(Value::list(res))
    }

    pub fn resolve(expr: &Expression, ctx: &Context) -> ResolveResult {
        Self::resolve_value(expr, ctx)
    }

    /// Evaluates `expr` entirely within the [`Value`] universe.
    ///
    /// Mirrors [`Value::resolve_val`] arm for arm without ever constructing a
    /// `Box<dyn Val>`, except at the host-function boundary, which still takes
    /// its arguments as `Cow<dyn Val>`. Both walkers are held to the same
    /// answers by the differential corpus in `tests/oracle.rs`.
    pub fn resolve_value(expr: &Expression, ctx: &Context) -> Result<Value, ExecutionError> {
        match &expr.expr {
            Expr::Literal(literal) => Ok(literal.to_value()),
            Expr::Call(call) => {
                if call.args.len() == 3 && call.func_name == operators::CONDITIONAL {
                    return if try_bool_value(Value::resolve_value(&call.args[0], ctx))? {
                        Value::resolve_value(&call.args[1], ctx)
                    } else {
                        Value::resolve_value(&call.args[2], ctx)
                    };
                }
                if call.args.len() == 2 {
                    match call.func_name.as_str() {
                        operators::LOGICAL_OR => {
                            let left = try_bool_value(Value::resolve_value(&call.args[0], ctx));
                            return if Ok(true) == left {
                                Ok(Value::Bool(true))
                            } else {
                                let right = match Value::resolve_value(&call.args[1], ctx)? {
                                    Value::Bool(b) => Some(b),
                                    _ => None,
                                };
                                match (left, right) {
                                    (Ok(false), Some(right)) => Ok(Value::Bool(right)),
                                    (Err(_), Some(true)) => Ok(Value::Bool(true)),
                                    (left, _) => Err(left.err().unwrap_or(NoSuchOverload)),
                                }
                            };
                        }
                        operators::LOGICAL_AND => {
                            let left = try_bool_value(Value::resolve_value(&call.args[0], ctx));
                            return if Ok(false) == left {
                                Ok(Value::Bool(false))
                            } else {
                                let right = match Value::resolve_value(&call.args[1], ctx)? {
                                    Value::Bool(b) => Some(b),
                                    _ => None,
                                };
                                match (left, right) {
                                    (Ok(true), Some(right)) => Ok(Value::Bool(right)),
                                    (Err(_), Some(false)) => Ok(Value::Bool(false)),
                                    (left, _) => Err(left.err().unwrap_or(NoSuchOverload)),
                                }
                            };
                        }
                        operators::EQUALS => {
                            let lhs = Value::resolve_value(&call.args[0], ctx)?;
                            let rhs = Value::resolve_value(&call.args[1], ctx)?;
                            return Ok(Value::Bool(lhs == rhs));
                        }
                        operators::NOT_EQUALS => {
                            let lhs = Value::resolve_value(&call.args[0], ctx)?;
                            let rhs = Value::resolve_value(&call.args[1], ctx)?;
                            return Ok(Value::Bool(lhs != rhs));
                        }
                        operators::INDEX | operators::OPT_INDEX => {
                            let mut is_optional = call.func_name == operators::OPT_INDEX;
                            let value = Value::resolve_value(&call.args[0], ctx)?;
                            let value = match as_optional(&value) {
                                Some(opt) => {
                                    is_optional = true;
                                    match opt.value() {
                                        Some(v) => v.clone(),
                                        None => return Ok(optional_none()),
                                    }
                                }
                                None => value,
                            };
                            let key = Value::resolve_value(&call.args[1], ctx)?;
                            let result = value_index(&value, &key);
                            return if is_optional {
                                Ok(match result {
                                    Ok(v) => optional_of(v),
                                    Err(_) => optional_none(),
                                })
                            } else {
                                result
                            };
                        }
                        operators::OPT_SELECT => {
                            let operand = Value::resolve_value(&call.args[0], ctx)?;
                            let field = match Value::resolve_value(&call.args[1], ctx)? {
                                Value::String(s) => Value::String(s),
                                _ => {
                                    return Err(ExecutionError::function_error(
                                        "_?._",
                                        "field must be string",
                                    ))
                                }
                            };
                            return Ok(match as_optional(&operand) {
                                // `Optional::map` keeps the outer `Some` and
                                // substitutes `optional.none` for a missing
                                // field, so a miss nests one optional inside
                                // another. Mirrored, not corrected, here.
                                Some(opt) => match opt.value() {
                                    None => optional_none(),
                                    Some(inner) => optional_of(
                                        value_index(inner, &field)
                                            .unwrap_or_else(|_| optional_none()),
                                    ),
                                },
                                None => optional_of(value_index(&operand, &field)?),
                            });
                        }
                        operators::ADD => return binary_op("add", call, ctx),
                        operators::SUBSTRACT => return binary_op("sub", call, ctx),
                        operators::DIVIDE => return binary_op("div", call, ctx),
                        operators::MULTIPLY => return binary_op("mul", call, ctx),
                        operators::MODULO => return binary_op("rem", call, ctx),
                        operators::LESS => {
                            return compare_op(call, ctx, |o| o == Ordering::Less);
                        }
                        operators::LESS_EQUALS => {
                            return compare_op(call, ctx, |o| o != Ordering::Greater);
                        }
                        operators::GREATER => {
                            return compare_op(call, ctx, |o| o == Ordering::Greater);
                        }
                        operators::GREATER_EQUALS => {
                            return compare_op(call, ctx, |o| o != Ordering::Less);
                        }
                        operators::IN => {
                            let lhs = Value::resolve_value(&call.args[0], ctx)?;
                            let rhs = Value::resolve_value(&call.args[1], ctx)?;
                            return Ok(Value::Bool(value_contains(&rhs, &lhs)?));
                        }
                        _ => (),
                    }
                }
                if call.args.len() == 1 {
                    match call.func_name.as_str() {
                        operators::LOGICAL_NOT => {
                            return match Value::resolve_value(&call.args[0], ctx)? {
                                Value::Bool(b) => Ok(Value::Bool(!b)),
                                _ => Err(ExecutionError::NoSuchOverload),
                            };
                        }
                        operators::NEGATE => {
                            let val = Value::resolve_value(&call.args[0], ctx)?;
                            return value_negate(val);
                        }
                        operators::NOT_STRICTLY_FALSE => {
                            return Ok(Value::Bool(
                                try_bool_value(Value::resolve_value(&call.args[0], ctx))
                                    .unwrap_or(true),
                            ));
                        }
                        _ => (),
                    }
                }
                match &call.target {
                    None => {
                        let args = resolve_args(&call.args, ctx)?;
                        if let Some(op) = ctx.env().find_overload(&call.func_name, &args) {
                            return op(args)?.as_ref().try_into();
                        }
                        let func = ctx.get_function(call.func_name.as_str()).ok_or_else(|| {
                            ExecutionError::UndeclaredReference(call.func_name.clone().into())
                        })?;
                        let mut ctx = FunctionContext::new(&call.func_name, None, ctx, args);
                        (func)(&mut ctx)
                    }
                    Some(target) => {
                        let args = resolve_args(&call.args, ctx)?;
                        let qualified_func = match &target.expr {
                            Expr::Ident(prefix) => {
                                let qualified_name = format!("{prefix}.{}", call.func_name);
                                if let Some(op) = ctx.env().find_overload(&qualified_name, &args) {
                                    return op(args)?.as_ref().try_into();
                                }
                                ctx.get_function(&qualified_name)
                            }
                            _ => None,
                        };
                        let (target, func, args) = match qualified_func {
                            None => {
                                let target = Value::resolve_value(target, ctx)?;
                                let mut args = args;
                                args.insert(0, Cow::Owned(target.try_into()?));
                                if let Some(op) =
                                    ctx.env().find_member_overload(&call.func_name, &args)
                                {
                                    return op(args)?.as_ref().try_into();
                                }
                                let target = args.remove(0);
                                let func =
                                    ctx.get_function(call.func_name.as_str()).ok_or_else(|| {
                                        ExecutionError::UndeclaredReference(
                                            call.func_name.clone().into(),
                                        )
                                    })?;
                                (Some(target), func, args)
                            }
                            Some(func) => (None, func, args),
                        };
                        let mut ctx = FunctionContext::new(&call.func_name, target, ctx, args);
                        (func)(&mut ctx)
                    }
                }
            }
            Expr::Ident(name) => ctx
                .get_variable(name)
                .ok_or_else(|| ExecutionError::UndeclaredReference(Arc::new(name.to_string()))),
            Expr::Select(select) => {
                let left = Value::resolve_value(select.operand.deref(), ctx)?;
                let field = select.field.as_str();

                if select.test {
                    match &left {
                        Value::Map(map) => {
                            Ok(Value::Bool(map.contains_key(&KeyRef::String(field))))
                        }
                        #[cfg(feature = "structs")]
                        Value::Struct(_) => Ok(Value::Bool(value_field(&left, field).is_ok())),
                        _ => value_field(&left, field),
                    }
                } else {
                    value_field(&left, field)
                }
            }
            Expr::List(list_expr) => {
                let mut list = Vec::with_capacity(list_expr.elements.len());
                for (idx, element) in list_expr.elements.iter().enumerate() {
                    let value = Value::resolve_value(element, ctx)?;
                    if list_expr.optional_indices.contains(&idx) {
                        match as_optional(&value) {
                            Some(opt) => {
                                if let Some(inner) = opt.value() {
                                    list.push(inner.clone());
                                }
                            }
                            None => list.push(value),
                        }
                    } else {
                        list.push(value);
                    }
                }
                Ok(Value::list(list))
            }
            Expr::Map(map_expr) => {
                let mut map = HashMap::with_capacity(map_expr.entries.len());
                for entry in map_expr.entries.iter() {
                    let (k, v, is_optional) = match &entry.expr {
                        EntryExpr::StructField(_) => panic!("WAT?"),
                        EntryExpr::MapEntry(e) => (&e.key, &e.value, e.optional),
                    };
                    let key = value_key(Value::resolve_value(k, ctx)?)?;
                    let value = Value::resolve_value(v, ctx)?;

                    if is_optional {
                        if let Some(opt) = as_optional(&value) {
                            if let Some(inner) = opt.value() {
                                map.insert(key, inner.clone());
                            }
                        } else {
                            map.insert(key, value);
                        }
                    } else {
                        map.insert(key, value);
                    }
                }
                Ok(Value::Map(Map::object(Arc::new(map))))
            }
            Expr::Comprehension(comprehension) => {
                let accu_init = Value::resolve_value(&comprehension.accu_init, ctx)?;
                let iter = Value::resolve_value(&comprehension.iter_range, ctx)?;
                let mut ctx = ctx.new_inner_scope();
                let mut items = value_iter(&iter)?.into_iter();

                if let Some(append) = AccuAppend::of(comprehension) {
                    if let Value::List(list) = accu_init {
                        // The accumulator stays here rather than in the
                        // context: `@result` is not a name CEL can parse, so
                        // the guard and the element cannot read it, and a
                        // nested comprehension binds its own in its own scope.
                        let mut list = list.into_vec();
                        for item in items {
                            ctx.add_variable_from_value(&comprehension.iter_var, item);
                            if let Some(guard) = append.guard {
                                if !try_bool_value(Value::resolve_value(guard, &ctx))? {
                                    continue;
                                }
                            }
                            list.push(Value::resolve_value(append.element, &ctx)?);
                        }
                        ctx.add_variable_from_value(&comprehension.accu_var, Value::list(list));
                        return Value::resolve_value(&comprehension.result, &ctx);
                    }
                    unreachable!("AccuAppend::of implies a list accumulator");
                }

                ctx.add_variable_from_value(&comprehension.accu_var, accu_init);
                for item in items.by_ref() {
                    if !try_bool_value(Value::resolve_value(&comprehension.loop_cond, &ctx))? {
                        break;
                    }
                    ctx.add_variable_from_value(&comprehension.iter_var, item);
                    let accu = Value::resolve_value(&comprehension.loop_step, &ctx)?;
                    ctx.add_variable_from_value(&comprehension.accu_var, accu);
                }
                Value::resolve_value(&comprehension.result, &ctx)
            }
            Expr::Struct(strct) => {
                let name = strct.type_name.clone();
                #[cfg(not(feature = "structs"))]
                {
                    Err(ExecutionError::InternalError(format!(
                        "Found struct {name}, feature not enabled!"
                    )))
                }
                #[cfg(feature = "structs")]
                {
                    let struct_def =
                        ctx.env()
                            .find_struct(&name)
                            .ok_or(ExecutionError::UnexpectedType {
                                got: name.to_owned(),
                                want: "known struct".to_owned(),
                            })?;
                    let mut fields = std::collections::BTreeMap::new();
                    for entry in &strct.entries {
                        match &entry.expr {
                            EntryExpr::StructField(expr) => {
                                let f = expr.field.clone();
                                let v = Value::resolve_value(&expr.value, ctx)?;
                                fields.insert(f, Cow::Owned(TryInto::<Box<dyn Val>>::try_into(v)?));
                            }
                            EntryExpr::MapEntry(entry) => {
                                return Err(ExecutionError::InternalError(format!(
                                    "Expected struct_field_expr, got {entry:?}"
                                )))
                            }
                        }
                    }
                    Ok(Value::Struct(Arc::new(struct_def.new_struct(fields)?)))
                }
            }
            Expr::Unspecified => panic!("Can't evaluate Unspecified Expr"),
        }
    }
}

/// Wraps `value` in an `optional`.
fn optional_of(value: Value) -> Value {
    Value::Opaque(Arc::new(OptionalValue::of(value)))
}

/// The empty `optional`.
fn optional_none() -> Value {
    Value::Opaque(Arc::new(OptionalValue::none()))
}

/// Views `value` as an optional, if it is one.
fn as_optional(value: &Value) -> Option<&OptionalValue> {
    match value {
        Value::Opaque(o) => o.downcast_ref::<OptionalValue>(),
        _ => None,
    }
}

fn try_bool_value(val: Result<Value, ExecutionError>) -> Result<bool, ExecutionError> {
    match val {
        Ok(Value::Bool(b)) => Ok(b),
        Ok(_) => Err(ExecutionError::NoSuchOverload),
        Err(err) => Err(err),
    }
}

/// Resolves a call's arguments and bridges them into the `dyn Val` universe.
///
/// The env overload table and [`FunctionContext`] both still take
/// `Cow<dyn Val>`; this is the only place [`Value::resolve_value`] touches the
/// old universe.
fn resolve_args<'a>(
    args: &[Expression],
    ctx: &Context,
) -> Result<Vec<Cow<'a, dyn Val>>, ExecutionError> {
    let mut out = Vec::with_capacity(args.len());
    for arg in args {
        let value = Value::resolve_value(arg, ctx)?;
        out.push(Cow::Owned(TryInto::<Box<dyn Val>>::try_into(value)?));
    }
    Ok(out)
}

/// Whether an incompatible right-hand side makes `op` on this receiver answer
/// `NoSuchOverload` rather than `UnsupportedBinaryOperator`.
///
/// Each operator trait in `common/types` picks its own error for a right-hand
/// side it cannot handle, and they do not agree: `Int` answers `NoSuchOverload`
/// for all five arithmetic operators, `List` does for `add` and `Duration` for
/// `sub`, and every other receiver answers `UnsupportedBinaryOperator`. That
/// split is not a rule anyone stated — it is which error each impl happened to
/// be written with — but it is observable, so it is reproduced rather than
/// tidied. [`Value`]'s native operators answer `UnsupportedBinaryOperator`
/// throughout, so only the listed cases are rewritten.
///
/// Derived by sweeping every (receiver, operator, argument) triple through both
/// evaluators, not by reading the impls: a capability trait's terminal error is
/// easy to misattribute to a neighbouring impl.
fn mismatch_is_no_such_overload(op: &'static str, value: &Value) -> bool {
    #[cfg(feature = "chrono")]
    let duration_sub = matches!(value, Value::Duration(_)) && op == "sub";
    #[cfg(not(feature = "chrono"))]
    let duration_sub = false;

    matches!(value, Value::Int(_))
        || (matches!(value, Value::List(_)) && op == "add")
        || duration_sub
}

fn binary_op(op: &'static str, call: &CallExpr, ctx: &Context) -> Result<Value, ExecutionError> {
    let lhs = Value::resolve_value(&call.args[0], ctx)?;
    let rhs = Value::resolve_value(&call.args[1], ctx)?;
    let rewrite = mismatch_is_no_such_overload(op, &lhs);
    let result = match op {
        "add" => lhs + rhs,
        "sub" => lhs - rhs,
        "div" => lhs / rhs,
        "mul" => lhs * rhs,
        "rem" => lhs % rhs,
        _ => unreachable!("unknown binary operator {op}"),
    };
    match result {
        Err(ExecutionError::UnsupportedBinaryOperator(..)) if rewrite => {
            Err(ExecutionError::NoSuchOverload)
        }
        other => other,
    }
}

/// Whether the receiver carries an ordering at all.
///
/// `PartialOrd for Value` orders `Null` against `Null` and would order a list
/// or a map by falling through, where `common/types` gives those no `Comparer`,
/// so ordering them is `NoSuchOverload`.
fn has_comparer(value: &Value) -> bool {
    #[cfg(feature = "chrono")]
    if matches!(value, Value::Duration(_) | Value::Timestamp(_)) {
        return true;
    }
    matches!(
        value,
        Value::Int(_)
            | Value::UInt(_)
            | Value::Float(_)
            | Value::String(_)
            | Value::Bytes(_)
            | Value::Bool(_)
    )
}

fn compare_op(
    call: &CallExpr,
    ctx: &Context,
    accept: impl FnOnce(Ordering) -> bool,
) -> Result<Value, ExecutionError> {
    let lhs = Value::resolve_value(&call.args[0], ctx)?;
    let rhs = Value::resolve_value(&call.args[1], ctx)?;
    if !has_comparer(&lhs) {
        return Err(ExecutionError::NoSuchOverload);
    }
    let ordering = lhs
        .partial_cmp(&rhs)
        .ok_or(ExecutionError::NoSuchOverload)?;
    Ok(Value::Bool(accept(ordering)))
}

fn value_negate(value: Value) -> Result<Value, ExecutionError> {
    match value {
        Value::Int(i) => i
            .checked_neg()
            .ok_or_else(|| ExecutionError::Overflow("negate", Value::Int(i), Value::Int(0)))
            .map(Value::Int),
        Value::Float(f) => Ok(Value::Float(-f)),
        Value::Bool(b) => Ok(Value::Bool(!b)),
        #[cfg(feature = "chrono")]
        Value::Duration(d) => Ok(Value::Duration(-d)),
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// Converts `value` into a map key.
fn value_key(value: Value) -> Result<Key, ExecutionError> {
    match value {
        Value::Int(i) => Ok(Key::Int(i)),
        Value::UInt(u) => Ok(Key::Uint(u)),
        Value::Bool(b) => Ok(Key::Bool(b)),
        Value::String(s) => Ok(Key::String(s)),
        other => Err(ExecutionError::unsupported_key_type(other)),
    }
}

/// `container.field`, looked up without materializing the field name.
///
/// `KeyRef::String` borrows, so a map field select costs no allocation. Going
/// through [`value_index`] would build an `Arc<String>` per access.
fn value_field(container: &Value, field: &str) -> Result<Value, ExecutionError> {
    match container {
        Value::Map(map) => map
            .get(&KeyRef::String(field))
            .map(|v| v.into_owned())
            .ok_or_else(|| ExecutionError::NoSuchKey(Arc::new(field.to_string()))),
        Value::List(_) => Err(ExecutionError::UnexpectedType {
            got: ValueType::String.to_string(),
            want: format!("{}|{}", ValueType::Int, ValueType::UInt),
        }),
        #[cfg(feature = "structs")]
        Value::Struct(_) => value_index(container, &Value::String(Arc::new(field.to_string()))),
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// `container[key]`, for every container the walker can index.
fn value_index(container: &Value, key: &Value) -> Result<Value, ExecutionError> {
    match container {
        Value::List(list) => {
            let idx = match key {
                Value::Int(i) => *i as usize,
                Value::UInt(u) => *u as usize,
                other => {
                    return Err(ExecutionError::UnexpectedType {
                        got: other.type_of().to_string(),
                        want: format!("{}|{}", ValueType::Int, ValueType::UInt),
                    })
                }
            };
            list.get(idx)
                .ok_or_else(|| ExecutionError::IndexOutOfBounds(key.clone()))
        }
        Value::Map(map) => {
            let k = value_key(key.clone())?;
            map.get(&k)
                .map(|v| v.into_owned())
                .ok_or_else(|| ExecutionError::NoSuchKey(Arc::new(key_display(&k))))
        }
        #[cfg(feature = "structs")]
        Value::Struct(s) => {
            use crate::common::traits::Indexer;
            let boxed: Box<dyn Val> = key.clone().try_into()?;
            s.get(boxed.as_ref())
                .and_then(|v| Value::try_from(v.as_ref()))
        }
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// The keys of `map`, as values a comprehension can bind.
fn map_keys(map: &Map) -> Vec<Value> {
    match map.storage() {
        MapStorage::Object(entries) => entries.keys().map(key_value).collect(),
        MapStorage::Record { schema, .. } => schema.keys.iter().map(key_value).collect(),
    }
}

fn key_value(key: &Key) -> Value {
    match key {
        Key::Int(i) => Value::Int(*i),
        Key::Uint(u) => Value::UInt(*u),
        Key::Bool(b) => Value::Bool(*b),
        Key::String(s) => Value::String(s.clone()),
    }
}

fn key_display(key: &Key) -> String {
    match key {
        Key::Int(i) => i.to_string(),
        Key::Uint(u) => u.to_string(),
        Key::Bool(b) => b.to_string(),
        Key::String(s) => s.as_str().to_string(),
    }
}

/// `needle in container`.
///
/// A needle that cannot be a map key is an error rather than a miss, which is
/// why the conversion is propagated instead of folded into `false`.
fn value_contains(container: &Value, needle: &Value) -> Result<bool, ExecutionError> {
    match container {
        Value::List(list) => Ok(list.contains(needle)),
        Value::Map(map) => Ok(map.contains_key(&value_key(needle.clone())?)),
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// The elements a comprehension iterates over.
fn value_iter(value: &Value) -> Result<Vec<Value>, ExecutionError> {
    match value {
        Value::List(list) => Ok(list.to_vec()),
        Value::Map(map) => Ok(map_keys(map)),
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

impl ops::Add<Value> for Value {
    type Output = ResolveResult;

    #[inline(always)]
    fn add(self, rhs: Value) -> Self::Output {
        match (self, rhs) {
            (Value::Int(l), Value::Int(r)) => l
                .checked_add(r)
                .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
                .map(Value::Int),

            (Value::UInt(l), Value::UInt(r)) => l
                .checked_add(r)
                .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
                .map(Value::UInt),

            (Value::Float(l), Value::Float(r)) => Value::Float(l + r).into(),

            (Value::List(l), Value::List(r)) => Ok(Value::List(l.concat(&r))),
            (Value::String(mut l), Value::String(r)) => {
                // If this is the only reference to `l`, we can append to it in place.
                // `l` is replaced with a clone otherwise.
                Arc::make_mut(&mut l).push_str(&r);
                Ok(Value::String(l))
            }
            (Value::Bytes(mut l), Value::Bytes(r)) => {
                Arc::make_mut(&mut l).extend_from_slice(&r);
                Ok(Value::Bytes(l))
            }
            #[cfg(feature = "chrono")]
            (Value::Duration(l), Value::Duration(r)) => l
                .checked_add(&r)
                .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
                .map(Value::Duration),
            #[cfg(feature = "chrono")]
            (Value::Timestamp(l), Value::Duration(r)) => checked_op(TsOp::Add, &l, &r),
            #[cfg(feature = "chrono")]
            (Value::Duration(l), Value::Timestamp(r)) => r
                .checked_add_signed(l)
                .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
                .map(Value::Timestamp),
            (left, right) => Err(ExecutionError::UnsupportedBinaryOperator(
                "add", left, right,
            )),
        }
    }
}

impl ops::Sub<Value> for Value {
    type Output = ResolveResult;

    #[inline(always)]
    fn sub(self, rhs: Value) -> Self::Output {
        match (self, rhs) {
            (Value::Int(l), Value::Int(r)) => l
                .checked_sub(r)
                .ok_or_else(|| ExecutionError::Overflow("sub", l.into(), r.into()))
                .map(Value::Int),

            (Value::UInt(l), Value::UInt(r)) => l
                .checked_sub(r)
                .ok_or_else(|| ExecutionError::Overflow("sub", l.into(), r.into()))
                .map(Value::UInt),

            (Value::Float(l), Value::Float(r)) => Value::Float(l - r).into(),

            #[cfg(feature = "chrono")]
            (Value::Duration(l), Value::Duration(r)) => l
                .checked_sub(&r)
                .ok_or_else(|| ExecutionError::Overflow("sub", l.into(), r.into()))
                .map(Value::Duration),
            #[cfg(feature = "chrono")]
            (Value::Timestamp(l), Value::Duration(r)) => checked_op(TsOp::Sub, &l, &r),
            #[cfg(feature = "chrono")]
            (Value::Timestamp(l), Value::Timestamp(r)) => {
                Value::Duration(l.signed_duration_since(r)).into()
            }
            (left, right) => Err(ExecutionError::UnsupportedBinaryOperator(
                "sub", left, right,
            )),
        }
    }
}

impl ops::Div<Value> for Value {
    type Output = ResolveResult;

    #[inline(always)]
    fn div(self, rhs: Value) -> Self::Output {
        match (self, rhs) {
            (Value::Int(l), Value::Int(r)) => {
                if r == 0 {
                    Err(ExecutionError::DivisionByZero(l.into()))
                } else {
                    l.checked_div(r)
                        .ok_or_else(|| ExecutionError::Overflow("div", l.into(), r.into()))
                        .map(Value::Int)
                }
            }

            (Value::UInt(l), Value::UInt(r)) => l
                .checked_div(r)
                .ok_or_else(|| ExecutionError::DivisionByZero(l.into()))
                .map(Value::UInt),

            (Value::Float(l), Value::Float(r)) => Value::Float(l / r).into(),

            (left, right) => Err(ExecutionError::UnsupportedBinaryOperator(
                "div", left, right,
            )),
        }
    }
}

impl ops::Mul<Value> for Value {
    type Output = ResolveResult;

    #[inline(always)]
    fn mul(self, rhs: Value) -> Self::Output {
        match (self, rhs) {
            (Value::Int(l), Value::Int(r)) => l
                .checked_mul(r)
                .ok_or_else(|| ExecutionError::Overflow("mul", l.into(), r.into()))
                .map(Value::Int),

            (Value::UInt(l), Value::UInt(r)) => l
                .checked_mul(r)
                .ok_or_else(|| ExecutionError::Overflow("mul", l.into(), r.into()))
                .map(Value::UInt),

            (Value::Float(l), Value::Float(r)) => Value::Float(l * r).into(),

            (left, right) => Err(ExecutionError::UnsupportedBinaryOperator(
                "mul", left, right,
            )),
        }
    }
}

impl ops::Rem<Value> for Value {
    type Output = ResolveResult;

    #[inline(always)]
    fn rem(self, rhs: Value) -> Self::Output {
        match (self, rhs) {
            (Value::Int(l), Value::Int(r)) => {
                if r == 0 {
                    Err(ExecutionError::RemainderByZero(l.into()))
                } else {
                    l.checked_rem(r)
                        .ok_or_else(|| ExecutionError::Overflow("rem", l.into(), r.into()))
                        .map(Value::Int)
                }
            }

            (Value::UInt(l), Value::UInt(r)) => l
                .checked_rem(r)
                .ok_or_else(|| ExecutionError::RemainderByZero(l.into()))
                .map(Value::UInt),

            (left, right) => Err(ExecutionError::UnsupportedBinaryOperator(
                "rem", left, right,
            )),
        }
    }
}

/// Op represents a binary arithmetic operation supported on a timestamp
///
#[cfg(feature = "chrono")]
enum TsOp {
    Add,
    Sub,
}

#[cfg(feature = "chrono")]
impl TsOp {
    fn str(&self) -> &'static str {
        match self {
            TsOp::Add => "add",
            TsOp::Sub => "sub",
        }
    }
}

/// Performs a checked arithmetic operation [`TsOp`] on a timestamp and a duration and ensures that
/// the resulting timestamp does not overflow the data type internal limits, as well as the timestamp
/// limits defined in the cel-spec. See [`MAX_TIMESTAMP`] and [`MIN_TIMESTAMP`] for more details.
#[cfg(feature = "chrono")]
fn checked_op(
    op: TsOp,
    lhs: &chrono::DateTime<chrono::FixedOffset>,
    rhs: &chrono::Duration,
) -> ResolveResult {
    // Add lhs and rhs together, checking for data type overflow
    let result = match op {
        TsOp::Add => lhs.checked_add_signed(*rhs),
        TsOp::Sub => lhs.checked_sub_signed(*rhs),
    }
    .ok_or_else(|| ExecutionError::Overflow(op.str(), (*lhs).into(), (*rhs).into()))?;

    // Check for cel-spec limits
    if result > *MAX_TIMESTAMP || result < *MIN_TIMESTAMP {
        Err(ExecutionError::Overflow(
            op.str(),
            (*lhs).into(),
            (*rhs).into(),
        ))
    } else {
        Value::Timestamp(result).into()
    }
}

#[cfg(test)]
mod tests {
    use crate::{objects::Key, Context, ExecutionError, Program, Value};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn test_indexed_map_access() {
        let mut context = Context::default();
        let mut headers = HashMap::new();
        headers.insert("Content-Type", "application/json".to_string());
        context.add_variable_from_value("headers", headers);

        let program = Program::compile("headers[\"Content-Type\"]").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, "application/json".into());
    }

    #[test]
    fn test_numeric_map_access() {
        let mut context = Context::default();
        let mut numbers = HashMap::new();
        numbers.insert(Key::Uint(1), "one".to_string());
        context.add_variable_from_value("numbers", numbers);

        let program = Program::compile("numbers[1u]").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, "one".into());
    }

    #[test]
    fn test_heterogeneous_compare() {
        let context = Context::default();

        let program = Program::compile("1 < uint(2)").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, true.into());

        let program = Program::compile("1 < 1.1").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, true.into());

        let program = Program::compile("uint(0) > -10").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(
            value,
            true.into(),
            "negative signed ints should be less than uints"
        );
    }

    #[test]
    fn test_float_compare() {
        let context = Context::default();

        let program = Program::compile("1.0 > 0.0").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, true.into());

        let program = Program::compile("double('NaN') == double('NaN')").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, false.into(), "NaN should not equal itself");

        let program = Program::compile("1.0 > double('NaN')").unwrap();
        let result = program.execute(&context);
        assert!(
            result.is_err(),
            "NaN should not be comparable with inequality operators"
        );
    }

    #[test]
    fn test_invalid_compare() {
        let context = Context::default();

        let program = Program::compile("{} == []").unwrap();
        let value = program.execute(&context).unwrap();
        assert_eq!(value, false.into());
    }

    #[test]
    fn test_size_fn_var() {
        let program = Program::compile("size(requests) + size == 5").unwrap();
        let mut context = Context::default();
        let requests = vec![Value::Int(42), Value::Int(42)];
        context
            .add_variable("requests", Value::list(requests))
            .unwrap();
        context.add_variable("size", Value::Int(3)).unwrap();
        assert_eq!(program.execute(&context).unwrap(), Value::Bool(true));
    }

    fn test_execution_error(program: &str, expected: ExecutionError) {
        let program = Program::compile(program).unwrap();
        let result = program.execute(&Context::default());
        assert_eq!(result.unwrap_err(), expected);
    }

    #[test]
    fn test_invalid_sub() {
        test_execution_error(
            "'foo' - 10",
            ExecutionError::UnsupportedBinaryOperator("sub", "foo".into(), Value::Int(10)),
        );
    }

    #[test]
    fn test_invalid_add() {
        test_execution_error(
            "'foo' + 10",
            ExecutionError::UnsupportedBinaryOperator("add", "foo".into(), Value::Int(10)),
        );
    }

    #[test]
    fn test_invalid_div() {
        test_execution_error(
            "'foo' / 10",
            ExecutionError::UnsupportedBinaryOperator("div", "foo".into(), Value::Int(10)),
        );
    }

    #[test]
    fn test_invalid_rem() {
        test_execution_error(
            "'foo' % 10",
            ExecutionError::UnsupportedBinaryOperator("rem", "foo".into(), Value::Int(10)),
        );
    }

    #[test]
    fn out_of_bound_list_access() {
        let program = Program::compile("list[10]").unwrap();
        let mut context = Context::default();
        context.add_variable("list", Value::list(vec![])).unwrap();
        let result = program.execute(&context);
        assert_eq!(
            result,
            Err(ExecutionError::IndexOutOfBounds(Value::Int(10)))
        );
    }

    #[test]
    fn out_of_bound_list_access_negative() {
        let program = Program::compile("list[-1]").unwrap();
        let mut context = Context::default();
        context.add_variable("list", Value::list(vec![])).unwrap();
        let result = program.execute(&context);
        assert_eq!(
            result,
            Err(ExecutionError::IndexOutOfBounds(Value::Int(-1)))
        );
    }

    #[test]
    fn list_access_uint() {
        let program = Program::compile("list[1u]").unwrap();
        let mut context = Context::default();
        context
            .add_variable("list", Value::list(vec![1.into(), 2.into()]))
            .unwrap();
        let result = program.execute(&context);
        assert_eq!(result, Ok(Value::Int(2.into())));
    }

    #[test]
    fn reference_to_value() {
        let test = "example".to_string();
        let direct: Value = test.as_str().into();
        assert_eq!(direct, Value::String(Arc::new(String::from("example"))));

        let vec = vec![test.as_str()];
        let indirect: Value = vec.into();
        assert_eq!(
            indirect,
            Value::list(vec![Value::String(Arc::new(String::from("example")))])
        );
    }

    #[test]
    fn test_short_circuit_and() {
        let mut context = Context::default();
        let data: HashMap<String, String> = HashMap::new();
        context.add_variable_from_value("data", data);

        let program = Program::compile("has(data.x) && data.x.startsWith(\"foo\")").unwrap();
        let value = program.execute(&context);
        println!("{value:?}");
        assert!(
            value.is_ok(),
            "The AND expression should support short-circuit evaluation."
        );
    }

    #[test]
    fn test_or_ignores_err_when_short_circuiting() {
        let mut context = Context::default();
        context.add_variable_from_value("foo", 42);
        context.add_variable_from_value("bar", 42);
        let program = Program::compile("foo || bar > 0").unwrap();
        let value = program.execute(&context);
        assert_eq!(value, Ok(true.into()));

        let program = Program::compile("foo || bar < 0").unwrap();
        let value = program.execute(&context);
        assert!(value.is_err());
    }

    #[test]
    fn test_and_ignores_err_when_short_circuiting() {
        let mut context = Context::default();
        context.add_variable_from_value("foo", 42);
        context.add_variable_from_value("bar", 42);
        let program = Program::compile("foo && bar < 0").unwrap();
        let value = program.execute(&context);
        assert_eq!(value, Ok(false.into()));

        let program = Program::compile("foo && bar > 0").unwrap();
        let value = program.execute(&context);
        assert!(value.is_err());
    }

    #[test]
    fn invalid_int_math() {
        use ExecutionError::*;

        let cases = [
            ("1 / 0", DivisionByZero(1.into())),
            ("1 % 0", RemainderByZero(1.into())),
            (
                &format!("{} + 1", i64::MAX),
                Overflow("add", i64::MAX.into(), 1.into()),
            ),
            (
                &format!("{} - 1", i64::MIN),
                Overflow("sub", i64::MIN.into(), 1.into()),
            ),
            (
                &format!("{} * 2", i64::MAX),
                Overflow("mul", i64::MAX.into(), 2.into()),
            ),
            (
                &format!("{} / -1", i64::MIN),
                Overflow("div", i64::MIN.into(), (-1).into()),
            ),
            (
                &format!("{} % -1", i64::MIN),
                Overflow("rem", i64::MIN.into(), (-1).into()),
            ),
        ];

        for (expr, err) in cases {
            test_execution_error(expr, err);
        }
    }

    #[test]
    fn invalid_uint_math() {
        use ExecutionError::*;

        let cases = [
            ("1u / 0u", DivisionByZero(1u64.into())),
            ("1u % 0u", RemainderByZero(1u64.into())),
            (
                &format!("{}u + 1u", u64::MAX),
                Overflow("add", u64::MAX.into(), 1u64.into()),
            ),
            ("0u - 1u", Overflow("sub", 0u64.into(), 1u64.into())),
            (
                &format!("{}u * 2u", u64::MAX),
                Overflow("mul", u64::MAX.into(), 2u64.into()),
            ),
        ];

        for (expr, err) in cases {
            test_execution_error(expr, err);
        }
    }

    #[test]
    fn test_index_missing_map_key() {
        let mut ctx = Context::default();
        let mut map = HashMap::new();
        map.insert("a".to_string(), Value::Int(1));
        ctx.add_variable_from_value("mymap", map);

        let p = Program::compile(r#"mymap["missing"]"#).expect("Must compile");
        let result = p.execute(&ctx);

        assert!(result.is_err(), "Should error on missing map key");
    }

    mod opaque {
        use crate::objects::{Map, Opaque, OpaqueVal, OptionalValue};
        use crate::parser::Parser;
        use crate::{Context, ExecutionError, FunctionContext, Program, Value};
        use serde::Serialize;
        use std::collections::HashMap;
        use std::fmt::Debug;
        use std::ops::Deref;
        use std::sync::Arc;

        #[derive(Debug, Eq, PartialEq, Serialize)]
        struct MyStruct {
            field: String,
        }

        impl Opaque for MyStruct {
            fn runtime_type_name(&self) -> &str {
                "my_struct"
            }

            #[cfg(feature = "json")]
            fn json(&self) -> Option<serde_json::Value> {
                Some(serde_json::to_value(self).unwrap())
            }
        }

        #[test]
        fn test_opaque_fn() {
            pub fn my_fn(ftx: &FunctionContext) -> Result<Value, ExecutionError> {
                if let Some(Some(opaque)) = ftx.this.as_ref().map(|v| v.downcast_ref::<OpaqueVal>())
                {
                    if opaque.val.runtime_type_name() == "my_struct" {
                        Ok(opaque
                            .val
                            .deref()
                            .downcast_ref::<MyStruct>()
                            .unwrap()
                            .field
                            .clone()
                            .into())
                    } else {
                        Err(ExecutionError::UnexpectedType {
                            got: opaque.val.runtime_type_name().to_string(),
                            want: "my_struct".to_string(),
                        })
                    }
                } else {
                    Err(ExecutionError::UnexpectedType {
                        got: format!("{:?}", ftx.this),
                        want: "Value::Opaque".to_string(),
                    })
                }
            }

            let value = Arc::new(MyStruct {
                field: String::from("value"),
            });

            let mut ctx = Context::default();
            ctx.add_variable_from_value("mine", Value::Opaque(value.clone()));
            ctx.add_function("myFn", my_fn);
            let prog = Program::compile("mine.myFn()").unwrap();
            assert_eq!(
                Ok(Value::String(Arc::new("value".into()))),
                prog.execute(&ctx)
            );
        }

        #[test]
        fn opaque_eq() {
            let value_1 = Arc::new(MyStruct {
                field: String::from("1"),
            });
            let value_2 = Arc::new(MyStruct {
                field: String::from("2"),
            });

            let mut ctx = Context::default();
            ctx.add_variable_from_value("v1", Value::Opaque(value_1.clone()));
            ctx.add_variable_from_value("v1b", Value::Opaque(value_1));
            ctx.add_variable_from_value("v2", Value::Opaque(value_2));
            assert_eq!(
                Program::compile("v2 == v1").unwrap().execute(&ctx),
                Ok(false.into())
            );
            assert_eq!(
                Program::compile("v1 == v1b").unwrap().execute(&ctx),
                Ok(true.into())
            );
            assert_eq!(
                Program::compile("v2 == v2").unwrap().execute(&ctx),
                Ok(true.into())
            );
        }

        #[test]
        fn test_value_holder_dbg() {
            let opaque = Arc::new(MyStruct {
                field: "not so opaque".to_string(),
            });
            let opaque = Value::Opaque(opaque);
            assert_eq!(
                "Opaque<my_struct>(MyStruct { field: \"not so opaque\" })",
                format!("{:?}", opaque)
            );
        }

        #[test]
        #[cfg(feature = "json")]
        fn test_json() {
            let value = Arc::new(MyStruct {
                field: String::from("value"),
            });
            let cel_value = Value::Opaque(value);
            let mut map = serde_json::Map::new();
            map.insert(
                "field".to_string(),
                serde_json::Value::String("value".to_string()),
            );
            assert_eq!(
                cel_value.json().expect("Must convert"),
                serde_json::Value::Object(map)
            );
        }

        #[test]
        fn test_optional() {
            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none()")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::none())))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(1)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::of(Value::Int(1)))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.ofNonZeroValue(0)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::none())))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.ofNonZeroValue(1)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::of(Value::Int(1)))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(1).value()")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(1))
            );
            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().value()")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Err(ExecutionError::FunctionError {
                    function: "value".to_string(),
                    message: "optional.none() dereference".to_string()
                })
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(1).hasValue()")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Bool(true))
            );
            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().hasValue()")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Bool(false))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(1).or(optional.of(2))")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::of(Value::Int(1)))))
            );
            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().or(optional.of(2))")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::of(Value::Int(2)))))
            );
            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().or(optional.none())")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Opaque(Arc::new(OptionalValue::none())))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(1).orValue(5)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(1))
            );
            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().orValue(5)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(5))
            );

            let mut ctx = Context::default();
            ctx.add_variable_from_value("msg", HashMap::from([("field", "value")]));

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("msg.?field")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &ctx),
                Ok(Value::Opaque(Arc::new(OptionalValue::of(Value::String(
                    Arc::new("value".to_string())
                )))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(msg).?field")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &ctx),
                Ok(Value::Opaque(Arc::new(OptionalValue::of(Value::String(
                    Arc::new("value".to_string())
                )))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().?field")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &ctx),
                Ok(Value::Opaque(Arc::new(OptionalValue::none())))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of(msg).?field.orValue('default')")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &ctx),
                Ok(Value::String(Arc::new("value".to_string())))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none().?field.orValue('default')")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &ctx),
                Ok(Value::String(Arc::new("default".to_string())))
            );

            let mut map_ctx = Context::default();
            let mut map = HashMap::new();
            map.insert("a".to_string(), Value::Int(1));
            map_ctx.add_variable_from_value("mymap", map);

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"mymap[?"missing"].orValue(99)"#)
                .expect("Must parse");
            assert_eq!(Value::resolve(&expr, &map_ctx), Ok(Value::Int(99)));

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"mymap[?"missing"].hasValue()"#)
                .expect("Must parse");
            assert_eq!(Value::resolve(&expr, &map_ctx), Ok(Value::Bool(false)));

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"mymap[?"a"].orValue(99)"#)
                .expect("Must parse");
            assert_eq!(Value::resolve(&expr, &map_ctx), Ok(Value::Int(1)));

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"mymap[?"a"].hasValue()"#)
                .expect("Must parse");
            assert_eq!(Value::resolve(&expr, &map_ctx), Ok(Value::Bool(true)));

            let mut list_ctx = Context::default();
            list_ctx.add_variable_from_value(
                "mylist",
                vec![Value::Int(1), Value::Int(2), Value::Int(3)],
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("mylist[?10].orValue(99)")
                .expect("Must parse");
            assert_eq!(Value::resolve(&expr, &list_ctx), Ok(Value::Int(99)));

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("mylist[?1].orValue(99)")
                .expect("Must parse");
            assert_eq!(Value::resolve(&expr, &list_ctx), Ok(Value::Int(2)));

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of([1, 2, 3])[1].orValue(99)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(2))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of([1, 2, 3])[4].orValue(99)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(99))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.none()[1].orValue(99)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(99))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("optional.of([1, 2, 3])[?1].orValue(99)")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Int(2))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("[1, 2, ?optional.of(3), 4]")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::list(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(3),
                    Value::Int(4)
                ]))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("[1, 2, ?optional.none(), 4]")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::list(vec![
                    Value::Int(1),
                    Value::Int(2),
                    Value::Int(4)
                ]))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("[?optional.of(1), ?optional.none(), ?optional.of(3)]")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::list(vec![Value::Int(1), Value::Int(3)]))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"[1, ?mymap[?"missing"], 3]"#)
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &map_ctx),
                Ok(Value::list(vec![Value::Int(1), Value::Int(3)]))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"[1, ?mymap[?"a"], 3]"#)
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &map_ctx),
                Ok(Value::list(vec![
                    Value::Int(1),
                    Value::Int(1),
                    Value::Int(3)
                ]))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse("[?optional.none(), ?optional.none()]")
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::list(vec![]))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"{"a": 1, "b": 2, ?"c": optional.of(3)}"#)
                .expect("Must parse");
            let mut expected_map = HashMap::new();
            expected_map.insert("a".into(), Value::Int(1));
            expected_map.insert("b".into(), Value::Int(2));
            expected_map.insert("c".into(), Value::Int(3));
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Map(Map::object(Arc::from(expected_map))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"{"a": 1, "b": 2, ?"c": optional.none()}"#)
                .expect("Must parse");
            let mut expected_map = HashMap::new();
            expected_map.insert("a".into(), Value::Int(1));
            expected_map.insert("b".into(), Value::Int(2));
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Map(Map::object(Arc::from(expected_map))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"{"a": 1, ?"b": optional.none(), ?"c": optional.of(3)}"#)
                .expect("Must parse");
            let mut expected_map = HashMap::new();
            expected_map.insert("a".into(), Value::Int(1));
            expected_map.insert("c".into(), Value::Int(3));
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Map(Map::object(Arc::from(expected_map))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"{"a": 1, ?"b": mymap[?"missing"]}"#)
                .expect("Must parse");
            let mut expected_map = HashMap::new();
            expected_map.insert("a".into(), Value::Int(1));
            assert_eq!(
                Value::resolve(&expr, &map_ctx),
                Ok(Value::Map(Map::object(Arc::from(expected_map))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"{"x": 10, ?"y": mymap[?"a"]}"#)
                .expect("Must parse");
            let mut expected_map = HashMap::new();
            expected_map.insert("x".into(), Value::Int(10));
            expected_map.insert("y".into(), Value::Int(1));
            assert_eq!(
                Value::resolve(&expr, &map_ctx),
                Ok(Value::Map(Map::object(Arc::from(expected_map))))
            );

            let expr = Parser::default()
                .enable_optional_syntax(true)
                .parse(r#"{?"a": optional.none(), ?"b": optional.none()}"#)
                .expect("Must parse");
            assert_eq!(
                Value::resolve(&expr, &Context::default()),
                Ok(Value::Map(Map::object(Arc::from(HashMap::new()))))
            );
        }
    }

    #[cfg(feature = "structs")]
    mod structs {
        use std::borrow::Cow;
        use std::sync::Arc;

        use crate::{
            common::{
                types::{self, CelBool, CelInt, CelString, CelStruct},
                value::Val,
            },
            env::StructDef,
            Context, Env, ExecutionError, Program, Value,
        };

        #[test]
        fn test_empty_struct() {
            let mut env = Env::stdlib();
            env.add_struct(StructDef::new(String::from("cel.MyStruct")));
            let program = Program::compile("cel.MyStruct {}").unwrap();
            let value = program.execute(&Context::with_env(Arc::new(env))).unwrap();
            match value {
                Value::Struct(s) => assert_eq!(s.name(), "cel.MyStruct"),
                _ => panic!("This can't be!"),
            }
        }

        #[test]
        fn test_struct() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.Problem"))
                    .add_field(String::from("solved"), types::BOOL_TYPE)
                    .add_field(String::from("answer"), types::INT_TYPE),
            );
            let program =
                Program::compile("cel.Problem { solved: 0 != null, answer: 21 * 2 }").unwrap();
            let value = program.execute(&Context::with_env(Arc::new(env))).unwrap();
            match value {
                Value::Struct(s) => {
                    assert_eq!(s.name(), "cel.Problem");
                    assert_eq!(
                        s.field_value("solved"),
                        Some(&CelBool::from(true) as &dyn Val)
                    );
                    assert_eq!(s.field_value("answer"), Some(&CelInt::from(42) as &dyn Val));
                    assert_eq!(s.field_values().len(), 2);
                    assert_eq!(
                        s.field_values().get("solved").cloned(),
                        Some(Arc::new(CelBool::from(true)) as Arc<dyn Val>)
                    );
                    assert_eq!(
                        s.field_values().get("answer").cloned(),
                        Some(Arc::new(CelInt::from(42)) as Arc<dyn Val>)
                    );
                }
                _ => panic!("This can't be!"),
            }
        }

        #[test]
        fn test_struct_field_access() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("some".into(), types::STRING_TYPE),
            );
            let program = Program::compile("cel.MyStruct { some: 'value' }.some").unwrap();
            let value = program.execute(&Context::with_env(env.into())).unwrap();
            assert_eq!(value, Value::String(Arc::new("value".to_owned())));
        }

        #[test]
        fn test_struct_no_such_field() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("some".into(), types::STRING_TYPE),
            );
            let program = Program::compile("cel.MyStruct { not_here: 'value' }").unwrap();
            let result = program.execute(&Context::with_env(env.into()));
            assert_eq!(
                result,
                Err(ExecutionError::NoSuchKey(
                    String::from("field `not_here` on struct `cel.MyStruct`").into()
                ))
            );
        }

        #[test]
        fn test_struct_with_default() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("some".into(), types::STRING_TYPE)
                    .add_field_with_default("here".into(), Box::new(CelString::from("yes"))),
            );
            let program = Program::compile("cel.MyStruct { some: 'value' }.here").unwrap();
            let result = program.execute(&Context::with_env(env.into()));
            assert_eq!(result, Ok(Value::String(Arc::new(String::from("yes")))));
        }

        #[test]
        fn test_struct_with_default_overwritten() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("some".into(), types::STRING_TYPE)
                    .add_field_with_default("here".into(), Box::new(CelString::from("yes"))),
            );
            let program =
                Program::compile("cel.MyStruct { some: 'value', here: 'totally' }.here").unwrap();
            let result = program.execute(&Context::with_env(env.into()));
            assert_eq!(result, Ok(Value::String(Arc::new(String::from("totally")))));
        }

        #[test]
        fn test_struct_has_macro() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("name".into(), types::STRING_TYPE)
                    .add_field("value".into(), types::INT_TYPE),
            );

            let mut my_struct = CelStruct::new("cel.MyStruct".to_owned());
            my_struct.add_field_value(
                "name".to_owned(),
                Cow::<dyn Val>::Owned(Box::new(CelString::from("test"))),
            );
            my_struct.add_field_value(
                "value".to_owned(),
                Cow::<dyn Val>::Owned(Box::new(CelInt::from(42))),
            );

            let mut context = Context::with_env(Arc::new(env));
            context
                .add_variable("my_var", Value::Struct(Arc::new(my_struct)))
                .unwrap();

            let program = Program::compile("has(my_var.name)").unwrap();
            let result = program.execute(&context).unwrap();
            assert_eq!(result, Value::Bool(true));

            let program = Program::compile("has(my_var.missing)").unwrap();
            let result = program.execute(&context).unwrap();
            assert_eq!(result, Value::Bool(false));

            let program =
                Program::compile("has(cel.MyStruct{name: 'foo', value: 1}.name)").unwrap();
            let result = program.execute(&context).unwrap();
            assert_eq!(result, Value::Bool(true));

            let program = Program::compile("has(cel.MyStruct{}.name)").unwrap();
            let result = program.execute(&context).unwrap();
            assert_eq!(result, Value::Bool(false));
        }

        #[test]
        fn test_struct_no_such_field_access() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("some".into(), types::STRING_TYPE),
            );
            let program = Program::compile("cel.MyStruct { some: 'value' }.not_here").unwrap();
            let result = program.execute(&Context::with_env(env.into()));
            assert_eq!(
                result,
                Err(ExecutionError::NoSuchKey(String::from("not_here").into()))
            );
        }

        #[test]
        fn unknown_struct() {
            let program = Program::compile("cel.MyStruct { some: 'value' }.not_here").unwrap();
            let result = program.execute(&Context::default());
            assert_eq!(
                result,
                Err(ExecutionError::UnexpectedType {
                    got: String::from("cel.MyStruct"),
                    want: String::from("known struct")
                })
            );
        }

        #[test]
        fn add_struct_variable_to_context() {
            let mut env = Env::stdlib();
            env.add_struct(
                StructDef::new(String::from("cel.MyStruct"))
                    .add_field("name".into(), types::STRING_TYPE)
                    .add_field("value".into(), types::INT_TYPE),
            );

            let mut my_struct = CelStruct::new("cel.MyStruct".to_owned());
            my_struct.add_field_value(
                "name".to_owned(),
                Cow::<dyn Val>::Owned(Box::new(CelString::from("test"))),
            );
            my_struct.add_field_value(
                "value".to_owned(),
                Cow::<dyn Val>::Owned(Box::new(CelInt::from(42))),
            );

            let mut context = Context::with_env(Arc::new(env));
            context
                .add_variable("my_var", Value::Struct(Arc::new(my_struct)))
                .unwrap();

            let program = Program::compile("my_var.name + ' ' + string(my_var.value)").unwrap();
            let result = program.execute(&context).unwrap();
            assert_eq!(result, Value::String(Arc::new("test 42".to_owned())));
        }
    }
}
