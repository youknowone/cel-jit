use crate::common::ast::{
    operators, CallExpr, ComprehensionExpr, EntryExpr, Expr, ListExpr, LiteralValue,
};
#[cfg(feature = "structs")]
use crate::common::types::CelStruct;
use crate::context::Context;
use crate::runtime::binop::{cel_add, map_contains_key, map_key_refs, values_equal};
use crate::runtime::convert::{
    intern_leaf, interned_as_keyref, interned_list_get, interned_map_get, interned_map_lookup_string,
};
use crate::runtime::error::{take_error, CelErrCode, ERROR_SENTINEL};
use crate::runtime::object::{
    bytes_len, list_int_at, list_len, map_len, new_optional, new_optional_none,
    string_byte_len, w_kind, CelKind, CelRef, ListStrategy, W_BoolObject,
    W_DoubleObject, W_IntColumn, W_IntObject, W_ListObject, W_OptionalObject,
    W_UIntObject,
};
use crate::runtime::object_array::{items_block_items_base, items_capacity};
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
use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::marker::PhantomData;
use std::ops;
use std::ops::Deref;
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
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

/// Tables this small are scanned as [`MapStorage::Entries`]; larger tables
/// are [`MapStorage::Object`].
const ORDERED_SCAN_LIMIT: usize = 8;

/// Insertion-ordered pairs in one `Arc` allocation. Used when the table is
/// small enough to scan.
#[derive(Clone)]
pub struct MapEntries {
    entries: Arc<[(Key, Value)]>,
}

impl MapEntries {
    pub(crate) fn new(entries: Box<[(Key, Value)]>) -> MapEntries {
        debug_assert!(
            entries.len() <= ORDERED_SCAN_LIMIT,
            "Entries tables are at most ORDERED_SCAN_LIMIT long"
        );
        MapEntries {
            entries: Arc::from(compact_replace_on_insert(entries)),
        }
    }

    pub(crate) fn entries_arc(&self) -> &Arc<[(Key, Value)]> {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (Key, Value)> {
        self.entries.iter()
    }
}

/// Compact `entries` with replace-on-insert: a later pair whose [`Key`]
/// matches an earlier one overwrites that earlier value and keeps its
/// position. Only used for tables that fit in [`ORDERED_SCAN_LIMIT`].
fn compact_replace_on_insert(entries: Box<[(Key, Value)]>) -> Box<[(Key, Value)]> {
    let mut entries = entries.into_vec();
    let n = entries.len();
    if n <= 1 {
        return entries.into_boxed_slice();
    }
    let mut write = 0usize;
    for read in 0..n {
        let existing = existing_filled_index(&entries[..write], &entries[read].0);
        replace_or_keep(&mut entries, &mut write, read, existing);
    }
    entries.truncate(write);
    entries.into_boxed_slice()
}

fn replace_or_keep(
    entries: &mut [(Key, Value)],
    write: &mut usize,
    read: usize,
    existing: Option<usize>,
) {
    if let Some(i) = existing {
        entries[i].1 = std::mem::replace(&mut entries[read].1, Value::Null);
    } else {
        if *write != read {
            entries.swap(*write, read);
        }
        *write += 1;
    }
}

fn existing_filled_index(filled: &[(Key, Value)], key: &Key) -> Option<usize> {
    let needle = key.as_keyref();
    filled.iter().position(|(k, _)| k.as_keyref() == needle)
}

fn pairs_position(pairs: &[(Key, Value)], key: &(dyn AsKeyRef + '_)) -> Option<usize> {
    debug_assert!(
        pairs.len() <= ORDERED_SCAN_LIMIT,
        "Entries lookup scans at most ORDERED_SCAN_LIMIT pairs"
    );
    let needle = key.as_keyref();
    pairs.iter().position(|(k, _)| k.as_keyref() == needle)
}

fn pairs_get<'a>(
    pairs: &'a [(Key, Value)],
    key: &(dyn AsKeyRef + '_),
) -> Option<&'a Value> {
    pairs_position(pairs, key).map(|i| &pairs[i].1)
}

fn ordered_tables_eq(a: &[(Key, Value)], b: &[(Key, Value)]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a == b {
        return true;
    }
    a.iter()
        .all(|(k, v)| pairs_get(b, k).is_some_and(|o| o == v))
}

/// Build a public map from `n` source slots. `n <= ORDERED_SCAN_LIMIT`
/// fills one `Arc` slice ([`MapStorage::Entries`]); a larger table is a
/// [`HashMap`]. `REPLACE` scans for a repeated key on the slice path;
/// the unique path skips that scan. `write` may skip (`None`). Written
/// pairs are dropped if `write` fails.
pub(crate) fn try_build_map<E, const REPLACE: bool>(
    n: usize,
    mut write: impl FnMut(usize) -> Result<Option<(Key, Value)>, E>,
) -> Result<Map, E> {
    if n <= ORDERED_SCAN_LIMIT {
        let arc = try_fill_arc::<(Key, Value), E, REPLACE>(
            n,
            write,
            |filled, pair| existing_filled_index(filled, &pair.0),
            |slot, pair| slot.1 = pair.1,
        )?;
        Ok(Map {
            storage: MapStorage::Entries(MapEntries { entries: arc }),
        })
    } else {
        let mut map = HashMap::with_capacity(n);
        for i in 0..n {
            if let Some((key, value)) = write(i)? {
                let prev = map.insert(key, value);
                if !REPLACE {
                    debug_assert!(prev.is_none(), "unique fill requires unique keys");
                }
            }
        }
        Ok(Map::object(Arc::new(map)))
    }
}

/// Write `write(0..cap)` into `slot`. `None` skips a source index. `REPLACE`
/// overwrites an already-written slot that `find_replace` names. Written
/// values are dropped if `write` fails. Returns how many slots were filled.
fn fill_uninit_slots<T, E, const REPLACE: bool>(
    slot: &mut [MaybeUninit<T>],
    mut write: impl FnMut(usize) -> Result<Option<T>, E>,
    mut find_replace: impl FnMut(&[T], &T) -> Option<usize>,
    mut apply_replace: impl FnMut(&mut T, T),
) -> Result<usize, E> {
    struct Guard<'a, T> {
        slot: &'a mut [MaybeUninit<T>],
        filled: usize,
    }
    impl<T> Drop for Guard<'_, T> {
        fn drop(&mut self) {
            for i in 0..self.filled {
                // SAFETY: `slot[i]` was written before `filled` advanced past i.
                unsafe { self.slot[i].assume_init_drop() };
            }
        }
    }
    let cap = slot.len();
    let mut guard = Guard { slot, filled: 0 };
    for i in 0..cap {
        let Some(item) = write(i)? else {
            continue;
        };
        if REPLACE {
            let filled = unsafe {
                // SAFETY: `slot[0..filled]` was written.
                std::slice::from_raw_parts(guard.slot.as_ptr() as *const T, guard.filled)
            };
            if let Some(at) = find_replace(filled, &item) {
                // SAFETY: `at` is an index into `0..filled`, which was written.
                apply_replace(unsafe { guard.slot[at].assume_init_mut() }, item);
                continue;
            }
        }
        guard.slot[guard.filled].write(item);
        guard.filled += 1;
    }
    let filled = guard.filled;
    core::mem::forget(guard);
    Ok(filled)
}

/// One `Arc<[T]>` allocation. If the final length is less than `cap`, a
/// second exact-size allocation holds the filled prefix.
fn try_fill_arc<T, E, const REPLACE: bool>(
    cap: usize,
    write: impl FnMut(usize) -> Result<Option<T>, E>,
    find_replace: impl FnMut(&[T], &T) -> Option<usize>,
    apply_replace: impl FnMut(&mut T, T),
) -> Result<Arc<[T]>, E> {
    if cap == 0 {
        return Ok(Arc::from([]));
    }
    let mut uninit: Arc<[MaybeUninit<T>]> = Arc::new_uninit_slice(cap);
    let slot = Arc::get_mut(&mut uninit).expect("unique");
    let filled = fill_uninit_slots::<T, E, REPLACE>(slot, write, find_replace, apply_replace)?;
    let arc = if filled == cap {
        // SAFETY: every index in 0..cap was written.
        unsafe { uninit.assume_init() }
    } else if filled == 0 {
        Arc::from([])
    } else {
        let mut exact: Arc<[MaybeUninit<T>]> = Arc::new_uninit_slice(filled);
        {
            let dst = Arc::get_mut(&mut exact).expect("unique");
            let src = Arc::get_mut(&mut uninit).expect("unique");
            for i in 0..filled {
                // SAFETY: `src[i]` is initialized; the read moves it so the
                // leftover `MaybeUninit` drop is a no-op.
                dst[i].write(unsafe { src[i].assume_init_read() });
            }
        }
        // SAFETY: every index in 0..filled was written.
        unsafe { exact.assume_init() }
    };
    Ok(arc)
}

/// How a [`Map`] holds its entries -- the map counterpart of [`ListStorage`].
#[derive(Clone)]
pub enum MapStorage {
    /// An owned table of boxed entries.
    Object(Arc<HashMap<Key, Value>>),
    /// Insertion-ordered pairs small enough to scan.
    Entries(MapEntries),
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
    /// An index into the fixed table of CEL type names, which is the whole of a
    /// type value: nothing indexes, adds or iterates one, so there is no
    /// payload for the index to stand in for.
    Type,
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
                    ScalarBank::Type => crate::common::types::type_const_value(word),
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

    /// A map over an insertion-ordered table of at most [`ORDERED_SCAN_LIMIT`]
    /// pairs.
    pub fn entries(entries: MapEntries) -> Map {
        debug_assert!(entries.len() <= ORDERED_SCAN_LIMIT);
        Map {
            storage: MapStorage::Entries(entries),
        }
    }

    /// Compact `entries` with replace-on-insert. Tables that fit in
    /// [`ORDERED_SCAN_LIMIT`] are [`MapStorage::Entries`]; larger tables
    /// are [`MapStorage::Object`].
    pub fn ordered(entries: Box<[(Key, Value)]>) -> Map {
        if entries.len() <= ORDERED_SCAN_LIMIT {
            Map {
                storage: MapStorage::Entries(MapEntries::new(entries)),
            }
        } else {
            let mut map = HashMap::with_capacity(entries.len());
            for (key, value) in Vec::from(entries) {
                map.insert(key, value);
            }
            Map::object(Arc::new(map))
        }
    }

    pub(crate) fn from_linked_entries(entries: Arc<[(Key, Value)]>) -> Map {
        debug_assert!(entries.len() <= ORDERED_SCAN_LIMIT);
        Map {
            storage: MapStorage::Entries(MapEntries { entries }),
        }
    }

    fn ordered_pairs(&self) -> Option<&[(Key, Value)]> {
        match &self.storage {
            MapStorage::Entries(e) => Some(&e.entries),
            _ => None,
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

    /// True when both maps name the same object table, entries table, or
    /// record row.
    pub fn ptr_eq(&self, other: &Map) -> bool {
        match (&self.storage, &other.storage) {
            (MapStorage::Object(a), MapStorage::Object(b)) => Arc::ptr_eq(a, b),
            (MapStorage::Entries(a), MapStorage::Entries(b)) => {
                Arc::ptr_eq(a.entries_arc(), b.entries_arc())
            }
            (
                MapStorage::Record {
                    schema: sa,
                    index: ia,
                },
                MapStorage::Record {
                    schema: sb,
                    index: ib,
                },
            ) => Arc::ptr_eq(sa, sb) && ia == ib,
            _ => false,
        }
    }

    pub(crate) fn object_arc(&self) -> Option<&Arc<HashMap<Key, Value>>> {
        match &self.storage {
            MapStorage::Object(a) => Some(a),
            _ => None,
        }
    }

    pub(crate) fn entries_arc(&self) -> Option<&Arc<[(Key, Value)]>> {
        match &self.storage {
            MapStorage::Entries(a) => Some(a.entries_arc()),
            _ => None,
        }
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            MapStorage::Object(map) => map.len(),
            MapStorage::Entries(e) => e.len(),
            MapStorage::Record { schema, .. } => schema.field_count(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key: &(dyn AsKeyRef + '_)) -> bool {
        map_has_exact_key(|k| self.has_exact_key(k), key.as_keyref())
    }

    fn has_exact_key(&self, key: KeyRef<'_>) -> bool {
        let key: &(dyn AsKeyRef + '_) = &key;
        match &self.storage {
            MapStorage::Object(map) => map.contains_key(key),
            MapStorage::Entries(e) => pairs_position(&e.entries, key).is_some(),
            MapStorage::Record { schema, .. } => schema.position(key).is_some(),
        }
    }

    /// The value corresponding to the key, boxed on the way out when the
    /// strategy does not already hold it. Implicitly converts between int and
    /// uint keys.
    pub fn get(&self, key: &(dyn AsKeyRef + '_)) -> Option<Cow<'_, Value>> {
        map_get_by_key(|k| self.get_exact(&k), key.as_keyref())
    }

    fn get_exact(&self, key: &(dyn AsKeyRef + '_)) -> Option<Cow<'_, Value>> {
        match &self.storage {
            MapStorage::Object(map) => map.get(key).map(Cow::Borrowed),
            MapStorage::Entries(e) => pairs_get(&e.entries, key).map(Cow::Borrowed),
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
            MapStorage::Entries(e) => MapIter::Entries(e.entries.iter()),
            MapStorage::Record { schema, index } => MapIter::Record {
                schema,
                index: *index,
                field: 0,
            },
        }
    }

    /// The entries as the owned table the rest of the language expects. Free
    /// for [`MapStorage::Object`], and the point at which a record row or an
    /// entries table pays for the hashed representation.
    pub fn to_hashmap(&self) -> HashMap<Key, Value> {
        match &self.storage {
            MapStorage::Object(map) => (**map).clone(),
            MapStorage::Entries(_) | MapStorage::Record { .. } => self
                .iter()
                .map(|(k, v)| (k.clone(), v.into_owned()))
                .collect(),
        }
    }
}

pub enum MapIter<'a> {
    Object(std::collections::hash_map::Iter<'a, Key, Value>),
    Entries(std::slice::Iter<'a, (Key, Value)>),
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
            MapIter::Entries(it) => it.next().map(|(k, v)| (k, Cow::Borrowed(v))),
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
            (MapStorage::Entries(_), MapStorage::Entries(_)) => ordered_tables_eq(
                self.ordered_pairs().expect("ordered"),
                other.ordered_pairs().expect("ordered"),
            ),
            _ => {
                if self.len() != other.len() {
                    return false;
                }
                let (iter_side, get_side) = match (&self.storage, &other.storage) {
                    (MapStorage::Entries(_), _) => (self, other),
                    (_, MapStorage::Entries(_)) => (other, self),
                    _ => (self, other),
                };
                iter_side
                    .iter()
                    .all(|(k, v)| get_side.get(k).is_some_and(|o| *o == *v))
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

/// Map index / field / `has`: exact `Key` first, then the other numeric
/// kind if the value fits. Interned maps call this over a linear scan;
/// the public map calls it over [`HashMap`]. `in` is [`map_has_exact_key`].
#[inline]
pub fn map_get_by_key<T>(
    mut exact: impl FnMut(KeyRef<'_>) -> Option<T>,
    needle: KeyRef<'_>,
) -> Option<T> {
    if let Some(v) = exact(needle) {
        return Some(v);
    }
    match needle {
        KeyRef::Int(k) => exact(KeyRef::Uint(u64::try_from(k).ok()?)),
        KeyRef::Uint(k) => exact(KeyRef::Int(i64::try_from(k).ok()?)),
        _ => None,
    }
}

/// `in` on a map: exact `Key` only, no cross-type numeric fallback.
/// Interned maps call this over a linear scan; the public map over
/// [`HashMap`] / record schema position.
#[inline]
pub fn map_has_exact_key(
    mut exact: impl FnMut(KeyRef<'_>) -> bool,
    needle: KeyRef<'_>,
) -> bool {
    exact(needle)
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
pub trait Opaque: Any + OpaqueEq + AsDebug {
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

/// Thin reference-counted slice: one allocation of a header followed by
/// `len` elements. `Arc<[T]>` is a fat pointer (16 bytes); putting one in
/// [`ListRef`] would make [`Value`] 32.
///
/// Every list buffer — object, ints, and shared column/record — starts with
/// this header so [`ListBuf::clone`] / [`ListBuf::drop`] bump the count at
/// the untagged address with no kind test. The static empty header starts
/// at 1 and is never released: [`RcSlice::empty`] takes an extra count.
///
/// # Safety
///
/// `ptr` addresses either the process-lifetime empty header or a unique
/// allocation of [`rc_slice_layout`] whose `len` initialized `T` values
/// follow the header. `Send`/`Sync` when `T: Send + Sync`.
struct RcSlice<T> {
    ptr: NonNull<RcSliceHeader>,
    _pd: PhantomData<T>,
}

#[repr(C)]
struct RcSliceHeader {
    strong: AtomicUsize,
    len: u32,
    kind: u8,
}

const LIST_TAG_OBJECT: usize = 0;
const LIST_TAG_INTS: usize = 1;
const LIST_TAG_SHARED: usize = 2;
/// Low bit on a [`ListBuf`] pointer: set for a tagged `Arc<ListStorage>`.
const LIST_SHARED_BIT: usize = 1;

const _: () = {
    assert!(core::mem::align_of::<ListStorage>() >= 2);
    assert!(core::mem::align_of::<RcSliceHeader>() >= 2);
};

static EMPTY_OBJECT: RcSliceHeader = RcSliceHeader {
    strong: AtomicUsize::new(1),
    len: 0,
    kind: LIST_TAG_OBJECT as u8,
};
static EMPTY_INTS: RcSliceHeader = RcSliceHeader {
    strong: AtomicUsize::new(1),
    len: 0,
    kind: LIST_TAG_INTS as u8,
};

fn rc_slice_layout<T>(len: usize) -> (Layout, usize) {
    let (layout, offset) = Layout::new::<RcSliceHeader>()
        .extend(Layout::array::<T>(len).expect("list buffer layout"))
        .expect("list buffer layout");
    (layout.pad_to_align(), offset)
}

fn rc_header_inc(header: &RcSliceHeader) {
    let old = header.strong.fetch_add(1, AtomicOrdering::Relaxed);
    if old > (isize::MAX as usize) {
        std::process::abort();
    }
}

fn is_empty_header(ptr: *const RcSliceHeader) -> bool {
    std::ptr::eq(ptr, &EMPTY_OBJECT) || std::ptr::eq(ptr, &EMPTY_INTS)
}

fn counted_empty(header: &'static RcSliceHeader) -> NonNull<RcSliceHeader> {
    let ptr = NonNull::from(header);
    // SAFETY: process-lifetime header; the initial 1 is never released.
    rc_header_inc(unsafe { ptr.as_ref() });
    ptr
}

// SAFETY: shared only through the atomic `strong` count; `T: Send + Sync`.
unsafe impl<T: Send + Sync> Send for RcSlice<T> {}
unsafe impl<T: Send + Sync> Sync for RcSlice<T> {}

impl<T> RcSlice<T> {
    fn empty() -> Self {
        let ptr = if core::mem::size_of::<T>() == core::mem::size_of::<i64>()
            && core::mem::align_of::<T>() == core::mem::align_of::<i64>()
        {
            counted_empty(&EMPTY_INTS)
        } else {
            counted_empty(&EMPTY_OBJECT)
        };
        RcSlice {
            ptr,
            _pd: PhantomData,
        }
    }

    fn len(&self) -> usize {
        // SAFETY: `ptr` is the empty header or a live allocation.
        unsafe { self.ptr.as_ref().len as usize }
    }

    #[allow(dead_code)]
    fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` is the empty header or a live allocation.
        unsafe { rc_slice_as_slice(self.ptr) }
    }

    fn from_vec(v: Vec<T>) -> Self {
        if v.is_empty() {
            return Self::empty();
        }
        match try_fill_rc::<T, std::convert::Infallible>(v.len(), {
            let mut iter = v.into_iter();
            move |_| Ok(iter.next())
        }) {
            Ok(s) => s,
            Err(e) => match e {},
        }
    }
}

impl<T> Clone for RcSlice<T> {
    fn clone(&self) -> Self {
        // SAFETY: empty header or a live allocation; count is at offset 0.
        rc_header_inc(unsafe { self.ptr.as_ref() });
        RcSlice {
            ptr: self.ptr,
            _pd: PhantomData,
        }
    }
}

impl<T> Drop for RcSlice<T> {
    fn drop(&mut self) {
        let ptr = self.ptr.as_ptr();
        // SAFETY: empty header or a live allocation.
        if unsafe { (*ptr).strong.fetch_sub(1, AtomicOrdering::Release) } != 1 {
            return;
        }
        std::sync::atomic::fence(AtomicOrdering::Acquire);
        if is_empty_header(ptr) {
            return;
        }
        // SAFETY: last owner of a heap buffer.
        unsafe { rc_slice_drop_in_place::<T>(self.ptr) };
    }
}

struct UninitRcSlice<T> {
    slice: ManuallyDrop<RcSlice<T>>,
    filled: usize,
    cap: usize,
}

impl<T> UninitRcSlice<T> {
    fn new(cap: usize) -> Self {
        if cap > u32::MAX as usize {
            panic!(
                "a list buffer past {} elements cannot be allocated",
                u32::MAX
            );
        }
        if cap == 0 {
            return UninitRcSlice {
                slice: ManuallyDrop::new(RcSlice::empty()),
                filled: 0,
                cap: 0,
            };
        }
        let (layout, _) = rc_slice_layout::<T>(cap);
        // SAFETY: `layout` is non-zero (`cap > 0`) and correctly aligned
        // for the header plus `cap` elements.
        let raw = unsafe { alloc(layout) };
        if raw.is_null() {
            handle_alloc_error(layout);
        }
        let header = raw.cast::<RcSliceHeader>();
        // SAFETY: `raw` is a unique allocation of `layout`, large enough
        // for the header.
        unsafe {
            header.write(RcSliceHeader {
                strong: AtomicUsize::new(1),
                len: 0,
                kind: 0,
            });
        }
        UninitRcSlice {
            slice: ManuallyDrop::new(RcSlice {
                // SAFETY: `raw` is a non-null allocation of `layout`.
                ptr: unsafe { NonNull::new_unchecked(header) },
                _pd: PhantomData,
            }),
            filled: 0,
            cap,
        }
    }

    fn slot(&mut self) -> &mut [MaybeUninit<T>] {
        if self.cap == 0 {
            return &mut [];
        }
        let (_, offset) = rc_slice_layout::<T>(self.cap);
        // SAFETY: unique allocation of [`rc_slice_layout`] for `cap`
        // elements; nothing else aliases this array until `finish`.
        unsafe {
            let data = self.slice.ptr.as_ptr().cast::<u8>().add(offset);
            std::slice::from_raw_parts_mut(data.cast::<MaybeUninit<T>>(), self.cap)
        }
    }

    fn finish(mut self) -> RcSlice<T> {
        let filled = self.filled;
        let cap = self.cap;
        if filled == 0 {
            return RcSlice::empty();
        }
        if filled != cap {
            let mut exact = UninitRcSlice::new(filled);
            // SAFETY: `self` uniquely owns `filled` initialized elements;
            // `exact` is a unique allocation of that many slots. The read
            // moves each element so `self`'s drop must not run them.
            unsafe {
                let src = {
                    let (_, offset) = rc_slice_layout::<T>(cap);
                    self.slice.ptr.as_ptr().cast::<u8>().add(offset).cast::<T>()
                };
                let dst = {
                    let (_, offset) = rc_slice_layout::<T>(filled);
                    exact
                        .slice
                        .ptr
                        .as_ptr()
                        .cast::<u8>()
                        .add(offset)
                        .cast::<T>()
                };
                for i in 0..filled {
                    dst.add(i).write(src.add(i).read());
                }
            }
            exact.filled = filled;
            self.filled = 0;
            return exact.finish();
        }
        // SAFETY: unique allocation; every slot in 0..cap was written.
        // `filled == cap <= u32::MAX` because [`UninitRcSlice::new`] rejected
        // a larger `cap`.
        unsafe {
            (*self.slice.ptr.as_ptr()).len = filled as u32;
        }
        // SAFETY: transferring the unique strong count to the returned handle.
        let slice = unsafe { ManuallyDrop::take(&mut self.slice) };
        self.filled = 0;
        self.cap = 0;
        core::mem::forget(self);
        slice
    }
}

impl<T> Drop for UninitRcSlice<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            // SAFETY: counted empty handle from [`RcSlice::empty`].
            unsafe { ManuallyDrop::drop(&mut self.slice) };
            return;
        }
        // SAFETY: unique allocation of `cap` slots; the first `filled`
        // elements were written and have not been moved. The `RcSlice`
        // field is not dropped: that would free the same allocation.
        unsafe {
            let (layout, offset) = rc_slice_layout::<T>(self.cap);
            let data = self
                .slice
                .ptr
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<T>();
            for i in 0..self.filled {
                std::ptr::drop_in_place(data.add(i));
            }
            dealloc(self.slice.ptr.as_ptr().cast(), layout);
        }
    }
}

/// Borrow the elements of a live [`RcSlice`] header.
///
/// # Safety
///
/// `ptr` is the empty header or a live allocation of `T`, and the borrow
/// does not outlive that allocation.
unsafe fn rc_slice_as_slice<'a, T>(ptr: NonNull<RcSliceHeader>) -> &'a [T] {
    // SAFETY: caller: `ptr` is the empty header or a live allocation.
    let n = unsafe { ptr.as_ref().len as usize };
    if n == 0 {
        return &[];
    }
    let (_, offset) = rc_slice_layout::<T>(n);
    // SAFETY: element array of `n` initialized `T` follows the header.
    let data = unsafe { ptr.as_ptr().cast::<u8>().add(offset).cast::<T>() };
    unsafe { std::slice::from_raw_parts(data, n) }
}

fn try_fill_rc<T, E>(
    cap: usize,
    write: impl FnMut(usize) -> Result<Option<T>, E>,
) -> Result<RcSlice<T>, E> {
    if cap == 0 {
        return Ok(RcSlice::empty());
    }
    let n = u32::try_from(cap).expect("a list buffer past u32::MAX elements") as usize;
    let mut uninit = UninitRcSlice::new(n);
    uninit.filled = fill_uninit_slots::<T, E, false>(
        uninit.slot(),
        write,
        |_, _| None,
        |_, _| {},
    )?;
    Ok(uninit.finish())
}

/// Thin pointer at an object buffer, an int buffer, or a shared
/// [`ListStorage`] (column / record).
///
/// Owned kinds keep a [`RcSliceHeader`] whose `kind` word names the payload.
/// A shared buffer is the `Arc<ListStorage>` pointer with
/// [`LIST_SHARED_BIT`] set, so clone/drop is one branch on that bit: tagged
/// bumps the `Arc`, untagged bumps the header. `PhantomData<*const ()>`
/// keeps the handle `!Send + !Sync` because a buffer owns [`Value`]s.
pub(crate) struct ListBuf {
    ptr: NonNull<u8>,
    _not_send_sync: PhantomData<*const ()>,
}

impl ListBuf {
    fn from_header(ptr: NonNull<RcSliceHeader>) -> Self {
        ListBuf {
            ptr: ptr.cast(),
            _not_send_sync: PhantomData,
        }
    }

    fn from_object(s: RcSlice<Value>) -> Self {
        if !is_empty_header(s.ptr.as_ptr()) {
            // SAFETY: unique heap object header.
            unsafe { (*s.ptr.as_ptr()).kind = LIST_TAG_OBJECT as u8 };
        }
        let ptr = s.ptr;
        core::mem::forget(s);
        ListBuf::from_header(ptr)
    }

    fn from_ints(s: RcSlice<i64>) -> Self {
        if !is_empty_header(s.ptr.as_ptr()) {
            // SAFETY: unique heap int header.
            unsafe { (*s.ptr.as_ptr()).kind = LIST_TAG_INTS as u8 };
        }
        let ptr = s.ptr;
        core::mem::forget(s);
        ListBuf::from_header(ptr)
    }

    fn from_shared(arc: Arc<ListStorage>) -> Self {
        let raw = Arc::into_raw(arc) as *mut u8;
        debug_assert_eq!(raw.addr() & LIST_SHARED_BIT, 0);
        ListBuf {
            // SAFETY: `Arc::into_raw` is aligned to `ListStorage` (>= 2), so
            // setting the low bit does not collide with a live address.
            // `map_addr` keeps the allocation's provenance.
            ptr: unsafe { NonNull::new_unchecked(raw.map_addr(|a| a | LIST_SHARED_BIT)) },
            _not_send_sync: PhantomData,
        }
    }

    fn is_shared(&self) -> bool {
        self.ptr.as_ptr().addr() & LIST_SHARED_BIT != 0
    }

    fn shared_ptr(&self) -> *const ListStorage {
        self.ptr.as_ptr().map_addr(|a| a & !LIST_SHARED_BIT) as *const ListStorage
    }

    fn tag(&self) -> usize {
        if self.is_shared() {
            return LIST_TAG_SHARED;
        }
        // SAFETY: untagged pointer is a live [`RcSliceHeader`].
        unsafe { (*self.ptr.as_ptr().cast::<RcSliceHeader>()).kind as usize }
    }

    fn raw(&self) -> NonNull<u8> {
        self.ptr
    }

    fn as_raw_tagged(&self) -> *const () {
        self.ptr.as_ptr() as *const ()
    }

    /// Increment the buffer and return a new handle. `ptr` is the header
    /// or tagged `Arc` stored by [`link_public_handle`].
    ///
    /// # Safety
    ///
    /// `ptr` is null, a live [`ListBuf`] header, or a tagged
    /// `Arc<ListStorage>` pointer produced by [`ListBuf::from_shared`].
    unsafe fn clone_from_raw(ptr: *const ()) -> Option<ListBuf> {
        if ptr.is_null() {
            return None;
        }
        let buf = ListBuf {
            ptr: NonNull::new(ptr as *mut u8)?,
            _not_send_sync: PhantomData,
        };
        let out = buf.clone();
        core::mem::forget(buf);
        Some(out)
    }

    fn len(&self) -> usize {
        match self.tag() {
            LIST_TAG_OBJECT => self.object_slice().map_or(0, <[Value]>::len),
            LIST_TAG_INTS => self.ints_slice().map_or(0, <[i64]>::len),
            LIST_TAG_SHARED => self.shared().map_or(0, ListStorage::len),
            _ => 0,
        }
    }

    fn object_slice(&self) -> Option<&[Value]> {
        match self.tag() {
            // SAFETY: tag names an object [`RcSlice<Value>`] header.
            LIST_TAG_OBJECT => Some(unsafe { rc_slice_as_slice(self.ptr.cast()) }),
            LIST_TAG_SHARED => match self.shared()? {
                ListStorage::Object(v) => Some(v.as_slice()),
                _ => None,
            },
            _ => None,
        }
    }

    fn ints_slice(&self) -> Option<&[i64]> {
        match self.tag() {
            // SAFETY: tag names an int [`RcSlice<i64>`] header.
            LIST_TAG_INTS => Some(unsafe { rc_slice_as_slice(self.ptr.cast()) }),
            LIST_TAG_SHARED => match self.shared()? {
                ListStorage::Ints(v) => Some(v.as_slice()),
                _ => None,
            },
            _ => None,
        }
    }

    fn shared(&self) -> Option<&ListStorage> {
        if !self.is_shared() {
            return None;
        }
        // SAFETY: tagged pointer produced by [`ListBuf::from_shared`]; the
        // `Arc` is live for the lifetime of this handle.
        Some(unsafe { &*self.shared_ptr() })
    }

    fn shared_identity(&self) -> Option<*const ListStorage> {
        self.is_shared().then(|| self.shared_ptr())
    }

    fn element_at(&self, index: usize) -> Value {
        if let Some(v) = self.object_slice() {
            return v[index].clone();
        }
        if let Some(v) = self.ints_slice() {
            return Value::Int(v[index]);
        }
        match self.shared() {
            Some(s) => s.element_at(index),
            None => panic!("list buffer has no element at {index}"),
        }
    }
}

impl Clone for ListBuf {
    fn clone(&self) -> Self {
        if self.is_shared() {
            // SAFETY: tagged pointer from [`ListBuf::from_shared`]; the `Arc`
            // is live.
            unsafe { Arc::increment_strong_count(self.shared_ptr()) };
        } else {
            // SAFETY: untagged pointer is a live [`RcSliceHeader`].
            rc_header_inc(unsafe { self.ptr.cast::<RcSliceHeader>().as_ref() });
        }
        ListBuf {
            ptr: self.ptr,
            _not_send_sync: PhantomData,
        }
    }
}

impl Drop for ListBuf {
    fn drop(&mut self) {
        if self.is_shared() {
            // SAFETY: tagged pointer from [`ListBuf::from_shared`]; this
            // handle owns one strong count.
            unsafe { Arc::decrement_strong_count(self.shared_ptr()) };
            return;
        }
        let header = self.ptr.cast::<RcSliceHeader>();
        // SAFETY: untagged pointer is a live [`RcSliceHeader`].
        if unsafe { header.as_ref().strong.fetch_sub(1, AtomicOrdering::Release) } != 1 {
            return;
        }
        std::sync::atomic::fence(AtomicOrdering::Acquire);
        // SAFETY: last owner of this owned buffer.
        unsafe { list_buf_drop_slow(header) };
    }
}

/// Last-owner drop: dispatch on [`RcSliceHeader::kind`], drop elements /
/// storage, deallocate.
///
/// # Safety
///
/// `ptr` is a unique remaining handle of a heap buffer (not a static empty
/// header, which never reaches a zero count).
#[inline(never)]
unsafe fn list_buf_drop_slow(ptr: NonNull<RcSliceHeader>) {
    if is_empty_header(ptr.as_ptr()) {
        return;
    }
    // SAFETY: unique heap header.
    let kind = unsafe { ptr.as_ref().kind as usize };
    match kind {
        LIST_TAG_OBJECT => {
            // SAFETY: object buffer; unique owner; `len` elements follow.
            unsafe { rc_slice_drop_in_place::<Value>(ptr) };
        }
        LIST_TAG_INTS => {
            // SAFETY: int buffer; unique owner; `len` elements follow.
            unsafe { rc_slice_drop_in_place::<i64>(ptr) };
        }
        _ => unreachable!("list buffer kind"),
    }
}

/// Drop `len` elements and free an [`RcSlice`] allocation.
///
/// # Safety
///
/// `ptr` is a unique heap [`RcSliceHeader`] of `T` whose `len` elements
/// were written when the buffer was closed.
unsafe fn rc_slice_drop_in_place<T>(ptr: NonNull<RcSliceHeader>) {
    let raw = ptr.as_ptr();
    // SAFETY: unique heap header.
    let n = unsafe { (*raw).len as usize };
    let (layout, offset) = rc_slice_layout::<T>(n);
    // SAFETY: element array of `n` initialized `T` follows the header.
    unsafe {
        let data = raw.cast::<u8>().add(offset).cast::<T>();
        for i in 0..n {
            std::ptr::drop_in_place(data.add(i));
        }
        dealloc(raw.cast(), layout);
    }
}

/// How a list's elements are stored.
///
/// A list built element by element owns a `Vec<Value>` and is what every
/// caller has always had. A list produced by the batch tier instead names a
/// window into a buffer the whole batch shares: the elements are already laid
/// out contiguously and unboxed, so a row costs no allocation and no
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
    /// Integers, owned by this list and boxed on the way out.
    /// `IntegerListStrategy`: what a list literal or a comprehension whose
    /// every element was an integer closes as. Unlike a [`ListStorage::Column`]
    /// the buffer is the list's own, so closing one moves it rather than
    /// copying it.
    Ints(Vec<i64>),
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
            ListStorage::Ints(v) => v.len(),
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
            ListStorage::Ints(v) => Value::Int(v[index]),
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
/// An owned object or int list is one allocation: a reference-counted header
/// followed by the inline element array. Column and record lists keep a
/// shared [`ListStorage`] so a batch of windows still shares one buffer:
/// [`ListRef::window`] tags that `Arc` and does not allocate.
///
/// Neither `Send` nor `Sync`. A list owns [`Value`]s, which are neither.
///
/// ```compile_fail
/// fn needs_send<T: Send>(_: T) {}
/// needs_send(cel::objects::ListRef::from(Vec::<cel::Value>::new()));
/// ```
///
/// ```compile_fail
/// fn needs_sync<T: Sync>(_: &T) {}
/// needs_sync(&cel::objects::ListRef::from(Vec::<cel::Value>::new()));
/// ```
///
/// Every consumer goes through the accessors below rather than matching a
/// storage variant, so adding a strategy does not reopen the call sites.
#[derive(Clone)]
pub struct ListRef {
    buf: ListBuf,
    /// 32-bit so a `Value` stays 24 bytes. A buffer is checked against this
    /// bound when the window is built rather than truncated silently.
    start: u32,
    len: u32,
}

impl ListRef {
    #[allow(dead_code)]
    pub(crate) fn storage(&self) -> Option<&ListStorage> {
        self.buf.shared()
    }

    pub(crate) fn public_ptr(&self) -> *const () {
        self.buf.as_raw_tagged()
    }

    pub(crate) fn ints_slice(&self) -> Option<&[i64]> {
        self.buf.ints_slice()
    }

    pub(crate) fn object_slice(&self) -> Option<&[Value]> {
        self.buf.object_slice()
    }

    #[allow(dead_code)]
    pub(crate) fn is_ints(&self) -> bool {
        self.buf.ints_slice().is_some()
    }

    pub(crate) fn is_whole(&self) -> bool {
        self.start == 0 && self.len() == self.buf.len()
    }

    /// Reconstruct a window from a bind-time public link. The offsets were
    /// recorded from a live [`ListRef`] over this buffer.
    pub(crate) fn from_linked(buf: ListBuf, start: u32, len: u32) -> ListRef {
        ListRef { buf, start, len }
    }

    /// # Safety
    ///
    /// `ptr` is null, a header stored by [`link_public_handle`], or a tagged
    /// `Arc<ListStorage>` pointer stored there.
    pub(crate) unsafe fn clone_from_public(ptr: *const ()) -> Option<ListBuf> {
        // SAFETY: forwarded to [`ListBuf::clone_from_raw`]; same contract.
        unsafe { ListBuf::clone_from_raw(ptr) }
    }

    pub(crate) fn try_fill_values<E>(
        n: usize,
        write: impl FnMut(usize) -> Result<Option<Value>, E>,
    ) -> Result<ListRef, E> {
        Ok(ListRef::from_object_buf(try_fill_rc(n, write)?))
    }

    pub(crate) fn try_fill_ints<E>(
        n: usize,
        write: impl FnMut(usize) -> Result<Option<i64>, E>,
    ) -> Result<ListRef, E> {
        Ok(ListRef::from_ints_buf(try_fill_rc(n, write)?))
    }

    fn from_object_buf(buf: RcSlice<Value>) -> ListRef {
        let len = u32::try_from(buf.len()).expect("a list past u32::MAX elements");
        ListRef {
            buf: ListBuf::from_object(buf),
            start: 0,
            len,
        }
    }

    fn from_ints_buf(buf: RcSlice<i64>) -> ListRef {
        let len = u32::try_from(buf.len()).expect("a list past u32::MAX elements");
        ListRef {
            buf: ListBuf::from_ints(buf),
            start: 0,
            len,
        }
    }

    fn from_storage_arc(storage: Arc<ListStorage>, start: usize, len: usize) -> ListRef {
        assert!(
            start + len <= storage.len(),
            "list window {start}..{} runs past the {} element buffer",
            start + len,
            storage.len()
        );
        let (Ok(start_u), Ok(len_u)) = (u32::try_from(start), u32::try_from(len)) else {
            panic!(
                "a list buffer past {} elements cannot be windowed",
                u32::MAX
            );
        };
        ListRef {
            buf: ListBuf::from_shared(storage),
            start: start_u,
            len: len_u,
        }
    }

    pub(crate) fn window_start(&self) -> usize {
        self.start as usize
    }

    /// True when `other` is the same window over the same buffer.
    pub fn ptr_eq(&self, other: &ListRef) -> bool {
        self.shares_storage_with(other) && self.start == other.start && self.len == other.len
    }

    /// The window `storage[start .. start + len]`.
    pub fn window(storage: Arc<ListStorage>, start: usize, len: usize) -> ListRef {
        ListRef::from_storage_arc(storage, start, len)
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
        (index < self.len()).then(|| self.buf.element_at(self.start as usize + index))
    }

    pub fn iter(&self) -> impl Iterator<Item = Value> + '_ {
        (0..self.len()).map(move |i| {
            self.get(i)
                .expect("index below len() is in bounds by construction")
        })
    }

    /// The boxed elements. Free for a whole object buffer, and the point at
    /// which an unboxed strategy pays for the representation the rest of the
    /// language expects.
    pub fn to_vec(&self) -> Vec<Value> {
        match self.whole_object() {
            Some(v) => v.to_vec(),
            None => self.iter().collect(),
        }
    }

    /// The elements as an owned `Vec`.
    pub fn into_vec(self) -> Vec<Value> {
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
    /// Two int lists, or an int list and an object list of ints, stay
    /// [`ListStorage::Ints`]: the elements are copied as words, not boxed.
    pub fn concat(self, other: &ListRef) -> ListRef {
        if let Some(out) = concat_int_lists(&self, other) {
            return out;
        }
        let n = self.len() + other.len();
        let mut left = self.iter();
        let mut right = other.iter();
        match ListRef::try_fill_values::<std::convert::Infallible>(n, |_| {
            Ok(left.next().or_else(|| right.next()))
        }) {
            Ok(out) => out,
            Err(e) => match e {},
        }
    }

    /// Whether both lists read the same buffer.
    ///
    /// This is the invariant the batch door depends on: every row of one output
    /// is a window onto one buffer, so a row costs no allocation. It is
    /// observable rather than private because a change that quietly went back
    /// to a buffer per row would otherwise pass every test.
    pub fn shares_storage_with(&self, other: &ListRef) -> bool {
        if self.buf.tag() != other.buf.tag() {
            return false;
        }
        match self.buf.shared_identity() {
            Some(a) => Some(a) == other.buf.shared_identity(),
            None => self.buf.raw() == other.buf.raw(),
        }
    }

    /// The boxed buffer, when this list is exactly all of one.
    fn whole_object(&self) -> Option<&[Value]> {
        let v = self.buf.object_slice()?;
        (self.start == 0 && self.len() == v.len()).then_some(v)
    }
}

#[inline(never)]
fn concat_int_lists(left: &ListRef, right: &ListRef) -> Option<ListRef> {
    let n = left.len() + right.len();
    if n == 0 {
        // `[] + []` must be the same empty object buffer as `[]`. An empty
        // fill would succeed as an int list without reading either side.
        return None;
    }
    let mut i = 0usize;
    ListRef::try_fill_ints::<()>(n, |_| {
        let src = if i < left.len() {
            int_at(left, i)
        } else {
            int_at(right, i - left.len())
        };
        i += 1;
        match src {
            Some(v) => Ok(Some(v)),
            None => Err(()),
        }
    })
    .ok()
}

fn int_at(list: &ListRef, i: usize) -> Option<i64> {
    let at = list.window_start() + i;
    if let Some(v) = list.ints_slice() {
        return v.get(at).copied();
    }
    if let Some(v) = list.object_slice() {
        return value_as_int(v.get(at)?);
    }
    value_as_int(&list.get(i)?)
}

fn value_as_int(v: &Value) -> Option<i64> {
    match v {
        Value::Int(i) => Some(*i),
        Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::Int => {
            Some(unsafe { (*w.cast::<W_IntObject>()).intval })
        }
        _ => None,
    }
}

impl Default for ListRef {
    fn default() -> Self {
        ListRef::from(Vec::new())
    }
}

impl<T: Into<ListStorage>> From<T> for ListRef {
    fn from(items: T) -> Self {
        match items.into() {
            ListStorage::Object(v) => ListRef::from_object_buf(RcSlice::from_vec(v)),
            ListStorage::Ints(v) => ListRef::from_ints_buf(RcSlice::from_vec(v)),
            other => {
                let len = other.len();
                ListRef::window(Arc::new(other), 0, len)
            }
        }
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
    /// A class-family leaf. The VM keeps values in this form so the public
    /// result is a `W_Root` pointer, not a rebuilt enum. `unpack` restores
    /// the typed variants for match sites that have not moved yet.
    Interned(CelRef),
}

const _: () = {
    assert!(core::mem::size_of::<Value>() == 24);
    assert!(core::mem::size_of::<Map>() == 24);
    assert!(core::mem::size_of::<ListRef>() == 16);
};

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
            Value::Interned(w) => write!(f, "{:?}", self.unpack_interned(*w)),
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
    /// Wrap a live class-family leaf as the public value.
    pub(crate) fn from_interned(w: CelRef) -> Self {
        Value::Interned(w)
    }

    /// The P5 handle: a prebuilt `int`.
    pub fn int(i: i64) -> Self {
        Value::from_interned(crate::runtime::object::new_int(i) as CelRef)
    }

    /// The P5 handle: a prebuilt `bool`.
    pub fn bool(b: bool) -> Self {
        Value::from_interned(crate::runtime::object::new_bool(b) as CelRef)
    }

    /// The P5 handle: the immortal `null`.
    pub fn null() -> Self {
        Value::from_interned(crate::runtime::object::new_null() as CelRef)
    }

    /// The class-family kind. A duration or timestamp that does not fit the
    /// leaf stays public and still reports its kind.
    pub fn kind(&self) -> crate::runtime::object::CelKind {
        if let Some(w) = crate::runtime::convert::intern_leaf(self) {
            return unsafe { crate::runtime::object::w_kind(w) };
        }
        #[cfg(feature = "chrono")]
        match self {
            Value::Duration(_) => return crate::runtime::object::CelKind::Duration,
            Value::Timestamp(_) => return crate::runtime::object::CelKind::Timestamp,
            _ => {}
        }
        panic!("intern_leaf is total for this variant")
    }

    /// Restore the typed variants. An interned leaf that convert cannot
    /// read back is `Null`.
    pub fn unpack(&self) -> Value {
        match self {
            Value::Interned(w) => crate::runtime::convert::interned_to_public(*w),
            other => other.clone(),
        }
    }

    /// [`unpack`] taking ownership, so a value that is already public is
    /// moved rather than cloned.
    pub(crate) fn into_public(self) -> Value {
        match self {
            Value::Interned(w) => crate::runtime::convert::interned_to_public(w),
            other => other,
        }
    }

    fn unpack_interned(&self, w: CelRef) -> Value {
        crate::runtime::convert::interned_to_public(w)
    }

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
            Value::Interned(w) => interned_value_type(*w),
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
            Value::Interned(w) => interned_is_zero(*w),
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

/// Unpack an interned leaf so a value stored in a public container, optional,
/// accumulator, or function argument never holds a pointer into the
/// evaluation region.
#[inline]
pub(crate) fn public_store(v: Value) -> Value {
    v.into_public()
}

impl From<&Value> for Value {
    fn from(value: &Value) -> Self {
        value.clone()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Interned(a), Value::Interned(b)) => interned_eq(*a, *b),
            (Value::Interned(a), other) => interned_eq_public(*a, other),
            (other, Value::Interned(b)) => interned_eq_public(*b, other),
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
        if matches!(self, Value::Interned(_)) || matches!(other, Value::Interned(_)) {
            return self.unpack().partial_cmp(&other.unpack());
        }
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
    #[inline(never)]
    fn of(comprehension: &'a ComprehensionExpr) -> Option<Self> {
        // The accumulator is held outside the context on this path, so nothing
        // evaluated per iteration may read it. The loop condition is the one
        // place a step of this shape still could — `exists` and `all` stop on
        // it — and both macros here emit a constant `true`.
        match &comprehension.loop_cond.expr {
            Expr::Literal(LiteralValue::Boolean(b)) if *b => {}
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

/// The bool-accumulator shape `all` and `exists` expand to.
///
/// `all(x, p)` is `@result && p` with a stop when `@result` is false;
/// `exists(x, p)` is `@result || p` with a stop when `@result` is true.
/// The accumulator is a bool the loop already holds, so the step does not
/// rebind it and the condition does not look it up.
struct BoolAccu<'a> {
    /// `true` for `all` (AND, stop on false); `false` for `exists` (OR, stop
    /// on true).
    and: bool,
    pred: &'a Expression,
}

impl<'a> BoolAccu<'a> {
    #[inline(never)]
    fn of(comprehension: &'a ComprehensionExpr) -> Option<Self> {
        let accu_var = comprehension.accu_var.as_str();
        let is_accu = |expr: &Expression| matches!(&expr.expr, Expr::Ident(n) if n == accu_var);
        if !is_accu(&comprehension.result) {
            return None;
        }
        let Expr::Call(step) = &comprehension.loop_step.expr else {
            return None;
        };
        if step.args.len() != 2 || !is_accu(&step.args[0]) {
            return None;
        }
        let and = if step.func_name == operators::LOGICAL_AND {
            true
        } else if step.func_name == operators::LOGICAL_OR {
            false
        } else {
            return None;
        };
        let Expr::Call(cond) = &comprehension.loop_cond.expr else {
            return None;
        };
        if cond.func_name != operators::NOT_STRICTLY_FALSE || cond.args.len() != 1 {
            return None;
        }
        let inner = &cond.args[0];
        if and {
            if !is_accu(inner) {
                return None;
            }
        } else {
            match &inner.expr {
                Expr::Call(not)
                    if not.func_name == operators::LOGICAL_NOT
                        && not.args.len() == 1
                        && is_accu(&not.args[0]) => {}
                _ => return None,
            }
        }
        Some(BoolAccu {
            and,
            pred: &step.args[1],
        })
    }
}

/// `all` / `exists` bool-accumulator loop, kept out of [`resolve_inner`]
/// so it does not sit on the map/filter path.
#[inline(never)]
fn eval_bool_accu(
    bool_accu: BoolAccu<'_>,
    accu_init: &Value,
    items: &mut IterItems,
    ctx: &mut Context,
    iter_var: &str,
) -> Result<Value, ExecutionError> {
    let mut accu = interned_as_bool(accu_init).ok_or(NoSuchOverload)?;
    while let Some(item) = items.next() {
        if bool_accu.and {
            if !accu {
                break;
            }
        } else if accu {
            break;
        }
        ctx.rebind(iter_var, item);
        accu = try_bool_value(resolve_inner(bool_accu.pred, ctx))?;
    }
    Ok(Value::Bool(accu))
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
    /// The crate's only walker. It was built arm for arm against the trait-object
    /// walker it replaced, the two held to the same answers by the differential
    /// corpus in `tests/oracle.rs`, which still gates this one alone.
    pub fn resolve_value(expr: &Expression, ctx: &Context) -> Result<Value, ExecutionError> {
        let scope = crate::runtime::heap::enter_eval_for(ctx);
        match resolve_inner(expr, ctx) {
            Ok(v) => Ok(scope.finish(v)),
            Err(e) => Err(e),
        }
    }
}

/// A list literal. All-int *literals* close as [`ListStorage::Ints`] from
/// the AST; any other element list starts in object storage sized to `n`.
/// Optional indices keep the public-element loop.
#[inline(never)]
fn eval_list_literal(list_expr: &ListExpr, ctx: &Context) -> Result<Value, ExecutionError> {
    if !list_expr.optional_indices.is_empty() {
        let n = list_expr.elements.len();
        let mut src = list_expr.elements.iter().enumerate();
        let list = ListRef::try_fill_values(n, |_| {
            let (idx, element) = src.next().expect("n source elements");
            let value = resolve_inner(element, ctx)?;
            if list_expr.optional_indices.contains(&idx) {
                Ok(match optional_view(&value) {
                    OptView::Empty => None,
                    OptView::Present(inner) => Some(public_store(inner)),
                    OptView::Plain => Some(public_store(value)),
                })
            } else {
                Ok(Some(public_store(value)))
            }
        })?;
        return Ok(Value::List(list));
    }
    let n = list_expr.elements.len();
    if n == 0 {
        return Ok(Value::list(Vec::<Value>::new()));
    }
    if list_expr
        .elements
        .iter()
        .all(|e| matches!(&e.expr, Expr::Literal(LiteralValue::Int(_))))
    {
        let mut src = list_expr.elements.iter();
        let list = ListRef::try_fill_ints::<ExecutionError>(n, |_| {
            let Expr::Literal(LiteralValue::Int(v)) = &src.next().expect("n ints").expr else {
                unreachable!()
            };
            Ok(Some(*v))
        })?;
        return Ok(Value::List(list));
    }
    let mut src = list_expr.elements.iter();
    let list = ListRef::try_fill_values(n, |_| {
        Ok(Some(public_store(resolve_inner(src.next().expect("n elements"), ctx)?)))
    })?;
    Ok(Value::List(list))
}

fn resolve_inner(expr: &Expression, ctx: &Context) -> Result<Value, ExecutionError> {
    match &expr.expr {
        Expr::Literal(literal) => Ok(literal.to_value()),
        Expr::Call(call) => {
            if call.args.len() == 3 && call.func_name == operators::CONDITIONAL {
                return if try_bool_value(resolve_inner(&call.args[0], ctx))? {
                    resolve_inner(&call.args[1], ctx)
                } else {
                    resolve_inner(&call.args[2], ctx)
                };
            }
            if call.args.len() == 2 {
                match call.func_name.as_str() {
                    operators::LOGICAL_OR => {
                        let left = try_bool_value(resolve_inner(&call.args[0], ctx));
                        return if Ok(true) == left {
                            Ok(Value::Bool(true))
                        } else {
                            let right = interned_as_bool(&resolve_inner(&call.args[1], ctx)?);
                            match (left, right) {
                                (Ok(false), Some(right)) => Ok(Value::Bool(right)),
                                (Err(_), Some(true)) => Ok(Value::Bool(true)),
                                (left, _) => Err(left.err().unwrap_or(NoSuchOverload)),
                            }
                        };
                    }
                    operators::LOGICAL_AND => {
                        let left = try_bool_value(resolve_inner(&call.args[0], ctx));
                        return if Ok(false) == left {
                            Ok(Value::Bool(false))
                        } else {
                            let right = interned_as_bool(&resolve_inner(&call.args[1], ctx)?);
                            match (left, right) {
                                (Ok(true), Some(right)) => Ok(Value::Bool(right)),
                                (Err(_), Some(false)) => Ok(Value::Bool(false)),
                                (left, _) => Err(left.err().unwrap_or(NoSuchOverload)),
                            }
                        };
                    }
                    operators::EQUALS => {
                        let lhs = resolve_inner(&call.args[0], ctx)?;
                        let rhs = resolve_inner(&call.args[1], ctx)?;
                        return Ok(Value::Bool(lhs == rhs));
                    }
                    operators::NOT_EQUALS => {
                        let lhs = resolve_inner(&call.args[0], ctx)?;
                        let rhs = resolve_inner(&call.args[1], ctx)?;
                        return Ok(Value::Bool(lhs != rhs));
                    }
                    operators::INDEX | operators::OPT_INDEX => {
                        let mut is_optional = call.func_name == operators::OPT_INDEX;
                        let value = resolve_inner(&call.args[0], ctx)?;
                        let value = match optional_view(&value) {
                            OptView::Present(inner) => {
                                is_optional = true;
                                inner
                            }
                            OptView::Empty => return Ok(optional_none()),
                            OptView::Plain => value,
                        };
                        let key = resolve_inner(&call.args[1], ctx)?;
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
                        let operand = resolve_inner(&call.args[0], ctx)?;
                        let field = match resolve_inner(&call.args[1], ctx)? {
                            Value::String(s) => Value::String(s),
                            _ => {
                                return Err(ExecutionError::function_error(
                                    "_?._",
                                    "field must be string",
                                ))
                            }
                        };
                        return Ok(match optional_view(&operand) {
                            // `Optional::map` keeps the outer `Some` and
                            // substitutes `optional.none` for a missing
                            // field, so a miss nests one optional inside
                            // another. Mirrored, not corrected, here.
                            OptView::Empty => optional_none(),
                            OptView::Present(inner) => optional_of(
                                value_index(&inner, &field).unwrap_or_else(|_| optional_none()),
                            ),
                            OptView::Plain => optional_of(value_index(&operand, &field)?),
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
                        let lhs = resolve_inner(&call.args[0], ctx)?;
                        let rhs = resolve_inner(&call.args[1], ctx)?;
                        return Ok(Value::Bool(value_contains(&rhs, &lhs)?));
                    }
                    _ => (),
                }
            }
            if call.args.len() == 1 {
                match call.func_name.as_str() {
                    operators::LOGICAL_NOT => {
                        return match interned_as_bool(&resolve_inner(&call.args[0], ctx)?) {
                            Some(b) => Ok(Value::Bool(!b)),
                            None => Err(ExecutionError::NoSuchOverload),
                        };
                    }
                    operators::NEGATE => {
                        let val = resolve_inner(&call.args[0], ctx)?;
                        return value_negate(val);
                    }
                    operators::NOT_STRICTLY_FALSE => {
                        return Ok(Value::Bool(
                            try_bool_value(resolve_inner(&call.args[0], ctx)).unwrap_or(true),
                        ));
                    }
                    _ => (),
                }
            }
            match &call.target {
                None => {
                    let args = resolve_args(&call.args, ctx)?;
                    if let Some(op) = ctx.env().find_overload(&call.func_name, &args) {
                        return op(args).map(public_store);
                    }
                    let args: Vec<Value> = args.into_iter().map(|v| v.unpack()).collect();
                    let func = ctx.get_function(call.func_name.as_str()).ok_or_else(|| {
                        ExecutionError::UndeclaredReference(call.func_name.clone().into())
                    })?;
                    let mut ctx = FunctionContext::new(&call.func_name, None, ctx, args);
                    (func)(&mut ctx).map(public_store)
                }
                Some(target) => {
                    let args = resolve_args(&call.args, ctx)?;
                    let qualified_func = match &target.expr {
                        // A comprehension variable shadows the package
                        // namespace, so a receiver the enclosing macro
                        // bound is a value and never a namespace:
                        // `xs.all(optional, optional.of(1))` calls `of` on
                        // the element. langdef.md, name resolution -- "in a
                        // comprehension like `[1].exists(x, x == 1)`, `x` is
                        // a local variable which shadows any identifier
                        // named `x` in ancestor scopes or the package
                        // namespace".
                        Expr::Ident(prefix) if !ctx.is_comprehension_variable(prefix) => {
                            // A namespaced call (`math.max(x)`) and a member
                            // call on a variable (`s.startsWith(x)`) parse
                            // identically, so every one of the latter asks
                            // this question too and almost always gets no.
                            // Asking it without joining the two names keeps
                            // the answer free.
                            if let Some(op) =
                                ctx.env()
                                    .find_qualified_overload(prefix, &call.func_name, &args)
                            {
                                return op(args).map(public_store);
                            }
                            ctx.get_qualified_function(prefix, &call.func_name)
                        }
                        _ => None,
                    };
                    let (target, func, args) = match qualified_func {
                        None => {
                            let target = public_store(resolve_inner(target, ctx)?);
                            let mut args = args;
                            args.insert(0, target);
                            if let Some(op) = ctx.env().find_member_overload(&call.func_name, &args)
                            {
                                return op(args).map(public_store);
                            }
                            let target = args.remove(0).unpack();
                            let args: Vec<Value> = args.into_iter().map(|v| v.unpack()).collect();
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
                    (func)(&mut ctx).map(public_store)
                }
            }
        }
        Expr::Ident(name) => ctx
            .load_ident(name)
            .ok_or_else(|| ExecutionError::UndeclaredReference(Arc::new(name.to_string()))),
        Expr::Select(select) => {
            let left = resolve_inner(select.operand.deref(), ctx)?;
            let field = select.field.as_str();

            if select.test {
                match &left {
                    Value::Map(map) => Ok(Value::Bool(map.contains_key(&KeyRef::String(field)))),
                    Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::Map => Ok(Value::Bool(
                        unsafe { interned_map_lookup_string(*w, field) }.is_some(),
                    )),
                    #[cfg(feature = "structs")]
                    Value::Struct(_) => Ok(Value::Bool(value_field(&left, field).is_ok())),
                    _ => value_field(&left, field),
                }
            } else {
                value_field(&left, field)
            }
        }
        Expr::List(list_expr) => eval_list_literal(list_expr, ctx),
        Expr::Map(map_expr) => {
            let n = map_expr.entries.len();
            let mut src = map_expr.entries.iter();
            let map = try_build_map::<_, true>(n, |_| {
                let entry = src.next().expect("n source entries");
                let (k, v, is_optional) = match &entry.expr {
                    EntryExpr::StructField(_) => panic!("WAT?"),
                    EntryExpr::MapEntry(e) => (&e.key, &e.value, e.optional),
                };
                let key = value_key(resolve_inner(k, ctx)?)?;
                let value = resolve_inner(v, ctx)?;
                let stored = if is_optional {
                    match optional_view(&value) {
                        OptView::Empty => None,
                        OptView::Present(inner) => Some(public_store(inner)),
                        OptView::Plain => Some(public_store(value)),
                    }
                } else {
                    Some(public_store(value))
                };
                Ok::<_, ExecutionError>(stored.map(|value| (key, value)))
            })?;
            Ok(Value::Map(map))
        }
        Expr::Comprehension(comprehension) => {
            let accu_init = resolve_inner(&comprehension.accu_init, ctx)?;
            let iter = resolve_inner(&comprehension.iter_range, ctx)?;
            let mut ctx = ctx.new_inner_scope();
            let mut items = iter_items(&iter)?;

            // Map/filter before `all`/`exists`: AccuAppend::of declines
            // `all` on the loop condition, so the map loop stays in this
            // function the way it did before BoolAccu existed.
            if let Some(append) = AccuAppend::of(comprehension) {
                if let Value::List(_) = accu_init {
                    // The accumulator stays here rather than in the
                    // context: `@result` is not a name CEL can parse, so
                    // the guard and the element cannot read it, and a
                    // nested comprehension binds its own in its own scope.
                    // Sized by the range up front, as the VM's
                    // `NewListFromArg` does: `filter` may leave some of
                    // it unused, `map` fills it exactly.
                    let n = items.len();
                    let list = ListRef::try_fill_values(n, |_| {
                        let item = items.next().expect("n source items");
                        ctx.rebind(&comprehension.iter_var, item);
                        if let Some(guard) = append.guard {
                            if !try_bool_value(resolve_inner(guard, &ctx))? {
                                return Ok(None);
                            }
                        }
                        Ok(Some(public_store(resolve_inner(append.element, &ctx)?)))
                    })?;
                    ctx.rebind(&comprehension.accu_var, Value::List(list));
                    return resolve_inner(&comprehension.result, &ctx);
                }
                unreachable!("AccuAppend::of implies a list accumulator");
            }

            if let Some(bool_accu) = BoolAccu::of(comprehension) {
                return eval_bool_accu(
                    bool_accu,
                    &accu_init,
                    &mut items,
                    &mut ctx,
                    &comprehension.iter_var,
                );
            }

            ctx.rebind(&comprehension.accu_var, public_store(accu_init));
            while let Some(item) = items.next() {
                if !try_bool_value(resolve_inner(&comprehension.loop_cond, &ctx))? {
                    break;
                }
                ctx.rebind(&comprehension.iter_var, item);
                let accu = public_store(resolve_inner(&comprehension.loop_step, &ctx)?);
                ctx.rebind(&comprehension.accu_var, accu);
            }
            resolve_inner(&comprehension.result, &ctx)
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
                            let v = public_store(resolve_inner(&expr.value, ctx)?);
                            fields.insert(f, v);
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

/// Wraps `value` in an `optional`.
pub(crate) fn optional_of(value: Value) -> Value {
    Value::Opaque(Arc::new(OptionalValue::of(public_store(value))))
}

/// The empty `optional`.
pub(crate) fn optional_none() -> Value {
    Value::Opaque(Arc::new(OptionalValue::none()))
}

/// Views `value` as an optional, if it is one.
pub(crate) fn as_optional(value: &Value) -> Option<&OptionalValue> {
    match value {
        Value::Opaque(o) => o.downcast_ref::<OptionalValue>(),
        _ => None,
    }
}

enum OptView {
    Plain,
    Empty,
    Present(Value),
}

/// An optional in either public or interned form, without rebuilding the inner
/// value as a public container.
fn optional_view(value: &Value) -> OptView {
    if let Value::Interned(w) = value {
        if unsafe { w_kind(*w) } != CelKind::Optional {
            return OptView::Plain;
        }
        let inner = unsafe { (*w.cast::<W_OptionalObject>()).w_value };
        return if inner.is_null() {
            OptView::Empty
        } else {
            OptView::Present(Value::from_interned(inner))
        };
    }
    match as_optional(value) {
        None => OptView::Plain,
        Some(opt) => match opt.value() {
            None => OptView::Empty,
            Some(inner) => OptView::Present(inner.clone()),
        },
    }
}

fn try_bool_value(val: Result<Value, ExecutionError>) -> Result<bool, ExecutionError> {
    match val {
        Ok(v) => interned_as_bool(&v).ok_or(ExecutionError::NoSuchOverload),
        Err(err) => Err(err),
    }
}

/// A bool in either public or interned form.
fn interned_as_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::Bool => {
            Some(unsafe { (*w.cast::<W_BoolObject>()).boolval } != 0)
        }
        _ => None,
    }
}

/// Resolves a call's arguments into the values the overload table matches on.
fn resolve_args(args: &[Expression], ctx: &Context) -> Result<Vec<Value>, ExecutionError> {
    args.iter()
        .map(|arg| resolve_inner(arg, ctx).map(public_store))
        .collect()
}

fn value_is_list(value: &Value) -> bool {
    match value {
        Value::List(_) => true,
        Value::Interned(w) => unsafe { w_kind(*w) == CelKind::List },
        _ => false,
    }
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
pub(crate) fn mismatch_is_no_such_overload(op: &'static str, value: &Value) -> bool {
    if let Value::Interned(w) = value {
        return match unsafe { w_kind(*w) } {
            CelKind::Int => true,
            CelKind::List if op == "add" => true,
            #[cfg(feature = "chrono")]
            CelKind::Duration if op == "sub" => true,
            _ => false,
        };
    }
    #[cfg(feature = "chrono")]
    let duration_sub = matches!(value, Value::Duration(_)) && op == "sub";
    #[cfg(not(feature = "chrono"))]
    let duration_sub = false;

    matches!(value, Value::Int(_))
        || (matches!(value, Value::List(_)) && op == "add")
        || duration_sub
}

fn binary_op(op: &'static str, call: &CallExpr, ctx: &Context) -> Result<Value, ExecutionError> {
    let lhs = resolve_inner(&call.args[0], ctx)?;
    let rhs = resolve_inner(&call.args[1], ctx)?;
    binary_values(op, lhs, rhs)
}

/// The operand-level half of [`binary_op`], shared with the bytecode VM, whose
/// operands arrive from the stack rather than from an expression.
pub(crate) fn binary_values(
    op: &'static str,
    lhs: Value,
    rhs: Value,
) -> Result<Value, ExecutionError> {
    if let Some(result) = numeric_binop(op, &lhs, &rhs) {
        return binop_mismatch(op, &lhs, result);
    }
    let no_such = mismatch_is_no_such_overload(op, &lhs);
    binop_mismatch_flag(
        no_such,
        match op {
            "add" => lhs + rhs,
            "sub" => lhs - rhs,
            "div" => lhs / rhs,
            "mul" => lhs * rhs,
            "rem" => lhs % rhs,
            _ => unreachable!("unknown binary operator {op}"),
        },
    )
}

/// [`binary_values`] without taking ownership. The fused VM ops read a
/// local and a pool entry in place; cloning them only to hand the same
/// integers to [`ops::Add`] is the cost those ops were written to drop.
pub(crate) fn binary_values_ref(
    op: &'static str,
    lhs: &Value,
    rhs: &Value,
) -> Result<Value, ExecutionError> {
    if let Some(result) = numeric_binop(op, lhs, rhs) {
        return binop_mismatch(op, lhs, result);
    }
    binop_mismatch(
        op,
        lhs,
        match op {
            "add" => lhs.clone() + rhs.clone(),
            "sub" => lhs.clone() - rhs.clone(),
            "div" => lhs.clone() / rhs.clone(),
            "mul" => lhs.clone() * rhs.clone(),
            "rem" => lhs.clone() % rhs.clone(),
            _ => unreachable!("unknown binary operator {op}"),
        },
    )
}

fn binop_mismatch(
    op: &'static str,
    lhs: &Value,
    result: Result<Value, ExecutionError>,
) -> Result<Value, ExecutionError> {
    binop_mismatch_flag(mismatch_is_no_such_overload(op, lhs), result)
}

fn binop_mismatch_flag(
    no_such: bool,
    result: Result<Value, ExecutionError>,
) -> Result<Value, ExecutionError> {
    match result {
        Err(ExecutionError::UnsupportedBinaryOperator(..)) if no_such => {
            Err(ExecutionError::NoSuchOverload)
        }
        other => other,
    }
}

/// Same-type numeric (and temporal) operators, with no allocation.
///
/// Cross-type and string/list/bytes stay on the owned [`ops`] impls, which
/// are the walker's answers and the ones the oracle pins.
fn interned_or_public_int(value: &Value) -> Option<i64> {
    match value {
        Value::Int(i) => Some(*i),
        Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::Int => {
            Some(unsafe { (*w.cast::<W_IntObject>()).intval })
        }
        _ => None,
    }
}

fn interned_or_public_uint(value: &Value) -> Option<u64> {
    match value {
        Value::UInt(u) => Some(*u),
        Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::UInt => {
            Some(unsafe { (*w.cast::<W_UIntObject>()).uintval })
        }
        _ => None,
    }
}

fn interned_or_public_float(value: &Value) -> Option<f64> {
    match value {
        Value::Float(f) => Some(*f),
        Value::Interned(w) if unsafe { w_kind(*w) } == CelKind::Double => {
            Some(unsafe { (*w.cast::<W_DoubleObject>()).floatval })
        }
        _ => None,
    }
}

fn numeric_binop(
    op: &'static str,
    lhs: &Value,
    rhs: &Value,
) -> Option<Result<Value, ExecutionError>> {
    if let (Some(l), Some(r)) = (interned_or_public_int(lhs), interned_or_public_int(rhs)) {
        return Some(int_binop(op, l, r));
    }
    if let (Some(l), Some(r)) = (interned_or_public_uint(lhs), interned_or_public_uint(rhs)) {
        return Some(uint_binop(op, l, r));
    }
    if let (Some(l), Some(r)) = (interned_or_public_float(lhs), interned_or_public_float(rhs)) {
        return Some(float_binop(op, l, r));
    }
    match (lhs, rhs) {
        #[cfg(feature = "chrono")]
        (Value::Duration(l), Value::Duration(r)) => Some(duration_binop(op, *l, *r)),
        #[cfg(feature = "chrono")]
        (Value::Timestamp(l), Value::Duration(r)) => Some(timestamp_duration_binop(op, l, r)),
        #[cfg(feature = "chrono")]
        (Value::Duration(l), Value::Timestamp(r)) if op == "add" => {
            Some(checked_op(TsOp::Add, r, l))
        }
        #[cfg(feature = "chrono")]
        (Value::Timestamp(l), Value::Timestamp(r)) if op == "sub" => {
            Some(Ok(Value::Duration(l.signed_duration_since(*r))))
        }
        _ => None,
    }
}

fn int_binop(op: &'static str, l: i64, r: i64) -> Result<Value, ExecutionError> {
    match op {
        "add" => l
            .checked_add(r)
            .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
            .map(Value::Int),
        "sub" => l
            .checked_sub(r)
            .ok_or_else(|| ExecutionError::Overflow("sub", l.into(), r.into()))
            .map(Value::Int),
        "mul" => l
            .checked_mul(r)
            .ok_or_else(|| ExecutionError::Overflow("mul", l.into(), r.into()))
            .map(Value::Int),
        "div" => {
            if r == 0 {
                Err(ExecutionError::DivisionByZero(l.into()))
            } else {
                l.checked_div(r)
                    .ok_or_else(|| ExecutionError::Overflow("div", l.into(), r.into()))
                    .map(Value::Int)
            }
        }
        "rem" => {
            if r == 0 {
                Err(ExecutionError::RemainderByZero(l.into()))
            } else {
                l.checked_rem(r)
                    .ok_or_else(|| ExecutionError::Overflow("rem", l.into(), r.into()))
                    .map(Value::Int)
            }
        }
        _ => unreachable!("unknown binary operator {op}"),
    }
}

fn uint_binop(op: &'static str, l: u64, r: u64) -> Result<Value, ExecutionError> {
    match op {
        "add" => l
            .checked_add(r)
            .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
            .map(Value::UInt),
        "sub" => l
            .checked_sub(r)
            .ok_or_else(|| ExecutionError::Overflow("sub", l.into(), r.into()))
            .map(Value::UInt),
        "mul" => l
            .checked_mul(r)
            .ok_or_else(|| ExecutionError::Overflow("mul", l.into(), r.into()))
            .map(Value::UInt),
        "div" => l
            .checked_div(r)
            .ok_or_else(|| ExecutionError::DivisionByZero(l.into()))
            .map(Value::UInt),
        "rem" => l
            .checked_rem(r)
            .ok_or_else(|| ExecutionError::RemainderByZero(l.into()))
            .map(Value::UInt),
        _ => unreachable!("unknown binary operator {op}"),
    }
}

fn float_binop(op: &'static str, l: f64, r: f64) -> Result<Value, ExecutionError> {
    match op {
        "add" => Ok(Value::Float(l + r)),
        "sub" => Ok(Value::Float(l - r)),
        "mul" => Ok(Value::Float(l * r)),
        "div" => Ok(Value::Float(l / r)),
        _ => Err(ExecutionError::UnsupportedBinaryOperator(
            op,
            Value::Float(l),
            Value::Float(r),
        )),
    }
}

#[cfg(feature = "chrono")]
fn duration_binop(
    op: &'static str,
    l: chrono::Duration,
    r: chrono::Duration,
) -> Result<Value, ExecutionError> {
    match op {
        "add" => l
            .checked_add(&r)
            .ok_or_else(|| ExecutionError::Overflow("add", l.into(), r.into()))
            .map(Value::Duration),
        "sub" => l
            .checked_sub(&r)
            .ok_or_else(|| ExecutionError::Overflow("sub", l.into(), r.into()))
            .map(Value::Duration),
        _ => Err(ExecutionError::UnsupportedBinaryOperator(
            op,
            l.into(),
            r.into(),
        )),
    }
}

#[cfg(feature = "chrono")]
fn timestamp_duration_binop(
    op: &'static str,
    l: &chrono::DateTime<chrono::FixedOffset>,
    r: &chrono::Duration,
) -> Result<Value, ExecutionError> {
    match op {
        "add" => checked_op(TsOp::Add, l, r),
        "sub" => checked_op(TsOp::Sub, l, r),
        _ => Err(ExecutionError::UnsupportedBinaryOperator(
            op,
            (*l).into(),
            (*r).into(),
        )),
    }
}

/// Whether the receiver carries an ordering at all.
///
/// `PartialOrd for Value` orders `Null` against `Null` and would order a list
/// or a map by falling through, where `common/types` gives those no `Comparer`,
/// so ordering them is `NoSuchOverload`.
fn has_comparer(value: &Value) -> bool {
    if let Value::Interned(w) = value {
        return matches!(
            unsafe { w_kind(*w) },
            CelKind::Int
                | CelKind::UInt
                | CelKind::Double
                | CelKind::Str
                | CelKind::Bytes
                | CelKind::Bool
                | CelKind::Duration
                | CelKind::Timestamp
        );
    }
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
    let lhs = resolve_inner(&call.args[0], ctx)?;
    let rhs = resolve_inner(&call.args[1], ctx)?;
    compare_values(&lhs, &rhs, accept)
}

/// The operand-level half of [`compare_op`], shared with the bytecode VM.
pub(crate) fn compare_values(
    lhs: &Value,
    rhs: &Value,
    accept: impl FnOnce(Ordering) -> bool,
) -> Result<Value, ExecutionError> {
    if !has_comparer(lhs) {
        return Err(ExecutionError::NoSuchOverload);
    }
    let ordering = lhs.partial_cmp(rhs).ok_or(ExecutionError::NoSuchOverload)?;
    Ok(Value::Bool(accept(ordering)))
}

pub(crate) fn value_negate(value: Value) -> Result<Value, ExecutionError> {
    match value.unpack() {
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
pub(crate) fn value_key(value: Value) -> Result<Key, ExecutionError> {
    match value.unpack() {
        Value::Int(i) => Ok(Key::Int(i)),
        Value::UInt(u) => Ok(Key::Uint(u)),
        Value::Bool(b) => Ok(Key::Bool(b)),
        Value::String(s) => Ok(Key::String(s)),
        other => Err(ExecutionError::unsupported_key_type(other)),
    }
}

/// Index an interned list or map without rebuilding the container.
///
/// `None` means this pair is not a cheap in-place read; the caller falls
/// through to the public-enum path. A `Some` is the finished answer,
/// including the same errors that path would raise.
fn try_interned_value_index(
    container: &Value,
    key: &Value,
) -> Option<Result<Value, ExecutionError>> {
    let Value::Interned(w) = container else {
        return None;
    };
    match unsafe { w_kind(*w) } {
        CelKind::List => Some(interned_list_index(*w, key)),
        CelKind::Map => interned_map_index(*w, key),
        _ => None,
    }
}

fn interned_list_index(w: CelRef, key: &Value) -> Result<Value, ExecutionError> {
    let index = interned_or_public_list_index(key)?;
    interned_list_item(w, index)
        .ok_or_else(|| ExecutionError::IndexOutOfBounds(list_index_error_key(key)))
}

fn interned_or_public_list_index(key: &Value) -> Result<i64, ExecutionError> {
    match key {
        Value::Int(i) => Ok(*i),
        Value::UInt(u) => Ok(*u as i64),
        Value::Interned(k) => match unsafe { w_kind(*k) } {
            CelKind::Int => Ok(unsafe { (*k.cast::<W_IntObject>()).intval }),
            CelKind::UInt => Ok(unsafe { (*k.cast::<W_UIntObject>()).uintval as i64 }),
            _ => Err(unexpected_list_index_type(key)),
        },
        _ => Err(unexpected_list_index_type(key)),
    }
}

fn unexpected_list_index_type(key: &Value) -> ExecutionError {
    ExecutionError::UnexpectedType {
        got: key.type_of().to_string(),
        want: format!("{}|{}", ValueType::Int, ValueType::UInt),
    }
}

/// The key the public list path puts in [`ExecutionError::IndexOutOfBounds`].
fn list_index_error_key(key: &Value) -> Value {
    match key {
        Value::Interned(_) => key.unpack(),
        other => other.clone(),
    }
}

fn interned_or_public_map_key(key: &Value) -> Option<KeyRef<'_>> {
    match key {
        Value::Int(i) => Some(KeyRef::Int(*i)),
        Value::UInt(u) => Some(KeyRef::Uint(*u)),
        Value::Bool(b) => Some(KeyRef::Bool(*b)),
        Value::String(s) => Some(KeyRef::String(s.as_str())),
        Value::Interned(w) => unsafe { interned_as_keyref(*w) },
        _ => None,
    }
}

fn keyref_display(key: &KeyRef<'_>) -> String {
    match key {
        KeyRef::Int(i) => i.to_string(),
        KeyRef::Uint(u) => u.to_string(),
        KeyRef::Bool(b) => b.to_string(),
        KeyRef::String(s) => (*s).to_string(),
    }
}

fn interned_map_index(w: CelRef, key: &Value) -> Option<Result<Value, ExecutionError>> {
    let needle = interned_or_public_map_key(key)?;
    let found = unsafe { interned_map_get(w, needle) };
    Some(match found {
        Some(item) => Ok(Value::from_interned(item)),
        None => Err(ExecutionError::NoSuchKey(Arc::new(keyref_display(&needle)))),
    })
}

fn interned_value_type(w: CelRef) -> ValueType {
    match unsafe { w_kind(w) } {
        CelKind::List => ValueType::List,
        CelKind::Map => ValueType::Map,
        CelKind::Int => ValueType::Int,
        CelKind::UInt => ValueType::UInt,
        CelKind::Double => ValueType::Float,
        CelKind::Str => ValueType::String,
        CelKind::Bytes => ValueType::Bytes,
        CelKind::Bool => ValueType::Bool,
        CelKind::Null => ValueType::Null,
        CelKind::Duration => ValueType::Duration,
        CelKind::Timestamp => ValueType::Timestamp,
        CelKind::Optional | CelKind::Type | CelKind::Opaque => ValueType::Opaque,
        #[cfg(feature = "structs")]
        CelKind::Struct => ValueType::Struct,
        CelKind::Frame => ValueType::Null,
    }
}

fn interned_is_zero(w: CelRef) -> bool {
    match unsafe { w_kind(w) } {
        CelKind::List => unsafe { list_len(w) == 0 },
        CelKind::Map => unsafe { map_len(w) == 0 },
        CelKind::Int => unsafe { (*w.cast::<W_IntObject>()).intval == 0 },
        CelKind::UInt => unsafe { (*w.cast::<W_UIntObject>()).uintval == 0 },
        CelKind::Double => unsafe { (*w.cast::<W_DoubleObject>()).floatval == 0.0 },
        CelKind::Str => unsafe { string_byte_len(w) == 0 },
        CelKind::Bytes => unsafe { bytes_len(w) == 0 },
        CelKind::Bool => unsafe { (*w.cast::<W_BoolObject>()).boolval == 0 },
        CelKind::Null => true,
        CelKind::Duration => Value::from_interned(w).unpack().is_zero(),
        _ => false,
    }
}

/// Length of a sizer: list, map, string, or bytes, interned or public.
pub(crate) fn value_len(value: &Value) -> Option<i64> {
    match value {
        Value::List(l) => Some(l.len() as i64),
        Value::Map(m) => Some(m.len() as i64),
        Value::String(s) => Some(s.len() as i64),
        Value::Bytes(b) => Some(b.len() as i64),
        Value::Interned(w) => match unsafe { w_kind(*w) } {
            CelKind::List => Some(unsafe { list_len(*w) }),
            CelKind::Map => Some(unsafe { map_len(*w) }),
            CelKind::Str => Some(unsafe { string_byte_len(*w) }),
            CelKind::Bytes => Some(unsafe { bytes_len(*w) }),
            _ => None,
        },
        _ => None,
    }
}

/// Finish an interned arithmetic op. An i64 overflow on a temporal leaf is
/// not the language overflow: the public implementation on the unpacked
/// operands decides, and a result with no leaf stays public.
pub(crate) fn interned_binop_result(w: CelRef) -> ResolveResult {
    if w != ERROR_SENTINEL {
        return Ok(Value::from_interned(w));
    }
    interned_arith_error()
}

fn interned_arith_error() -> ResolveResult {
    let Some(err) = take_error() else {
        return Err(ExecutionError::InternalError(
            "raised without an error".to_owned(),
        ));
    };
    let lhs = unsafe { crate::runtime::convert::ref_to_value(err.lhs) }.unwrap_or(Value::Null);
    let rhs = if err.rhs == ERROR_SENTINEL {
        Value::Int(0)
    } else {
        unsafe { crate::runtime::convert::ref_to_value(err.rhs) }.unwrap_or(Value::Null)
    };
    if err.code == CelErrCode::Overflow {
        #[cfg(feature = "chrono")]
        if let Some(result) = temporal_public_arith(err.op, &lhs, &rhs) {
            return result;
        }
    }
    Err(raised_to_execution(err.code, err.op, lhs, rhs))
}

#[cfg(feature = "chrono")]
fn temporal_public_arith(op: &'static str, lhs: &Value, rhs: &Value) -> Option<ResolveResult> {
    if op == "negate" && matches!(lhs, Value::Duration(_)) {
        return Some(value_negate(lhs.clone()));
    }
    if matches!(
        (lhs, rhs),
        (Value::Duration(_), Value::Duration(_))
            | (Value::Timestamp(_), Value::Duration(_))
            | (Value::Duration(_), Value::Timestamp(_))
            | (Value::Timestamp(_), Value::Timestamp(_))
    ) {
        return Some(binary_values_ref(op, lhs, rhs));
    }
    None
}

fn raised_to_execution(
    code: CelErrCode,
    op: &'static str,
    lhs: Value,
    rhs: Value,
) -> ExecutionError {
    match code {
        CelErrCode::Overflow => ExecutionError::Overflow(op, lhs, rhs),
        CelErrCode::DivisionByZero => ExecutionError::DivisionByZero(lhs),
        CelErrCode::RemainderByZero => ExecutionError::RemainderByZero(lhs),
        CelErrCode::UnsupportedBinaryOperator => {
            if mismatch_is_no_such_overload(op, &lhs) {
                ExecutionError::NoSuchOverload
            } else {
                ExecutionError::UnsupportedBinaryOperator(op, lhs, rhs)
            }
        }
        CelErrCode::NoSuchOverload => ExecutionError::NoSuchOverload,
        CelErrCode::NoneDereference => {
            ExecutionError::function_error(op, "optional.none() dereference")
        }
    }
}

/// `container.field`, looked up without materializing the field name.
///
/// `KeyRef::String` borrows, so a map field select costs no allocation. Going
/// through [`value_index`] would build an `Arc<String>` per access.
pub(crate) fn value_field(container: &Value, field: &str) -> Result<Value, ExecutionError> {
    if let Value::Interned(w) = container {
        return match unsafe { w_kind(*w) } {
            CelKind::Map => match unsafe { interned_map_lookup_string(*w, field) } {
                Some(item) => Ok(Value::from_interned(item)),
                None => Err(ExecutionError::NoSuchKey(Arc::new(field.to_string()))),
            },
            CelKind::List => Err(ExecutionError::UnexpectedType {
                got: ValueType::String.to_string(),
                want: format!("{}|{}", ValueType::Int, ValueType::UInt),
            }),
            #[cfg(feature = "structs")]
            CelKind::Struct => value_index(container, &Value::String(Arc::new(field.to_string()))),
            _ => Err(ExecutionError::NoSuchOverload),
        };
    }
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
///
/// An interned list or map is indexed in place. Rebuilding a public
/// `List`/`Map` is the fallback for every other shape, and for interned
/// cases that are not cheap to read from the leaf.
pub(crate) fn value_index(container: &Value, key: &Value) -> Result<Value, ExecutionError> {
    if let Some(result) = try_interned_value_index(container, key) {
        return result;
    }
    let container = container.unpack();
    let key = key.unpack();
    match &container {
        Value::List(list) => {
            let idx = match &key {
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
        Value::Struct(s) => match &key {
            Value::String(field) => s
                .field_value(field)
                .cloned()
                .ok_or_else(|| ExecutionError::NoSuchKey(Arc::new(field.as_str().to_owned()))),
            other => Err(ExecutionError::UnsupportedIndex(
                other.clone(),
                container.clone(),
            )),
        },
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// The keys of `map`, as values a comprehension can bind.
fn map_keys(map: &Map) -> Vec<Value> {
    match map.storage() {
        MapStorage::Object(entries) => entries.keys().map(key_value).collect(),
        MapStorage::Entries(_) => map
            .ordered_pairs()
            .expect("ordered")
            .iter()
            .map(|(k, _)| key_value(k))
            .collect(),
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
pub(crate) fn value_contains(container: &Value, needle: &Value) -> Result<bool, ExecutionError> {
    if let Value::Interned(w) = container {
        let n = match needle {
            Value::Interned(n) => *n,
            other => intern_leaf(other).ok_or(ExecutionError::NoSuchOverload)?,
        };
        return interned_contains(*w, n);
    }
    let needle = needle.unpack();
    match container {
        Value::List(list) => Ok(list.contains(&needle)),
        Value::Map(map) => Ok(map.contains_key(&value_key(needle)?)),
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// `needle in container` on interned operands.
///
/// A string container is `NoSuchOverload`: CEL has no `x in "string"`. A
/// needle that cannot be a map key is `UnsupportedKeyType`, not a miss. An
/// ints-list answers only for an int needle; any other kind is a miss.
#[inline]
pub(crate) fn interned_contains(container: CelRef, needle: CelRef) -> Result<bool, ExecutionError> {
    match unsafe { w_kind(container) } {
        CelKind::List => {
            if let Some(ints) = interned_ints_slice(container) {
                if unsafe { w_kind(needle) } == CelKind::Int {
                    let n = unsafe { (*needle.cast::<W_IntObject>()).intval };
                    return Ok(ints.contains(&n));
                }
            }
            Ok(interned_list_contains_in_place(container, needle))
        }
        CelKind::Map => {
            if unsafe { interned_as_keyref(needle) }.is_none() {
                value_key(Value::from_interned(needle).unpack())?;
            }
            Ok(unsafe { map_contains_key(container, needle) })
        }
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

/// `a?.b` on an interned receiver.
///
/// A miss on a plain map/struct is `NoSuchKey`. A miss on an optional
/// receiver folds to `optional.of(optional.none)`. Anything else is
/// `NoSuchOverload` so the caller can decline.
pub(crate) fn interned_opt_select(w: CelRef, field: &str) -> Result<CelRef, ExecutionError> {
    let (inner, optional) = if unsafe { w_kind(w) } == CelKind::Optional {
        let inner = unsafe { (*w.cast::<W_OptionalObject>()).w_value };
        if inner.is_null() {
            return Ok(new_optional_none() as CelRef);
        }
        (inner, true)
    } else {
        (w, false)
    };
    let found = match unsafe { w_kind(inner) } {
        CelKind::Map => unsafe { interned_map_lookup_string(inner, field) },
        #[cfg(feature = "structs")]
        CelKind::Struct => unsafe { crate::runtime::object::struct_lookup_field(inner, field) },
        _ => return Err(ExecutionError::NoSuchOverload),
    };
    match found {
        Some(item) => Ok(new_optional(item) as CelRef),
        None if optional => Ok(new_optional(new_optional_none() as CelRef) as CelRef),
        None => Err(ExecutionError::NoSuchKey(Arc::new(field.to_string()))),
    }
}

/// Scan without copying the list into a buffer. A hit at index *k* costs *k*
/// equality tests, not a traversal of the tail.
fn interned_list_contains_in_place(w: CelRef, needle: CelRef) -> bool {
    let n = unsafe { list_len(w) };
    let mut i = 0i64;
    while i < n {
        if let Some(item) = unsafe { interned_list_get(w, i) } {
            if unsafe { values_equal(item, needle) } {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn interned_eq(a: CelRef, b: CelRef) -> bool {
    if let Some(eq) = unsafe { crate::runtime::object::interned_list_eq(a, b) } {
        return eq;
    }
    unsafe { values_equal(a, b) }
}

fn interned_eq_public(w: CelRef, other: &Value) -> bool {
    if let Some(ints) = interned_ints_slice(w) {
        if let Value::List(list) = other {
            return public_list_eq_ints(list, ints);
        }
    }
    match intern_leaf(other) {
        Some(b) => interned_eq(w, b),
        None => match unsafe { crate::runtime::convert::ref_to_value(w) } {
            Ok(unpacked) => unpacked == *other,
            Err(_) => false,
        },
    }
}

fn public_list_eq_ints(list: &ListRef, ints: &[i64]) -> bool {
    if list.len() != ints.len() {
        return false;
    }
    if let Some(v) = list.ints_slice() {
        let start = list.window_start();
        return v.get(start..start + ints.len()) == Some(ints);
    }
    (0..ints.len()).all(|i| list.get(i) == Some(Value::Int(ints[i])))
}

fn interned_ints_slice<'a>(w: CelRef) -> Option<&'a [i64]> {
    unsafe { crate::runtime::object::list_ints_slice(w) }
}

/// The elements a comprehension iterates over.
pub(crate) fn value_iter(value: &Value) -> Result<Vec<Value>, ExecutionError> {
    let mut items = iter_items(value)?;
    let mut out = Vec::with_capacity(items.len());
    while let Some(item) = items.next() {
        out.push(item);
    }
    Ok(out)
}

/// One pass over a list or map. An interned Ints list is read in place as
/// unboxed ints; an interned object list yields the stored refs from a
/// slice resolved once before the loop.
enum IterItems {
    Vec(std::vec::IntoIter<Value>),
    InternedInts { ints: &'static [i64], i: usize },
    InternedRefs { refs: &'static [CelRef], i: usize },
    InternedList { w: CelRef, i: i64, n: i64 },
}

impl IterItems {
    fn len(&self) -> usize {
        match self {
            IterItems::Vec(it) => it.len(),
            IterItems::InternedInts { ints, i } => ints.len().saturating_sub(*i),
            IterItems::InternedRefs { refs, i } => refs.len().saturating_sub(*i),
            IterItems::InternedList { i, n, .. } => (*n - *i).max(0) as usize,
        }
    }

    fn next(&mut self) -> Option<Value> {
        match self {
            IterItems::Vec(it) => it.next(),
            IterItems::InternedInts { ints, i } => {
                let v = ints.get(*i)?;
                *i += 1;
                Some(Value::Int(*v))
            }
            IterItems::InternedRefs { refs, i } => {
                let w = refs.get(*i)?;
                *i += 1;
                Some(Value::from_interned(*w))
            }
            IterItems::InternedList { w, i, n } => {
                if *i >= *n {
                    return None;
                }
                let item = interned_list_item(*w, *i)?;
                *i += 1;
                Some(item)
            }
        }
    }
}

fn iter_items(value: &Value) -> Result<IterItems, ExecutionError> {
    if let Value::Interned(w) = value {
        return match unsafe { w_kind(*w) } {
            CelKind::List => interned_list_items(*w),
            CelKind::Map => Ok(IterItems::Vec(
                unsafe { map_key_refs(*w) }
                    .into_iter()
                    .map(Value::from_interned)
                    .collect::<Vec<_>>()
                    .into_iter(),
            )),
            _ => Err(ExecutionError::NoSuchOverload),
        };
    }
    match value {
        Value::List(list) => Ok(IterItems::Vec(list.to_vec().into_iter())),
        Value::Map(map) => Ok(IterItems::Vec(map_keys(map).into_iter())),
        _ => Err(ExecutionError::NoSuchOverload),
    }
}

fn interned_list_items(w: CelRef) -> Result<IterItems, ExecutionError> {
    unsafe {
        let leaf = &*w.cast::<W_ListObject>();
        let start = leaf.start as usize;
        let n = leaf.length as usize;
        if leaf.strategy == ListStrategy::Ints && !leaf.storage.is_null() {
            let col = &*leaf.storage.cast::<W_IntColumn>();
            if !col.data.is_null() && start.saturating_add(n) <= col.length as usize {
                let ints = std::slice::from_raw_parts(col.data.add(start), n);
                return Ok(IterItems::InternedInts { ints, i: 0 });
            }
        }
        if leaf.strategy == ListStrategy::Object {
            if n == 0 {
                return Ok(IterItems::InternedRefs { refs: &[], i: 0 });
            }
            let base = items_block_items_base(leaf.items);
            if !base.is_null() && start.saturating_add(n) <= items_capacity(leaf.items) {
                let refs = std::slice::from_raw_parts(base.add(start), n);
                return Ok(IterItems::InternedRefs { refs, i: 0 });
            }
        }
        Ok(IterItems::InternedList {
            w,
            i: 0,
            n: leaf.length,
        })
    }
}

/// One element of an interned list. Ints stay unboxed; object/window items
/// stay the stored ref.
fn interned_list_item(w: CelRef, index: i64) -> Option<Value> {
    unsafe {
        let leaf = &*w.cast::<W_ListObject>();
        if index < 0 || index >= leaf.length {
            return None;
        }
        match leaf.strategy {
            ListStrategy::Ints => list_int_at(w, index).map(Value::Int),
            ListStrategy::Object | ListStrategy::Window => {
                interned_list_get(w, index).map(Value::from_interned)
            }
        }
    }
}

impl ops::Add<Value> for Value {
    type Output = ResolveResult;

    #[inline(always)]
    fn add(self, rhs: Value) -> Self::Output {
        if matches!(self, Value::Interned(_)) || matches!(rhs, Value::Interned(_)) {
            // List `+` is [`ListRef::concat`]. Interning a public list to
            // reach `w_list_add` would box every int and rebuild the result
            // at the door; unpack (a linked bound list is an Arc clone) and
            // use the one implementation.
            if value_is_list(&self) && value_is_list(&rhs) {
                return self.unpack().add(rhs.unpack());
            }
            if let (Some(a), Some(b)) = (intern_leaf(&self), intern_leaf(&rhs)) {
                let w = unsafe { cel_add(a, b) };
                return interned_binop_result(w);
            }
            return self.unpack().add(rhs.unpack());
        }
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
                if let Some(s) = Arc::get_mut(&mut l) {
                    s.push_str(&r);
                    Ok(Value::String(l))
                } else {
                    let mut out = String::with_capacity(l.len() + r.len());
                    out.push_str(&l);
                    out.push_str(&r);
                    Ok(Value::String(Arc::new(out)))
                }
            }
            (Value::Bytes(mut l), Value::Bytes(r)) => {
                if let Some(s) = Arc::get_mut(&mut l) {
                    s.extend_from_slice(&r);
                    Ok(Value::Bytes(l))
                } else {
                    let mut out = Vec::with_capacity(l.len() + r.len());
                    out.extend_from_slice(&l);
                    out.extend_from_slice(&r);
                    Ok(Value::Bytes(Arc::new(out)))
                }
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
        if matches!(self, Value::Interned(_)) || matches!(rhs, Value::Interned(_)) {
            return self.unpack().sub(rhs.unpack());
        }
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
        if matches!(self, Value::Interned(_)) || matches!(rhs, Value::Interned(_)) {
            return self.unpack().div(rhs.unpack());
        }
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
        if matches!(self, Value::Interned(_)) || matches!(rhs, Value::Interned(_)) {
            return self.unpack().mul(rhs.unpack());
        }
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
        if matches!(self, Value::Interned(_)) || matches!(rhs, Value::Interned(_)) {
            return self.unpack().rem(rhs.unpack());
        }
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
    use super::{ListRef, ListStorage};
    use crate::{objects::Key, Context, ExecutionError, Program, Value};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn concat_of_two_int_lists_stays_ints() {
        let a = ListRef::whole(Arc::new(ListStorage::Ints(vec![1, 2, 3])));
        let b = ListRef::whole(Arc::new(ListStorage::Ints(vec![4, 5])));
        let out = a.concat(&b);
        assert!(out.is_ints());
        assert_eq!(
            out.to_vec(),
            vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
                Value::Int(4),
                Value::Int(5)
            ]
        );
    }

    #[test]
    fn concat_of_int_list_and_object_int_stays_ints() {
        let a = ListRef::whole(Arc::new(ListStorage::Ints(vec![1, 2, 3])));
        let b = ListRef::from(vec![Value::Int(15)]);
        let out = a.concat(&b);
        assert!(out.is_ints());
        assert_eq!(
            out.to_vec(),
            vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(15)]
        );
    }

    #[test]
    fn empty_list_clone_and_drop() {
        let a = ListRef::from(Vec::<Value>::new());
        assert!(a.is_empty());
        let b = a.clone();
        assert!(a.ptr_eq(&b));
        drop(a);
        assert_eq!(b.to_vec(), Vec::<Value>::new());
        drop(b);
        let ints = ListRef::whole(Arc::new(ListStorage::Ints(Vec::new())));
        assert!(ints.is_empty());
        assert!(ints.is_ints());
    }

    #[test]
    fn empty_concat_matches_empty_list() {
        let empty = ListRef::from(Vec::<Value>::new());
        let sum = empty.clone().concat(&empty);
        assert!(empty.ptr_eq(&sum));
    }

    #[test]
    fn zero_sized_window_clone_and_drop() {
        let storage = Arc::new(ListStorage::Ints(vec![1, 2, 3]));
        let z = ListRef::window(Arc::clone(&storage), 1, 0);
        assert_eq!(z.len(), 0);
        assert_eq!(z.to_vec(), Vec::<Value>::new());
        let z2 = z.clone();
        assert!(z.shares_storage_with(&z2));
        drop(z);
        assert_eq!(z2.to_vec(), Vec::<Value>::new());
        let whole = ListRef::whole(storage);
        assert_eq!(whole.len(), 3);
        assert!(z2.shares_storage_with(&whole));
    }

    #[test]
    fn one_thousand_windows_over_one_column_allocate_zero_times() {
        let storage = Arc::new(ListStorage::Column(super::ValueColumn::Scalar {
            bank: super::ScalarBank::Int,
            words: Arc::from([1i64, 2, 3].as_slice()),
        }));
        let mut windows = Vec::with_capacity(1000);
        for _ in 0..1000 {
            windows.push(ListRef::window(Arc::clone(&storage), 0, 3));
        }
        assert_eq!(Arc::strong_count(&storage), 1001);
        for w in windows.windows(2) {
            assert!(w[0].shares_storage_with(&w[1]));
        }
        drop(windows);
        assert_eq!(Arc::strong_count(&storage), 1);
    }

    #[test]
    fn windowing_a_list_is_a_refcount_bump() {
        let storage = Arc::new(ListStorage::Ints(vec![1, 2, 3, 4]));
        let a = ListRef::window(Arc::clone(&storage), 0, 2);
        let b = ListRef::window(Arc::clone(&storage), 2, 2);
        assert!(a.shares_storage_with(&b));
        assert!(matches!(a.storage(), Some(ListStorage::Ints(_))));
        assert_eq!(a.to_vec(), vec![Value::Int(1), Value::Int(2)]);
        assert_eq!(b.to_vec(), vec![Value::Int(3), Value::Int(4)]);
        drop(storage);
        assert_eq!(a.get(1), Some(Value::Int(2)));
        assert_eq!(b.get(0), Some(Value::Int(3)));
    }

    #[test]
    fn rc_slice_clone_drop_empty_and_slice() {
        let empty = super::RcSlice::<i64>::empty();
        assert!(empty.as_slice().is_empty());
        let cloned = empty.clone();
        drop(empty);
        assert!(cloned.as_slice().is_empty());
        drop(cloned);
        let s = super::RcSlice::from_vec(vec![1i64, 2, 3]);
        assert_eq!(s.as_slice(), &[1, 2, 3]);
        let c = s.clone();
        drop(s);
        assert_eq!(c.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn list_fill_drops_prefix_on_error() {
        let s = Arc::new("keep".to_string());
        let err = ListRef::try_fill_values::<&'static str>(2, |i| {
            if i == 0 {
                Ok(Some(Value::String(s.clone())))
            } else {
                Err("fail")
            }
        });
        assert_eq!(err, Err("fail"));
        assert_eq!(Arc::strong_count(&s), 1);
    }

    #[derive(Debug)]
    struct DropProbe(Arc<std::sync::atomic::AtomicUsize>);
    impl PartialEq for DropProbe {
        fn eq(&self, _: &Self) -> bool {
            true
        }
    }
    impl Eq for DropProbe {}
    impl super::Opaque for DropProbe {
        fn runtime_type_name(&self) -> &str {
            "test.DropProbe"
        }
    }
    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn probe() -> (Value, Arc<std::sync::atomic::AtomicUsize>) {
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            Value::Opaque(Arc::new(DropProbe(Arc::clone(&hits)))),
            hits,
        )
    }

    #[test]
    fn object_list_clone_drop_reaches_zero_once() {
        let (v, hits) = probe();
        let list = ListRef::from(vec![v]);
        let clone = list.clone();
        drop(list);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 0);
        drop(clone);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn ints_list_clone_drop_and_window() {
        let list = ListRef::try_fill_ints::<()>(3, |i| Ok(Some(i as i64 + 1))).unwrap();
        let a = list.clone();
        drop(list);
        assert_eq!(a.to_vec(), vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        let storage = Arc::new(ListStorage::Ints(vec![10, 20, 30]));
        let w = ListRef::window(Arc::clone(&storage), 1, 1);
        assert_eq!(w.to_vec(), vec![Value::Int(20)]);
        let w2 = w.clone();
        drop(w);
        drop(storage);
        assert_eq!(w2.get(0), Some(Value::Int(20)));
        drop(a);
        drop(w2);
    }

    #[test]
    fn shared_list_clone_drop_reaches_zero_once() {
        let (v, hits) = probe();
        let storage = Arc::new(ListStorage::Object(vec![v]));
        let a = ListRef::window(Arc::clone(&storage), 0, 1);
        let b = ListRef::window(storage, 0, 1);
        assert!(a.shares_storage_with(&b));
        drop(a);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 0);
        drop(b);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn empty_and_windows_over_each_kind() {
        let empty = ListRef::from(Vec::<Value>::new());
        assert!(empty.is_empty());
        drop(empty.clone());
        drop(empty);

        let (v, hits) = probe();
        let object = ListRef::from(vec![v]);
        let object_w = ListRef::from_linked(object.clone().buf, 0, 0);
        assert_eq!(object_w.len(), 0);
        drop(object);
        drop(object_w);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);

        let ints = ListRef::try_fill_ints::<()>(2, |i| Ok(Some(i as i64))).unwrap();
        let ints_w = ListRef::from_linked(ints.clone().buf, 1, 0);
        assert_eq!(ints_w.len(), 0);
        drop(ints);
        drop(ints_w);

        let col = Arc::new(ListStorage::Column(super::ValueColumn::Scalar {
            bank: super::ScalarBank::Int,
            words: Arc::from([1i64, 2, 3].as_slice()),
        }));
        let col_w = ListRef::window(Arc::clone(&col), 1, 0);
        assert_eq!(col_w.len(), 0);
        let col_full = ListRef::window(col, 0, 3);
        assert!(col_w.shares_storage_with(&col_full));
        drop(col_w);
        assert_eq!(col_full.len(), 3);
        drop(col_full);
    }

    #[test]
    fn fill_ints_drops_on_error() {
        let err = ListRef::try_fill_ints::<&'static str>(2, |i| {
            if i == 0 {
                Ok(Some(1))
            } else {
                Err("fail")
            }
        });
        assert_eq!(err, Err("fail"));
    }

    #[test]
    fn list_clone_drop_on_this_thread() {
        // ListRef is !Send + !Sync, so clone/drop of an owned header stays
        // on this thread. The header count stays atomic: a pooled list
        // leaf's public link points at the `ListBuf` in `CelCode::consts`,
        // and two threads executing one `Arc<Program>` clone/drop it.
        // Arc<ListStorage> still uses its own atomic because the batch
        // tier shares one column among windows.
        let (v, hits) = probe();
        let object = ListRef::from(vec![v]);
        let ints = ListRef::try_fill_ints::<()>(4, |i| Ok(Some(i as i64))).unwrap();
        let (v2, hits2) = probe();
        let shared = ListRef::whole(Arc::new(ListStorage::Object(vec![v2])));
        for _ in 0..4 {
            drop(object.clone());
            drop(ints.clone());
            drop(shared.clone());
        }
        drop(object);
        drop(ints);
        drop(shared);
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(hits2.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    /// `math.max(x)` and `s.startsWith(x)` parse the same, so the walker asks
    /// whether a namespaced function exists before treating the target as a
    /// receiver. Nothing covered that arm; these pin what it decides.
    #[test]
    fn namespaced_call_beats_a_same_named_variable() {
        let mut ctx = Context::default();
        ctx.add_function("math.max", |a: i64, b: i64| if a > b { a } else { b });
        // A variable that shares the namespace's name, and one whose own name is
        // a prefix of an unrelated stdlib function (`size`, `startsWith`).
        ctx.add_variable_from_value("math", Value::from("not a namespace"));
        ctx.add_variable_from_value("s", Value::from("hello world"));

        // The namespaced function wins over the identically named variable.
        let prog = Program::compile("math.max(2, 3)").unwrap();
        assert_eq!(Ok(Value::Int(3)), prog.execute(&ctx));

        // A member call on a variable is unaffected, including when its name is
        // a prefix of a registered non-namespaced function.
        let prog = Program::compile(r#"s.startsWith("hello")"#).unwrap();
        assert_eq!(Ok(Value::Bool(true)), prog.execute(&ctx));

        // An unregistered namespaced name falls through to the receiver path
        // and reports the method, not the joined name.
        let prog = Program::compile("math.min(2, 3)").unwrap();
        assert!(matches!(
            prog.execute(&ctx),
            Err(ExecutionError::UndeclaredReference(name)) if name.as_str() == "min"
        ));
    }

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
        use crate::objects::{Map, Opaque, OptionalValue};
        use crate::parser::Parser;
        use crate::{Context, ExecutionError, FunctionContext, Program, Value};
        use serde::Serialize;
        use std::collections::HashMap;
        use std::fmt::Debug;
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
                if let Some(Value::Opaque(opaque)) = ftx.this.as_ref() {
                    if opaque.runtime_type_name() == "my_struct" {
                        Ok(opaque
                            .downcast_ref::<MyStruct>()
                            .unwrap()
                            .field
                            .clone()
                            .into())
                    } else {
                        Err(ExecutionError::UnexpectedType {
                            got: opaque.runtime_type_name().to_string(),
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
        use std::sync::Arc;

        use crate::{
            common::types::{self, CelStruct},
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
                    assert_eq!(s.field_value("solved"), Some(&Value::Bool(true)));
                    assert_eq!(s.field_value("answer"), Some(&Value::Int(42)));
                    assert_eq!(s.field_values().len(), 2);
                    assert_eq!(
                        s.field_values().get("solved").cloned(),
                        Some(Value::Bool(true))
                    );
                    assert_eq!(
                        s.field_values().get("answer").cloned(),
                        Some(Value::Int(42))
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
                    .add_field_with_default(
                        "here".into(),
                        Value::String(Arc::new("yes".to_owned())),
                    ),
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
                    .add_field_with_default(
                        "here".into(),
                        Value::String(Arc::new("yes".to_owned())),
                    ),
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
                Value::String(Arc::new("test".to_owned())),
            );
            my_struct.add_field_value("value".to_owned(), Value::Int(42));

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
                Value::String(Arc::new("test".to_owned())),
            );
            my_struct.add_field_value("value".to_owned(), Value::Int(42));

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
