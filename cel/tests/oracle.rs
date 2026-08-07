//! **The differential oracle** — the P1 keystone of the `cel-unboxed-values`
//! design (task #52), and the gate every later phase leans on.
//!
//! ## What it is, and the one property that makes it work
//!
//! `tests/oracle_corpus.txt` is a checked-in **data** file: CEL source on the
//! left, the expected result on the right, written in a canonical notation that
//! is deliberately **not** CEL and **not** Rust. This file renders whatever an
//! evaluator produced into that same notation and compares the two as strings.
//!
//! Freezing the expectation as data is the load-bearing choice. An oracle that
//! is itself an evaluator — a second walker written in the same style as the
//! subject — is worthless here, because the phase that rewrites the subject
//! rewrites the oracle in the same commit and the gate silently evaporates. A
//! text file cannot be refactored by accident.
//!
//! ## Why not `parity_sweep_binary_operators`
//!
//! The pre-existing sweep (`src/majit/mod.rs`) is not this and cannot become
//! it. It lives inside the `#[cfg(test)]` module of the columnar tier that P9
//! deletes; `sweep_case` lowers each expression first and returns
//! `SweepVerdict::Declined` when the batch lowering refuses it, so the
//! expressions an oracle exists to cover are exactly the ones it never
//! compares; its only assertion is that more than 200 cases agreed; and its
//! operand universe is a batch schema of scalar columns, with no map, list,
//! null, comprehension, error, duration or opaque anywhere in it.
//!
//! ## The differential
//!
//! `EVALUATORS` is the list of doors under test, and registering one is the
//! only change a new evaluator needs on this side — the corpus is already the
//! shared expectation, so every case becomes an agreement check between each
//! registered evaluator and the frozen data.
//!
//! P2 used it exactly that way: `resolve_value` was registered beside the
//! `dyn Val` walker, the two were held to the whole corpus while the new one
//! was built arm by arm, and the walker row was removed only once it was
//! deleted. The list holds one row again today; that is a statement about the
//! crate, not about this file.
//!
//! ## Coverage
//!
//! Every case declares one or more `feat:` tags. `REQUIRED_COVERAGE` names the
//! CEL feature axis and the minimum number of cases each point on it must
//! carry. A feature that silently drops out of the corpus fails the build
//! instead of quietly shrinking the gate — which is the failure mode a coverage
//! assertion exists to prevent. Unknown tags are rejected too, so a typo cannot
//! satisfy a requirement by accident.
//!
//! ## What this oracle does NOT cover
//!
//! Stated plainly, because an overclaimed gate is worse than a small one:
//!
//! * **The JIT / columnar tier.** Nothing here builds with `--features jit-*`
//!   in mind; `EVALUATORS` holds interpreter doors only.
//! * **Error payloads other than the ones `render_error` spells out.** Most
//!   variants render to their name alone, so a case pinning `NoSuchOverload`
//!   does not pin *which* overload was missing.
//! * **Non-determinism.** Map iteration order is normalised away by sorting;
//!   an expression whose result genuinely varies run to run cannot live here.
//! * **`f64` bit patterns.** Doubles render through `{:?}`, so `-0.0` and `0.0`
//!   are distinguishable but the last-ulp behaviour of a long chain is not
//!   pinned beyond what that rendering shows.
//! * **Parse-error detail.** A syntax case pins only that compilation failed.
//! * **A successfully built `Value::Struct`**, and any opaque other than
//!   `OptionalValue`. `render` handles both so the file builds with those
//!   features on; the corpus reaches a struct literal only to pin *when* the
//!   type is refused, because `fixed_context` registers no struct definition.
//! * **Anything behind a cargo feature that is off.** Cases carry `cfg:` and
//!   are skipped when the feature is absent; the corresponding coverage
//!   requirements are dropped with them, and the run says so. A leading `!`
//!   negates, so a divergence that only appears with a feature OFF can be
//!   pinned as the same expression with two answers.

use std::collections::BTreeMap;
use std::sync::Arc;

use cel::common::types::TypeValue;
use cel::objects::{Key, OptionalValue};
use cel::parser::{Expression, Parser};
use cel::{Context, ExecutionError, Program, Value};

// ---------------------------------------------------------------------------
// the doors under test
// ---------------------------------------------------------------------------

/// One evaluation door. The differential is over this list: every corpus case
/// runs through every entry and every entry must produce the frozen answer.
///
/// The signature takes an already-parsed `Expression` so that parsing config
/// (which the corpus controls per case, see `parse:`) is not tangled with
/// evaluation semantics (which is what the corpus is about). `Program::execute`
/// is `Value::resolve` on the program's expression — `public_door_agrees`
/// below pins that, so registering the walker here does not lose the
/// caller-facing door.
struct Evaluator {
    name: &'static str,
    eval: fn(&Expression, &Context) -> Result<Value, ExecutionError>,
}

/// The crate has one evaluator again, now that the `dyn Val` walker is gone.
///
/// The corpus outlives it: it was written against two evaluators and stays the
/// frozen expectation for whatever replaces this one. A second row goes back
/// here the moment there is a second door — the bytecode VM of P4 onwards — and
/// nothing else about this file has to change for that.
const EVALUATORS: &[Evaluator] = &[
    Evaluator {
        name: "value-walker",
        eval: Value::resolve_value,
    },
    Evaluator {
        name: "bytecode-vm",
        eval: cel::vm::eval,
    },
];

// ---------------------------------------------------------------------------
// the canonical rendering — the ONLY code that reads a `Value`'s shape
// ---------------------------------------------------------------------------

/// Projects a `Value` into the corpus notation.
///
/// Deliberately total and deliberately boring: it is the single point where
/// this test knows anything about the value universe, so a phase that changes
/// that universe changes exactly one function here and the corpus stays put.
fn render(value: &Value) -> String {
    match value {
        Value::Int(i) => format!("int({i})"),
        Value::UInt(u) => format!("uint({u})"),
        Value::Float(f) => format!("double({})", render_double(*f)),
        Value::Bool(b) => format!("bool({b})"),
        Value::String(s) => format!("string({})", quote(s)),
        Value::Bytes(b) => format!("bytes({})", hex(b)),
        Value::Null => "null".to_string(),
        Value::List(list) => {
            let items: Vec<String> = list.iter().map(|v| render(&v)).collect();
            format!("list[{}]", items.join(", "))
        }
        Value::Map(map) => {
            // Sorted by rendered key: `MapStorage::Object` is a `HashMap`, so
            // iteration order is not a property of the value and must not leak
            // into the expectation.
            let mut entries: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", render_key(k), render(&v)))
                .collect();
            entries.sort();
            format!("map{{{}}}", entries.join(", "))
        }
        Value::Opaque(o) => match o.downcast_ref::<OptionalValue>() {
            Some(opt) => match opt.value() {
                Some(v) => format!("optional({})", render(v)),
                None => "optional.none".to_string(),
            },
            // The DENOTED type's name. A type value's own `runtime_type_name`
            // is `type` for every one of them, so the generic arm below would
            // render `type(1)` and `type('a')` identically and the corpus could
            // not tell the two answers apart.
            None => match o.downcast_ref::<TypeValue>() {
                Some(t) => format!("type({})", t.name()),
                None => format!("opaque({})", o.runtime_type_name()),
            },
        },
        #[cfg(feature = "chrono")]
        Value::Duration(d) => match d.num_nanoseconds() {
            Some(ns) => format!("duration({ns}ns)"),
            None => format!("duration({}s+)", d.num_seconds()),
        },
        #[cfg(feature = "chrono")]
        Value::Timestamp(t) => format!("timestamp({})", t.to_rfc3339()),
        // Rendered so the arm exists under `--features structs`; the corpus
        // carries no struct case (see this file's header).
        #[cfg(feature = "structs")]
        Value::Struct(s) => {
            let fields: Vec<String> = s
                .field_values()
                .into_iter()
                .map(|(name, v)| format!("{name}: {}", render(&v)))
                .collect();
            format!("struct({}{{{}}})", s.name(), fields.join(", "))
        }
    }
}

fn render_key(key: &Key) -> String {
    match key {
        Key::Int(i) => format!("int({i})"),
        Key::Uint(u) => format!("uint({u})"),
        Key::Bool(b) => format!("bool({b})"),
        Key::String(s) => format!("string({})", quote(s)),
    }
}

/// `{:?}` on `f64` round-trips and distinguishes `0.0` from `-0.0`; the
/// non-finite spellings are pinned by hand so they cannot drift with a
/// formatting change.
fn render_double(f: f64) -> String {
    if f.is_nan() {
        "nan".to_string()
    } else if f == f64::INFINITY {
        "inf".to_string()
    } else if f == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{f:?}")
    }
}

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Projects an `ExecutionError` into the corpus notation.
///
/// Written out variant by variant rather than through `Debug`, so the corpus
/// pins a *classification* and not a message string. The variants that carry a
/// stable, value-free payload (a name, a count) render it; the rest render to
/// their name alone, which is the limit stated in this file's header.
#[allow(deprecated)]
fn render_error(err: &ExecutionError) -> String {
    let body = match err {
        ExecutionError::InvalidArgumentCount { expected, actual } => {
            format!("InvalidArgumentCount: expected={expected} actual={actual}")
        }
        ExecutionError::UnsupportedTargetType { .. } => "UnsupportedTargetType".to_string(),
        ExecutionError::NotSupportedAsMethod { method, .. } => {
            format!("NotSupportedAsMethod: {method}")
        }
        ExecutionError::UnsupportedKeyType(_) => "UnsupportedKeyType".to_string(),
        ExecutionError::UnexpectedType { got, want } => {
            format!("UnexpectedType: got={got} want={want}")
        }
        ExecutionError::NoSuchKey(k) => format!("NoSuchKey: {k}"),
        ExecutionError::NoSuchOverload => "NoSuchOverload".to_string(),
        ExecutionError::UndeclaredReference(n) => format!("UndeclaredReference: {n}"),
        ExecutionError::MissingArgumentOrTarget => "MissingArgumentOrTarget".to_string(),
        ExecutionError::ValuesNotComparable(_, _) => "ValuesNotComparable".to_string(),
        ExecutionError::UnsupportedUnaryOperator(op, _) => {
            format!("UnsupportedUnaryOperator: {op}")
        }
        ExecutionError::UnsupportedBinaryOperator(op, _, _) => {
            format!("UnsupportedBinaryOperator: {op}")
        }
        ExecutionError::UnsupportedMapIndex(_) => "UnsupportedMapIndex".to_string(),
        ExecutionError::UnsupportedListIndex(_) => "UnsupportedListIndex".to_string(),
        ExecutionError::UnsupportedIndex(_, _) => "UnsupportedIndex".to_string(),
        ExecutionError::UnsupportedFunctionCallIdentifierType(_) => {
            "UnsupportedFunctionCallIdentifierType".to_string()
        }
        ExecutionError::UnsupportedFieldsConstruction(_) => {
            "UnsupportedFieldsConstruction".to_string()
        }
        ExecutionError::FunctionError { function, .. } => format!("FunctionError: {function}"),
        ExecutionError::DivisionByZero(_) => "DivisionByZero".to_string(),
        ExecutionError::RemainderByZero(_) => "RemainderByZero".to_string(),
        ExecutionError::Overflow(op, _, _) => format!("Overflow: {op}"),
        ExecutionError::IndexOutOfBounds(_) => "IndexOutOfBounds".to_string(),
        ExecutionError::InternalError(_) => "InternalError".to_string(),
        // `ExecutionError` is `#[non_exhaustive]`. A variant added upstream
        // lands here loudly rather than being silently folded into a
        // neighbouring classification.
        other => format!("UNRENDERED({other})"),
    };
    format!("error({body})")
}

// ---------------------------------------------------------------------------
// the fixed context
// ---------------------------------------------------------------------------

/// The bindings every corpus case sees. One context for the whole corpus,
/// documented in the corpus header as well: a case that needs a new binding
/// adds it here rather than carrying its own environment, so the file stays a
/// list of expressions and answers instead of a program.
fn fixed_context() -> Context<'static> {
    let mut ctx = Context::default();

    ctx.add_variable_from_value("i", Value::Int(42));
    ctx.add_variable_from_value("neg", Value::Int(-7));
    ctx.add_variable_from_value("u", Value::UInt(7));
    ctx.add_variable_from_value("d", Value::Float(1.5));
    ctx.add_variable_from_value("b", Value::Bool(true));
    ctx.add_variable_from_value("s", Value::String(Arc::new("hello".to_string())));
    ctx.add_variable_from_value("by", Value::Bytes(Arc::new(b"hi".to_vec())));
    ctx.add_variable_from_value("nil", Value::Null);

    ctx.add_variable_from_value(
        "xs",
        Value::list(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
            Value::Int(4),
        ]),
    );
    ctx.add_variable_from_value(
        "strs",
        Value::list(vec![
            Value::String(Arc::new("a".to_string())),
            Value::String(Arc::new("bb".to_string())),
            Value::String(Arc::new("ccc".to_string())),
        ]),
    );
    ctx.add_variable_from_value("empty", Value::list(Vec::<Value>::new()));

    let mut m = std::collections::HashMap::new();
    m.insert(Key::String(Arc::new("a".to_string())), Value::Int(1));
    m.insert(Key::String(Arc::new("b".to_string())), Value::Int(2));
    ctx.add_variable_from_value("m", Value::Map(cel::objects::Map::object(Arc::new(m))));

    let mut inner = std::collections::HashMap::new();
    inner.insert(Key::String(Arc::new("k".to_string())), Value::Int(5));
    let mut outer = std::collections::HashMap::new();
    outer.insert(
        Key::String(Arc::new("inner".to_string())),
        Value::Map(cel::objects::Map::object(Arc::new(inner))),
    );
    ctx.add_variable_from_value(
        "nested",
        Value::Map(cel::objects::Map::object(Arc::new(outer))),
    );

    let person = |name: &str, age: i64| {
        let mut p = std::collections::HashMap::new();
        p.insert(
            Key::String(Arc::new("name".to_string())),
            Value::String(Arc::new(name.to_string())),
        );
        p.insert(Key::String(Arc::new("age".to_string())), Value::Int(age));
        Value::Map(cel::objects::Map::object(Arc::new(p)))
    };
    ctx.add_variable_from_value(
        "people",
        Value::list(vec![person("ann", 30), person("bob", 20)]),
    );

    ctx.add_variable_from_value(
        "opt_some",
        Value::Opaque(Arc::new(OptionalValue::of(Value::Int(9)))),
    );
    ctx.add_variable_from_value("opt_none", Value::Opaque(Arc::new(OptionalValue::none())));

    // A host function and a host function that fails: the residual-call axis.
    ctx.add_function("double_it", |v: i64| v * 2);
    ctx.add_function("boom", || -> Result<i64, ExecutionError> {
        Err(ExecutionError::function_error("boom", "deliberate"))
    });

    ctx
}

// ---------------------------------------------------------------------------
// the corpus
// ---------------------------------------------------------------------------

const CORPUS: &str = include_str!("oracle_corpus.txt");

/// One frozen case. `want` is the corpus text verbatim — nothing parses it,
/// which is precisely why it cannot drift with the subject.
struct Case {
    line: usize,
    expr: String,
    feats: Vec<String>,
    want: String,
    cfg: Option<String>,
    /// Which parser configuration compiles this case. `None` is
    /// `Program::compile` — the door a caller holds, whose `Parser::default()`
    /// has optional syntax OFF. `Some("optional")` turns it on, which is the
    /// only way `_[?_]` and `_?._` reach the evaluator at all.
    parse: Option<String>,
}

/// The CEL feature axis, and the floor each point on it must carry.
///
/// This is the coverage assertion. Editing the corpus down until a feature is
/// no longer exercised fails here by name.
const REQUIRED_COVERAGE: &[(&str, usize)] = &[
    ("literal_int", 2),
    ("literal_uint", 2),
    ("literal_double", 2),
    ("literal_bool", 2),
    ("literal_string", 2),
    ("literal_bytes", 1),
    ("literal_null", 2),
    ("literal_list", 2),
    ("literal_map", 2),
    ("arith", 6),
    ("arith_overflow", 2),
    ("arith_div_zero", 2),
    ("compare", 6),
    ("compare_hetero", 3),
    ("equality", 4),
    ("logic_and", 3),
    ("logic_or", 3),
    ("logic_not", 2),
    ("ternary", 3),
    ("index_list", 3),
    ("index_map", 3),
    ("field_select", 3),
    ("in_operator", 3),
    ("size", 3),
    ("macro_has", 3),
    ("macro_all", 2),
    ("macro_exists", 2),
    ("macro_exists_one", 2),
    ("macro_map", 3),
    ("macro_filter", 2),
    ("comprehension_nested", 1),
    ("map_range", 3),
    ("type_fn", 3),
    ("dyn_fn", 1),
    ("conversion", 4),
    ("string_fn", 4),
    ("regex", 1),
    ("optional", 4),
    ("opt_syntax", 2),
    ("opt_select", 3),
    ("duration", 3),
    ("timestamp", 3),
    ("opaque", 2),
    ("struct_literal", 3),
    ("host_function", 2),
    ("error", 8),
    ("variable", 3),
    ("syntax_error", 1),
];

/// Coverage points that only exist when a cargo feature is on. When the
/// feature is off the cases carrying them are skipped, so the requirement is
/// dropped with them rather than failing a build that never could have met it.
const FEATURE_GATED_COVERAGE: &[(&str, &str)] = &[
    ("duration", "chrono"),
    ("timestamp", "chrono"),
    ("regex", "regex"),
];

fn cfg_enabled(name: &str) -> bool {
    match name {
        "chrono" => cfg!(feature = "chrono"),
        "regex" => cfg!(feature = "regex"),
        "structs" => cfg!(feature = "structs"),
        "json" => cfg!(feature = "json"),
        "bytes" => cfg!(feature = "bytes"),
        other => panic!("corpus names an unknown cfg `{other}`"),
    }
}

/// Whether a case's `cfg:` selects this build, where a leading `!` negates.
///
/// The negation is not symmetry for its own sake. A divergence between two
/// evaluators can be *feature-shaped* — the two answered differently only with
/// a feature OFF — and a corpus that can only say "needs feature X" has no way
/// to pin the answer a build WITHOUT it must give. That is exactly how the
/// struct-literal refusal ordering stayed invisible.
fn case_selected(cfg: &str) -> bool {
    match cfg.strip_prefix('!') {
        Some(name) => !cfg_enabled(name),
        None => cfg_enabled(cfg),
    }
}

fn parse_corpus() -> Vec<Case> {
    let mut cases = Vec::new();
    let mut cur: BTreeMap<&str, (usize, String)> = BTreeMap::new();

    let flush = |cur: &mut BTreeMap<&str, (usize, String)>, cases: &mut Vec<Case>| {
        if cur.is_empty() {
            return;
        }
        let (line, expr) = cur
            .remove("expr")
            .unwrap_or_else(|| panic!("corpus record without an `expr:` near line {:?}", cur));
        let (_, feats) = cur
            .remove("feat")
            .unwrap_or_else(|| panic!("corpus record at line {line} has no `feat:`"));
        let (_, want) = cur
            .remove("want")
            .unwrap_or_else(|| panic!("corpus record at line {line} has no `want:`"));
        let cfg = cur.remove("cfg").map(|(_, v)| v);
        let parse = cur.remove("parse").map(|(_, v)| v);
        if let Some(extra) = cur.keys().next() {
            panic!("corpus record at line {line} has an unknown key `{extra}:`");
        }
        cur.clear();
        cases.push(Case {
            line,
            expr,
            feats: feats.split_whitespace().map(str::to_string).collect(),
            want,
            cfg,
            parse,
        });
    };

    for (idx, raw) in CORPUS.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw.trim_end();
        if line.trim_start().starts_with('#') {
            continue;
        }
        if line.trim().is_empty() {
            flush(&mut cur, &mut cases);
            continue;
        }
        let Some((key, value)) = line.split_once(": ") else {
            panic!("corpus line {lineno} is neither blank, a comment, nor `key: value`: {line:?}");
        };
        let key = match key {
            "expr" => "expr",
            "feat" => "feat",
            "want" => "want",
            "cfg" => "cfg",
            "parse" => "parse",
            other => panic!("corpus line {lineno} has unknown key `{other}`"),
        };
        if cur.insert(key, (lineno, value.to_string())).is_some() {
            panic!("corpus line {lineno} repeats key `{key}` inside one record");
        }
    }
    flush(&mut cur, &mut cases);
    cases
}

/// Compiles a case under the parser configuration it asks for. `None` means a
/// failed parse, which the corpus spells `parse_error`.
fn parse_case(case: &Case) -> Option<Expression> {
    match case.parse.as_deref() {
        None => Program::compile(&case.expr)
            .ok()
            .map(|p| p.expression().clone()),
        Some("optional") => Parser::default()
            .enable_optional_syntax(true)
            .parse(&case.expr)
            .ok(),
        Some(other) => panic!(
            "corpus line {} names an unknown parser `{other}`",
            case.line
        ),
    }
}

/// Runs one case through one door and renders whatever came back.
fn observe(evaluator: &Evaluator, case: &Case, ctx: &Context) -> String {
    let Some(expr) = parse_case(case) else {
        return "parse_error".to_string();
    };
    match (evaluator.eval)(&expr, ctx) {
        Ok(v) => render(&v),
        Err(e) => render_error(&e),
    }
}

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

#[test]
fn corpus_agrees_with_every_evaluator() {
    let cases = parse_corpus();
    let ctx = fixed_context();

    let mut disagreements = Vec::new();
    let mut checked = 0usize;
    let mut skipped = 0usize;

    for case in &cases {
        if let Some(cfg) = &case.cfg {
            if !case_selected(cfg) {
                skipped += 1;
                continue;
            }
        }
        for evaluator in EVALUATORS {
            let got = observe(evaluator, case, &ctx);
            checked += 1;
            if got != case.want {
                disagreements.push(format!(
                    "  line {} [{}]  {}\n      want: {}\n      got:  {}",
                    case.line, evaluator.name, case.expr, case.want, got
                ));
            }
        }
    }

    assert!(
        disagreements.is_empty(),
        "{} of {checked} evaluations disagree with the frozen corpus:\n{}",
        disagreements.len(),
        disagreements.join("\n")
    );

    println!(
        "oracle: {} cases x {} evaluator(s) = {checked} evaluations agreed ({skipped} skipped by cfg)",
        cases.len(),
        EVALUATORS.len()
    );
}

/// `EVALUATORS` holds `Value::resolve`, but the door a caller actually holds is
/// `Program::execute`. This pins them together, so the corpus gates the public
/// API and not just an internal entry point.
#[test]
fn public_door_agrees() {
    let cases = parse_corpus();
    let ctx = fixed_context();

    let mut split = Vec::new();
    for case in &cases {
        if case.parse.is_some() {
            continue; // `Program::compile` cannot express these; see `parse:`.
        }
        if case.cfg.as_deref().map(case_selected) == Some(false) {
            continue;
        }
        let Ok(program) = Program::compile(&case.expr) else {
            continue;
        };
        let via_program = match program.execute(&ctx) {
            Ok(v) => render(&v),
            Err(e) => render_error(&e),
        };
        let via_resolve = match Value::resolve(program.expression(), &ctx) {
            Ok(v) => render(&v),
            Err(e) => render_error(&e),
        };
        if via_program != via_resolve {
            split.push(format!(
                "  line {}  {}\n      Program::execute: {via_program}\n      Value::resolve:   {via_resolve}",
                case.line, case.expr
            ));
        }
    }

    assert!(
        split.is_empty(),
        "`Program::execute` and `Value::resolve` disagree, so the corpus no longer \
         gates the public door:\n{}",
        split.join("\n")
    );
}

#[test]
fn every_required_feature_is_covered() {
    let cases = parse_corpus();

    let known: Vec<&str> = REQUIRED_COVERAGE.iter().map(|(name, _)| *name).collect();
    let mut counts: BTreeMap<&str, usize> = known.iter().map(|n| (*n, 0)).collect();

    let mut unknown = Vec::new();
    for case in &cases {
        let live = case.cfg.as_deref().map(case_selected).unwrap_or(true);
        for feat in &case.feats {
            match counts.get_mut(feat.as_str()) {
                Some(slot) => {
                    if live {
                        *slot += 1;
                    }
                }
                None => unknown.push(format!("  line {}: `{feat}`", case.line)),
            }
        }
    }

    assert!(
        unknown.is_empty(),
        "corpus declares feature tags that are not on the axis in REQUIRED_COVERAGE \
         (a typo here would silently satisfy nothing):\n{}",
        unknown.join("\n")
    );

    let mut short = Vec::new();
    for (feat, floor) in REQUIRED_COVERAGE {
        if let Some((_, cfg)) = FEATURE_GATED_COVERAGE.iter().find(|(f, _)| f == feat) {
            if !cfg_enabled(cfg) {
                continue;
            }
        }
        let have = counts[feat];
        if have < *floor {
            short.push(format!("  {feat}: {have} case(s), floor is {floor}"));
        }
    }

    assert!(
        short.is_empty(),
        "the corpus no longer covers every point on the CEL feature axis — a feature \
         dropping out of the corpus must fail the build, not shrink the gate:\n{}",
        short.join("\n")
    );
}

#[test]
fn corpus_is_well_formed() {
    let cases = parse_corpus();
    assert!(!cases.is_empty(), "the corpus is empty");

    // Keyed on the parser and the feature configuration too: the same source
    // under a different `parse:` or `cfg:` is a different case, not a
    // duplicate. A feature-shaped divergence is stated as one expression with
    // two answers, so without `cfg` in the key it could not be written down.
    let mut seen: BTreeMap<(&str, Option<&str>, Option<&str>), usize> = BTreeMap::new();
    let mut dupes = Vec::new();
    for case in &cases {
        let key = (
            case.expr.as_str(),
            case.parse.as_deref(),
            case.cfg.as_deref(),
        );
        if let Some(first) = seen.insert(key, case.line) {
            dupes.push(format!(
                "  line {} repeats the expression first seen at line {first}: {}",
                case.line, case.expr
            ));
        }
    }
    assert!(
        dupes.is_empty(),
        "duplicate expressions inflate the coverage counts without widening the gate:\n{}",
        dupes.join("\n")
    );

    for case in &cases {
        assert!(
            !case.feats.is_empty(),
            "corpus record at line {} declares no features",
            case.line
        );
        assert!(
            !case.want.trim().is_empty(),
            "corpus record at line {} has an empty `want:`",
            case.line
        );
    }
}
