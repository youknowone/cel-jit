//! **A census of what `majit::lower::lower_typed` accepts**, taken over the
//! widest checked-in statement of what cel must evaluate: the 250 `expr:`
//! records of `tests/oracle_corpus.txt`.
//!
//! This file measures. It asserts almost nothing, on purpose — a census that
//! fails when the number moves becomes a number to refit rather than a
//! measurement, and the fraction it prints is meant to be read, not defended.
//!
//! ## Why a raw "N of 250" would be the wrong number
//!
//! The two-bank machine's [`ValType`] has seven variants — `Int`, `Bool`,
//! `UInt`, `Float`, `Str`, `Timestamp`, `Duration` — and no bank for bytes,
//! null or optional. A corpus expression reading `by`, `nil`, `opt_some` or
//! `opt_none` therefore cannot lower under ANY schema anyone could write, so
//! counting it as a decline measures the corpus's input universe rather than
//! `lower_typed`'s op coverage. Those records are separated into their own
//! `DOMAIN` bucket and the census reports TWO fractions:
//!
//! * **raw** — `LOWERED / (LOWERED + DECLINED + DOMAIN)`, what a caller holding
//!   this corpus and this machine would actually see;
//! * **in-domain** — `LOWERED / (LOWERED + DECLINED)`, what the lowering itself
//!   covers once the expressions no schema can name are set aside.
//!
//! Neither is reported alone. The gap between them IS a finding.
//!
//! ## The schema is deliberately maximal
//!
//! `maximal_schema` declares every path derivable from `oracle.rs`'s
//! `fixed_context()` whose leaf has a `ValType`, including the flattened list
//! ELEMENT columns a list is spelled as here (`xs[]`, `people[].age`, built
//! with [`elem_slot_path`] rather than as string literals — the helper is the
//! definition). Declaring the most a schema can declare means a decline in
//! this census is a statement about the LOWERING and not about a stingy
//! environment. Whole-value paths (`m`, `nested`, `people`, `xs`, …) are
//! deliberately left undeclared, because no `ValType` names a map or a list;
//! `lower_typed`'s refusal of them is exactly the information wanted.
//!
//! ## How the decline histogram is bucketed
//!
//! A `LowerError::reason` embeds names and numbers, so the raw strings are
//! near-singletons. `normalise_reason` applies exactly two rewrites, and no
//! others:
//!
//! 1. every backtick-delimited span becomes `` `_` `` (so ``undeclared path
//!    `xs` `` and ``undeclared path `m` `` are one row);
//! 2. every run of ASCII digits becomes `N`.
//!
//! Anything else that differs — a bank name, a type class — stays a distinct
//! row, because those are different reasons and not different spellings of one.
//!
//! ## The honesty clause — read this before quoting the number
//!
//! `oracle_corpus.txt` is a CONFORMANCE corpus. It was written to pin
//! semantics, so it is deliberately dense in error cases, overload refusals,
//! syntactic edge cases, and one-of-each feature probes — a third of it exists
//! to make something fail in a particular way. It is NOT, and was never meant
//! to be, a sample of how CEL is written in production, where the same handful
//! of comparison-and-conjunction shapes recur over declared scalar fields.
//!
//! Whatever fraction this census prints is therefore a **lower bound** on what
//! real CEL traffic would lower, by an unmeasured margin. Nobody should quote
//! it as "X% of real CEL lowers"; the only claim it supports is "X% of the
//! conformance corpus lowers", which is a different and much more pessimistic
//! sentence.
#![cfg(feature = "jit")]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use cel::common::ast::{EntryExpr, Expr, IdedExpr};
use cel::majit::lower::{elem_slot_path, lower_typed, Schema, ValType};
use cel::parser::Parser;

// ---------------------------------------------------------------------------
// the corpus
// ---------------------------------------------------------------------------

/// One corpus record, reduced to what this census needs.
struct Record {
    line: usize,
    expr: String,
    /// The corpus text of `want:`. Used to recognise the rows declared not to
    /// parse, and to cross-tab lowering against the cases that RAISE.
    want: String,
    /// The cargo feature this case is stated against, `!` negating.
    cfg: Option<String>,
    /// Which parser configuration the case is stated against. Not cosmetic:
    /// the corpus carries `m[?"a"]` twice, once under `parse: optional` and
    /// once without, where it is `want: parse_error`.
    parse: Option<String>,
}

impl Record {
    fn parser(&self) -> Parser {
        match self.parse.as_deref() {
            None => Parser::default(),
            Some("optional") => Parser::default().enable_optional_syntax(true),
            Some(other) => panic!(
                "line {} names an unknown parser configuration `{other}`",
                self.line
            ),
        }
    }
}

/// Pulls the `expr:`, `want:`, `cfg:` and `parse:` lines out of every corpus
/// record. The corpus is line-oriented; a record runs from its `expr:` line to
/// the next one.
fn corpus_records() -> Vec<Record> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("oracle_corpus.txt");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let mut records: Vec<Record> = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if let Some(expr) = line.strip_prefix("expr:") {
            records.push(Record {
                line: index + 1,
                expr: expr.trim().to_string(),
                want: String::new(),
                cfg: None,
                parse: None,
            });
            continue;
        }
        let Some(record) = records.last_mut() else {
            continue;
        };
        if let Some(want) = line.strip_prefix("want:") {
            record.want = want.trim().to_string();
        } else if let Some(cfg) = line.strip_prefix("cfg:") {
            record.cfg = Some(cfg.trim().to_string());
        } else if let Some(parse) = line.strip_prefix("parse:") {
            record.parse = Some(parse.trim().to_string());
        }
    }
    records
}

/// Whether a cargo feature the corpus names is on in this build. Mirrors
/// `oracle.rs` — an unknown name is a corpus error and fails loudly rather than
/// silently selecting or skipping.
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
fn case_selected(cfg: &str) -> bool {
    match cfg.strip_prefix('!') {
        Some(name) => !cfg_enabled(name),
        None => cfg_enabled(cfg),
    }
}

// ---------------------------------------------------------------------------
// the maximal schema
// ---------------------------------------------------------------------------

/// Every path `fixed_context()` binds whose leaf has a [`ValType`].
///
/// Scalars and constant `Select` chains are named directly; a list is named by
/// its flattened element columns, which is what a list IS on this machine (see
/// [`elem_slot_path`]). What is absent is absent because no bank can hold it:
/// `by` (bytes), `nil` (null) and the two optionals have no variant, and the
/// whole-value paths `m`, `nested`, `nested.inner`, `xs`, `strs`, `empty`,
/// `people` are maps and lists rather than scalars.
fn maximal_schema() -> Schema {
    let mut schema = Schema::new();
    for (path, ty) in [
        ("i", ValType::Int),
        ("neg", ValType::Int),
        ("u", ValType::UInt),
        ("d", ValType::Float),
        ("b", ValType::Bool),
        ("s", ValType::Str),
        ("m.a", ValType::Int),
        ("m.b", ValType::Int),
        ("nested.inner.k", ValType::Int),
    ] {
        schema.insert(path.to_string(), ty);
    }
    for (list, field, ty) in [
        ("xs", None, ValType::Int),
        ("strs", None, ValType::Str),
        ("empty", None, ValType::Int),
        ("people", Some("name"), ValType::Str),
        ("people", Some("age"), ValType::Int),
    ] {
        schema.insert(elem_slot_path(list, field), ty);
    }
    schema
}

/// The bindings no schema can declare, whatever the machine's op coverage.
const OUT_OF_DOMAIN_BINDINGS: &[&str] = &["by", "nil", "opt_some", "opt_none"];

// ---------------------------------------------------------------------------
// free identifiers
// ---------------------------------------------------------------------------

/// Every `Ident` name reachable from `e`.
///
/// An OVER-approximation of the free set: a comprehension's bound variables are
/// collected too, because separating them would need a scope walk this census
/// does not need. That is sound for its one use — deciding whether a record
/// reads an out-of-domain binding — provided no name in
/// [`OUT_OF_DOMAIN_BINDINGS`] is ever a bound variable in the corpus. It is
/// not: the corpus's macro variables are `x`, `y`, `p`, `k`, `optional` and
/// `size`, and `assert_no_out_of_domain_name_is_bound` re-checks that claim
/// against the corpus text rather than leaving it as a comment.
///
/// The base of a `Select` chain needs no special case: it is an `Ident` node in
/// the operand position, so the plain recursion reaches it.
fn collect_idents(e: &IdedExpr, out: &mut BTreeSet<String>) {
    match &e.expr {
        Expr::Ident(name) => {
            out.insert(name.clone());
        }
        Expr::Select(sel) => collect_idents(&sel.operand, out),
        Expr::Call(call) => {
            if let Some(target) = &call.target {
                collect_idents(target, out);
            }
            for arg in &call.args {
                collect_idents(arg, out);
            }
        }
        Expr::Comprehension(comp) => {
            for part in [
                &comp.iter_range,
                &comp.accu_init,
                &comp.loop_cond,
                &comp.loop_step,
                &comp.result,
            ] {
                collect_idents(part, out);
            }
        }
        Expr::List(list) => {
            for element in &list.elements {
                collect_idents(element, out);
            }
        }
        Expr::Map(map) => {
            for entry in &map.entries {
                match &entry.expr {
                    EntryExpr::MapEntry(kv) => {
                        collect_idents(&kv.key, out);
                        collect_idents(&kv.value, out);
                    }
                    EntryExpr::StructField(field) => collect_idents(&field.value, out),
                }
            }
        }
        Expr::Struct(structure) => {
            for entry in &structure.entries {
                match &entry.expr {
                    EntryExpr::MapEntry(kv) => {
                        collect_idents(&kv.key, out);
                        collect_idents(&kv.value, out);
                    }
                    EntryExpr::StructField(field) => collect_idents(&field.value, out),
                }
            }
        }
        Expr::Literal(_) | Expr::Unspecified => {}
    }
}

// ---------------------------------------------------------------------------
// decline normalisation
// ---------------------------------------------------------------------------

/// Collapses a `LowerError::reason` to its bucket key. The two rewrites are
/// spelled out in this file's header, and are the whole rule.
fn normalise_reason(reason: &str) -> String {
    let mut out = String::with_capacity(reason.len());
    let mut chars = reason.chars().peekable();
    let mut prev_was_digit = false;
    while let Some(c) = chars.next() {
        if c == '`' {
            // A backtick-delimited span, name and all, becomes one placeholder.
            for inner in chars.by_ref() {
                if inner == '`' {
                    break;
                }
            }
            out.push_str("`_`");
            prev_was_digit = false;
        } else if c.is_ascii_digit() {
            if !prev_was_digit {
                out.push('N');
            }
            prev_was_digit = true;
        } else {
            out.push(c);
            prev_was_digit = false;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// the census
// ---------------------------------------------------------------------------

#[test]
fn lower_typed_coverage_over_the_oracle_corpus() {
    let records = corpus_records();

    // A corpus this test cannot find would make it measure nothing, which is
    // the failure mode that looks most like success.
    assert!(
        records.len() > 100,
        "only {} corpus expressions found -- the corpus moved or the record \
         format changed, and this census would otherwise be measuring nothing",
        records.len()
    );

    let schema = maximal_schema();

    let mut parse_error = 0usize;
    let mut cfg_off = 0usize;
    let mut unparsed: Vec<String> = Vec::new();
    let mut domain: Vec<String> = Vec::new();
    let mut lowered = 0usize;
    let mut declined = 0usize;

    let mut decline_histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut sum_reducible = 0usize;
    let mut row_projection = 0usize;
    let mut lowered_wanting_error = 0usize;
    let mut declined_wanting_error = 0usize;
    let mut lowered_error_cases: Vec<String> = Vec::new();

    for record in &records {
        let (line, source) = (record.line, &record.expr);

        // A row the corpus declares unparseable is the parser's subject: there
        // is no tree to lower.
        if record.want == "parse_error" {
            parse_error += 1;
            continue;
        }
        if let Some(cfg) = &record.cfg {
            if !case_selected(cfg) {
                cfg_off += 1;
                continue;
            }
        }

        let Ok(expr) = record.parser().parse(source) else {
            unparsed.push(format!("  line {line}: {source}"));
            continue;
        };

        let mut idents = BTreeSet::new();
        collect_idents(&expr, &mut idents);
        let out_of_domain: Vec<&str> = OUT_OF_DOMAIN_BINDINGS
            .iter()
            .copied()
            .filter(|name| idents.contains(*name))
            .collect();
        if !out_of_domain.is_empty() {
            domain.push(format!(
                "  line {line}: {source}   [{}]",
                out_of_domain.join(" ")
            ));
            continue;
        }

        let wants_error = record.want.starts_with("error(");
        match lower_typed(&expr, &schema) {
            Ok(low) => {
                lowered += 1;
                if low.sum_reducible().is_ok() {
                    sum_reducible += 1;
                }
                if low.is_row_projection() {
                    row_projection += 1;
                }
                if wants_error {
                    lowered_wanting_error += 1;
                    lowered_error_cases
                        .push(format!("  line {line}: {source}\n      {}", record.want));
                }
            }
            Err(error) => {
                declined += 1;
                *decline_histogram
                    .entry(normalise_reason(&error.reason))
                    .or_default() += 1;
                if wants_error {
                    declined_wanting_error += 1;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // the report
    // -----------------------------------------------------------------------

    let total = parse_error + cfg_off + unparsed.len() + domain.len() + lowered + declined;
    let raw_denominator = lowered + declined + domain.len();
    let in_domain_denominator = lowered + declined;
    let pct = |num: usize, den: usize| -> String {
        if den == 0 {
            "n/a".to_string()
        } else {
            format!("{:.1}%", 100.0 * num as f64 / den as f64)
        }
    };

    println!("\n=== lower_typed coverage over tests/oracle_corpus.txt ===\n");
    println!("bucket census");
    println!("  PARSE_ERROR  {parse_error:>4}   declared `want: parse_error`; no tree to lower");
    println!("  CFG_OFF      {cfg_off:>4}   `cfg:` does not select this build");
    println!(
        "  UNPARSED     {:>4}   expected to parse and did not (a DEFECT if nonzero)",
        unparsed.len()
    );
    println!(
        "  DOMAIN       {:>4}   reads {}; no ValType names it under ANY schema",
        domain.len(),
        OUT_OF_DOMAIN_BINDINGS.join("/")
    );
    println!("  LOWERED      {lowered:>4}   lower_typed returned Ok");
    println!("  DECLINED     {declined:>4}   lower_typed returned Err");
    println!("  ------------------");
    println!("  TOTAL        {total:>4}");

    println!("\nacceptance fractions (both, never one alone)");
    println!(
        "  raw        {lowered}/{raw_denominator} = {}   LOWERED / (LOWERED + DECLINED + DOMAIN)",
        pct(lowered, raw_denominator)
    );
    println!(
        "  in-domain  {lowered}/{in_domain_denominator} = {}   LOWERED / (LOWERED + DECLINED)",
        pct(lowered, in_domain_denominator)
    );

    println!("\ndecline histogram (reasons normalised: backtick spans -> `_`, digit runs -> N)");
    let mut rows: Vec<(&String, &usize)> = decline_histogram.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    for (reason, count) in &rows {
        println!("  {count:>4}  {reason}");
    }
    println!("  ({} distinct normalised reasons)", rows.len());

    println!("\nthe two gates AFTER lowering (of the {lowered} LOWERED)");
    println!(
        "  sum_reducible()    {sum_reducible:>4}   ({} of LOWERED) -- reaches the batch reduce path",
        pct(sum_reducible, lowered)
    );
    println!(
        "  is_row_projection(){row_projection:>4}   ({} of LOWERED)",
        pct(row_projection, lowered)
    );
    println!(
        "  lowered but NOT sum_reducible: {}",
        lowered - sum_reducible
    );

    println!("\ncross-tab: does the corpus row expect an error(...)?");
    println!(
        "  LOWERED  with want: error(...)   {lowered_wanting_error:>4}   ({} of LOWERED)",
        pct(lowered_wanting_error, lowered)
    );
    println!(
        "  LOWERED  with a value answer     {:>4}",
        lowered - lowered_wanting_error
    );
    println!("  DECLINED with want: error(...)   {declined_wanting_error:>4}");
    if !lowered_error_cases.is_empty() {
        println!("  the lowered rows that RAISE in the tree-walker:");
        for case in &lowered_error_cases {
            println!("{case}");
        }
    }

    if !domain.is_empty() {
        println!(
            "\nDOMAIN rows ({}), which no schema can admit:",
            domain.len()
        );
        for row in &domain {
            println!("{row}");
        }
    }

    println!(
        "\nREMINDER: oracle_corpus.txt is a CONFORMANCE corpus, not a usage \n\
         distribution. Every fraction above is a LOWER BOUND on what real CEL \n\
         traffic would lower. See this file's header before quoting one.\n"
    );

    // -----------------------------------------------------------------------
    // the assertions -- deliberately three, and no coverage threshold
    // -----------------------------------------------------------------------

    assert_eq!(
        total,
        records.len(),
        "the six buckets must partition the corpus"
    );
    assert!(
        unparsed.is_empty(),
        "{} of {} corpus expressions did not parse under any configuration:\n{}",
        unparsed.len(),
        records.len(),
        unparsed.join("\n")
    );
}

/// The premise [`collect_idents`] over-approximates on: no out-of-domain
/// binding is ever a macro's bound variable in the corpus, so counting bound
/// variables as free cannot move a record into the DOMAIN bucket by mistake.
///
/// Checked against the corpus text rather than asserted in prose, so a future
/// case spelled `xs.map(nil, ...)` fails here instead of silently biasing the
/// census.
#[test]
fn no_out_of_domain_name_is_bound_by_a_corpus_macro() {
    let mut offenders = Vec::new();
    for record in corpus_records() {
        let Ok(expr) = record.parser().parse(&record.expr) else {
            continue;
        };
        let mut bound = BTreeSet::new();
        collect_bound_vars(&expr, &mut bound);
        for name in OUT_OF_DOMAIN_BINDINGS {
            if bound.contains(*name) {
                offenders.push(format!("  line {}: {}", record.line, record.expr));
                break;
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "an out-of-domain name is a bound variable, so the free-identifier \
         over-approximation in `collect_idents` is no longer sound:\n{}",
        offenders.join("\n")
    );
}

/// Every name a comprehension binds anywhere in `e`.
fn collect_bound_vars(e: &IdedExpr, out: &mut BTreeSet<String>) {
    match &e.expr {
        Expr::Comprehension(comp) => {
            out.insert(comp.iter_var.clone());
            if let Some(second) = &comp.iter_var2 {
                out.insert(second.clone());
            }
            out.insert(comp.accu_var.clone());
            for part in [
                &comp.iter_range,
                &comp.accu_init,
                &comp.loop_cond,
                &comp.loop_step,
                &comp.result,
            ] {
                collect_bound_vars(part, out);
            }
        }
        Expr::Select(sel) => collect_bound_vars(&sel.operand, out),
        Expr::Call(call) => {
            if let Some(target) = &call.target {
                collect_bound_vars(target, out);
            }
            for arg in &call.args {
                collect_bound_vars(arg, out);
            }
        }
        Expr::List(list) => {
            for element in &list.elements {
                collect_bound_vars(element, out);
            }
        }
        Expr::Map(map) => {
            for entry in &map.entries {
                match &entry.expr {
                    EntryExpr::MapEntry(kv) => {
                        collect_bound_vars(&kv.key, out);
                        collect_bound_vars(&kv.value, out);
                    }
                    EntryExpr::StructField(field) => collect_bound_vars(&field.value, out),
                }
            }
        }
        Expr::Struct(structure) => {
            for entry in &structure.entries {
                match &entry.expr {
                    EntryExpr::MapEntry(kv) => {
                        collect_bound_vars(&kv.key, out);
                        collect_bound_vars(&kv.value, out);
                    }
                    EntryExpr::StructField(field) => collect_bound_vars(&field.value, out),
                }
            }
        }
        Expr::Ident(_) | Expr::Literal(_) | Expr::Unspecified => {}
    }
}
