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
//! null or optional.
//!
//! The `DOMAIN` bucket holds the records that fail for THAT reason rather than
//! for a missing operator: a record reading `by`, `nil`, `opt_some` or
//! `opt_none` cannot lower under ANY schema anyone could write, so counting it
//! as a decline would measure the corpus's input universe rather than
//! `lower_typed`'s op coverage. The test is [`collect_out_of_domain`], over the
//! parsed AST, and it looks at BINDINGS ONLY.
//!
//! ## Why bindings and not literals
//!
//! `b"hi"` and `null` are values the machine cannot hold either, so the obvious
//! move is to treat an out-of-domain LITERAL the same way. That was tried and
//! reverted; the reasons are worth keeping, because each costs a measurement to
//! rediscover.
//!
//! **A binding is fatal on its own; a literal is not.** A binding has to become
//! a column, the schema is what names a column's bank, and there is no bank to
//! name — so no schema over any context admits it. A literal can be erased by
//! constant folding before any bank is chosen: `null == null` carries a null
//! literal and LOWERS today, to a folded `true`. Treating literals as fatal
//! moves 7 records into `DOMAIN` and takes that one out of `LOWERED`, which is
//! a record the machine demonstrably handles.
//!
//! **Conditioning on the decline instead splits a single failure class in
//! two.** The repair for that counterexample is "a literal counts only when the
//! lowering also declined", and it is worse than the problem. It tests that a
//! decline happened SOMEWHERE in the record, never that the literal caused it —
//! so it lands `type(null)` in `DOMAIN` and `type(1)` in `DECLINED`, though
//! both fail for one reason: `type()` folds to a `Value::Opaque` and no bank
//! holds it. The only thing telling them apart is an incidental null literal
//! that is not why either declined.
//!
//! That is the objection, and it is not about how many rows move. The bucket
//! already under-approximates its predicate (next section), and a UNIFORM
//! under-approximation is honest. One that admits some members of a failure
//! class and excludes others on an incidental syntactic feature is an artifact:
//! a reader seeing `type(null)` filed as "no schema can help" beside `type(1)`
//! filed as a decline will conclude something the data does not support. In
//! passing it also shrinks `constant of type `opaque`` from 17 to 14, taking
//! 15% off the census's one actionable finding.
//!
//! Attributing by CAUSE would fix that, and cause is not available here: the
//! only channel carrying it is the `LowerError` text, which is barred because
//! it goes stale the moment a message is reworded. So the choice is between two
//! UNIFORM rules — bindings only, or bindings plus literals-with-decline — and
//! only the first is uniform.
//!
//! **The message text is not an alternative channel.** Seven records carry an
//! out-of-domain literal:
//!
//! ```text
//!   line 89    b"hi"                     line 656   type(null)
//!   line 93    null                      line 775   type(null) == null_type
//!   line 97    null == null              line 1152  b"ab" + b"cd"
//!   line 652   type(b"hi")
//! ```
//!
//! Only two say so in their `LowerError` — line 89 (`bytes literal`) and line
//! 93 (`null literal`). Line 97 does not decline at all, and the remaining four
//! decline for a downstream reason, so any rule reading the reason string sees
//! two of seven and silently misses the rest.
//!
//! So `DOMAIN` stays the narrow syntactic test. It is uniform, and it is
//! decidable without running the thing it exists to explain — neither of which
//! any wider version manages. The choice costs nothing besides: `raw` reads
//! 55.1% under either rule (see the invariance note below), so the fraction
//! worth quoting does not depend on it.
//!
//! ## `DOMAIN` under-approximates its own predicate, deliberately
//!
//! Stated plainly so the number is not over-read: the bucket is a SUFFICIENT
//! SYNTACTIC TEST for "cannot lower under any schema", not that predicate
//! itself. Other declines satisfy the predicate too — a folded constant of an
//! unrepresentable type, a struct literal — because no schema declaration
//! changes what a folded constant's type is. Widening the test would move those
//! records out of `DECLINED` without a single change to `lower_typed`, which is
//! exactly why the fraction to quote is the one that cannot be moved that way.
//!
//! With `DOMAIN` separated, the census reports TWO fractions:
//!
//! * **raw** — `LOWERED / (LOWERED + DECLINED + DOMAIN)`, what a caller holding
//!   this corpus and this machine would actually see;
//! * **in-domain** — `LOWERED / (LOWERED + DECLINED)`, what the lowering itself
//!   covers once the expressions no schema can name are set aside.
//!
//! Neither is reported alone. The gap between them IS a finding.
//!
//! And they are not equally solid. Moving a record between `DECLINED` and
//! `DOMAIN` reclassifies it inside the SAME denominator, so **`raw` is
//! INVARIANT under where the `DOMAIN` boundary is drawn** and only `in-domain`
//! responds to it. `raw` is a measurement; `in-domain` is a measurement plus a
//! judgement about that boundary. **When the two disagree, quote `raw`** — it
//! is the one a wider definition of `DOMAIN` cannot inflate.
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
//! Normalising is what makes the histogram countable, and it is also what makes
//! it unactionable on its own: "26 declines about a constant of some type" is a
//! size without a name, and nobody can turn it into work. So each bucket also
//! prints the DISTINCT RAW reasons underneath it with their own counts, which
//! is where the type names, the function names and the paths survive. Every
//! bucket gets its sub-rows, with no threshold — a threshold would hide exactly
//! the long tail that says what is missing. The one exception is a bucket whose
//! sole raw reason is the bucket string itself (normalisation was a no-op
//! because the reason carries no backticks and no digits); printing that twice
//! would be noise.
//!
//! Every row also carries `(n err)`: how many of its records the corpus writes
//! as `want: error(...)`. That column is what tells a coverage GAP apart from
//! correct behaviour, and a count alone cannot. A row of expressions the corpus
//! wrote to RAISE is one where refusing to lower may be the right answer rather
//! than a hole — the tree-walker is going to produce an error for them either
//! way — whereas a row with no error cases is asking for work. See
//! [`DeclineCell`]. The whole-population version of the same number is in the
//! cross-tab below the histogram; this is that number resolved per reason.
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
// out-of-domain reads
// ---------------------------------------------------------------------------

/// Every out-of-domain BINDING `e` reads: a name in [`OUT_OF_DOMAIN_BINDINGS`].
///
/// Bindings only, and that narrowness is the point — see the census header's
/// "why bindings and not literals" for the measurement that settled it. A
/// binding is decidable from the AST alone and needs nothing from
/// `lower_typed`, so this walk cannot be wrong about a record for a reason the
/// lowering could later reword.
///
/// An OVER-approximation: a comprehension's bound variables are collected too,
/// because separating them would need a scope walk this census does not need.
/// That is sound provided no name in [`OUT_OF_DOMAIN_BINDINGS`] is ever a bound
/// variable in the corpus, which
/// `no_out_of_domain_name_is_bound_by_a_corpus_macro` re-checks against the
/// corpus text rather than leaving as a comment.
///
/// The base of a `Select` chain needs no special case: it is an `Ident` node in
/// the operand position, so the plain recursion reaches it.
fn collect_out_of_domain(e: &IdedExpr, out: &mut BTreeSet<String>) {
    match &e.expr {
        Expr::Ident(name) => {
            if OUT_OF_DOMAIN_BINDINGS.contains(&name.as_str()) {
                out.insert(name.clone());
            }
        }
        Expr::Select(sel) => collect_out_of_domain(&sel.operand, out),
        Expr::Call(call) => {
            if let Some(target) = &call.target {
                collect_out_of_domain(target, out);
            }
            for arg in &call.args {
                collect_out_of_domain(arg, out);
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
                collect_out_of_domain(part, out);
            }
        }
        Expr::List(list) => {
            for element in &list.elements {
                collect_out_of_domain(element, out);
            }
        }
        Expr::Map(map) => {
            for entry in &map.entries {
                match &entry.expr {
                    EntryExpr::MapEntry(kv) => {
                        collect_out_of_domain(&kv.key, out);
                        collect_out_of_domain(&kv.value, out);
                    }
                    EntryExpr::StructField(field) => collect_out_of_domain(&field.value, out),
                }
            }
        }
        Expr::Struct(structure) => {
            for entry in &structure.entries {
                match &entry.expr {
                    EntryExpr::MapEntry(kv) => {
                        collect_out_of_domain(&kv.key, out);
                        collect_out_of_domain(&kv.value, out);
                    }
                    EntryExpr::StructField(field) => collect_out_of_domain(&field.value, out),
                }
            }
        }
        Expr::Literal(_) | Expr::Unspecified => {}
    }
}

// ---------------------------------------------------------------------------
// decline normalisation
// ---------------------------------------------------------------------------

/// One histogram cell: how many records landed on a reason, and how many of
/// those the corpus declares should RAISE (`want: error(...)`).
///
/// The second number is what separates a coverage GAP from correct behaviour. A
/// bucket whose records are all error cases is one where declining may be the
/// right answer — the corpus wrote those rows to fail — and a bucket with no
/// error cases is asking for work. A total alone cannot tell them apart.
#[derive(Default, Clone, Copy)]
struct DeclineCell {
    total: usize,
    wants_error: usize,
}

impl DeclineCell {
    /// The cell covering every cell in `cells`, for rolling raw reasons up into
    /// their normalised bucket.
    fn sum<'a>(cells: impl IntoIterator<Item = &'a DeclineCell>) -> DeclineCell {
        cells
            .into_iter()
            .fold(DeclineCell::default(), |mut acc, cell| {
                acc.total += cell.total;
                acc.wants_error += cell.wants_error;
                acc
            })
    }
}

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

    // Normalised bucket -> the distinct RAW reasons in it -> how many each.
    // Two levels rather than one: the outer key is what makes the histogram
    // countable, the inner keys are what make it a work list.
    let mut decline_histogram: BTreeMap<String, BTreeMap<String, DeclineCell>> = BTreeMap::new();
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

        // A binding settles it without asking the lowering anything: no schema
        // can name the column, so the answer is the same for every schema. This
        // test is deliberately narrow; the header says why nothing else is in
        // it.
        let mut out_of_domain = BTreeSet::new();
        collect_out_of_domain(&expr, &mut out_of_domain);
        if !out_of_domain.is_empty() {
            let label = out_of_domain.iter().cloned().collect::<Vec<_>>().join(" ");
            domain.push(format!("  line {line}: {source}   [{label}]"));
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
                let cell = decline_histogram
                    .entry(normalise_reason(&error.reason))
                    .or_default()
                    .entry(error.reason.clone())
                    .or_default();
                cell.total += 1;
                if wants_error {
                    cell.wants_error += 1;
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
    println!("  ---");
    println!("  `raw` is INVARIANT under where the DOMAIN boundary is drawn: moving a");
    println!("  record between DECLINED and DOMAIN reclassifies it inside the SAME");
    println!("  denominator, so only `in-domain` responds. raw is a measurement;");
    println!("  in-domain is a measurement PLUS a judgement about that boundary.");
    println!("  WHEN THE TWO DISAGREE, QUOTE raw -- widening the definition of DOMAIN");
    println!("  can inflate in-domain without a single change to lower_typed.");

    println!("\ndecline histogram (reasons normalised: backtick spans -> `_`, digit runs -> N)");
    println!("  indented rows are the DISTINCT RAW reasons inside the bucket above them");
    println!("  (n err) = how many of that row's records carry `want: error(...)`, i.e. are");
    println!("  corpus rows written to RAISE -- where declining may be correct, not a gap");
    let mut rows: Vec<(&String, DeclineCell, &BTreeMap<String, DeclineCell>)> = decline_histogram
        .iter()
        .map(|(reason, raw)| (reason, DeclineCell::sum(raw.values()), raw))
        .collect();
    rows.sort_by(|a, b| b.1.total.cmp(&a.1.total).then_with(|| a.0.cmp(b.0)));
    let mut distinct_raw = 0usize;
    for (reason, cell, raw) in &rows {
        println!(
            "  {:>4} ({:>2} err)  {reason}",
            cell.total, cell.wants_error
        );
        distinct_raw += raw.len();
        // A bucket holding exactly one raw reason equal to the bucket string
        // is one normalisation did nothing to: there is no name to recover,
        // and printing the same line indented would say it twice.
        if raw.len() == 1 && raw.contains_key(*reason) {
            continue;
        }
        let mut raw_rows: Vec<(&String, &DeclineCell)> = raw.iter().collect();
        raw_rows.sort_by(|a, b| b.1.total.cmp(&a.1.total).then_with(|| a.0.cmp(b.0)));
        for (text, sub) in raw_rows {
            println!(
                "        {:>4} ({:>2} err)  {text}",
                sub.total, sub.wants_error
            );
        }
    }
    println!(
        "  ({} distinct normalised reasons, {distinct_raw} distinct raw)",
        rows.len()
    );

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
    println!(
        "  DECLINED with want: error(...)   {declined_wanting_error:>4}   \
         (per reason: the (n err) column above)"
    );
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

/// The premise [`collect_out_of_domain`] over-approximates on: no out-of-domain
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
         over-approximation in `collect_out_of_domain` is no longer sound:\n{}",
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
