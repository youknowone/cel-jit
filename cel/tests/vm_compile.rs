//! The compiler's gate: every expression the differential corpus holds
//! compiles, or is refused with a typed error.
//!
//! The corpus is the frozen list of expressions cel is expected to evaluate,
//! so it is also the widest available statement of what the compiler must
//! accept. Reusing it here costs nothing and means new coverage lands in both
//! places at once.
//!
//! What this asserts is deliberately weaker than the oracle's: no answers are
//! checked, because nothing executes yet. What it does catch is a shape the
//! compiler cannot encode at all -- and, because a `CompileError` is a value
//! rather than a panic, it distinguishes "refused" from "crashed".

use std::path::Path;

use cel::parser::Parser;
use cel::vm::compile;

/// One corpus record, reduced to what this test needs.
struct Record {
    line: usize,
    expr: String,
    /// The corpus text of `want:`, used only to recognise the cases that are
    /// declared not to parse.
    want: String,
    /// Which parser configuration the case is stated against.
    ///
    /// This is not cosmetic: the corpus carries `m[?"a"]` twice, once under
    /// `parse: optional` where it is a map index and once without, where it
    /// is `want: parse_error`. Parsing every row with optional syntax on
    /// would quietly turn that second row into a contradiction.
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

/// Pulls the `expr:`, `want:` and `parse:` lines out of every corpus record.
///
/// The corpus is line-oriented and comments start with `#`. A record runs
/// from its `expr:` line to the next one.
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
                parse: None,
            });
        } else if let Some(want) = line.strip_prefix("want:") {
            if let Some(record) = records.last_mut() {
                record.want = want.trim().to_string();
            }
        } else if let Some(parse) = line.strip_prefix("parse:") {
            if let Some(record) = records.last_mut() {
                record.parse = Some(parse.trim().to_string());
            }
        }
    }
    records
}

#[test]
fn every_corpus_expression_compiles_or_is_refused_with_a_typed_error() {
    let records = corpus_records();

    // A corpus this test cannot find would make it pass on nothing, which is
    // the failure mode that looks most like success.
    assert!(
        records.len() > 100,
        "only {} corpus expressions found -- the corpus moved or the record \
         format changed, and this test would otherwise be checking nothing",
        records.len()
    );

    let mut refused = Vec::new();
    let mut unparsed = Vec::new();
    let mut compiled = 0usize;
    let mut expected_parse_errors = 0usize;

    for record in &records {
        let (line, source) = (&record.line, &record.expr);

        // A case the corpus declares unparseable is the parser's subject, not
        // the compiler's -- there is no tree to compile.
        if record.want == "parse_error" {
            assert!(
                record.parser().parse(source).is_err(),
                "line {line} is declared `want: parse_error` but parses: {source}"
            );
            expected_parse_errors += 1;
            continue;
        }

        let Ok(expr) = record.parser().parse(source) else {
            unparsed.push(format!("  line {line}: {source}"));
            continue;
        };
        match compile(&expr) {
            Ok(_) => compiled += 1,
            Err(error) => refused.push(format!("  line {line}: {source}\n      {error}")),
        }
    }

    assert!(
        refused.is_empty(),
        "{} of {} corpus expressions did not compile:\n{}",
        refused.len(),
        records.len(),
        refused.join("\n")
    );

    // A row the parser rejects is a corpus or parser question, not the
    // compiler's -- but it silently shrinks what this test covers, so it is
    // named rather than counted.
    assert!(
        unparsed.is_empty(),
        "{} of {} corpus expressions did not parse under any configuration:\n{}",
        unparsed.len(),
        records.len(),
        unparsed.join("\n")
    );

    assert_eq!(compiled + expected_parse_errors, records.len());
}

/// The compiler reports, rather than panics, on an expression tree the parser
/// cannot produce.
///
/// `Expr::Unspecified` is the case the walker panics on
/// (`Can't evaluate Unspecified Expr`). Compiling has to be total over the
/// AST type, not over the subset the parser happens to emit.
#[test]
fn an_unspecified_node_is_refused_without_panicking() {
    use cel::common::ast::Expr;
    use cel::IdedExpr;

    let error = compile(&IdedExpr {
        id: 1,
        expr: Expr::Unspecified,
    })
    .expect_err("an unspecified expression has no value to compile");

    // The message names the node, so a failure is locatable in the source.
    assert!(error.to_string().contains("unspecified"), "{error}");
}
