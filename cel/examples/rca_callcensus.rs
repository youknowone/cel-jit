//! What every compiled batch loop carries per element: its body length, its
//! residual calls, and its guards.
//!
//! The `%`-by-a-power-of-two wall was found by differencing two compiled
//! traces op by op; this widens that instrument from one pair to a corpus, so
//! the next wall is picked by census rather than by guess.
//!
//! Body length ALONE over-reports, and by a lot: the backend skips an
//! operation that has no side effect and whose result nothing live reads
//! (`backend/x86/regalloc.py:383-386`), so a pure op in the body may cost
//! nothing at all. A `Call*` and a `Guard*` are the two kinds that always
//! survive — a call is work the trace could not inline, and a guard is a
//! comparison, a resume point and a bridge target that no dead-code screen
//! removes. So those two are counted for every loop and ranked at the end,
//! and the body histogram is printed for the loops that lead each ranking.
//!
//! Run it:
//!
//! ```text
//! cargo run --release --package cel --no-default-features \
//!   --features regex,chrono,jit-cranelift --example rca_callcensus
//! ```

use std::collections::BTreeMap;

use cel::majit::batch::{Batch, BatchProgram, ColumnRef, Tier};
use cel::majit::bytecode::float_bank::{reset_jit_stats, reset_persistent_state};
use cel::majit::lower::{Schema, ValType};
use majit_metainterp::embed::Census;

const ROWS: usize = 512;
const LIST_LEN: i64 = 16;

struct Data {
    x: Vec<i64>,
    y: Vec<i64>,
    u: Vec<u64>,
    f: Vec<f64>,
    g: Vec<f64>,
    b: Vec<bool>,
    s: Vec<String>,
    t: Vec<i64>,
    dur: Vec<i64>,
    lens: Vec<i64>,
    nums: Vec<i64>,
    strs: Vec<String>,
}

impl Data {
    fn new() -> Self {
        let n = ROWS;
        Self {
            x: (0..n as i64).collect(),
            y: (0..n as i64).map(|i| i % 7 + 1).collect(),
            u: (0..n as u64).collect(),
            f: (0..n).map(|i| i as f64 * 0.5).collect(),
            g: (0..n).map(|i| i as f64 * 0.25 + 1.0).collect(),
            b: (0..n).map(|i| i % 2 == 0).collect(),
            s: (0..n)
                .map(|i| ["ab", "cd", "ef", "gh", "ij"][i % 5].to_string())
                .collect(),
            t: (0..n as i64)
                .map(|i| 1_700_000_000_000_000_000 + i * 1_000_000_007)
                .collect(),
            dur: (0..n as i64).map(|i| i * 1_000_000_007).collect(),
            lens: vec![LIST_LEN; n],
            nums: (0..n as i64 * LIST_LEN).map(|i| i % 97).collect(),
            strs: (0..n as i64 * LIST_LEN)
                .map(|i| ["p", "q", "r"][(i % 3) as usize].to_string())
                .collect(),
        }
    }

    fn schema(&self) -> Schema {
        [
            ("x", ValType::Int),
            ("y", ValType::Int),
            ("u", ValType::UInt),
            ("f", ValType::Float),
            ("g", ValType::Float),
            ("b", ValType::Bool),
            ("s", ValType::Str),
            ("t", ValType::Timestamp),
            ("dur", ValType::Duration),
            ("nums[]", ValType::Int),
            ("strs[]", ValType::Str),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
    }

    fn batch(&self) -> Batch<'_> {
        Batch::new(ROWS)
            .column("x", ColumnRef::Int(&self.x))
            .column("y", ColumnRef::Int(&self.y))
            .column("u", ColumnRef::UInt(&self.u))
            .column("f", ColumnRef::Float(&self.f))
            .column("g", ColumnRef::Float(&self.g))
            .column("b", ColumnRef::Bool(&self.b))
            .column("s", ColumnRef::Str(&self.s))
            .column("t", ColumnRef::Timestamp(&self.t))
            .column("dur", ColumnRef::Duration(&self.dur))
            .column(
                "nums",
                ColumnRef::List {
                    lens: &self.lens,
                    fields: vec![(None, ColumnRef::Int(&self.nums))],
                },
            )
            .column(
                "strs",
                ColumnRef::List {
                    lens: &self.lens,
                    fields: vec![(None, ColumnRef::Str(&self.strs))],
                },
            )
    }
}

/// One compiled loop's census.
struct Row {
    label: String,
    loop_ix: usize,
    body: usize,
    calls: usize,
    guards: usize,
    hist: BTreeMap<String, usize>,
    src: String,
}

/// Print `row`'s body histogram, most frequent first.
fn print_hist(row: &Row) {
    println!(
        "   {} loop[{}] body={} calls={} guards={}  {}",
        row.label, row.loop_ix, row.body, row.calls, row.guards, row.src
    );
    let mut items: Vec<_> = row.hist.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (name, c) in items {
        println!("       {c:4}  {name}");
    }
}

/// One expression's census: one [`Row`] per compiled loop.
fn census(label: &str, src: &str, data: &Data) -> Vec<Row> {
    let schema = data.schema();
    let lowered = match BatchProgram::compile(src, &schema) {
        Ok(l) => l,
        Err(e) => {
            println!("-- {label:28} DECLINES  {src}\n     {e}");
            return Vec::new();
        }
    };
    let batch = data.batch();
    let bound = match lowered.bind_per_row(&batch) {
        Ok(b) => b,
        Err(e) => {
            println!("-- {label:28} NO BIND   {src}\n     {e}");
            return Vec::new();
        }
    };
    reset_persistent_state();
    reset_jit_stats();
    for _ in 0..48 {
        if let Err(e) = bound.collect_on(Tier::Jit) {
            println!("-- {label:28} RUN ERR   {src}\n     {e}");
            return Vec::new();
        }
    }
    let loops = Census::compiled_opcode_log();
    if loops.is_empty() {
        println!("-- {label:28} NO LOOP   {src}");
        return Vec::new();
    }
    // `RCACC_DUMP=<label>` prints the whole trace, `Label` markers included.
    // The histogram cannot answer where the body starts when a trace holds more
    // than one `Label`, and a nested comprehension does.
    let dump = std::env::var("RCACC_DUMP").is_ok_and(|d| d == label);
    let mut rows = Vec::new();
    for (i, ops) in loops.into_iter().enumerate() {
        if dump {
            println!("== {label} loop[{i}] {} ops  {src}", ops.len());
            for (j, op) in ops.iter().enumerate() {
                println!("   [{j:3}] {op:?}");
            }
        }
        // The last Label starts the steady-state body; everything before it is
        // the preamble the peeled iteration hoists invariants into.
        let body_at = ops
            .iter()
            .rposition(|op| format!("{op:?}") == "Label")
            .map_or(0, |p| p + 1);
        let body = &ops[body_at..];
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        for op in body {
            *hist.entry(format!("{op:?}")).or_default() += 1;
        }
        let sum_of = |prefix: &str| -> usize {
            hist.iter()
                .filter(|(k, _)| k.starts_with(prefix))
                .map(|(_, c)| *c)
                .sum()
        };
        let calls = sum_of("Call");
        let guards = sum_of("Guard");
        let flag = if calls > 0 { "CALL" } else { "    " };
        println!(
            "{flag} {label:28} loop[{i}] body={:3} calls={calls} guards={guards}  {src}",
            body.len()
        );
        rows.push(Row {
            label: label.to_string(),
            loop_ix: i,
            body: body.len(),
            calls,
            guards,
            hist,
            src: src.to_string(),
        });
    }
    rows
}

fn main() {
    let data = Data::new();
    let corpus: &[(&str, &str)] = &[
        ("int/add", "x + y"),
        ("int/mul_add", "x * 3 + 1"),
        ("int/sub", "x - y * 2"),
        ("int/neg", "-x"),
        ("int/div_pow2", "x / 2"),
        ("int/div_odd", "x / 3"),
        ("int/mod_pow2", "x % 2"),
        ("int/mod_odd", "x % 3"),
        ("int/div_var", "x / y"),
        ("int/mod_var", "x % y"),
        ("uint/div_pow2", "u / 2u"),
        ("uint/mod_pow2", "u % 8u"),
        ("uint/mod_odd", "u % 3u"),
        ("float/mul_add", "f * 1.5 + g"),
        ("float/div", "f / 2.0"),
        ("bool/and", "x > 5 && y < 3"),
        ("bool/not", "!b"),
        ("cond/ternary", "x > 5 ? x * 2 : y + 1"),
        ("str/eq", "s == \"ab\""),
        ("str/ne", "s != \"zz\""),
        ("str/startswith", "s.startsWith(\"a\")"),
        ("str/size", "size(s)"),
        ("str/in_list", "s in [\"ab\", \"cd\"]"),
        ("int/in_list", "x in [1, 2, 3]"),
        ("list/size", "size(nums)"),
        ("list/index", "nums[0] + nums[1]"),
        ("list/map", "nums.map(i, i * 2)"),
        ("list/map_div", "nums.map(i, i / 2)"),
        ("list/map_mod", "nums.map(i, i % 3)"),
        ("list/filter_mod2", "nums.filter(i, i % 2 == 0)"),
        ("list/filter_gt", "nums.filter(i, i > 3)"),
        ("list/exists", "nums.exists(i, i > 5)"),
        ("list/all", "nums.all(i, i >= 0)"),
        ("list/str_exists", "strs.exists(i, i == \"q\")"),
        ("list/str_filter", "strs.filter(i, i != \"r\")"),
        ("ts/hours", "t.getHours()"),
        ("ts/minutes", "t.getMinutes()"),
        ("ts/seconds", "t.getSeconds()"),
        ("ts/dayofmonth", "t.getDayOfMonth()"),
        ("ts/dayofweek", "t.getDayOfWeek()"),
        ("dur/seconds", "dur.getSeconds()"),
        ("dur/hours", "dur.getHours()"),
        ("str/concat", "s + \"!\""),
        ("str/endswith", "s.endsWith(\"b\")"),
        ("str/contains", "s.contains(\"b\")"),
        ("str/matches", "s.matches(\"a.\")"),
        ("str/from_int", "string(x)"),
        ("str/lt", "s < \"cd\""),
        ("cast/int_from_float", "int(f)"),
        ("cast/float_from_int", "double(x)"),
        ("cast/uint", "uint(x)"),
        ("temporal/cmp", "t > timestamp(\"2024-01-01T00:00:00Z\")"),
        ("temporal/sub", "t - t"),
        ("temporal/add_dur", "t + dur"),
        ("temporal/dur_cmp", "dur > duration(\"1s\")"),
        ("list/in", "x in nums"),
        ("list/nested", "nums.filter(i, i > 3).map(j, j * 2)"),
        ("list/map_str", "strs.map(i, i + \"!\")"),
        ("list/index_var", "nums[x % 4]"),
        ("bool/ternary_str", "b ? s : \"zz\""),
        ("math/abs_like", "x < 0 ? -x : x"),
        ("float/cmp_chain", "f > 1.0 && g < 100.0"),
    ];
    let mut rows: Vec<Row> = Vec::new();
    for (label, src) in corpus {
        rows.extend(census(label, src, &data));
    }

    // The two survivors of the backend's dead-code screen, ranked. A body
    // number on its own does not say what a loop costs; these two do.
    println!("\n== loops carrying a residual call");
    rows.sort_by(|a, b| b.calls.cmp(&a.calls).then(a.label.cmp(&b.label)));
    for row in rows.iter().filter(|r| r.calls > 0) {
        print_hist(row);
    }
    if rows.iter().all(|r| r.calls == 0) {
        println!("   none");
    }

    println!("\n== loops by guard count (top 8)");
    rows.sort_by(|a, b| b.guards.cmp(&a.guards).then(a.label.cmp(&b.label)));
    for row in rows.iter().take(8) {
        print_hist(row);
    }

    // Body length is the weakest of the three readings, but it is the one that
    // says where to look for the other two.
    println!("\n== loops by body length (top 8)");
    rows.sort_by(|a, b| b.body.cmp(&a.body).then(a.label.cmp(&b.label)));
    for row in rows.iter().take(8) {
        print_hist(row);
    }
}
