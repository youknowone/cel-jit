//! Which compiled batch loops still carry a residual call per element.
//!
//! The `%`-by-a-power-of-two wall was found by differencing two compiled
//! traces op by op; this widens that instrument from one pair to a corpus, so
//! the next wall is picked by census rather than by guess. For every
//! expression it prints the compiled loop's body length and, whenever the body
//! holds a `Call*`, the whole body histogram — a call in the steady-state body
//! is a per-element cost the trace could not inline.
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

/// One expression's census: `(body ops, calls in body)` per compiled loop.
fn census(label: &str, src: &str, data: &Data) {
    let schema = data.schema();
    let lowered = match BatchProgram::compile(src, &schema) {
        Ok(l) => l,
        Err(e) => {
            println!("-- {label:28} DECLINES  {src}\n     {e}");
            return;
        }
    };
    let batch = data.batch();
    let bound = match lowered.bind_per_row(&batch) {
        Ok(b) => b,
        Err(e) => {
            println!("-- {label:28} NO BIND   {src}\n     {e}");
            return;
        }
    };
    reset_persistent_state();
    reset_jit_stats();
    for _ in 0..48 {
        if let Err(e) = bound.collect_on(Tier::Jit) {
            println!("-- {label:28} RUN ERR   {src}\n     {e}");
            return;
        }
    }
    let loops = Census::compiled_opcode_log();
    if loops.is_empty() {
        println!("-- {label:28} NO LOOP   {src}");
        return;
    }
    for (i, ops) in loops.into_iter().enumerate() {
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
        let calls: usize = hist
            .iter()
            .filter(|(k, _)| k.starts_with("Call"))
            .map(|(_, c)| *c)
            .sum();
        let flag = if calls > 0 { "CALL" } else { "    " };
        println!(
            "{flag} {label:28} loop[{i}] body={:3} calls={calls}  {src}",
            body.len()
        );
        if calls > 0 {
            let mut items: Vec<_> = hist.into_iter().collect();
            items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for (name, c) in items {
                println!("       {c:4}  {name}");
            }
        }
    }
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
    ];
    for (label, src) in corpus {
        census(label, src, &data);
    }
}
