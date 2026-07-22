//! Proof-of-concept meta-tracing JIT tier for cel-rust, built on the in-repo
//! `majit` framework (a Rust port of RPython's tracing JIT). Tracked as
//! cell-majit (issue #357).
//!
//! majit does not attach to cel-rust's live `Value::resolve_val` evaluator:
//! that path returns `Cow<'a, dyn Val>` and dispatches through trait objects /
//! `downcast_ref`, which is outside the restricted Rust subset majit can
//! meta-trace. Instead, a CEL `Program` (a fixed AST = green constant) is
//! lowered to a flat `i64`-word bytecode, and a small mainloop authored in the
//! traceable subset evaluates it over a batch of inputs. The de-risk phase
//! established the perf envelope (scalar arith 9-16x, slot-resolved policy
//! predicate ~6x over a clean interpreter, comprehensions 2-4x only when the
//! list length is a green constant and can be unrolled).
//!
//! ## M1 — smoke check ([`smoke`])
//!
//! A self-contained register-machine mainloop cloned from
//! `majit/examples/tinyframe`. It de-risks, before any CEL-IR work:
//!   * the `majit` crates build as cross-workspace path deps of `cel`,
//!   * the cranelift backend links here,
//!   * a hot loop actually traces + compiles (observed via
//!     `set_on_compile_loop`).
//!
//! ## M2 — CEL AST -> traceable bytecode ([`lower`], [`bytecode`])
//!
//! [`lower::lower`] compiles the supported `Expr` subset (`Int`/`Boolean`
//! literals, slot-resolved `Ident`/`Select`, arithmetic `+ - * / %`, unary
//! `-`, comparisons, boolean `&& || !`) to the flat `i64`-word program of
//! [`bytecode`]. Anything outside the subset returns [`lower::LowerError`], the
//! signal to fall back to the stock tree-walking evaluator. Correctness is
//! pinned by cross-checking the lowered program against the real
//! `Program::execute` on the same inputs (see the tests below).
//!
//! ## M3+ — batch evaluation + green-length comprehension unroll (planned)
//!
//! Wrap the lowered body in the batch-over-rows loop (the majit merge point),
//! add a batch API, and unroll green-length comprehensions.

pub mod bytecode;
pub mod lower;
pub mod smoke;

#[cfg(test)]
mod tests {
    use super::bytecode::{clean_interp, eval_batch_sum, run_jit, COMPILES};
    use super::lower::lower;
    use crate::{Context, Program, Value};
    use core::sync::atomic::Ordering;
    use std::collections::HashMap;

    #[derive(Debug, Clone, Copy)]
    enum Bind {
        Int(i64),
        Bool(bool),
    }

    impl Bind {
        fn as_i64(self) -> i64 {
            match self {
                Bind::Int(v) => v,
                Bind::Bool(b) => b as i64,
            }
        }
    }

    /// Cross-check: a lowered CEL expression, run on both the clean interpreter
    /// and the majit mainloop (JIT compilation disabled — this validates the
    /// lowering, not the trace), yields the same scalar the stock
    /// `Program::execute` tree-walker does for the same variable bindings.
    fn check(expr_src: &str, binds: &[(&str, Bind)]) {
        let program = Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let lowered = lower(program.expression())
            .unwrap_or_else(|e| panic!("lower `{expr_src}`: {e}"));

        // Stock tree-walker reference.
        let mut ctx = Context::default();
        for (name, b) in binds {
            match b {
                Bind::Int(v) => ctx.add_variable_from_value(*name, *v),
                Bind::Bool(v) => ctx.add_variable_from_value(*name, *v),
            }
        }
        let cel_val = program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"));
        let cel_i = match cel_val {
            Value::Bool(b) => b as i64,
            Value::Int(i) => i,
            other => panic!("`{expr_src}`: unexpected result {other:?}"),
        };

        // majit inputs aligned to the lowering's slot order.
        let map: HashMap<&str, i64> = binds.iter().map(|(n, b)| (*n, b.as_i64())).collect();
        let inputs: Vec<i64> = lowered
            .slots
            .iter()
            .map(|s| *map.get(s.path.as_str()).unwrap_or_else(|| panic!("no binding for slot `{}`", s.path)))
            .collect();
        let prog = lowered.program_for(&inputs);

        let clean = clean_interp(&prog, lowered.num_regs);
        assert_eq!(clean, cel_i, "clean interp vs stock for `{expr_src}` {binds:?}");
        let jit = run_jit(&prog, lowered.num_regs, u32::MAX);
        assert_eq!(jit, cel_i, "majit (jit-off) vs stock for `{expr_src}` {binds:?}");
    }

    #[test]
    fn scalar_policy_predicate() {
        for (a, b, c) in [
            (5i64, 3i64, false),
            (3, 5, false),
            (5, 3, true),
            (1, 1, false),
            (-2, -3, true),
            (i64::MIN + 1, i64::MAX, false),
        ] {
            check(
                "a >= b && !c",
                &[("a", Bind::Int(a)), ("b", Bind::Int(b)), ("c", Bind::Bool(c))],
            );
        }
    }

    #[test]
    fn arithmetic() {
        check(
            "(a + b) * c - 2",
            &[("a", Bind::Int(3)), ("b", Bind::Int(4)), ("c", Bind::Int(5))],
        );
        check(
            "a * b + c",
            &[("a", Bind::Int(-6)), ("b", Bind::Int(7)), ("c", Bind::Int(11))],
        );
        check("-a + b", &[("a", Bind::Int(9)), ("b", Bind::Int(4))]);
    }

    #[test]
    fn division_and_modulo() {
        // Constant fold (cometkim's `simple_arithmetic`): 1 + 2*3 - 4/2 == 5.
        check("1 + 2 * 3 - 4 / 2", &[]);
        // Variable division / modulo over a nonzero, non-overflowing domain.
        for (a, b) in [(20i64, 3i64), (-20, 3), (20, -3), (-20, -3), (7, 7), (0, 5)] {
            check("a / b", &[("a", Bind::Int(a)), ("b", Bind::Int(b))]);
            check("a % b", &[("a", Bind::Int(a)), ("b", Bind::Int(b))]);
        }
        // Nested (cometkim's `nested_expr`) on divisor-nonzero inputs.
        check(
            "((a + b) * (c - d)) / ((e + f) - (g * h))",
            &[
                ("a", Bind::Int(9)), ("b", Bind::Int(4)), ("c", Bind::Int(7)), ("d", Bind::Int(2)),
                ("e", Bind::Int(300)), ("f", Bind::Int(211)), ("g", Bind::Int(3)), ("h", Bind::Int(5)),
            ],
        );
    }

    #[test]
    fn comparisons_and_booleans() {
        let rows = [(0i64, 0i64), (1, 2), (2, 1), (-5, -5), (100, -100)];
        for (a, b) in rows {
            for expr in [
                "a >= b",
                "a > b",
                "a <= b",
                "a < b",
                "a == b",
                "a != b",
                "a > b || a == b",
                "a < b && b < 100",
            ] {
                check(expr, &[("a", Bind::Int(a)), ("b", Bind::Int(b))]);
            }
        }
    }

    #[test]
    fn conditional_ternary() {
        for x in [15i64, 5, 10, 11, -3, 0] {
            check("x > 10 ? x * 2 : x + 5", &[("x", Bind::Int(x))]);
        }
    }

    #[test]
    fn list_index_constant() {
        let program = Program::compile("list[0] + list[2] + list[4]").unwrap();
        let lowered = lower(program.expression()).expect("constant list index is lowerable");
        let paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(paths, ["list[0]", "list[2]", "list[4]"]);

        let mut ctx = Context::default();
        ctx.add_variable_from_value("list", vec![10i64, 20, 30, 40, 50]);
        let cel = match program.execute(&ctx).unwrap() {
            Value::Int(i) => i,
            o => panic!("unexpected {o:?}"),
        };
        let inputs = vec![10i64, 30, 50]; // list[0], list[2], list[4]
        let prog = lowered.program_for(&inputs);
        assert_eq!(clean_interp(&prog, lowered.num_regs), cel);
        assert_eq!(run_jit(&prog, lowered.num_regs, u32::MAX), cel);
    }

    #[test]
    fn select_chain_slots() {
        // Member-access policy lowers; slots resolve to the dotted paths in
        // first-encounter order (no execute — map construction is covered by
        // M3's batch harness).
        let program = Program::compile("account.balance >= txn.amount && !account.frozen").unwrap();
        let lowered = lower(program.expression()).expect("member-access policy is lowerable");
        let paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(paths, ["account.balance", "txn.amount", "account.frozen"]);
    }

    #[test]
    fn comprehension_all_exists() {
        // Literal-list `all`/`exists` fold to a bool over green-constant length.
        check("[1, 2, 3, 4, 5].all(x, x > 0)", &[]);
        check("[1, -2, 3].all(x, x > 0)", &[]);
        check("[1, 2, 3].exists(x, x > 2)", &[]);
        check("[1, 2, 3].exists(x, x > 9)", &[]);
        check("[].all(x, x > 0)", &[]);
        check("[1, 2, 3].exists_one(x, x > 2)", &[]);
        check("[1, 2, 3].exists_one(x, x > 0)", &[]);
        // Predicate referencing an outer slot alongside the iter var.
        check("[1, 2, 3].all(x, x < n)", &[("n", Bind::Int(5))]);
        check("[1, 2, 3].all(x, x < n)", &[("n", Bind::Int(2))]);
    }

    /// Cross-check the columnar batch evaluator. `eval_batch_sum` on both the
    /// interpreter tier (jit-off) and the compiled tier (jit-on) must equal the
    /// sum of the stock tree-walker's per-row result, and the jit-on run must
    /// actually compile the hot loop. `slot_paths` pins the lowering's slot
    /// order; `rows[i][k]` is slot `k`'s value in row `i` (int/bool as `i64`).
    /// This exercises the real throughput path — each column is read at the red
    /// row index via `raw_load`, not baked as a per-row constant.
    fn check_batch(expr_src: &str, slot_paths: &[&str], bool_slots: &[bool], rows: &[Vec<i64>]) {
        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let lowered =
            lower(program.expression()).unwrap_or_else(|e| panic!("lower `{expr_src}`: {e}"));
        let paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(paths, slot_paths, "slot order for `{expr_src}`");

        // Oracle: sum the stock tree-walker's per-row result.
        let mut expected = 0i64;
        for row in rows {
            let mut ctx = Context::default();
            for ((name, &v), &is_bool) in slot_paths.iter().zip(row).zip(bool_slots) {
                if is_bool {
                    ctx.add_variable_from_value(*name, v != 0);
                } else {
                    ctx.add_variable_from_value(*name, v);
                }
            }
            expected += match program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"))
            {
                Value::Bool(b) => b as i64,
                Value::Int(i) => i,
                other => panic!("`{expr_src}`: unexpected {other:?}"),
            };
        }

        // Transpose rows into per-slot i64 columns.
        let columns: Vec<Vec<i64>> = (0..slot_paths.len())
            .map(|k| rows.iter().map(|r| r[k]).collect())
            .collect();
        let col_refs: Vec<&[i64]> = columns.iter().map(|c| c.as_slice()).collect();

        let off = eval_batch_sum(&lowered, &col_refs, u32::MAX);
        assert_eq!(off, expected, "batch jit-off vs stock for `{expr_src}`");
        COMPILES.store(0, Ordering::Relaxed);
        let on = eval_batch_sum(&lowered, &col_refs, 8);
        assert_eq!(on, expected, "batch jit-on vs stock for `{expr_src}`");
        assert!(
            COMPILES.load(Ordering::Relaxed) >= 1,
            "batch `{expr_src}` must compile the hot loop"
        );
    }

    /// Deterministic per-row column data: an LCG mapped into `[lo, hi]` per slot.
    fn gen_rows(n: usize, ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
        let mut x: u64 = 0x2545F4914F6CDD1D;
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(ranges.len());
            for &(lo, hi) in ranges {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let span = (hi - lo + 1) as u64;
                row.push(lo + ((x >> 33) % span) as i64);
            }
            rows.push(row);
        }
        rows
    }

    #[test]
    fn batch_policy_count() {
        // Count rows where `a >= b` over real i64 columns read at the red index.
        let rows = gen_rows(3000, &[(-50, 50), (-50, 50)]);
        check_batch("a >= b", &["a", "b"], &[false, false], &rows);
    }

    #[test]
    fn batch_arithmetic_sum() {
        // Sum `(a + b) * c - d` over columns (small ranges keep it overflow-free
        // so debug-checked arithmetic and the compiled trace agree).
        let rows = gen_rows(3000, &[(0, 40), (0, 40), (-20, 20), (0, 100)]);
        check_batch(
            "(a + b) * c - d",
            &["a", "b", "c", "d"],
            &[false, false, false, false],
            &rows,
        );
    }

    #[test]
    fn batch_conditional_sum() {
        // Sum the ternary `x > 10 ? x * 2 : x + 5` over a single column.
        let rows = gen_rows(3000, &[(-5, 25)]);
        check_batch("x > 10 ? x * 2 : x + 5", &["x"], &[false], &rows);
    }

    #[test]
    fn batch_bool_slot_policy() {
        // A bool-typed slot column (`frozen`): `a >= b && !frozen`.
        let rows = gen_rows(3000, &[(-30, 30), (-30, 30), (0, 1)]);
        check_batch(
            "a >= b && !frozen",
            &["a", "b", "frozen"],
            &[false, false, true],
            &rows,
        );
    }

    #[test]
    fn out_of_subset_bails() {
        // list-returning / string / double / member-fn / list-valued
        // comprehension (`map` builds a list) all fall back to the tree-walker.
        for expr in [
            "[1, 2, 3]",
            "'a' + 'b'",
            "x.size()",
            "1.5 + a",
            "[1, 2, 3].map(x, x * 2)",
            "[1, 2, 3].filter(x, x > 1)",
            "x.all(x, x > 0)",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower(program.expression()).is_err(),
                "`{expr}` must be rejected as out of subset"
            );
        }
    }
}
