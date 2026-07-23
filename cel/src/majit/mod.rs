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
//! ## M3 — batch evaluation + green-length comprehension unroll
//!
//! [`bytecode::eval_batch_sum`] wraps the lowered body in a batch-over-rows loop
//! (the majit merge point), reading each context column at the red row index via
//! a compiled `raw_load` (the buffer bases held loop-invariant in the register
//! file). Green-length comprehensions unroll into the straight-line fold. The
//! flagship int policy `balance >= amount && !frozen` runs ~119x over the stock
//! tree-walker (see `examples/majit_columnar_batch`).
//!
//! ## M4 — `double` columns (the two-bank machine)
//!
//! [`lower::lower_typed`] lowers the same subset under a [`lower::Schema`]
//! declaring which paths are `double`, allocating float slots/temps in a
//! parallel `fregs` bank ([`bytecode::float_bank`]). A float comparison crosses
//! banks (`f64` operands, an int `0`/`1` result). Loop-invariant literal loads
//! are hoisted to a prelude that runs once. [`bytecode::eval_batch_sum_f`] is the
//! float batch path; the flagship float policy `price >= 100.0 && qty < 50.0`
//! runs ~94x over the tree-walker, bit-exact (see
//! `examples/majit_columnar_batch_float`).

pub mod bytecode;
pub mod lower;
pub mod smoke;

#[cfg(test)]
mod tests {
    use super::bytecode::{clean_interp, eval_batch_sum, eval_batch_sum_f, run_jit, Column, COMPILES};
    use super::lower::{lower, lower_typed, Schema, ValType};
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
        // Assert the compile counter *increased* across the jit-on run rather
        // than resetting it to 0 first: the counter is a shared global, so a
        // concurrent batch test's reset could otherwise mask a real compile.
        let before = COMPILES.load(Ordering::Relaxed);
        let on = eval_batch_sum(&lowered, &col_refs, 8);
        assert_eq!(on, expected, "batch jit-on vs stock for `{expr_src}`");
        assert!(
            COMPILES.load(Ordering::Relaxed) > before,
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

    /// Deterministic per-slot f64 column data in roughly [-1, 1].
    fn gen_float_cols(n: usize, seed: u64) -> Vec<f64> {
        let mut x = seed;
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mant = x & ((1u64 << 52) - 1);
            let v = f64::from_bits((0x3ffu64 << 52) | mant) - 1.0 + (k as f64 * 1e-12);
            out.push(if k & 1 == 0 { v } else { -v });
        }
        out
    }

    /// The two-bank float VM (`float_bank`) evaluated on a hand-built float
    /// policy: count rows where `a[i] >= b[i]` reading real `f64` columns at
    /// the red index. Oracle is the stock tree-walker's float comparison. This
    /// pins the float path (float column load + a bank-crossing float compare)
    /// before the lowerer emits it. clean == jit-off == jit-on, jit-on compiles.
    #[test]
    fn float_vm_count_ge() {
        use super::bytecode::float_bank::{clean_interp_f, run_jit_f, COMPILES as COMPILES_F};
        use super::bytecode::{
            OP_ADD, OP_COL_LOAD_F, OP_FGE, OP_JUMP_IF_ABOVE, OP_LOAD_CONST, OP_MUL, OP_RETURN,
        };

        let n = 3000usize;
        let cola = gen_float_cols(n, 0x2545_F491_4F6C_DD1D);
        let colb = gen_float_cols(n, 0x9E37_79B9_7F4A_7C15);

        // Oracle: the stock tree-walker's `a >= b` on f64 vars, summed.
        let program = Program::compile("a >= b").unwrap();
        let mut expected = 0i64;
        for i in 0..n {
            let mut ctx = Context::default();
            ctx.add_variable_from_value("a", cola[i]);
            ctx.add_variable_from_value("b", colb[i]);
            match program.execute(&ctx).unwrap() {
                Value::Bool(b) => expected += b as i64,
                other => panic!("unexpected {other:?}"),
            }
        }

        // int regs: i=0 acc=1 n=2 one=3 stride=4 ea=5 base_a=6 base_b=7 bool=8
        // float regs: fa=0 fb=1
        let base_a = cola.as_ptr() as i64;
        let base_b = colb.as_ptr() as i64;
        let mut prog: Vec<i64> = vec![
            OP_LOAD_CONST, 0, 0,
            OP_LOAD_CONST, 0, 1,
            OP_LOAD_CONST, n as i64, 2,
            OP_LOAD_CONST, 1, 3,
            OP_LOAD_CONST, 8, 4,
            OP_LOAD_CONST, base_a, 6,
            OP_LOAD_CONST, base_b, 7,
        ];
        let body_pc = prog.len() as i64;
        assert_eq!(body_pc, 21);
        prog.extend_from_slice(&[
            OP_MUL, 0, 4, 5,
            OP_COL_LOAD_F, 6, 5, 0,
            OP_COL_LOAD_F, 7, 5, 1,
            OP_FGE, 0, 1, 8,
            OP_ADD, 1, 8, 1,
            OP_ADD, 0, 3, 0,
            OP_JUMP_IF_ABOVE, 2, 0, body_pc,
            OP_RETURN, 1,
        ]);

        let (ni, nf) = (9usize, 2usize);
        assert_eq!(clean_interp_f(&prog, ni, nf), expected, "clean vs oracle");
        assert_eq!(run_jit_f(&prog, ni, nf, u32::MAX), expected, "jit-off vs oracle");
        let before = COMPILES_F.load(Ordering::Relaxed);
        assert_eq!(run_jit_f(&prog, ni, nf, 8), expected, "jit-on vs oracle");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "float batch must compile the hot loop"
        );
        core::hint::black_box((&cola, &colb));
    }

    /// One input column for a typed-lowering batch test: an int/bool column or
    /// a `double` column. Owns its data so the test keeps the buffers alive.
    enum ColData {
        Int(Vec<i64>),
        Float(Vec<f64>),
    }

    impl ColData {
        fn len(&self) -> usize {
            match self {
                ColData::Int(c) => c.len(),
                ColData::Float(c) => c.len(),
            }
        }
        fn ty(&self) -> ValType {
            match self {
                ColData::Int(_) => ValType::Int,
                ColData::Float(_) => ValType::Float,
            }
        }
        fn column(&self) -> Column<'_> {
            match self {
                ColData::Int(c) => Column::Int(c),
                ColData::Float(c) => Column::Float(c),
            }
        }
    }

    /// Deterministic per-column f64 data in `[lo, hi)` from an LCG, exact f64
    /// (built via `from_bits`) so the tree-walker oracle sees the same bits.
    fn gen_f64(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
        let mut x = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mant = x & ((1u64 << 52) - 1);
            let u = f64::from_bits((0x3ffu64 << 52) | mant) - 1.0; // [0, 1)
            out.push(lo + u * (hi - lo));
        }
        out
    }

    /// Deterministic per-column i64 data in `[lo, hi]` from an LCG.
    fn gen_i64(n: usize, seed: u64, lo: i64, hi: i64) -> Vec<i64> {
        let mut x = seed;
        let span = (hi - lo + 1) as u64;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                lo + ((x >> 33) % span) as i64
            })
            .collect()
    }

    /// Cross-check the typed (two-bank) columnar batch evaluator. The schema is
    /// read off `cols` (int vs `double`), which also pins the lowering's slot
    /// order. The clean two-bank interpreter, the majit interpreter tier, and
    /// the compiled tier must all equal the stock tree-walker's per-row sum, and
    /// the compiled run must actually trace the hot loop.
    fn check_batch_f(expr_src: &str, cols: &[(&str, ColData)]) {
        use super::bytecode::float_bank::{clean_interp_f, COMPILES as COMPILES_F};

        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let schema: Schema = cols.iter().map(|(n, d)| (n.to_string(), d.ty())).collect();
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("lower_typed `{expr_src}`: {e}"));

        // Slot order + bank pin.
        let paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        let want_paths: Vec<&str> = cols.iter().map(|(n, _)| *n).collect();
        assert_eq!(paths, want_paths, "slot order for `{expr_src}`");
        for (slot, (_, d)) in lowered.slots.iter().zip(cols) {
            assert_eq!(slot.ty, d.ty(), "slot `{}` bank for `{expr_src}`", slot.path);
        }

        let n = cols.first().map_or(0, |(_, d)| d.len());
        for (name, d) in cols {
            assert_eq!(d.len(), n, "column `{name}` length for `{expr_src}`");
        }

        // Oracle: sum the stock tree-walker's per-row result.
        let mut expected = 0i64;
        for i in 0..n {
            let mut ctx = Context::default();
            for (name, d) in cols {
                match d {
                    ColData::Int(c) => ctx.add_variable_from_value(*name, c[i]),
                    ColData::Float(c) => ctx.add_variable_from_value(*name, c[i]),
                }
            }
            expected += match program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"))
            {
                Value::Bool(b) => b as i64,
                Value::Int(v) => v,
                other => panic!("`{expr_src}`: unexpected {other:?}"),
            };
        }

        let columns: Vec<Column> = cols.iter().map(|(_, d)| d.column()).collect();

        // Clean two-bank interpreter over the built batch program.
        let bases: Vec<i64> = columns.iter().map(|c| c.base()).collect();
        let (prog, ni, nf) = lowered.batch_sum_program(&bases, n as i64);
        assert_eq!(clean_interp_f(&prog, ni, nf), expected, "clean vs stock for `{expr_src}`");
        core::hint::black_box(&columns);

        // majit interpreter tier, then compiled tier. The compile counter is a
        // shared, monotonic global; asserting it *increased* across the jit-on
        // run (rather than resetting it to 0 first) is robust to other float
        // tests compiling concurrently.
        let off = eval_batch_sum_f(&lowered, &columns, u32::MAX);
        assert_eq!(off, expected, "batch jit-off vs stock for `{expr_src}`");
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_f(&lowered, &columns, 8);
        assert_eq!(on, expected, "batch jit-on vs stock for `{expr_src}`");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "float batch `{expr_src}` must compile the hot loop"
        );
    }

    /// Cross-check a **float-valued** typed batch (a float aggregate). The oracle
    /// sums the tree-walker's per-row `f64` in row order; the clean two-bank
    /// interpreter, the majit interpreter tier, and the compiled tier must each
    /// reproduce that sum bit for bit (float addition is order-sensitive, so
    /// compare bits, never a tolerance). The compiled run must trace the loop.
    fn check_batch_float(expr_src: &str, cols: &[(&str, ColData)]) {
        use super::bytecode::eval_batch_sum_float;
        use super::bytecode::float_bank::{clean_interp_f, COMPILES as COMPILES_F};

        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let schema: Schema = cols.iter().map(|(n, d)| (n.to_string(), d.ty())).collect();
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("lower_typed `{expr_src}`: {e}"));
        assert_eq!(
            lowered.result_bank,
            ValType::Float,
            "`{expr_src}` must lower to a float result"
        );

        // Slot order + bank pin.
        let paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        let want_paths: Vec<&str> = cols.iter().map(|(n, _)| *n).collect();
        assert_eq!(paths, want_paths, "slot order for `{expr_src}`");
        for (slot, (_, d)) in lowered.slots.iter().zip(cols) {
            assert_eq!(slot.ty, d.ty(), "slot `{}` bank for `{expr_src}`", slot.path);
        }

        let n = cols.first().map_or(0, |(_, d)| d.len());
        for (name, d) in cols {
            assert_eq!(d.len(), n, "column `{name}` length for `{expr_src}`");
        }

        // Oracle: sum the stock tree-walker's per-row f64 in row order.
        let mut expected = 0.0f64;
        for i in 0..n {
            let mut ctx = Context::default();
            for (name, d) in cols {
                match d {
                    ColData::Int(c) => ctx.add_variable_from_value(*name, c[i]),
                    ColData::Float(c) => ctx.add_variable_from_value(*name, c[i]),
                }
            }
            expected += match program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"))
            {
                Value::Float(v) => v,
                other => panic!("`{expr_src}`: unexpected {other:?}"),
            };
        }

        let columns: Vec<Column> = cols.iter().map(|(_, d)| d.column()).collect();

        // Clean two-bank interpreter over the built batch program.
        let bases: Vec<i64> = columns.iter().map(|c| c.base()).collect();
        let (prog, ni, nf) = lowered.batch_sum_program(&bases, n as i64);
        let clean = f64::from_bits(clean_interp_f(&prog, ni, nf) as u64);
        assert_eq!(clean.to_bits(), expected.to_bits(), "clean vs stock for `{expr_src}`");
        core::hint::black_box(&columns);

        // majit interpreter tier, then compiled tier (monotonic compile-counter).
        let off = eval_batch_sum_float(&lowered, &columns, u32::MAX);
        assert_eq!(off.to_bits(), expected.to_bits(), "batch jit-off vs stock for `{expr_src}`");
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_float(&lowered, &columns, 8);
        assert_eq!(on.to_bits(), expected.to_bits(), "batch jit-on vs stock for `{expr_src}`");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "float aggregate `{expr_src}` must compile the hot loop"
        );
    }

    #[test]
    fn batch_float_aggregate() {
        // Float-valued top-level result -> float accumulator (OP_RETURN_F). The
        // running total sums the per-row f64 in row order, bit-exact across the
        // clean/interp/compiled tiers.
        let n = 3000;
        let price = gen_f64(n, 0x0FED_CBA9_8765_4321, 0.0, 100.0);
        let qty = gen_f64(n, 0x1357_9BDF_2468_ACE0, 0.0, 50.0);
        // sum(price * qty)
        check_batch_float(
            "price * qty",
            &[
                ("price", ColData::Float(price.clone())),
                ("qty", ColData::Float(qty.clone())),
            ],
        );
        // sum(price * qty + price) — two float ops feeding the accumulator
        check_batch_float(
            "price * qty + price",
            &[("price", ColData::Float(price)), ("qty", ColData::Float(qty))],
        );
        // sum(price * 2.0) — a hoisted float constant inside a float aggregate
        let p2 = gen_f64(n, 0x2468_ACE0_1357_9BDF, -50.0, 50.0);
        check_batch_float("price * 2.0", &[("price", ColData::Float(p2))]);
    }

    #[test]
    fn batch_float_policy_count() {
        // Flagship float policy: count rows where a float column clears a float
        // constant threshold AND another stays under a float limit. Float column
        // loads + float-vs-const compares -> int bools -> int AND -> count.
        let n = 3000;
        let price = gen_f64(n, 0x2545_F491_4F6C_DD1D, 0.0, 200.0);
        let qty = gen_f64(n, 0x9E37_79B9_7F4A_7C15, 0.0, 100.0);
        check_batch_f(
            "price >= 100.0 && qty < 50.0",
            &[("price", ColData::Float(price)), ("qty", ColData::Float(qty))],
        );
    }

    #[test]
    fn batch_float_col_vs_col() {
        // Float column vs float column comparison driven through the lowerer.
        let n = 3000;
        let a = gen_f64(n, 0xAAAA_5555_AAAA_5555, -1.0, 1.0);
        let b = gen_f64(n, 0xBBBB_4444_BBBB_4444, -1.0, 1.0);
        check_batch_f("a >= b", &[("a", ColData::Float(a)), ("b", ColData::Float(b))]);
    }

    #[test]
    fn batch_float_arith_policy() {
        // Float arithmetic (FMUL) then a float-const compare.
        let n = 3000;
        let price = gen_f64(n, 0x1111_2222_3333_4444, 0.0, 100.0);
        let qty = gen_f64(n, 0x5555_6666_7777_8888, 0.0, 100.0);
        check_batch_f(
            "price * qty >= 2500.0",
            &[("price", ColData::Float(price)), ("qty", ColData::Float(qty))],
        );
    }

    #[test]
    fn batch_mixed_bank_policy() {
        // Both banks in one policy: an int/bool column AND a float-const compare.
        // Exercises OP_COL_LOAD (int) and OP_COL_LOAD_F (float) side by side.
        let n = 3000;
        let flagged = gen_i64(n, 0xCAFE_F00D_CAFE_F00D, 0, 1);
        let price = gen_f64(n, 0xF00D_CAFE_F00D_CAFE, 0.0, 200.0);
        check_batch_f(
            "flagged >= 1 && price >= 100.0",
            &[("flagged", ColData::Int(flagged)), ("price", ColData::Float(price))],
        );
    }

    #[test]
    fn probe_float_const_only() {
        // Isolates a float constant (OP_LOAD_CONST_F), no AND.
        let n = 3000;
        let a = gen_f64(n, 0x1234_5678_9ABC_DEF0, -1.0, 1.0);
        check_batch_f("a >= 0.0", &[("a", ColData::Float(a))]);
    }

    #[test]
    fn probe_and_no_const() {
        // Isolates AND over float-derived bools, no float constant.
        let n = 3000;
        let a = gen_f64(n, 0x1111_1111_1111_1111, -1.0, 1.0);
        let b = gen_f64(n, 0x2222_2222_2222_2222, -1.0, 1.0);
        check_batch_f(
            "a >= b && b >= a",
            &[("a", ColData::Float(a)), ("b", ColData::Float(b))],
        );
    }

    #[test]
    fn typed_lowering_bails() {
        // Float modulo, mixed int/float arithmetic, and a float ternary arm
        // still bail. (A float-valued top-level result now compiles into a float
        // accumulator — see `batch_float_aggregate`; a mixed int/float
        // comparison widens via cast_int_to_float — see `batch_mixed_col_compare`.)
        let schema: Schema = [("p".to_string(), ValType::Float), ("q".to_string(), ValType::Int)]
            .into_iter()
            .collect();
        for expr in ["p % 2.0 >= 1.0", "p + q", "p >= 1.0 ? p : p"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering"
            );
        }
    }

    #[test]
    fn batch_mixed_col_compare() {
        // A float column vs an int column: the int side is widened per row via
        // cast_int_to_float, matching the tree-walker's `int as f64`. Both
        // operand orders (int column on either side).
        let n = 3000;
        let price = gen_f64(n, 0x2020_2020_2020_2020, 0.0, 10.0);
        let level = gen_i64(n, 0x3030_3030_3030_3030, 0, 10);
        check_batch_f(
            "price >= level",
            &[("price", ColData::Float(price)), ("level", ColData::Int(level))],
        );
        let price2 = gen_f64(n, 0x4040_4040_4040_4040, 0.0, 10.0);
        let level2 = gen_i64(n, 0x5050_5050_5050_5050, 0, 10);
        check_batch_f(
            "level < price",
            &[("level", ColData::Int(level2)), ("price", ColData::Float(price2))],
        );
    }

    #[test]
    fn batch_mixed_literal_compare() {
        // An int literal compared to a float column is promoted to a double
        // constant (`int as f64`), matching the tree-walker. Both orders.
        let n = 3000;
        let price = gen_f64(n, 0x0F0F_0F0F_0F0F_0F0F, 0.0, 200.0);
        let qty = gen_f64(n, 0xF0F0_F0F0_F0F0_F0F0, 0.0, 100.0);
        check_batch_f(
            "price >= 100 && qty < 50",
            &[("price", ColData::Float(price.clone())), ("qty", ColData::Float(qty.clone()))],
        );
        check_batch_f(
            "100 <= price && 50 > qty",
            &[("price", ColData::Float(price)), ("qty", ColData::Float(qty))],
        );
    }

    #[test]
    fn batch_mixed_literal_equality() {
        // int-literal equality against a float column (whole-valued rows so the
        // predicate actually fires), promoted to a double constant.
        let n = 3000;
        let score = gen_i64(n, 0x1357_9BDF_2468_ACE0, 0, 5)
            .into_iter()
            .map(|v| v as f64)
            .collect::<Vec<f64>>();
        check_batch_f("score == 3", &[("score", ColData::Float(score))]);
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
