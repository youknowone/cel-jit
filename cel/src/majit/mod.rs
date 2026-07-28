//! Proof-of-concept meta-tracing JIT tier for cel-rust, built on the in-repo
//! `majit` framework (a Rust port of RPython's tracing JIT). Tracked as
//! cell-majit (issue #357).
//!
//! majit does not attach to cel-rust's live `Value::resolve_val` evaluator:
//! that path returns `Cow<'a, dyn Val>` and dispatches through trait objects /
//! `downcast_ref`, which is outside the restricted Rust subset the
//! `#[jit_interp]` front-end can meta-trace. (majit's other front-end,
//! `majit-translate`, traces real Rust from LLBC and does handle class
//! dispatch; `CONVERGENCE.md` in this directory is the plan for getting cel
//! onto it.) Instead, a CEL `Program` (a fixed AST = green constant) is
//! lowered to a flat `i64`-word bytecode, and a small mainloop authored in the
//! traceable subset evaluates it over a batch of inputs. The de-risk phase
//! established the perf envelope (scalar arith 9-16x, slot-resolved policy
//! predicate ~6x over a clean interpreter, comprehensions 2-4x only when the
//! list length is a green constant and can be unrolled).
//!
//! ## M1 — the framework links and compiles a loop
//!
//! That the `majit` crates build as cross-workspace path deps of `cel`, that
//! the cranelift backend links here, and that a hot loop actually traces and
//! compiles were first shown on a self-contained register machine cloned from
//! `majit/examples/tinyframe`. That second mainloop is gone: the same
//! properties are now asserted on the real mainloop over real CEL in
//! `tests/majit_trace_evidence.rs`, which also pins that the compiled trace
//! RUNS the loop rather than deopting per iteration.
//!
//! ## M2 — CEL AST -> traceable bytecode ([`lower`], [`bytecode`])
//!
//! [`lower::lower_typed`] compiles the supported `Expr` subset (`Int`/`Boolean`
//! literals, slot-resolved `Ident`/`Select`, arithmetic `+ - * / %`, unary
//! `-`, comparisons, boolean `&& || !`) to the flat `i64`-word program of
//! [`bytecode`]. Anything outside the subset returns [`lower::LowerError`], the
//! signal to fall back to the stock tree-walking evaluator. Correctness is
//! pinned by cross-checking the lowered program against the real
//! `Program::execute` on the same inputs (see the tests below).
//!
//! ## M3 — batch evaluation + green-length comprehension unroll
//!
//! [`bytecode::eval_batch_sum_f`] wraps the lowered body in a batch-over-rows loop
//! (the majit merge point), reading each context column at the red row index via
//! a compiled `raw_load` (the buffer bases held loop-invariant in the register
//! file). Green-length comprehensions unroll into the straight-line fold. The
//! The columnar path can run far faster end-to-end than the stock tree-walker,
//! but that is a cross-model batch result, not a JIT-only or request-latency
//! multiplier. `examples/majit_ab` is the default fair suite: it keeps real CEL
//! request latency, same-bytecode JIT throughput, and cold/break-even results in
//! separate panels.
//!
//! ## M4 — `double` columns (the two-bank machine)
//!
//! [`lower::lower_typed`] lowers the same subset under a [`lower::Schema`]
//! declaring which paths are `double`, allocating float slots/temps in a
//! parallel `fregs` bank ([`bytecode::float_bank`]). A float comparison crosses
//! banks (`f64` operands, an int `0`/`1` result). Loop-invariant literal loads
//! are hoisted to a prelude that runs once. [`bytecode::eval_batch_sum_f`] is the
//! float batch path. The experimental float example reports its cross-model
//! batch ratio and same-bytecode JIT-only ratio separately, bit-exact (see
//! `examples/majit_columnar_batch_float`).
//!
//! ## M5 — filling out the numeric columnar subset
//!
//! The two-bank machine is extended to the rest of the winnable numeric domain,
//! each addition cross-checked bit-exact across the clean / interp / compiled
//! tiers:
//!   * **int↔float compares** widen the int side per row (`cast_int_to_float`),
//!     matching the tree-walker's `int as f64` promotion.
//!   * **float aggregates** — a float-valued top-level result sums into a float
//!     accumulator (`OP_RETURN_F`); the running total is a loop-carried
//!     dependency, so the compiled trace sums in row order, bit for bit.
//!   * **float ternary** `c ? t : f` blends the two arms over their raw `f64`
//!     bit patterns (`OP_FSELECT`, a mask select), overflow-free and without the
//!     reassociation an arithmetic blend would need.
//!   * **uint columns** share the int register file (the raw 64-bit pattern):
//!     add/sub/mul and eq/ne reuse the int ops, ordering compares unsigned
//!     (`OP_ULT`/`OP_ULE`, `>`/`>=` via an operand swap). Division/modulo go
//!     through the `int.udiv`/`int.umod` oopspec residual calls, which is where
//!     upstream put them when it deleted its unsigned division resops.
//!
//! ## M6 — runtime-length lists (a nested red loop)
//!
//! A comprehension over a list column whose length is a per-row value has no
//! green trip count, so it cannot unroll — and unrolling is not what upstream
//! does either: PyPy only unrolls a `jit.isconstant` length
//! (`rlib/jit.py`'s `loop_unrolling_heuristic`) and otherwise just traces the
//! loop. So the lowering emits a real inner loop whose back-edge is its own
//! `can_enter_jit` point; the pc-green mainloop then gives the element loop a
//! trace identity separate from the row loop's.
//!
//! The list is stored the columnar (Arrow) way rather than as a value: one
//! flattened element column per field read (`items[]`, `items[].price`), laid
//! end to end across the batch, plus two derived per-row columns —
//! `size(items)` and `offset(items)`. Locating a row's elements is then
//! arithmetic, and the element load is a `raw_load` at `(offset + j) * 8`
//! exactly as a row load is one at `row * 8`. Field access on the loop variable
//! resolves to those element columns, which is what the literal-list unroll
//! could never do. An element column's length is the batch's flattened element
//! count, not the row count, so the row count is a parameter of
//! [`bytecode::eval_batch_sum_f`] rather than something read off a column.
//!
//! Everything outside this columnar subset — bytes, maps, lists of lists,
//! list-valued results, member/method calls, `in`, custom functions — is a
//! structural loss for a batch JIT and returns [`lower::LowerError`], falling
//! back to the stock tree-walker. The win is confined to what a compiled trace
//! over aligned columns can express.

pub mod batch;
pub mod bytecode;
pub mod lower;

#[cfg(test)]
mod tests {
    use super::bytecode::{clean_batch_sum_f, eval_batch_sum_f, Column};
    use super::lower::{lower_typed, size_slot_source, Schema, ValType};
    use crate::{Context, Program, Value};
    use core::sync::atomic::Ordering;
    use std::collections::HashMap;

    #[derive(Debug, Clone, Copy)]
    enum Bind {
        Int(i64),
        Bool(bool),
    }

    impl Bind {
        fn ty(self) -> ValType {
            match self {
                Bind::Int(_) => ValType::Int,
                Bind::Bool(_) => ValType::Bool,
            }
        }

        fn as_i64(self) -> i64 {
            match self {
                Bind::Int(v) => v,
                Bind::Bool(b) => b as i64,
            }
        }
    }

    /// Cross-check ONE row: a lowered CEL expression run on both the clean
    /// two-bank interpreter and the majit mainloop with compilation disabled
    /// (this validates the LOWERING, not the trace) yields the same scalar the
    /// stock `Program::execute` tree-walker does for the same bindings. A
    /// one-row batch is the single-row program on this machine — the loop runs
    /// once and the accumulator holds the row's value.
    fn check(expr_src: &str, binds: &[(&str, Bind)]) {
        let cols: Vec<(&str, ColData)> = binds
            .iter()
            .map(|(n, b)| (*n, ColData::Int(vec![b.as_i64()])))
            .collect();
        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let schema: Schema = binds.iter().map(|(n, b)| (n.to_string(), b.ty())).collect();
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("lower_typed `{expr_src}`: {e}"));

        // Stock tree-walker reference. A `Bind::Bool` must bind a real bool, not
        // its 0/1 image, or the walker would compare an int where CEL sees a
        // bool and the oracle would stop being one.
        let mut ctx = Context::default();
        for (name, b) in binds {
            match b {
                Bind::Int(v) => ctx.add_variable_from_value(*name, *v),
                Bind::Bool(v) => ctx.add_variable_from_value(*name, *v),
            }
        }
        let cel_i = match program
            .execute(&ctx)
            .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"))
        {
            Value::Bool(b) => b as i64,
            Value::Int(i) => i,
            other => panic!("`{expr_src}`: unexpected result {other:?}"),
        };

        let columns: Vec<Column> = cols.iter().map(|(_, d)| d.column()).collect();
        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, 1),
            Some(cel_i),
            "clean interp vs stock for `{expr_src}` {binds:?}"
        );
        assert_eq!(
            eval_batch_sum_f(&lowered, &columns, 1, u32::MAX),
            Some(cel_i),
            "majit (jit-off) vs stock for `{expr_src}` {binds:?}"
        );
    }

    /// Regression: an overflow-checked op (`OP_ADD_OVF`) whose `GuardNoOverflow`
    /// fails inside a *compiled* trace must resume through the blackhole on the
    /// virtualizable `[int; virt]` regs. This mid-body guard is the first on
    /// this machine to land in vable-array resume territory; before the
    /// deopt-time vinfo seed + `token_offset==0` inert token-clear it panicked.
    /// The wrapped result must match the oracle — and the resume must RECORD the
    /// overflow in the trap register, which `OP_TRAP_STORE` then publishes: that
    /// write is what lets the batch driver refuse to answer where the
    /// tree-walker raises.
    #[test]
    fn overflow_deopt_on_compiled_trace() {
        use super::bytecode::float_bank::{clean_interp_f, run_jit_f, COMPILES as COMPILES_F};
        use super::bytecode::{
            OP_ADD, OP_ADD_OVF, OP_JUMP_IF_ABOVE, OP_LOAD_CONST, OP_RETURN, OP_TRAP_STORE,
        };
        // regs: i=0, n=1, acc=2, inc=3, one=4, trap_flag=5, trap_addr=6.
        // `inc = MAX/4` makes `acc` overflow a handful of iterations in — after
        // the threshold-3 loop has compiled, so the overflow guard fails in the
        // compiled trace.
        let n: i64 = 30;
        let inc: i64 = i64::MAX / 4;
        let mut trap: Box<i64> = Box::new(0);
        let trap_addr = (&mut *trap) as *mut i64 as i64;
        // One instruction per line: the operand grouping IS the program.
        #[rustfmt::skip]
        let prog: Vec<i64> = vec![
            OP_LOAD_CONST, 0, 0,
            OP_LOAD_CONST, n, 1,
            OP_LOAD_CONST, 0, 2,
            OP_LOAD_CONST, inc, 3,
            OP_LOAD_CONST, 1, 4,
            OP_LOAD_CONST, 0, 5,
            OP_LOAD_CONST, trap_addr, 6,
            // loop_start @ pc = 21
            OP_ADD_OVF, 2, 3, 2, 5,     // acc = ovfchecked(acc + inc), trap -> r5
            OP_ADD, 0, 4, 0,            // i = i + 1
            OP_JUMP_IF_ABOVE, 1, 0, 21, // while n > i
            OP_TRAP_STORE, 6, 5,        // *trap_addr = trap_flag
            OP_RETURN, 2,
        ];
        let before = COMPILES_F.load(Ordering::Relaxed);
        let jit = run_jit_f(&prog, 7, 0, 3);
        let jit_trap = *trap;
        *trap = 0;
        let clean = clean_interp_f(&prog, 7, 0);
        assert_eq!(
            jit, clean,
            "compiled-tier overflow deopt must match wrapping oracle"
        );
        assert_eq!(
            jit_trap, 1,
            "the compiled tier's overflow deopt must set the trap flag"
        );
        assert_eq!(*trap, 1, "the reference tier must set the trap flag too");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "loop must tier-compile so the overflow lands in the compiled trace",
        );
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
                &[
                    ("a", Bind::Int(a)),
                    ("b", Bind::Int(b)),
                    ("c", Bind::Bool(c)),
                ],
            );
        }
    }

    #[test]
    fn arithmetic() {
        check(
            "(a + b) * c - 2",
            &[
                ("a", Bind::Int(3)),
                ("b", Bind::Int(4)),
                ("c", Bind::Int(5)),
            ],
        );
        check(
            "a * b + c",
            &[
                ("a", Bind::Int(-6)),
                ("b", Bind::Int(7)),
                ("c", Bind::Int(11)),
            ],
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
                ("a", Bind::Int(9)),
                ("b", Bind::Int(4)),
                ("c", Bind::Int(7)),
                ("d", Bind::Int(2)),
                ("e", Bind::Int(300)),
                ("f", Bind::Int(211)),
                ("g", Bind::Int(3)),
                ("h", Bind::Int(5)),
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

    /// A caller may FLATTEN the indices it uses into columns of their own,
    /// declaring `list[k]` for each. That is a different convention from
    /// binding the list itself (see `constant_index_reads_the_rows_own_list`),
    /// and the schema is what picks between them: a declared `list[k]` is a row
    /// column and reads like any other, with no bounds check because the caller
    /// has already resolved the index.
    #[test]
    fn list_index_constant() {
        let program = Program::compile("list[0] + list[2] + list[4]").unwrap();
        let schema: Schema = [
            ("list[0]".to_string(), ValType::Int),
            ("list[2]".to_string(), ValType::Int),
            ("list[4]".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        let lowered =
            lower_typed(program.expression(), &schema).expect("constant list index is lowerable");
        let paths: Vec<&str> = lowered.slots.iter().map(|s| s.path.as_str()).collect();
        assert_eq!(paths, ["list[0]", "list[2]", "list[4]"]);

        let mut ctx = Context::default();
        ctx.add_variable_from_value("list", vec![10i64, 20, 30, 40, 50]);
        let cel = match program.execute(&ctx).unwrap() {
            Value::Int(i) => i,
            o => panic!("unexpected {o:?}"),
        };
        // One column per constant-index slot: list[0], list[2], list[4].
        let (c0, c2, c4) = (vec![10i64], vec![30i64], vec![50i64]);
        let columns = [Column::Int(&c0), Column::Int(&c2), Column::Int(&c4)];
        assert_eq!(clean_batch_sum_f(&lowered, &columns, 1), Some(cel));
        assert_eq!(eval_batch_sum_f(&lowered, &columns, 1, u32::MAX), Some(cel));
    }

    #[test]
    fn select_chain_slots() {
        // Member-access policy lowers; slots resolve to the dotted paths in
        // first-encounter order (no execute — map construction is covered by
        // M3's batch harness).
        let program = Program::compile("account.balance >= txn.amount && !account.frozen").unwrap();
        let schema: Schema = [
            ("account.balance".to_string(), ValType::Int),
            ("txn.amount".to_string(), ValType::Int),
            ("account.frozen".to_string(), ValType::Bool),
        ]
        .into_iter()
        .collect();
        let lowered =
            lower_typed(program.expression(), &schema).expect("member-access policy is lowerable");
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

    /// Cross-check the columnar batch evaluator over ROW-MAJOR test data:
    /// `slot_paths` pins the lowering's slot order and `rows[i][k]` is slot
    /// `k`'s value in row `i` (int/bool as `i64`). Transposes into columns and
    /// delegates to [`check_batch_f`], so it inherits the full three-tier
    /// bit-exact contract. `bool_slots` marks which columns the tree-walker must
    /// see as real `bool`s — on the machine they are `0`/`1` in the int bank
    /// either way.
    fn check_batch(expr_src: &str, slot_paths: &[&str], bool_slots: &[bool], rows: &[Vec<i64>]) {
        let cols: Vec<(&str, ColData)> = slot_paths
            .iter()
            .zip(bool_slots)
            .enumerate()
            .map(|(k, (name, &is_bool))| {
                let c: Vec<i64> = rows.iter().map(|r| r[k]).collect();
                (
                    *name,
                    if is_bool {
                        ColData::Bool(c)
                    } else {
                        ColData::Int(c)
                    },
                )
            })
            .collect();
        check_batch_f(expr_src, &cols);
    }

    /// Deterministic per-row column data: an LCG mapped into `[lo, hi]` per slot.
    fn gen_rows(n: usize, ranges: &[(i64, i64)]) -> Vec<Vec<i64>> {
        let mut x: u64 = 0x2545F4914F6CDD1D;
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(ranges.len());
            for &(lo, hi) in ranges {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
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
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        // One instruction per line: the operand grouping IS the program.
        #[rustfmt::skip]
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
        #[rustfmt::skip]
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
        assert_eq!(
            run_jit_f(&prog, ni, nf, u32::MAX),
            expected,
            "jit-off vs oracle"
        );
        let before = COMPILES_F.load(Ordering::Relaxed);
        assert_eq!(run_jit_f(&prog, ni, nf, 8), expected, "jit-on vs oracle");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "float batch must compile the hot loop"
        );
        core::hint::black_box((&cola, &colb));
    }

    /// One input column for a typed-lowering batch test: an int/bool column, a
    /// `uint` column (stored as its i64 bit pattern), or a `double` column. Owns
    /// its data so the test keeps the buffers alive.
    enum ColData {
        Int(Vec<i64>),
        /// A `bool` column, stored as `0`/`1`. Its STORAGE is identical to
        /// [`ColData::Int`] (bools live in the int bank as `0`/`1`), but it is a
        /// different declared type on both ends: the schema says
        /// [`ValType::Bool`], so `&&`/`!`/`?:` accept it and arithmetic does
        /// not, and the oracle binds a real `Value::Bool` rather than its `0`/`1`
        /// image.
        Bool(Vec<i64>),
        UInt(Vec<i64>),
        Float(Vec<f64>),
        /// A string column. Handed to the machine as strings; `prepare_batch`
        /// ranks the batch's distinct values into the `i64` ids it runs on.
        Str(Vec<String>),
        /// A timestamp column, as `i64` nanoseconds since the Unix epoch. Read
        /// directly as an int column (no interning); the oracle rebuilds a
        /// `Value::Timestamp` from each nanos value.
        Timestamp(Vec<i64>),
        /// A duration column, as `i64` nanoseconds. Read directly as an int
        /// column; the oracle rebuilds a `Value::Duration` from each nanos value.
        Duration(Vec<i64>),
    }

    impl ColData {
        fn len(&self) -> usize {
            match self {
                ColData::Int(c)
                | ColData::Bool(c)
                | ColData::UInt(c)
                | ColData::Timestamp(c)
                | ColData::Duration(c) => c.len(),
                ColData::Float(c) => c.len(),
                ColData::Str(c) => c.len(),
            }
        }
        fn ty(&self) -> ValType {
            match self {
                ColData::Int(_) => ValType::Int,
                ColData::Bool(_) => ValType::Bool,
                ColData::UInt(_) => ValType::UInt,
                ColData::Float(_) => ValType::Float,
                ColData::Str(_) => ValType::Str,
                ColData::Timestamp(_) => ValType::Timestamp,
                ColData::Duration(_) => ValType::Duration,
            }
        }
        fn column(&self) -> Column<'_> {
            match self {
                ColData::Int(c)
                | ColData::Bool(c)
                | ColData::UInt(c)
                | ColData::Timestamp(c)
                | ColData::Duration(c) => Column::Int(c),
                ColData::Float(c) => Column::Float(c),
                ColData::Str(c) => Column::Str(c),
            }
        }
    }

    /// Deterministic per-column f64 data in `[lo, hi)` from an LCG, exact f64
    /// (built via `from_bits`) so the tree-walker oracle sees the same bits.
    fn gen_f64(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
        let mut x = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                lo + ((x >> 33) % span) as i64
            })
            .collect()
    }

    /// Deterministic full-range u64 column data returned as its i64 bit pattern.
    /// Roughly half the values have the high bit set (u64 > i64::MAX), so a
    /// signed comparison would order them differently from the unsigned oracle —
    /// making the uint compare tests genuinely discriminating.
    fn gen_u64_bits(n: usize, seed: u64) -> Vec<i64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                x as i64
            })
            .collect()
    }

    /// The tree-walker's value for element `k` of a column — the oracle's view
    /// of one cell.
    fn cell_value(d: &ColData, k: usize) -> Value {
        match d {
            ColData::Int(c) => Value::Int(c[k]),
            ColData::Bool(c) => Value::Bool(c[k] != 0),
            ColData::UInt(c) => Value::UInt(c[k] as u64),
            ColData::Float(c) => Value::Float(c[k]),
            ColData::Str(c) => Value::String(std::sync::Arc::new(c[k].clone())),
            ColData::Timestamp(c) => {
                Value::Timestamp(chrono::DateTime::from_timestamp_nanos(c[k]).fixed_offset())
            }
            ColData::Duration(c) => Value::Duration(chrono::Duration::nanoseconds(c[k])),
        }
    }

    /// Bind row `i` of every column into a fresh tree-walker context — the
    /// oracle's view of one row.
    fn row_context<'a>(cols: &'a [(&str, ColData)], i: usize) -> Context<'a> {
        let mut ctx = Context::default();
        for (name, d) in cols {
            ctx.add_variable_from_value(*name, cell_value(d, i));
        }
        ctx
    }

    /// Cross-check that a typed batch REFUSES to answer — the other half of the
    /// [`check_batch_f`] contract. A refusal is only correct when the
    /// tree-walker itself raises, so that is asserted first; then all three
    /// tiers must return `None`, meaning the trap flag survived the loop and
    /// reached the driver. The compiled tier must still trace the loop:
    /// refusing is a guard exit taken on some row, not a failure to compile.
    fn check_batch_f_refuses(expr_src: &str, cols: &[(&str, ColData)]) {
        use super::bytecode::float_bank::COMPILES as COMPILES_F;

        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let schema: Schema = cols.iter().map(|(n, d)| (n.to_string(), d.ty())).collect();
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("lower_typed `{expr_src}`: {e}"));

        let n = cols.first().map_or(0, |(_, d)| d.len());
        assert!(
            (0..n).any(|i| program.execute(&row_context(cols, i)).is_err()),
            "`{expr_src}`: the tree-walker answers every row, so refusing would be wrong"
        );

        let columns: Vec<Column> = cols.iter().map(|(_, d)| d.column()).collect();
        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, n),
            None,
            "clean must refuse `{expr_src}`"
        );
        assert_eq!(
            eval_batch_sum_f(&lowered, &columns, n, u32::MAX),
            None,
            "batch jit-off must refuse `{expr_src}`"
        );
        // Start the compiled tier cold: the driver persists across calls, so a
        // loop an earlier case already compiled would not compile again.
        super::bytecode::float_bank::reset_persistent_state();
        let before = COMPILES_F.load(Ordering::Relaxed);
        assert_eq!(
            eval_batch_sum_f(&lowered, &columns, n, 8),
            None,
            "batch jit-on must refuse `{expr_src}`"
        );
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "float batch `{expr_src}` must still compile the hot loop"
        );
    }

    /// Cross-check the typed (two-bank) columnar batch evaluator. The schema is
    /// read off `cols` (int vs `double`), which also pins the lowering's slot
    /// order. The clean two-bank interpreter, the majit interpreter tier, and
    /// the compiled tier must all equal the stock tree-walker's per-row sum, and
    /// the compiled run must actually trace the hot loop.
    fn check_batch_f(expr_src: &str, cols: &[(&str, ColData)]) {
        use super::bytecode::float_bank::COMPILES as COMPILES_F;

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
            assert_eq!(
                slot.ty,
                d.ty(),
                "slot `{}` bank for `{expr_src}`",
                slot.path
            );
        }

        let n = cols.first().map_or(0, |(_, d)| d.len());
        for (name, d) in cols {
            assert_eq!(d.len(), n, "column `{name}` length for `{expr_src}`");
        }

        // Oracle: sum the stock tree-walker's per-row result. A `uint` result
        // rides the int accumulator as its raw bit pattern, exactly as the
        // machine's plain `OP_ADD` reduction does.
        let mut expected = 0i64;
        for i in 0..n {
            let ctx = row_context(cols, i);
            expected += match program
                .execute(&ctx)
                .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"))
            {
                Value::Bool(b) => b as i64,
                Value::Int(v) => v,
                Value::UInt(v) => v as i64,
                other => panic!("`{expr_src}`: unexpected {other:?}"),
            };
        }

        let columns: Vec<Column> = cols.iter().map(|(_, d)| d.column()).collect();

        // Clean two-bank interpreter over the built batch program.
        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, n),
            Some(expected),
            "clean vs stock for `{expr_src}`"
        );

        // majit interpreter tier, then compiled tier. The compile counter is a
        // shared, monotonic global; asserting it *increased* across the jit-on
        // run (rather than resetting it to 0 first) is robust to other float
        // tests compiling concurrently.
        let off = eval_batch_sum_f(&lowered, &columns, n, u32::MAX);
        assert_eq!(
            off,
            Some(expected),
            "batch jit-off vs stock for `{expr_src}`"
        );
        // Start the compiled tier cold: the driver persists across calls, so a
        // loop an earlier case already compiled would not compile again.
        super::bytecode::float_bank::reset_persistent_state();
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_f(&lowered, &columns, n, 8);
        assert_eq!(on, Some(expected), "batch jit-on vs stock for `{expr_src}`");
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
        use super::bytecode::float_bank::COMPILES as COMPILES_F;

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
            assert_eq!(
                slot.ty,
                d.ty(),
                "slot `{}` bank for `{expr_src}`",
                slot.path
            );
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
                ctx.add_variable_from_value(*name, cell_value(d, i));
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
        let clean = clean_batch_sum_f(&lowered, &columns, n)
            .map(|bits| f64::from_bits(bits as u64))
            .unwrap_or_else(|| panic!("clean tier trapped on `{expr_src}`"));
        assert_eq!(
            clean.to_bits(),
            expected.to_bits(),
            "clean vs stock for `{expr_src}`"
        );

        // majit interpreter tier, then compiled tier (monotonic compile-counter).
        let off = eval_batch_sum_float(&lowered, &columns, n, u32::MAX)
            .unwrap_or_else(|| panic!("jit-off tier trapped on `{expr_src}`"));
        assert_eq!(
            off.to_bits(),
            expected.to_bits(),
            "batch jit-off vs stock for `{expr_src}`"
        );
        // Start the compiled tier cold: the driver persists across calls, so a
        // loop an earlier case already compiled would not compile again.
        super::bytecode::float_bank::reset_persistent_state();
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_float(&lowered, &columns, n, 8)
            .unwrap_or_else(|| panic!("jit-on tier trapped on `{expr_src}`"));
        assert_eq!(
            on.to_bits(),
            expected.to_bits(),
            "batch jit-on vs stock for `{expr_src}`"
        );
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "float aggregate `{expr_src}` must compile the hot loop"
        );
    }

    /// Deterministic per-column string data drawn from a small `choices` set via
    /// an LCG, so the tree-walker oracle and the interned id column see the same
    /// values.
    fn gen_str(n: usize, seed: u64, choices: &[&str]) -> Vec<String> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                choices[((x >> 33) as usize) % choices.len()].to_string()
            })
            .collect()
    }

    /// Cross-check a typed batch containing **string** columns. Each string
    /// column reaches the machine as strings, and `prepare_batch` ranks the
    /// batch's distinct values into the `i64` ids it compares — an
    /// order-preserving, injective encoding, so an id compare equals a content
    /// compare bit for bit for ordering as well as equality. The clean / interp
    /// / compiled tiers must all equal the stock tree-walker's per-row bool/int
    /// sum, and the compiled run must trace the loop.
    fn check_batch_str(expr_src: &str, cols: &[(&str, ColData)]) {
        use super::bytecode::float_bank::COMPILES as COMPILES_F;

        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let schema: Schema = cols.iter().map(|(n, d)| (n.to_string(), d.ty())).collect();
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("lower_typed `{expr_src}`: {e}"));

        // Slot order + bank pin. A `size(<string column>)` slot is DERIVED by
        // this harness rather than declared, so it is excluded from the order
        // check and pinned to the int bank instead.
        let declared: std::collections::HashMap<&str, &ColData> =
            cols.iter().map(|(n, d)| (*n, d)).collect();
        // The declared columns must be exactly the ones the expression reads, in
        // first-use order — counting a `size(x)` slot as a read of `x`, since it
        // is derived from that column and needs no separate declaration.
        let mut referenced: Vec<&str> = Vec::new();
        for slot in &lowered.slots {
            let src = size_slot_source(&slot.path).unwrap_or(slot.path.as_str());
            if !referenced.contains(&src) {
                referenced.push(src);
            }
        }
        let want_paths: Vec<&str> = cols.iter().map(|(n, _)| *n).collect();
        assert_eq!(referenced, want_paths, "slot order for `{expr_src}`");
        for slot in &lowered.slots {
            match size_slot_source(&slot.path) {
                None => {
                    let d = declared.get(slot.path.as_str()).unwrap_or_else(|| {
                        panic!(
                            "slot `{}` has no declared column for `{expr_src}`",
                            slot.path
                        )
                    });
                    assert_eq!(
                        slot.ty,
                        d.ty(),
                        "slot `{}` bank for `{expr_src}`",
                        slot.path
                    );
                }
                Some(src) => {
                    assert_eq!(
                        slot.ty,
                        ValType::Int,
                        "derived slot `{}` must be int-banked for `{expr_src}`",
                        slot.path
                    );
                    assert!(
                        matches!(declared.get(src), Some(ColData::Str(_))),
                        "derived slot `{}` needs a declared string column `{src}` for `{expr_src}`",
                        slot.path
                    );
                }
            }
        }

        let n = cols.first().map_or(0, |(_, d)| d.len());
        for (name, d) in cols {
            assert_eq!(d.len(), n, "column `{name}` length for `{expr_src}`");
        }

        // Derived length columns: `str::len()` per row, exactly what
        // `String::size` returns. Materialized from the SAME strings the ids come
        // from, so the two columns cannot drift apart.
        let mut len_storage: std::collections::HashMap<&str, Vec<i64>> =
            std::collections::HashMap::new();
        for slot in &lowered.slots {
            if let Some(src) = size_slot_source(&slot.path) {
                if let Some(ColData::Str(c)) = declared.get(src) {
                    len_storage.insert(src, c.iter().map(|s| s.len() as i64).collect());
                }
            }
        }
        // Build one column per SLOT, keyed by path: a `size(x)` slot reads the
        // derived length column, anything else its own buffer. A string column
        // is handed over AS STRINGS — the ids are ranks over the whole batch,
        // which `prepare_batch` is the only thing positioned to compute.
        let columns: Vec<Column> = lowered
            .slots
            .iter()
            .map(|slot| match size_slot_source(&slot.path) {
                Some(src) => Column::Int(&len_storage[src]),
                None => declared[slot.path.as_str()].column(),
            })
            .collect();

        // Oracle: sum the stock tree-walker's per-row result (bool/int).
        let mut expected = 0i64;
        for i in 0..n {
            let mut ctx = Context::default();
            for (name, d) in cols {
                ctx.add_variable_from_value(*name, cell_value(d, i));
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

        // Clean two-bank interpreter over the built batch program.
        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, n),
            Some(expected),
            "clean vs stock for `{expr_src}`"
        );

        // majit interpreter tier, then compiled tier (monotonic compile counter).
        let off = eval_batch_sum_f(&lowered, &columns, n, u32::MAX);
        assert_eq!(
            off,
            Some(expected),
            "batch jit-off vs stock for `{expr_src}`"
        );
        // Start the compiled tier cold: the driver persists across calls, so a
        // loop an earlier case already compiled would not compile again.
        super::bytecode::float_bank::reset_persistent_state();
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_f(&lowered, &columns, n, 8);
        assert_eq!(on, Some(expected), "batch jit-on vs stock for `{expr_src}`");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "string batch `{expr_src}` must compile the hot loop"
        );
    }

    /// A runtime-length **list** column, in the flattened (Arrow) layout the
    /// machine reads: `lens[r]` elements for row `r`, all rows' elements laid
    /// end to end in one column per field. `offset(list)` is the exclusive
    /// prefix sum of `lens`, so locating a row's elements is arithmetic.
    struct ListCol {
        /// Per-row element count — the `size(list)` column.
        lens: Vec<i64>,
        /// `(field, flattened element column)`. `None` names the elements
        /// themselves (a list of scalars, slot path `list[]`); `Some(f)` names
        /// one record field (slot path `list[].f`). Every column is
        /// `lens.iter().sum()` long.
        fields: Vec<(Option<&'static str>, ColData)>,
    }

    impl ListCol {
        /// Exclusive prefix sums of [`ListCol::lens`] — the `offset(list)`
        /// column.
        fn offsets(&self) -> Vec<i64> {
            let mut acc = 0;
            self.lens
                .iter()
                .map(|&l| {
                    let o = acc;
                    acc += l;
                    o
                })
                .collect()
        }

        /// The tree-walker's view of row `r`: a list of scalars when the sole
        /// field is unnamed, otherwise a list of records.
        fn row_value(&self, r: usize) -> Value {
            use crate::objects::{Key, Map};
            use std::sync::Arc;

            let off = self.offsets()[r] as usize;
            let elems: Vec<Value> = (off..off + self.lens[r] as usize)
                .map(|k| match self.fields.as_slice() {
                    [(None, d)] => cell_value(d, k),
                    named => {
                        let map: HashMap<Key, Value> = named
                            .iter()
                            .map(|(f, d)| {
                                let f = f.expect("a record list names every field");
                                (Key::String(Arc::new(f.to_string())), cell_value(d, k))
                            })
                            .collect();
                        Value::Map(Map { map: Arc::new(map) })
                    }
                })
                .collect();
            Value::List(Arc::new(elems))
        }
    }

    /// Cross-check a typed batch whose expression iterates a **runtime-length
    /// list** — the nested-loop shape, where the element count is a column
    /// value rather than a green constant. The list's element columns are
    /// flattened across the whole batch and read at `(offset + j) * 8` inside
    /// the inner loop, so this also pins the derived `size(..)` / `offset(..)`
    /// row columns and the row-vs-element slot split.
    ///
    /// `expect_refusal` selects the contract: either all three tiers equal the
    /// tree-walker's per-row sum, or all three refuse (`None`) because the
    /// walker itself raises on some row. Either way the compiled run must
    /// actually trace a hot loop.
    fn check_batch_list_impl(
        expr_src: &str,
        rows: &[(&str, ColData)],
        lists: &[(&str, ListCol)],
        expect_refusal: bool,
    ) {
        use super::bytecode::float_bank::COMPILES as COMPILES_F;
        use super::lower::{elem_slot_path, elem_slot_source, offset_slot_source, SlotKind};

        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let mut schema: Schema = rows.iter().map(|(n, d)| (n.to_string(), d.ty())).collect();
        for (name, lc) in lists {
            for (field, d) in &lc.fields {
                schema.insert(elem_slot_path(name, *field), d.ty());
            }
        }
        let lowered = lower_typed(program.expression(), &schema)
            .unwrap_or_else(|e| panic!("lower_typed `{expr_src}`: {e}"));

        let n = rows
            .first()
            .map_or_else(|| lists[0].1.lens.len(), |(_, d)| d.len());
        for (name, d) in rows {
            assert_eq!(d.len(), n, "column `{name}` length for `{expr_src}`");
        }
        for (name, lc) in lists {
            assert_eq!(lc.lens.len(), n, "list `{name}` row count for `{expr_src}`");
            let total: i64 = lc.lens.iter().sum();
            for (field, d) in &lc.fields {
                assert_eq!(
                    d.len() as i64,
                    total,
                    "list `{name}` field {field:?} element count for `{expr_src}`"
                );
            }
        }

        let declared_rows: HashMap<&str, &ColData> = rows.iter().map(|(n, d)| (*n, d)).collect();
        let declared_lists: HashMap<&str, &ListCol> =
            lists.iter().map(|(n, lc)| (*n, lc)).collect();
        let offsets: HashMap<&str, Vec<i64>> =
            lists.iter().map(|(n, lc)| (*n, lc.offsets())).collect();

        // One column per SLOT: an element column for a `list[]`/`list[].f`
        // slot, the derived length / offset column for `size(list)` /
        // `offset(list)`, and the declared column for anything else.
        let columns: Vec<Column> = lowered
            .slots
            .iter()
            .map(|slot| {
                if let Some((list, field)) = elem_slot_source(&slot.path) {
                    assert!(
                        matches!(slot.kind, SlotKind::Element { .. }),
                        "slot `{}` must be an element slot for `{expr_src}`",
                        slot.path
                    );
                    let lc = declared_lists
                        .get(list)
                        .unwrap_or_else(|| panic!("slot `{}` names no list", slot.path));
                    let (_, d) = lc
                        .fields
                        .iter()
                        .find(|(f, _)| *f == field)
                        .unwrap_or_else(|| panic!("list `{list}` declares no field {field:?}"));
                    return d.column();
                }
                assert_eq!(
                    slot.kind,
                    SlotKind::Row,
                    "slot `{}` must be a row slot for `{expr_src}`",
                    slot.path
                );
                if let Some(src) = size_slot_source(&slot.path) {
                    return Column::Int(&declared_lists[src].lens);
                }
                if let Some(src) = offset_slot_source(&slot.path) {
                    return Column::Int(&offsets[src]);
                }
                declared_rows
                    .get(slot.path.as_str())
                    .unwrap_or_else(|| panic!("slot `{}` has no declared column", slot.path))
                    .column()
            })
            .collect();

        // Oracle: the stock tree-walker, one row at a time, with each list
        // rebuilt as a real `Value::List` from the same flattened data.
        let row_ctx = |i: usize| {
            let mut ctx = Context::default();
            for (name, d) in rows {
                ctx.add_variable_from_value(*name, cell_value(d, i));
            }
            for (name, lc) in lists {
                ctx.add_variable_from_value(*name, lc.row_value(i));
            }
            ctx
        };
        let expected = if expect_refusal {
            assert!(
                (0..n).any(|i| program.execute(&row_ctx(i)).is_err()),
                "`{expr_src}`: the tree-walker answers every row, so refusing would be wrong"
            );
            None
        } else {
            let mut sum = 0i64;
            for i in 0..n {
                sum += match program
                    .execute(&row_ctx(i))
                    .unwrap_or_else(|e| panic!("execute `{expr_src}`: {e:?}"))
                {
                    Value::Bool(b) => b as i64,
                    Value::Int(v) => v,
                    Value::UInt(v) => v as i64,
                    other => panic!("`{expr_src}`: unexpected {other:?}"),
                };
            }
            Some(sum)
        };

        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, n),
            expected,
            "clean vs stock for `{expr_src}`"
        );
        assert_eq!(
            eval_batch_sum_f(&lowered, &columns, n, u32::MAX),
            expected,
            "batch jit-off vs stock for `{expr_src}`"
        );
        // Start the compiled tier cold: the driver persists across calls, so a
        // loop an earlier case already compiled would not compile again.
        super::bytecode::float_bank::reset_persistent_state();
        let before = COMPILES_F.load(Ordering::Relaxed);
        assert_eq!(
            eval_batch_sum_f(&lowered, &columns, n, 8),
            expected,
            "batch jit-on vs stock for `{expr_src}`"
        );
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "list batch `{expr_src}` must compile a hot loop"
        );
    }

    /// [`check_batch_list_impl`] with the answering contract.
    fn check_batch_list(expr_src: &str, rows: &[(&str, ColData)], lists: &[(&str, ListCol)]) {
        check_batch_list_impl(expr_src, rows, lists, false);
    }

    /// [`check_batch_list_impl`] with the refusing contract.
    fn check_batch_list_refuses(
        expr_src: &str,
        rows: &[(&str, ColData)],
        lists: &[(&str, ListCol)],
    ) {
        check_batch_list_impl(expr_src, rows, lists, true);
    }

    /// Deterministic per-row element counts in `[0, max]`, so the batch mixes
    /// empty rows (the zero-trip guard) with rows of several elements (the
    /// back-edge).
    fn gen_lens(n: usize, seed: u64, max: i64) -> Vec<i64> {
        gen_i64(n, seed, 0, max)
    }

    #[test]
    fn batch_size() {
        // `size(s)` reads a DERIVED length column: the machine carries a string
        // as an `i64` id and has no bytes to count, so the batch builder
        // materializes `str::len()` per row from the same strings it interns.
        // The walker's `String::size` is exactly `str::len()` — UTF-8 BYTES, not
        // code points — so the multi-byte choices below are the interesting case
        // and a code-point count would fail here.
        let n = 3000;
        let words = ["a", "bb", "ccc", "", "héllo", "日본어", "🎉"];
        let s = gen_str(n, 0x5EED_1234_ABCD_9876, &words);
        for expr in [
            "size(s) > 2",
            "size(s)",
            "s.size() > 2",
            "size(s) == 0",
            "size(s) * 2 - 1",
        ] {
            check_batch_str(expr, &[("s", ColData::Str(s.clone()))]);
        }
        // Value and length of the same column together: two slots, one declared
        // column, and the id/length columns must stay row-aligned.
        check_batch_str(
            "s == \"ccc\" || size(s) > 4",
            &[("s", ColData::Str(s.clone()))],
        );
        // A literal list has a green length that folds to a constant.
        check_batch_str("size([1, 2, 3]) + size(s)", &[("s", ColData::Str(s))]);
    }

    #[test]
    fn size_bails() {
        // Only a `string` or `list` column has a materialized length column,
        // and an int column has no length at all. A LITERAL string or list is
        // not a column and does not need one — its length is green and folds
        // (see `size_folds_a_literal_argument`).
        let schema: Schema = [
            ("s".to_string(), ValType::Str),
            ("i".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in ["size(i) > 1", "i.size() > 1"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering"
            );
        }
    }

    /// A literal argument's length is known while lowering — `str::len` is the
    /// same UTF-8 BYTE count the walker reports, including for multi-byte
    /// characters, so the fold must not count chars.
    #[test]
    fn size_folds_a_literal_argument() {
        let cols: Vec<(&'static str, ColData)> = vec![("i", ColData::Int(vec![1, 2, 3, 4]))];
        for expr in [
            "size('abc') > 1",
            "size('') == 0",
            "size('\u{00e9}') == 2",
            "size('\u{d55c}\u{ae00}') == 6",
            "size([1, 2, 3]) == 3",
            "size('ab') + i > 3",
        ] {
            assert_eq!(
                sweep_case(expr, &cols),
                SweepVerdict::Agreed,
                "`{expr}` folds to a constant and must match the walker"
            );
        }
    }

    #[test]
    fn batch_string_equality() {
        // String ==/!= lower to an id compare (OP_EQ/OP_NE over the
        // int-file ids). The oracle compares actual strings; the id compare is
        // bit-exact against it across the clean / interp / compiled tiers.
        let n = 3000;
        let roles = ["admin", "user", "guest", "root", "auditor"];
        let role = gen_str(n, 0x3A5B_7C9D_1E2F_0405, &roles);
        // Column vs a present literal, both == and !=.
        check_batch_str("role == \"admin\"", &[("role", ColData::Str(role.clone()))]);
        check_batch_str("role != \"admin\"", &[("role", ColData::Str(role.clone()))]);
        // A literal absent from the column: every row is unequal (count 0 for
        // ==, n for !=), and it must still compile.
        check_batch_str(
            "role == \"superadmin\"",
            &[("role", ColData::Str(role.clone()))],
        );
        // Column vs column.
        let other = gen_str(n, 0x9182_7364_5A4B_3C2D, &roles);
        check_batch_str(
            "a == b",
            &[
                ("a", ColData::Str(role.clone())),
                ("b", ColData::Str(other.clone())),
            ],
        );
        check_batch_str(
            "a != b",
            &[("a", ColData::Str(role)), ("b", ColData::Str(other))],
        );
    }

    #[test]
    fn batch_string_mixed_with_int() {
        // A string equality combined with an int comparison via `&&` — the
        // string id compare and the int compare share the int register file.
        let n = 3000;
        let roles = ["admin", "user", "guest"];
        let role = gen_str(n, 0x1122_3344_5566_7788, &roles);
        let age = gen_i64(n, 0x8877_6655_4433_2211, 0, 80);
        check_batch_str(
            "role == \"admin\" && age >= 18",
            &[("role", ColData::Str(role)), ("age", ColData::Int(age))],
        );
    }

    /// A string literal's id is a **broadcast scalar**, not a program constant:
    /// it reaches the machine in a seeded register, and no word of the program
    /// carries it.
    ///
    /// This is what keeps the warm driver warm. `prepare_batch` interns the code
    /// words and the JIT keys its compiled loop on them, so a per-batch value
    /// baked into an immediate would re-key the trace on every batch and compile
    /// the loop again each time. And the id IS per-batch now that it is a rank:
    /// `"m"` is id 1 among `["a", "m", "z"]` and id 0 among `["m", "z"]`.
    ///
    /// The assertion that pins it: two expressions differing ONLY in the literal
    /// must produce byte-identical words.
    #[test]
    fn string_literal_id_rides_a_register_not_the_words() {
        use super::lower::SeedKind;
        let schema: Schema = [("role".to_string(), ValType::Str)].into_iter().collect();
        let lower = |src: &str| {
            let program = Program::compile(src).unwrap();
            lower_typed(program.expression(), &schema).expect("string equality lowers")
        };

        let admin = lower("role == \"admin\"");
        let guest = lower("role == \"superadmin\"");
        assert_eq!(
            admin.scalar_seeds[0].kind,
            SeedKind::StrId("admin".to_string())
        );
        assert_eq!(
            guest.scalar_seeds[0].kind,
            SeedKind::StrId("superadmin".to_string())
        );

        let (a_shape, g_shape) = (admin.batch_sum_shape(true), guest.batch_sum_shape(true));
        assert_eq!(
            a_shape.code, g_shape.code,
            "two literals, one shape: the id is not in the words"
        );
        assert_eq!(a_shape.seed.num_scalars(), 1);
        assert_eq!(
            admin.scalar_seeds[0].reg, guest.scalar_seeds[0].reg,
            "and both arrive in the same register"
        );
    }

    /// The same batch, one literal, two different sets of neighbours: the
    /// literal's rank moves, and the answer does not.
    ///
    /// A rank is a property of the batch, which is what forced the id out of the
    /// words. This is the case that would silently break if a literal's id were
    /// ever cached across batches.
    #[test]
    fn a_literals_rank_moves_with_the_batch_and_the_answer_does_not() {
        use super::batch::{Batch, BatchProgram, ColumnRef};
        let schema: Schema = [("role".to_string(), ValType::Str)].into_iter().collect();
        let program = BatchProgram::compile("role < \"m\"", &schema).expect("string `<` lowers");

        // `"m"` sorts last here and in the middle there, so its rank differs.
        for (rows, want) in [
            (vec!["a", "b", "c"], 3),
            (vec!["a", "z", "n", "b"], 2),
            (vec!["z", "y"], 0),
        ] {
            let col: Vec<String> = rows.iter().map(|s| s.to_string()).collect();
            let batch = Batch::new(col.len()).column("role", ColumnRef::Str(&col));
            let got = program.bind(&batch).expect("bind").sum().expect("sum");
            assert_eq!(got, Value::Int(want), "rows {rows:?}");
        }
    }

    /// The four pure string predicates, cross-checked against the tree-walker
    /// on every tier. Each is answered once per DISTINCT string at bind and
    /// read per row from a table indexed by the id, so the assertion that
    /// matters is that per-distinct and per-row give the same answers.
    #[test]
    fn batch_string_predicates() {
        let n = 3000;
        // Prefixes, suffixes and infixes that overlap, plus the empty string,
        // so a table that confused two ids would show up.
        let words = [
            "admin",
            "ad",
            "administrator",
            "badmin",
            "guest",
            "gu",
            "",
            "ADMIN",
        ];
        let col = gen_str(n, 0x0BAD_5EED_1234_5678, &words);
        for expr in [
            "s.startsWith('ad')",
            "s.startsWith('')",
            "s.startsWith('zzz')",
            "s.endsWith('min')",
            "s.endsWith('')",
            "s.contains('dmi')",
            "s.contains('')",
            "s.matches('^a.*n$')",
            "s.matches('[Aa]dmin')",
            // Combined with the id compare and an ordering, so the table read
            // and the rank compare share a batch.
            "s.startsWith('ad') && s != 'ad'",
            "s.contains('d') || s < 'b'",
        ] {
            check_batch_str(expr, &[("s", ColData::Str(col.clone()))]);
        }
    }

    /// A predicate's argument must be a literal (there is no single table for a
    /// per-row argument), and an invalid regex must DECLINE rather than answer:
    /// the tree-walker raises on it, and swallowing that would be a wrong
    /// answer rather than a missing one.
    #[test]
    fn string_predicate_declines_keep_the_walkers_errors() {
        let schema: Schema = [
            ("s".to_string(), ValType::Str),
            ("t".to_string(), ValType::Str),
        ]
        .into_iter()
        .collect();
        let lower = |expr: &str| {
            let program = Program::compile(expr).unwrap();
            lower_typed(program.expression(), &schema)
        };
        for expr in ["s.startsWith(t)", "s.contains(t)", "s.matches('[')"] {
            assert!(lower(expr).is_err(), "`{expr}` must decline");
        }
        // And the walker really does raise on that regex, so declining is what
        // keeps the two tiers agreeing.
        let program = Program::compile("s.matches('[')").unwrap();
        let mut ctx = Context::default();
        ctx.add_variable_from_value("s", Value::String("a".to_string().into()));
        assert!(program.execute(&ctx).is_err(), "invalid regex must raise");
    }

    /// `-b` is not a CEL overload, but the evaluator this tier accelerates
    /// answers it as `!b`, so the tier answers it the same way. A JIT that
    /// disagreed with its own interpreter would be wrong whichever one matches
    /// the spec.
    #[test]
    fn unary_minus_on_bool_matches_the_evaluator() {
        for b in [true, false] {
            let program = Program::compile("-b").unwrap();
            let mut ctx = Context::default();
            ctx.add_variable_from_value("b", Value::Bool(b));
            assert_eq!(
                program.execute(&ctx).unwrap(),
                Value::Bool(!b),
                "evaluator ground truth for `-{b}`"
            );
        }
        let n = 600;
        let flags = gen_i64(n, 0x1357_9BDF_2468_ACE0, 0, 1);
        check_batch_f("-b", &[("b", ColData::Bool(flags))]);
    }

    /// String ordering lowers now that ids are ranks; what still bails is a
    /// string-VALUED result, which is the reduction's limit and not the
    /// comparison's — there is no sum of strings.
    #[test]
    fn string_ordering_lowers_and_a_string_result_still_bails() {
        let schema: Schema = [
            ("a".to_string(), ValType::Str),
            ("b".to_string(), ValType::Str),
        ]
        .into_iter()
        .collect();
        let lower = |expr: &str| {
            let program = Program::compile(expr).unwrap();
            lower_typed(program.expression(), &schema)
        };
        for expr in ["a < b", "a <= b", "a > b", "a >= b", "a == b", "a != b"] {
            assert!(lower(expr).is_ok(), "`{expr}` must lower to a rank compare");
        }
        // `a` LOWERS — it is a column read — but its result is a string, and
        // the batch loop reduces by sum. Two separate refusals.
        assert!(lower("a").unwrap().sum_reducible().is_err());
        assert!(
            lower("a + b").is_err(),
            "concatenation has no characters to produce"
        );
    }

    /// Ordering over a real batch, cross-checked against the tree-walker across
    /// every tier. This is the assertion the rank encoding exists for: the
    /// walker compares CONTENT, the machine compares ids, and they must agree.
    #[test]
    fn batch_string_ordering() {
        let n = 3000;
        // Deliberately not sorted, not uniform in length, and sharing prefixes,
        // so a rank that merely grouped equal strings would not survive.
        let words = [
            "admin",
            "ad",
            "administrator",
            "auditor",
            "guest",
            "g",
            "root",
            "Root",
            "",
        ];
        let a = gen_str(n, 0x5150_1234_ABCD_9876, &words);
        let b = gen_str(n, 0xFEED_FACE_0BAD_C0DE, &words);
        for op in ["<", "<=", ">", ">="] {
            check_batch_str(
                &format!("a {op} b"),
                &[
                    ("a", ColData::Str(a.clone())),
                    ("b", ColData::Str(b.clone())),
                ],
            );
            // Against a literal, both present in the column and absent from it.
            for lit in ["guest", "zzz"] {
                check_batch_str(
                    &format!("a {op} \"{lit}\""),
                    &[("a", ColData::Str(a.clone()))],
                );
            }
        }
    }

    /// Deterministic i64-nanosecond column in `[base, base + span)` from an LCG,
    /// for timestamp / duration columns.
    fn gen_nanos(n: usize, seed: u64, base: i64, span: i64) -> Vec<i64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Keep the top 63 bits: an LCG's low bits are short-period, and
                // `>> 33` would cap every draw at ~2^31 ns (~2.1s), collapsing a
                // multi-year span into a single instant.
                base + ((x >> 1) % span as u64) as i64
            })
            .collect()
    }

    #[test]
    fn batch_timestamp_compare() {
        // Timestamps are i64 nanoseconds since the epoch; the signed int order
        // equals the chronological order, so all six comparisons are bit-exact
        // against the tree-walker (which compares Value::Timestamp instants). The
        // `timestamp("...")` literal folds to a nanos constant via the same
        // parse_from_rfc3339 the walker uses.
        let n = 3000;
        // Column spans ~2023-11 .. ~2024-07; the literal 2024-01-01 sits inside.
        let base = 1_700_000_000_000_000_000;
        let span = 20_000_000_000_000_000;
        let event = gen_nanos(n, 0x71E5_7A11_9B0C_2D3E, base, span);
        let created = gen_nanos(n, 0xC0DE_F00D_1234_5678, base, span);
        // Column vs a `timestamp(...)` literal — `<`, `>=`, `==`.
        check_batch_f(
            "event < timestamp(\"2024-01-01T00:00:00Z\")",
            &[("event", ColData::Timestamp(event.clone()))],
        );
        check_batch_f(
            "event >= timestamp(\"2024-01-01T00:00:00Z\")",
            &[("event", ColData::Timestamp(event.clone()))],
        );
        // Column vs column.
        check_batch_f(
            "event < created",
            &[
                ("event", ColData::Timestamp(event.clone())),
                ("created", ColData::Timestamp(created.clone())),
            ],
        );
        check_batch_f(
            "event == created",
            &[
                ("event", ColData::Timestamp(event)),
                ("created", ColData::Timestamp(created)),
            ],
        );
    }

    #[test]
    fn batch_duration_compare() {
        // Durations are i64 nanoseconds; the same signed-int order holds, and
        // `duration("1h")` folds to a nanos constant via the walker's parser.
        let n = 3000;
        let elapsed = gen_nanos(n, 0x2222_3333_4444_5555, 0, 7_200_000_000_000); // [0, 2h)
        let budget = gen_nanos(n, 0x9999_8888_7777_6666, 0, 7_200_000_000_000);
        check_batch_f(
            "elapsed > duration(\"1h\")",
            &[("elapsed", ColData::Duration(elapsed.clone()))],
        );
        check_batch_f(
            "elapsed <= budget",
            &[
                ("elapsed", ColData::Duration(elapsed)),
                ("budget", ColData::Duration(budget)),
            ],
        );
    }

    #[test]
    fn batch_int_division() {
        // Regression: the two-bank machine never implemented `OP_DIV`/`OP_MOD`,
        // so every int `/` or `%` that reached the typed (columnar) path died
        // with `bad op`. Only the single-bank machine had them, and no columnar
        // test divided, so the gap went unseen. Negative operands included: cel
        // `/` truncates toward zero while majit lowers a bare `/` to floor.
        let n = 2000;
        let a = gen_nanos(n, 0x5151_2626_3737_4848, -5000, 10_000);
        let b = gen_nanos(n, 0x1234_5678_9abc_def0, 1, 97); // nonzero divisor
        for expr in ["a / b", "a % b"] {
            check_batch_f(
                expr,
                &[
                    ("a", ColData::Int(a.clone())),
                    ("b", ColData::Int(b.clone())),
                ],
            );
        }
        // Constant divisor: the literal folds to a prelude register, which is
        // also the shape the duration accessors emit.
        for expr in ["a / 7", "a % 7", "(a + 1) / 7 - a % 3"] {
            check_batch_f(expr, &[("a", ColData::Int(a.clone()))]);
        }
    }

    #[test]
    fn batch_int_division_at_i64_min() {
        // Regression: the traced `/` and `%` divide the operands' MAGNITUDES
        // (floor and truncation agree there) and reapply the sign. `|i64::MIN|`
        // is 2^63, which is not an i64 — read as a signed magnitude it comes
        // back NEGATIVE, and the sign reapplication then flipped the answer, so
        // `i64::MIN / 2` produced `+2^62` in the compiled tier while the walker
        // and the clean tier said `-2^62`. Dividing the magnitudes UNSIGNED
        // makes the bit pattern exact. Every legal `i64::MIN` divisor is
        // covered; `-1` is the illegal one and belongs to the refusal test.
        // The divisor cycle avoids `-1`, the one illegal divisor for
        // `i64::MIN` (that corner belongs to `int_division_refuses`).
        let divisors = [2i64, -2, 3, -3, 1, 7, -7, 97, -97, 5, -5, i64::MIN];
        let n = 240;
        let b: Vec<i64> = (0..n).map(|i| divisors[i % divisors.len()]).collect();
        // Mostly `i64::MIN`, with the neighbours mixed in so the column is not
        // one constant the optimizer could specialize the whole loop on.
        let a: Vec<i64> = (0..n)
            .map(|i| match i % 8 {
                3 => i64::MIN + 1,
                6 => i64::MAX,
                _ => i64::MIN,
            })
            .collect();
        // The quotients and remainders here reach ±2^63, which would overflow
        // the machine's plain `OP_ADD` reduction (and the oracle's `+=`), so
        // reduce each one mod a large prime: still magnitude-sensitive to every
        // bit that matters, but summable. The sign predicates pin the exact
        // symptom the bug had — a flipped sign.
        for expr in [
            "(a / b) % 1000000007",
            "(a % b) % 1000000007",
            "a / b < 0",
            "a % b < 0",
            "a / b > 0",
        ] {
            check_batch_f(
                expr,
                &[
                    ("a", ColData::Int(a.clone())),
                    ("b", ColData::Int(b.clone())),
                ],
            );
        }
    }

    #[test]
    fn int_division_refuses() {
        // `/` and `%` are PARTIAL in the tree-walker: a zero divisor raises
        // `DivisionByZero`/`RemainderByZero` and `i64::MIN / -1` raises
        // `Overflow` (`common/types/int.rs:119-143`). Both were an undeclared
        // "assumed domain" — the zero divisor PANICKED the process (Rust integer
        // division by zero panics in every build) and the overflow corner
        // answered `i64::MIN`. Now each is the RPython `ll_int_py_div_ovf_zer`
        // guard pair, whose failure records the trap and abandons the batch.
        // Long enough that the row loop still gets hot: refusing is a guard exit
        // on one row, not a reason for the trace never to compile.
        let n = 240;
        let a: Vec<i64> = (0..n).map(|i| i as i64 + 10).collect();
        // A single zero, late enough that the loop is already compiled.
        let mut zero_divisor: Vec<i64> = (0..n).map(|i| (i % 7) as i64 + 1).collect();
        zero_divisor[200] = 0;
        // `i64::MIN / -1` is the overflow corner: one row carries it.
        let mut min_dividend = a.clone();
        min_dividend[200] = i64::MIN;
        let mut minus_one: Vec<i64> = (0..n).map(|i| (i % 7) as i64 + 1).collect();
        minus_one[200] = -1;
        for (expr, a, b) in [
            ("a / b", a.clone(), zero_divisor.clone()),
            ("a % b", a.clone(), zero_divisor),
            ("a / b", min_dividend.clone(), minus_one.clone()),
            ("a % b", min_dividend, minus_one),
        ] {
            check_batch_f_refuses(expr, &[("a", ColData::Int(a)), ("b", ColData::Int(b))]);
        }
    }

    #[test]
    fn batch_uint_division() {
        // Unsigned `/` and `%`. The columns are full-range u64 (roughly half
        // with the high bit set), which is exactly where a signed division
        // answers something else — so this fails outright if the lowering reuses
        // the signed opcode. RPython has NO unsigned division resop (`UINT_
        // FLOORDIV` was deleted in 2016); these lower to the `int.udiv` /
        // `int.umod` oopspec residual calls instead, the same shape the signed
        // `/` already used.
        let n = 2000;
        let a = gen_u64_bits(n, 0x9E37_79B9_7F4A_7C15);
        // Divisor column: full-range but never zero.
        let b: Vec<i64> = gen_u64_bits(n, 0x2545_F491_4F6C_DD1D)
            .into_iter()
            .map(|v| if v == 0 { 1 } else { v })
            .collect();
        // A full-range quotient or remainder would overflow the machine's plain
        // `OP_ADD` reduction, so reduce each row mod a large prime first — which
        // is itself another unsigned division.
        for expr in [
            "(a / b) % 1000000007u",
            "(a % b) % 1000000007u",
            "a / b == 0u",
            "a % b == a",
        ] {
            check_batch_f(
                expr,
                &[
                    ("a", ColData::UInt(a.clone())),
                    ("b", ColData::UInt(b.clone())),
                ],
            );
        }
        // Constant divisors, including one above 2^63 where the signed quotient
        // would be negative and the unsigned one is 0 or 1.
        for expr in [
            "(a / 7u) % 1000000007u",
            "a % 7u",
            "a / 9223372036854775809u",
            "(a % 9223372036854775809u) % 1000000007u",
        ] {
            check_batch_f(expr, &[("a", ColData::UInt(a.clone()))]);
        }
    }

    #[test]
    fn uint_division_refuses() {
        // The unsigned zero divisor is the only partial case: every pair of u64
        // operands with a nonzero divisor has a representable quotient, so there
        // is no unsigned peer of the `i64::MIN / -1` corner.
        let n = 240;
        let a: Vec<i64> = (0..n).map(|i| i as i64 + 10).collect();
        let mut b: Vec<i64> = (0..n).map(|i| (i % 7) as i64 + 1).collect();
        b[200] = 0;
        for expr in ["a / b", "a % b"] {
            check_batch_f_refuses(
                expr,
                &[
                    ("a", ColData::UInt(a.clone())),
                    ("b", ColData::UInt(b.clone())),
                ],
            );
        }
    }

    #[test]
    fn batch_duration_accessors() {
        // `d.getHours()` and friends are `chrono::Duration::num_*` = a
        // toward-zero divide of the i64-nanos payload. The span straddles zero so
        // NEGATIVE durations are exercised: that is where truncation and floor
        // disagree, the exact class the `OP_DIV` sign-mask fix exists for. Both
        // the compiled and interpreter tiers must match the tree-walker oracle.
        let n = 3000;
        let elapsed = gen_nanos(
            n,
            0x0f1e_2d3c_4b5a_6978,
            -7_200_000_000_000,
            14_400_000_000_000,
        );
        for expr in [
            "elapsed.getHours()",
            "elapsed.getMinutes()",
            "elapsed.getSeconds()",
            "elapsed.getMilliseconds()",
            // The accessor result is a plain int, so it composes with the rest of
            // the int subset (arithmetic, comparison, ternary).
            "elapsed.getSeconds() * 2 + 1",
            "elapsed.getMinutes() >= 0",
            "elapsed.getHours() > 0 ? elapsed.getMinutes() : 0 - elapsed.getMinutes()",
        ] {
            check_batch_f(expr, &[("elapsed", ColData::Duration(elapsed.clone()))]);
        }
    }

    #[test]
    fn batch_timestamp_accessors() {
        // Calendar fields from an i64-nanos instant: a floored day split plus
        // Hinnant's civil-from-days. The span deliberately straddles the epoch,
        // so PRE-1970 instants (negative nanos, where the day count must floor
        // rather than truncate) are covered, and it is wide enough to cross leap
        // years and year ends. Every field is checked against the tree-walker.
        let n = 4000;
        // ~1962-01 .. ~1977-12, i.e. both sides of the epoch.
        let ts = gen_nanos(
            n,
            0x7a6b_5c4d_3e2f_1009,
            -252_460_800_000_000_000,
            504_921_600_000_000_000,
        );
        for expr in [
            "t.getFullYear()",
            "t.getMonth()",
            "t.getDate()",
            "t.getDayOfMonth()",
            "t.getDayOfYear()",
            "t.getDayOfWeek()",
            "t.getHours()",
            "t.getMinutes()",
            "t.getSeconds()",
            "t.getMilliseconds()",
            // Composes with the rest of the int subset.
            "t.getFullYear() * 100 + t.getMonth()",
            "t.getDayOfWeek() == 0 || t.getDayOfWeek() == 6",
            "t.getHours() >= 9 && t.getHours() < 18 ? t.getMinutes() : 0",
        ] {
            check_batch_f(expr, &[("t", ColData::Timestamp(ts.clone()))]);
        }
    }

    #[test]
    fn batch_timestamp_accessors_recent_epoch() {
        // A second window entirely after the epoch, spanning a leap day
        // (2024-02-29) and a year boundary, so the leap-year arm of
        // civil-from-days is exercised on positive day counts too.
        let n = 4000;
        let ts = gen_nanos(
            n,
            0x1122_3344_5566_7788,
            1_703_980_800_000_000_000, // 2023-12-31T00:00:00Z
            86_400_000_000_000_000,    // 1000 days
        );
        for expr in [
            "t.getFullYear()",
            "t.getMonth()",
            "t.getDate()",
            "t.getDayOfYear()",
            "t.getDayOfWeek()",
        ] {
            check_batch_f(expr, &[("t", ColData::Timestamp(ts.clone()))]);
        }
    }

    #[test]
    fn temporal_accessor_bails() {
        // The accessor names are registered ONLY as member overloads
        // (`common/types/duration.rs:191-222`, `timestamp.rs:278-357`), so
        // everything outside those overloads must bail to the tree-walker rather
        // than answer:
        //   * the global spelling is an UndeclaredReference in the walker,
        //   * a non-temporal receiver is a type error,
        //   * the calendar names have no `duration` overload,
        //   * a folded timestamp literal has lost its UTC offset,
        //   * the accessors take no arguments.
        let schema: Schema = [
            ("t".to_string(), ValType::Timestamp),
            ("d".to_string(), ValType::Duration),
            ("i".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in [
            // Registered only as member overloads, so the global spelling is an
            // UndeclaredReference in the walker — lowering it would answer where
            // the walker raises.
            "getHours(d)",
            "getFullYear(t)",
            // Non-temporal receiver: a type error in the walker.
            "i.getSeconds()",
            // The accessors take no arguments.
            "d.getHours(1)",
            // Calendar fields have no `duration` overload.
            "d.getDayOfWeek()",
            "d.getFullYear()",
            // A folded timestamp literal drops the RFC-3339 offset the walker
            // keeps, so its calendar fields are not ours to answer.
            "timestamp(\"2024-03-05T06:07:08+09:00\").getHours()",
            "timestamp(\"2024-03-05T06:07:08Z\").getFullYear()",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering"
            );
        }
    }

    /// The numeric conversions that are pure moves on the two-bank machine.
    /// `double(int)` is one `cast_int_to_float`; `int(uint)` and `uint(int)` are
    /// raw reinterpretations of the same 64-bit pattern, so they emit nothing at
    /// all and only relabel the bank; the same-type spellings are the identity.
    /// The uint columns deliberately include values above 2^63, where the signed
    /// and unsigned readings differ, so a mislabelled bank cannot pass.
    #[test]
    fn batch_numeric_conversions() {
        let n = 3000;
        let i = gen_i64(n, 0x4a3b_2c1d_0e9f_8a7b, -5_000, 10_000);
        let f = gen_f64(n, 0x9f8e_7d6c_5b4a_3928, -5_000.0, 10_000.0);
        // Straddles 2^63: read as i64 these are negative, as u64 they are huge.
        let u: Vec<i64> = gen_i64(n, 0x1357_9bdf_2468_ace0, -5_000, 10_000)
            .into_iter()
            .enumerate()
            .map(|(k, v)| {
                if k % 3 == 0 {
                    v.wrapping_add(i64::MIN)
                } else {
                    v
                }
            })
            .collect();

        check_batch_f("double(i) > 100.0", &[("i", ColData::Int(i.clone()))]);
        for expr in ["double(i) + f > 0.0", "double(i) == f"] {
            check_batch_f(
                expr,
                &[
                    ("i", ColData::Int(i.clone())),
                    ("f", ColData::Float(f.clone())),
                ],
            );
        }
        for expr in ["int(u) < 0", "int(u) > 100", "uint(int(u)) > 100u"] {
            check_batch_f(expr, &[("u", ColData::UInt(u.clone()))]);
        }
        for expr in ["uint(i) > 100u", "int(uint(i)) > 100"] {
            check_batch_f(expr, &[("i", ColData::Int(i.clone()))]);
        }
        // `int(double)` truncates toward zero and saturates at the i64 bounds.
        // The float column spans both signs so the toward-zero rounding is
        // exercised on negatives, where a floor would differ.
        for expr in ["int(f) > 100", "int(f) < 0", "int(f)"] {
            check_batch_f(expr, &[("f", ColData::Float(f.clone()))]);
        }
        check_batch_f(
            "int(f) + i > 0",
            &[
                ("f", ColData::Float(f.clone())),
                ("i", ColData::Int(i.clone())),
            ],
        );
        // Identity spellings.
        check_batch_f("int(i) > 100", &[("i", ColData::Int(i.clone()))]);
        check_batch_f("double(f) > 100.0", &[("f", ColData::Float(f))]);
        check_batch_f("uint(u) > 100u", &[("u", ColData::UInt(u))]);
    }

    #[test]
    fn numeric_conversion_bails() {
        // What is left after the four numeric casts: a string or temporal
        // argument, which the walker answers with a parse or a `FunctionError`
        // rather than a number. (The numeric pairings all lower — see
        // `numeric_conversions_at_the_saturation_bounds`.)
        let schema: Schema = [
            ("i".to_string(), ValType::Int),
            ("f".to_string(), ValType::Float),
            ("u".to_string(), ValType::UInt),
            ("s".to_string(), ValType::Str),
            ("t".to_string(), ValType::Timestamp),
        ]
        .into_iter()
        .collect();
        for expr in [
            "int(s) > 1",
            "double(s) > 1.0",
            "int(\"123\") > 1",
            "int(t) > 1",
            "double(t) > 1.0",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering"
            );
        }
    }

    /// Member syntax must never reach a GLOBAL overload. The tree-walker looks
    /// the two spellings up in disjoint namespaces (`objects.rs:1331` global vs
    /// `:1364` member), and of the whole stdlib only `size` is registered both
    /// ways, so `x.double()` / `"...".timestamp()` are `UndeclaredReference`
    /// errors there. Answering them would be a JIT-only result for an expression
    /// the walker rejects.
    #[test]
    fn member_syntax_cannot_reach_global_overloads() {
        let schema: Schema = [
            ("i".to_string(), ValType::Int),
            ("f".to_string(), ValType::Float),
            ("s".to_string(), ValType::Str),
        ]
        .into_iter()
        .collect();
        for expr in [
            "i.double() > 1.0",
            "f.int() > 1",
            "i.uint() > 1u",
            "\"2024-01-01T00:00:00Z\".timestamp() > timestamp(\"2020-01-01T00:00:00Z\")",
            "\"3s\".duration() > duration(\"1s\")",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail: member syntax does not reach a global overload"
            );
        }
    }

    /// `bool` is its own declared type, not a spelling of `int`.
    ///
    /// Before [`ValType::Bool`] existed, an `int` column reached `&&`/`||`/`!`
    /// and `?:` because the only test those sites made was "is it in the int
    /// bank", so `a && b` with `a = 1, b = 2` answered `3` (`OP_AND` is bitwise)
    /// where the tree-walker raises `NoSuchOverload` — a JIT-only answer to an
    /// expression CEL rejects. The lowering must now decline every one of these.
    #[test]
    fn bool_is_a_type_not_an_int() {
        let schema: Schema = [
            ("b".to_string(), ValType::Bool),
            ("c".to_string(), ValType::Bool),
            ("i".to_string(), ValType::Int),
            ("j".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in [
            // Logical ops are bool-only: `1 && 2`, `1 || 2`, `!1` are all
            // NoSuchOverload in the walker.
            "i && j",
            "i || j",
            "!i",
            "b && i",
            "i && b",
            // The ternary condition is bool-only.
            "i ? i : j",
            // Arithmetic on bool is UnsupportedBinaryOperator.
            "b + c",
            "b - c",
            "b * c",
            "b + i",
            // Cross-type ordering is NoSuchOverload.
            "i < b",
            "b > i",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering (bool is not int)"
            );
        }

        // Cross-type `==` does NOT compare the bits: an int and a bool are
        // never equal whatever they hold, so the walker answers `false` and the
        // lowering folds to that constant rather than declining.
        for i in [0i64, 1, 2, -1] {
            for x in [false, true] {
                let pair = &[("i", Bind::Int(i)), ("b", Bind::Bool(x))];
                check("i == b", pair);
                check("i != b", pair);
                check("b == i", pair);
            }
        }

        // The same operators on real bools lower and agree with the walker.
        for (x, y) in [(false, false), (false, true), (true, false), (true, true)] {
            let pair = &[("b", Bind::Bool(x)), ("c", Bind::Bool(y))];
            for expr in ["b && c", "b || c", "b < c", "b <= c", "b == c", "b != c"] {
                check(expr, pair);
            }
            check("!b", &[("b", Bind::Bool(x))]);
            check(
                "b ? i : j",
                &[
                    ("b", Bind::Bool(x)),
                    ("i", Bind::Int(7)),
                    ("j", Bind::Int(-3)),
                ],
            );
        }
    }

    /// A path the schema does not declare is a decline, not an implicit `int`.
    /// The bank decides which operators the path is legal under, so guessing one
    /// answers an expression the caller never typed.
    #[test]
    fn undeclared_path_bails() {
        let schema: Schema = [("i".to_string(), ValType::Int)].into_iter().collect();
        for expr in ["i + missing", "missing > 0", "i > 0 && missing"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail: `missing` is undeclared"
            );
        }
    }

    #[test]
    fn temporal_mixed_bails() {
        // A timestamp vs duration comparison is NoSuchOverload and a temporal vs
        // int is a type error, so both bail the LOWERING. So do the products
        // and quotients, which have no temporal overload at all, and
        // `duration - timestamp`, which CEL does not define even though its
        // mirror image is. A bare temporal column lowers and is refused a step
        // later, by the sum.
        let schema: Schema = [
            ("t".to_string(), ValType::Timestamp),
            ("d".to_string(), ValType::Duration),
            ("i".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        let lower = |expr: &str| {
            let program = Program::compile(expr).unwrap();
            lower_typed(program.expression(), &schema)
        };
        // `d + t` is in the list on purpose: `t + d` lowers, but the evaluator
        // dispatches on the LEFT operand and `Duration` has no timestamp arm,
        // so the mirror image is UnsupportedBinaryOperator.
        for expr in [
            "t < d", "t < i", "t * d", "d / d", "d % d", "d - t", "t + t", "d + t",
        ] {
            assert!(
                lower(expr).is_err(),
                "`{expr}` must bail the typed lowering (mixed / undefined overload)"
            );
        }
        assert!(
            lower("t").unwrap().sum_reducible().is_err(),
            "a bare temporal column lowers; the sum is what refuses it"
        );
    }

    /// Temporal arithmetic, cross-checked against the tree-walker on every
    /// tier. Both banks are i64 nanoseconds, but the walker computes in chrono,
    /// so this is the test that the two agree inside the domain the lowering
    /// narrowed to.
    #[test]
    fn batch_temporal_arithmetic() {
        let n = 3000;
        // ±20 years around the epoch: comfortably inside the ±146-year bound a
        // single operation allows, so no batch here is refused.
        const YEAR: i64 = 365 * 24 * 3_600_000_000_000;
        let t1 = gen_nanos(n, 0x1111_2222_3333_4444, -20 * YEAR, 40 * YEAR);
        let t2 = gen_nanos(n, 0x5555_6666_7777_8888, -20 * YEAR, 40 * YEAR);
        let d1 = gen_nanos(n, 0x9999_AAAA_BBBB_CCCC, -YEAR, 2 * YEAR);
        let d2 = gen_nanos(n, 0xDDDD_EEEE_FFFF_0000, -YEAR, 2 * YEAR);
        let pick = |nm: &str| -> ColData {
            match nm {
                "t1" => ColData::Timestamp(t1.clone()),
                "t2" => ColData::Timestamp(t2.clone()),
                "d1" => ColData::Duration(d1.clone()),
                "d2" => ColData::Duration(d2.clone()),
                other => panic!("unknown column `{other}`"),
            }
        };
        for (expr, names) in [
            ("t1 - t2 > d1", &["t1", "t2", "d1"][..]),
            ("t2 - t1 <= d2", &["t2", "t1", "d2"][..]),
            ("t1 + d1 > t2", &["t1", "d1", "t2"][..]),
            ("t1 - d1 < t2", &["t1", "d1", "t2"][..]),
            ("d1 + d2 > d1", &["d1", "d2"][..]),
            ("d1 - d2 < d1", &["d1", "d2"][..]),
            // A folded literal as one operand, and a two-operation chain.
            ("t1 + duration('24h') > t2", &["t1", "t2"][..]),
            ("(t1 - t2) + d1 > d2", &["t1", "t2", "d1", "d2"][..]),
        ] {
            let cols: Vec<(&str, ColData)> = names.iter().map(|nm| (*nm, pick(nm))).collect();
            check_batch_f(expr, &cols);
        }
    }

    /// The domain narrowing is what makes the machine and chrono agree, so it
    /// has to be real: a column outside it must REFUSE, not answer.
    #[test]
    fn temporal_arithmetic_refuses_a_batch_outside_its_domain() {
        use super::batch::{Batch, BatchError, BatchProgram, ColumnRef};
        let schema: Schema = [
            ("a".to_string(), ValType::Duration),
            ("b".to_string(), ValType::Duration),
        ]
        .into_iter()
        .collect();
        let program = BatchProgram::compile("a + b > a", &schema).expect("temporal add lowers");
        let bound = program
            .lowered()
            .temporal_bound
            .expect("one add, one bound");
        assert_eq!(bound, i64::MAX / 2, "one operation combines two operands");

        // Two durations that each FIT i64 nanoseconds but whose sum does not.
        // chrono answers this; the machine would overflow, so it must refuse.
        let big = vec![bound + 10];
        let batch = Batch::new(1)
            .column("a", ColumnRef::Duration(&big))
            .column("b", ColumnRef::Duration(&big));
        assert!(
            matches!(
                program.bind(&batch),
                Err(BatchError::TemporalOutOfDomain { .. })
            ),
            "a value past the bound must refuse the batch"
        );
        // And the tree-walker really does answer it, which is why refusing —
        // rather than trapping, which means "the walker raised" — is right.
        let mut ctx = Context::default();
        let one = chrono::Duration::nanoseconds(bound + 10);
        ctx.add_variable_from_value("a", Value::Duration(one));
        ctx.add_variable_from_value("b", Value::Duration(one));
        let walker = Program::compile("a + b > a").unwrap();
        assert_eq!(walker.execute(&ctx).unwrap(), Value::Bool(true));

        // Inside the bound the same expression answers.
        let ok = vec![bound / 2];
        let batch = Batch::new(1)
            .column("a", ColumnRef::Duration(&ok))
            .column("b", ColumnRef::Duration(&ok));
        assert_eq!(program.bind(&batch).unwrap().sum().unwrap(), Value::Int(1));
    }

    #[test]
    fn batch_int_in_set() {
        // `x in [literals]` unrolls to an OR-chain of equalities over the int
        // register file, bit-exact across the clean / interp / compiled tiers.
        let n = 3000;
        let age = gen_i64(n, 0x5151_2626_3737_4848, 0, 80);
        check_batch_f("age in [18, 21, 65]", &[("age", ColData::Int(age.clone()))]);
        // Two membership sets combined with `&&`.
        let dept = gen_i64(n, 0xA1B2_C3D4_E5F6_0718, 0, 6);
        check_batch_f(
            "age in [18, 21] && dept in [1, 2, 3]",
            &[("age", ColData::Int(age)), ("dept", ColData::Int(dept))],
        );
    }

    #[test]
    fn batch_string_in_set() {
        // String membership: an OR-chain of id equalities. Literals in
        // the set feed the injectivity check alongside the column values.
        let n = 3000;
        let roles = ["admin", "user", "guest", "root", "auditor"];
        let role = gen_str(n, 0x3131_4242_5353_6464, &roles);
        check_batch_str(
            "role in [\"admin\", \"root\"]",
            &[("role", ColData::Str(role.clone()))],
        );
        // A set element absent from the column still compiles (never matches).
        check_batch_str(
            "role in [\"admin\", \"superuser\"]",
            &[("role", ColData::Str(role))],
        );
    }

    #[test]
    fn in_empty_and_bails() {
        // `x in []` is const false (still compiles the loop); a heterogeneous
        // element or a non-literal container bails to the tree-walker.
        let n = 3000;
        let age = gen_i64(n, 0x1212_3434_5656_7878, 0, 80);
        check_batch_f("age in []", &[("age", ColData::Int(age))]);

        let schema: Schema = [("age".to_string(), ValType::Int)].into_iter().collect();
        // An int column tested against a string element: bank mismatch bails.
        let program = Program::compile("age in [\"x\"]").unwrap();
        assert!(
            lower_typed(program.expression(), &schema).is_err(),
            "heterogeneous @in must bail the typed lowering"
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
            &[
                ("price", ColData::Float(price)),
                ("qty", ColData::Float(qty)),
            ],
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
            &[
                ("price", ColData::Float(price)),
                ("qty", ColData::Float(qty)),
            ],
        );
    }

    #[test]
    fn batch_float_col_vs_col() {
        // Float column vs float column comparison driven through the lowerer.
        let n = 3000;
        let a = gen_f64(n, 0xAAAA_5555_AAAA_5555, -1.0, 1.0);
        let b = gen_f64(n, 0xBBBB_4444_BBBB_4444, -1.0, 1.0);
        check_batch_f(
            "a >= b",
            &[("a", ColData::Float(a)), ("b", ColData::Float(b))],
        );
    }

    #[test]
    fn batch_float_arith_policy() {
        // Float arithmetic (FMUL) then a float-const compare.
        let n = 3000;
        let price = gen_f64(n, 0x1111_2222_3333_4444, 0.0, 100.0);
        let qty = gen_f64(n, 0x5555_6666_7777_8888, 0.0, 100.0);
        check_batch_f(
            "price * qty >= 2500.0",
            &[
                ("price", ColData::Float(price)),
                ("qty", ColData::Float(qty)),
            ],
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
            &[
                ("flagged", ColData::Int(flagged)),
                ("price", ColData::Float(price)),
            ],
        );
    }

    #[test]
    fn batch_float_ternary() {
        // Float-armed ternary lowers to a bit-mask FSELECT (int condition,
        // float arms). The blend is over the raw f64 bit patterns, so it is
        // bit-exact against the tree-walker across all three tiers, and the
        // float-valued result feeds the float accumulator.
        let n = 3000;
        let price = gen_f64(n, 0x6E1F_2A3B_4C5D_6E7F, 0.0, 200.0);
        let qty = gen_f64(n, 0x7F6E_5D4C_3B2A_1F0E, 0.0, 100.0);
        // condition compares a float column, both arms are float expressions.
        check_batch_float(
            "price >= 100.0 ? price * 2.0 : qty",
            &[
                ("price", ColData::Float(price.clone())),
                ("qty", ColData::Float(qty.clone())),
            ],
        );
        // a plain float column vs float column arm selection.
        check_batch_float(
            "price >= qty ? price : qty",
            &[
                ("price", ColData::Float(price)),
                ("qty", ColData::Float(qty)),
            ],
        );
    }

    #[test]
    fn batch_float_negate() {
        // Float unary negation (`OP_FNEG`) traces through the two-bank mainloop.
        // `-` flips the f64 sign bit (always exact), so a negated column and a
        // negated ternary arm stay bit-exact across the clean / interp / compiled
        // tiers, and the compiled tier must trace the loop (the negate arm no
        // longer aborts the trace).
        let n = 3000;
        let price = gen_f64(n, 0x2B3C_4D5E_6F70_8191, -200.0, 200.0);
        let qty = gen_f64(n, 0x9182_7364_5546_3728, 0.0, 100.0);
        // Bare float negation.
        check_batch_float("-price", &[("price", ColData::Float(price.clone()))]);
        // Negation inside a float ternary arm (FSELECT blends a negated value).
        check_batch_float(
            "price >= qty ? -price : qty",
            &[
                ("price", ColData::Float(price)),
                ("qty", ColData::Float(qty)),
            ],
        );
    }

    #[test]
    fn batch_uint_compare() {
        // Full-range u64 columns compared unsigned. About half the rows have the
        // high bit set, so a signed compare would count differently; the oracle
        // (Value::UInt orders unsigned) pins the unsigned semantics across the
        // clean / interp / compiled tiers.
        let n = 3000;
        let a = gen_u64_bits(n, 0x51ED_2701_AABB_CCDD);
        let b = gen_u64_bits(n, 0xC0FF_EE00_1234_5678);
        // Column vs a uint constant at the sign boundary (2^63), all four
        // orderings (the `>`/`>=` forms exercise the operand-swap path).
        for expr in [
            "a >= 9223372036854775808u",
            "a > 9223372036854775808u",
            "a <= 9223372036854775808u",
            "a < 9223372036854775808u",
        ] {
            check_batch_f(expr, &[("a", ColData::UInt(a.clone()))]);
        }
        // Column vs column, ordering plus eq/ne (uint eq/ne reuse the int ops).
        for expr in ["a < b", "a <= b", "a > b", "a >= b", "a == b", "a != b"] {
            check_batch_f(
                expr,
                &[
                    ("a", ColData::UInt(a.clone())),
                    ("b", ColData::UInt(b.clone())),
                ],
            );
        }
    }

    /// The overflow contract, end to end: on a batch where some row's `int`
    /// arithmetic overflows, the tree-walker RAISES, so no sum is a correct
    /// answer and every tier must refuse to produce one.
    ///
    /// This is the batch transposition of PyPy's `int_add_ovf` +
    /// `guard_no_overflow`: the guard exits the compiled trace, the blackhole
    /// resumes into the `None` arm, that arm records the event, and the driver
    /// turns the record into "no result — use the tree-walker", which is what
    /// actually raises. The rows are chosen so the overflow appears LATE, well
    /// after the loop has tier-compiled, so the refusal comes from a guard
    /// failing inside compiled code and not merely from the interpreter tier.
    #[test]
    fn batch_int_overflow_refuses() {
        use super::bytecode::float_bank::COMPILES as COMPILES_F;
        let n = 3000;
        let base_a = gen_i64(n, 0x3141_5926_5358_9793, 1, 1000);
        let base_b = gen_i64(n, 0x2718_2818_2845_9045, 1, 1000);

        let schema: Schema = [
            ("a".to_string(), ValType::Int),
            ("b".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        // Bounded rows everywhere except the tail, whose operands are picked to
        // overflow the operator under test (`a - b` needs a huge NEGATIVE `a`,
        // which the `a + b` pair would not produce).
        for (expr, tail_a, tail_b) in [
            ("a + b", i64::MAX - 1, i64::MAX - 1),
            ("a - b", i64::MIN + 1, i64::MAX - 1),
            ("a * b", i64::MAX / 2, 3),
        ] {
            let mut a = base_a.clone();
            let mut b = base_b.clone();
            a[n - 1] = tail_a;
            b[n - 1] = tail_b;
            let program = Program::compile(expr).unwrap();
            let lowered = lower_typed(program.expression(), &schema)
                .unwrap_or_else(|e| panic!("lower_typed `{expr}`: {e}"));

            // The oracle really does raise on the offending row — without this
            // the test would be asserting a refusal nobody asked for.
            let mut ctx = Context::default();
            ctx.add_variable_from_value("a", a[n - 1]);
            ctx.add_variable_from_value("b", b[n - 1]);
            assert!(
                program.execute(&ctx).is_err(),
                "`{expr}` must overflow the tree-walker on the tail row"
            );

            let data = [ColData::Int(a.clone()), ColData::Int(b.clone())];
            let columns: Vec<Column> = data.iter().map(|d| d.column()).collect();
            assert_eq!(
                clean_batch_sum_f(&lowered, &columns, n),
                None,
                "clean tier must refuse `{expr}`"
            );
            assert_eq!(
                eval_batch_sum_f(&lowered, &columns, n, u32::MAX),
                None,
                "jit-off tier must refuse `{expr}`"
            );
            // Start the compiled tier cold: the driver persists across calls, so a
            // loop an earlier case already compiled would not compile again.
            super::bytecode::float_bank::reset_persistent_state();
            let before = COMPILES_F.load(Ordering::Relaxed);
            assert_eq!(
                eval_batch_sum_f(&lowered, &columns, n, 8),
                None,
                "jit-on tier must refuse `{expr}`"
            );
            assert!(
                COMPILES_F.load(Ordering::Relaxed) > before,
                "`{expr}` must tier-compile so the refusal comes from a compiled guard"
            );
        }
    }

    /// The flip side: a batch whose rows all stay in range must still answer.
    /// `OP_*_OVF` replaced the plain wrapping ops on every user `+ - *`, so this
    /// pins that the guard is free when it does not fire.
    #[test]
    fn batch_int_arith_in_range_still_answers() {
        let n = 3000;
        let a = gen_i64(n, 0x0bad_c0de_dead_beef, -1_000_000, 1_000_000);
        let b = gen_i64(n, 0x00c0_ffee_0bad_f00d, -1_000_000, 1_000_000);
        for expr in ["a + b", "a - b", "a * b", "a * b + a - b"] {
            check_batch_f(
                expr,
                &[
                    ("a", ColData::Int(a.clone())),
                    ("b", ColData::Int(b.clone())),
                ],
            );
        }
    }

    #[test]
    fn batch_uint_arithmetic() {
        // uint `+ - *` are checked against the UNSIGNED bounds. Reusing the
        // signed `Int*Ovf` guard would be wrong in BOTH directions, so both are
        // pinned here: this test's operands are all above `i64::MAX`, where a
        // signed guard would refuse every row, and `uint_arith_refuses` covers
        // the reverse (`0u - 1u`, fine signed and not unsigned).
        let n = 2000;
        let a: Vec<i64> = gen_i64(n, 0x0bad_c0de_1234_5678, 0, 1_000_000_000)
            .into_iter()
            .map(|v| (v as u64 + (1u64 << 63)) as i64)
            .collect();
        let b = gen_i64(n, 0xfeed_face_8765_4321, 0, 1_000_000);
        // A full-width sum would overflow the machine's plain `OP_ADD`
        // reduction, so reduce each row first.
        for expr in [
            "(a + b) % 1000000007u",
            "(a - b) % 1000000007u",
            "a + b >= a",
            "a - b <= a",
        ] {
            check_batch_f(
                expr,
                &[
                    ("a", ColData::UInt(a.clone())),
                    ("b", ColData::UInt(b.clone())),
                ],
            );
        }
        // Products that exceed `i64::MAX` but still fit a `u64` — the case the
        // unsigned `uint_mul_high` test accepts and a signed one would not.
        let c = gen_i64(n, 0x1357_9bdf_2468_ace0, 3_000_000_000, 4_000_000_000);
        let d = gen_i64(n, 0x2468_ace0_1357_9bdf, 3_000_000_000, 4_000_000_000);
        for expr in ["(c * d) % 1000000007u", "c * d > 9223372036854775807u"] {
            check_batch_f(
                expr,
                &[
                    ("c", ColData::UInt(c.clone())),
                    ("d", ColData::UInt(d.clone())),
                ],
            );
        }
    }

    #[test]
    fn uint_arith_refuses() {
        // The unsigned bounds, one row each: a carry out of bit 63, a borrow
        // below zero, and a product wider than 64 bits. `0u - 1u` is the
        // direction a signed guard would have waved through.
        let n = 240;
        let ones = vec![1i64; n];
        let filler: Vec<i64> = (0..n).map(|i| i as i64 + 10).collect();
        let mut carries = filler.clone();
        carries[200] = -1; // u64::MAX
        let mut borrows = filler.clone();
        borrows[200] = 0;
        let mut wide = filler.clone();
        wide[200] = 1i64 << 40;
        let mut wide_peer = filler;
        wide_peer[200] = 1i64 << 40;
        for (expr, a, b) in [
            ("a + b", carries, ones.clone()),
            ("a - b", borrows, ones),
            ("a * b", wide, wide_peer),
        ] {
            check_batch_f_refuses(expr, &[("a", ColData::UInt(a)), ("b", ColData::UInt(b))]);
        }
    }

    #[test]
    fn uint_negate_bails() {
        // `-uint` is NoSuchOverload in CEL (Negator is int/double only), so the
        // typed lowering must bail rather than emit a float-bank OP_FNEG for a
        // uint operand that lives in the int register file.
        let schema: Schema = [("a".to_string(), ValType::UInt)].into_iter().collect();
        let program = Program::compile("-a").unwrap();
        assert!(
            lower_typed(program.expression(), &schema).is_err(),
            "`-a` on a uint column must bail the typed lowering"
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
        // Float modulo, mixed int/float arithmetic, and a mixed-bank ternary arm
        // still bail. (A float-valued top-level result now compiles into a float
        // accumulator — see `batch_float_aggregate`; a mixed int/float comparison
        // widens via cast_int_to_float — see `batch_mixed_col_compare`; a
        // same-bank float ternary lowers to FSELECT — see `batch_float_ternary`.)
        let schema: Schema = [
            ("p".to_string(), ValType::Float),
            ("q".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in ["p % 2.0 >= 1.0", "p + q", "p >= 1.0 ? p : q"] {
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
            &[
                ("price", ColData::Float(price)),
                ("level", ColData::Int(level)),
            ],
        );
        let price2 = gen_f64(n, 0x4040_4040_4040_4040, 0.0, 10.0);
        let level2 = gen_i64(n, 0x5050_5050_5050_5050, 0, 10);
        check_batch_f(
            "level < price",
            &[
                ("level", ColData::Int(level2)),
                ("price", ColData::Float(price2)),
            ],
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
            &[
                ("price", ColData::Float(price.clone())),
                ("qty", ColData::Float(qty.clone())),
            ],
        );
        check_batch_f(
            "100 <= price && 50 > qty",
            &[
                ("price", ColData::Float(price)),
                ("qty", ColData::Float(qty)),
            ],
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
        // list-returning / string-arith / member-fn on a non-string column /
        // mixed int-float arithmetic / list-valued comprehension (`map` builds a
        // list) / comprehension over a non-list column all fall back to the
        // tree-walker. Every path is undeclared here, so the schema is empty and
        // each slot defaults to the int bank.
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
                lower_typed(program.expression(), &Schema::new()).is_err(),
                "`{expr}` must be rejected as out of subset"
            );
        }
    }

    /// A list-of-records column and the batch-wide flattened field it declares.
    fn record_list(lens: Vec<i64>, fields: Vec<(Option<&'static str>, ColData)>) -> ListCol {
        ListCol { lens, fields }
    }

    #[test]
    fn batch_list_record_field() {
        // The headline shape: a comprehension over a RUNTIME-length list with a
        // field access on the loop variable. The element count is a column
        // value, so there is no green trip count and no unroll — the lowering
        // emits a real inner loop whose back-edge is its own `can_enter_jit`
        // point, and the element column is read at `(offset + j) * 8`.
        let n = 400;
        let lens = gen_lens(n, 0x11A5_7C01_D0DE_0001, 3);
        let total = lens.iter().sum::<i64>() as usize;
        let price = gen_i64(total, 0x11A5_7C01_D0DE_0002, 0, 30);
        let items = || {
            vec![(
                "items",
                record_list(
                    lens.clone(),
                    vec![(Some("price"), ColData::Int(price.clone()))],
                ),
            )]
        };
        for expr in [
            "items.all(i, i.price > 10)",
            "items.exists(i, i.price > 25)",
            "items.exists_one(i, i.price == 7)",
            // The derived length column, shared with the comprehension's own
            // trip count.
            "size(items)",
            "size(items) > 1 && items.all(i, i.price > 5)",
            // Arithmetic on the element inside the inner loop.
            "items.all(i, i.price * 2 - 1 > 10)",
        ] {
            check_batch_list(expr, &[], &items());
        }
    }

    #[test]
    fn batch_list_scalar_elements() {
        // A list of bare scalars: the loop variable IS the element, so it
        // resolves to the unnamed element column `nums[]`.
        let n = 400;
        let lens = gen_lens(n, 0x5CA1_A200_0001, 4);
        let total = lens.iter().sum::<i64>() as usize;
        let nums = gen_i64(total, 0x5CA1_A200_0002, -20, 20);
        let cols = || {
            vec![(
                "nums",
                record_list(lens.clone(), vec![(None, ColData::Int(nums.clone()))]),
            )]
        };
        for expr in [
            "nums.exists(i, i > 5)",
            "nums.all(i, i > -100)",
            "nums.all(i, i % 2 == 0)",
            "size(nums) == 0",
        ] {
            check_batch_list(expr, &[], &cols());
        }
    }

    #[test]
    fn batch_list_multi_field_and_row_column() {
        // Two element columns plus an ordinary ROW column read inside the inner
        // loop: the row load stays in the outer prologue and the element loads
        // stay in the inner loop, and the two indices must not be confused.
        let n = 400;
        let lens = gen_lens(n, 0xF1E1_D500_0001, 3);
        let total = lens.iter().sum::<i64>() as usize;
        let price = gen_i64(total, 0xF1E1_D500_0002, 0, 20);
        let qty = gen_i64(total, 0xF1E1_D500_0003, 1, 5);
        let limit = gen_i64(n, 0xF1E1_D500_0004, 0, 15);
        let items = || {
            vec![(
                "items",
                record_list(
                    lens.clone(),
                    vec![
                        (Some("price"), ColData::Int(price.clone())),
                        (Some("qty"), ColData::Int(qty.clone())),
                    ],
                ),
            )]
        };
        for expr in [
            "items.all(i, i.price * i.qty > 10)",
            "items.exists(i, i.price > limit)",
            "items.all(i, i.price > limit) && limit > 5",
        ] {
            check_batch_list(expr, &[("limit", ColData::Int(limit.clone()))], &items());
        }
    }

    #[test]
    fn batch_list_float_field() {
        // A `double` element field rides the float bank: the element load is an
        // `OP_COL_LOAD_F` at the inner index, and the comparison crosses banks
        // exactly as a row-column float compare does.
        let n = 400;
        let lens = gen_lens(n, 0xF10A_7000_0001, 3);
        let total = lens.iter().sum::<i64>() as usize;
        let amount = gen_f64(total, 0xF10A_7000_0002, 0.0, 4.0);
        let costs = || {
            vec![(
                "costs",
                record_list(
                    lens.clone(),
                    vec![(Some("amount"), ColData::Float(amount.clone()))],
                ),
            )]
        };
        for expr in [
            "costs.all(i, i.amount > 1.5)",
            "costs.exists(i, i.amount < 0.5)",
        ] {
            check_batch_list(expr, &[], &costs());
        }
    }

    #[test]
    fn batch_list_two_comprehensions_over_one_list() {
        // Two comprehensions over the SAME list must each get their own element
        // register: they read the column at their own inner index, so sharing
        // one register would let the second loop see the first loop's last
        // element. The rows below mix lengths, so a shared register would
        // disagree with the walker.
        let n = 400;
        let lens = gen_lens(n, 0x2C0F_0001, 3);
        let total = lens.iter().sum::<i64>() as usize;
        let price = gen_i64(total, 0x2C0F_0002, 0, 30);
        check_batch_list(
            "items.all(i, i.price > 0) && items.exists(i, i.price > 20)",
            &[],
            &[(
                "items",
                record_list(lens, vec![(Some("price"), ColData::Int(price))]),
            )],
        );
    }

    #[test]
    fn list_arith_refuses() {
        // Overflow INSIDE the inner loop must reach the driver: the trap flag is
        // set on one element of one row, survives both loops, and is published
        // once after the row loop. The walker raises there, so no sum is the
        // right answer.
        let n = 240;
        let lens = vec![1i64; n];
        let mut price = vec![3i64; n];
        // Late enough that the loops are long since compiled when the guard
        // finally exits.
        price[200] = i64::MAX / 2;
        check_batch_list_refuses(
            "items.all(i, i.price * 4 > 0)",
            &[],
            &[(
                "items",
                record_list(lens, vec![(Some("price"), ColData::Int(price))]),
            )],
        );
    }

    #[test]
    fn list_lowering_bails() {
        // The subset boundary around runtime lists.
        let schema: Schema = [
            ("items[].price".to_string(), ValType::Int),
            ("groups[].n".to_string(), ValType::Int),
            ("x".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in [
            // A list of lists needs per-ELEMENT offsets; a flat row column
            // cannot express those, and one level of nesting is also what keeps
            // the inner loop body straight-line.
            "items.all(i, groups.exists(g, g.n > i.price))",
            // A list is not a value on this machine — it is a (size, offset)
            // pair plus element columns — so it can never reach a register.
            "items == items",
            // A scalar column is not iterable, list or not.
            "x.all(y, y > 0)",
            // `map` / `filter` accumulate a list.
            "items.map(i, i.price)",
            "items.filter(i, i.price > 1)",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering"
            );
        }
        // Two lists SIDE BY SIDE are in subset — one inner loop each. This pins
        // the nesting bail above to nesting, not to "more than one list".
        for expr in [
            "items.all(i, i.price > x) && groups.exists(g, g.n > 0)",
            "items.all(i, i.price > 0)",
            // Both arms of a ternary reduce to the derived length columns.
            "x > 0 ? size(items) : size(groups)",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_ok(),
                "`{expr}` must lower"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Lowering <-> tree-walker parity sweep
    // ---------------------------------------------------------------------
    //
    // The per-feature tests above check expressions someone thought to write.
    // `1 && 2` was not one of them: it lowered, the machine answered `3`, and
    // the tree-walker raises `NoSuchOverload` — a JIT-only answer that stood
    // because no test crossed an `int` operand with a boolean operator.
    //
    // This sweep crosses the whole operand matrix with the whole operator set
    // mechanically. It asserts ONE property, the only one that matters for a
    // drop-in replacement:
    //
    //   whenever the lowering ACCEPTS an expression and the machine ANSWERS,
    //   the answer equals the tree-walker's — and the tree-walker must have
    //   had an answer to give.
    //
    // Declining is always allowed (the caller falls back to the walker), and so
    // is refusing mid-batch (the trap flag reaching the driver). Only answering
    // differently, or answering at all where the walker raises, is a defect.

    /// What one sweep expression did. Only [`SweepVerdict::Agreed`] compares
    /// values; the other two are legal outcomes counted for coverage, so a
    /// change that quietly declines everything shows up as a collapsed census
    /// rather than as a green run.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum SweepVerdict {
        /// The lowering refused the expression: it stays with the tree-walker.
        Declined,
        /// The machine ran but trapped, so the batch driver returned no answer.
        Refused,
        /// The machine answered and the answer equals the tree-walker's.
        Agreed,
    }

    /// Every operand the sweep crosses: two columns of each declared type so a
    /// same-bank binary op has two distinct operands, plus one literal of each
    /// type so literal folding is covered on both sides of every operator.
    ///
    /// Values are chosen so the tree-walker itself never raises on the numeric
    /// operators: no zero divisor, and magnitudes far from the i64/u64 bounds.
    /// Overflow and division-by-zero have their own refusal tests; mixing them
    /// in here would push most of the matrix down the `Refused` path and stop
    /// the sweep from comparing values.
    /// One literal of each type, so literal folding is covered on both sides of
    /// every operator.
    const LITERALS: [&str; 5] = ["6", "6u", "1.5", "true", "\"ab\""];

    fn sweep_operands() -> (Vec<&'static str>, Vec<(&'static str, ColData)>) {
        let cols: Vec<(&'static str, ColData)> = vec![
            ("i", ColData::Int(vec![7, -3, 11, 2])),
            ("j", ColData::Int(vec![2, 5, -1, 4])),
            ("u", ColData::UInt(vec![3, 9, 1, 6])),
            ("v", ColData::UInt(vec![2, 4, 8, 5])),
            ("f", ColData::Float(vec![2.5, -0.5, 1.25, 3.0])),
            ("g", ColData::Float(vec![0.5, 4.0, -2.0, 1.5])),
            ("b", ColData::Bool(vec![1, 0, 1, 0])),
            ("c", ColData::Bool(vec![1, 1, 0, 0])),
            (
                "s",
                ColData::Str(
                    ["ab", "cd", "ab", "ef"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            ),
            (
                "r",
                ColData::Str(
                    ["ab", "zz", "cd", "ef"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            ),
            (
                "t",
                ColData::Timestamp(vec![
                    1_700_000_000_000_000_000,
                    1_600_000_000_000_000_000,
                    1_800_000_000_000_000_000,
                    0,
                ]),
            ),
            (
                "w",
                ColData::Timestamp(vec![
                    1_700_000_000_000_000_000,
                    1_650_000_000_000_000_000,
                    -1_000_000_000,
                    1_000_000_000,
                ]),
            ),
            (
                "d",
                ColData::Duration(vec![1_000_000_000, 2_500_000_000, -500_000_000, 0]),
            ),
            (
                "e",
                ColData::Duration(vec![3_000_000_000, -1_000_000_000, 1_000_000_000, 7]),
            ),
        ];
        let mut srcs: Vec<&'static str> = cols.iter().map(|(n, _)| *n).collect();
        srcs.extend(LITERALS);
        (srcs, cols)
    }

    /// Run one sweep expression through both evaluators and check the parity
    /// property. Panics with the expression on any divergence.
    fn sweep_case(expr_src: &str, cols: &[(&str, ColData)]) -> SweepVerdict {
        let n = cols[0].1.len();

        let program =
            Program::compile(expr_src).unwrap_or_else(|e| panic!("parse `{expr_src}`: {e:?}"));
        let schema: Schema = cols
            .iter()
            .map(|(nm, d)| (nm.to_string(), d.ty()))
            .collect();
        let lowered = match lower_typed(program.expression(), &schema) {
            Ok(l) => l,
            Err(_) => return SweepVerdict::Declined,
        };
        // Lowering and reduction refuse separately; this harness runs the batch
        // sum, so it needs both.
        if lowered.sum_reducible().is_err() {
            return SweepVerdict::Declined;
        }

        // Columns in SLOT order, materializing the two derived kinds the schema
        // does not declare: a string column's `i64` ids, and a
        // `size(<string>)` length column.
        let declared: HashMap<&str, &ColData> = cols.iter().map(|(nm, d)| (*nm, d)).collect();
        let mut derived: Vec<Vec<i64>> = Vec::new();
        let mut plan: Vec<Result<&ColData, usize>> = Vec::new();
        for slot in &lowered.slots {
            match size_slot_source(&slot.path) {
                Some(src) => {
                    let Some(ColData::Str(c)) = declared.get(src) else {
                        panic!(
                            "`{expr_src}`: size slot `{}` has no string column",
                            slot.path
                        )
                    };
                    derived.push(c.iter().map(|s| s.len() as i64).collect());
                    plan.push(Err(derived.len() - 1));
                }
                None => match declared.get(slot.path.as_str()) {
                    Some(d) => plan.push(Ok(d)),
                    None => panic!("`{expr_src}`: slot `{}` has no column", slot.path),
                },
            }
        }
        let columns: Vec<Column> = plan
            .iter()
            .map(|p| match p {
                Ok(d) => d.column(),
                Err(k) => Column::Int(&derived[*k]),
            })
            .collect();

        // The tree-walker's per-row results, in row order. `None` marks a row it
        // raised on: answering that row at all would be a JIT-only answer.
        let oracle: Vec<Option<Value>> = (0..n)
            .map(|k| program.execute(&row_context(cols, k)).ok())
            .collect();
        let walker_raised = oracle.iter().any(|v| v.is_none());

        let float_result = lowered.result_bank == ValType::Float;
        let got = if float_result {
            super::bytecode::eval_batch_sum_float(&lowered, &columns, n, u32::MAX)
                .map(|f| f.to_bits() as i64)
        } else {
            clean_batch_sum_f(&lowered, &columns, n)
        };
        let Some(got) = got else {
            return SweepVerdict::Refused;
        };
        assert!(
            !walker_raised,
            "`{expr_src}`: the machine answered {got} where the tree-walker raises"
        );

        // Reduce the walker's rows the same way the batch loop does: an int-bank
        // result sums as i64 (a `uint` rides the accumulator as its raw bit
        // pattern), a float result sums as f64 in row order and is compared by
        // BITS — float addition is order-sensitive, so a tolerance would hide
        // exactly the reassociation this is here to catch.
        let want = if float_result {
            let mut acc = 0.0f64;
            for v in oracle.iter().flatten() {
                match v {
                    Value::Float(x) => acc += x,
                    other => panic!("`{expr_src}`: float result vs walker {other:?}"),
                }
            }
            acc.to_bits() as i64
        } else {
            let mut acc = 0i64;
            for v in oracle.iter().flatten() {
                acc += match v {
                    Value::Bool(x) => *x as i64,
                    Value::Int(x) => *x,
                    Value::UInt(x) => *x as i64,
                    other => panic!(
                        "`{expr_src}`: lowered to bank {:?} but the walker answers {other:?}",
                        lowered.result_bank
                    ),
                };
            }
            acc
        };
        assert_eq!(got, want, "`{expr_src}`: machine vs tree-walker");

        // The majit interpreter tier must reproduce the clean tier's answer.
        // The COMPILED tier is deliberately not run here: it would trace and
        // compile once per expression, and what this sweep is testing is the
        // LOWERING's type rules, which are the same words on every tier. The
        // compiled tier is covered per feature by the `check_batch_*` helpers.
        if !float_result {
            assert_eq!(
                eval_batch_sum_f(&lowered, &columns, n, u32::MAX),
                Some(got),
                "`{expr_src}`: majit interp tier vs clean tier"
            );
        }
        SweepVerdict::Agreed
    }

    /// The tree-walker's answers for the cross-type cases the lowering folds.
    ///
    /// None of these follow from the operator table: `1 == 1u` is `true`
    /// (`int`/`uint`/`double` are ONE equality class and compare numerically),
    /// `1 == "ab"` is `false` rather than an error (two different classes are
    /// never equal), and `1 || true` is `true` though `1 || false` is
    /// `NoSuchOverload` (the logical operators absorb the other operand whole).
    /// The lowering encodes each of these. Pin them here so a change to the
    /// walker's semantics fails loudly instead of leaving the JIT tier as a
    /// second, quietly disagreeing implementation.
    #[test]
    fn cross_type_ground_truth() {
        let cases: [(&str, Result<Value, ()>); 30] = [
            // One numeric class, compared numerically — not by bit pattern.
            ("1 == 1u", Ok(Value::Bool(true))),
            ("1 != 1u", Ok(Value::Bool(false))),
            ("1 < 2u", Ok(Value::Bool(true))),
            ("2u < 1", Ok(Value::Bool(false))),
            ("1u == 1.0", Ok(Value::Bool(true))),
            ("1u < 2.0", Ok(Value::Bool(true))),
            ("2.0 < 1u", Ok(Value::Bool(false))),
            // The int's sign decides across int/uint: every negative int is
            // below every uint, and `u64::MAX` is not `-1` however the bits
            // read. Neither machine compare answers this on its own.
            ("1u < -1", Ok(Value::Bool(false))),
            ("-1 < 1u", Ok(Value::Bool(true))),
            ("18446744073709551615u == -1", Ok(Value::Bool(false))),
            ("18446744073709551615u > 1", Ok(Value::Bool(true))),
            // int/double widens the int, losing precision exactly as `as f64`
            // does — the JIT tier's OP_I2F must not be more exact than this.
            (
                "9007199254740993 == 9007199254740992.0",
                Ok(Value::Bool(true)),
            ),
            // Different classes: equal is false, unequal is true whatever the
            // operands hold, and ordering is an error.
            ("1 == 'ab'", Ok(Value::Bool(false))),
            ("1 != 'ab'", Ok(Value::Bool(true))),
            ("true == 1", Ok(Value::Bool(false))),
            ("1 == true", Ok(Value::Bool(false))),
            (
                "1 == timestamp('2020-01-01T00:00:00Z')",
                Ok(Value::Bool(false)),
            ),
            (
                "timestamp('2020-01-01T00:00:00Z') == duration('1s')",
                Ok(Value::Bool(false)),
            ),
            ("1 < 'ab'", Err(())),
            ("1 < true", Err(())),
            // The logical operators absorb the other operand whole — its value,
            // its errors AND its type — but only the absorbing constant does.
            ("1 || true", Ok(Value::Bool(true))),
            ("true || 1", Ok(Value::Bool(true))),
            ("1 && false", Ok(Value::Bool(false))),
            ("false && 1", Ok(Value::Bool(false))),
            ("1 || false", Err(())),
            ("1 && true", Err(())),
            // Ordering IS defined on strings; a uint arm rides the ternary.
            ("'ab' < 'cd'", Ok(Value::Bool(true))),
            ("true ? 1u : 2u", Ok(Value::UInt(1))),
            // Negation is defined on neither uint nor string.
            ("-1u", Err(())),
            ("-'ab'", Err(())),
        ];
        for (src, want) in cases {
            let program = Program::compile(src).unwrap();
            let got = program.execute(&Context::default()).map_err(|_| ());
            assert_eq!(got, want, "tree-walker ground truth for `{src}`");
        }
    }

    /// Every expression the sweep matrix can build, bucketed by why the
    /// **lowering** declined it — and, of those, how many the tree-walker
    /// ANSWERS.
    ///
    /// A decline only costs coverage where the walker has an answer to give;
    /// where the walker raises, declining is the correct outcome, and counting
    /// it would reward the lowering for refusing expressions CEL rejects. This
    /// asserts the answerable declines do not GROW: covering another pairing
    /// lowers the number, and a regression that starts refusing legal
    /// expressions raises it. The breakdown rides the failure message, so a
    /// break names the family that moved.
    ///
    /// ⚠️Counted separately, and reported either way, is the second refusal:
    /// an expression that LOWERS but whose result the batch loop's sum cannot
    /// consume (`LoweredF::sum_reducible`). That is the reduction's limit, not
    /// the lowering's, and folding the two together would let a lowering gain
    /// look like a reduction gain or hide one behind the other.
    #[test]
    fn coverage_gap_does_not_grow() {
        use std::collections::BTreeMap;
        // What is left of the LOWERING gap: a string id is a rank, so ordering
        // reads it and the four predicates index a bind-time table off it —
        // but a rank still cannot give back the CHARACTERS, so `s + s` and
        // `string(x)` have none to produce.
        //
        // Temporal arithmetic (`d + d`, `t + d`, `t - t`) declines at the
        // operator: the operands are i64 nanoseconds but the walker's chrono
        // arithmetic is wider than i64 nanoseconds, so a machine add can
        // overflow where the walker answers. Covering it needs a bind-time
        // domain check, not just an opcode.
        const GAP_CEILING: usize = 25;

        let (srcs, cols) = sweep_operands();
        let schema: Schema = cols
            .iter()
            .map(|(nm, d)| (nm.to_string(), d.ty()))
            .collect();
        let rows = cols[0].1.len();

        let mut exprs: Vec<String> = Vec::new();
        for a in &srcs {
            for b in &srcs {
                for op in [
                    "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||",
                ] {
                    exprs.push(format!("{a} {op} {b}"));
                }
                exprs.push(format!("{a} ? {b} : {b}"));
            }
            exprs.push(format!("!{a}"));
            exprs.push(format!("-{a}"));
            // The operator matrix says nothing about CEL's FUNCTION surface,
            // and the two are separate coverage questions: the conversions and
            // the string methods each reach the lowering by a different door.
            // Applying every one to every operand leaves the walker to say
            // which pairings have an answer, exactly as above.
            for f in ["int", "uint", "double", "string", "size"] {
                exprs.push(format!("{f}({a}) == {f}({a})"));
            }
            for m in ["startsWith", "endsWith", "contains", "matches"] {
                exprs.push(format!("{a}.{m}('ab')"));
            }
            for g in ["getFullYear", "getHours", "getDayOfWeek"] {
                exprs.push(format!("{a}.{g}() > 1"));
            }
        }

        let mut gap: BTreeMap<String, (usize, String)> = BTreeMap::new();
        let mut gap_total = 0usize;
        // The second refusal, tracked apart: lowered, but not something the
        // loop's sum can accumulate.
        let mut unreduced: BTreeMap<String, (usize, String)> = BTreeMap::new();
        let mut unreduced_total = 0usize;
        for expr in &exprs {
            let program = Program::compile(expr).unwrap();
            let lowered = lower_typed(program.expression(), &schema);
            // Only a decline the walker ANSWERS costs anything.
            if (0..rows).any(|k| program.execute(&row_context(&cols, k)).is_err()) {
                continue;
            }
            let reason = match &lowered {
                Err(e) => {
                    gap_total += 1;
                    &mut gap
                }
                .entry(e.reason.clone()),
                Ok(l) => match l.sum_reducible() {
                    Ok(()) => continue,
                    Err(e) => {
                        unreduced_total += 1;
                        unreduced.entry(e.reason)
                    }
                },
            };
            let slot = reason.or_insert((0, expr.clone()));
            slot.0 += 1;
        }

        let render = |m: BTreeMap<String, (usize, String)>| {
            let mut buckets: Vec<(usize, String, String)> =
                m.into_iter().map(|(k, (n, ex))| (n, k, ex)).collect();
            buckets.sort_by_key(|(n, _, _)| std::cmp::Reverse(*n));
            buckets
                .iter()
                .map(|(n, reason, ex)| format!("\n  {n:5}  {reason}   e.g. `{ex}`"))
                .collect::<String>()
        };
        let breakdown = render(gap);
        let unreduced_breakdown = render(unreduced);
        assert!(
            gap_total <= GAP_CEILING,
            "{gap_total} of {} sweep expressions are answered by the tree-walker \
             and declined by the LOWERING (ceiling {GAP_CEILING}):{breakdown}\n\
             \n{unreduced_total} more lower but are not sum-reducible:\
             {unreduced_breakdown}",
            exprs.len(),
        );
    }

    /// Comparisons that cross the int/uint and uint/double banks, over operands
    /// the sweep matrix's small numbers never reach.
    ///
    /// The int/uint pairing is where a machine compare is simply wrong in both
    /// directions: signed reads `u64::MAX` as `-1`, unsigned reads `-1` as
    /// `u64::MAX`, and the tree-walker answers neither. Every row here has an
    /// operand on the far side of `i64::MAX` or below zero, so a lowering that
    /// picked one machine compare and hoped fails on it.
    #[test]
    fn cross_bank_numeric_comparison_at_the_boundaries() {
        let cols: Vec<(&'static str, ColData)> = vec![
            ("i", ColData::Int(vec![-1, 0, 1, i64::MIN])),
            ("j", ColData::Int(vec![i64::MAX, -7, 0, 3])),
            (
                "u",
                ColData::UInt(vec![u64::MAX as i64, 0, 1, (i64::MAX as u64 + 1) as i64]),
            ),
            ("f", ColData::Float(vec![-1.0, 0.0, 1.0, 9.3e18])),
        ];
        for op in ["==", "!=", "<", "<=", ">", ">="] {
            for (a, b) in [
                ("i", "u"),
                ("u", "i"),
                ("j", "u"),
                ("u", "j"),
                ("u", "f"),
                ("f", "u"),
                ("i", "f"),
                ("f", "i"),
            ] {
                let expr = format!("{a} {op} {b}");
                assert_eq!(
                    sweep_case(&expr, &cols),
                    SweepVerdict::Agreed,
                    "`{expr}` must lower and agree with the tree-walker"
                );
            }
            // An int LITERAL against a uint settles the sign while lowering:
            // non-negative leaves the bare unsigned compare, negative settles
            // the comparison outright. Both must still answer what the walker
            // answers, including for uints above `i64::MAX`.
            for lit in ["0", "1", "-1", "-9223372036854775808"] {
                for expr in [format!("u {op} {lit}"), format!("{lit} {op} u")] {
                    assert_eq!(
                        sweep_case(&expr, &cols),
                        SweepVerdict::Agreed,
                        "`{expr}` must lower and agree with the tree-walker"
                    );
                }
            }
        }
    }

    /// Two operands of different type classes are never equal and always
    /// unequal, so `==`/`!=` fold to a constant instead of declining — but the
    /// operands still evaluate, so a row either side traps on still refuses.
    #[test]
    fn cross_class_equality_folds_but_still_evaluates_its_operands() {
        let cols: Vec<(&'static str, ColData)> = vec![
            ("i", ColData::Int(vec![7, -3, 11, 2])),
            ("j", ColData::Int(vec![2, 5, -1, 4])),
            ("b", ColData::Bool(vec![1, 0, 1, 0])),
            (
                "s",
                ColData::Str(
                    ["ab", "cd", "ab", "ef"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            ),
            ("t", ColData::Timestamp(vec![1, 2, 3, 4])),
            ("d", ColData::Duration(vec![1, 2, 3, 4])),
        ];
        for (a, b) in [
            ("i", "b"),
            ("i", "s"),
            ("i", "t"),
            ("b", "s"),
            ("t", "d"),
            ("s", "t"),
        ] {
            for op in ["==", "!="] {
                for expr in [format!("{a} {op} {b}"), format!("{b} {op} {a}")] {
                    assert_eq!(
                        sweep_case(&expr, &cols),
                        SweepVerdict::Agreed,
                        "`{expr}` folds to a constant and must match the walker"
                    );
                }
            }
            // Ordering across classes has no answer to fold to.
            for op in ["<", "<=", ">", ">="] {
                let expr = format!("{a} {op} {b}");
                assert_eq!(
                    sweep_case(&expr, &cols),
                    SweepVerdict::Declined,
                    "`{expr}` is NoSuchOverload and must decline"
                );
            }
        }
        // The fold does not skip the operands: `i / 0` still traps the row it
        // would have compared, exactly as the walker raises on it.
        assert_eq!(
            sweep_case("(i / (j - j)) == b", &cols),
            SweepVerdict::Refused,
            "the constant answer must not outrank the division the row still performs"
        );
    }

    /// `||` answers `true` and `&&` answers `false` as soon as one operand is
    /// that literal, whatever the others are — CEL absorbs the other operand's
    /// value, errors and type. Without the absorbing literal the type error
    /// stands.
    #[test]
    fn logical_operators_absorb_the_other_operand() {
        let cols: Vec<(&'static str, ColData)> = vec![
            ("i", ColData::Int(vec![7, -3, 11, 2])),
            ("b", ColData::Bool(vec![1, 0, 1, 0])),
            (
                "s",
                ColData::Str(
                    ["ab", "cd", "ab", "ef"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                ),
            ),
        ];
        for expr in [
            "i || true",
            "true || i",
            "s || true",
            "i && false",
            "false && i",
            "b || true",
            "b && false",
        ] {
            assert_eq!(
                sweep_case(expr, &cols),
                SweepVerdict::Agreed,
                "`{expr}` is absorbed by its literal and must match the walker"
            );
        }
        for expr in ["i || false", "false || i", "i && true", "true && i"] {
            assert_eq!(
                sweep_case(expr, &cols),
                SweepVerdict::Declined,
                "`{expr}` has no absorbing literal and stays a type error"
            );
        }
    }

    /// The numeric conversions, at the bounds where the four casts differ.
    ///
    /// The walker converts with a plain Rust `as`, so every one of them is
    /// TOTAL: the two widenings wrap or lose precision rather than raise, and
    /// the two narrowings truncate toward zero and saturate. That is why none
    /// of them needs a trap guard — and why `uint` cannot borrow the signed
    /// narrowing, which saturates at different bounds.
    #[test]
    fn numeric_conversions_at_the_saturation_bounds() {
        for (src, want) in [
            // int <-> uint reinterpret the same 64 bits, in both directions.
            ("int(18446744073709551615u)", Value::Int(-1)),
            ("int(9223372036854775808u)", Value::Int(i64::MIN)),
            ("uint(-1)", Value::UInt(u64::MAX)),
            // uint -> double goes through `u64`, so it does not turn negative
            // above 2^63; precision is lost the way `as f64` loses it.
            (
                "double(18446744073709551615u)",
                Value::Float(u64::MAX as f64),
            ),
            (
                "double(9007199254740993u)",
                Value::Float(9007199254740992.0),
            ),
            // double -> int and double -> uint saturate at DIFFERENT bounds.
            ("int(1.0e20)", Value::Int(i64::MAX)),
            ("int(-1.0e20)", Value::Int(i64::MIN)),
            ("uint(1.0e20)", Value::UInt(u64::MAX)),
            ("uint(-1.5)", Value::UInt(0)),
            ("uint(10.5)", Value::UInt(10)),
        ] {
            let program = Program::compile(src).unwrap();
            assert_eq!(
                program.execute(&Context::default()).map_err(|_| ()),
                Ok(want),
                "tree-walker ground truth for `{src}`"
            );
        }

        // The same conversions on COLUMN operands, through the machine.
        let cols: Vec<(&'static str, ColData)> = vec![
            ("i", ColData::Int(vec![-1, 0, 7, i64::MIN])),
            (
                "u",
                ColData::UInt(vec![u64::MAX as i64, 0, 7, (i64::MAX as u64 + 1) as i64]),
            ),
            ("f", ColData::Float(vec![-1.5, 0.0, 10.5, 1.0e20])),
        ];
        for expr in [
            "int(u) < 0",
            "uint(i) > 0u",
            "double(u) > 1.5",
            "double(i) > 1.5",
            "int(f) > 0",
            "uint(f) > 0u",
        ] {
            assert_eq!(
                sweep_case(expr, &cols),
                SweepVerdict::Agreed,
                "`{expr}` must lower and agree with the tree-walker"
            );
        }
    }

    /// `list[k]` on a bound LIST column reads the row's OWN element, which
    /// sits at a different place in every row, and refuses the row where the
    /// index is past that row's span.
    ///
    /// Both halves matter: reading `elems[k]` instead of `elems[offset + k]`
    /// would answer a neighbouring row's value, and answering an out-of-range
    /// index at all would be a JIT-only answer — the tree-walker raises there.
    #[test]
    fn constant_index_reads_the_rows_own_list() {
        use super::batch::{Batch, BatchProgram, ColumnRef, Tier};

        // Jagged: row 0 = [5, 6], row 1 = [7], row 2 = [8, 9, 10]. Only row 1
        // is short, so `list[1]` is in range on two rows out of three.
        let lens: Vec<i64> = vec![2, 1, 3];
        let elems: Vec<i64> = vec![5, 6, 7, 8, 9, 10];

        let walker_sum = |src: &str| -> Option<i64> {
            let program = Program::compile(src).unwrap();
            let mut total = 0;
            let mut off = 0usize;
            for len in &lens {
                let n = *len as usize;
                let list: Vec<Value> = elems[off..off + n].iter().map(|v| Value::Int(*v)).collect();
                off += n;
                let mut ctx = Context::default();
                ctx.add_variable_from_value("items", Value::List(list.into()));
                total += match program.execute(&ctx).ok()? {
                    Value::Int(i) => i,
                    Value::Bool(b) => b as i64,
                    other => panic!("`{src}`: unexpected {other:?}"),
                };
            }
            Some(total)
        };

        let schema: Schema = [("items[]".to_string(), ValType::Int)]
            .into_iter()
            .collect();
        for src in [
            "items[0]",
            "items[1]",
            "items[2]",
            "items[0] > 6",
            "items[0] + items[1]",
        ] {
            let program = BatchProgram::compile(src, &schema)
                .unwrap_or_else(|e| panic!("lower `{src}`: {e}"));
            let batch = Batch::new(lens.len()).column(
                "items",
                ColumnRef::List {
                    lens: &lens,
                    fields: vec![(None, ColumnRef::Int(&elems))],
                },
            );
            let bound = program
                .bind(&batch)
                .unwrap_or_else(|e| panic!("bind `{src}`: {e}"));
            let got = match bound.sum_on(Tier::Clean) {
                Ok(Value::Int(i)) => Some(i),
                Ok(other) => panic!("`{src}`: unexpected {other:?}"),
                Err(_) => None,
            };
            assert_eq!(got, walker_sum(src), "`{src}`: machine vs tree-walker");
            // The tracing interpreter must reach the same verdict.
            assert_eq!(
                bound.sum_on(Tier::Interpreter).ok().map(|v| match v {
                    Value::Int(i) => i,
                    other => panic!("`{src}`: unexpected {other:?}"),
                }),
                walker_sum(src),
                "`{src}`: majit interp tier vs tree-walker"
            );
        }

        // And the COMPILED tier, which is where the bounds check has to
        // survive being traced: it is a forward jump inside the row body, so a
        // trace that recorded only the in-range path would answer the
        // out-of-range rows instead of refusing them. Enough rows to cross the
        // trace threshold many times over, with the short rows spread through
        // so the compiled loop meets both paths.
        let mut long_lens = Vec::new();
        let mut long_elems = Vec::new();
        for r in 0..400i64 {
            let n = if r % 7 == 3 { 1 } else { 3 };
            long_lens.push(n);
            for k in 0..n {
                long_elems.push(r * 10 + k);
            }
        }
        let long_batch = Batch::new(long_lens.len()).column(
            "items",
            ColumnRef::List {
                lens: &long_lens,
                fields: vec![(None, ColumnRef::Int(&long_elems))],
            },
        );
        // `items[0]` is in range on every row; `items[1]` is not on the short
        // ones, so the compiled loop must refuse the whole batch.
        let in_range: i64 = {
            let mut off = 0usize;
            let mut acc = 0;
            for len in &long_lens {
                acc += long_elems[off];
                off += *len as usize;
            }
            acc
        };
        use super::bytecode::float_bank::COMPILES as COMPILES_F;
        use core::sync::atomic::Ordering;

        let program = BatchProgram::compile("items[0]", &schema).unwrap();
        let bound = program.bind(&long_batch).unwrap();
        let before = COMPILES_F.load(Ordering::Relaxed);
        assert_eq!(
            bound.sum_on(Tier::Jit).unwrap(),
            Value::Int(in_range),
            "the compiled tier must read each row's own first element"
        );
        // Otherwise the assertions above only re-ran the tracing interpreter
        // and said nothing about the compiled loop.
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "the loop must actually have been traced and compiled"
        );
        let program = BatchProgram::compile("items[1]", &schema).unwrap();
        let bound = program.bind(&long_batch).unwrap();
        assert!(
            bound.sum_on(Tier::Jit).is_err(),
            "the compiled tier must refuse the rows whose list is too short"
        );
    }

    /// The same read through a list of STRUCTS, where the index picks the
    /// element and the field picks the column. `resolve_path` stops at the
    /// index, so this shape is recognised before it is asked.
    #[test]
    fn constant_index_then_field() {
        use super::batch::{Batch, BatchProgram, ColumnRef, Tier};

        let lens: Vec<i64> = vec![2, 1, 3];
        let price: Vec<i64> = vec![5, 6, 7, 8, 9, 10];
        let schema: Schema = [("items[].price".to_string(), ValType::Int)]
            .into_iter()
            .collect();
        let batch = Batch::new(3).column(
            "items",
            ColumnRef::List {
                lens: &lens,
                fields: vec![(Some("price"), ColumnRef::Int(&price))],
            },
        );

        let program = BatchProgram::compile("items[0].price", &schema)
            .expect("a constant index into a struct list lowers");
        assert_eq!(
            program.bind(&batch).unwrap().sum_on(Tier::Clean).unwrap(),
            Value::Int(5 + 7 + 8),
            "each row's FIRST element, not the buffer's first three",
        );

        // Past the shortest row's span: the row refuses, so the batch does.
        let program = BatchProgram::compile("items[1].price", &schema).unwrap();
        assert!(
            program.bind(&batch).unwrap().sum_on(Tier::Clean).is_err(),
            "row 1 has one element, so `items[1]` is out of range there",
        );
    }

    /// Cross every operand with every binary operator and check parity.
    #[test]
    fn parity_sweep_binary_operators() {
        let (srcs, cols) = sweep_operands();
        let mut census = [0usize; 3];
        for a in &srcs {
            for b in &srcs {
                for op in [
                    "+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||",
                ] {
                    let expr = format!("{a} {op} {b}");
                    let v = sweep_case(&expr, &cols);
                    census[v as usize] += 1;
                }
            }
        }
        // The sweep is only evidence while it still ANSWERS things. These floors
        // are well under the current counts; they exist so a change that makes
        // the lowering decline everything fails here instead of going green.
        assert!(
            census[SweepVerdict::Agreed as usize] > 200,
            "parity sweep census {census:?} — too few compared answers to be evidence"
        );
    }

    /// The unary operators and the ternary, over the same operand matrix.
    #[test]
    fn parity_sweep_unary_and_ternary() {
        let (srcs, cols) = sweep_operands();
        let mut agreed = 0usize;
        for a in &srcs {
            for expr in [format!("!{a}"), format!("-{a}")] {
                if sweep_case(&expr, &cols) == SweepVerdict::Agreed {
                    agreed += 1;
                }
            }
            for b in &srcs {
                let expr = format!("{a} ? {b} : {b}");
                if sweep_case(&expr, &cols) == SweepVerdict::Agreed {
                    agreed += 1;
                }
            }
        }
        assert!(
            agreed > 20,
            "unary/ternary sweep agreed on only {agreed} expressions — too few to be evidence"
        );
    }
}
