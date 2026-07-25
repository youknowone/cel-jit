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
//!     (`OP_ULT`/`OP_ULE`, `>`/`>=` via an operand swap). Division/modulo bail
//!     (the trace IR has no unsigned floordiv/mod).
//!
//! Everything outside this numeric/bool columnar subset — strings, bytes, maps,
//! lists, member/method calls, `in`, timestamps, custom functions — is a
//! structural loss for a batch JIT and returns [`lower::LowerError`], falling
//! back to the stock tree-walker. The win is confined to what a compiled
//! straight-line trace over aligned columns can express.

pub mod bytecode;
pub mod lower;
pub mod smoke;

#[cfg(test)]
mod tests {
    use super::bytecode::{clean_interp, eval_batch_sum, eval_batch_sum_f, run_jit, Column, COMPILES};
    use super::lower::{lower, lower_typed, size_slot_source, Schema, ValType};
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
        let before = COMPILES.load(Ordering::Relaxed);
        let jit = run_jit(&prog, 7, 3);
        let jit_trap = *trap;
        *trap = 0;
        let clean = clean_interp(&prog, 7);
        assert_eq!(jit, clean, "compiled-tier overflow deopt must match wrapping oracle");
        assert_eq!(jit_trap, 1, "the compiled tier's overflow deopt must set the trap flag");
        assert_eq!(*trap, 1, "the reference tier must set the trap flag too");
        assert!(
            COMPILES.load(Ordering::Relaxed) > before,
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

    /// One input column for a typed-lowering batch test: an int/bool column, a
    /// `uint` column (stored as its i64 bit pattern), or a `double` column. Owns
    /// its data so the test keeps the buffers alive.
    enum ColData {
        Int(Vec<i64>),
        UInt(Vec<i64>),
        Float(Vec<f64>),
        /// A string column. Interned to an `i64` content-hash column
        /// (`intern_hash`) by the batch harness before it reaches the machine;
        /// [`ColData::column`] therefore refuses it (the id vec is derived, not
        /// borrowable from here).
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
                | ColData::UInt(c)
                | ColData::Timestamp(c)
                | ColData::Duration(c) => Column::Int(c),
                ColData::Float(c) => Column::Float(c),
                ColData::Str(_) => {
                    panic!("Str column must be interned to an id column before `column()`")
                }
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

    /// Deterministic full-range u64 column data returned as its i64 bit pattern.
    /// Roughly half the values have the high bit set (u64 > i64::MAX), so a
    /// signed comparison would order them differently from the unsigned oracle —
    /// making the uint compare tests genuinely discriminating.
    fn gen_u64_bits(n: usize, seed: u64) -> Vec<i64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                x as i64
            })
            .collect()
    }

    /// Run the clean two-bank interpreter over a freshly built batch program.
    /// Returns `None` on an int-arithmetic overflow, the same contract as
    /// [`super::bytecode::eval_batch_sum_f`], so all three tiers are compared on
    /// equal terms — a tier that silently wrapped where another trapped would
    /// show up as a `Some`/`None` mismatch, not as a plausible wrong number.
    fn clean_batch_sum_f(
        lowered: &super::lower::LoweredF,
        columns: &[Column],
        n: usize,
    ) -> Option<i64> {
        use super::bytecode::float_bank::clean_interp_f;
        let bases: Vec<i64> = columns.iter().map(|c| c.base()).collect();
        let mut trap: Box<i64> = Box::new(0);
        let trap_addr = (&mut *trap) as *mut i64 as i64;
        let (prog, ni, nf) = lowered.batch_sum_program_trapping(&bases, n as i64, trap_addr);
        let out = clean_interp_f(&prog, ni, nf);
        core::hint::black_box(columns);
        if *trap != 0 {
            None
        } else {
            Some(out)
        }
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
                    ColData::UInt(c) => ctx.add_variable_from_value(*name, c[i] as u64),
                    ColData::Float(c) => ctx.add_variable_from_value(*name, c[i]),
                    ColData::Str(c) => ctx.add_variable_from_value(*name, c[i].clone()),
                    ColData::Timestamp(c) => ctx.add_variable_from_value(
                        *name,
                        Value::Timestamp(
                            chrono::DateTime::from_timestamp_nanos(c[i]).fixed_offset(),
                        ),
                    ),
                    ColData::Duration(c) => ctx.add_variable_from_value(
                        *name,
                        Value::Duration(chrono::Duration::nanoseconds(c[i])),
                    ),
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
        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, n),
            Some(expected),
            "clean vs stock for `{expr_src}`"
        );

        // majit interpreter tier, then compiled tier. The compile counter is a
        // shared, monotonic global; asserting it *increased* across the jit-on
        // run (rather than resetting it to 0 first) is robust to other float
        // tests compiling concurrently.
        let off = eval_batch_sum_f(&lowered, &columns, u32::MAX);
        assert_eq!(off, Some(expected), "batch jit-off vs stock for `{expr_src}`");
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_f(&lowered, &columns, 8);
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
                    ColData::UInt(c) => ctx.add_variable_from_value(*name, c[i] as u64),
                    ColData::Float(c) => ctx.add_variable_from_value(*name, c[i]),
                    ColData::Str(c) => ctx.add_variable_from_value(*name, c[i].clone()),
                    ColData::Timestamp(c) => ctx.add_variable_from_value(
                        *name,
                        Value::Timestamp(
                            chrono::DateTime::from_timestamp_nanos(c[i]).fixed_offset(),
                        ),
                    ),
                    ColData::Duration(c) => ctx.add_variable_from_value(
                        *name,
                        Value::Duration(chrono::Duration::nanoseconds(c[i])),
                    ),
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
        let clean = clean_batch_sum_f(&lowered, &columns, n)
            .map(|bits| f64::from_bits(bits as u64))
            .unwrap_or_else(|| panic!("clean tier trapped on `{expr_src}`"));
        assert_eq!(clean.to_bits(), expected.to_bits(), "clean vs stock for `{expr_src}`");

        // majit interpreter tier, then compiled tier (monotonic compile-counter).
        let off = eval_batch_sum_float(&lowered, &columns, u32::MAX)
            .unwrap_or_else(|| panic!("jit-off tier trapped on `{expr_src}`"));
        assert_eq!(off.to_bits(), expected.to_bits(), "batch jit-off vs stock for `{expr_src}`");
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_float(&lowered, &columns, 8)
            .unwrap_or_else(|| panic!("jit-on tier trapped on `{expr_src}`"));
        assert_eq!(on.to_bits(), expected.to_bits(), "batch jit-on vs stock for `{expr_src}`");
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
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                choices[((x >> 33) as usize) % choices.len()].to_string()
            })
            .collect()
    }

    /// Cross-check a typed batch containing **string** columns. Each string
    /// column is interned to an `i64` content-hash column (`intern_hash`); the
    /// hash is verified injective over every distinct string present (column
    /// values + the expression's literals), so an id compare equals a content
    /// compare bit for bit. The clean / interp / compiled tiers must all equal
    /// the stock tree-walker's per-row bool/int sum, and the compiled run must
    /// trace the loop.
    fn check_batch_str(expr_src: &str, cols: &[(&str, ColData)]) {
        use super::bytecode::float_bank::COMPILES as COMPILES_F;
        use super::lower::intern_hash;

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
                        panic!("slot `{}` has no declared column for `{expr_src}`", slot.path)
                    });
                    assert_eq!(slot.ty, d.ty(), "slot `{}` bank for `{expr_src}`", slot.path);
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

        // Intern each string column to an i64 content-hash column, and gather
        // every (string, hash) pair (literals + column values) for the
        // injectivity check.
        let mut all_strs: Vec<(&str, i64)> = Vec::new();
        for lit in &lowered.str_literals {
            all_strs.push((lit.as_str(), intern_hash(lit)));
        }
        let mut id_storage: std::collections::HashMap<&str, Vec<i64>> =
            std::collections::HashMap::new();
        for (name, d) in cols {
            if let ColData::Str(c) = d {
                let ids: Vec<i64> = c.iter().map(|s| intern_hash(s)).collect();
                for (s, &h) in c.iter().zip(&ids) {
                    all_strs.push((s.as_str(), h));
                }
                id_storage.insert(*name, ids);
            }
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
        // Injectivity: no two distinct strings may share a hash (a real collision
        // bails to the tree-walker in production; the test data is collision-free
        // so the assert documents the id-compare == content-compare invariant).
        let mut seen: std::collections::HashMap<i64, &str> = std::collections::HashMap::new();
        for &(s, h) in &all_strs {
            match seen.get(&h) {
                Some(&prev) => {
                    assert_eq!(prev, s, "hash collision for `{expr_src}`: `{prev}` vs `{s}`")
                }
                None => {
                    seen.insert(h, s);
                }
            }
        }

        // Build one column per SLOT, keyed by path: a `size(x)` slot reads the
        // derived length column, a Str slot its interned id column, anything else
        // its own buffer.
        let columns: Vec<Column> = lowered
            .slots
            .iter()
            .map(|slot| match size_slot_source(&slot.path) {
                Some(src) => Column::Int(&len_storage[src]),
                None => match declared[slot.path.as_str()] {
                    ColData::Str(_) => Column::Int(&id_storage[slot.path.as_str()]),
                    d => d.column(),
                },
            })
            .collect();

        // Oracle: sum the stock tree-walker's per-row result (bool/int).
        let mut expected = 0i64;
        for i in 0..n {
            let mut ctx = Context::default();
            for (name, d) in cols {
                match d {
                    ColData::Int(c) => ctx.add_variable_from_value(*name, c[i]),
                    ColData::UInt(c) => ctx.add_variable_from_value(*name, c[i] as u64),
                    ColData::Float(c) => ctx.add_variable_from_value(*name, c[i]),
                    ColData::Str(c) => ctx.add_variable_from_value(*name, c[i].clone()),
                    ColData::Timestamp(c) => ctx.add_variable_from_value(
                        *name,
                        Value::Timestamp(
                            chrono::DateTime::from_timestamp_nanos(c[i]).fixed_offset(),
                        ),
                    ),
                    ColData::Duration(c) => ctx.add_variable_from_value(
                        *name,
                        Value::Duration(chrono::Duration::nanoseconds(c[i])),
                    ),
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

        // Clean two-bank interpreter over the built batch program.
        assert_eq!(
            clean_batch_sum_f(&lowered, &columns, n),
            Some(expected),
            "clean vs stock for `{expr_src}`"
        );
        core::hint::black_box(&id_storage);

        // majit interpreter tier, then compiled tier (monotonic compile counter).
        let off = eval_batch_sum_f(&lowered, &columns, u32::MAX);
        assert_eq!(off, Some(expected), "batch jit-off vs stock for `{expr_src}`");
        let before = COMPILES_F.load(Ordering::Relaxed);
        let on = eval_batch_sum_f(&lowered, &columns, 8);
        assert_eq!(on, Some(expected), "batch jit-on vs stock for `{expr_src}`");
        assert!(
            COMPILES_F.load(Ordering::Relaxed) > before,
            "string batch `{expr_src}` must compile the hot loop"
        );
    }

    #[test]
    fn batch_size() {
        // `size(s)` reads a DERIVED length column: the machine carries a string
        // as a content hash and has no bytes to count, so the batch builder
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
        // Only a `string` column has a materialized length column. A list/map
        // column has no representation on this machine at all, and a non-column
        // argument has nothing to derive from.
        let schema: Schema = [
            ("s".to_string(), ValType::Str),
            ("i".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in ["size(i) > 1", "i.size() > 1", "size(\"abc\") > 1"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering"
            );
        }
    }

    #[test]
    fn batch_string_equality() {
        // String ==/!= lower to a content-hash compare (OP_EQ/OP_NE over the
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
        check_batch_str("role == \"superadmin\"", &[("role", ColData::Str(role.clone()))]);
        // Column vs column.
        let other = gen_str(n, 0x9182_7364_5A4B_3C2D, &roles);
        check_batch_str(
            "a == b",
            &[("a", ColData::Str(role.clone())), ("b", ColData::Str(other.clone()))],
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

    #[test]
    fn string_ordering_bails() {
        // Strings support only equality here; ordering (`<` etc.) needs sorted
        // ids, so the typed lowering bails and the tree-walker handles it. A bare
        // string result is likewise not sum-reducible and bails.
        let schema: Schema =
            [("a".to_string(), ValType::Str), ("b".to_string(), ValType::Str)]
                .into_iter()
                .collect();
        for expr in ["a < b", "a <= b", "a > b", "a >= b", "a"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering (string ordering / bare result)"
            );
        }
    }

    /// Deterministic i64-nanosecond column in `[base, base + span)` from an LCG,
    /// for timestamp / duration columns.
    fn gen_nanos(n: usize, seed: u64, base: i64, span: i64) -> Vec<i64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    fn batch_duration_accessors() {
        // `d.getHours()` and friends are `chrono::Duration::num_*` = a
        // toward-zero divide of the i64-nanos payload. The span straddles zero so
        // NEGATIVE durations are exercised: that is where truncation and floor
        // disagree, the exact class the `OP_DIV` sign-mask fix exists for. Both
        // the compiled and interpreter tiers must match the tree-walker oracle.
        let n = 3000;
        let elapsed = gen_nanos(n, 0x0f1e_2d3c_4b5a_6978, -7_200_000_000_000, 14_400_000_000_000);
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
            .map(|(k, v)| if k % 3 == 0 { v.wrapping_add(i64::MIN) } else { v })
            .collect();

        check_batch_f("double(i) > 100.0", &[("i", ColData::Int(i.clone()))]);
        for expr in ["double(i) + f > 0.0", "double(i) == f"] {
            check_batch_f(
                expr,
                &[("i", ColData::Int(i.clone())), ("f", ColData::Float(f.clone()))],
            );
        }
        for expr in ["int(u) < 0", "int(u) > 100", "uint(int(u)) > 100u"] {
            check_batch_f(expr, &[("u", ColData::UInt(u.clone()))]);
        }
        for expr in ["uint(i) > 100u", "int(uint(i)) > 100"] {
            check_batch_f(expr, &[("i", ColData::Int(i.clone()))]);
        }
        // Identity spellings.
        check_batch_f("int(i) > 100", &[("i", ColData::Int(i.clone()))]);
        check_batch_f("double(f) > 100.0", &[("f", ColData::Float(f))]);
        check_batch_f("uint(u) > 100u", &[("u", ColData::UInt(u))]);
    }

    #[test]
    fn numeric_conversion_bails() {
        // `double(uint)` needs `u64 as f64`, which differs from the signed cast
        // above 2^63 and has no trace opcode. `int(double)`/`uint(double)` need
        // `CastFloatToInt`, which the IR has but the `#[jit_interp]` macro does
        // not lower yet. String and temporal arguments are a parse or a
        // `FunctionError` in the walker.
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
            "double(u) > 1.0",
            "int(f) > 1",
            "uint(f) > 1u",
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

    #[test]
    fn temporal_mixed_bails() {
        // A timestamp vs duration comparison is NoSuchOverload, a temporal vs int
        // is a type error, temporal arithmetic is out of subset, and a bare
        // temporal result is not sum-reducible — all bail to the tree-walker.
        let schema: Schema = [
            ("t".to_string(), ValType::Timestamp),
            ("d".to_string(), ValType::Duration),
            ("i".to_string(), ValType::Int),
        ]
        .into_iter()
        .collect();
        for expr in ["t < d", "t < i", "t - t", "t + d", "d + d", "t"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering (mixed/arith/bare temporal)"
            );
        }
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
        // String membership: an OR-chain of content-hash equalities. Literals in
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
            &[("price", ColData::Float(price)), ("qty", ColData::Float(qty))],
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
            &[("price", ColData::Float(price)), ("qty", ColData::Float(qty))],
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
                &[("a", ColData::UInt(a.clone())), ("b", ColData::UInt(b.clone()))],
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

        let schema: Schema =
            [("a".to_string(), ValType::Int), ("b".to_string(), ValType::Int)]
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
                eval_batch_sum_f(&lowered, &columns, u32::MAX),
                None,
                "jit-off tier must refuse `{expr}`"
            );
            let before = COMPILES_F.load(Ordering::Relaxed);
            assert_eq!(
                eval_batch_sum_f(&lowered, &columns, 8),
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
                &[("a", ColData::Int(a.clone())), ("b", ColData::Int(b.clone()))],
            );
        }
    }

    #[test]
    fn uint_arith_bails() {
        // Every uint arithmetic operator bails; only comparisons lower.
        //
        // `+ - *` USED to reuse the signed opcodes on the grounds that they are
        // bit-identical mod 2^64 — true of the value, false of the CHECK. The
        // tree-walker uses `u64::checked_*` (`common/types/uint.rs:78-196`) and
        // raises on unsigned overflow, which the signed `Int*Ovf` guard answers
        // wrongly in both directions: `2^63 + 1` is fine unsigned but overflows
        // signed, and `0u - 1u` is the reverse. The trace IR has no unsigned
        // overflow opcode (and detecting unsigned `*` overflow by hand needs a
        // division), so the whole group falls back rather than wrap silently.
        //
        // `/ %` bail for the neighbouring reason: no unsigned floordiv/mod
        // opcode. Both wait on the same parent majit-macros addition.
        let schema: Schema =
            [("a".to_string(), ValType::UInt), ("b".to_string(), ValType::UInt)]
                .into_iter()
                .collect();
        for expr in [
            "a + b >= 1500u",
            "a - b >= 1500u",
            "a * b < 250000u",
            "a / b >= 1u",
            "a % b >= 1u",
        ] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower_typed(program.expression(), &schema).is_err(),
                "`{expr}` must bail the typed lowering (no unsigned arithmetic opcodes)"
            );
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
        let schema: Schema = [("p".to_string(), ValType::Float), ("q".to_string(), ValType::Int)]
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
