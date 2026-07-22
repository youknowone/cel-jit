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
//! literals, slot-resolved `Ident`/`Select`, arithmetic `+ - *`, unary `-`,
//! comparisons, boolean `&& || !`) to the flat `i64`-word program of
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
    use super::bytecode::{clean_interp, run_jit};
    use super::lower::lower;
    use crate::{Context, Program, Value};
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
    fn out_of_subset_bails() {
        for expr in ["[1, 2, 3]", "'a' + 'b'", "[1, 2, 3].all(x, x > 0)", "x.size()", "1.5 + a"] {
            let program = Program::compile(expr).unwrap();
            assert!(
                lower(program.expression()).is_err(),
                "`{expr}` must be rejected as out of subset"
            );
        }
    }
}
