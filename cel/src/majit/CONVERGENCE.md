# Converging cel onto one interpreter

Status: design, not implemented. Written 2026-07-26.

## The problem

cel-jit currently has **two evaluators**:

| | evaluator | traced? | covers |
|---|---|---|---|
| 1 | `Value::resolve_val` (`objects.rs`) — the recursive AST tree-walker | no | all of CEL |
| 2 | `majit::bytecode::float_bank::run_mainloop_f` — a flat `i64`/`f64` register machine | yes | a scalar columnar subset |

Evaluator 2 was written *because* evaluator 1 could not be traced: `resolve_val`
returns `Cow<'a, dyn Val>` and dispatches through trait objects and
`downcast_ref`. So the majit tier is additive — **+6577 lines, 0 deletions**,
with the only edits to pre-existing cel code being `lib.rs` +3 and
`Cargo.toml` +44 — and `Program::execute` never reaches it.

That shape is not what a meta-tracing JIT is for. RPython has exactly one
interpreter; the JIT traces *that* interpreter. Writing a second, traceable
mirror of the first is the thing meta-tracing exists to make unnecessary. The
deviation is real and this document is the plan to close it.

## What must NOT be done

**Deleting the tree-walker is not the fix.** The traceable subset can never be
total — `matches` is a regex call, custom functions are opaque, and bytes/map/
optional/struct have no machine representation. In RPython the interpreter is
also where every deopt lands; a fallback is orthodox. What is *not* orthodox is
that the fallback is a **different** interpreter from the traced one.

The fix is convergence: evaluator 2 must absorb evaluator 1 and become the
single production evaluator, with the JIT tracing it.

## The blocker is the front-end, not the JIT

majit has **two front-ends**, and cel is on the wrong one.

**Front-end A — the `#[jit_interp]` proc macro** (`majit-macros`). Parses a Rust
`fn` with `syn` and lowers a restricted subset directly to jitcode. It accepts
13 `syn::Expr` kinds (`Assign, Binary, Block, Call, Cast, If, Lit, Match,
MethodCall, Paren, Path, Struct, Unary`) and **never emits `GuardClass`** —
`rg GuardClass majit-macros/src/` is empty. No trait objects, no heap objects,
no class dispatch. RPython has no counterpart to this macro at all; it is a
shortcut for hand-written mainloops (cel, the wasmi kernel, the examples).
cel uses this one.

**Front-end B — `majit-translate`** — the actual RPython pipeline
(`flowspace/` → `annotator/` → `rtyper/` → `codewriter/`, ported line by line)
over Charon-extracted LLBC of **real Rust**. `GuardClass` lives here. This is
what pyre uses; `rg '#\[jit_interp' pyre/` finds only comments.

⇒ `dyn Val` is not fundamentally untraceable. RPython traces `W_Root` subclass
dispatch every day: a virtual call becomes `guard_class(w_obj, W_IntObject)`
plus the inlined body. cel cannot do it *because it is on front-end A*.

## Prerequisite: cel needs a bytecode interpreter

RPython's JIT is a **bytecode**-interpreter JIT. `jit_merge_point` is keyed on
greens `(pc, code)`; that is what gives a trace its identity, what the
back-edge counter counts, and what deopt resumes into. PyPy's interpreter is a
bytecode VM for exactly this reason.

cel's `resolve_val` is a recursive AST walk. There is no pc, no dispatch loop,
and no merge point to place. So convergence has a prerequisite that is not
about the JIT at all:

> **cel must gain a real bytecode compiler and a bytecode VM over full
> `Value`s, and that VM must become `Program::execute`.**

This step is independently justifiable — a flat bytecode VM normally beats a
`Cow<dyn Val>` recursive walk on its own — and it is the step that makes
everything after it possible.

## Plan

**Step 1 — one interpreter, no JIT.**
CEL AST → a full-fidelity bytecode covering *all* of CEL (not the traceable
subset): lists, maps, strings, bytes, optionals, custom functions, regex.
A stack VM over `Value` executes it. `Program::execute` becomes that VM and the
recursive walker is deleted. Nothing about majit is involved. Gate: the entire
existing cel test suite, unchanged.

**Step 2 — a merge point on the dispatch loop.**
Put `jit_merge_point` with `greens = [pc, code]` on the new VM's loop and
`can_enter_jit` on its back-edges, exactly as `run_mainloop_f` does today for
its subset.

**Step 3 — move to front-end B.**
Extract LLBC for the `cel` crate and drive it through `majit-translate` instead
of the `#[jit_interp]` macro. `Value`'s dispatch becomes `guard_class` +
inlined `getfield`, the same treatment pyre's object model gets. This is the
step that makes the *whole* language traceable rather than a subset, because
the tracer now sees the real interpreter instead of a hand-written mirror.

**Step 4 — demote the columnar path to what it is.**
`lower_typed` + the two-bank machine + `eval_batch_sum*` are a **vectorized
batch executor**, not a JIT. Keep them as an explicit columnar API if the
throughput is wanted, but stop reporting their data-model win as a JIT win: the
flagship's 121x splits into 13.1x (stock → clean VM, i.e. slot resolution and
no boxing) and 9.2x (clean VM → compiled trace, the actual JIT). Only the
second number is what Step 3 would deliver on the real interpreter.

## Costs and risks, stated up front

- **Step 1 is a rewrite of cel's evaluator.** cel-jit is a fork of
  `cel-rust/cel-rust`; replacing `resolve_val` is a permanent divergence from
  upstream. That is a project decision, not a technical one.
- **`Cow<'a, dyn Val>` has lifetimes; RPython does not.** The rtyper expects
  GC-managed boxes. Values will have to be owned/boxed the way pyre's
  `PyObjectRef` is, which is itself part of Step 1.
- **`majit-translate`'s rtyper still carries pyre-only transitional adapters**
  (`translator/rtyper/legacy_*.rs`, gated by `cutover.rs`). cel would be its
  second client and would expose whatever those adapters paper over.
- **LLBC extraction must be wired for the cel crate.** `local_crates.rs` seeds
  its alias roots from the loaded LLBC set, so adding `cel` is mechanical, but
  the build plumbing and the known LLBC landmines (`i128` breaks decoding;
  stale corpora silently mask changes) come with it.
- **Step 3 buys correctness of *coverage*, not automatically speed.** The
  measured JIT effect on the current subset is ~9x over a clean VM. Whether a
  `Value`-boxing VM traces to anything near that is unknown until measured, and
  the runtime-list result below is a warning that some shapes lose outright.

## Known open defect that Step 4 must not paper over

The runtime-length list comprehension shipped in `0758724` is **correct but a
net performance loss**: `items.all(i, i.price > 10)` measures 0.0–0.4x of the
clean VM and 0.3–0.8x of the tree-walker at every list length, with a flat
~1–3 µs per-row cost that says the row loop cannot stay in compiled code across
the inner loop. Whatever happens to the columnar path, that shape must not be
elected on a performance path until the per-row exit is understood.
