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
~1–3 µs per-row cost.

That cost is now diagnosed, and `tests/majit_trace_evidence.rs` pins it.
Counters from `float_bank::{COMPILES, GUARD_FAILS, TRACE_ABORTS}`:

| workload | compiles | guard_fails | aborts |
|---|---|---|---|
| flat int predicate, 50000 rows | 1 | 1 | 0 |
| flat float predicate, 50000 rows | 1 | 1 | 0 |
| nested list, 4000 rows × 8 elements | 2 | **3999** | **1** |

One deopt per row. Under `MAJIT_LOG=1` the inner element loop traces to
`CloseLoop` and compiles; the outer row loop hits that inner merge point twice
and closes there too, as the cross-loop cut.

That cut used to be **refused at optimize time** —

```
abort trace (InvalidLoop: next_iteration_args longer than inputargs
             (full-body-walk cross-loop cut over a forced heap virtual))
abort compile: root loop entry/jump arity mismatch input=3 jump=29
```

— the tripwire at `majit-metainterp/src/optimizeopt/optimizer.rs` (the
`inputarg_type_at` check): the cut label declared 3 inputargs while the closing
JUMP carried 29. That is fixed (see below); the cut now compiles, which is the
`compiles` 1 → 2 above. The per-row cost survives it, on a second defect.

**Checked against RPython source** (`rpython/jit/metainterp/`, present on this
machine), because whether this is a design limit or a port defect decides
everything:

1. **Cutting at the second encounter of a foreign green key is orthodox.**
   `reached_loop_header` (`pyjitpl.py:3018-3060`) scans `current_merge_points`
   for a matching green key; on no match it *appends* the foreign merge point
   and keeps tracing. The second encounter matches that entry and compiles the
   INNER loop from there, peeling the outer prefix as preamble. majit's split at
   trip count 3 is exactly this.
2. **The arity mismatch is structurally impossible in RPython.**
   `live_arg_boxes = greenboxes + redboxes` plus, for a virtualizable jitdriver,
   `+= self.virtualizable_boxes; .pop()` (`pyjitpl.py:2981-2989`). That one list
   feeds the merge-point registration (:3060), `compile_loop` (:3039) and
   `compile_trace`'s JUMP (:3213), and line **3020 asserts
   `len(original_boxes) == len(live_arg_boxes)`**. `virtualizable_boxes` holds
   one box per array element (`virtualizable.py:94-98`), so an RPython header
   for `regs: [int; virt]` carries all the element boxes — 29, never 3.

⇒ So this is **not** a design gap in meta-tracing, and it is not cel's lowering
either: two merge points for a nested loop is PyPy's own model. It is front-end
A building the merge-point registration differently from the close.

**Registration never expands the virtualizable; the close does.**

- Close: `collect_jump_args_with_boxes` (generated by
  `majit-macros/src/jit_interp/codegen_state.rs`, and the `JitState` hook at
  `majit-metainterp/src/jit_state.rs:619-642`) emits every `<arr>_ptr`/
  `<arr>_len` header, then splices the whole element shadow, dropping the
  trailing identity — its comment cites `pyjitpl.py:2982-2989` verbatim. 29.
- Registration: `majit-metainterp/src/pyjitpl/dispatch.rs` builds the merge
  point's `original_boxes` from `sym.state_field_ref` / `sym.state_ref_field_ref`
  — **scalar state fields only** — and falls back to the operand payload when
  there are none: `let original_boxes = if red_boxes.is_empty() { live_arg_boxes }
  else { red_boxes };`. Its comment says the intent is exactly "so the inner
  loop's LABEL inputarg arity matches the JUMP (compile.py:334)", but it only
  covers interpreters that HAVE scalar state fields. `VmStateF` is
  `{ regs: [int; virt], fregs: [float; virt] }` — no scalars — so the fallback
  fires and the registered shape is the greens plus one UNexpanded vable ref
  box. 3.

The 3 is therefore not the trace's entry shape: a trace that starts at its own
header is minted expanded, which is why the flat row loop and the inner element
loop both compile. The 3 is the **cut** label, rebuilt from the registration
when the outer trace is cut at the inner loop's second visit.

**Fixed** in the parent majit repo (`majit-metainterp`), not here: the macro now
emits one `__jit_loop_carried_boxes` and both the close
(`JitState::collect_jump_args_with_boxes`) and the registration (the new
`JitCodeSym::loop_carried_boxes`, reachable from the jitcode dispatch loop) go
through it, mirroring `pyjitpl.py:2981-2989`'s single `live_arg_boxes`. The
boxes are typed end to end because the registered ones become a cut trace's
LABEL inputargs. `MAJIT_LOG=1` now reports `cut_trace_from: original_boxes=29`
against the 29-arg JUMP, and the cut compiles.

## The defect behind it: the cut's storage key

The outer loop now compiles and still nothing enters it. `compile_loop` stores a
cross-loop cut under `ctx.cut_inner_green_key`, which the dispatch loop derives
as `green_key_from_code_ptr(ctx.green_key_raw.0, pc)` — and `green_key_raw.0` is
`JitState::code_ptr()`, which **defaults to 0** and is overridden by nobody on
front-end A. So the cut lands under a pc-only hash while the interpreter
presents `S::green_key([pc, program])` at that merge point. Measured with
`MAJIT_LOG=1 MAJIT_MPTRACE=1`:

```
[jit] start tracing at key=10921439234107011841   (inner loop, pc=64) → compiles
[jit] start tracing at key=8274026927361047311    (outer loop, pc=42)
@@@MPTRACE add-mp pc=64 header_pc=42 inner_key=6467483736705779522
[jit] cut_trace_from: start.op_index=14 original_boxes=29 trace_ops=24
[jit][compile-loop] trace_id=2 header_pc=6467483736705779522
```

`6467483736705779522` ≠ `10921439234107011841` for the same `(pc=64, program)`.

RPython would not reach the cut here at all. `reached_loop_header`
(`pyjitpl.py:3001-3007`) runs *before* the `current_merge_points` scan:

```python
ptoken = self.get_procedure_token(greenboxes)
if has_compiled_targets(ptoken):
    self.compile_trace(live_arg_boxes, ptoken)
```

`greenboxes` is the merge point just reached, so an outer trace that walks into
an already-compiled inner loop ends with a JUMP into that loop's procedure. The
inner element loop always compiles first here (it takes two back-edges per row
to the outer's one), so that is the branch this shape belongs in. The dispatch
loop implements neither it nor a greens-derived cut key; `compiled_key_for_greens`
/ `record_loop_header_greens` already exist for the bridge path and are the
obvious material for the latter.

Consequences for this document:

- The per-row cost is **one compiled-trace entry plus one guard deopt**, not
  anything about the columnar data model. It is a majit-side defect, so Step 4
  demoting the columnar path does not make it go away — the same shape will
  appear on the real interpreter of Step 1 the moment a CEL expression contains
  a loop inside a loop, which `x.all(i, i.items.all(j, ...))` does.
- It is one more argument for Step 3. Front-end B is the pipeline where that
  merge-point/JUMP construction is the ported RPython one.

Until it is fixed, the nested shape must not be elected on a performance path.
