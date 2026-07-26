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

## The nested-loop defect (FIXED — kept for the RCA)

The runtime-length list comprehension shipped in `0758724` was **correct but a
net performance loss**: `items.all(i, i.price > 10)` measured 0.0–0.4x of the
clean VM and 0.3–0.8x of the tree-walker at every list length, with a flat
~1–3 µs per-row cost.

Two stacked `majit-metainterp` defects, both now fixed. Counters from
`float_bank::{COMPILES, GUARD_FAILS, TRACE_ABORTS}`, pinned by
`tests/majit_trace_evidence.rs`:

| workload | compiles | guard_fails | aborts |
|---|---|---|---|
| flat int predicate, 50000 rows | 1 | 1 | 0 |
| flat float predicate, 50000 rows | 1 | 1 | 0 |
| nested list, 4000 rows × 8 elements | 2 | **9** (was 3999) | **0** (was 1) |

100000 rows × 8 elements goes from a net loss to **1.68x** over the clean
bytecode VM; the compiled trace itself runs at the flat case's ~10x, with the
one-shot compile amortising from roughly 65k rows.

Under `MAJIT_LOG=1` the inner element loop traces to `CloseLoop` and compiles;
the outer row loop then hits that inner merge point twice and used to close
there too, as the cross-loop cut.

**Defect 1 — the cut was refused at optimize time:**

```
abort trace (InvalidLoop: next_iteration_args longer than inputargs
             (full-body-walk cross-loop cut over a forced heap virtual))
abort compile: root loop entry/jump arity mismatch input=3 jump=29
```

— the tripwire at `majit-metainterp/src/optimizeopt/optimizer.rs` (the
`inputarg_type_at` check): the cut label declared 3 inputargs while the closing
JUMP carried 29.

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

## Defect 2 — the cut's storage key

With the arity fixed the outer loop compiled and still nothing entered it.
`compile_loop` stores a cross-loop cut under `ctx.cut_inner_green_key`, which the
dispatch loop derives as `green_key_from_code_ptr(ctx.green_key_raw.0, pc)` —
and `green_key_raw.0` is `JitState::code_ptr()`, which **defaults to 0** and is
overridden by nobody on front-end A. So the cut lands under a pc-only hash while
the interpreter presents `S::green_key([pc, program])` at that merge point.
Measured with `MAJIT_LOG=1 MAJIT_MPTRACE=1`:

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
an already-compiled inner loop ends with a JUMP into that loop's procedure — the
cut is only for an inner loop nobody has compiled yet, and there `compile_loop`
attaches the result to `original_boxes[:num_green_args]`, the inner greenkey the
interpreter actually presents. The inner element loop always compiles first here
(two back-edges per row to the outer's one), so this shape belongs in the
`compile_trace` branch.

**Fixed**, the `has_compiled_targets` half: `TraceCtx::compiled_key_for_greens_fn`
(installed alongside `has_compiled_targets_fn` at the three trace-start sites,
wrapping the existing `MetaInterp::compiled_key_for_greens`) lets the dispatch
loop resolve the merge point's greens, and the cut is declined when a compiled
loop already lives there. The trace keeps tracing and closes at its own header
with the inner loop's body inlined — the same shape it produces at trip counts
too low to reach the merge point twice, which was already the fast case.

Still open: the **JUMP-into-ptoken** half of :3001-3007, and a greens-derived cut
key. Neither is reachable from this workload now, since declining the cut already
closes at the outer header. `compile_trace_from_interp` exists and is unused, and
`compile_trace_entry_data` declines an entry-bridge close for `header_pc != 0` on
the grounds that it would drop the trace's own back-edge — RPython accepts that
loss and recovers the back-edge through the bridge that leaves the inner loop's
exit guard, so closing this gap means revisiting that decline.

Consequences for this document:

- The per-row cost was **one compiled-trace entry plus one guard deopt**, not
  anything about the columnar data model — it was a majit-side defect, and Step 4
  demoting the columnar path would not have made it go away. The same shape
  reaches the real interpreter of Step 1 the moment a CEL expression puts a loop
  inside a loop, which `x.all(i, i.items.all(j, ...))` does.
- Both fixes landed in the dispatch loop that front-end B also runs, so Step 3
  inherits them.

---

## Defect 3 — the exit guard could not bridge, so a varying trip count deopted forever

The census that pinned defects 1 and 2 used a **constant** trip count, which
hid what the fix actually bought. The outer trace inlines the inner loop and
guards its trip count; every row whose list is a different length fails that
guard. Over 100k rows, 4000-element batches (before this section's fix):

| lengths | guard_fails | aborts | JIT vs the clean VM |
|---|---|---|---|
| 8, 8, 8, …    | 9      | 0  | 1.31x |
| 8, 9, 8, 9, … | 50004  | 1  | 0.06–0.13x |
| 4..12 cycling | 88888  | 8  | 0.03–0.09x |
| 0..32 spread  | 165621 | 12 | 0.04–0.05x |

A guard that fails that often is supposed to grow a bridge. None ever formed:

```
[bridge] start_bridge_tracing (green resume) key=… trace=1 fail=1 resume_pc=95 ok=true
Abort during bridge tracing            ← with ZERO ops recorded
```

`start_bridge_tracing` succeeded and the walk aborted on its **first statement**.
The macro-generated `__trace_*` opens with

```rust
let Some(__vable_argbox) = __ctx.standard_virtualizable_jitcode_argbox() else {
    return TraceAction::Abort;
};
```

and on a bridge `ctx.virtualizable_boxes` was `None`. `pyjitpl.py:3449
rebuild_state_after_failure` ends with

```python
if vinfo is not None:
    self.virtualizable_boxes = virtualizable_boxes
    self.check_synchronized_virtualizable()
```

where `virtualizable_boxes` is what `resume.py:1370
consume_virtualizable_boxes` decoded out of the guard's vable section. pyre does
that assignment in each front-end's `setup_bridge_sym`; pyre-jit-trace has
`seed_virtualizable_boxes`, and the `#[jit_interp]` macro had nothing — its own
STATUS comment says it seeds int/ref scalars only. So **no front-end A state
with a `[.. ; virt]` array has ever formed a guard-exit bridge**, on any
interpreter.

The guard's vable stream was there (26 entries, exactly the parent loop's box
count) and decoded fine — except for its first entry, the virtualizable
identity, which came back as `Box(0, Int)` = 25 rather than the `&state`
pointer. `initialize_virtualizable` mints that box as

```rust
OpRef::input_arg_ref(info.identity_ref_bank_index.unwrap_or(index_of_virtualizable))
```

and `identity_ref_bank_index` is a JitCode ref **register** (the macro sets it to
1 because the dispatch lowering binds `program` to ref reg 0 and `&state` to ref
reg 1), while trace inputargs are numbered **flat across banks**. `InputArgRef(1)`
is inputarg #1, an int; the optimizer resolved it straight through to
`InputArgInt(1)`:

```
[callee-rca][store-final-vable] vable=[(InputArgRef(1), InputArgInt(1), false, Int), …]
```

The alias was invisible while the identity was only read back through
`virtualizable_values[-1]` (whose concrete is written separately, and is
correct), but it is what `capture_resumedata` writes into every guard.

**Fixed**, three pieces:

1. `initialize_virtualizable` recovers the identity the way `pyjitpl.py:3295
   virtualizable_box = original_boxes[index]` does — as the red that *holds* the
   virtualizable — by matching the live `vable_ptr` against `live_values`, and
   mints `OpRef::input_arg_typed(idx, Type::Ref)`. The vable section now reads
   `(InputArgRef(2), InputArgRef(2), false, Ref)`.
2. `majit_metainterp::seed_bridge_virtualizable_boxes` rebuilds the shadow from
   the decoded stream (`[identity, statics…, array items…]`, array lengths read
   off the live object per `virtualizable.py:150-153`), and the macro's
   `setup_bridge_sym` calls it for any state with a virt array.
3. `start_bridge_tracing` resolves the live virtualizable through the same
   `virtualizable_heap_ptr` hook trace entry uses and puts it on the ctx, so the
   seed can run `check_synchronized_virtualizable`: a decoded identity that is
   not the live object declines the bridge (`compile.py:725-729
   compile.giveup()`) instead of being dereferenced. Without that check a
   0..32 spread SIGSEGVs on `state.regs.len()` through a bogus pointer.

Result, same 100k-row runs:

| lengths | guard_fails | JIT vs the clean VM |
|---|---|---|
| 8, 8, 8, …    | 9     | **1.37–1.41x** |
| 8, 9, 8, 9, … | 210   | **1.14–1.60x** |
| 4..12 cycling | 1609  | **0.56–0.57x** |
| 0..32 spread  | 12959 | **0.45–0.50x** |

Pinned by `nested_list_loop_varying_trip_count` in
`cel/tests/majit_trace_evidence.rs`.

### Still open

- **The preamble's copy of the exit guard.** The spread case's remaining deopts
  are rows that leave through the peeled preamble rather than the loop body.
  That guard's vable section names the identity as failarg 0, but the deadframe
  slot the backend writes there holds something else, so piece 3's check gives
  up on the bridge. Odd trip counts (3, 5, …) hit it deterministically.
- **The JUMP-into-ptoken half of :3001-3007.** Now that bridges form it is
  reachable and measurably better on non-uniform lengths (`cycle 4..12`
  1.69–1.72x and `0..32` 0.71–0.74x, all at a flat ~201 deopts, versus 0.56x /
  0.45x above), because the trip count stays in the inner loop's own back-edge
  instead of being baked into the outer trace. It is NOT landed: it routes every
  row's exit through the guard above, so trip counts 3 and 5 go from 6 and 15
  deopts to 2101 and 4192. It becomes a strict win once the preamble guard
  bridges.
