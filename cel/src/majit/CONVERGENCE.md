# Converging cel onto one interpreter

Status: written 2026-07-26 as design. Step 1 has since landed (see the note
under it); Steps 2-4 are still design.

Superseded in part, 2026-08-07. Evaluator 1 below is no longer
`Value::resolve_val` returning `Cow<'a, dyn Val>`: the trait-object universe has
been deleted and the tree walker is `Value::resolve_value`, which walks the
`Value` enum. Everything this document says about `dyn Val`, `Cow` lifetimes and
`downcast_ref` describes the tree as it stood on 2026-07-26. The two-evaluator
problem itself is unchanged. One consequence is not merely descriptive: the
document's argument that Step 1 pays for itself on speed was an argument against
the deleted walker, and measurement against the new one refutes it. See the note
under Step 1.

Superseded again, 2026-08-09: `vm` is a default cargo feature, so
`Program::execute` is the bytecode VM. Wherever this document says
`Program::execute` is the tree walker, that describes the pre-2026-08-09 default
and the walker is now reached as `Value::resolve` / `Value::resolve_value`, or
through `Program::execute` under `--no-default-features`.

> ⛔⛔ **THIS DOCUMENT IS NOT THE DESIGN OF RECORD. Read the `cel-unboxed-values`
> skill first.** Noted 2026-08-13. That document plans the same epic in phases
> P0–P9 with measured gates, and the two numbering schemes have to be lined up
> before anything here is executed:
>
> | here | there | state |
> |---|---|---|
> | Step 1 (replace the walker) | P2 | **landed** |
> | Step 2 / 2b (merge point, jitdriver spec) | **P8** | gated — see below |
> | Step 3 / 3a / 3b (front-end B) | P4, P6–P8 | P4 landed; P6–P8 gated |
> | Step 4 (demote the columnar path) | P9 | gated behind P8 |
>
> **The gate: the design of record says STOP AT P5 for the JIT half.** Its
> economics measurement (task #88) puts a compiled cel artefact's fixed per-call
> cost at 34–92 µs on both backends against a clean-VM CEL evaluation of
> 43–130 ns, and sets an explicit re-entry criterion for P6–P8 — *fixed per-call
> cost under ~1 µs on both backends*. **Steps 2b, 3b and 4 of this document all
> sit past that stop**, so executing this document's plan in order walks straight
> through a documented NO-GO. Do not read the 2a/3a work recorded below as
> progress toward 2b; it is progress toward knowing the portal is not the problem.
>
> **What is actually next is P5** — the `Arc`-free, header-first `W_Root` class
> family. The B2 number recorded under Step 3 below points at exactly this from a
> second direction: the largest single rtyper prepass failure across cel's
> closure is **`sync::Arc::deref`, 16 of 88** — the value universe's `Arc`
> itself. P5 is the phase that deletes it.
>
> **P5's fixed-size half has since landed** (2026-08-13, `cel/src/runtime`):
> `CelObject`, `CelClass`, `lltype::malloc_typed`, the eight fixed-size leaves
> (`int uint double bool null duration timestamp type`), the arithmetic chains,
> and `==`/`!=` plus the four orderings. It is additive — nothing is reachable
> from `Value`, which still carries `Arc<String>` / `Arc<Vec<u8>>` /
> `Arc<dyn Opaque>` / `Arc<CelStruct>`. What remains of P5 is blocked, not
> skipped: the variable-length leaves need **M5** (varsize allocation lowering),
> and `W_OpaqueObject` needs the heap and the D12 side table.
>
> **The premise underneath all of it is now measured, not assumed.** The family
> is only worth its two header words if `fuse_boxing_alloc` turns each
> constructor into a `NewWithVtable` the optimizer can delete, and all of that
> pass's decline paths are a bare `continue`. Seeding the pipeline census at the
> chains themselves (`majit-translate/tests/test_cel_census.rs`,
> `cel_census_pipeline_runtime_add` / `_less`, measured at cel-jit `3db70c6`):
>
> | portal | `new` | `newwithvtable` |
> |---|---|---|
> | `runtime::binop::cel_add` | 1 | **5** |
> | `runtime::binop::cel_less` | 0 | **1** |
> | `objects::Value::resolve_value` (control) | 88 | 0 |
>
> Five is every constructor in `cel_add`'s closure — `new_int`, `new_uint`,
> `new_double`, `new_duration`, `new_timestamp` — and one is `new_bool` under
> `cel_less`. `lltype::malloc_typed` and `pyobject::get_instantiate` drop out of
> the jitcode list entirely, consumed by the fuse. **The leaves fuse.**
>
> The count is what makes that a statement about *every* constructor rather
> than about five fusions somewhere. Each constructor body holds exactly one
> `malloc_typed`, and each is its own graph, so the five constructor graphs
> contribute exactly five allocation sites; five of them are `newwithvtable`, so
> none is left over. ⚠ Which means `cel_add`'s residual `new` is in some other
> graph, and it is **unidentified**. The obvious candidate — `raise`'s
> `Some(CelError { .. })` enum shell, which §2(d) says materializes — is refuted
> by `cel_less`: `error::raise` is a jitcode under both portals, and `cel_less`
> reports `new` 0.
>
> ⛔ **Both numbers were unreadable until two harness defects were fixed, and
> each returned a clean zero rather than an error.**
>
> - `run_pipeline_census` passed `HostStaticAddrs::default()`. `pytypes` is the
>   only channel carrying a class static's address, and `resolve_vtable_addr`
>   declines without one — so the first run reported `0` about the harness's
>   configuration, not about any constructor.
> - `section1_cells` looked up `new_with_vtable` in the opname histogram, which
>   spells the op `newwithvtable`. That cell read `0` for every portal, fused or
>   not. It is the same dump-vs-insns-table split the file already prints both
>   spellings for on `vtablemethodptr` — the second instance of one defect.
>
> ⚠ The corrected lookup does **not** overturn the §1 walker row: the walker
> measures `88 / 0` under both spellings, because it has no allocation of this
> shape to fuse. The blindness only ever mattered for a portal that fuses.
>
> ⛔ **One real wall in the new code, found by the same run.**
> `runtime::error::RAISED` — the `thread_local!` holding the out-of-band error
> slot — is not registered in `PyreCallRegistry`, so `raise`'s pyre-side lift
> fails and it stays residual on every failing arm. `error.rs` already documents
> the slot as thread-local "for now", belonging to the heap and moving there
> with it; this measures what the placement costs. The orthodox destination is
> the one §6.3 names — the slot on an execution context passed as an argument,
> the shape `OperationError` has upstream — and it needs the context object,
> so it is P6 work rather than a spelling change here.
>
> P5 is SEMVER MAJOR and §8 of the design of record requires M1–M8 to land on
> one named pyre branch agreed with the user before P5 starts. **M1 itself is
> already landed**: `pytype_static_addr`'s doc records that the bucket carries
> only the address while the root comes from the static's own declared type at
> the read site, naming `charon-corpus`'s `CelClass` as the case it serves, and
> `cel_boxing_cluster_fuses_once_the_class_address_resolves` asserts one
> `NewWithVtable` carrying the real class pointer. The gate the design called
> "the sole gate on the boxing fuse" is open.
>
> **D12 — decided 2026-08-13: register a full finalizer.** The design of record
> left the `W_OpaqueObject` host-object lifetime open between a finalizer and
> the contract *"opaque host objects live as long as the heap"*. Take the
> finalizer, with the side table owning the `Arc<dyn Opaque>` and `host_index` a
> slab slot. Three grounds, each checked against code rather than inherited:
>
> - **The mechanism is not a majit change.** `MiniMarkGC::register_finalizer`,
>   `finalizer_next_dead`, `deal_with_objects_with_finalizers` and the
>   `FINALIZER_REGISTERED` dedup flag are all present and unit-tested in
>   majit-gc. The light-destructor path (`TypeInfo::destructor`) exists too and
>   is the *wrong* tool, not a missing one.
> - **Upstream puts this exact case on the full queue.** `W_CPPInstance`
>   (`pypy/module/_cppyy/interp_cppyy.py`) wraps a host object whose destruction
>   runs arbitrary host code and uses `register_finalizer`, registering only
>   when it owns the object. `finalizer-order.rst` restricts light destructors
>   to "objects that just need to free an extra block of raw memory" and forbids
>   calling any external C function from one — dropping an `Arc<dyn Opaque>`
>   runs a user `Drop`, so it fails that test by the doc's own sentence.
> - **CEL mints opaques during evaluation, so the contract would leak
>   unboundedly.** `impl_conversions!` (`macros.rs`) emits an
>   `IntoResolveResult` impl per row, and one row is `Arc<dyn Opaque>`, so a
>   user-registered host function may *return* a fresh opaque per call — inside
>   a comprehension, per element. That is the public extension point and it
>   survives P5, unlike the two in-repo producers (`optional.none()` /
>   `optional.of()` in `functions.rs`), which leave the opaque route for their
>   own leaves. And there is no reset point to clear the table at: §7 forbids a
>   dropped-at-return arena because `Program::execute` returns an unlifetimed
>   value and eleven `ExecutionError` variants carry one.
>
> **Accepted failure mode:** loss of prompt deterministic release. Today the
> last `Value` drop frees the host object at once; afterwards release waits for
> unreachability at a major plus the drain, so an opaque holding a scarce
> resource on an idle heap is held until the next evaluation or teardown.
> Document that hosts needing prompt release keep their own `Arc` and treat the
> binding as a borrow. The floor is benign: because the side table owns the
> `Arc`, the worst case — majors never firing — degrades exactly to the
> heap-lifetime contract, so this is a strict improvement over it rather than a
> competing bet.
>
> **Drain placement is part of the decision.** The collector-side trigger only
> schedules; the `Arc` drops run at the `Program::execute` boundary, *not* at
> the in-VM dispatch safepoint, because a user `Drop` may re-enter cel. Upstream
> avoids the same hazard by draining between bytecodes via `UserDelAction`.
>
> **Coupled constraint, recorded with D12:** post-P5 `dyn Opaque` must be
> contracted not to embed a `Value`/`CelRef`. The side table is not in §7's root
> set, so an entry holding one dangles; adding the table to the root set instead
> would invert D12 by making every opaque's referents heap-lifetime. Today's
> trait admits embedding — `OptionalValue` is an `Opaque` holding a `Value` —
> so this has to be a documented contract, not a compile-time bound.

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

> The two line counts are as of 2026-07-26 and are not maintained; `cel/src/majit`
> is 13561 lines on 2026-08-13. **The clause that matters has not moved:
> `Program::execute` still never reaches the tier.** Measured the same day: the
> crate's only `jit_merge_point!` / `can_enter_jit!` are in
> `cel/src/majit/bytecode.rs`, on `float_bank::run_mainloop_f`, and
> `cel/src/vm/` — the evaluator `Program::execute` actually runs — mentions
> majit in 0 of its 6 files. That is Step 2 restated as a census: the JIT traces
> the mirror, not the interpreter.

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

It is the step that makes everything after it possible.

> **This paragraph used to say the step was independently justifiable, because a
> flat bytecode VM normally beats a `Cow<dyn Val>` recursive walk on its own.
> That justification is void and the claim is now false.** It was written against
> the `Cow<dyn Val>` walker, and P2 deleted that walker. Measured against its
> replacement, `Value::resolve_value`, the VM's *run* half — compilation
> excluded, the code object owned by the `Program` — costs 1.5–3.9x the walker on
> flat expressions. Step 1 is therefore a performance regression on flat CEL, and
> has to be argued on convergence alone: it is the prerequisite for Steps 2–4,
> not a win in itself. The deficit is tracked as its own item.
>
> The one place the VM already wins is the comprehension accumulator, and only
> since it stopped rebuilding the result list per element: `xs.map(x, ..)` over
> 10 000 rows fell from 50 018 allocations to 18, against the walker's 19.

## Plan

**Step 1 — one interpreter, no JIT.**
CEL AST → a full-fidelity bytecode covering *all* of CEL (not the traceable
subset): lists, maps, strings, bytes, optionals, custom functions, regex.
A stack VM over `Value` executes it. `Program::execute` becomes that VM and the
recursive walker is deleted. Nothing about majit is involved. Gate: the entire
existing cel test suite, unchanged.

> **Landed 2026-08-09, without the deletion.** `cel::vm` is the bytecode
> compiler and stack VM, and `vm` is a default cargo feature, so
> `Program::execute` is that VM. The walker is NOT deleted — "What must NOT be
> done" above is why — and is reached as `Value::resolve` /
> `Value::resolve_value`, which is what `tests/oracle.rs` and
> `tests/vm_walker_sweep.rs` call directly, so both evaluators are gated
> whichever way the feature is set. `--no-default-features --features
> regex,chrono` is the configuration that hands a *caller* the walker;
> `.github/workflows/rust.yml` runs a test leg for it, because with `vm` on by
> default nothing else runs `Program::execute` on the walker.
>
> The gate held: `cargo test` and `cargo test --features vm` were run on the
> same tree before the flip and returned identical results — 127 lib + 2 + 4 +
> 2 + 2 integration + 15 doc tests, 0 failed in both.

**Step 2 — a merge point on the dispatch loop.**
Put `jit_merge_point` with `greens = [pc, code]` on the new VM's loop and
`can_enter_jit` on its back-edges, exactly as `run_mainloop_f` does today for
its subset.

> **Step 2 does not execute in this position, and it splits. Reviewed
> 2026-08-13.** "Exactly as `run_mainloop_f` does" means *under `#[jit_interp]`*
> — those two macros are inert on their own (`macro_rules! jit_merge_point`
> expands to nothing; the proc macro is what turns them into
> `driver.merge_point` / `driver.back_edge`). So Step 2 as written IS front-end A
> applied to `cel::vm`, i.e. the thing Step 3 exists to replace, attempted one
> step early. Worse, the only way to make front-end A digest a `Value`
> interpreter is to carry the values as `ref(T)`/`opaque(T)` handles with
> residual calls — a flat mirror state struct, which is **the second evaluator
> again**, the deviation this document exists to delete.
>
> What is genuinely blocking is narrower than "front-end A cannot take cel", and
> that broader claim is false on its face: front-end A *is* attached and running,
> on `float_bank::run_mainloop_f`. The blocker is the one this document's own
> section title already names — front-end A never emits `GuardClass` and has no
> heap-object model, so a traced `Value` dispatch cannot specialise `Value::Int`
> from `Value::String` at the merge point, which is the whole point of tracing it.
>
> There is also a prerequisite that belongs to **both** front ends and to neither
> of them exclusively: **the portal shape does not exist yet.** `cel_eval_loop`
> is a free function but takes no `pc` and does not contain the loop; the loop is
> `Vm::run` (a method, `pc` a local) and the opcode match is a third function,
> `Vm::step`.
>
> ⛔ **The reason given for 2a below was wrong, and 2a is not the refactor it
> describes. Refuted 2026-08-13.** The claim was that "`greens = [pc, code]` is
> unspellable against any current argument list, and front-end B binds
> greens/reds **by parameter name**", so the loop and its dispatch had to be
> reshaped into one free function whose parameters are the greens and reds.
> Both clauses are false, and each is refuted by front-end B's own code:
>
> - **Binding is positional off the marker call, not by parameter name.**
>   `jtransform`'s marker handler is a direct port of
>   `num_green_args = len(jitdriver.greens); greens = args[1:1+num_green_args];
>   reds = args[1+num_green_args:]` — the `greens: Vec<String>` in
>   `JitDriverSpec` supplies the *count* (and the dotted `red.field` greenfield
>   spelling), not a lookup key. `autodetect_jit_markers_redvars` finds the
>   `jit_merge_point` in *any* block of the portal graph and slices its argument
>   list; nothing consults the portal's signature.
> - **The greens need not be parameters of anything.** pyre's own production
>   portal — the one `generated.rs` configures with
>   `greens = [next_instr, is_being_profiled, pycode]` — is
>   `eval_loop_jit(frame: &mut PyFrame) -> LoopResult`. It takes **one**
>   argument. `pycode` is a local (`let code = …pyframe_get_pycode(frame)`),
>   `next_instr` is a loop local, and the single parameter is the *red*. cel's
>   `Vm::run(&mut self)` is already that shape, with `&mut Vm` as the red.
>
> So no reshaping of the production evaluator is warranted on this basis, and
> `Vm::run` can be the portal as it stands: the merge point needs `code` bound to
> a local beside the existing `pc`, which is 2b's work, not a prerequisite to it.
>
> **Nor is what looked like a second, genuine fault one.** Seeded at `vm::eval`,
> the closure is 29 jitcodes holding `cel_eval_loop`, `Vm::new` and
> `Vm::public_error` — everything `cel_eval_loop` calls **except** `Vm::run` —
> and so nothing of the loop or the opcode match; seeded at `Vm::run` it is 95
> jitcodes including `Vm::step` and `Vm::unwind`. `Vm::run` is never named in the
> `vm::eval` run at all: not as a jitcode, not as a rtyper skip, not as a prepass
> failure. That is `policy.py:48-84` working, not a defect. `look_inside_graph`
> computes `contains_loop = !find_backedges(graph).is_empty()` and returns
> `res && !contains_loop` unless the graph is `unroll_safe`, recording the
> refusal in `unsafe_loopy_graphs`; a refused graph becomes a residual call and
> never enters the closure. `Vm::run` **is** the dispatch loop, so it is refused
> by construction. `cel_eval_loop` (`match vm.run() { … }`, no loop) and
> `public_error` (no loop) are not, which is the whole of the asymmetry — and it
> is why `Compiler::emit` and the other `&mut self` methods come through fine.
>
> This is exactly why upstream registers an interpreter loop as a **portal** —
> a seed for `find_all_graphs` — rather than expecting it to be discovered as a
> callee, and why pyre configures `eval_loop_jit` as one. So the conclusion is
> the useful one: **2a is dissolved entirely, and 2b's portal is
> `vm::interp::Vm::run` exactly as it stands.** `cel_census_pipeline_vm_run` is
> already that configuration, and its 95-jitcode closure is what a cel
> `JitDriverSpec` would see.
>
> Read the plan as **2a → 3a → 2b → 3b → 4**:
>
> | | |
> |---|---|
> | **2a** | ~~portal shape (cel only, no majit)~~ — **dissolved**; see above. The reshape is not required and the discovery gap it was to fix is `policy.py`'s loopy-graph refusal working as designed. `Vm::run` is already portal-shaped |
> | **3a** | front-end B ANALYSIS over `cel.ullbc` — ~~corpus stale~~ refreshed and run; `cel::vm` lowers with zero walls |
> | **2b** | the merge point + jitdriver spec — needs 3a's answers |
> | **3b** | front-end B EXECUTION (guard_class, jitcodes) |
> | **4** | demote the columnar path |
>
> 3a is reachable today: `majit-translate`'s pipeline already runs over
> `cel-jit/build/llbc/cel.ullbc` in-process from its own cel census test, with
> empty fnaddr bindings and default host statics. It needs no portal, which is
> why it can size 2a instead of waiting on it — and it did: running it first is
> what produced both the B2 number above and 2a's refutation, at no cost to the
> production evaluator.

**Step 3 — move to front-end B.**
Extract LLBC for the `cel` crate and drive it through `majit-translate` instead
of the `#[jit_interp]` macro. `Value`'s dispatch becomes `guard_class` +
inlined `getfield`, the same treatment pyre's object model gets. This is the
step that makes the *whole* language traceable rather than a subset, because
the tracer now sees the real interpreter instead of a hand-written mirror.

> ⚠ **"the same treatment pyre's object model gets" is true of pyre's BEHAVIOUR,
> not of the route.** Reviewed 2026-08-13: pyre does reach `guard_class`, but
> through its hand-written tracer, not through `majit-translate`'s codewriter.
> The codewriter leg — the typeptr rewrite in the getfield/setfield transform,
> its insn byte, the assembler encoding, the dispatch arm and the blackhole impl
> — is the one missing piece; everything on either side of it already exists
> (the metainterp's `guard_class`, the heapcache's known-class tracking, the
> optimizer's `optimize_guard_class`, and GuardClass codegen in all four
> backends). So **cel on front-end B would be the first consumer of a route pyre
> itself does not exercise.** That is a real risk and it is the difference
> between Step 3 delivering class dispatch and Step 3 delivering residual calls.
>
> A second risk this list already gestures at ("cel would be its second client
> and would expose whatever those adapters paper over") is measurable rather
> than speculative: run the rtyper's Skip histogram on a cel pipeline run. If
> cel's graphs mostly fall through to the legacy annotator/resolver, front-end B
> hands cel the flat legacy walker rather than `guard_class`-quality typing, and
> most of this step's payoff evaporates. Turn the flag into a number before
> committing to the step.
>
> **The number, measured 2026-08-13 (B2).** `PYRE_RTYPER_VERBOSE=1` over
> `cel_census_pipeline_vm_run`, i.e. the closure seeded at `vm::interp::Vm::run`:
> **88 of 95 graphs take the legacy path**; 7 clear the two-phase gate. The
> partition is exact — every skip carries the single reason
> `two-phase: subject not annotated/rtyped in prepass`, and the prepass's own
> `[PREPASS histogram]` reports 85 phase-A + 3 phase-B failures, so
> 88 + 7 = 95 with nothing unaccounted. **The risk is confirmed as stated: today
> front-end B would hand cel the flat legacy walker for 93% of its closure.**
>
> What the number does *not* say is that this is architectural. The prepass
> classifies every failure itself; this is its own `[PREPASS histogram]`, summing
> to 85 + 3:
>
> | count | phase | orthodox disposition |
> |---|---|---|
> | 64 | A | `FUNCPATH-OTHER` (registry / residual) |
> | 12 | A | `CLASSDEF-LESS-FOREIGN` (foreign-type getattr) |
> | 3 | A | `UNION-PAIR-PORT` |
> | 2 | A | `RESIDUAL-CLOSURE` (`dont_look_inside`) |
> | 2 | A | `UNCLASSIFIED` |
> | 1 | A | `BLOCKED-BLOCK` |
> | 1 | A | `METHOD-RESOLUTION` |
> | 3 | B | `GETCLSFIELD` |
>
> Re-reading the same 85 messages by their *innermost* cause — the last
> `not registered in PyreCallRegistry` in a nested chain, since an outer frame
> only re-reports its callee's failure — splits that 64 further: 59 name a
> specific unregistered host path and 7 are a contained `compute_at_fixpoint`
> panic. Ranked by symbol, the head is **`sync::Arc::deref` at 16 of 88**, ahead
> of `BTreeMap::get`, `core::clone::impls::<Impl>::clone`, `Into::into`,
> `core::slice::<Impl>::get` and `::last_mut` at 3 each. `Arc` is how cel spells
> every shared `Value` payload, so that one registration is the largest single
> lever here, and the row as a whole is a registration backlog, not a design wall.
>
> The `CLASSDEF-LESS-FOREIGN` row is the opposite kind: a classdef-less
> `getattr("__pos_0")` is enum/tuple payload access, which is precisely the
> `Value` dispatch Step 3 exists to type. Those 12 are the ones that would still
> be legacy-walked after all the registry work, and they are the honest measure of
> how much of Step 3's payoff needs the codewriter leg rather than a registry entry.
>
> ⛔ And `build/llbc/cel.ullbc` on this box is STALE — written 2026-08-07,
> against `cel/src/vm/*` last moved 2026-08-09, so it predates the `vm`-default
> flip. This list's own risk line ("stale corpora silently mask changes") has
> already happened once here. Gate on `scripts/extract-llbc.py --check cel`,
> run bare, before reading it.

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

## Defect 4 — the preamble's copy of the exit guard, and the `[.. ; virt]` header

Defect 3 left the spread case giving up on most of its bridges. The rows that
gave up were the ones leaving through the **peeled preamble's** copy of the exit
guard rather than through the loop body's copy: two copies of ONE guard, with
byte-identical resume stream, `fail_arg_types` and frame pc, disagreeing on what
deadframe slot 0 held at runtime — a `&state` pointer through the body copy, a
small integer (`3`, `16`) through the preamble copy.

Which trip counts broke was deterministic and *moved with the trace-eagerness
threshold* — `(E-1) | threshold && E >= 3`, so {3,5,9} at threshold 8 and
{3,5,7,13} at threshold 12. That rules the selection law out as the defect: it
only decides whether the bad bridge→loop JUMP gets built at all.

The defect was in `#[jit_interp]`'s state layout. For each `[.. ; virt]` array
the macro synthesized a `(<arr>_ptr, <arr>_len)` inputarg pair, and `extract_live`
filled every `<arr>_ptr` with the same `self as *const Self`. `VmStateF` has two
virt arrays, so the loop **named its virtualizable twice** — entry contract
`n=29` with refs at positions 0 AND 2. A bridge's contract is not synthesized; it
is decoded from the guard's vable section, `[identity, elements…]` — `n=26` with
a ref only at 0. The bridge's JUMP into the loop's procedure token re-read
`sym.fregs_ptr`, which in its own 26-slot contract is an **Int**, so the JUMP
carried `InputArgInt` at position 2 — precisely the position the preamble guard
names as the vable identity. Piece 3 of defect 3 then correctly refused to
dereference it and gave up on the bridge.

**Fixed** by removing the synthesis rather than patching the bridge, because
RPython does not have it:

- `warmspot.py:529-538` — `jd.index_of_virtualizable = jitdriver.reds.index(vname)`.
  The virtualizable is ONE red.
- `virtualizable.py:139-144 load_list_of_boxes` — the vable list names it once,
  identity last.
- `virtualizable.py:150-153` — every array's length is read off the live object,
  never boxed.

So the macro now mints a single `__vable_identity` slot, `<arr>_len` inputargs
are gone, and `StateFieldLayout::total_slots` loses its `2·N` term. Both
contracts became `n=26` with a ref only at 0. This also unmasked a latent bug it
had been hiding: `initialize_virtualizable` scanned `original_boxes`
(`[Void; num_green_args] ++ live_values`) and fed the position it found to a
**reds-only** `input_arg_typed`; with a duplicated identity the first match
happened to land on a ref anyway. It now scans `live_values`.

Census effect at 4000 rows: the 0..32 spread goes 12959 → **959** deopts.

**Refuted along the way** (kept so they are not re-investigated): peel/numbering
(both copies number the same canonical box — `resume.rs:3768` replacement-walks
before numbering); the backend deadframe (dense, base 64 on both sides);
`resolve_failarg_opref` demoted-home/stale-ref asymmetry (every slot 0 resolved
through plain `ssa`); and a third arg-vector construction in
`close_into_merge_point_token` (it uses the same `collect_jump_args_with_boxes`
as the other close paths).

## Measured result: the tier is now a win, not a loss

`cel/examples/majit_nested_bench.rs` sweeps each shape over a geometric ladder of
row counts and reports the MINIMUM of interleaved rounds — interference can only
make a round slower, so the fastest round is the robust estimator, and a median
on a shared box swings the baseline several-fold. Compilation was inside the
timed region of every batch when these numbers were taken, so a single batch size
could not separate "the compiled code is slow" from "the batch was too short to
pay for compiling"; the ladder can. (The section below removes that fixed cost;
the ladder is still what shows it is gone.)

The same binary, built once against pre-defect-1 majit and once against HEAD
(three HEAD runs, one base run, all on the same loaded box):

| shape | deopts before | deopts after | jit/clean before | jit/clean after @640k | fitted steady |
|---|---|---|---|---|---|
| constant 8      | 1 per row | 9    | 0.04–0.05x | **7.8–9.2x** | 20–26x |
| alternating 8/9 | 1 per row | 210  | 0.02–0.04x | **4.3–5.1x** | 8–10x |
| cycle 4..12     | 1 per row | 1609 | 0.04–0.05x | **2.5–2.8x** | 6–7x |
| spread 0..32    | 1 per row | 959  | 0.08x      | **4.6–7.2x** | 8–13x |
| constant 64     | 1 per row | 10   | 0.23–0.26x | **4.8–5.1x** | (degenerate) |

"before" is flat across every batch size and the sweep prints `no swept size
where the JIT total wins` for all five — one deopt per row never amortises, so
turning the tier on made cel slower than not having it. "after" climbs with batch
size because the only fixed cost left is tracing and compiling, fitted at
10–25 ms, putting break-even at roughly 60k–180k rows. For comparison the FLAT
single-loop shape (`majit_ab`) runs 11.15x at 2M rows off a ~1.7 ms compile.

The floor check says the columnar pipeline is worth having in the first place —
`constant 8` at 640k rows:

| tier | ns per row/eval |
|---|---|
| the tree walker (`Program::execute` before `vm` became default) | 1498.89 |
| clean bytecode VM over the lowered program | 177.97 |
| **compiled majit trace** | **22.70** |

so the lowering alone buys ~8x and the JIT buys ~8x on top of that, ~66x
end to end. Pre-fix majit ran this at 3276 ns/row — slower than the tree-walker.

## The per-call regime, which is a different unit and answers differently

Measured 2026-08-13 at cel-jit `d60a60c`, `cel/examples/majit_vs_cometkim_percall`
in its documented regime (`--profile bench --features jit-cranelift`): one
expression, one FIXED activation, ONE evaluation timed. The ladder above times a
batch of many rows; this times a single `execute`, which is the unit the design
of record's stop-at-P5 gate is stated in. **They do not substitute for each
other, and the tier answers them oppositely.**

⚠ The denominator is **29** — the harness prints it (`28/29 lower to the
compiled tier; 29/29 are answered`, `custom_function` declining with
*"unsupported for majit lowering: call `add`"*). Earlier notes quoting `N/28`,
and the file's own doc comment saying "18 benchmark expressions", are both
against a denominator this run does not have.

**The lowering — the landed half of the epic — is the win here.** `clean`
(the plain bytecode VM over the lowered program, no tracing machinery) against
`stock` (the tree walker), over the 28 timed cases:

* faster on **22 of 28**, up to `all_comprehension` 8.01x, `exists_comprehension`
  6.78x, `real_world_policy` 4.78x, `string_operations` 3.75x;
* slower on 6, and one class is not marginal: `variable_access/hashmap` and
  `variable_access/resolver` both run **0.11x** — the walker answers them in
  ~5.4 ns and the VM takes ~50 ns. A bare variable read is where dispatch
  overhead has nothing to amortise against.

**The compiled tier does not engage at this unit.** Only **12 of 28** rows
compiled at all; for the other 16 the `majit` column is the tracing interpreter
printed under the compiled tier's heading, at a fixed **539.8–780.7 ns**
(median 583.0). At one row per call the batch loop has no back edge to get hot
on. Of the 12 that did compile, the tier wins only above roughly 100 elements:

| case | stock ns | majit ns | ratio |
|---|---|---|---|
| `map_list_scaling/10000` | 221084.2 | 11835.0 | **18.68x** |
| `filter_list_scaling/10000` | 329411.5 | 30671.2 | **10.74x** |
| `comprehension_scaling/500` | 23679.4 | 2970.8 | **7.97x** |
| `map_list_scaling/100` | 2639.2 | 1122.8 | **2.35x** |
| `comprehension_scaling/10` | 807.2 | 1072.6 | 0.75x |
| `filter_list_scaling/10` | 494.1 | 1138.6 | 0.43x |

Crossover sits between 10 and 100 elements on all three ladders.

⛔ This is a second, independent measurement pointing the same way as the design
of record's NO-GO: for a single evaluation the compiled tier is not what answers
the call, and the fixed cost that stops it is not in the compiled code. It does
**not** reproduce that gate's own figure — task #88 puts a compiled artefact's
fixed per-call cost at 34–92 µs, two orders above the ~583 ns floor seen here —
so the two are agreeing in direction on different quantities, and neither
number should be quoted for the other.

## The fixed cost: the driver and the program now outlive a batch

The break-even above was set entirely by trace + compile, and the arithmetic said
no per-compile trim could reach it: the compiled tier beat the clean VM by
155 ns/row, so winning under 10k rows needed the whole trace-and-compile to fit
in 1.55 ms, against a measured ~10 ms (optimize 1.65 ms, cranelift backend
3.72 ms at ~18 us/op, driver setup ~1.2 ms). Trimming all of it at once still
lands at 3.5–5 ms.

The cost was not the compiler's. It was that cel paid it **again for every
batch**, which upstream never does: the compiled procedure token lives on the
greens-keyed JitCell (`warmstate.py:157-199 wref_procedure_token`) and
`memmgr.py:23-69` keeps it for `max_age` generations. Two things forced the
repeat, and both had to go:

1. `batch_sum_program_trapping` emitted an `OP_LOAD_CONST` for each column base,
   for `n` and for the trap-word address. The green key is the program POINTER
   plus pc (`trace_ctx.rs green_key_raw`), so different data meant different
   words meant a different key. They are now plain reds, seeded into the initial
   register bank by `BatchSeed::regs`. Red data columns are the upstream norm:
   `rsre_core.py:384-385` keeps the regex pattern green and the subject string
   red, `micronumpy/loop.py:88-89` keeps array base storage red. They must stay
   *plain* reds — `promote()` inserts a `guard_value`, and a guard that fails
   every batch generates a bridge per batch (`rlib/jit.py`), which is per-batch
   recompilation renamed.
2. `run_mainloop_f` built its own `JitDriver`, so the compiled loop died with the
   call. The mainloop now takes `&mut JitDriver`, and `float_bank` keeps drivers
   in a thread-local map, so the driver outlives the call. This is the wasmi
   kernel's arrangement (`kernel.rs:1610`, `:3677-3689`).

   The address the key is built from stays put for a separate reason, and it
   takes **two** things, which is the part worth stating because getting one of
   them was measured to give wrong answers.

   The green key stores the code pointer AS A NUMBER (`with_typed_decision_key`,
   `key.values[2] = code_ptr as i64`), so two programs that ever hold one address
   build byte-identical keys and `comparekey` finds them EQUAL. That is a true
   collision, not a hash collision: there is no residual field to disagree on,
   and it surfaces as an earlier expression's answer rather than a crash.

   1. `LoweredF` builds each `BatchShape` once, behind a `OnceLock`, and *owns*
      the words, so the address cannot move under a live loop.
   2. A pooled driver holds an `Arc` on every program it has been keyed on
      (`float_bank::PooledDriver`), so the address cannot be *recycled* under one
      either.

   (1) alone is not enough and this was demonstrated, not argued: with the words
   owned by the lowering but not held by the driver, `majit::tests::
   batch_string_construction` goes from green to a run-varying wrong answer,
   because `DRIVERS` outlives every `LoweredF`. **An owner closes an
   address-recycling hazard only if it outlives every key that names it.**
   `a_dead_programs_address_does_not_carry_its_compiled_loop` pins it.

   An earlier arrangement interned the words in a capped thread-local table.
   That table's retention — never freeing one entry, and taking `DRIVERS` with
   it when the cap flushed — was doing (2)'s job by accident; it read like a
   cache and was load-bearing as an ownership edge.

Same binary, same box, five shapes over the same ladder:

| shape | jit/clean @640k before → after | fitted steady before → after | break-even |
|---|---|---|---|
| constant 8      | 7.8–9.2x → **28.74x** | 20–26x → 29.26x | 22–32k rows → **358** |
| alternating 8/9 | 4.3–5.1x → **9.57x**  | 8–10x → 9.64x   | → **620** |
| cycle 4..12     | 2.5–2.8x → **5.83x**  | 6–7x → 5.86x    | → **577** |
| spread 0..32    | 4.6–7.2x → **7.68x**  | 8–13x → 7.69x   | → **271** |
| constant 64     | 4.8–5.1x → **13.08x** | (degenerate)    | — |

The signature is that measured @640k now EQUALS fitted steady (28.74 vs 29.26,
9.57 vs 9.64, 5.83 vs 5.86, 7.68 vs 7.69): with no fixed cost left there is
nothing to amortise, so every size runs at the same rate. Trace + compile fits at
0.02–0.05 ms, and every shape wins at the smallest swept size — 10k rows, where
all five used to lose. Paying the compile in full, `constant 8` at 10k still runs
12.00x the clean VM.

Read the `cmp` column when comparing shapes in one run: the five shapes share one
expression, so the first shape to reach 10k rows compiles the loop and the rest
reuse it. That is the real shape of the win for cel's use case — one policy,
many batches — but it means a shape's own compile shows up only where `cmp` is
non-zero. A second batch of one expression compiles 0 times and still answers its
own columns' question, which `same_expression_second_batch_reuses_the_loop` pins.

A second data shape on a warm driver takes the existing loop's exit guard until
that guard is hot and then attaches a BRIDGE (`Traces bridged: 1` under
`MAJIT_STATS=1`), which is why the trace-census tests reset between shapes: they
pin per-shape tracing, not the reuse path.

Also landed here: cranelift's IR verifier now runs only under `debug_assertions`
(backend 3719 → 2783 us, steady-state unchanged), matching `compile.py:242-244`,
which runs the equivalent checks under `if not we_are_translated()`. And
`new_driver_f` passes `periodic_invalidation = false`, since there is no
quasi-immutable state here and the timer would re-invalidate what a persistent
driver just compiled.

### Still open

- ~~**`cycle 4..12` is the weakest shape** at 5.8x against 29x for a constant
  trip count.~~ **The ordering no longer holds.** Re-run 2026-08-13 on the same
  640k ladder, all five shapes in ONE interleaved run — which is what makes the
  *ordering* readable even though the box was loaded, since a within-run
  comparison pays the same interference on every arm — `cycle 4..12` is third of
  five at 6.25x, behind `alternating 8/9` (5.87x) and `constant 8` (5.92x), and
  ahead of `spread 0..32` (9.41x) and `constant 64` (15.37x). The 29x is not
  reproducible on this tree either, and that is NOT recorded here as a
  regression: the box was at load 36-89, three repeats of one panel swung 2.8x,
  both this crate and majit have moved, and the 177.97 ns/row clean-arm figure
  the 29x would be checked against comes from a different run than the ratio
  table it sits in. A quiet box would settle it; nothing short of one will.
- ~~**The JUMP-into-ptoken half of :3001-3007.** … It is NOT landed~~ —
  **LANDED**, in the parent repo, as PR #1125 (`majit: extract reusable JIT
  infrastructure from CEL`). The arm is `already_compiled_here` in
  `majit-metainterp`'s jitcode dispatch: it resolves the merge point's greens
  through `merge_point_green_key_hash`, asks `has_compiled_targets_fn`, and on a
  hit publishes `close_jump_into_key` / `close_greens` / `close_green_pc` for the
  driver instead of cutting a second copy of that loop. A key whose attempt
  already declined is latched by `cross_loop_close_declined`, so the optimizer is
  not re-run over a growing trace for a deterministic decline.
  The TIGHTENED census this asked for exists and is the way to re-measure it:
  `xloop_close_decision_reached` / `xloop_close_target_compiled` /
  `xloop_close_published` (`MC_DIAG` slots 68/69/70, also surfaced by
  `pyre-wasm-runner`). The first is bumped BEFORE the branch on purpose, so a
  zero in the other two separates "the walk never reached the decision" from
  "it reached it and the target was not compiled".
  ⛔ The deopt figures this item used to quote (trip count 3: 2101 → 401 deopts;
  `spread 0..32` 959 → 1763) were taken BEFORE the green-key unification and
  describe jumps into loops filed under keys nothing enters. They do not carry
  over and must not be cited again without a re-measurement.
- **Single-activation API.** None of this touches `Program::execute(&Context)`,
  which is how CEL is actually called. The JIT cell in `majit_ab`'s primary panel
  stays N/A until an `execute_jit(program, activation)` exists — and `majit_ab`
  reports the three real expressions as `lowerable`, so what is absent is the
  door, not the coverage.
  **The door is not the next step — but the reason first written here was wrong,
  and the correction changes what to fix.** The measurement stands: same binary,
  same run, 2026-08-13, stock cached `Program::execute` is 183 ns/eval while
  `warm_break_even`'s persistent-driver tier costs 708 ns for a ONE-row batch,
  against the clean VM's 42 ns. What was wrong is the causal sentence — that the
  708 ns is the cost of entering and leaving a compiled artifact.

  **At one row of that fixture nothing is compiled and nothing is entered.**
  Four independent legs, all on this tree:

  1. `LoweredF`'s batch lowering closes the loop as a do-while — `OP_ADD_IMM
     r_i,1` then `OP_JUMP_IF_ABOVE r_n, r_i, body_pc` — a BOTTOM test. At n=1 the
     comparison is `1 > 1`, false, so the back edge is not taken.
  2. The crate's only `can_enter_jit!` sits in `run_mainloop_f`'s
     `OP_JUMP_IF_ABOVE` arm under `if tgt < pc`, i.e. on that back edge alone.
  3. `JitDriver::merge_point` opens `if !self.meta.is_tracing() { return; }`, so
     the merge point cannot enter compiled code on its own; entry is `back_edge`.
  4. The COLD panel of the same run prints its own `compiles` column, and it
     reads **0 at rows=1 and rows=8**, first becoming 1 at rows=16.

  `warm_break_even`'s fixture is `engine_program` — `balance >= amount &&
  !frozen`, straight-line and scalar. ⚠ The reading does NOT generalise to a
  comprehension shape: a one-row batch of `items.all(i, i.price > 10)` DOES cross
  `can_enter_jit!`, because `LowerCtxF::emit_back_edge` puts a back edge in the
  spliced body whose trip count is the per-row `size(L)`, independent of n.

  So the 666 ns above the clean VM is `run_jit_persistent_f` overhead **with the
  JIT idle** — the `DRIVERS` pool remove/insert pair and its key hashing, the
  per-program `Arc` insert, `run_mainloop_f`'s two per-call bank allocations
  (`regs.to_vec()`, `vec![0.0; num_fregs]`), and one `is_tracing()` check per
  opcode from the expanded mainloop. None of that is the compiled-entry path.

  ⇒ **The door stays closed, but because the target is unmeasured, not because it
  is known to be expensive.** Two cheap probes separate the two costs, and until
  they are run nobody knows which to attack: (a) print
  `size_of::<PooledDriver>()`, on which the pool-churn hypothesis entirely rests;
  (b) `dispatch` in `batch.rs` already routes `Tier::Interpreter` and `Tier::Jit`
  through the SAME `run_jit_persistent_f`, differing only in `threshold_for`
  (`u32::MAX` vs the default) — so timing the Interpreter tier beside the other
  two splits it exactly: `Interpreter − Clean` is harness overhead with tracing
  permanently off, `Jit − Interpreter` is tracing plus compiled entry. The
  genuine entry cost must then be read at the smallest n where `compiles > 0`,
  never at n=1.

## Two evaluators means the second one must be checked against the first

Until the convergence above lands, the tier's correctness claim rests entirely on
"evaluator 2 answers what evaluator 1 answers". Three things now hold that up.

**`bool` is a type.** `ValType` had `Int`, `UInt`, `Float`, `Str`, `Timestamp`,
`Duration` — and no `Bool`, because a bool is `0`/`1` in the int register file
and storage was the only question asked. So the tests at `&&`, `||`, `!` and the
ternary condition were "is this operand in the int bank", which an `int` passes.
`a && b` with `a = 1, b = 2` lowered to the bitwise `OP_AND` and answered `3`;
the tree-walker raises `NoSuchOverload`. Reproducible on the clean tier, so it
was never a trace defect. `ValType::Bool` now exists and those operators require
it — CEL says `1 && 2`, `!1` and `1 ? x : y` are all `NoSuchOverload`,
`true + true` is unsupported, and `1 == true` is `false` rather than a bit
compare, while ordering IS defined (`false < true`) and lowers to the signed int
ops.

**An undeclared path declines.** `slot()` used to default an absent schema entry
to `ValType::Int`. The bank decides which operators a path is legal under, and
the caller builds its columns from the same declaration, so a default is a
silently chosen meaning. `select_chain_slots` had been asserting that
`account.balance >= txn.amount && !account.frozen` was "lowerable" with an empty
schema — i.e. with a bool read as an int. Only the derived `size(...)` /
`offset(...)` columns still get their type from the lowering, because they are
counts.

**A parity sweep, not a list of remembered cases.** `parity_sweep_binary_operators`
crosses two columns of every `ValType` plus one literal of each type with the
thirteen binary operators — 4693 expressions — and asserts the one property that
matters: when the lowering accepts and the machine answers, the answer equals
the tree-walker's, and the walker must have had an answer to give. Declining and
refusing mid-batch are legal and are counted (4159 / 6 / 528) rather than
asserted, so a change that quietly declines everything collapses the census
instead of going green. Restoring the int operand to the `&&` fold fails it with
`` `i && i`: the machine answered 17 where the tree-walker raises ``.

## The batch tier has an API now

The tier had internals and no API, so every caller repeated the encoding by
hand: column vectors in the lowering's slot order, base pointers, `intern_hash`
loops with their own injectivity check, Arrow offset buffers, register counts.
That hand-encoding is where the `bool`-as-`int` declaration came from.

`cel::majit::batch` is that encoding written once, in CEL's types:

```rust
let program = BatchProgram::compile("balance >= amount && !frozen", &schema)?;
let batch = Batch::new(n)
    .column("balance", ColumnRef::Int(&balance))
    .column("amount", ColumnRef::Int(&amount))
    .column("frozen", ColumnRef::Bool(&frozen));
let matching_rows = program.bind(&batch)?.sum()?;
```

`compile` is per expression, `bind` is per batch — the split the warm driver
needs. `bind` materializes what the schema does not declare (`0`/`1` for bools,
content-hash ids, `size(...)` lengths, `offset(...)` prefix sums), checks hash
injectivity, and builds the batch program once; `sum_on(tier)` runs it. Slot
order, banks and base pointers never reach the caller. Errors are split into the
permanent (`Lower`) and the data-dependent (`MissingColumn`, `ColumnType`,
`RowCount`, `HashCollision`, `Trapped`), all meaning "use `Program::execute`".

This is a **batch aggregate** API, not the single-activation one the section
above is still waiting on: the answer is `sum over rows of expr(row)`, because a
running total in a loop-carried accumulator is what the compiled trace has. A
per-row output column would need a store opcode whose PyPy justification has not
been verified.

Moving `majit_columnar_batch` onto it moved its JIT figure from 1.23 to 0.62
ns/row: it had been calling `run_jit_seeded_f`, which builds a driver per call,
so the flagship example had never measured the warm driver. `majit_ab` stays on
the raw entry points on purpose — both of its panels measure the cold path, a
fresh driver per run, which the API cannot express.
