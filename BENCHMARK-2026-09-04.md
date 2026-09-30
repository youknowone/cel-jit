# cel-jit vs cometkim's cel-jit — per-call benchmark report

**Date** 2026-09-04 · **Machine** Apple M5 Max (Mac17,6), quiet (load average ~4) ·
**Toolchain** rustc 1.98.0 · **Profile** `[profile.bench]` (lto, codegen-units = 1, opt-level 3) on both sides

| side | tree | what it is |
| --- | --- | --- |
| `ck-*` | `cel-rust/cel-rust` PR #233 head `4d57618` | cel 0.11.6 tree-walker + his Cranelift AOT backend |
| `our-*` | `youknowone/cel-jit` branch `majit` at `848d74c` | our tree-walker, our bytecode VM, our batch machine |

Every figure is **nanoseconds for ONE `execute` call** on a fixed activation built
outside the timer. Nothing is divided or multiplied by the element count. Where a
list is involved, the whole list is processed inside that one call.

---

## 1. Method, and what was done to make it fair

His benchmark (`cel-jit/benches/comparison.rs`) times
`b.iter(|| compiled.execute(&ctx))` under criterion. Ours times the same unit
under a thread-CPU timer: batches of at least 20 ms of CPU on the measuring
thread, best batch reported, so time spent descheduled by other work is not
charged to the measurement.

To remove the timer as a variable, **his 18 cases were re-measured inside his own
checkout with our timer** (`cel-jit/examples/percall_ck.rs`, added to the checkout,
not committed), and **his criterion bench was also run verbatim** beside it. The
two timers disagree as follows, on the same binary and the same cases:

| scale | criterion reads |
| --- | --- |
| single-digit ns (`simple_arithmetic` compiled, 6.8 vs 5.5) | +24 % |
| tens of ns (`variable_access/resolver` walker, 5.7 vs 4.2) | +8 % to +37 % |
| hundreds of ns and above (`list_map`, `real_world_policy`, the ladders) | within ±3 % |

Every `ck-*` number in this report is the one-harness number, so the comparison
carries no methodology term. His criterion numbers are quoted only where noted.

**Every timed closure unwraps its result.** A case whose compiled program answers
with an error is printed as *not answered*, not as a fast number.

### Deviations that remain, stated rather than hidden

1. **The two walkers are different versions.** `ck-walk` is cel 0.11.6;
   `our-walk` is our fork of the 0.14 line. On comprehensions the upstream walker
   became much slower between those versions, so *`ck-walk` is not a baseline this
   project improved.* The like-for-like axis is **his best tier against our best
   tier**: `ck-jit` vs `our-auto`.
2. **Activation handling differs by design.** His compiled function resolves every
   variable by name out of the `Context` on each call. Our batch tier resolves the
   activation to slots and encodes it into columns once, at `bind`. `bind` is
   measured separately and reported in §5 so a reader can put that cost back.
3. **Four of his cases do not answer.** `comprehension_scaling` (`items.filter(x, x
   % 2 == 0).map(x, x * 2)`) returns `Undeclared reference to '@result'` from his
   backend at all four sizes. His criterion does not unwrap, so it reports those as
   timings; they are timings of an early error return.
4. **The existing boards do not pass host functions to lowering.** Their
   `custom_function` row calls `BatchProgram::from_program`, which has no
   `Context`, so the batch tier declines and the row is answered by the walker.
   This is a harness limitation, not a library limitation:
   `BatchProgram::from_program_in` receives the registered functions and lowers
   the same expression successfully (§6).

---

## 2. Per-call results

`our-VM` is `Program::execute` — the door a caller who names no features goes
through. `our-auto` is the batch machine choosing its own tier. Bold marks the
best number in the row.

| case | ck-walk | ck-jit | our-walk | our-VM | our-auto | tier `auto` chose |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| simple_arithmetic | 40.9 | 5.5 | 41.1 | 70.6 | **4.7** | interpreter (`clean`) |
| comparison | 33.7 | 5.4 | 28.2 | 48.0 | **4.5** | interpreter (`clean`) |
| conditional | 32.5 | 19.0 | 26.9 | 50.0 | **12.2** | interpreter (`clean`) |
| nested_expression | 138.2 | 127.8 | 144.7 | 197.5 | **19.2** | interpreter (`clean`) |
| variable_access/hashmap | 5.6 | 11.3 | **4.2** | 23.4 | 4.7 | interpreter (`clean`) |
| variable_access/resolver | **4.2** | 10.4 | 4.6 | 23.6 | 4.7 | interpreter (`clean`) |
| member_access | 124.1 | 150.9 | 81.4 | 124.8 | **12.0** | interpreter (`clean`) |
| list_indexing | 58.2 | 66.7 | 61.1 | 124.3 | **18.2** | interpreter (`clean`) |
| list_filter | 1421.3 | 991.2 | 413.3 | 408.8 | **75.2** | interpreter (`clean`) |
| list_map | 857.9 | 744.5 | 262.1 | 232.2 | **65.1** | interpreter (`clean`) |
| all_comprehension | 473.8 | 168.7 | 331.4 | 257.2 | **4.6** | interpreter (`clean`) |
| exists_comprehension | 359.4 | 146.0 | 273.7 | 191.6 | **4.7** | interpreter (`clean`) |
| string_operations | 223.3 | 219.7 | 159.4 | 190.2 | **4.4** | interpreter (`clean`) |
| custom_function | **69.0** | 414.0 | 113.4 | 146.3 | — | walker (lowering declines) |
| real_world_policy | 419.9 | 475.0 | 293.2 | 406.5 | **17.4** | interpreter (`clean`) |

`variable_access/resolver`: 4.2 vs 4.7 ns is within the run-to-run spread of this
harness; read the two as a tie.

**Summary of the like-for-like axis.** `our-auto` is faster than `ck-jit` on all
25 cases his backend answers. Our bytecode VM *alone*, with no batch machine,
beats his compiled tier on **16** of those 25 and loses on nine:
`simple_arithmetic`, `comparison`, `conditional`, `nested_expression`, both
`variable_access` cases, `list_indexing`, `all_comprehension` and
`exists_comprehension`. **`custom_function` is the one case where every one of our
evaluators loses to his side's walker.**

**Note the last column.** Across the whole of this table `auto` did not use the
JIT — it routed every row to the batch machine's *interpreter*. §6 takes that up.

---

## 3. The size ladders, and how not to read them

`list.map(x, x * 2)` over a list of `N` elements, one call:

| N | ck-jit ns/call | vs previous rung | ck-jit ns/element | our-VM ns/call | vs previous rung | our-VM ns/element |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 183 | — | 183.0 | 99 | — | 99.3 |
| 10 | 1 517 | 8.3× | 151.7 | 259 | 2.6× | 25.9 |
| 100 | 26 975 | 17.8× | 269.8 | 1 685 | 6.5× | 16.9 |
| 1 000 | 1 427 200 | 52.9× | 1 427.2 | 15 639 | 9.3× | 15.6 |
| 10 000 | 118 990 000 | 83.4× | 11 899.0 | 154 491 | 9.9× | 15.5 |

> **Read this as an algorithm difference, not a code-generation difference.**
> Ten times the elements costs his side 53–83× and ours 9.3–9.9×. cel 0.11.6's
> `map`/`filter` rebuild the accumulator list on every iteration, which is
> quadratic, and a JIT cannot change the complexity of the loop it compiles: his
> walker and his compiled tier sit on the same curve (121.68 ms vs 118.99 ms at
> N = 10 000, a 1.02× difference, from his own criterion run). Our side became
> linear through interpreter and lowering work, and *our own tree-walker is linear
> too* (180 µs at N = 10 000), which is what shows the change is not the JIT.
>
> Consequently the ratio between the two sides grows in proportion to N — 2.9× at
> N = 1, 11× at 10, 130× at 100, 2 507× at 1 000, 24 632× at 10 000. **A number
> from the tail of that ladder says how far apart the two complexities are, and
> nothing about compiler quality.** Compare code generation on the N = 1–10 rungs
> and on the scalar cases in §2 instead.
>
> Sanity check on the units, because a ratio that tracks N is also what a
> per-element figure mistaken for a per-call one would produce: our column is not
> flat in N (99 → 154 491 ns), so it is a per-call figure; a per-element figure
> would be the 15.5 ns in the last column. The same ladder measured through a
> second, unrelated harness reproduces the same growth.

The other two ladders behave the same way (`filter_list_scaling`: 182 ns → 32.6 ms
on his side, 114 ns → 278 µs on ours). His `comprehension_scaling` does not answer.

---

## 4. Batch throughput (a different question)

The per-call table above answers *"how long does one evaluation take?"*. Where a
caller has many rows, the batch machine answers a different question. 50 000 rows,
ns per row, our tree-walker as the control:

| case | stock ns/row | clean VM ns/row | compiled ns/row | compiled ÷ clean |
| --- | ---: | ---: | ---: | ---: |
| simple_arithmetic | 44.4 | 3.72 | 0.24 | 15.22× |
| comparison | 29.3 | 2.01 | 0.25 | 8.03× |
| conditional | 64.8 | 4.66 | 0.83 | 5.64× |
| nested_expression | 141.2 | 11.38 | 4.86 | 2.34× |
| member_access | 102.7 | 4.90 | 0.46 | 10.70× |
| list_indexing | 76.3 | 8.10 | 1.05 | 7.73× |
| list_filter | 407.0 | 42.58 | 9.66 | 4.41× |
| list_map | 264.3 | 10.59 | 5.40 | 1.96× |
| all_comprehension | 344.6 | 4.69 | 0.46 | 10.10× |
| string_operations | 156.2 | 1.87 | 0.29 | 6.41× |
| real_world_policy | 167.8 | 15.06 | 1.09 | 13.83× |

17 of his 18 expressions reach the compiled tier; all 18 are answered.
Only the last column isolates *compilation* — the ratio against the walker also
contains the data-model change (slot resolution, unboxed columns).

**This table is not comparable to his numbers** and none are quoted in it: his
regime is one call on a fixed activation, this one is throughput over varying rows.

---

## 5. Putting the activation cost back

Our batch tier's advantage in §2 partly reflects that it resolves the activation
once, at `bind`, where his compiled function resolves names on every call. `bind`
is the whole per-activation encoding — more than name resolution — so adding all of
it back is a pessimistic bound on that design difference:

| case | our-auto | bind | of which name resolution | of which columnar encode |
| --- | ---: | ---: | ---: | ---: |
| simple_arithmetic | 4.7 | 57.0 | 9.4 | 43.8 |
| conditional | 12.2 | 78.0 | 21.2 | 55.6 |
| member_access | 12.0 | 107.1 | 40.7 | 63.2 |
| all_comprehension | 4.6 | 57.3 | 9.6 | 44.1 |
| list_map | 65.1 | 68.4 | 9.7 | 55.9 |

A caller that evaluates a fresh activation exactly once really does pay this, and
several §2 wins do not survive it. A caller that evaluates the same shape
repeatedly pays it once.

---

## 6. What this report does and does not establish

### `auto` is a tier selector, and on the per-call table it did not pick the JIT

`auto` chooses between the batch machine's plain-Rust interpreter over lowered
columnar code (`clean`) and its compiled trace (`jit`). Across the 28 per-call
cases it chose:

| tier | cases |
| --- | --- |
| `clean`, not compiled | 18 — every row of the §2 table, plus the N = 1 and N = 10 ladder rungs |
| `jit`, compiled | 10 — the three ladders from N = 100 up |

So the §2 wins are **not** wins by a JIT. They come from lowering (constant
folding, slot resolution, unboxed operands) and from a faster interpreter over
that lowered code. Naming the tier `auto` chose is the point of the last column.

**The reason is not that the JIT fails to fire.** It fires on every one of these
calls. The board's `enter/call` column reads `1.00` for every row of §2, which
counts calls that entered compiled code. A bottom-tested row loop at one row
takes no back edge, so the loop's own door never warms — but a second door,
keyed on function entry, counts CALLS rather than rows, and the repeated one-row
calls a per-call board makes are exactly what warms it.

What loses is the price of going through that door. Differencing two batch sizes
separates the compiled tier's per-row work from its per-call fixed cost:

| case | compiled work per row | fixed cost per call | interpreter, whole call |
| --- | ---: | ---: | ---: |
| simple_arithmetic | 0.3 ns | 111.2 ns | 4.8 ns |
| conditional | 0.7 ns | 106.4 ns | 12.3 ns |
| member_access | 0.4 ns | 108.2 ns | 11.7 ns |
| real_world_policy | 0.8 ns | 127.8 ns | 17.5 ns |
| list_map | 1.1 ns | 121.3 ns | 62.4 ns |

The compiled code is 15× to 60× cheaper per row than interpreting the same
lowered program. Entering it costs 85 to 138 ns, which is more than the entire
interpreted call. **At one row the JIT is the slowest thing in this codebase, and
it is the door and not the code that makes it so.** `auto` reads that correctly
and routes to the interpreter.

### Measured directly: the crossover, and the door warming up

The account above is inferred from two boards that each sit at one end of the
range. A third bench, `cel/examples/jit_regime.rs`, measures the range itself:
the same bound program run by both tiers at every batch size, and a one-row
program called twenty thousand times with the counters read as it warms.

**Where compiling starts to pay** (`Tier::Clean` and `Tier::Jit` on one bound
batch; `enter/call` was 1.00 and `gfail/call` 0.00 at every size, so nothing
below is a deopt artifact):

| case | crossover | jit ns at 1 row | interpreter ns/row | compiled ns/row (floor) |
| --- | ---: | ---: | ---: | ---: |
| `list.exists(e, e > 100)` | **4 rows** | 154 | 84–107 | 12.9 |
| policy, four conjuncts | **8 rows** | 116 | 23–25 | 3.2 |
| `x > 10 ? x * 2 : x + 5` | **16 rows** | 138 | 12–16 | 2.8 |
| `((a+b)*(c-d))/((e+f)-(g*h))` | **16 rows** | 126 | 21–24 | 7.0 |
| `add(x, y) + multiply(a, b)` | **32 rows** | 136 | 13–15 | 6.1 |

Four to thirty-two rows, where a decomposition of the same machine in August 2026
put the break-even at three to five thousand. The compiled tier is not waiting
for a batch any more; it is waiting for a handful of rows.

One case reverses at the top: `nested_arithmetic` improves to 7.0 ns per row at
512 rows and falls back to 13.7 at 8 192, while the interpreter beside it stays
flat. Its guard-failure rate is zero, so it is not deoptimizing; eight integer
columns at 8 192 rows is half a megabyte of input, and the compiled tier is the
only one of the two fast enough to notice. Recorded as an observation, not
diagnosed.

**The door warming up**, one row per call, no warm-up given, counters read as it
happens:

| calls so far | window ns/call | compiles | entered compiled code |
| ---: | ---: | ---: | ---: |
| 1 | 542–1 750 | 0 | 0 |
| 10 | 108 000–217 000 | 1 | 3 |
| 100 | 144–176 | 1 | 93 |
| 1 000 | 136–179 | 1 | 993 |
| 20 000 | 137–171 | 1 | 19 993 |

Read the second row as the compile itself — one to two milliseconds, charged to
the nine calls in that window — and the rest as steady state. **After roughly
seven one-row calls the program is compiled, and every call from the eighth on
enters compiled code.** A program whose row body contains its own loop
(`list.exists`) compiles on call one instead, from that inner loop's back edge.

And it stays slower than not compiling at all: settled, 132–165 ns through the
compiled door against 18–98 ns interpreted, 1.7× to 7.4×. So the answer to *"the
requests keep coming, doesn't it eventually get hot?"* is that it gets hot almost
immediately, and being hot is not what is missing. The entry path is.

### Where compilation does pay, measured against our own interpreter

The honest measure of what compilation buys is our compiled tier against our
`clean` tier — the same lowered program, same columns, one running it and one
compiled from it:

| case | clean ns/call | jit ns/call | compilation buys |
| --- | ---: | ---: | ---: |
| map_list_scaling/100 | 528.0 | 201.2 | 2.62× |
| map_list_scaling/1000 | 3 802.3 | 615.2 | 6.18× |
| map_list_scaling/10000 | 36 583.7 | 4 967.2 | 7.37× |
| filter_list_scaling/1000 | 6 357.6 | 567.8 | 11.20× |
| filter_list_scaling/10000 | 59 134.2 | 4 565.9 | 12.95× |
| comprehension_scaling/500 | 3 836.6 | 450.5 | 8.52× |

and, in the 50 000-row batch regime of §4, 1.96× to 15.22× depending on the
expression. **That is this project's JIT result: 2–15×, and only once there is a
loop with enough trips, or enough rows, to be worth tracing.**

### Does this prove an advantage on real CEL workloads?

Partly, and the part it does not prove should be stated plainly.

**What it does support.** For a caller who compiles once and evaluates many rows
of the same shape — log and event filtering, feature extraction, bulk policy
scoring — the compiled tier is 2–15× over an already-lowered interpreter, and the
whole stack is far ahead of the PR-233 tree. For a caller who evaluates one row
at a time, our default door (`Program::execute`) is competitive with a Cranelift
AOT backend, winning 16 of 25 and losing 9.

**What it does not support.**

1. **The dominant real-world CEL shape is one expression, one fresh activation,
   once per request** — admission control, authorization conditions, routing
   rules. In that shape the JIT fires and still loses, because the door costs
   more than the work. The evidence above says the product is fast there; it says
   the JIT is not why, and it names the entry cost as what would have to change.
2. **A fresh activation per call is not free.** §5 measures `bind` at 57 to 273 ns,
   which is more than several `auto` figures in §2 are. Where the activation is
   genuinely new every call, the honest per-request number is `our-VM`, not
   `our-auto`, and against `ck-jit` that column loses 9 of 25.
3. **Seven of the fifteen §2 rows do not depend on their input.**
   `simple_arithmetic`, `comparison`, `all_comprehension`, `exists_comprehension`
   and `string_operations` are constant expressions, and both `variable_access`
   cases are a single variable read. Our lowering answers those at bind time by
   projection or constant fold, which is why they read 4.4 to 4.7 ns. His AOT
   folds some of the same ones (5.5 ns on `simple_arithmetic`) and not others
   (168.7 ns on `all_comprehension`). These rows measure a lowering feature, and
   no real policy evaluates a constant.
4. **This is a microbenchmark set, not a workload.** Eighteen expressions chosen
   to exercise features. `real_world_policy` is the only one shaped like a
   deployed policy, and it is a single row answered by the interpreter.
5. **One coverage figure in this report is a harness artifact, not a limit.**
   `custom_function` is reported as declining, and both existing boards do
   observe that — but both call `BatchProgram::from_program`, which is not given
   the `Context` holding the user's functions. `from_program_in` is, and takes
   them. So that row measures a benchmark that did not pass the functions along,
   and the coverage claim it supports should be withdrawn until re-measured.
   Host functions are common in real deployments, which makes this the wrong
   thing to be wrong about.

**A fair one-sentence claim from this data:** on cometkim's expression set our
lowered interpreter matches or beats his Cranelift AOT backend per call without
compiling anything, and our compiler adds a further 2–15× once a workload has
loops or rows to amortize it — which is a claim about the *stack*, with the JIT
earning its place only in the batch regime.

## 7. Reproducing

```sh
# our side (from the cel-jit checkout)
cargo run --profile bench -p cel --features jit-cranelift --example majit_vs_cometkim_percall
cargo run --profile bench -p cel --features jit-cranelift --example majit_vs_cometkim
cargo run --profile bench -p cel --features jit-cranelift --example jit_regime

# his side (from a PR-233 checkout: fetch cel-rust/cel-rust pull/233 head 4d57618)
cargo bench -p cel-jit --bench comparison           # his own harness
cargo run --profile bench -p cel-jit --example percall_ck   # his cases, our timer
```

Measure on an idle machine. Under load above roughly 12, even thread-CPU figures
stop being repeatable; the numbers in this report were taken at load average ~4.
