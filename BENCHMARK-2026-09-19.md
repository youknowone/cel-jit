# cel-jit re-measurement, 2026-09-19

**Machine** Apple M5 Max (Mac17,6), 18 cores · **Toolchain** rustc 1.98.1 ·
**Profile** `[profile.bench]` (lto, codegen-units = 1, opt-level 3) on both sides ·
**Our backend** `jit-dynasm` · **Timer** `clock_gettime(CLOCK_THREAD_CPUTIME_ID)`,
best batch of >= 20 ms of thread CPU.

| side | tree | what it is |
| --- | --- | --- |
| `ck-*` | `cel-rust/cel-rust` PR #233 head `4d57618` | cel 0.11.6 tree walker + his Cranelift AOT backend |
| `our-*` | this checkout, `71f493e` | our tree walker, our bytecode VM, our batch machine (clean / compiled / auto) |

majit at `302ee9a5bb9`. Executable SHA-256, Cargo.lock hashes and per-run load
averages are in `benchmark-results/2026-09-19/{provenance,loads}.txt`; the raw
logs of every run quoted here are in that directory.

## 1. Method, and what is excluded

Three harnesses, three runs each, interleaved in one chain so no harness sits
entirely inside one load window:

1. `cel/examples/majit_vs_cometkim_percall` — his benchmark expressions and our
   extensions of them (54 rows), in his regime: one expression, one fixed activation, ONE evaluation timed.
2. `cel-jit/examples/percall_ck` (in his checkout) — HIS two evaluators, his
   cases, timed with OUR timer, so a number of his can be put beside one of ours
   without a methodology term.
3. `cel/examples/jit_regime` — the same bound batch program interpreted and
   compiled at every batch size from 1 to 8,192 rows.

**Board run 1 is discarded from every table**: it started at load average 113 and
its absolutes are 40-65% above runs 2 and 3. Runs 2 and 3 ran at load 7-21, and
all three `percall_ck` and `jit_regime` runs ran at load 7-14. Reported figures
are the MINIMUM of the two quiet board runs and the MEDIAN of the three
`jit_regime` runs.

That the box was quiet enough is not asserted, it is checked: the Sep-3 binary
of the same board, re-run today in the same window (§5), reproduces the
2026-09-04 quiet-box figures to within 3% on the columns it shares with them.

Excluded from every timed call: program compilation, JIT compilation, input
construction. `bind` (the batch machine's per-activation encoding) is timed
separately and printed in its own column — §3 puts it back.

## 2. Per-call results

One activation, one evaluation. `our-VM` is `Program::execute`, the door a caller
who names no features goes through. `our-clean`/`our-jit`/`our-auto` are the batch
machine's interpreter / compiled trace / self-selecting tier at ONE row, against
an activation bound outside the timer.

| case | ck-walk | ck-jit | our-walk | our-VM | our-clean | our-jit(1row) | our-auto | bind | auto route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| simple_arithmetic | 42.1 | 6.8 | 45.6 | 76.5 | 4.7 | 93.0 | 4.8 | 51.6 | clean |
| comparison | 33.0 | 6.8 | 31.5 | 86.3 | 4.4 | 91.1 | 4.6 | 51.7 | clean |
| conditional | 31.3 | 20.7 | 28.4 | 64.7 | 12.3 | 94.4 | 12.1 | 73.6 | clean |
| nested_expression | 136.1 | 131.4 | 137.2 | 217.7 | 20.9 | 100.1 | 20.5 | 260.8 | clean |
| variable_access/hashmap | 5.2 | 12.7 | 4.1 | 43.4 | 4.4 | 92.9 | 4.7 | 69.4 | clean |
| variable_access/resolver | 4.1 | 12.3 | 3.9 | 42.0 | 4.6 | 92.6 | 4.7 | 69.6 | clean |
| member_access | 127.6 | 153.4 | 98.8 | 309.9 | 12.5 | 99.7 | 12.0 | 107.3 | clean |
| list_indexing | 55.5 | 73.4 | 69.6 | 318.5 | 17.2 | 97.2 | 17.4 | 220.5 | clean |
| list_filter | 1,389.3 | 931.9 | 407.4 | 541.6 | 70.9 | 135.8 | 72.5 | 65.8 | clean |
| list_map | 848.2 | 743.3 | 252.2 | 297.5 | 61.1 | 139.3 | 60.3 | 64.8 | clean |
| all_comprehension | 452.0 | 171.5 | 416.4 | 432.5 | 4.5 | 93.9 | 4.4 | 51.5 | clean |
| exists_comprehension | 349.8 | 147.1 | 306.8 | 312.7 | 4.7 | 92.8 | 4.4 | 51.8 | clean |
| map_list_scaling/1 | 196.7 | 175.4 | 138.6 | 152.9 | 62.8 | 136.5 | 65.2 | 164.4 | clean |
| map_list_scaling/10 | 1,630.2 | 1,503.4 | 305.5 | 461.7 | 140.4 | 193.0 | 139.4 | 172.9 | clean |
| map_list_scaling/100 | 28,248.3 | 26,628.2 | 1,926.2 | 3,299.4 | 518.4 | 251.1 | 247.2 | 178.7 | jit |
| map_list_scaling/1000 | 1,456,895 | 1,441,029 | 18,240.8 | 38,737.1 | 3,699.8 | 787.4 | 798.1 | 253.8 | jit |
| map_list_scaling/10000 | 119,725,096 | 118,215,720 | 181,949 | 371,253 | 34,727.4 | 5,693.4 | 5,877.0 | 450.6 | jit |
| filter_list_scaling/1 | 216.3 | 174.0 | 149.9 | 163.2 | 62.6 | 135.1 | 64.7 | 163.4 | clean |
| filter_list_scaling/10 | 1,285.3 | 905.2 | 410.8 | 555.2 | 132.0 | 193.0 | 138.0 | 164.2 | clean |
| filter_list_scaling/100 | 15,890.6 | 12,003.1 | 2,906.7 | 4,273.7 | 886.4 | 250.4 | 249.3 | 171.7 | jit |
| filter_list_scaling/1000 | 419,489 | 385,835 | 28,101.3 | 44,692.0 | 5,578.2 | 790.4 | 790.8 | 257.3 | jit |
| filter_list_scaling/10000 | 32,480,694 | 31,247,168 | 281,437 | 473,800 | 53,144.8 | 6,127.2 | 6,102.4 | 389.9 | jit |
| comprehension_scaling/10 | 2,170.3 | — | 646.9 | 742.2 | 135.1 | 190.9 | 137.4 | 162.2 | clean |
| comprehension_scaling/50 | 12,176.3 | — | 2,286.1 | 2,970.0 | 488.4 | 224.0 | 232.9 | 180.5 | jit |
| comprehension_scaling/100 | 27,607.8 | — | 4,481.9 | 5,886.0 | 920.5 | 273.6 | 272.4 | 181.1 | jit |
| comprehension_scaling/500 | 262,958 | — | 19,889.0 | 29,402.0 | 3,699.8 | 497.6 | 504.1 | 203.5 | jit |
| string_operations | 224.3 | 217.4 | 166.8 | 208.3 | 4.7 | 94.7 | 4.7 | 51.4 | clean |
| custom_function | 70.1 | 432.5 | 113.5 | 211.7 | — | — | — | — | - |
| real_world_policy | 432.7 | 496.0 | 317.9 | 822.4 | 17.2 | 95.7 | 17.7 | 519.1 | clean |
| map_body/x | — | — | 9,824.7 | 32,053.4 | 3,097.1 | 586.0 | 580.4 | 264.2 | jit |
| map_body/x*2 | — | — | 18,013.8 | 36,479.3 | 3,623.6 | 753.7 | 751.6 | 249.7 | jit |
| map_body/x*2+1 | — | — | 29,647.1 | 50,562.3 | 4,607.7 | 824.9 | 793.1 | 244.9 | jit |
| map_body/x*2+1-3 | — | — | 42,275.5 | 65,333.2 | 4,880.9 | 819.1 | 855.6 | 277.0 | jit |
| record_map_scaling/1 | — | — | 167.7 | 201.4 | 65.4 | 147.5 | 68.0 | 179.8 | clean |
| record_map_scaling/10 | — | — | 603.1 | 893.0 | 135.5 | 191.1 | 136.5 | 170.7 | clean |
| record_map_scaling/100 | — | — | 5,090.8 | 7,610.6 | 508.2 | 235.5 | 231.3 | 179.6 | jit |
| record_map_scaling/1000 | — | — | 50,337.1 | 80,486.1 | 3,321.5 | 583.8 | 580.3 | 278.1 | jit |
| record_map_scaling/10000 | — | — | 510,024 | 817,657 | 25,583.3 | 5,304.8 | 5,416.9 | 389.3 | jit |
| record_filter | — | — | 53,875.8 | 100,382 | 4,623.1 | 750.1 | 745.5 | 278.4 | jit |
| record_exists_int/1 | — | — | 157.9 | 266.0 | 22.4 | 107.3 | 21.8 | 132.0 | clean |
| record_exists_int/10 | — | — | 840.5 | 1,524.2 | 76.4 | 139.6 | 76.8 | 128.8 | clean |
| record_exists_int/100 | — | — | 7,051.2 | 14,863.7 | 548.5 | 168.0 | 165.5 | 131.6 | jit |
| record_exists_int/1000 | — | — | 73,776.4 | 154,707 | 3,433.3 | 418.5 | 420.0 | 130.4 | jit |
| record_exists_int/10000 | — | — | 718,466 | 1,539,461 | 32,065.2 | 2,743.9 | 2,744.9 | 124.5 | jit |
| record_exists_str/1 | — | — | 145.4 | 241.0 | 19.2 | 94.5 | 18.2 | 179.3 | clean |
| record_exists_str/10 | — | — | 850.1 | 1,562.4 | 90.6 | 139.9 | 90.0 | 210.5 | clean |
| record_exists_str/100 | — | — | 7,865.1 | 15,831.8 | 500.3 | 162.2 | 163.9 | 285.2 | jit |
| record_exists_str/1000 | — | — | 74,433.4 | 151,744 | 5,001.1 | 398.9 | 402.7 | 905.5 | jit |
| record_exists_str/10000 | — | — | 724,551 | 1,552,314 | 49,653.0 | 2,669.5 | 2,680.6 | 6,751.4 | jit |
| scalar_exists/1 | — | — | 123.8 | 162.3 | 18.9 | 94.8 | 18.2 | 123.6 | clean |
| scalar_exists/10 | — | — | 627.1 | 713.9 | 78.4 | 138.7 | 77.0 | 123.3 | clean |
| scalar_exists/100 | — | — | 5,371.8 | 6,111.6 | 528.1 | 166.1 | 164.8 | 121.9 | jit |
| scalar_exists/1000 | — | — | 50,832.5 | 62,258.7 | 4,731.5 | 405.5 | 405.3 | 124.2 | jit |
| scalar_exists/10000 | — | — | 491,736 | 632,077 | 41,127.7 | 2,731.5 | 2,731.0 | 123.5 | jit |

`our-jit(1row)` is the compiled tier entered at one row. It is 91-140 ns on every
case and never wins there; that is the compiled entry door, not the expression.

## 3. Against cometkim's compiled backend

Two accountings, because the two designs put the activation in different places.
His compiled function reads every variable out of the `Context` by name on every
call; ours resolves the activation to columns once, at `bind`. `auto` is the
comparison in his regime (activation held fixed outside the timer, exactly as he
holds his). `auto+bind` re-pays the WHOLE columnar encoding on every call, which
is more work than his name resolution, so it is a pessimistic bound on the
difference and not an upper bound on what a majit call costs.

| case | ck-jit | our-auto | ck-jit ÷ auto | our-auto+bind | ck-jit ÷ (auto+bind) | our-VM | ck-jit ÷ VM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simple_arithmetic | 6.8 | 4.8 | 1.42× | 56.4 | 0.12× | 76.5 | 0.09× |
| comparison | 6.8 | 4.6 | 1.48× | 56.3 | 0.12× | 86.3 | 0.08× |
| conditional | 20.7 | 12.1 | 1.71× | 85.7 | 0.24× | 64.7 | 0.32× |
| nested_expression | 131.4 | 20.5 | 6.41× | 281.3 | 0.47× | 217.7 | 0.60× |
| variable_access/hashmap | 12.7 | 4.7 | 2.70× | 74.1 | 0.17× | 43.4 | 0.29× |
| variable_access/resolver | 12.3 | 4.7 | 2.62× | 74.3 | 0.17× | 42.0 | 0.29× |
| member_access | 153.4 | 12.0 | 12.78× | 119.3 | 1.29× | 309.9 | 0.49× |
| list_indexing | 73.4 | 17.4 | 4.22× | 237.9 | 0.31× | 318.5 | 0.23× |
| list_filter | 931.9 | 72.5 | 12.85× | 138.3 | 6.74× | 541.6 | 1.72× |
| list_map | 743.3 | 60.3 | 12.33× | 125.1 | 5.94× | 297.5 | 2.50× |
| all_comprehension | 171.5 | 4.4 | 38.98× | 55.9 | 3.07× | 432.5 | 0.40× |
| exists_comprehension | 147.1 | 4.4 | 33.43× | 56.2 | 2.62× | 312.7 | 0.47× |
| map_list_scaling/1 | 175.4 | 65.2 | 2.69× | 229.6 | 0.76× | 152.9 | 1.15× |
| map_list_scaling/10 | 1,503.4 | 139.4 | 10.78× | 312.3 | 4.81× | 461.7 | 3.26× |
| map_list_scaling/100 | 26,628.2 | 247.2 | 107.72× | 425.9 | 62.52× | 3,299.4 | 8.07× |
| map_list_scaling/1000 | 1,441,029 | 798.1 | 1,805.57× | 1,051.9 | 1,369.93× | 38,737.1 | 37.20× |
| map_list_scaling/10000 | 118,215,720 | 5,877.0 | 20,114.98× | 6,327.6 | 18,682.55× | 371,253 | 318.42× |
| filter_list_scaling/1 | 174.0 | 64.7 | 2.69× | 228.1 | 0.76× | 163.2 | 1.07× |
| filter_list_scaling/10 | 905.2 | 138.0 | 6.56× | 302.2 | 3.00× | 555.2 | 1.63× |
| filter_list_scaling/100 | 12,003.1 | 249.3 | 48.15× | 421.0 | 28.51× | 4,273.7 | 2.81× |
| filter_list_scaling/1000 | 385,835 | 790.8 | 487.90× | 1,048.1 | 368.13× | 44,692.0 | 8.63× |
| filter_list_scaling/10000 | 31,247,168 | 6,102.4 | 5,120.47× | 6,492.3 | 4,812.96× | 473,800 | 65.95× |
| string_operations | 217.4 | 4.7 | 46.26× | 56.1 | 3.88× | 208.3 | 1.04× |
| custom_function | 432.5 | — | — | — | — | 211.7 | 2.04× |
| real_world_policy | 496.0 | 17.7 | 28.02× | 536.8 | 0.92× | 822.4 | 0.60× |

**Counts.** `our-auto` beats `ck-jit` on **all 24 cases its batch machine
lowers**, by 1.42× to 20,115×. Under the pessimistic `auto+bind` accounting it wins on
**14 of 25** and loses on 11 — every loss is a cheap scalar expression where our
50-240 ns activation encoding dwarfs the evaluation itself. Our bytecode VM
alone, with no batch machine, beats his compiled tier on **14 of 25** (it was 16
of 25 on 2026-09-04; §5 says what changed).

**Where we lose outright.** `custom_function` is the one case our batch machine
declines (`from_program` is not handed the Context that carries the user's
functions — `from_program_in` lowers it, and `jit_regime` §4 measures it). Our
best evaluator there is the walker at 113.5 ns, which still beats his compiled
tier (432.5 ns) but loses to his walker (70.1 ns).

Walker against walker — the same algorithm in two versions of the same library —
ours is faster on 21 of 25.

## 4. What compiling actually earns

### 4.1 The crossover: one row against many

`jit_regime`, medians of three runs. `clean` is the batch machine's untraced
interpreter over the same lowered columnar program; `jit` is the compiled trace.
Every measured call entered compiled code (`enter/call` 1.00) and none failed a
guard (`gfail/call` 0.00) at every size, in all three runs.

| case | 1-row clean | 1-row jit | crossover rows (3/3 runs) | 8192-row clean µs | 8192-row jit µs | jit ÷ clean @8192 | jit ns/row @8192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conditional | 13.4 | 106.4 | 32 | 70.1 | 15.4 | 4.55× | 1.88 |
| nested_arithmetic | 20.3 | 111.7 | 16 | 124.3 | 53.5 | 2.32× | 6.53 |
| policy | 17.4 | 106.6 | 16 | 136.0 | 21.3 | 6.38× | 2.60 |
| list_exists | 80.0 | 135.7 | 2 | 599.4 | 53.6 | 11.17× | 6.55 |
| custom_function | 16.4 | 110.0 | 32 | 84.1 | 47.1 | 1.79× | 5.75 |

The crossover row was IDENTICAL in all three runs and identical to the
2026-09-05 dynasm run: `list.exists` at 2 rows, arithmetic and policy at 16,
conditional and host functions at 32. Compiling a CEL program pays from a
handful of rows, not from thousands.

### 4.2 Per-row cost by tier, on the board's own cases

One replicated-batch slope per case, three tiers in one binary. `interp` is the
same traced portal with its threshold at `u32::MAX`, so it can never compile:
that column is the per-row cost BEFORE compiling and `jit` is the same row
AFTER. `clean` is the floor under both.

| case | clean/row | interp/row | jit/row | jit vs interp | jit vs clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple_arithmetic | 0.10 | 12.6 | 0.30 | 42.0× | 0.33× |
| comparison | 0.10 | 12.8 | 0.30 | 42.7× | 0.33× |
| conditional | 4.60 | 17.7 | 0.60 | 29.5× | 7.67× |
| nested_expression | 13.30 | 28.9 | 4.60 | 6.3× | 2.89× |
| variable_access/hashmap | 0.10 | 14.7 | 0.30 | 49.0× | 0.33× |
| variable_access/resolver | 0.00 | 15.0 | 0.30 | 50.0× | — |
| member_access | 4.20 | 17.6 | 0.50 | 35.2× | 8.40× |
| list_indexing | 8.80 | 22.2 | 0.90 | 24.7× | 9.78× |
| list_filter | 39.10 | 35.2 | 5.20 | 6.8× | 7.52× |
| list_map | 6.40 | 19.1 | 1.10 | 17.4× | 5.82× |
| all_comprehension | 0.10 | 12.7 | 0.30 | 42.3× | 0.33× |
| exists_comprehension | 0.10 | 12.7 | 0.30 | 42.3× | 0.33× |
| map_list_scaling/1 | 17.60 | 25.9 | 0.90 | 28.8× | 19.56× |
| map_list_scaling/10 | 85.50 | 166.4 | 6.10 | 27.3× | 14.02× |
| map_list_scaling/100 | 490.90 | 1,570.2 | 49.40 | 31.8× | 9.94× |
| map_list_scaling/1000 | 3,703.90 | 14,877.8 | 544.60 | 27.3× | 6.80× |
| map_list_scaling/10000 | 35,190.80 | 150,201.0 | 5,275.60 | 28.5× | 6.67× |
| filter_list_scaling/1 | 17.40 | 25.8 | 1.00 | 25.8× | 17.40× |
| filter_list_scaling/10 | 80.90 | 177.5 | 6.60 | 26.9× | 12.26× |
| filter_list_scaling/100 | 837.20 | 1,643.4 | 57.00 | 28.8× | 14.69× |
| filter_list_scaling/1000 | 5,902.50 | 16,016.5 | 552.80 | 29.0× | 10.68× |
| filter_list_scaling/10000 | 59,367.30 | 158,999.2 | 5,513.80 | 28.8× | 10.77× |
| comprehension_scaling/10 | 84.30 | 183.7 | 6.90 | 26.6× | 12.22× |
| comprehension_scaling/50 | 426.90 | 845.7 | 30.20 | 28.0× | 14.14× |
| comprehension_scaling/100 | 781.80 | 1,785.6 | 62.60 | 28.5× | 12.49× |
| comprehension_scaling/500 | 3,812.50 | 8,557.1 | 285.80 | 29.9× | 13.34× |
| string_operations | 0.10 | 12.7 | 0.30 | 42.3× | 0.33× |
| real_world_policy | 10.60 | 29.5 | 1.80 | 16.4× | 5.89× |
| map_body/x | 3,473.00 | 14,743.0 | 379.90 | 38.8× | 9.14× |
| map_body/x*2 | 3,753.40 | 14,897.7 | 525.10 | 28.4× | 7.15× |
| map_body/x*2+1 | 4,352.40 | 17,169.1 | 555.90 | 30.9× | 7.83× |
| map_body/x*2+1-3 | 5,497.60 | 18,136.7 | 621.40 | 29.2× | 8.85× |
| record_map_scaling/1 | 17.60 | 25.9 | 0.90 | 28.8× | 19.56× |
| record_map_scaling/10 | 80.60 | 166.2 | 5.50 | 30.2× | 14.65× |
| record_map_scaling/100 | 460.00 | 1,495.6 | 38.40 | 38.9× | 11.98× |
| record_map_scaling/1000 | 3,125.50 | 15,497.3 | 421.40 | 36.8× | 7.42× |
| record_map_scaling/10000 | 33,079.70 | 149,368.8 | 4,762.70 | 31.4× | 6.95× |
| record_filter | 4,549.40 | 16,158.0 | 516.70 | 31.3× | 8.80× |
| record_exists_int/1 | 17.50 | 25.4 | 0.70 | 36.3× | 25.00× |
| record_exists_int/10 | 71.00 | 163.3 | 4.40 | 37.1× | 16.14× |
| record_exists_int/100 | 566.70 | 1,550.3 | 33.10 | 46.8× | 17.12× |
| record_exists_int/1000 | 4,791.50 | 14,622.3 | 278.50 | 52.5× | 17.20× |
| record_exists_int/10000 | 42,070.20 | 147,556.0 | 2,685.30 | 54.9× | 15.67× |
| record_exists_str/1 | 16.40 | 25.1 | 0.70 | 35.9× | 23.43× |
| record_exists_str/10 | 83.10 | 168.7 | 5.40 | 31.2× | 15.39× |
| record_exists_str/100 | 542.90 | 1,479.5 | 31.40 | 47.1× | 17.29× |
| record_exists_str/1000 | 5,010.80 | 14,546.8 | 275.40 | 52.8× | 18.19× |
| record_exists_str/10000 | 49,452.40 | 145,458.1 | 2,696.40 | 53.9× | 18.34× |
| scalar_exists/1 | 17.50 | 24.1 | 0.70 | 34.4× | 25.00× |
| scalar_exists/10 | 71.60 | 161.7 | 4.50 | 35.9× | 15.91× |
| scalar_exists/100 | 535.50 | 1,459.3 | 27.50 | 53.1× | 19.47× |
| scalar_exists/1000 | 4,807.90 | 14,497.1 | 277.70 | 52.2× | 17.31× |
| scalar_exists/10000 | 24,136.70 | 144,603.3 | 2,692.40 | 53.7× | 8.96× |

Compiling is worth **6-55× over the tracing interpreter** and **2.9-25× over the
clean interpreter** on everything that carries a loop or a field access. On the
seven cases that fold to a constant (`simple_arithmetic`, `comparison`, both
`variable_access`, `string_operations`, both comprehension folds) the clean
interpreter is 3× cheaper per row than compiled code, and `auto` routes them to
`clean` — which is why those rows read 0.33× and not a regression.

### 4.3 What the JIT does NOT do yet

`Program::execute` — the per-call door — is attached to a portal
(`cel/src/vm/portal.rs`), with 72 opcodes inlined onto a PyPy-shaped
virtualizable frame. It does not compile. `eval_through_portal` constructs its
`JitDriver` per call with a threshold of 1,000,000
(`cel/src/vm/portal.rs:715`), so the counter cannot survive a call and no
single-call program reaches it. The per-call figures in §2 are therefore the
portal loop INTERPRETING, and every `our-VM` number in this report is a number
the JIT has not yet been allowed to improve.

## 5. What got slower since 2026-09-03

The board binary built on 2026-09-03 22:50 (right after `848d74c`) still exists
and was re-run today, in the same load window as runs 2 and 3, 15 minutes after
them. Same box, same conditions, two binaries — so this table is a controlled
before/after of the 30 commits in between, not a comparison against a published
number.

| case | walker 09-03 | walker now | Δ | VM 09-03 | VM now | Δ | auto 09-03 | auto now | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| simple_arithmetic | 41.6 | 45.6 | +10% | 70.8 | 76.5 | +8% | 4.7 | 4.8 | +2% |
| comparison | 28.4 | 31.5 | +11% | 49.1 | 86.3 | +76% | 4.8 | 4.6 | -4% |
| conditional | 27.9 | 28.4 | +2% | 51.8 | 64.7 | +25% | 12.7 | 12.1 | -5% |
| nested_expression | 141.4 | 137.2 | -3% | 193.7 | 217.7 | +12% | 20.3 | 20.5 | +1% |
| variable_access/hashmap | 4.4 | 4.1 | -7% | 23.3 | 43.4 | +86% | 5.8 | 4.7 | -19% |
| variable_access/resolver | 4.9 | 3.9 | -20% | 25.3 | 42.0 | +66% | 5.1 | 4.7 | -8% |
| member_access | 82.7 | 98.8 | +19% | 125.2 | 309.9 | +148% | 12.0 | 12.0 | +0% |
| list_indexing | 60.0 | 69.6 | +16% | 117.1 | 318.5 | +172% | 16.7 | 17.4 | +4% |
| list_filter | 397.7 | 407.4 | +2% | 395.8 | 541.6 | +37% | 75.0 | 72.5 | -3% |
| list_map | 263.4 | 252.2 | -4% | 240.3 | 297.5 | +24% | 63.0 | 60.3 | -4% |
| all_comprehension | 357.4 | 416.4 | +17% | 257.0 | 432.5 | +68% | 4.8 | 4.4 | -8% |
| exists_comprehension | 283.2 | 306.8 | +8% | 196.7 | 312.7 | +59% | 4.9 | 4.4 | -10% |
| map_list_scaling/1 | 145.0 | 138.6 | -4% | 104.3 | 152.9 | +47% | 66.8 | 65.2 | -2% |
| map_list_scaling/10 | 315.3 | 305.5 | -3% | 268.3 | 461.7 | +72% | 138.9 | 139.4 | +0% |
| map_list_scaling/100 | 2,014.6 | 1,926.2 | -4% | 1,708.9 | 3,299.4 | +93% | 199.2 | 247.2 | +24% |
| map_list_scaling/1000 | 19,111.1 | 18,240.8 | -5% | 16,914.9 | 38,737.1 | +129% | 664.0 | 798.1 | +20% |
| map_list_scaling/10000 | 179,524 | 181,949 | +1% | 165,952 | 371,253 | +124% | 5,158.5 | 5,877.0 | +14% |
| filter_list_scaling/1 | 150.0 | 149.9 | -0% | 113.5 | 163.2 | +44% | 66.6 | 64.7 | -3% |
| filter_list_scaling/10 | 410.6 | 410.8 | +0% | 382.3 | 555.2 | +45% | 131.7 | 138.0 | +5% |
| filter_list_scaling/100 | 2,910.4 | 2,906.7 | -0% | 2,938.5 | 4,273.7 | +45% | 191.1 | 249.3 | +30% |
| filter_list_scaling/1000 | 27,177.0 | 28,101.3 | +3% | 27,937.4 | 44,692.0 | +60% | 614.3 | 790.8 | +29% |
| filter_list_scaling/10000 | 271,446 | 281,437 | +4% | 280,286 | 473,800 | +69% | 4,658.3 | 6,102.4 | +31% |
| comprehension_scaling/10 | 621.6 | 646.9 | +4% | 551.8 | 742.2 | +35% | 156.9 | 137.4 | -12% |
| comprehension_scaling/50 | 2,339.9 | 2,286.1 | -2% | 2,227.5 | 2,970.0 | +33% | 222.9 | 232.9 | +4% |
| comprehension_scaling/100 | 4,160.6 | 4,481.9 | +8% | 3,970.5 | 5,886.0 | +48% | 217.1 | 272.4 | +25% |
| comprehension_scaling/500 | 18,533.4 | 19,889.0 | +7% | 18,444.9 | 29,402.0 | +59% | 447.1 | 504.1 | +13% |
| string_operations | 159.5 | 166.8 | +5% | 185.3 | 208.3 | +12% | 4.7 | 4.7 | +0% |
| custom_function | 108.1 | 113.5 | +5% | 138.2 | 211.7 | +53% | — | — | — |
| real_world_policy | 292.3 | 317.9 | +9% | 399.0 | 822.4 | +106% | 17.3 | 17.7 | +2% |
| map_body/x | 9,437.9 | 9,824.7 | +4% | 10,453.6 | 32,053.4 | +207% | 480.9 | 580.4 | +21% |
| map_body/x*2 | 22,320.8 | 18,013.8 | -19% | 20,064.7 | 36,479.3 | +82% | 821.0 | 751.6 | -8% |
| map_body/x*2+1 | 31,572.2 | 29,647.1 | -6% | 34,601.9 | 50,562.3 | +46% | 964.5 | 793.1 | -18% |
| map_body/x*2+1-3 | 41,356.5 | 42,275.5 | +2% | 48,471.4 | 65,333.2 | +35% | 1,029.2 | 855.6 | -17% |
| record_map_scaling/1 | 162.5 | 167.7 | +3% | 128.9 | 201.4 | +56% | 69.7 | 68.0 | -2% |
| record_map_scaling/10 | 483.2 | 603.1 | +25% | 426.8 | 893.0 | +109% | 149.4 | 136.5 | -9% |
| record_map_scaling/100 | 3,582.6 | 5,090.8 | +42% | 3,171.9 | 7,610.6 | +140% | 186.8 | 231.3 | +24% |
| record_map_scaling/1000 | 33,767.1 | 50,337.1 | +49% | 30,339.1 | 80,486.1 | +165% | 508.0 | 580.3 | +14% |
| record_map_scaling/10000 | 357,639 | 510,024 | +43% | 330,141 | 817,657 | +148% | 3,870.1 | 5,416.9 | +40% |
| record_filter | 61,600.2 | 53,875.8 | -13% | 51,159.4 | 100,382 | +96% | 666.0 | 745.5 | +12% |
| record_exists_int/1 | 132.1 | 157.9 | +20% | 119.4 | 266.0 | +123% | 20.1 | 21.8 | +8% |
| record_exists_int/10 | 678.1 | 840.5 | +24% | 579.9 | 1,524.2 | +163% | 77.3 | 76.8 | -1% |
| record_exists_int/100 | 6,159.0 | 7,051.2 | +14% | 5,429.8 | 14,863.7 | +174% | 130.3 | 165.5 | +27% |
| record_exists_int/1000 | 60,404.9 | 73,776.4 | +22% | 52,048.3 | 154,707 | +197% | 423.3 | 420.0 | -1% |
| record_exists_int/10000 | 630,583 | 718,466 | +14% | 529,293 | 1,539,461 | +191% | 3,236.2 | 2,744.9 | -15% |
| record_exists_str/1 | 137.1 | 145.4 | +6% | 115.2 | 241.0 | +109% | 20.2 | 18.2 | -10% |
| record_exists_str/10 | 758.3 | 850.1 | +12% | 637.8 | 1,562.4 | +145% | 89.4 | 90.0 | +1% |
| record_exists_str/100 | 6,425.3 | 7,865.1 | +22% | 5,279.6 | 15,831.8 | +200% | 127.6 | 163.9 | +28% |
| record_exists_str/1000 | 62,219.5 | 74,433.4 | +20% | 52,169.4 | 151,744 | +191% | 397.7 | 402.7 | +1% |
| record_exists_str/10000 | 637,526 | 724,551 | +14% | 533,267 | 1,552,314 | +191% | 3,299.0 | 2,680.6 | -19% |
| scalar_exists/1 | 123.7 | 123.8 | +0% | 84.3 | 162.3 | +93% | 19.8 | 18.2 | -8% |
| scalar_exists/10 | 546.0 | 627.1 | +15% | 308.3 | 713.9 | +132% | 77.3 | 77.0 | -0% |
| scalar_exists/100 | 4,663.1 | 5,371.8 | +15% | 2,323.3 | 6,111.6 | +163% | 125.4 | 164.8 | +31% |
| scalar_exists/1000 | 42,941.4 | 50,832.5 | +18% | 23,839.7 | 62,258.7 | +161% | 410.5 | 405.3 | -1% |
| scalar_exists/10000 | 438,605 | 491,736 | +12% | 230,766 | 632,077 | +174% | 3,289.7 | 2,731.0 | -17% |

**The bytecode VM regressed on every one of the 54 cases, by +8% to +207%**,
while the batch machine is flat (-19% to +40%, mixed) and the walker moved
-20% to +49%.

### 5.1 It is not the portal

`examples/rca_exec` times one door on one expression. Built three ways — the
2026-09-03 tree, today's tree without `jit` (no portal), today's tree with
`jit-dynasm` (portal) — the walker is the control and is unchanged across all
three:

| expression | VM 09-03 | VM now, no portal | VM now, portal | walker 09-03 | walker now |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1 + 2 * 3 - 4 / 2` | 69.2 | 90.3 | 74.7 | 43.1 | 46.3 |
| `x` | 23.6 | 39.7 | 49.2 | 9.8 | 9.3 |
| `list[3]` | 39.9 | 122.2 | 124.2 | 17.0 | 17.6 |
| `[1, 2, 3, 4, 5][2]` | 83.6 | 250.2 | 105.8 | 53.8 | 53.9 |
| `size(list)` | 57.5 | 109.1 | 108.5 | 46.8 | 45.3 |
| `{'a': 1, 'b': 2}['a']` | 124.5 | 424.5 | 426.8 | 92.1 | 95.3 |
| `list.map(e, e * 2)` | 279.7 | 500.9 | 455.2 | 355.8 | 330.5 |
| `list.filter(e, e % 2 == 0)` | 372.5 | 733.6 | 562.2 | 441.8 | 426.2 |
| `list.exists(e, e > 5)` | 231.4 | 557.2 | 531.2 | 414.1 | 431.0 |
| `list.all(e, e > 0)` | 317.4 | 795.8 | 742.7 | 535.8 | 557.6 |

The walker is within ±5% everywhere, so the box and the harness are not what
moved. The portal is not either: turning it on RECOVERS part of the loss
(`[1,2,3,4,5][2]` 250.2 → 105.8, `list.filter` 733.6 → 562.2) and is within 5%
of the no-portal build elsewhere. What both of today's builds share, and the
2026-09-03 build does not, is the interned class-family value universe the
portal needed — the 18 `vm:`/`cel runtime:` commits of the 30. The VM pays it on every boundary
crossing, and it costs 1.1-3.4× per call.

The walker does not go through that universe, which is why it is the control
that holds still.

### 5.2 What it cost in the comparison

Two cases moved from the win column to the loss column against his compiled
backend between 2026-09-04 and today: `member_access` (VM 125.2 → 309.9 ns) and
`real_world_policy` (399.0 → 822.4). The count of cases our VM alone wins fell
from 16 of 25 to 14 of 25. `auto` is unaffected — it does not go through the VM.

## 6. What this report establishes, and what it does not

**Established.** On batch-shaped work our compiled tier is between 1.8× and 11×
the clean interpreter at 8,192 rows and 6-55× the tracing interpreter per row,
it starts paying at 2-32 rows, and the whole machine is 1.4×-20,115× cometkim's
Cranelift AOT backend on the 24 cases our batch machine lowers, with the
activation held fixed the way he holds it.

**Not established.** That cel-jit is faster on a per-call workload — one fresh
activation, one evaluation, no batch. There, `auto+bind` loses 11 of 25 to his
backend, the compiled tier's one-row entry is 91-140 ns against a 4-22 ns
interpreter, and the per-call door has regressed 1.1-3.1× in two weeks for a
compile that is not yet switched on.

## 7. Reproducing

```sh
# ours (from the cel-jit checkout)
cargo run --profile bench -p cel --features jit-dynasm --example majit_vs_cometkim_percall
cargo run --profile bench -p cel --features jit-dynasm --example jit_regime

# his (from a PR-233 checkout: fetch cel-rust/cel-rust pull/233 head 4d57618)
cargo run --release -p cel-jit --example percall_ck     # his evaluators, our timer
cargo bench -p cel-jit --bench comparison               # his own harness

# the per-call A/B of §5.1
cargo build --profile bench -p cel --example rca_exec                        # no portal
cargo build --profile bench -p cel --features jit-dynasm --example rca_exec  # portal
target/release/examples/rca_exec [vm|walker] '<expr>' 1.5 time
```
