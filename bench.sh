#!/usr/bin/env bash
# Repeatable cel majit benchmark runner.
#
# The default fair suite reports:
#   1. real CEL single-activation request latency (stock; JIT is N/A until it
#      exposes an equivalent API),
#   2. JIT-only throughput over identical lowered bytecode and columns, and
#   3. cold/break-even totals for a fresh JIT driver.
#
# Columnar examples remain available explicitly, but their stock/JIT ratio is a
# cross-model batch comparison, not a CEL request-latency or JIT-only result.
# This runs the selected example in release with the `jit` feature. cd's
# to its own dir first so it works regardless of where it is invoked from.
#
# `majit_nested_bench` covers the nested comprehension shape, which the default
# suite's flat single-loop engine panel does not reach. It sweeps a ladder of
# batch sizes so tracing/compilation can be separated from the compiled trace's
# own per-row cost, and prints a tree-walker reference as a floor.
#
# Usage:
#   ./bench.sh                         # fair request/engine/cold suite (default)
#   ./bench.sh majit_nested_bench      # nested comprehension, size ladder
#   ./bench.sh majit_nested_bench 640000 9   # <max_rows> <rounds>
#   ./bench.sh majit_columnar_batch    # explicit columnar batch experiment
#   ./bench.sh majit_vs_cometkim       # explicitly cross-regime historical probe
set -euo pipefail
cd "$(dirname "$0")"
ex="${1:-majit_ab}"
exec cargo run --release --quiet --package cel --features jit --example "$ex" "${@:2}"
