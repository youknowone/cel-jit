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
#   CEL_PROFILE=bench ./bench.sh majit_vs_cometkim_percall   # cometkim's own regime
#
# `majit_vs_cometkim_percall` is the one example whose numbers are meant to be
# read beside another project's: it measures cometkim's unit — one expression,
# one fixed activation, one call — so it must be built the way his criterion
# bench is, under `[profile.bench]` (lto, one codegen unit). CEL_PROFILE selects
# that; it defaults to `release`, which is what every other example here reports.
#
# CEL_BACKEND selects the JIT backend: cranelift (default) or dynasm. `jit`
# alone names none — cargo features are additive, so a `jit` that named one
# could never un-name it and every number here would silently describe that one
# backend. A run states which backend it measured, because the two are not
# interchangeable: they have already diverged on cel three times.
set -euo pipefail
cd "$(dirname "$0")"
ex="${1:-majit_ab}"
be="${CEL_BACKEND:-cranelift}"
prof="${CEL_PROFILE:-release}"
case "$be" in
  cranelift|dynasm) ;;
  *) echo "CEL_BACKEND must be cranelift or dynasm, got '$be'" >&2; exit 2 ;;
esac
case "$prof" in
  release|bench) ;;
  *) echo "CEL_PROFILE must be release or bench, got '$prof'" >&2; exit 2 ;;
esac
echo "# backend: $be   profile: $prof   example: $ex" >&2
exec cargo run --profile "$prof" --quiet --package cel --features "jit-$be" --example "$ex" "${@:2}"
