#!/usr/bin/env bash
# Repeatable cel majit benchmark runner.
#
# The cel majit tier reads context columns through a compiled trace and beats
# the stock per-row tree-walker. This runs the relevant example in release with
# the `majit-jit` feature. cd's to its own dir first so it works regardless of
# where it's invoked from (cel-rust is a nested workspace; running `-p cel`
# from the parent pyre-wasmi workspace fails).
#
# Usage:
#   ./bench.sh                    # columnar batch table (default)
#   ./bench.sh majit_vs_cometkim  # per-expression throughput vs cometkim's AOT
#   ./bench.sh majit_ab           # A/B jit-on vs jit-off on the loop kernels
set -euo pipefail
cd "$(dirname "$0")"
ex="${1:-majit_columnar_batch}"
exec cargo run --release --quiet --package cel --features majit-jit --example "$ex" "${@:2}"
