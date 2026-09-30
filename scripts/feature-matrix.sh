#!/usr/bin/env bash
#
# Checks that every feature combination `cel/Cargo.toml` advertises actually
# builds.
#
# The crate's `[features]` table is a public interface: each entry is a
# configuration a downstream user may select, and until this script existed
# none of them was exercised alone. `rust.yml` ran `cargo test --features
# regex` and `--features chrono`, but those are *additive on top of the
# defaults*, and the defaults are `regex` + `chrono` — so both invocations
# built the exact same feature set as a bare `cargo test`, and the
# single-feature builds were never compiled at all. That is how an
# unconditional `use chrono::FixedOffset` sat in `ser.rs`'s test module, and
# how `#[cfg(feature = "chrono")]` attributes orphaned by a deleted `pub use`
# came to gate `pub enum Kind` and `use crate::objects::Value`.
#
# The feature list is READ FROM THE MANIFEST, not hardcoded here. A hardcoded
# list is a gate that stops looking the moment someone adds a feature, which is
# the failure mode this script exists to end.
#
# Targets: `--all-targets`, not a bare `cargo check`. The originally reported
# error is inside `#[cfg(test)] mod tests`, which a bare `cargo check` never
# compiles. `--all-targets` covers lib, tests, benches and examples — and
# `dhat-heap` is a benches-only feature, so without it that feature would be
# checked vacuously.
#
# Portability: bash 3.2 (stock macOS) and no `grep`/`rg` dependency — `awk` is
# the only text tool used, so this runs the same on a CI runner and a laptop.
#
# Usage: scripts/feature-matrix.sh [--jit]
#   --jit  also check the JIT backend selectors. Off by default: they pull the
#          `majit` crates and dominate the runtime of the whole matrix.

set -uo pipefail

cd "$(dirname "$0")/.."

MANIFEST=cel/Cargo.toml
WITH_JIT=0
[ "${1:-}" = "--jit" ] && WITH_JIT=1

# Feature names from the `[features]` table: the section runs to the next
# `[section]` header or EOF, and a feature is a name at the start of a line
# followed by `=`.
FEATURES=$(
  awk '/^\[features\]/ { inside = 1; next }
       /^\[/           { inside = 0 }
       inside && /^[A-Za-z0-9_-]+[[:space:]]*=/ { sub(/[[:space:]]*=.*/, ""); print }' "$MANIFEST"
)

if [ -z "$FEATURES" ]; then
  echo "FAIL: no features parsed out of $MANIFEST — the parser broke, not the matrix" >&2
  exit 2
fi

# Manifest and target directory must agree, checked BEFORE any matrix row.
#
# A file under `cel/examples/` with no `[[example]]` block is still discovered
# and built by `--all-targets` — just with no `required-features`, so it is
# compiled in every configuration below. Three of them once broke nine rows with
# `error[E0433]: cannot find majit in cel`, a symptom that cost a full census
# and `git merge-base` archaeology to trace back to a manifest omission. Running
# this first turns that into one line naming the file.
#
# Examples specifically, not tests: a test that needs a feature says so in the
# file (`#![cfg(feature = "jit")]`), but an example is a binary with a `main`
# and `required-features` is the only place to spell it.
#
# Set equality by NAME, in both directions. A count cannot do this: one block
# naming a file that no longer exists plus one undeclared file net to equal, and
# the check passes over two defects at once.
declared_f=$(mktemp)
on_disk_f=$(mktemp)

awk '/^\[\[example\]\]/ { inblock = 1; next }
     inblock && /^name[[:space:]]*=/ {
       match($0, /"[^"]*"/)
       print substr($0, RSTART + 1, RLENGTH - 2)
       inblock = 0
     }' "$MANIFEST" >"$declared_f"

for p in cel/examples/*.rs; do
  [ -e "$p" ] || continue
  b=${p##*/}
  echo "${b%.rs}"
done >"$on_disk_f"

if [ ! -s "$on_disk_f" ]; then
  echo "FAIL: no files matched cel/examples/*.rs — the glob broke, not the manifest" >&2
  rm -f "$declared_f" "$on_disk_f"
  exit 2
fi

# `declared` on the left, `on_disk` on the right, each direction named for the
# defect it is rather than for the set operation.
undeclared=$(awk 'NR == FNR { d[$0]; next } !($0 in d) { printf "%s ", $0 }' "$declared_f" "$on_disk_f")
phantom=$(awk 'NR == FNR { k[$0]; next } !($0 in k) { printf "%s ", $0 }' "$on_disk_f" "$declared_f")
n_disk=$(awk 'END { print NR }' "$on_disk_f")
rm -f "$declared_f" "$on_disk_f"

if [ -n "$undeclared" ] || [ -n "$phantom" ]; then
  echo "FAIL: $MANIFEST and cel/examples/ disagree." >&2
  [ -n "$undeclared" ] && echo "  no [[example]] block, so built in EVERY feature configuration: $undeclared" >&2
  [ -n "$phantom" ] && echo "  [[example]] block naming a file that does not exist: $phantom" >&2
  echo "  Add a block (with required-features if it needs one), or drop the stale block." >&2
  exit 1
fi

echo "manifest/targets: $n_disk examples on disk, all declared; no block names a missing file"

fail=0

# check <label> <expectation> <cargo args...>
#   expectation: `builds`  — must compile
#                `refuses` — must fail, and the failure must name the backend
#                            selection error, not some unrelated breakage
check() {
  label="$1"
  expect="$2"
  shift 2
  log=$(mktemp)
  cargo check -p cel --all-targets "$@" >"$log" 2>&1
  status=$?

  first_error=$(awk '/^error/ { print; exit }' "$log")
  [ -n "$first_error" ] || first_error="exit $status"

  if [ "$expect" = builds ]; then
    if [ "$status" -eq 0 ]; then
      verdict="ok"
    else
      verdict="FAILED: $first_error"
      fail=1
    fi
  else
    # The documented refusal names both backends in one `compile_error!`.
    named_backends=$(awk '/cranelift/ && /dynasm/ { print "yes"; exit }' "$log")
    if [ "$status" -eq 0 ]; then
      verdict="FAILED: expected a refusal, it built"
      fail=1
    elif [ "$named_backends" = yes ]; then
      verdict="ok (refused, as documented)"
    else
      verdict="FAILED: refused for the wrong reason: $first_error"
      fail=1
    fi
  fi

  printf '%-46s %s\n' "$label" "$verdict"
  rm -f "$log"
}

echo "cel feature matrix (features parsed from $MANIFEST):"
echo "  $(echo "$FEATURES" | tr '\n' ' ')"
echo

check "--no-default-features" builds --no-default-features
check "(default)" builds

for f in $FEATURES; do
  case "$f" in
    default) continue ;;
    # Bare `jit` names no backend. `majit-metainterp` turns that into a
    # `compile_error!` on purpose, so the JIT tier can never fail open into a
    # silently-not-JITting build. Pin the refusal.
    jit)
      check "--no-default-features --features jit" refuses \
            --no-default-features --features jit
      continue
      ;;
    jit-dynasm|jit-cranelift)
      if [ "$WITH_JIT" -eq 0 ]; then
        printf '%-46s %s\n' "--no-default-features --features $f" "skipped (pass --jit)"
        continue
      fi
      ;;
  esac
  check "--no-default-features --features $f" builds --no-default-features --features "$f"
done

check "--all-features" builds --all-features

echo
if [ "$fail" -ne 0 ]; then
  echo "feature matrix: RED"
  exit 1
fi
echo "feature matrix: green"
