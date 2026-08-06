#!/usr/bin/env python3
"""cel driver for the Charon ULLBC extraction engine.

Declares cel's crate table and delegates to the neutral engine that lives in
the pyre repository (`<pyre-root>/scripts/llbc_extract.py`, which carries zero
crate names precisely so an external consumer can bring its own driver).
Artefacts land under `<cel-root>/build/llbc`.

`cel.ullbc` is the input `majit-translate`'s front end B reads
(`analyze_multiple_pipeline_from_llbc_with_modules`,
`majit-translate/src/lib.rs:512`). Nothing in cel's cargo build consumes it
today — cel does not depend on `majit-translate` — so this driver is run by
hand, not by a `build.rs`.

    scripts/extract-llbc.py            # -> build/llbc/cel.ullbc
    scripts/extract-llbc.py --force    # ignore the fingerprint, re-extract
    CARGO_FEATURES=cranelift scripts/extract-llbc.py

Locating the engine and Charon:

  * `PYRE_ROOT` (env) overrides where the pyre checkout is; the default assumes
    the sibling layout this repo is developed in (`<super>/cel-jit`,
    `<super>/scripts`, `<super>/majit`, …).
  * Charon itself is resolved by the engine from `PYRE_SHARED_BUILD` /
    `CHARON_DEST`, defaulting to `<pyre-root>/../.pyre-build/charon/<platform>`;
    install it with `<pyre-root>/scripts/install-charon.py`.

⚠ FINGERPRINT SCOPE. The stamp covers cel's own tracked sources, this driver,
the resolved cargo/charon flags, and the installed Charon version (the engine
reads `.installed-version`). It does NOT cover the engine module itself — that
file lives in the pyre repository, and `git ls-files` cannot reach outside this
one. After editing `<pyre-root>/scripts/llbc_extract.py`, re-extract with
`--force` (or `LLBC_FORCE_REEXTRACT=1`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYRE_ROOT = Path(os.environ.get("PYRE_ROOT", ROOT.parent))

_engine_dir = PYRE_ROOT / "scripts"
if not (_engine_dir / "llbc_extract.py").is_file():
    raise SystemExit(
        f"extract-llbc.py: no extraction engine at {_engine_dir / 'llbc_extract.py'}\n"
        "  set PYRE_ROOT to the pyre checkout that provides scripts/llbc_extract.py"
    )
sys.path.insert(0, str(_engine_dir))

from llbc_extract import CrateSpec, run_cli  # noqa: E402


SPECS: dict[str, CrateSpec] = {
    "cel": CrateSpec(
        name="cel",
        crate_dir=ROOT / "cel",
        output_name="cel.ullbc",
        # cel's backend selector is `jit-<backend>`, so the engine's feature
        # placeholder (default `dynasm`, override with `CARGO_FEATURES`) is
        # spliced into the name rather than passed bare. A bare `--features
        # jit` is a hard `compile_error!` by design
        # (`majit-metainterp/src/pyjitpl.rs:40-41`), so the backend must be
        # named. The artefact is unaffected by which one: `cel.ullbc` carries
        # only cel's own item bodies, and no cel source is `cfg`'d on the
        # backend — the selector exists so the crate links at all.
        cargo_args=["--features", "jit-{features}"],
        # ⛔ `cel::parser` MUST stay opaque. Whole-crate extraction with the
        # parser transparent aborts the charon driver:
        #
        #     warning: Unexpected error: could not find region '1_0
        #     thread 'rustc' has overflowed its stack
        #     fatal runtime error: stack overflow, aborting
        #     ... (signal: 6, SIGABRT)
        #
        # and Charon writes NO artefact, so a caller that trusts the exit
        # status of a pipeline instead of the artefact sees a false green.
        # `RUST_MIN_STACK=1GiB` does not move it, so it is not plain recursion
        # depth: `cel/src/parser/parser.rs` is the ANTLR-generated CEL parser
        # (75 KB of deeply-nested generated types over `antlr4rust`).
        #
        # Making that one module opaque keeps its declarations and drops its
        # bodies, which is exactly right for this artefact's consumer: the
        # traced graph closure starts at an evaluation portal and never calls
        # the parser — parsing happens once, in `Program::compile`, outside
        # any JIT portal.
        charon_args=["--opaque", "cel::parser"],
        # cel's dependency graph is resolved through a git rev for the majit
        # crates, so a `cargo metadata` walk would need the network on a cold
        # cache and would return no path dependencies anyway. The artefact's
        # declared inputs are exactly cel's own tracked sources plus the
        # lockfile (which pins the majit rev).
        #
        # ⚠ The majit rev IS an input: `run_mainloop_f`'s signature names
        # `majit_metainterp::JitDriver<VmStateF>`, so majit's type layouts land
        # in `cel.ullbc`. `Cargo.lock` covers the DECLARED rev. It does not
        # cover the uncommitted `.cargo/config.toml` `[patch]` that redirects
        # those git deps to the enclosing pyre worktree while cel-jit lives
        # inside it — a local override this driver cannot see, and one that has
        # been observed to come and go under a concurrent session. When the
        # patch is on and off you get two different `cel.ullbc` for one
        # fingerprint. Re-extract with `--force` after toggling it, and read
        # the extraction log: a patched build compiles
        # `majit-* (…/pyre-wasmi/majit/…)`, an unpatched one compiles
        # `majit-* (https://github.com/youknowone/pyre.git?rev=…)`.
        fingerprint_pathspecs=[
            "cel/Cargo.toml",
            "cel/src/",
        ],
        # No cross-target layout sidecar: nothing reads cel layouts for a
        # target other than the extraction host.
        layout_targets=(),
    ),
    # Secondary artefact, NOT the production input: the same crate reduced to
    # the call-graph closures of the three portals the P0.b census measures.
    # `cel.ullbc` is ~100 MB and the front end lowers every function in it;
    # this one is ~10x smaller and turns a census run from tens of minutes
    # into tens of seconds, which is what makes the census usable as an
    # iteration instrument rather than a once-a-day batch job.
    #
    # `--start-from` translates only the named items and what they reach.
    # Charon rejects an impl pattern that is not the first path element, so
    # `Value::resolve_val` is named by its containing module.
    #
    # ⚠ Not interchangeable with `cel.ullbc`: a portal the list does not name
    # will not resolve, and the reduced type universe can change what
    # `register_trait_families` sees. Measure the production artefact before
    # believing a delta.
    "cel-portals": CrateSpec(
        name="cel-portals",
        crate_dir=ROOT / "cel",
        output_name="cel-portals.ullbc",
        cargo_args=["--features", "jit-{features}"],
        charon_args=[
            "--start-from",
            "cel::majit::bytecode::float_bank::run_mainloop_f",
            "--start-from",
            "cel::majit::bytecode::float_bank::clean_interp_seeded_f",
            "--start-from",
            "cel::objects",
        ],
        fingerprint_pathspecs=[
            "cel/Cargo.toml",
            "cel/src/",
        ],
        layout_targets=(),
    ),
}

# Only the whole-crate artefact by default; `cel-portals` is opt-in.
DEFAULT_CRATES = ["cel"]

BASE_PATHSPECS = [
    "Cargo.lock",
    "Cargo.toml",
    "scripts/extract-llbc.py",
]


def main() -> None:
    run_cli(
        SPECS,
        DEFAULT_CRATES,
        root=ROOT,
        out_dir=ROOT / "build" / "llbc",
        base_pathspecs=BASE_PATHSPECS,
        charon_root=PYRE_ROOT,
        layout_targets=(),
    )


if __name__ == "__main__":
    main()
