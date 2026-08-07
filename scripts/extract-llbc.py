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

⚠ FINGERPRINT SCOPE. The stamp has two source fields, because this artefact has
two kinds of input and only one of them is reachable by a pathspec:

  * `source=` — cel's own tracked sources, this driver, the resolved
    cargo/charon flags, and the installed Charon version (the engine reads
    `.installed-version`). Resolved through `git ls-files`, run in THIS repo.
  * `external=` — everything `git ls-files` structurally cannot name: the
    engine module in the pyre repository, the patched `majit` workspace, and
    the two gitignored files that decide what gets built (`Cargo.lock`,
    `.cargo/config.toml`). Declared by `EXTERNAL_INPUTS` below and hashed by
    content.

The split is not cosmetic: a `--check` failure names the field, so "my own
sources moved" and "a dependency in another checkout moved" are different
messages. See `EXTERNAL_INPUTS` for what is in the second set and why.
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
        # Declared rather than walked: cel's manifest resolves the majit crates
        # through a git rev, so a `cargo metadata` walk needs the network on a
        # cold cache. This list is the half of the input set that lives in this
        # repo; the majit half is declared out-of-band in `EXTERNAL_INPUTS`,
        # because `git ls-files` run here cannot name a path in another
        # repository at all.
        #
        # ⛔ Two claims that used to stand here were false, and both said the
        # coverage was wider than it was:
        #
        #   * "plus the lockfile (which pins the majit rev)" — `Cargo.lock` is
        #     gitignored here, so the `BASE_PATHSPECS` entry naming it matched
        #     nothing (#132). It was the stated justification for skipping the
        #     more expensive walk, so the cheap option rested on coverage that
        #     did not exist. The engine's `refuse_inert_pathspecs` now refuses
        #     that shape for every driver, not just this one.
        #   * "a walk … would return no path dependencies anyway" — MEASURED
        #     false while the `[patch]` below is on: the walk returns a 10-crate
        #     closure, 9 of them out-of-root under `<pyre-root>/majit`. The
        #     claim was only ever true of an unpatched build.
        #
        # ⚠ The majit sources ARE an input: `run_mainloop_f`'s signature names
        # `majit_metainterp::JitDriver<VmStateF>`, so majit's type layouts land
        # in `cel.ullbc` — and `majit-macros` is a proc macro, so it changes
        # cel's own item bodies, not merely layouts. The `.cargo/config.toml`
        # `[patch]` that redirects those git deps to the enclosing pyre worktree
        # is itself gitignored, and has been observed to come and go under a
        # concurrent session; with the patch on and off you get two different
        # `cel.ullbc`. All three of those — the majit tree, the patch file, the
        # lockfile — are in `EXTERNAL_INPUTS` now, so toggling the patch moves
        # `external=` and `--check` reports stale on its own. To confirm which
        # build you got, read the extraction log: a patched build compiles
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

# ⛔ `Cargo.lock` is NOT here, and its absence is deliberate rather than an
# oversight. It is gitignored in this repo (`.gitignore:2`), and the engine
# builds its input set as `ls_files() | ls_files("--others",
# "--exclude-standard")` — tracked ∪ untracked-not-ignored — so an ignored file
# is in neither. Naming it produced the appearance of coverage and nothing else.
# Listing it again would restore the appearance, not the coverage. It is covered
# now, through `EXTERNAL_INPUTS` below, which hashes by content instead of
# asking git — the only channel that can reach a file git refuses to list.
BASE_PATHSPECS = [
    "Cargo.toml",
    "scripts/extract-llbc.py",
]


# Inputs hashed into `external=` instead of `source=`, because `git ls-files`
# run in THIS repo cannot name any of them: three live in the pyre checkout, and
# two are gitignored here. Absolute paths; the engine puts a root-relative label
# in the digest, so relocating the whole `<super>/{pyre,cel-jit}` cohabitation as
# a unit does not move the fingerprint, while rearranging the repos relative to
# each other does.
EXTERNAL_INPUTS = [
    # The extraction engine. Its code decides what Charon is asked to translate
    # and how the stamp is computed, so editing it changes the artefact — and
    # it is one directory outside this repo, which is the whole reason the
    # `external=` channel exists (#119).
    PYRE_ROOT / "scripts" / "llbc_extract.py",
    # The patched majit workspace, declared WHOLE and not crate by crate. That
    # is deliberate over-coverage: an edit under `majit/` that cel's build never
    # compiles costs one re-extraction, whereas any of the three ways a
    # hand-maintained crate list goes silently wrong costs a wrong answer.
    # Measured, all three, on this tree:
    #
    #   * MEMBERSHIP IS FEATURE-DEPENDENT. `cargo metadata --features
    #     jit-cranelift` gives a 10-crate closure whose 9 out-of-root members
    #     include `majit-backend-cranelift` and NOT `majit-backend-dynasm`.
    #     `jit-dynasm` — this driver's default `CARGO_FEATURES` — swaps them.
    #     One static list is wrong for one of the two backends.
    #   * A NEW CRATE IN THE CLOSURE IS INVISIBLE to a list written today: the
    #     `[patch]` names three crates and the other six arrive as their path
    #     deps, so majit can grow the set without touching anything here.
    #   * A PER-CRATE `src/` LIST WOULD MISS `majit-macros` ENTIRELY. Deriving
    #     one the way the engine's walk does drops it, because the walk filters
    #     targets by `{"lib","bin","custom-build"} & kinds` and a proc macro's
    #     kind is `["proc-macro"]`. That is not hypothetical — pyre's own
    #     `--list-inputs` is 699 entries containing exactly one `majit-macros`
    #     path, its `Cargo.toml`, and no source file. It is also the worst crate
    #     to miss: a proc macro's expansion IS the extracted crate's bodies.
    #
    # ⚠ The path is hardcoded while the truth lives in `.cargo/config.toml`'s
    # `[patch]`. Re-pointing the patch at a different tree is DETECTED (that
    # file is declared below, so the digest moves), but not FOLLOWED — the new
    # tree would be undeclared until this line is updated. With the patch off
    # entirely, majit comes from the git rev in `Cargo.lock`, also declared, and
    # this entry then over-covers a tree that is not an input.
    PYRE_ROOT / "majit",
]

# Declared separately because absence is a legitimate state that this driver
# must keep working in, and is itself a fingerprint-relevant signal: the engine
# hashes a missing entry as `<absent>`, so patch-on and patch-off produce
# different digests. `refuse_absent_external_inputs` therefore does not apply.
OPTIONAL_EXTERNAL_INPUTS = [
    # Gitignored (`.gitignore:2`), and the reason #132 existed: it pins the
    # majit git rev plus every crates.io version cel resolves.
    ROOT / "Cargo.lock",
    # The `[patch]` override, gitignored (`.gitignore:8`). Declared as the
    # DIRECTORY so a second file appearing beside `config.toml` (cargo also
    # reads an extensionless `config`) is covered without editing this list.
    ROOT / ".cargo",
]


def refuse_absent_external_inputs(inputs: list[Path]) -> None:
    """Refuse a declared external input that is not on disk.

    ⛔ THIS FUNCTION USES `exists()` AS ITS VERDICT, and the engine's
    `refuse_inert_pathspecs` pointedly does not — it refuses on git's answer and
    lets `exists()` pick only the wording. That is not an inconsistency to
    reconcile, so do not make the two agree. One question, one oracle: a
    pathspec becomes an input only by way of `git ls-files`, so git decides
    there; an external input is hashed by reading its bytes, so being on disk IS
    the question here. It is the same reason the engine probe at the top of this
    file is an `is_file()`. Conflating the two predicates is what #132 was, and
    the rule that came out of it is not "prefer git" — it is that the defect was
    using `exists()` to answer a question about git.

    The sibling now lives in the engine rather than beside this function, so
    nothing local shows the contrast. That is why it is spelled out here.

    The engine hashes a missing external input as `<absent>`, which is right for
    one its dependency walk DISCOVERED: a deleted dependency has to move the
    digest. It is wrong for one a driver DECLARED, because there a typo and a
    deletion are indistinguishable, and the typo's digest is stable, non-empty
    and covers nothing — #132's failure with the sign flipped, inert there and
    alive-but-empty here. `OPTIONAL_EXTERNAL_INPUTS` holds the entries whose
    absence really is a state rather than a mistake.
    """
    absent = [path for path in inputs if not path.exists()]
    if not absent:
        return

    lines = ["extract-llbc.py: EXTERNAL_INPUTS names paths that do not exist:"]
    for path in absent:
        lines.append(
            f"  {path} — the engine would hash it as `<absent>`, so this entry"
            f" would contribute a constant while the rest of the set kept the"
            f" digest moving, and nothing would ever say so. Fix the path, or"
            f" move it to OPTIONAL_EXTERNAL_INPUTS if being absent is a state"
            f" this repo is meant to build in."
        )
    raise SystemExit("\n".join(lines))


def main() -> None:
    # No inert-pathspec check here: `run_cli` runs the engine's own
    # `refuse_inert_pathspecs` before every subcommand, over `base_pathspecs`
    # and every spec's `fingerprint_pathspecs` — the same scope this driver used
    # to check for itself. Verified against cel's real declaration rather than
    # assumed: it passes as declared, and refuses both an ignored-file entry and
    # a typo'd one, in each of the two scopes. Only the external-input guard is
    # still driver-local, because the engine has no equivalent.
    refuse_absent_external_inputs(EXTERNAL_INPUTS)
    run_cli(
        SPECS,
        DEFAULT_CRATES,
        root=ROOT,
        out_dir=ROOT / "build" / "llbc",
        base_pathspecs=BASE_PATHSPECS,
        charon_root=PYRE_ROOT,
        layout_targets=(),
        external_inputs=tuple(EXTERNAL_INPUTS + OPTIONAL_EXTERNAL_INPUTS),
    )


if __name__ == "__main__":
    main()
