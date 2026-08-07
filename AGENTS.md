# cel-jit working notes

## Never read `build/llbc/` without `--check`

```sh
scripts/extract-llbc.py --check cel        # exit 0 iff current, nonzero otherwise
scripts/extract-llbc.py --check cel cel-portals
```

`build/llbc/cel.ullbc` is a ~100 MB Charon artefact produced by
`scripts/extract-llbc.py` and read by whatever you point at it — a census, a
front-end B run, `PYRE_MIR_FRONTEND_LLBC`. It is git-ignored, so it is never
carried by a checkout and never updated by one either: you get whatever the last
extraction on this box left behind.

Nothing about opening the file tells you which sources produced it, and a stale
one does not fail. It answers — in detail, and about code that no longer exists.
Observed 2026-08-07: the on-box `cel.ullbc` predated the P2 series and still
carried `as_adder`, `CelString`, `CelInt`, `clone_as_boxed` and `resolve_val`,
every one of which P2 deleted. A census over it would have reported the old
`dyn Val` families and appeared to confirm a premise that fresh source refutes.

`--check` compares the artefact's `.fingerprint` stamp against the tree and
reports the verdict through the exit status. It is the same comparison a
fingerprint-unchanged extraction already makes to decide whether to skip; the
only new thing is that a consumer can now run it. Every "nothing to compare"
path — no stamp, empty stamp, stamp missing fields — is a refusal, not a pass,
because a harness reads the exit status and not the output.

It never extracts. Re-extraction is whole-crate Charon at ~8.4 GB RSS and it
writes into the working tree, so the refusal names the command and leaves the
scheduling to you:

```sh
CARGO_FEATURES=cranelift scripts/extract-llbc.py --force cel
```

Point the check at a copy with `LLBC_DEST=<dir>` when the live artefacts must
not be disturbed (e.g. an extraction is running).

### What re-extraction costs, and what triggers it

The stamp covers `Cargo.lock`, `Cargo.toml`, `scripts/extract-llbc.py`,
`cel/Cargo.toml` and `cel/src/`. **The driver is in that list**, so a
comment-only edit to `scripts/extract-llbc.py` invalidates a 100 MB artefact and
demands a multi-GB re-extraction. Keep tooling notes in this file instead — it
is outside the fingerprint.

Two inputs the stamp cannot see:

* `<pyre-root>/scripts/llbc_extract.py`, the shared engine — it lives in another
  repository and `git ls-files` cannot reach it.
* `.cargo/config.toml`'s `[patch]`, which redirects the majit git deps at the
  enclosing worktree. It is uncommitted, so the same stamp can describe two
  different artefacts depending on whether the patch was in place.

After either changes, re-extract with `--force`.
