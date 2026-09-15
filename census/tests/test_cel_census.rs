//! Front-end B census over cel's Charon LLBC.
//!
//! This is a measurement rather than an acceptance gate: it prints what the
//! translator can and cannot lower and does not assert census totals.
//!
//! The probes answer independent questions:
//!
//! * `cel_census_call_sites` lowers **every** local body with
//!   `lower_fun_decl` and classifies every call site it produces. It needs no
//!   portal, so its coverage is the whole artefact rather than one BFS
//!   closure. This is the probe that answers which call sites are walls.
//! * `cel_census_pipeline_*` runs the production analyzer from a portal seed,
//!   which additionally exercises the codewriter and annotator. Coverage is
//!   the portal's graph closure only.
//!
//! ```sh
//! cargo test --release -p cel-census --test test_cel_census \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! Do not pass `--ignored`: `cfg_attr(debug_assertions, ignore)` makes these
//! ordinary tests in release builds, so `--ignored` would select none of them.
//!
//! `--test-threads=1` is not decoration: each pipeline invocation re-seeds
//! process-global registries (`STRUCT_ORIGIN_REGISTRY`, …), so two probes
//! running concurrently race on them.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use majit_charon_reader::Llbc;
use majit_translate::flowspace::model::HOST_ENV;
use majit_translate::front::mir::{
    collect_unsafe_fn_stubs_from_llbc, lower_fun_decl, lower_fun_decl_with_static_addrs, LowerError,
};
use majit_translate::{
    AnalyzeConfig, CallPath, CallTarget, HostStaticAddrs, JitDriverSpec, OpKind, PipelineConfig,
};

/// `build/llbc/cel.ullbc` at the repository root, or `CEL_CENSUS_LLBC`.
/// Returns `None` when absent so the test skips instead of failing in a
/// checkout that has never run the extractor.
fn cel_llbc_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("CEL_CENSUS_LLBC") {
        return Some(PathBuf::from(p));
    }
    let path = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../build/llbc/cel.ullbc"
    ));
    path.exists().then_some(path)
}

/// Assert the pipeline will read the **cel** artefact, and say which one.
///
/// `MAJIT_MIR_FRONTEND_LLBC` is one of the two LLBC sources
/// `build_semantic_program_via_active_frontend` still has — explicit analyzer
/// paths, then this variable. The auto-discovery of
/// `<workspace>/build/llbc/{pyre-object,pyre-interpreter,pyre-jit}.ullbc` that
/// used to sit under them is gone, so a census that fails to set the variable
/// supplies nothing at all.
///
/// An unset variable does not fail loudly on its own: the pipeline panics at
/// `no LLBC source resolved`, the census absorbs that panic into a cell, and the
/// test reports `ok` — in 0.21s against 98.56s for a reading that actually loads
/// the corpus. This asserts rather than prints for that reason; a diagnostic
/// nobody reads would be the same failure one layer up.
///
/// `--test-threads=1` is required because the variable is process-global, so
/// tests running in parallel interleave it and the failure presents as one
/// census silently reading another's corpus. The post-run call below is what
/// catches a suite run without that flag, because it re-checks after the
/// pipeline has had a chance to observe a value some other test replaced.
/// FNV-1a 64 over the artefact's bytes.
///
/// Inlined rather than taken as a dependency because the whole point is that
/// two *different runs* can be compared, so the function must be stable across
/// Rust versions and machines. `DefaultHasher` explicitly promises the
/// opposite, and a hash that silently changes meaning between runs is worse
/// than none: it reads like a corpus that moved.
fn content_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Returns the artefact's content hash, so the caller can compare the
/// before-run and after-run readings.
///
/// Path and hash are printed on one line, and the path is not replaced by a
/// label. A hash captured under a label like `corpus sha256` and a
/// hash captured from a file are indistinguishable once they are two hex
/// strings in a message: nothing about them says they came from different
/// files, and a before/after "transition" assembled that way is two quantities
/// each measured once. Only the path travelling next to the digest prevents it.
fn assert_llbc_is_cel(expected: &PathBuf, when: &str) -> u64 {
    let resolved = std::env::var("MAJIT_MIR_FRONTEND_LLBC").unwrap_or_else(|_| {
        panic!(
            "{when}: MAJIT_MIR_FRONTEND_LLBC is unset — the pipeline would auto-discover pyre's LLBC"
        )
    });
    assert_eq!(
        PathBuf::from(&resolved),
        *expected,
        "{when}: MAJIT_MIR_FRONTEND_LLBC does not name the artefact this census set"
    );
    // Content, not filename: a basename check would pass on a pyre corpus that
    // happened to be copied to this path.
    let provenance = expected.with_extension("ullbc.provenance");
    // Fail closed when the sidecar is absent. This is an anticipated state:
    // `llbc_extract.py` reports that an artefact may predate provenance or may
    // have been written by another extractor. Accepting that state would make
    // this check unable to establish which corpus produced the census.
    let text = std::fs::read_to_string(&provenance).unwrap_or_else(|e| {
        panic!(
            "{when}: {} has no readable provenance sidecar ({e}) — refusing to call this a cel \
             extraction on the strength of its filename",
            expected.display()
        )
    });
    assert!(
        text.lines().any(|l| l.trim() == "crate=cel"),
        "{when}: {} is not a cel extraction — {} does not say `crate=cel`",
        expected.display(),
        provenance.display()
    );
    let bytes = std::fs::read(expected)
        .unwrap_or_else(|e| panic!("{when}: cannot read {} — {e}", expected.display()));
    let hash = content_hash(&bytes);
    eprintln!(
        "[census] {when}: {} len={} fnv1a64={:016x}",
        expected.display(),
        bytes.len(),
        hash
    );
    hash
}

fn skip_note() {
    eprintln!(
        "skipping: cel.ullbc missing — run `python3 scripts/extract-llbc.py cel` \
         at the repository root, or set CEL_CENSUS_LLBC"
    );
}

fn bump(counts: &mut BTreeMap<&'static str, usize>, key: &'static str) {
    *counts.entry(key).or_default() += 1;
}

/// Lower every local body and classify every call site in the result.
///
/// The wall classes are named as such: a wall stops the graph, whereas an
/// ordinary residual call is a normal lowering that the codewriter continues
/// past. `__dyn_call` is documented at `front/mir.rs:16542` as "not a
/// lowering, it is a placeholder: an unregistered synthetic path that stops
/// whatever graph reaches it".
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: lowers the whole cel LLBC; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_call_sites() {
    let Some(path) = cel_llbc_path() else {
        skip_note();
        return;
    };
    let llbc = Llbc::load(&path).expect("load cel llbc");

    let mut fns_total = 0usize;
    let mut fns_bodyless = 0usize;
    let mut fns_failed: BTreeMap<String, usize> = BTreeMap::new();
    let mut graphs = Vec::new();
    // Leaf names of bodies that lowered. Used only to split FunctionPath call
    // sites into "callee has a graph here" and "callee does not"; leaf-keyed,
    // so the ambiguity count below is reported alongside it rather than
    // silently absorbed.
    let mut lowered_leaves: BTreeMap<String, usize> = BTreeMap::new();

    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        fns_total += 1;
        if fd.unstructured().is_none() {
            fns_bodyless += 1;
            continue;
        }
        match lower_fun_decl(&llbc, fd) {
            Ok(graph) => {
                if let Some(leaf) = graph.name.rsplit("::").next() {
                    *lowered_leaves.entry(leaf.to_string()).or_default() += 1;
                }
                graphs.push(graph);
            }
            Err(err) => {
                let class = match &err {
                    LowerError::FunctionNotFound(_) => "FunctionNotFound".to_string(),
                    LowerError::Schema(_) => "Schema".to_string(),
                    // Keep the leading clause only: the tail carries block and
                    // local numbers, which would make every failure unique and
                    // turn the histogram into a list.
                    LowerError::Unsupported(msg) => {
                        let head: String = msg.chars().take(60).collect();
                        format!("Unsupported: {head}")
                    }
                };
                *fns_failed.entry(class).or_default() += 1;
            }
        }
    }

    let mut sites = BTreeMap::new();
    let mut residual_callees: BTreeMap<String, usize> = BTreeMap::new();
    let mut indirect_sites: BTreeMap<String, usize> = BTreeMap::new();
    // Name wall owners and indirect sites so equal totals cannot hide a change
    // in which graphs or trait methods make up the population.
    let mut wall_owners: BTreeMap<String, usize> = BTreeMap::new();
    for graph in &graphs {
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, .. } => match target {
                        CallTarget::FunctionPath { segments } => {
                            let leaf = segments.last().map(String::as_str).unwrap_or("");
                            if leaf == "__dyn_call" {
                                bump(&mut sites, "WALL  call __dyn_call (graph-stopping)");
                                *wall_owners.entry(graph.name.clone()).or_default() += 1;
                            } else if lowered_leaves.contains_key(leaf) {
                                bump(&mut sites, "      call FunctionPath, callee lowered here");
                            } else {
                                bump(&mut sites, "      call FunctionPath, no local graph");
                                *residual_callees.entry(leaf.to_string()).or_default() += 1;
                            }
                        }
                        CallTarget::Method { .. } => {
                            bump(&mut sites, "      call Method (receiver dispatch)")
                        }
                        CallTarget::SyntheticTransparentCtor { .. } => bump(
                            &mut sites,
                            "      call SyntheticTransparentCtor (enum shell)",
                        ),
                        CallTarget::Indirect {
                            trait_root,
                            method_name,
                        } => {
                            bump(&mut sites, "      call Indirect (vtable arm)");
                            // Lowering to `Indirect` and resolving its family are
                            // separate stages, so preserve the member names.
                            *indirect_sites
                                .entry(format!("{trait_root}::{method_name}"))
                                .or_default() += 1;
                        }
                        CallTarget::UnsupportedExpr => {
                            bump(&mut sites, "WALL  call UnsupportedExpr");
                            *wall_owners.entry(graph.name.clone()).or_default() += 1;
                        }
                    },
                    OpKind::IndirectCall { graphs, .. } => match graphs {
                        Some(candidates) if !candidates.is_empty() => {
                            bump(&mut sites, "      indirect-call, candidate graphs present")
                        }
                        Some(_) => bump(&mut sites, "      indirect-call, empty candidate list"),
                        None => {
                            bump(&mut sites, "WALL  indirect-call, graphs=None");
                            *wall_owners.entry(graph.name.clone()).or_default() += 1;
                        }
                    },
                    _ => {}
                }
            }
        }
    }

    let ambiguous_leaves = lowered_leaves.values().filter(|n| **n > 1).count();
    let total_sites: usize = sites.values().sum();
    let walls: usize = sites
        .iter()
        .filter(|(k, _)| k.starts_with("WALL"))
        .map(|(_, n)| *n)
        .sum();

    eprintln!("=== cel front-end B census: {} ===", path.display());
    // Include process identity and the relevant gate value so separate census
    // runs can be distinguished without assuming anything about caches.
    eprintln!(
        "pid {}  MAJIT_FNPTR_INDIRECT={}",
        std::process::id(),
        std::env::var("MAJIT_FNPTR_INDIRECT").unwrap_or_else(|_| "<unset>".into())
    );
    eprintln!("local fns visited        {fns_total}");
    eprintln!("  no body (opaque)       {fns_bodyless}");
    eprintln!("  lowered                {}", graphs.len());
    eprintln!(
        "  refused                {}",
        fns_failed.values().sum::<usize>()
    );
    for (class, n) in &fns_failed {
        eprintln!("      {n:6}  {class}");
    }
    eprintln!("call sites in lowered graphs  {total_sites}");
    for (class, n) in &sites {
        eprintln!("      {n:6}  {class}");
    }
    eprintln!("walls                    {walls} of {total_sites}");
    eprintln!(
        "leaf-keyed control: {ambiguous_leaves} of {} lowered leaves are owned by >1 body, \
         so the FunctionPath split above is approximate by that much",
        lowered_leaves.len()
    );
    eprintln!("wall sites, by owning graph:");
    for (name, n) in &wall_owners {
        eprintln!("      {n:6}  {name}");
    }
    eprintln!("vtable (Indirect) sites, by trait::method:");
    for (name, n) in &indirect_sites {
        eprintln!("      {n:6}  {name}");
    }
    let mut top: Vec<(&String, &usize)> = residual_callees.iter().collect();
    top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    eprintln!("top callees with no local graph:");
    for (name, n) in top.into_iter().take(20) {
        eprintln!("      {n:6}  {name}");
    }
}

/// Opnames in a `JitCode::dump()`, one per assembled instruction.
///
/// `format_assembler` (`codewriter/format.rs:112-`) writes one line per
/// `FlatOp`, optionally prefixed by a `%4d  ` bytecode position, and every
/// non-label arm opens with the opname. Labels are `L<n>:` and are the only
/// lines that are not instructions, so they are the only thing filtered.
fn dump_opnames(dump: &str) -> impl Iterator<Item = &str> {
    dump.lines().filter_map(|line| {
        let mut tokens = line.split_whitespace();
        let first = tokens.next()?;
        // The position prefix is a bare integer; the opname is the next token.
        let name = if first.bytes().all(|b| b.is_ascii_digit()) {
            tokens.next()?
        } else {
            first
        };
        (!name.ends_with(':')).then_some(name)
    })
}

/// Print reproducible pipeline-shape measurements for one portal.
///
/// Two measurements are deliberately absent rather than approximated:
///
/// * the ULLBC `Drop` / `Call` terminator / fn counts are over the portal's
///   ULLBC closure, which this result does not carry — a whole-artefact count
///   would be a different denominator wearing the same name;
/// * a "real-computation ops" percentage needs a separately specified opname
///   classification, so the full histogram is printed instead.
fn section1_cells(label: &str, result: &majit_translate::pipeline::ProgramPipelineResult) {
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for jitcode in &result.jitcodes {
        for name in dump_opnames(&jitcode.dump()) {
            *hist.entry(name.to_string()).or_default() += 1;
        }
    }
    let ops: usize = hist.values().sum();
    // Print counts both with and without the liveness and end-of-block
    // pseudo-ops so consumers can choose an explicit instruction definition.
    let live_markers = hist.get("-live-").copied().unwrap_or(0);
    let block_markers = hist.get("---").copied().unwrap_or(0);
    let count_prefix = |p: &str| -> usize {
        hist.iter()
            .filter(|(k, _)| k.starts_with(p))
            .map(|(_, n)| n)
            .sum()
    };
    let exact = |k: &str| -> usize { hist.get(k).copied().unwrap_or(0) };

    eprintln!("--- §1 cells [{label}] ---");
    eprintln!("  jitcodes                  {}", result.jitcodes.len());
    eprintln!("  ops (all dump lines)      {ops}");
    eprintln!(
        "  ops less `-live-`         {}      <- the rule that reproduces §1",
        ops - live_markers
    );
    eprintln!(
        "  ops less `-live-` + `---` {}",
        ops - live_markers - block_markers
    );
    eprintln!(
        "  residual_call* : inline_call*   {} : {}",
        count_prefix("residual_call"),
        count_prefix("inline_call")
    );
    eprintln!("  guard_class               {}", exact("guard_class"));
    // The dump and instruction table use different spellings for this op, so
    // print both rather than silently reporting zero for one representation.
    eprintln!(
        "  vtablemethodptr (dump)    {}   [insns-table spelling `vtable_method_ptr`: {}]",
        exact("vtablemethodptr"),
        exact("vtable_method_ptr")
    );
    // Same split as `vtablemethodptr` above: the dump spells this op
    // `newwithvtable` and the insns table spells it `new_with_vtable`, and the
    // histogram queried here is the dump's. Looking up only the insns-table
    // spelling reads as a hard zero — a missing lookup, indistinguishable from a
    // fuse that declined — so print both and let the reader see which one is
    // populated.
    eprintln!(
        "  new / newwithvtable       {} / {}   [insns-table spelling `new_with_vtable`: {}]",
        exact("new"),
        exact("newwithvtable"),
        exact("new_with_vtable")
    );
    eprintln!(
        "  indirectcalltarget_indices  {}",
        result.indirectcalltarget_indices.len()
    );
    eprintln!("  opname histogram ({} distinct):", hist.len());
    let mut rows: Vec<(&String, &usize)> = hist.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    for (name, n) in rows {
        eprintln!("      {n:6}  {name}");
    }
}

fn run_pipeline_census(label: &str, portal: CallPath) {
    run_pipeline_census_with_pytypes(label, portal, &[]);
}

/// [`run_pipeline_census`], with the class statics' addresses supplied.
///
/// `pytypes` is not decoration for a boxing census — it is the gate.
/// `resolve_vtable_addr` has to turn `&CEL_INT_CLASS` into an address to put in
/// `NewWithVtable.vtable`, and `HostStaticAddrs.pytypes` is the only channel
/// that carries one (`test_mir_frontend.rs
/// boxing_cluster_fuses_from_the_host_supplied_class_address` asserts that
/// supplying it is by itself sufficient). With the default empty slice the fuse
/// declines with a bare `continue`, so a `new_with_vtable` of `0` reports the
/// missing address rather than anything about the constructor — which is why
/// the two entry points are separate rather than one with a default.
fn run_pipeline_census_with_pytypes<'a>(
    label: &str,
    portal: CallPath,
    pytypes: &'a [(&'a str, i64)],
) {
    let Some(path) = cel_llbc_path() else {
        skip_note();
        return;
    };
    // SAFETY: serialized test binary (`--test-threads=1`); set before the
    // pipeline reads it and before any worker spawns.
    unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", &path) };
    // Supersedes a bare "LLBC in use: <path>" line: the path alone cannot
    // distinguish this corpus from one rewritten under the same name.
    let hash_before = assert_llbc_is_cel(&path, "before the pipeline");

    let config = AnalyzeConfig {
        pipeline: PipelineConfig {
            transform: Default::default(),
            jit_drivers: vec![JitDriverSpec {
                portal,
                greens: Vec::new(),
                reds: Vec::new(),
                green_kinds: Vec::new(),
                red_kinds: Vec::new(),
                autoreds: false,
                virtualizables: Vec::new(),
                red_types: Vec::new(),
            }],
            register_trait_families: Vec::new(),
        },
    };

    // A panic here is expected output, not a bug to be silenced. The pipeline
    // is designed to fail loud on a shape it cannot digest, and it prints its
    // census histograms before reaching that point — so catching the unwind
    // and reporting the message is what makes this a measurement at all.
    // Anyone "fixing" this into a quiet fallback destroys the exact signal the
    // harness exists to collect.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        majit_translate::analyze_multiple_pipeline_with_modules(
            &[],
            &config,
            None,
            &|_, _| None,
            &[],
            HostStaticAddrs {
                pytypes,
                ..Default::default()
            },
        )
    }));

    // Re-checked AFTER the run, not only before: the value the pipeline
    // actually observed is the one live during it, and a parallel test could
    // only have replaced it in that window.
    //
    // ⭐ Comparing the two HASHES closes a hazard the variable check cannot see
    // at all. `MAJIT_MIR_FRONTEND_LLBC` can name the same path start to finish
    // while a concurrent extraction rewrites the bytes underneath it — a real
    // near-miss on this tree, where a re-extraction landed ~110s after a census
    // had read the artefact. Nothing recorded the ordering, so it was
    // reconstructible only from a log mtime that happened to sit near the
    // rewrite. This makes it decidable instead.
    let hash_after = assert_llbc_is_cel(&path, "after the pipeline");
    assert_eq!(
        hash_before, hash_after,
        "the LLBC was rewritten while this census was reading it — the cells above describe \
         no single corpus"
    );

    match outcome {
        Ok(result) => {
            eprintln!("=== cel pipeline census [{label}]: completed ===");
            eprintln!("jitcodes emitted: {}", result.jitcodes.len());
            let names: BTreeSet<String> = result
                .jitcodes_by_path
                .keys()
                .map(|k| k.canonical_key())
                .collect();
            eprintln!("jitcode paths ({}): {names:#?}", names.len());
            let mut insns: Vec<&String> = result.insns.keys().collect();
            insns.sort_unstable();
            eprintln!("insn vocabulary ({}): {insns:?}", insns.len());
            section1_cells(label, &result);
        }
        Err(err) => {
            let msg = err
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| err.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic>");
            eprintln!("=== cel pipeline census [{label}]: panicked ===");
            eprintln!("panic: {msg}");
        }
    }
}

/// Census the typed bytecode VM portal `clean_interp_seeded_f`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_typed_vm() {
    run_pipeline_census(
        "clean_interp_seeded_f",
        CallPath::from_segments(["majit", "bytecode", "float_bank", "clean_interp_seeded_f"]),
    );
}

/// Census the float-bank main loop `run_mainloop_f`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_mainloop() {
    run_pipeline_census(
        "run_mainloop_f",
        CallPath::from_segments(["majit", "bytecode", "float_bank", "run_mainloop_f"]),
    );
}

/// `cel::vm::eval`, the bytecode VM entry. A free function, which matters:
/// `register_configured_jitdrivers` (`lib.rs:1942`) asserts the portal
/// resolves to an exact graph in `call_control.function_graphs()`, and the
/// walker's own entry (`cel::objects::<Impl>::resolve_value`) is an inherent
/// method rather than a free function.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_vm_eval() {
    run_pipeline_census("vm::eval", CallPath::from_segments(["vm", "eval"]));
}

/// Census the AST walker. Associated functions must include their owning type
/// because only free functions receive widened alias paths.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_walker() {
    run_pipeline_census(
        "objects::Value::resolve_value",
        CallPath::from_segments(["objects", "Value", "resolve_value"]),
    );
}

/// Census the class-family arithmetic chain `runtime::binop::cel_add`.
///
/// This portal exists to read one cell: `new / new_with_vtable`. The class
/// family in `cel::runtime` is built on the premise that its constructors fuse
/// — that `fuse_boxing_alloc` recognises the `lltype::malloc_typed` body and
/// mints a `NewWithVtable` the optimizer can delete — and all three of the
/// conditions that premise rests on fail with a bare `continue`, so nothing
/// reports a constructor that did not fuse. A portal seeded here puts the
/// constructors in a closure where the cell is countable.
///
/// `cel_add` reaches `new_int`, `new_uint`, `new_double`, `new_duration` and
/// `new_timestamp` through its arms; `new_bool` is under the ordering chain,
/// which is why the next probe exists rather than this one standing alone.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_add() {
    run_pipeline_census_with_pytypes(
        "runtime::binop::cel_add",
        CallPath::from_segments(["runtime", "binop", "cel_add"]),
        CEL_CLASS_ADDRS,
    );
}

/// Census the class-family ordering chain `runtime::binop::cel_less`.
///
/// The `new_bool` half of the fuse question, plus the only chain whose arms
/// return an `i64` code rather than a value. See
/// [`cel_census_pipeline_runtime_add`].
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_less() {
    run_pipeline_census_with_pytypes(
        "runtime::binop::cel_less",
        CallPath::from_segments(["runtime", "binop", "cel_less"]),
        CEL_CLASS_ADDRS,
    );
}

/// Census the optional constructors through `optional.ofNonZeroValue`.
///
/// The family's first MANAGED payload, and so the case least entitled to be
/// assumed from the five scalar constructors [`cel_census_pipeline_runtime_add`]
/// covers: `new_optional` stores a `CelRef` *argument* into the allocation where
/// every scalar leaf stores an `i64`.
///
/// This portal rather than `cel_optional_of` because its closure holds BOTH
/// constructors — the zero arm allocates through `new_optional_none`, the other
/// through `new_optional` — so one run says whether each fuses. They are
/// separate functions on purpose (`new_optional_none` spells its null at the
/// allocation site), which is exactly why one fusing is not evidence about the
/// other.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_optional() {
    run_pipeline_census_with_pytypes(
        "runtime::optional::cel_optional_of_non_zero_value",
        CallPath::from_segments(["runtime", "optional", "cel_optional_of_non_zero_value"]),
        CEL_CLASS_ADDRS,
    );
}

/// Census the variable-length leaf `new_list`.
///
/// `W_ListObject` is the leaf whose fuse result the encoding argued in
/// `runtime::object_array`'s module doc rests on: the payload is a separately
/// allocated block precisely so the leaf stays a fixed-size struct the fuse can
/// match. If it does not fuse, the block split bought nothing.
///
/// Its payload pointer is NOT a managed edge — the block is allocated outside
/// the traced heap, and `runtime::registration` registers no offset for it. The
/// family's only managed edge is `W_OptionalObject::w_value`.
///
/// The seed is the constructor itself rather than a caller, unlike
/// [`cel_census_pipeline_runtime_add`] and [`cel_census_pipeline_runtime_optional`],
/// because the three variable-length constructors have no production call site
/// yet — every use in the crate is under `#[cfg(test)]`, which the extractor
/// does not carry. There is no `cel_*` chain that reaches them, so a chain
/// portal would report a closure that never allocates one. `fuse_boxing_alloc`
/// runs per graph over the constructor's own body, so the constructor's graph
/// being in the closure is what makes the cell countable; a caller is not
/// needed for that, only for the wider closure the other probes also measure.
///
/// Two cells answer the question. `new / newwithvtable` counts the leaf
/// allocations that fused. The block allocation cannot fuse at all —
/// `new_items_block` is size-parameterised, lives outside `runtime::lltype` and
/// is not spelled `malloc_typed`, so the matcher never even inspects it — so it
/// survives as a call, and `residual_call* : inline_call*` is which KIND of call
/// it survived as. Those are different outcomes: a residual call stops the
/// closure at the block allocator, an inline call carries it in.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_new_list() {
    run_pipeline_census_with_pytypes(
        "runtime::object::new_list",
        CallPath::from_segments(["runtime", "object", "new_list"]),
        CEL_CLASS_ADDRS,
    );
}

/// Census the variable-length leaf `new_bytes`.
///
/// Its own seed rather than a widening of [`cel_census_pipeline_runtime_new_list`]:
/// the two closures are disjoint below the leaf — `new_list` reaches
/// `new_items_block`, `new_bytes` reaches `new_bytes_block` — so neither run says
/// anything about the other's block call, and the leaves are separate structs
/// with separate field layouts the fuse resolves independently.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_new_bytes() {
    run_pipeline_census_with_pytypes(
        "runtime::object::new_bytes",
        CallPath::from_segments(["runtime", "object", "new_bytes"]),
        CEL_CLASS_ADDRS,
    );
}

/// Census the variable-length leaf `new_string`.
///
/// `new_string` shares `new_bytes_block` with `new_bytes` but not its leaf:
/// `W_StringObject` is a distinct struct with distinct field names, and the fuse
/// resolves the payload stores by field name off that layout. See
/// [`cel_census_pipeline_runtime_new_bytes`] for why each leaf gets its own seed.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_new_string() {
    run_pipeline_census_with_pytypes(
        "runtime::object::new_string",
        CallPath::from_segments(["runtime", "object", "new_string"]),
        CEL_CLASS_ADDRS,
    );
}

/// Census the items block's own constructor, seeded AT it.
///
/// [`cel_census_pipeline_runtime_new_list`] cannot answer anything about this
/// function's body. `new_items_block` copies its elements in a `while` loop, so
/// its graph carries a backedge, and `JitPolicy::look_inside_graph` rejects any
/// loopy graph — `CallControl::find_all_graphs_bfs` therefore never adds it to
/// `candidate_graphs`, `CallControl::graphs_from` answers `None`, and
/// `guess_call_kind` classifies the call `Residual`. Seeded from `new_list` it
/// is a `residual_call_r_r` and nothing inside it is ever dumped, so that census
/// is silent about the element store rather than negative about it.
///
/// A portal seed bypasses exactly that gate: `find_all_graphs_bfs` inserts the
/// jitdrivers' own portal graphs into `candidate_graphs` directly, with no
/// `look_inside_graph` call, because a portal is the thing being compiled rather
/// than a callee being judged. So this seed makes the loopy body dumpable
/// WITHOUT relaxing the policy, changing cel, or touching the front end.
///
/// **Read the jitcode count first.** A portal that produced no jitcode and an
/// element store that lowered to nothing are the same empty histogram, and only
/// the count separates them. `register_configured_jitdrivers` asserts the portal
/// path resolves, so a typo fails loudly rather than quietly measuring nothing —
/// but a resolved portal that still emits zero jitcodes would not, and that is
/// the reading this comment exists to prevent.
///
/// `CEL_CLASS_ADDRS` is passed although this closure allocates no class-headed
/// object: the block has no vtable, so nothing here should fuse. Supplying the
/// table anyway keeps a `new_with_vtable` of `0` meaning "nothing fused" rather
/// than "no address was available to fuse with".
///
/// # The reading of record, and what it can be attributed to
///
/// First run: 3 jitcodes (`new_items_block`, `items_block_items_base`,
/// `alloc_block`), vocabulary carrying `setarrayitem_gc_r`, histogram with one
/// `arraywrite`, one `arrayread` and two `arraylen` — the loop body. The
/// element store lowers to a reference array store.
///
/// That reading was taken against a tree where `majit-translate` was **dirty**,
/// and the split matters more than the fact:
///
/// * Reconstructable — `codewriter/policy.rs`, `model.rs`, `lib.rs`, `parse.rs`,
///   `codewriter/call.rs`, `codewriter/codewriter.rs` and
///   `front/result_exc.rs` were in the state that became the decline-instrument
///   commit, so their measured content is recoverable from it.
/// * **Not** reconstructable — `decline.rs` and `translator/rtyper/cutover.rs`
///   were edited again after the rlib was built, so whatever they held at build
///   time is gone. `cutover::is_known_unported` participates in the
///   residual-versus-candidate decision, so this is not a nil concern.
/// * Pinned regardless — `front/mir.rs`, which holds
///   `is_list_items_elem_ptr_add_parts` and therefore actually decides the
///   store, was clean; and the input was `cel.ullbc` extracted from a clean
///   cel-jit tree (`dirty_status=ok`).
///
/// So the result is strong enough to act on and **not yet citable as a grade**:
/// a confirming re-run against a clean tree is outstanding. Record the same
/// split for any future reading rather than the bare cells — a number whose
/// tree nobody can reconstruct expires the moment someone asks which code
/// produced it.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_runtime_new_items_block() {
    run_pipeline_census_with_pytypes(
        "runtime::object_array::new_items_block",
        CallPath::from_segments(["runtime", "object_array", "new_items_block"]),
        CEL_CLASS_ADDRS,
    );
}

/// The class statics of `cel::runtime::object`, at placeholder addresses.
///
/// Translation-time values only: the census lowers and never executes, and
/// these reach nothing but `NewWithVtable.vtable`. They are distinct so that a
/// fused site names which class it captured rather than one shared constant
/// standing in for twelve.
///
/// The table has to name every class the seeded closure can allocate: a class
/// missing here has no address for `resolve_vtable_addr` to resolve, and the
/// fuse declines with a bare `continue` — a zero that is about this table
/// rather than about the constructor.
///
/// The last three are the variable-length leaves. They are appended at the next
/// free addresses rather than inserted in declaration order, because these
/// values are placeholders whose only requirement is distinctness — renumbering
/// the nine above would change every existing row for no reading.
const CEL_CLASS_ADDRS: &[(&str, i64)] = &[
    ("CEL_INT_CLASS", 0x0001_0000),
    ("CEL_UINT_CLASS", 0x0001_0100),
    ("CEL_DOUBLE_CLASS", 0x0001_0200),
    ("CEL_BOOL_CLASS", 0x0001_0300),
    ("CEL_NULL_CLASS", 0x0001_0400),
    ("CEL_DURATION_CLASS", 0x0001_0500),
    ("CEL_TIMESTAMP_CLASS", 0x0001_0600),
    ("CEL_TYPE_CLASS", 0x0001_0700),
    ("CEL_OPTIONAL_CLASS", 0x0001_0800),
    ("CEL_BYTES_CLASS", 0x0001_0900),
    ("CEL_STRING_CLASS", 0x0001_0a00),
    ("CEL_LIST_CLASS", 0x0001_0b00),
    ("CEL_MAP_CLASS", 0x0001_0c00),
    ("CEL_STRUCT_CLASS", 0x0001_0d00),
    ("CEL_OPAQUE_CLASS", 0x0001_0e00),
    ("CEL_INT_COLUMN_CLASS", 0x0001_0f00),
    ("CEL_FRAME_CLASS", 0x0001_1000),
];

/// Per-symbol lowering-wall table for the `cel::vm` evaluator.
///
/// The whole-crate census reports aggregate opacity; this table instead shows
/// which evaluator bodies lower and where each one stops.
///
/// It is a census, not a gate. The only assertion is that the module was found
/// at all — a filter that silently matches nothing would otherwise print an
/// empty table and read as "no walls".
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: lowers the whole cel LLBC; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_vm_walls() {
    let Some(path) = cel_llbc_path() else {
        skip_note();
        return;
    };
    let llbc = Llbc::load(&path).expect("load cel llbc");

    // Keep one entry per body, not per name: Charon emits a separate `FunDecl`
    // per monomorphization, and a name-keyed map would overwrite siblings.
    let mut rows: Vec<(String, String)> = Vec::new();
    let mut clean = 0usize;
    let mut bodyless = 0usize;
    let mut walled = 0usize;
    let mut wall_classes: BTreeMap<String, usize> = BTreeMap::new();
    // Call sites INSIDE bodies that lowered. A body can lower and still be
    // useless to trace if every arm leaves through an opaque call, so "clean"
    // is reported with its residual/indirect counts rather than alone.
    let mut dyn_call_sites = 0usize;
    let mut indirect_sites = 0usize;
    let mut method_sites = 0usize;
    let mut unsupported_sites = 0usize;

    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        let name = fd.item_meta.name_path();
        if !name.contains("cel::vm") {
            continue;
        }
        if fd.unstructured().is_none() {
            bodyless += 1;
            rows.push((name, "bodyless (no MIR in this artefact)".to_string()));
            continue;
        }
        match lower_fun_decl(&llbc, fd) {
            Ok(graph) => {
                let mut dyns = 0usize;
                let mut inds = 0usize;
                let mut meths = 0usize;
                let mut residuals = 0usize;
                // Counted on its own rather than folded into the residual
                // bucket: a call whose TARGET EXPRESSION did not lower is a
                // different fact from a call to a callee with no local graph,
                // and only the first says the front end could not read the
                // call at all.
                let mut unsupported = 0usize;
                for block in &graph.blocks {
                    for op in &block.operations {
                        if let OpKind::Call { target, .. } = &op.kind {
                            match target {
                                CallTarget::FunctionPath { segments } => {
                                    if segments.last().map(String::as_str) == Some("__dyn_call") {
                                        dyns += 1;
                                    } else {
                                        residuals += 1;
                                    }
                                }
                                CallTarget::Indirect { .. } => inds += 1,
                                CallTarget::Method { .. } => meths += 1,
                                CallTarget::UnsupportedExpr => unsupported += 1,
                                CallTarget::SyntheticTransparentCtor { .. } => {}
                            }
                        }
                    }
                }
                clean += 1;
                dyn_call_sites += dyns;
                indirect_sites += inds;
                method_sites += meths;
                unsupported_sites += unsupported;
                rows.push((
                    name,
                    format!(
                        "lowered   blocks={:<4} dyn_call={dyns:<3} indirect={inds:<3} \
                         method={meths:<3} unsupported={unsupported:<3} residual={residuals}",
                        graph.blocks.len(),
                    ),
                ));
            }
            Err(err) => {
                // The leading clause only. The tail carries block and local
                // numbers, which would make every wall unique and turn the
                // class histogram below into a copy of the row list.
                let class = match &err {
                    LowerError::FunctionNotFound(_) => "FunctionNotFound".to_string(),
                    LowerError::Schema(_) => "Schema".to_string(),
                    LowerError::Unsupported(msg) => {
                        format!("Unsupported: {}", msg.chars().take(70).collect::<String>())
                    }
                };
                walled += 1;
                *wall_classes.entry(class.clone()).or_default() += 1;
                rows.push((name, format!("WALL      {class}")));
            }
        }
    }

    assert!(
        !rows.is_empty(),
        "no `cel::vm` bodies in {}: the name filter matched nothing, which is a \
         broken instrument rather than a clean module. Check the artefact's \
         naming (whole-crate vs --start-from) before reading any zero here.",
        path.display()
    );

    println!(
        "=== cel::vm lowering walls, artefact {} ===",
        path.display()
    );
    rows.sort();
    for (name, verdict) in &rows {
        println!("  {name}\n      {verdict}");
    }
    rows.sort();
    let distinct = rows.iter().map(|(n, _)| n).collect::<BTreeSet<_>>().len();
    println!(
        "--- totals over {} `cel::vm` bodies ({distinct} distinct names) ---",
        rows.len()
    );
    println!("  lowered            : {clean}");
    println!("  walled             : {walled}");
    println!("  bodyless           : {bodyless}");
    println!("  call sites inside the lowered bodies:");
    println!("    __dyn_call       : {dyn_call_sites}");
    println!("    Indirect (vtable): {indirect_sites}");
    println!("    Method (receiver): {method_sites}");
    println!("    UnsupportedExpr  : {unsupported_sites}");
    if !wall_classes.is_empty() {
        println!("--- wall classes ---");
        for (class, n) in &wall_classes {
            println!("  {n:>4}  {class}");
        }
    }
}

/// Seed the pipeline directly at the dispatch loop.
///
/// `cel_census_pipeline_vm_eval` seeds at `vm::eval` and its jitcode closure
/// contains `cel_eval_loop`, `Vm::new` and `Vm::public_error` — everything
/// `cel_eval_loop` calls except `Vm::run`, and so nothing of the loop or the
/// opcode match. Meanwhile `cel_census_vm_walls` shows both bodies lower with
/// zero walls in isolation. Two causes fit that pair: the method call does not
/// carry the closure across, or the annotator failures recorded on that run
/// truncated it before it got there.
///
/// Seeding here removes the first hop, so it separates them: a closure that
/// now contains `Vm::step` puts the fault on reachability from `vm::eval`,
/// which is what the portal reshape (Step 2a) exists to fix; a closure that
/// still does not, or a run that dies the same way, puts it on the annotator
/// and Step 2a would not have helped.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_pipeline_vm_run() {
    run_pipeline_census(
        "vm::interp::Vm::run",
        CallPath::from_segments(["vm", "interp", "Vm", "run"]),
    );
}

// ---------------------------------------------------------------------------
// Route-B callee-registration residue
// ---------------------------------------------------------------------------

/// Paths of `cutover.rs`'s `FOREIGN_STDLIB_EXTERNALS`, `::`-joined.
///
/// Copied because the table is `pub(crate)` and an integration test cannot
/// reach it. Only the paths are copied: the result lltype decides whether
/// `build_stub_pygraph_for_lltype` *builds* the stub, not whether the path is a
/// candidate, so a path here whose stub declines reads as registered when it is
/// not. That is the only direction this copy can be wrong in, and it
/// understates residue rather than inventing it.
const FOREIGN_STDLIB_EXTERN_PATHS: &[&str] = &[
    "core::mem::swap",
    "core::mem::drop",
    "core::mem::forget",
    "core::mem::size_of",
    "std::mem::size_of",
    "alloc::alloc::handle_alloc_error",
    "sync::atomic::AtomicBool::store",
    "sync::atomic::AtomicU8::store",
    "cell::Cell::set",
    "core::num::<Impl>::saturating_mul",
    "core::f64::<Impl>::is_infinite",
    "core::slice::<Impl>::is_empty",
    "core::slice::<Impl>::contains",
    "std::f64::<Impl>::floor",
    "std::f64::<Impl>::ceil",
    "std::f64::<Impl>::powf",
    "core::f64::<Impl>::to_bits",
    "core::f64::<Impl>::abs",
    "core::f64::<Impl>::is_nan",
    "core::f64::<Impl>::NAN",
    "core::f64::<Impl>::INFINITY",
];

/// The three universes, in print order. `DYING` is first because it is the
/// control.
///
/// The predicate is the **owning body's** module path, and each table lists the
/// call targets those bodies emit. Scoping the other way — filtering by the
/// *callee's* module — cannot express the control at all: `Arc` lives under
/// `alloc::sync`, so a table of callees under `cel::objects::` would hold no
/// `Arc` row no matter how many the value layer makes.
const ROUTE_B_SCOPES: &[(&str, &str, &str)] = &[
    (
        "DYING",
        "cel::objects::",
        "deleted by route B — THE CONTROL",
    ),
    ("POST_B", "cel::runtime::", "route B's destination"),
    ("SURVIVOR", "cel::vm::", "survives the swap unchanged"),
];

/// Front-end synthetic call heads, with the arity guard each arm carries.
///
/// `translate_op` (`flowspace_adapter.rs:696`, `:727`, `:1677`, `:1733`,
/// `:1771`, `:1804`, `:1847`, `:1873`, `:1882`, `:1923`, `:1968`) matches these
/// BEFORE it reaches `PyreCallRegistry`. They are heads the front end mints,
/// not callees anything could register, so counting them as residue would put
/// the largest rows in every table on a question that is not asked.
///
/// `(head, segment count, arg count)`; `None` means the arm does not constrain
/// that count. A head whose guard misses falls through to the registry and
/// HOST_ENV lookups below, exactly as it does in `translate_op`.
const FRONT_END_SHIM_ARMS: &[(&str, Option<usize>, Option<usize>)] = &[
    ("__str_const", Some(2), Some(0)),
    ("__const_int_array", None, Some(0)),
    ("__array_repeat", Some(1), Some(2)),
    ("__cast_pointer", Some(2), Some(1)),
    ("__pyre_cast_instance", Some(2), None),
    ("__pyre_cast_address", Some(1), None),
    ("__pyre_range", Some(1), None),
    ("__pyre_stringbuilder_new", Some(1), None),
    ("__pyre_stringbuilder_append", Some(1), None),
    ("__pyre_stringbuilder_build", Some(1), None),
    ("__len", Some(1), None),
];

/// Foreign-container callees `translate_op` re-routes to an rtyper list
/// operation before consulting the registry
/// (`is_vec_ctor_segments`, `is_vec_from_elem_segments`,
/// `is_vec_push_segments`, `is_vec_extend_from_slice_segments`,
/// `is_slice_reverse_segments`, and the `__len` arm's `core::slice` spelling).
///
/// `vec::Vec::default` is deliberately absent — there is no recogniser for it,
/// so it is residue, and adding it here to tidy the table would erase a real
/// row.
const FOREIGN_CONTAINER_OPS: &[&str] = &[
    "vec::Vec::new",
    "vec::Vec::with_capacity",
    "vec::Vec::push",
    "vec::Vec::extend_from_slice",
    "alloc::vec::from_elem",
    "core::slice::<Impl>::reverse",
    "core::slice::<Impl>::len",
];

/// How a callee path resolves, by the channel that answers it.
///
/// The order is `translate_op`'s own dispatch order: front-end shims, then
/// `PyreCallRegistry` (whose three population passes run
/// graphs → unsafe stubs → foreign externs, all before any lookup), then the
/// HOST_ENV layers.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Registration {
    /// `CallTarget::Method`. Never charged against the registry: a method
    /// target is resolved through the receiver's classdef
    /// (`SomeInstance.classdef` → `MethodDesc`), and the registry is not
    /// consulted. Charging its failures here would move a classdef question
    /// into the registration column and shrink the residue number.
    NotApplicable,
    /// A head from [`FRONT_END_SHIM_ARMS`].
    FrontEndShim,
    /// A path from [`FOREIGN_CONTAINER_OPS`].
    ForeignContainerOp,
    /// A local decl carrying a body, so `CallControl::function_graphs` can
    /// carry it. Body presence, not a completed lowering — see
    /// [`classify_callee`] for what that costs the residue direction.
    Graph,
    /// A local decl with no body that `collect_unsafe_fn_stubs_from_llbc`
    /// accepted, so the stub channel is the only one that holds the key.
    UnsafeStub,
    /// In [`FOREIGN_STDLIB_EXTERN_PATHS`].
    ForeignExtern,
    /// Single-segment `HOST_ENV.lookup_builtin` hit — resolution layer 2.
    HostBuiltin,
    /// `HOST_ENV.import_module(prefix).module_get(leaf)` hit — layer 3b.
    HostModuleAttr,
    /// `["simple_call", <exception class>]`, the `exc_from_raise`
    /// reconstruction — layer 3c.
    HostExcClass,
    /// `__dyn_call`: residue, but named. `front/mir.rs:16542` calls it "not a
    /// lowering, it is a placeholder: an unregistered synthetic path that stops
    /// whatever graph reaches it", so it is a wall rather than a callee anyone
    /// could register.
    DynCallWall,
    /// Residue.
    Unregistered,
}

impl Registration {
    fn label(self) -> &'static str {
        match self {
            Registration::NotApplicable => "n/a (method)",
            Registration::FrontEndShim => "shim",
            Registration::ForeignContainerOp => "container-op",
            Registration::Graph => "graph",
            Registration::UnsafeStub => "unsafe-stub",
            Registration::ForeignExtern => "extern",
            Registration::HostBuiltin => "host-builtin",
            Registration::HostModuleAttr => "host-attr",
            Registration::HostExcClass => "host-exc",
            Registration::DynCallWall => "WALL __dyn_call",
            Registration::Unregistered => "UNREGISTERED",
        }
    }

    /// Whether this verdict is callee-registration residue.
    fn is_residue(self) -> bool {
        matches!(self, Registration::Unregistered | Registration::DynCallWall)
    }
}

/// The artefact's `FunDecl` status for a callee path.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LocalDecl {
    /// Local decl carrying a body — the thing that can become a graph.
    Body,
    /// Local decl with `body: null`. Opaque, so it never reaches
    /// `function_graphs`.
    Bodyless,
    /// No local decl at this path.
    Absent,
    /// A `CallTarget::Method` row. The column is a `FunctionPath` lookup, and a
    /// method target has no path to look up — printing `no` here would assert
    /// an absence nobody checked.
    NotApplicable,
}

impl LocalDecl {
    fn label(self) -> &'static str {
        match self {
            LocalDecl::Body => "yes",
            LocalDecl::Bodyless => "bodyless",
            LocalDecl::Absent => "no",
            LocalDecl::NotApplicable => "n/a",
        }
    }
}

/// Local-crate decl facts, merged across a path's monomorphization siblings.
#[derive(Default)]
struct LocalFnFacts {
    has_body: bool,
    /// A sibling at this path that `collect_unsafe_fn_stubs_from_llbc`
    /// accepted. Read back from the collector rather than re-derived from
    /// `signature.is_unsafe`: being unsafe is only the collector's selector,
    /// and it declines on three further conditions.
    stub_registered: bool,
    siblings: usize,
}

/// One callee, aggregated over every site in one scope.
#[derive(Default)]
struct CalleeAgg {
    sites: usize,
    owners: BTreeSet<String>,
    /// The callee's segments, kept rather than re-split from the display path:
    /// a `__str_const` literal can itself contain `::`.
    segments: Vec<String>,
    /// Every arg count seen at this callee. The shim arms guard on arity, and
    /// aggregating by path alone would let one conforming site vouch for a
    /// sibling that misses the guard.
    arg_counts: BTreeSet<usize>,
}

/// A printed table row.
struct ResidueRow {
    path: String,
    method: bool,
    sites: usize,
    owners: usize,
    local: LocalDecl,
    registration: Registration,
}

/// Per-scope accumulation.
#[derive(Default)]
struct ScopeTally {
    lowered: usize,
    bodyless: usize,
    walled: usize,
    wall_classes: BTreeMap<String, usize>,
    fn_callees: BTreeMap<String, CalleeAgg>,
    method_callees: BTreeMap<String, CalleeAgg>,
    /// Call kinds that have no registry question at all.
    other_kinds: BTreeMap<&'static str, usize>,
    /// `trait_root::method_name` of every `CallTarget::Indirect` site.
    indirect_families: BTreeMap<String, usize>,
}

/// Rows per table before truncation kicks in.
const RESIDUE_TABLE_ROWS: usize = 30;

/// Whether a path names `Arc` at a segment boundary.
///
/// Segment-exact rather than a substring test: the crate root Charon writes
/// varies (`alloc::sync::Arc::…` here, bare `sync::…` in the foreign-externals
/// table), and a substring match would also take `Arcane`.
fn names_arc(path: &str) -> bool {
    path.split("::").any(|seg| seg == "Arc")
}

/// Which channel answers this callee, in `translate_op`'s dispatch order.
///
/// Every way this function can be wrong points the same way — it labels a
/// resolvable path `UNREGISTERED` — save for two exceptions. A shim arm added
/// to `translate_op` and not mirrored here, a `lookup_with_leaf_match` hit, and
/// the unmodelled `register_foreign_opaque_method_externals` channel all
/// inflate the residue. So the residue printed is an upper bound, and the
/// direction of the error is the one that does not flatter route B.
///
/// The two exceptions understate it instead. One is the copied
/// [`FOREIGN_STDLIB_EXTERN_PATHS`] table, documented there. The other is
/// [`Registration::Graph`]: its test is that the callee's decl carries a body,
/// not that the body lowers, so a callee that walls reads as `graph` when
/// `function_graphs` has no entry for it. Measured on this artefact by
/// `cel_census_call_sites`: 2 of the 2066 bodied local decls refuse to lower.
fn classify_callee(
    agg: &CalleeAgg,
    local_fns: &BTreeMap<String, LocalFnFacts>,
) -> (LocalDecl, Registration) {
    let segments = &agg.segments;
    let path = segments.join("::");
    let facts = local_fns.get(&path);
    let local = match facts {
        Some(f) if f.has_body => LocalDecl::Body,
        Some(_) => LocalDecl::Bodyless,
        None => LocalDecl::Absent,
    };
    let head = segments.first().map(String::as_str).unwrap_or("");
    let leaf = segments.last().map(String::as_str).unwrap_or("");

    let shim = FRONT_END_SHIM_ARMS.iter().any(|(name, segs, args)| {
        *name == head
            && segs.is_none_or(|n| n == segments.len())
            // Every site must satisfy the arity guard, not just one.
            && args.is_none_or(|n| agg.arg_counts.iter().all(|seen| *seen == n))
    });
    // `str::<Impl>::len` is spelled with a variable prefix, so it is matched
    // here rather than listed in FOREIGN_CONTAINER_OPS.
    let str_len = segments.len() >= 3
        && segments[segments.len() - 3] == "str"
        && segments[segments.len() - 2] == "<Impl>"
        && leaf == "len";

    let registration = if leaf == "__dyn_call" {
        Registration::DynCallWall
    } else if shim {
        Registration::FrontEndShim
    } else if FOREIGN_CONTAINER_OPS.contains(&path.as_str()) || str_len {
        Registration::ForeignContainerOp
    } else if local == LocalDecl::Body {
        // Ahead of the stub arm, which is the population order the registry
        // itself runs: graphs, then unsafe stubs, then foreign externs, and
        // `register_unsafe_fn_stubs` yields to a key the graph pass already
        // holds. Being `unsafe` holds nothing back — every one of this
        // artefact's 623 unsafe decls carries a body — so a body is a `graph`
        // whether or not the stub collector also accepted it.
        Registration::Graph
    } else if facts.is_some_and(|f| f.stub_registered) {
        Registration::UnsafeStub
    } else if FOREIGN_STDLIB_EXTERN_PATHS.contains(&path.as_str()) {
        Registration::ForeignExtern
    } else if segments.len() == 1 && HOST_ENV.lookup_builtin(head).is_some() {
        Registration::HostBuiltin
    } else if segments.len() >= 2
        && HOST_ENV
            .import_module(&segments[..segments.len() - 1].join("."))
            .and_then(|m| m.module_get(leaf))
            .is_some()
    {
        Registration::HostModuleAttr
    } else if segments.len() == 2
        && head == "simple_call"
        && HOST_ENV.lookup_builtin(leaf).is_some()
    {
        Registration::HostExcClass
    } else {
        Registration::Unregistered
    };
    (local, registration)
}

fn residue_rows(tally: &ScopeTally, local_fns: &BTreeMap<String, LocalFnFacts>) -> Vec<ResidueRow> {
    let mut rows: Vec<ResidueRow> = tally
        .fn_callees
        .iter()
        .map(|(path, agg)| {
            let (local, registration) = classify_callee(agg, local_fns);
            ResidueRow {
                path: path.clone(),
                method: false,
                sites: agg.sites,
                owners: agg.owners.len(),
                local,
                registration,
            }
        })
        .chain(tally.method_callees.iter().map(|(path, agg)| ResidueRow {
            path: path.clone(),
            method: true,
            sites: agg.sites,
            owners: agg.owners.len(),
            local: LocalDecl::NotApplicable,
            registration: Registration::NotApplicable,
        }))
        .collect();
    rows.sort_by(|a, b| b.sites.cmp(&a.sites).then(a.path.cmp(&b.path)));
    rows
}

fn print_residue_row(row: &ResidueRow) {
    println!(
        "  {:>6}  {:>5}  {:<6}  {:<8}  {:<15}  {}",
        row.sites,
        row.owners,
        if row.method { "method" } else { "fn" },
        row.local.label(),
        row.registration.label(),
        row.path,
    );
}

/// Print one scope's table, truncated to [`RESIDUE_TABLE_ROWS`].
///
/// The dropped rows are itemised by verdict rather than summarised as a count:
/// a bare "… and 412 more" is indistinguishable from "… and 412 more, 300 of
/// them unregistered", and the second is the whole reading.
fn print_residue_table(label: &str, prefix: &str, role: &str, rows: &[ResidueRow]) {
    println!();
    println!("--- {label}  (bodies under `{prefix}`) — {role} ---");
    println!(
        "  {:>6}  {:>5}  {:<6}  {:<8}  {:<15}  {}",
        "sites", "from", "kind", "local", "registry", "callee"
    );
    for row in rows.iter().take(RESIDUE_TABLE_ROWS) {
        print_residue_row(row);
    }
    if rows.len() > RESIDUE_TABLE_ROWS {
        let dropped = &rows[RESIDUE_TABLE_ROWS..];
        let mut by_verdict: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
        for row in dropped {
            let e = by_verdict.entry(row.registration.label()).or_default();
            e.0 += 1;
            e.1 += row.sites;
        }
        println!(
            "  [truncated at {RESIDUE_TABLE_ROWS} rows] {} further callees, {} further sites:",
            dropped.len(),
            dropped.iter().map(|r| r.sites).sum::<usize>(),
        );
        for (verdict, (n_rows, n_sites)) in &by_verdict {
            println!("      {n_rows:>5} callees / {n_sites:>6} sites  {verdict}");
        }
    }
}

/// Per-scope totals, split so the method population is never summed into the
/// registry population.
fn print_residue_totals(label: &str, tally: &ScopeTally, rows: &[ResidueRow]) {
    let sites_of = |pred: fn(&ResidueRow) -> bool| -> (usize, usize) {
        let hit: Vec<&ResidueRow> = rows.iter().filter(|r| pred(r)).collect();
        (hit.len(), hit.iter().map(|r| r.sites).sum())
    };
    let (residue_rows_n, residue_sites) = sites_of(|r| r.registration.is_residue());
    let (method_rows, method_sites) = sites_of(|r| r.method);
    println!("  totals [{label}]:");
    println!(
        "      bodies                lowered={} walled={} bodyless={}",
        tally.lowered, tally.walled, tally.bodyless
    );
    println!(
        "      RESIDUE               {residue_rows_n:>5} callees / {residue_sites:>6} sites  (upper bound)"
    );
    println!(
        "      method targets        {method_rows:>5} callees / {method_sites:>6} sites  (classdef question, NOT residue)"
    );
    println!("      resolved, by channel:");
    let mut by_channel: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for row in rows
        .iter()
        .filter(|r| !r.registration.is_residue() && !r.method)
    {
        let e = by_channel.entry(row.registration.label()).or_default();
        e.0 += 1;
        e.1 += row.sites;
    }
    for (channel, (n_rows, n_sites)) in &by_channel {
        println!("          {n_rows:>5} callees / {n_sites:>6} sites  {channel}");
    }
    if !tally.other_kinds.is_empty() {
        println!("      call kinds with no registry question:");
        for (kind, n) in &tally.other_kinds {
            println!("          {n:>6}  {kind}");
        }
    }
    if !tally.indirect_families.is_empty() {
        println!("      vtable families reached:");
        for (family, n) in &tally.indirect_families {
            println!("          {n:>6}  {family}");
        }
    }
    if !tally.wall_classes.is_empty() {
        println!("      lowering walls:");
        for (class, n) in &tally.wall_classes {
            println!("          {n:>6}  {class}");
        }
    }
}

/// What callee-registration residue survives route B.
///
/// Route B swaps `cel::vm` off `cel::objects::Value` — 21 `Arc` fields — and
/// onto the Arc-free `cel::runtime` object family. This is a **static walk**:
/// every local body under one of [`ROUTE_B_SCOPES`] is lowered with
/// `lower_fun_decl_with_static_addrs`, and every call target it emits is
/// aggregated and classified against the channels `translate_op` resolves a
/// `CallTarget::FunctionPath` through. No pipeline runs, no portal is seeded,
/// and no stub fixpoint is iterated.
///
/// # Read `DYING` first. It is the control.
///
/// `DYING` scopes the bodies route B **deletes**, so it is where the `Arc`
/// traffic lives. If its table shows no `sync::Arc::*` rows at a plausible
/// multiplicity, this instrument is mis-scoped and **no other table here means
/// anything** — a `POST_B` residue of zero read off a broken scope is the same
/// output a correct scope would produce for a universe that really is clean.
/// The control is therefore checked before `POST_B` and `SURVIVOR` are printed
/// at all, and a failure stops the probe rather than annotating it.
///
/// # `CallTarget::Method` is not a registration question
///
/// A method target never consults `PyreCallRegistry`: it resolves through the
/// receiver's `SomeInstance.classdef` → `MethodDesc` chain, and when it fails it
/// fails as a missing classdef. It gets its own `kind` column, its registry cell
/// reads `n/a (method)`, and its sites are reported on their own line rather
/// than added to the residue. Folding the two together would move classdef
/// failures into the registration count — inflating what route B has to fix in
/// one column while leaving nothing in the other, which reads as a smaller
/// registration residue than there is.
///
/// # What the two resolution columns can and cannot say
///
/// `local` is the artefact's `FunDecl` status for the callee path, read off a
/// `body.is_some()` index rather than a successful lowering: a body Charon
/// emitted as `{"Error": …}` reads as present here.
///
/// `registry` names the channel that answers the call. Three are registry
/// passes (`graph`, `unsafe-stub`, `extern`), three are HOST_ENV layers
/// (`host-builtin`, `host-attr`, `host-exc`), and two are front-end arms that
/// run before the registry is consulted at all (`shim`, `container-op`).
/// Only `UNREGISTERED` and the named `__dyn_call` wall count as residue.
///
/// `unsafe-stub` is not read off `signature.is_unsafe`: that flag is only the
/// selector `collect_unsafe_fn_stubs_from_llbc` starts from, and it declines
/// on three further conditions. The column runs that collector over the same
/// artefact and reads back which paths it accepted, so a fn it declined stays
/// residue instead of being credited to a channel that never registered it.
///
/// Two channels are not modelled: `lookup_with_leaf_match`'s short-path
/// fallback, and `register_foreign_opaque_method_externals`, whose collector is
/// `pub(crate)`. Both can only turn a resolvable path into an `UNREGISTERED`
/// row, as can any recogniser added to `translate_op` without a row here. So
/// **every way this probe is wrong inflates the residue** — save for the two
/// exceptions named at [`classify_callee`] — and the number printed is an
/// upper bound, which is the direction that does not flatter route B.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: lowers three cel modules of the whole LLBC; use `cargo test --release --test test_cel_census`"
)]
fn cel_census_route_b_residue() {
    let Some(path) = cel_llbc_path() else {
        skip_note();
        return;
    };
    let bytes = std::fs::read(&path).expect("read cel llbc");
    let hash = content_hash(&bytes);
    let llbc = Llbc::load(&path).expect("load cel llbc");

    // Path → merged decl facts. `body.is_some()` rather than
    // `unstructured()`: the latter deserializes every body in the artefact,
    // and this index only needs presence. Monomorphization siblings share a
    // `name_path`, so the flags are OR-merged and the collision count is
    // reported below rather than absorbed.
    let mut local_fns: BTreeMap<String, LocalFnFacts> = BTreeMap::new();
    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        let facts = local_fns.entry(fd.item_meta.name_path()).or_default();
        facts.has_body |= fd.body.is_some();
        facts.siblings += 1;
    }
    // Ask the collector which paths the stub channel holds. `signature.
    // is_unsafe` is only its selector: it also declines a reference return, a
    // return whose token `residual_return_shell` cannot model, and a multiword
    // `ref` return. A declined fn is registered by nobody, so reading the
    // selector alone would print `unsafe-stub` over residue.
    for (segments, _, _) in collect_unsafe_fn_stubs_from_llbc(&llbc) {
        if let Some(facts) = local_fns.get_mut(&segments.join("::")) {
            facts.stub_registered = true;
        }
    }
    let colliding = local_fns.values().filter(|f| f.siblings > 1).count();

    let mut tallies: BTreeMap<&'static str, ScopeTally> = BTreeMap::new();
    for fd in llbc.iter_local_fns() {
        if fd.is_global_initializer.is_some() {
            continue;
        }
        let owner = fd.item_meta.name_path();
        let Some((label, _, _)) = ROUTE_B_SCOPES
            .iter()
            .find(|(_, prefix, _)| owner.starts_with(prefix))
        else {
            continue;
        };
        let tally = tallies.entry(label).or_default();
        if fd.unstructured().is_none() {
            tally.bodyless += 1;
            continue;
        }
        // `CEL_CLASS_ADDRS` rather than a bare `lower_fun_decl`: a class
        // singleton read (`&CEL_INT_CLASS`) lowers to a
        // `__pyre_cast_instance[<root>]` narrow when its address is supplied
        // and to a residual 0-arg call when it is not (`mir.rs:6072`,
        // `pytype_static_addr`). With the default empty table the twelve cel
        // class statics would each read as an unregistered callee — residue
        // about the missing address, not about route B. The table is passed to
        // all three scopes even though only `cel::runtime` allocates
        // class-headed objects, so the tables stay comparable.
        let graph = match lower_fun_decl_with_static_addrs(
            &llbc,
            fd,
            HostStaticAddrs {
                pytypes: CEL_CLASS_ADDRS,
                ..Default::default()
            },
        ) {
            Ok(graph) => graph,
            Err(err) => {
                // Leading clause only: the tail carries block and local
                // numbers, which would make every wall its own class.
                let class = match &err {
                    LowerError::FunctionNotFound(_) => "FunctionNotFound".to_string(),
                    LowerError::Schema(_) => "Schema".to_string(),
                    LowerError::Unsupported(msg) => {
                        format!("Unsupported: {}", msg.chars().take(70).collect::<String>())
                    }
                };
                tally.walled += 1;
                *tally.wall_classes.entry(class).or_default() += 1;
                continue;
            }
        };
        tally.lowered += 1;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, args, .. } => match target {
                        CallTarget::FunctionPath { segments } => {
                            // A `__fn_const` head is the marker for a call
                            // through a function constant, not part of the
                            // callee's identity; the registry is keyed on the
                            // path underneath it.
                            let segs = majit_translate::model::fn_const_segments(target)
                                .unwrap_or(segments.as_slice());
                            let agg = tally.fn_callees.entry(segs.join("::")).or_default();
                            agg.sites += 1;
                            agg.owners.insert(owner.clone());
                            agg.segments = segs.to_vec();
                            agg.arg_counts.insert(args.len());
                        }
                        CallTarget::Method {
                            name,
                            receiver_root,
                            ..
                        } => {
                            // `resolved_path` is stamped at codewriter time and
                            // this walk never gets there, so the receiver root
                            // is all the identity a static row can carry.
                            let key = format!(
                                "{}::{name}",
                                receiver_root.as_deref().unwrap_or("<unresolved-receiver>")
                            );
                            let agg = tally.method_callees.entry(key).or_default();
                            agg.sites += 1;
                            agg.owners.insert(owner.clone());
                            agg.arg_counts.insert(args.len());
                        }
                        CallTarget::Indirect {
                            trait_root,
                            method_name,
                        } => {
                            bump(&mut tally.other_kinds, "Indirect (vtable arm)");
                            *tally
                                .indirect_families
                                .entry(format!("{trait_root}::{method_name}"))
                                .or_default() += 1;
                        }
                        CallTarget::SyntheticTransparentCtor { .. } => {
                            bump(&mut tally.other_kinds, "SyntheticTransparentCtor")
                        }
                        CallTarget::UnsupportedExpr => bump(
                            &mut tally.other_kinds,
                            "UnsupportedExpr (target did not lower)",
                        ),
                    },
                    OpKind::IndirectCall { graphs, .. } => match graphs {
                        Some(c) if !c.is_empty() => {
                            bump(&mut tally.other_kinds, "indirect-call, candidates present")
                        }
                        Some(_) => bump(&mut tally.other_kinds, "indirect-call, empty candidates"),
                        None => bump(&mut tally.other_kinds, "indirect-call, graphs=None"),
                    },
                    _ => {}
                }
            }
        }
    }

    println!("=== cel route-B callee-registration residue ===");
    println!(
        "artefact {} len={} fnv1a64={hash:016x}",
        path.display(),
        bytes.len()
    );
    println!(
        "local fns indexed {} paths, {colliding} of them shared by >1 monomorphization",
        local_fns.len()
    );
    println!(
        "scope predicate: the OWNING body's module. Rows are the call targets those bodies emit."
    );
    println!(
        "READ `DYING` FIRST — it is the control. It scopes the bodies route B deletes, so it is"
    );
    println!(
        "where the Arc traffic must appear. No Arc rows there means the scope is wrong, and the"
    );
    println!("other two tables are then void rather than merely suspect.");
    println!(
        "`CallTarget::Method` rows carry `n/a (method)`: a method target resolves through the"
    );
    println!("receiver's classdef and never consults PyreCallRegistry.");

    let empty = ScopeTally::default();
    let dying = tallies.get("DYING").unwrap_or(&empty);
    let dying_rows = residue_rows(dying, &local_fns);
    let (_, dying_prefix, dying_role) = ROUTE_B_SCOPES[0];
    print_residue_table("DYING", dying_prefix, dying_role, &dying_rows);
    print_residue_totals("DYING", dying, &dying_rows);

    let arc_rows: Vec<&ResidueRow> = dying_rows.iter().filter(|r| names_arc(&r.path)).collect();
    let arc_sites: usize = arc_rows.iter().map(|r| r.sites).sum();
    println!();
    println!("--- CONTROL: Arc-naming rows in DYING (printed in full, never truncated) ---");
    for row in &arc_rows {
        print_residue_row(row);
    }
    println!(
        "  control verdict: {} distinct Arc callees / {arc_sites} sites — {}",
        arc_rows.len(),
        if arc_rows.is_empty() { "FAIL" } else { "PASS" }
    );
    assert!(
        !arc_rows.is_empty(),
        "control FAILED: no `Arc` call target under `{dying_prefix}`. The value layer route B \
         deletes is built on 21 Arc fields, so a zero here is a mis-scoped instrument, not a \
         clean universe. POST_B and SURVIVOR are not printed — their numbers would be \
         unreadable."
    );

    for (label, prefix, role) in &ROUTE_B_SCOPES[1..] {
        let tally = tallies.get(label).unwrap_or(&empty);
        let rows = residue_rows(tally, &local_fns);
        print_residue_table(label, prefix, role, &rows);
        print_residue_totals(label, tally, &rows);
    }
}
