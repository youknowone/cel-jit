//! Emits `CEL_MAJIT_PROVENANCE`: which majit actually got compiled in.
//!
//! `cel/Cargo.toml` pins the majit crates to a git revision, and
//! `../.cargo/config.toml` — untracked, gitignored, LOCAL ONLY — `[patch]`es
//! that pin to the enclosing pyre worktree. Both resolve silently, so a JIT
//! measurement taken under the pin and one taken under the patch produce the
//! same output text while describing majits days apart. Every consumer of
//! `@@@STATS` was ambiguous between them.
//!
//! # Why this reads `Cargo.lock` and not the config file
//!
//! "`.cargo/config.toml` exists and mentions majit, therefore the patch is in
//! effect" is an inference, and inference is what produced the defect this
//! guards against. `Cargo.lock` is cargo's own *recorded resolution*: the
//! `majit-metainterp` package entry carries `source = "git+…#<sha>"` when the
//! pin won and carries **no `source` key at all** when it resolved to a path,
//! which is what a `[patch]` redirect produces. That is an observation.
//!
//! The config file is then read only for *where* the patch pointed — the same
//! input cargo itself read, not a guess about what cargo did.
//!
//! # Why the dirty flag is content-based
//!
//! `git status --porcelain` is a stat-cache test, not a content test, and it
//! has been observed in this worktree reporting `M` for files whose `git diff`
//! was empty. It also resolves upward: `majit/` is a subdirectory of the
//! `pyre-wasmi` repo, not a repo of its own, so an unscoped `status` reports
//! the whole tree — including permanently untracked siblings — and the flag
//! would be stuck on forever. A flag that is always set carries zero bits.
//!
//! So: `git diff --quiet HEAD -- .` scoped to the majit tree. `+dirty` is the
//! common case, not an edge case, which is exactly why its false positives
//! would go unnoticed.
//!
//! Anything undeterminable emits `unknown`. It never emits a clean-looking
//! token it cannot stand behind — a provenance string that overstates is worse
//! than none, because it gets cited.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().unwrap_or(&manifest_dir).to_path_buf();

    let lock_path = workspace_root.join("Cargo.lock");
    let config_path = workspace_root.join(".cargo/config.toml");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!("cargo:rerun-if-changed={}", config_path.display());
    println!("cargo:rerun-if-changed=Cargo.toml");

    let token = resolve(&lock_path, &config_path).unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=CEL_MAJIT_PROVENANCE={token}");
}

fn resolve(lock_path: &Path, config_path: &Path) -> Option<String> {
    match majit_source(&std::fs::read_to_string(lock_path).ok()?)? {
        // The pin won. The revision is the fragment after `#`.
        Source::Git(url) => {
            let rev = url.rsplit_once('#').map(|(_, rev)| rev).unwrap_or(&url);
            Some(format!("pin:{}", short(rev)))
        }
        // No `source` key: cargo resolved majit to a path, i.e. the `[patch]`
        // redirect won. Read the config only now, for the location.
        Source::Path => {
            let dir = patched_majit_dir(config_path)?;
            // Watch the whole majit tree: without this the token caches, and a
            // cached provenance token asserts a version the binary may not
            // have — this defect one level up, and harder to see.
            println!("cargo:rerun-if-changed={}", dir.display());
            let head = git(&dir, &["rev-parse", "--short", "HEAD"])?;
            let clean = Command::new("git")
                .arg("-C")
                .arg(&dir)
                .args(["diff", "--quiet", "HEAD", "--", "."])
                .status()
                .ok()?
                .success();
            Some(format!("local:{head}{}", if clean { "" } else { "+dirty" }))
        }
    }
}

enum Source {
    Git(String),
    Path,
}

/// The `source` of `Cargo.lock`'s `majit-metainterp` package entry.
fn majit_source(lock: &str) -> Option<Source> {
    let mut in_entry = false;
    for line in lock.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            // A `[[package]]` boundary reached while inside the entry means
            // majit-metainterp declared no `source` — a path resolution.
            if in_entry {
                return Some(Source::Path);
            }
            continue;
        }
        if line == r#"name = "majit-metainterp""# {
            in_entry = true;
        } else if in_entry {
            if let Some(rest) = line.strip_prefix("source = ") {
                return Some(Source::Git(rest.trim_matches('"').to_string()));
            }
        }
    }
    in_entry.then_some(Source::Path)
}

/// The majit tree the `[patch]` table points at — the parent of the patched
/// `majit-metainterp` path, since all three majit crates are redirected
/// together and the dirty flag must cover every one of them.
fn patched_majit_dir(config_path: &Path) -> Option<PathBuf> {
    let config = std::fs::read_to_string(config_path).ok()?;
    let line = config
        .lines()
        .find(|l| l.trim_start().starts_with("majit-metainterp"))?;
    let (_, rest) = line.split_once("path")?;
    let raw = rest.split('"').nth(1)?;
    let dir = config_path.parent()?.parent()?.join(raw);
    Some(dir.parent()?.to_path_buf())
}

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn short(rev: &str) -> String {
    rev.chars().take(8).collect()
}
