//! Vocabulary oracle over the user-facing surface.
//!
//! A green test suite is weak evidence of rename completeness: the old
//! names still compile fine, and docs that quietly tell users to import
//! the engine crate still pass. The oracle is a vocabulary scan — the
//! README, docs, examples, and rendered API surface must never carry the
//! pre-rename product names, and must never require the engine crate's
//! name.

use std::fs;
use std::path::{Path, PathBuf};

/// Terms that must not appear anywhere on the user-facing surface, with the
/// reason each is forbidden.
const FORBIDDEN: &[(&str, &str)] = &[
    (
        "horon_engine::",
        "users must never be told to import the engine crate",
    ),
    ("g_hyper", "pre-rename engine crate name"),
    ("g_file", "pre-rename crate name"),
    ("GeoFile", "pre-rename handle type name"),
    ("gHyper", "pre-rename engine product name"),
    ("gFile", "pre-rename product name"),
];

/// Historical statements that stay true under their original names — release
/// records are facts about the past, and rewriting them to satisfy this scan
/// would falsify history. Each entry is (file suffix, exact line substring).
const PRESERVED_HISTORY: &[(&str, &str)] = &[
    (
        "docs/QUANTIZED_SEMANTIC.md",
        "shipped in format v4 (gFile v0.7.0)",
    ),
    ("docs/HTT_FORMAT.md", "*Implemented (gFile v0.5.0):*"),
];

/// The user-facing surface: everything a user reads before (or instead of)
/// the source code — plus `src`, which carries the rendered API docs, the
/// doc-tests, and the CLI help text.
const SURFACE: &[&str] = &["README.md", "docs", "examples", "src"];

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) {
    if root.is_file() {
        out.push(root.to_path_buf());
        return;
    }
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        collect_files(&path, out);
    }
}

#[test]
fn user_facing_surface_never_carries_pre_rename_vocabulary() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();

    for surface in SURFACE {
        let mut files = Vec::new();
        collect_files(&manifest_dir.join(surface), &mut files);
        for file in files {
            let bytes = fs::read(&file).unwrap_or_default();
            let text = String::from_utf8_lossy(&bytes);
            // Implementation code under src/ may legitimately name the
            // engine crate — it is a dependency. What must stay clean is
            // the text a user sees: rendered doc comments (and, via the
            // other surfaces, README, docs, and examples).
            let doc_comments_only =
                *surface == "src" && file.extension().is_some_and(|e| e == "rs");
            for (lineno, line) in text.lines().enumerate() {
                if doc_comments_only {
                    let t = line.trim_start();
                    if !t.starts_with("///") && !t.starts_with("//!") {
                        continue;
                    }
                }
                let preserved = PRESERVED_HISTORY.iter().any(|(suffix, substring)| {
                    file.ends_with(suffix) && line.contains(substring)
                });
                if preserved {
                    continue;
                }
                for (term, why) in FORBIDDEN {
                    if line.contains(term) {
                        violations.push(format!(
                            "{}:{}: contains `{}` ({})\n    {}",
                            file.strip_prefix(&manifest_dir).unwrap_or(&file).display(),
                            lineno + 1,
                            term,
                            why,
                            line.trim()
                        ));
                    }
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "forbidden vocabulary on the user-facing surface:\n{}",
        violations.join("\n")
    );
}
