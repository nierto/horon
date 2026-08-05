//! The committed conformance corpus is in sync with the code that produced it.
//!
//! `tests/corpus/` is the oracle every independent `.htt` implementation is
//! graded against, so it has to be trustworthy in two ways:
//!
//!   1. **Reproducible** — regenerating produces byte-identical files, which
//!      is the same no-nondeterministic-leaks property `determinism.rs`
//!      asserts for the format at large.
//!   2. **Current** — the expected-state dumps still describe what this
//!      implementation recovers. A reader-behaviour change that silently
//!      shifts the oracle would let every wrapper inherit the drift.
//!
//! Regenerating into a scratch directory and comparing bytes covers both. If
//! this fails after a deliberate format or reader change, rerun
//! `cargo run --example gen_corpus` and commit the corpus in the same commit.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

fn committed_corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("corpus")
}

fn file_names(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("read corpus directory")
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .collect()
}

#[test]
fn corpus_regenerates_byte_identically() {
    let committed = committed_corpus();
    assert!(
        committed.is_dir(),
        "conformance corpus missing — run `cargo run --example gen_corpus`"
    );

    let scratch = TempDir::new().unwrap();
    let status = Command::new(env!("CARGO"))
        .args(["run", "--quiet", "--example", "gen_corpus", "--"])
        .arg(scratch.path())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .status()
        .expect("run the corpus generator");
    assert!(status.success(), "corpus generator failed");

    let fresh_names = file_names(scratch.path());
    let committed_names = file_names(&committed);
    assert_eq!(
        committed_names, fresh_names,
        "corpus file set differs — regenerate and commit `tests/corpus/`"
    );

    for name in &committed_names {
        let a = std::fs::read(committed.join(name)).unwrap();
        let b = std::fs::read(scratch.path().join(name)).unwrap();
        assert_eq!(
            a,
            b,
            "corpus artifact `{}` differs from freshly generated output — \
             rerun `cargo run --example gen_corpus` and commit the result \
             (committed {} bytes, generated {} bytes)",
            name,
            a.len(),
            b.len()
        );
    }
}

#[test]
fn corpus_covers_the_reader_feature_matrix() {
    // Guards against a case being dropped from the generator: every format
    // surface a wrapper has to implement should be represented by at least
    // one committed file.
    let names = file_names(&committed_corpus());
    let has = |needle: &str| names.iter().any(|n| n.contains(needle) && n.ends_with(".htt"));

    for required in [
        "empty",
        "structural_only",
        "uncompressed",
        "zstd",
        "wal_live",
        "wal_batched",
        "deletes_and_updates",
        "unicode_keys",
        "deep_hierarchy",
        "gacl",
        "meaning_addressed",
        "quantized",
        "epochs",
        "history_sidecar",
        "torn_tail",
    ] {
        assert!(has(required), "conformance corpus is missing a `{}` case", required);
    }

    // Every .htt must ship the expected-state dump that grades it.
    for name in names.iter().filter(|n| n.ends_with(".htt")) {
        let expected = name.replace(".htt", ".expected.json");
        assert!(
            names.contains(&expected),
            "corpus file `{}` has no `{}` alongside it",
            name,
            expected
        );
    }
}
