//! Placement state must survive the storage boundary.
//!
//! The all-zero semantic tail is the on-disk encoding of "not set", which
//! forces two API rules: an explicit placement of all zeros is rejected
//! (it would silently unplace on reload), and clearing a placement is its
//! own operation whose result is identical in memory, after WAL replay,
//! and after compaction.
//!
//! Also covers the WAL-completeness rule: implicit ancestor creation is
//! logged, so a replica built from the WAL alone matches the primary.

use std::path::PathBuf;

use horon::{Horon, HoronConfig, WalTail};
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn cfg(dims: u8) -> HoronConfig {
    HoronConfig {
        semantic_dims: dims,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

fn coords_with(dims: u8, dim: usize, raw: i128) -> Vec<u8> {
    let mut coords = vec![0u8; dims as usize * 16];
    coords[dim * 16..dim * 16 + 16].copy_from_slice(&raw.to_le_bytes());
    coords
}

// ---------------------------------------------------------------------------
// Rejection of the unrepresentable placement
// ---------------------------------------------------------------------------

#[test]
fn all_zero_placement_is_rejected() {
    let dims: u8 = 20;
    let gf = Horon::open_with_config(temp_path(), cfg(dims)).unwrap();
    gf.put("/n", b"data").unwrap();

    let zeros = vec![0u8; dims as usize * 16];
    let err = gf.set_semantic("/n", zeros).unwrap_err();
    assert!(
        err.to_string().contains("clear_semantic"),
        "rejection must point at the explicit alternative, got: {}",
        err
    );

    // A short vector zero-extends — an empty one is therefore all-zero too.
    assert!(gf.set_semantic("/n", Vec::new()).is_err());

    // Rejection must not have perturbed state: the node stays unplaced.
    assert!(gf.get_semantic("/n").unwrap().is_empty());
}

#[test]
fn near_origin_placement_is_accepted() {
    let dims: u8 = 20;
    let gf = Horon::open_with_config(temp_path(), cfg(dims)).unwrap();
    gf.put("/n", b"data").unwrap();

    // One epsilon in one dimension is a legal placement.
    let coords = coords_with(dims, 16, 1);
    gf.set_semantic("/n", coords.clone()).unwrap();
    assert_eq!(gf.get_semantic("/n").unwrap(), coords);
}

// ---------------------------------------------------------------------------
// clear_semantic: one meaning, every path
// ---------------------------------------------------------------------------

#[test]
fn clear_semantic_unplaces_in_memory() {
    let dims: u8 = 20;
    let gf = Horon::open_with_config(temp_path(), cfg(dims)).unwrap();
    gf.put("/a/placed", b"data").unwrap();
    gf.set_semantic("/a/placed", coords_with(dims, 16, 1 << 60)).unwrap();

    let query = coords_with(dims, 16, 1 << 60);
    let before = gf.nearest_semantic(&query, 5, 16..dims as usize).unwrap();
    assert!(
        before.iter().any(|(k, _)| k == "/a/placed"),
        "placed node must be visible to semantic queries"
    );

    gf.clear_semantic("/a/placed").unwrap();

    assert!(gf.get_semantic("/a/placed").unwrap().is_empty(), "cleared in memory");
    let after = gf.nearest_semantic(&query, 5, 16..dims as usize).unwrap();
    assert!(
        !after.iter().any(|(k, _)| k == "/a/placed"),
        "cleared node must not appear in semantic queries"
    );
    assert_eq!(gf.get("/a/placed").unwrap(), b"data", "data untouched by clear");
}

#[test]
fn cleared_state_survives_wal_replay_and_compaction_identically() {
    let dims: u8 = 20;
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, cfg(dims)).unwrap();
        gf.put("/a/cleared", b"one").unwrap();
        gf.set_semantic("/a/cleared", coords_with(dims, 16, 1 << 60)).unwrap();
        gf.clear_semantic("/a/cleared").unwrap();
        gf.put("/a/kept", b"two").unwrap();
        gf.set_semantic("/a/kept", coords_with(dims, 17, 1 << 59)).unwrap();
        gf.flush().unwrap();
    }

    // Path 1: reload replays the live WAL (place + clear as entries).
    {
        let gf = Horon::open_with_config(&path, cfg(dims)).unwrap();
        assert!(
            gf.get_semantic("/a/cleared").unwrap().is_empty(),
            "WAL replay must reproduce the cleared state"
        );
        assert_eq!(gf.get_semantic("/a/kept").unwrap(), coords_with(dims, 17, 1 << 59));
        gf.compact().unwrap();
    }

    // Path 2: reload from the compacted snapshot.
    {
        let gf = Horon::open_with_config(&path, cfg(dims)).unwrap();
        assert!(
            gf.get_semantic("/a/cleared").unwrap().is_empty(),
            "snapshot load must agree with WAL replay about the cleared state"
        );
        assert_eq!(gf.get_semantic("/a/kept").unwrap(), coords_with(dims, 17, 1 << 59));
    }
}

#[test]
fn clear_semantic_on_quantized_file() {
    let dims: u8 = 40;
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, HoronConfig {
            semantic_dims: dims,
            compression: false,
            auto_compact_threshold: 0,
            quantized_semantic: true,
            ..Default::default()
        })
        .unwrap();
        gf.put("/q", b"data").unwrap();
        let err = gf.set_semantic("/q", vec![0u8; dims as usize * 16]);
        assert!(err.is_err(), "all-zero rejection applies to quantized files too");

        gf.set_semantic("/q", coords_with(dims, 20, 1 << 55)).unwrap();
        gf.clear_semantic("/q").unwrap();
        gf.flush().unwrap();
    }
    let gf = Horon::open_with_config(&path, cfg(dims)).unwrap();
    assert!(gf.get_semantic("/q").unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// WAL completeness: ancestors are logged
// ---------------------------------------------------------------------------

#[test]
fn replica_from_wal_tail_matches_primary() {
    let dims: u8 = 20;
    let gf = Horon::open_with_config(temp_path(), cfg(dims)).unwrap();
    gf.put("/deep/a/b/c/leaf", b"payload").unwrap();
    gf.put("/deep/a/b/c/leaf2", b"payload2").unwrap();
    gf.put("/other/x", b"y").unwrap();
    gf.flush().unwrap();

    let mut primary: Vec<String> = gf.list("/").unwrap();
    primary.sort();

    let mut replica: Vec<String> = match gf.wal_entries_since(1).unwrap() {
        WalTail::Entries(entries) => {
            let mut keys: Vec<String> = entries.iter().map(|e| e.key.clone()).collect();
            keys.sort();
            keys.dedup();
            keys
        }
        WalTail::SnapshotRequired { .. } => panic!("no compaction happened"),
    };
    replica.sort();

    assert_eq!(
        primary, replica,
        "every node the primary has must be described by the WAL"
    );
}

#[test]
fn ancestor_entries_precede_their_children_and_replay_cleanly() {
    let dims: u8 = 20;
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, cfg(dims)).unwrap();
        gf.put("/p/q/r/leaf", b"v1").unwrap();
        // Second put under the same subtree must NOT re-log the ancestors.
        gf.put("/p/q/r/leaf2", b"v2").unwrap();
        gf.flush().unwrap();

        match gf.wal_entries_since(1).unwrap() {
            WalTail::Entries(entries) => {
                let keys: Vec<&str> = entries.iter().map(|e| e.key.as_str()).collect();
                assert_eq!(
                    keys,
                    vec!["/p", "/p/q", "/p/q/r", "/p/q/r/leaf", "/p/q/r/leaf2"],
                    "ancestors once, parents-first, then leaves in order"
                );
            }
            WalTail::SnapshotRequired { .. } => panic!("no compaction happened"),
        }
    }

    // The logged ancestors must replay without error and reproduce the tree.
    let gf = Horon::open_with_config(&path, cfg(dims)).unwrap();
    let mut keys = gf.list("/").unwrap();
    keys.sort();
    assert_eq!(keys, vec!["/p", "/p/q", "/p/q/r", "/p/q/r/leaf", "/p/q/r/leaf2"]);
    assert_eq!(
        gf.get_meta("/p").unwrap().get("content_type").map(String::as_str),
        Some("application/x-directory"),
        "replayed ancestor carries the directory content type"
    );
}
