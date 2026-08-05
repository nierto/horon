//! Partial mmap reads + the meaning-addressed layout (format v3).
//!
//! The thesis under test: **the tree IS the data, the data IS the location.**
//! In a v3 file, a node's byte offset is a function of its semantic
//! coordinates, so a semantic query touches only the byte-neighborhood of
//! the query point — measured here by `last_semantic_scan_count()`.
//!
//! Run: GMATH_PROFILE=embedded cargo test --test meaning_addressed

use std::collections::HashMap;
use std::path::PathBuf;

use horon::{Horon, HoronConfig, HoronError};
use horon_engine::Store;
use tempfile::NamedTempFile;

const SEM_DIMS: u8 = 20; // 16 reserved + 4 user

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn plain(partial: bool) -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: SEM_DIMS,
        compression: false,
        auto_compact_threshold: 0,
        partial_reads: partial,
        ..Default::default()
    }
}

fn ma_config(partial: bool) -> HoronConfig {
    HoronConfig {
        meaning_addressed: true,
        semantic_bounds: (0.0, 1000.0),
        ..plain(partial)
    }
}

/// Encode user-dim values (dims 16..) as a full semantic vector.
fn coords(user_vals: &[f64]) -> Vec<u8> {
    use g_math::fixed_point::FixedPoint;
    let mut out = vec![0u8; SEM_DIMS as usize * 16];
    for (d, v) in user_vals.iter().enumerate() {
        let off = (16 + d) * 16;
        out[off..off + 16].copy_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out
}

// ===========================================================================
// partial mode parity with full mode
// ===========================================================================

fn build_v2_file(path: &std::path::Path) {
    let gf = Horon::open_with_config(path, plain(false)).unwrap();
    for i in 0..30 {
        let key = format!("/data/g{}/n{:02}", i % 3, i);
        gf.put(&key, format!("value-{}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "tag", &format!("t{}", i % 7)).unwrap();
        gf.set_semantic(&key, coords(&[i as f64 * 10.0, 500.0 - i as f64, 3.0, 4.0])).unwrap();
    }
    gf.compact().unwrap();
    // Some post-compaction WAL entries too.
    for i in 30..36 {
        gf.put(&format!("/data/wal/n{:02}", i), format!("wal-{}", i).as_bytes()).unwrap();
    }
    drop(gf);
}

#[test]
fn partial_mode_matches_full_mode() {
    let path = temp_path();
    build_v2_file(&path);
    let p_full = temp_path();
    let p_part = temp_path();
    std::fs::copy(&path, &p_full).unwrap();
    std::fs::copy(&path, &p_part).unwrap();

    let full = Horon::open_with_config(&p_full, plain(false)).unwrap();
    let part = Horon::open_with_config(&p_part, plain(true)).unwrap();

    assert_eq!(full.len(), part.len(), "len differs");

    let keys = full.list("/").unwrap();
    let pkeys = part.list("/").unwrap();
    assert_eq!(
        keys.iter().collect::<std::collections::BTreeSet<_>>(),
        pkeys.iter().collect::<std::collections::BTreeSet<_>>(),
        "key sets differ"
    );

    for key in &keys {
        assert_eq!(full.get(key).unwrap(), part.get(key).unwrap(), "data: {}", key);
        assert_eq!(
            full.get_semantic(key).unwrap(),
            part.get_semantic(key).unwrap(),
            "semantic: {}", key
        );
        // Compare user metadata only (full mode synthesizes fresh timestamps).
        let fm: HashMap<String, String> = full.get_meta(key).unwrap();
        let pm: HashMap<String, String> = part.get_meta(key).unwrap();
        assert_eq!(fm.get("tag"), pm.get("tag"), "meta tag: {}", key);
        assert_eq!(full.exists(key), part.exists(key));
    }

    assert_eq!(
        full.children("/data").unwrap().into_iter().collect::<std::collections::BTreeSet<_>>(),
        part.children("/data").unwrap().into_iter().collect::<std::collections::BTreeSet<_>>(),
    );
}

#[test]
fn partial_mode_writes_survive_full_reopen() {
    let path = temp_path();
    build_v2_file(&path);

    {
        let part = Horon::open_with_config(&path, plain(true)).unwrap();
        part.put("/data/new/fresh", b"born in partial mode").unwrap();
        part.put("/data/g0/n00", b"updated in partial mode").unwrap(); // snapshot-resident update
        part.set_meta("/data/g1/n01", "extra", "promoted").unwrap();   // promote + meta
        part.remove("/data/g2/n02").unwrap();                          // tombstone

        // Visible through the partial handle immediately:
        assert_eq!(part.get("/data/g0/n00").unwrap(), b"updated in partial mode");
        assert!(!part.exists("/data/g2/n02"));
        assert_eq!(part.get_meta("/data/g1/n01").unwrap().get("extra").map(String::as_str), Some("promoted"));
        // The update must have preserved the node's snapshot metadata/semantics:
        assert_eq!(part.get_meta("/data/g0/n00").unwrap().get("tag").map(String::as_str), Some("t0"));
        assert!(!part.get_semantic("/data/g0/n00").unwrap().is_empty());
    }

    // Full reopen replays the same WAL over the old snapshot.
    let full = Horon::open_with_config(&path, plain(false)).unwrap();
    assert_eq!(full.get("/data/new/fresh").unwrap(), b"born in partial mode");
    assert_eq!(full.get("/data/g0/n00").unwrap(), b"updated in partial mode");
    assert!(!full.exists("/data/g2/n02"));
    assert_eq!(full.get_meta("/data/g1/n01").unwrap().get("extra").map(String::as_str), Some("promoted"));
}

#[test]
fn partial_mode_compaction_remaps_and_stays_correct() {
    let path = temp_path();
    build_v2_file(&path);

    let part = Horon::open_with_config(&path, plain(true)).unwrap();
    part.put("/data/new/one", b"n1").unwrap();
    part.remove("/data/g0/n03").unwrap();
    let len_before = part.len();

    part.compact().unwrap();

    // Same handle keeps serving after the remap.
    assert_eq!(part.len(), len_before, "len changed across compaction");
    assert_eq!(part.get("/data/new/one").unwrap(), b"n1");
    assert!(!part.exists("/data/g0/n03"));
    assert_eq!(part.get("/data/g1/n01").unwrap(), b"value-1");
    assert_eq!(part.get_meta("/data/g1/n04").unwrap().get("tag").map(String::as_str), Some("t4"));

    // Post-compaction writes + reads still coherent.
    part.put("/data/new/two", b"n2").unwrap();
    assert_eq!(part.get("/data/new/two").unwrap(), b"n2");
    drop(part);

    let full = Horon::open_with_config(&path, plain(false)).unwrap();
    assert_eq!(full.get("/data/new/one").unwrap(), b"n1");
    assert_eq!(full.get("/data/new/two").unwrap(), b"n2");
    assert!(!full.exists("/data/g0/n03"));
}

// ===========================================================================
// meaning-addressed layout: the thesis in action
// ===========================================================================

/// Three well-separated clusters in user-dim space, many nodes each.
fn build_ma_file(path: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let gf = Horon::open_with_config(path, ma_config(false)).unwrap();
    let mut all = Vec::new();
    // 500 nodes per cluster, grouped 50 per parent: the engine's golden-angle
    // sibling placement loses angular separation at extreme fan-out (1000+
    // siblings under one parent quantize to colliding signatures — the engine
    // a known fan-out limitation), so keep a realistic tree shape.
    let centers = [(100.0, 100.0), (500.0, 500.0), (900.0, 200.0)];
    for (c, (cx, cy)) in centers.iter().enumerate() {
        for i in 0..500 {
            let key = format!("/c{}/g{}/node_{:03}", c, i / 50, i);
            let x = cx + (i % 25) as f64;
            let y = cy + (i / 25) as f64;
            let sem = coords(&[x, y, 50.0, 50.0]);
            gf.put(&key, format!("payload {} {}", c, i).as_bytes()).unwrap();
            gf.set_semantic(&key, sem.clone()).unwrap();
            all.push((key, sem));
        }
    }
    gf.compact().unwrap(); // writes the v3 pure-Hilbert snapshot
    drop(gf);
    all
}

#[test]
fn ma_file_is_v3_and_full_mode_reads_it() {
    let path = temp_path();
    let all = build_ma_file(&path);

    assert_eq!(std::fs::read(&path).unwrap()[4], 3, "meaning-addressed file must be format v3");

    // Full-geometry open: depth-sorted replay must reconstruct everything.
    let full = Horon::open_with_config(&path, plain(false)).unwrap();
    for (key, sem) in &all {
        assert!(full.exists(key), "missing {}", key);
        assert_eq!(&full.get_semantic(key).unwrap(), sem, "semantic {}", key);
    }
}

#[test]
fn thesis_semantic_query_reads_only_a_byte_neighborhood() {
    let path = temp_path();
    let all = build_ma_file(&path);
    let total = all.len();

    let part = Horon::open_with_config(&path, ma_config(true)).unwrap();

    // Query near cluster 1's center.
    let query = coords(&[503.0, 504.0, 50.0, 50.0]);
    let k = 10;
    let got = part.nearest_semantic(&query, k, 16..18).unwrap();

    // Ground truth by brute force over what we inserted.
    let mut expected: Vec<(String, g_math::fixed_point::FixedPoint)> = all
        .iter()
        .map(|(key, sem)| (key.clone(), Store::semantic_distance(&query, sem, 16..18)))
        .collect();
    expected.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    expected.truncate(k);

    // The cluster grid produces distance TIES; ordering among equal
    // distances is unspecified. Compare the distance sequences, and verify
    // every returned key's reported distance is its true distance.
    let truth: HashMap<&String, g_math::fixed_point::FixedPoint> = all
        .iter()
        .map(|(key, sem)| (key, Store::semantic_distance(&query, sem, 16..18)))
        .collect();
    assert_eq!(got.len(), k);
    for (i, (key, dist)) in got.iter().enumerate() {
        assert!(
            (dist.to_f64() - expected[i].1.to_f64()).abs() < 1e-9,
            "distance sequence diverges at {}: got {} expected {}",
            i, dist, expected[i].1
        );
        assert!(
            (truth[key].to_f64() - dist.to_f64()).abs() < 1e-9,
            "reported distance for {} is wrong", key
        );
    }

    // THE THESIS METRIC: the query decoded only a neighborhood, not the file.
    let scanned = part.last_semantic_scan_count().unwrap();
    println!("=== MEANING-ADDRESSED THESIS ===");
    println!("  nodes in file:      {}", total);
    println!("  entries examined:   {}", scanned);
    println!("  fraction touched:   {:.1}%", 100.0 * scanned as f64 / total as f64);
    assert!(
        scanned < total / 5,
        "semantic query should touch a byte-neighborhood, not the whole file \
         (scanned {} of {})",
        scanned, total
    );

    // All three clusters, same story:
    for (probe, label) in [
        (coords(&[105.0, 103.0, 50.0, 50.0]), "cluster 0"),
        (coords(&[902.0, 205.0, 50.0, 50.0]), "cluster 2"),
    ] {
        let r = part.nearest_semantic(&probe, 5, 16..18).unwrap();
        assert_eq!(r.len(), 5, "{}", label);
        let scanned = part.last_semantic_scan_count().unwrap();
        assert!(scanned < total / 5, "{}: scanned {} of {}", label, scanned, total);
    }
}

#[test]
fn ma_partial_survives_writes_and_compaction() {
    let path = temp_path();
    build_ma_file(&path);

    let part = Horon::open_with_config(&path, ma_config(true)).unwrap();
    // New node in cluster-1 territory at coordinates no grid node occupies
    // (grid coords are integers); must appear in query results (overlay merge).
    part.put("/c1/newcomer", b"fresh").unwrap();
    part.set_semantic("/c1/newcomer", coords(&[505.5, 505.5, 50.0, 50.0])).unwrap();

    let got = part.nearest_semantic(&coords(&[505.5, 505.5, 50.0, 50.0]), 3, 16..18).unwrap();
    assert_eq!(got[0].0, "/c1/newcomer", "overlay node must win the query");

    // Compact folds it into the v3 snapshot; queries still work after remap.
    part.compact().unwrap();
    let got = part.nearest_semantic(&coords(&[505.5, 505.5, 50.0, 50.0]), 3, 16..18).unwrap();
    assert_eq!(got[0].0, "/c1/newcomer", "post-compaction query lost the node");
    assert_eq!(std::fs::read(&path).unwrap()[4], 3, "compaction must keep v3");
}

// ===========================================================================
// Config validation
// ===========================================================================

#[test]
fn invalid_mode_combinations_are_rejected() {
    let is_config_err = |r: Result<Horon, HoronError>| matches!(r, Err(HoronError::Config(_)));

    assert!(is_config_err(Horon::open_with_config(temp_path(), HoronConfig {
        meaning_addressed: true,
        compression: true,
        semantic_dims: SEM_DIMS,
        ..Default::default()
    })), "meaning_addressed + compression must be rejected");

    assert!(is_config_err(Horon::open_with_config(temp_path(), HoronConfig {
        meaning_addressed: true,
        compression: false,
        semantic_dims: 16, // no user dims
        ..Default::default()
    })), "meaning_addressed without user dims must be rejected");

    assert!(is_config_err(Horon::open_with_config(temp_path(), HoronConfig {
        partial_reads: true,
        compression: true,
        ..Default::default()
    })), "partial_reads + compression must be rejected");

    assert!(is_config_err(Horon::open_with_config(temp_path(), HoronConfig {
        partial_reads: true,
        compression: false,
        gacl: true,
        ..Default::default()
    })), "partial_reads + gacl must be rejected");
}
