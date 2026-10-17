//! Partial-mode `HoronReader`: the mmap read surface, verified against the
//! full-materialization reader and the writer over the same file.
//!
//! The resolution logic is shared (`partial_*` functions), so these tests
//! guard the *construction* and *refresh* paths — the parts the reader owns.

use horon::{DurabilityMode, Horon, HoronConfig, HoronReader, ReaderMode, RefreshOutcome};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);
const SEM_DIMS: u8 = 24;

fn temp_path() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "horon_preader_{}_{}.htt",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    p
}

fn coords(vals: &[f64]) -> Vec<u8> {
    use g_math::fixed_point::Q64_64 as FixedPoint;
    let mut out = vec![0u8; SEM_DIMS as usize * 16];
    for (d, v) in vals.iter().enumerate() {
        let off = (16 + d) * 16;
        out[off..off + 16].copy_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out
}

fn ma_cfg() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: SEM_DIMS,
        compression: false,
        auto_compact_threshold: 0,
        meaning_addressed: true,
        semantic_bounds: (0.0, 1.0),
        durability: DurabilityMode::Relaxed,
        ..Default::default()
    }
}

/// Snapshot + WAL-era writes + a delete + an update: every state a read
/// must resolve across view, overlay, and tombstones.
fn build_fixture(path: &std::path::Path) {
    let gf = Horon::open_with_config(path, ma_cfg()).unwrap();
    for i in 0..120 {
        let key = format!("/d/{i:03}");
        gf.put(&key, format!("v{i}").as_bytes()).unwrap();
        let x = (i as f64 * 997.0) % 1000.0 / 1000.0;
        let y = (i as f64 * 331.0) % 1000.0 / 1000.0;
        gf.set_semantic(&key, coords(&[x, y, 0.5, 0.5])).unwrap();
    }
    gf.compact().unwrap(); // v3 snapshot
    // WAL era: new node, an update, a delete
    gf.put("/d/900", b"wal-era").unwrap();
    gf.put("/d/001", b"updated").unwrap();
    gf.remove("/d/002").unwrap();
    drop(gf);
}

#[test]
fn partial_reader_matches_full_reader_everywhere() {
    let path = temp_path();
    build_fixture(&path);

    let full = HoronReader::open(&path).unwrap();
    let part = HoronReader::open_with_mode(&path, ReaderMode::Partial).unwrap();

    assert_eq!(full.len(), part.len(), "len");
    // Full mode lists in store-iteration order, partial in sorted order (a
    // pre-existing writer asymmetry) — the parity claim is content.
    let mut keys = full.list("/").unwrap();
    keys.sort();
    assert_eq!(keys, part.list("/").unwrap(), "list");
    let mut kids = full.children("/d").unwrap();
    kids.sort();
    assert_eq!(kids, part.children("/d").unwrap(), "children");

    for key in &keys {
        assert_eq!(full.get(key).unwrap(), part.get(key).unwrap(), "get {key}");
        assert_eq!(
            full.get_semantic(key).unwrap(),
            part.get_semantic(key).unwrap(),
            "semantic {key}"
        );
        assert_eq!(full.exists(key), part.exists(key), "exists {key}");
    }
    assert!(!part.exists("/d/002"), "tombstoned key resurfaced");
    assert_eq!(part.get("/d/001").unwrap(), b"updated", "WAL update lost");
    assert_eq!(part.get("/d/900").unwrap(), b"wal-era", "WAL insert lost");

    // Semantic k-NN parity: partial (exact mmap scan) vs full reader.
    let q = coords(&[0.31, 0.62, 0.5, 0.5]);
    let f = full.nearest_semantic(&q, 7, 16..20).unwrap();
    let p = part.nearest_semantic(&q, 7, 16..20).unwrap();
    assert_eq!(f, p, "semantic k-NN diverged between modes");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn partial_reader_rejects_incompatible_files() {
    // Compressed
    let path = temp_path();
    {
        let gf = Horon::open_with_config(
            &path,
            HoronConfig { auto_compact_threshold: 0, ..Default::default() },
        )
        .unwrap();
        gf.put("/a", b"1").unwrap();
        gf.compact().unwrap();
    }
    let err = match HoronReader::open_partial(&path) {
        Err(e) => e,
        Ok(_) => panic!("compressed file must be rejected"),
    };
    assert!(
        err.to_string().contains("uncompressed"),
        "wrong rejection: {err}"
    );
    let _ = std::fs::remove_file(&path);

    // GACL
    let path = temp_path();
    {
        let gf = Horon::open_with_config(
            &path,
            HoronConfig {
                compression: false,
                gacl: true,
                auto_compact_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        gf.put("/a", b"1").unwrap();
    }
    let err = match HoronReader::open_partial(&path) {
        Err(e) => e,
        Ok(_) => panic!("GACL file must be rejected"),
    };
    assert!(err.to_string().contains("GACL"), "wrong rejection: {err}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn structural_queries_error_loudly_in_partial_mode() {
    use g_math::fixed_point::Q64_64 as FixedPoint;
    let path = temp_path();
    build_fixture(&path);
    let part = HoronReader::open_partial(&path).unwrap();
    let err = part.nearest(&[FixedPoint::ZERO; 4]).unwrap_err();
    assert!(err.to_string().contains("partial reader mode"), "{err}");
    let err = part.neighbors("/d/001", 3).unwrap_err();
    assert!(err.to_string().contains("partial reader mode"), "{err}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn partial_refresh_applies_increments_and_reloads_past_compaction() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, ma_cfg()).unwrap();
    for i in 0..40 {
        gf.put(&format!("/d/{i:03}"), b"x").unwrap();
    }
    gf.compact().unwrap();

    let mut part = HoronReader::open_partial(&path).unwrap();
    assert_eq!(part.refresh().unwrap(), RefreshOutcome::UpToDate);

    // Incremental: new WAL entries land in the overlay, mmap view untouched.
    gf.put("/d/900", b"late").unwrap();
    gf.remove("/d/003").unwrap();
    match part.refresh().unwrap() {
        RefreshOutcome::Applied(n) => assert!(n >= 2, "expected >=2, got {n}"),
        other => panic!("expected Applied, got {other:?}"),
    }
    assert_eq!(part.get("/d/900").unwrap(), b"late");
    assert!(!part.exists("/d/003"), "tombstone not applied");

    // Compaction folds the WAL; the reader must rebuild IN PARTIAL MODE and
    // release the old inode.
    gf.put("/d/901", b"post").unwrap();
    gf.compact().unwrap();
    gf.put("/d/902", b"after").unwrap();
    let outcome = part.refresh().unwrap();
    assert_eq!(outcome, RefreshOutcome::Reloaded, "compaction not detected");
    assert_eq!(part.get("/d/901").unwrap(), b"post");
    assert_eq!(part.get("/d/902").unwrap(), b"after");
    // Still partial after the rebuild: structural queries still refuse.
    use g_math::fixed_point::Q64_64 as FixedPoint;
    assert!(part.nearest(&[FixedPoint::ZERO; 4]).is_err(), "mode not preserved");

    drop(gf);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn many_partial_readers_alongside_a_live_writer() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, ma_cfg()).unwrap();
    for i in 0..60 {
        gf.put(&format!("/d/{i:03}"), format!("v{i}").as_bytes()).unwrap();
    }
    gf.compact().unwrap();

    // Readers open while the writer holds its exclusive lock.
    let readers: Vec<_> = (0..4)
        .map(|_| HoronReader::open_partial(&path).unwrap())
        .collect();
    for r in &readers {
        assert_eq!(r.get("/d/007").unwrap(), b"v7");
    }
    // Writer keeps writing with mapped readers live.
    gf.put("/d/900", b"live").unwrap();
    assert_eq!(gf.get("/d/900").unwrap(), b"live");
    // A second writer is still refused.
    assert!(Horon::open_with_config(&path, ma_cfg()).is_err());

    drop(gf);
    let _ = std::fs::remove_file(&path);
}
