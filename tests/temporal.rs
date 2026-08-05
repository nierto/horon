//! Temporal epochs: seal → archive → replay.
//!
//! Covers: epoch counter persistence across compaction and reopen; sidecar
//! archival (and its absence with retention off); HoronHistory epochs/as_of/
//! trajectory/delta correctness against scripted state; speculative epoch
//! labeling; graceful degradation when sidecars are deleted.

use horon::{
    DeltaKind, DurabilityMode, Horon, HoronConfig, HistoryRetention, HoronHistory,
};
use g_math::fixed_point::FixedPoint;
use std::path::PathBuf;
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

/// 20 semantic dims (16 reserved + 4 user); encode user dims 16..20.
fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out.resize(20 * 16, 0);
    out
}

fn config(retention: HistoryRetention) -> HoronConfig {
    HoronConfig {
        semantic_dims: 20,
        auto_compact_threshold: 0, // manual compaction only
        durability: DurabilityMode::Relaxed,
        history_retention: retention,
        ..Default::default()
    }
}

fn sidecars(path: &PathBuf) -> Vec<PathBuf> {
    let dir = path.parent().unwrap();
    let stem = path.file_name().unwrap().to_str().unwrap();
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(&format!("{}.h", stem)) && !n.ends_with(".tmp"))
        })
        .collect();
    out.sort();
    out
}

/// The full lifecycle: seal → mutate → seal → compact → mutate → seal.
/// as_of() must reproduce the exact state at every seal.
#[test]
fn epochs_replay_exact_states_across_compaction() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config(HistoryRetention::Archive)).unwrap();

        // Epoch 1: two nodes.
        gf.put("/a", b"a1").unwrap();
        gf.put("/b", b"b1").unwrap();
        gf.set_semantic("/a", coords(&[0.1, 0.1, 0.0, 0.0])).unwrap();
        gf.set_semantic("/b", coords(&[0.9, 0.9, 0.0, 0.0])).unwrap();
        assert_eq!(gf.seal_epoch().unwrap(), 1);

        // Epoch 2: /a moves, /c appears.
        gf.set_semantic("/a", coords(&[0.5, 0.5, 0.0, 0.0])).unwrap();
        gf.put("/c", b"c1").unwrap();
        assert_eq!(gf.seal_epoch().unwrap(), 2);

        // Compaction archives epochs 1–2 history into a sidecar.
        assert!(gf.compact().unwrap());
        assert_eq!(sidecars(&path).len(), 1, "one history segment after compact");

        // Epoch 3 (post-compaction): /b deleted, /c gets coords.
        gf.remove("/b").unwrap();
        gf.set_semantic("/c", coords(&[0.2, 0.8, 0.0, 0.0])).unwrap();
        assert_eq!(gf.seal_epoch().unwrap(), 3);
        gf.flush().unwrap();
    }

    let hist = HoronHistory::open(&path).unwrap();
    assert!(hist.is_complete(), "retention was on from creation");
    let epochs = hist.epochs();
    assert_eq!(epochs.iter().map(|e| e.id).collect::<Vec<_>>(), vec![1, 2, 3]);
    assert!(epochs.iter().all(|e| !e.speculative));

    // Epoch 1: /a at (0.1, 0.1), /b present, no /c.
    let s1 = hist.as_of(1).unwrap();
    assert_eq!(s1.len(), 2);
    assert_eq!(s1.get("/a").unwrap().data, b"a1");
    assert!(s1.get("/c").is_none());
    let a1 = decode2(&s1.get("/a").unwrap().semantic);
    assert!((a1.0 - 0.1).abs() < 1e-9 && (a1.1 - 0.1).abs() < 1e-9);

    // Epoch 2: /a moved to (0.5, 0.5), /c present without coords.
    let s2 = hist.as_of(2).unwrap();
    assert_eq!(s2.len(), 3);
    let a2 = decode2(&s2.get("/a").unwrap().semantic);
    assert!((a2.0 - 0.5).abs() < 1e-9 && (a2.1 - 0.5).abs() < 1e-9);
    assert!(s2.get("/c").unwrap().semantic.is_empty());

    // Epoch 3: /b gone, /c at (0.2, 0.8).
    let s3 = hist.as_of(3).unwrap();
    assert_eq!(s3.len(), 2);
    assert!(s3.get("/b").is_none());
    let c3 = decode2(&s3.get("/c").unwrap().semantic);
    assert!((c3.0 - 0.2).abs() < 1e-9 && (c3.1 - 0.8).abs() < 1e-9);
}

fn decode2(sem: &[u8]) -> (f64, f64) {
    let d = |i: usize| {
        FixedPoint::from_raw(i128::from_le_bytes(
            sem[i * 16..(i + 1) * 16].try_into().unwrap(),
        ))
        .to_f64()
    };
    (d(16), d(17))
}

/// The epoch counter must survive compaction (re-stamp) and reopen (replay).
#[test]
fn epoch_counter_survives_compaction_and_reopen() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config(HistoryRetention::Off)).unwrap();
        gf.put("/x", b"1").unwrap();
        gf.seal_epoch().unwrap();
        gf.seal_epoch().unwrap();
        assert_eq!(gf.current_epoch(), 2);
        assert!(gf.compact().unwrap());
        // Counter survives even with retention OFF (the re-stamp is cheap).
        assert_eq!(gf.current_epoch(), 2);
        assert_eq!(gf.seal_epoch().unwrap(), 3);
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open_with_config(&path, config(HistoryRetention::Off)).unwrap();
        assert_eq!(gf.current_epoch(), 3, "counter restored by WAL replay");
        assert_eq!(gf.seal_epoch().unwrap(), 4);
    }
}

/// Retention off: compaction leaves no sidecars — the simple htt pays nothing.
#[test]
fn retention_off_creates_no_sidecars() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config(HistoryRetention::Off)).unwrap();
    gf.put("/x", b"1").unwrap();
    gf.put("/y", b"2").unwrap();
    assert!(gf.compact().unwrap());
    assert!(sidecars(&path).is_empty());
}

/// Deleting sidecars degrades to a plain htt: the file still opens and
/// serves current state; HoronHistory reports incomplete and as_of refuses.
#[test]
fn deleting_sidecars_degrades_gracefully() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config(HistoryRetention::Archive)).unwrap();
        gf.put("/x", b"1").unwrap();
        gf.seal_epoch().unwrap();
        assert!(gf.compact().unwrap());
        gf.put("/y", b"2").unwrap();
        gf.seal_epoch().unwrap();
        gf.flush().unwrap();
    }
    for s in sidecars(&path) {
        std::fs::remove_file(s).unwrap();
    }
    // Plain open serves the current state.
    {
        let gf = Horon::open_with_config(&path, config(HistoryRetention::Archive)).unwrap();
        assert_eq!(gf.get("/x").unwrap(), b"1");
        assert_eq!(gf.get("/y").unwrap(), b"2");
        assert_eq!(gf.current_epoch(), 2);
    }
    // Temporal reads know they can't reconstruct the past exactly.
    let hist = HoronHistory::open(&path).unwrap();
    assert!(!hist.is_complete());
    assert!(hist.as_of(2).is_err(), "incomplete history must refuse as_of");
}

/// A scripted coordinate path must come back exactly, per epoch, and
/// speculative epochs must be flagged and skipped by trajectory sampling.
#[test]
fn trajectory_and_speculative_epochs() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config(HistoryRetention::Archive)).unwrap();

    gf.put("/course", b"x").unwrap();
    gf.set_semantic("/course", coords(&[0.1, 0.0, 0.0, 0.0])).unwrap();
    gf.seal_epoch().unwrap(); // e1: 0.1

    gf.set_semantic("/course", coords(&[0.4, 0.0, 0.0, 0.0])).unwrap();
    // Intra-epoch excursion: up to 0.9 and back down before the seal —
    // the epoch sample is the LAST write before the seal.
    gf.set_semantic("/course", coords(&[0.9, 0.0, 0.0, 0.0])).unwrap();
    gf.set_semantic("/course", coords(&[0.5, 0.0, 0.0, 0.0])).unwrap();
    gf.seal_epoch().unwrap(); // e2: 0.5

    gf.set_semantic("/course", coords(&[0.7, 0.0, 0.0, 0.0])).unwrap();
    gf.seal_speculative_epoch().unwrap(); // e3: speculative — skipped

    gf.set_semantic("/course", coords(&[0.6, 0.0, 0.0, 0.0])).unwrap();
    gf.seal_epoch().unwrap(); // e4: 0.6
    gf.flush().unwrap();
    drop(gf);

    let hist = HoronHistory::open(&path).unwrap();
    let epochs = hist.epochs();
    assert_eq!(epochs.len(), 4);
    assert!(epochs[2].speculative && epochs[2].id == 3);

    let traj = hist.trajectory("/course", &(16..17));
    let ids: Vec<u64> = traj.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, vec![1, 2, 4], "speculative epoch 3 must be skipped");
    let vals: Vec<f64> = traj.iter().map(|(_, v)| v[0]).collect();
    assert!((vals[0] - 0.1).abs() < 1e-9);
    assert!((vals[1] - 0.5).abs() < 1e-9, "sample is the last write before the seal");
    assert!((vals[2] - 0.6).abs() < 1e-9);
}

/// delta(a, b) reports added, removed, and moved keys with displacement.
#[test]
fn delta_reports_added_removed_moved() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config(HistoryRetention::Archive)).unwrap();

    gf.put("/stays", b"x").unwrap();
    gf.set_semantic("/stays", coords(&[0.2, 0.2, 0.0, 0.0])).unwrap();
    gf.put("/leaves", b"x").unwrap();
    gf.seal_epoch().unwrap(); // e1

    gf.set_semantic("/stays", coords(&[0.5, 0.6, 0.0, 0.0])).unwrap();
    gf.remove("/leaves").unwrap();
    gf.put("/arrives", b"x").unwrap();
    gf.seal_epoch().unwrap(); // e2
    gf.flush().unwrap();
    drop(gf);

    let hist = HoronHistory::open(&path).unwrap();
    let deltas = hist.delta(1, 2, &(16..18)).unwrap();

    let find = |k: &str| deltas.iter().find(|d| d.key == k).unwrap();
    assert!(matches!(find("/arrives").kind, DeltaKind::Added));
    assert!(matches!(find("/leaves").kind, DeltaKind::Removed));
    match &find("/stays").kind {
        DeltaKind::Moved { displacement, distance } => {
            assert!((displacement[0] - 0.3).abs() < 1e-9);
            assert!((displacement[1] - 0.4).abs() < 1e-9);
            assert!((distance - 0.5).abs() < 1e-9, "3-4-5 triangle");
        }
        other => panic!("expected Moved, got {:?}", other),
    }
    assert_eq!(deltas.len(), 3, "unexpected extra deltas: {:?}", deltas);
}

/// Two compactions produce two segments; history spans both.
#[test]
fn multiple_segments_merge_seamlessly() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config(HistoryRetention::Archive)).unwrap();
        gf.put("/n", b"1").unwrap();
        gf.seal_epoch().unwrap();
        assert!(gf.compact().unwrap());
        gf.put("/n2", b"2").unwrap();
        gf.seal_epoch().unwrap();
        assert!(gf.compact().unwrap());
        gf.put("/n3", b"3").unwrap();
        gf.seal_epoch().unwrap();
        gf.flush().unwrap();
    }
    assert_eq!(sidecars(&path).len(), 2);

    let hist = HoronHistory::open(&path).unwrap();
    assert!(hist.is_complete());
    assert_eq!(hist.epochs().len(), 3);
    assert_eq!(hist.as_of(1).unwrap().len(), 1); // /n
    assert_eq!(hist.as_of(3).unwrap().len(), 3); // /n + /n2 + /n3
}
