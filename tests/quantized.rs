//! Quantized semantic storage (`docs/QUANTIZED_SEMANTIC.md`).
//!
//! Covers: write-through canonicalization (in-memory ≡ post-reload, the
//! determinism-across-reload guarantee); rejection of out-of-range user
//! dims, nonzero reserved dims (without GACL), and oversized vectors;
//! on-disk size (13.3× tail shrink at 40 dims); compaction round-trip;
//! GACL composition (bands bit-exact, access decisions unchanged);
//! temporal history sidecars (segment v2) and trajectories; partial reads;
//! meaning-addressed composition; and header flag/version gating.

use horon::quant::{dequantize_raw, quantize_raw, SemLayout};
use horon::{
    DurabilityMode, Horon, HoronConfig, HistoryRetention, HoronHistory,
};
use g_math::fixed_point::FixedPoint;
use std::path::PathBuf;
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

const DIMS: u8 = 40; // typical application shape: 16 reserved + 24 user

fn qconfig() -> HoronConfig {
    HoronConfig {
        semantic_dims: DIMS,
        compression: false,
        auto_compact_threshold: 0,
        durability: DurabilityMode::Relaxed,
        quantized_semantic: true,
        ..Default::default()
    }
}

/// Full-width coords with user dims (16+) set from f64 values.
fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out.resize(DIMS as usize * 16, 0);
    out
}

/// Raw Q64.64 of user dim `u` (0-based from dim 16) in a coords vector.
fn user_raw(coords: &[u8], u: usize) -> i128 {
    let s = (16 + u) * 16;
    i128::from_le_bytes(coords[s..s + 16].try_into().unwrap())
}

#[test]
fn write_through_and_reload_identical() {
    let path = temp_path();
    let stored;
    {
        let gf = Horon::open_with_config(&path, qconfig()).unwrap();
        gf.put("/a", b"x").unwrap();
        gf.set_semantic("/a", coords(&[0.3, 0.7, 0.15])).unwrap();

        // The store holds the on-grid value, not the caller's original:
        // every user dim is an exact multiple of 1/19683.
        stored = gf.get_semantic("/a").unwrap();
        for u in 0..24 {
            let raw = user_raw(&stored, u);
            let q = quantize_raw(raw).unwrap();
            assert_eq!(dequantize_raw(q), raw, "user dim {} not on grid", u);
        }
        // ~4.3-digit contract: grid value within half a step of the input.
        let half_step = (1i128 << 64) / (2 * 19_683);
        let want = FixedPoint::from_f64(0.3).raw();
        assert!((user_raw(&stored, 0) - want).abs() <= half_step);
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open_with_config(&path, qconfig()).unwrap();
        assert_eq!(
            gf.get_semantic("/a").unwrap(),
            stored,
            "in-memory state must equal post-reload state byte-for-byte"
        );
    }
}

#[test]
fn ranking_survives_reload() {
    let path = temp_path();
    let before;
    {
        let gf = Horon::open_with_config(&path, qconfig()).unwrap();
        for i in 0..30 {
            let key = format!("/n{}", i);
            gf.put(&key, b"d").unwrap();
            let v = 0.02 * i as f64;
            gf.set_semantic(&key, coords(&[v, 1.0 - v, 0.5])).unwrap();
        }
        before = gf
            .nearest_semantic(&coords(&[0.31, 0.69, 0.5]), 5, 16..19)
            .unwrap();
        gf.flush().unwrap();
    }
    let gf = Horon::open_with_config(&path, qconfig()).unwrap();
    let after = gf
        .nearest_semantic(&coords(&[0.31, 0.69, 0.5]), 5, 16..19)
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(before[0].0, "/n15"); // 0.30 is the nearest grid point to 0.31
}

#[test]
fn rejects_out_of_range_reserved_and_oversized() {
    let gf = Horon::open_with_config(temp_path(), qconfig()).unwrap();
    gf.put("/a", b"x").unwrap();

    // User dim beyond ±1.49987.
    let err = gf.set_semantic("/a", coords(&[2.0])).unwrap_err();
    assert!(err.to_string().contains("TQ1.9 range"), "got: {}", err);
    // Boundary value is accepted; one step above is not.
    gf.set_semantic("/a", coords(&[29_524.0 / 19_683.0])).unwrap();

    // Nonzero reserved dim without GACL.
    let mut c = coords(&[0.5]);
    c[0..16].copy_from_slice(&FixedPoint::from_f64(0.25).raw().to_le_bytes());
    let err = gf.set_semantic("/a", c).unwrap_err();
    assert!(err.to_string().contains("reserved"), "got: {}", err);

    // More coords than the file has dims.
    let long = vec![0u8; (DIMS as usize + 1) * 16];
    let err = gf.set_semantic("/a", long).unwrap_err();
    assert!(err.to_string().contains("semantic dims"), "got: {}", err);

    // A failed set_semantic must not have half-applied.
    let ok = gf.get_semantic("/a").unwrap();
    assert_eq!(user_raw(&ok, 0), dequantize_raw(29_524));
}

#[test]
fn config_requires_user_dims() {
    let cfg = HoronConfig {
        semantic_dims: 16,
        quantized_semantic: true,
        ..Default::default()
    };
    assert!(Horon::open_with_config(temp_path(), cfg).is_err());
}

#[test]
fn on_disk_size_shrinks() {
    let (qpath, fpath) = (temp_path(), temp_path());
    let build = |path: &PathBuf, quantized: bool| {
        let cfg = HoronConfig {
            quantized_semantic: quantized,
            ..qconfig()
        };
        let gf = Horon::open_with_config(path, cfg).unwrap();
        for i in 0..100 {
            let key = format!("/n{}", i);
            gf.put(&key, b"payload").unwrap();
            gf.set_semantic(&key, coords(&[0.1, 0.2, 0.3, 0.4])).unwrap();
        }
        gf.compact().unwrap();
    };
    build(&qpath, true);
    build(&fpath, false);
    let (q, f) = (
        std::fs::metadata(&qpath).unwrap().len() as i64,
        std::fs::metadata(&fpath).unwrap().len() as i64,
    );
    // 100 nodes × (640 − 48) bytes of tail shrink, snapshot only (the WAL
    // was folded by compact). Everything else in the two files is identical.
    assert_eq!(f - q, 100 * (640 - 48), "quantized {} vs full {}", q, f);
}

#[test]
fn compaction_roundtrip() {
    let path = temp_path();
    let stored;
    {
        let gf = Horon::open_with_config(&path, qconfig()).unwrap();
        for i in 0..20 {
            let key = format!("/n{}", i);
            gf.put(&key, b"d").unwrap();
            gf.set_semantic(&key, coords(&[0.05 * i as f64, 0.9])).unwrap();
        }
        gf.compact().unwrap();
        // Writes after compaction land in the fresh WAL.
        gf.put("/tail", b"t").unwrap();
        gf.set_semantic("/tail", coords(&[1.25, 0.0, 0.33])).unwrap();
        stored = (
            gf.get_semantic("/n7").unwrap(),
            gf.get_semantic("/tail").unwrap(),
        );
        gf.flush().unwrap();
    }
    let gf = Horon::open_with_config(&path, qconfig()).unwrap();
    assert_eq!(gf.get_semantic("/n7").unwrap(), stored.0);
    assert_eq!(gf.get_semantic("/tail").unwrap(), stored.1);
    // Compact again on the reopened handle — still stable.
    gf.compact().unwrap();
    assert_eq!(gf.get_semantic("/tail").unwrap(), stored.1);
}

#[test]
fn gacl_bands_stay_bit_exact() {
    use horon::Credentials;
    let path = temp_path();
    let cfg = HoronConfig {
        gacl: true,
        ..qconfig()
    };
    // A deliberately tight identity-style band that 2-byte quantization
    // could never represent: width 1e-3 at ~0.42.
    let (lo, hi) = (0.4210_f64, 0.4220_f64);
    let band;
    {
        let gf = Horon::open_with_config(&path, cfg).unwrap();
        gf.put("/secret", b"s").unwrap();
        let mut c = coords(&[0.5]);
        c[0..16].copy_from_slice(&FixedPoint::from_f64(lo).raw().to_le_bytes());
        c[16..32].copy_from_slice(&FixedPoint::from_f64(hi).raw().to_le_bytes());
        // can_read checks domain/classification/identity too — open those
        // bands ([0, 1]) so only the tight read band decides.
        let one = FixedPoint::from_int(1).raw().to_le_bytes();
        for hi_dim in [7usize, 9, 11] {
            c[hi_dim * 16..hi_dim * 16 + 16].copy_from_slice(&one);
        }
        gf.set_semantic("/secret", c.clone()).unwrap();
        band = (
            i128::from_le_bytes(c[0..16].try_into().unwrap()),
            i128::from_le_bytes(c[16..32].try_into().unwrap()),
        );
        gf.flush().unwrap();
    }
    let cfg = HoronConfig {
        gacl: true,
        ..qconfig()
    };
    let gf = Horon::open_with_config(&path, cfg).unwrap();
    // Reserved dims survive at FULL precision (no quantization).
    let c = gf.get_semantic("/secret").unwrap();
    assert_eq!(i128::from_le_bytes(c[0..16].try_into().unwrap()), band.0);
    assert_eq!(i128::from_le_bytes(c[16..32].try_into().unwrap()), band.1);
    // And access decisions honor the exact band edges.
    let mut inside = Credentials::root();
    inside.read = FixedPoint::from_f64(0.4215);
    gf.set_credentials(inside);
    assert!(gf.get("/secret").is_ok());
    let mut outside = Credentials::root();
    outside.read = FixedPoint::from_f64(0.4225);
    gf.set_credentials(outside);
    assert!(gf.get("/secret").is_err());
}

#[test]
fn temporal_epochs_and_sidecars() {
    let path = temp_path();
    let cfg = HoronConfig {
        history_retention: HistoryRetention::Archive,
        ..qconfig()
    };
    {
        let gf = Horon::open_with_config(&path, cfg).unwrap();
        gf.put("/k", b"v").unwrap();
        gf.set_semantic("/k", coords(&[0.10, 0.90])).unwrap();
        gf.seal_epoch().unwrap();
        gf.set_semantic("/k", coords(&[0.20, 0.80])).unwrap();
        gf.seal_epoch().unwrap();
        gf.compact().unwrap(); // archives the pre-fence WAL → v2 segment
        gf.set_semantic("/k", coords(&[0.30, 0.70])).unwrap();
        gf.seal_epoch().unwrap();
        gf.flush().unwrap();
    }
    let h = HoronHistory::open(&path).unwrap();
    assert!(h.is_complete());
    assert_eq!(h.epochs().len(), 3);
    let traj = h.trajectory("/k", &(16..18));
    assert_eq!(traj.len(), 3);
    // Trajectory samples are the on-grid values (ranking-grade ~4.3 digits).
    for (i, expect) in [(0usize, 0.10f64), (1, 0.20), (2, 0.30)] {
        assert!(
            (traj[i].1[0] - expect).abs() < 1.0 / 19_683.0,
            "epoch {} sample {} vs {}",
            i,
            traj[i].1[0],
            expect
        );
    }
    // The archived segment is self-describing: version 2 with the layout
    // flags byte (quantized, no GACL) at offset 6.
    let seg = PathBuf::from(format!("{}.h000001", path.display()));
    let bytes = std::fs::read(&seg).expect("sidecar segment");
    assert_eq!(bytes[4], 2, "segment v2");
    assert_eq!(bytes[6], 0x01, "layout flags: quantized, no GACL");
}

#[test]
fn partial_reads_decode_quantized_tails() {
    let path = temp_path();
    let stored;
    {
        let gf = Horon::open_with_config(&path, qconfig()).unwrap();
        for i in 0..25 {
            let key = format!("/n{}", i);
            gf.put(&key, b"d").unwrap();
            gf.set_semantic(&key, coords(&[0.04 * i as f64, 0.5])).unwrap();
        }
        gf.compact().unwrap(); // partial mode reads the snapshot via mmap
        stored = gf.get_semantic("/n9").unwrap();
        gf.flush().unwrap();
    }
    let cfg = HoronConfig {
        partial_reads: true,
        ..qconfig()
    };
    let gf = Horon::open_with_config(&path, cfg).unwrap();
    assert_eq!(gf.get_semantic("/n9").unwrap(), stored);
    let near = gf
        .nearest_semantic(&coords(&[0.36, 0.5]), 1, 16..18)
        .unwrap();
    assert_eq!(near[0].0, "/n9");
}

#[test]
fn composes_with_meaning_addressing() {
    let path = temp_path();
    let cfg = HoronConfig {
        meaning_addressed: true,
        ..qconfig()
    };
    let stored;
    {
        let gf = Horon::open_with_config(&path, cfg).unwrap();
        for i in 0..25 {
            let key = format!("/n{}", i);
            gf.put(&key, b"d").unwrap();
            gf.set_semantic(&key, coords(&[0.04 * i as f64, 0.5])).unwrap();
        }
        gf.compact().unwrap(); // v4 + Hilbert-ordered snapshot
        stored = gf.get_semantic("/n12").unwrap();
        gf.flush().unwrap();
    }
    let cfg = HoronConfig {
        meaning_addressed: true,
        partial_reads: true,
        ..qconfig()
    };
    let gf = Horon::open_with_config(&path, cfg).unwrap();
    assert_eq!(gf.get_semantic("/n12").unwrap(), stored);
    // Windowed (Hilbert-addressed) semantic query over quantized tails.
    let near = gf
        .nearest_semantic(&coords(&[0.49, 0.5]), 1, 16..18)
        .unwrap();
    assert_eq!(near[0].0, "/n12");
}

#[test]
fn header_gates_flag_to_v4() {
    use horon::header::GeoHeader;
    // Hand-build a v2 header with the quantized flag: must be rejected.
    let mut h = GeoHeader::new(4, 40, FixedPoint::from_int(1).raw(), false);
    h.flags |= 0x80; // FLAG_QUANTIZED_SEMANTIC
    let bytes = h.to_bytes();
    let err = GeoHeader::from_bytes(&bytes).unwrap_err();
    assert!(err.to_string().contains("v4"), "got: {}", err);
}

#[test]
fn wal_record_sizes() {
    use horon::{WalEntry, WalPayload};
    let layout = SemLayout { dims: 40, quantized: true, gacl: false };
    let entry = WalEntry {
        seq: 1,
        op: 0x05, // SET_SEMANTIC
        key: "/k".to_string(),
        payload: WalPayload::SetSemantic { coords: coords(&[0.5, 0.25]) },
    };
    let mut quantized = Vec::new();
    entry.write_to(&mut quantized, &layout).unwrap();
    let mut full = Vec::new();
    entry.write_to(&mut full, &SemLayout::plain(40)).unwrap();
    // seq(4) + op(1) + key_len(2) + key(2) + tail + crc(4)
    assert_eq!(quantized.len(), 4 + 1 + 2 + 2 + 48 + 4);
    assert_eq!(full.len(), 4 + 1 + 2 + 2 + 640 + 4);
    // And the quantized record decodes back to the canonical full width.
    let parsed = WalEntry::read_from(&mut std::io::Cursor::new(&quantized), &layout)
        .unwrap()
        .unwrap();
    match parsed.payload {
        WalPayload::SetSemantic { coords: c } => {
            assert_eq!(c.len(), 640);
            // Grid value of 0.5 = 9842/19683 (round-half-away of 9841.5).
            assert_eq!(user_raw(&c, 0), dequantize_raw(quantize_raw(FixedPoint::from_f64(0.5).raw()).unwrap()));
        }
        _ => panic!("expected SetSemantic"),
    }
}
