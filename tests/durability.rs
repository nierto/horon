//! Durability fault-injection tests.
//!
//! These tests attack the file, not the API: torn WAL headers, corrupt
//! snapshots, garbage tails, kill -9 mid-run, and concurrent process opens.
//! True power-loss (device cache) durability cannot be tested in-process;
//! what CAN be tested is that nothing relies on Drop/page-cache flushes and
//! that recovery never trusts unverified bytes.
//!
//! Run: GMATH_PROFILE=embedded cargo test --test durability

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use horon::{DurabilityMode, Horon, HoronConfig, HoronError};
use horon::header::GeoHeader;
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

/// Uncompressed, immediate-flush config — raw bytes are easy to attack.
fn plain_config() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

/// Byte offset of the 8-byte WAL header: file header + snapshot section.
fn wal_header_offset(bytes: &[u8]) -> usize {
    let header = GeoHeader::from_bytes(bytes[..32].try_into().unwrap()).unwrap();
    let mut cursor = Cursor::new(&bytes[32..]);
    horon::snapshot::read_snapshot(
        &mut cursor,
        header.compression_enabled(),
        &horon::quant::SemLayout::plain(header.semantic_dims as usize),
        header.version >= 2,
    )
    .unwrap();
    32 + cursor.position() as usize
}

fn write_n_entries(path: &Path, n: usize) {
    let gf = Horon::open_with_config(path, plain_config()).unwrap();
    for i in 0..n {
        gf.put(&format!("/k/e_{}", i), format!("v{}", i).as_bytes()).unwrap();
    }
    drop(gf);
}

fn count_entries(path: &Path) -> usize {
    let gf = Horon::open_with_config(path, plain_config()).unwrap();
    (0..1000)
        .filter(|i| gf.exists(&format!("/k/e_{}", i)))
        .count()
}

// ===========================================================================
// the WAL header is advisory; committed entries survive a torn header
// ===========================================================================

#[test]
fn torn_wal_header_undercount_recovers_all_entries() {
    let path = temp_path();
    write_n_entries(&path, 12);

    let mut bytes = std::fs::read(&path).unwrap();
    let off = wal_header_offset(&bytes);
    // Torn write scenario: header says 3 entries, disk holds 12 valid ones.
    bytes[off..off + 4].copy_from_slice(&3u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    assert_eq!(count_entries(&path), 12, "committed entries were dropped by a torn header");
}

#[test]
fn torn_wal_header_overcount_is_harmless() {
    let path = temp_path();
    write_n_entries(&path, 12);

    let mut bytes = std::fs::read(&path).unwrap();
    let off = wal_header_offset(&bytes);
    bytes[off..off + 4].copy_from_slice(&500u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    assert_eq!(count_entries(&path), 12);
}

#[test]
fn garbage_base_seq_recovers_entries() {
    let path = temp_path();
    write_n_entries(&path, 8);

    let mut bytes = std::fs::read(&path).unwrap();
    let off = wal_header_offset(&bytes);
    // The first valid entry anchors the sequence chain, not the header.
    bytes[off + 4..off + 8].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    assert_eq!(count_entries(&path), 8);
}

#[test]
fn wal_scan_stops_cleanly_at_garbage_tail() {
    let path = temp_path();
    write_n_entries(&path, 10);

    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend(std::iter::repeat(0xAB).take(64));
    std::fs::write(&path, &bytes).unwrap();

    assert_eq!(count_entries(&path), 10, "garbage tail must truncate, not corrupt or fail open");
}

// ===========================================================================
// durability modes actually differ; default is Batched
// ===========================================================================

#[test]
fn default_durability_is_batched() {
    assert_eq!(HoronConfig::default().durability, DurabilityMode::Batched);
    assert_eq!(DurabilityMode::default(), DurabilityMode::Batched);
}

#[test]
fn fsync_mode_overrides_batching() {
    // With a 64-entry batch configured, Fsync mode must still put every
    // entry on disk immediately; Batched must hold them in the pending batch.
    let observe = |mode: DurabilityMode| -> usize {
        let path = temp_path();
        let gf = Horon::open_with_config(&path, HoronConfig {
            wal_batch_size: 64,
            wal_flush_interval_ms: 0,
            durability: mode,
            ..plain_config()
        }).unwrap();
        for i in 0..5 {
            gf.put(&format!("/k/e_{}", i), b"x").unwrap();
        }
        // Copy the file while the handle is still open (before Drop flushes)
        // and count what a fresh reader recovers from the copy.
        let copy = temp_path();
        std::fs::copy(&path, &copy).unwrap();
        drop(gf);
        count_entries(&copy)
    };

    assert_eq!(observe(DurabilityMode::Fsync), 5, "Fsync must persist every append immediately");
    assert_eq!(observe(DurabilityMode::Batched), 0, "Batched must hold entries in the pending batch");
}

// ===========================================================================
// snapshot integrity: CRC and length bounds
// ===========================================================================

#[test]
fn snapshot_crc_detects_corruption() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, plain_config()).unwrap();
        for i in 0..10 {
            gf.put(&format!("/k/e_{}", i), format!("value_{}", i).as_bytes()).unwrap();
        }
        gf.compact().unwrap();
    }

    let mut bytes = std::fs::read(&path).unwrap();
    // Flip one byte inside the snapshot entry data (past the 8-byte section lens).
    bytes[32 + 8 + 21] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    match Horon::open_with_config(&path, plain_config()) {
        Err(HoronError::ChecksumMismatch { context, .. }) => {
            assert!(context.contains("snapshot"), "wrong CRC context: {}", context);
        }
        Err(e) => panic!("expected snapshot ChecksumMismatch, got: {}", e),
        Ok(_) => panic!("corrupt snapshot must not open silently"),
    }
}

#[test]
fn corrupt_snapshot_length_field_is_clean_error() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, plain_config()).unwrap();
        gf.put("/k/e_0", b"x").unwrap();
        gf.compact().unwrap();
    }

    let mut bytes = std::fs::read(&path).unwrap();
    // snap_byte_len := u32::MAX — must error cleanly, not allocate 4GB.
    bytes[32..36].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    assert!(
        Horon::open_with_config(&path, plain_config()).is_err(),
        "absurd length field must be a clean error"
    );
}

#[test]
fn v1_fixture_opens_and_upgrades_to_v2_on_compact() {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests").join("fixtures").join("catalog.htt");
    let path = temp_path();
    std::fs::copy(&src, &path).unwrap();

    let version_of = |p: &Path| std::fs::read(p).unwrap()[4];
    assert_eq!(version_of(&path), 1, "fixture should be a v1 file");

    {
        let gf = Horon::open_with_config(&path, HoronConfig {
            lazy_geometry: true,
            ..Default::default()
        }).unwrap();
        assert!(gf.len() > 100);
        gf.compact().unwrap();
    }
    assert_eq!(version_of(&path), 2, "compaction should upgrade the file to v2");

    // And the upgraded file must still open with everything intact.
    let gf = Horon::open_with_config(&path, HoronConfig {
        lazy_geometry: true,
        ..Default::default()
    }).unwrap();
    assert!(gf.len() > 100);
}

// ===========================================================================
// inter-process lock and orphan tempfile cleanup
// ===========================================================================

#[test]
fn orphan_tmp_removed_on_open() {
    let path = temp_path();
    write_n_entries(&path, 3);

    let tmp = path.with_extension("htt.tmp");
    std::fs::write(&tmp, b"crashed compaction leftovers").unwrap();

    let _gf = Horon::open_with_config(&path, plain_config()).unwrap();
    assert!(!tmp.exists(), "orphaned .htt.tmp must be removed on open");
}

/// Child process body for `second_process_cannot_open_locked_file`.
/// Only runs when spawned by the parent test (env var set); asserts that
/// opening a file locked by the parent fails with `Locked`.
#[test]
fn lock_probe_child() {
    let Ok(path) = std::env::var("GFILE_LOCK_PROBE_PATH") else { return };
    match Horon::open_with_config(&path, plain_config()) {
        Err(HoronError::Locked(_)) => {} // expected — exit success
        Err(e) => panic!("expected Locked error, got: {}", e),
        Ok(_) => panic!("second process must not be able to open a locked file"),
    }
}

#[cfg(unix)]
#[test]
fn second_process_cannot_open_locked_file() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, plain_config()).unwrap();

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "lock_probe_child", "--nocapture"])
        .env("GFILE_LOCK_PROBE_PATH", &path)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "child should observe Locked and exit cleanly (it panicked instead)"
    );
    drop(gf);

    // After the lock holder is gone, a new open must succeed.
    let _gf2 = Horon::open_with_config(&path, plain_config()).unwrap();
}

// ===========================================================================
// kill -9: recovery must not depend on Drop or clean shutdown
// ===========================================================================

/// Child process body for `kill9_child_recovers_fsynced_writes`: writes 20
/// entries in Fsync mode, then dies by abort() — no Drop, no flush, no
/// destructors. Everything durable must already be on disk.
#[test]
fn crash_child_writer() {
    let Ok(path) = std::env::var("GFILE_CRASH_PATH") else { return };
    let gf = Horon::open_with_config(&path, HoronConfig {
        durability: DurabilityMode::Fsync,
        ..plain_config()
    }).unwrap();
    for i in 0..20 {
        gf.put(&format!("/k/e_{}", i), format!("v{}", i).as_bytes()).unwrap();
    }
    std::process::abort();
}

#[test]
fn kill9_child_recovers_fsynced_writes() {
    let path = temp_path();

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child_writer", "--nocapture"])
        .env("GFILE_CRASH_PATH", &path)
        .status()
        .unwrap();
    assert!(!status.success(), "child is supposed to die by abort()");

    assert_eq!(
        count_entries(&path),
        20,
        "every Fsync-acknowledged write must survive an abort()"
    );
}

// ===========================================================================
// parser robustness: random corruption and truncation must never panic
// ===========================================================================

/// Deterministic xorshift64* PRNG — no rand dependency, reproducible failures.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

fn build_corpus_file(compressed: bool) -> Vec<u8> {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, HoronConfig {
        dimension: 4,
        semantic_dims: 20,
        compression: compressed,
        auto_compact_threshold: 0,
        ..Default::default()
    }).unwrap();
    for i in 0..25 {
        let key = format!("/corpus/group_{}/item_{}", i % 4, i);
        gf.put(&key, format!("payload_{}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "tag", &format!("t{}", i)).unwrap();
        let mut coords = vec![0u8; 20 * 16];
        let off = 16 * 16;
        // i+1: all-zero coordinates encode "not set" and are rejected.
        coords[off..off + 16].copy_from_slice(&((i as i128 + 1) << 64).to_le_bytes());
        gf.set_semantic(&key, coords).unwrap();
    }
    gf.compact().unwrap(); // snapshot populated
    for i in 25..35 {
        gf.put(&format!("/corpus/tail_{}", i), b"wal entry").unwrap(); // WAL populated
    }
    drop(gf);
    std::fs::read(&path).unwrap()
}

/// Opening a corrupted file must return Ok or Err — never panic or OOM.
fn assert_open_never_panics(bytes: Vec<u8>, what: &str) {
    let path = temp_path();
    std::fs::write(&path, &bytes).unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = Horon::open_with_config(&path, plain_config());
    }));
    assert!(result.is_ok(), "open PANICKED on {}", what);
}

#[test]
fn random_byte_corruption_never_panics() {
    for compressed in [false, true] {
        let pristine = build_corpus_file(compressed);
        let mut rng = XorShift(0x9E3779B97F4A7C15);

        // Full sweep in CI; sampled locally to keep the dev loop fast.
        let iters = if std::env::var("CI").is_ok() { 250 } else { 60 };
        for iter in 0..iters {
            let mut bytes = pristine.clone();
            let flips = 1 + (rng.next() % 8) as usize;
            for _ in 0..flips {
                let pos = (rng.next() % bytes.len() as u64) as usize;
                bytes[pos] ^= (rng.next() % 255 + 1) as u8;
            }
            assert_open_never_panics(
                bytes,
                &format!("mutation iter {} (compressed={})", iter, compressed),
            );
        }
    }
}

#[test]
fn truncation_at_any_boundary_never_panics() {
    for compressed in [false, true] {
        let pristine = build_corpus_file(compressed);
        // Every truncation boundary region; denser sweep in CI.
        let step = if std::env::var("CI").is_ok() { 7 } else { 37 };
        for len in (0..pristine.len()).step_by(step) {
            assert_open_never_panics(
                pristine[..len].to_vec(),
                &format!("truncation at {} of {} (compressed={})", len, pristine.len(), compressed),
            );
        }
    }
}
