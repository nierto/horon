//! =============================================================================
//! Horon Comprehensive Integration Tests
//! =============================================================================
//!
//! Tests the full .htt persistence pipeline end-to-end:
//!   - File creation and cold-start recovery
//!   - WAL correctness across all operation types
//!   - Snapshot compaction and reload
//!   - Crash recovery (truncated WAL)
//!   - Compression (zstd snapshot + WAL)
//!   - Concurrent-safe file semantics
//!   - GACL band logic and inheritance
//!   - Semantic coordinate roundtrip
//!   - Edge cases and stress conditions
//!
//! Run with: cargo test --test comprehensive -- --nocapture

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use horon::{Horon, HoronConfig, AccessBand, Credentials};
use horon::gacl::NodeAccessBands;
use horon::format::*;
use horon::header::GeoHeader;
use horon::snapshot::NodeEntry;
use horon::wal::{WalEntry, WalPayload};
use g_math::fixed_point::Q64_64 as FixedPoint;
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn config_no_compression() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

fn config_compressed() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: true,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

fn config_with_semantics(sem: u8) -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: sem,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

fn config_auto_compact(threshold: u32) -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: false,
        auto_compact_threshold: threshold,
        ..Default::default()
    }
}

// ===========================================================================
// 1. FILE CREATION & STRUCTURE
// ===========================================================================

#[test]
fn test_empty_file_structure() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.flush().unwrap();
    }

    let size = fs::metadata(&path).unwrap().len();
    assert_eq!(size as usize, MIN_FILE_SIZE, "empty file should be exactly header + snap_header + wal_header");

    // Verify header bytes directly
    let raw = fs::read(&path).unwrap();
    assert_eq!(&raw[0..4], b"HTT\0");
    assert_eq!(raw[4], VERSION);
    assert_eq!(raw[6], 4); // dimension
}

#[test]
fn test_empty_file_compressed() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_compressed()).unwrap();
        gf.flush().unwrap();
    }

    let raw = fs::read(&path).unwrap();
    let flags = raw[5];
    assert!(flags & FLAG_COMPRESSION != 0, "compression flag should be set");
    assert!(flags & FLAG_WAL_COMPRESSED != 0, "WAL compressed flag should be set");
}

#[test]
fn test_empty_file_with_semantic_dims() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(16)).unwrap();
        gf.flush().unwrap();
    }

    let raw = fs::read(&path).unwrap();
    assert_eq!(raw[7], 16); // semantic_dims
    assert!(raw[5] & FLAG_SEMANTIC != 0, "semantic flag should be set");
}

#[test]
fn test_header_crc_validated_on_open() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.flush().unwrap();
    }

    // Corrupt the dimension byte (offset 6) without updating CRC
    let mut raw = fs::read(&path).unwrap();
    raw[6] = 99;
    fs::write(&path, &raw).unwrap();

    let result = Horon::open(&path);
    assert!(result.is_err(), "corrupted header should fail to open");
}

#[test]
fn test_header_tau_preserved() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        drop(gf);
    }

    let raw = fs::read(&path).unwrap();
    let tau_raw = i128::from_le_bytes(raw[8..24].try_into().unwrap());
    let expected = FixedPoint::from_int(1).raw();
    assert_eq!(tau_raw, expected, "tau should be Q64.64 representation of 1.0");
}

// ===========================================================================
// 2. BASIC CRUD OPERATIONS
// ===========================================================================

#[test]
fn test_put_and_get() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/hello", b"world").unwrap();
    assert_eq!(gf.get("/hello").unwrap(), b"world");
}

#[test]
fn test_put_creates_intermediates() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/a/b/c", b"deep").unwrap();
    assert!(gf.exists("/a"));
    assert!(gf.exists("/a/b"));
    assert!(gf.exists("/a/b/c"));
    assert_eq!(gf.len(), 3);
}

#[test]
fn test_upsert_overwrites_data() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/key", b"v1").unwrap();
    gf.put("/key", b"v2").unwrap();
    assert_eq!(gf.get("/key").unwrap(), b"v2");
    assert_eq!(gf.len(), 1);
}

#[test]
fn test_remove() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/a", b"1").unwrap();
    gf.put("/b", b"2").unwrap();
    gf.remove("/a").unwrap();

    assert!(!gf.exists("/a"));
    assert!(gf.exists("/b"));
    assert_eq!(gf.len(), 1);
}

#[test]
fn test_remove_nonexistent_is_error() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
    assert!(gf.remove("/ghost").is_err());
}

#[test]
fn test_get_nonexistent_is_error() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
    assert!(gf.get("/ghost").is_err());
}

#[test]
fn test_set_and_get_meta() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/doc", b"content").unwrap();
    gf.set_meta("/doc", "author", "alice").unwrap();
    gf.set_meta("/doc", "type", "markdown").unwrap();

    let meta = gf.get_meta("/doc").unwrap();
    assert_eq!(meta.get("author").unwrap(), "alice");
    assert_eq!(meta.get("type").unwrap(), "markdown");
}

#[test]
fn test_children_and_list() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/a/x", b"1").unwrap();
    gf.put("/a/y", b"2").unwrap();
    gf.put("/a/z", b"3").unwrap();
    gf.put("/b", b"4").unwrap();

    let children: HashSet<String> = gf.children("/a").unwrap().into_iter().collect();
    assert!(children.contains("/a/x"));
    assert!(children.contains("/a/y"));
    assert!(children.contains("/a/z"));
    assert!(!children.contains("/b"));

    let all = gf.list("/").unwrap();
    assert!(all.len() >= 4); // /a, /a/x, /a/y, /a/z, /b
}

#[test]
fn test_is_empty_and_len() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    assert!(gf.is_empty());
    assert_eq!(gf.len(), 0);

    gf.put("/x", b"data").unwrap();
    assert!(!gf.is_empty());
    assert_eq!(gf.len(), 1);
}

// ===========================================================================
// 3. WAL PERSISTENCE & RECOVERY
// ===========================================================================

#[test]
fn test_wal_recovery_insert() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/a", b"data_a").unwrap();
        gf.put("/b", b"data_b").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.get("/a").unwrap(), b"data_a");
        assert_eq!(gf.get("/b").unwrap(), b"data_b");
        assert_eq!(gf.len(), 2);
    }
}

#[test]
fn test_wal_recovery_delete() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/a", b"1").unwrap();
        gf.put("/b", b"2").unwrap();
        gf.put("/c", b"3").unwrap();
        gf.remove("/b").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert!(gf.exists("/a"));
        assert!(!gf.exists("/b"));
        assert!(gf.exists("/c"));
        assert_eq!(gf.len(), 2);
    }
}

#[test]
fn test_wal_recovery_update() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/key", b"v1").unwrap();
        gf.put("/key", b"v2").unwrap();
        gf.put("/key", b"v3").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.get("/key").unwrap(), b"v3");
        assert_eq!(gf.len(), 1);
    }
}

#[test]
fn test_wal_recovery_set_meta() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/doc", b"content").unwrap();
        gf.set_meta("/doc", "author", "alice").unwrap();
        gf.set_meta("/doc", "version", "42").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        let meta = gf.get_meta("/doc").unwrap();
        assert_eq!(meta.get("author").unwrap(), "alice");
        assert_eq!(meta.get("version").unwrap(), "42");
    }
}

#[test]
fn test_wal_mixed_operations_recovery() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/a", b"1").unwrap();
        gf.put("/b", b"2").unwrap();
        gf.put("/c", b"3").unwrap();
        gf.set_meta("/a", "tag", "first").unwrap();
        gf.remove("/b").unwrap();
        gf.put("/d", b"4").unwrap();
        gf.put("/a", b"updated").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.get("/a").unwrap(), b"updated");
        assert!(!gf.exists("/b"));
        assert_eq!(gf.get("/c").unwrap(), b"3");
        assert_eq!(gf.get("/d").unwrap(), b"4");
    }
}

#[test]
fn test_wal_entry_count_tracking() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    assert_eq!(gf.wal_len(), 0);
    gf.put("/a", b"1").unwrap();
    assert_eq!(gf.wal_len(), 1);
    gf.put("/b", b"2").unwrap();
    assert_eq!(gf.wal_len(), 2);
    gf.set_meta("/a", "k", "v").unwrap();
    assert_eq!(gf.wal_len(), 3);
    gf.remove("/a").unwrap();
    assert_eq!(gf.wal_len(), 4);
}

// ===========================================================================
// 4. SNAPSHOT COMPACTION
// ===========================================================================

#[test]
fn test_compact_basic() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        for i in 0..20 {
            gf.put(&format!("/n{}", i), format!("d{}", i).as_bytes()).unwrap();
        }
        assert!(gf.wal_len() > 0);
        // compact() reports whether it actually ran (skip signaling).
        assert!(gf.compact().unwrap(), "a serial compact must report that it ran");
        assert_eq!(gf.wal_len(), 0);
        assert_eq!(gf.len(), 20);
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.len(), 20);
        assert_eq!(gf.get("/n7").unwrap(), b"d7");
        assert_eq!(gf.wal_len(), 0);
    }
}

#[test]
fn test_compact_preserves_metadata() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/doc", b"content").unwrap();
        gf.set_meta("/doc", "author", "alice").unwrap();
        gf.set_meta("/doc", "version", "3").unwrap();
        gf.compact().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        let meta = gf.get_meta("/doc").unwrap();
        assert_eq!(meta.get("author").unwrap(), "alice");
        assert_eq!(meta.get("version").unwrap(), "3");
    }
}

#[test]
fn test_compact_excludes_deleted_nodes() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/keep", b"yes").unwrap();
        gf.put("/remove_me", b"no").unwrap();
        gf.remove("/remove_me").unwrap();
        gf.compact().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert!(gf.exists("/keep"));
        assert!(!gf.exists("/remove_me"));
        assert_eq!(gf.len(), 1);
    }
}

#[test]
fn test_compact_then_continue_writing() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    // Write, compact, write more, close, reopen
    gf.put("/pre", b"before").unwrap();
    gf.compact().unwrap();
    gf.put("/post", b"after").unwrap();
    gf.flush().unwrap();
    drop(gf);

    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.get("/pre").unwrap(), b"before");
    assert_eq!(gf.get("/post").unwrap(), b"after");
}

#[test]
fn test_multiple_compactions() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    for round in 0..5 {
        for i in 0..10 {
            gf.put(&format!("/r{}n{}", round, i), b"data").unwrap();
        }
        gf.compact().unwrap();
        assert_eq!(gf.wal_len(), 0);
    }

    drop(gf);
    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.len(), 50);
    assert_eq!(gf.get("/r3n7").unwrap(), b"data");
}

#[test]
fn test_auto_compact_threshold() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_auto_compact(10)).unwrap();

    // Insert 10 nodes — should trigger auto-compaction (which now runs on
    // a background thread, so poll for completion instead of asserting
    // synchronously).
    for i in 0..10 {
        gf.put(&format!("/n{}", i), b"d").unwrap();
    }

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while gf.wal_len() > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(gf.wal_len(), 0, "auto-compact should have fired in the background");
    assert_eq!(gf.len(), 10);

    drop(gf);
    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.len(), 10);
}

// ===========================================================================
// 5. COMPRESSION
// ===========================================================================

#[test]
fn test_compressed_roundtrip() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_compressed()).unwrap();
        for i in 0..100 {
            gf.put(&format!("/node_{}", i), format!("data_{}", i).as_bytes()).unwrap();
        }
        gf.compact().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.len(), 100);
        assert_eq!(gf.get("/node_42").unwrap(), b"data_42");
        assert_eq!(gf.get("/node_99").unwrap(), b"data_99");
    }
}

#[test]
fn test_compressed_smaller_than_uncompressed() {
    let path_z = temp_path();
    let path_raw = temp_path();

    // Write identical data to both
    for (path, compressed) in [(&path_z, true), (&path_raw, false)] {
        let config = HoronConfig {
            dimension: 4,
            semantic_dims: 0,
            compression: compressed,
            auto_compact_threshold: 0,
            ..Default::default()
        };
        let gf = Horon::open_with_config(path, config).unwrap();
        for i in 0..200 {
            gf.put(
                &format!("/item_{}", i),
                format!("repetitive data for item number {}", i).as_bytes(),
            ).unwrap();
        }
        gf.compact().unwrap();
    }

    let size_z = fs::metadata(&path_z).unwrap().len();
    let size_raw = fs::metadata(&path_raw).unwrap().len();
    assert!(
        size_z < size_raw,
        "compressed ({} bytes) should be smaller than raw ({} bytes)",
        size_z, size_raw
    );
}

// ===========================================================================
// 6. CRASH RECOVERY (TRUNCATED WAL)
// ===========================================================================

#[test]
fn test_truncated_wal_entry_is_discarded() {
    let path = temp_path();

    // Write some valid entries
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/good_a", b"aaa").unwrap();
        gf.put("/good_b", b"bbb").unwrap();
        gf.flush().unwrap();
    }

    // Append garbage bytes to simulate a partial write
    {
        let mut file = fs::OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03]).unwrap();
    }

    // Reopen — should recover the two good entries
    {
        let gf = Horon::open(&path).unwrap();
        assert!(gf.exists("/good_a"));
        assert!(gf.exists("/good_b"));
        assert_eq!(gf.len(), 2);
    }
}

#[test]
fn test_corrupted_wal_crc_truncates() {
    let path = temp_path();

    // Write entries
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/first", b"ok").unwrap();
        gf.put("/second", b"ok").unwrap();
        gf.put("/third", b"ok").unwrap();
        gf.flush().unwrap();
    }

    // Corrupt the last few bytes of the file (CRC of last WAL entry)
    {
        let mut raw = fs::read(&path).unwrap();
        let len = raw.len();
        raw[len - 1] ^= 0xFF;
        raw[len - 2] ^= 0xFF;
        fs::write(&path, &raw).unwrap();
    }

    // Reopen — third entry should be discarded due to bad CRC
    {
        let gf = Horon::open(&path).unwrap();
        assert!(gf.exists("/first"));
        assert!(gf.exists("/second"));
        // Third may or may not survive depending on which entry the corruption hits
        // But we should have at least the first two
        assert!(gf.len() >= 2);
    }
}

// ===========================================================================
// 7. SEMANTIC COORDINATES
// ===========================================================================

#[test]
fn test_semantic_dims_preserved_in_file() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(16)).unwrap();
        gf.put("/node", b"data").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.get("/node").unwrap(), b"data");
    }
}

#[test]
fn test_semantic_dims_wal_recovery() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(16)).unwrap();
        gf.put("/a", b"1").unwrap();
        gf.put("/b", b"2").unwrap();
        gf.remove("/a").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert!(!gf.exists("/a"));
        assert_eq!(gf.get("/b").unwrap(), b"2");
    }
}

#[test]
fn test_semantic_dims_compact_and_reopen() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(16)).unwrap();
        for i in 0..30 {
            gf.put(&format!("/s{}", i), format!("d{}", i).as_bytes()).unwrap();
        }
        gf.compact().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.len(), 30);
        assert_eq!(gf.get("/s15").unwrap(), b"d15");
    }
}

// ===========================================================================
// 8. GACL LOGIC
// ===========================================================================

#[test]
fn test_gacl_band_boundary_conditions() {
    // Exact boundary values
    let band = AccessBand::from_f64(0.5, 0.8);
    assert!(band.permits(FixedPoint::from_f64(0.5)));   // lower bound inclusive
    assert!(band.permits(FixedPoint::from_f64(0.8)));   // upper bound inclusive
    assert!(band.permits(FixedPoint::from_f64(0.65)));  // middle
    assert!(!band.permits(FixedPoint::from_f64(0.49))); // just below
    assert!(!band.permits(FixedPoint::from_f64(0.81))); // just above
}

#[test]
fn test_gacl_narrow_idempotent() {
    let parent = AccessBand::from_f64(0.3, 0.9);
    let child = AccessBand::from_f64(0.5, 0.7);
    let narrowed = child.narrow(&parent);

    // Child is already within parent — narrow should be identity
    assert!(narrowed.permits(FixedPoint::from_f64(0.5)));
    assert!(narrowed.permits(FixedPoint::from_f64(0.7)));
    assert!(!narrowed.permits(FixedPoint::from_f64(0.3)));
}

#[test]
fn test_gacl_narrow_inverted_band() {
    // If child lo > parent hi after narrowing, band is empty
    let parent = AccessBand::from_f64(0.2, 0.4);
    let child = AccessBand::from_f64(0.6, 0.8);
    let narrowed = child.narrow(&parent);

    // narrowed.lo = max(0.6, 0.2) = 0.6, narrowed.hi = min(0.8, 0.4) = 0.4
    // lo > hi → empty band, nothing should permit
    assert!(!narrowed.permits(FixedPoint::from_f64(0.5)));
    assert!(!narrowed.permits(FixedPoint::from_f64(0.3)));
    assert!(!narrowed.permits(FixedPoint::from_f64(0.7)));
}

#[test]
fn test_gacl_multi_group_credentials() {
    let viewer = Credentials {
        read: FixedPoint::from_f64(0.5),
        write: FixedPoint::from_f64(0.0),
        exec: FixedPoint::from_f64(0.0),
        domain: FixedPoint::from_f64(0.3),
        classification: FixedPoint::from_f64(0.0),
        identity: FixedPoint::from_f64(0.42),
    };
    let editor = Credentials {
        read: FixedPoint::from_f64(0.3),
        write: FixedPoint::from_f64(0.7),
        exec: FixedPoint::from_f64(0.3),
        domain: FixedPoint::from_f64(0.5),
        classification: FixedPoint::from_f64(0.3),
        identity: FixedPoint::from_f64(0.42),
    };

    let combined = Credentials::from_groups(&[viewer, editor]);

    // Should take max of each: read=0.5, write=0.7, exec=0.3, domain=0.5
    let node = NodeAccessBands {
        read: AccessBand::from_f64(0.0, 1.0),
        write: AccessBand::from_f64(0.5, 1.0),
        exec: AccessBand::open(),
        domain: AccessBand::from_f64(0.3, 0.7),
        classification: AccessBand::open(),
        identity: AccessBand::open(),
    };

    assert!(combined.can_access(&node));
    assert!(combined.can_read(&node));
}

#[test]
fn test_gacl_access_denied() {
    let lowly = Credentials {
        read: FixedPoint::from_f64(0.2),
        write: FixedPoint::from_f64(0.1),
        exec: FixedPoint::from_f64(0.0),
        domain: FixedPoint::from_f64(0.3),
        classification: FixedPoint::from_f64(0.1),
        identity: FixedPoint::from_f64(0.5),
    };

    let restricted = NodeAccessBands {
        read: AccessBand::from_f64(0.5, 1.0),   // needs ≥ 0.5 read
        write: AccessBand::from_f64(0.8, 1.0),
        exec: AccessBand::open(),
        domain: AccessBand::from_f64(0.3, 0.7),
        classification: AccessBand::from_f64(0.5, 1.0),
        identity: AccessBand::open(),
    };

    assert!(!lowly.can_access(&restricted));
    assert!(!lowly.can_read(&restricted));
}

#[test]
fn test_gacl_semantic_bytes_roundtrip() {
    let bands = NodeAccessBands {
        read: AccessBand::from_f64(0.2, 0.8),
        write: AccessBand::from_f64(0.5, 1.0),
        exec: AccessBand::from_f64(0.0, 0.5),
        domain: AccessBand::from_f64(0.3, 0.7),
        classification: AccessBand::from_f64(0.6, 0.9),
        identity: AccessBand::from_f64(0.42, 0.43),
    };

    let bytes = bands.to_semantic_bytes(16);
    assert_eq!(bytes.len(), 16 * 16); // 16 semantic dims × 16 bytes each

    let parsed = NodeAccessBands::from_semantic_bytes(&bytes).unwrap();

    // Verify roundtrip precision
    let cred = FixedPoint::from_f64(0.5);
    assert_eq!(bands.read.permits(cred), parsed.read.permits(cred));
    assert_eq!(bands.write.permits(cred), parsed.write.permits(cred));
    assert_eq!(bands.domain.permits(cred), parsed.domain.permits(cred));
}

#[test]
fn test_gacl_inheritance_chain() {
    // Grandparent → parent → child: monotonic restriction
    let grandparent = NodeAccessBands {
        read: AccessBand::from_f64(0.0, 1.0),
        write: AccessBand::from_f64(0.3, 1.0),
        exec: AccessBand::open(),
        domain: AccessBand::open(),
        classification: AccessBand::open(),
        identity: AccessBand::open(),
    };

    let parent = NodeAccessBands {
        read: AccessBand::from_f64(0.2, 0.9),
        write: AccessBand::from_f64(0.5, 1.0),
        exec: AccessBand::from_f64(0.1, 0.8),
        domain: AccessBand::open(),
        classification: AccessBand::open(),
        identity: AccessBand::open(),
    }.narrow(&grandparent);

    let child = NodeAccessBands::public().narrow(&parent);

    // Child should be at least as restrictive as parent
    // parent.read = narrow([0.2,0.9], [0.0,1.0]) = [0.2, 0.9]
    // child.read  = narrow([0.0,1.0], [0.2,0.9]) = [0.2, 0.9]
    assert!(!child.read.permits(FixedPoint::from_f64(0.1)));
    assert!(child.read.permits(FixedPoint::from_f64(0.5)));

    // parent.write = narrow([0.5,1.0], [0.3,1.0]) = [0.5, 1.0]
    assert!(!child.write.permits(FixedPoint::from_f64(0.4)));
    assert!(child.write.permits(FixedPoint::from_f64(0.7)));
}

// ===========================================================================
// 9. SPATIAL QUERIES (through persistence)
// ===========================================================================

#[test]
fn test_nearest_query_through_persistence() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        gf.put("/a", b"data").unwrap();
        gf.put("/b", b"data").unwrap();
        gf.put("/c", b"data").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        let (path, dist) = gf.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert_eq!(path, "/");
        assert!(dist.to_f64() < 0.01);
    }
}

#[test]
fn test_neighbors_query_through_persistence() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        for i in 0..10 {
            gf.put(&format!("/n{}", i), b"d").unwrap();
        }
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        let neighbors = gf.neighbors("/n0", 3).unwrap();
        assert!(!neighbors.is_empty());
        assert!(neighbors.len() <= 3);
    }
}

// ===========================================================================
// 10. BINARY FORMAT PRIMITIVES
// ===========================================================================

#[test]
fn test_header_all_flags_combinations() {
    for compressed in [false, true] {
        for sem in [0u8, 1, 16, 255] {
            let header = GeoHeader::new(4, sem, FixedPoint::from_int(1).raw(), compressed);
            let bytes = header.to_bytes();
            let parsed = GeoHeader::from_bytes(&bytes).unwrap();

            assert_eq!(parsed.dimension, 4);
            assert_eq!(parsed.semantic_dims, sem);
            assert_eq!(parsed.compression_enabled(), compressed);
            assert_eq!(parsed.has_semantic_dims(), sem > 0);
        }
    }
}

#[test]
fn test_snapshot_node_entry_large_payload() {
    let big_data = vec![0xABu8; 100_000];
    let entry = NodeEntry {
        key: "/big".to_string(),
        data: big_data.clone(),
        metadata: vec![("size".to_string(), "100000".to_string())],
        semantic_coords: vec![],
    };

    let mut buf = Vec::new();
    entry.write_to(&mut buf, &horon::quant::SemLayout::plain(0)).unwrap();

    let mut cursor = std::io::Cursor::new(&buf);
    let parsed = NodeEntry::read_from(&mut cursor, &horon::quant::SemLayout::plain(0)).unwrap();

    assert_eq!(parsed.data, big_data);
    assert_eq!(parsed.metadata[0].1, "100000");
}

#[test]
fn test_snapshot_many_metadata_pairs() {
    let metadata: Vec<(String, String)> = (0..100)
        .map(|i| (format!("key_{}", i), format!("value_{}", i)))
        .collect();

    let entry = NodeEntry {
        key: "/meta_heavy".to_string(),
        data: b"payload".to_vec(),
        metadata: metadata.clone(),
        semantic_coords: vec![],
    };

    let mut buf = Vec::new();
    entry.write_to(&mut buf, &horon::quant::SemLayout::plain(0)).unwrap();

    let mut cursor = std::io::Cursor::new(&buf);
    let parsed = NodeEntry::read_from(&mut cursor, &horon::quant::SemLayout::plain(0)).unwrap();

    assert_eq!(parsed.metadata.len(), 100);
    assert_eq!(parsed.metadata[50].0, "key_50");
}

#[test]
fn test_wal_all_op_types_sequential() {
    let mut buf = Vec::new();

    let ops: Vec<WalEntry> = vec![
        WalEntry { seq: 1, op: OP_INSERT, key: "/a".into(), payload: WalPayload::Insert(NodeEntry {
            key: "/a".into(), data: b"data_a".to_vec(), metadata: vec![], semantic_coords: vec![],
        })},
        WalEntry { seq: 2, op: OP_UPDATE, key: "/a".into(), payload: WalPayload::Update {
            data: b"data_a_v2".to_vec(), metadata: vec![],
        }},
        WalEntry { seq: 3, op: OP_SET_META, key: "/a".into(), payload: WalPayload::SetMeta {
            meta_key: "tag".into(), meta_value: "test".into(),
        }},
        WalEntry { seq: 4, op: OP_DELETE, key: "/a".into(), payload: WalPayload::Delete },
    ];

    for op in &ops {
        op.write_to(&mut buf, &horon::quant::SemLayout::plain(0)).unwrap();
    }

    let mut cursor = std::io::Cursor::new(&buf);
    for expected in &ops {
        let parsed = WalEntry::read_from(&mut cursor, &horon::quant::SemLayout::plain(0)).unwrap().unwrap();
        assert_eq!(parsed.seq, expected.seq);
        assert_eq!(parsed.op, expected.op);
        assert_eq!(parsed.key, expected.key);
    }

    // No more entries
    assert!(WalEntry::read_from(&mut cursor, &horon::quant::SemLayout::plain(0)).unwrap().is_none());
}

// ===========================================================================
// 11. EDGE CASES & STRESS
// ===========================================================================

#[test]
fn test_empty_key_data() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    // Empty data is valid
    gf.put("/empty", b"").unwrap();
    assert_eq!(gf.get("/empty").unwrap(), b"");
}

#[test]
fn test_large_payload() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    let big = vec![0x42u8; 1_000_000]; // 1 MB
    gf.put("/big", &big).unwrap();
    gf.flush().unwrap();

    drop(gf);

    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.get("/big").unwrap(), big);
}

#[test]
fn test_unicode_keys_and_metadata() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/日本語/パス", "日本語データ".as_bytes()).unwrap();
    gf.set_meta("/日本語/パス", "著者", "太郎").unwrap();
    gf.flush().unwrap();

    drop(gf);

    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.get("/日本語/パス").unwrap(), "日本語データ".as_bytes());
    let meta = gf.get_meta("/日本語/パス").unwrap();
    assert_eq!(meta.get("著者").unwrap(), "太郎");
}

#[test]
fn test_deep_path_hierarchy() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    let deep_path = "/a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p";
    gf.put(deep_path, b"deep").unwrap();
    assert!(gf.exists(deep_path));

    // All intermediates should exist
    assert!(gf.exists("/a/b/c/d/e/f/g/h"));
    assert!(gf.len() >= 16); // 16 path components
}

#[test]
fn test_many_nodes_stress() {
    let path = temp_path();
    let n = 1000;

    {
        let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
        for i in 0..n {
            gf.put(&format!("/node_{}", i), format!("data_{}", i).as_bytes()).unwrap();
        }
        gf.compact().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.len(), n);

        // Spot-check random entries
        for i in [0, 42, 99, 500, 999] {
            assert_eq!(
                gf.get(&format!("/node_{}", i)).unwrap(),
                format!("data_{}", i).as_bytes()
            );
        }
    }
}

#[test]
fn test_repeated_compact_cycles() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    for cycle in 0..10 {
        for i in 0..20 {
            let key = format!("/c{}_n{}", cycle, i);
            gf.put(&key, b"x").unwrap();
        }
        gf.compact().unwrap();
    }

    assert_eq!(gf.len(), 200);
    drop(gf);

    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.len(), 200);
}

#[test]
fn test_insert_delete_insert_same_key() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    gf.put("/toggle", b"v1").unwrap();
    gf.remove("/toggle").unwrap();
    gf.put("/toggle", b"v2").unwrap();

    assert_eq!(gf.get("/toggle").unwrap(), b"v2");
    gf.flush().unwrap();

    drop(gf);

    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.get("/toggle").unwrap(), b"v2");
}

// ===========================================================================
// 12. FILE SIZE ACCOUNTING
// ===========================================================================

#[test]
fn test_file_grows_with_wal() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();
    gf.flush().unwrap();

    let size_empty = fs::metadata(&path).unwrap().len();

    gf.put("/a", b"data").unwrap();
    gf.flush().unwrap();
    let size_one = fs::metadata(&path).unwrap().len();

    gf.put("/b", b"more data").unwrap();
    gf.flush().unwrap();
    let size_two = fs::metadata(&path).unwrap().len();

    assert!(size_one > size_empty, "WAL entry should increase file size");
    assert!(size_two > size_one, "second WAL entry should increase further");
}

#[test]
fn test_compact_can_shrink_file() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config_no_compression()).unwrap();

    // Write many entries then delete most
    for i in 0..100 {
        gf.put(&format!("/n{}", i), b"data").unwrap();
    }
    for i in 10..100 {
        gf.remove(&format!("/n{}", i)).unwrap();
    }
    gf.flush().unwrap();

    let size_before_compact = fs::metadata(&path).unwrap().len();

    gf.compact().unwrap();

    let size_after_compact = fs::metadata(&path).unwrap().len();

    assert!(
        size_after_compact < size_before_compact,
        "compaction should shrink file after mass deletes: {} >= {}",
        size_after_compact, size_before_compact
    );
}

// ===========================================================================
// 13. CONCURRENCY (Phase E)
// ===========================================================================

#[test]
fn test_concurrent_reads_and_writes() {
    let path = temp_path();
    let gf = Arc::new(Horon::open_with_config(&path, config_no_compression()).unwrap());

    let mut handles = vec![];
    for t in 0..8 {
        let gf = Arc::clone(&gf);
        handles.push(std::thread::spawn(move || {
            for i in 0..10 {
                gf.put(&format!("/t{}/n{}", t, i), b"data").unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // All 80 unique leaf keys should exist
    for t in 0..8 {
        for i in 0..10 {
            assert!(gf.exists(&format!("/t{}/n{}", t, i)));
        }
    }

    // Compact and reopen
    gf.compact().unwrap();
    gf.flush().unwrap();
    let len_before = gf.len();
    drop(gf);

    let gf = Horon::open(&path).unwrap();
    assert_eq!(gf.len(), len_before);
    for t in 0..8 {
        for i in 0..10 {
            assert_eq!(gf.get(&format!("/t{}/n{}", t, i)).unwrap(), b"data");
        }
    }
}

#[test]
fn test_concurrent_mixed_ops() {
    let path = temp_path();
    let gf = Arc::new(Horon::open_with_config(&path, config_no_compression()).unwrap());

    // Pre-populate some keys
    for i in 0..20 {
        gf.put(&format!("/shared/{}", i), format!("v{}", i).as_bytes()).unwrap();
    }

    let mut handles = vec![];

    // Writers
    for t in 0..4 {
        let gf = Arc::clone(&gf);
        handles.push(std::thread::spawn(move || {
            for i in 0..5 {
                gf.put(&format!("/w{}/n{}", t, i), b"written").unwrap();
            }
        }));
    }

    // Readers
    for _ in 0..4 {
        let gf = Arc::clone(&gf);
        handles.push(std::thread::spawn(move || {
            for i in 0..20 {
                let _ = gf.get(&format!("/shared/{}", i));
                let _ = gf.exists(&format!("/shared/{}", i));
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify writer results
    for t in 0..4 {
        for i in 0..5 {
            assert_eq!(gf.get(&format!("/w{}/n{}", t, i)).unwrap(), b"written");
        }
    }
}

#[test]
fn test_concurrent_compact_during_writes() {
    let path = temp_path();
    let gf = Arc::new(Horon::open_with_config(&path, config_no_compression()).unwrap());

    // Pre-populate
    for i in 0..20 {
        gf.put(&format!("/pre/{}", i), b"data").unwrap();
    }

    // Spawn writers and a compactor concurrently
    let mut handles = vec![];

    for t in 0..4 {
        let gf = Arc::clone(&gf);
        handles.push(std::thread::spawn(move || {
            for i in 0..5 {
                gf.put(&format!("/during/{}/n{}", t, i), b"concurrent").unwrap();
            }
        }));
    }

    let gf_compact = Arc::clone(&gf);
    handles.push(std::thread::spawn(move || {
        gf_compact.compact().unwrap();
    }));

    for h in handles {
        h.join().unwrap();
    }

    // All pre-populated and concurrent keys should exist
    for i in 0..20 {
        assert!(gf.exists(&format!("/pre/{}", i)));
    }
    for t in 0..4 {
        for i in 0..5 {
            assert!(gf.exists(&format!("/during/{}/n{}", t, i)));
        }
    }

    // Reopen and verify
    gf.flush().unwrap();
    drop(gf);

    let gf = Horon::open(&path).unwrap();
    for i in 0..20 {
        assert_eq!(gf.get(&format!("/pre/{}", i)).unwrap(), b"data");
    }
    for t in 0..4 {
        for i in 0..5 {
            assert_eq!(gf.get(&format!("/during/{}/n{}", t, i)).unwrap(), b"concurrent");
        }
    }
}

#[test]
fn test_wal_batch_mode() {
    let path = temp_path();
    let config = HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: false,
        auto_compact_threshold: 0,
        wal_batch_size: 5,
        ..Default::default()
    };
    let gf = Horon::open_with_config(&path, config).unwrap();

    for i in 0..12 {
        gf.put(&format!("/batch/{}", i), b"data").unwrap();
    }

    // Flush remaining entries
    gf.flush().unwrap();

    // 12 leaf INSERTs + 1 for the implicitly created /batch ancestor (WAL
    // completeness: ancestor creation is logged so replicas converge).
    assert_eq!(gf.wal_len(), 13);

    // Reopen and verify all entries persisted
    drop(gf);
    let gf = Horon::open(&path).unwrap();
    for i in 0..12 {
        assert_eq!(gf.get(&format!("/batch/{}", i)).unwrap(), b"data");
    }
}

#[test]
fn test_send_sync_compile_check() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Horon>();
    assert_sync::<Horon>();
}

// ===========================================================================
// 14. WAL BLOCK COMPRESSION
// ===========================================================================

#[test]
fn test_compressed_wal_create_and_reopen() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_compressed()).unwrap();
        gf.put("/a", b"alpha").unwrap();
        gf.put("/b", b"beta").unwrap();
        gf.put("/c", b"gamma").unwrap();
        gf.set_meta("/a", "tag", "first").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert_eq!(gf.get("/a").unwrap(), b"alpha");
        assert_eq!(gf.get("/b").unwrap(), b"beta");
        assert_eq!(gf.get("/c").unwrap(), b"gamma");
        let meta = gf.get_meta("/a").unwrap();
        assert_eq!(meta.get("tag").unwrap(), "first");
    }
}

#[test]
fn test_compressed_wal_delete_recovery() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_compressed()).unwrap();
        gf.put("/keep", b"yes").unwrap();
        gf.put("/remove", b"no").unwrap();
        gf.remove("/remove").unwrap();
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        assert!(gf.exists("/keep"));
        assert!(!gf.exists("/remove"));
    }
}

#[test]
fn test_compressed_wal_compact_and_continue() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, config_compressed()).unwrap();
        for i in 0..30 {
            gf.put(&format!("/pre/{}", i), b"before").unwrap();
        }
        gf.compact().unwrap();
        // Write more after compaction (new block-compressed WAL)
        for i in 0..10 {
            gf.put(&format!("/post/{}", i), b"after").unwrap();
        }
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open(&path).unwrap();
        for i in 0..30 {
            assert_eq!(gf.get(&format!("/pre/{}", i)).unwrap(), b"before");
        }
        for i in 0..10 {
            assert_eq!(gf.get(&format!("/post/{}", i)).unwrap(), b"after");
        }
    }
}

#[test]
fn test_compressed_wal_batch_mode() {
    let path = temp_path();
    let config = HoronConfig {
        compression: true,
        wal_batch_size: 10,
        auto_compact_threshold: 0,
        ..Default::default()
    };
    {
        let gf = Horon::open_with_config(&path, config).unwrap();
        for i in 0..25 {
            gf.put(&format!("/batch/{}", i), b"data").unwrap();
        }
        gf.flush().unwrap();
        // 25 leaf INSERTs + 1 for the implicitly created /batch ancestor.
        assert_eq!(gf.wal_len(), 26);
    }
    {
        let gf = Horon::open(&path).unwrap();
        // 25 leaf entries + 1 intermediate (/batch)
        assert_eq!(gf.len(), 26);
        for i in 0..25 {
            assert_eq!(gf.get(&format!("/batch/{}", i)).unwrap(), b"data");
        }
    }
}

#[test]
fn test_compressed_wal_concurrent_compact() {
    let path = temp_path();
    let config = HoronConfig {
        compression: true,
        auto_compact_threshold: 0,
        ..Default::default()
    };
    let gf = Arc::new(Horon::open_with_config(&path, config).unwrap());

    // Pre-populate
    for i in 0..20 {
        gf.put(&format!("/pre/{}", i), b"data").unwrap();
    }

    // Concurrent writes + compaction
    let mut handles = vec![];
    for t in 0..4 {
        let gf = Arc::clone(&gf);
        handles.push(std::thread::spawn(move || {
            for i in 0..5 {
                gf.put(&format!("/during/{}/n{}", t, i), b"concurrent").unwrap();
            }
        }));
    }
    let gf_c = Arc::clone(&gf);
    handles.push(std::thread::spawn(move || {
        gf_c.compact().unwrap();
    }));

    for h in handles {
        h.join().unwrap();
    }

    // Verify all data
    for i in 0..20 {
        assert!(gf.exists(&format!("/pre/{}", i)));
    }
    for t in 0..4 {
        for i in 0..5 {
            assert!(gf.exists(&format!("/during/{}/n{}", t, i)));
        }
    }

    // Reopen and verify persistence
    gf.flush().unwrap();
    drop(gf);
    let gf = Horon::open(&path).unwrap();
    for i in 0..20 {
        assert_eq!(gf.get(&format!("/pre/{}", i)).unwrap(), b"data");
    }
    for t in 0..4 {
        for i in 0..5 {
            assert_eq!(gf.get(&format!("/during/{}/n{}", t, i)).unwrap(), b"concurrent");
        }
    }
}

#[test]
fn test_compressed_wal_file_smaller() {
    let path_z = temp_path();
    let path_raw = temp_path();

    // Write identical data — enough WAL entries for compression to matter
    for (path, compressed) in [(&path_z, true), (&path_raw, false)] {
        let config = HoronConfig {
            compression: compressed,
            auto_compact_threshold: 0,
            ..Default::default()
        };
        let gf = Horon::open_with_config(path, config).unwrap();
        for i in 0..200 {
            gf.put(
                &format!("/item_{}", i),
                format!("repetitive data for item number {}", i).as_bytes(),
            ).unwrap();
        }
        gf.flush().unwrap();
    }

    let size_z = fs::metadata(&path_z).unwrap().len();
    let size_raw = fs::metadata(&path_raw).unwrap().len();
    assert!(
        size_z < size_raw,
        "compressed WAL ({} bytes) should be smaller than raw ({} bytes)",
        size_z, size_raw
    );
}

// ===========================================================================
// 14. GACL ENFORCEMENT (end-to-end through Horon)
// ===========================================================================

fn gacl_config() -> HoronConfig {
    HoronConfig {
        semantic_dims: 40,
        compression: false,
        auto_compact_threshold: 0,
        gacl: true,
        ..Default::default()
    }
}

/// Build semantic coords with GACL bands set in dims 0-11.
fn coords_with_bands(bands: &NodeAccessBands, total_dims: usize) -> Vec<u8> {
    bands.to_semantic_bytes(total_dims)
}

#[test]
fn test_gacl_enforcement_read_denied() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    // Insert a node and restrict read access to band [0.5, 1.0]
    gf.put("/secret", b"classified").unwrap();
    let bands = NodeAccessBands {
        read: AccessBand::from_f64(0.5, 1.0),
        ..NodeAccessBands::public()
    };
    gf.set_semantic("/secret", coords_with_bands(&bands, 40)).unwrap();

    // Without credentials: everything accessible
    assert!(gf.get("/secret").is_ok());

    // With low credentials: denied
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.3),
        write: FixedPoint::from_f64(1.0),
        exec: FixedPoint::from_f64(1.0),
        domain: FixedPoint::from_f64(1.0),
        classification: FixedPoint::from_f64(1.0),
        identity: FixedPoint::from_f64(1.0),
    });
    assert!(gf.get("/secret").is_err());
    assert!(!gf.exists("/secret")); // invisible

    // With sufficient credentials: allowed
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.7),
        write: FixedPoint::from_f64(1.0),
        exec: FixedPoint::from_f64(1.0),
        domain: FixedPoint::from_f64(1.0),
        classification: FixedPoint::from_f64(1.0),
        identity: FixedPoint::from_f64(1.0),
    });
    assert!(gf.get("/secret").is_ok());
    assert!(gf.exists("/secret"));
}

#[test]
fn test_gacl_enforcement_write_denied() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    // Insert a node and restrict write access to band [0.8, 1.0]
    gf.put("/protected", b"v1").unwrap();
    let bands = NodeAccessBands {
        write: AccessBand::from_f64(0.8, 1.0),
        ..NodeAccessBands::public()
    };
    gf.set_semantic("/protected", coords_with_bands(&bands, 40)).unwrap();

    // Set credentials with insufficient write
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(1.0),
        write: FixedPoint::from_f64(0.5), // below 0.8
        exec: FixedPoint::from_f64(1.0),
        domain: FixedPoint::from_f64(1.0),
        classification: FixedPoint::from_f64(1.0),
        identity: FixedPoint::from_f64(1.0),
    });

    // Read works (write band doesn't gate reads)
    assert!(gf.get("/protected").is_ok());
    // Write denied
    assert!(gf.put("/protected", b"v2").is_err());
    // Remove denied
    assert!(gf.remove("/protected").is_err());
    // set_meta denied
    assert!(gf.set_meta("/protected", "key", "val").is_err());
}

#[test]
fn test_gacl_enforcement_children_filtered() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    gf.put("/dir/public", b"visible").unwrap();
    gf.put("/dir/secret", b"hidden").unwrap();

    // Restrict /dir/secret
    let restricted = NodeAccessBands {
        read: AccessBand::from_f64(0.9, 1.0),
        ..NodeAccessBands::public()
    };
    gf.set_semantic("/dir/secret", coords_with_bands(&restricted, 40)).unwrap();

    // Low-privilege user
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.5),
        write: FixedPoint::from_f64(0.5),
        exec: FixedPoint::from_f64(0.5),
        domain: FixedPoint::from_f64(0.5),
        classification: FixedPoint::from_f64(0.5),
        identity: FixedPoint::from_f64(0.5),
    });

    let children = gf.children("/dir").unwrap();
    assert!(children.contains(&"/dir/public".to_string()));
    assert!(!children.contains(&"/dir/secret".to_string()));
}

#[test]
fn test_gacl_no_flag_means_no_enforcement() {
    // File without GACL flag — credentials have no effect
    let path = temp_path();
    let config = HoronConfig {
        semantic_dims: 40,
        compression: false,
        auto_compact_threshold: 0,
        gacl: false, // <-- no GACL
        ..Default::default()
    };
    let gf = Horon::open_with_config(&path, config).unwrap();

    gf.put("/node", b"data").unwrap();
    // Set restrictive bands
    let bands = NodeAccessBands {
        read: AccessBand::closed(),
        ..NodeAccessBands::public()
    };
    gf.set_semantic("/node", coords_with_bands(&bands, 40)).unwrap();

    // Even with credentials, no enforcement because flag is off
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.5),
        write: FixedPoint::from_f64(0.5),
        exec: FixedPoint::from_f64(0.5),
        domain: FixedPoint::from_f64(0.5),
        classification: FixedPoint::from_f64(0.5),
        identity: FixedPoint::from_f64(0.5),
    });

    // All operations succeed — GACL is not enforced
    assert!(gf.get("/node").is_ok());
    assert!(gf.exists("/node"));
}

#[test]
fn test_gacl_no_coords_means_public() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    // Node without semantic coords — should be public
    gf.put("/open", b"data").unwrap();

    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.1),
        write: FixedPoint::from_f64(0.1),
        exec: FixedPoint::from_f64(0.1),
        domain: FixedPoint::from_f64(0.1),
        classification: FixedPoint::from_f64(0.1),
        identity: FixedPoint::from_f64(0.1),
    });

    // No bands = public, so even low credentials work
    assert!(gf.get("/open").is_ok());
    assert!(gf.put("/open", b"updated").is_ok());
}

#[test]
fn test_gacl_clear_credentials_disables_enforcement() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    gf.put("/secret", b"data").unwrap();
    let bands = NodeAccessBands {
        read: AccessBand::from_f64(0.9, 1.0),
        ..NodeAccessBands::public()
    };
    gf.set_semantic("/secret", coords_with_bands(&bands, 40)).unwrap();

    // Low creds: denied
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.1),
        write: FixedPoint::from_f64(1.0),
        exec: FixedPoint::from_f64(1.0),
        domain: FixedPoint::from_f64(1.0),
        classification: FixedPoint::from_f64(1.0),
        identity: FixedPoint::from_f64(1.0),
    });
    assert!(gf.get("/secret").is_err());

    // Clear credentials: enforcement off
    gf.clear_credentials();
    assert!(gf.get("/secret").is_ok());
}

#[test]
fn test_gacl_insert_always_allowed() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    // Even with minimal credentials, inserting NEW nodes is always allowed
    // (new nodes have no bands yet → public)
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.0),
        write: FixedPoint::from_f64(0.0),
        exec: FixedPoint::from_f64(0.0),
        domain: FixedPoint::from_f64(0.0),
        classification: FixedPoint::from_f64(0.0),
        identity: FixedPoint::from_f64(0.0),
    });

    // Insert succeeds — node doesn't exist yet, no bands to check
    assert!(gf.put("/new_node", b"data").is_ok());
}

#[test]
fn test_gacl_survives_compact_and_reopen() {
    let path = temp_path();

    // Create GACL-enabled file
    {
        let gf = Horon::open_with_config(&path, gacl_config()).unwrap();
        gf.put("/secret", b"classified").unwrap();
        let bands = NodeAccessBands {
            read: AccessBand::from_f64(0.8, 1.0),
            ..NodeAccessBands::public()
        };
        gf.set_semantic("/secret", coords_with_bands(&bands, 40)).unwrap();
        gf.compact().unwrap();
    }

    // Reopen — GACL flag should persist in header
    {
        let gf = Horon::open(&path).unwrap();
        assert!(gf.gacl_active() == false); // no credentials set yet

        // Set low credentials — should be denied
        gf.set_credentials(Credentials {
            read: FixedPoint::from_f64(0.3),
            write: FixedPoint::from_f64(1.0),
            exec: FixedPoint::from_f64(1.0),
            domain: FixedPoint::from_f64(1.0),
            classification: FixedPoint::from_f64(1.0),
            identity: FixedPoint::from_f64(1.0),
        });
        assert!(gf.get("/secret").is_err());

        // Root credentials — should succeed
        gf.set_credentials(Credentials::root());
        assert_eq!(gf.get("/secret").unwrap(), b"classified");
    }
}

// --- GACL hardening regressions ---

fn gacl_fail_closed_config() -> HoronConfig {
    HoronConfig {
        semantic_dims: 40,
        compression: false,
        auto_compact_threshold: 0,
        gacl: true,
        gacl_fail_closed: true,
        ..Default::default()
    }
}

#[test]
fn test_gacl_fail_closed_denies_without_credentials() {
    // With fail-closed set, GACL-enabled + no credentials must deny, not
    // silently expose everything (the default fail-open behavior).
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_fail_closed_config()).unwrap();

    gf.put("/data", b"contents").unwrap();
    // No credentials set: fail-closed denies reads and writes to existing nodes.
    assert!(gf.get("/data").is_err(), "fail-closed must deny reads without credentials");
    assert!(!gf.exists("/data"), "fail-closed node must be invisible without credentials");
    assert!(gf.put("/data", b"v2").is_err(), "fail-closed must deny writes without credentials");

    // Spatial queries return nothing rather than leaking nodes.
    assert!(gf.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).is_err());
    assert!(gf.children("/").unwrap().is_empty());

    // Once credentials are supplied, access resumes normally. The earlier
    // write was denied, so the original contents remain.
    gf.set_credentials(Credentials::root());
    assert_eq!(gf.get("/data").unwrap(), b"contents");
    assert!(gf.put("/data", b"v2").is_ok());
    assert_eq!(gf.get("/data").unwrap(), b"v2");
}

#[test]
fn test_gacl_fail_open_is_the_default() {
    // Default (fail_closed = false): GACL-enabled + no credentials allows all,
    // preserving existing behavior.
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();
    gf.put("/data", b"contents").unwrap();
    assert!(gf.get("/data").is_ok(), "default fail-open must allow reads without credentials");
}

#[test]
fn test_gacl_closed_band_denies_default_credential() {
    // A node with a fully-closed read band must deny even the default/unset
    // credential 0.0 — the [0,0]-permits-0.0 bug.
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    gf.put("/locked", b"secret").unwrap();
    let bands = NodeAccessBands {
        read: AccessBand::closed(),
        ..NodeAccessBands::public()
    };
    gf.set_semantic("/locked", coords_with_bands(&bands, 40)).unwrap();

    // A requester with the zero credential must still be denied.
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.0),
        write: FixedPoint::from_f64(1.0),
        exec: FixedPoint::from_f64(1.0),
        domain: FixedPoint::from_f64(1.0),
        classification: FixedPoint::from_f64(1.0),
        identity: FixedPoint::from_f64(1.0),
    });
    assert!(gf.get("/locked").is_err(), "closed band must deny credential 0.0");
    assert!(!gf.exists("/locked"));
}

#[test]
fn test_gacl_nearest_expands_past_inaccessible_cluster() {
    // Restrict the root and a cluster of nodes, leaving exactly one accessible
    // node. Since the root sits at the origin it is the nearest raw candidate
    // to a [0,0,0,0] query, so a fixed candidate window that stopped at the
    // nearest few would return only inaccessible nodes. The expanding window
    // must push past them to the single accessible node.
    let path = temp_path();
    let gf = Horon::open_with_config(&path, gacl_config()).unwrap();

    let restricted = NodeAccessBands {
        read: AccessBand::from_f64(0.9, 1.0),
        ..NodeAccessBands::public()
    };
    // Restrict the root itself.
    gf.set_semantic("/", coords_with_bands(&restricted, 40)).unwrap();
    // Many restricted nodes near the origin.
    for i in 0..30 {
        let key = format!("/r{}", i);
        gf.put(&key, b"x").unwrap();
        gf.set_semantic(&key, coords_with_bands(&restricted, 40)).unwrap();
    }
    // Exactly one accessible (public) node.
    gf.put("/allowed", b"y").unwrap();

    // Low-privilege reader: cannot see any restricted node or the root.
    gf.set_credentials(Credentials {
        read: FixedPoint::from_f64(0.2),
        write: FixedPoint::from_f64(0.2),
        exec: FixedPoint::from_f64(0.2),
        domain: FixedPoint::from_f64(0.2),
        classification: FixedPoint::from_f64(0.2),
        identity: FixedPoint::from_f64(0.2),
    });

    let (key, _dist) = gf.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
    assert_eq!(key, "/allowed", "expanding window must reach the one accessible node");
    assert!(gf.get(&key).is_ok(), "returned node must be readable by the caller");
}

/// A semantic vector shorter (or longer) than `semantic_dims × 16` must not
/// be serialized verbatim: the WAL/snapshot tail is fixed-size, so replay
/// reads exactly that width — a short record would misalign the stream,
/// fail the next CRC, and recovery would silently truncate every later
/// committed write. Short vectors zero-extend (the documented decode
/// semantics for missing dims); oversized vectors are rejected.
#[test]
fn short_semantic_vector_cannot_corrupt_wal() {
    let path = temp_path();
    let stored;
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(20)).unwrap();
        gf.put("/a", b"first").unwrap();
        // 2 dims worth of bytes in a 20-dim file (multiple of 16, so the
        // store-level alignment guard alone does not catch it).
        let mut short = Vec::new();
        short.extend_from_slice(&FixedPoint::from_f64(0.25).raw().to_le_bytes());
        short.extend_from_slice(&FixedPoint::from_f64(0.75).raw().to_le_bytes());
        gf.set_semantic("/a", short.clone()).unwrap();
        // Writes AFTER the short record are the ones a misaligned replay loses.
        gf.put("/b", b"second").unwrap();
        gf.set_meta("/b", "k", "v").unwrap();

        // In-memory view is already the zero-extended full-width vector, so
        // pre- and post-reload reads agree.
        stored = gf.get_semantic("/a").unwrap();
        assert_eq!(stored.len(), 20 * 16);
        assert_eq!(&stored[..32], &short[..]);
        assert!(stored[32..].iter().all(|&b| b == 0));
        gf.flush().unwrap();
    }
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(20)).unwrap();
        assert_eq!(gf.get("/a").unwrap(), b"first");
        assert_eq!(
            gf.get("/b").unwrap(),
            b"second",
            "entries after a short SET_SEMANTIC record must survive replay"
        );
        assert_eq!(gf.get_meta("/b").unwrap().get("k").map(String::as_str), Some("v"));
        assert_eq!(gf.get_semantic("/a").unwrap(), stored);
    }
    // Oversized vectors are rejected loudly (truncation would drop data).
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(20)).unwrap();
        let long = vec![0u8; 21 * 16];
        let err = gf.set_semantic("/a", long).unwrap_err();
        assert!(err.to_string().contains("semantic dims"), "got: {}", err);
    }
}

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::Q64_64> {
    vals.iter().map(|&v| g_math::fixed_point::Q64_64::from_f64(v)).collect()
}
