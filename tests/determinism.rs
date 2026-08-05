//! Cross-platform determinism.
//!
//! The claim behind WAL-as-replication: the same operation script produces a
//! byte-identical .htt file on any conforming platform (x86, ARM, ...),
//! because all geometry is gMath Q64.64 fixed-point and the file format has
//! no nondeterministic leaks (timestamps, map ordering).
//!
//! Two layers of proof:
//!   1. `same_script_same_bytes` — two independent builds in this process
//!      produce byte-identical files (catches per-run leaks).
//!   2. `golden_file_crc` — the file bytes match a committed golden CRC
//!      (catches per-PLATFORM divergence when CI runs this on x86 and ARM).
//!
//! If a deliberate format/layout change breaks the golden, update GOLDEN_CRC
//! in the same commit and say so in the commit message.

use std::path::PathBuf;

use horon::{Horon, HoronConfig};
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

/// Fixed operation script: hierarchy + payloads + user metadata + semantic
/// coords + deletes + a compaction. Everything derived from loop indices —
/// no clocks, no randomness.
fn build_scripted_file() -> Vec<u8> {
    let path = temp_path();
    let sem_dims: u8 = 20;
    let gf = Horon::open_with_config(&path, HoronConfig {
        dimension: 4,
        semantic_dims: sem_dims,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }).unwrap();

    for i in 0..30 {
        let key = format!("/det/branch_{}/leaf_{:02}", i % 3, i);
        gf.put(&key, format!("payload-{:04}", i * 7).as_bytes()).unwrap();
        gf.set_meta(&key, "tier", &format!("{}", i % 5)).unwrap();
        gf.set_meta(&key, "alpha", "constant").unwrap();

        let mut coords = vec![0u8; sem_dims as usize * 16];
        for d in 0..4 {
            let off = (16 + d) * 16;
            let val = (((i * 31 + d * 7) % 97) as i128) << 60; // fractional Q64.64
            coords[off..off + 16].copy_from_slice(&val.to_le_bytes());
        }
        gf.set_semantic(&key, coords).unwrap();
    }
    for i in (0..30).step_by(5) {
        gf.remove(&format!("/det/branch_{}/leaf_{:02}", i % 3, i)).unwrap();
    }
    gf.compact().unwrap();
    for i in 30..40 {
        gf.put(&format!("/det/wal_tail_{}", i), format!("tail-{}", i).as_bytes()).unwrap();
    }
    gf.flush().unwrap();
    drop(gf);

    std::fs::read(&path).unwrap()
}

/// The same fixed script against a quantized-semantic file (format
/// v4). Coord values are scaled into the TQ1.9 range (k/128, k < 97); the
/// quantize→dequantize write-through is pure integer arithmetic, so the
/// quantized layout carries its own golden CRC.
fn build_scripted_quantized_file() -> Vec<u8> {
    let path = temp_path();
    let sem_dims: u8 = 20;
    let gf = Horon::open_with_config(&path, HoronConfig {
        dimension: 4,
        semantic_dims: sem_dims,
        compression: false,
        auto_compact_threshold: 0,
        quantized_semantic: true,
        ..Default::default()
    }).unwrap();

    for i in 0..30 {
        let key = format!("/det/branch_{}/leaf_{:02}", i % 3, i);
        gf.put(&key, format!("payload-{:04}", i * 7).as_bytes()).unwrap();
        gf.set_meta(&key, "tier", &format!("{}", i % 5)).unwrap();
        gf.set_meta(&key, "alpha", "constant").unwrap();

        let mut coords = vec![0u8; sem_dims as usize * 16];
        for d in 0..4 {
            let off = (16 + d) * 16;
            let val = (((i * 31 + d * 7) % 97) as i128) << 57; // k/128 < 0.76
            coords[off..off + 16].copy_from_slice(&val.to_le_bytes());
        }
        gf.set_semantic(&key, coords).unwrap();
    }
    for i in (0..30).step_by(5) {
        gf.remove(&format!("/det/branch_{}/leaf_{:02}", i % 3, i)).unwrap();
    }
    gf.compact().unwrap();
    for i in 30..40 {
        gf.put(&format!("/det/wal_tail_{}", i), format!("tail-{}", i).as_bytes()).unwrap();
    }
    gf.flush().unwrap();
    drop(gf);

    std::fs::read(&path).unwrap()
}

#[test]
fn same_script_same_bytes() {
    let a = build_scripted_file();
    let b = build_scripted_file();
    assert_eq!(a.len(), b.len(), "independent builds differ in length");
    assert!(a == b, "independent builds of the same script must be byte-identical");
}

#[test]
fn same_script_same_bytes_quantized() {
    let a = build_scripted_quantized_file();
    let b = build_scripted_quantized_file();
    assert_eq!(a.len(), b.len(), "independent quantized builds differ in length");
    assert!(a == b, "independent quantized builds must be byte-identical");
}

#[test]
fn golden_file_crc_quantized() {
    const GOLDEN_CRC_QUANTIZED: u32 = 0xAA7E_4AD3;
    let bytes = build_scripted_quantized_file();
    let crc = crc32fast::hash(&bytes);
    assert_eq!(
        crc, GOLDEN_CRC_QUANTIZED,
        "quantized file bytes diverge from the committed golden (len {}, got 0x{:08X}). \
         On a new platform this means the determinism claim is BROKEN; after a \
         deliberate format change, update GOLDEN_CRC_QUANTIZED in the same commit.",
        bytes.len(), crc
    );
}

#[test]
fn golden_file_crc() {
    const GOLDEN_CRC: u32 = 0x69C5_E988;
    let bytes = build_scripted_file();
    let crc = crc32fast::hash(&bytes);
    assert_eq!(
        crc, GOLDEN_CRC,
        "file bytes diverge from the committed golden (len {}, got 0x{:08X}). \
         On a new platform this means the determinism claim is BROKEN; after a \
         deliberate format change, update GOLDEN_CRC in the same commit.",
        bytes.len(), crc
    );
}
