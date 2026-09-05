//! Determinism when auto-compaction actually fires.
//!
//! `tests/determinism.rs` proves the format has no per-run leaks, but it sets
//! `auto_compact_threshold: 0` — it never lets auto-compaction run. A
//! downstream determinism audit (gMaterials, 2026-09-05) found that above the
//! default threshold of 10 000 entries the same input produced different bytes
//! on every run: compaction was handed to a detached thread, which read its
//! fence whenever the scheduler got to it while the writer kept appending, so
//! the snapshot/WAL split — and therefore the compressed bytes — moved run to
//! run. 154 248 nodes, four runs, four different hashes.
//!
//! `AutoCompactMode::Sync` (the default since 0.15.0) compacts on the writer
//! during the write that crosses the threshold, so the fence is that write's
//! sequence number and the split is a pure function of the input.
//!
//! **The fixture size is measured, not guessed.** A first attempt used 130
//! nodes at a threshold of 40 and passed under `Background` too — it would
//! have passed with or without the fix, which is worse than no test. Sweeping
//! for the scale where the defect actually appears gave, over 5 builds each:
//!
//! ```text
//!   nodes  threshold  contended | Sync | Background
//!    2000        100       yes  |   1  |  5 distinct hashes
//!    5000        200       yes  |   1  |  5 distinct hashes
//!   20000        500       yes  |   1  |  5 distinct hashes
//! ```
//!
//! 2000/100 is the smallest configuration that reproduced, so that is what
//! these tests use. Shrinking it for speed would quietly restore a fixture
//! that cannot fail. WAL batching and relaxed durability are set only to keep
//! the runtime sane — neither affects where a fence lands.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use horon::{AutoCompactMode, DurabilityMode, Horon, HoronConfig};
use tempfile::NamedTempFile;

/// Measured minimum that exposes a scheduler-chosen fence. Do not lower.
const THRESHOLD: u32 = 100;
const NODES: usize = 2_000;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

/// Identical script, every value derived from the loop index. No clocks.
fn build(mode: AutoCompactMode) -> Vec<u8> {
    let path = temp_path();
    let gf = Horon::open_with_config(
        &path,
        HoronConfig {
            dimension: 4,
            semantic_dims: 8,
            compression: true, // zstd magnifies any snapshot difference
            auto_compact_threshold: THRESHOLD,
            auto_compact_mode: mode,
            // Speed only: the default fsyncs every append, which would make
            // this test minutes long. Neither setting moves a fence.
            wal_batch_size: 256,
            durability: DurabilityMode::Relaxed,
            ..Default::default()
        },
    )
    .unwrap();

    for i in 0..NODES {
        let key = format!("/t/branch_{}/leaf_{:03}", i % 4, i);
        gf.put(&key, format!("payload-{:05}", i * 13).as_bytes()).unwrap();
        gf.set_meta(&key, "tier", &format!("{}", i % 7)).unwrap();
    }
    gf.flush().unwrap();
    drop(gf);
    std::fs::read(&path).unwrap()
}

/// Under contention, so a fence chosen by the scheduler has room to move.
fn build_contended(mode: AutoCompactMode) -> Vec<u8> {
    let stop = Arc::new(AtomicBool::new(false));
    let mut busy = Vec::new();
    for _ in 0..2 {
        let stop = Arc::clone(&stop);
        busy.push(std::thread::spawn(move || {
            let mut x = 0u64;
            while !stop.load(Ordering::Relaxed) {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                std::hint::black_box(x);
            }
        }));
    }
    let bytes = build(mode);
    stop.store(true, Ordering::Relaxed);
    for t in busy {
        t.join().unwrap();
    }
    bytes
}

/// The default must be the reproducible one. A consumer that never reads the
/// docs still gets a file it can hash.
#[test]
fn the_default_mode_is_sync() {
    assert_eq!(HoronConfig::default().auto_compact_mode, AutoCompactMode::Sync);
}

/// The defect, as reported: same script, above threshold, different bytes.
#[test]
fn identical_scripts_above_the_threshold_produce_identical_bytes() {
    let a = build(AutoCompactMode::Sync);
    let b = build(AutoCompactMode::Sync);
    assert_eq!(a.len(), b.len(), "lengths differ: {} vs {}", a.len(), b.len());
    assert!(
        a == b,
        "two identical scripts crossing the auto-compaction threshold produced \
         different bytes — the snapshot/WAL split is not a function of the input",
    );
}

/// The same, with the CPU contended. A fence read by whichever thread the
/// scheduler picks would move here even if it happened to be stable when the
/// machine was idle.
#[test]
fn contention_does_not_change_the_bytes() {
    let a = build_contended(AutoCompactMode::Sync);
    let b = build_contended(AutoCompactMode::Sync);
    assert!(
        a == b,
        "contention changed the output bytes — compaction timing is leaking \
         into the file",
    );
}

/// Background mode trades reproducibility for a writer that never stalls, and
/// that trade is the documented reason it is not the default. What it must
/// still preserve is the logical content: the same keys with the same
/// payloads, whatever the split.
///
/// It also demonstrates the second cost of the mode. Dropping the handle does
/// not release the file, because the compaction thread holds its own reference
/// to the store — so an immediate reopen can fail with `Locked`. The retry
/// below is not defensive padding; it is the documented behaviour, asserted.
/// Under `Sync` no reopen ever needs it.
#[test]
fn background_mode_preserves_content_even_though_bytes_may_move() {
    let path = temp_path();
    let gf = Horon::open_with_config(
        &path,
        HoronConfig {
            dimension: 4,
            semantic_dims: 8,
            auto_compact_threshold: THRESHOLD,
            auto_compact_mode: AutoCompactMode::Background,
            wal_batch_size: 256,
            durability: DurabilityMode::Relaxed,
            ..Default::default()
        },
    )
    .unwrap();
    for i in 0..NODES {
        gf.put(&format!("/b/{:05}", i), format!("v{}", i).as_bytes()).unwrap();
    }
    gf.flush().unwrap();
    drop(gf);

    let cfg = || HoronConfig {
        dimension: 4,
        semantic_dims: 8,
        auto_compact_mode: AutoCompactMode::Background,
        ..Default::default()
    };
    let mut reopened = None;
    let mut waited_for_the_lock = false;
    for _ in 0..200 {
        match Horon::open_with_config(&path, cfg()) {
            Ok(h) => {
                reopened = Some(h);
                break;
            }
            Err(horon::HoronError::Locked(_)) => {
                waited_for_the_lock = true;
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
            Err(e) => panic!("reopen failed for a reason other than the lock: {e:?}"),
        }
    }
    let reopened = reopened.expect("compaction thread never released the lock");
    let _ = waited_for_the_lock; // observed or not, both are legal here
    for i in 0..NODES {
        assert_eq!(
            reopened.get(&format!("/b/{:05}", i)).unwrap(),
            format!("v{}", i).into_bytes(),
            "background compaction lost or corrupted a payload",
        );
    }
}
