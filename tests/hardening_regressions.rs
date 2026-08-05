//! Regression tests for hardening fixes.
//!
//! Each test pins a verified silent-data-loss bug: torn WAL tails swallowing
//! post-recovery writes, write-side length wraps corrupting the log, and
//! partial-mode compaction resurrecting deletes. Every one of them lost data
//! without reporting an error, which is why they get dedicated tests rather
//! than living inside a broader suite.

use std::io::Write;
use std::path::PathBuf;

use horon::{Horon, HoronConfig};
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn plain_config() -> HoronConfig {
    HoronConfig {
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

/// recovery must TRUNCATE the torn WAL tail. Without truncation,
/// writes made after a recovery land behind the garbage and the next open
/// (which stops scanning at the garbage) silently loses fsync-acked data.
#[test]
fn recovery_truncates_torn_tail_so_new_writes_survive() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, plain_config()).unwrap();
        gf.put("/a", b"1").unwrap();
        gf.put("/b", b"2").unwrap();
        gf.put("/c", b"3").unwrap();
        gf.flush().unwrap();
    }
    // Simulate a crash mid-append: garbage bytes at the tail.
    {
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x99]).unwrap();
    }
    // Recover, then write MORE data (fsync-acked).
    {
        let gf = Horon::open_with_config(&path, plain_config()).unwrap();
        assert_eq!(gf.len(), 3, "recovery should keep the 3 committed entries");
        gf.put("/d", b"4").unwrap();
        gf.flush().unwrap();
    }
    // The post-recovery write must survive the next open.
    let gf = Horon::open_with_config(&path, plain_config()).unwrap();
    assert_eq!(
        gf.get("/d").unwrap(),
        b"4".to_vec(),
        "post-recovery write was lost — torn tail not truncated"
    );
    assert_eq!(gf.len(), 4);
}

/// oversized keys/data must be rejected at the write path — a key
/// longer than the u16 length field silently wrapped at serialization time,
/// corrupting the WAL and destroying the committed entries behind it.
#[test]
fn oversized_writes_rejected_before_corrupting_the_log() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, plain_config()).unwrap();
    gf.put("/ok", b"fine").unwrap();

    let huge_key = format!("/{}", "k".repeat(70_000));
    assert!(gf.put(&huge_key, b"x").is_err(), "70k-byte key must be rejected");

    let huge_meta = "v".repeat(70_000);
    assert!(
        gf.set_meta("/ok", "m", &huge_meta).is_err(),
        "70k-byte metadata value must be rejected"
    );

    gf.put("/after", b"still fine").unwrap();
    gf.flush().unwrap();
    drop(gf);

    // Everything committed around the rejected writes survives reopen.
    let gf = Horon::open_with_config(&path, plain_config()).unwrap();
    assert_eq!(gf.get("/ok").unwrap(), b"fine".to_vec());
    assert_eq!(gf.get("/after").unwrap(), b"still fine".to_vec());
    assert!(gf.get(&huge_key).is_err());
}

/// a delete issued while a partial-mode compaction is folding the
/// snapshot must stay deleted — clearing all tombstones resurrected it.
#[test]
fn partial_mode_delete_survives_compaction() {
    let path = temp_path();
    let config = || HoronConfig {
        compression: false,
        partial_reads: true,
        auto_compact_threshold: 0,
        ..Default::default()
    };
    // Build a file whose entries are snapshot-resident.
    {
        let gf = Horon::open_with_config(&path, config()).unwrap();
        gf.put("/keep", b"k").unwrap();
        gf.put("/gone", b"g").unwrap();
        gf.compact().unwrap();
    }
    // Reopen in partial mode: delete a snapshot-resident key, then compact.
    // (Single-threaded stand-in for the concurrent interleaving: the
    // tombstone exists while compaction merges the snapshot.)
    {
        let gf = Horon::open_with_config(&path, config()).unwrap();
        gf.remove("/gone").unwrap();
        assert!(gf.get("/gone").is_err(), "deleted before compaction");
        gf.compact().unwrap();
        assert!(
            gf.get("/gone").is_err(),
            "compaction resurrected a deleted key (tombstones cleared)"
        );
        assert_eq!(gf.get("/keep").unwrap(), b"k".to_vec());
    }
    // And it stays gone across reopen.
    let gf = Horon::open_with_config(&path, config()).unwrap();
    assert!(gf.get("/gone").is_err());
    assert_eq!(gf.get("/keep").unwrap(), b"k".to_vec());
}

/// A `put_data_only` node must survive `compact()` + reopen.
///
/// Data-only inserts skip the geometric path, so they never reserve a slot in
/// their parent's sibling sequence. Reopening without `lazy_geometry` replays
/// every node through the full geometric path, where the data-only node is
/// assigned an index from the parent's counter — which could be an index some
/// sibling had already recorded in `_child_index`. Both quantized to the same
/// point, the insert was refused as a signature collision, and the whole file
/// failed to open with `Failed to add node to tensor network`: the data was
/// written, acknowledged, compacted, and then unreachable.
///
/// Placement now probes past an occupied slot instead of refusing.
#[test]
fn data_only_node_survives_compaction_and_reopen() {
    let path = temp_path();

    {
        let gf = Horon::open_with_config(&path, plain_config()).unwrap();
        // Geometric siblings first: these record `_child_index` values.
        for i in 0..20 {
            gf.put(&format!("/catalog/course/c{i}"), b"geo").unwrap();
        }
        // A data-only node whose ancestor (/catalog/_meta) is created by the
        // data-only path and so never reserves a sibling slot under /catalog.
        gf.put_data_only("/catalog/_meta/epochs", b"payload").unwrap();
        gf.set_meta("/catalog/_meta/epochs", "kind", "index").unwrap();
        gf.compact().unwrap();
    }

    // The reopen that used to fail outright.
    let gf = Horon::open_with_config(&path, plain_config())
        .expect("a file containing a data-only node must reopen");

    // Reopening is not enough — the data has to still be there and correct.
    assert_eq!(gf.get("/catalog/_meta/epochs").unwrap(), b"payload".to_vec());
    assert_eq!(
        gf.get_meta("/catalog/_meta/epochs").unwrap().get("kind").map(String::as_str),
        Some("index"),
        "metadata on the data-only node must survive"
    );
    for i in 0..20 {
        assert_eq!(
            gf.get(&format!("/catalog/course/c{i}")).unwrap(),
            b"geo".to_vec(),
            "geometric siblings must be unharmed by the probe"
        );
    }
}

/// The same file must reopen repeatedly, not just once: an incremental
/// pipeline opens, writes, compacts and closes on every run.
#[test]
fn data_only_node_survives_repeated_reopen_cycles() {
    let path = temp_path();

    for round in 0..4 {
        let gf = Horon::open_with_config(&path, plain_config())
            .unwrap_or_else(|e| panic!("round {round} failed to open: {e:?}"));
        gf.put(&format!("/catalog/course/c{round}"), b"geo").unwrap();
        gf.put_data_only("/catalog/_meta/epochs", format!("round-{round}").as_bytes())
            .unwrap();
        gf.compact().unwrap();
    }

    let gf = Horon::open_with_config(&path, plain_config()).unwrap();
    assert_eq!(gf.get("/catalog/_meta/epochs").unwrap(), b"round-3".to_vec());
    for round in 0..4 {
        assert_eq!(gf.get(&format!("/catalog/course/c{round}")).unwrap(), b"geo".to_vec());
    }
}
