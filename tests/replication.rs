//! WAL tailing/subscription: "the file IS the replication protocol."
//!
//! A replica catches up with `wal_entries_since(from_seq)` and stays current
//! with `subscribe_wal()`. Subscribe first, catch up second: the two streams
//! meet exactly at the returned `next_seq` — no gap, no overlap. Live
//! entries are delivered strictly after the primary's fsync, so a replica
//! never holds data the primary could lose.
//!
//! Run: GMATH_PROFILE=embedded cargo test --test replication

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use horon::{Horon, HoronConfig, WalEntry, WalPayload, WalTail};
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn config() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 20,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

/// Apply a tailed WAL entry to a replica through the public API.
fn apply_to_replica(replica: &Horon, entry: &WalEntry) {
    match &entry.payload {
        WalPayload::Insert(node) => {
            replica.put(&entry.key, &node.data).unwrap();
            for (mk, mv) in &node.metadata {
                let _ = replica.set_meta(&entry.key, mk, mv);
            }
            if node.semantic_coords.iter().any(|&b| b != 0) {
                let _ = replica.set_semantic(&entry.key, node.semantic_coords.clone());
            }
        }
        WalPayload::Update { data, metadata } => {
            replica.put(&entry.key, data).unwrap();
            for (mk, mv) in metadata {
                let _ = replica.set_meta(&entry.key, mk, mv);
            }
        }
        WalPayload::Delete => {
            let _ = replica.remove(&entry.key);
        }
        WalPayload::SetMeta { meta_key, meta_value } => {
            let _ = replica.set_meta(&entry.key, meta_key, meta_value);
        }
        WalPayload::SetSemantic { coords } => {
            let _ = replica.set_semantic(&entry.key, coords.clone());
        }
        WalPayload::Epoch { .. } => {
            // Epoch markers carry no node state; a replica would advance its
            // own counter via seal_epoch when mirroring temporal semantics.
        }
    }
}

#[test]
fn catch_up_returns_exactly_the_requested_range() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config()).unwrap();
    for i in 0..10 {
        gf.put(&format!("/k/n{}", i), format!("v{}", i).as_bytes()).unwrap();
    }

    // Everything from the beginning (seqs start at 1). The first put under
    // /k/ logs the implicitly created /k ancestor, so 10 puts = 11 entries.
    match gf.wal_entries_since(1).unwrap() {
        WalTail::Entries(entries) => {
            assert_eq!(entries.len(), 11);
            // Sequence order, contiguous.
            for w in entries.windows(2) {
                assert_eq!(w[1].seq, w[0].seq + 1);
            }
        }
        other => panic!("expected entries, got {:?}", other),
    }

    // A mid-stream fence.
    match gf.wal_entries_since(6).unwrap() {
        WalTail::Entries(entries) => {
            assert_eq!(entries.len(), 6);
            assert_eq!(entries[0].seq, 6);
        }
        other => panic!("expected entries, got {:?}", other),
    }

    // A future fence: nothing yet, not an error.
    match gf.wal_entries_since(99).unwrap() {
        WalTail::Entries(entries) => assert!(entries.is_empty()),
        other => panic!("expected empty entries, got {:?}", other),
    }
}

#[test]
fn compaction_signals_snapshot_required() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config()).unwrap();
    for i in 0..10 {
        gf.put(&format!("/k/n{}", i), b"x").unwrap();
    }
    gf.compact().unwrap(); // folds the ancestor + 10 leaves into the snapshot

    let base = match gf.wal_entries_since(1).unwrap() {
        WalTail::SnapshotRequired { base_seq } => {
            assert!(base_seq > 1, "base_seq should have advanced past the fold");
            base_seq
        }
        other => panic!("expected SnapshotRequired, got {:?}", other),
    };

    // From the new base onward, tailing works again (/k already exists, so
    // this logs exactly one entry).
    gf.put("/k/after", b"post-compaction").unwrap();
    match gf.wal_entries_since(base).unwrap() {
        WalTail::Entries(entries) => {
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].key, "/k/after");
        }
        other => panic!("expected entries, got {:?}", other),
    }
}

#[test]
fn live_subscription_delivers_in_order_after_durability() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config()).unwrap();
    gf.put("/before", b"not delivered").unwrap();

    let (next_seq, rx) = gf.subscribe_wal().unwrap();
    for i in 0..20 {
        gf.put(&format!("/live/n{}", i), format!("L{}", i).as_bytes()).unwrap();
    }

    // 20 leaves + the /live ancestor logged by the first put = 21 entries.
    let mut received = Vec::new();
    while let Ok(entry) = rx.recv_timeout(Duration::from_secs(2)) {
        received.push(entry);
        if received.len() == 21 {
            break;
        }
    }
    assert_eq!(received.len(), 21);
    assert_eq!(received[0].seq, next_seq, "stream must start exactly at next_seq");
    for w in received.windows(2) {
        assert_eq!(w[1].seq, w[0].seq + 1, "sequence gap in live stream");
    }
    assert_eq!(received[0].key, "/live", "ancestor precedes its first child");
    assert_eq!(received[1].key, "/live/n0");

    // Dropping the receiver unsubscribes; further writes must not error.
    drop(rx);
    gf.put("/live/after_unsub", b"ok").unwrap();
    gf.flush().unwrap();
}

#[test]
fn full_replication_round_trip() {
    let primary_path = temp_path();
    let replica_path = temp_path();
    let primary = Horon::open_with_config(&primary_path, config()).unwrap();
    let replica = Horon::open_with_config(&replica_path, config()).unwrap();

    // History before the replica appears.
    for i in 0..15 {
        let key = format!("/hist/n{:02}", i);
        primary.put(&key, format!("h{}", i).as_bytes()).unwrap();
        primary.set_meta(&key, "origin", "history").unwrap();
    }
    primary.remove("/hist/n03").unwrap();

    // Replica arrives: subscribe FIRST, then catch up below next_seq.
    let (next_seq, rx) = primary.subscribe_wal().unwrap();
    match primary.wal_entries_since(1).unwrap() {
        WalTail::Entries(entries) => {
            for e in entries.iter().filter(|e| e.seq < next_seq) {
                apply_to_replica(&replica, e);
            }
        }
        other => panic!("expected entries, got {:?}", other),
    }

    // Live writes race in from another thread while the replica tails.
    let p = Arc::new(primary);
    let writer = {
        let p = Arc::clone(&p);
        std::thread::spawn(move || {
            for i in 0..25 {
                let key = format!("/live/n{:02}", i);
                p.put(&key, format!("l{}", i).as_bytes()).unwrap();
                if i % 5 == 0 {
                    p.set_meta(&key, "tag", "five").unwrap();
                }
            }
            p.remove("/live/n07").unwrap();
        })
    };

    // Tail until the writer is done and the stream drains.
    writer.join().unwrap();
    p.flush().unwrap();
    while let Ok(entry) = rx.recv_timeout(Duration::from_millis(500)) {
        apply_to_replica(&replica, &entry);
    }

    // The replica must now hold the primary's exact logical state.
    assert_eq!(p.len(), replica.len(), "node counts diverge");
    for key in p.list("/").unwrap() {
        assert!(replica.exists(&key), "replica missing {}", key);
        assert_eq!(p.get(&key).unwrap(), replica.get(&key).unwrap(), "data: {}", key);
        assert_eq!(
            p.get_semantic(&key).unwrap(),
            replica.get_semantic(&key).unwrap(),
            "semantic: {}", key
        );
        let pm = p.get_meta(&key).unwrap();
        let rm = replica.get_meta(&key).unwrap();
        assert_eq!(pm.get("origin"), rm.get("origin"), "meta origin: {}", key);
        assert_eq!(pm.get("tag"), rm.get("tag"), "meta tag: {}", key);
    }
    assert!(!replica.exists("/hist/n03"), "replicated delete lost");
    assert!(!replica.exists("/live/n07"), "live delete lost");
}
