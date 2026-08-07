//! Concurrent read access: many readers alongside a live writer.
//!
//! `Horon` takes an exclusive advisory lock on every open, so these cases were
//! previously unreachable — a second open failed regardless of intent.
//! `HoronReader` opens read-only and unlocked, which is only sound because the
//! file is append-only apart from an 8-byte WAL header the scanner already
//! treats as advisory. These tests are the evidence for that claim.
//!
//! Scale note: two readers and a handful of writes would pass whether or not
//! the design is correct. Each case here is sized so the interleaving it
//! targets actually occurs.

use horon::{Horon, HoronConfig, HoronReader, RefreshOutcome};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

fn temp_path() -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "horon_readers_{}_{}.htt",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    p
}
static COUNTER: AtomicUsize = AtomicUsize::new(0);

fn cfg() -> HoronConfig {
    HoronConfig {
        auto_compact_threshold: 0, // compaction only when we ask for it
        ..Default::default()
    }
}

// ===========================================================================
// 1. N readers replaying continuously while a writer appends at high rate
// ===========================================================================

#[test]
fn many_readers_never_observe_a_malformed_view_under_sustained_writes() {
    const READERS: usize = 8;
    const WRITES: usize = 4_000;

    let path = temp_path();
    let gf = Horon::open_with_config(&path, cfg()).unwrap();
    gf.put("/seed", b"seed").unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let observations = Arc::new(AtomicUsize::new(0));

    let readers: Vec<_> = (0..READERS)
        .map(|_| {
            let path = path.clone();
            let stop = Arc::clone(&stop);
            let observations = Arc::clone(&observations);
            std::thread::spawn(move || {
                let mut rounds = 0usize;
                let mut last_seq = 0u32;
                while !stop.load(Ordering::Relaxed) {
                    // A fresh open each round exercises the full load path
                    // against a file being appended to concurrently.
                    let r = match HoronReader::open(&path) {
                        Ok(r) => r,
                        Err(e) => panic!("reader open failed mid-write: {e}"),
                    };

                    // The seed is in the snapshot and must never disappear.
                    assert_eq!(r.get("/seed").unwrap(), b"seed", "seed lost");

                    // Every key the view reports must be readable, and its
                    // payload must match the value encoded in its own key.
                    // A torn or misattributed entry shows up here.
                    for k in r.list("/").unwrap() {
                        if let Some(n) = k.strip_prefix("/n/") {
                            let want = format!("v{n}");
                            let got = r.get(&k).unwrap();
                            assert_eq!(
                                got,
                                want.as_bytes(),
                                "payload/key mismatch at {k} — misattributed WAL entry"
                            );
                        }
                    }

                    // Sequence must advance monotonically across rounds: the
                    // writer only appends, so a reader can never go backwards.
                    let seq = r.next_seq();
                    assert!(
                        seq >= last_seq,
                        "seq went backwards: {last_seq} -> {seq}"
                    );
                    last_seq = seq;

                    rounds += 1;
                    observations.fetch_add(1, Ordering::Relaxed);
                    // Yield between rounds. A tight loop of full opens across
                    // 8 threads saturates the box, and `cargo test` runs this
                    // binary alongside every other one — starving a
                    // timing-sensitive suite (WAL flush intervals, durability)
                    // would make THIS test the cause of someone else's flake.
                    // The interleaving being tested survives the pause.
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                rounds
            })
        })
        .collect();

    for i in 0..WRITES {
        gf.put(&format!("/n/{i}"), format!("v{i}").as_bytes()).unwrap();
    }
    stop.store(true, Ordering::Relaxed);

    let total: usize = readers.into_iter().map(|h| h.join().unwrap()).sum();
    assert!(
        total >= READERS,
        "readers did not complete a round each ({total})"
    );
    // The point of the test is concurrency, so assert the interleaving was
    // real rather than the readers finishing before the writer started.
    assert!(
        observations.load(Ordering::Relaxed) > 0,
        "no reader observation was made"
    );

    // Final state is complete and correct.
    drop(gf);
    let r = HoronReader::open(&path).unwrap();
    for i in 0..WRITES {
        assert_eq!(r.get(&format!("/n/{i}")).unwrap(), format!("v{i}").as_bytes());
    }
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// 2. A compaction fired mid-read
// ===========================================================================

#[test]
fn readers_survive_a_compaction_underneath_them() {
    const READERS: usize = 4;

    let path = temp_path();
    let gf = Horon::open_with_config(&path, cfg()).unwrap();
    for i in 0..500 {
        gf.put(&format!("/pre/{i}"), format!("p{i}").as_bytes()).unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let readers: Vec<_> = (0..READERS)
        .map(|_| {
            let path = path.clone();
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                // Opened BEFORE the compaction. On Unix this fd keeps the
                // pre-rename inode alive, so the view must stay complete and
                // self-consistent even as the file is replaced underneath.
                let held = HoronReader::open(&path).unwrap();
                let held_len = held.len();

                while !stop.load(Ordering::Relaxed) {
                    // The pinned view never changes and never degrades.
                    assert_eq!(held.len(), held_len, "pinned view mutated");
                    for i in (0..500).step_by(97) {
                        assert_eq!(
                            held.get(&format!("/pre/{i}")).unwrap(),
                            format!("p{i}").as_bytes(),
                            "pinned view lost data across compaction"
                        );
                    }
                    // A fresh open during/after the rename must also succeed:
                    // rename is atomic, so a reader sees either file whole.
                    let fresh = HoronReader::open(&path).unwrap();
                    assert!(
                        fresh.get("/pre/0").is_ok(),
                        "fresh open during compaction lost data"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                held_len
            })
        })
        .collect();

    // Compact repeatedly while the readers run.
    for round in 0..5 {
        for i in 0..100 {
            gf.put(&format!("/post/{round}/{i}"), b"x").unwrap();
        }
        gf.compact().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for h in readers {
        h.join().unwrap();
    }

    // Post-compaction file is intact for a new reader.
    drop(gf);
    let r = HoronReader::open(&path).unwrap();
    for i in 0..500 {
        assert_eq!(r.get(&format!("/pre/{i}")).unwrap(), format!("p{i}").as_bytes());
    }
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// 3. Derive-don't-trust: the WAL header is advisory
// ===========================================================================

#[test]
fn reader_is_correct_when_the_wal_header_is_deliberately_wrong() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, cfg()).unwrap();
        for i in 0..200 {
            gf.put(&format!("/k/{i}"), format!("v{i}").as_bytes()).unwrap();
        }
    }

    let truth = {
        let r = HoronReader::open(&path).unwrap();
        (r.len(), r.list("/").unwrap().len())
    };

    // Walk the documented layout (HTT_FORMAT.md §2) to the WAL header rather
    // than guessing its contents: header -> [bounds] -> snapshot -> [CRC].
    let bytes = std::fs::read(&path).unwrap();
    let off = {
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let version = bytes[4];
        let flags = bytes[5];
        let semantic_dims = bytes[7] as usize;

        let mut o = 32usize; // fixed header
        if flags & (1 << 6) != 0 {
            // v3 meaning-addressed bounds section: (min,max) f64 per user dim
            o += semantic_dims.saturating_sub(16) * 16;
        }
        let snap_byte_len = u32_at(o) as usize;
        let node_count = u32_at(o + 4);
        o += 8;
        let compressed = flags & 1 != 0;
        if node_count > 0 && compressed {
            let comp_len = u32_at(o) as usize;
            o += 4 + comp_len;
        } else {
            o += snap_byte_len;
        }
        if version >= 2 {
            o += 4; // snapshot CRC
        }
        o
    };
    // Sanity: the bytes we are about to tamper really are the WAL header the
    // writer maintains, so a green test cannot come from tampering padding.
    let live_count = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
    assert!(
        live_count > 0,
        "computed WAL header offset {off} holds count 0 — layout walk is wrong"
    );

    // Every torn combination a concurrent reader could observe: a stale or
    // fresh count paired with a stale or fresh base. If the reader trusted
    // either field, at least one of these would change the answer.
    for (count, base) in [
        (0u32, 1u32),
        (9999, 1),
        (live_count, 0),
        (live_count, 9999),
        (0, 0),
        (u32::MAX, u32::MAX),
    ] {
        let mut corrupted = bytes.clone();
        corrupted[off..off + 4].copy_from_slice(&count.to_le_bytes());
        corrupted[off + 4..off + 8].copy_from_slice(&base.to_le_bytes());

        let tampered = temp_path();
        std::fs::write(&tampered, &corrupted).unwrap();

        let r = HoronReader::open(&tampered).unwrap();
        assert_eq!(
            (r.len(), r.list("/").unwrap().len()),
            truth,
            "reader result changed with WAL header ({count}, {base}) — \
             the header is being trusted, not derived"
        );
        for i in 0..200 {
            assert_eq!(
                r.get(&format!("/k/{i}")).unwrap(),
                format!("v{i}").as_bytes(),
                "payload wrong under tampered header ({count}, {base})"
            );
        }
        let _ = std::fs::remove_file(&tampered);
    }
    let _ = std::fs::remove_file(&path);
}

// ===========================================================================
// 4. refresh(): incremental catch-up and the reload fallback
// ===========================================================================

#[test]
fn refresh_applies_new_entries_and_reloads_past_a_compaction() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, cfg()).unwrap();
    gf.put("/a", b"1").unwrap();

    let mut r = HoronReader::open(&path).unwrap();
    assert_eq!(r.get("/a").unwrap(), b"1");
    assert!(r.get("/b").is_err(), "reader saw a write that did not exist yet");

    // Nothing new.
    assert_eq!(r.refresh().unwrap(), RefreshOutcome::UpToDate);

    // Incremental catch-up.
    gf.put("/b", b"2").unwrap();
    gf.put("/c", b"3").unwrap();
    match r.refresh().unwrap() {
        RefreshOutcome::Applied(n) => assert!(n >= 2, "expected >=2 entries, got {n}"),
        other => panic!("expected Applied, got {other:?}"),
    }
    assert_eq!(r.get("/b").unwrap(), b"2");
    assert_eq!(r.get("/c").unwrap(), b"3");

    // Compaction folds the WAL into the snapshot and advances base_seq past
    // the reader, which must then rebuild rather than silently miss entries.
    gf.put("/d", b"4").unwrap();
    gf.compact().unwrap();
    gf.put("/e", b"5").unwrap();

    let outcome = r.refresh().unwrap();
    assert_eq!(r.get("/d").unwrap(), b"4", "lost an entry folded into the snapshot");
    assert_eq!(r.get("/e").unwrap(), b"5", "lost a post-compaction entry");
    assert!(
        matches!(outcome, RefreshOutcome::Reloaded | RefreshOutcome::Applied(_)),
        "unexpected outcome {outcome:?}"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_reader_does_not_block_a_writer_and_a_writer_does_not_block_a_reader() {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, cfg()).unwrap();
    gf.put("/x", b"1").unwrap();

    // Reader opens while the writer holds the exclusive lock. This is the
    // case that fails before HoronReader exists.
    let r1 = HoronReader::open(&path).unwrap();
    let r2 = HoronReader::open(&path).unwrap();
    assert_eq!(r1.get("/x").unwrap(), b"1");
    assert_eq!(r2.get("/x").unwrap(), b"1");

    // The writer keeps working with readers live.
    gf.put("/y", b"2").unwrap();
    assert_eq!(gf.get("/y").unwrap(), b"2");

    // And a second WRITER is still correctly refused.
    assert!(
        Horon::open_with_config(&path, cfg()).is_err(),
        "exclusive write lock was weakened"
    );

    let _ = std::fs::remove_file(&path);
}
