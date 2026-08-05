//! =============================================================================
//! Horon Throughput Benchmarks
//! =============================================================================
//!
//! Isolates Horon persistence costs from the geometric engine's costs:
//!   - Pure WAL serialization (no Store, just byte format)
//!   - Horon overhead vs raw Store (marginal cost of persistence)
//!   - Cold-start recovery (WAL replay vs snapshot load)
//!   - Compaction (raw vs zstd)
//!   - Read/query throughput on pre-built trees
//!   - GACL computation overhead
//!
//! Two independent per-write costs to keep separate when reading these numbers:
//!
//!  1. ENGINE GEOMETRY (`Store::put` / `Horon::put`): building the Sarkar
//!     embedding + VP-tree + Klein point + power-diagram cell for the new node.
//!     Measured here at ~6 ms/node (release, with wide sibling fan-out; it
//!     scales with sibling count and depth). `put_data_only` skips ALL of it
//!     and costs ~8 µs/node — three orders of magnitude cheaper. This cost is
//!     pure in-memory engine; it involves no disk.
//!
//!  2. Horon DURABILITY (WAL append): under the default `DurabilityMode::Batched`
//!     with `wal_batch_size: 0`, EVERY append fsyncs the WAL — ~9 ms/append on a
//!     typical SSD, independent of geometry. A node write is one append; a
//!     `set_semantic` is another. So a naive bulk load pays ~2 fsyncs/node on
//!     top of the geometry, and the fsyncs usually dominate.
//!
//! For bulk loading, set `durability: DurabilityMode::Relaxed` (a final
//! `compact()` writes the durable, fsynced snapshot) and use `put_data_only`
//! when the file is queried only by meaning (semantic / meaning-addressed).
//! That combination is what takes a 100k-node build from ~30 min to ~2 s — see
//! `examples/scan_counts.rs`. Benchmarks here use small node counts and
//! pre-built trees to keep runtimes sane.
//!
//! Run with: GMATH_PROFILE=embedded cargo bench --bench throughput

use std::fs;
use std::io::Cursor;
use std::path::PathBuf;

/// Zero-dim plain semantic layout for tail-less bench entries.
const L0: horon::quant::SemLayout = horon::quant::SemLayout { dims: 0, quantized: false, gacl: false };

use criterion::{
    black_box, criterion_group, criterion_main,
    Criterion, BenchmarkId, Throughput,
};

use horon::{Horon, HoronConfig};
use horon::format::*;
use horon::snapshot::NodeEntry;
use horon::wal::{WalEntry, WalPayload};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn config_raw() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

fn config_zstd() -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: 0,
        compression: true,
        auto_compact_threshold: 0,
        ..Default::default()
    }
}

/// Pre-populate a Horon with `n` flat nodes. Returns path.
fn prepopulate(n: usize, config: HoronConfig) -> PathBuf {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config).unwrap();
    for i in 0..n {
        gf.put(&format!("/node_{}", i), format!("data_{}", i).as_bytes()).unwrap();
    }
    gf.flush().unwrap();
    drop(gf);
    path
}

/// Pre-populate and compact. Returns path.
fn prepopulate_compacted(n: usize, config: HoronConfig) -> PathBuf {
    let path = temp_path();
    let gf = Horon::open_with_config(&path, config).unwrap();
    for i in 0..n {
        gf.put(&format!("/node_{}", i), format!("data_{}", i).as_bytes()).unwrap();
    }
    gf.compact().unwrap();
    drop(gf);
    path
}

// ===========================================================================
// GROUP 1: Pure WAL Serialization (no Store — isolates format overhead)
// ===========================================================================

fn bench_wal_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_serialization");

    // Single INSERT entry: serialize to buffer
    let insert_entry = WalEntry {
        seq: 1,
        op: OP_INSERT,
        key: "/test/node".to_string(),
        payload: WalPayload::Insert(NodeEntry {
            key: "/test/node".to_string(),
            data: b"hello world, this is some payload data".to_vec(),
            metadata: vec![
                ("author".to_string(), "alice".to_string()),
                ("type".to_string(), "text".to_string()),
            ],
            semantic_coords: vec![],
        }),
    };

    group.bench_function("write_insert_entry", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(256);
            insert_entry.write_to(&mut buf, &L0).unwrap();
            black_box(buf.len());
        })
    });

    // Single DELETE entry
    let delete_entry = WalEntry {
        seq: 2, op: OP_DELETE, key: "/test/node".to_string(),
        payload: WalPayload::Delete,
    };

    group.bench_function("write_delete_entry", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(64);
            delete_entry.write_to(&mut buf, &L0).unwrap();
            black_box(buf.len());
        })
    });

    // SET_META entry
    let meta_entry = WalEntry {
        seq: 3, op: OP_SET_META, key: "/test/node".to_string(),
        payload: WalPayload::SetMeta {
            meta_key: "author".to_string(),
            meta_value: "alice".to_string(),
        },
    };

    group.bench_function("write_set_meta_entry", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(128);
            meta_entry.write_to(&mut buf, &L0).unwrap();
            black_box(buf.len());
        })
    });

    // INSERT with 16 semantic dims (256 extra bytes)
    let sem_entry = WalEntry {
        seq: 1,
        op: OP_INSERT,
        key: "/semantic/node".to_string(),
        payload: WalPayload::Insert(NodeEntry {
            key: "/semantic/node".to_string(),
            data: b"payload".to_vec(),
            metadata: vec![],
            semantic_coords: vec![0u8; 256], // 16 dims × 16 bytes
        }),
    };

    group.bench_function("write_insert_16sem", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(512);
            sem_entry.write_to(&mut buf, &L0).unwrap();
            black_box(buf.len());
        })
    });

    // Batch: serialize 100 INSERT entries
    let entries: Vec<WalEntry> = (0..100).map(|i| WalEntry {
        seq: i,
        op: OP_INSERT,
        key: format!("/batch/node_{}", i),
        payload: WalPayload::Insert(NodeEntry {
            key: format!("/batch/node_{}", i),
            data: format!("data for node {}", i).into_bytes(),
            metadata: vec![],
            semantic_coords: vec![],
        }),
    }).collect();

    group.throughput(Throughput::Elements(100));
    group.bench_function("write_100_inserts", |b| {
        b.iter(|| {
            let mut buf = Vec::with_capacity(8192);
            for e in &entries {
                e.write_to(&mut buf, &L0).unwrap();
            }
            black_box(buf.len());
        })
    });

    // Read back 100 entries
    let mut serialized = Vec::new();
    for e in &entries {
        e.write_to(&mut serialized, &L0).unwrap();
    }

    group.bench_function("read_100_inserts", |b| {
        b.iter(|| {
            let mut cursor = Cursor::new(&serialized);
            for _ in 0..100 {
                let e = WalEntry::read_from(&mut cursor, &L0).unwrap().unwrap();
                black_box(&e);
            }
        })
    });

    group.finish();
}

// ===========================================================================
// GROUP 2: Snapshot Serialization (isolated)
// ===========================================================================

fn bench_snapshot_serialization(c: &mut Criterion) {
    use horon::snapshot;

    let mut group = c.benchmark_group("snapshot_serialization");

    for &n in &[10, 50, 100, 500] {
        let entries: Vec<NodeEntry> = (0..n).map(|i| NodeEntry {
            key: format!("/node_{}", i),
            data: format!("data for node {} with some content", i).into_bytes(),
            metadata: vec![("idx".to_string(), i.to_string())],
            semantic_coords: vec![],
        }).collect();

        group.throughput(Throughput::Elements(n as u64));

        // Write uncompressed
        group.bench_with_input(
            BenchmarkId::new("write_raw", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut buf = Vec::with_capacity(n * 100);
                    snapshot::write_snapshot(&mut buf, &entries, false, true, &L0).unwrap();
                    black_box(buf.len());
                })
            },
        );

        // Write zstd
        group.bench_with_input(
            BenchmarkId::new("write_zstd", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut buf = Vec::with_capacity(n * 100);
                    snapshot::write_snapshot(&mut buf, &entries, true, true, &L0).unwrap();
                    black_box(buf.len());
                })
            },
        );

        // Read uncompressed
        let mut raw_buf = Vec::new();
        snapshot::write_snapshot(&mut raw_buf, &entries, false, true, &L0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("read_raw", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut cursor = Cursor::new(&raw_buf);
                    let parsed = snapshot::read_snapshot(&mut cursor, false, &L0, true).unwrap();
                    black_box(parsed.len());
                })
            },
        );

        // Read compressed
        let mut zstd_buf = Vec::new();
        snapshot::write_snapshot(&mut zstd_buf, &entries, true, true, &L0).unwrap();

        group.bench_with_input(
            BenchmarkId::new("read_zstd", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut cursor = Cursor::new(&zstd_buf);
                    let parsed = snapshot::read_snapshot(&mut cursor, true, &L0, true).unwrap();
                    black_box(parsed.len());
                })
            },
        );
    }

    group.finish();
}

// ===========================================================================
// GROUP 3: End-to-End Insert (Store + WAL — shows full cost)
// ===========================================================================

fn bench_e2e_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("e2e_insert");
    group.sample_size(10);

    // Full-stack insert: Store::put + WAL append
    for &n in &[10, 25, 50] {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("full_stack", n),
            &n,
            |b, &n| {
                b.iter_with_setup(
                    || {
                        let path = temp_path();
                        Horon::open_with_config(&path, config_raw()).unwrap()
                    },
                    |gf| {
                        for i in 0..n {
                            gf.put(&format!("/n{}", i), b"data").unwrap();
                        }
                        black_box(&gf);
                    },
                );
            },
        );
    }

    // Marginal insert cost at different tree sizes
    for &base in &[10, 50, 100] {
        let path = temp_path();
        let gf = Horon::open_with_config(&path, config_raw()).unwrap();
        for i in 0..base {
            gf.put(&format!("/pre_{}", i), b"x").unwrap();
        }

        let mut idx = base;
        group.bench_with_input(
            BenchmarkId::new("marginal_at", base),
            &base,
            |b, _| {
                b.iter(|| {
                    gf.put(&format!("/m{}", idx), b"x").unwrap();
                    idx += 1;
                    black_box(&gf);
                })
            },
        );
    }

    group.finish();
}

// ===========================================================================
// GROUP 4: Cold-Start Recovery
// ===========================================================================

fn bench_cold_start(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_start");
    group.sample_size(10);

    for &n in &[10, 50, 100] {
        // WAL replay
        let wal_path = prepopulate(n, config_raw());
        group.bench_with_input(
            BenchmarkId::new("wal_replay_raw", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let gf = Horon::open(black_box(&wal_path)).unwrap();
                    black_box(gf.len());
                })
            },
        );

        // Snapshot load (raw)
        let snap_path = prepopulate_compacted(n, config_raw());
        group.bench_with_input(
            BenchmarkId::new("snapshot_raw", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let gf = Horon::open(black_box(&snap_path)).unwrap();
                    black_box(gf.len());
                })
            },
        );

        // Snapshot load (zstd)
        let zstd_path = prepopulate_compacted(n, config_zstd());
        group.bench_with_input(
            BenchmarkId::new("snapshot_zstd", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let gf = Horon::open(black_box(&zstd_path)).unwrap();
                    black_box(gf.len());
                })
            },
        );
    }

    group.finish();
}

// ===========================================================================
// GROUP 5: Compaction
// ===========================================================================

fn bench_compaction(c: &mut Criterion) {
    let mut group = c.benchmark_group("compaction");
    group.sample_size(10);

    for &n in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("raw", n),
            &n,
            |b, &n| {
                b.iter_with_setup(
                    || {
                        let path = temp_path();
                        let gf = Horon::open_with_config(&path, config_raw()).unwrap();
                        for i in 0..n {
                            gf.put(&format!("/n{}", i), format!("d{}", i).as_bytes()).unwrap();
                        }
                        gf
                    },
                    |gf| {
                        gf.compact().unwrap();
                        black_box(&gf);
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("zstd", n),
            &n,
            |b, &n| {
                b.iter_with_setup(
                    || {
                        let path = temp_path();
                        let gf = Horon::open_with_config(&path, config_zstd()).unwrap();
                        for i in 0..n {
                            gf.put(&format!("/n{}", i), format!("d{}", i).as_bytes()).unwrap();
                        }
                        gf
                    },
                    |gf| {
                        gf.compact().unwrap();
                        black_box(&gf);
                    },
                );
            },
        );
    }

    group.finish();
}

// ===========================================================================
// GROUP 6: Read Throughput (hot path — in-memory, no disk)
// ===========================================================================

fn bench_read_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_throughput");

    for &n in &[50, 100, 500] {
        let path = temp_path();
        let gf = Horon::open_with_config(&path, config_raw()).unwrap();
        for i in 0..n {
            gf.put(&format!("/node_{}", i), format!("data_{}", i).as_bytes()).unwrap();
        }

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("get", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n {
                        black_box(gf.get(&format!("/node_{}", i)).unwrap());
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("exists", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n {
                        black_box(gf.exists(&format!("/node_{}", i)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("get_meta", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n {
                        black_box(gf.get_meta(&format!("/node_{}", i)).unwrap());
                    }
                })
            },
        );
    }

    group.finish();
}

// ===========================================================================
// GROUP 7: Spatial Query Latency (pre-built tree)
// ===========================================================================

fn bench_spatial(c: &mut Criterion) {
    use g_math::fixed_point::FixedPoint;

    let mut group = c.benchmark_group("spatial");
    group.sample_size(20);

    // Query points are fixed point, like every coordinate the store holds.
    // Hoisted out of the timed closures so the measurement is the query, not
    // a float conversion no real caller performs.
    let origin = [FixedPoint::ZERO; 4];
    let off_center = [
        FixedPoint::from_f64(0.3),
        FixedPoint::from_f64(0.2),
        FixedPoint::from_f64(-0.1),
        FixedPoint::from_f64(0.1),
    ];

    for &n in &[20, 50, 100] {
        let path = temp_path();
        let gf = Horon::open_with_config(&path, config_raw()).unwrap();
        for i in 0..n {
            gf.put(&format!("/n{}", i), b"d").unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("nearest_origin", n),
            &n,
            |b, _| b.iter(|| black_box(gf.nearest(&origin).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("nearest_off_center", n),
            &n,
            |b, _| b.iter(|| black_box(gf.nearest(&off_center).unwrap())),
        );

        group.bench_with_input(
            BenchmarkId::new("neighbors_k3", n),
            &n,
            |b, _| b.iter(|| black_box(gf.neighbors("/n0", 3).unwrap())),
        );
    }

    group.finish();
}

// ===========================================================================
// GROUP 8: GACL Overhead (pure computation, no I/O)
// ===========================================================================

fn bench_gacl(c: &mut Criterion) {
    use horon::gacl::{NodeAccessBands, AccessBand, Credentials};
    use g_math::fixed_point::FixedPoint;

    let mut group = c.benchmark_group("gacl");

    let band = AccessBand::from_f64(0.3, 0.8);
    let cred = FixedPoint::from_f64(0.5);
    group.bench_function("band_permits", |b| {
        b.iter(|| black_box(band.permits(black_box(cred))))
    });

    let creds = Credentials::root();
    let bands = NodeAccessBands {
        read: AccessBand::from_f64(0.5, 1.0),
        write: AccessBand::from_f64(0.7, 1.0),
        exec: AccessBand::from_f64(0.0, 1.0),
        domain: AccessBand::from_f64(0.3, 0.8),
        classification: AccessBand::from_f64(0.5, 1.0),
        identity: AccessBand::open(),
    };
    group.bench_function("can_access_6_bands", |b| {
        b.iter(|| black_box(creds.can_access(black_box(&bands))))
    });

    let parent = NodeAccessBands {
        read: AccessBand::from_f64(0.3, 0.9),
        write: AccessBand::from_f64(0.5, 1.0),
        exec: AccessBand::open(),
        domain: AccessBand::from_f64(0.2, 0.8),
        classification: AccessBand::open(),
        identity: AccessBand::open(),
    };
    let child = NodeAccessBands::public();
    group.bench_function("narrow_inheritance", |b| {
        b.iter(|| black_box(child.narrow(black_box(&parent))))
    });

    group.bench_function("to_semantic_bytes_16", |b| {
        b.iter(|| black_box(bands.to_semantic_bytes(16)))
    });

    let bytes = bands.to_semantic_bytes(16);
    group.bench_function("from_semantic_bytes_16", |b| {
        b.iter(|| black_box(NodeAccessBands::from_semantic_bytes(black_box(&bytes))))
    });

    let groups: Vec<Credentials> = (0..5).map(|i| {
        let v = (i as f64 + 1.0) / 6.0;
        Credentials {
            read: FixedPoint::from_f64(v),
            write: FixedPoint::from_f64(v * 0.8),
            exec: FixedPoint::from_f64(v * 0.5),
            domain: FixedPoint::from_f64(v * 0.9),
            classification: FixedPoint::from_f64(v * 0.7),
            identity: FixedPoint::from_f64(0.42),
        }
    }).collect();

    group.bench_function("from_groups_5", |b| {
        b.iter(|| black_box(Credentials::from_groups(black_box(&groups))))
    });

    group.finish();
}

// ===========================================================================
// GROUP 9: File Size Report (not timed — prints compression ratios)
// ===========================================================================

fn bench_file_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("file_size");
    group.sample_size(10);

    // Just measure compaction time; file sizes logged to stderr
    for &n in &[50, 100] {
        let raw_path = prepopulate_compacted(n, config_raw());
        let zstd_path = prepopulate_compacted(n, config_zstd());

        let raw_size = fs::metadata(&raw_path).unwrap().len();
        let zstd_size = fs::metadata(&zstd_path).unwrap().len();

        eprintln!(
            "  {} nodes: raw={} bytes, zstd={} bytes, ratio={:.1}x",
            n, raw_size, zstd_size,
            raw_size as f64 / zstd_size as f64
        );

        // Bench opening each to compare load time
        group.bench_with_input(
            BenchmarkId::new("open_raw", n),
            &n,
            |b, _| b.iter(|| {
                let gf = Horon::open(black_box(&raw_path)).unwrap();
                black_box(gf.len());
            }),
        );

        group.bench_with_input(
            BenchmarkId::new("open_zstd", n),
            &n,
            |b, _| b.iter(|| {
                let gf = Horon::open(black_box(&zstd_path)).unwrap();
                black_box(gf.len());
            }),
        );
    }

    group.finish();
}

// ===========================================================================
// Register
// ===========================================================================

criterion_group!(
    benches,
    bench_wal_serialization,
    bench_snapshot_serialization,
    bench_e2e_insert,
    bench_cold_start,
    bench_compaction,
    bench_read_throughput,
    bench_spatial,
    bench_gacl,
    bench_file_sizes,
);
criterion_main!(benches);
