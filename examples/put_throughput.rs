//! Write-throughput gate for ancestor-materialization in the WAL.
//!
//! Measures `put()` throughput across tree shapes, from realistic to
//! adversarial. Run before and after a write-path change on the same
//! machine and compare:
//!
//! ```text
//! cargo run --release --example put_throughput
//! ```
//!
//! The gate cares about the realistic shapes (flat, catalog, filesystem);
//! the deep shapes exist to expose the worst case, not to veto.

use std::path::PathBuf;
use std::time::Instant;

use horon::{Horon, HoronConfig};

fn temp_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("htt_put_bench_{}_{}.htt", tag, std::process::id()))
}

fn bench(label: &str, tag: &str, keys: &[String]) {
    let path = temp_path(tag);
    let _ = std::fs::remove_file(&path);
    let gf = Horon::open_with_config(&path, HoronConfig {
        semantic_dims: 20,
        compression: true,
        auto_compact_threshold: 0,
        ..Default::default()
    })
    .unwrap();

    let start = Instant::now();
    for key in keys {
        gf.put(key, b"benchmark-payload-0123456789").unwrap();
    }
    gf.flush().unwrap();
    let elapsed = start.elapsed();

    let total_nodes = gf.list("/").unwrap().len();
    let file_bytes = std::fs::metadata(&path).unwrap().len();
    let per_op = elapsed.as_secs_f64() / keys.len() as f64;
    println!(
        "{:34} {:6} puts  {:7} nodes  {:9.0} puts/s  {:7.1} µs/put  file {:8} B",
        label,
        keys.len(),
        total_nodes,
        keys.len() as f64 / elapsed.as_secs_f64(),
        per_op * 1e6,
        file_bytes,
    );
    drop(gf);
    let _ = std::fs::remove_file(&path);
}

fn main() {
    let n = 3000usize;
    println!("put_throughput — {} puts per shape, semantic_dims=20, zstd on, WAL only (no compaction)\n", n);

    bench(
        "flat            /x/leaf_N",
        "flat",
        &(0..n).map(|i| format!("/x/leaf_{:05}", i)).collect::<Vec<_>>(),
    );
    bench(
        "catalog         /c/cat_16/item_N",
        "catalog",
        &(0..n).map(|i| format!("/c/cat_{:02}/item_{:05}", i % 16, i)).collect::<Vec<_>>(),
    );
    bench(
        "filesystem      /f/d_6/s_5/file_N",
        "fs",
        &(0..n).map(|i| format!("/f/d_{}/s_{}/file_{:05}", i % 6, (i / 6) % 5, i)).collect::<Vec<_>>(),
    );
    bench(
        "deep-shared     /a/b/c/d/e_N/leaf_N",
        "deep",
        &(0..n).map(|i| format!("/a/b/c/d/e_{:04}/leaf_{:04}", i, i)).collect::<Vec<_>>(),
    );
    bench(
        "deep-unique(8)  /r_N/l2/../leaf   [adversarial]",
        "worst",
        &(0..n).map(|i| format!("/r_{:04}/l2/l3/l4/l5/l6/l7/leaf", i)).collect::<Vec<_>>(),
    );
}
