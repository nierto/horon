//! Measurement: meaning-addressed windowed semantic queries vs
//! brute-force full scans, at 10k / 50k / 100k entries.
//!
//! Builds a meaning-addressed (format v3) file with clustered user
//! coordinates, opens it in partial (mmap) mode, and reports for top-10
//! semantic queries: entries touched (`last_semantic_scan_count()`) vs the
//! full-scan baseline (= all entries), plus wall-clock per query.
//!
//! Run: GMATH_PROFILE=embedded cargo run --release --example scan_counts

use std::path::PathBuf;
use std::time::Instant;

use horon::{DurabilityMode, Horon, HoronConfig};
use g_math::fixed_point::FixedPoint;

const SEM_DIMS: u8 = 20; // 16 reserved + 4 user

fn coords(user_vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; SEM_DIMS as usize * 16];
    for (d, v) in user_vals.iter().enumerate() {
        let off = (16 + d) * 16;
        out[off..off + 16].copy_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out
}

fn ma_config(partial: bool) -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: SEM_DIMS,
        compression: false,
        auto_compact_threshold: 0,
        partial_reads: partial,
        meaning_addressed: true,
        semantic_bounds: (0.0, 1000.0),
        // Bulk build: the default (Batched) fsyncs the WAL on EVERY append,
        // so a 100k-node load pays ~200k fsyncs (~minutes of pure disk sync).
        // This file is made durable by the final compact() (which writes and
        // syncs the snapshot), so per-entry WAL durability during the build is
        // wasted — relax it. Query opens do no writes, so this is harmless
        // there.
        durability: DurabilityMode::Relaxed,
        ..Default::default()
    }
}

/// Deterministic pseudo-random value in [0, 1).
fn rnd(i: usize, salt: usize) -> f64 {
    let x = (i.wrapping_mul(2654435761) ^ salt.wrapping_mul(40503)) as u32;
    (x as f64) / (u32::MAX as f64)
}

fn main() {
    for &n in &[10_000usize, 50_000, 100_000] {
        let path = PathBuf::from(format!("/tmp/scan_counts_{}.htt", n));
        let _ = std::fs::remove_file(&path);

        // Build: 100 clusters, points jittered ±5 around each center.
        let build_start = Instant::now();
        {
            let gf = Horon::open_with_config(&path, ma_config(false)).unwrap();
            for i in 0..n {
                let cluster = i % 100;
                let cx = 10.0 + (cluster % 10) as f64 * 100.0;
                let cy = 10.0 + (cluster / 10) as f64 * 100.0;
                let c = [
                    cx + rnd(i, 1) * 10.0 - 5.0,
                    cy + rnd(i, 2) * 10.0 - 5.0,
                    rnd(i, 3) * 1000.0,
                    rnd(i, 4) * 1000.0,
                ];
                // Nest under per-cluster parents to keep fan-out bounded.
                // Data-only insert: this file is queried purely by semantic
                // Hilbert windows (meaning-addressed v3), so the ~ms/node
                // hyperbolic embedding that gf.put() builds is never used —
                // skipping it turns a ~hour build into seconds.
                // Data-only insert: this file is queried purely by semantic
                // Hilbert windows (meaning-addressed v3), so the hyperbolic
                // embedding that gf.put() would build is never used.
                let key = format!("/e/c{:03}/{:06}", cluster, i);
                gf.put_data_only(&key, b"x").unwrap();
                gf.set_semantic(&key, coords(&c)).unwrap();
            }
            gf.compact().unwrap(); // writes the v3 Hilbert-ordered snapshot
        }
        let build_secs = build_start.elapsed().as_secs_f64();

        // Query in partial (windowed) mode.
        let gf = Horon::open_with_config(&path, ma_config(true)).unwrap();
        let mut touched_total = 0usize;
        let mut window_time = 0.0f64;
        let queries = 20;
        for q in 0..queries {
            let cluster = (q * 7) % 100;
            let qc = [
                10.0 + (cluster % 10) as f64 * 100.0 + 2.0,
                10.0 + (cluster / 10) as f64 * 100.0 - 2.0,
                500.0,
                500.0,
            ];
            let t = Instant::now();
            let hits = gf.nearest_semantic(&coords(&qc), 10, 16..18).unwrap();
            window_time += t.elapsed().as_secs_f64();
            assert_eq!(hits.len(), 10);
            touched_total += gf.last_semantic_scan_count().unwrap_or(0);
        }
        let avg_touched = touched_total / queries;

        println!(
            "n={:>6}  build+compact {:>6.1}s  windowed: avg {:>6} entries touched ({:>5.2}% of file), {:>8.3} ms/query (brute-force baseline: {} entries = 100%)",
            n,
            build_secs,
            avg_touched,
            100.0 * avg_touched as f64 / n as f64,
            1000.0 * window_time / queries as f64,
            n,
        );

        let _ = std::fs::remove_file(&path);
    }
}
