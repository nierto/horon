//! Recall of meaning-addressed partial-mode k-NN on REALISTIC spatial queries.
//!
//! The adversarial probe (dim_slice_probe) used deliberately scattered marks.
//! This asks the fair question: for ordinary nearest-neighbour queries over
//! the ADDRESSED dims — the case the design targets — does the mmap window
//! return the same top-k as a full exact scan?
//!
//! recall@k = |partial ∩ full| / k, averaged over many random queries.
//!
//!     cargo run --release --example dim_slice_recall

use horon::{DurabilityMode, Horon, HoronConfig};

const SEM_DIMS: u8 = 24; // 16 reserved + 8 user — ALL addressed
const N: usize = 3000;
const K: usize = 10;
const QUERIES: usize = 40;

fn coords(vals: &[f64]) -> Vec<u8> {
    use g_math::fixed_point::FixedPoint;
    let mut out = vec![0u8; SEM_DIMS as usize * 16];
    for (d, v) in vals.iter().enumerate() {
        let off = (16 + d) * 16;
        out[off..off + 16].copy_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out
}

fn cfg(partial: bool) -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: SEM_DIMS,
        compression: false,
        auto_compact_threshold: 0,
        partial_reads: partial,
        meaning_addressed: std::env::var("V2").is_err(),
        semantic_bounds: (0.0, 1.0),
        durability: DurabilityMode::Relaxed,
        ..Default::default()
    }
}

// Deterministic pseudo-random in [0,1) — no rand dependency, reproducible.
fn prng(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64) / (u32::MAX as f64 / 2.0)
}

fn main() {
    let mut path = std::env::temp_dir();
    path.push(format!("horon_recall_{}.htt", std::process::id()));
    let _ = std::fs::remove_file(&path);

    let mut seed = 0x5EED_u64;
    {
        let gf = Horon::open_with_config(&path, cfg(false)).unwrap();
        for i in 0..N {
            let v: Vec<f64> = (0..8).map(|_| prng(&mut seed).fract()).collect();
            let key = format!("/n/{i:05}");
            gf.put(&key, b"x").unwrap();
            gf.set_semantic(&key, coords(&v)).unwrap();
        }
        gf.compact().unwrap();
    }

    let slice = 16..24; // the full addressed range

    // Horon takes an exclusive lock, so the two modes cannot be open at
    // once — collect ground truth first, then reopen in partial mode.
    let mut queries = Vec::new();
    let mut qseed = 0xC0FFEE_u64;
    for _ in 0..QUERIES {
        let q: Vec<f64> = (0..8).map(|_| prng(&mut qseed).fract()).collect();
        queries.push(coords(&q));
    }

    let truths: Vec<Vec<String>> = {
        let full = Horon::open_with_config(&path, cfg(false)).unwrap();
        queries
            .iter()
            .map(|qc| {
                full.nearest_semantic(qc, K, slice.clone())
                    .unwrap()
                    .into_iter()
                    .map(|(k, _)| k)
                    .collect()
            })
            .collect()
    };

    let part = Horon::open_with_config(&path, cfg(true)).unwrap();
    let mut recalls = Vec::new();
    let mut scans = Vec::new();
    for (qc, truth) in queries.iter().zip(&truths) {
        let got = part.nearest_semantic(qc, K, slice.clone()).unwrap();
        scans.push(part.last_semantic_scan_count().unwrap_or(0));
        let tset: std::collections::HashSet<&str> = truth.iter().map(|k| k.as_str()).collect();
        let hits = got.iter().filter(|(k, _)| tset.contains(k.as_str())).count();
        recalls.push(hits as f64 / K as f64);
    }

    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let worst = recalls.iter().cloned().fold(1.0_f64, f64::min);
    let perfect = recalls.iter().filter(|r| **r >= 1.0).count();
    let scan_mean = scans.iter().sum::<usize>() as f64 / scans.len() as f64;

    println!("meaning-addressed partial-mode k-NN, ALL dims addressed (16..24)");
    println!("  {N} nodes, k={K}, {QUERIES} random spatial queries\n");
    println!("  mean recall@{K}:   {:.1}%", mean * 100.0);
    println!("  worst query:      {:.1}%", worst * 100.0);
    println!("  perfect queries:  {perfect}/{QUERIES}");
    println!("  mean scanned:     {:.0} of {N}  ({:.1}%)", scan_mean, scan_mean / N as f64 * 100.0);

    let _ = std::fs::remove_file(&path);
}
