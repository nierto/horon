//! =============================================================================
//! Hilbert Locality & Lazy Geometry Integration Tests
//! =============================================================================
//!
//! Validates three things:
//!   1. lazy_geometry opens existing .htt files correctly (backward compat)
//!   2. Hilbert mapping preserves locality for real semantic coordinates
//!   3. The full premise: semantic coords from .htt → Hilbert index → memory
//!      locality (nearby meaning = nearby storage offsets)
//!
//! Run with: cargo test --test hilbert_locality -- --nocapture

use std::path::PathBuf;

use horon::{Horon, HoronConfig};
use horon::hilbert::{HilbertMapper, HilbertIndex};
use tempfile::NamedTempFile;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("catalog.htt")
}

fn config_lazy() -> HoronConfig {
    HoronConfig {
        lazy_geometry: true,
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

fn config_lazy_semantics(sem: u8) -> HoronConfig {
    HoronConfig {
        dimension: 4,
        semantic_dims: sem,
        compression: false,
        auto_compact_threshold: 0,
        lazy_geometry: true,
        ..Default::default()
    }
}

/// Parse semantic coord bytes (16 bytes per dim) into f64 values.
/// Each dim is i128 LE Q64.64 — we just convert to f64 for comparison.
fn semantic_bytes_to_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(16)
        .map(|chunk| {
            let raw = i128::from_le_bytes(chunk.try_into().unwrap());
            // Q64.64: divide by 2^64
            raw as f64 / (1u128 << 64) as f64
        })
        .collect()
}

/// Normalize semantic f64 values to [0, 1] range for Hilbert mapping.
/// Returns (normalized_values, min, max) per dimension.
fn normalize_coords(all_coords: &[Vec<f64>], dim_range: std::ops::Range<usize>) -> Vec<Vec<f64>> {
    if all_coords.is_empty() {
        return vec![];
    }

    let ndims = dim_range.len();

    // Find min/max per dimension
    let mut mins = vec![f64::MAX; ndims];
    let mut maxs = vec![f64::MIN; ndims];
    for coords in all_coords {
        for (i, dim_idx) in dim_range.clone().enumerate() {
            if let Some(&v) = coords.get(dim_idx) {
                mins[i] = mins[i].min(v);
                maxs[i] = maxs[i].max(v);
            }
        }
    }

    // Normalize to [0, 1]
    all_coords
        .iter()
        .map(|coords| {
            (0..ndims)
                .map(|i| {
                    let dim_idx = dim_range.start + i;
                    let v = coords.get(dim_idx).copied().unwrap_or(0.0);
                    let range = maxs[i] - mins[i];
                    if range < 1e-12 {
                        0.5
                    } else {
                        (v - mins[i]) / range
                    }
                })
                .collect()
        })
        .collect()
}

/// Euclidean distance between two coordinate slices.
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ===========================================================================
// 1. Backward compatibility: existing .htt opens with lazy_geometry
// ===========================================================================

#[test]
fn existing_htt_opens_with_lazy_geometry() {
    let src = fixture_path();
    let path = temp_path();
    std::fs::copy(&src, &path).unwrap();

    let gf = Horon::open_with_config(&path, config_lazy()).unwrap();
    let n = gf.len();
    println!("Lazy open: {} nodes", n);
    assert!(n > 0, "lazy_geometry should still load all nodes");
}

#[test]
fn lazy_geometry_data_accessible_and_consistent() {
    // Verify lazy_geometry loads data correctly from an existing .htt file:
    // every key should have accessible (non-empty) data, and reopening
    // the same file should produce identical results (deterministic).
    let src = fixture_path();
    let path1 = temp_path();
    let path2 = temp_path();
    std::fs::copy(&src, &path1).unwrap();
    std::fs::copy(&src, &path2).unwrap();

    let gf1 = Horon::open_with_config(&path1, config_lazy()).unwrap();
    let gf2 = Horon::open_with_config(&path2, config_lazy()).unwrap();

    assert_eq!(gf1.len(), gf2.len(), "two lazy opens must produce same node count");

    let keys = gf1.list("/").unwrap();
    println!("Checking {} keys across two lazy opens", keys.len());

    let mut data_nodes = 0;
    let mut semantic_nodes = 0;
    for key in &keys {
        let d1 = gf1.get(key).unwrap();
        let d2 = gf2.get(key).unwrap();
        assert_eq!(d1, d2, "data mismatch between two lazy opens for key '{}'", key);
        if !d1.is_empty() {
            data_nodes += 1;
        }

        let s1 = gf1.get_semantic(key).unwrap();
        let s2 = gf2.get_semantic(key).unwrap();
        assert_eq!(s1, s2, "semantic mismatch between two lazy opens for key '{}'", key);
        if !s1.is_empty() {
            semantic_nodes += 1;
        }
    }
    println!(
        "Verified: {} total keys, {} with data, {} with semantic coords",
        keys.len(), data_nodes, semantic_nodes
    );
    assert!(data_nodes > 100, "expected substantial data nodes in fixture");
    assert!(semantic_nodes > 100, "expected substantial semantic nodes in fixture");
}

// ===========================================================================
// 1b. Sarkar reconstruction: compact → reopen produces identical data
// ===========================================================================

/// Build `groups * per_group` leaves under shared parents plus `deep` deeper
/// nodes, compact, then assert a full Sarkar reopen matches a lazy reopen for
/// every key, in both payload and semantic coordinates.
///
/// The bug: DashMap iteration order in snapshots caused child_index
/// misassignment on reload, corrupting key→data mappings.
/// The fix: store _child_index in metadata, sort snapshot by depth, use
/// put_positioned on replay.
///
/// Shared by the every-push regression test and the full-scale one so the two
/// cannot drift apart; only the fixture size differs.
fn assert_sarkar_reconstruction(groups: usize, per_group: usize, deep: usize) {
    let path = temp_path();
    let sem_dims: u8 = 20;

    // Phase 1: Create file with many siblings under the same parents
    {
        let config = HoronConfig {
            dimension: 4,
            semantic_dims: sem_dims,
            compression: true,
            auto_compact_threshold: 0,
            ..Default::default()
        };
        let gf = Horon::open_with_config(&path, config).unwrap();

        // Wide fan-out under shared parents is the trigger; total node count is
        // what makes the iteration order actually diverge. Every payload and
        // every semantic value is unique, so any key->data or key->coords
        // crossover is detectable.
        for g in 0..groups {
            for i in 0..per_group {
                let idx = g * per_group + i;
                let key = format!("/group_{:02}/item_{:04}", g, i);
                gf.put(&key, format!("data_{:05}", idx).as_bytes()).unwrap();

                let mut coords = vec![0u8; sem_dims as usize * 16];
                let dim16_offset = 16 * 16;
                if dim16_offset + 16 <= coords.len() {
                    // +1: all-zero coordinates encode "not set" and are rejected.
                    let val = ((idx + 1) as i128) << 64;
                    coords[dim16_offset..dim16_offset + 16].copy_from_slice(&val.to_le_bytes());
                }
                gf.set_semantic(&key, coords).unwrap();
            }
        }

        // Deeper nodes exercise the depth sort in the replay path.
        for i in 0..deep {
            let key = format!("/group_{:02}/item_{:04}/detail_{}", i % groups, i % per_group, i);
            gf.put(&key, format!("detail_{:04}", i).as_bytes()).unwrap();
        }

        // Compact — this triggers snapshot write with _child_index metadata
        gf.compact().unwrap();
    }

    // Phase 2: Reopen with full Sarkar and lazy, compare
    let path_full = temp_path();
    let path_lazy = temp_path();
    std::fs::copy(&path, &path_full).unwrap();
    std::fs::copy(&path, &path_lazy).unwrap();

    let gf_full = Horon::open(&path_full).unwrap();
    let gf_lazy = Horon::open_with_config(&path_lazy, HoronConfig {
        lazy_geometry: true,
        ..Default::default()
    }).unwrap();

    assert_eq!(
        gf_full.len(), gf_lazy.len(),
        "full and lazy should have same node count"
    );

    let keys_lazy = gf_lazy.list("/").unwrap();

    let mut mismatches = 0;
    for key in &keys_lazy {
        let data_full = gf_full.get(key).unwrap();
        let data_lazy = gf_lazy.get(key).unwrap();
        if data_full != data_lazy {
            mismatches += 1;
            println!(
                "MISMATCH: key={}, full={:?}, lazy={:?}",
                key,
                String::from_utf8_lossy(&data_full),
                String::from_utf8_lossy(&data_lazy),
            );
        }

        // Also check semantic coords
        let sem_full = gf_full.get_semantic(key).unwrap();
        let sem_lazy = gf_lazy.get_semantic(key).unwrap();
        if sem_full != sem_lazy {
            mismatches += 1;
            println!("SEMANTIC MISMATCH: key={}", key);
        }
    }

    println!(
        "Sarkar reconstruction: {} keys, {} mismatches",
        keys_lazy.len(),
        mismatches
    );
    assert_eq!(
        mismatches, 0,
        "full Sarkar reopen should match lazy reopen after compaction with _child_index fix"
    );
}

/// Every-push guard. Small enough for CI, and it does catch a wholesale
/// regression of the replay path, but see the full-scale test below: this size
/// never reproduced the original bug on its own.
#[test]
fn sarkar_reconstruction_after_compaction() {
    assert_sarkar_reconstruction(2, 25, 10); // 50 leaves
}

/// The scale the corruption actually manifested at. It was silent above ~7370
/// nodes, where DashMap iteration order diverges from insertion order widely
/// enough to misassign child_index; small fixtures pass whether or not the fix
/// is present, so this is the test that carries the real evidence.
///
/// Ignored by default because it takes roughly six minutes. Run it before any
/// release that touches compaction, snapshot layout, or Sarkar replay:
///
/// ```text
/// cargo test --release --test hilbert_locality -- --ignored --nocapture
/// ```
#[test]
#[ignore = "10_000 nodes, ~6 min; the scale the bug manifested at. Run with --ignored"]
fn sarkar_reconstruction_at_original_bug_scale() {
    assert_sarkar_reconstruction(20, 500, 200); // 10_000 leaves
}

// ===========================================================================
// 2. Lazy geometry: semantic queries work, spatial queries degrade gracefully
// ===========================================================================

#[test]
fn lazy_geometry_semantic_queries_work() {
    let path = temp_path();
    let sem_dims: u8 = 20;

    // Create a file with semantic coords
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(sem_dims)).unwrap();
        gf.put("/a", b"alpha").unwrap();
        gf.put("/b", b"bravo").unwrap();
        gf.put("/c", b"charlie").unwrap();

        // Set semantic coords: 20 dims × 16 bytes = 320 bytes each
        let mut coords_a = vec![0u8; (sem_dims as usize) * 16];
        let mut coords_b = vec![0u8; (sem_dims as usize) * 16];
        let mut coords_c = vec![0u8; (sem_dims as usize) * 16];

        // Make /a and /b close, /c far away (in dim 16+)
        // Dim 16 (byte offset 16*16 = 256): a=10, b=12, c=200
        let dim16_offset = 16 * 16; // skip GACL dims 0-15
        if dim16_offset + 16 <= coords_a.len() {
            let val_a = (10i128) << 64;
            let val_b = (12i128) << 64;
            let val_c = (200i128) << 64;
            coords_a[dim16_offset..dim16_offset + 16].copy_from_slice(&val_a.to_le_bytes());
            coords_b[dim16_offset..dim16_offset + 16].copy_from_slice(&val_b.to_le_bytes());
            coords_c[dim16_offset..dim16_offset + 16].copy_from_slice(&val_c.to_le_bytes());
        }

        gf.set_semantic("/a", coords_a).unwrap();
        gf.set_semantic("/b", coords_b).unwrap();
        gf.set_semantic("/c", coords_c).unwrap();
    }

    // Reopen with lazy_geometry
    let gf = Horon::open_with_config(&path, config_lazy_semantics(sem_dims)).unwrap();
    assert_eq!(gf.len(), 3);

    // Semantic neighbor query should still work
    let neighbors = gf.neighbors_semantic("/a", 2, 16..20).unwrap();
    println!("neighbors of /a: {:?}", neighbors);
    assert!(!neighbors.is_empty(), "semantic queries should work with lazy_geometry");

    // /b should be closer to /a than /c
    if neighbors.len() >= 2 {
        let first_key = &neighbors[0].0;
        assert_eq!(first_key, "/b", "/b should be nearest to /a in semantic space");
    }
}

// ===========================================================================
// 3. The premise: Hilbert mapping preserves semantic locality
// ===========================================================================

#[test]
fn hilbert_locality_synthetic_clusters() {
    // Create 3 clusters in 4D. Hilbert curves preserve locality statistically
    // (nearby points tend to get nearby indices) but have "seams" where
    // adjacent cells in space get distant indices. So we test average behavior:
    // average intra-cluster distance should be much less than average inter-cluster.
    let mapper = HilbertMapper::new(4, 12);

    let cluster_a: Vec<[f64; 4]> = vec![
        [0.10, 0.10, 0.10, 0.10],
        [0.11, 0.12, 0.10, 0.09],
        [0.09, 0.11, 0.11, 0.10],
        [0.12, 0.10, 0.09, 0.11],
    ];
    let cluster_b: Vec<[f64; 4]> = vec![
        [0.80, 0.80, 0.80, 0.80],
        [0.81, 0.79, 0.80, 0.81],
        [0.79, 0.80, 0.81, 0.79],
        [0.80, 0.81, 0.79, 0.80],
    ];
    let cluster_c: Vec<[f64; 4]> = vec![
        [0.50, 0.10, 0.90, 0.50],
        [0.51, 0.11, 0.89, 0.51],
        [0.49, 0.09, 0.91, 0.49],
        [0.50, 0.10, 0.90, 0.51],
    ];

    let idx_a: Vec<HilbertIndex> = cluster_a.iter().map(|c| mapper.coords_to_index(c)).collect();
    let idx_b: Vec<HilbertIndex> = cluster_b.iter().map(|c| mapper.coords_to_index(c)).collect();
    let idx_c: Vec<HilbertIndex> = cluster_c.iter().map(|c| mapper.coords_to_index(c)).collect();

    // Average intra-cluster Hilbert distance
    let mut intra_dists = Vec::new();
    for cluster in [&idx_a, &idx_b, &idx_c] {
        for i in 0..cluster.len() {
            for j in (i + 1)..cluster.len() {
                let d = (cluster[i].0 as i128 - cluster[j].0 as i128).unsigned_abs();
                intra_dists.push(d as f64);
            }
        }
    }
    let avg_intra = intra_dists.iter().sum::<f64>() / intra_dists.len() as f64;

    // Average inter-cluster Hilbert distance
    let mut inter_dists = Vec::new();
    let all_clusters = [&idx_a, &idx_b, &idx_c];
    for ci in 0..3 {
        for cj in (ci + 1)..3 {
            for a in all_clusters[ci] {
                for b in all_clusters[cj] {
                    let d = (a.0 as i128 - b.0 as i128).unsigned_abs();
                    inter_dists.push(d as f64);
                }
            }
        }
    }
    let avg_inter = inter_dists.iter().sum::<f64>() / inter_dists.len() as f64;
    let ratio = avg_inter / avg_intra;

    println!(
        "Synthetic clusters: avg_intra={:.0}, avg_inter={:.0}, ratio={:.1}x",
        avg_intra, avg_inter, ratio
    );
    assert!(
        ratio > 1.5,
        "average inter-cluster distance should significantly exceed intra-cluster. Got ratio={:.2}",
        ratio
    );
}

#[test]
fn hilbert_locality_from_scale_htt() {
    // Open a scale .htt, extract semantic coords, map through the Hilbert
    // curve, and verify locality preservation.
    let src = fixture_path();
    let path = temp_path();
    std::fs::copy(&src, &path).unwrap();

    let gf = Horon::open_with_config(&path, config_lazy()).unwrap();
    let keys = gf.list("/").unwrap();

    // Collect semantic coords from all nodes that have them
    let mut key_coords: Vec<(String, Vec<f64>)> = Vec::new();
    for key in &keys {
        let sem = gf.get_semantic(key).unwrap();
        if sem.len() >= 16 {
            let coords = semantic_bytes_to_f64(&sem);
            if coords.len() > 16 {
                // Only use user-defined dims (16+), skip GACL dims
                key_coords.push((key.clone(), coords));
            }
        }
    }

    println!(
        "Loaded {} nodes with semantic coords (of {} total)",
        key_coords.len(),
        keys.len()
    );

    if key_coords.len() < 10 {
        println!("SKIP: not enough nodes with semantic coords to test locality");
        return;
    }

    // Use dims 16..20 (first 4 user-defined semantic dims) for Hilbert mapping
    let dim_range = 16..20;
    let ndims = dim_range.len();
    let all_coords: Vec<Vec<f64>> = key_coords.iter().map(|(_, c)| c.clone()).collect();
    let normalized = normalize_coords(&all_coords, dim_range.clone());

    let mapper = HilbertMapper::new(ndims, 12);

    // Map each node to its Hilbert index
    let mut entries: Vec<(String, Vec<f64>, HilbertIndex)> = key_coords
        .iter()
        .zip(normalized.iter())
        .map(|((key, raw_coords), norm)| {
            let idx = mapper.coords_to_index(norm);
            (key.clone(), raw_coords.clone(), idx)
        })
        .collect();

    // Sort by Hilbert index (simulating storage order)
    entries.sort_by(|a, b| a.2.cmp(&b.2));

    // THE PREMISE TEST: for each node, compare its semantic distance to its
    // Hilbert neighbors (adjacent in storage) vs random nodes.
    //
    // If Hilbert preserves locality, the average semantic distance to storage
    // neighbors should be significantly less than to random nodes.
    let mut neighbor_dists: Vec<f64> = Vec::new();
    let mut random_dists: Vec<f64> = Vec::new();

    for i in 1..entries.len() - 1 {
        // Semantic distance to storage neighbors (Hilbert-adjacent)
        let slice_self: Vec<f64> = dim_range.clone().filter_map(|d| entries[i].1.get(d).copied()).collect();
        let slice_prev: Vec<f64> = dim_range.clone().filter_map(|d| entries[i - 1].1.get(d).copied()).collect();
        let slice_next: Vec<f64> = dim_range.clone().filter_map(|d| entries[i + 1].1.get(d).copied()).collect();

        neighbor_dists.push(euclidean_dist(&slice_self, &slice_prev));
        neighbor_dists.push(euclidean_dist(&slice_self, &slice_next));

        // Semantic distance to a "far" node (stride by half the list)
        let far_idx = (i + entries.len() / 2) % entries.len();
        let slice_far: Vec<f64> = dim_range.clone().filter_map(|d| entries[far_idx].1.get(d).copied()).collect();
        random_dists.push(euclidean_dist(&slice_self, &slice_far));
    }

    let avg_neighbor = neighbor_dists.iter().sum::<f64>() / neighbor_dists.len() as f64;
    let avg_random = random_dists.iter().sum::<f64>() / random_dists.len() as f64;
    let ratio = avg_random / avg_neighbor;

    println!("=== HILBERT LOCALITY RESULTS (scale .htt fixture) ===");
    println!("  Nodes tested:              {}", entries.len());
    println!("  Avg neighbor distance:     {:.6}", avg_neighbor);
    println!("  Avg random distance:       {:.6}", avg_random);
    println!("  Locality ratio:            {:.2}x", ratio);
    println!("  (>1.0 = locality preserved, higher = better)");

    // Print a sample of the Hilbert ordering
    println!("\n  Sample Hilbert ordering (first 10):");
    for (i, (key, _, idx)) in entries.iter().take(10).enumerate() {
        println!("    [{:3}] idx={:>12}  key={}", i, idx.value(), key);
    }

    assert!(
        ratio > 1.0,
        "Hilbert neighbors should be semantically closer than random nodes. \
         Got ratio={:.2} (need >1.0)",
        ratio
    );

    // A strong locality result would be ratio > 1.5
    if ratio > 1.5 {
        println!("\n  STRONG locality ({:.2}x) — the premise holds.", ratio);
    } else {
        println!(
            "\n  WEAK locality ({:.2}x) — works but may need more dims or bits.",
            ratio
        );
    }
}

// ===========================================================================
// 4. Hilbert chunk offset: verify determinism and monotonic mapping
// ===========================================================================

#[test]
fn hilbert_chunk_offsets_from_htt() {
    // Open .htt, compute Hilbert indices, simulate chunk allocation,
    // verify that nodes with similar semantics land in nearby chunks.
    let src = fixture_path();
    let path = temp_path();
    std::fs::copy(&src, &path).unwrap();

    let gf = Horon::open_with_config(&path, config_lazy()).unwrap();
    let keys = gf.list("/").unwrap();

    let mut key_coords: Vec<(String, Vec<f64>)> = Vec::new();
    for key in &keys {
        let sem = gf.get_semantic(key).unwrap();
        if sem.len() >= 16 {
            let coords = semantic_bytes_to_f64(&sem);
            if coords.len() > 16 {
                key_coords.push((key.clone(), coords));
            }
        }
    }

    if key_coords.len() < 4 {
        println!("SKIP: not enough nodes for chunk test");
        return;
    }

    let dim_range = 16..20;
    let ndims = dim_range.len();
    let all_coords: Vec<Vec<f64>> = key_coords.iter().map(|(_, c)| c.clone()).collect();
    let normalized = normalize_coords(&all_coords, dim_range.clone());

    let mapper = HilbertMapper::new(ndims, 12);
    let chunk_size: usize = 4096; // 4KB chunks

    let mut chunk_assignments: Vec<(String, u64, HilbertIndex)> = key_coords
        .iter()
        .zip(normalized.iter())
        .map(|((key, _), norm)| {
            let idx = mapper.coords_to_index(norm);
            let offset = idx.chunk_offset(chunk_size);
            (key.clone(), offset, idx)
        })
        .collect();

    chunk_assignments.sort_by_key(|(_, offset, _)| *offset);

    // Count how many adjacent (in Hilbert order) pairs share the same chunk
    let mut same_chunk = 0;
    let mut total_pairs = 0;
    for w in chunk_assignments.windows(2) {
        let chunk_a = w[0].1 / chunk_size as u64;
        let chunk_b = w[1].1 / chunk_size as u64;
        if chunk_a == chunk_b {
            same_chunk += 1;
        }
        total_pairs += 1;
    }

    // Count unique chunks used
    let mut chunks_used: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for (_, offset, _) in &chunk_assignments {
        chunks_used.insert(offset / chunk_size as u64);
    }

    println!("=== CHUNK ALLOCATION RESULTS ===");
    println!("  Nodes:            {}", chunk_assignments.len());
    println!("  Chunk size:       {} bytes", chunk_size);
    println!("  Unique chunks:    {}", chunks_used.len());
    println!(
        "  Co-located pairs: {}/{} ({:.1}%)",
        same_chunk,
        total_pairs,
        100.0 * same_chunk as f64 / total_pairs as f64
    );
    println!("  Packing ratio:    {:.1} nodes/chunk",
        chunk_assignments.len() as f64 / chunks_used.len() as f64
    );

    // Determinism check: same coords → same chunk
    if let Some((key, _, _)) = chunk_assignments.first() {
        let sem = gf.get_semantic(key).unwrap();
        let coords = semantic_bytes_to_f64(&sem);
        let norm = normalize_coords(&[coords], dim_range.clone());
        let idx1 = mapper.coords_to_index(&norm[0]);
        let idx2 = mapper.coords_to_index(&norm[0]);
        assert_eq!(idx1, idx2, "Hilbert mapping must be deterministic");
        println!("  Determinism:      PASS");
    }
}

// ===========================================================================
// 5. Roundtrip: create .htt → write with semantics → close → lazy reopen
//    → Hilbert map → verify locality is preserved through full pipeline
// ===========================================================================

#[test]
fn full_pipeline_create_close_lazy_reopen_hilbert() {
    let path = temp_path();
    let sem_dims: u8 = 24; // 12 GACL + 12 user

    // Phase 1: Create .htt with clustered data
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(sem_dims)).unwrap();

        // Cluster A: "databases" — similar coords in user dims 16-19
        for i in 0..5 {
            let key = format!("/db/node_{}", i);
            gf.put(&key, format!("db_data_{}", i).as_bytes()).unwrap();
            let mut coords = vec![0u8; sem_dims as usize * 16];
            let base = 16 * 16; // dim 16 offset
            for d in 0..4 {
                let offset = base + d * 16;
                let val = ((100 + i * 2 + d as i32) as i128) << 64;
                coords[offset..offset + 16].copy_from_slice(&val.to_le_bytes());
            }
            gf.set_semantic(&key, coords).unwrap();
        }

        // Cluster B: "api_endpoints" — different region of semantic space
        for i in 0..5 {
            let key = format!("/api/endpoint_{}", i);
            gf.put(&key, format!("api_data_{}", i).as_bytes()).unwrap();
            let mut coords = vec![0u8; sem_dims as usize * 16];
            let base = 16 * 16;
            for d in 0..4 {
                let offset = base + d * 16;
                let val = ((500 + i * 2 + d as i32) as i128) << 64;
                coords[offset..offset + 16].copy_from_slice(&val.to_le_bytes());
            }
            gf.set_semantic(&key, coords).unwrap();
        }

        // Compact to snapshot
        gf.compact().unwrap();
    }

    // Phase 2: Reopen with lazy_geometry — verify all data survived
    let gf = Horon::open_with_config(&path, config_lazy_semantics(sem_dims)).unwrap();
    // 10 data nodes + intermediate path nodes (/db, /api) created by the engine
    assert!(gf.len() >= 10, "should have at least 10 nodes, got {}", gf.len());

    let db_keys: Vec<String> = gf.list("/db/").unwrap();
    let api_keys: Vec<String> = gf.list("/api/").unwrap();
    assert!(db_keys.len() >= 5, "should have at least 5 db keys, got {}", db_keys.len());
    assert!(api_keys.len() >= 5, "should have at least 5 api keys, got {}", api_keys.len());

    // Phase 3: Hilbert mapping — clusters should separate
    let dim_range = 16..20;
    let ndims = dim_range.len();
    let mapper = HilbertMapper::new(ndims, 12);

    let mut all_sem: Vec<(String, Vec<f64>)> = Vec::new();
    for key in db_keys.iter().chain(api_keys.iter()) {
        let sem = gf.get_semantic(key).unwrap();
        if sem.is_empty() {
            continue; // skip intermediate path nodes without semantic coords
        }
        let coords = semantic_bytes_to_f64(&sem);
        all_sem.push((key.clone(), coords));
    }

    let raw_coords: Vec<Vec<f64>> = all_sem.iter().map(|(_, c)| c.clone()).collect();
    let normalized = normalize_coords(&raw_coords, dim_range.clone());

    let mut entries: Vec<(String, HilbertIndex)> = all_sem
        .iter()
        .zip(normalized.iter())
        .map(|((key, _), norm)| {
            let idx = mapper.coords_to_index(norm);
            (key.clone(), idx)
        })
        .collect();

    entries.sort_by(|a, b| a.1.cmp(&b.1));

    println!("=== FULL PIPELINE: Hilbert ordering ===");
    for (i, (key, idx)) in entries.iter().enumerate() {
        let tag = if key.starts_with("/db/") { "DB " } else { "API" };
        println!("  [{:2}] {} idx={:>12}  {}", i, tag, idx.value(), key);
    }

    // Verify clusters are contiguous in Hilbert order:
    // All DB nodes should appear in a run, all API nodes in another run.
    let labels: Vec<&str> = entries
        .iter()
        .map(|(k, _)| if k.starts_with("/db/") { "db" } else { "api" })
        .collect();

    // Count label transitions — perfect clustering = exactly 1 transition
    let transitions: usize = labels.windows(2).filter(|w| w[0] != w[1]).count();
    println!("  Label transitions: {} (1 = perfect clustering)", transitions);
    assert!(
        transitions <= 3,
        "Clusters should be mostly contiguous in Hilbert order, got {} transitions",
        transitions
    );
}

// ===========================================================================
// 6. Phase 1 integration: compaction writes (depth, hilbert, key)-ordered
//    snapshots — semantically similar entries are byte-adjacent on disk,
//    parents still precede children, and replay is unaffected.
// ===========================================================================

/// Read the raw snapshot entries from a .htt file, in on-disk order.
fn read_raw_snapshot(path: &std::path::Path) -> Vec<horon::snapshot::NodeEntry> {
    use horon::header::GeoHeader;
    use std::io::Read;

    let mut f = std::fs::File::open(path).unwrap();
    let mut hdr_buf = [0u8; 32];
    f.read_exact(&mut hdr_buf).unwrap();
    let header = GeoHeader::from_bytes(&hdr_buf).unwrap();

    horon::snapshot::read_snapshot(
        &mut f,
        header.compression_enabled(),
        &horon::quant::SemLayout::plain(header.semantic_dims as usize),
        header.version >= 2,
    )
    .unwrap()
}

/// Write a node whose user dim `d` (0-based from DIM_USER_DEFINED_START)
/// carries integer value `val`.
fn put_with_user_dims(gf: &Horon, key: &str, data: &[u8], sem_dims: u8, vals: &[i128]) {
    gf.put(key, data).unwrap();
    let mut coords = vec![0u8; sem_dims as usize * 16];
    for (d, val) in vals.iter().enumerate() {
        let offset = (16 + d) * 16;
        if offset + 16 <= coords.len() {
            coords[offset..offset + 16].copy_from_slice(&(*val << 64).to_le_bytes());
        }
    }
    gf.set_semantic(key, coords).unwrap();
}

#[test]
fn compaction_orders_snapshot_by_hilbert_within_depth() {
    let path = temp_path();
    let sem_dims: u8 = 20;

    {
        let gf = Horon::open_with_config(&path, config_with_semantics(sem_dims)).unwrap();

        // Two tight semantic clusters at the SAME depth, interleaved by key so
        // that alphabetical order would mix them: keys alternate a/b prefixes.
        for i in 0..8 {
            // "red" cluster around 100, keys alternate to defeat key-ordering
            put_with_user_dims(&gf, &format!("/x/item_{:02}", i * 2), b"red", sem_dims,
                &[100 + i, 100 + i, 100, 100]);
            // "blue" cluster around 900
            put_with_user_dims(&gf, &format!("/x/item_{:02}", i * 2 + 1), b"blue", sem_dims,
                &[900 + i, 900 + i, 900, 900]);
        }

        gf.compact().unwrap();
    }

    let entries = read_raw_snapshot(&path);
    assert!(!entries.is_empty());

    // Invariant 1: depth non-decreasing (parents before children).
    let depths: Vec<usize> = entries.iter().map(|e| e.key.matches('/').count()).collect();
    for w in depths.windows(2) {
        assert!(w[0] <= w[1], "snapshot must be depth-sorted: {:?}", depths);
    }

    // Invariant 2: within the leaf depth, the two clusters are contiguous —
    // Hilbert ordering groups them even though keys interleave.
    let labels: Vec<&str> = entries
        .iter()
        .filter(|e| e.key.starts_with("/x/item_"))
        .map(|e| if e.data == b"red" { "red" } else { "blue" })
        .collect();
    assert_eq!(labels.len(), 16);
    let transitions = labels.windows(2).filter(|w| w[0] != w[1]).count();
    println!("cluster transitions in on-disk order: {} (alphabetical would be 15)", transitions);
    assert_eq!(
        transitions, 1,
        "clusters must be contiguous in the snapshot, got order: {:?}",
        labels
    );
}

#[test]
fn hilbert_ordered_snapshot_replays_identically() {
    let path = temp_path();
    let sem_dims: u8 = 20;

    // Mixed content: clustered leaves, deeper children, and no-coord nodes.
    {
        let gf = Horon::open_with_config(&path, config_with_semantics(sem_dims)).unwrap();
        for i in 0..10 {
            put_with_user_dims(&gf, &format!("/c/n_{}", i), format!("v{}", i).as_bytes(),
                sem_dims, &[(i % 2) * 800 + i, 50, 50, 50]);
        }
        gf.put("/c/plain", b"no coords at all").unwrap();
        gf.put("/c/n_3/deep", b"child under a leaf").unwrap();
        gf.set_meta("/c/n_7", "tag", "keep").unwrap();
        gf.compact().unwrap();
    }

    // Reopen full-Sarkar and lazy; both must agree with each other completely.
    let path_full = temp_path();
    let path_lazy = temp_path();
    std::fs::copy(&path, &path_full).unwrap();
    std::fs::copy(&path, &path_lazy).unwrap();

    let gf_full = Horon::open_with_config(&path_full, config_with_semantics(sem_dims)).unwrap();
    let gf_lazy = Horon::open_with_config(&path_lazy, config_lazy_semantics(sem_dims)).unwrap();

    assert_eq!(gf_full.len(), gf_lazy.len());
    for key in gf_lazy.list("/").unwrap() {
        assert_eq!(gf_full.get(&key).unwrap(), gf_lazy.get(&key).unwrap(), "data: {}", key);
        assert_eq!(
            gf_full.get_semantic(&key).unwrap(),
            gf_lazy.get_semantic(&key).unwrap(),
            "semantic: {}", key
        );
    }
    assert_eq!(gf_full.get_meta("/c/n_7").unwrap().get("tag").map(String::as_str), Some("keep"));
    assert_eq!(gf_full.get("/c/n_3/deep").unwrap(), b"child under a leaf");

    // Second compaction cycle on the reopened file must also be stable.
    gf_full.put("/c/n_10", b"post-reopen write").unwrap();
    gf_full.compact().unwrap();
    drop(gf_full);
    let gf_again = Horon::open_with_config(&path_full, config_with_semantics(sem_dims)).unwrap();
    assert_eq!(gf_again.get("/c/n_10").unwrap(), b"post-reopen write");
    assert_eq!(gf_again.get("/c/n_3/deep").unwrap(), b"child under a leaf");
}

#[test]
fn files_without_user_dims_keep_key_order() {
    // A file with no semantic dims configured must fall back to the old
    // (depth, key) order exactly — rank 0 for everyone, key tiebreaker rules.
    let path = temp_path();
    {
        let gf = Horon::open_with_config(&path, HoronConfig {
            dimension: 4,
            semantic_dims: 0,
            compression: false,
            auto_compact_threshold: 0,
            ..Default::default()
        }).unwrap();
        for name in ["zeta", "alpha", "mid"] {
            gf.put(&format!("/p/{}", name), name.as_bytes()).unwrap();
        }
        gf.compact().unwrap();
    }

    let entries = read_raw_snapshot(&path);
    let leaf_keys: Vec<&str> = entries
        .iter()
        .filter(|e| e.key.starts_with("/p/"))
        .map(|e| e.key.as_str())
        .collect();
    assert_eq!(leaf_keys, vec!["/p/alpha", "/p/mid", "/p/zeta"]);
}
