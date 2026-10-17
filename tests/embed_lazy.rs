//! Embed-on-demand in its motivating habitat: a lazily-opened `.htt` file gains
//! spatial queries on demand, without a full-geometry reopen.
//!
//! Covers: lazy_geometry mode (embed_all → nearest/neighbors work),
//! identity preservation through the upgrade, geometry being derived state
//! (reopen → data-only again → re-embeddable), and partial_reads mode
//! (snapshot-resident nodes promoted then embedded).

use horon::{DurabilityMode, Horon, HoronConfig};
use g_math::fixed_point::Q64_64 as FixedPoint;
use std::path::PathBuf;
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out.resize(18 * 16, 0);
    out
}

fn build_file(path: &PathBuf) {
    let gf = Horon::open_with_config(
        path,
        HoronConfig {
            semantic_dims: 18,
            auto_compact_threshold: 0,
            durability: DurabilityMode::Relaxed,
            ..Default::default()
        },
    )
    .unwrap();
    gf.put("/course", b"parent").unwrap();
    for i in 0..8 {
        let key = format!("/course/c{}", i);
        gf.put(&key, format!("data-{}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "idx", &i.to_string()).unwrap();
        gf.set_semantic(&key, coords(&[i as f64 * 0.1, 0.5])).unwrap();
    }
    gf.compact().unwrap();
}

#[test]
fn lazy_file_gains_spatial_queries_on_demand() {
    let path = temp_path();
    build_file(&path);

    let gf = Horon::open_with_config(
        &path,
        HoronConfig {
            lazy_geometry: true,
            auto_compact_threshold: 0,
            ..Default::default()
        },
    )
    .unwrap();

    // Lazy: data + semantic work, spatial does not.
    assert_eq!(gf.get("/course/c3").unwrap(), b"data-3");
    let sem = gf.nearest_semantic(&coords(&[0.3, 0.5]), 1, 16..18).unwrap();
    assert_eq!(sem[0].0, "/course/c3");
    assert!(
        gf.neighbors("/course/c3", 2).is_err()
            || gf.neighbors("/course/c3", 2).unwrap().is_empty(),
        "lazy nodes must not be spatially visible before embed"
    );

    // Embed on demand: everything under /course (root "/" embeds via
    // ancestor recursion — it exists as a data-only node in lazy replay).
    let upgraded = gf.embed_all("/course").unwrap();
    assert_eq!(upgraded, 9, "parent + 8 children");

    // Spatial world is live now.
    let near = gf.neighbors("/course/c3", 3).unwrap();
    assert!(!near.is_empty(), "no spatial neighbors after embed");
    assert!(near.iter().all(|k| k.starts_with("/course")), "{:?}", near);

    // Identity untouched.
    assert_eq!(gf.get("/course/c3").unwrap(), b"data-3");
    assert_eq!(gf.get_meta("/course/c3").unwrap().get("idx").unwrap(), "3");
    let sem = gf.nearest_semantic(&coords(&[0.3, 0.5]), 1, 16..18).unwrap();
    assert_eq!(sem[0].0, "/course/c3");

    // Idempotent.
    assert_eq!(gf.embed_all("/course").unwrap(), 0);
}

#[test]
fn geometry_is_derived_state_not_persisted() {
    let path = temp_path();
    build_file(&path);

    {
        let gf = Horon::open_with_config(
            &path,
            HoronConfig { lazy_geometry: true, ..Default::default() },
        )
        .unwrap();
        assert!(gf.embed("/course/c1").unwrap());
        assert!(!gf.neighbors("/course/c1", 1).unwrap().is_empty());
        gf.flush().unwrap();
    }

    // Reopen lazily: the embedding is gone (derived state), the data is not.
    let gf = Horon::open_with_config(
        &path,
        HoronConfig { lazy_geometry: true, ..Default::default() },
    )
    .unwrap();
    assert_eq!(gf.get("/course/c1").unwrap(), b"data-1");
    assert!(
        gf.neighbors("/course/c1", 1).is_err()
            || gf.neighbors("/course/c1", 1).unwrap().is_empty(),
        "geometry must not survive a lazy reopen"
    );
    // And it can be re-embedded.
    assert!(gf.embed("/course/c1").unwrap());
    assert!(!gf.neighbors("/course/c1", 1).unwrap().is_empty());
}

#[test]
fn partial_reads_mode_promotes_then_embeds() {
    let path = temp_path();
    // partial_reads requires an uncompressed snapshot.
    {
        let gf = Horon::open_with_config(
            &path,
            HoronConfig {
                semantic_dims: 18,
                compression: false,
                auto_compact_threshold: 0,
                durability: DurabilityMode::Relaxed,
                ..Default::default()
            },
        )
        .unwrap();
        gf.put("/course", b"parent").unwrap();
        for i in 0..5 {
            let key = format!("/course/c{}", i);
            gf.put(&key, format!("data-{}", i).as_bytes()).unwrap();
            gf.set_semantic(&key, coords(&[i as f64 * 0.1, 0.5])).unwrap();
        }
        gf.compact().unwrap();
    }

    let gf = Horon::open_with_config(
        &path,
        HoronConfig {
            partial_reads: true,
            compression: false,
            ..Default::default()
        },
    )
    .unwrap();

    // Snapshot-resident, mmap-served, no geometry.
    assert_eq!(gf.get("/course/c2").unwrap(), b"data-2");

    // Embed promotes the node (and its ancestors) into the overlay first.
    assert!(gf.embed("/course/c2").unwrap());
    assert!(gf.embed("/course/c4").unwrap());
    let near = gf.neighbors("/course/c2", 1).unwrap();
    assert!(!near.is_empty(), "embedded partial-mode nodes must see each other");

    // Data still correct after promote + embed.
    assert_eq!(gf.get("/course/c2").unwrap(), b"data-2");
}
