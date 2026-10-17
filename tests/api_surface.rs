//! API-surface lockfile: everything a user of this crate needs is nameable
//! through this crate alone, with signatures recorded here.
//!
//! Each `fn`-pointer binding locks a method's existence and exact signature
//! at compile time — removing a method, changing a parameter, or breaking a
//! re-export turns this file into a build failure instead of a silent
//! surface change. This file deliberately imports nothing from the engine
//! crate: if it compiles, the facade is self-sufficient.

use horon::{
    Horon, HoronConfig, HoronError, HoronResult, SemanticDisk, SemanticOutlier, Store,
    StoreError,
};
use g_math::fixed_point::Q64_64 as FixedPoint;
use std::ops::Range;

#[test]
fn query_surface_is_nameable_without_the_engine_crate() {
    // The query methods live on `Horon`'s deref target, so each lock is a
    // closure calling through `Horon` — exactly the call a user writes —
    // coerced to a `fn` pointer with the full signature spelled out.

    // Point queries
    let _nearest: fn(&Horon, &[FixedPoint]) -> HoronResult<(String, FixedPoint)> =
        |gf, coords| gf.nearest(coords);
    let _nearest_k: fn(&Horon, &[FixedPoint], usize) -> HoronResult<Vec<(String, FixedPoint)>> =
        |gf, coords, k| gf.nearest_k(coords, k);
    let _neighbors: fn(&Horon, &str, usize) -> HoronResult<Vec<String>> =
        |gf, path, k| gf.neighbors(path, k);
    let _find_within: fn(&Horon, &str, FixedPoint) -> HoronResult<Vec<String>> =
        |gf, path, radius| gf.find_within(path, radius);
    let _position: fn(&Horon, &str) -> HoronResult<Vec<FixedPoint>> =
        |gf, key| gf.position(key);

    // Semantic-slice queries
    let _nearest_semantic: fn(
        &Horon,
        &[u8],
        usize,
        Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> =
        |gf, coords, k, dims| gf.nearest_semantic(coords, k, dims);
    let _neighbors_semantic: fn(
        &Horon,
        &str,
        usize,
        Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> =
        |gf, path, k, dims| gf.neighbors_semantic(path, k, dims);
    let _find_similar: fn(
        &Horon,
        &str,
        usize,
        Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> =
        |gf, key, k, dims| gf.find_similar(key, k, dims);
    let _find_outliers: fn(
        &Horon,
        &str,
        FixedPoint,
        Range<usize>,
    ) -> HoronResult<Vec<SemanticOutlier>> =
        |gf, prefix, z, dims| gf.find_outliers(prefix, z, dims);

    // Associated function — nameable directly on `Horon`.
    let _semantic_distance: fn(&[u8], &[u8], Range<usize>) -> FixedPoint =
        Horon::semantic_distance;

    // The escape hatch stays available, and its type is ours to name.
    let _store: fn(&Horon) -> &Store = |gf| gf.store();
}

#[test]
fn semantic_disk_is_usable_without_the_engine_crate() {
    // Locks both the re-export and its constructor signature — including
    // that its error type is nameable from this crate.
    let _build: fn(&[(&str, usize)]) -> Result<SemanticDisk, StoreError> = SemanticDisk::build;
}

/// Semantic coordinates: 16 GACL/reserved dims of zeros, then user dims.
fn coords(vals: &[i32]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for v in vals {
        out.extend_from_slice(&FixedPoint::from_int(*v).raw().to_le_bytes());
    }
    out
}

#[test]
fn forwarded_queries_reach_the_right_engine_methods() {
    let dir = tempfile::tempdir().unwrap();
    let config = HoronConfig {
        semantic_dims: 18,
        ..Default::default()
    };
    let gf = Horon::open_with_config(dir.path().join("smoke.htt"), config).unwrap();

    gf.put("/a/1", b"one").unwrap();
    gf.put("/a/2", b"two").unwrap();
    gf.put("/a/3", b"three").unwrap();

    // nearest_k: k results, sorted by ascending distance.
    let origin: Vec<FixedPoint> = vec![FixedPoint::from_int(0); 4];
    let hits = gf.nearest_k(&origin, 2).unwrap();
    assert_eq!(hits.len(), 2);
    assert!(hits[0].1 <= hits[1].1);

    // position: one coordinate per structural dimension, inside the disk.
    let pos = gf.position("/a/1").unwrap();
    assert_eq!(pos.len(), 4);

    // find_within: an embedded node is within a generous radius of itself's
    // siblings — and the anchor itself is excluded.
    let within = gf.find_within("/a/1", FixedPoint::from_int(1000)).unwrap();
    assert!(within.iter().any(|k| k == "/a/2"));
    assert!(!within.iter().any(|k| k == "/a/1"));

    // find_similar: nearest semantic neighbor over the user dims, self
    // excluded. (All-zero coordinates encode "not set", so every placement
    // is offset from the origin.)
    gf.set_semantic("/a/1", coords(&[1, 1])).unwrap();
    gf.set_semantic("/a/2", coords(&[2, 1])).unwrap();
    gf.set_semantic("/a/3", coords(&[50, 50])).unwrap();
    let similar = gf.find_similar("/a/1", 1, 16..18).unwrap();
    assert_eq!(similar[0].0, "/a/2");

    // semantic_distance: zero to self, positive and symmetric across points.
    let (a, b) = (coords(&[1, 1]), coords(&[4, 5]));
    let zero = FixedPoint::from_int(0);
    assert_eq!(Horon::semantic_distance(&a, &a, 16..18), zero);
    let d = Horon::semantic_distance(&a, &b, 16..18);
    assert!(d > zero);
    assert_eq!(Horon::semantic_distance(&b, &a, 16..18), d);
}

#[test]
fn engine_error_payload_is_matchable_via_this_crate() {
    // A caller handling `HoronError` can destructure the wrapped engine
    // error down to its variants without depending on the engine crate.
    let err = HoronError::Store(StoreError::NotFound("k".to_string()));
    match err {
        HoronError::Store(StoreError::NotFound(key)) => assert_eq!(key, "k"),
        other => panic!("expected Store(NotFound), got {other:?}"),
    }
}
