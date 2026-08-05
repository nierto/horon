//! The `.htt` lifecycle in one file: create → use → modify → reopen.
//!
//! This is the README's quick-start example in runnable form — if this
//! stops compiling or passing, the README is lying.
//!
//! Domain here: a server fleet, where each node's semantic dims are
//! normalized telemetry (cpu load, request rate) — "which machines behave
//! like this one?" is a semantic nearest-neighbor query. The same shape
//! fits course catalogs, sensor networks, document corpora: hierarchy in
//! the path, meaning in the dims.
//!
//! ```text
//! GMATH_PROFILE=embedded cargo run --example lifecycle
//! ```

use horon::{Horon, HoronConfig, HoronResult};
use g_math::fixed_point::FixedPoint;

/// Semantic coordinates are raw Q64.64 bytes, 16 per dimension. Dims 0–15
/// are reserved (access control); your dims start at 16. This helper places
/// f64 values into the user dims of a full-width vector.
fn coords(total_dims: usize, user_vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; total_dims * 16];
    for (i, v) in user_vals.iter().enumerate() {
        let off = (16 + i) * 16;
        out[off..off + 16].copy_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out
}

fn main() -> HoronResult<()> {
    let path = std::env::temp_dir().join("lifecycle_demo.htt");
    let _ = std::fs::remove_file(&path);

    // ---- CREATE ------------------------------------------------------
    // 16 reserved dims + 2 user dims (cpu load, request rate — normalized).
    // Every write below is WAL-logged before it takes effect — kill the
    // process anywhere, nothing committed is lost.
    let dims = 18u8;
    {
        let htt = Horon::open_with_config(&path, HoronConfig {
            semantic_dims: dims,
            ..Default::default()
        })?;

        htt.put("/fleet/eu/web-01", b"nginx 1.27")?;
        htt.put("/fleet/eu/web-02", b"nginx 1.27")?;
        htt.put("/fleet/eu/batch-01", b"cron runner")?;
        htt.set_meta("/fleet/eu/web-01", "role", "edge")?;
        htt.set_semantic("/fleet/eu/web-01", coords(dims as usize, &[0.80, 0.70]))?;
        htt.set_semantic("/fleet/eu/web-02", coords(dims as usize, &[0.75, 0.65]))?;
        htt.set_semantic("/fleet/eu/batch-01", coords(dims as usize, &[0.20, 0.05]))?;

        // ---- USE -------------------------------------------------------
        let data = htt.get("/fleet/eu/web-01")?;
        let machines = htt.children("/fleet/eu")?;
        // "Which machines behave like web-01?" — nearest by telemetry dims:
        let similar = htt.neighbors_semantic("/fleet/eu/web-01", 2, 16..dims as usize)?;
        println!("payload:  {}", String::from_utf8_lossy(&data));
        println!("machines: {:?}", machines);
        println!("similar:  {:?}", similar); // web-02 ranks far before batch-01

        // ---- MODIFY ----------------------------------------------------
        htt.put("/fleet/eu/web-01", b"nginx 1.28")?; // redeploy: put = upsert
        htt.set_meta("/fleet/eu/web-01", "role", "edge-canary")?;
        // New telemetry after a scale-out shifted its load profile:
        htt.set_semantic("/fleet/eu/web-01", coords(dims as usize, &[0.55, 0.60]))?;
        htt.remove("/fleet/eu/batch-01")?; // decommissioned
        htt.compact()?; // fold the WAL into a fresh snapshot
    } // drop = flush + unlock

    // ---- REOPEN ------------------------------------------------------
    // Everything above was replayed from disk; same state, same answers.
    let htt = Horon::open(&path)?;
    assert_eq!(htt.get("/fleet/eu/web-01")?, b"nginx 1.28");
    assert_eq!(
        htt.get_meta("/fleet/eu/web-01")?.get("role").map(String::as_str),
        Some("edge-canary")
    );
    assert!(!htt.exists("/fleet/eu/batch-01"));
    println!("reopened: {} nodes, state intact", htt.len());

    let _ = std::fs::remove_file(&path);
    Ok(())
}
