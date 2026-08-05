//! GeoStore — production-ready wrapper around Horon.
//!
//! Drop this into any project that needs a persistent, thread-safe,
//! spatially-aware hierarchical store. Clone to share across threads.
//!
//! ```text
//! GMATH_PROFILE=embedded cargo run --example geostore
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;

use horon::{
    Horon, HoronConfig, HoronResult, DurabilityMode,
};

// ---------------------------------------------------------------------------
// GeoStore: the wrapper
// ---------------------------------------------------------------------------

/// A shared, thread-safe persistent store backed by a `.htt` file.
///
/// `Clone` is cheap (Arc bump). Pass into thread::spawn, async tasks,
/// HTTP handlers, gRPC services — anywhere you'd pass an `Arc<T>`.
#[derive(Clone)]
pub struct GeoStore {
    inner: Arc<Horon>,
}

impl GeoStore {
    /// Open or create with sane defaults.
    ///
    /// - zstd compression on
    /// - auto-compact at 10k WAL entries
    /// - immediate flush (no batching)
    /// - relaxed durability (OS page cache)
    pub fn open<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Ok(Self {
            inner: Arc::new(Horon::open(path)?),
        })
    }

    /// Open with explicit tuning.
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: HoronConfig,
    ) -> HoronResult<Self> {
        Ok(Self {
            inner: Arc::new(Horon::open_with_config(path, config)?),
        })
    }

    // --- Common profiles ---

    /// Profile: embedded / edge device.
    ///
    /// Small WAL before auto-compact, fsync every write, no batching.
    /// Prioritizes crash safety over throughput.
    pub fn open_embedded<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Self::open_with_config(path, HoronConfig {
            auto_compact_threshold: 1_000,
            durability: DurabilityMode::Fsync,
            ..Default::default()
        })
    }

    /// Profile: application server / service registry.
    ///
    /// Batch WAL writes for throughput, fsync on batch flush,
    /// larger compaction window. Good for moderate write rates
    /// (hundreds/sec) with many concurrent readers.
    pub fn open_server<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Self::open_with_config(path, HoronConfig {
            auto_compact_threshold: 50_000,
            wal_batch_size: 64,
            wal_flush_interval_ms: 100,
            durability: DurabilityMode::Batched,
            ..Default::default()
        })
    }

    /// Profile: analytics / read-heavy workload.
    ///
    /// Relaxed durability, large compaction window.
    /// Writes are infrequent; reads dominate.
    pub fn open_analytics<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Self::open_with_config(path, HoronConfig {
            auto_compact_threshold: 100_000,
            wal_batch_size: 256,
            wal_flush_interval_ms: 500,
            durability: DurabilityMode::Relaxed,
            ..Default::default()
        })
    }

    // --- CRUD (delegated) ---

    pub fn put(&self, key: &str, data: &[u8]) -> HoronResult<()> {
        self.inner.put(key, data)
    }

    pub fn get(&self, key: &str) -> HoronResult<Vec<u8>> {
        self.inner.get(key)
    }

    pub fn remove(&self, key: &str) -> HoronResult<()> {
        self.inner.remove(key)
    }

    pub fn exists(&self, key: &str) -> bool {
        self.inner.exists(key)
    }

    pub fn set_meta(&self, key: &str, name: &str, value: &str) -> HoronResult<()> {
        self.inner.set_meta(key, name, value)
    }

    pub fn get_meta(&self, key: &str) -> HoronResult<HashMap<String, String>> {
        self.inner.get_meta(key)
    }

    // --- Hierarchy ---

    pub fn children(&self, path: &str) -> HoronResult<Vec<String>> {
        self.inner.children(path)
    }

    pub fn list(&self, prefix: &str) -> HoronResult<Vec<String>> {
        self.inner.list(prefix)
    }

    // --- Spatial queries ---

    /// Find the node closest to the given coordinates.
    pub fn nearest(&self, coords: &[g_math::fixed_point::FixedPoint])
        -> HoronResult<(String, g_math::fixed_point::FixedPoint)> {
        self.inner.nearest(coords)
    }

    /// Find k nearest neighbors of an existing node.
    pub fn neighbors(&self, key: &str, k: usize) -> HoronResult<Vec<String>> {
        self.inner.neighbors(key, k)
    }

    // --- Persistence control ---

    pub fn compact(&self) -> HoronResult<bool> {
        self.inner.compact()
    }

    pub fn compact_async(&self) -> JoinHandle<HoronResult<bool>> {
        self.inner.compact_async()
    }

    pub fn flush(&self) -> HoronResult<()> {
        self.inner.flush()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn wal_len(&self) -> u32 {
        self.inner.wal_len()
    }
}

// ---------------------------------------------------------------------------
// Demo: three deployment scenarios
// ---------------------------------------------------------------------------

fn main() {
    // -----------------------------------------------------------------------
    // 1. Service registry
    // -----------------------------------------------------------------------
    println!("=== Service Registry ===\n");
    {
        let db = GeoStore::open_server("/tmp/geostore_services.htt").unwrap();

        // Register services under an org hierarchy
        db.put("/org/platform/auth",    b"auth-service-v2.3").unwrap();
        db.put("/org/platform/gateway", b"api-gateway-v1.8").unwrap();
        db.put("/org/ml/inference",     b"bert-serving-v4.1").unwrap();
        db.put("/org/ml/training",      b"trainer-gpu-v2.0").unwrap();
        db.put("/org/data/ingest",      b"kafka-bridge-v3.2").unwrap();
        db.put("/org/data/warehouse",   b"clickhouse-proxy-v1.1").unwrap();

        // Tag with metadata
        db.set_meta("/org/ml/inference", "gpu", "a100").unwrap();
        db.set_meta("/org/ml/inference", "replicas", "8").unwrap();
        db.set_meta("/org/data/ingest", "throughput", "500k-msg-sec").unwrap();

        // Discover: "what ML services do we have?"
        let ml_services = db.children("/org/ml").unwrap();
        println!("  ML services: {:?}", ml_services);

        // Discover: "everything under /org"
        let all = db.list("/org").unwrap();
        println!("  All registered: {} services", all.len());

        // Spatial: "which node is closest to the origin in hyperbolic space?"
        // (In a real system, coordinates encode capability vectors)
        let (closest, dist) = db.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        println!("  Nearest to origin: {} (dist={:.4})", closest, dist);

        // Spatial: "who are the 3 nearest neighbors to the auth service?"
        let neighbors = db.neighbors("/org/platform/auth", 3).unwrap();
        println!("  Auth neighbors: {:?}", neighbors);

        db.flush().unwrap();
    }

    // -----------------------------------------------------------------------
    // 2. Hierarchical config store (like etcd but embedded + spatial)
    // -----------------------------------------------------------------------
    println!("\n=== Config Store ===\n");
    {
        let db = GeoStore::open_embedded("/tmp/geostore_config.htt").unwrap();

        // Global defaults
        db.put("/config/global/timeout_ms", b"5000").unwrap();
        db.put("/config/global/retry_count", b"3").unwrap();
        db.put("/config/global/log_level", b"info").unwrap();

        // Environment overrides
        db.put("/config/prod/timeout_ms", b"10000").unwrap();
        db.put("/config/prod/log_level", b"warn").unwrap();
        db.put("/config/staging/log_level", b"debug").unwrap();

        // Service-specific
        db.put("/config/prod/auth/timeout_ms", b"30000").unwrap();
        db.put("/config/prod/auth/max_sessions", b"10000").unwrap();

        // Read config with inheritance: most specific wins
        fn resolve_config(db: &GeoStore, env: &str, service: &str, key: &str) -> String {
            // Try service-specific, then env, then global
            let paths = [
                format!("/config/{}/{}/{}", env, service, key),
                format!("/config/{}/{}", env, key),
                format!("/config/global/{}", key),
            ];
            for path in &paths {
                if let Ok(data) = db.get(path) {
                    return String::from_utf8_lossy(&data).to_string();
                }
            }
            "<unset>".to_string()
        }

        let timeout = resolve_config(&db, "prod", "auth", "timeout_ms");
        let log_lvl = resolve_config(&db, "prod", "auth", "log_level");
        let retries = resolve_config(&db, "prod", "auth", "retry_count");

        println!("  prod/auth/timeout_ms  = {} (service override)", timeout);
        println!("  prod/auth/log_level   = {} (env override)", log_lvl);
        println!("  prod/auth/retry_count = {} (global default)", retries);

        // List all config keys in staging
        let staging_keys = db.list("/config/staging").unwrap();
        println!("  Staging overrides: {:?}", staging_keys);

        db.flush().unwrap();
    }

    // -----------------------------------------------------------------------
    // 3. Multi-threaded content store with concurrent access
    // -----------------------------------------------------------------------
    println!("\n=== Concurrent Content Store ===\n");
    {
        let db = GeoStore::open_server("/tmp/geostore_content.htt").unwrap();

        // Spawn writer threads
        let mut handles = vec![];
        for team in ["frontend", "backend", "infra", "security"] {
            let db = db.clone(); // Arc bump, cheap
            let team = team.to_string();
            handles.push(std::thread::spawn(move || {
                for i in 0..10 {
                    let key = format!("/docs/{}/page_{}", team, i);
                    let content = format!("{} documentation page {}", team, i);
                    db.put(&key, content.as_bytes()).unwrap();
                    db.set_meta(&key, "team", &team).unwrap();
                    db.set_meta(&key, "version", &i.to_string()).unwrap();
                }
            }));
        }

        // Concurrent readers while writers are active
        for _ in 0..4 {
            let db = db.clone();
            handles.push(std::thread::spawn(move || {
                // Readers tolerate missing keys (writers may not be done)
                for i in 0..10 {
                    let _ = db.get(&format!("/docs/frontend/page_{}", i));
                    let _ = db.exists(&format!("/docs/backend/page_{}", i));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        println!("  Total entries: {}", db.len());
        println!("  WAL entries: {}", db.wal_len());

        // Background compaction while we continue reading
        let compact_handle = db.compact_async();

        // Reads continue during compaction (lock-free via DashMap)
        let teams = db.children("/docs").unwrap();
        println!("  Team namespaces: {:?}", teams);
        for team in &teams {
            let pages = db.children(team).unwrap();
            println!("    {} has {} pages", team, pages.len());
        }

        compact_handle.join().unwrap().unwrap();
        println!("  After compaction: WAL entries = {}", db.wal_len());

        db.flush().unwrap();
    }

    // Cleanup
    let _ = std::fs::remove_file("/tmp/geostore_services.htt");
    let _ = std::fs::remove_file("/tmp/geostore_config.htt");
    let _ = std::fs::remove_file("/tmp/geostore_content.htt");

    println!("\nDone.");
}

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}
