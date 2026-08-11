//! Insert-rate probe: N puts through the full Horon path, Relaxed durability.
//!
//! DEPTH=k builds chains of depth k instead of a flat fan-out: n/k chains,
//! each grown one level per put, so inserts are spread across depths 1..=k.
//! The flat and deep shapes exercise different bucket geometry in
//! find_bucket, which is exactly what a flat-only benchmark cannot see.
use horon::{DurabilityMode, Horon, HoronConfig};
fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12_000);
    let depth: usize = std::env::var("DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut path = std::env::temp_dir();
    path.push(format!("horon_putrate_{}.htt", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let t0 = std::time::Instant::now();
    {
        let gf = Horon::open_with_config(&path, HoronConfig {
            auto_compact_threshold: 0,
            durability: DurabilityMode::Relaxed,
            ..Default::default()
        }).unwrap();
        if depth <= 1 {
            for i in 0..n {
                gf.put(&format!("/d/{i:06}"), b"x").unwrap();
            }
        } else {
            let chains = n / depth;
            for c in 0..chains {
                let mut key = format!("/c{c:05}");
                gf.put(&key, b"x").unwrap();
                for l in 1..depth {
                    key.push_str(&format!("/n{l}"));
                    gf.put(&key, b"x").unwrap();
                }
            }
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("n={n} depth={depth}  build={secs:.1}s  {:.2} ms/insert", secs * 1000.0 / n as f64);
    let _ = std::fs::remove_file(&path);
}
