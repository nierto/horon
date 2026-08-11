//! Insert-rate probe: N puts through the full Horon path, Relaxed durability.
use horon::{DurabilityMode, Horon, HoronConfig};
fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12_000);
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
        for i in 0..n { gf.put(&format!("/d/{i:06}"), b"x").unwrap(); }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!("n={n}  build={secs:.1}s  {:.2} ms/insert", secs * 1000.0 / n as f64);
    let _ = std::fs::remove_file(&path);
}
