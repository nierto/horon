//! Append to a `.htt` slowly enough that external readers catch it mid-flight.
//!
//! Used by the cross-language differential oracle in `htt-wrappers`: the
//! Python and JavaScript readers sample this file while it is being written
//! and assert that every view they observe is a valid prefix of the write
//! sequence. Because those readers take no lock, this also demonstrates the
//! property `HoronReader` relies on — a live `.htt` is safe to read.
//!
//!     cargo run --release --example live_writer -- <path> <count> <delay_ms>

use horon::{Horon, HoronConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: live_writer <path> <count> <delay_ms>");
    let count: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(500);
    let delay_ms: u64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(4);

    let _ = std::fs::remove_file(path);
    let gf = Horon::open_with_config(
        path,
        HoronConfig {
            auto_compact_threshold: 0,
            ..Default::default()
        },
    )
    .expect("open");

    for i in 0..count {
        gf.put(&format!("/n/{i}"), format!("v{i}").as_bytes())
            .expect("put");
        if delay_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay_ms));
        }
    }
    println!("wrote {count} entries to {path}");
}
