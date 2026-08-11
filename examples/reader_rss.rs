//! F26 Phase D: do N partial readers actually SHARE one physical copy?
//!
//! Parent builds a file, then spawns itself 4x per mode; each child opens
//! the file, touches every key, and reports its VmRSS. If partial mode
//! shares pages via the page cache, per-child RSS stays near-flat while
//! full mode pays a private materialized copy per process.
//!
//!     cargo run --release --example reader_rss -- [nodes]

use horon::{DurabilityMode, Horon, HoronConfig, HoronReader, ReaderMode};

fn vm_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmRSS:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Proportional set size: shared pages divided by their sharer count. This
/// is the honest "physical memory" number — VmRSS counts a shared mmap page
/// once per process, PSS charges each sharer 1/n of it.
fn pss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/smaps_rollup")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("Pss:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

fn child(path: &str, mode: &str) {
    let reader = match mode {
        "partial" => HoronReader::open_partial(path).unwrap(),
        _ => HoronReader::open_with_mode(path, ReaderMode::Full).unwrap(),
    };
    let keys = reader.list("/").unwrap();
    let mut bytes = 0usize;
    for k in &keys {
        bytes += reader.get(k).map(|d| d.len()).unwrap_or(0);
    }
    println!("{} {} {} {}", keys.len(), bytes, vm_rss_kb(), pss_kb());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("child") {
        return child(&args[2], &args[3]);
    }
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);

    let mut path = std::env::temp_dir();
    path.push(format!("horon_rss_{}.htt", std::process::id()));
    let _ = std::fs::remove_file(&path);
    {
        let gf = Horon::open_with_config(&path, HoronConfig {
            compression: false,
            auto_compact_threshold: 0,
            durability: DurabilityMode::Relaxed,
            ..Default::default()
        }).unwrap();
        // 2 KiB payloads: data dwarfs fixed per-process overhead.
        let payload = vec![0xABu8; 2048];
        for i in 0..n {
            gf.put(&format!("/d/{i:06}"), &payload).unwrap();
        }
        gf.compact().unwrap();
    }
    let file_kb = std::fs::metadata(&path).map(|m| m.len() / 1024).unwrap_or(0);
    println!("file: {file_kb} KiB, {n} nodes, 4 concurrent readers per mode\n");

    let exe = std::env::current_exe().unwrap();
    for mode in ["full", "partial"] {
        let children: Vec<_> = (0..4)
            .map(|_| {
                std::process::Command::new(&exe)
                    .args(["child", path.to_str().unwrap(), mode])
                    .stdout(std::process::Stdio::piped())
                    .spawn()
                    .unwrap()
            })
            .collect();
        let mut rss = Vec::new();
        for c in children {
            let out = c.wait_with_output().unwrap();
            let line = String::from_utf8_lossy(&out.stdout);
            let f: Vec<u64> = line.split_whitespace().filter_map(|v| v.parse().ok()).collect();
            rss.push((f.get(2).copied().unwrap_or(0), f.get(3).copied().unwrap_or(0)));
        }
        let rss_total: u64 = rss.iter().map(|r| r.0).sum();
        let pss_total: u64 = rss.iter().map(|r| r.1).sum();
        println!("  {mode:<8} per-child (RSS, PSS) KiB: {rss:?}");
        println!("           totals: RSS {rss_total} KiB   PSS {pss_total} KiB   <- PSS is physical");
    }
    let _ = std::fs::remove_file(&path);
}
