//! Synthetic dataset generator — builds `.htt` files at realistic scale.
//!
//! Where `gen_corpus` produces tiny files that pin down *format* behaviour,
//! this produces large files that exercise *data* behaviour: Hilbert locality,
//! compaction over thousands of nodes, lazy-open cost, semantic query quality.
//! Everything is generated from a seed, so a given `--seed` always yields a
//! byte-identical file and a committed fixture can be reproduced by anyone.
//!
//! Three profiles, each modelling a different shape of hierarchical data:
//!
//! * `catalog` — a taxonomy of categories and items whose semantic
//!   coordinates cluster by category. The shape most applications have:
//!   entities grouped by kind, similar entities near each other.
//! * `filesystem` — a directory tree with coordinates derived from file size,
//!   depth and age, mirroring what `htt import-fs` produces from a real
//!   directory.
//! * `vectors` — a flat corpus of dense embeddings, the shape `htt import-vec`
//!   produces from an embedding pipeline.
//!
//! ```text
//! cargo run --example gen_fixture -- --profile catalog --nodes 1500 --v1
//! cargo run --example gen_fixture -- --profile filesystem --out /tmp/fs.htt
//! cargo run --example gen_fixture -- --profile vectors --dims 40 --quantized
//! ```
//!
//! Coordinates are generated in Q64.64 integer space directly — no floating
//! point anywhere in the value path, so output does not depend on the host's
//! FPU behaviour.

use std::fs;
use std::path::PathBuf;
use std::process::exit;

use horon::{Horon, HoronConfig};

/// Q64.64 representation of 1.0. All generated user dimensions live in
/// `[0, ONE]`, which keeps them inside the meaning-addressed default bounds
/// and well inside the TQ1.9 quantization range.
const ONE: i128 = 1i128 << 64;

/// First user-controlled semantic dimension; 0..16 are reserved for access
/// bands and other reserved axes.
const USER_DIM_START: usize = 16;

// ---------------------------------------------------------------------------
// Deterministic PRNG (SplitMix64)
// ---------------------------------------------------------------------------

/// Seeded, portable, and dependency-free. The same seed must produce the same
/// file on every platform, so this is deliberately a fixed integer algorithm
/// rather than anything from the `rand` ecosystem.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform value in `[0, bound)`.
    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 { 0 } else { self.next_u64() % bound }
    }

    /// Uniform Q64.64 value in `[0, ONE)` — a u64 of fractional bits is
    /// exactly the Q64.64 encoding of `[0.0, 1.0)`.
    fn unit(&mut self) -> i128 {
        self.next_u64() as i128
    }

    /// Signed offset within `±spread`.
    fn jitter(&mut self, spread: i128) -> i128 {
        if spread == 0 {
            return 0;
        }
        let magnitude = (self.next_u64() as i128) % (spread + 1);
        if self.next_u64() & 1 == 0 { magnitude } else { -magnitude }
    }
}

/// Clamp into the `[0, ONE]` band every profile writes within.
fn clamp_unit(v: i128) -> i128 {
    v.clamp(0, ONE)
}

fn coords_from(values: &[(usize, i128)], dims: u8) -> Vec<u8> {
    let mut coords = vec![0u8; dims as usize * 16];
    for (dim, value) in values {
        if *dim < dims as usize {
            let off = dim * 16;
            coords[off..off + 16].copy_from_slice(&value.to_le_bytes());
        }
    }
    coords
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/// Obviously-synthetic category names: no generated file should ever be
/// mistakable for data belonging to a real project.
const CATEGORIES: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "omicron", "pi",
];

const EXTENSIONS: &[&str] = &["txt", "md", "rs", "json", "csv", "bin"];

/// A taxonomy whose semantic coordinates cluster by category: every category
/// gets a centroid, every item sits near its category's centroid. This is what
/// makes the file useful for locality testing — Hilbert ordering should place
/// same-category items in adjacent bytes.
fn build_catalog(gf: &Horon, dims: u8, nodes: usize, rng: &mut Rng) {
    let category_count = CATEGORIES.len().min(1 + nodes / 40).max(2);
    let user_dims = (dims as usize).saturating_sub(USER_DIM_START);
    let cluster_dims = user_dims.min(8);

    // One centroid per category, drawn once so every item can be placed
    // relative to it.
    let centroids: Vec<Vec<i128>> = (0..category_count)
        .map(|_| (0..cluster_dims).map(|_| rng.unit()).collect())
        .collect();

    // Wide enough that neighbouring clusters overlap: a fixture whose
    // clusters are trivially separable would flatter any locality test it is
    // used in.
    let spread = ONE / 8;

    for i in 0..nodes {
        let cat = i % category_count;
        let key = format!("/catalog/{}/item_{:05}", CATEGORIES[cat], i);
        let payload = format!(
            "{{\"id\":{},\"category\":\"{}\",\"weight\":{}}}",
            i,
            CATEGORIES[cat],
            rng.below(1000)
        );
        gf.put(&key, payload.as_bytes()).unwrap();
        gf.set_meta(&key, "category", CATEGORIES[cat]).unwrap();
        gf.set_meta(&key, "tier", &format!("{}", i % 5)).unwrap();

        let values: Vec<(usize, i128)> = (0..cluster_dims)
            .map(|d| {
                let v = clamp_unit(centroids[cat][d] + rng.jitter(spread));
                (USER_DIM_START + d, v)
            })
            .collect();
        gf.set_semantic(&key, coords_from(&values, dims)).unwrap();
    }
}

/// A directory tree with the dimensions `htt import-fs` derives from real
/// files: size, depth and age. Useful for the filesystem-indexing showcase
/// without touching anybody's actual disk.
fn build_filesystem(gf: &Horon, dims: u8, nodes: usize, rng: &mut Rng) {
    let top = 6usize;
    let mid = 5usize;

    for i in 0..nodes {
        let d1 = i % top;
        let d2 = (i / top) % mid;
        let ext = EXTENSIONS[i % EXTENSIONS.len()];
        let key = format!("/fs/dir_{:02}/sub_{:02}/file_{:05}.{}", d1, d2, i, ext);

        // Sizes span several orders of magnitude, as a real tree does.
        let size = 1u64 << (rng.below(18) + 4);
        let payload = format!("size={} ext={}", size, ext);
        gf.put(&key, payload.as_bytes()).unwrap();
        gf.set_meta(&key, "extension", ext).unwrap();
        gf.set_meta(&key, "bytes", &format!("{}", size)).unwrap();

        // dim 0: log2(size) normalised over a 32-doubling range
        // dim 1: depth, normalised over 8 levels
        // dim 2: age in days, normalised over ~3 years
        let log_size = (63 - size.leading_zeros()) as i128;
        let depth = key.matches('/').count() as i128;
        let age_days = rng.below(1100) as i128;
        let values = vec![
            (USER_DIM_START, clamp_unit(log_size * ONE / 32)),
            (USER_DIM_START + 1, clamp_unit(depth * ONE / 8)),
            (USER_DIM_START + 2, clamp_unit(age_days * ONE / 1100)),
        ];
        gf.set_semantic(&key, coords_from(&values, dims)).unwrap();
    }
}

/// A flat corpus of dense vectors — every user dimension carries signal, the
/// shape an embedding pipeline exports through `htt import-vec`.
fn build_vectors(gf: &Horon, dims: u8, nodes: usize, rng: &mut Rng) {
    let user_dims = (dims as usize).saturating_sub(USER_DIM_START);
    // A handful of latent topics, so the corpus has structure to find rather
    // than being uniform noise.
    let topics = 8usize.min(nodes.max(1));
    let centroids: Vec<Vec<i128>> = (0..topics)
        .map(|_| (0..user_dims).map(|_| rng.unit()).collect())
        .collect();
    let spread = ONE / 12;

    for i in 0..nodes {
        let topic = i % topics;
        let key = format!("/vectors/doc_{:06}", i);
        gf.put(&key, format!("doc-{:06}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "topic", &format!("t{}", topic)).unwrap();

        let values: Vec<(usize, i128)> = (0..user_dims)
            .map(|d| {
                let v = clamp_unit(centroids[topic][d] + rng.jitter(spread));
                (USER_DIM_START + d, v)
            })
            .collect();
        gf.set_semantic(&key, coords_from(&values, dims)).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Format v1 emission
// ---------------------------------------------------------------------------

/// Rewrite a freshly compacted v2 file as v1.
///
/// Per the format spec the sole difference is the trailing snapshot CRC that
/// v2 added, so downgrading is: drop those four bytes, set the version byte,
/// and recompute the header CRC. Generating v1 keeps the
/// upgrade-on-compaction path covered by a fixture that can be regenerated,
/// rather than by a historical file nobody can rebuild.
fn downgrade_to_v1(path: &PathBuf) {
    let mut bytes = fs::read(path).expect("read generated file");
    assert_eq!(bytes[4], 2, "expected a v2 file to downgrade");

    let flags = bytes[5];
    let compressed = flags & 0b1 != 0;
    let snap_byte_len =
        u32::from_le_bytes([bytes[32], bytes[33], bytes[34], bytes[35]]) as usize;

    // Offset of the snapshot CRC: header + snapshot header + snapshot data.
    let mut off = 32 + 8;
    if compressed && snap_byte_len > 0 {
        let clen = u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
            as usize;
        off += 4 + clen;
    } else {
        off += snap_byte_len;
    }

    bytes.drain(off..off + 4);
    bytes[4] = 1;
    let crc = crc32fast::hash(&bytes[0..28]);
    bytes[28..32].copy_from_slice(&crc.to_le_bytes());
    fs::write(path, bytes).expect("write v1 file");
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "gen_fixture — build a synthetic .htt file at scale\n\n\
         USAGE:\n\
         \x20 cargo run --example gen_fixture -- [options]\n\n\
         OPTIONS:\n\
         \x20 --profile <catalog|filesystem|vectors>  data shape (default: catalog)\n\
         \x20 --nodes N        number of leaf nodes (default: 1500)\n\
         \x20 --dims D         total semantic dimensions, >16 (default: 40)\n\
         \x20 --seed S         PRNG seed; same seed = same bytes (default: 1)\n\
         \x20 --out PATH       output file (default: tests/fixtures/<profile>.htt)\n\
         \x20 --v1             emit format v1 (exercises upgrade-on-compaction)\n\
         \x20 --uncompressed   store the snapshot raw instead of zstd\n\
         \x20 --quantized      TQ1.9 semantic tails (format v4)\n"
    );
    exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut profile = "catalog".to_string();
    let mut nodes = 1500usize;
    let mut dims = 40u8;
    let mut seed = 1u64;
    let mut out: Option<PathBuf> = None;
    let mut v1 = false;
    let mut compression = true;
    let mut quantized = false;

    let mut i = 0;
    while i < args.len() {
        let value = |i: usize| -> String {
            args.get(i + 1).cloned().unwrap_or_else(|| usage())
        };
        match args[i].as_str() {
            "--profile" => { profile = value(i); i += 2 }
            "--nodes" => { nodes = value(i).parse().unwrap_or_else(|_| usage()); i += 2 }
            "--dims" => { dims = value(i).parse().unwrap_or_else(|_| usage()); i += 2 }
            "--seed" => { seed = value(i).parse().unwrap_or_else(|_| usage()); i += 2 }
            "--out" => { out = Some(PathBuf::from(value(i))); i += 2 }
            "--v1" => { v1 = true; i += 1 }
            "--uncompressed" => { compression = false; i += 1 }
            "--quantized" => { quantized = true; i += 1 }
            _ => usage(),
        }
    }

    if dims as usize <= USER_DIM_START {
        eprintln!("error: --dims must exceed {} (the reserved region)", USER_DIM_START);
        exit(1);
    }
    if v1 && quantized {
        eprintln!("error: --v1 and --quantized are mutually exclusive versions");
        exit(1);
    }

    let path = out.unwrap_or_else(|| {
        PathBuf::from("tests").join("fixtures").join(format!("{}.htt", profile))
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    let _ = fs::remove_file(&path);

    let mut rng = Rng::new(seed);
    {
        let gf = Horon::open_with_config(&path, HoronConfig {
            dimension: 4,
            semantic_dims: dims,
            compression,
            auto_compact_threshold: 0,
            quantized_semantic: quantized,
            ..Default::default()
        })
        .expect("create fixture");

        match profile.as_str() {
            "catalog" => build_catalog(&gf, dims, nodes, &mut rng),
            "filesystem" => build_filesystem(&gf, dims, nodes, &mut rng),
            "vectors" => build_vectors(&gf, dims, nodes, &mut rng),
            other => {
                eprintln!("error: unknown profile `{}`", other);
                exit(1);
            }
        }

        // Fold everything into the snapshot: a fixture should exercise the
        // snapshot path, and leaves the WAL free for a test to write into.
        gf.compact().unwrap();
    }

    if v1 {
        downgrade_to_v1(&path);
    }

    let size = fs::metadata(&path).unwrap().len();
    let version = fs::read(&path).unwrap()[4];
    println!(
        "{} — profile={} nodes={} dims={} seed={} version={} {} bytes",
        path.display(),
        profile,
        nodes,
        dims,
        seed,
        version,
        size
    );
}
