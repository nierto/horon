//! Conformance corpus generator.
//!
//! Produces the `.htt` files that every independent implementation — the PHP,
//! JavaScript and Python readers, and any third party working from
//! `docs/HTT_FORMAT.md` alone — is graded against. Each generated file is
//! paired with a `.expected.json` describing the logical state a conforming
//! reader must recover from it.
//!
//! The corpus is deliberately synthetic and tiny: every payload is derived
//! from loop indices, so the whole set is byte-reproducible (same guarantee
//! `tests/determinism.rs` asserts for the format at large) and small enough
//! to commit.
//!
//! Run with:
//!
//! ```text
//! cargo run --example gen_corpus            # writes tests/corpus/
//! cargo run --example gen_corpus -- <dir>   # writes <dir>/
//! ```
//!
//! The expected-state dumps are produced by this implementation, which is the
//! reference by definition: they encode what Rust recovers, and a wrapper
//! disagreeing with them is either wrong or has found a spec defect worth
//! filing. They are not an independent oracle.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use horon::{AccessBand, Horon, HoronConfig, HistoryRetention, NodeAccessBands};

/// Bumped when the dump schema changes shape, so wrapper test harnesses can
/// refuse a corpus they do not understand.
const CORPUS_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Case table
// ---------------------------------------------------------------------------

struct Case {
    /// Basename, without extension. Numbered so the directory reads in
    /// roughly increasing order of reader difficulty.
    name: &'static str,
    /// What a reader proves by parsing this file correctly.
    notes: &'static str,
    build: fn(&Path),
}

const CASES: &[Case] = &[
    Case {
        name: "01_empty",
        notes: "No nodes at all: empty snapshot, empty WAL. Exercises the \
                zero-length snapshot layout (always uncompressed, even when \
                the compression flag is set) and the CRC over zero bytes.",
        build: build_empty,
    },
    Case {
        name: "02_structural_only",
        notes: "semantic_dims = 0, so the has-semantic flag is clear and node \
                entries carry no sem_coords field at all. A reader that \
                unconditionally reads coordinates will desynchronise here.",
        build: build_structural_only,
    },
    Case {
        name: "03_uncompressed",
        notes: "Baseline: raw snapshot entries, raw WAL entries, semantic \
                coordinates present at full Q64.64 width. The simplest \
                complete file.",
        build: build_uncompressed,
    },
    Case {
        name: "04_zstd",
        notes: "The same logical content as 03, with the snapshot stored as a \
                single zstd frame. Proves compressed-snapshot handling in \
                isolation (the WAL is empty after compaction).",
        build: build_zstd,
    },
    Case {
        name: "05_wal_live",
        notes: "Uncompacted: most state lives in WAL entries that must be \
                replayed over the snapshot. Written with the default \
                flush-every-append, so each compressed WAL block holds a \
                single entry.",
        build: build_wal_live,
    },
    Case {
        name: "06_wal_batched",
        notes: "As 05 but with wal_batch_size = 8, so blocks hold multiple \
                entries. Catches readers that assume one entry per block \
                (see the block framing note in the format spec).",
        build: build_wal_batched,
    },
    Case {
        name: "07_deletes_and_updates",
        notes: "Live WAL containing INSERT, UPDATE, SET_META and DELETE ops. \
                The expected state is what survives replay, not what was \
                written — a reader that ignores DELETE will report extra \
                nodes.",
        build: build_deletes_and_updates,
    },
    Case {
        name: "08_unicode_keys",
        notes: "Non-ASCII paths (accents, CJK, emoji) and non-ASCII metadata \
                values. Key lengths are byte lengths, not character counts.",
        build: build_unicode_keys,
    },
    Case {
        name: "09_deep_hierarchy",
        notes: "Deep nesting plus wide sibling fan-out, including nodes \
                created implicitly as ancestors. Snapshot entries are \
                parents-before-children; a reader must not assume any \
                stronger ordering.",
        build: build_deep_hierarchy,
    },
    Case {
        name: "10_gacl",
        notes: "Access bands encoded into the reserved semantic dimensions \
                and the GACL flag set in the header. Readers are expected to \
                surface the raw coordinates; enforcement is cooperative and \
                out of scope for a reader.",
        build: build_gacl,
    },
    Case {
        name: "11_meaning_addressed",
        notes: "Format v3: snapshot entries in pure global-Hilbert order \
                rather than depth order. Requires an uncompressed snapshot. \
                A reader must reconstruct the hierarchy from the keys alone.",
        build: build_meaning_addressed,
    },
    Case {
        name: "12_quantized",
        notes: "Format v4: user dimensions stored as TQ1.9 at 2 bytes each. \
                The dumped coordinates are the DECODED full-width values, so \
                a conforming reader must implement dequantisation.",
        build: build_quantized,
    },
    Case {
        name: "13_epochs",
        notes: "EPOCH markers (op 0x06) interleaved with data ops, including \
                a speculative seal. Markers change no node state; a reader \
                that treats them as node ops will corrupt its replay.",
        build: build_epochs,
    },
    Case {
        name: "14_history_sidecar",
        notes: "Compacted with Archive retention, so the pre-fence WAL span \
                is archived into a .h000001 sidecar. The main file must be \
                fully readable on its own; the sidecar is optional history.",
        build: build_history_sidecar,
    },
    Case {
        name: "15_torn_tail",
        notes: "A valid file with a truncated final WAL entry appended. \
                Recovery must stop at the first bad CRC and keep every \
                preceding entry — the expected state is the post-recovery \
                state, and the file is left torn on purpose.",
        build: build_torn_tail,
    },
];

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Q64.64 fractional value, kept well inside the TQ1.9 range (±1.49987) so
/// the same helper serves quantized and full-width files.
fn frac(i: usize) -> [u8; 16] {
    (((i % 20) as i128) << 59).to_le_bytes()
}

/// Semantic coordinate vector with the user dimensions (16+) populated.
fn user_coords(dims: u8, seed: usize) -> Vec<u8> {
    let mut coords = vec![0u8; dims as usize * 16];
    for d in 16..dims as usize {
        let off = d * 16;
        coords[off..off + 16].copy_from_slice(&frac(seed + d));
    }
    coords
}

fn open(path: &Path, config: HoronConfig) -> Horon {
    Horon::open_with_config(path, config).expect("create corpus file")
}

fn build_empty(path: &Path) {
    let gf = open(path, HoronConfig { compression: false, ..Default::default() });
    gf.compact().unwrap();
}

fn build_structural_only(path: &Path) {
    let gf = open(path, HoronConfig {
        semantic_dims: 0,
        compression: false,
        ..Default::default()
    });
    for i in 0..6 {
        let key = format!("/plain/node_{:02}", i);
        gf.put(&key, format!("payload-{}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "kind", "structural").unwrap();
    }
    gf.compact().unwrap();
}

fn build_uncompressed(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false,
        ..Default::default()
    });
    seed_tree(&gf, dims);
    gf.compact().unwrap();
}

fn build_zstd(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: true,
        ..Default::default()
    });
    seed_tree(&gf, dims);
    gf.compact().unwrap();
}

fn build_wal_live(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: true,
        auto_compact_threshold: 0,
        ..Default::default()
    });
    // A small compacted base, then everything else stays in the WAL.
    gf.put("/base/anchor", b"anchor").unwrap();
    gf.compact().unwrap();
    seed_tree(&gf, dims);
    gf.flush().unwrap();
}

fn build_wal_batched(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: true,
        auto_compact_threshold: 0,
        wal_batch_size: 8,
        ..Default::default()
    });
    gf.put("/base/anchor", b"anchor").unwrap();
    gf.compact().unwrap();
    for i in 0..20 {
        let key = format!("/batch/item_{:02}", i);
        gf.put(&key, format!("payload-{:03}", i).as_bytes()).unwrap();
        gf.set_semantic(&key, user_coords(dims, i)).unwrap();
    }
    gf.flush().unwrap();
}

fn build_deletes_and_updates(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    });
    gf.put("/base/anchor", b"anchor").unwrap();
    gf.compact().unwrap();

    for i in 0..9 {
        let key = format!("/churn/item_{}", i);
        gf.put(&key, format!("original-{}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "state", "original").unwrap();
    }
    // Rewrite three payloads (UPDATE replaces data and metadata wholesale).
    for i in [1, 4, 7] {
        let key = format!("/churn/item_{}", i);
        gf.put(&key, format!("rewritten-{}", i).as_bytes()).unwrap();
        gf.set_meta(&key, "state", "rewritten").unwrap();
    }
    // Remove three others, one of which was rewritten above.
    for i in [2, 4, 6] {
        gf.remove(&format!("/churn/item_{}", i)).unwrap();
    }
    gf.flush().unwrap();
}

fn build_unicode_keys(path: &Path) {
    let gf = open(path, HoronConfig {
        semantic_dims: 0,
        compression: false,
        ..Default::default()
    });
    let keys = [
        "/café/menu",
        "/café/menú/día",
        "/文書/第一章",
        "/文書/第二章",
        "/emoji/🌍/🗺",
        "/mixed/naïve_π_42",
    ];
    for (i, key) in keys.iter().enumerate() {
        gf.put(key, format!("payload-{}", i).as_bytes()).unwrap();
        gf.set_meta(key, "título", "valeur — accentuée").unwrap();
    }
    gf.compact().unwrap();
}

fn build_deep_hierarchy(path: &Path) {
    let gf = open(path, HoronConfig {
        semantic_dims: 0,
        compression: false,
        ..Default::default()
    });
    // One deep spine; leaf nodes are written without their ancestors, which
    // the store materialises implicitly.
    gf.put("/deep/a/b/c/d/e/f/leaf", b"deep").unwrap();
    for i in 0..8 {
        gf.put(&format!("/deep/wide/child_{:02}", i), format!("wide-{}", i).as_bytes()).unwrap();
    }
    for i in 0..3 {
        gf.put(&format!("/deep/a/b/sibling_{}", i), b"sib").unwrap();
    }
    gf.compact().unwrap();
}

fn build_gacl(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false,
        gacl: true,
        ..Default::default()
    });

    gf.put("/secure/public", b"anyone").unwrap();
    gf.set_semantic("/secure/public", NodeAccessBands::public().to_semantic_bytes(dims as usize))
        .unwrap();

    let mut restricted = NodeAccessBands::public();
    restricted.read = AccessBand::from_f64(0.5, 1.0);
    restricted.write = AccessBand::from_f64(0.8, 1.0);
    gf.put("/secure/restricted", b"privileged").unwrap();
    gf.set_semantic("/secure/restricted", restricted.to_semantic_bytes(dims as usize)).unwrap();

    let mut closed = NodeAccessBands::public();
    closed.read = AccessBand::closed();
    gf.put("/secure/sealed", b"nobody").unwrap();
    gf.set_semantic("/secure/sealed", closed.to_semantic_bytes(dims as usize)).unwrap();

    gf.compact().unwrap();
}

fn build_meaning_addressed(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false, // required by the meaning-addressed layout
        meaning_addressed: true,
        ..Default::default()
    });
    for i in 0..12 {
        let key = format!("/corpus/doc_{:02}", i);
        gf.put(&key, format!("document-{:02}", i).as_bytes()).unwrap();
        gf.set_semantic(&key, user_coords(dims, i * 3)).unwrap();
    }
    gf.compact().unwrap();
}

fn build_quantized(path: &Path) {
    let dims: u8 = 40;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false,
        quantized_semantic: true,
        ..Default::default()
    });
    for i in 0..10 {
        let key = format!("/vectors/embedding_{:02}", i);
        gf.put(&key, format!("vec-{:02}", i).as_bytes()).unwrap();
        gf.set_semantic(&key, user_coords(dims, i)).unwrap();
    }
    gf.compact().unwrap();
}

fn build_epochs(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false,
        auto_compact_threshold: 0,
        ..Default::default()
    });
    gf.put("/base/anchor", b"anchor").unwrap();
    gf.compact().unwrap();

    for round in 0..3 {
        for i in 0..3 {
            let key = format!("/timeline/round_{}/item_{}", round, i);
            gf.put(&key, format!("r{}-i{}", round, i).as_bytes()).unwrap();
            gf.set_semantic(&key, user_coords(dims, round * 5 + i)).unwrap();
        }
        gf.seal_epoch().unwrap();
    }
    // A speculative seal labels a projected state; it is still just a marker.
    gf.put("/timeline/projected", b"what-if").unwrap();
    gf.seal_speculative_epoch().unwrap();
    gf.flush().unwrap();
}

fn build_history_sidecar(path: &Path) {
    let dims: u8 = 20;
    let gf = open(path, HoronConfig {
        semantic_dims: dims,
        compression: false,
        auto_compact_threshold: 0,
        history_retention: HistoryRetention::Archive,
        ..Default::default()
    });
    for round in 0..2 {
        for i in 0..4 {
            let key = format!("/archive/item_{}", i);
            gf.put(&key, format!("round-{}-value-{}", round, i).as_bytes()).unwrap();
            gf.set_semantic(&key, user_coords(dims, round * 7 + i)).unwrap();
        }
        gf.seal_epoch().unwrap();
    }
    // Compaction archives the pre-fence WAL span into <file>.h000001.
    gf.compact().unwrap();
    gf.put("/archive/after_compaction", b"live").unwrap();
    gf.flush().unwrap();
}

fn build_torn_tail(path: &Path) {
    let dims: u8 = 20;
    {
        let gf = open(path, HoronConfig {
            semantic_dims: dims,
            compression: false, // raw entries, so the tear is a plain truncation
            auto_compact_threshold: 0,
            ..Default::default()
        });
        gf.put("/base/anchor", b"anchor").unwrap();
        gf.compact().unwrap();
        for i in 0..6 {
            let key = format!("/intact/entry_{}", i);
            gf.put(&key, format!("survives-{}", i).as_bytes()).unwrap();
        }
        gf.flush().unwrap();
    }

    // Append a half-written entry: a plausible header followed by bytes that
    // stop short of the payload and CRC the entry claims.
    let mut bytes = fs::read(path).unwrap();
    let seq: u32 = 9_999;
    bytes.extend_from_slice(&seq.to_le_bytes());
    bytes.push(0x01); // INSERT
    let key = b"/torn/never_committed";
    bytes.extend_from_slice(&(key.len() as u16).to_le_bytes());
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&(64u32).to_le_bytes()); // claims 64 payload bytes
    bytes.extend_from_slice(b"only-a-few"); // ...and delivers ten
    fs::write(path, bytes).unwrap();
}

/// Shared content for the plain full-width cases.
fn seed_tree(gf: &Horon, dims: u8) {
    for i in 0..12 {
        let key = format!("/tree/branch_{}/leaf_{:02}", i % 3, i);
        gf.put(&key, format!("payload-{:04}", i * 7).as_bytes()).unwrap();
        gf.set_meta(&key, "tier", &format!("{}", i % 5)).unwrap();
        gf.set_meta(&key, "kind", "leaf").unwrap();
        gf.set_semantic(&key, user_coords(dims, i)).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Expected-state dump
// ---------------------------------------------------------------------------

/// Metadata keys the in-memory store synthesises on replay and the writer
/// deliberately excludes from the file (`created_at`/`updated_at` are wall
/// clocks, `key`/`size` are derivable). Rust's `get_meta` returns them, but no
/// reader working from the bytes can — so the expected state must not claim
/// them. Everything else, including `_child_index` and `content_type`, IS
/// persisted and a conforming reader must surface it.
const SYNTHESIZED_META: &[&str] = &["key", "size", "created_at", "updated_at"];

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Decode the 32-byte header straight from disk, independently of the
/// library's own structs — the dump should describe the bytes a wrapper will
/// actually meet.
fn dump_header(bytes: &[u8]) -> String {
    let flags = bytes[5];
    let algo = match (flags >> 1) & 0b11 {
        0 => "zstd",
        1 => "zlib",
        2 => "lz4",
        _ => "reserved",
    };
    let compressed = flags & 0b1 != 0;
    let u32_at = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    format!(
        "  \"header\": {{\n\
         \x20   \"magic_hex\": \"{}\",\n\
         \x20   \"version\": {},\n\
         \x20   \"flags_byte\": {},\n\
         \x20   \"flags\": {{\n\
         \x20     \"compressed\": {},\n\
         \x20     \"algorithm\": {},\n\
         \x20     \"has_semantic\": {},\n\
         \x20     \"wal_block_compressed\": {},\n\
         \x20     \"gacl\": {},\n\
         \x20     \"meaning_addressed\": {},\n\
         \x20     \"quantized\": {}\n\
         \x20   }},\n\
         \x20   \"dimension\": {},\n\
         \x20   \"semantic_dims\": {},\n\
         \x20   \"tau_hex\": \"{}\",\n\
         \x20   \"node_count\": {},\n\
         \x20   \"header_crc\": {}\n\
         \x20 }},",
        hex(&bytes[0..4]),
        bytes[4],
        flags,
        compressed,
        if compressed { format!("\"{}\"", algo) } else { "null".to_string() },
        flags & (1 << 3) != 0,
        flags & (1 << 4) != 0,
        flags & (1 << 5) != 0,
        flags & (1 << 6) != 0,
        flags & (1 << 7) != 0,
        bytes[6],
        bytes[7],
        hex(&bytes[8..24]),
        u32_at(24),
        u32_at(28),
    )
}

/// Read a corpus file back and describe the logical state a conforming reader
/// must recover. Nodes and metadata are sorted so the dump is canonical.
fn dump_expected(name: &str, notes: &str, htt: &Path, sidecars: &[String]) -> String {
    let raw = fs::read(htt).unwrap();

    // Dump from a copy: opening performs recovery, and the torn-tail case must
    // stay torn on disk.
    let scratch = htt.with_extension("dumpcopy");
    fs::copy(htt, &scratch).unwrap();
    let gf = Horon::open_with_config(&scratch, HoronConfig {
        lazy_geometry: true,
        auto_compact_threshold: 0,
        ..Default::default()
    })
    .expect("reopen corpus file");

    let mut keys = gf.list("/").unwrap();
    keys.sort();

    let mut nodes = String::new();
    for (i, key) in keys.iter().enumerate() {
        let data = gf.get(key).unwrap_or_default();
        let meta: BTreeMap<String, String> = gf
            .get_meta(key)
            .unwrap_or_default()
            .into_iter()
            .filter(|(k, _)| !SYNTHESIZED_META.contains(&k.as_str()))
            .collect();
        let meta_json = meta
            .iter()
            .map(|(k, v)| format!("[\"{}\", \"{}\"]", json_escape(k), json_escape(v)))
            .collect::<Vec<_>>()
            .join(", ");
        let semantic = gf.get_semantic(key).map(|c| hex(&c)).unwrap_or_default();

        nodes.push_str(&format!(
            "    {{\n\
             \x20     \"key\": \"{}\",\n\
             \x20     \"data_hex\": \"{}\",\n\
             \x20     \"meta\": [{}],\n\
             \x20     \"semantic_hex\": \"{}\"\n\
             \x20   }}{}\n",
            json_escape(key),
            hex(&data),
            meta_json,
            semantic,
            if i + 1 == keys.len() { "" } else { "," },
        ));
    }

    let live = keys.len();
    let epoch = gf.current_epoch();
    drop(gf);
    let _ = fs::remove_file(&scratch);

    let sidecar_json = sidecars
        .iter()
        .map(|s| format!("\"{}\"", json_escape(s)))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
        "{{\n\
         \x20 \"corpus_version\": {},\n\
         \x20 \"file\": \"{}.htt\",\n\
         \x20 \"notes\": \"{}\",\n\
         {}\n\
         \x20 \"file_bytes\": {},\n\
         \x20 \"sidecars\": [{}],\n\
         \x20 \"current_epoch\": {},\n\
         \x20 \"live_nodes\": {},\n\
         \x20 \"nodes\": [\n{}  ]\n\
         }}\n",
        CORPUS_VERSION,
        name,
        json_escape(notes),
        dump_header(&raw),
        raw.len(),
        sidecar_json,
        epoch,
        live,
        nodes,
    )
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn main() {
    let out_dir: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/corpus"));
    fs::create_dir_all(&out_dir).expect("create corpus directory");

    let mut manifest = Vec::new();

    for case in CASES {
        let htt = out_dir.join(format!("{}.htt", case.name));
        // Regenerate from scratch: builders open existing files rather than
        // replacing them.
        let _ = fs::remove_file(&htt);
        for existing in sidecars_of(&htt) {
            let _ = fs::remove_file(out_dir.join(existing));
        }

        (case.build)(&htt);

        let sidecars = sidecars_of(&htt);
        let expected = dump_expected(case.name, case.notes, &htt, &sidecars);
        fs::write(out_dir.join(format!("{}.expected.json", case.name)), expected)
            .expect("write expected dump");

        let size = fs::metadata(&htt).unwrap().len();
        println!("{:<24} {:>7} bytes{}", case.name, size, if sidecars.is_empty() {
            String::new()
        } else {
            format!("  (+{} sidecar)", sidecars.len())
        });
        manifest.push((case.name, case.notes, sidecars));
    }

    let entries = manifest
        .iter()
        .map(|(name, notes, sidecars)| {
            let sc = sidecars
                .iter()
                .map(|s| format!("\"{}\"", json_escape(s)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "    {{ \"name\": \"{}\", \"file\": \"{}.htt\", \"expected\": \"{}.expected.json\", \"sidecars\": [{}], \"notes\": \"{}\" }}",
                name, name, name, sc, json_escape(notes)
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    fs::write(
        out_dir.join("manifest.json"),
        format!(
            "{{\n  \"corpus_version\": {},\n  \"cases\": [\n{}\n  ]\n}}\n",
            CORPUS_VERSION, entries
        ),
    )
    .expect("write manifest");

    println!("\n{} cases written to {}", CASES.len(), out_dir.display());
}

/// History sidecars written alongside a corpus file, oldest first.
fn sidecars_of(htt: &Path) -> Vec<String> {
    let dir = match htt.parent() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let stem = match htt.file_name().and_then(|s| s.to_str()) {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut found: Vec<String> = fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().to_str().map(String::from))
        .filter(|n| n.starts_with(&format!("{}.h", stem)))
        .collect();
    found.sort();
    found
}
