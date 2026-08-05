//! htt — command-line manager for `.htt` files (import pipelines).
//!
//! ```text
//! htt import-csv    <in.csv> <out.htt> --path-cols A,B [--key-col C]
//!                   [--dim-cols X,Y,Z] [--meaning-addressed]
//! htt import-fs     <root-dir> <out.htt> [--max-bytes N] [--meaning-addressed]
//! htt import-json   <in.json|in.yaml> <out.htt> [--dim-fields a,b]
//!                   [--meaning-addressed] [--quantized]
//! htt import-sqlite <db.sqlite> <out.htt> [--tables t1,t2]
//!                   [--dim-cols t.c,...] [--key-cols t.c,...]
//!                   [--meaning-addressed] [--quantized]
//! htt import-vec    <in.jsonl> <out.htt> [--id-field id] [--vector-field vector]
//!                   [--take-dims N] [--meaning-addressed] [--quantized]
//! htt inspect       <file.htt>
//! htt query         <file.htt> <key> [k]
//! ```
//!
//! Build: cargo build --release --features import

use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::exit;

use horon::header::GeoHeader;
use horon::{Horon, HoronConfig};
use g_math::fixed_point::FixedPoint;

const USER_DIM_START: usize = 16;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("import-csv") => import_csv(&args[1..]),
        Some("import-fs") => import_fs(&args[1..]),
        Some("import-json") => import_json(&args[1..]),
        Some("import-sqlite") => import_sqlite(&args[1..]),
        Some("import-vec") => import_vec(&args[1..]),
        Some("inspect") => inspect(&args[1..]),
        Some("query") => query(&args[1..]),
        _ => {
            eprintln!(
                "htt — manage .htt files\n\n\
                 USAGE:\n\
                 \x20 htt import-csv <in.csv> <out.htt> --path-cols A,B [--key-col C]\n\
                 \x20                [--dim-cols X,Y,Z] [--meaning-addressed]\n\
                 \x20 htt import-fs  <root-dir> <out.htt> [--max-bytes N] [--meaning-addressed]\n\
                 \x20 htt import-json <in.json|.yaml> <out.htt> [--dim-fields a,b]\n\
                 \x20                [--meaning-addressed] [--quantized]\n\
                 \x20 htt import-sqlite <db.sqlite> <out.htt> [--tables t1,t2]\n\
                 \x20                [--dim-cols t.c,...] [--key-cols t.c,...]\n\
                 \x20                [--meaning-addressed] [--quantized]\n\
                 \x20 htt import-vec <in.jsonl> <out.htt> [--id-field id]\n\
                 \x20                [--vector-field vector] [--take-dims N]\n\
                 \x20                [--meaning-addressed] [--quantized]\n\
                 \x20 htt inspect    <file.htt>\n\
                 \x20 htt query      <file.htt> <key> [k]\n\n\
                 import-csv: --path-cols columns become the folder hierarchy, --dim-cols\n\
                 numeric columns become semantic dimensions (auto-normalized), all other\n\
                 columns become metadata. --meaning-addressed stores the file so that\n\
                 similar rows are physically adjacent bytes (format v3).\n\n\
                 import-fs: mirrors a directory tree; dimensions are log2(size), depth,\n\
                 and age in days. Files over --max-bytes (default 65536) are skipped.\n\n\
                 import-json: nested objects/arrays become paths, scalar members become\n\
                 metadata, --dim-fields numeric members become semantic dimensions.\n\
                 YAML is detected by .yaml/.yml extension. Parses the whole document\n\
                 in memory (unlike the streaming csv importer).\n\n\
                 import-sqlite: foreign keys become the hierarchy — child rows nest\n\
                 under the parent row they reference (self-referencing tables nest\n\
                 recursively); row columns become metadata, --dim-cols become semantic\n\
                 dimensions. Buffers rows in memory.\n\n\
                 import-vec: one JSON object per line with an embedding vector — the\n\
                 quantized-storage pairing: --quantized stores dims at 2 bytes each (TQ1.9, range\n\
                 ±1.49987 — unit-norm embeddings fit comfortably); recommended for\n\
                 vector corpora. --take-dims N keeps the leading N components\n\
                 (legitimate for matryoshka-style embeddings, lossy otherwise)."
            );
            exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {}", e);
        exit(1);
    }
}

// ---------------------------------------------------------------------------
// Small arg helpers (no clap — this is an experimentation tool)
// ---------------------------------------------------------------------------

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

/// Flags that consume the following argument as their value. Needed so
/// `positional` does not mistake a flag's value for a positional argument
/// (e.g. `--path-cols A,B in.csv` must treat `A,B` as the flag value, not as
/// positional #0).
const VALUE_FLAGS: &[&str] = &[
    "--path-cols", "--dim-cols", "--key-col", "--max-bytes",
    "--dim-fields", "--tables", "--key-cols", "--id-field", "--vector-field", "--take-dims",
];

fn positional(args: &[String], n: usize) -> Result<&str, String> {
    let mut positionals = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if a.starts_with("--") {
            // Skip the flag, and its value too if it is value-taking.
            i += if VALUE_FLAGS.contains(&a.as_str()) { 2 } else { 1 };
            continue;
        }
        positionals.push(a.as_str());
        i += 1;
    }
    positionals
        .get(n)
        .copied()
        .ok_or_else(|| format!("missing positional argument #{}", n + 1))
}

/// Path-segment sanitizer: no separators, no empties.
fn sanitize(seg: &str) -> String {
    let s: String = seg
        .trim()
        .chars()
        .map(|c| match c {
            '/' | '\\' => '-',
            c if c.is_whitespace() => '_',
            c => c,
        })
        .collect();
    if s.is_empty() { "_".to_string() } else { s }
}


/// Parse a decimal dimension value exactly.
///
/// `f64::parse` is used only to reject non-numeric and non-finite input; the
/// value itself comes from `FixedPoint::from_str`, which reads the decimal
/// digits straight into Q64.64. Going through f64 would quantize every
/// imported number to 53 bits before it reached a 64-bit fractional type.
fn parse_dim(raw: &str) -> Result<FixedPoint, DimParseError> {
    let probe: f64 = raw.parse().map_err(|_| DimParseError::NotNumeric)?;
    if !probe.is_finite() {
        return Err(DimParseError::NotFinite);
    }
    Ok(FixedPoint::from_str(raw))
}

/// Why a dimension value was rejected. Kept distinct because the CLI reports
/// them differently and users act on the difference.
#[derive(Debug, PartialEq)]
enum DimParseError {
    NotNumeric,
    NotFinite,
}

/// Encode user-dim values as a full semantic vector (16 reserved + dims).
fn encode_dims(vals: &[FixedPoint], total_dims: usize) -> Vec<u8> {
    let mut out = vec![0u8; total_dims * 16];
    for (d, v) in vals.iter().enumerate() {
        let off = (USER_DIM_START + d) * 16;
        out[off..off + 16].copy_from_slice(&v.raw().to_le_bytes());
    }
    out
}

/// log2 in fixed point: `ln(x) / ln(2)`. gMath has no log2, and the float
/// one would put a transcendental on the path that produces coordinates.
fn log2_fixed(x: FixedPoint) -> FixedPoint {
    let ln2 = FixedPoint::from_str("0.69314718055994530942");
    if x <= FixedPoint::from_int(0) {
        return FixedPoint::from_int(0);
    }
    x.ln() / ln2
}

/// Pad a [lo, hi] span by 5% each side so nothing sits on the edge. A
/// degenerate span (single value, or lo == hi) becomes a unit interval.
fn pad_bounds(lo: FixedPoint, hi: FixedPoint) -> (FixedPoint, FixedPoint) {
    let one = FixedPoint::from_int(1);
    if hi <= lo {
        return (lo - one, lo + one);
    }
    let pad = (hi - lo) / FixedPoint::from_int(20); // 5%
    (lo - pad, hi + pad)
}

/// Global padded bounds over every dim value.
fn padded_bounds(values: &[FixedPoint]) -> (FixedPoint, FixedPoint) {
    if values.is_empty() {
        return (FixedPoint::from_int(0), FixedPoint::from_int(1));
    }
    let mut lo = values[0];
    let mut hi = values[0];
    for &v in values {
        if v < lo { lo = v; }
        if v > hi { hi = v; }
    }
    pad_bounds(lo, hi)
}

fn build_config(
    user_dims: usize,
    meaning_addressed: bool,
    bounds: (FixedPoint, FixedPoint),
    quantized: bool,
) -> HoronConfig {
    // The v3 bounds section stores f64 pairs, so this is the one conversion
    // the format requires; every computation above it is fixed point.
    let bounds = (bounds.0.to_f64(), bounds.1.to_f64());
    HoronConfig {
        dimension: 4,
        semantic_dims: (USER_DIM_START + user_dims) as u8,
        compression: false, // required by meaning-addressed & partial modes
        auto_compact_threshold: 0,
        meaning_addressed,
        semantic_bounds: bounds,
        quantized_semantic: quantized,
        ..Default::default()
    }
}

/// Shared guards for importers that produce semantic dimensions: the on-disk
/// dim count is a u8 (reserved + user must fit), and `--quantized` needs at
/// least one user dim to quantize.
fn check_dims(user_dims: usize, quantized: bool) -> Result<(), String> {
    let total = USER_DIM_START + user_dims;
    if total > u8::MAX as usize {
        return Err(format!(
            "too many dimensions: {} user + {} reserved = {} exceeds the {}-dimension              format limit (use --take-dims / fewer dim fields)",
            user_dims, USER_DIM_START, total, u8::MAX
        ));
    }
    if quantized && user_dims == 0 {
        return Err("--quantized requires at least one semantic dimension".into());
    }
    Ok(())
}

/// TQ1.9 value range guard for `--quantized` imports: fail during the bounds
/// pass, before any file is created, naming the offending value.
fn check_tq19_range(v: FixedPoint, what: &str) -> Result<(), String> {
    const TQ19_MAX: f64 = 29_524.0 / 19_683.0;
    let v = v.to_f64();
    if v.abs() > TQ19_MAX {
        return Err(format!(
            "{}: value {} is outside the TQ1.9 range ±{:.5} required by --quantized              (normalize the source values, or drop --quantized)",
            what, v, TQ19_MAX
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// import-csv
// ---------------------------------------------------------------------------

fn import_csv(args: &[String]) -> Result<(), String> {
    let input = positional(args, 0)?;
    let output = positional(args, 1)?;
    let path_cols: Vec<&str> = flag_value(args, "--path-cols")
        .ok_or("--path-cols is required (comma-separated column names)")?
        .split(',')
        .map(str::trim)
        .collect();
    let dim_cols: Vec<&str> = flag_value(args, "--dim-cols")
        .map(|v| v.split(',').map(str::trim).collect())
        .unwrap_or_default();
    let key_col = flag_value(args, "--key-col");
    let meaning_addressed = has_flag(args, "--meaning-addressed");
    if meaning_addressed && dim_cols.is_empty() {
        return Err("--meaning-addressed requires --dim-cols".into());
    }

    let headers = csv_headers(input)?;
    let col_idx = |name: &str| -> Result<usize, String> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("column '{}' not found (headers: {})", name, headers.join(", ")))
    };
    let path_idx: Vec<usize> = path_cols.iter().map(|c| col_idx(c)).collect::<Result<_, _>>()?;
    let dim_idx: Vec<usize> = dim_cols.iter().map(|c| col_idx(c)).collect::<Result<_, _>>()?;
    let key_idx = key_col.map(|c| col_idx(c)).transpose()?;

    // The on-disk semantic_dims field is a u8: reserved dims + user dims must
    // fit in it, or the count would silently wrap and corrupt the file.
    let total_dims = USER_DIM_START + dim_idx.len();
    if total_dims > u8::MAX as usize {
        return Err(format!(
            "too many --dim-cols: {} user dims + {} reserved = {} exceeds the {}-dimension format limit",
            dim_idx.len(), USER_DIM_START, total_dims, u8::MAX
        ));
    }

    if Path::new(output).exists() {
        return Err(format!("refusing to overwrite existing file: {}", output));
    }

    // Pass 1 (streaming): running min/max over dim values for global
    // normalization bounds. Rows are not buffered — only the min/max survive,
    // so memory does not grow with the CSV. Non-numeric or non-finite dim
    // values (NaN/inf parse as valid f64 and would poison the bounds) are
    // rejected here with a row number.
    let bounds = if dim_idx.is_empty() {
        (FixedPoint::from_int(0), FixedPoint::from_int(1))
    } else {
        let mut seen = false;
        let mut lo = FixedPoint::from_int(0);
        let mut hi = FixedPoint::from_int(0);
        let mut reader = csv::Reader::from_path(input).map_err(|e| e.to_string())?;
        for (n, rec) in reader.records().enumerate() {
            let row = rec.map_err(|e| e.to_string())?;
            for &i in &dim_idx {
                let raw = row.get(i).unwrap_or("").trim();
                let v = parse_dim(raw).map_err(|e| match e {
                    DimParseError::NotNumeric => format!(
                        "row {}: '{}' in dim column '{}' is not numeric",
                        n + 2, raw, headers[i]
                    ),
                    DimParseError::NotFinite => format!(
                        "row {}: '{}' in dim column '{}' is not a finite number",
                        n + 2, raw, headers[i]
                    ),
                })?;
                if !seen {
                    lo = v;
                    hi = v;
                    seen = true;
                } else {
                    if v < lo { lo = v; }
                    if v > hi { hi = v; }
                }
            }
        }
        if seen { pad_bounds(lo, hi) } else { (FixedPoint::from_int(0), FixedPoint::from_int(1)) }
    };

    let gf = Horon::open_with_config(output, build_config(dim_idx.len(), meaning_addressed, bounds, false))
        .map_err(|e| e.to_string())?;

    // Pass 2 (streaming): insert. Keys are tracked in a set only to report
    // collisions (rows that map to the same key silently upsert); this holds
    // the distinct key strings, far smaller than buffering every record.
    let mut seen: HashSet<String> = HashSet::new();
    let mut collisions = 0usize;
    let mut imported = 0usize;
    let mut reader = csv::Reader::from_path(input).map_err(|e| e.to_string())?;
    for (n, rec) in reader.records().enumerate() {
        let row = rec.map_err(|e| e.to_string())?;
        let mut key = String::new();
        for &i in &path_idx {
            key.push('/');
            key.push_str(&sanitize(row.get(i).unwrap_or("")));
        }
        key.push('/');
        match key_idx {
            Some(i) => key.push_str(&sanitize(row.get(i).unwrap_or(""))),
            None => key.push_str(&format!("row_{:05}", n)),
        }

        if !seen.insert(key.clone()) {
            collisions += 1;
        }

        let payload: Vec<String> = row.iter().map(str::to_string).collect();
        gf.put(&key, payload.join(",").as_bytes()).map_err(|e| e.to_string())?;

        for (i, header) in headers.iter().enumerate() {
            if path_idx.contains(&i) || dim_idx.contains(&i) || key_idx == Some(i) {
                continue;
            }
            let val = row.get(i).unwrap_or("").trim();
            if !val.is_empty() {
                let _ = gf.set_meta(&key, header, val);
            }
        }
        if !dim_idx.is_empty() {
            // Pass 1 already validated these as finite numbers.
            let vals: Vec<FixedPoint> = dim_idx
                .iter()
                .map(|&i| {
                    parse_dim(row.get(i).unwrap_or("0").trim())
                        .unwrap_or_else(|_| FixedPoint::from_int(0))
                })
                .collect();
            gf.set_semantic(&key, encode_dims(&vals, total_dims))
                .map_err(|e| e.to_string())?;
        }
        imported += 1;
    }
    gf.compact().map_err(|e| e.to_string())?;

    println!("imported {} rows → {}", imported, output);
    println!("  hierarchy:  /{}/…", path_cols.join("/"));
    println!("  dimensions: {} ({})", dim_idx.len(), dim_cols.join(", "));
    println!("  bounds:     {:.3} .. {:.3}", bounds.0.to_f64(), bounds.1.to_f64());
    println!("  layout:     {}", if meaning_addressed { "meaning-addressed (v3)" } else { "standard (v2)" });
    if collisions > 0 {
        println!(
            "  ⚠ {} row(s) collided onto an existing key and overwrote it \
             (add or widen --key-col for a unique key per row)",
            collisions
        );
    }
    println!("  file size:  {} bytes", std::fs::metadata(output).map(|m| m.len()).unwrap_or(0));
    println!("\ntry:  htt query {} <key> 5", output);
    Ok(())
}

/// Read just the header row of a CSV (opens, reads headers, closes).
fn csv_headers(input: &str) -> Result<Vec<String>, String> {
    let mut reader = csv::Reader::from_path(input).map_err(|e| e.to_string())?;
    Ok(reader
        .headers()
        .map_err(|e| e.to_string())?
        .iter()
        .map(str::to_string)
        .collect())
}

// ---------------------------------------------------------------------------
// import-fs
// ---------------------------------------------------------------------------

fn import_fs(args: &[String]) -> Result<(), String> {
    let root = PathBuf::from(positional(args, 0)?);
    let output = positional(args, 1)?;
    let max_bytes: u64 = flag_value(args, "--max-bytes")
        .map(|v| v.parse().map_err(|_| "--max-bytes must be a number"))
        .transpose()?
        .unwrap_or(65_536);
    let meaning_addressed = has_flag(args, "--meaning-addressed");

    // Pass 1: walk, collect candidate files + dim values.
    struct Item {
        rel: String,
        size: u64,
        depth: FixedPoint,
        age_days: FixedPoint,
    }
    let mut items = Vec::new();
    let mut skipped = 0usize;
    let now = std::time::SystemTime::now();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name == ".git" || name == "target" || name == "node_modules" {
                continue;
            }
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() {
                if meta.len() > max_bytes {
                    skipped += 1;
                    continue;
                }
                let rel = path
                    .strip_prefix(&root)
                    .map_err(|e| e.to_string())?
                    .to_string_lossy()
                    .replace('\\', "/");
                // Integer seconds divided in fixed point — the float
                // division was the only step between the OS value and a
                // stored coordinate.
                let age_days = meta
                    .modified()
                    .ok()
                    .and_then(|m| now.duration_since(m).ok())
                    .map(|d| {
                        FixedPoint::from_int(d.as_secs().min(i32::MAX as u64) as i32)
                            / FixedPoint::from_int(86_400)
                    })
                    .unwrap_or_else(|| FixedPoint::from_int(0));
                items.push(Item {
                    depth: FixedPoint::from_int(rel.matches('/').count() as i32),
                    size: meta.len(),
                    age_days,
                    rel,
                });
            }
        }
    }
    if items.is_empty() {
        return Err("no files found (all skipped or directory empty)".into());
    }

    let mut all_vals = Vec::new();
    for it in &items {
        all_vals.push(log2_fixed(FixedPoint::from_int(it.size.min(i32::MAX as u64) as i32) + FixedPoint::from_int(1)));
        all_vals.push(it.depth);
        all_vals.push(it.age_days);
    }
    let bounds = padded_bounds(&all_vals);

    if Path::new(output).exists() {
        return Err(format!("refusing to overwrite existing file: {}", output));
    }
    let gf = Horon::open_with_config(output, build_config(3, meaning_addressed, bounds, false))
        .map_err(|e| e.to_string())?;

    for it in &items {
        let key = format!("/{}", it.rel.split('/').map(sanitize).collect::<Vec<_>>().join("/"));
        let data = std::fs::read(root.join(&it.rel)).map_err(|e| e.to_string())?;
        gf.put(&key, &data).map_err(|e| e.to_string())?;
        let _ = gf.set_meta(&key, "bytes", &it.size.to_string());
        if let Some(ext) = Path::new(&it.rel).extension() {
            let _ = gf.set_meta(&key, "ext", &ext.to_string_lossy());
        }
        let dims = [
            log2_fixed(FixedPoint::from_int(it.size.min(i32::MAX as u64) as i32) + FixedPoint::from_int(1)),
            it.depth,
            it.age_days,
        ];
        gf.set_semantic(&key, encode_dims(&dims, USER_DIM_START + 3))
            .map_err(|e| e.to_string())?;
    }
    gf.compact().map_err(|e| e.to_string())?;

    println!("imported {} files → {} ({} skipped over {} bytes)", items.len(), output, skipped, max_bytes);
    println!("  dimensions: log2(size), depth, age_days");
    println!("  bounds:     {:.3} .. {:.3}", bounds.0.to_f64(), bounds.1.to_f64());
    println!("  layout:     {}", if meaning_addressed { "meaning-addressed (v3)" } else { "standard (v2)" });
    println!("  file size:  {} bytes", std::fs::metadata(output).map(|m| m.len()).unwrap_or(0));
    Ok(())
}


// ---------------------------------------------------------------------------
// import-json — nested JSON/YAML → paths
// ---------------------------------------------------------------------------

/// Parse a JSON or YAML document (by extension) into a `serde_json::Value`.
fn parse_document(input: &str) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(input).map_err(|e| format!("{}: {}", input, e))?;
    if input.ends_with(".yaml") || input.ends_with(".yml") {
        serde_yaml::from_str(&text).map_err(|e| format!("{}: {}", input, e))
    } else {
        serde_json::from_str(&text).map_err(|e| format!("{}: {}", input, e))
    }
}

/// Zero-padded array-index segment so lexicographic key order matches
/// element order (width from the largest index).
fn index_segment(i: usize, len: usize) -> String {
    let width = len.saturating_sub(1).max(1).to_string().len().max(1);
    format!("{:0width$}", i, width = width)
}

/// Object keys in sorted order — explicit, so determinism never depends on
/// the map backing serde_json happens to be built with.
fn sorted_keys(obj: &serde_json::Map<String, serde_json::Value>) -> Vec<&String> {
    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    keys
}

/// Pass 1: every numeric `--dim-fields` value in the tree (for bounds),
/// erroring on dim fields that exist but are not numbers.
fn collect_json_dims(
    v: &serde_json::Value,
    at: &str,
    dim_fields: &[&str],
    out: &mut Vec<FixedPoint>,
) -> Result<(), String> {
    match v {
        serde_json::Value::Object(obj) => {
            for key in sorted_keys(obj) {
                let child = &obj[key];
                if dim_fields.contains(&key.as_str()) {
                    let n = child.as_f64().ok_or_else(|| {
                        format!("{}/{}: dim field is not a finite number", at, key)
                    })?;
                    if !n.is_finite() {
                        return Err(format!("{}/{}: dim field is not finite", at, key));
                    }
                    out.push(FixedPoint::from_f64(n));
                } else if child.is_object() || child.is_array() {
                    collect_json_dims(child, &format!("{}/{}", at, sanitize(key)), dim_fields, out)?;
                }
            }
        }
        serde_json::Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                if item.is_object() || item.is_array() {
                    let seg = index_segment(i, items.len());
                    collect_json_dims(item, &format!("{}/{}", at, seg), dim_fields, out)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Pass 2: emit nodes. Objects/arrays become paths; an object's scalar
/// members become its metadata; array scalar elements become leaf nodes
/// (payload = the scalar); `--dim-fields` numeric members become the object
/// node's semantic dimensions (consumed — not repeated as metadata).
fn emit_json(
    gf: &Horon,
    path: &str,
    v: &serde_json::Value,
    dim_fields: &[&str],
    total_dims: usize,
    nodes: &mut usize,
) -> Result<(), String> {
    match v {
        serde_json::Value::Object(obj) => {
            if path != "/" {
                gf.put(path, b"").map_err(|e| e.to_string())?;
                *nodes += 1;
            }
            let mut dims: Vec<FixedPoint> = vec![FixedPoint::from_int(0); dim_fields.len()];
            let mut has_dims = false;
            for key in sorted_keys(obj) {
                let child = &obj[key];
                if let Some(d) = dim_fields.iter().position(|f| f == key) {
                    if let Some(n) = child.as_f64() {
                        dims[d] = FixedPoint::from_f64(n);
                        has_dims = true;
                        continue; // consumed as a dimension
                    }
                }
                let child_path = if path == "/" {
                    format!("/{}", sanitize(key))
                } else {
                    format!("{}/{}", path, sanitize(key))
                };
                match child {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        emit_json(gf, &child_path, child, dim_fields, total_dims, nodes)?;
                    }
                    serde_json::Value::Null => {}
                    scalar => {
                        let val = match scalar {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        gf.set_meta(path, key, &val).map_err(|e| e.to_string())?;
                    }
                }
            }
            if has_dims && path != "/" {
                gf.set_semantic(path, encode_dims(&dims, total_dims))
                    .map_err(|e| e.to_string())?;
            }
        }
        serde_json::Value::Array(items) => {
            if path != "/" {
                gf.put(path, b"").map_err(|e| e.to_string())?;
                *nodes += 1;
            }
            for (i, item) in items.iter().enumerate() {
                let seg = index_segment(i, items.len());
                let child_path = if path == "/" {
                    format!("/{}", seg)
                } else {
                    format!("{}/{}", path, seg)
                };
                match item {
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        emit_json(gf, &child_path, item, dim_fields, total_dims, nodes)?;
                    }
                    scalar => {
                        let val = match scalar {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        gf.put(&child_path, val.as_bytes()).map_err(|e| e.to_string())?;
                        *nodes += 1;
                    }
                }
            }
        }
        _ => return Err("root must be a JSON/YAML object or array".into()),
    }
    Ok(())
}

fn import_json(args: &[String]) -> Result<(), String> {
    let input = positional(args, 0)?;
    let output = positional(args, 1)?;
    let dim_fields: Vec<&str> = flag_value(args, "--dim-fields")
        .map(|v| v.split(',').map(str::trim).collect())
        .unwrap_or_default();
    let meaning_addressed = has_flag(args, "--meaning-addressed");
    let quantized = has_flag(args, "--quantized");
    if meaning_addressed && dim_fields.is_empty() {
        return Err("--meaning-addressed requires --dim-fields".into());
    }
    check_dims(dim_fields.len(), quantized)?;

    let doc = parse_document(input)?;
    if !doc.is_object() && !doc.is_array() {
        return Err("root must be a JSON/YAML object or array".into());
    }

    // Pass 1: bounds (and --quantized range check) over every dim value.
    let mut dim_vals = Vec::new();
    collect_json_dims(&doc, "", &dim_fields, &mut dim_vals)?;
    if quantized {
        for &v in &dim_vals {
            check_tq19_range(v, input)?;
        }
    }
    let bounds = if dim_vals.is_empty() { (FixedPoint::from_int(0), FixedPoint::from_int(1)) } else { padded_bounds(&dim_vals) };

    if Path::new(output).exists() {
        return Err(format!("refusing to overwrite existing file: {}", output));
    }
    let gf = Horon::open_with_config(
        output,
        build_config(dim_fields.len(), meaning_addressed, bounds, quantized),
    )
    .map_err(|e| e.to_string())?;

    let mut nodes = 0usize;
    emit_json(&gf, "/", &doc, &dim_fields, USER_DIM_START + dim_fields.len(), &mut nodes)?;
    gf.compact().map_err(|e| e.to_string())?;

    println!("imported {} nodes → {}", nodes, output);
    println!("  dimensions: {} ({})", dim_fields.len(), dim_fields.join(", "));
    println!("  bounds:     {:.3} .. {:.3}", bounds.0.to_f64(), bounds.1.to_f64());
    println!("  layout:     {}", layout_label(meaning_addressed, quantized));
    println!("  file size:  {} bytes", std::fs::metadata(output).map(|m| m.len()).unwrap_or(0));
    Ok(())
}

fn layout_label(meaning_addressed: bool, quantized: bool) -> &'static str {
    match (quantized, meaning_addressed) {
        (true, true) => "meaning-addressed + quantized (v4)",
        (true, false) => "quantized (v4)",
        (false, true) => "meaning-addressed (v3)",
        (false, false) => "standard (v2)",
    }
}

// ---------------------------------------------------------------------------
// import-sqlite — FK-hierarchy → paths
// ---------------------------------------------------------------------------

/// One cell, carried losslessly enough for keys/metadata/payload.
#[derive(Clone, Debug)]
enum Cell {
    Int(i64),
    Real(f64),
    Text(String),
    Blob(usize),
    Null,
}

impl Cell {
    fn from_ref(v: rusqlite::types::ValueRef<'_>) -> Self {
        use rusqlite::types::ValueRef;
        match v {
            ValueRef::Integer(i) => Cell::Int(i),
            ValueRef::Real(f) => Cell::Real(f),
            ValueRef::Text(t) => Cell::Text(String::from_utf8_lossy(t).into_owned()),
            ValueRef::Blob(b) => Cell::Blob(b.len()),
            ValueRef::Null => Cell::Null,
        }
    }

    /// Stable string form used for keys and FK joins. `None` for NULL.
    fn key_str(&self) -> Option<String> {
        match self {
            Cell::Int(i) => Some(i.to_string()),
            Cell::Real(f) => Some(format!("{}", f)),
            Cell::Text(t) => Some(t.clone()),
            Cell::Blob(n) => Some(format!("blob{}", n)),
            Cell::Null => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Cell::Int(i) => Some(*i as f64),
            Cell::Real(f) => Some(*f),
            _ => None,
        }
    }

    fn json(&self) -> serde_json::Value {
        match self {
            Cell::Int(i) => serde_json::Value::from(*i),
            Cell::Real(f) => serde_json::Value::from(*f),
            Cell::Text(t) => serde_json::Value::from(t.as_str()),
            Cell::Blob(n) => serde_json::Value::from(format!("<blob {} bytes>", n)),
            Cell::Null => serde_json::Value::Null,
        }
    }
}

/// A table's chosen parent link: FK columns here, referenced columns there.
struct ParentLink {
    to_table: String,
    from_cols: Vec<String>,
    /// Referenced columns; empty = the parent's implicit PK (its key cols).
    to_cols: Vec<String>,
}

struct TableInfo {
    name: String,
    key_cols: Vec<String>, // empty = rowid
    parent: Option<ParentLink>,
    dim_cols: Vec<String>,
}

fn sqlite_err<E: std::fmt::Display>(e: E) -> String {
    format!("sqlite: {}", e)
}

fn import_sqlite(args: &[String]) -> Result<(), String> {
    let input = positional(args, 0)?;
    let output = positional(args, 1)?;
    let meaning_addressed = has_flag(args, "--meaning-addressed");
    let quantized = has_flag(args, "--quantized");

    let parse_specs = |flag: &str| -> Result<Vec<(String, String)>, String> {
        flag_value(args, flag)
            .map(|v| {
                v.split(',')
                    .map(|s| {
                        s.trim()
                            .split_once('.')
                            .map(|(t, c)| (t.to_string(), c.to_string()))
                            .ok_or_else(|| format!("{} entries must be table.column, got '{}'", flag, s))
                    })
                    .collect()
            })
            .unwrap_or_else(|| Ok(Vec::new()))
    };
    let dim_specs = parse_specs("--dim-cols")?;
    let key_specs = parse_specs("--key-cols")?;

    let conn = rusqlite::Connection::open_with_flags(
        input,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .map_err(sqlite_err)?;

    // Included tables, sorted for determinism.
    let mut tables: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            .map_err(sqlite_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(sqlite_err)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_err)?
    };
    tables.sort();
    if let Some(filter) = flag_value(args, "--tables") {
        let want: Vec<&str> = filter.split(',').map(str::trim).collect();
        for w in &want {
            if !tables.iter().any(|t| t == w) {
                return Err(format!("--tables: '{}' not found (has: {})", w, tables.join(", ")));
            }
        }
        tables.retain(|t| want.contains(&t.as_str()));
    }
    if tables.is_empty() {
        return Err("no tables to import".into());
    }

    // Per-table structure: columns, PK, chosen parent FK, dims.
    let mut infos: Vec<TableInfo> = Vec::new();
    for t in &tables {
        let mut columns = Vec::new();
        let mut pk: Vec<(i64, String)> = Vec::new();
        {
            let mut stmt = conn
                .prepare(&format!("PRAGMA table_info(\"{}\")", t))
                .map_err(sqlite_err)?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((r.get::<_, String>(1)?, r.get::<_, i64>(5)?))
                })
                .map_err(sqlite_err)?;
            for row in rows {
                let (name, pk_ord) = row.map_err(sqlite_err)?;
                if pk_ord > 0 {
                    pk.push((pk_ord, name.clone()));
                }
                columns.push(name);
            }
        }
        pk.sort();
        let key_override: Vec<String> = key_specs
            .iter()
            .filter(|(kt, _)| kt == t)
            .map(|(_, c)| c.clone())
            .collect();
        let key_cols = if !key_override.is_empty() {
            key_override
        } else {
            pk.into_iter().map(|(_, c)| c).collect()
        };

        // First FK (lowest id) that points at an included table.
        let mut parent: Option<ParentLink> = None;
        {
            let mut stmt = conn
                .prepare(&format!("PRAGMA foreign_key_list(\"{}\")", t))
                .map_err(sqlite_err)?;
            let rows = stmt
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,                    // id
                        r.get::<_, i64>(1)?,                    // seq
                        r.get::<_, String>(2)?,                 // to table
                        r.get::<_, String>(3)?,                 // from col
                        r.get::<_, Option<String>>(4)?,         // to col (None = implicit PK)
                    ))
                })
                .map_err(sqlite_err)?;
            let mut fks: Vec<(i64, i64, String, String, Option<String>)> =
                rows.collect::<Result<Vec<_>, _>>().map_err(sqlite_err)?;
            fks.sort_by_key(|(id, seq, ..)| (*id, *seq));
            let mut by_id: Vec<(i64, ParentLink)> = Vec::new();
            for (id, _seq, to_table, from_col, to_col) in fks {
                if let Some((last_id, link)) = by_id.last_mut() {
                    if *last_id == id {
                        link.from_cols.push(from_col);
                        if let Some(c) = to_col {
                            link.to_cols.push(c);
                        }
                        continue;
                    }
                }
                by_id.push((
                    id,
                    ParentLink {
                        to_table,
                        from_cols: vec![from_col],
                        to_cols: to_col.into_iter().collect(),
                    },
                ));
            }
            // PRAGMA foreign_key_list numbers constraints in REVERSE
            // declaration order (id 0 = last declared — verified against
            // bundled SQLite). Users read "the first FK wins" as declaration
            // order, so iterate highest id first.
            for (_, link) in by_id.into_iter().rev() {
                if tables.iter().any(|x| *x == link.to_table) {
                    parent = Some(link);
                    break;
                }
            }
        }

        let dim_cols: Vec<String> = dim_specs
            .iter()
            .filter(|(dt, _)| dt == t)
            .map(|(_, c)| c.clone())
            .collect();
        for c in &dim_cols {
            if !columns.contains(c) {
                return Err(format!("--dim-cols: {}.{} not found", t, c));
            }
        }
        infos.push(TableInfo { name: t.clone(), key_cols, parent, dim_cols });
    }

    // Topological order over parent links (self-references stay in place;
    // rows are ordered within the table instead).
    let order: Vec<usize> = {
        let mut placed = vec![false; infos.len()];
        let mut order = Vec::new();
        loop {
            let mut progressed = false;
            for i in 0..infos.len() {
                if placed[i] {
                    continue;
                }
                let ready = match &infos[i].parent {
                    None => true,
                    Some(p) if p.to_table == infos[i].name => true, // self-FK
                    Some(p) => infos
                        .iter()
                        .position(|x| x.name == p.to_table)
                        .map(|j| placed[j])
                        .unwrap_or(true),
                };
                if ready {
                    placed[i] = true;
                    order.push(i);
                    progressed = true;
                }
            }
            if order.len() == infos.len() {
                break;
            }
            if !progressed {
                let cyclic: Vec<&str> = infos
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !placed[*i])
                    .map(|(_, t)| t.name.as_str())
                    .collect();
                return Err(format!(
                    "foreign-key cycle between tables: {} (import them separately with --tables)",
                    cyclic.join(", ")
                ));
            }
        }
        order
    };

    let user_dims: usize = infos.iter().map(|t| t.dim_cols.len()).max().unwrap_or(0);
    check_dims(user_dims, quantized)?;
    if meaning_addressed && user_dims == 0 {
        return Err("--meaning-addressed requires --dim-cols".into());
    }

    // Read every table's rows once (buffered — this importer is not
    // streaming; noted in the usage text).
    struct Row {
        cells: Vec<(String, Cell)>,
        key: String,
        parent_lookup: Option<(String, String, String)>, // (table, cols-sig, values)
    }
    let mut all_rows: Vec<Vec<Row>> = Vec::new();
    let mut dim_vals: Vec<FixedPoint> = Vec::new();
    for &i in &order {
        let info = &infos[i];
        let use_rowid = info.key_cols.is_empty();
        let sql = if use_rowid {
            format!("SELECT rowid AS __htt_rowid, * FROM \"{}\"", info.name)
        } else {
            format!("SELECT * FROM \"{}\"", info.name)
        };
        let mut stmt = conn.prepare(&sql).map_err(|e| {
            format!("{} (table '{}' may be WITHOUT ROWID and need --key-cols)", sqlite_err(e), info.name)
        })?;
        let col_names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
        let mut rows_out: Vec<Row> = Vec::new();
        let mut rows = stmt.query([]).map_err(sqlite_err)?;
        while let Some(row) = rows.next().map_err(sqlite_err)? {
            let mut cells = Vec::with_capacity(col_names.len());
            for (c, name) in col_names.iter().enumerate() {
                cells.push((name.clone(), Cell::from_ref(row.get_ref(c).map_err(sqlite_err)?)));
            }
            let get = |name: &str| cells.iter().find(|(n, _)| n == name).map(|(_, c)| c);

            let key = if use_rowid {
                get("__htt_rowid").and_then(Cell::key_str).unwrap_or_default()
            } else {
                let parts: Vec<String> = info
                    .key_cols
                    .iter()
                    .map(|c| get(c).and_then(Cell::key_str).unwrap_or_else(|| "null".into()))
                    .collect();
                parts.join("-")
            };

            let parent_lookup = info.parent.as_ref().and_then(|p| {
                let vals: Option<Vec<String>> = p
                    .from_cols
                    .iter()
                    .map(|c| get(c).and_then(Cell::key_str))
                    .collect();
                vals.map(|v| {
                    let sig = if p.to_cols.is_empty() { "__pk".to_string() } else { p.to_cols.join(",") };
                    (p.to_table.clone(), sig, v.join("\x1f"))
                })
            });

            for c in &info.dim_cols {
                match get(c).and_then(Cell::as_f64) {
                    Some(v) if v.is_finite() => {
                        let v = FixedPoint::from_f64(v);
                        if quantized {
                            check_tq19_range(v, &format!("{}.{}", info.name, c))?;
                        }
                        dim_vals.push(v);
                    }
                    Some(_) => return Err(format!("{}.{}: non-finite dim value", info.name, c)),
                    None => {} // NULL/non-numeric dims default to 0.0 at insert
                }
            }
            rows_out.push(Row { cells, key, parent_lookup });
        }
        rows_out.sort_by(|a, b| a.key.cmp(&b.key));
        all_rows.push(rows_out);
    }

    let bounds = if dim_vals.is_empty() { (FixedPoint::from_int(0), FixedPoint::from_int(1)) } else { padded_bounds(&dim_vals) };
    if Path::new(output).exists() {
        return Err(format!("refusing to overwrite existing file: {}", output));
    }
    let gf = Horon::open_with_config(
        output,
        build_config(user_dims, meaning_addressed, bounds, quantized),
    )
    .map_err(|e| e.to_string())?;

    // Address book: (table, cols-sig, values) → node path, registered for
    // every column tuple some child references.
    let mut lookup_specs: Vec<(String, Vec<String>)> = Vec::new(); // (table, cols; empty=pk)
    for info in &infos {
        if let Some(p) = &info.parent {
            lookup_specs.push((p.to_table.clone(), p.to_cols.clone()));
        }
    }
    let mut address: std::collections::HashMap<(String, String, String), String> =
        std::collections::HashMap::new();
    let mut orphans = 0usize;
    let mut imported = 0usize;

    let insert_row = |info: &TableInfo,
                          row: &Row,
                          parent_path: Option<&str>,
                          address: &mut std::collections::HashMap<(String, String, String), String>|
     -> Result<String, String> {
        let path = match parent_path {
            Some(pp) => format!("{}/{}/{}", pp, sanitize(&info.name), sanitize(&row.key)),
            None => format!("/{}/{}", sanitize(&info.name), sanitize(&row.key)),
        };
        // Payload: the row as JSON with sorted keys (insertion is sorted, so
        // the map backing cannot affect the bytes).
        let mut sorted: Vec<&(String, Cell)> = row.cells.iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let mut obj = serde_json::Map::new();
        for (name, cell) in &sorted {
            if name != "__htt_rowid" {
                obj.insert(name.clone(), cell.json());
            }
        }
        gf.put(&path, serde_json::Value::Object(obj).to_string().as_bytes())
            .map_err(|e| e.to_string())?;

        let get = |name: &str| row.cells.iter().find(|(n, _)| n == name).map(|(_, c)| c);
        let fk_cols: Vec<&String> = info
            .parent
            .as_ref()
            .map(|p| p.from_cols.iter().collect())
            .unwrap_or_default();
        for (name, cell) in &row.cells {
            if name == "__htt_rowid"
                || info.key_cols.contains(name)
                || info.dim_cols.contains(name)
                || fk_cols.iter().any(|c| *c == name)
            {
                continue;
            }
            match cell {
                Cell::Null | Cell::Blob(_) => {}
                other => {
                    if let Some(v) = other.key_str() {
                        let _ = gf.set_meta(&path, name, &v);
                    }
                }
            }
        }
        if !info.dim_cols.is_empty() {
            let vals: Vec<FixedPoint> = info
                .dim_cols
                .iter()
                .map(|c| FixedPoint::from_f64(get(c).and_then(Cell::as_f64).unwrap_or(0.0)))
                .collect();
            gf.set_semantic(&path, encode_dims(&vals, USER_DIM_START + user_dims))
                .map_err(|e| e.to_string())?;
        }
        // Register this row under every address spec children may use.
        for (t, cols) in &lookup_specs {
            if *t != info.name {
                continue;
            }
            let (sig, vals): (String, Option<Vec<String>>) = if cols.is_empty() {
                (
                    "__pk".to_string(),
                    info.key_cols
                        .iter()
                        .map(|c| get(c).and_then(Cell::key_str))
                        .collect(),
                )
            } else {
                (
                    cols.join(","),
                    cols.iter().map(|c| get(c).and_then(Cell::key_str)).collect(),
                )
            };
            if let Some(vals) = vals {
                address.insert((info.name.clone(), sig, vals.join("\x1f")), path.clone());
            }
        }
        Ok(path)
    };

    for (oi, &i) in order.iter().enumerate() {
        let info = &infos[i];
        let rows = &all_rows[oi];
        let self_ref = info
            .parent
            .as_ref()
            .map(|p| p.to_table == info.name)
            .unwrap_or(false);
        if !self_ref {
            for row in rows {
                let parent_path = match &row.parent_lookup {
                    Some((t, sig, vals)) => {
                        match address.get(&(t.clone(), sig.clone(), vals.clone())) {
                            Some(p) => Some(p.clone()),
                            None => {
                                orphans += 1; // dangling FK → import as a root
                                None
                            }
                        }
                    }
                    None => None,
                };
                insert_row(info, row, parent_path.as_deref(), &mut address)?;
                imported += 1;
            }
        } else {
            // Self-referencing table: multi-pass until every row's parent
            // exists; leftovers (row-level cycles) become roots.
            let mut pending: Vec<&Row> = rows.iter().collect();
            loop {
                let mut next: Vec<&Row> = Vec::new();
                let mut progressed = false;
                for row in pending {
                    let parent_path = match &row.parent_lookup {
                        None => None,
                        Some((t, sig, vals)) => {
                            match address.get(&(t.clone(), sig.clone(), vals.clone())) {
                                Some(p) => Some(p.clone()),
                                None => {
                                    next.push(row);
                                    continue;
                                }
                            }
                        }
                    };
                    insert_row(info, row, parent_path.as_deref(), &mut address)?;
                    imported += 1;
                    progressed = true;
                }
                if next.is_empty() {
                    break;
                }
                if !progressed {
                    orphans += next.len();
                    for row in next {
                        insert_row(info, row, None, &mut address)?;
                        imported += 1;
                    }
                    break;
                }
                pending = next;
            }
        }
    }
    gf.compact().map_err(|e| e.to_string())?;

    println!("imported {} rows from {} tables → {}", imported, order.len(), output);
    for &i in &order {
        let info = &infos[i];
        let parent = info
            .parent
            .as_ref()
            .map(|p| format!(" (nests under {})", p.to_table))
            .unwrap_or_default();
        println!("  table {}{}", info.name, parent);
    }
    println!("  dimensions: {} (per-table --dim-cols; unset dims are 0)", user_dims);
    println!("  bounds:     {:.3} .. {:.3}", bounds.0.to_f64(), bounds.1.to_f64());
    println!("  layout:     {}", layout_label(meaning_addressed, quantized));
    if orphans > 0 {
        println!("  ⚠ {} row(s) had a dangling/cyclic FK and were imported as roots", orphans);
    }
    println!("  file size:  {} bytes", std::fs::metadata(output).map(|m| m.len()).unwrap_or(0));
    Ok(())
}

// ---------------------------------------------------------------------------
// import-vec — vector-index exports (JSONL embeddings) → semantic dims
// ---------------------------------------------------------------------------

fn import_vec(args: &[String]) -> Result<(), String> {
    let input = positional(args, 0)?;
    let output = positional(args, 1)?;
    let id_field = flag_value(args, "--id-field").unwrap_or("id");
    let vector_field = flag_value(args, "--vector-field").unwrap_or("vector");
    let take_dims: Option<usize> = flag_value(args, "--take-dims")
        .map(|v| v.parse().map_err(|_| "--take-dims must be a number"))
        .transpose()?;
    let meaning_addressed = has_flag(args, "--meaning-addressed");
    let quantized = has_flag(args, "--quantized");

    let read_lines = || -> Result<Vec<(usize, serde_json::Value)>, String> {
        let text = std::fs::read_to_string(input).map_err(|e| format!("{}: {}", input, e))?;
        let mut out = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("line {}: {}", n + 1, e))?;
            if !v.is_object() {
                return Err(format!("line {}: expected a JSON object", n + 1));
            }
            out.push((n + 1, v));
        }
        Ok(out)
    };

    let vector_of = |n: usize, obj: &serde_json::Value| -> Result<Vec<FixedPoint>, String> {
        let arr = obj
            .get(vector_field)
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("line {}: no '{}' array", n, vector_field))?;
        let mut vals = Vec::with_capacity(arr.len());
        for v in arr {
            let f = v
                .as_f64()
                .filter(|f| f.is_finite())
                .ok_or_else(|| format!("line {}: non-numeric vector component", n))?;
            vals.push(FixedPoint::from_f64(f));
        }
        if let Some(t) = take_dims {
            vals.truncate(t);
        }
        if vals.is_empty() {
            return Err(format!("line {}: empty vector", n));
        }
        Ok(vals)
    };

    // Pass 1: dimension count consistency, bounds, --quantized range.
    let records = read_lines()?;
    if records.is_empty() {
        return Err("no records found".into());
    }
    let mut dims: Option<usize> = None;
    let mut all_vals = Vec::new();
    for (n, obj) in &records {
        let vals = vector_of(*n, obj)?;
        match dims {
            None => dims = Some(vals.len()),
            Some(d) if d != vals.len() => {
                return Err(format!(
                    "line {}: vector has {} dims, previous lines had {} \
                     (use --take-dims to normalize)",
                    n, vals.len(), d
                ));
            }
            _ => {}
        }
        if quantized {
            for &v in &vals {
                check_tq19_range(v, &format!("line {}", n))?;
            }
        }
        all_vals.extend(vals);
    }
    let d = dims.unwrap();
    check_dims(d, quantized)?;
    let bounds = padded_bounds(&all_vals);

    if Path::new(output).exists() {
        return Err(format!("refusing to overwrite existing file: {}", output));
    }
    let gf = Horon::open_with_config(
        output,
        build_config(d, meaning_addressed, bounds, quantized),
    )
    .map_err(|e| e.to_string())?;

    // Pass 2: insert. Payload = the original JSON line (lossless); scalar
    // fields become metadata; one-level-nested object fields flatten to
    // "field.sub" metadata.
    let mut seen: HashSet<String> = HashSet::new();
    let mut collisions = 0usize;
    for (n, obj) in &records {
        let key = match obj.get(id_field) {
            Some(serde_json::Value::String(s)) => format!("/{}", sanitize(s)),
            Some(serde_json::Value::Number(num)) => format!("/{}", sanitize(&num.to_string())),
            _ => format!("/line_{:06}", n),
        };
        if !seen.insert(key.clone()) {
            collisions += 1;
        }
        gf.put(&key, obj.to_string().as_bytes()).map_err(|e| e.to_string())?;

        if let Some(map) = obj.as_object() {
            for field in sorted_keys(map) {
                if field == vector_field || field == id_field {
                    continue;
                }
                match &map[field] {
                    serde_json::Value::Object(sub) => {
                        for sk in sorted_keys(sub) {
                            let sv = &sub[sk];
                            if !sv.is_object() && !sv.is_array() && !sv.is_null() {
                                let val = match sv {
                                    serde_json::Value::String(s) => s.clone(),
                                    other => other.to_string(),
                                };
                                let _ = gf.set_meta(&key, &format!("{}.{}", field, sk), &val);
                            }
                        }
                    }
                    serde_json::Value::Array(_) | serde_json::Value::Null => {}
                    scalar => {
                        let val = match scalar {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        let _ = gf.set_meta(&key, field, &val);
                    }
                }
            }
        }
        let vals = vector_of(*n, obj)?;
        gf.set_semantic(&key, encode_dims(&vals, USER_DIM_START + d))
            .map_err(|e| e.to_string())?;
    }
    gf.compact().map_err(|e| e.to_string())?;

    println!("imported {} vectors → {}", records.len(), output);
    println!("  dimensions: {}{}", d,
        take_dims.map(|t| format!(" (--take-dims {} applied)", t)).unwrap_or_default());
    println!("  bounds:     {:.3} .. {:.3}", bounds.0.to_f64(), bounds.1.to_f64());
    println!("  layout:     {}", layout_label(meaning_addressed, quantized));
    if collisions > 0 {
        println!("  ⚠ {} record(s) collided onto an existing key and overwrote it", collisions);
    }
    println!("  file size:  {} bytes", std::fs::metadata(output).map(|m| m.len()).unwrap_or(0));
    if !quantized && d >= 8 {
        println!("  hint: embedding corpora usually fit TQ1.9 — rerun with --quantized for 8× smaller dims");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// inspect / query
// ---------------------------------------------------------------------------

fn read_header(path: &str) -> Result<GeoHeader, String> {
    // Read only the 32-byte header prefix — never the whole file, which may be
    // gigabytes when all we need is the header for `inspect`/`query`.
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 32];
    file.read_exact(&mut buf)
        .map_err(|_| "file too small to be a .htt".to_string())?;
    GeoHeader::from_bytes(&buf).map_err(|e| e.to_string())
}

fn open_fast(path: &str) -> Result<Horon, String> {
    // Lazy geometry: instant open, all key/semantic queries available.
    Horon::open_with_config(path, HoronConfig {
        lazy_geometry: true,
        ..Default::default()
    })
    .map_err(|e| e.to_string())
}

fn inspect(args: &[String]) -> Result<(), String> {
    let path = positional(args, 0)?;
    let header = read_header(path)?;
    let gf = open_fast(path)?;

    println!("{}", path);
    println!("  format:      v{}{}{}", header.version,
        if header.flags & 0x40 != 0 { " (meaning-addressed)" } else { "" },
        if header.quantized_semantic() { " (quantized semantic)" } else { "" });
    println!("  dimensions:  {} structural + {} semantic ({} user)",
        header.dimension, header.semantic_dims,
        (header.semantic_dims as usize).saturating_sub(USER_DIM_START));
    println!("  compression: {}", if header.compression_enabled() { "zstd" } else { "none" });
    println!("  gacl:        {}", header.gacl_enabled());
    println!("  nodes:       {} (live: {})", header.node_count, gf.len());
    println!("  wal entries: {}", gf.wal_len());
    println!("  file size:   {} bytes", std::fs::metadata(path).map(|m| m.len()).unwrap_or(0));

    let keys = gf.list("/").map_err(|e| e.to_string())?;
    println!("  sample keys:");
    for key in keys.iter().take(10) {
        println!("    {}", key);
    }
    if keys.len() > 10 {
        println!("    … and {} more", keys.len() - 10);
    }
    Ok(())
}

fn query(args: &[String]) -> Result<(), String> {
    let path = positional(args, 0)?;
    let key = positional(args, 1)?;
    let k: usize = positional(args, 2).ok().and_then(|s| s.parse().ok()).unwrap_or(5);

    let header = read_header(path)?;
    let user_dims = (header.semantic_dims as usize).saturating_sub(USER_DIM_START);
    if user_dims == 0 {
        return Err("file has no user semantic dimensions to query by".into());
    }
    let gf = open_fast(path)?;
    if !gf.exists(key) {
        return Err(format!("key not found: {}", key));
    }

    let range = USER_DIM_START..USER_DIM_START + user_dims;
    let results = gf.neighbors_semantic(key, k, range).map_err(|e| e.to_string())?;

    println!("most similar to {} :", key);
    for (nkey, dist) in results {
        let meta = gf.get_meta(&nkey).unwrap_or_default();
        let extras: Vec<String> = meta
            .iter()
            .filter(|(mk, _)| !matches!(mk.as_str(), "key" | "size" | "created_at" | "updated_at" | "_child_index"))
            .take(3)
            .map(|(mk, mv)| format!("{}={}", mk, mv))
            .collect();
        println!("  {:>10.4}  {}  {}", dist, nkey, extras.join("  "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn tmp(ext: &str) -> String {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir()
            .join(format!("htt_test_{}_{}.{}", std::process::id(), n, ext))
            .to_string_lossy()
            .into_owned()
    }

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|x| x.to_string()).collect()
    }


    // ---- import-json ----

    #[test]
    fn import_json_nested_objects_become_paths() {
        let json = tmp("json");
        let out = tmp("htt");
        std::fs::write(
            &json,
            r#"{"courses":{"cbt":{"x":0.8,"name":"CBT Basics","sub":{"y":2}},
                "act":{"x":0.2,"name":"ACT Intro"}},
               "tags":["alpha","beta"]}"#.replace('\n', " "),
        )
        .unwrap();
        let args = s(&[&json, &out, "--dim-fields", "x"]);
        import_json(&args).unwrap();

        let gf = open_fast(&out).unwrap();
        // Objects → paths; scalars → metadata; dim fields consumed as dims.
        assert!(gf.exists("/courses/cbt"));
        assert!(gf.exists("/courses/cbt/sub"));
        let meta = gf.get_meta("/courses/cbt").unwrap();
        assert_eq!(meta.get("name").map(String::as_str), Some("CBT Basics"));
        assert!(meta.get("x").is_none(), "dim field must not double as metadata");
        // Array scalars → leaf nodes with the scalar as payload.
        assert_eq!(gf.get("/tags/0").unwrap(), b"alpha");
        assert_eq!(gf.get("/tags/1").unwrap(), b"beta");
        // Dims are queryable: cbt (0.8) is nearer to 0.7 than act (0.2).
        let q = encode_dims(&[FixedPoint::from_f64(0.7)], USER_DIM_START + 1);
        let near = gf.nearest_semantic(&q, 1, USER_DIM_START..USER_DIM_START + 1).unwrap();
        assert_eq!(near[0].0, "/courses/cbt");
        let _ = std::fs::remove_file(&json);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn import_json_reads_yaml_and_quantizes() {
        let yaml = tmp("yaml");
        let out = tmp("htt");
        std::fs::write(&yaml, "a:\n  x: 0.5\n  note: hi\nb:\n  x: 0.25\n").unwrap();
        let args = s(&[&yaml, &out, "--dim-fields", "x", "--quantized"]);
        import_json(&args).unwrap();

        let header = read_header(&out).unwrap();
        assert_eq!(header.version, 4);
        assert!(header.quantized_semantic());
        let gf = open_fast(&out).unwrap();
        // Stored dim is the on-grid TQ1.9 value (0.5 → 9842/19683).
        let sem = gf.get_semantic("/a").unwrap();
        let raw = i128::from_le_bytes(
            sem[USER_DIM_START * 16..USER_DIM_START * 16 + 16].try_into().unwrap(),
        );
        assert_eq!(raw, horon::quant::dequantize_raw(9842));
        assert_eq!(gf.get_meta("/a").unwrap().get("note").map(String::as_str), Some("hi"));
        let _ = std::fs::remove_file(&yaml);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn import_json_quantized_rejects_out_of_range() {
        let json = tmp("json");
        let out = tmp("htt");
        std::fs::write(&json, r#"{"a":{"x":2.5}}"#).unwrap();
        let args = s(&[&json, &out, "--dim-fields", "x", "--quantized"]);
        let err = import_json(&args).unwrap_err();
        assert!(err.contains("TQ1.9"), "got: {}", err);
        assert!(!Path::new(&out).exists(), "must fail before creating the file");
        let _ = std::fs::remove_file(&json);
    }

    // ---- import-sqlite ----

    fn make_db(path: &str) {
        let conn = rusqlite::Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE dept(id INTEGER PRIMARY KEY, name TEXT);
             CREATE TABLE emp(id INTEGER PRIMARY KEY, dept_id INTEGER REFERENCES dept(id),
                              mgr_id INTEGER REFERENCES emp(id), name TEXT, score REAL);
             INSERT INTO dept VALUES (1,'eng'),(2,'ops');
             INSERT INTO emp VALUES (10,1,NULL,'ada',0.9);
             INSERT INTO emp VALUES (11,1,10,'bob',0.4);
             INSERT INTO emp VALUES (12,2,NULL,'cyd',0.7);",
        )
        .unwrap();
    }

    #[test]
    fn import_sqlite_fk_hierarchy() {
        let db = tmp("sqlite");
        let out = tmp("htt");
        make_db(&db);
        let args = s(&[&db, &out, "--dim-cols", "emp.score"]);
        import_sqlite(&args).unwrap();

        let gf = open_fast(&out).unwrap();
        // emp declares dept_id before mgr_id; the FIRST-DECLARED FK wins the
        // nesting (pragma ids are reversed — handled in import_sqlite), so
        // every emp row nests under its dept row.
        assert!(gf.exists("/dept/1/emp/10"), "keys: {:?}", gf.list("/").unwrap());
        assert!(gf.exists("/dept/1/emp/11"));
        assert!(gf.exists("/dept/2/emp/12"));
        let ada = &"/dept/1/emp/10".to_string();
        // Metadata and payload survive.
        let meta = gf.get_meta(ada).unwrap();
        assert_eq!(meta.get("name").map(String::as_str), Some("ada"));
        let payload: serde_json::Value =
            serde_json::from_slice(&gf.get(ada).unwrap()).unwrap();
        assert_eq!(payload["score"], serde_json::json!(0.9));
        // Dims queryable: nearest to 0.85 over emp.score is ada (0.9).
        let q = encode_dims(&[FixedPoint::from_f64(0.85)], USER_DIM_START + 1);
        let near = gf.nearest_semantic(&q, 1, USER_DIM_START..USER_DIM_START + 1).unwrap();
        assert_eq!(near[0].0, *ada);
        let _ = std::fs::remove_file(&db);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn import_sqlite_self_fk_nests_rows() {
        let db = tmp("sqlite");
        let out = tmp("htt");
        let conn = rusqlite::Connection::open(&db).unwrap();
        conn.execute_batch(
            "CREATE TABLE node(id INTEGER PRIMARY KEY, parent INTEGER REFERENCES node(id), tag TEXT);
             INSERT INTO node VALUES (1,NULL,'root'),(2,1,'child'),(3,2,'grandchild');",
        )
        .unwrap();
        drop(conn);
        let args = s(&[&db, &out]);
        import_sqlite(&args).unwrap();

        let gf = open_fast(&out).unwrap();
        assert!(gf.exists("/node/1"));
        assert!(gf.exists("/node/1/node/2"));
        assert!(gf.exists("/node/1/node/2/node/3"));
        assert_eq!(
            gf.get_meta("/node/1/node/2").unwrap().get("tag").map(String::as_str),
            Some("child")
        );
        let _ = std::fs::remove_file(&db);
        let _ = std::fs::remove_file(&out);
    }

    // ---- import-vec ----

    #[test]
    fn import_vec_quantized_grid_and_ranking() {
        let jsonl = tmp("jsonl");
        let out = tmp("htt");
        std::fs::write(
            &jsonl,
            concat!(
                "{\"id\":\"a\",\"vector\":[0.1,0.9],\"payload\":{\"tag\":\"x\"}}\n",
                "{\"id\":\"b\",\"vector\":[0.8,0.2],\"kind\":\"doc\"}\n",
                "{\"id\":\"c\",\"vector\":[0.15,0.85]}\n",
            ),
        )
        .unwrap();
        let args = s(&[&jsonl, &out, "--quantized"]);
        import_vec(&args).unwrap();

        let header = read_header(&out).unwrap();
        assert_eq!(header.version, 4);
        assert_eq!(header.semantic_dims as usize, USER_DIM_START + 2);
        let gf = open_fast(&out).unwrap();
        // Nearest to a's neighborhood is c, not b.
        let near = gf.neighbors_semantic("/a", 1, USER_DIM_START..USER_DIM_START + 2).unwrap();
        assert_eq!(near[0].0, "/c");
        // Flattened nested metadata + plain scalar metadata.
        assert_eq!(gf.get_meta("/a").unwrap().get("payload.tag").map(String::as_str), Some("x"));
        assert_eq!(gf.get_meta("/b").unwrap().get("kind").map(String::as_str), Some("doc"));
        // Payload is the original JSON line.
        let payload: serde_json::Value = serde_json::from_slice(&gf.get("/b").unwrap()).unwrap();
        assert_eq!(payload["vector"][0], serde_json::json!(0.8));
        // Stored dims are on the TQ1.9 grid.
        let sem = gf.get_semantic("/a").unwrap();
        let raw = i128::from_le_bytes(
            sem[USER_DIM_START * 16..USER_DIM_START * 16 + 16].try_into().unwrap(),
        );
        assert_eq!(raw, horon::quant::dequantize_raw(horon::quant::quantize_raw(raw).unwrap()));
        let _ = std::fs::remove_file(&jsonl);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn import_vec_dimension_mismatch_and_take_dims() {
        let jsonl = tmp("jsonl");
        let out = tmp("htt");
        std::fs::write(
            &jsonl,
            "{\"id\":\"a\",\"vector\":[0.1,0.2,0.3]}\n{\"id\":\"b\",\"vector\":[0.4,0.5]}\n",
        )
        .unwrap();
        let err = import_vec(&s(&[&jsonl, &out])).unwrap_err();
        assert!(err.contains("dims"), "got: {}", err);
        // --take-dims 2 normalizes the widths.
        import_vec(&s(&[&jsonl, &out, "--take-dims", "2"])).unwrap();
        let header = read_header(&out).unwrap();
        assert_eq!(header.semantic_dims as usize, USER_DIM_START + 2);
        let _ = std::fs::remove_file(&jsonl);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn positional_skips_value_flag_values() {
        // A value-flag's value must not be mistaken for a positional, even when
        // the flag appears before the positionals.
        let args = s(&["--path-cols", "A,B", "in.csv", "out.htt", "--meaning-addressed"]);
        assert_eq!(positional(&args, 0).unwrap(), "in.csv");
        assert_eq!(positional(&args, 1).unwrap(), "out.htt");
        assert!(positional(&args, 2).is_err());

        // Boolean flag does not swallow the following positional.
        let args = s(&["--meaning-addressed", "in.csv", "out.htt"]);
        assert_eq!(positional(&args, 0).unwrap(), "in.csv");
        assert_eq!(positional(&args, 1).unwrap(), "out.htt");
    }

    #[test]
    fn pad_bounds_degenerate_is_unit_interval() {
        let fp = |v: i32| FixedPoint::from_int(v);
        assert_eq!(pad_bounds(fp(5), fp(5)), (fp(4), fp(6)));
        assert_eq!(pad_bounds(fp(3), fp(1)), (fp(2), fp(4))); // hi <= lo
        let (lo, hi) = pad_bounds(fp(0), fp(10));
        assert!(lo < fp(0) && hi > fp(10));
    }

    #[test]
    fn padded_bounds_handles_degenerate_input() {
        // NaN and infinity are unrepresentable in Q64.64, so they are rejected
        // at parse time (see parse_dim) rather than filtered here — the
        // remaining degenerate cases are an empty set and a zero-width span.
        let fp = |v: f64| FixedPoint::from_f64(v);
        assert_eq!(
            padded_bounds(&[fp(1.0), fp(2.0)]),
            pad_bounds(fp(1.0), fp(2.0))
        );
        assert_eq!(
            padded_bounds(&[]),
            (FixedPoint::from_int(0), FixedPoint::from_int(1))
        );
        // Zero-width span becomes a unit interval around the value.
        assert_eq!(
            padded_bounds(&[fp(5.0), fp(5.0)]),
            (fp(4.0), fp(6.0))
        );
    }

    #[test]
    fn parse_dim_reads_decimals_exactly() {
        // The point of parse_dim: "0.1" becomes the Q64.64 nearest to decimal
        // 0.1, not the Q64.64 nearest to the f64 nearest to 0.1.
        assert_eq!(parse_dim("0.1").unwrap(), FixedPoint::from_str("0.1"));
        assert_eq!(parse_dim("nan"), Err(DimParseError::NotFinite));
        assert_eq!(parse_dim("inf"), Err(DimParseError::NotFinite));
        assert_eq!(parse_dim("abc"), Err(DimParseError::NotNumeric));
    }

    #[test]
    fn import_csv_rejects_non_finite_dim_value() {
        let csv = tmp("csv");
        let out = tmp("htt");
        std::fs::write(&csv, "name,x\na,1.0\nb,nan\n").unwrap();
        let args = s(&[&csv, &out, "--path-cols", "name", "--dim-cols", "x"]);
        let err = import_csv(&args).unwrap_err();
        assert!(err.contains("finite"), "expected a finiteness error, got: {}", err);
        let _ = std::fs::remove_file(&csv);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn import_csv_streams_and_deduplicates_keys() {
        let csv = tmp("csv");
        let out = tmp("htt");
        // Two rows share the same key-col value → one key, last write wins.
        std::fs::write(&csv, "id,val\nk1,first\nk1,second\nk2,other\n").unwrap();
        let args = s(&[&csv, &out, "--path-cols", "id", "--key-col", "id"]);
        import_csv(&args).unwrap();

        let gf = open_fast(&out).unwrap();
        // k1 appears once (deduplicated); k2 once.
        assert!(gf.exists("/k1/k1"));
        assert!(gf.exists("/k2/k2"));
        let _ = std::fs::remove_file(&csv);
        let _ = std::fs::remove_file(&out);
    }
}
