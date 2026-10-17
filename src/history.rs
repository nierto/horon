//! Temporal history: sidecar segments + the `HoronHistory` reader (temporal epochs).
//!
//! Design: `docs/TEMPORAL_EPOCHS.md`. The WAL already records the manifold's
//! movement; with `HistoryRetention::Archive`, compaction archives the
//! pre-fence WAL into zstd-compressed sidecar segments instead of discarding
//! it. This module owns the segment format and the read-only temporal
//! queries over the full retained log:
//!
//! - [`HoronHistory::epochs`] — every sealed epoch (id, seq, speculative)
//! - [`HoronHistory::as_of`] — full node state at an epoch seal
//! - [`HoronHistory::trajectory`] — one key's semantic coordinates across epochs
//! - [`HoronHistory::delta`] — who moved between two epochs, and how far
//!
//! # Segment format (`<file>.h000001`, `<file>.h000002`, …)
//!
//! ```text
//! magic "HTTH" (4) | version (1) | semantic_dims (1) | layout_flags (1) | reserved (1)
//! first_seq (4 LE) | end_seq (4 LE, exclusive) | raw_len (4) | comp_len (4)
//! zstd-compressed entry bytes (comp_len)
//! crc32 of the RAW (uncompressed) entry bytes (4)
//! ```
//!
//! `layout_flags` (v2 segments — quantized files): bit 0 = quantized
//! tails, bit 1 = GACL reserved region present. v1 segments (plain files)
//! keep the byte zero.
//!
//! Entries are stored plain-serialized (never WAL-block-compressed),
//! regardless of the main file's WAL compression — one reader path, and the
//! segment is self-describing given its own `semantic_dims` byte.
//! Overlapping seq spans between segments (possible after a crash between
//! archive and truncate) are harmless: the reader deduplicates by sequence
//! number, and duplicate entries are byte-identical.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};

use crate::error::{HoronError, HoronResult};
use crate::format::{
    EPOCH_FLAG_RESTAMP, EPOCH_FLAG_SPECULATIVE, FLAG_MEANING_ADDRESSED, HEADER_SIZE,
    HIST_FLAG_GACL, HIST_FLAG_QUANTIZED, HIST_HEADER_SIZE, HIST_MAGIC, HIST_VERSION,
    HIST_VERSION_QUANTIZED, MAX_HIST_BYTES, DIM_USER_DEFINED_START,
};
use crate::header::GeoHeader;
use crate::quant::SemLayout;
use crate::wal::{self, WalEntry, WalPayload};

// ---------------------------------------------------------------------------
// Segment paths + write (used by compaction)
// ---------------------------------------------------------------------------

/// Sidecar path for segment `n`: `<file>.h000042`.
pub(crate) fn segment_path(base: &Path, n: u32) -> PathBuf {
    PathBuf::from(format!("{}.h{:06}", base.display(), n))
}

/// All existing sidecar segments for `base`, sorted by segment number.
pub(crate) fn list_segments(base: &Path) -> Vec<(u32, PathBuf)> {
    let Some(dir) = base.parent() else { return Vec::new() };
    let Some(stem) = base.file_name().and_then(|s| s.to_str()) else { return Vec::new() };
    let prefix = format!("{}.h", stem);
    let mut out = Vec::new();
    let Ok(read) = fs::read_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir })
    else {
        return Vec::new();
    };
    for entry in read.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if let Some(suffix) = name.strip_prefix(&prefix) {
            if suffix.len() == 6 && suffix.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(n) = suffix.parse::<u32>() {
                    out.push((n, entry.path()));
                }
            }
        }
    }
    out.sort_by_key(|(n, _)| *n);
    out
}

/// Next free segment number for `base` (1-based).
pub(crate) fn next_segment_number(base: &Path) -> u32 {
    list_segments(base).last().map_or(1, |(n, _)| n + 1)
}

/// Write a history segment crash-safely (tmp + fsync + rename + dir fsync).
/// Entries are plain-serialized and zstd-compressed as a whole.
pub(crate) fn write_segment(
    base: &Path,
    n: u32,
    layout: SemLayout,
    first_seq: u32,
    end_seq: u32,
    entries: &[WalEntry],
) -> HoronResult<PathBuf> {
    let final_path = segment_path(base, n);
    let tmp_path = PathBuf::from(format!("{}.tmp", final_path.display()));

    let mut raw = Vec::new();
    for e in entries {
        e.write_to(&mut raw, &layout)?;
    }
    let compressed = zstd::bulk::compress(&raw, 3)
        .map_err(|e| HoronError::CompressionError(e.to_string()))?;

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    file.write_all(&HIST_MAGIC)?;
    // v2 segments (quantized files) record the tail layout in the byte at
    // offset 6, so each segment stays self-describing.
    let (version, layout_flags) = if layout.quantized {
        let mut f = HIST_FLAG_QUANTIZED;
        if layout.gacl {
            f |= HIST_FLAG_GACL;
        }
        (HIST_VERSION_QUANTIZED, f)
    } else {
        (HIST_VERSION, 0)
    };
    file.write_all(&[version, layout.dims as u8, layout_flags, 0])?;
    file.write_all(&first_seq.to_le_bytes())?;
    file.write_all(&end_seq.to_le_bytes())?;
    file.write_all(&(raw.len() as u32).to_le_bytes())?;
    file.write_all(&(compressed.len() as u32).to_le_bytes())?;
    file.write_all(&compressed)?;
    file.write_all(&crc32fast::hash(&raw).to_le_bytes())?;
    file.sync_all()?;
    fs::rename(&tmp_path, &final_path)?;
    // Directory-entry durability piggybacks on compaction's closing
    // fsync_dir (same directory); a crash before that point simply re-runs
    // the archive on the next compaction (overlap is deduplicated on read).
    Ok(final_path)
}

/// Remove orphaned segment tempfiles left by a crash mid-archive.
pub(crate) fn cleanup_orphan_segment_tmp(base: &Path) {
    let Some(dir) = base.parent() else { return };
    let Some(stem) = base.file_name().and_then(|s| s.to_str()) else { return };
    let prefix = format!("{}.h", stem);
    let Ok(read) = fs::read_dir(if dir.as_os_str().is_empty() { Path::new(".") } else { dir })
    else {
        return;
    };
    for entry in read.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with(&prefix) && name.ends_with(".tmp") {
            let _ = fs::remove_file(entry.path());
            log::warn!("removed orphaned history segment tempfile {}", name);
        }
    }
}

/// Read one segment: returns (first_seq, end_seq, entries).
fn read_segment(path: &Path, expected: SemLayout) -> HoronResult<(u32, u32, Vec<WalEntry>)> {
    let mut file = File::open(path)?;
    let mut header = [0u8; HIST_HEADER_SIZE];
    file.read_exact(&mut header)?;

    if header[0..4] != HIST_MAGIC {
        return Err(HoronError::InvalidFormat(format!(
            "{}: not a history segment (bad magic)",
            path.display()
        )));
    }
    // v1: plain full-width tails. v2: quantized layout recorded at offset 6.
    let layout = match header[4] {
        HIST_VERSION => SemLayout::plain(header[5] as usize),
        HIST_VERSION_QUANTIZED => SemLayout {
            dims: header[5] as usize,
            quantized: header[6] & HIST_FLAG_QUANTIZED != 0,
            gacl: header[6] & HIST_FLAG_GACL != 0,
        },
        v => {
            return Err(HoronError::InvalidFormat(format!(
                "{}: unsupported history segment version {}",
                path.display(),
                v
            )))
        }
    };
    if layout != expected {
        return Err(HoronError::InvalidFormat(format!(
            "{}: segment layout {:?} does not match main file's {:?}",
            path.display(),
            layout,
            expected
        )));
    }
    let first_seq = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let end_seq = u32::from_le_bytes(header[12..16].try_into().unwrap());
    let raw_len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    let comp_len = u32::from_le_bytes(header[20..24].try_into().unwrap()) as usize;
    if raw_len > MAX_HIST_BYTES || comp_len > MAX_HIST_BYTES {
        return Err(HoronError::InvalidFormat(format!(
            "{}: segment length field exceeds maximum — corrupt header",
            path.display()
        )));
    }

    let compressed = crate::format::read_bounded_vec(&mut file, comp_len, "history segment")?;
    let raw = zstd::bulk::decompress(&compressed, raw_len)
        .map_err(|e| HoronError::CompressionError(e.to_string()))?;

    let mut crc_buf = [0u8; 4];
    file.read_exact(&mut crc_buf)?;
    let stored = u32::from_le_bytes(crc_buf);
    let computed = crc32fast::hash(&raw);
    if stored != computed {
        return Err(HoronError::ChecksumMismatch {
            expected: stored,
            actual: computed,
            context: format!("history segment {}", path.display()),
        });
    }

    let mut entries = Vec::new();
    let mut cursor = std::io::Cursor::new(&raw);
    while let Some(entry) = WalEntry::read_from(&mut cursor, &layout)? {
        entries.push(entry);
    }
    Ok((first_seq, end_seq, entries))
}

// ---------------------------------------------------------------------------
// HoronHistory — the temporal reader
// ---------------------------------------------------------------------------

/// One sealed epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochInfo {
    /// Logical epoch counter (1-based; monotonically increasing).
    pub id: u64,
    /// WAL sequence of the original seal — the sample point.
    pub seq: u32,
    /// Sealed as speculative (projected / what-if, not recorded history).
    pub speculative: bool,
}

/// A node's state inside an [`HoronStateView`].
#[derive(Debug, Clone, Default)]
pub struct StateNode {
    /// Node payload bytes.
    pub data: Vec<u8>,
    /// Metadata pairs, in application order (later pairs shadow earlier).
    pub metadata: Vec<(String, String)>,
    /// Raw semantic coordinates (Q64.64, 16 bytes/dim); empty if never set.
    pub semantic: Vec<u8>,
}

/// Full node state at one epoch seal, reconstructed by replay.
#[derive(Debug, Default)]
pub struct HoronStateView {
    nodes: BTreeMap<String, StateNode>,
}

impl HoronStateView {
    /// Node state for a key.
    pub fn get(&self, key: &str) -> Option<&StateNode> {
        self.nodes.get(key)
    }

    /// All keys, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.nodes.keys().map(|k| k.as_str())
    }

    /// Number of live nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the view holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// How a key changed between two epochs (see [`HoronHistory::delta`]).
#[derive(Debug, Clone)]
pub struct KeyDelta {
    /// The node key.
    pub key: String,
    /// What happened to it.
    pub kind: DeltaKind,
}

/// The kind of change a [`KeyDelta`] records.
#[derive(Debug, Clone)]
pub enum DeltaKind {
    /// Key exists at the later epoch but not the earlier one.
    Added,
    /// Key exists at the earlier epoch but not the later one.
    Removed,
    /// Key exists at both; its semantic coordinates moved.
    Moved {
        /// Per-dimension displacement (later − earlier) over the queried range.
        displacement: Vec<f64>,
        /// Euclidean length of the displacement.
        distance: f64,
    },
}

/// Read-only temporal reader over a .htt file plus its history sidecars.
///
/// Opens the main file (header + live WAL; the snapshot is *not* used —
/// state is reconstructed purely by replaying the retained log) and every
/// `<file>.hNNNNNN` segment, merges entries by sequence number, and answers
/// epoch-grained temporal queries. This is an analysis path: one sequential
/// scan at open, O(retained history) memory.
pub struct HoronHistory {
    /// All retained entries, seq-deduplicated, in sequence order.
    entries: Vec<WalEntry>,
    /// Original seals only (compaction re-stamps dropped), ascending by id.
    epochs: Vec<EpochInfo>,
    /// Semantic dimension count from the main file header.
    semantic_dims: usize,
    /// True when history reaches back to the file's first write (seq 1).
    complete: bool,
}

impl HoronHistory {
    /// Open a file and its history segments for temporal reading.
    ///
    /// Fails on a corrupt segment (bad magic/CRC/dims). Missing segments do
    /// not fail the open — they leave [`Self::is_complete`] false, and
    /// [`Self::as_of`] refuses epochs older than the retained span.
    pub fn open<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;

        // Main file header.
        let mut hbuf = [0u8; HEADER_SIZE];
        file.read_exact(&mut hbuf)?;
        let header = GeoHeader::from_bytes(&hbuf)?;
        let semantic_dims = header.semantic_dims as usize;
        let layout = SemLayout::from_header(&header);

        // Skip the bounds section if present (flag-gated: a v4 quantized
        // file without meaning-addressing has no bounds section).
        if header.flags & FLAG_MEANING_ADDRESSED != 0 {
            let bounds_len = semantic_dims.saturating_sub(DIM_USER_DEFINED_START) * 16;
            file.seek(SeekFrom::Current(bounds_len as i64))?;
        }

        // Skip the snapshot section without decoding it.
        let mut buf4 = [0u8; 4];
        file.read_exact(&mut buf4)?;
        let snap_byte_len = u32::from_le_bytes(buf4) as u64;
        file.read_exact(&mut buf4)?;
        let node_count = u32::from_le_bytes(buf4);
        if node_count > 0 && header.compression_enabled() {
            file.read_exact(&mut buf4)?;
            let comp_len = u32::from_le_bytes(buf4) as u64;
            file.seek(SeekFrom::Current(comp_len as i64))?;
        } else {
            file.seek(SeekFrom::Current(snap_byte_len as i64))?;
        }
        if header.version >= 2 {
            file.seek(SeekFrom::Current(4))?; // snapshot CRC
        }

        // Live WAL entries (EOF/torn-tolerant, like recovery).
        let (_count, _base_seq) = wal::read_wal_header(&mut file)?;
        let mut merged: BTreeMap<u32, WalEntry> = BTreeMap::new();
        if header.wal_compressed() {
            let algo = header.compression_algo();
            while let Ok(Some((block, n))) = wal::read_wal_block(&mut file, algo) {
                let mut cursor = std::io::Cursor::new(&block);
                for _ in 0..n {
                    match WalEntry::read_from(&mut cursor, &layout) {
                        Ok(Some(e)) => {
                            merged.entry(e.seq).or_insert(e);
                        }
                        _ => break,
                    }
                }
            }
        } else {
            while let Ok(Some(e)) = WalEntry::read_from(&mut file, &layout) {
                merged.entry(e.seq).or_insert(e);
            }
        }

        // Sidecar segments (oldest first; duplicates deduped by seq).
        for (_n, seg_path) in list_segments(path) {
            let (_first, _end, entries) = read_segment(&seg_path, layout)?;
            for e in entries {
                merged.entry(e.seq).or_insert(e);
            }
        }

        let complete = merged.keys().next().is_none_or(|&first| first == 1);
        let entries: Vec<WalEntry> = merged.into_values().collect();

        // Collect original seals; drop compaction re-stamps (they only
        // ferry the counter — the original seal is the sample point).
        let mut epochs: Vec<EpochInfo> = Vec::new();
        for e in &entries {
            if let WalPayload::Epoch { epoch_id, flags } = &e.payload {
                if flags & EPOCH_FLAG_RESTAMP != 0 {
                    continue;
                }
                // First occurrence wins (lowest seq = the true seal).
                if !epochs.iter().any(|info| info.id == *epoch_id) {
                    epochs.push(EpochInfo {
                        id: *epoch_id,
                        seq: e.seq,
                        speculative: flags & EPOCH_FLAG_SPECULATIVE != 0,
                    });
                }
            }
        }
        epochs.sort_by_key(|i| i.id);

        Ok(Self { entries, epochs, semantic_dims, complete })
    }

    /// True when the retained log reaches back to the file's first write —
    /// only then can [`Self::as_of`] reconstruct any epoch exactly. False
    /// means segments are missing (retention enabled late, or sidecars
    /// deleted).
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Every sealed epoch (original seals; compaction re-stamps excluded),
    /// ascending by id. Speculative epochs are included and flagged.
    pub fn epochs(&self) -> &[EpochInfo] {
        &self.epochs
    }

    /// Semantic dimension count of the underlying file.
    pub fn semantic_dims(&self) -> usize {
        self.semantic_dims
    }

    /// Reconstruct the full node state as of an epoch seal (inclusive).
    ///
    /// Requires complete history (`is_complete()`), since state is a replay
    /// from the file's first write.
    pub fn as_of(&self, epoch_id: u64) -> HoronResult<HoronStateView> {
        let info = self.epoch_info(epoch_id)?;
        if !self.complete {
            return Err(HoronError::InvalidOperation(
                "history is incomplete (missing sidecar segments or retention \
                 enabled after writes) — as_of cannot reconstruct exact state"
                    .to_string(),
            ));
        }
        let mut view = HoronStateView::default();
        for e in &self.entries {
            if e.seq > info.seq {
                break;
            }
            Self::apply(&mut view, e);
        }
        Ok(view)
    }

    /// One key's semantic coordinates (decoded over `dim_range`) sampled at
    /// every **non-speculative** epoch seal, ascending by epoch. Epochs where
    /// the key does not exist (or has no coordinates yet) are omitted.
    pub fn trajectory(&self, key: &str, dim_range: &Range<usize>) -> Vec<(u64, Vec<f64>)> {
        let mut out = Vec::new();
        let mut current: Option<Vec<u8>> = None;
        let mut exists = false;
        let mut seals = self
            .epochs
            .iter()
            .filter(|i| !i.speculative)
            .peekable();

        for e in &self.entries {
            // Emit samples for every seal that precedes this entry.
            while let Some(info) = seals.peek() {
                if e.seq > info.seq {
                    if exists {
                        if let Some(coords) = &current {
                            out.push((info.id, decode_range(coords, dim_range)));
                        }
                    }
                    seals.next();
                } else {
                    break;
                }
            }
            if e.key == key {
                match &e.payload {
                    WalPayload::Insert(node) => {
                        exists = true;
                        if !node.semantic_coords.is_empty()
                            && node.semantic_coords.iter().any(|&b| b != 0)
                        {
                            current = Some(node.semantic_coords.clone());
                        }
                    }
                    WalPayload::Update { .. } => exists = true,
                    WalPayload::Delete => {
                        exists = false;
                        current = None;
                    }
                    WalPayload::SetSemantic { coords } => current = Some(coords.clone()),
                    _ => {}
                }
            }
        }
        // Seals at or after the last entry.
        for info in seals {
            if exists {
                if let Some(coords) = &current {
                    out.push((info.id, decode_range(coords, dim_range)));
                }
            }
        }
        out
    }

    /// What changed between two epochs, over `dim_range`: keys added,
    /// removed, or moved in semantic space. Requires complete history.
    pub fn delta(
        &self,
        epoch_a: u64,
        epoch_b: u64,
        dim_range: &Range<usize>,
    ) -> HoronResult<Vec<KeyDelta>> {
        let a = self.as_of(epoch_a)?;
        let b = self.as_of(epoch_b)?;
        let mut out = Vec::new();

        for (key, node_b) in &b.nodes {
            match a.nodes.get(key) {
                None => out.push(KeyDelta { key: key.clone(), kind: DeltaKind::Added }),
                Some(node_a) => {
                    let ca = decode_range(&node_a.semantic, dim_range);
                    let cb = decode_range(&node_b.semantic, dim_range);
                    let displacement: Vec<f64> =
                        ca.iter().zip(&cb).map(|(x, y)| y - x).collect();
                    let distance =
                        displacement.iter().map(|d| d * d).sum::<f64>().sqrt();
                    if distance > 0.0 {
                        out.push(KeyDelta {
                            key: key.clone(),
                            kind: DeltaKind::Moved { displacement, distance },
                        });
                    }
                }
            }
        }
        for key in a.nodes.keys() {
            if !b.nodes.contains_key(key) {
                out.push(KeyDelta { key: key.clone(), kind: DeltaKind::Removed });
            }
        }
        Ok(out)
    }

    fn epoch_info(&self, epoch_id: u64) -> HoronResult<EpochInfo> {
        self.epochs
            .iter()
            .find(|i| i.id == epoch_id)
            .copied()
            .ok_or_else(|| {
                HoronError::InvalidOperation(format!("unknown epoch {}", epoch_id))
            })
    }

    /// Apply one entry to a state view (pure data; no geometry).
    fn apply(view: &mut HoronStateView, e: &WalEntry) {
        match &e.payload {
            WalPayload::Insert(node) => {
                let s = view.nodes.entry(e.key.clone()).or_default();
                s.data = node.data.clone();
                s.metadata = node.metadata.clone();
                if !node.semantic_coords.is_empty()
                    && node.semantic_coords.iter().any(|&b| b != 0)
                {
                    s.semantic = node.semantic_coords.clone();
                }
            }
            WalPayload::Update { data, metadata } => {
                let s = view.nodes.entry(e.key.clone()).or_default();
                s.data = data.clone();
                for (mk, mv) in metadata {
                    s.metadata.push((mk.clone(), mv.clone()));
                }
            }
            WalPayload::Delete => {
                view.nodes.remove(&e.key);
            }
            WalPayload::SetMeta { meta_key, meta_value } => {
                if let Some(s) = view.nodes.get_mut(&e.key) {
                    s.metadata.push((meta_key.clone(), meta_value.clone()));
                }
            }
            WalPayload::SetSemantic { coords } => {
                if let Some(s) = view.nodes.get_mut(&e.key) {
                    s.semantic = coords.clone();
                }
            }
            WalPayload::Epoch { .. } => {}
        }
    }
}

/// Decode a dimension range of raw Q64.64 coordinates into f64 values.
/// Dimensions past the end of `coords` decode as 0.0.
fn decode_range(coords: &[u8], dim_range: &Range<usize>) -> Vec<f64> {
    dim_range
        .clone()
        .map(|dim| {
            let start = dim * 16;
            let end = start + 16;
            if coords.len() >= end {
                g_math::fixed_point::Q64_64::from_raw(i128::from_le_bytes(
                    coords[start..end].try_into().unwrap(),
                ))
                .to_f64()
            } else {
                0.0
            }
        })
        .collect()
}
