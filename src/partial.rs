//! Partial (mmap-backed) snapshot reads — the partial-reads half of meaning-addressed
//! storage.
//!
//! Instead of materializing every snapshot entry into RAM at open, the file
//! is memory-mapped and a single structural scan builds a small index:
//! `key → (offset, len, hilbert)`. Payloads stay on disk until a read
//! actually touches them; the OS page cache does the rest.
//!
//! For meaning-addressed files (format v3), entries are stored in pure
//! global-Hilbert order, so the index is simultaneously the chunk directory:
//! a semantic query maps to a position on the curve, and its candidates are
//! a contiguous byte range of the file.
//!
//! Only uncompressed snapshots can be partially read — a zstd single frame
//! must be decompressed whole.
//!
//! # mmap safety argument
//!
//! `SnapView` owns the `Mmap` privately and lives behind
//! `RwLock<SnapView>` in the partial-mode state. Every access decodes and
//! **copies out** while holding the read guard. Remap happens in exactly one
//! place — compaction — which maps the newly renamed file, builds a fresh
//! `SnapView`, and swaps it in under the **write** lock (and under the WAL
//! lock): the swap blocks until every reader releases its guard, so the old
//! mapping is only dropped once nobody can touch it. No in-process
//! use-after-remap is possible.
//!
//! What this does NOT protect against: a *non-cooperating external process*
//! truncating or rewriting the file under the mapping (the `flock` is
//! advisory, and a no-op on non-unix). That can SIGBUS on the next page
//! fault. Precondition: do not point partial mode at a file other writers
//! can mutate.
//!
//! Parsing over the mmap is allocation-free: every length field is walked
//! with `checked_add` + `slice::get`, so a corrupt file yields a clean
//! error, never an oversized allocation or panic.

use std::collections::HashMap;
use std::io::Cursor;

use memmap2::Mmap;

use crate::error::{HoronError, HoronResult};
use crate::quant::SemLayout;
use crate::snapshot::NodeEntry;

/// Location (and Hilbert address) of one snapshot entry inside the mmap.
#[derive(Debug, Clone)]
pub struct EntryLoc {
    /// Key of the node stored at this location.
    pub key: String,
    /// Byte offset of the entry, relative to the start of the raw entry
    /// region (i.e. after the 8-byte snapshot section header).
    pub off: usize,
    /// Length of the serialized entry in bytes.
    pub len: usize,
    /// Global-bounds Hilbert index of the entry's user semantic dims
    /// (0 when the file is not meaning-addressed or the entry has no coords).
    pub hilbert: u128,
}

/// An open snapshot view: the mmap plus the structural index.
pub struct SnapView {
    mmap: Mmap,
    /// Absolute file offset where the raw entry region starts.
    raw_start: usize,
    /// Entries in on-disk order (for v3 files: pure Hilbert order).
    pub entries: Vec<EntryLoc>,
    /// key → index into `entries`.
    pub by_key: HashMap<String, usize>,
    layout: SemLayout,
}

impl SnapView {
    /// Build a view over an uncompressed snapshot: structurally scan every
    /// entry recording offsets, keys, and (optionally) Hilbert addresses —
    /// without materializing payloads.
    ///
    /// `raw_start`/`raw_len` delimit the raw entry region in the file.
    /// `hilbert_of` computes the global Hilbert address from an entry's
    /// semantic coordinate bytes (None for non-meaning-addressed files).
    pub fn scan(
        mmap: Mmap,
        raw_start: usize,
        raw_len: usize,
        node_count: usize,
        layout: SemLayout,
        hilbert_of: Option<&dyn Fn(&[u8]) -> u128>,
    ) -> HoronResult<Self> {
        let region = mmap
            .get(raw_start..raw_start + raw_len)
            .ok_or_else(|| HoronError::InvalidFormat(
                "snapshot region exceeds file length".to_string(),
            ))?;

        let mut entries = Vec::with_capacity(node_count.min(1_000_000));
        let mut by_key = HashMap::with_capacity(node_count.min(1_000_000));
        let mut pos = 0usize;

        for _ in 0..node_count {
            let start = pos;

            // key_len + key
            let key_len = read_u16(region, &mut pos)? as usize;
            let key_bytes = read_slice(region, &mut pos, key_len)?;
            let key = std::str::from_utf8(key_bytes)
                .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8 key: {}", e)))?
                .to_string();

            // data_len + data (skipped — this is the point)
            let data_len = read_u32(region, &mut pos)? as usize;
            skip(region, &mut pos, data_len)?;

            // metadata
            let meta_count = read_u16(region, &mut pos)? as usize;
            for _ in 0..meta_count {
                let mk_len = read_u16(region, &mut pos)? as usize;
                skip(region, &mut pos, mk_len)?;
                let mv_len = read_u16(region, &mut pos)? as usize;
                skip(region, &mut pos, mv_len)?;
            }

            // semantic coords — read for the Hilbert address, no copy
            // (disk encoding; `hilbert_of` decodes if the file is quantized)
            let sem = read_slice(region, &mut pos, layout.disk_bytes())?;
            let hilbert = hilbert_of.map(|f| f(sem)).unwrap_or(0);

            let idx = entries.len();
            entries.push(EntryLoc { key: key.clone(), off: start, len: pos - start, hilbert });
            by_key.insert(key, idx);
        }

        Ok(Self { mmap, raw_start, entries, by_key, layout })
    }

    /// Decode the full entry at index `i` from the mmap.
    pub fn decode(&self, i: usize) -> HoronResult<NodeEntry> {
        let loc = self.entries.get(i).ok_or_else(|| {
            HoronError::InvalidFormat(format!("entry index {} out of range", i))
        })?;
        let abs = self.raw_start + loc.off;
        let bytes = self.mmap.get(abs..abs + loc.len).ok_or_else(|| {
            HoronError::InvalidFormat("entry range exceeds file length".to_string())
        })?;
        NodeEntry::read_from(&mut Cursor::new(bytes), &self.layout)
    }

    /// The semantic coordinate bytes of entry `i` in their ON-DISK encoding
    /// (they sit at the fixed-size tail of the entry). For quantized files
    /// the caller decodes via [`SnapView::layout`].
    pub fn semantic_of(&self, i: usize) -> HoronResult<&[u8]> {
        let loc = self.entries.get(i).ok_or_else(|| {
            HoronError::InvalidFormat(format!("entry index {} out of range", i))
        })?;
        let abs_end = self.raw_start + loc.off + loc.len;
        // An entry shorter than the semantic tail would underflow this range.
        let abs_start = abs_end.checked_sub(self.layout.disk_bytes()).ok_or_else(|| {
            HoronError::InvalidFormat("entry shorter than its semantic tail".to_string())
        })?;
        self.mmap
            .get(abs_start..abs_end)
            .ok_or_else(|| HoronError::InvalidFormat(
                "semantic range exceeds file length".to_string(),
            ))
    }

    /// The semantic-tail layout this view was scanned with.
    pub fn layout(&self) -> SemLayout {
        self.layout
    }

    /// Position on the Hilbert-ordered entry list where `target` would
    /// insert — the center of a window search. Only meaningful for v3 files
    /// (entries stored in Hilbert order).
    pub fn hilbert_position(&self, target: u128) -> usize {
        self.entries
            .partition_point(|e| e.hilbert < target)
    }
}

fn read_u16(region: &[u8], pos: &mut usize) -> HoronResult<u16> {
    let b = read_slice(region, pos, 2)?;
    Ok(u16::from_le_bytes([b[0], b[1]]))
}

fn read_u32(region: &[u8], pos: &mut usize) -> HoronResult<u32> {
    let b = read_slice(region, pos, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn read_slice<'a>(region: &'a [u8], pos: &mut usize, len: usize) -> HoronResult<&'a [u8]> {
    let end = pos.checked_add(len).ok_or_else(|| {
        HoronError::InvalidFormat("entry offset overflow".to_string())
    })?;
    let s = region.get(*pos..end).ok_or_else(|| {
        HoronError::InvalidFormat("truncated snapshot entry".to_string())
    })?;
    *pos = end;
    Ok(s)
}

fn skip(region: &[u8], pos: &mut usize, len: usize) -> HoronResult<()> {
    read_slice(region, pos, len).map(|_| ())
}
