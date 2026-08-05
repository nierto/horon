//! Write-Ahead Log — append-only mutation log with per-entry CRC.
//!
//! Every Store mutation (insert, update, delete, set_meta, set_semantic)
//! is serialized as a WAL entry. On recovery, entries are replayed in
//! sequence order. The first entry with a bad CRC is treated as a partial
//! write — the WAL is truncated there.

use std::io::{Read, Write};

use crate::error::{HoronError, HoronResult};
use crate::format::*;
use crate::quant::SemLayout;
use crate::snapshot::NodeEntry;

/// A single WAL entry.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Monotonically increasing sequence number assigned at append time.
    pub seq: u32,
    /// Operation code (one of the `format::OP_*` constants).
    pub op: u8,
    /// Key of the node the operation targets.
    pub key: String,
    /// Operation-specific payload.
    pub payload: WalPayload,
}

/// Op-specific payload.
#[derive(Debug, Clone)]
pub enum WalPayload {
    /// INSERT: full node entry.
    Insert(NodeEntry),
    /// UPDATE: replacement data + metadata.
    Update {
        /// Replacement payload bytes for the node.
        data: Vec<u8>,
        /// Replacement metadata pairs (replaces all existing metadata).
        metadata: Vec<(String, String)>,
    },
    /// DELETE: no additional data.
    Delete,
    /// SET_META: single key/value.
    SetMeta {
        /// Metadata key to set.
        meta_key: String,
        /// Value to associate with `meta_key`.
        meta_value: String,
    },
    /// SET_SEMANTIC: full semantic coordinate replacement.
    SetSemantic {
        /// New semantic coordinates as raw Q64.64 bytes (16 bytes per dimension).
        coords: Vec<u8>,
    },
    /// EPOCH: seal marker (temporal epochs). No node state change; replay advances the
    /// in-memory epoch counter. `key` is empty — epochs are file-scoped.
    Epoch {
        /// Monotonically increasing logical epoch counter (not wall-clock).
        epoch_id: u64,
        /// `EPOCH_FLAG_*` bits (bit 0: speculative).
        flags: u8,
    },
}

impl WalEntry {
    /// Serialize this entry to bytes (excluding the trailing CRC).
    ///
    /// Semantic coordinates are held full-width (16 bytes/dim) in memory;
    /// `layout` decides the on-disk tail encoding (quantization).
    fn write_body<W: Write>(&self, w: &mut W, layout: &SemLayout) -> HoronResult<()> {
        // seq (u32)
        w.write_all(&self.seq.to_le_bytes())?;
        // op (u8)
        w.write_all(&[self.op])?;
        // key
        let key_bytes = self.key.as_bytes();
        w.write_all(&(key_bytes.len() as u16).to_le_bytes())?;
        w.write_all(key_bytes)?;

        match &self.payload {
            WalPayload::Insert(entry) => {
                // data
                w.write_all(&(entry.data.len() as u32).to_le_bytes())?;
                w.write_all(&entry.data)?;
                // metadata
                w.write_all(&(entry.metadata.len() as u16).to_le_bytes())?;
                for (mk, mv) in &entry.metadata {
                    let mk_b = mk.as_bytes();
                    let mv_b = mv.as_bytes();
                    w.write_all(&(mk_b.len() as u16).to_le_bytes())?;
                    w.write_all(mk_b)?;
                    w.write_all(&(mv_b.len() as u16).to_le_bytes())?;
                    w.write_all(mv_b)?;
                }
                // semantic coords
                if layout.quantized {
                    if layout.disk_bytes() > 0 {
                        w.write_all(&layout.encode_tail(&entry.semantic_coords)?)?;
                    }
                } else if !entry.semantic_coords.is_empty() {
                    w.write_all(&entry.semantic_coords)?;
                }
            }
            WalPayload::Update { data, metadata } => {
                w.write_all(&(data.len() as u32).to_le_bytes())?;
                w.write_all(data)?;
                w.write_all(&(metadata.len() as u16).to_le_bytes())?;
                for (mk, mv) in metadata {
                    let mk_b = mk.as_bytes();
                    let mv_b = mv.as_bytes();
                    w.write_all(&(mk_b.len() as u16).to_le_bytes())?;
                    w.write_all(mk_b)?;
                    w.write_all(&(mv_b.len() as u16).to_le_bytes())?;
                    w.write_all(mv_b)?;
                }
            }
            WalPayload::Delete => {
                // No additional fields
            }
            WalPayload::SetMeta { meta_key, meta_value } => {
                let mk_b = meta_key.as_bytes();
                let mv_b = meta_value.as_bytes();
                w.write_all(&(mk_b.len() as u16).to_le_bytes())?;
                w.write_all(mk_b)?;
                w.write_all(&(mv_b.len() as u16).to_le_bytes())?;
                w.write_all(mv_b)?;
            }
            WalPayload::SetSemantic { coords } => {
                if layout.quantized {
                    w.write_all(&layout.encode_tail(coords)?)?;
                } else {
                    w.write_all(coords)?;
                }
            }
            WalPayload::Epoch { epoch_id, flags } => {
                w.write_all(&epoch_id.to_le_bytes())?;
                w.write_all(&[*flags])?;
            }
        }

        Ok(())
    }

    /// Serialize entry with trailing CRC32.
    pub fn write_to<W: Write>(&self, w: &mut W, layout: &SemLayout) -> HoronResult<()> {
        let mut body = Vec::new();
        self.write_body(&mut body, layout)?;

        let crc = crc32fast::hash(&body);
        w.write_all(&body)?;
        w.write_all(&crc.to_le_bytes())?;

        Ok(())
    }

    /// Deserialize an entry from bytes. Returns None if CRC check fails
    /// (indicating a partial write / truncation point). The payload's
    /// semantic coordinates come back full-width regardless of the disk
    /// encoding (`layout` — quantization).
    pub fn read_from<R: Read>(
        r: &mut R,
        layout: &SemLayout,
    ) -> HoronResult<Option<Self>> {
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];

        // We need to buffer the body for CRC verification
        // Read seq
        if r.read_exact(&mut buf4).is_err() {
            return Ok(None); // EOF — no more entries
        }
        let mut body = Vec::new();
        body.extend_from_slice(&buf4);
        let seq = u32::from_le_bytes(buf4);

        // op
        let mut op_buf = [0u8; 1];
        r.read_exact(&mut op_buf)?;
        body.push(op_buf[0]);
        let op = op_buf[0];

        // key
        r.read_exact(&mut buf2)?;
        body.extend_from_slice(&buf2);
        let key_len = u16::from_le_bytes(buf2) as usize;
        let mut key_buf = vec![0u8; key_len];
        r.read_exact(&mut key_buf)?;
        body.extend_from_slice(&key_buf);
        let key = String::from_utf8(key_buf)
            .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8 key: {}", e)))?;

        // Op-specific body reading (accumulates into body for CRC)
        let payload = match op {
            OP_INSERT => {
                let entry = read_insert_body(r, &mut body, layout)?;
                WalPayload::Insert(entry)
            }
            OP_UPDATE => {
                let (data, metadata) = read_update_body(r, &mut body)?;
                WalPayload::Update { data, metadata }
            }
            OP_DELETE => {
                WalPayload::Delete
            }
            OP_SET_META => {
                let (mk, mv) = read_meta_body(r, &mut body)?;
                WalPayload::SetMeta { meta_key: mk, meta_value: mv }
            }
            OP_SET_SEMANTIC => {
                // The CRC covers the on-disk bytes; the payload is decoded
                // to full width afterwards.
                let mut disk = vec![0u8; layout.disk_bytes()];
                r.read_exact(&mut disk)?;
                body.extend_from_slice(&disk);
                let coords = if layout.quantized {
                    layout.decode_tail(&disk)?
                } else {
                    disk
                };
                WalPayload::SetSemantic { coords }
            }
            OP_EPOCH => {
                let mut buf8 = [0u8; 8];
                r.read_exact(&mut buf8)?;
                body.extend_from_slice(&buf8);
                let epoch_id = u64::from_le_bytes(buf8);
                let mut flag_buf = [0u8; 1];
                r.read_exact(&mut flag_buf)?;
                body.push(flag_buf[0]);
                WalPayload::Epoch { epoch_id, flags: flag_buf[0] }
            }
            _ => {
                return Err(HoronError::InvalidFormat(
                    format!("unknown WAL op code: 0x{:02X}", op)
                ));
            }
        };

        // Read and verify CRC
        r.read_exact(&mut buf4)?;
        let stored_crc = u32::from_le_bytes(buf4);
        let computed_crc = crc32fast::hash(&body);

        if stored_crc != computed_crc {
            // Partial write detected — truncation point
            return Ok(None);
        }

        Ok(Some(WalEntry { seq, op, key, payload }))
    }
}

/// Read INSERT body fields, appending raw bytes to `body` for CRC.
fn read_insert_body<R: Read>(
    r: &mut R,
    body: &mut Vec<u8>,
    layout: &SemLayout,
) -> HoronResult<NodeEntry> {
    let mut buf2 = [0u8; 2];
    let mut buf4 = [0u8; 4];

    // data
    r.read_exact(&mut buf4)?;
    body.extend_from_slice(&buf4);
    let data_len = u32::from_le_bytes(buf4) as usize;
    if data_len > MAX_ENTRY_DATA {
        return Err(HoronError::InvalidFormat(format!(
            "WAL entry data length {} exceeds maximum {} — corrupt length field",
            data_len, MAX_ENTRY_DATA
        )));
    }
    let data = crate::format::read_bounded_vec(r, data_len, "WAL entry data")?;
    body.extend_from_slice(&data);

    // metadata
    r.read_exact(&mut buf2)?;
    body.extend_from_slice(&buf2);
    let meta_count = u16::from_le_bytes(buf2) as usize;
    let mut metadata = Vec::with_capacity(meta_count);
    for _ in 0..meta_count {
        r.read_exact(&mut buf2)?;
        body.extend_from_slice(&buf2);
        let mk_len = u16::from_le_bytes(buf2) as usize;
        let mut mk_buf = vec![0u8; mk_len];
        r.read_exact(&mut mk_buf)?;
        body.extend_from_slice(&mk_buf);

        r.read_exact(&mut buf2)?;
        body.extend_from_slice(&buf2);
        let mv_len = u16::from_le_bytes(buf2) as usize;
        let mut mv_buf = vec![0u8; mv_len];
        r.read_exact(&mut mv_buf)?;
        body.extend_from_slice(&mv_buf);

        let mk = String::from_utf8(mk_buf)
            .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8: {}", e)))?;
        let mv = String::from_utf8(mv_buf)
            .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8: {}", e)))?;
        metadata.push((mk, mv));
    }

    // semantic coords (CRC over disk bytes; payload decoded to full width)
    let disk_bytes = layout.disk_bytes();
    let mut semantic_coords = vec![0u8; disk_bytes];
    if disk_bytes > 0 {
        r.read_exact(&mut semantic_coords)?;
        body.extend_from_slice(&semantic_coords);
        if layout.quantized {
            semantic_coords = layout.decode_tail(&semantic_coords)?;
        }
    }

    Ok(NodeEntry {
        key: String::new(), // key already parsed by caller
        data,
        metadata,
        semantic_coords,
    })
}

/// Read UPDATE body fields.
fn read_update_body<R: Read>(
    r: &mut R,
    body: &mut Vec<u8>,
) -> HoronResult<(Vec<u8>, Vec<(String, String)>)> {
    let mut buf2 = [0u8; 2];
    let mut buf4 = [0u8; 4];

    // data
    r.read_exact(&mut buf4)?;
    body.extend_from_slice(&buf4);
    let data_len = u32::from_le_bytes(buf4) as usize;
    if data_len > MAX_ENTRY_DATA {
        return Err(HoronError::InvalidFormat(format!(
            "WAL entry data length {} exceeds maximum {} — corrupt length field",
            data_len, MAX_ENTRY_DATA
        )));
    }
    let data = crate::format::read_bounded_vec(r, data_len, "WAL entry data")?;
    body.extend_from_slice(&data);

    // metadata
    r.read_exact(&mut buf2)?;
    body.extend_from_slice(&buf2);
    let meta_count = u16::from_le_bytes(buf2) as usize;
    let mut metadata = Vec::with_capacity(meta_count);
    for _ in 0..meta_count {
        r.read_exact(&mut buf2)?;
        body.extend_from_slice(&buf2);
        let mk_len = u16::from_le_bytes(buf2) as usize;
        let mut mk_buf = vec![0u8; mk_len];
        r.read_exact(&mut mk_buf)?;
        body.extend_from_slice(&mk_buf);

        r.read_exact(&mut buf2)?;
        body.extend_from_slice(&buf2);
        let mv_len = u16::from_le_bytes(buf2) as usize;
        let mut mv_buf = vec![0u8; mv_len];
        r.read_exact(&mut mv_buf)?;
        body.extend_from_slice(&mv_buf);

        let mk = String::from_utf8(mk_buf)
            .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8: {}", e)))?;
        let mv = String::from_utf8(mv_buf)
            .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8: {}", e)))?;
        metadata.push((mk, mv));
    }

    Ok((data, metadata))
}

/// Read SET_META body fields.
fn read_meta_body<R: Read>(
    r: &mut R,
    body: &mut Vec<u8>,
) -> HoronResult<(String, String)> {
    let mut buf2 = [0u8; 2];

    r.read_exact(&mut buf2)?;
    body.extend_from_slice(&buf2);
    let mk_len = u16::from_le_bytes(buf2) as usize;
    let mut mk_buf = vec![0u8; mk_len];
    r.read_exact(&mut mk_buf)?;
    body.extend_from_slice(&mk_buf);

    r.read_exact(&mut buf2)?;
    body.extend_from_slice(&buf2);
    let mv_len = u16::from_le_bytes(buf2) as usize;
    let mut mv_buf = vec![0u8; mv_len];
    r.read_exact(&mut mv_buf)?;
    body.extend_from_slice(&mv_buf);

    let mk = String::from_utf8(mk_buf)
        .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8: {}", e)))?;
    let mv = String::from_utf8(mv_buf)
        .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8: {}", e)))?;

    Ok((mk, mv))
}

/// Maximum decompressed WAL block size (256 MB safety cap).
const MAX_BLOCK_DECOMPRESSED: usize = 256 * 1024 * 1024;

/// Write a compressed WAL block.
///
/// Block format: `entry_count (u16 LE) + compressed_len (u32 LE) + compressed_data`.
pub fn write_wal_block<W: Write>(
    w: &mut W,
    entries_data: &[u8],
    entry_count: u16,
    algo: u8,
) -> HoronResult<()> {
    let compressed = crate::compression::compress(entries_data, algo)?;
    w.write_all(&entry_count.to_le_bytes())?;
    w.write_all(&(compressed.len() as u32).to_le_bytes())?;
    w.write_all(&compressed)?;
    Ok(())
}

/// Read a single compressed WAL block. Returns `(decompressed_bytes, entry_count)`
/// or `None` on clean EOF.
pub fn read_wal_block<R: Read>(
    r: &mut R,
    algo: u8,
) -> HoronResult<Option<(Vec<u8>, u16)>> {
    // Read block_entry_count (u16)
    let mut buf2 = [0u8; 2];
    if r.read_exact(&mut buf2).is_err() {
        return Ok(None); // clean EOF
    }
    let entry_count = u16::from_le_bytes(buf2);

    if entry_count == 0 || entry_count > WAL_BLOCK_SIZE as u16 {
        return Ok(None); // invalid block header — treat as truncation
    }

    // Read compressed_len (u32)
    let mut buf4 = [0u8; 4];
    if r.read_exact(&mut buf4).is_err() {
        return Ok(None); // truncated block header
    }
    let compressed_len = u32::from_le_bytes(buf4) as usize;

    if compressed_len == 0 || compressed_len > MAX_BLOCK_DECOMPRESSED {
        return Ok(None); // invalid or absurd size
    }

    // Read compressed data (allocation bounded by bytes actually present)
    let compressed = match crate::format::read_bounded_vec(r, compressed_len, "WAL block") {
        Ok(c) => c,
        Err(_) => return Ok(None), // truncated compressed data
    };

    // Decompress
    let decompressed = crate::compression::decompress(&compressed, algo, MAX_BLOCK_DECOMPRESSED)?;
    Ok(Some((decompressed, entry_count)))
}

/// Write WAL section header.
pub fn write_wal_header<W: Write>(w: &mut W, entry_count: u32, base_seq: u32) -> HoronResult<()> {
    w.write_all(&entry_count.to_le_bytes())?;
    w.write_all(&base_seq.to_le_bytes())?;
    Ok(())
}

/// Read WAL section header. Returns (entry_count, base_seq).
pub fn read_wal_header<R: Read>(r: &mut R) -> HoronResult<(u32, u32)> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let entry_count = u32::from_le_bytes(buf4);
    r.read_exact(&mut buf4)?;
    let base_seq = u32::from_le_bytes(buf4);
    Ok((entry_count, base_seq))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Zero-dim plain layout — the tail-less common case for these tests.
    fn l0() -> SemLayout {
        SemLayout::plain(0)
    }

    fn make_insert_entry(seq: u32, key: &str, data: &[u8]) -> WalEntry {
        WalEntry {
            seq,
            op: OP_INSERT,
            key: key.to_string(),
            payload: WalPayload::Insert(NodeEntry {
                key: key.to_string(),
                data: data.to_vec(),
                metadata: vec![],
                semantic_coords: vec![],
            }),
        }
    }

    #[test]
    fn test_wal_entry_insert_roundtrip() {
        let entry = make_insert_entry(1, "/test", b"hello");

        let mut buf = Vec::new();
        entry.write_to(&mut buf, &l0()).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = WalEntry::read_from(&mut cursor, &l0()).unwrap().unwrap();

        assert_eq!(parsed.seq, 1);
        assert_eq!(parsed.op, OP_INSERT);
        assert_eq!(parsed.key, "/test");
        match parsed.payload {
            WalPayload::Insert(e) => assert_eq!(e.data, b"hello"),
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn test_wal_entry_delete_roundtrip() {
        let entry = WalEntry {
            seq: 5,
            op: OP_DELETE,
            key: "/gone".to_string(),
            payload: WalPayload::Delete,
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf, &l0()).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = WalEntry::read_from(&mut cursor, &l0()).unwrap().unwrap();

        assert_eq!(parsed.seq, 5);
        assert_eq!(parsed.op, OP_DELETE);
        assert!(matches!(parsed.payload, WalPayload::Delete));
    }

    #[test]
    fn test_wal_entry_set_meta_roundtrip() {
        let entry = WalEntry {
            seq: 10,
            op: OP_SET_META,
            key: "/doc".to_string(),
            payload: WalPayload::SetMeta {
                meta_key: "author".to_string(),
                meta_value: "alice".to_string(),
            },
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf, &l0()).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = WalEntry::read_from(&mut cursor, &l0()).unwrap().unwrap();

        assert_eq!(parsed.seq, 10);
        match parsed.payload {
            WalPayload::SetMeta { meta_key, meta_value } => {
                assert_eq!(meta_key, "author");
                assert_eq!(meta_value, "alice");
            }
            _ => panic!("expected SetMeta"),
        }
    }

    #[test]
    fn test_wal_corrupted_crc_returns_none() {
        let entry = make_insert_entry(1, "/test", b"data");
        let mut buf = Vec::new();
        entry.write_to(&mut buf, &l0()).unwrap();

        // Corrupt last byte (part of CRC)
        let len = buf.len();
        buf[len - 1] ^= 0xFF;

        let mut cursor = Cursor::new(&buf);
        let result = WalEntry::read_from(&mut cursor, &l0()).unwrap();
        assert!(result.is_none(), "corrupted CRC should return None");
    }

    #[test]
    fn test_wal_multiple_entries() {
        let entries = vec![
            make_insert_entry(1, "/a", b"aaa"),
            make_insert_entry(2, "/b", b"bbb"),
            WalEntry {
                seq: 3,
                op: OP_DELETE,
                key: "/a".to_string(),
                payload: WalPayload::Delete,
            },
        ];

        let mut buf = Vec::new();
        for e in &entries {
            e.write_to(&mut buf, &l0()).unwrap();
        }

        let mut cursor = Cursor::new(&buf);
        let e1 = WalEntry::read_from(&mut cursor, &l0()).unwrap().unwrap();
        let e2 = WalEntry::read_from(&mut cursor, &l0()).unwrap().unwrap();
        let e3 = WalEntry::read_from(&mut cursor, &l0()).unwrap().unwrap();

        assert_eq!(e1.seq, 1);
        assert_eq!(e2.seq, 2);
        assert_eq!(e3.seq, 3);
        assert_eq!(e3.op, OP_DELETE);

        // No more entries
        let e4 = WalEntry::read_from(&mut cursor, &l0()).unwrap();
        assert!(e4.is_none());
    }

    #[test]
    fn test_wal_header_roundtrip() {
        let mut buf = Vec::new();
        write_wal_header(&mut buf, 42, 100).unwrap();

        let mut cursor = Cursor::new(&buf);
        let (count, base) = read_wal_header(&mut cursor).unwrap();
        assert_eq!(count, 42);
        assert_eq!(base, 100);
    }

    #[test]
    fn test_wal_block_roundtrip() {
        let entries: Vec<WalEntry> = (1..=10)
            .map(|i| make_insert_entry(i, &format!("/n{}", i), b"data"))
            .collect();

        // Serialize entries into raw bytes
        let mut raw = Vec::new();
        for e in &entries {
            e.write_to(&mut raw, &l0()).unwrap();
        }

        // Write as compressed block
        let mut block_buf = Vec::new();
        write_wal_block(&mut block_buf, &raw, entries.len() as u16, ALGO_ZSTD).unwrap();

        // Read back
        let mut cursor = Cursor::new(&block_buf);
        let (decompressed, count) = read_wal_block(&mut cursor, ALGO_ZSTD).unwrap().unwrap();
        assert_eq!(count, 10);
        assert_eq!(decompressed, raw);

        // Parse individual entries from decompressed block
        let mut inner = Cursor::new(&decompressed);
        for i in 1..=10u32 {
            let parsed = WalEntry::read_from(&mut inner, &l0()).unwrap().unwrap();
            assert_eq!(parsed.seq, i);
        }
        assert!(WalEntry::read_from(&mut inner, &l0()).unwrap().is_none());
    }

    #[test]
    fn test_wal_block_single_entry() {
        let entry = make_insert_entry(1, "/single", b"one");
        let mut raw = Vec::new();
        entry.write_to(&mut raw, &l0()).unwrap();

        let mut block_buf = Vec::new();
        write_wal_block(&mut block_buf, &raw, 1, ALGO_ZSTD).unwrap();

        let mut cursor = Cursor::new(&block_buf);
        let (decompressed, count) = read_wal_block(&mut cursor, ALGO_ZSTD).unwrap().unwrap();
        assert_eq!(count, 1);
        assert_eq!(decompressed, raw);
    }

    #[test]
    fn test_wal_block_full_64() {
        let entries: Vec<WalEntry> = (1..=64)
            .map(|i| make_insert_entry(i, &format!("/node_{}", i), b"payload"))
            .collect();

        let mut raw = Vec::new();
        for e in &entries {
            e.write_to(&mut raw, &l0()).unwrap();
        }

        let mut block_buf = Vec::new();
        write_wal_block(&mut block_buf, &raw, 64, ALGO_ZSTD).unwrap();

        let mut cursor = Cursor::new(&block_buf);
        let (decompressed, count) = read_wal_block(&mut cursor, ALGO_ZSTD).unwrap().unwrap();
        assert_eq!(count, 64);
        assert_eq!(decompressed, raw);
    }

    #[test]
    fn test_wal_block_eof_returns_none() {
        let cursor = Cursor::new(Vec::<u8>::new());
        let result = read_wal_block(&mut cursor.clone(), ALGO_ZSTD).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_wal_block_truncated_header() {
        // Only 1 byte — partial block_entry_count
        let mut cursor = Cursor::new(vec![0x01]);
        let result = read_wal_block(&mut cursor, ALGO_ZSTD).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_wal_block_multiple_blocks() {
        let mut all_blocks = Vec::new();
        let mut expected_entries = Vec::new();

        // Write 3 blocks: 64 + 64 + 12
        for block_idx in 0..3u32 {
            let count = if block_idx < 2 { 64 } else { 12 };
            let entries: Vec<WalEntry> = (0..count)
                .map(|i| {
                    let seq = block_idx * 64 + i + 1;
                    make_insert_entry(seq, &format!("/b{}/n{}", block_idx, i), b"x")
                })
                .collect();

            let mut raw = Vec::new();
            for e in &entries {
                e.write_to(&mut raw, &l0()).unwrap();
            }

            write_wal_block(&mut all_blocks, &raw, count as u16, ALGO_ZSTD).unwrap();
            expected_entries.extend(entries);
        }

        // Read all 3 blocks back
        let mut cursor = Cursor::new(&all_blocks);
        let mut read_count = 0u32;
        for block_idx in 0..3 {
            let (decompressed, count) = read_wal_block(&mut cursor, ALGO_ZSTD).unwrap().unwrap();
            let expected_count: u16 = if block_idx < 2 { 64 } else { 12 };
            assert_eq!(count, expected_count);

            let mut inner = Cursor::new(&decompressed);
            for _ in 0..count {
                let parsed = WalEntry::read_from(&mut inner, &l0()).unwrap().unwrap();
                assert_eq!(parsed.seq, read_count + 1);
                read_count += 1;
            }
        }
        assert_eq!(read_count, 140);

        // No more blocks
        assert!(read_wal_block(&mut cursor, ALGO_ZSTD).unwrap().is_none());
    }
}
