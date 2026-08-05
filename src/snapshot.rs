//! Snapshot section — serialization and deserialization of node entries.
//!
//! A snapshot is a point-in-time representation of all live nodes,
//! stored in insertion order so that Sarkar reconstruction on replay
//! produces identical geometric state.

use std::io::{Read, Write, Cursor};

use crate::error::{HoronError, HoronResult};
use crate::format::{MAX_ENTRY_DATA, MAX_SNAPSHOT_BYTES};
use crate::quant::SemLayout;

/// A single node entry in a snapshot or INSERT WAL op.
#[derive(Debug, Clone)]
pub struct NodeEntry {
    /// Hierarchical path key identifying the node.
    pub key: String,
    /// Raw payload bytes stored at the node.
    pub data: Vec<u8>,
    /// Metadata key/value pairs attached to the node.
    pub metadata: Vec<(String, String)>,
    /// Semantic coordinates as raw Q64.64 bytes (16 bytes per dimension) —
    /// always full-width in memory; the on-disk tail encoding is decided by
    /// the [`SemLayout`] passed to `write_to`/`read_from` (quantization).
    /// Empty if no semantic dimensions.
    pub semantic_coords: Vec<u8>,
}

impl NodeEntry {
    /// Serialize this entry to bytes.
    pub fn write_to<W: Write>(&self, w: &mut W, layout: &SemLayout) -> HoronResult<usize> {
        let mut written = 0;

        // key_len (u16) + key
        let key_bytes = self.key.as_bytes();
        w.write_all(&(key_bytes.len() as u16).to_le_bytes())?;
        w.write_all(key_bytes)?;
        written += 2 + key_bytes.len();

        // data_len (u32) + data
        w.write_all(&(self.data.len() as u32).to_le_bytes())?;
        w.write_all(&self.data)?;
        written += 4 + self.data.len();

        // meta_count (u16) + metadata pairs
        w.write_all(&(self.metadata.len() as u16).to_le_bytes())?;
        written += 2;
        for (mk, mv) in &self.metadata {
            let mk_bytes = mk.as_bytes();
            let mv_bytes = mv.as_bytes();
            w.write_all(&(mk_bytes.len() as u16).to_le_bytes())?;
            w.write_all(mk_bytes)?;
            w.write_all(&(mv_bytes.len() as u16).to_le_bytes())?;
            w.write_all(mv_bytes)?;
            written += 2 + mk_bytes.len() + 2 + mv_bytes.len();
        }

        // semantic coords (fixed-size tail, length known from the layout)
        if layout.quantized {
            if layout.disk_bytes() > 0 {
                let disk = layout.encode_tail(&self.semantic_coords)?;
                w.write_all(&disk)?;
                written += disk.len();
            }
        } else if !self.semantic_coords.is_empty() {
            w.write_all(&self.semantic_coords)?;
            written += self.semantic_coords.len();
        }

        Ok(written)
    }

    /// Deserialize an entry from bytes. Semantic coordinates come back
    /// full-width regardless of the disk encoding.
    pub fn read_from<R: Read>(r: &mut R, layout: &SemLayout) -> HoronResult<Self> {
        // key
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf2)?;
        let key_len = u16::from_le_bytes(buf2) as usize;
        let mut key_buf = vec![0u8; key_len];
        r.read_exact(&mut key_buf)?;
        let key = String::from_utf8(key_buf)
            .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8 key: {}", e)))?;

        // data
        r.read_exact(&mut buf4)?;
        let data_len = u32::from_le_bytes(buf4) as usize;
        if data_len > MAX_ENTRY_DATA {
            return Err(HoronError::InvalidFormat(format!(
                "entry data length {} exceeds maximum {} — corrupt length field",
                data_len, MAX_ENTRY_DATA
            )));
        }
        let data = crate::format::read_bounded_vec(r, data_len, "entry data")?;

        // metadata
        r.read_exact(&mut buf2)?;
        let meta_count = u16::from_le_bytes(buf2) as usize;
        let mut metadata = Vec::with_capacity(meta_count);
        for _ in 0..meta_count {
            r.read_exact(&mut buf2)?;
            let mk_len = u16::from_le_bytes(buf2) as usize;
            let mut mk_buf = vec![0u8; mk_len];
            r.read_exact(&mut mk_buf)?;
            let mk = String::from_utf8(mk_buf)
                .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8 meta key: {}", e)))?;

            r.read_exact(&mut buf2)?;
            let mv_len = u16::from_le_bytes(buf2) as usize;
            let mut mv_buf = vec![0u8; mv_len];
            r.read_exact(&mut mv_buf)?;
            let mv = String::from_utf8(mv_buf)
                .map_err(|e| HoronError::InvalidFormat(format!("invalid UTF-8 meta value: {}", e)))?;

            metadata.push((mk, mv));
        }

        // semantic coords
        let disk_bytes = layout.disk_bytes();
        let mut semantic_coords = vec![0u8; disk_bytes];
        if disk_bytes > 0 {
            r.read_exact(&mut semantic_coords)?;
            if layout.quantized {
                semantic_coords = layout.decode_tail(&semantic_coords)?;
            }
        }

        Ok(Self { key, data, metadata, semantic_coords })
    }
}

/// Serialize a snapshot section (header + entries).
///
/// `with_crc` (format v2): append a CRC32 of the raw (uncompressed) entry
/// bytes after the data section, so corruption is detected on read rather
/// than silently loading wrong data.
pub fn write_snapshot<W: Write>(
    w: &mut W,
    entries: &[NodeEntry],
    compress: bool,
    with_crc: bool,
    layout: &SemLayout,
) -> HoronResult<()> {
    // Serialize entries to buffer first — needed for byte length and CRC.
    let mut raw = Vec::new();
    for entry in entries {
        entry.write_to(&mut raw, layout)?;
    }

    // An empty snapshot always takes the raw branch: the reader early-returns
    // on node_count == 0 without consuming a compressed_len field, so writing
    // a compressed empty section would misalign the WAL header that follows.
    let compress = compress && !entries.is_empty();

    // snap_byte_len (decompressed size) + snap_node_count
    w.write_all(&(raw.len() as u32).to_le_bytes())?;
    w.write_all(&(entries.len() as u32).to_le_bytes())?;

    if compress {
        let compressed = zstd::bulk::compress(&raw, 3)
            .map_err(|e| HoronError::CompressionError(e.to_string()))?;
        w.write_all(&(compressed.len() as u32).to_le_bytes())?;
        w.write_all(&compressed)?;
    } else {
        w.write_all(&raw)?;
    }

    if with_crc {
        w.write_all(&crc32fast::hash(&raw).to_le_bytes())?;
    }
    Ok(())
}

/// Read a snapshot section, returning all node entries.
///
/// `with_crc` must match the file's format version (v2+ carries a trailing
/// CRC32 of the raw entry bytes). Length fields are bounds-checked so a
/// corrupt file produces a clean error instead of a huge allocation.
pub fn read_snapshot<R: Read>(
    r: &mut R,
    compressed: bool,
    layout: &SemLayout,
    with_crc: bool,
) -> HoronResult<Vec<NodeEntry>> {
    let mut buf4 = [0u8; 4];

    // snap_byte_len (decompressed size)
    r.read_exact(&mut buf4)?;
    let snap_byte_len = u32::from_le_bytes(buf4) as usize;
    if snap_byte_len > MAX_SNAPSHOT_BYTES {
        return Err(HoronError::InvalidFormat(format!(
            "snapshot byte length {} exceeds maximum {} — corrupt length field",
            snap_byte_len, MAX_SNAPSHOT_BYTES
        )));
    }

    // snap_node_count — each serialized entry is at least 8 bytes, so a
    // count that implies more entries than the section can hold is corrupt.
    r.read_exact(&mut buf4)?;
    let snap_node_count = u32::from_le_bytes(buf4) as usize;
    if snap_node_count > snap_byte_len / 8 + 1 {
        return Err(HoronError::InvalidFormat(format!(
            "snapshot node count {} impossible for {} section bytes",
            snap_node_count, snap_byte_len
        )));
    }

    if snap_node_count == 0 {
        if with_crc {
            r.read_exact(&mut buf4)?;
            let stored = u32::from_le_bytes(buf4);
            let computed = crc32fast::hash(&[]);
            if stored != computed {
                return Err(HoronError::ChecksumMismatch {
                    expected: stored,
                    actual: computed,
                    context: "snapshot section (empty)".to_string(),
                });
            }
        }
        return Ok(Vec::new());
    }

    let raw_data = if compressed {
        // Read compressed_len + compressed_data
        r.read_exact(&mut buf4)?;
        let compressed_len = u32::from_le_bytes(buf4) as usize;
        if compressed_len > MAX_SNAPSHOT_BYTES {
            return Err(HoronError::InvalidFormat(format!(
                "snapshot compressed length {} exceeds maximum {} — corrupt length field",
                compressed_len, MAX_SNAPSHOT_BYTES
            )));
        }
        let compressed =
            crate::format::read_bounded_vec(r, compressed_len, "compressed snapshot")?;

        // Bounded streaming decode (shared with WAL blocks); the snapshot
        // header declares the exact decompressed size, so require equality.
        let raw = crate::compression::decompress(
            &compressed,
            crate::format::ALGO_ZSTD,
            snap_byte_len,
        )?;
        if raw.len() != snap_byte_len {
            return Err(HoronError::InvalidFormat(format!(
                "snapshot decompressed to {} bytes, header declared {}",
                raw.len(),
                snap_byte_len
            )));
        }
        raw
    } else {
        crate::format::read_bounded_vec(r, snap_byte_len, "snapshot section")?
    };

    if with_crc {
        r.read_exact(&mut buf4)?;
        let stored = u32::from_le_bytes(buf4);
        let computed = crc32fast::hash(&raw_data);
        if stored != computed {
            return Err(HoronError::ChecksumMismatch {
                expected: stored,
                actual: computed,
                context: "snapshot section".to_string(),
            });
        }
    }

    let mut cursor = Cursor::new(&raw_data);
    let mut entries = Vec::with_capacity(snap_node_count.min(1_000_000));
    for _ in 0..snap_node_count {
        entries.push(NodeEntry::read_from(&mut cursor, layout)?);
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_entry_roundtrip() {
        let entry = NodeEntry {
            key: "/test/path".to_string(),
            data: b"hello world".to_vec(),
            metadata: vec![
                ("author".to_string(), "alice".to_string()),
                ("type".to_string(), "text".to_string()),
            ],
            semantic_coords: vec![],
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf, &SemLayout::plain(0)).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = NodeEntry::read_from(&mut cursor, &SemLayout::plain(0)).unwrap();

        assert_eq!(parsed.key, entry.key);
        assert_eq!(parsed.data, entry.data);
        assert_eq!(parsed.metadata, entry.metadata);
    }

    #[test]
    fn test_node_entry_with_semantic_coords() {
        let coords = vec![0u8; 48]; // 3 semantic dims × 16 bytes
        let entry = NodeEntry {
            key: "/node".to_string(),
            data: b"data".to_vec(),
            metadata: vec![],
            semantic_coords: coords.clone(),
        };

        let mut buf = Vec::new();
        entry.write_to(&mut buf, &SemLayout::plain(0)).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = NodeEntry::read_from(&mut cursor, &SemLayout::plain(3)).unwrap();

        assert_eq!(parsed.semantic_coords, coords);
    }

    #[test]
    fn test_snapshot_roundtrip_uncompressed() {
        let entries = vec![
            NodeEntry {
                key: "/a".to_string(),
                data: b"aaa".to_vec(),
                metadata: vec![],
                semantic_coords: vec![],
            },
            NodeEntry {
                key: "/a/b".to_string(),
                data: b"bbb".to_vec(),
                metadata: vec![("k".to_string(), "v".to_string())],
                semantic_coords: vec![],
            },
        ];

        let mut buf = Vec::new();
        write_snapshot(&mut buf, &entries, false, true, &SemLayout::plain(0)).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = read_snapshot(&mut cursor, false, &SemLayout::plain(0), true).unwrap();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].key, "/a");
        assert_eq!(parsed[1].key, "/a/b");
        assert_eq!(parsed[1].metadata[0], ("k".to_string(), "v".to_string()));
    }

    #[test]
    fn test_snapshot_roundtrip_compressed() {
        let entries: Vec<NodeEntry> = (0..100).map(|i| NodeEntry {
            key: format!("/node_{}", i),
            data: format!("data for node {}", i).into_bytes(),
            metadata: vec![("idx".to_string(), i.to_string())],
            semantic_coords: vec![],
        }).collect();

        let mut buf = Vec::new();
        write_snapshot(&mut buf, &entries, true, true, &SemLayout::plain(0)).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = read_snapshot(&mut cursor, true, &SemLayout::plain(0), true).unwrap();

        assert_eq!(parsed.len(), 100);
        assert_eq!(parsed[42].key, "/node_42");
    }

    #[test]
    fn test_empty_snapshot() {
        let mut buf = Vec::new();
        write_snapshot(&mut buf, &[], false, true, &SemLayout::plain(0)).unwrap();

        let mut cursor = Cursor::new(&buf);
        let parsed = read_snapshot(&mut cursor, false, &SemLayout::plain(0), true).unwrap();
        assert!(parsed.is_empty());
    }
}
