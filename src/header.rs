//! .htt file header — 32-byte fixed structure.
//!
//! Layout:
//! ```text
//! Offset  Size  Field
//!  0       4    magic          b"HTT\0"
//!  4       1    version        Format version (1–3)
//!  5       1    flags          Bit field
//!  6       1    dimension      Structural dimension
//!  7       1    semantic_dims  Semantic dimension count
//!  8      16    tau            Q64.64 as i128 LE
//! 24       4    node_count     u32 LE
//! 28       4    header_crc     CRC32 of bytes 0..27
//! ```

use crate::error::{HoronError, HoronResult};
use crate::format::*;

/// Parsed file header.
#[derive(Debug, Clone)]
pub struct GeoHeader {
    /// Format version (1–3; see [`crate::format::VERSION`]).
    pub version: u8,
    /// Flags bit field (compression, algorithm, semantic dims, GACL, meaning-addressed).
    pub flags: u8,
    /// Structural dimension of the Poincaré disk embedding.
    pub dimension: u8,
    /// Number of semantic dimensions per node (0 = none).
    pub semantic_dims: u8,
    /// Sarkar embedding scale factor tau, as raw Q64.64 fixed-point bits.
    pub tau_raw: i128,
    /// Number of node entries in the snapshot section.
    pub node_count: u32,
}

impl GeoHeader {
    /// Create a new header with the given parameters.
    pub fn new(dimension: u8, semantic_dims: u8, tau_raw: i128, compressed: bool) -> Self {
        Self::with_gacl(dimension, semantic_dims, tau_raw, compressed, false)
    }

    /// Create a header with optional GACL enforcement.
    pub fn with_gacl(
        dimension: u8,
        semantic_dims: u8,
        tau_raw: i128,
        compressed: bool,
        gacl: bool,
    ) -> Self {
        let mut flags = 0u8;
        if compressed {
            flags |= FLAG_COMPRESSION;
            flags |= FLAG_WAL_COMPRESSED;
            // zstd is algo 0, bits 1-2 = 00 — already zero
        }
        if semantic_dims > 0 {
            flags |= FLAG_SEMANTIC;
        }
        if gacl {
            flags |= FLAG_GACL;
        }
        Self {
            version: VERSION,
            flags,
            dimension,
            semantic_dims,
            tau_raw,
            node_count: 0,
        }
    }

    /// Whether compression is enabled.
    pub fn compression_enabled(&self) -> bool {
        self.flags & FLAG_COMPRESSION != 0
    }

    /// Compression algorithm code (only meaningful if compression enabled).
    pub fn compression_algo(&self) -> u8 {
        (self.flags & FLAG_ALGO_MASK) >> FLAG_ALGO_SHIFT
    }

    /// Whether semantic dimensions are present.
    pub fn has_semantic_dims(&self) -> bool {
        self.flags & FLAG_SEMANTIC != 0
    }

    /// Whether WAL entries are block-compressed.
    pub fn wal_compressed(&self) -> bool {
        self.flags & FLAG_WAL_COMPRESSED != 0
    }

    /// Whether GACL enforcement is enabled for this file.
    pub fn gacl_enabled(&self) -> bool {
        self.flags & FLAG_GACL != 0
    }

    /// Whether semantic tails are stored quantized (format v4).
    pub fn quantized_semantic(&self) -> bool {
        self.flags & FLAG_QUANTIZED_SEMANTIC != 0
    }

    /// Serialize header to 32 bytes.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];

        // Magic
        buf[0..4].copy_from_slice(&MAGIC);

        // Version + flags + dimensions
        buf[4] = self.version;
        buf[5] = self.flags;
        buf[6] = self.dimension;
        buf[7] = self.semantic_dims;

        // Tau (i128 LE)
        buf[8..24].copy_from_slice(&self.tau_raw.to_le_bytes());

        // Node count (u32 LE)
        buf[24..28].copy_from_slice(&self.node_count.to_le_bytes());

        // CRC32 of bytes 0..27
        let crc = crc32fast::hash(&buf[0..28]);
        buf[28..32].copy_from_slice(&crc.to_le_bytes());

        buf
    }

    /// Parse header from 32 bytes.
    pub fn from_bytes(buf: &[u8; HEADER_SIZE]) -> HoronResult<Self> {
        // Verify magic
        if buf[0..4] != MAGIC {
            return Err(HoronError::InvalidFormat(
                format!("bad magic: {:?}", &buf[0..4])
            ));
        }

        // Verify CRC
        let stored_crc = u32::from_le_bytes([buf[28], buf[29], buf[30], buf[31]]);
        let computed_crc = crc32fast::hash(&buf[0..28]);
        if stored_crc != computed_crc {
            return Err(HoronError::ChecksumMismatch {
                expected: stored_crc,
                actual: computed_crc,
                context: "file header".to_string(),
            });
        }

        let version = buf[4];
        if !(MIN_SUPPORTED_VERSION..=MAX_SUPPORTED_VERSION).contains(&version) {
            return Err(HoronError::InvalidFormat(
                format!("unsupported version: {} (supported: {}..={})",
                    version, MIN_SUPPORTED_VERSION, MAX_SUPPORTED_VERSION)
            ));
        }

        let flags = buf[5];
        let dimension = buf[6];
        let semantic_dims = buf[7];

        if flags & FLAG_MEANING_ADDRESSED != 0 && version < VERSION_MEANING_ADDRESSED {
            return Err(HoronError::InvalidFormat(
                "meaning-addressed flag requires format v3".to_string(),
            ));
        }

        if flags & FLAG_QUANTIZED_SEMANTIC != 0 && version < VERSION_QUANTIZED {
            return Err(HoronError::InvalidFormat(
                "quantized-semantic flag requires format v4".to_string(),
            ));
        }

        let tau_raw = i128::from_le_bytes(buf[8..24].try_into().unwrap());
        let node_count = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);

        // Validate compression algo if compression enabled
        if flags & FLAG_COMPRESSION != 0 {
            let algo = (flags & FLAG_ALGO_MASK) >> FLAG_ALGO_SHIFT;
            if algo > ALGO_LZ4 {
                return Err(HoronError::UnsupportedCompression(algo));
            }
        }

        Ok(Self {
            version,
            flags,
            dimension,
            semantic_dims,
            tau_raw,
            node_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = GeoHeader::new(4, 12, 0x0001_0000_0000_0000, true);
        let bytes = header.to_bytes();
        let parsed = GeoHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, VERSION);
        assert_eq!(parsed.dimension, 4);
        assert_eq!(parsed.semantic_dims, 12);
        assert_eq!(parsed.tau_raw, 0x0001_0000_0000_0000);
        assert!(parsed.compression_enabled());
        assert_eq!(parsed.compression_algo(), ALGO_ZSTD);
        assert!(parsed.has_semantic_dims());
        assert_eq!(parsed.node_count, 0);
    }

    #[test]
    fn test_header_no_compression() {
        let header = GeoHeader::new(4, 0, 0x0001_0000_0000_0000, false);
        let bytes = header.to_bytes();
        let parsed = GeoHeader::from_bytes(&bytes).unwrap();

        assert!(!parsed.compression_enabled());
        assert!(!parsed.has_semantic_dims());
    }

    #[test]
    fn test_header_bad_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"NOPE");
        let err = GeoHeader::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, HoronError::InvalidFormat(_)));
    }

    #[test]
    fn test_header_bad_crc() {
        let header = GeoHeader::new(4, 0, 0, false);

        // Corrupt node_count (byte 24 — inside the CRC-covered 0..28 region)
        let mut tampered = header.to_bytes();
        tampered[24] = 0xFF;
        let err = GeoHeader::from_bytes(&tampered).unwrap_err();
        assert!(matches!(err, HoronError::ChecksumMismatch { .. }));

        // Corrupt dimension (byte 6)
        let mut tampered = header.to_bytes();
        tampered[6] = 99;
        let err = GeoHeader::from_bytes(&tampered).unwrap_err();
        assert!(matches!(err, HoronError::ChecksumMismatch { .. }));
    }

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<[u8; HEADER_SIZE]>(), 32);
    }
}
