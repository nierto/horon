//! Compression utilities for .htt format.
//!
//! Supports zstd (default), with the algorithm selectable via the flags byte.

use crate::error::{HoronError, HoronResult};
use crate::format::*;

/// Compress a block of data using the specified algorithm.
pub fn compress(data: &[u8], algo: u8) -> HoronResult<Vec<u8>> {
    match algo {
        ALGO_ZSTD => {
            zstd::bulk::compress(data, 3)
                .map_err(|e| HoronError::CompressionError(e.to_string()))
        }
        _ => Err(HoronError::UnsupportedCompression(algo)),
    }
}

/// Decompress a block of data using the specified algorithm.
/// `max_size` caps the decompressed size; the output buffer grows with the
/// actual decompressed bytes rather than pre-allocating `max_size`, so a
/// crafted tiny block cannot force a large allocation up front.
pub fn decompress(data: &[u8], algo: u8, max_size: usize) -> HoronResult<Vec<u8>> {
    use std::io::Read;
    match algo {
        ALGO_ZSTD => {
            let mut out = Vec::new();
            let decoder = zstd::stream::read::Decoder::new(data)
                .map_err(|e| HoronError::CompressionError(e.to_string()))?;
            decoder
                .take(max_size as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|e| HoronError::CompressionError(e.to_string()))?;
            if out.len() > max_size {
                return Err(HoronError::CompressionError(format!(
                    "decompressed size exceeds maximum {}",
                    max_size
                )));
            }
            Ok(out)
        }
        _ => Err(HoronError::UnsupportedCompression(algo)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zstd_roundtrip() {
        let data = b"hello world, this is a test of zstd compression in Horon";
        let compressed = compress(data, ALGO_ZSTD).unwrap();
        let decompressed = decompress(&compressed, ALGO_ZSTD, data.len()).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_unsupported_algo() {
        let result = compress(b"data", 3);
        assert!(matches!(result, Err(HoronError::UnsupportedCompression(3))));
    }
}
