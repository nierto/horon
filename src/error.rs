//! Error types for Horon operations.

use std::fmt;

/// Errors that can occur during Horon operations.
#[derive(Debug)]
pub enum HoronError {
    /// File I/O error.
    Io(std::io::Error),
    /// Invalid .htt file (bad magic, unsupported version, corrupted header).
    InvalidFormat(String),
    /// CRC mismatch on header, snapshot entry, or WAL entry.
    ChecksumMismatch {
        /// CRC32 value recorded in the file.
        expected: u32,
        /// CRC32 value computed over the bytes actually read.
        actual: u32,
        /// Which structure the mismatch was detected in (e.g. "header", "WAL entry").
        context: String,
    },
    /// WAL entry is corrupted (partial write detected during recovery).
    CorruptedWalEntry {
        /// Sequence number of the corrupted entry.
        seq: u32,
        /// Description of what was wrong with the entry.
        message: String,
    },
    /// Unsupported compression algorithm.
    UnsupportedCompression(u8),
    /// Compression/decompression failure.
    CompressionError(String),
    /// engine Store error (passthrough).
    Store(horon_engine::store::StoreError),
    /// Access denied by geometric ACL.
    AccessDenied {
        /// Key of the node the requester attempted to access.
        key: String,
        /// Why access was refused (which band check failed).
        reason: String,
    },
    /// Configuration error.
    Config(String),
    /// The file is already open (advisory-locked) by another process.
    Locked(String),
    /// The requested write cannot be represented in the on-disk format
    /// (e.g. key longer than the u16 length field allows).
    InvalidOperation(String),
}

impl fmt::Display for HoronError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::InvalidFormat(msg) => write!(f, "invalid .htt format: {}", msg),
            Self::InvalidOperation(msg) => write!(f, "invalid operation: {}", msg),
            Self::ChecksumMismatch { expected, actual, context } => {
                write!(f, "CRC mismatch in {}: expected 0x{:08X}, got 0x{:08X}", context, expected, actual)
            }
            Self::CorruptedWalEntry { seq, message } => {
                write!(f, "corrupted WAL entry at seq {}: {}", seq, message)
            }
            Self::UnsupportedCompression(algo) => {
                write!(f, "unsupported compression algorithm: 0x{:02X}", algo)
            }
            Self::CompressionError(msg) => write!(f, "compression error: {}", msg),
            Self::Store(e) => write!(f, "store error: {}", e),
            Self::AccessDenied { key, reason } => {
                write!(f, "access denied for '{}': {}", key, reason)
            }
            Self::Config(msg) => write!(f, "configuration error: {}", msg),
            Self::Locked(path) => {
                write!(f, "file is locked by another process: {}", path)
            }
        }
    }
}

impl std::error::Error for HoronError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Store(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for HoronError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<horon_engine::store::StoreError> for HoronError {
    fn from(e: horon_engine::store::StoreError) -> Self {
        Self::Store(e)
    }
}

/// Convenience alias for `Result` with [`HoronError`] as the error type.
pub type HoronResult<T> = Result<T, HoronError>;
