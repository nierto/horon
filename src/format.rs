//! .htt binary format constants and shared types.

/// Magic bytes: "HTT\0" (0x48 0x54 0x54 0x00) — Hyperbolic Tree Tensors.
pub const MAGIC: [u8; 4] = [0x48, 0x54, 0x54, 0x00];

/// Default format version written for ordinary files.
///
/// v1: original layout. v2 (2026-07): snapshot section carries a trailing
/// CRC32 of its raw (uncompressed) entry bytes. v2 readers accept v1 files;
/// compaction upgrades v1 files to v2. v3 (2026-07): meaning-addressed
/// layout — a global-normalization-bounds section follows the file header,
/// and snapshot entries are stored in pure Hilbert order (see
/// FLAG_MEANING_ADDRESSED). v3 is only written when that flag is set.
/// v4 (2026-07): quantized semantic tails (see FLAG_QUANTIZED_SEMANTIC);
/// only written when that flag is set.
pub const VERSION: u8 = 2;

/// Version written for meaning-addressed files (FLAG_MEANING_ADDRESSED).
pub const VERSION_MEANING_ADDRESSED: u8 = 3;

/// Version written for quantized-semantic files (FLAG_QUANTIZED_SEMANTIC).
/// Takes precedence over v3 when both flags are set.
pub const VERSION_QUANTIZED: u8 = 4;

/// Newest format version this build can read.
pub const MAX_SUPPORTED_VERSION: u8 = 4;

/// Oldest format version this build can read.
pub const MIN_SUPPORTED_VERSION: u8 = 1;

/// Upper bound for the snapshot section's raw or compressed byte length.
/// A corrupt length field must produce a clean error, not a huge allocation.
pub const MAX_SNAPSHOT_BYTES: usize = 1024 * 1024 * 1024;

/// Upper bound for a single entry's data payload length (snapshot or WAL).
pub const MAX_ENTRY_DATA: usize = 256 * 1024 * 1024;

/// File header size in bytes.
pub const HEADER_SIZE: usize = 32;

/// Snapshot header size in bytes.
pub const SNAP_HEADER_SIZE: usize = 8;

/// WAL header size in bytes.
pub const WAL_HEADER_SIZE: usize = 8;

/// Minimum file size: header + snap header + wal header.
pub const MIN_FILE_SIZE: usize = HEADER_SIZE + SNAP_HEADER_SIZE + 4 + WAL_HEADER_SIZE;
// v2: the snapshot section carries a trailing 4-byte CRC32, even when empty.

/// Maximum entries per compressed WAL block.
pub const WAL_BLOCK_SIZE: usize = 64;

// --- Flags byte (offset 5) ---

/// Flag bit 0: compression enabled.
pub const FLAG_COMPRESSION: u8 = 0x01;

/// Flag bits 1-2: compression algorithm (masked after shifting).
pub const FLAG_ALGO_MASK: u8 = 0x06;
/// Right-shift that moves the FLAG_ALGO_MASK bits down to a 2-bit algorithm code.
pub const FLAG_ALGO_SHIFT: u8 = 1;

/// Flag bit 3: has semantic dimensions.
pub const FLAG_SEMANTIC: u8 = 0x08;

/// Flag bit 4: WAL entries are block-compressed (implies bit 0).
pub const FLAG_WAL_COMPRESSED: u8 = 0x10;

/// Flag bit 6: meaning-addressed layout (format v3). When set, a bounds
/// section (user_dims × 16 bytes: per-dim min/max f64 LE) follows the file
/// header, and snapshot entries are stored in pure global-Hilbert order —
/// a node's position on disk is a function of its semantic coordinates.
/// Full-geometry readers must depth-sort entries in memory before Sarkar
/// replay. Requires an uncompressed snapshot.
pub const FLAG_MEANING_ADDRESSED: u8 = 0x40;

/// Flag bit 5: GACL enforcement enabled. When set, reads/writes check
/// semantic dimensions 0–11 against caller credentials. When clear,
/// all nodes are accessible regardless of their GACL bands.
pub const FLAG_GACL: u8 = 0x20;

/// Flag bit 7: quantized semantic tails (format v4 — see
/// `docs/QUANTIZED_SEMANTIC.md`). When set, each entry's semantic tail
/// stores user dims (16+) as TQ1.9 — 2 bytes/dim (i16 LE, value × 3⁹,
/// range ±29524/19683) — and the reserved GACL region (dims 0–15) is
/// present (full-width Q64.64) only when FLAG_GACL is also set, elided
/// otherwise. This is the LAST free flag bit; any further format feature
/// needs a v5 extended header.
pub const FLAG_QUANTIZED_SEMANTIC: u8 = 0x80;

// --- Compression algorithm codes (2 bits) ---

/// zstd compression (the default when compression is enabled).
pub const ALGO_ZSTD: u8 = 0;
/// zlib (DEFLATE) compression.
pub const ALGO_ZLIB: u8 = 1;
/// LZ4 compression.
pub const ALGO_LZ4: u8 = 2;

// --- WAL operation codes ---

/// Create node (error if exists).
pub const OP_INSERT: u8 = 0x01;

/// Replace data + metadata on existing node.
pub const OP_UPDATE: u8 = 0x02;

/// Remove node.
pub const OP_DELETE: u8 = 0x03;

/// Set one metadata key/value on existing node.
pub const OP_SET_META: u8 = 0x04;

/// Update semantic coordinates on existing node.
pub const OP_SET_SEMANTIC: u8 = 0x05;

/// Epoch marker (temporal epochs). No node state change: replay
/// advances the in-memory epoch counter. Payload: epoch_id (u64 LE) +
/// flags (u8). Files that never call `seal_epoch()` never contain this op,
/// so they remain readable by tools that predate epochs.
pub const OP_EPOCH: u8 = 0x06;

/// OP_EPOCH flag bit 0: the sealed state is speculative (projected /
/// what-if), not recorded history. The flag labels — it does not isolate:
/// speculative work belongs in a copy of the file, and this bit ensures a
/// labeled span can never silently pass as recorded fact.
pub const EPOCH_FLAG_SPECULATIVE: u8 = 0x01;

/// OP_EPOCH flag bit 1: a compaction re-stamp. Compaction seeds the fresh
/// WAL with a marker restating the current epoch so the counter survives
/// truncation; re-stamps carry no new seal moment and are ignored by
/// temporal sampling (the original seal, at its original sequence, is the
/// authoritative sample point).
pub const EPOCH_FLAG_RESTAMP: u8 = 0x02;

// --- History sidecar segments (temporal epochs) ---

/// Magic bytes for history sidecar segments: "HTTH".
pub const HIST_MAGIC: [u8; 4] = [0x48, 0x54, 0x54, 0x48];

/// History segment format version (plain full-width entries).
pub const HIST_VERSION: u8 = 1;

/// History segment version for quantized-semantic files: the byte at
/// offset 6 (formerly reserved) records the tail layout — bit 0 quantized,
/// bit 1 GACL — so segments stay self-describing. v1 readers reject v2
/// segments loudly rather than misparsing the narrower tails.
pub const HIST_VERSION_QUANTIZED: u8 = 2;

/// History segment layout-flags (offset 6, v2 segments): quantized tails.
pub const HIST_FLAG_QUANTIZED: u8 = 0x01;

/// History segment layout-flags (offset 6, v2 segments): GACL reserved
/// region present at full width.
pub const HIST_FLAG_GACL: u8 = 0x02;

/// History segment header size: magic(4) + version(1) + semantic_dims(1) +
/// reserved(2) + first_seq(4) + end_seq(4) + raw_len(4) + comp_len(4).
pub const HIST_HEADER_SIZE: usize = 24;

/// Upper bound for a history segment's raw (uncompressed) byte length —
/// a corrupt length field must produce a clean error, not a huge allocation.
pub const MAX_HIST_BYTES: usize = 1024 * 1024 * 1024;

// --- Reserved semantic dimension indices ---

/// Read access band (lo/hi).
pub const DIM_READ_LO: usize = 0;
/// Upper bound of the read access band.
pub const DIM_READ_HI: usize = 1;

/// Write access band (lo/hi).
pub const DIM_WRITE_LO: usize = 2;
/// Upper bound of the write access band.
pub const DIM_WRITE_HI: usize = 3;

/// Execute/invoke access band (lo/hi).
pub const DIM_EXEC_LO: usize = 4;
/// Upper bound of the execute/invoke access band.
pub const DIM_EXEC_HI: usize = 5;

/// Organizational domain band (lo/hi).
pub const DIM_DOMAIN_LO: usize = 6;
/// Upper bound of the organizational domain band.
pub const DIM_DOMAIN_HI: usize = 7;

/// Data classification band (lo/hi).
pub const DIM_CLASS_LO: usize = 8;
/// Upper bound of the data classification band.
pub const DIM_CLASS_HI: usize = 9;

/// User/service identity band (lo/hi).
pub const DIM_IDENTITY_LO: usize = 10;
/// Upper bound of the user/service identity band.
pub const DIM_IDENTITY_HI: usize = 11;

/// First user-defined semantic dimension index.
pub const DIM_USER_DEFINED_START: usize = 16;

/// Read exactly `len` bytes into a `Vec` without trusting `len` for the
/// allocation: the buffer grows only as bytes actually arrive from the
/// reader, so a corrupt length field in a small file costs a clean error,
/// not a `len`-sized allocation made before the read can fail.
pub(crate) fn read_bounded_vec<R: std::io::Read>(
    r: &mut R,
    len: usize,
    what: &str,
) -> crate::error::HoronResult<Vec<u8>> {
    use std::io::Read;
    // Capacity hint capped at 64 KiB: one allocation for the common case,
    // while a hostile length field can still only force a tiny upfront alloc.
    let mut buf = Vec::with_capacity(len.min(64 * 1024));
    r.take(len as u64).read_to_end(&mut buf)?;
    if buf.len() != len {
        return Err(crate::error::HoronError::InvalidFormat(format!(
            "truncated {}: expected {} bytes, got {}",
            what, len, buf.len()
        )));
    }
    Ok(buf)
}
