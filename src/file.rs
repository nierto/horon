//! Horon — persistent engine Store backed by a .htt file.
//!
//! Thread-safe: all methods take `&self`. Interior mutability is provided by
//! `Mutex<WalWriter>` for the WAL and the engine's `DashMap` + `StripedLock` for
//! the Store. `Horon` is `Send + Sync` and can be shared via `Arc<Horon>`.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::ops::Deref;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::thread::JoinHandle;
use g_math::fixed_point::FixedPoint;
use std::time::{Duration, Instant};

use horon_engine::{SemanticOutlier, Store, StoreConfig};

use crate::error::{HoronError, HoronResult};
use crate::format::*;
use crate::gacl::{Credentials, NodeAccessBands};
use crate::header::GeoHeader;
use crate::hilbert::HilbertMapper;
use crate::partial::SnapView;
use crate::quant::SemLayout;
use crate::snapshot::{self, NodeEntry};
use crate::wal::{self, WalEntry, WalPayload};

// Compile-time assertion: Horon is Send + Sync (required for Arc<Horon>).
const _: () = {
    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}
    fn _check() {
        _assert_send::<Horon>();
        _assert_sync::<Horon>();
    }
};

/// Durability mode for WAL writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurabilityMode {
    /// Flush + fsync after EVERY append, overriding any batch configuration
    /// (safest, ~1–5ms per write). For financial / audit-critical data.
    Fsync,
    /// fsync when a batch flushes (batch size or interval reached; every
    /// append when batching is off). The default — survives power loss
    /// except for the entries still pending in the current batch.
    ///
    /// Note that the default `wal_batch_size` of 0 means "flush every append",
    /// so under the default config this fsyncs on every write. That is correct
    /// for a live transactional store but is the wrong choice for a bulk load:
    /// each node is ≥1 append (a `put` plus each `set_semantic`/`set_meta`), so
    /// loading N nodes costs ≥N fsyncs (~ms each) — often minutes for 100k+
    /// nodes, dwarfing the actual in-memory work. Use `Relaxed` (below) or a
    /// large `wal_batch_size` for bulk imports.
    Batched,
    /// Never fsync — relies on the OS page cache. Survives process kills
    /// but NOT power loss.
    ///
    /// The right choice for bulk imports and rebuild pipelines that finish with
    /// a `compact()`: compaction writes and fsyncs the snapshot, so the loaded
    /// data is made durable once at the end rather than fsynced per append.
    /// This is what makes a large meaning-addressed build fast (see
    /// `examples/scan_counts.rs`).
    Relaxed,
}

impl Default for DurabilityMode {
    fn default() -> Self {
        Self::Batched
    }
}

/// WAL history retention across compaction (temporal epochs — see
/// `docs/TEMPORAL_EPOCHS.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistoryRetention {
    /// Compaction truncates the WAL (the default). Byte-identical to
    /// pre-epochs behavior; no sidecars, no cost. The simple htt pays nothing.
    #[default]
    Off,
    /// Compaction archives the pre-fence WAL into a zstd-compressed sidecar
    /// segment (`<file>.h000001`, `<file>.h000002`, …) before truncating.
    /// The live read/write/recovery paths are untouched — the cost is cold
    /// disk, O(coordinate changes). Deleting the sidecars degrades the file
    /// to a plain htt that has simply forgotten its past. Read the history
    /// with [`crate::history::HoronHistory`].
    Archive,
}

/// Result of a WAL catch-up read (see `Horon::wal_entries_since`).
#[derive(Debug)]
pub enum WalTail {
    /// Committed entries with `seq >= from_seq`, in sequence order.
    Entries(Vec<WalEntry>),
    /// Compaction folded `from_seq` into the snapshot — the replica must
    /// bootstrap from a snapshot/file copy, then tail from `base_seq`.
    SnapshotRequired {
        /// First WAL sequence still present after compaction — the replica
        /// resumes tailing from here once it has bootstrapped.
        base_seq: u32,
    },
}

/// Configuration for Horon.
pub struct HoronConfig {
    /// Structural dimension for the Poincaré disk (default: 4).
    pub dimension: u8,
    /// Number of semantic dimensions (default: 16 — 12 access + 4 reserved).
    pub semantic_dims: u8,
    /// Enable zstd compression (default: true).
    pub compression: bool,
    /// Auto-compact when WAL exceeds this many entries (0 = disabled).
    pub auto_compact_threshold: u32,
    /// WAL batch size — flush after this many entries (0 = immediate, default).
    pub wal_batch_size: u32,
    /// WAL flush interval in milliseconds (0 = no time-based flush).
    ///
    /// The interval is checked **only when a new entry is appended** — there
    /// is no background timer thread. So if writes stop while entries are
    /// pending, they are not flushed by wall-clock time alone; they are
    /// flushed by the next append, an explicit [`flush`](Horon::flush) /
    /// [`compact`](Horon::compact), or on drop. Use `DurabilityMode::Fsync`
    /// (or `wal_batch_size = 1`) if every write must be durable immediately.
    pub wal_flush_interval_ms: u64,
    /// Durability mode for WAL writes.
    pub durability: DurabilityMode,
    /// Enable GACL enforcement (default: false).
    /// When true, reads/writes check semantic dimensions 0–11 against credentials.
    /// Has no effect unless credentials are also set via `set_credentials()`.
    pub gacl: bool,
    /// Fail closed when GACL is enabled but no credentials are set (default:
    /// false = fail open).
    ///
    /// With the default fail-open behavior, turning on `gacl` but forgetting
    /// to call `set_credentials()` leaves every node accessible — the checks
    /// short-circuit to "allow" when credentials are absent. Set this to
    /// `true` to instead deny all access (reads, writes, and spatial results)
    /// until credentials are supplied, so a missing `set_credentials()` fails
    /// safe rather than wide open. Has no effect unless `gacl` is also set.
    /// (GACL remains cooperative query-scoping, not a security boundary — see
    /// the module docs — so this hardens the default, it does not make GACL an
    /// enforcement barrier against an adversary with raw file access.)
    pub gacl_fail_closed: bool,
    /// Skip geometric embedding on load (default: false).
    ///
    /// When true, snapshot and WAL replay use `put_data_only()` which skips
    /// Sarkar embedding, VP-tree, and power diagram construction. Opens in
    /// milliseconds instead of minutes for large files (5000+ nodes).
    ///
    /// Semantic queries (`nearest_semantic`, `neighbors_semantic`,
    /// `get_semantic`) work normally. Spatial queries (`nearest`,
    /// `neighbors`) will not return results for data-only nodes.
    pub lazy_geometry: bool,
    /// Serve reads from a memory-mapped snapshot instead of materializing
    /// every entry into RAM at open (default: false).
    ///
    /// Payloads stay on disk until a read touches them; memory holds only a
    /// key→offset index plus written-since-open data. Implies lazy geometry
    /// (no hyperbolic spatial queries). Requires an uncompressed snapshot
    /// and is currently incompatible with GACL enforcement.
    pub partial_reads: bool,
    /// Create the file with the meaning-addressed layout (format v3,
    /// default: false; file-creation-time flag, like `gacl`).
    ///
    /// Snapshot entries are stored in pure global-Hilbert order: a node's
    /// position on disk becomes a function of its user semantic coordinates
    /// (dims 16+), normalized by `semantic_bounds`. With `partial_reads`,
    /// semantic queries read only the byte-neighborhood of the query point.
    /// Requires `compression: false` and `semantic_dims > 16`.
    pub meaning_addressed: bool,
    /// Global normalization bounds (min, max) applied to every user semantic
    /// dimension for meaning-addressed placement. Must be stable for the
    /// file's lifetime — they define the address space. Default (0.0, 1.0).
    /// Note: `f64` here is dictated by the format, not by the API — the v3
    /// bounds section stores these as f64 LE pairs (see `docs/HTT_FORMAT.md`
    /// §1). Everything else in the public surface takes and returns
    /// `FixedPoint`; changing this one would be a format revision.
    pub semantic_bounds: (f64, f64),
    /// WAL history retention across compaction (default: `Off` —
    /// byte-identical to pre-epochs behavior). See [`HistoryRetention`].
    pub history_retention: HistoryRetention,
    /// Allow APPROXIMATE semantic k-NN in `partial_reads` mode (default:
    /// false — results are exact).
    ///
    /// Meaning-addressed files can answer `nearest_semantic` from a window
    /// around the query's Hilbert address instead of scanning the snapshot.
    /// That is much faster and **not exact**: a space-filling curve has
    /// seams, so the window can miss true neighbours. Measured recall@10 on
    /// uniform 8-dim data is ~44%, and the shortfall is silent.
    ///
    /// Exact is the default because a query that quietly returns the wrong
    /// answer cannot be reasoned about. Enable this only where approximate
    /// ranking is acceptable and you have measured the recall your data
    /// actually gives.
    pub approximate_semantic: bool,
    /// Create the file with quantized semantic tails (format v4 —
    /// `docs/QUANTIZED_SEMANTIC.md`; file-creation-time flag, like `gacl`).
    ///
    /// User dims (16+) are stored as TQ1.9 (2 bytes/dim, ~4.3 significant
    /// digits, range ±1.49987) and the reserved GACL region is elided
    /// unless `gacl` is also set (then it stays full-width — access bands
    /// remain bit-exact). Distances over user dims become ranking-grade
    /// approximate; determinism is fully preserved (fixed integer
    /// quantization, write-through canonicalization). `set_semantic`
    /// rejects out-of-range user dims and (without GACL) nonzero reserved
    /// dims. Requires `semantic_dims > 16`.
    pub quantized_semantic: bool,
}

impl Default for HoronConfig {
    fn default() -> Self {
        Self {
            dimension: 4,
            semantic_dims: 16,
            compression: true,
            auto_compact_threshold: 10_000,
            wal_batch_size: 0,
            wal_flush_interval_ms: 0,
            durability: DurabilityMode::Batched,
            gacl: false,
            gacl_fail_closed: false,
            lazy_geometry: false,
            partial_reads: false,
            meaning_addressed: false,
            semantic_bounds: (0.0, 1.0),
            history_retention: HistoryRetention::Off,
            approximate_semantic: false,
            quantized_semantic: false,
        }
    }
}

/// State for `partial_reads` mode: the mmap-backed snapshot view, deletes of
/// snapshot-resident keys (tombstones), and query instrumentation.
struct PartialState {
    /// The mmap + structural index. Swapped wholesale after compaction.
    view: RwLock<SnapView>,
    /// Snapshot-resident keys deleted since open/compaction. Keys written
    /// since open live in the Store overlay and shadow the snapshot.
    tombstones: RwLock<HashSet<String>>,
    /// Entries examined by the most recent semantic query — the
    /// "how many bytes did meaning-addressing save us" instrument.
    scanned: AtomicUsize,
}

/// WAL writer state — all mutable WAL state lives here behind `Mutex<WalWriter>`.
///
/// Lock hierarchy (no nesting, no deadlock):
///   engine internal locks → all released → `Mutex<WalWriter>` → released
struct WalWriter {
    file: File,
    next_seq: u32,
    /// Number of entries flushed to disk since last compaction.
    entry_count: u32,
    /// Byte offset where WAL entry data starts (after the 8-byte WAL header).
    wal_data_offset: u64,
    // --- Batch state ---
    pending_serialized: Vec<Vec<u8>>,
    pending_count: u32,
    batch_size: u32,
    flush_interval: Duration,
    last_flush: Instant,
    durability: DurabilityMode,
    // --- Block compression ---
    wal_compressed: bool,
    compression_algo: u8,
    /// Semantic-tail layout (quantization) — needed to serialize entries.
    layout: SemLayout,
    // --- Replication ---
    /// Entries pending flush, retained for subscriber fan-out. Only
    /// populated while subscribers exist (no overhead otherwise).
    pending_entries: Vec<WalEntry>,
    /// Live WAL subscribers. Entries are delivered AFTER fsync — a replica
    /// never runs ahead of the primary's durability. Dropped receivers are
    /// pruned on the next flush.
    subscribers: Vec<std::sync::mpsc::Sender<WalEntry>>,
}

impl WalWriter {
    /// Append a WAL entry. Flushes to disk when batch conditions are met.
    fn append(&mut self, entry: &WalEntry) -> HoronResult<()> {
        // The on-disk sequence number is a u32. Fail loudly before it would
        // wrap: a wrapped seq (back to 0) silently breaks the monotonic-order
        // assumption that the compaction fence and WAL replication both rely
        // on. At this ceiling the file must be rebuilt (copy its logical state
        // into a fresh file) to reset the sequence. Reaching it takes ~4
        // billion writes between compactions' seq continuity.
        if self.next_seq == u32::MAX {
            return Err(HoronError::InvalidOperation(
                "WAL sequence space exhausted (u32 seq at its maximum); \
                 rebuild the file to reset the sequence".to_string(),
            ));
        }
        let mut buf = Vec::new();
        entry.write_to(&mut buf, &self.layout)?;
        self.pending_serialized.push(buf);
        if !self.subscribers.is_empty() {
            self.pending_entries.push(entry.clone());
        }
        self.pending_count += 1;
        self.next_seq += 1;

        // Fsync mode overrides batching: every append is flushed and synced
        // immediately, regardless of batch size or interval configuration.
        let should_flush = self.durability == DurabilityMode::Fsync
            || self.batch_size == 0
            || self.pending_count >= self.batch_size
            || (self.flush_interval.as_millis() > 0
                && self.last_flush.elapsed() >= self.flush_interval);

        if should_flush {
            self.flush_pending()?;
        }

        Ok(())
    }

    /// Flush all pending entries to disk and update the on-disk WAL header.
    fn flush_pending(&mut self) -> HoronResult<()> {
        if self.pending_serialized.is_empty() {
            return Ok(());
        }

        if self.wal_compressed {
            for chunk in self.pending_serialized.chunks(WAL_BLOCK_SIZE) {
                let mut block_bytes = Vec::new();
                for entry_bytes in chunk {
                    block_bytes.extend_from_slice(entry_bytes);
                }
                wal::write_wal_block(
                    &mut self.file,
                    &block_bytes,
                    chunk.len() as u16,
                    self.compression_algo,
                )?;
            }
        } else {
            for entry_bytes in &self.pending_serialized {
                self.file.write_all(entry_bytes)?;
            }
        }
        self.entry_count += self.pending_serialized.len() as u32;

        // Update WAL header on disk (seek back, write, seek to end)
        let current_pos = self.file.stream_position()?;
        self.file.seek(SeekFrom::Start(self.wal_data_offset - WAL_HEADER_SIZE as u64))?;
        let base_seq = self.next_seq - self.entry_count;
        wal::write_wal_header(&mut self.file, self.entry_count, base_seq)?;
        self.file.seek(SeekFrom::Start(current_pos))?;

        match self.durability {
            DurabilityMode::Fsync | DurabilityMode::Batched => {
                self.file.sync_data()?;
            }
            DurabilityMode::Relaxed => {}
        }

        self.pending_serialized.clear();
        self.pending_count = 0;
        self.last_flush = Instant::now();

        // Fan out to subscribers — strictly after the durability point
        // above, so a replica can never hold an entry the primary might
        // lose. Dead receivers are pruned.
        if !self.subscribers.is_empty() {
            for entry in self.pending_entries.drain(..) {
                self.subscribers.retain(|tx| tx.send(entry.clone()).is_ok());
            }
        } else {
            self.pending_entries.clear();
        }

        Ok(())
    }

    /// Total entries (on-disk + pending).
    fn total_entries(&self) -> u32 {
        self.entry_count + self.pending_serialized.len() as u32
    }
}

/// A persistent engine Store backed by a `.htt` file.
///
/// Thread-safe: all methods take `&self`. `Horon` is a cheap handle over
/// an internally reference-counted core, so it can be shared directly or via
/// `Arc<Horon>` — both work.
///
/// - **Reads**: lock-free (in-memory via the engine's DashMap)
/// - **Writes**: parallel at Store level (DashMap + StripedLock<64>),
///   serialized at WAL level (`Mutex<WalWriter>`)
/// - **Compaction**: at most one concurrent compaction (`AtomicBool` gate);
///   auto-compaction runs on a background thread, never on the writer
pub struct Horon {
    core: Arc<HoronCore>,
}

impl Deref for Horon {
    type Target = HoronCore;
    fn deref(&self) -> &HoronCore {
        &self.core
    }
}

impl Horon {
    /// Open an existing .htt file or create a new one.
    pub fn open<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Self::open_with_config(path, HoronConfig::default())
    }

    /// Open with custom configuration.
    ///
    /// For existing files, format settings (dimension, semantic_dims, compression)
    /// are read from the header. Runtime settings (auto_compact_threshold,
    /// wal_batch_size, durability) are taken from `config`.
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: HoronConfig,
    ) -> HoronResult<Self> {
        let core = Arc::new(HoronCore::open_core(path.as_ref(), config)?);
        // The core keeps a Weak to itself so writer threads can hand a strong
        // reference to background compaction without an outer Arc<Horon>.
        let _ = core.self_weak.set(Arc::downgrade(&core));
        Ok(Self { core })
    }

    /// Compute the Euclidean distance between two raw semantic coordinate
    /// vectors across a dimensional slice, without touching any store.
    ///
    /// Lives on `Horon` (not the deref target) because it takes no
    /// receiver: it must be callable as `Horon::semantic_distance`.
    pub fn semantic_distance(
        coords_a: &[u8],
        coords_b: &[u8],
        dim_range: std::ops::Range<usize>,
    ) -> FixedPoint {
        Store::semantic_distance(coords_a, coords_b, dim_range)
    }
}

/// The shared state behind [`Horon`]. All operations live here; `Horon`
/// derefs to this type, so call everything through `Horon`.
#[doc(hidden)]
pub struct HoronCore {
    store: Store,
    path: PathBuf,
    header: GeoHeader,
    wal: Mutex<WalWriter>,
    config: HoronConfig,
    compacting: AtomicBool,
    /// GACL credentials for the current session. When `Some` AND the header
    /// has `FLAG_GACL` set, all reads/writes check access bands in dims 0–11.
    /// When `None` or FLAG_GACL is unset, all nodes are accessible.
    credentials: RwLock<Option<Credentials>>,
    /// Most recent compaction failure, if any. Auto-compaction runs behind
    /// write calls and must not fail them, so its errors are recorded here
    /// (and logged) instead of being silently dropped.
    last_compaction_error: Mutex<Option<String>>,
    /// Weak self-reference, set once at construction. Lets `&self` methods
    /// hand a strong reference to background compaction threads.
    self_weak: OnceLock<Weak<HoronCore>>,
    /// `Some` when opened with `partial_reads` — the mmap view + overlay
    /// bookkeeping. `None` = classic full-materialization mode.
    partial: Option<PartialState>,
    /// `Some(min, max)` when the file is meaning-addressed (format v3):
    /// the global normalization bounds read from / written to the bounds
    /// section. Immutable for the file's lifetime.
    ma_bounds: Option<(f64, f64)>,
    /// Current epoch counter. 0 = no epoch ever sealed. Restored from
    /// WAL replay (`OP_EPOCH` entries) at open; survives compaction via a
    /// re-stamped marker in the fresh WAL.
    current_epoch: AtomicU64,
}

const MAX_HILBERT_DIMS: usize = 8;
const HILBERT_BITS: u32 = 12;

/// Number of addressed user dims for a given semantic_dims count.
fn addressed_dims(semantic_dims: usize) -> usize {
    semantic_dims
        .saturating_sub(DIM_USER_DEFINED_START)
        .min(MAX_HILBERT_DIMS)
}

/// Decode the addressed user dims (16+, capped at 8) of a raw semantic
/// vector into f64 values.
fn decode_user_dims(sem: &[u8], semantic_dims: usize) -> Vec<FixedPoint> {
    (0..addressed_dims(semantic_dims))
        .map(|d| {
            let start = (DIM_USER_DEFINED_START + d) * 16;
            let end = start + 16;
            if sem.len() >= end {
                // Straight from the stored Q64.64 raw. The old path divided
                // by 2^64 into an f64 and lost roughly half the bits, so two
                // distinct stored coordinates could collapse onto the same
                // address.
                FixedPoint::from_raw(i128::from_le_bytes(sem[start..end].try_into().unwrap()))
            } else {
                FixedPoint::from_int(0)
            }
        })
        .collect()
}

/// Map raw user-dim values to a Hilbert address under global bounds.
fn hilbert_from_values(vals: &[FixedPoint], bounds: (f64, f64)) -> u128 {
    if vals.is_empty() {
        return 0;
    }
    // The bounds are f64 because the v3 bounds section stores them that way
    // (see docs/HTT_FORMAT.md §1). They are converted once, here, and every
    // subsequent operation is fixed point — so the address is a function of
    // the stored coordinates rather than of float arithmetic.
    let lo = FixedPoint::from_f64(bounds.0);
    let hi = FixedPoint::from_f64(bounds.1);
    let range = hi - lo;
    let zero = FixedPoint::from_int(0);
    let one = FixedPoint::from_int(1);
    let half = one / FixedPoint::from_int(2);
    let epsilon = FixedPoint::from_f64(1e-12);

    let norm: Vec<FixedPoint> = vals
        .iter()
        .map(|v| {
            if range < epsilon {
                half
            } else {
                let n = (*v - lo) / range;
                if n < zero { zero } else if n > one { one } else { n }
            }
        })
        .collect();
    HilbertMapper::new(vals.len(), HILBERT_BITS)
        .coords_to_index_fixed(&norm)
        .value()
}

/// Decode a dimension slice of a raw Q64.64 coordinate vector.
///
/// Mirrors the engine's `decode_semantic_slice`. Kept here so a ranking loop
/// can hoist the (constant) query decode out of the per-candidate path.
fn decode_dim_slice(coords: &[u8], r: &std::ops::Range<usize>) -> Vec<FixedPoint> {
    r.clone()
        .map(|dim| {
            let start = dim * 16;
            let end = start + 16;
            if coords.len() >= end {
                FixedPoint::from_raw(i128::from_le_bytes(coords[start..end].try_into().unwrap()))
            } else {
                FixedPoint::from_int(0)
            }
        })
        .collect()
}

/// Squared distance between a pre-decoded query slice and raw candidate bytes.
///
/// Ranking-only kernel: sqrt is monotone, so ordering by this is identical to
/// ordering by the true distance, at ~150 ns instead of ~23 us. Callers must
/// convert the survivors before returning distances to a user.
fn semantic_distance_sq(
    qv: &[FixedPoint],
    coords: &[u8],
    r: &std::ops::Range<usize>,
) -> FixedPoint {
    let cv = decode_dim_slice(coords, r);
    g_math::fixed_point::imperative::fused::euclidean_distance_squared(qv, &cv)
}

/// Compute the global-bounds Hilbert address of a semantic coordinate
/// vector (user dims 16+, capped at 8 axes, 12 bits per axis). This is THE
/// meaning→location function of format v3.
fn global_hilbert(sem: &[u8], semantic_dims: usize, bounds: (f64, f64)) -> u128 {
    hilbert_from_values(&decode_user_dims(sem, semantic_dims), bounds)
}

impl HoronCore {
    fn open_core(path: &Path, config: HoronConfig) -> HoronResult<Self> {
        if path.exists() {
            Self::open_existing(path, &config)
        } else {
            Self::create_new(path, config)
        }
    }

    fn validate_mode_combos(config: &HoronConfig) -> HoronResult<()> {
        if config.dimension != 4 {
            // The header persists a configurable dimension, but the underlying
            // engine Store currently always embeds in 4 dimensions — accepting
            // any other value would write a header the spatial API can never
            // honor (hardening audit). Reject loudly until Store is parameterized.
            return Err(HoronError::Config(format!(
                "dimension {} is not supported in this release (only 4)",
                config.dimension
            )));
        }
        if config.partial_reads && config.compression {
            return Err(HoronError::Config(
                "partial_reads requires compression: false (a zstd frame cannot be partially read)".into(),
            ));
        }
        if config.partial_reads && config.gacl {
            return Err(HoronError::Config(
                "partial_reads is not yet compatible with GACL enforcement".into(),
            ));
        }
        if config.quantized_semantic
            && config.semantic_dims as usize <= DIM_USER_DEFINED_START
        {
            return Err(HoronError::Config(format!(
                "quantized_semantic requires semantic_dims > {} (user dims to quantize)",
                DIM_USER_DEFINED_START
            )));
        }
        if config.meaning_addressed {
            if config.compression {
                return Err(HoronError::Config(
                    "meaning_addressed requires compression: false".into(),
                ));
            }
            if config.semantic_dims as usize <= DIM_USER_DEFINED_START {
                return Err(HoronError::Config(format!(
                    "meaning_addressed requires semantic_dims > {} (user dims to address by)",
                    DIM_USER_DEFINED_START
                )));
            }
            let (lo, hi) = config.semantic_bounds;
            if !(hi > lo) {
                return Err(HoronError::Config(
                    "semantic_bounds must satisfy max > min".into(),
                ));
            }
        }
        Ok(())
    }

    /// Serialize the bounds section (v3): user_dims × (min f64 LE, max f64 LE).
    fn bounds_section_bytes(semantic_dims: usize, bounds: (f64, f64)) -> Vec<u8> {
        let user_dims = semantic_dims.saturating_sub(DIM_USER_DEFINED_START);
        let mut out = Vec::with_capacity(user_dims * 16);
        for _ in 0..user_dims {
            out.extend_from_slice(&bounds.0.to_le_bytes());
            out.extend_from_slice(&bounds.1.to_le_bytes());
        }
        out
    }

    fn read_bounds_section(
        file: &mut File,
        semantic_dims: usize,
    ) -> HoronResult<(f64, f64)> {
        use std::io::Read;
        let user_dims = semantic_dims.saturating_sub(DIM_USER_DEFINED_START);
        let mut buf = vec![0u8; user_dims * 16];
        file.read_exact(&mut buf)?;
        if user_dims == 0 {
            return Err(HoronError::InvalidFormat(
                "meaning-addressed file with no user semantic dims".into(),
            ));
        }
        // Uniform bounds today: read dim 0 (all dims written identical).
        let lo = f64::from_le_bytes(buf[0..8].try_into().unwrap());
        let hi = f64::from_le_bytes(buf[8..16].try_into().unwrap());
        if !(hi > lo) || !lo.is_finite() || !hi.is_finite() {
            return Err(HoronError::InvalidFormat(
                "corrupt bounds section (max <= min or non-finite)".into(),
            ));
        }
        Ok((lo, hi))
    }

    fn create_new(path: &Path, config: HoronConfig) -> HoronResult<Self> {
        Self::validate_mode_combos(&config)?;
        let tau_raw = g_math::fixed_point::FixedPoint::from_int(1).raw();
        let mut header = GeoHeader::with_gacl(
            config.dimension,
            config.semantic_dims,
            tau_raw,
            config.compression,
            config.gacl,
        );
        if config.meaning_addressed {
            header.version = VERSION_MEANING_ADDRESSED;
            header.flags |= FLAG_MEANING_ADDRESSED;
        }
        if config.quantized_semantic {
            header.version = VERSION_QUANTIZED;
            header.flags |= FLAG_QUANTIZED_SEMANTIC;
        }
        let layout = SemLayout::from_header(&header);

        // Open WITHOUT truncating, take the advisory lock, then truncate.
        // Truncating before the lock let a create/create race destroy the
        // winner's freshly written header (hardening audit).
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        try_lock_exclusive(&file, path)?;
        file.set_len(0)?;
        cleanup_orphan_tmp(path);

        file.write_all(&header.to_bytes())?;
        if config.meaning_addressed {
            file.write_all(&Self::bounds_section_bytes(
                config.semantic_dims as usize,
                config.semantic_bounds,
            ))?;
        }
        snapshot::write_snapshot(&mut file, &[], false, true, &layout)?;
        wal::write_wal_header(&mut file, 0, 1)?;
        let wal_data_offset = file.stream_position()?;
        // Make the freshly created file durable: content + directory entry.
        file.sync_all()?;
        fsync_dir(path)?;

        let ma_bounds = config.meaning_addressed.then_some(config.semantic_bounds);
        // A fresh file has an empty snapshot; partial mode starts with an
        // empty view over it.
        let partial = if config.partial_reads {
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let raw_start = HEADER_SIZE
                + ma_bounds.map_or(0, |_| {
                    (config.semantic_dims as usize - DIM_USER_DEFINED_START) * 16
                })
                + SNAP_HEADER_SIZE;
            let view = SnapView::scan(mmap, raw_start, 0, 0, layout, None)?;
            Some(PartialState {
                view: RwLock::new(view),
                tombstones: RwLock::new(HashSet::new()),
                scanned: AtomicUsize::new(0),
            })
        } else {
            None
        };

        // Exact: the header stores tau as Q64.64, so it goes straight back
        // into the store. An f64 round-trip here would silently replay a
        // file at a tau it was never written with.
        let store = Store::with_config(
            StoreConfig::new().tau(g_math::fixed_point::FixedPoint::from_raw(tau_raw)),
        );

        let wal = WalWriter {
            file,
            next_seq: 1,
            entry_count: 0,
            wal_data_offset,
            pending_serialized: Vec::new(),
            pending_entries: Vec::new(),
            subscribers: Vec::new(),
            pending_count: 0,
            batch_size: config.wal_batch_size,
            flush_interval: Duration::from_millis(config.wal_flush_interval_ms),
            last_flush: Instant::now(),
            durability: config.durability,
            wal_compressed: config.compression,
            compression_algo: if config.compression { ALGO_ZSTD } else { 0 },
            layout,
        };

        Ok(Self {
            store,
            path: path.to_path_buf(),
            header,
            wal: Mutex::new(wal),
            config,
            compacting: AtomicBool::new(false),
            credentials: RwLock::new(None),
            last_compaction_error: Mutex::new(None),
            self_weak: OnceLock::new(),
            partial,
            ma_bounds,
            current_epoch: AtomicU64::new(0),
        })
    }

    fn open_existing(path: &Path, config: &HoronConfig) -> HoronResult<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;
        try_lock_exclusive(&file, path)?;
        cleanup_orphan_tmp(path);

        let mut header_bytes = [0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut file, &mut header_bytes)?;
        let header = GeoHeader::from_bytes(&header_bytes)?;

        let layout = SemLayout::from_header(&header);
        let compressed = header.compression_enabled();
        let snapshot_has_crc = header.version >= 2;
        let meaning_addressed = header.flags & FLAG_MEANING_ADDRESSED != 0;

        // v3: the bounds section sits between the header and the snapshot.
        let ma_bounds = if meaning_addressed {
            Some(Self::read_bounds_section(&mut file, header.semantic_dims as usize)?)
        } else {
            None
        };

        if config.partial_reads {
            if compressed {
                return Err(HoronError::Config(
                    "partial_reads requires an uncompressed snapshot".into(),
                ));
            }
            if header.gacl_enabled() || config.gacl {
                return Err(HoronError::Config(
                    "partial_reads is not yet compatible with GACL enforcement".into(),
                ));
            }
            return Self::open_partial(path, file, header, ma_bounds, config);
        }

        let snap_entries = {
            let mut entries =
                snapshot::read_snapshot(&mut file, compressed, &layout, snapshot_has_crc)?;
            // Meaning-addressed files store entries in pure Hilbert order;
            // Sarkar replay needs parents before children, so restore depth
            // order in memory (child_index metadata makes the rest
            // order-insensitive).
            if meaning_addressed {
                entries.sort_by(|a, b| {
                    let da = a.key.matches('/').count();
                    let db = b.key.matches('/').count();
                    da.cmp(&db).then(a.key.cmp(&b.key))
                });
            }
            entries
        };
        let (wal_entry_count, wal_base_seq) = wal::read_wal_header(&mut file)?;
        let wal_data_offset = file.stream_position()?;

        // Build store with tau from header
        let store = Store::with_config(
            StoreConfig::new().tau(g_math::fixed_point::FixedPoint::from_raw(header.tau_raw)),
        );
        load_snapshot_into_store(&store, &snap_entries, config.lazy_geometry)?;

        // Replay WAL entries — see scan_wal for the torn-header rationale.
        let lazy = config.lazy_geometry;
        // Epoch counter restoration (temporal epochs): the highest OP_EPOCH id seen in
        // the replayed WAL is the file's current epoch.
        let replayed_epoch = std::cell::Cell::new(0u64);
        let (next_seq, valid_wal_count, wal_valid_end) = scan_wal(
            &mut file,
            &header,
            &layout,
            wal_base_seq,
            wal_entry_count,
            |entry| {
                if let WalPayload::Epoch { epoch_id, .. } = &entry.payload {
                    replayed_epoch.set((*epoch_id).max(replayed_epoch.get()));
                }
                replay_entry(&store, entry, lazy)
            },
        )?;

        // Recovery truncates the torn tail: without this, appends land AFTER
        // the garbage and the next open's scan (which stops at the garbage)
        // silently loses them (hardening audit).
        if wal_valid_end < file.metadata()?.len() {
            log::warn!(
                "truncating torn WAL tail at byte {} (file was {} bytes)",
                wal_valid_end,
                file.metadata()?.len()
            );
            file.set_len(wal_valid_end)?;
            file.sync_all()?;
        }
        file.seek(SeekFrom::End(0))?;

        let partial: Option<PartialState> = None;
        let file_config = HoronConfig {
            dimension: header.dimension,
            semantic_dims: header.semantic_dims,
            compression: compressed,
            auto_compact_threshold: config.auto_compact_threshold,
            wal_batch_size: config.wal_batch_size,
            wal_flush_interval_ms: config.wal_flush_interval_ms,
            durability: config.durability,
            gacl: header.gacl_enabled(),
            gacl_fail_closed: config.gacl_fail_closed,
            lazy_geometry: config.lazy_geometry,
            approximate_semantic: config.approximate_semantic,
            partial_reads: config.partial_reads,
            meaning_addressed,
            semantic_bounds: ma_bounds.unwrap_or(config.semantic_bounds),
            history_retention: config.history_retention,
            quantized_semantic: header.quantized_semantic(),
        };

        let wal = WalWriter {
            file,
            next_seq,
            entry_count: valid_wal_count,
            wal_data_offset,
            pending_serialized: Vec::new(),
            pending_entries: Vec::new(),
            subscribers: Vec::new(),
            pending_count: 0,
            batch_size: config.wal_batch_size,
            flush_interval: Duration::from_millis(config.wal_flush_interval_ms),
            last_flush: Instant::now(),
            durability: config.durability,
            wal_compressed: header.wal_compressed(),
            compression_algo: header.compression_algo(),
            layout,
        };

        Ok(Self {
            store,
            path: path.to_path_buf(),
            header,
            wal: Mutex::new(wal),
            config: file_config,
            compacting: AtomicBool::new(false),
            credentials: RwLock::new(None),
            last_compaction_error: Mutex::new(None),
            self_weak: OnceLock::new(),
            partial,
            ma_bounds,
            current_epoch: AtomicU64::new(replayed_epoch.get()),
        })
    }

    /// Open in partial (mmap) mode: snapshot payloads stay on disk; only a
    /// structural index is built. `file` is positioned just past the header
    /// (and bounds section, if any).
    fn open_partial(
        path: &Path,
        mut file: File,
        header: GeoHeader,
        ma_bounds: Option<(f64, f64)>,
        config: &HoronConfig,
    ) -> HoronResult<Self> {
        use std::io::Read;

        let layout = SemLayout::from_header(&header);
        let semantic_dims = header.semantic_dims as usize;
        let snapshot_has_crc = header.version >= 2;

        // Snapshot section header (manual — we index, not materialize).
        let mut b4 = [0u8; 4];
        file.read_exact(&mut b4)?;
        let snap_byte_len = u32::from_le_bytes(b4) as usize;
        if snap_byte_len > MAX_SNAPSHOT_BYTES {
            return Err(HoronError::InvalidFormat(format!(
                "snapshot byte length {} exceeds maximum {}",
                snap_byte_len, MAX_SNAPSHOT_BYTES
            )));
        }
        file.read_exact(&mut b4)?;
        let node_count = u32::from_le_bytes(b4) as usize;
        if node_count > snap_byte_len / 8 + 1 {
            return Err(HoronError::InvalidFormat(format!(
                "snapshot node count {} impossible for {} section bytes",
                node_count, snap_byte_len
            )));
        }

        let raw_start = file.stream_position()? as usize;
        file.seek(SeekFrom::Start((raw_start + snap_byte_len) as u64))?;
        let stored_crc = if snapshot_has_crc {
            file.read_exact(&mut b4)?;
            Some(u32::from_le_bytes(b4))
        } else {
            None
        };
        let (wal_entry_count, wal_base_seq) = wal::read_wal_header(&mut file)?;
        let wal_data_offset = file.stream_position()?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Verify the snapshot CRC by streaming over the mmap — one
        // sequential pass, no heap materialization. (The structural scan
        // below touches the same pages anyway.)
        if let Some(stored) = stored_crc {
            let region = mmap.get(raw_start..raw_start + snap_byte_len).ok_or_else(|| {
                HoronError::InvalidFormat("snapshot region exceeds file length".into())
            })?;
            let computed = crc32fast::hash(region);
            if stored != computed {
                return Err(HoronError::ChecksumMismatch {
                    expected: stored,
                    actual: computed,
                    context: "snapshot section".to_string(),
                });
            }
        }

        // Structural scan → index (+ Hilbert addresses for v3 files).
        let hilbert_fn;
        let hilbert_of: Option<&dyn Fn(&[u8]) -> u128> = match ma_bounds {
            Some(b) => {
                // The scan hands the ON-DISK tail; decode quantized tails to
                // full width so the Hilbert pipeline sees canonical bytes.
                hilbert_fn = move |sem: &[u8]| {
                    if layout.quantized {
                        match layout.decode_tail(sem) {
                            Ok(full) => global_hilbert(&full, semantic_dims, b),
                            Err(_) => 0,
                        }
                    } else {
                        global_hilbert(sem, semantic_dims, b)
                    }
                };
                Some(&hilbert_fn)
            }
            None => None,
        };
        let view = SnapView::scan(
            mmap, raw_start, snap_byte_len, node_count, layout, hilbert_of,
        )?;

        // Overlay store (data-only; no geometry in partial mode) + WAL replay.
        let store = Store::with_config(
            StoreConfig::new().tau(g_math::fixed_point::FixedPoint::from_raw(header.tau_raw)),
        );
        let mut tombstones = HashSet::new();
        // Epoch counter restoration (temporal epochs) — see full-mode open.
        let replayed_epoch = std::cell::Cell::new(0u64);
        let (next_seq, valid_wal_count, wal_valid_end) = scan_wal(
            &mut file,
            &header,
            &layout,
            wal_base_seq,
            wal_entry_count,
            |entry| {
                if let WalPayload::Epoch { epoch_id, .. } = &entry.payload {
                    replayed_epoch.set((*epoch_id).max(replayed_epoch.get()));
                }
                apply_wal_partial(&store, &view, &mut tombstones, entry)
            },
        )?;
        // Recovery truncates the torn tail (see full-mode open for rationale).
        if wal_valid_end < file.metadata()?.len() {
            log::warn!(
                "truncating torn WAL tail at byte {} (file was {} bytes)",
                wal_valid_end,
                file.metadata()?.len()
            );
            file.set_len(wal_valid_end)?;
            file.sync_all()?;
        }
        file.seek(SeekFrom::End(0))?;

        let partial = Some(PartialState {
            view: RwLock::new(view),
            tombstones: RwLock::new(tombstones),
            scanned: AtomicUsize::new(0),
        });

        let file_config = HoronConfig {
            dimension: header.dimension,
            semantic_dims: header.semantic_dims,
            compression: false,
            approximate_semantic: config.approximate_semantic,
            auto_compact_threshold: config.auto_compact_threshold,
            wal_batch_size: config.wal_batch_size,
            wal_flush_interval_ms: config.wal_flush_interval_ms,
            durability: config.durability,
            gacl: false,
            gacl_fail_closed: config.gacl_fail_closed,
            lazy_geometry: true,
            partial_reads: true,
            meaning_addressed: ma_bounds.is_some(),
            semantic_bounds: ma_bounds.unwrap_or(config.semantic_bounds),
            history_retention: config.history_retention,
            quantized_semantic: header.quantized_semantic(),
        };

        let wal_writer = WalWriter {
            file,
            next_seq,
            entry_count: valid_wal_count,
            wal_data_offset,
            pending_serialized: Vec::new(),
            pending_entries: Vec::new(),
            subscribers: Vec::new(),
            pending_count: 0,
            batch_size: config.wal_batch_size,
            flush_interval: Duration::from_millis(config.wal_flush_interval_ms),
            last_flush: Instant::now(),
            durability: config.durability,
            wal_compressed: header.wal_compressed(),
            compression_algo: header.compression_algo(),
            layout,
        };

        Ok(Self {
            store,
            path: path.to_path_buf(),
            header,
            wal: Mutex::new(wal_writer),
            config: file_config,
            compacting: AtomicBool::new(false),
            credentials: RwLock::new(None),
            last_compaction_error: Mutex::new(None),
            self_weak: OnceLock::new(),
            partial,
            ma_bounds,
            current_epoch: AtomicU64::new(replayed_epoch.get()),
        })
    }

    /// Semantic-tail layout of this file (quantization aware).
    fn sem_layout(&self) -> SemLayout {
        SemLayout::from_header(&self.header)
    }

    // ---- GACL ----

    /// Set credentials for this session. When the file has GACL enabled
    /// (header flag), all subsequent reads/writes will check access bands
    /// in semantic dimensions 0–11 against these credentials.
    ///
    /// Has no effect on files without GACL.
    pub fn set_credentials(&self, creds: Credentials) {
        let mut guard = self.credentials.write().unwrap_or_else(|e| e.into_inner());
        *guard = Some(creds);
    }

    /// Clear credentials — disables GACL enforcement for this session.
    pub fn clear_credentials(&self) {
        let mut guard = self.credentials.write().unwrap_or_else(|e| e.into_inner());
        *guard = None;
    }

    /// Whether GACL enforcement is active for query filtering.
    ///
    /// Active when the header flag is set AND either credentials are present,
    /// or the file is configured fail-closed (in which case filtering runs and
    /// denies everything until credentials are supplied).
    pub fn gacl_active(&self) -> bool {
        if !self.header.gacl_enabled() {
            return false;
        }
        let guard = self.credentials.read().unwrap_or_else(|e| e.into_inner());
        guard.is_some() || self.config.gacl_fail_closed
    }

    /// Check read access for a key. Returns Ok(()) if access is granted,
    /// or Err(AccessDenied) if the caller lacks read permission.
    ///
    /// Access is always granted when:
    /// - GACL flag is not set in the header, OR
    /// - No credentials have been set, OR
    /// - The node has no semantic coordinates (= public), OR
    /// - The node's GACL bands (dims 0–11) are all open
    fn check_read(&self, key: &str) -> HoronResult<()> {
        if !self.header.gacl_enabled() {
            return Ok(());
        }
        let guard = self.credentials.read().unwrap_or_else(|e| e.into_inner());
        let creds = match guard.as_ref() {
            Some(c) => c,
            None => {
                // No credentials: fail open (allow) or fail closed (deny) per
                // config. Fail-closed keeps a forgotten set_credentials() from
                // silently exposing every node.
                return self.no_credentials_result(key);
            }
        };

        let bands = self.node_bands(key);
        if creds.can_read(&bands) {
            Ok(())
        } else {
            Err(HoronError::AccessDenied {
                key: key.to_string(),
                reason: "read access denied by GACL".to_string(),
            })
        }
    }

    /// Check write access for a key.
    fn check_write(&self, key: &str) -> HoronResult<()> {
        if !self.header.gacl_enabled() {
            return Ok(());
        }
        let guard = self.credentials.read().unwrap_or_else(|e| e.into_inner());
        let creds = match guard.as_ref() {
            Some(c) => c,
            None => return self.no_credentials_result(key),
        };

        let bands = self.node_bands(key);
        if creds.can_access(&bands) {
            Ok(())
        } else {
            Err(HoronError::AccessDenied {
                key: key.to_string(),
                reason: "write access denied by GACL".to_string(),
            })
        }
    }

    /// Access decision when GACL is enabled but no credentials are set:
    /// `Ok(())` under the default fail-open policy, `AccessDenied` when the
    /// file is configured fail-closed.
    fn no_credentials_result(&self, key: &str) -> HoronResult<()> {
        if self.config.gacl_fail_closed {
            Err(HoronError::AccessDenied {
                key: key.to_string(),
                reason: "GACL enabled but no credentials set (fail-closed)".to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Extract GACL bands from a node's semantic coordinates.
    /// Returns public bands if the node has no coords or insufficient dims.
    fn node_bands(&self, key: &str) -> NodeAccessBands {
        match self.store.get_semantic(key) {
            Ok(sem) if sem.len() >= 12 * 16 => {
                NodeAccessBands::from_semantic_bytes(&sem)
                    .unwrap_or_else(NodeAccessBands::public)
            }
            _ => NodeAccessBands::public(),
        }
    }

    /// Retain only the read-accessible items, honoring the fail-open /
    /// fail-closed policy when no credentials are set. Generic over the item
    /// shape via a key extractor, so it serves both `(key, dist)` result lists
    /// and bare path lists.
    fn retain_readable<T>(&self, items: Vec<T>, key_of: impl Fn(&T) -> &str) -> Vec<T> {
        if !self.header.gacl_enabled() {
            return items;
        }
        let guard = self.credentials.read().unwrap_or_else(|e| e.into_inner());
        let creds = match guard.as_ref() {
            Some(c) => c,
            // No credentials: fail open (return all) or fail closed (drop all).
            None => return if self.config.gacl_fail_closed { Vec::new() } else { items },
        };

        items.into_iter().filter(|item| {
            let bands = self.node_bands(key_of(item));
            creds.can_read(&bands)
        }).collect()
    }

    /// Filter path lists by read access.
    fn filter_readable_paths(&self, paths: Vec<String>) -> Vec<String> {
        self.retain_readable(paths, |k| k.as_str())
    }

    /// Collect the `k` nearest read-accessible items, expanding the candidate
    /// window until `k` survive access-filtering or the store is exhausted.
    ///
    /// A fixed over-fetch (e.g. `k * 3`) silently drops accessible nodes that
    /// sit just past the window when nearer nodes are inaccessible — the query
    /// then returns fewer than `k` results, or none, even though more exist.
    /// The window grows geometrically and is capped at the total node count,
    /// so the cost is bounded by the store size and paid only when access
    /// filtering actually removes candidates.
    fn collect_k_readable<T>(
        &self,
        k: usize,
        key_of: impl Fn(&T) -> &str + Copy,
        mut fetch: impl FnMut(usize) -> HoronResult<Vec<T>>,
    ) -> HoronResult<Vec<T>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let total = self.store.len().max(1);
        let mut window = k.saturating_mul(3).max(16);
        loop {
            let raw = fetch(window)?;
            let fetched = raw.len();
            let filtered = self.retain_readable(raw, key_of);
            if filtered.len() >= k || fetched >= total || window >= total {
                return Ok(filtered.into_iter().take(k).collect());
            }
            window = window.saturating_mul(4).min(total);
        }
    }

    // ---- Public API (mirrors Store) ----

    /// Reject writes the on-disk format cannot represent. Key and metadata
    /// lengths are u16 fields and data length is a bounded u32 field; an
    /// oversized value would silently wrap at serialization time, corrupting
    /// the WAL and destroying committed entries behind it.
    fn validate_write(key: &str, data_len: usize, meta: Option<(&str, &str)>) -> HoronResult<()> {
        if key.len() > u16::MAX as usize {
            return Err(HoronError::InvalidOperation(format!(
                "key length {} exceeds format maximum {}",
                key.len(),
                u16::MAX
            )));
        }
        if data_len > crate::format::MAX_ENTRY_DATA {
            return Err(HoronError::InvalidOperation(format!(
                "data length {} exceeds format maximum {}",
                data_len,
                crate::format::MAX_ENTRY_DATA
            )));
        }
        if let Some((mk, mv)) = meta {
            if mk.len() > u16::MAX as usize || mv.len() > u16::MAX as usize {
                return Err(HoronError::InvalidOperation(format!(
                    "metadata key/value length {}/{} exceeds format maximum {}",
                    mk.len(),
                    mv.len(),
                    u16::MAX
                )));
            }
        }
        Ok(())
    }

    /// Store data at a key (upsert). Thread-safe.
    ///
    /// Store is mutated first (engine internal locks), then WAL is appended
    /// under `Mutex<WalWriter>`. No lock nesting between crates.
    ///
    /// This apply-then-log order is required by the compaction fence: an
    /// entry's sequence number is only assigned once the in-memory store
    /// already reflects it, so a compaction that fences at that sequence sees
    /// a consistent store snapshot. One consequence: if the WAL append itself
    /// fails (a durability error, e.g. disk I/O), the in-memory change has
    /// already been applied and stays visible — a later successful compaction
    /// persists it, but a crash before any flush would lose it. The `Err`
    /// therefore means "not durably committed", not "no change made".
    pub fn put(&self, key: &str, data: &[u8]) -> HoronResult<()> {
        self.put_inner(key, data, true)
    }

    /// Store data at a key without building hyperbolic geometry (upsert).
    ///
    /// Mirrors [`Store::put_data_only`]: skips the Sarkar embedding, VP-tree,
    /// Klein, and power-diagram construction that ordinary [`put`](Self::put)
    /// pays (~milliseconds per node). Semantic queries (`nearest_semantic`,
    /// `neighbors_semantic`, `get_semantic`) and meaning-addressed layout
    /// still work — those are driven by the stored semantic coordinates, not
    /// the hyperbolic embedding. Only spatial queries (`nearest`, `neighbors`)
    /// will not see nodes inserted this way until the file is reopened without
    /// `lazy_geometry` (which replays them through the full `put` path).
    ///
    /// Use this for bulk-loading meaning-addressed / semantic-only files,
    /// where the hyperbolic geometry is never queried.
    pub fn put_data_only(&self, key: &str, data: &[u8]) -> HoronResult<()> {
        self.put_inner(key, data, false)
    }

    /// Ancestor paths of `key`, shallowest first, excluding the root and the
    /// key itself: `/a/b/c` → `["/a", "/a/b"]`.
    fn ancestor_paths(key: &str) -> Vec<String> {
        let parts: Vec<&str> = key.split('/').filter(|p| !p.is_empty()).collect();
        (1..parts.len())
            .map(|depth| format!("/{}", parts[..depth].join("/")))
            .collect()
    }

    fn put_inner(&self, key: &str, data: &[u8], geometry: bool) -> HoronResult<()> {
        Self::validate_write(key, data.len(), None)?;
        let is_update = self.exists_inner(key);
        // Inserting under a fresh path implicitly creates the ancestors. Those
        // creations must reach the WAL as explicit entries, or a replica built
        // by tailing the WAL is missing nodes the primary has ("the file is
        // the replication protocol" requires the log to describe all state).
        let new_ancestors: Vec<String> = if is_update {
            Vec::new()
        } else {
            Self::ancestor_paths(key)
                .into_iter()
                .filter(|ancestor| !self.exists_inner(ancestor))
                .collect()
        };
        // For updates, check write access on existing node.
        // For inserts, node has no bands yet → public → always allowed.
        if is_update {
            self.check_write(key)?;
        }
        if let Some(p) = &self.partial {
            // Promote a snapshot-resident node first so its metadata and
            // semantics survive the update, then write data-only (no
            // geometry in partial mode) and clear any tombstone.
            {
                let view = p.view.read().unwrap_or_else(|e| e.into_inner());
                let _ = promote_from_view(&self.store, &view, key);
                ensure_ancestors(&self.store, &view, key);
            }
            Store::put_data_only(&self.store, key, data)?;
            p.tombstones.write().unwrap_or_else(|e| e.into_inner()).remove(key);
        } else if geometry {
            self.store.put(key, data)?;
        } else {
            self.store.put_data_only(key, data)?;
        }

        let should_compact = {
            let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
            let sem_bytes = self.header.semantic_dims as usize * 16;
            // Log the ancestors this put created, parents-first, before the
            // leaf's own entry (replay requires parents before children).
            // Metadata mirrors what compaction persists for these nodes:
            // synthesized wall-clock fields filtered, the rest sorted for
            // deterministic bytes. A concurrent sibling put can race us into
            // logging the same ancestor twice — replay skips the duplicate
            // (AlreadyExists is tolerated there for exactly this shape).
            for ancestor in &new_ancestors {
                let mut metadata: Vec<(String, String)> = self
                    .store
                    .get_meta(ancestor)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|(k, _)| {
                        k != "key" && k != "size" && k != "created_at" && k != "updated_at"
                    })
                    .collect();
                metadata.sort();
                let entry = WalEntry {
                    seq: wal.next_seq,
                    op: OP_INSERT,
                    key: ancestor.clone(),
                    payload: WalPayload::Insert(NodeEntry {
                        key: ancestor.clone(),
                        data: Vec::new(),
                        metadata,
                        semantic_coords: vec![0u8; sem_bytes],
                    }),
                };
                wal.append(&entry)?;
            }
            let entry = if is_update {
                WalEntry {
                    seq: wal.next_seq,
                    op: OP_UPDATE,
                    key: key.to_string(),
                    payload: WalPayload::Update {
                        data: data.to_vec(),
                        metadata: vec![],
                    },
                }
            } else {
                WalEntry {
                    seq: wal.next_seq,
                    op: OP_INSERT,
                    key: key.to_string(),
                    payload: WalPayload::Insert(NodeEntry {
                        key: key.to_string(),
                        data: data.to_vec(),
                        metadata: vec![],
                        semantic_coords: vec![0u8; sem_bytes],
                    }),
                }
            };
            wal.append(&entry)?;
            self.config.auto_compact_threshold > 0
                && wal.total_entries() >= self.config.auto_compact_threshold
        };

        if should_compact {
            // Off the writer thread; errors are logged and recorded in
            // last_compaction_error() — never fails the write itself.
            self.spawn_auto_compact();
        }

        Ok(())
    }

    /// Retrieve data by key. Lock-free (in-memory via DashMap).
    /// Returns `AccessDenied` if GACL is active and credentials lack read access.
    pub fn get(&self, key: &str) -> HoronResult<Vec<u8>> {
        self.check_read(key)?;
        if let Some(p) = &self.partial {
            return partial_get(&self.store, p, key);
        }
        Ok(self.store.get(key)?)
    }

    /// Remove a key. Thread-safe.
    /// Returns `AccessDenied` if GACL is active and credentials lack write access.
    pub fn remove(&self, key: &str) -> HoronResult<()> {
        self.check_write(key)?;
        if let Some(p) = &self.partial {
            let in_store = self.store.exists(key);
            let in_snapshot = {
                let view = p.view.read().unwrap_or_else(|e| e.into_inner());
                view.by_key.contains_key(key)
            };
            let tombstoned = p
                .tombstones
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .contains(key);
            if !in_store && (!in_snapshot || tombstoned) {
                return Err(HoronError::Store(
                    horon_engine::store::StoreError::NotFound(key.to_string()),
                ));
            }
            if in_store {
                self.store.remove(key)?;
            }
            p.tombstones
                .write()
                .unwrap_or_else(|e| e.into_inner())
                .insert(key.to_string());
        } else {
            self.store.remove(key)?;
        }

        let should_compact = {
            let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
            let entry = WalEntry {
                seq: wal.next_seq,
                op: OP_DELETE,
                key: key.to_string(),
                payload: WalPayload::Delete,
            };
            wal.append(&entry)?;
            self.config.auto_compact_threshold > 0
                && wal.total_entries() >= self.config.auto_compact_threshold
        };

        if should_compact {
            // Off the writer thread; errors are logged and recorded in
            // last_compaction_error() — never fails the write itself.
            self.spawn_auto_compact();
        }

        Ok(())
    }

    /// Check if a key exists. Lock-free.
    /// When GACL is active, returns `false` for nodes the caller cannot read.
    pub fn exists(&self, key: &str) -> bool {
        if !self.exists_inner(key) {
            return false;
        }
        self.check_read(key).is_ok()
    }

    /// Existence across overlay store, tombstones, and mmap index.
    fn exists_inner(&self, key: &str) -> bool {
        if self.store.exists(key) {
            return true;
        }
        if let Some(p) = &self.partial {
            return partial_exists(&self.store, p, key);
        }
        false
    }

    /// Set metadata on a key. Thread-safe.
    /// Returns `AccessDenied` if GACL is active and credentials lack write access.
    pub fn set_meta(&self, key: &str, name: &str, value: &str) -> HoronResult<()> {
        Self::validate_write(key, 0, Some((name, value)))?;
        self.check_write(key)?;
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            promote_from_view(&self.store, &view, key)?;
        }
        self.store.set_meta(key, name, value)?;

        let should_compact = {
            let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
            let entry = WalEntry {
                seq: wal.next_seq,
                op: OP_SET_META,
                key: key.to_string(),
                payload: WalPayload::SetMeta {
                    meta_key: name.to_string(),
                    meta_value: value.to_string(),
                },
            };
            wal.append(&entry)?;
            self.config.auto_compact_threshold > 0
                && wal.total_entries() >= self.config.auto_compact_threshold
        };

        if should_compact {
            // Off the writer thread; errors are logged and recorded in
            // last_compaction_error() — never fails the write itself.
            self.spawn_auto_compact();
        }

        Ok(())
    }

    /// Get metadata for a key. Lock-free.
    /// Returns `AccessDenied` if GACL is active and credentials lack read access.
    pub fn get_meta(&self, key: &str) -> HoronResult<HashMap<String, String>> {
        self.check_read(key)?;
        if let Some(p) = &self.partial {
            if self.store.exists(key) {
                return Ok(self.store.get_meta(key)?);
            }
            if !p.tombstones.read().unwrap_or_else(|e| e.into_inner()).contains(key) {
                let view = p.view.read().unwrap_or_else(|e| e.into_inner());
                if let Some(&idx) = view.by_key.get(key) {
                    let entry = view.decode(idx)?;
                    let mut map: HashMap<String, String> =
                        entry.metadata.into_iter().collect();
                    map.insert("key".to_string(), entry.key.clone());
                    map.insert("size".to_string(), entry.data.len().to_string());
                    return Ok(map);
                }
            }
            return Err(HoronError::Store(
                horon_engine::store::StoreError::NotFound(key.to_string()),
            ));
        }
        Ok(self.store.get_meta(key)?)
    }

    /// Set semantic coordinates on a key. Thread-safe.
    ///
    /// Coordinates are raw Q64.64 bytes (16 bytes per dimension).
    /// Use `FixedPoint::from_f64(value).raw().to_le_bytes()` to encode each.
    /// Vectors shorter than `semantic_dims × 16` bytes are zero-extended;
    /// longer ones are rejected (the on-disk tail is fixed-size).
    ///
    /// Returns `AccessDenied` if GACL is active and credentials lack write access.
    /// Note: this also gates changes to GACL bands (dims 0–11), preventing
    /// unauthorized users from opening up restricted nodes.
    pub fn set_semantic(&self, key: &str, coords: Vec<u8>) -> HoronResult<()> {
        Self::validate_write(key, 0, None)?;
        self.check_write(key)?;
        let mut coords = coords;
        let layout = self.sem_layout();
        if layout.quantized {
            // Write-through canonicalization: validate (range; reserved dims
            // zero unless GACL) and snap user dims to the TQ1.9 grid, so the
            // in-memory store equals the post-reload state byte-for-byte.
            layout.canonicalize(&mut coords)?;
        } else if coords.len() != layout.mem_bytes() {
            // The WAL/snapshot tail is fixed-size (semantic_dims × 16 bytes):
            // a shorter vector serialized verbatim would misalign the stream
            // on replay (the reader consumes exactly the tail width), failing
            // the next CRC and silently truncating every later committed
            // write. Zero-extend short vectors — the documented decode
            // semantics for missing dims — and reject oversized ones
            // (truncation would drop data).
            if coords.len() > layout.mem_bytes() {
                return Err(HoronError::InvalidOperation(format!(
                    "coords cover {} bytes but the file has {} semantic dims ({} bytes)",
                    coords.len(),
                    layout.dims,
                    layout.mem_bytes()
                )));
            }
            coords.resize(layout.mem_bytes(), 0);
        }
        if coords.iter().all(|&b| b == 0) {
            // The all-zero tail is the on-disk encoding of "not set": a node
            // stored at the exact origin would silently lose its placement on
            // reload (snapshot load and WAL replay both collapse zero tails).
            // Refuse the write instead of storing a placement the file cannot
            // represent.
            return Err(HoronError::InvalidOperation(
                "all-zero semantic coordinates encode \"not set\" and cannot be stored as a \
                 placement; use clear_semantic() to remove a placement, or offset at least one \
                 dimension from zero for a placement near the origin"
                    .to_string(),
            ));
        }
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            promote_from_view(&self.store, &view, key)?;
        }
        self.store.set_semantic(key, coords.clone())?;

        let should_compact = {
            let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
            let entry = WalEntry {
                seq: wal.next_seq,
                op: OP_SET_SEMANTIC,
                key: key.to_string(),
                payload: WalPayload::SetSemantic {
                    coords: coords,
                },
            };
            wal.append(&entry)?;
            self.config.auto_compact_threshold > 0
                && wal.total_entries() >= self.config.auto_compact_threshold
        };

        if should_compact {
            // Off the writer thread; errors are logged and recorded in
            // last_compaction_error() — never fails the write itself.
            self.spawn_auto_compact();
        }

        Ok(())
    }

    /// Remove a node's semantic placement.
    ///
    /// The on-disk encoding of "no placement" is an all-zero semantic tail,
    /// so this appends a SET_SEMANTIC entry of zeros and unsets the
    /// coordinates in memory: the node keeps its data and metadata but stops
    /// participating in semantic queries, and the state survives reload
    /// identically via both the WAL and snapshot paths.
    ///
    /// On a GACL file the access bands live in the same vector, so clearing
    /// a placement also removes the node's bands; re-establish them with
    /// [`set_semantic`](Self::set_semantic) if needed.
    pub fn clear_semantic(&self, key: &str) -> HoronResult<()> {
        Self::validate_write(key, 0, None)?;
        self.check_write(key)?;
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            promote_from_view(&self.store, &view, key)?;
        }
        self.store.set_semantic(key, Vec::new())?;

        let should_compact = {
            let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
            let entry = WalEntry {
                seq: wal.next_seq,
                op: OP_SET_SEMANTIC,
                key: key.to_string(),
                payload: WalPayload::SetSemantic {
                    coords: vec![0u8; self.sem_layout().mem_bytes()],
                },
            };
            wal.append(&entry)?;
            self.config.auto_compact_threshold > 0
                && wal.total_entries() >= self.config.auto_compact_threshold
        };

        if should_compact {
            self.spawn_auto_compact();
        }

        Ok(())
    }

    /// Get semantic coordinates for a key. Lock-free.
    ///
    /// Returns raw Q64.64 bytes (16 bytes per dimension). Empty if not set.
    /// Returns `AccessDenied` if GACL is active and credentials lack read access.
    pub fn get_semantic(&self, key: &str) -> HoronResult<Vec<u8>> {
        self.check_read(key)?;
        if let Some(p) = &self.partial {
            return partial_get_semantic(&self.store, p, key);
        }
        Ok(self.store.get_semantic(key)?)
    }

    /// List immediate children. Lock-free.
    /// When GACL is active, only children the caller can read are returned.
    pub fn children(&self, path: &str) -> HoronResult<Vec<String>> {
        if let Some(p) = &self.partial {
            return partial_children(&self.store, p, path);
        }
        let kids = self.store.children(path)?;
        Ok(self.filter_readable_paths(kids))
    }

    /// List all keys under prefix. Lock-free.
    /// When GACL is active, only nodes the caller can read are returned.
    pub fn list(&self, prefix: &str) -> HoronResult<Vec<String>> {
        if let Some(p) = &self.partial {
            return partial_list(&self.store, p, prefix);
        }
        let keys = self.store.list(prefix)?;
        Ok(self.filter_readable_paths(keys))
    }

    /// Find nearest node to coordinates. Lock-free.
    /// When GACL is active, skips inaccessible nodes.
    pub fn nearest(&self, coords: &[FixedPoint]) -> HoronResult<(String, FixedPoint)> {
        if !self.gacl_active() {
            return Ok(self.store.nearest(coords)?);
        }
        // Expand the candidate window until an accessible node is found or the
        // store is exhausted, so a cluster of inaccessible near nodes cannot
        // hide a slightly-farther accessible one.
        let closest = self.collect_k_readable(1, |(k, _): &(String, FixedPoint)| k.as_str(), |n| {
            Ok(self.store.nearest_k(coords, n)?)
        })?;
        closest.into_iter().next()
            .ok_or_else(|| HoronError::AccessDenied {
                key: "(nearest query)".to_string(),
                reason: "no accessible nodes found near query point".to_string(),
            })
    }

    /// Find k nearest neighbors of an existing node. Lock-free.
    /// When GACL is active, only accessible neighbors are returned.
    pub fn neighbors(&self, path: &str, k: usize) -> HoronResult<Vec<String>> {
        if !self.gacl_active() {
            return Ok(self.store.neighbors(path, k)?);
        }
        self.collect_k_readable(k, |p: &String| p.as_str(), |n| {
            Ok(self.store.neighbors(path, n)?)
        })
    }

    /// Find k nearest nodes by Euclidean distance across a dimensional slice.
    ///
    /// A dimensional slice selects which semantic dimensions to compare.
    /// Different slices answer different questions from the same data:
    /// - `16..33`: category preference topology (who takes what)
    /// - `33..40`: operational topology (availability, popularity)
    /// - `16..40`: combined (full picture)
    ///
    /// Returns `(key, distance)` sorted by distance ascending. Lock-free.
    pub fn nearest_semantic(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: std::ops::Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> {
        if self.partial.is_some() {
            return self.nearest_semantic_partial(query_coords, k, dim_range);
        }
        if !self.gacl_active() {
            return Ok(self.store.nearest_semantic(query_coords, k, dim_range)?);
        }
        self.collect_k_readable(k, |(key, _): &(String, FixedPoint)| key.as_str(), |n| {
            Ok(self.store.nearest_semantic(query_coords, n, dim_range.clone())?)
        })
    }

    /// Find k nearest nodes to an existing node by semantic dimensional distance.
    /// The queried node is excluded. Lock-free.
    /// When GACL is active, only accessible nodes are returned.
    pub fn neighbors_semantic(
        &self,
        path: &str,
        k: usize,
        dim_range: std::ops::Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> {
        if self.partial.is_some() {
            let coords = self.get_semantic(path)?;
            if coords.is_empty() {
                return Ok(Vec::new());
            }
            let results = self.nearest_semantic_partial(&coords, k + 1, dim_range)?;
            return Ok(results.into_iter().filter(|(key, _)| key != path).take(k).collect());
        }
        if !self.gacl_active() {
            return Ok(self.store.neighbors_semantic(path, k, dim_range)?);
        }
        self.collect_k_readable(k, |(key, _): &(String, FixedPoint)| key.as_str(), |n| {
            Ok(self.store.neighbors_semantic(path, n, dim_range.clone())?)
        })
    }

    /// Find the k nearest stored nodes to an arbitrary point in hyperbolic
    /// space. Lock-free.
    ///
    /// Like [`Horon::nearest`] but returns multiple candidates, sorted by
    /// ascending hyperbolic distance. When GACL is active, only accessible
    /// nodes are returned.
    pub fn nearest_k(
        &self,
        coords: &[FixedPoint],
        k: usize,
    ) -> HoronResult<Vec<(String, FixedPoint)>> {
        if !self.gacl_active() {
            return Ok(self.store.nearest_k(coords, k)?);
        }
        self.collect_k_readable(k, |(key, _): &(String, FixedPoint)| key.as_str(), |n| {
            Ok(self.store.nearest_k(coords, n)?)
        })
    }

    /// Find the k stored nodes most similar to an existing node across a
    /// dimensional slice — "what's like this one?". Lock-free.
    ///
    /// This is [`Horon::neighbors_semantic`] under a task-shaped name:
    /// the queried node is excluded and results are sorted by ascending
    /// semantic distance. When GACL is active, only accessible nodes are
    /// returned.
    pub fn find_similar(
        &self,
        key: &str,
        k: usize,
        dim_range: std::ops::Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> {
        self.neighbors_semantic(key, k, dim_range)
    }

    /// Find semantic outliers among the nodes under a key prefix: nodes whose
    /// average distance to their nearest peers is anomalously large relative
    /// to the population. Lock-free.
    ///
    /// See `Store::find_outliers` for the statistics. When GACL is active,
    /// inaccessible nodes are dropped from the returned list; the population
    /// statistics themselves are computed over the full prefix population,
    /// exactly as they would be for a caller with full access.
    pub fn find_outliers(
        &self,
        prefix: &str,
        z_threshold: FixedPoint,
        dim_range: std::ops::Range<usize>,
    ) -> HoronResult<Vec<SemanticOutlier>> {
        let outliers = self.store.find_outliers(prefix, z_threshold, dim_range)?;
        Ok(self.retain_readable(outliers, |o| o.key.as_str()))
    }

    /// Find all nodes within a hyperbolic distance of an existing node.
    /// Lock-free. When GACL is active, only accessible nodes are returned.
    pub fn find_within(&self, path: &str, radius: FixedPoint) -> HoronResult<Vec<String>> {
        let keys = self.store.find_within(path, radius)?;
        Ok(self.filter_readable_paths(keys))
    }

    /// The hyperbolic (Poincaré) position of a stored key. Lock-free.
    ///
    /// Errors for unknown keys and for data-only nodes (no embedding).
    /// Returns `AccessDenied` if GACL is active and credentials lack read
    /// access.
    pub fn position(&self, key: &str) -> HoronResult<Vec<FixedPoint>> {
        self.check_read(key)?;
        Ok(self.store.position(key)?)
    }

    /// Number of stored entries (excludes root).
    pub fn len(&self) -> usize {
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            let tombs = p.tombstones.read().unwrap_or_else(|e| e.into_inner());
            let snapshot_only = view
                .entries
                .iter()
                .filter(|e| !self.store.exists(&e.key) && !tombs.contains(&e.key))
                .count();
            return self.store.len() + snapshot_only;
        }
        self.store.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        if self.partial.is_some() {
            // Snapshot-resident entries live in the mmap view, not the
            // overlay store — count them too.
            return self.len() == 0;
        }
        self.store.is_empty()
    }

    /// Access the underlying engine Store.
    pub fn store(&self) -> &Store {
        &self.store
    }

    /// Semantic k-NN over the mmap'd snapshot plus the overlay store.
    ///
    /// For meaning-addressed files (v3) queried on user dims, candidates are
    /// selected by an expanding window around the query's Hilbert address —
    /// only that byte-neighborhood of the file is decoded. The window is a
    /// locality heuristic (space-filling curves have seams), verified by
    /// exact distances and widened adaptively; it degrades to a full scan
    /// when the window exhausts. Non-v3 files always full-scan (still
    /// without materializing the snapshot into RAM).
    fn nearest_semantic_partial(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: std::ops::Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> {
        let p = self.partial.as_ref().expect("partial mode");
        partial_nearest_semantic(
            &self.store,
            p,
            self.ma_bounds,
            self.config.approximate_semantic,
            query_coords,
            k,
            dim_range,
        )
    }
}

/// Partial-mode semantic k-NN over the mmap view — shared by [`Horon`] and
/// [`HoronReader`] so both answer through one implementation. Exact by
/// default (full proxy-space scan, zero materialization); `allow_window`
/// opts into the approximate Hilbert byte-neighborhood.
#[allow(clippy::too_many_arguments)]
fn partial_nearest_semantic(
    overlay: &Store,
    p: &PartialState,
    ma_bounds: Option<(f64, f64)>,
    allow_window: bool,
    query_coords: &[u8],
    k: usize,
    dim_range: std::ops::Range<usize>,
) -> HoronResult<Vec<(String, FixedPoint)>> {
    {
        if k == 0 {
            return Ok(Vec::new());
        }
        let view = p.view.read().unwrap_or_else(|e| e.into_inner());
        let tombs = p.tombstones.read().unwrap_or_else(|e| e.into_inner());
        let n = view.entries.len();

        // Candidate selection.
        // The window is an APPROXIMATION and is off unless asked for. It is
        // also only meaningful when the queried slice is one the layout
        // actually addresses: the window is centred by the first
        // MAX_HILBERT_DIMS user dims, so ranking by any other slice places it
        // somewhere unrelated to the answer.
        let addressed_end = DIM_USER_DEFINED_START + MAX_HILBERT_DIMS;
        let windowed = allow_window
            && ma_bounds.is_some()
            && dim_range.start >= DIM_USER_DEFINED_START
            && dim_range.end <= addressed_end;
        // `best` holds SQUARED distances until the survivors are converted.
        let mut best: Vec<(String, FixedPoint)> = Vec::new();
        let mut scanned = 0usize;
        // Decode the query slice once; it is constant across the whole scan.
        let qv = decode_dim_slice(query_coords, &dim_range);

        let consider = |i: usize, view: &SnapView, best: &mut Vec<(String, FixedPoint)>,
                            scanned: &mut usize| -> HoronResult<()> {
            let e = &view.entries[i];
            if tombs.contains(&e.key) || overlay.exists(&e.key) {
                return Ok(()); // deleted or shadowed by the overlay
            }
            let sem = view.semantic_of(i)?; // on-disk encoding
            if !sem.iter().any(|&b| b != 0) {
                return Ok(()); // no coords = not in semantic space
            }
            *scanned += 1;
            let layout = view.layout();
            // Rank by SQUARED distance. sqrt is monotone, so the top-k is
            // identical, and a Q64.64 sqrt costs ~23 us against ~150 ns for
            // the squared kernel — 160x, and it is paid once per candidate.
            // The k survivors are converted to true distances below, so the
            // returned values are unchanged.
            let dsq = if layout.quantized {
                let full = layout.decode_tail(sem)?;
                semantic_distance_sq(&qv, &full, &dim_range)
            } else {
                semantic_distance_sq(&qv, sem, &dim_range)
            };
            best.push((e.key.clone(), dsq));
            Ok(())
        };

        if windowed && n > 0 {
            let bounds = ma_bounds.unwrap();
            let sem_dims = view.layout().dims;
            let qvals = decode_user_dims(query_coords, sem_dims);
            let span = bounds.1 - bounds.0;

            // Multi-probe expanding windows. A space-filling curve has
            // seams: one spatial neighborhood can map to several curve
            // segments. Probing the query point AND small ± offsets along
            // each addressed axis lands a window in each nearby segment;
            // exact distances then verify every candidate. Widen until we
            // hold k candidates with margin or the probes cover the file.
            let mut half = (2 * k).max(16);
            let mut delta = FixedPoint::from_f64(span / 256.0);
            let mut seen: HashSet<usize> = HashSet::new();
            loop {
                best.clear();
                seen.clear();
                scanned = 0;

                let mut probes: Vec<Vec<FixedPoint>> = vec![qvals.clone()];
                for d in 0..qvals.len() {
                    for sign in [-1i32, 1] {
                        let mut v = qvals.clone();
                        v[d] = v[d] + FixedPoint::from_int(sign) * delta;
                        probes.push(v);
                    }
                }

                let mut covered_all = true;
                for probe in &probes {
                    let center = view.hilbert_position(hilbert_from_values(probe, bounds));
                    let lo = center.saturating_sub(half);
                    let hi = (center + half).min(n);
                    if lo > 0 || hi < n {
                        covered_all = false;
                    }
                    for i in lo..hi {
                        if seen.insert(i) {
                            consider(i, &view, &mut best, &mut scanned)?;
                        }
                    }
                }
                if best.len() >= k * 2 || covered_all {
                    break;
                }
                half *= 2;
                delta = delta * FixedPoint::from_int(2);
            }
        } else {
            for i in 0..n {
                consider(i, &view, &mut best, &mut scanned)?;
            }
        }
        p.scanned.store(scanned, Ordering::Relaxed);

        // Squared ranking is order-identical to true distance, so taking the
        // top k here loses nothing — then pay the sqrt k times, not n times.
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        best.truncate(k);
        for (key, d) in best.iter_mut() {
            *d = match view.by_key.get(key).copied() {
                Some(i) => {
                    let sem = view.semantic_of(i)?;
                    let layout = view.layout();
                    if layout.quantized {
                        let full = layout.decode_tail(sem)?;
                        Store::semantic_distance(query_coords, &full, dim_range.clone())
                    } else {
                        Store::semantic_distance(query_coords, sem, dim_range.clone())
                    }
                }
                // Not in the mmap view (shouldn't happen — it came from it);
                // fall back to the squared value's root rather than lie.
                None => d.sqrt(),
            };
        }

        // Merge the overlay store's own semantic results (promoted/new nodes).
        // Those already carry TRUE distances, so merge after the conversion.
        let overlay_hits = overlay
            .nearest_semantic(query_coords, k, dim_range)
            .unwrap_or_default();
        best.extend(overlay_hits);

        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        best.truncate(k);
        Ok(best)
    }
}

impl HoronCore {
    /// Snapshot entries examined by the most recent semantic query in
    /// partial mode — the meaning-addressed thesis metric: nearby meaning
    /// should mean a small byte-neighborhood, not a full scan.
    pub fn last_semantic_scan_count(&self) -> Option<usize> {
        self.partial.as_ref().map(|p| p.scanned.load(Ordering::Relaxed))
    }

    // ---- WAL tailing / replication ----

    /// Subscribe to the live WAL stream. Returns `(next_seq, receiver)`:
    /// every entry with `seq >= next_seq` will be delivered on the receiver,
    /// strictly AFTER it is durable on the primary (post-fsync), in
    /// sequence order.
    ///
    /// Replication pattern: subscribe first, then catch up with
    /// [`wal_entries_since`](Self::wal_entries_since) for everything below
    /// `next_seq` — the two streams meet exactly, no gap and no overlap.
    /// Pending un-flushed entries are flushed at subscription time to make
    /// that invariant hold. Drop the receiver to unsubscribe.
    ///
    /// GACL note: the WAL stream is the **physical** replication log and is
    /// **not** access-filtered — every committed entry is delivered regardless
    /// of the subscriber's credentials (a replica must receive the full log to
    /// stay byte-consistent with the primary; filtering would make it
    /// diverge). Treat WAL subscription as a trusted-peer channel. GACL bands
    /// scope the query surface (`nearest*`, `get`, `list`), not replication.
    pub fn subscribe_wal(
        &self,
    ) -> HoronResult<(u32, std::sync::mpsc::Receiver<WalEntry>)> {
        let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        wal.flush_pending()?;
        let (tx, rx) = std::sync::mpsc::channel();
        wal.subscribers.push(tx);
        Ok((wal.next_seq, rx))
    }

    /// Read committed WAL entries with `seq >= from_seq` from disk — the
    /// catch-up half of replication.
    ///
    /// If compaction has folded `from_seq` into the snapshot, the WAL no
    /// longer reaches back that far: `SnapshotRequired` tells the replica to
    /// bootstrap by copying the file (or full state) and then tail from the
    /// returned `base_seq`.
    ///
    /// Like [`subscribe_wal`](Self::subscribe_wal), this returns the physical
    /// log unfiltered by GACL — it is the replication catch-up channel, not a
    /// query surface.
    pub fn wal_entries_since(&self, from_seq: u32) -> HoronResult<WalTail> {
        let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        wal.flush_pending()?;

        let base_seq = wal.next_seq - wal.entry_count;
        if from_seq < base_seq {
            return Ok(WalTail::SnapshotRequired { base_seq });
        }

        let end_pos = wal.file.stream_position()?;
        let data_offset = wal.wal_data_offset;
        wal.file.seek(SeekFrom::Start(data_offset))?;

        let layout = self.sem_layout();
        let mut out = Vec::new();
        let wal_compressed = wal.wal_compressed;
        let algo = wal.compression_algo;
        let entry_count = wal.entry_count;
        let scan_result = scan_wal_raw(
            &mut wal.file,
            wal_compressed,
            algo,
            &layout,
            base_seq,
            entry_count,
            |entry| {
                if entry.seq >= from_seq {
                    out.push(entry.clone());
                }
                Ok(())
            },
        );
        // Always restore the append position, even if the scan errored.
        // (Read-only path: the valid-end offset is intentionally unused.)
        wal.file.seek(SeekFrom::Start(end_pos))?;
        scan_result?;

        Ok(WalTail::Entries(out))
    }

    // ---- Persistence operations ----

    /// Compact: fold snapshot + WAL into a new snapshot, reset WAL.
    ///
    /// Writes continue during compaction — concurrent entries are captured via
    /// a sequence-number fence and re-appended after the new snapshot.
    /// Only one compaction runs at a time (AtomicBool gate).
    ///
    /// Returns `Ok(true)` when this call performed the compaction and
    /// `Ok(false)` when it was skipped because another compaction was already
    /// in progress (that other run does the work). A skip is not a failure, so
    /// callers that must guarantee their own compaction ran should check the
    /// boolean rather than treating any `Ok` as "I compacted".
    pub fn compact(&self) -> HoronResult<bool> {
        if self.compacting.compare_exchange(
            false, true, Ordering::SeqCst, Ordering::SeqCst,
        ).is_err() {
            return Ok(false);
        }
        let result = self.compact_inner();
        self.compacting.store(false, Ordering::SeqCst);
        {
            let mut last = self
                .last_compaction_error
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            match &result {
                Ok(()) => *last = None,
                Err(e) => {
                    log::error!("compaction failed: {} — WAL will keep growing until a compaction succeeds", e);
                    *last = Some(e.to_string());
                }
            }
        }
        result.map(|()| true)
    }

    /// The most recent compaction failure, or `None` if the last compaction
    /// succeeded (or none has run). Auto-compaction triggered by writes
    /// records its errors here instead of failing the write.
    pub fn last_compaction_error(&self) -> Option<String> {
        self.last_compaction_error
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Background compaction — spawns a thread and returns a join handle.
    /// The joined result mirrors [`compact`](Self::compact): `Ok(true)` if it
    /// compacted, `Ok(false)` if another compaction was already running.
    pub fn compact_async(&self) -> JoinHandle<HoronResult<bool>> {
        let this = self
            .self_weak
            .get()
            .and_then(Weak::upgrade)
            .expect("Horon must be constructed via Horon::open()");
        std::thread::spawn(move || this.compact())
    }

    /// Fire-and-forget background compaction for auto-compact triggers.
    /// Runs off the writer thread; errors are logged and recorded in
    /// `last_compaction_error()` by `compact()` itself. The thread holds a
    /// strong reference to the core, so the file stays open (and the final
    /// WAL flush deferred) until the compaction finishes.
    fn spawn_auto_compact(&self) {
        // Cheap dedup: if a compaction is already running, the CAS in
        // compact() would make the new thread a no-op anyway — skip spawning.
        if self.compacting.load(Ordering::SeqCst) {
            return;
        }
        match self.self_weak.get().and_then(Weak::upgrade) {
            Some(core) => {
                std::thread::spawn(move || {
                    let _ = core.compact();
                });
            }
            // Unreachable via the public API; fall back to synchronous.
            None => {
                let _ = self.compact();
            }
        }
    }

    fn compact_inner(&self) -> HoronResult<()> {
        let layout = self.sem_layout();
        // In-memory entries always carry full-width tails; `layout` decides
        // the on-disk encoding at serialization time.
        let sem_bytes = layout.mem_bytes();
        let compressed = self.config.compression;

        // Step 1: Record fence under WAL lock (brief hold)
        let (fence_seq, fence_file_pos) = {
            let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
            wal.flush_pending()?;
            let fence_seq = wal.next_seq;
            let fence_file_pos = wal.file.stream_position()?;
            (fence_seq, fence_file_pos)
        };
        // WAL lock released — writes continue

        // Step 2: Snapshot from Store (lock-free DashMap iteration). In
        // partial mode the Store is only an overlay — merge in every
        // snapshot-resident entry that isn't shadowed or tombstoned,
        // decoding straight from the mmap.
        let mut entries = Vec::new();
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            let tombs = p.tombstones.read().unwrap_or_else(|e| e.into_inner());
            for (i, loc) in view.entries.iter().enumerate() {
                if !self.store.exists(&loc.key) && !tombs.contains(&loc.key) {
                    entries.push(view.decode(i)?);
                }
            }
        }
        let all_keys = self.store.list("/")?;
        entries.reserve(all_keys.len());
        for key in &all_keys {
            // A key deleted between list() and here must not abort the
            // whole compaction — skip it; its DELETE is in the WAL.
            let data = match self.store.get(key) {
                Ok(d) => d,
                Err(horon_engine::store::StoreError::NotFound(_)) => continue,
                Err(e) => return Err(e.into()),
            };
            let meta = self.store.get_meta(key)?;
            // Deterministic snapshot bytes — the claim behind
            // WAL-as-replication is that the same logical state produces
            // byte-identical files on any machine. Two leaks are closed here:
            // HashMap iteration order (sort) and the synthetic wall-clock
            // fields get_meta() injects (filter — replay discards them
            // anyway, so they never survived a reload).
            let mut metadata: Vec<(String, String)> = meta
                .into_iter()
                .filter(|(k, _)| {
                    k != "key" && k != "size" && k != "created_at" && k != "updated_at"
                })
                .collect();
            metadata.sort();
            let semantic_coords = self.store.get_semantic(key)
                .unwrap_or_default();
            let semantic_coords = if semantic_coords.len() == sem_bytes {
                semantic_coords
            } else {
                // Pad or truncate to expected size
                let mut padded = vec![0u8; sem_bytes];
                let copy_len = semantic_coords.len().min(sem_bytes);
                padded[..copy_len].copy_from_slice(&semantic_coords[..copy_len]);
                padded
            };
            entries.push(NodeEntry {
                key: key.clone(),
                data,
                metadata,
                semantic_coords,
            });
        }

        // Ordering.
        //
        // Meaning-addressed files (v3): PURE global-Hilbert order — a node's
        // position on disk is a function of its semantic coordinates. The
        // full-geometry reader restores depth order in memory before Sarkar
        // replay, so no replay invariant is lost.
        //
        // Ordinary files: (depth, per-snapshot hilbert, key). Depth is the
        // replay invariant (parents before children); within a depth level
        // Hilbert makes semantically similar entries byte-adjacent; key
        // keeps the order total and deterministic.
        let entries: Vec<NodeEntry> = if let Some(bounds) = self.ma_bounds {
            let sem_dims = self.header.semantic_dims as usize;
            let mut indexed: Vec<(u128, NodeEntry)> = entries
                .into_iter()
                .map(|e| (global_hilbert(&e.semantic_coords, sem_dims, bounds), e))
                .collect();
            indexed.sort_by(|(ra, a), (rb, b)| ra.cmp(rb).then(a.key.cmp(&b.key)));
            indexed.into_iter().map(|(_, e)| e).collect()
        } else {
            let ranks = hilbert_snapshot_ranks(&entries, self.header.semantic_dims as usize);
            let mut indexed: Vec<(u128, NodeEntry)> = ranks.into_iter().zip(entries).collect();
            indexed.sort_by(|(rank_a, a), (rank_b, b)| {
                let depth_a = a.key.matches('/').count();
                let depth_b = b.key.matches('/').count();
                depth_a
                    .cmp(&depth_b)
                    .then(rank_a.cmp(rank_b))
                    .then(a.key.cmp(&b.key))
            });
            indexed.into_iter().map(|(_, e)| e).collect()
        };

        // Step 3: Write snapshot to tempfile
        let tmp_path = self.path.with_extension("htt.tmp");
        let mut tmp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        // Lock the tempfile now: after the rename it IS the live file, and
        // reusing this handle below means the lock transfers with no gap.
        try_lock_exclusive(&tmp_file, &tmp_path)?;

        let mut header = self.header.clone();
        header.node_count = entries.len() as u32;
        // Compaction rewrites the whole file, so upgrade to the current
        // format version — v4 for quantized files, v3 for meaning-addressed,
        // else v2.
        header.version = if layout.quantized {
            VERSION_QUANTIZED
        } else if self.ma_bounds.is_some() {
            VERSION_MEANING_ADDRESSED
        } else {
            VERSION
        };
        tmp_file.write_all(&header.to_bytes())?;
        if let Some(bounds) = self.ma_bounds {
            tmp_file.write_all(&Self::bounds_section_bytes(
                self.header.semantic_dims as usize,
                bounds,
            ))?;
        }
        let snap_raw_len: usize = {
            // Needed to rebuild the partial view after the rename.
            let mut probe = Vec::new();
            for e in &entries {
                e.write_to(&mut probe, &layout)?;
            }
            probe.len()
        };
        snapshot::write_snapshot(&mut tmp_file, &entries, compressed, true, &layout)?;

        // Step 4: Acquire WAL lock for file swap
        let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        wal.flush_pending()?;

        // Read concurrent WAL entries (written since fence)
        let concurrent_count = wal.next_seq - fence_seq;
        let mut concurrent_entries = Vec::new();
        if concurrent_count > 0 {
            wal.file.seek(SeekFrom::Start(fence_file_pos))?;
            if wal.wal_compressed {
                let algo = wal.compression_algo;
                while concurrent_entries.len() < concurrent_count as usize {
                    match wal::read_wal_block(&mut wal.file, algo)? {
                        Some((block_data, block_entry_count)) => {
                            let mut cursor = std::io::Cursor::new(&block_data);
                            for _ in 0..block_entry_count {
                                if let Some(entry) = WalEntry::read_from(&mut cursor, &layout)? {
                                    concurrent_entries.push(entry);
                                }
                            }
                        }
                        None => break,
                    }
                }
            } else {
                for _ in 0..concurrent_count {
                    match WalEntry::read_from(&mut wal.file, &layout)? {
                        Some(entry) => concurrent_entries.push(entry),
                        None => break,
                    }
                }
            }
        }

        // Temporal epochs: archive the pre-fence WAL span into a history sidecar BEFORE
        // the rename discards it (still in the fallible-and-abortable zone).
        // The archived span is [old base_seq, fence_seq); concurrent entries
        // stay in the live WAL only, so spans do not overlap in the normal
        // path. A crash after this write but before the rename re-archives
        // the same span next time under a new segment number — harmless,
        // HoronHistory deduplicates by sequence number.
        if self.config.history_retention == HistoryRetention::Archive {
            let old_header_pos = wal.wal_data_offset - 8;
            wal.file.seek(SeekFrom::Start(old_header_pos))?;
            let (_old_count, old_base_seq) = wal::read_wal_header(&mut wal.file)?;
            if fence_seq > old_base_seq {
                let mut archived: Vec<WalEntry> = Vec::new();
                if wal.wal_compressed {
                    let algo = wal.compression_algo;
                    'archive: while wal.file.stream_position()? < fence_file_pos {
                        match wal::read_wal_block(&mut wal.file, algo)? {
                            Some((block, n)) => {
                                let mut cursor = std::io::Cursor::new(&block);
                                for _ in 0..n {
                                    match WalEntry::read_from(&mut cursor, &layout)? {
                                        Some(e) if e.seq < fence_seq => archived.push(e),
                                        _ => break 'archive,
                                    }
                                }
                            }
                            None => break,
                        }
                    }
                } else {
                    while wal.file.stream_position()? < fence_file_pos {
                        match WalEntry::read_from(&mut wal.file, &layout)? {
                            Some(e) if e.seq < fence_seq => archived.push(e),
                            _ => break,
                        }
                    }
                }
                if !archived.is_empty() {
                    let n = crate::history::next_segment_number(&self.path);
                    crate::history::write_segment(
                        &self.path,
                        n,
                        layout,
                        old_base_seq,
                        fence_seq,
                        &archived,
                    )?;
                }
            }
        }

        // Write WAL header + concurrent entries to tempfile
        wal::write_wal_header(&mut tmp_file, concurrent_entries.len() as u32, fence_seq)?;
        let new_wal_data_offset = tmp_file.stream_position()?;
        if wal.wal_compressed && !concurrent_entries.is_empty() {
            let algo = wal.compression_algo;
            let serialized: Vec<Vec<u8>> = concurrent_entries
                .iter()
                .map(|e| {
                    let mut buf = Vec::new();
                    e.write_to(&mut buf, &layout).unwrap();
                    buf
                })
                .collect();
            for chunk in serialized.chunks(WAL_BLOCK_SIZE) {
                let mut block_bytes = Vec::new();
                for entry_bytes in chunk {
                    block_bytes.extend_from_slice(entry_bytes);
                }
                wal::write_wal_block(&mut tmp_file, &block_bytes, chunk.len() as u16, algo)?;
            }
        } else {
            for entry in &concurrent_entries {
                entry.write_to(&mut tmp_file, &layout)?;
            }
        }
        tmp_file.flush()?;
        tmp_file.sync_all()?;

        // Keep using the tempfile handle — after the rename it IS the live
        // file (same inode), so no reopen is needed and the advisory lock
        // carries over with no window for another process to sneak in.
        let mut new_file = tmp_file;
        new_file.seek(SeekFrom::End(0))?;

        // Build the new partial view over the tempfile BEFORE the rename, so a
        // failure here aborts without having swapped the on-disk file. The
        // mmap holds the inode, which the rename keeps (rename preserves the
        // inode), so it stays valid across the rename. Doing this after the
        // rename would leave the on-disk file replaced while the in-memory WAL
        // handle still pointed at the old, now-unlinked inode — subsequent
        // writes would vanish on reopen (post-rename atomicity, hardening audit).
        let new_view = if self.partial.is_some() {
            let mmap = unsafe { memmap2::Mmap::map(&new_file)? };
            let sem_dims = self.header.semantic_dims as usize;
            let raw_start = HEADER_SIZE
                + self.ma_bounds.map_or(0, |_| {
                    (sem_dims - DIM_USER_DEFINED_START) * 16
                })
                + SNAP_HEADER_SIZE;
            let hilbert_fn;
            let hilbert_of: Option<&dyn Fn(&[u8]) -> u128> = match self.ma_bounds {
                Some(b) => {
                    // Same decode-first rule as the open_partial scan.
                    hilbert_fn = move |sem: &[u8]| {
                        if layout.quantized {
                            match layout.decode_tail(sem) {
                                Ok(full) => global_hilbert(&full, sem_dims, b),
                                Err(_) => 0,
                            }
                        } else {
                            global_hilbert(sem, sem_dims, b)
                        }
                    };
                    Some(&hilbert_fn)
                }
                None => None,
            };
            Some(SnapView::scan(
                mmap,
                raw_start,
                snap_raw_len,
                entries.len(),
                layout,
                hilbert_of,
            )?)
        } else {
            None
        };

        // Point of no return: everything above is fallible-and-abortable;
        // everything below adopts the new file into the in-memory WAL state
        // and must not early-return before that adoption completes, or the two
        // would diverge. `fsync_dir` is therefore deferred to the very end —
        // after adoption — so that even a dir-fsync failure leaves in-memory
        // and the visible on-disk file in agreement (it degrades durability,
        // not consistency).
        //
        // rename is atomic on POSIX; the directory entry becomes durable at
        // the closing fsync_dir.
        fs::rename(&tmp_path, &self.path)?;

        // Swap in the pre-built partial view (infallible now).
        if let Some(p) = &self.partial {
            if let Some(view) = new_view {
                *p.view.write().unwrap_or_else(|e| e.into_inner()) = view;
            }
            // Drop only tombstones for keys the new snapshot no longer
            // contains. Clearing ALL of them resurrected keys deleted while
            // this compaction ran: their DELETE was re-appended to the new
            // WAL, but the merged snapshot still held the entry, so the live
            // handle served deleted data (hardening audit). Lock order (tombstones
            // write, then view read) is nested only here; all other sites
            // take these locks one at a time.
            {
                let mut tombs = p.tombstones.write().unwrap_or_else(|e| e.into_inner());
                let view = p.view.read().unwrap_or_else(|e| e.into_inner());
                tombs.retain(|k| view.by_key.contains_key(k));
            }
        }

        // Update WalWriter state
        wal.file = new_file;
        wal.entry_count = concurrent_entries.len() as u32;
        wal.wal_data_offset = new_wal_data_offset;
        wal.pending_serialized.clear();
        wal.pending_count = 0;
        // next_seq continues — no reset

        // Make the rename durable now that in-memory state has adopted the new
        // file. A failure here means reduced crash-durability for this one
        // compaction (the caller will retry), not an inconsistent handle.
        fsync_dir(&self.path)?;

        // Temporal epochs: re-stamp the epoch counter into the fresh WAL so it survives
        // the truncation. Uses the normal append path (durability mode,
        // subscriber fan-out, and the u32 seq guard all apply). Only when
        // epochs are in use — a file that never sealed an epoch stays
        // byte-identical to pre-epochs compaction output.
        let epoch = self.current_epoch.load(Ordering::SeqCst);
        if epoch > 0 {
            let entry = WalEntry {
                seq: wal.next_seq,
                op: OP_EPOCH,
                key: String::new(),
                payload: WalPayload::Epoch { epoch_id: epoch, flags: EPOCH_FLAG_RESTAMP },
            };
            wal.append(&entry)?;
            wal.flush_pending()?;
        }

        Ok(())
    }

    /// Flush all pending WAL entries to disk.
    pub fn flush(&self) -> HoronResult<()> {
        let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        wal.flush_pending()?;
        wal.file.flush()?;
        Ok(())
    }

    // ---- Temporal epochs (docs/TEMPORAL_EPOCHS.md) ----

    /// Seal the current state as a new epoch and return its id.
    ///
    /// Appends an `OP_EPOCH` marker to the WAL — a logical, monotonically
    /// increasing counter (not wall-clock, preserving determinism). The
    /// marker means "the manifold state as of here is a coherent
    /// calibration": temporal readers ([`crate::history::HoronHistory`])
    /// sample state at epoch seals. Pending WAL entries are flushed through
    /// first, so the seal always sits after everything written before it.
    ///
    /// Without `history_retention: Archive`, epochs still work within the
    /// live WAL, but compaction discards the pre-fence history (the epoch
    /// *counter* itself survives compaction either way).
    pub fn seal_epoch(&self) -> HoronResult<u64> {
        self.seal_epoch_inner(0)
    }

    /// Seal a **speculative** epoch (projected / what-if state, not recorded
    /// history). The flag labels — it does not isolate: speculative writes
    /// belong in a copy of the file, and this bit ensures a labeled span can
    /// never silently pass as recorded fact. Temporal sampling APIs skip
    /// speculative epochs unless explicitly asked.
    pub fn seal_speculative_epoch(&self) -> HoronResult<u64> {
        self.seal_epoch_inner(EPOCH_FLAG_SPECULATIVE)
    }

    fn seal_epoch_inner(&self, flags: u8) -> HoronResult<u64> {
        let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        // The counter is only advanced under the WAL lock, so load-add-store
        // here is race-free.
        let epoch_id = self.current_epoch.load(Ordering::SeqCst) + 1;
        let entry = WalEntry {
            seq: wal.next_seq,
            op: OP_EPOCH,
            key: String::new(),
            payload: WalPayload::Epoch { epoch_id, flags },
        };
        wal.append(&entry)?;
        // A seal is a durability boundary: flush it through regardless of
        // the batch configuration.
        wal.flush_pending()?;
        self.current_epoch.store(epoch_id, Ordering::SeqCst);
        Ok(epoch_id)
    }

    /// The current epoch counter (0 = no epoch ever sealed).
    pub fn current_epoch(&self) -> u64 {
        self.current_epoch.load(Ordering::SeqCst)
    }

    // ---- On-demand geometry (embed-on-demand) ----

    /// Upgrade a lazily-loaded key to a full geometric embedding, in place,
    /// so spatial queries (`nearest`, `neighbors`, `find_in_radius`) can see
    /// it without a full-geometry reopen.
    ///
    /// Files opened with `lazy_geometry` (or `partial_reads`) replay nodes
    /// data-only — semantic queries work, spatial ones do not. This computes
    /// the node's Sarkar placement on demand: missing ancestors embed first;
    /// key, value, metadata, and semantic coordinates are preserved. In
    /// partial-reads mode, snapshot-resident nodes (and their ancestors) are
    /// promoted into the overlay first.
    ///
    /// Geometry is DERIVED state and is not written to the WAL or the file:
    /// after a reopen the node is data-only again until re-embedded. The
    /// placement is deterministic for a fixed call sequence within a session.
    /// Returns whether this call performed the upgrade.
    pub fn embed(&self, key: &str) -> HoronResult<bool> {
        self.check_read(key)?;
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            ensure_ancestors(&self.store, &view, key);
            promote_from_view(&self.store, &view, key)?;
        }
        Ok(self.store.embed_existing(key)?)
    }

    /// Embed the prefix node and every key under it (convenience) —
    /// see [`Self::embed`]. Returns how many keys this call upgraded.
    pub fn embed_all(&self, prefix: &str) -> HoronResult<usize> {
        let mut upgraded = 0;
        let mut keys = self.list(prefix)?;
        if !keys.iter().any(|k| k == prefix) {
            keys.push(prefix.to_string());
        }
        keys.sort();
        for key in keys {
            match self.embed(&key) {
                Ok(true) => upgraded += 1,
                Ok(false) => {}
                // list() can surface keys GACL hides from this session;
                // skip them rather than failing the sweep.
                Err(HoronError::AccessDenied { .. }) => {}
                Err(e) => return Err(e),
            }
        }
        Ok(upgraded)
    }

    /// WAL entry count since last compaction (including pending unflushed entries).
    pub fn wal_len(&self) -> u32 {
        let wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        wal.total_entries()
    }

    /// File path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for HoronCore {
    fn drop(&mut self) {
        // Recover a poisoned lock like every other WAL site — skipping the
        // final flush after an unrelated panic would drop pending entries.
        let mut wal = self.wal.lock().unwrap_or_else(|e| e.into_inner());
        let _ = wal.flush_pending();
        let _ = wal.file.flush();
    }
}

/// fsync the directory containing `path`, making a preceding create or
/// rename durable. POSIX: rename() is atomic but its directory entry is not
/// durable until the parent directory is fsynced. No-op on non-unix.
fn fsync_dir(path: &Path) -> HoronResult<()> {
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            let dir = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };
            File::open(dir)?.sync_all()?;
        }
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Take a non-blocking exclusive advisory lock (flock) on the file, so two
/// processes cannot open the same .htt read-write and corrupt each other.
/// The lock lives as long as the file descriptor. No-op on non-unix
/// platforms (documented limitation).
fn try_lock_exclusive(file: &File, path: &Path) -> HoronResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            if err.kind() == std::io::ErrorKind::WouldBlock {
                return Err(HoronError::Locked(path.display().to_string()));
            }
            return Err(HoronError::Io(err));
        }
    }
    #[cfg(not(unix))]
    let _ = (file, path);
    Ok(())
}

/// Remove an orphaned compaction tempfile left behind by a crash.
/// Only called after the main file's advisory lock is held, so no live
/// compaction can own the tempfile.
fn cleanup_orphan_tmp(path: &Path) {
    crate::history::cleanup_orphan_segment_tmp(path);
    let tmp = path.with_extension("htt.tmp");
    if tmp.exists() {
        match fs::remove_file(&tmp) {
            Ok(()) => log::warn!(
                "removed orphaned compaction tempfile {} (crash during a previous compaction)",
                tmp.display()
            ),
            Err(e) => log::warn!("could not remove orphaned tempfile {}: {}", tmp.display(), e),
        }
    }
}

/// Compute a Hilbert index per snapshot entry from its user semantic
/// dimensions (dims `DIM_USER_DEFINED_START..`, capped at 8 axes — the
/// mapper's practical resolution/index-width sweet spot).
///
/// Coordinates are normalized per-snapshot (min/max over this entry set).
/// That is deliberately NOT a stable global address — Phase 1 only *orders*
/// entries within a snapshot, so relative position is all that matters and
/// per-snapshot bounds are deterministic for a given node set. Entries
/// without user dims (or files with none configured) all rank 0 and fall
/// back to the key tiebreaker.
///
/// f64 is safe here: only IEEE-basic ops (sub/div/mul/round), which are
/// bit-deterministic across conforming platforms — no transcendentals.
fn hilbert_snapshot_ranks(entries: &[NodeEntry], semantic_dims: usize) -> Vec<u128> {
    const MAX_HILBERT_DIMS: usize = 8;
    const HILBERT_BITS: u32 = 12;

    let user_dims = semantic_dims
        .saturating_sub(DIM_USER_DEFINED_START)
        .min(MAX_HILBERT_DIMS);
    if user_dims == 0 || entries.is_empty() {
        return vec![0; entries.len()];
    }

    // Decode user dims straight from the stored Q64.64 raw — no f64 step,
    // so two coordinates the format can distinguish cannot collapse onto one
    // address, and the snapshot ordering is a function of the bytes rather
    // than of float arithmetic.
    let decode = |coords: &[u8]| -> Vec<FixedPoint> {
        (0..user_dims)
            .map(|d| {
                let start = (DIM_USER_DEFINED_START + d) * 16;
                let end = start + 16;
                if coords.len() >= end {
                    FixedPoint::from_raw(i128::from_le_bytes(
                        coords[start..end].try_into().unwrap(),
                    ))
                } else {
                    FixedPoint::from_int(0)
                }
            })
            .collect()
    };
    let decoded: Vec<Vec<FixedPoint>> =
        entries.iter().map(|e| decode(&e.semantic_coords)).collect();

    let one = FixedPoint::from_int(1);
    let half = one / FixedPoint::from_int(2);
    let epsilon = FixedPoint::from_f64(1e-12);

    let mut mins = vec![FixedPoint::from_raw(i128::MAX); user_dims];
    let mut maxs = vec![FixedPoint::from_raw(i128::MIN); user_dims];
    for coords in &decoded {
        for i in 0..user_dims {
            if coords[i] < mins[i] {
                mins[i] = coords[i];
            }
            if coords[i] > maxs[i] {
                maxs[i] = coords[i];
            }
        }
    }

    let mapper = HilbertMapper::new(user_dims, HILBERT_BITS);
    decoded
        .iter()
        .map(|coords| {
            let norm: Vec<FixedPoint> = (0..user_dims)
                .map(|i| {
                    let range = maxs[i] - mins[i];
                    if range < epsilon {
                        half
                    } else {
                        (coords[i] - mins[i]) / range
                    }
                })
                .collect();
            mapper.coords_to_index_fixed(&norm).value()
        })
        .collect()
}

/// Scan WAL entries from the file's current position, applying each valid
/// entry, and return `(next_seq, valid_count)`.
///
/// The 8-byte WAL header is rewritten in place on every flush and carries no
/// checksum — a torn header write could undercount and silently drop
/// committed entries if we trusted it. So the header count is ADVISORY: the
/// scan runs until clean EOF, a bad per-entry CRC, a sequence-number break,
/// or a parse failure (torn tail). Committed, CRC-valid entries are never
/// dropped by a torn header.
/// Materialize decoded snapshot entries into a `Store`.
///
/// Shared by the read-write open path and [`HoronReader`] so the two cannot
/// drift: a reader that reconstructed geometry differently from the writer
/// would report different `nearest()` answers for the same file, and nothing
/// would flag it. `_child_index` replay is what makes this deterministic
/// regardless of entry order (see the Sarkar reconstruction test).
fn load_snapshot_into_store(
    store: &Store,
    snap_entries: &[snapshot::NodeEntry],
    lazy_geometry: bool,
) -> HoronResult<()> {
    for entry in snap_entries {
        if lazy_geometry {
            Store::put_data_only(store, &entry.key, &entry.data).map_err(HoronError::from)?;
        } else {
            // Use stored child_index for deterministic Sarkar reconstruction
            let child_index = entry
                .metadata
                .iter()
                .find(|(k, _)| k == "_child_index")
                .and_then(|(_, v)| v.parse::<u32>().ok());

            if let Some(idx) = child_index {
                store
                    .put_positioned(&entry.key, &entry.data, idx)
                    .map_err(HoronError::from)?;
            } else {
                Store::put(store, &entry.key, &entry.data).map_err(HoronError::from)?;
            }
        }
        for (mk, mv) in &entry.metadata {
            if mk != "key"
                && mk != "size"
                && mk != "created_at"
                && mk != "updated_at"
                && mk != "_child_index"
            {
                let _ = store.set_meta(&entry.key, mk, mv);
            }
        }
        if !entry.semantic_coords.is_empty() && entry.semantic_coords.iter().any(|&b| b != 0) {
            let _ = store.set_semantic(&entry.key, entry.semantic_coords.clone());
        }
    }
    Ok(())
}

fn scan_wal<F>(
    file: &mut File,
    header: &GeoHeader,
    layout: &SemLayout,
    wal_base_seq: u32,
    wal_entry_count: u32,
    apply: F,
) -> HoronResult<(u32, u32, u64)>
where
    F: FnMut(&WalEntry) -> HoronResult<()>,
{
    scan_wal_raw(
        file,
        header.wal_compressed(),
        header.compression_algo(),
        layout,
        wal_base_seq,
        wal_entry_count,
        apply,
    )
}

/// `scan_wal` without a header — for callers that know the compression
/// settings directly (e.g. the live WalWriter).
fn scan_wal_raw<F>(
    file: &mut File,
    wal_compressed: bool,
    algo: u8,
    layout: &SemLayout,
    wal_base_seq: u32,
    wal_entry_count: u32,
    mut apply: F,
) -> HoronResult<(u32, u32, u64)>
where
    F: FnMut(&WalEntry) -> HoronResult<()>,
{
    let mut next_seq = wal_base_seq;
    let mut valid: u32 = 0;
    let mut expected_seq: Option<u32> = None;
    // Byte offset (from file start) where the valid WAL region ends. Callers
    // recovering a file truncate to this so appends land after the last valid
    // entry instead of after a torn tail the next scan would stop at.
    let mut valid_end = file.stream_position()?;

    let accept = |entry: WalEntry,
                  expected_seq: &mut Option<u32>,
                  next_seq: &mut u32,
                  valid: &mut u32,
                  apply: &mut F|
     -> HoronResult<bool> {
        if let Some(exp) = *expected_seq {
            if entry.seq != exp {
                log::warn!(
                    "WAL scan stopped: sequence break (expected {}, found {})",
                    exp, entry.seq
                );
                return Ok(false);
            }
        }
        *expected_seq = Some(entry.seq + 1);
        *next_seq = entry.seq + 1;
        apply(&entry)?;
        *valid += 1;
        Ok(true)
    };

    if wal_compressed {
        // Blocks are accepted all-or-nothing: a block with a torn entry or an
        // internal sequence break is discarded whole, so the valid region
        // always ends on a block boundary (writers emit whole blocks
        // atomically; a partially-valid block only arises from corruption).
        'blocks: loop {
            match wal::read_wal_block(file, algo) {
                Ok(Some((block_data, block_entry_count))) => {
                    // Parse and validate the whole block before applying.
                    let mut cursor = std::io::Cursor::new(&block_data);
                    let mut entries = Vec::with_capacity(block_entry_count as usize);
                    for _ in 0..block_entry_count {
                        match WalEntry::read_from(&mut cursor, layout) {
                            Ok(Some(entry)) => entries.push(entry),
                            Ok(None) => break 'blocks,
                            Err(e) => {
                                log::warn!("WAL scan stopped: torn entry in block ({})", e);
                                break 'blocks;
                            }
                        }
                    }
                    // Sequence continuity across the block (including the
                    // boundary with the previous block).
                    let first = match entries.first() {
                        Some(e) => e.seq,
                        None => break 'blocks,
                    };
                    if let Some(exp) = expected_seq {
                        if first != exp {
                            log::warn!(
                                "WAL scan stopped: sequence break at block boundary (expected {}, found {})",
                                exp, first
                            );
                            break 'blocks;
                        }
                    }
                    if entries.iter().enumerate().any(|(i, e)| e.seq != first + i as u32) {
                        log::warn!("WAL scan stopped: sequence break inside block");
                        break 'blocks;
                    }
                    for entry in entries {
                        // Cannot fail the seq check (pre-validated above).
                        accept(entry, &mut expected_seq, &mut next_seq, &mut valid, &mut apply)?;
                    }
                    valid_end = file.stream_position()?;
                }
                Ok(None) => break,
                Err(e) => {
                    log::warn!("WAL scan stopped: unreadable block ({})", e);
                    break;
                }
            }
        }
    } else {
        loop {
            match WalEntry::read_from(file, layout) {
                Ok(Some(entry)) => {
                    if !accept(entry, &mut expected_seq, &mut next_seq, &mut valid, &mut apply)? {
                        break;
                    }
                    valid_end = file.stream_position()?;
                }
                Ok(None) => break,
                Err(e) => {
                    log::warn!("WAL scan stopped: torn tail entry ({})", e);
                    break;
                }
            }
        }
    }

    if valid != wal_entry_count {
        log::warn!(
            "WAL header count {} differs from scan result {} — header is advisory, scan wins",
            wal_entry_count, valid
        );
    }
    Ok((next_seq, valid, valid_end))
}


// ---------------------------------------------------------------------------
// Partial-mode read resolution, shared by Horon and HoronReader
// ---------------------------------------------------------------------------
// Each function resolves a read across the overlay store, the tombstone set,
// and the mmap snapshot view — extracted from Horon's partial branches so the
// lock-free reader answers reads through the SAME code instead of a copy that
// could drift. The writer's behavior is unchanged: its methods delegate here.

fn partial_get(store: &Store, p: &PartialState, key: &str) -> HoronResult<Vec<u8>> {
    if store.exists(key) {
        return Ok(store.get(key)?);
    }
    if !p.tombstones.read().unwrap_or_else(|e| e.into_inner()).contains(key) {
        let view = p.view.read().unwrap_or_else(|e| e.into_inner());
        if let Some(&idx) = view.by_key.get(key) {
            return Ok(view.decode(idx)?.data);
        }
    }
    Err(HoronError::Store(
        horon_engine::store::StoreError::NotFound(key.to_string()),
    ))
}

fn partial_exists(store: &Store, p: &PartialState, key: &str) -> bool {
    if store.exists(key) {
        return true;
    }
    if p.tombstones.read().unwrap_or_else(|e| e.into_inner()).contains(key) {
        return false;
    }
    p.view
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .by_key
        .contains_key(key)
}

fn partial_get_semantic(store: &Store, p: &PartialState, key: &str) -> HoronResult<Vec<u8>> {
    if store.exists(key) {
        return Ok(store.get_semantic(key)?);
    }
    if !p.tombstones.read().unwrap_or_else(|e| e.into_inner()).contains(key) {
        let view = p.view.read().unwrap_or_else(|e| e.into_inner());
        if let Some(&idx) = view.by_key.get(key) {
            let sem = view.semantic_of(idx)?; // on-disk encoding
            let layout = view.layout();
            let full = if layout.quantized {
                layout.decode_tail(sem)?
            } else {
                sem.to_vec()
            };
            // Parity with full mode: all-zero coords = "not set".
            if full.iter().any(|&b| b != 0) {
                return Ok(full);
            }
            return Ok(Vec::new());
        }
    }
    Err(HoronError::Store(
        horon_engine::store::StoreError::NotFound(key.to_string()),
    ))
}

fn partial_children(store: &Store, p: &PartialState, path: &str) -> HoronResult<Vec<String>> {
    let base = if path == "/" { String::new() } else { path.trim_end_matches('/').to_string() };
    let direct_child = |k: &str| -> bool {
        match k.strip_prefix(&base) {
            Some(rest) => rest.len() > 1 && rest.starts_with('/') && !rest[1..].contains('/'),
            None => false,
        }
    };
    // The overlay's data-only nodes don't maintain parent child
    // lists, so enumerate the store by prefix instead of children().
    let mut out: std::collections::BTreeSet<String> = store
        .list("/")
        .unwrap_or_default()
        .into_iter()
        .filter(|k| direct_child(k))
        .collect();
    let view = p.view.read().unwrap_or_else(|e| e.into_inner());
    let tombs = p.tombstones.read().unwrap_or_else(|e| e.into_inner());
    for e in &view.entries {
        if direct_child(&e.key) && !tombs.contains(&e.key) {
            out.insert(e.key.clone());
        }
    }
    Ok(out.into_iter().collect())
}

fn partial_list(store: &Store, p: &PartialState, prefix: &str) -> HoronResult<Vec<String>> {
    let mut out: std::collections::BTreeSet<String> =
        store.list(prefix).unwrap_or_default().into_iter().collect();
    let view = p.view.read().unwrap_or_else(|e| e.into_inner());
    let tombs = p.tombstones.read().unwrap_or_else(|e| e.into_inner());
    for e in &view.entries {
        if e.key.starts_with(prefix) && !tombs.contains(&e.key) {
            out.insert(e.key.clone());
        }
    }
    Ok(out.into_iter().collect())
}

/// Copy a snapshot-resident entry into the overlay Store (partial mode).
/// A node must be fully in the Store before per-field mutations (set_meta,
/// set_semantic, updates) so reads see a complete merged view.
fn promote_from_view(store: &Store, view: &SnapView, key: &str) -> HoronResult<bool> {
    if store.exists(key) {
        return Ok(true);
    }
    let Some(&idx) = view.by_key.get(key) else {
        return Ok(false);
    };
    let entry = view.decode(idx)?;
    Store::put_data_only(store, &entry.key, &entry.data).map_err(HoronError::from)?;
    for (mk, mv) in &entry.metadata {
        if mk != "key" && mk != "size" && mk != "created_at" && mk != "updated_at" {
            let _ = store.set_meta(&entry.key, mk, mv);
        }
    }
    if !entry.semantic_coords.is_empty() && entry.semantic_coords.iter().any(|&b| b != 0) {
        let _ = store.set_semantic(&entry.key, entry.semantic_coords.clone());
    }
    Ok(true)
}

/// Create any missing ancestors of `key` in the overlay store (partial
/// mode): `put_data_only` does not auto-create parents the way the full
/// geometric path does, and `children()` must be able to enumerate them.
fn ensure_ancestors(store: &Store, view: &SnapView, key: &str) {
    let mut path = String::new();
    let segments: Vec<&str> = key.trim_matches('/').split('/').collect();
    for seg in &segments[..segments.len().saturating_sub(1)] {
        path.push('/');
        path.push_str(seg);
        if !store.exists(&path) && !view.by_key.contains_key(&path) {
            let _ = Store::put_data_only(store, &path, b"");
        }
    }
}

/// Apply a WAL entry in partial mode: the Store is an overlay over the
/// mmap'd snapshot, and deletes of snapshot-resident keys become tombstones.
fn apply_wal_partial(
    store: &Store,
    view: &SnapView,
    tombstones: &mut HashSet<String>,
    entry: &WalEntry,
) -> HoronResult<()> {
    match &entry.payload {
        WalPayload::Insert(node) => {
            ensure_ancestors(store, view, &entry.key);
            Store::put_data_only(store, &entry.key, &node.data).map_err(HoronError::from)?;
            for (mk, mv) in &node.metadata {
                let _ = store.set_meta(&entry.key, mk, mv);
            }
            if !node.semantic_coords.is_empty() && node.semantic_coords.iter().any(|&b| b != 0) {
                let _ = store.set_semantic(&entry.key, node.semantic_coords.clone());
            }
            tombstones.remove(&entry.key);
        }
        WalPayload::Update { data, metadata } => {
            // Promote first so snapshot metadata/semantics survive the update.
            let _ = promote_from_view(store, view, &entry.key);
            ensure_ancestors(store, view, &entry.key);
            Store::put_data_only(store, &entry.key, data).map_err(HoronError::from)?;
            for (mk, mv) in metadata {
                let _ = store.set_meta(&entry.key, mk, mv);
            }
            tombstones.remove(&entry.key);
        }
        WalPayload::Delete => {
            if store.exists(&entry.key) {
                let _ = store.remove(&entry.key);
            }
            tombstones.insert(entry.key.clone());
        }
        WalPayload::SetMeta { meta_key, meta_value } => {
            if promote_from_view(store, view, &entry.key)? {
                let _ = store.set_meta(&entry.key, meta_key, meta_value);
            } else {
                log::warn!("WAL SetMeta for unknown key '{}' — skipped", entry.key);
            }
        }
        WalPayload::SetSemantic { coords } => {
            if promote_from_view(store, view, &entry.key)? {
                let _ = store.set_semantic(&entry.key, coords.clone());
            } else {
                log::warn!("WAL SetSemantic for unknown key '{}' — skipped", entry.key);
            }
        }
        WalPayload::Epoch { .. } => {
            // No node state change — the open closure tracks the counter.
        }
    }
    Ok(())
}

/// Replay a single WAL entry into the Store.
///
/// INSERT may hit AlreadyExists when a snapshot-WAL overlap occurs during
/// compaction — this is harmless and treated as a no-op.
fn replay_entry(store: &Store, entry: &WalEntry, lazy_geometry: bool) -> HoronResult<()> {
    match &entry.payload {
        WalPayload::Insert(node) => {
            let result = if lazy_geometry {
                Store::put_data_only(store, &entry.key, &node.data)
            } else {
                let child_index = node.metadata.iter()
                    .find(|(k, _)| k == "_child_index")
                    .and_then(|(_, v)| v.parse::<u32>().ok());
                if let Some(idx) = child_index {
                    store.put_positioned(&entry.key, &node.data, idx)
                } else {
                    Store::put(store, &entry.key, &node.data)
                }
            };
            match result {
                Ok(()) => {}
                Err(horon_engine::StoreError::AlreadyExists(_)) => {
                    // Snapshot-WAL overlap after compaction — safe to skip
                }
                Err(e) => return Err(HoronError::from(e)),
            }
            for (mk, mv) in &node.metadata {
                if mk != "_child_index" {
                    let _ = store.set_meta(&entry.key, mk, mv);
                }
            }
        }
        WalPayload::Update { data, metadata } => {
            // Updates go to existing nodes — no Sarkar embedding needed
            let put = if lazy_geometry { Store::put_data_only } else { Store::put };
            put(store, &entry.key, data)?;
            for (mk, mv) in metadata {
                let _ = store.set_meta(&entry.key, mk, mv);
            }
        }
        WalPayload::Delete => {
            let _ = store.remove(&entry.key);
        }
        WalPayload::SetMeta { meta_key, meta_value } => {
            let _ = store.set_meta(&entry.key, meta_key, meta_value);
        }
        WalPayload::SetSemantic { coords } => {
            if coords.iter().all(|&b| b == 0) {
                // The zero tail encodes "not set" — parity with snapshot
                // load, which skips all-zero tails. Without this, a cleared
                // placement replayed from the WAL would come back as "placed
                // at the origin" while the compacted form of the same state
                // comes back unset.
                let _ = store.set_semantic(&entry.key, Vec::new());
            } else {
                let _ = store.set_semantic(&entry.key, coords.clone());
            }
        }
        WalPayload::Epoch { .. } => {
            // No node state change — the open closure tracks the counter.
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {

/// Exact fixed-point coordinates from decimal literals.
fn fp(vals: &[f64]) -> Vec<g_math::fixed_point::FixedPoint> {
    vals.iter().map(|&v| g_math::fixed_point::FixedPoint::from_f64(v)).collect()
}

    use super::*;
    use tempfile::NamedTempFile;

    fn temp_path() -> PathBuf {
        NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
    }

    #[test]
    fn test_create_and_reopen() {
        let path = temp_path();

        {
            let gf = Horon::open(&path).unwrap();
            gf.put("/hello", b"world").unwrap();
            gf.put("/foo/bar", b"baz").unwrap();
            gf.set_meta("/hello", "author", "alice").unwrap();
            gf.flush().unwrap();
            assert_eq!(gf.len(), 3);
        }

        {
            let gf = Horon::open(&path).unwrap();
            assert_eq!(gf.get("/hello").unwrap(), b"world");
            assert_eq!(gf.get("/foo/bar").unwrap(), b"baz");
            assert_eq!(gf.len(), 3);
        }
    }

    #[test]
    fn test_wal_recovery() {
        let path = temp_path();

        {
            let gf = Horon::open(&path).unwrap();
            gf.put("/a", b"1").unwrap();
            gf.put("/b", b"2").unwrap();
            gf.remove("/a").unwrap();
            gf.flush().unwrap();
        }

        {
            let gf = Horon::open(&path).unwrap();
            assert!(!gf.exists("/a"));
            assert_eq!(gf.get("/b").unwrap(), b"2");
            assert_eq!(gf.len(), 1);
        }
    }

    #[test]
    fn test_compact() {
        let path = temp_path();

        {
            let gf = Horon::open(&path).unwrap();
            for i in 0..50 {
                gf.put(&format!("/node_{}", i), format!("data_{}", i).as_bytes()).unwrap();
            }
            assert!(gf.wal_len() > 0);
            gf.compact().unwrap();
            assert_eq!(gf.wal_len(), 0);
            assert_eq!(gf.len(), 50);
        }

        {
            let gf = Horon::open(&path).unwrap();
            assert_eq!(gf.len(), 50);
            assert_eq!(gf.get("/node_42").unwrap(), b"data_42");
        }
    }

    #[test]
    fn test_upsert() {
        let path = temp_path();

        let gf = Horon::open(&path).unwrap();
        gf.put("/key", b"v1").unwrap();
        gf.put("/key", b"v2").unwrap();
        assert_eq!(gf.get("/key").unwrap(), b"v2");
        assert_eq!(gf.len(), 1);
    }

    #[test]
    fn test_nearest() {
        let path = temp_path();

        let gf = Horon::open(&path).unwrap();
        gf.put("/a", b"data").unwrap();
        gf.put("/b", b"data").unwrap();

        let (nearest_path, dist) = gf.nearest(&fp(&[0.0, 0.0, 0.0, 0.0])).unwrap();
        assert_eq!(nearest_path, "/");
        assert!(dist.to_f64() < 0.1);
    }

    #[test]
    fn test_empty_file_size() {
        let path = temp_path();
        {
            let gf = Horon::open(&path).unwrap();
            gf.flush().unwrap();
            drop(gf);
        }
        let size = std::fs::metadata(&path).unwrap().len();
        assert_eq!(size as usize, MIN_FILE_SIZE);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<Horon>();
        assert_sync::<Horon>();
    }

    #[test]
    fn test_concurrent_put() {
        let path = temp_path();
        let gf = Arc::new(Horon::open(&path).unwrap());

        let mut handles = vec![];
        for t in 0..4 {
            let gf = Arc::clone(&gf);
            handles.push(std::thread::spawn(move || {
                for i in 0..5 {
                    gf.put(&format!("/t{}/n{}", t, i), b"data").unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        for t in 0..4 {
            for i in 0..5 {
                assert!(gf.exists(&format!("/t{}/n{}", t, i)));
            }
        }
    }
}

// ===========================================================================
// HoronReader — concurrent read-only access
// ===========================================================================

/// What a [`HoronReader::refresh`] had to do to catch up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshOutcome {
    /// No new committed entries; the view was already current.
    UpToDate,
    /// `n` new WAL entries were applied incrementally.
    Applied(usize),
    /// The writer compacted past this reader's position, folding unseen
    /// entries into the snapshot, so the view was rebuilt from scratch.
    /// Equivalent to [`WalTail::SnapshotRequired`] on the writer side.
    Reloaded,
}

/// A read-only, lock-free view of a `.htt` file.
///
/// [`Horon`] takes an exclusive advisory lock on every open, so only one
/// process may hold a file at all — even for reading. `HoronReader` opens the
/// file read-only and takes **no lock**, so any number of readers may run
/// alongside a live writer.
///
/// # Why this is safe
///
/// Everything in a live file is append-only or replaced atomically:
/// the header and snapshot are immutable between compactions, WAL entries are
/// appended, and compaction writes a tempfile and `rename`s it. The one
/// in-place mutation is the 8-byte WAL header, and the scanner already treats
/// it as advisory — it derives the entry sequence from the entries themselves
/// and stops at the first CRC failure, so a torn count cannot mislead a
/// reader. A reader that opened before a compaction keeps reading the old
/// inode: a complete, consistent, immutable view.
///
/// # Constraints
///
/// - **Snapshot semantics.** The view is fixed as of `open`. Call
///   [`refresh`](Self::refresh) to pick up committed writes.
/// - **No GACL enforcement.** Queries are unscoped. GACL is cooperative
///   query-scoping rather than a security boundary (anyone holding the file
///   can read all of it), so this grants no access the file did not already
///   give — but use [`Horon`] when you want band filtering.
/// - **Unix.** Lock-free reading relies on advisory locks. On platforms where
///   locks are mandatory, an unlocked read of a locked file may fail.
/// - **A long-lived reader pins its inode.** If the writer compacts, the
///   pre-compaction file cannot be reclaimed until every reader holding it
///   closes or refreshes.
///
/// Reads never truncate a torn WAL tail — that recovery is a write, and
/// belongs to the writer.
pub struct HoronReader {
    store: Store,
    path: PathBuf,
    header: GeoHeader,
    layout: SemLayout,
    /// Sequence the next unseen WAL entry would carry.
    next_seq: u32,
    /// Present in [`ReaderMode::Partial`]: the mmap snapshot view plus the
    /// WAL overlay/tombstones, resolved through the same `partial_*`
    /// functions the writer uses.
    partial: Option<PartialState>,
    /// v3 global normalization bounds (partial mode, meaning-addressed files).
    ma_bounds: Option<(f64, f64)>,
}

/// How a [`HoronReader`] materializes the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReaderMode {
    /// Decode the whole snapshot into memory (0.9.0 behaviour): full read
    /// surface including structural hyperbolic queries.
    Full,
    /// Memory-map the snapshot; RAM holds only a key→offset index plus
    /// writes-since-snapshot. N processes share one physical copy via the
    /// page cache. Implies lazy geometry: structural hyperbolic queries
    /// (`nearest`, `neighbors`) are unavailable; semantic queries are exact
    /// by default. Requires an uncompressed, non-GACL file — the same
    /// constraints as the writer's `partial_reads`.
    Partial,
}

impl HoronReader {
    /// Open a `.htt` file read-only, without taking a lock
    /// ([`ReaderMode::Full`]).
    pub fn open<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Self::open_with_mode(path, ReaderMode::Full)
    }

    /// Open read-only in [`ReaderMode::Partial`]: mmap the snapshot instead
    /// of materializing it. See [`ReaderMode`] for the constraints.
    pub fn open_partial<P: AsRef<Path>>(path: P) -> HoronResult<Self> {
        Self::open_with_mode(path, ReaderMode::Partial)
    }

    /// Open read-only in the given mode, without taking a lock.
    pub fn open_with_mode<P: AsRef<Path>>(path: P, mode: ReaderMode) -> HoronResult<Self> {
        let path = path.as_ref().to_path_buf();
        match mode {
            ReaderMode::Full => {
                let (store, header, layout, next_seq) = Self::load(&path)?;
                Ok(Self { store, path, header, layout, next_seq, partial: None, ma_bounds: None })
            }
            ReaderMode::Partial => Self::load_partial(path),
        }
    }

    /// Build the partial (mmap) view: same construction as the writer's
    /// `partial_reads` mode, minus the exclusive lock, minus torn-tail
    /// truncation (recovery is a write and belongs to the writer).
    fn load_partial(path: PathBuf) -> HoronResult<Self> {
        use std::io::Read;
        let mut file = File::open(&path)?;

        let mut header_bytes = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header = GeoHeader::from_bytes(&header_bytes)?;
        let layout = SemLayout::from_header(&header);

        if header.compression_enabled() {
            return Err(HoronError::Config(
                "partial mode requires an uncompressed snapshot (a zstd frame cannot be partially read)".into(),
            ));
        }
        if header.gacl_enabled() {
            return Err(HoronError::Config(
                "partial mode is not compatible with GACL enforcement".into(),
            ));
        }

        let ma_bounds = if header.flags & FLAG_MEANING_ADDRESSED != 0 {
            Some(HoronCore::read_bounds_section(&mut file, header.semantic_dims as usize)?)
        } else {
            None
        };

        // Snapshot section header (manual — index, not materialize), CRC
        // over the mmap, structural scan. Mirrors the writer's open_partial.
        let semantic_dims = header.semantic_dims as usize;
        let snapshot_has_crc = header.version >= 2;
        let mut b4 = [0u8; 4];
        file.read_exact(&mut b4)?;
        let snap_byte_len = u32::from_le_bytes(b4) as usize;
        if snap_byte_len > MAX_SNAPSHOT_BYTES {
            return Err(HoronError::InvalidFormat(format!(
                "snapshot byte length {} exceeds maximum {}",
                snap_byte_len, MAX_SNAPSHOT_BYTES
            )));
        }
        file.read_exact(&mut b4)?;
        let node_count = u32::from_le_bytes(b4) as usize;
        if node_count > snap_byte_len / 8 + 1 {
            return Err(HoronError::InvalidFormat(format!(
                "snapshot node count {} impossible for {} section bytes",
                node_count, snap_byte_len
            )));
        }
        let raw_start = file.stream_position()? as usize;
        file.seek(SeekFrom::Start((raw_start + snap_byte_len) as u64))?;
        let stored_crc = if snapshot_has_crc {
            file.read_exact(&mut b4)?;
            Some(u32::from_le_bytes(b4))
        } else {
            None
        };
        let (wal_entry_count, wal_base_seq) = wal::read_wal_header(&mut file)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if let Some(stored) = stored_crc {
            let region = mmap.get(raw_start..raw_start + snap_byte_len).ok_or_else(|| {
                HoronError::InvalidFormat("snapshot region exceeds file length".into())
            })?;
            let computed = crc32fast::hash(region);
            if stored != computed {
                return Err(HoronError::ChecksumMismatch {
                    expected: stored,
                    actual: computed,
                    context: "snapshot section".to_string(),
                });
            }
        }

        let hilbert_fn;
        let hilbert_of: Option<&dyn Fn(&[u8]) -> u128> = match ma_bounds {
            Some(b) => {
                hilbert_fn = move |sem: &[u8]| {
                    if layout.quantized {
                        match layout.decode_tail(sem) {
                            Ok(full) => global_hilbert(&full, semantic_dims, b),
                            Err(_) => 0,
                        }
                    } else {
                        global_hilbert(sem, semantic_dims, b)
                    }
                };
                Some(&hilbert_fn)
            }
            None => None,
        };
        let view = SnapView::scan(
            mmap, raw_start, snap_byte_len, node_count, layout, hilbert_of,
        )?;

        // WAL overlay: writes since the snapshot, tombstones for deletes.
        // Derive-don't-trust as everywhere: the scan stops at the first bad
        // CRC and takes sequence numbers from the entries.
        let store = Store::with_config(
            StoreConfig::new().tau(g_math::fixed_point::FixedPoint::from_raw(header.tau_raw)),
        );
        let mut tombstones = HashSet::new();
        let (next_seq, _valid, _end) = scan_wal(
            &mut file,
            &header,
            &layout,
            wal_base_seq,
            wal_entry_count,
            |entry| apply_wal_partial(&store, &view, &mut tombstones, entry),
        )?;

        let partial = Some(PartialState {
            view: RwLock::new(view),
            tombstones: RwLock::new(tombstones),
            scanned: AtomicUsize::new(0),
        });
        Ok(Self { store, path, header, layout, next_seq, partial, ma_bounds })
    }

    /// Read the whole file into a fresh `Store`. No lock, no truncation.
    fn load(path: &Path) -> HoronResult<(Store, GeoHeader, SemLayout, u32)> {
        let mut file = File::open(path)?;

        let mut header_bytes = [0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut file, &mut header_bytes)?;
        let header = GeoHeader::from_bytes(&header_bytes)?;

        let layout = SemLayout::from_header(&header);
        let compressed = header.compression_enabled();
        let snapshot_has_crc = header.version >= 2;
        let meaning_addressed = header.flags & FLAG_MEANING_ADDRESSED != 0;

        if meaning_addressed {
            // Skip the v3 bounds section: readers do not place nodes, so the
            // bounds are not needed to reconstruct the view.
            let bounds_len = (header.semantic_dims as usize)
                .saturating_sub(DIM_USER_DEFINED_START)
                * 16;
            file.seek(SeekFrom::Current(bounds_len as i64))?;
        }

        let mut snap_entries =
            snapshot::read_snapshot(&mut file, compressed, &layout, snapshot_has_crc)?;
        if meaning_addressed {
            // Hilbert order on disk; Sarkar replay needs parents first.
            snap_entries.sort_by(|a, b| {
                let da = a.key.matches('/').count();
                let db = b.key.matches('/').count();
                da.cmp(&db).then(a.key.cmp(&b.key))
            });
        }

        let (wal_entry_count, wal_base_seq) = wal::read_wal_header(&mut file)?;

        let store = Store::with_config(
            StoreConfig::new().tau(g_math::fixed_point::FixedPoint::from_raw(header.tau_raw)),
        );
        load_snapshot_into_store(&store, &snap_entries, false)?;

        let (next_seq, _valid, _end) = scan_wal(
            &mut file,
            &header,
            &layout,
            wal_base_seq,
            wal_entry_count,
            |entry| replay_entry(&store, entry, false),
        )?;

        Ok((store, header, layout, next_seq))
    }

    /// Pick up committed writes since the last load.
    ///
    /// Applies new WAL entries incrementally. If the writer compacted past
    /// this reader's position — folding entries it never saw into the
    /// snapshot — the view is rebuilt instead, reported as
    /// [`RefreshOutcome::Reloaded`].
    pub fn refresh(&mut self) -> HoronResult<RefreshOutcome> {
        let mut file = File::open(&self.path)?;

        let mut header_bytes = [0u8; HEADER_SIZE];
        std::io::Read::read_exact(&mut file, &mut header_bytes)?;
        let header = GeoHeader::from_bytes(&header_bytes)?;
        let compressed = header.compression_enabled();
        let meaning_addressed = header.flags & FLAG_MEANING_ADDRESSED != 0;

        if meaning_addressed {
            let bounds_len = (header.semantic_dims as usize)
                .saturating_sub(DIM_USER_DEFINED_START)
                * 16;
            file.seek(SeekFrom::Current(bounds_len as i64))?;
        }

        // Skip the snapshot body without decoding it — only the WAL header
        // beyond it is needed to decide incremental-vs-reload. Same traversal
        // HoronHistory uses.
        let mut buf4 = [0u8; 4];
        std::io::Read::read_exact(&mut file, &mut buf4)?;
        let snap_byte_len = u32::from_le_bytes(buf4) as u64;
        std::io::Read::read_exact(&mut file, &mut buf4)?;
        let node_count = u32::from_le_bytes(buf4);
        if node_count > 0 && compressed {
            std::io::Read::read_exact(&mut file, &mut buf4)?;
            let comp_len = u32::from_le_bytes(buf4) as u64;
            file.seek(SeekFrom::Current(comp_len as i64))?;
        } else {
            file.seek(SeekFrom::Current(snap_byte_len as i64))?;
        }
        if header.version >= 2 {
            file.seek(SeekFrom::Current(4))?; // snapshot CRC
        }
        let (wal_entry_count, wal_base_seq) = wal::read_wal_header(&mut file)?;

        // Entries this reader never saw are now inside the snapshot. The
        // rebuild keeps the reader's mode: a partial reader also drops its
        // old mmap here, releasing the pre-compaction inode it was pinning.
        if wal_base_seq > self.next_seq {
            let rebuilt = if self.partial.is_some() {
                Self::load_partial(self.path.clone())?
            } else {
                let (store, header, layout, next_seq) = Self::load(&self.path)?;
                Self { store, path: self.path.clone(), header, layout, next_seq, partial: None, ma_bounds: None }
            };
            *self = rebuilt;
            return Ok(RefreshOutcome::Reloaded);
        }

        let from = self.next_seq;
        let mut applied = 0usize;
        let next_seq = if let Some(p) = &self.partial {
            // Same-snapshot catch-up: apply new entries to the overlay and
            // tombstones through the writer's own replay function; the mmap
            // view is untouched (the snapshot has not changed).
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            let mut tombs = p.tombstones.write().unwrap_or_else(|e| e.into_inner());
            let (next_seq, _valid, _end) = scan_wal(
                &mut file,
                &self.header,
                &self.layout,
                wal_base_seq,
                wal_entry_count,
                |entry| {
                    if entry.seq >= from {
                        applied += 1;
                        apply_wal_partial(&self.store, &view, &mut tombs, entry)
                    } else {
                        Ok(())
                    }
                },
            )?;
            next_seq
        } else {
            let (next_seq, _valid, _end) = scan_wal(
                &mut file,
                &self.header,
                &self.layout,
                wal_base_seq,
                wal_entry_count,
                |entry| {
                    if entry.seq >= from {
                        applied += 1;
                        replay_entry(&self.store, entry, false)
                    } else {
                        Ok(())
                    }
                },
            )?;
            next_seq
        };
        self.next_seq = next_seq;

        Ok(if applied == 0 {
            RefreshOutcome::UpToDate
        } else {
            RefreshOutcome::Applied(applied)
        })
    }

    /// Sequence number the next unseen WAL entry would carry.
    pub fn next_seq(&self) -> u32 {
        self.next_seq
    }

    /// Retrieve the data stored at `key`.
    pub fn get(&self, key: &str) -> HoronResult<Vec<u8>> {
        if let Some(p) = &self.partial {
            return partial_get(&self.store, p, key);
        }
        Ok(self.store.get(key)?)
    }

    /// Whether `key` exists in this view.
    pub fn exists(&self, key: &str) -> bool {
        if let Some(p) = &self.partial {
            return partial_exists(&self.store, p, key);
        }
        self.store.exists(key)
    }

    /// Raw semantic coordinates for `key`, or empty when unset.
    pub fn get_semantic(&self, key: &str) -> HoronResult<Vec<u8>> {
        if let Some(p) = &self.partial {
            return partial_get_semantic(&self.store, p, key);
        }
        Ok(self.store.get_semantic(key)?)
    }

    /// Direct children of `path`.
    pub fn children(&self, path: &str) -> HoronResult<Vec<String>> {
        if let Some(p) = &self.partial {
            return partial_children(&self.store, p, path);
        }
        Ok(self.store.children(path)?)
    }

    /// Every key under `prefix`, recursively.
    pub fn list(&self, prefix: &str) -> HoronResult<Vec<String>> {
        if let Some(p) = &self.partial {
            return partial_list(&self.store, p, prefix);
        }
        Ok(self.store.list(prefix)?)
    }

    /// Nearest node to a structural coordinate.
    ///
    /// Unavailable in [`ReaderMode::Partial`] (it implies lazy geometry);
    /// semantic queries are the supported surface there.
    pub fn nearest(&self, coords: &[FixedPoint]) -> HoronResult<(String, FixedPoint)> {
        if self.partial.is_some() {
            return Err(HoronError::Config(
                "structural queries are unavailable in partial reader mode (lazy geometry); use ReaderMode::Full or a semantic query".into(),
            ));
        }
        Ok(self.store.nearest(coords)?)
    }

    /// k nearest neighbours of an existing node.
    ///
    /// Unavailable in [`ReaderMode::Partial`] — see [`Self::nearest`].
    pub fn neighbors(&self, path: &str, k: usize) -> HoronResult<Vec<String>> {
        if self.partial.is_some() {
            return Err(HoronError::Config(
                "structural queries are unavailable in partial reader mode (lazy geometry); use ReaderMode::Full or a semantic query".into(),
            ));
        }
        Ok(self.store.neighbors(path, k)?)
    }

    /// The k nearest nodes to `query_coords` over a dimension slice.
    ///
    /// In partial mode this runs over the mmap without materializing the
    /// snapshot — exact (full proxy-space scan); the approximate Hilbert
    /// window is writer-side opt-in and not offered here.
    pub fn nearest_semantic(
        &self,
        query_coords: &[u8],
        k: usize,
        dim_range: std::ops::Range<usize>,
    ) -> HoronResult<Vec<(String, FixedPoint)>> {
        if let Some(p) = &self.partial {
            return partial_nearest_semantic(
                &self.store, p, self.ma_bounds, false, query_coords, k, dim_range,
            );
        }
        Ok(self.store.nearest_semantic(query_coords, k, dim_range)?)
    }

    /// Snapshot entries examined by the most recent partial-mode semantic
    /// query (`None` in full mode).
    pub fn last_semantic_scan_count(&self) -> Option<usize> {
        self.partial.as_ref().map(|p| p.scanned.load(Ordering::Relaxed))
    }

    /// Number of live nodes in this view.
    pub fn len(&self) -> usize {
        if let Some(p) = &self.partial {
            let view = p.view.read().unwrap_or_else(|e| e.into_inner());
            let tombs = p.tombstones.read().unwrap_or_else(|e| e.into_inner());
            let snap_live = view
                .by_key
                .keys()
                .filter(|k| !tombs.contains(*k) && !self.store.exists(k))
                .count();
            return snap_live + self.store.len();
        }
        self.store.len()
    }

    /// Whether this view holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}
