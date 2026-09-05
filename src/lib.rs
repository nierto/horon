//! # Horon
//!
//! Hierarchical data storage in a single `.htt` file, with write-ahead
//! logging, zstd compression, and geometric access control.
//!
//! Horon wraps a deterministic in-memory hyperbolic `Store` with durable
//! persistence:
//! every mutation is appended to a WAL, snapshots are produced by compaction,
//! and cold start replays the log to reconstruct identical geometric state
//! on any platform (guaranteed by gMath's deterministic fixed-point arithmetic).
//!
//! ## Quick Start
//!
//! ```no_run
//! use horon::Horon;
//! use g_math::fixed_point::FixedPoint;
//!
//! // Open or create a .htt file
//! let htt = Horon::open("data.htt").unwrap();
//!
//! // Standard Store operations — persisted automatically
//! htt.put("/config/db", b"postgres://localhost").unwrap();
//! let data = htt.get("/config/db").unwrap();
//!
//! // Coordinates and distances are Q64.64 fixed point, never floats, so the
//! // same query returns bit-identical results on every platform. Conversion
//! // from a decimal literal is explicit: the lossy step stays visible at the
//! // call site instead of hiding inside the API.
//! let origin: Vec<FixedPoint> = (0..4).map(|_| FixedPoint::from_int(0)).collect();
//! let (path, dist) = htt.nearest(&origin).unwrap();
//!
//! // Compact WAL into snapshot
//! htt.compact().unwrap();
//! ```

#![warn(missing_docs)]

pub mod error;
pub mod format;
pub mod header;
pub mod snapshot;
pub mod wal;
pub mod compression;
pub mod gacl;
pub mod file;
pub mod hilbert;
pub mod history;
pub mod partial;
pub mod quant;

// Re-exports
pub use error::{HoronError, HoronResult};
// Engine surface reachable through this crate alone: the store type behind
// `Horon::store()`, the error payload carried by `HoronError::Store`,
// the outlier record returned by `Horon::find_outliers`, and the semantic
// concept disk. Users never have to name the engine crate.
pub use horon_engine::{SemanticDisk, SemanticOutlier, Store, StoreError};
pub use gacl::{Credentials, AccessBand, NodeAccessBands};
pub use file::{
    AutoCompactMode, DurabilityMode, HistoryRetention, Horon, HoronConfig, HoronReader, ReaderMode,
    RefreshOutcome, WalTail,
};
pub use history::{DeltaKind, EpochInfo, HoronHistory, HoronStateView, KeyDelta, StateNode};
pub use quant::SemLayout;
pub use wal::{WalEntry, WalPayload};
