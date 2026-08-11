<p align="center">
  <a href="https://geodineum.com">
    <img src="https://geodineum.com/wp-content/uploads/2026/07/logo_geodineum_launch.png" alt="Geodineum" width="128">
  </a>
</p>

# Horon

**The file is the database. The layout is the index.**

[![Crates.io](https://img.shields.io/crates/v/horon)](https://crates.io/crates/horon)
[![Documentation](https://docs.rs/horon/badge.svg)](https://docs.rs/horon)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Built by **Niels Erik Toren** · part of the [Geodineum](https://github.com/nierto) ecosystem.

## The idea

Persist a deterministic hierarchical store to one `.htt` file: a documented
binary snapshot plus a write-ahead log, every mutation logged before it takes
effect. The file survives crashes and power loss and recovers itself on open.
Copy the file and you have copied the database, its history, and its indexes.

## Why the layout is the index

The engine gives structure a *location in space*: trees embed into
hyperbolic geometry, where exponential volume matches exponential branching
([horon-engine](https://github.com/nierto/horon-engine)). Horon extends that
to a *location in the file*: with the meaning-addressed layout (format v3),
a node's byte position becomes a function of its semantic coordinates
through a space-filling curve over fixed bounds — comparable data becomes
byte-adjacent on disk. Readers memory-map the file, so N processes share one
physical copy through the page cache, and the working set is the pages a
query touches rather than the dataset.

## Determinism

All arithmetic is [gMath](https://github.com/nierto/gMath) fixed point. A
file written on one platform replays to bit-identical state on any other.
That property is what lets the WAL double as a replication protocol. CI runs
the determinism gate on x86-64 and arm64.

## Public build surface

The supported API is what `horon` re-exports at the crate root:

- **`Horon`** / **`HoronConfig`** / **`DurabilityMode`** / **`HistoryRetention`**: open, read, write, query, compact.
- **`HoronError`** / **`HoronResult`**: the error surface. `HoronError::Store` carries a `StoreError`, also re-exported.
- **`Store`** / **`StoreError`** / **`SemanticOutlier`** / **`SemanticDisk`**: the engine surface, reachable through this crate alone.
- **`HoronHistory`** (+ `EpochInfo`, `HoronStateView`, `KeyDelta`, `DeltaKind`, `StateNode`): read-only temporal queries.
- **`Credentials`** / **`AccessBand`** / **`NodeAccessBands`**: GACL bands and session credentials.
- **`WalEntry`** / **`WalPayload`** / **`WalTail`**: the replication stream types.
- **`SemLayout`**: the semantic-tail layout descriptor (full-width vs quantized).

Everything else (`format`, `header`, `snapshot`, `wal`, `partial`,
`hilbert`, `quant` internals) is format plumbing: stable through the binary
specification, not through its Rust signatures. Per-symbol truth lives in
the docblocks (`cargo doc --open`).

## The `htt` CLI

Built with feature `import`, this is the no-code surface: `import-csv`,
`import-fs`, `import-json` (nested JSON/YAML), `import-sqlite` (foreign keys
become the hierarchy), `import-vec` (JSONL embedding exports), plus
`inspect` and `query`.

## Capabilities

Each line is a shipped capability and the thing it makes possible:

- **Embedded zero-server database.** Path-keyed puts and gets, metadata,
  hierarchy walks, in one copyable file. *Build state once, query it
  offline forever.*
- **Concurrent lock-free readers.** `HoronReader` opens without the writer's
  exclusive lock: N reader processes alongside one live writer, with
  `refresh()` for incremental catch-up. `ReaderMode::Partial` memory-maps
  the snapshot instead of materializing it — measured with 4 concurrent
  readers, 18 MiB physical per process against 148 MiB materialized, one
  shared copy in the page cache. *Query a corpus from every process at
  once, without loading it anywhere.*
- **Semantic k-NN over dimension slices.** Attach coordinate vectors, query
  nearest neighbors over exactly the axes that matter. *Recommendations by
  domain that ignore occupancy and popularity axes.*
- **Meaning-addressed layout (format v3).** Similar entries become
  physically adjacent bytes; with `partial_reads`, semantic queries run
  over the mmap without materializing the snapshot — exact by default,
  and `approximate_semantic` opts into a Hilbert-window mode that touches
  only a byte-neighborhood (faster, and it can miss neighbours; measure
  the recall your data gives before relying on it). *Browse a corpus
  larger than RAM through an mmap window.*
- **Quantized semantic storage (format v4).** User dims stored as 2-byte
  balanced ternary instead of 16-byte fixed point: ranking-grade precision,
  byte-exact determinism. *`import-vec --quantized` turns a vector-DB
  export into a compact semantic file in one command.*
- **Temporal epochs.** Seal calibration moments; replay any sealed state,
  trace one key's drift, diff two epochs. *The movement of the data is the
  insight.*
- **WAL replication.** Subscribe to committed entries and catch up by
  sequence number; the file is the wire protocol. *A replica is a file copy
  plus a tail.*
- **Geometric access control.** Access bands encoded in reserved semantic
  dims, checked against session credentials. *Cooperative multi-tenant
  scoping of one shared file. Not a security boundary; see limits.*
- **Import pipelines.** CSV, directory trees, nested JSON/YAML, SQLite (FK
  hierarchy becomes paths), and JSONL embeddings, each one command. *Any of
  these becomes a queryable `.htt` without writing code.*
- **Crash recovery by construction.** Per-entry CRCs, torn-tail truncation,
  atomic compaction, fault-injection tested. *Kill -9 mid-write; reopen;
  committed data is there.*

## Contract

The binary format is the integration contract:
[docs/HTT_FORMAT.md](docs/HTT_FORMAT.md). Any language can read or write
`.htt` from that document alone. Design deep-dives:
[docs/TEMPORAL_EPOCHS.md](docs/TEMPORAL_EPOCHS.md) (epochs and history
sidecars), [docs/QUANTIZED_SEMANTIC.md](docs/QUANTIZED_SEMANTIC.md)
(quantized tails, including the honest limits).

## Quick start

Create, use, modify, reopen. The `lifecycle` example runs this sequence
with assertions: `GMATH_PROFILE=embedded cargo run --example lifecycle`.

```rust
use horon::{Horon, HoronConfig};

// CREATE: a server fleet. Hierarchy in the path, telemetry in the dims
// (16 reserved + 2 user). Every write is WAL-logged before it takes effect.
let htt = Horon::open_with_config("fleet.htt", HoronConfig {
    semantic_dims: 18,
    ..Default::default()
})?;
htt.put("/fleet/eu/web-01", b"nginx 1.27")?;
htt.set_meta("/fleet/eu/web-01", "role", "edge")?;
htt.set_semantic("/fleet/eu/web-01", coords)?; // raw Q64.64, 16 bytes/dim, dims 16+ are yours

// USE: retrieval, hierarchy, "which machines behave like this one?"
let data = htt.get("/fleet/eu/web-01")?;
let machines = htt.children("/fleet/eu")?;
let similar = htt.neighbors_semantic("/fleet/eu/web-01", 5, 16..18)?;

// MODIFY: put is upsert; coordinates and metadata replace in place.
htt.put("/fleet/eu/web-01", b"nginx 1.28")?;      // redeploy
htt.set_semantic("/fleet/eu/web-01", new_coords)?; // load profile shifted
htt.remove("/fleet/eu/batch-01")?;                 // decommissioned
htt.compact()?; // fold the WAL into a fresh snapshot

// REOPEN: drop the handle; the next open replays to identical state.
drop(htt);
let htt = Horon::open("fleet.htt")?;
```

The same shape fits course catalogs, sensor networks, and document corpora:
hierarchy in the path, meaning in the dims.

Or without code:

```sh
cargo build --release --features import
htt import-csv courses.csv courses.htt --path-cols category --dim-cols x,y
htt query courses.htt /category/some-course 5
```

Build with `GMATH_PROFILE=embedded`; the determinism contract is defined on
that profile. For an idiomatic thread-safe wrapper, see
`examples/geostore.rs`.

## Limits worth knowing

- One **writer** per file: `Horon::open` takes an exclusive advisory lock
  (unix), so a second writer fails cleanly. Readers are unbounded —
  `HoronReader` opens lock-free alongside the writer, in `Full` mode
  (snapshot materialized) or `Partial` mode (mmap; RAM holds a key→offset
  index plus writes-since-open, so the file may exceed memory). On
  non-unix platforms the lock is a no-op and single-process access is your
  responsibility; lock-free reading is unix-only today.
- `partial_reads` (mmap-backed snapshot) is mutually exclusive with two
  other options, and the constructor rejects the combination rather than
  degrading silently:
  - `compression: true` — a zstd frame cannot be partially read, so a
    compressed snapshot must be materialized whole.
  - `gacl: true` — access-band filtering needs the resolved node, which the
    mmap view does not materialize until a read touches it.

  It also implies `lazy_geometry`, so hyperbolic spatial queries are
  unavailable in that mode: the trade is RAM for reach. Choose per file at
  open time; the flag is not persisted.
- GACL is cooperative query-scoping, not a security boundary. Anyone
  holding the file can read all of it; use OS permissions or encryption for
  confidentiality.
- Quantized files trade user-dim precision for size: distances are
  ranking-grade, out-of-range values are rejected (never silently clamped),
  and pre-v4 readers refuse the file loudly.
- Semantic dimensions are capped at 255 per file (16 reserved + up to 239
  user).
- Default durability fsyncs every write. Bulk loads should use
  `DurabilityMode::Relaxed` and finish with `compact()`; documented on
  `DurabilityMode`.

## Collaborate

Contributions are welcome. Open issues and pick up work on the ecosystem
board at [geodineum.com](https://geodineum.com); issues tagged
`good-first-issue` are a good place to start.

- Fork, branch, and open a pull request against `main`.
- Any change to the on-disk format must update
  [docs/HTT_FORMAT.md](docs/HTT_FORMAT.md) in the same commit. The spec is
  the contract other implementations are written against.
- Format changes must also regenerate the conformance corpus
  (`cargo run --example gen_corpus`) and the determinism goldens, in that
  same commit.

## Author & support

Built by **Niels Erik Toren**.

If you want to support the work:

| Currency | Address |
|---|---|
| Bitcoin (BTC) | `bc1qwf78fjgapt2gcts4mwf3gnfkclvqgtlg4gpu4d` |
| Ethereum (ETH) | `0xf38b517Dd2005d93E0BDc1e9807665074c5eC731` / `nierto.eth` |
| Monero (XMR) | `8BPaSoq1pEJH4LgbGNQ92kFJA3oi2frE4igHvdP9Lz2giwhFo2VnNvGT8XABYasjtoVY2Qb3LVHv6CP3qwcJ8UnyRtjWRZ5` |

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express
or implied. Use of this software is entirely at your own risk. In no event
shall the author or contributors be held liable for any damages arising from
the use or inability to use this software.

## License

Apache-2.0 (see [LICENSE](LICENSE)).
