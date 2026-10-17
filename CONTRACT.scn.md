# CONTRACT.scn :: horon agent primer

Condensed prime for an agent building against `horon`. Authoritative format
spec: [docs/HTT_FORMAT.md](docs/HTT_FORMAT.md). Per-symbol truth: docblocks
(`cargo doc --open`). Code wins over docs.

## ::PRIME

```
Horon := durable single-file hierarchical store | .htt binary format | WAL-first facade over a hyperbolic spatial engine
Implements: path-keyed CRUD + spatial/semantic k-NN via in-memory Store + append-only WAL + snapshot compaction
Maintains: deterministic state: g_math Q64.64 fixed point, bit-identical replay on every platform; the WAL doubles as the replication protocol
Coordinates: horon-engine (geometry) + g_math (arithmetic, 0.7, Q64_64); the format is defined by HTT_FORMAT.md, not by this crate
```

## ::ANCHOR

```
DOMAIN: write-ahead-log, snapshot, compaction, crash-recovery, CRC32, zstd, mmap, advisory-flock, embedded-database, single-file, replication, k-NN, dimension-slice, poincare-disk, fixed-point-Q64.64, balanced-ternary-quantization, hilbert-curve, temporal-epochs, geometric-ACL
PATTERN: facade, apply-then-log, copy-out-mmap-read, lazy-index, session-credentials, opt-in-at-creation, golden-file-oracle
RUST: &self-everywhere, Arc-shareable, Send+Sync, cheap-clone-handle, feature-gated-CLI
CONCEPT: determinism-by-construction, format-vs-product-separation, one-writer, cooperative-access-control, punctuated-calibration
```

## ::CONTEXT

```
ARCHITECTURAL_LAYER: persistence facade (the product surface; engine below, reachable but unadvertised)
STACK: g_math (arithmetic) -> horon-engine (in-memory Store) -> horon (.htt durability) -> htt CLI / independent readers
LIFECYCLE: Horon::open(path) -> put/get/query -> compact() -> drop (advisory lock released)
BUILD: cargo build [--features import  # enables the htt CLI]
TEST:  cargo test --all-features       # corpus must regenerate byte-identically
```

## ::INTERFACE

Compressed signatures. `HoronResult<T>` = `Result<T, HoronError>`; errors on
every fallible call.

```
Horon::open(path) | Horon::open_with_config(path, HoronConfig) -> HoronResult<Horon>
put(key, &[u8]) | put_data_only(key, &[u8])          # data_only skips geometry: ~µs vs ~ms
get(key) -> Vec<u8> | remove(key) | exists(key) -> bool
children(path) -> Vec<String> | list(prefix) -> Vec<String>
set_meta(key, k, v) | get_meta(key) -> HashMap<String, String>
set_semantic(key, Vec<u8>) | get_semantic(key) -> Vec<u8> | clear_semantic(key)
  # coords = raw Q64.64 LE, 16 B/dim; dims 0..11 GACL bands, 12..15 reserved, 16+ user
  # all-zero vector = "not set" sentinel and is rejected; keep >=1 dim off zero
nearest(&[FixedPoint]) -> (String, FixedPoint) | nearest_k(coords, k) -> Vec<(String, FixedPoint)>
neighbors(path, k) -> Vec<String> | find_within(path, radius: FixedPoint) -> Vec<String>
position(key) -> Vec<FixedPoint>
nearest_semantic(&[u8], k, Range<usize>) | neighbors_semantic(path, k, Range) | find_similar(key, k, Range)
find_outliers(prefix, z: FixedPoint, Range) -> Vec<SemanticOutlier>
Horon::semantic_distance(a, b, Range) -> FixedPoint  # associated fn, no store touched
embed(key) -> bool | embed_all(prefix) -> usize      # upgrade data-only keys to embeddings
set_credentials(Credentials) | clear_credentials() | gacl_active() -> bool
compact() -> bool | compact_async() -> JoinHandle | flush()
seal_epoch() -> u64 | seal_speculative_epoch() -> u64 | current_epoch() -> u64
subscribe_wal(..) | wal_entries_since(seq) -> WalTail | wal_len() -> u32
len() | is_empty() | path() | store() -> &Store      # engine escape hatch (type re-exported)

HoronHistory::open(path): epochs() -> &[EpochInfo] | as_of(id) -> HoronStateView
  | trajectory(key, &Range) -> Vec<(u64, Vec<f64>)> | delta(a, b)   # read-only temporal
```

CLI (feature `import`): `htt import-csv | import-fs | import-json | import-sqlite | import-vec [--quantized] | inspect | query`

## ::PATTERNS

```
EMBEDDED_DB:  open -> put/set_meta/set_semantic -> query -> compact -> drop; one copyable file
BULK_LOAD:    HoronConfig{durability: DurabilityMode::Relaxed} + put_data_only -> compact() once at end
              (default durability fsyncs EVERY append at ~ms each; N nodes cost >=N fsyncs)
SLICE_QUERY:  set_semantic(40 dims) -> nearest_semantic(q, k, 16..33); the slice picks the question
SIMILARITY:   find_similar(key, k, dims) | anomalies: find_outliers(prefix, z, dims)
REPLICATION:  subscribe_wal / wal_entries_since(seq) -> ship entries -> replica replays; the file is the wire protocol
TEMPORAL:     HoronConfig{history_retention: HistoryRetention::Archive} -> seal_epoch() at calibration moments
              -> HoronHistory as_of/trajectory/delta; default Off = byte-identical file, no sidecars
GACL:         file created with GACL -> set_credentials(creds) -> reads/writes/queries filtered per session
BIG_FILES:    meaning-addressed (v3) + partial_reads -> exact mmap queries, zero materialization; approximate_semantic=true -> Hilbert-window (byte-neighborhood, can miss neighbours); quantized (v4) -> 2 B per user dim
NO_CODE:      htt import-* input.x out.htt -> htt query out.htt /path k
```

## ::INVARIANTS

```
GUARANTEE: bit-identical replay: same operations, same bytes, any platform
REQUIRES:  g_math 0.7 Q64_64 (Fixed<i128, 64>); one g_math minor across horon and horon-engine
ENSURES:   WAL doubles as replication; golden CRCs pin the format
NEVER:     floats in the compute path; f64 params/returns on new API (FixedPoint only)

GUARANTEE: the format is owned by its specification
REQUIRES:  any on-disk change updates docs/HTT_FORMAT.md + regenerates tests/corpus
           + determinism goldens, in the same commit
ENSURES:   independent readers built from the spec alone stay valid
NEVER:     change magic "HTT\0", the .htt extension, or flag semantics silently

GUARANTEE: crash safety by construction
REQUIRES:  apply-then-log order; per-entry CRC32; atomic compaction (tmp + fsync + rename)
ENSURES:   kill -9 mid-write, reopen, committed data is there; torn tails truncate loudly

GUARANTEE: one writer per file
REQUIRES:  unix advisory flock (non-unix: no-op, single-process access is the caller's job)
NEVER:     treat GACL as a security boundary; cooperative query-scoping only

CONSTRAINT: semantic_dims <= 255 (16 reserved + <=239 user) | key/meta lengths <= u16 | bounded entry data
CONSTRAINT: quantized v4 rejects out-of-range values (never clamps); pre-v4 readers refuse v4 loudly
CONSTRAINT: geometry is derived state, not persisted; lazily-loaded stores re-embed (embed/embed_all)
```

## ::GRAPH

```
DEPENDS_ON:   horon-engine (sibling path dep until published; version 0.8), g_math 0.7, zstd, crc32fast, memmap2
PROVIDES_TO:  htt CLI (feature import) | any language reader written from HTT_FORMAT.md
RE_EXPORTS:   Store, StoreError, SemanticOutlier, SemanticDisk; the engine is reachable through this crate alone
ORACLES:      tests/corpus_conformance.rs (corpus byte-identity) | tests/determinism.rs (golden CRCs)
              | tests/api_surface.rs (compile-time signature lockfile) | tests/vocabulary.rs (naming)
```

---

Built by **Niels Erik Toren** · [support](README.md#author--support) · Apache-2.0
