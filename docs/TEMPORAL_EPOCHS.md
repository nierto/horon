# Temporal Epochs: opt-in history retention + temporal reader

Status: **shipped 2026-07-10**.
Companion: `horon-engine`'s `docs/SEMANTIC_INDEX.md` — shares the epoch model.
Format details: `HTT_FORMAT.md` §4.1. Tests: `tests/temporal.rs`.

## The insight

The WAL already records the movement of the manifold: every `set_semantic`
is a seq-ordered, CRC-guarded `OP_SET_SEMANTIC` entry; every structural
change is an `OP_INSERT`/`OP_DELETE`. Trajectory information is **already
being written** — and then `compact()` truncates it. The temporal pillar is
the decision to stop discarding history we already pay to record, plus the
minimal machinery to address it: epoch markers, retention across compaction,
and a reader for time-scoped queries.

Deltas beat snapshot series on every axis that matters here:

- **Cost:** an epoch update touches only what moved (`set_semantic` ≈ 8 µs,
  d × 16 B payload). A snapshot series re-embeds every node (Sarkar ≈ 6.4 ms
  per node) and re-states everything that didn't change, every time.
  Delta log = O(changes); snapshot series = O(everything × epochs).
- **Fidelity:** diffing snapshots recovers only net displacement between
  sample times (an excursion up-and-back is invisible). The WAL records the
  actual path through coordinate space, in causal order.
- **Identity:** node identity is the path — stable across epochs by
  construction — so a trajectory is just the sequence of a key's
  `SET_SEMANTIC` entries.

## Non-negotiable: the simple htt pays nothing

Temporal is **opt-in**. With retention off (the default), behavior is
byte-identical to today: `compact()` truncates the WAL, no epoch entries
exist unless written, no sidecars appear, no read/write/recovery path
changes. A user of htt as a fast hierarchical KV store never encounters
this layer. Verified by an off-mode identity test (same op sequence with
and without the feature compiled path → identical main-file bytes).

## Design

### 1. Epoch markers — `OP_EPOCH` (0x06)

A new WAL op, appended by `Horon::seal_epoch()`:

```text
payload: epoch_id (u64 LE) | flags (u8)    // bit 0: SPECULATIVE
key:     "" (empty — epochs are file-scoped, not node-scoped)
```

- `epoch_id` is a monotonically increasing logical counter assigned by
  `seal_epoch()` — **not wall-clock time**, preserving determinism
  (replays are bit-identical; map epoch → date in metadata if a human
  label is wanted).
- Replay treats `OP_EPOCH` as a state no-op that advances the in-memory
  `current_epoch`. After compaction (any retention mode), the fresh WAL is
  seeded with one `OP_EPOCH` restating the current counter — flagged
  `RESTAMP` so readers know it ferries the counter and is not a seal
  moment — so the epoch survives truncation without any header change.
- `SPECULATIVE` (via `seal_speculative_epoch()`) marks projected/what-if
  seals written by external tools (forecasting is explicitly **out of
  scope** — see below) so fiction can never silently pass as recorded
  history. Honest semantics: **the flag labels, it does not isolate** —
  speculative *writes* physically remain in the log, so speculative work
  belongs in a copy of the file. Sampling APIs (`trajectory`) skip
  speculative seals; `as_of` answers for any epoch you explicitly name.

**Compatibility (honest statement):** `OP_EPOCH` is a new WAL op code.
Files that never call `seal_epoch()` never contain it and remain readable
by every existing tool. A pre-epochs reader opening a live WAL that *does*
contain `OP_EPOCH` fails with unknown-op — acceptable for private-repo
stage; noted in HTT_FORMAT.md. The main-file header, version byte, and
snapshot layout are untouched.

### 2. Retention across compaction — sidecar history segments

`HoronConfig` gains `history_retention: HistoryRetention` —
`Off` (default) | `Archive`. With `Archive`, `compact()` moves the old WAL
into a sealed, zstd-compressed sidecar instead of truncating it:

```text
data.htt          — main file, format unchanged, always self-sufficient
data.htt.h000001  — history segment 1 (oldest), zstd, own header + CRC
data.htt.h000002  — history segment 2, ...
```

- **Crash safety:** archive-then-truncate, in that order; the segment is
  written to a temp name, fsynced, renamed (same discipline as compact's
  snapshot rename). A crash between archive and truncate leaves a duplicate
  span — segments carry `(first_seq, last_seq)` in their header, so the
  reader detects and skips overlap. Idempotent, never lossy.
- **Degradation:** deleting the sidecars produces a plain, fully working
  htt file that has simply forgotten its past. History is a layer above
  the format, not a fork of it.
- Live read/query/recovery paths are untouched — recovery still replays
  only the main file's post-snapshot WAL tail.

### 3. Temporal reader — `HoronHistory`

Read-only, opens main file + sidecars, one sequential scan (segments are
cold; this is an analysis path, not a hot path):

```rust
HoronHistory::open(path) -> Self
  .epochs() -> Vec<EpochInfo>                      // id, seq span, speculative
  .as_of(epoch) -> HoronStateView                    // full state at that seal
  .trajectory(key, dim_range) -> Vec<(epoch, Vec<FixedPoint>)>
  .delta(epoch_a, epoch_b) -> Vec<KeyDelta>        // who moved, which dims, how far
```

- `as_of` = replay segments + live WAL up to the epoch marker (the
  "rerun the population" operation, mechanized). O(history) — acceptable
  cold; per-epoch checkpoints are a later optimization if needed.
- `trajectory` = filter the scan by key; the per-epoch coordinate is the
  key's **last** `SET_SEMANTIC` at or before each seal (intra-epoch churn
  is preserved in the log and can be exposed later; the epoch-grain API
  ships first).
- `delta` = the input to drift, demand-void-velocity, and
  intervention-response analytics (targeted segment vs. untouched controls
  in the same file).

### 4. Interaction with the engine's semantic index

`seal_epoch()` marks the calibrate → query transition. Within an epoch the
manifold is frozen, so the engine's lazy per-slice VP-trees stay valid; the
epoch counter is the natural invalidation signal. The two features are
independently useful and independently shippable — the index works without epochs
(any semantic write bumps its internal counter), and epochs work without
the index.

## Known limits (stated up front)

- `SET_SEMANTIC` stores the full coordinate vector (40 dims → 640 B) even
  for a 1-dim change. Fine at current scales; a sparse per-dim delta op is
  a possible later refinement, **out of scope** for v1.
- WAL `seq` is `u32`; a multi-year hot temporal file could approach it.
  The existing sequence guard errors cleanly rather than wrapping; segment
  headers use the same seq type. Documented, not blocking.
- `as_of()` cost grows linearly with retained history until checkpoints
  are added.
- **Projection/forecasting is not part of htt.** The substrate records and
  replays deterministically; extrapolation (trend fitting, seasonal models,
  speculative-epoch generation) belongs to the application/analysis layer.
  htt's only concession is the `SPECULATIVE` flag, so projected states are
  permanently distinguishable from recorded ones.

## Verification plan

- Off-mode identity: same op sequence, retention off → main file
  byte-identical to pre-epochs behavior; no sidecars.
- Round-trip: build → seal → mutate → seal → compact → mutate → seal;
  `as_of(e)` equals a state snapshot captured at each seal.
- Crash-safety sweep (CI=1 densified): kill between archive and truncate →
  reopen, no data loss, overlap skipped, `epochs()` correct.
- Trajectory correctness: known coordinate script → exact expected
  per-epoch vectors, including keys created/deleted mid-series.
- Speculative isolation: speculative epochs excluded by default, included
  on request, never merged into non-speculative reads.
