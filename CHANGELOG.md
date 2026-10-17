# Changelog

## 0.16.0 - 2026-10-17

### Changed

- **g_math 0.6 → 0.7**, with horon-engine. 0.7 makes the fixed-point format a
  type and removes `FixedPoint` and `GMATH_PROFILE`; horon uses
  `g_math::fixed_point::Q64_64`, the Q64.64 format the embedded profile
  selected. Not a disk-format change: the corpus and both determinism suites
  regenerate byte-identically. Recomputed distances and scores can differ
  from 0.15.1 in the last place (0.7 rounds the transcendentals correctly),
  so do not mix query results across the boundary in one dataset.
- Picks up horon-engine's fixes for a small `tau` (which now panicked on the
  first insert instead of silently wrapping) and for outside-disk queries.

## 0.15.1

### Fixed

- **`get_meta` returned a different shape under `partial_reads`.** The partial
  path builds its map from the persisted entry and was still inserting `key`
  and `size`, which `horon-engine` 0.8.0 stopped synthesising on the normal
  path. The same call therefore returned four keys or two depending on a config
  flag the caller may not have set deliberately. Both paths now return
  caller-supplied metadata only. Missed in 0.15.0 because nothing exercises
  `get_meta` under partial reads.

## 0.15.0

Determinism by default. Reported by gMaterials (2026-09-05), which builds a
154 248-node store and hashes the file: four runs over identical input gave
four different hashes.

### Fixed

- **`.htt` bytes depended on thread scheduling above the auto-compaction
  threshold.** Crossing `auto_compact_threshold` (default 10 000 entries)
  handed compaction to a detached `std::thread`. That thread recorded its
  fence — `wal.next_seq` — whenever the scheduler got to it, while the writer
  kept appending. Where the fence lands decides which entries fold into the
  snapshot and which stay in the WAL, and the snapshot is compressed as a unit,
  so two runs over the same input produced different files. Logical content was
  never affected; only the layout, and therefore any hash of it.

  Auto-compaction now runs **synchronously on the writer**, during the write
  that crosses the threshold, so the fence is that write's sequence number and
  the split is a pure function of the input. Cost: one compaction stall per
  `auto_compact_threshold` writes.

  Below the threshold nothing changed, which is why every existing test passed —
  `tests/determinism.rs` sets `auto_compact_threshold: 0`.

### Added

- **`AutoCompactMode`**, on `HoronConfig::auto_compact_mode`. `Sync` is the
  default and is the reproducible one. `Background` restores the pre-0.15.0
  detached thread for writers that must never stall, and documents both of its
  costs: bytes vary run to run, and the file stays locked briefly after drop
  because the compaction thread outlives the handle.

- **`tests/determinism_above_threshold.rs`** — the case the suite was missing.
  Its fixture size is measured, not guessed: 130 nodes at threshold 40 passed
  under `Background` too, so it would have passed with or without the fix.
  Sweeping for the scale that actually reproduces gave 2000 nodes at threshold
  100 under contention, where `Background` produced 5 distinct hashes in 5
  builds and `Sync` produced 1. Shrinking that fixture would quietly restore a
  test that cannot fail.

### Changed

**Breaking**: `HoronConfig` gains a field, so struct-literal construction
without `..Default::default()` needs updating. Requires `horon-engine 0.8`,
whose `get_meta` no longer injects wall-clock fields.
