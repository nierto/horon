# Quantized Semantic Storage (TQ1.9)

**Status:** shipped in format v4 (gFile v0.7.0) · opt-in at file creation ·
default OFF is byte-identical to v2/v3 output.

## What it is

An opt-in on-disk encoding for semantic coordinates: **user dims (16+) are
stored as TQ1.9 balanced-ternary fixed-point — 2 bytes per dimension instead
of 16** — and the reserved GACL region (dims 0–15) is **elided entirely
unless the file has GACL enabled** (in which case it stays full-width Q64.64,
so access-control semantics remain exact).

TQ1.9 is gMath's `TritQ1_9` encoding: an `i16` holding `value × 3⁹`
(scale 19 683), range ±29524/19683 ≈ **±1.49987**, uniform step
1/19683 ≈ **5.08 × 10⁻⁵** (~4.3 decimal digits). Horon owns the
Q64.64 ↔ TQ1.9 conversion (`src/quant.rs`) because gMath provides no direct
`FixedPoint` encoder; the scale constant is imported from gMath and the
rounding rule (round-half-away-from-zero) matches `TritQ1_9::from_rational`.
Ties are exact in both directions (divisors 2⁶⁴ and 19683 share no factor
with the ½-step), and `quantize(dequantize(q)) == q` for every one of the
59 049 valid raw values — pinned by an exhaustive test.

## Sizes (40-dim file, production catalog shape)

| layout                         | tail bytes/entry | vs 640 B |
|--------------------------------|------------------|----------|
| v2/v3 full-width               | 640              | 1×       |
| v4 quantized, GACL off         | **48**           | **13.3×**|
| v4 quantized, GACL on          | 304              | 2.1×     |

The tail appears in three record types (snapshot entry, `INSERT`,
`SET_SEMANTIC`), so the same ratio applies to temporal trajectory records: a
40-dim `SET_SEMANTIC` payload drops from 640 B to 48 B.

**Honest note on the original estimate:** the "8× smaller" estimate assumed
uniform quantization of all dims. Reserved-region elision beats it (13.3×)
for non-GACL files; GACL files get only 2.1× because their access bands
deliberately stay full-width. The hoped-for "zero-multiply distance kernels"
do not exist in g_math for distances (only dot products, behind an
`inference` feature that requires rayon) — no kernel claim ships with quantization.

## The contract

- **Ranking-grade distances.** Stored user dims carry ~4.3 significant
  digits. Semantic query *rankings* may differ from a full-width file where
  true distances differ by less than the quantization noise; near-equal
  distances collapse into exact ties (broken deterministically by key).
- **Still deterministic.** Quantization is a fixed, pure-integer function —
  no floats, no platform variance. Same writes → byte-identical file, on any
  platform, including across close/reopen (see write-through below).
- **Range is enforced, not clamped.** `set_semantic` on a quantized file
  **rejects** user-dim values outside ±1.49987 (`InvalidOperation`, names
  the offending dim). Silent saturation would corrupt rankings invisibly.
  Max-normalized affinities ([0, 1]) fit with headroom.
- **Reserved dims (0–15) must be zero** in a quantized file without GACL —
  they are not stored, so `set_semantic` rejects nonzero values there
  rather than silently dropping them. With GACL, they are stored at full
  Q64.64 precision and access decisions are bit-exact.
- **Write-through canonicalization.** The in-memory store holds
  `dequantize(quantize(v))` — the on-grid value — not the caller's original.
  `get_semantic` returns the same bytes before and after reopen; in-memory
  query results equal post-reload results. This is what makes determinism
  survive the storage boundary.

## Where quantization lives (and doesn't)

Entirely at the Horon serialization edge. `NodeEntry` / `WalEntry` hold
**full-width canonical bytes in memory**; encoding to the 2-byte layout
happens only in `write_to`, decoding only in `read_from` (parameterized by
`SemLayout`, `src/quant.rs`). Consequences, all by construction:

- The engine is untouched — its API still receives 16-byte-aligned Q64.64.
  VP-tree pruning across the semantic index, disk and analytics stays sound because quantize-on-store
  makes storage the single source of truth for both index build and query.
- WAL replay, partial-mode promotion, compaction fence re-serialization,
  and replication fan-out (`subscribe_wal` / `wal_entries_since`) all
  operate on full-width entries; re-encoding is stable because grid values
  round-trip exactly.
- All three Hilbert decode sites (`decode_user_dims`,
  `hilbert_snapshot_ranks`, the partial-scan closure) consume full-width
  bytes; the partial-scan closure decodes the disk tail first. Meaning-
  addressing composes with quantization (v4 + `FLAG_MEANING_ADDRESSED`).

## Format

- **Flag bit 7** (`FLAG_QUANTIZED_SEMANTIC = 0x80`) + **format v4**;
  the flag requires v4, mirroring the bit-6/v3 gate. This spends the last
  flag bit — any future format feature needs a v5 extended header.
- Tail layout when the flag is set:
  `[reserved: min(dims,16) × 16 B Q64.64 — present only when FLAG_GACL]`
  `[user: (dims−16) × 2 B i16 LE TQ1.9]`.
- History sidecars: segment version 2, with the layout recorded in the
  formerly-reserved byte at offset 6 (bit 0 quantized, bit 1 GACL) —
  segments stay self-describing; v1 readers reject v2 segments loudly.
- Requires `semantic_dims > 16` (a quantized file with only reserved dims
  would store nothing). Composes with compression, GACL, meaning-addressing,
  partial reads, and history retention.
- Simple-file-pays-nothing: without the flag, every code path serializes
  exactly as v2/v3 — the pre-quantization determinism golden CRC is unchanged, and a
  second golden CRC pins the quantized layout.

## Rejected forks (design decisions, 2026-07-11)

1. **Uniform 2 B for all dims incl. GACL bands** — simplest, exactly the
   estimated 8×, but access-band edges would move by up to 2.5 × 10⁻⁵,
   making access decisions quantization-dependent. Access is exact-compare
   semantics, not ranking-grade. Rejected.
2. **User dims 2 B, reserved always full-width** — access exact, but only
   2.1× on catalog-shaped files; the headline claim dies. Rejected.
3. **Through-stack i16 dims in the engine** — 8× RAM too, future quantized
   kernels; but changes the engine's alignment contract and decode choke point
   for a kernel that doesn't exist yet. Deferred, not rejected — revisit if
   RAM or benchmarks demand.
4. **Saturate on out-of-range** — never blocks a write, but silently
   flattens any pipeline emitting values beyond ±1.5. Rejected for the
   honesty of the failure mode.
5. **Re-quantize-on-compact migration** — in-place format mutation of live
   files, durability tests and a downgrade story, paid before any file
   needs migrating. Creation-time opt-in only; the compaction hook
   (`geofile.rs` pad block) remains the place to add it if a real file
   ever needs converting.

## Known limits

- `find_outliers` z-scores are ratios of small differences; in
  low-variance populations quantization noise can flip flags near the
  threshold. Ranking-grade contract covers it; measure before tightening
  any threshold on quantized data.
- The semantic disk drops negative weights while the index's Euclidean metric
  uses them signed. Quantized files inherit this split unchanged; dims
  intended for the disk should be non-negative affinities.
- A pre-v4 reader (e.g. an older standalone `HttReader`) rejects v4
  files at the version check — loud, by design. Quantization is per-file
  opt-in; existing files never change.
