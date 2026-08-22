# HTT Binary Format Specification (versions 1–4)

## Overview

`.htt` (Hyperbolic Tree Tensors) is a persistence format for hierarchical,
semantically-addressed stores. Horon reads and writes `.htt`; the format is
defined by this document, not by any implementation. It combines a point-in-time
snapshot with an append-only write-ahead log (WAL). The WAL is the default and
primary persistence mechanism — snapshots are an optimization produced by
compaction.

**Design principles:**
- Raw binary, no external schema dependency (no protobuf, bincode, etc.)
- Any language can read/write with just this spec document
- WAL-first: every mutation is an append; snapshots are derived by compaction
- gMath determinism: identical replay → identical geometric state on any platform
- Compression optional, algorithm-tagged, block-level for WAL

**Key insight:** the store's geometric state (Sarkar coordinates, Klein points, power
cells, VP-trees, point location grid) is fully reconstructible from tree topology +
config. The file only stores keys, payloads, metadata, and semantic coordinates.
On load, geometric state is rebuilt by replaying inserts through the Sarkar
construction — gMath's 0-ULP fixed-point guarantees identical results everywhere.

---

## 1. File Structure

```
┌──────────────────────────────────┐
│ File Header          (32 bytes)  │
├──────────────────────────────────┤
│ Bounds Section       (v3 only)   │  ← present only when flag bit 6
├──────────────────────────────────┤    (meaning-addressed) is set
│ Snapshot Section     (variable)  │
├──────────────────────────────────┤
│ WAL Section          (variable)  │
└──────────────────────────────────┘
```

The bounds section exists only in meaning-addressed files (flag bit 6,
format v3+): `user_dims × 16` bytes of per-dimension global normalization
min/max as f64 LE pairs, immediately after the 32-byte header. Readers MUST
account for it before parsing the snapshot header — see §9's version-3
entry. Ordinary files go straight from header to snapshot.

A freshly created file has an empty snapshot and the WAL grows with each
mutation. Compaction folds the WAL into a new snapshot and resets the WAL.

---

## 2. File Header (32 bytes, fixed)

All multi-byte integers are **little-endian**.

```
Offset  Size  Field          Description
─────────────────────────────────────────────────────────
 0       4    magic          b"HTT\0" (0x48 0x54 0x54 0x00)
 4       1    version        Format version (1–4)
 5       1    flags          Bit field (see below)
 6       1    dimension      Poincaré disk structural dimension (e.g. 4)
 7       1    semantic_dims  Number of semantic dimensions (0 = none)
 8      16    tau            Sarkar spacing parameter, Q64.64 as i128 LE
24       4    node_count     u32 LE — total live nodes after full replay
28       4    header_crc     CRC32 of bytes 0..27
─────────────────────────────────────────────────────────
Total: 32 bytes
```

### Flags byte (offset 5)

```
Bit  Meaning
───────────────────────────────────────
 0   Compression enabled (0 = none, 1 = enabled)
1-2  Compression algorithm (when bit 0 = 1):
       00 = zstd (RFC 8878) — default
       01 = zlib/deflate (RFC 1951)
       10 = lz4
       11 = reserved
 3   Has semantic dimensions (0 = structural only)
 4   WAL entries are block-compressed (set together with bit 0)
 5   GACL enforcement enabled for this file
 6   Meaning-addressed layout (requires format v3)
 7   Quantized semantic tails (requires format v4) — see §7.3
```

Bit 7 is the LAST flag bit; any future format feature needs a v5 extended
header.

### Field rationale

- **version u8**: 255 format revisions. Breaking change = new version.
- **dimension u8**: max 255. engine default is 4; deployments have used up to 23. No realistic need past 255.
- **semantic_dims u8**: max 255 semantic axes. Combined with dimension, total embedding space ≤ 510D.
- **tau as Q64.64**: full fixed-point precision preserved. 16 bytes = one FixedPoint value.
- **node_count u32**: max ~4 billion live nodes per file. Updated on compaction.
- **header_crc**: CRC32 of header bytes 0-27. Detects header corruption on open.

---

## 3. Snapshot Section

Immediately follows the header at offset 32.

### Snapshot Header (8 bytes)

```
Offset  Size  Field
──────────────────────────────────
 0       4    snap_byte_len    u32 LE — total bytes of snapshot data
                               (decompressed size if compression enabled)
 4       4    snap_node_count  u32 LE — number of node entries
```

### Snapshot Data

When compression is **disabled**: raw node entries follow immediately.
When compression is **enabled**: a `compressed_len` u32 LE, then a single
compressed frame containing all node entries. An empty snapshot always uses
the uncompressed layout regardless of the compression flag.

**Format v2**: a `snapshot_crc` u32 LE follows the snapshot data (after the
raw entries, or after the compressed frame) — CRC32 of the raw
(decompressed) entry bytes. Verified on open; a mismatch fails the open
with a checksum error instead of silently loading corrupt data. Present
even for empty snapshots (CRC of zero bytes).

Node entries are stored **parents-before-children** — the one property
Sarkar replay requires. Within that constraint the order depends on how the
snapshot was produced:

- plain writes produce **insertion order**;
- compaction of a file with user semantic dims sorts by
  `(depth, hilbert_index, key)` — still parents-first, with semantically
  similar entries byte-adjacent within each depth level;
- v3 meaning-addressed files use **pure global-Hilbert order** instead
  (full-geometry readers depth-sort in memory before replay — see §10).

Readers MUST NOT assume any ordering beyond what the version guarantees.
Each entry:

```
Field           Type      Condition       Notes
────────────────────────────────────────────────────────────────
key_len         u16 LE    always          Path length in bytes
key             [u8]      always          UTF-8 path (e.g. "/a/b/c")
data_len        u32 LE    always          Payload length in bytes
data            [u8]      always          Raw payload bytes
meta_count      u16 LE    always          Number of metadata k/v pairs
  mk_len        u16 LE    × meta_count   Metadata key length
  mk            [u8]      × meta_count   Metadata key (UTF-8)
  mv_len        u16 LE    × meta_count   Metadata value length
  mv            [u8]      × meta_count   Metadata value (UTF-8)
sem_coords      [u8]      semantic_dims>0 semantic_dims × 16 bytes
                                          (Q64.64 per dimension, LE)
```

**Parent is NOT stored.** The parent of `/a/b/c` is `/a/b` — derivable from
the key itself. The root `/` has no parent, and is not a node: it never
appears as an entry and readers must not synthesize it.

### Metadata namespaces

Metadata keys beginning with `_` are **reserved for the format**. They appear
in the metadata block like any other pair, but they are the writer's own
state, not user data: readers SHOULD NOT present them as user metadata, and
any tool that rewrites a node MUST preserve them byte-for-byte. Currently
defined:

- `_child_index` — the node's sibling position, written so that replay is
  order-insensitive beyond the parents-first guarantee. Corrupting or
  dropping it degrades deterministic Sarkar reconstruction.

Writers MUST reject user attempts to set `_`-prefixed keys.

Conversely, some metadata returned by the reference read API is **synthesized
at load time and never stored in the file**: `key`, `size`, `created_at`,
`updated_at`. No reader working from the bytes can recover them (the
timestamps are the *loading* process's wall clock, not creation times), and
no conformance expectation may reference them. Everything else in the
metadata block — including `content_type` — is persisted and MUST be
surfaced.

An empty snapshot has `snap_byte_len = 0` and `snap_node_count = 0`
(v2: followed by the 4-byte CRC of zero bytes).

---

## 4. WAL Section

Immediately follows the snapshot section.

### WAL Header (8 bytes)

```
Offset  Size  Field
──────────────────────────────────
 0       4    wal_entry_count  u32 LE — total WAL entries
 4       4    wal_base_seq     u32 LE — sequence number of first entry
```

The sequence counter is monotonically increasing across compactions.
After compaction, `wal_base_seq` continues from where it left off.

### WAL Entries

When compression is **disabled**: raw entries follow sequentially.
When compression is **enabled**: entries are grouped into blocks (see §5).

Each WAL entry:

```
Field           Type      Condition         Notes
────────────────────────────────────────────────────────────────
seq             u32 LE    always            Monotonic sequence number
op              u8        always            Operation code (see below)
key_len         u16 LE    always            Path length
key             [u8]      always            UTF-8 path

--- INSERT (op = 0x01) ---
data_len        u32 LE
data            [u8]                        Payload bytes
meta_count      u16 LE
  mk_len/mk/mv_len/mv                      (same as snapshot format)
sem_coords      [u8]      semantic_dims>0   semantic_dims × 16 bytes

--- UPDATE (op = 0x02) ---
data_len        u32 LE
data            [u8]                        New payload bytes
meta_count      u16 LE                      Full metadata replacement
  mk_len/mk/mv_len/mv

--- DELETE (op = 0x03) ---
(no additional fields)

--- SET_META (op = 0x04) ---
mk_len          u16 LE                      Single metadata key
mk              [u8]
mv_len          u16 LE                      Single metadata value
mv              [u8]

--- SET_SEMANTIC (op = 0x05) ---
sem_coords      [u8]      semantic_dims>0   semantic_dims × 16 bytes
                                            Full semantic coord replacement

--- EPOCH (op = 0x06) ---
epoch_id        u64 LE                      Logical epoch counter (1-based,
                                            monotonic; NOT wall-clock)
flags           u8                          bit 0: SPECULATIVE
                                            bit 1: RESTAMP (compaction
                                            counter carry-over, not a seal)

--- end op-specific ---

entry_crc       u32       always            CRC32 of this entry
                                            (bytes from seq through last
                                            op-specific field, excluding
                                            entry_crc itself)
```

### Operation codes

```
Code  Name          Effect
──────────────────────────────────────
0x01  INSERT        Create node (error if exists)
0x02  UPDATE        Replace data+metadata (error if not exists)
0x03  DELETE        Remove node
0x04  SET_META      Set one metadata key/value on existing node
0x05  SET_SEMANTIC  Update semantic coordinates on existing node
0x06  EPOCH         Seal marker (no node state change; replay advances
                    the in-memory epoch counter). See §4.1.
0x07-0xFF           Reserved
```

### Implicit ancestors

Inserting `/a/b/c` implicitly creates `/a` and `/a/b` as nodes: empty data,
metadata `content_type: application/x-directory`, no semantic placement.
Where that creation is recorded depends on the writer's age:

- **Current writers** log an explicit INSERT for each ancestor they create,
  parents-first, immediately before the entry that triggered it. The WAL is
  therefore complete: a replica built purely by tailing it converges to the
  primary. A concurrent-writer race can log the same ancestor twice; replay
  MUST tolerate INSERT of an existing key by skipping it.
- **Files written before this rule** contain no ancestor entries in the WAL
  (compacted snapshots always contained them). Readers MUST therefore
  synthesize missing ancestors during WAL replay — for every replayed key,
  ensure each ancestor path exists as a directory node as described above.

Ancestor synthesis is idempotent over explicit entries, so a single replay
implementation (synthesize after replay) handles both eras correctly.

### 4.1 Temporal epochs and history sidecars

`seal_epoch()` appends an `EPOCH` (0x06) marker. Files that never seal an
epoch never contain the op and remain readable by pre-epochs tools; a
pre-epochs reader encountering `EPOCH` in a live WAL fails with unknown-op
(accepted while the format was pre-publication). Compaction seeds the fresh WAL with a
RESTAMP-flagged marker restating the current counter so the epoch survives
truncation; re-stamps are not seal moments and are ignored by temporal
sampling. The SPECULATIVE flag labels projected/what-if seals — it labels,
it does not isolate (speculative writes belong in a copy of the file).

With `history_retention: Archive`, compaction archives the pre-fence WAL
span into a sidecar **history segment** instead of discarding it:

```
<file>.h000001, <file>.h000002, …   (numbered, oldest first)

Offset  Size  Field
──────────────────────────────────────────────
 0       4    magic         "HTTH" (0x48 0x54 0x54 0x48)
 4       1    version       currently 1
 5       1    semantic_dims must match the main file header
 6       2    reserved      zero
 8       4    first_seq     u32 LE — first archived sequence
12       4    end_seq       u32 LE — exclusive end of the span
16       4    raw_len       u32 LE — uncompressed entry bytes
20       4    comp_len      u32 LE — zstd-compressed length
24     ...    zstd(entries) plain-serialized WAL entries (§4 layout,
                            never WAL-block-framed, regardless of the
                            main file's compression)
 +       4    crc32         of the RAW (uncompressed) entry bytes
```

Segments are written crash-safely (tmp + fsync + rename) *before* the
compaction rename, so a crash in between merely re-archives the same span
under the next number — readers deduplicate by sequence. The main file is
always self-sufficient: deleting the sidecars yields a plain htt that has
simply forgotten its past. See `docs/TEMPORAL_EPOCHS.md` for the design and
the `HoronHistory` reader (`epochs` / `as_of` / `trajectory` / `delta`).

### Per-entry CRC

Every WAL entry ends with a CRC32 of its own bytes. On recovery, the reader
verifies each entry's CRC. The first entry with a bad CRC indicates a partial
write (crash/power loss) — the reader truncates the WAL at that point.
All preceding entries are guaranteed intact.

---

## 5. Compression

When the compression flag is set (flags bit 0 = 1):

### Snapshot compression

The entire snapshot data section is a single compressed frame.
The `snap_byte_len` field records the **decompressed** size.
After the snapshot header, the file contains:

```
compressed_len  u32 LE    Compressed frame size in bytes
compressed_data [u8]      Compressed frame (zstd/zlib/lz4)
```

### WAL block compression

WAL entries are grouped into blocks of up to **64 entries**. Each block is an
independent compressed frame (decompressible without prior blocks).

```
Block structure:
  block_entry_count  u16 LE    Entries in this block (1-64)
  compressed_len     u32 LE    Compressed frame size
  compressed_data    [u8]      Compressed frame containing raw entries
```

**A block is written once and never modified.** Each flush takes the
entries pending at that moment and emits them as one or more fresh blocks,
chunked at a maximum of 64 entries per block — appending to the file, never
rewriting an existing block. Block sizes therefore mirror the writer's
flush batching: a writer that flushes every append produces single-entry
blocks (the default configuration does exactly this), a writer batching N
entries produces N-entry blocks. Readers MUST NOT assume any particular
entry count per block beyond the 1–64 range, and writers MUST NOT
decompress, extend, or rewrite a previously written block — the append-only
property is what makes a crash mid-write affect only the final block.

**Rationale:** up to 64 entries × ~200 bytes avg = ~12KB per block. Large
enough for good compression ratio, small enough for fast decompression. A
reader can decompress individual blocks without touching the rest of the
WAL.

### Algorithm selection

The algorithm is encoded in flags bits 1-2. Readers MUST support zstd
(the default). Support for zlib and lz4 is optional.

Writers SHOULD use zstd unless there is a specific reason not to.

---

## 6. Lifecycle

### Create

1. Write header (32 bytes): magic, version (the highest version required by
   the enabled features — 2 for a plain file, 3 with meaning-addressing, 4
   with quantized tails), flags, dimension, semantic_dims, tau,
   node_count=0, header_crc
2. Write empty snapshot header: snap_byte_len=0, snap_node_count=0
3. Write WAL header: wal_entry_count=0, wal_base_seq=1

### Write (put/remove/set_meta)

1. Construct WAL entry with next sequence number
2. Compute entry_crc
3. Append entry to WAL section (compressed files group each flushed batch
   into one block of up to 64 entries)
4. Increment wal_entry_count in WAL header (advisory — see §4)
5. Optionally fsync (configurable: per-op, batched, or manual).
   `node_count` in the file header is updated by compaction only.

### Read (cold start)

1. Verify header magic and header_crc
2. Read snapshot: bulk-insert all nodes in order (Sarkar reconstruction)
3. Replay WAL entries in sequence order, verifying each entry_crc
4. Stop at first bad CRC (truncate corrupted tail, log warning)
5. Geometric state is now fully built — ready to serve queries

### Compaction

1. Build current state by reading snapshot + replaying WAL (or use in-memory state)
2. Write new snapshot section with all live nodes, ordered per §3's Snapshot
   Data rules (parents-first always; `(depth, hilbert_index, key)` for files
   with user semantic dims; pure global-Hilbert for v3)
3. Reset WAL: wal_entry_count=0, wal_base_seq=last_seq+1
4. Update header: node_count, header_crc
5. fsync

Compaction can be triggered by:
- `store.compact()` — explicit
- WAL entry count ≥ `auto_compact_threshold` — automatic (0 disables;
  runs on a background thread off the writer)

### Recovery

Same as cold start. The per-entry CRC guarantees that partial writes from
crashes are detected and truncated cleanly. No manual intervention needed.

---

## 7. Semantic Coordinates

When `semantic_dims > 0` (flags bit 3 = 1), each INSERT entry and each snapshot
node entry includes `semantic_dims × 16` bytes of Q64.64 fixed-point coordinates
(or the quantized tail of §7.3 when flag bit 7 is set — the same three record
types, a different fixed-size tail).

These represent user-defined semantic axes (for example category, capability,
or telemetry dimensions).

**Structural and semantic coordinates are separate**, and are indexed
separately. They are not concatenated into one vector.

**On load**, the reader:
1. Reconstructs structural coordinates via Sarkar construction (from tree
   topology) — these are never stored in the file
2. Reads semantic coordinates from the file and attaches them to the node
3. Builds the *structural* spatial index over structural coordinates only
4. Serves semantic queries from a *separate* index over the semantic vectors
   (dimension slices), and concept queries from the semantic disk, which
   derives its own position as a barycenter of concept anchors

Consequently a node has up to three distinct positions: its structural place in
the tree's disk, its raw semantic vector, and — if a concept taxonomy is
configured — a derived position in a second disk. A query answers against
exactly one of them.

> **Corrected 2026-08-22.** Earlier revisions of this section described the
> reader as concatenating `[structural | semantic]` and building one index over
> the result. No release has ever done that; the two coordinate systems have
> always been stored and indexed separately. The description, not the
> behaviour, was wrong.

The SET_SEMANTIC op (0x05) allows updating semantic coordinates without
replacing the node's data or metadata.

### The zero tail means "not set"

Every record's semantic tail is fixed-size and always present, so the format
has no out-of-band way to say "this node has no coordinates": **the all-zero
tail IS the encoding of "not set"**. Three rules follow:

1. Readers MUST treat an all-zero tail (after dequantization, for v4) as
   "no coordinates", not as a placement at the origin. A node with a zero
   tail does not participate in semantic queries.
2. Writers MUST reject an explicit placement of all zeros (mirroring §7.3's
   reject-don't-saturate rule) — a placement at the exact origin is not
   representable. Placements near the origin need at least one nonzero
   dimension.
3. A SET_SEMANTIC record whose tail is all zeros is the **clear** operation:
   it removes the node's placement. Replay MUST restore "not set", giving
   identical state through the WAL and snapshot paths.

On GACL files the access bands share the semantic vector (dims 0–11), so a
node with bands but no user placement has a nonzero tail and is unaffected
by rule 1; clearing such a node removes its bands along with the placement.

### 7.1 Reserved Dimension Table

Semantic dimension indices 0-15 are reserved for standardized access control
and classification. Indices 16-254 are user-defined (content semantics, tags,
domain-specific axes).

Access dimensions use **lo/hi pairs** — each access concept occupies two
consecutive dimensions defining the band of credential values that grant access.

```
Index  Name                  Type     Description
──────────────────────────────────────────────────────────────────
 0     read_access_lo        lo       Read permission band lower bound
 1     read_access_hi        hi       Read permission band upper bound
 2     write_access_lo       lo       Write permission band lower bound
 3     write_access_hi       hi       Write permission band upper bound
 4     exec_access_lo        lo       Execute/invoke permission band lower bound
 5     exec_access_hi        hi       Execute/invoke permission band upper bound
 6     domain_scope_lo       lo       Organizational domain band lower bound
 7     domain_scope_hi       hi       Organizational domain band upper bound
 8     classification_lo     lo       Data classification band lower bound
 9     classification_hi     hi       Data classification band upper bound
10     identity_lo           lo       User/service identity band lower bound
11     identity_hi           hi       User/service identity band upper bound
12-15  (reserved)            —        Future access concepts
──────────────────────────────────────────────────────────────────
16+    user-defined          any      Content semantics, tags, categories,
                                      topic vectors, similarity axes, etc.
```

### 7.2 Geometric Access Control (GACL)

Access rules are encoded directly in a node's semantic coordinates.

> **What ships (and the scope of this section).** GACL is *cooperative
> query-scoping*, checked on access: session credentials are set via
> `set_credentials()`, and every read/write compares them against the node's
> bands in dims 0–11 (see README "GACL"). Unreadable nodes are found by the
> index and then **filtered out of results** — they are not made invisible to
> the index. It is **not** a security boundary: raw file access, or querying
> without credentials on a file not opened fail-closed, sees everything. Use OS
> permissions or encryption for real confidentiality.
>
> A stronger "geometric invisibility" model — folding access into the distance
> function so the index never visits unauthorized nodes — was explored and is
> **not** planned; the decision was to keep GACL as a
> Horon-level cooperative filter. The subsections below document the band and
> credential *encoding* that ships; passages describing nodes as
> "geometrically distant / unreachable" describe that abandoned research idea,
> not behavior — they are retained only to explain the coordinate scheme.

**Core principle:** each node's access dimensions define a band [lo, hi] per
access concept. A requester's credential vector contains their position in each
access dimension. Access is granted when the credential falls within every
band; otherwise the node is filtered from that requester's results.

#### Credential Vectors

Users and services are assigned positions in access dimensions based on their
roles, groups, and individual identity:

```
Group hierarchy (positions in a single access axis):
  public       = 0.00       Bands that include 0.00 → public access
  employee     = 0.25       Bands [0.00, 1.00] → employee access
  contractor   = 0.35       Bands [0.25, 1.00] → employee + contractor
  engineering  = 0.50       Bands [0.50, 1.00] → engineering and above
  security     = 0.70       Bands [0.70, 1.00] → security and above
  admin        = 0.85       Bands [0.85, 1.00] → admin and root only
  root         = 1.00       Bands [1.00, 1.00] → root only
```

Cross-cutting groups use separate dimension pairs. A node in
`/engineering/classified` might have:

```
read_access:     [0.50, 1.00]   engineering+ can read
write_access:    [0.85, 1.00]   admin+ can write
domain_scope:    [0.40, 0.60]   engineering domain only
classification:  [0.70, 1.00]   security clearance required
```

#### Multi-Group Membership

A user in multiple groups takes the MAX (most permissive) position per access
dimension across all their groups. Their credential vector is the union:

```
alice: groups = [engineering, security]
  read_credential  = max(0.50, 0.70) = 0.70
  domain_credential = engineering position = 0.50
```

#### Individual Permissions

For user-specific access, the identity dimensions (10-11) encode individual
principals. Each user/service gets a deterministic position:

```
user_position = stable_hash(user_id) / MAX_HASH   → value in [0, 1)

alice = 0.4217
bob   = 0.8831
```

A node readable only by alice:
```
identity_lo = 0.4210
identity_hi = 0.4220    ← tight band, only alice falls within
```

A node readable by alice and bob:
```
identity_lo = 0.0000
identity_hi = 1.0000    ← wide band (use group dims for restriction instead)
— or use two INSERT ops with different identity bands (multi-ACE)
```

#### Inheritance Rules

The intended monotonic-restriction rule is that a child's effective band is
the intersection with its parent's, so a child can only ever be *more*
restrictive:

```
child.access_lo = max(child.access_lo, parent.access_lo)   ← can only narrow
child.access_hi = min(child.access_hi, parent.access_hi)   ← can only narrow
```

> **Not auto-enforced.** `NodeAccessBands::narrow()` computes this
> intersection, but the shipped access check reads each node's stored bands
> **as-is** and does not walk ancestors — so a child written with wider bands
> than its parent *is* checked against those wider bands. Apply `narrow()` at
> write time (or precompute effective bands) if you want hierarchical
> restriction. There is no geometric "Sarkar cone containment" guarantee that
> descendants respect an ancestor's bands; that was part of the abandoned
> geometric-invisibility model.

#### Query Model

The shipped model is **session-scoped**: call `set_credentials()` on the
Horon handle, and subsequent reads, writes, listings, and spatial/semantic
queries are filtered against those credentials until `clear_credentials()`.
Without credentials the bands are ignored (full access) — unless the file was
opened with `gacl_fail_closed`, in which case access is denied until
credentials are supplied.

The per-call, distance-integrated variants once sketched here
(`nearest_with_creds`, …, where access was to be *built into the distance
function*) belonged to the abandoned geometric-invisibility model and are **not
planned**. Filtering happens after the query, not inside the metric.

#### Comparison with Traditional ACL

> **Status caveat:** the right-hand column is the *abandoned*
> geometric-invisibility model (§7.2 lead), shown for contrast only — it does
> **not** describe shipped behavior and is **not planned**. Shipped GACL is
> check-on-access: "Never found (invisible)" and "Built into distance calc" do
> not hold; unauthorized nodes are found and then filtered. Shipped GACL's
> real properties are the middle column's "found then denied" plus an O(1)
> per-node coordinate compare (cheaper check, post-filter — not invisibility).

| Property              | Metadata ACL         | Geometric ACL (research, unshipped) |
|-----------------------|----------------------|-------------------------------------|
| Check cost            | O(k) principals      | O(1) coordinate compare  |
| Unauthorized nodes    | Found then denied    | Never found (invisible)  |
| Spatial query filter  | Post-filter          | Built into distance calc |
| Inheritance           | Tree walk            | Coordinate propagation   |
| Multi-attribute       | Multiple lookups     | Single distance calc     |
| Cross-cutting groups  | Separate dimensions  | Separate dimensions      |
| Audit                 | Parse ACL strings    | Compare coordinates      |
| Storage               | Variable-length text | Fixed: 2 dims × 16 bytes |

### 7.3 Quantized Semantic Tails (format v4)

Opt-in at file creation (flag bit 7 + version 4; design and contract in
`docs/QUANTIZED_SEMANTIC.md`). The semantic tail of every snapshot entry,
INSERT, and SET_SEMANTIC record becomes:

```
[reserved region]  min(semantic_dims, 16) × 16 bytes   Q64.64 i128 LE
                   PRESENT ONLY when flag bit 5 (GACL) is set;
                   elided (implied zero) otherwise
[user region]      (semantic_dims − 16) × 2 bytes      TQ1.9 i16 LE
```

TQ1.9: an i16 holding `value × 3⁹` (scale 19683), range ±29524/19683
≈ ±1.49987, uniform step 1/19683 ≈ 5.08e-5 (~4.3 decimal digits). Encoding
rounds half away from zero; decoding back to Q64.64 rounds half away from
zero (`round(q × 2⁶⁴ / 19683)`). The round-trip is exact for every valid
i16, which is what keeps replay/compaction/replication re-encoding stable.

Writers MUST reject (not saturate) user-dim values outside the range, and
MUST reject nonzero reserved dims when the reserved region is elided.
Writers also canonicalize on write (store `decode(encode(v))` in memory),
so in-memory state equals post-reload state — determinism survives the
storage boundary. Distances over user dims become ranking-grade
(~4.3 significant digits); byte-level determinism is unchanged.

Requires `semantic_dims > 16`. Composes with compression, GACL,
meaning-addressing (v4 wins the version byte; the bounds section and
Hilbert ordering are computed over the DECODED full-width values), partial
reads, and history retention (sidecar segments bump to version 2 and
record the layout in the byte at offset 6: bit 0 quantized, bit 1 GACL).

Sizes at 40 dims: 640 B/tail (plain) → 48 B (quantized, no GACL, 13.3×)
or 304 B (quantized + GACL — access bands deliberately stay full-width).

---

## 8. Replication

The WAL IS the replication protocol. To replicate a store:

1. **Full sync**: send the .htt file (snapshot + WAL)
2. **Incremental sync**: peer reports its last sequence number; sender ships
   WAL entries from that point forward
3. **Determinism guarantee**: gMath fixed-point is designed to produce
   identical geometric state after replay on any platform. Verified today
   on x86-64 Linux (`tests/determinism.rs` against a committed golden CRC);
   the ARM CI leg is queued to run at publication. Other platforms
   (WASM, RISC-V) are expected to match but are not yet verified.

No separate wire format is needed. WAL entries are the replication units.

*Implemented (gFile v0.5.0):* `subscribe_wal()` delivers committed entries
post-fsync in sequence order; `wal_entries_since(seq)` serves catch-up reads
and returns `SnapshotRequired { base_seq }` when compaction has folded the
requested fence into the snapshot (bootstrap by file copy, then tail).

---

## 9. Versioning and Evolution

- **version=1**: original spec — no snapshot CRC
- **version=2** (2026-07): snapshot section carries a trailing CRC32 of its
  raw entry bytes. v2 readers accept v1 files; compaction rewrites the whole
  file and upgrades it to v2. The WAL header's `entry_count` is demoted to
  advisory — readers recover entries by scanning to the first bad CRC,
  sequence break, or torn tail, so a torn header write cannot drop
  committed entries.
- **version=3** (2026-07, flag bit 6 — meaning-addressed layout): a bounds
  section follows the 32-byte header (`user_dims × 16` bytes: per-dimension
  global normalization min/max, f64 LE), and snapshot entries are stored in
  **pure global-Hilbert order** — a node's byte position is a function of
  its user semantic coordinates (dims 16+, capped at 8 axes, 12 bits/axis,
  normalized by the bounds). Requires an uncompressed snapshot.
  Full-geometry readers MUST depth-sort entries in memory before Sarkar
  replay (the stored `_child_index` metadata makes replay order-insensitive
  beyond parents-first). Partial (mmap) readers get semantic locality for
  free: a semantic query's candidates are a contiguous byte range around
  the query's Hilbert address. v3 is only written when the flag is set;
  ordinary files stay v2.
- **version=4** (2026-07, flag bit 7 — quantized semantic tails): user
  semantic dims stored as 2-byte TQ1.9, reserved GACL region elided unless
  GACL is on (see §7.3). Only written when the flag is set; ordinary files
  stay v2 (or v3 when meaning-addressed). A v4 file may also carry flag
  bit 6 — meaning-addressed order composes with quantized tails.
- New fields: add to reserved header bytes or append new sections after WAL
- New op codes: use 0x06-0xFF range
- Breaking changes: increment version byte; old readers reject unknown versions
- The format is designed to be extended without breaking existing readers

Two 2026-07 changes are **writer behaviour**, not format revisions — the
bytes they produce were always valid, so no version bump: explicit ancestor
INSERTs (§4, Implicit ancestors) and the zero-tail placement rules (§7).
Readers built to this document handle files from both before and after.

---

## 10. Reference Sizes

Minimum file size (empty store, v2): 32 (header) + 8 (snap) + 4 (snapshot CRC) + 8 (wal) = **52 bytes** (v1, without the snapshot CRC: 48 bytes)

Typical node entry overhead: 2 (key_len) + 4 (data_len) + 2 (meta_count) = **8 bytes** + key + data + metadata

WAL entry overhead: 4 (seq) + 1 (op) + 2 (key_len) + 4 (entry_crc) = **11 bytes** + key + op-specific data

Example: 10,000 nodes, 100-byte avg key, 200-byte avg data, 2 metadata pairs:
- Snapshot: ~3.5 MB uncompressed, ~800 KB with zstd (~4x ratio on structured data)
- WAL: depends on mutation rate; compaction keeps it bounded

---

## 11. Concurrency and Locking

> **PROVISIONAL.** The Unix half of this section is verified against the
> reference implementation and an empirical cross-process probe
> (`examples/lock_probe.rs`). The Windows half awaits the same probe on real
> Windows hardware and may be revised. It is included now because omitting
> the locking contract entirely is the more dangerous error: a writer that
> does not participate in locking corrupts files silently.

The format's concurrency model is **single writer, multiple readers**,
enforced by whole-file kernel locks — not by the format itself. All
protection is cooperative between lock-taking processes, so the contract
below is normative for every implementation that writes:

- A **writer** MUST hold an exclusive whole-file lock on the `.htt` for the
  duration of its write session: `flock(LOCK_EX)` on Unix; on Windows the
  intended mechanism is `LockFileEx` over the whole file (pending
  verification). Implementations MUST use the platform's standard whole-file
  lock — a sidecar lockfile or any other private convention is invisible to
  other implementations and provides no protection.
- A **reader** SHOULD hold a shared lock (`flock(LOCK_SH)`) for a stable
  view. On Unix a reader MAY instead read without locking and accept the
  consequences: it can observe a torn tail mid-append, which §4's per-entry
  CRC recovery handles (truncate at first bad CRC). Lock-free reading is a
  Unix-only degraded mode; on Windows, kernel enforcement is expected to
  fail unlocked reads of a locked file.
- Kernel locks are released automatically when the holding process exits,
  so a crashed writer can never wedge a file (verified by probe).
- Network filesystems (NFS, SMB) do not reliably propagate these locks and
  are outside the interop guarantee: concurrent access to a `.htt` on a
  network mount is unsupported.

## Appendix A: CRC32

CRC32 uses the ISO 3309 / ITU-T V.42 polynomial (same as Ethernet, PNG, gzip):
`0xEDB88320` (reflected) / `0x04C11DB7` (normal).

This is the standard `crc32` available in every language's standard library.

## Appendix B: Byte Order

All multi-byte integers are little-endian. Q64.64 values are stored as i128 LE
(16 bytes, two's complement, integer part in high 64 bits, fractional in low 64).
