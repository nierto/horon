# Test fixtures

Every file here is **generated, not captured**. No fixture contains data
belonging to a real person, project or dataset, and none should ever be
added by hand — if a test needs a new shape, add a profile to
`examples/gen_fixture.rs` and generate it.

## catalog.htt

A 1500-item taxonomy whose semantic coordinates cluster by category, stored
as format v1 so the upgrade-on-compaction path stays covered.

```sh
cargo run --example gen_fixture -- --profile catalog --nodes 1500 --dims 40 --seed 1 --v1
```

The generator is deterministic: the same seed produces byte-identical output,
so this file can be rebuilt exactly by anyone. Used by `hilbert_locality.rs`
(locality and chunk-offset tests) and `durability.rs` (v1 → v2 upgrade).

Regenerate it whenever a format change makes it stale, and say so in the
commit message — the tests assert on node counts and on a locality ratio, so
a regenerated fixture is a deliberate change, not a mechanical one.

## Other shapes

`gen_fixture` also builds `filesystem` (directory-tree shape, the dimensions
`htt import-fs` derives) and `vectors` (flat embedding corpus, the
`htt import-vec` shape). Neither is committed — generate them into a temporary
directory when a test or benchmark needs one.

For small files that pin down *format* behaviour rather than data behaviour,
see `tests/corpus/`, which is the conformance oracle for independent `.htt`
implementations.
