# g_math 0.5.0 upgrade guide (Horon)

You are on `g_math = "0.4.31"`, shared with horon-private, horon-engine,
horon-engine-private and ganalyze-wp-bridge. Cargo will not move you to 0.5.0
on its own: `"0.4.31"` means `>=0.4.31, <0.5.0`. You upgrade when you choose to.

## Verdict

Upgrading is safe for your stored data and is NOT a disk-format break.
It IS a behavioural change you should announce to your own consumers.

## Your persisted data is unaffected

Horon writes `FixedPoint::from_f64(v).raw().to_le_bytes()` into cells. We
compared v0.4.34 and v0.5.0 byte for byte: `from_f64`, `to_f64`, `from_int` and
`from_str` are unchanged. A value converted and stored today produces the exact
same bits under 0.5.0, so existing cells, WAL entries and file headers stay
valid. No migration, no regeneration.

Your `TritQ1_9` / `SCALE_TQ1_9` use in `quant.rs` is also unaffected. 0.5.0
resized the balanced-ternary UGOD tiers, but TQ1.9 is a separate standalone
type and was deliberately left alone.

## What does change

Computed results can move by one unit in the last place, because binary
multiply and divide now round to nearest everywhere instead of truncating.
On your embedded (Q64.64) profile:

- multiply: changes only on exact halfway cases, which are vanishingly rare
- divide: changes on roughly half of inexact quotients

So anything you compute at query time and then compare, hash or assert on can
differ in its final digit. Distances, scores and normalised vectors are the
likely places.

## Announce it downstream

Yes, tell your consumers. Not as a data break, but as a precision change:
"recomputed values may differ in the last digit from values computed by an
earlier release; do not mix results across the boundary in one dataset."
If any Horon API returns a computed float or fixed-point value that a caller
hashes or diffs, that caller needs the same warning.

## Worth having

0.5.0 fixes silent wrong answers that could have bitten you:

- arithmetic near the top of a tier could wrap into a plausible but wrong
  value instead of promoting (for example, on embedded, 9e18 + 9e18 returned
  0.0)
- symbolic values coerced to binary could wrap mod 2^64 into a wrong value

## Checklist

1. Bump the pin in all five repos together; they are meant to match.
2. Re-run your test suite. Expect golden-value assertions on computed
   quantities to need a one-ulp tolerance or a refreshed expectation.
3. Do not regenerate stored cells. They are already correct.

Full notice: https://github.com/nierto/gMath/releases/tag/v0.5.0
