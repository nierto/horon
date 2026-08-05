//! TQ1.9 quantized semantic tails (`docs/QUANTIZED_SEMANTIC.md`).
//!
//! Opt-in on-disk encoding for semantic coordinates (format v4,
//! `FLAG_QUANTIZED_SEMANTIC`): user dims (16+) are stored as TQ1.9
//! balanced-ternary fixed-point — an `i16` holding `value × 3⁹` — 2 bytes
//! per dimension instead of 16. The reserved GACL region (dims 0–15) is
//! elided entirely unless the file has GACL enabled, in which case it stays
//! full-width Q64.64 so access decisions remain bit-exact.
//!
//! The codec is pure integer arithmetic (round-half-away-from-zero, matching
//! gMath's `TritQ1_9`), so quantized files are exactly as deterministic as
//! full-width ones — just ranking-grade (~4.3 significant digits) in their
//! user dims. `quantize_raw(dequantize_raw(q)) == q` for every valid `q`
//! (pinned exhaustively below), which is what lets in-memory canonical
//! values re-encode byte-identically across replay, compaction, and
//! replication.
//!
//! In-memory representation is ALWAYS full-width (16 bytes/dim Q64.64):
//! `NodeEntry` / `WalEntry` hold canonical full-width bytes, and only
//! `write_to` / `read_from` translate to/from the disk tail via [`SemLayout`].

use crate::error::{HoronError, HoronResult};
use crate::format::{DIM_USER_DEFINED_START, FLAG_GACL, FLAG_QUANTIZED_SEMANTIC};
use crate::header::GeoHeader;

/// TQ1.9 scale: 3⁹. One quantization step is 1/19683 ≈ 5.08e-5.
/// Kept equal to gMath's `SCALE_TQ1_9` (asserted in tests).
pub const TQ19_SCALE: i128 = 19_683;

/// Largest representable TQ1.9 raw magnitude: (3¹⁰ − 1)/2.
/// Value range is ±29524/19683 ≈ ±1.49987.
pub const TQ19_MAX_RAW: i16 = 29_524;

/// Quantize a raw Q64.64 value to a TQ1.9 raw (`value × 3⁹`), rounding
/// half away from zero. `None` when the value falls outside ±29524/19683.
pub fn quantize_raw(raw: i128) -> Option<i16> {
    // Reject magnitudes ≥ 2.0 before multiplying so `raw * TQ19_SCALE`
    // cannot overflow i128 (2.0 is already far outside the ±1.49987 range).
    if raw.unsigned_abs() > 2u128 << 64 {
        return None;
    }
    let scaled = raw * TQ19_SCALE;
    let half = 1i128 << 63;
    // Floor-shift of (|x| + half) is round-half-up on the magnitude, which
    // is round-half-away-from-zero once the sign is restored. Ties are
    // exact: scaled ≡ 2^63 (mod 2^64) rounds away, same as TritQ1_9.
    let q = if scaled >= 0 {
        (scaled + half) >> 64
    } else {
        -((-scaled + half) >> 64)
    };
    if q.unsigned_abs() > TQ19_MAX_RAW as u128 {
        None
    } else {
        Some(q as i16)
    }
}

/// Dequantize a TQ1.9 raw back to Q64.64, rounding half away from zero.
/// The result is within half a Q64.64 ULP of q/19683, so re-quantizing
/// always returns `q` (the round-trip error 19683/2⁶⁵ is far below the
/// half-step needed to move a quantization bucket).
pub fn dequantize_raw(q: i16) -> i128 {
    let x = (q as i128) * (1i128 << 64);
    let neg = x < 0;
    let mag = x.unsigned_abs();
    let div = TQ19_SCALE as u128;
    let (quot, rem) = (mag / div, mag % div);
    // Odd divisor: rem*2 == div is impossible, so there is no tie to break.
    let quot = if rem * 2 >= div { quot + 1 } else { quot };
    if neg {
        -(quot as i128)
    } else {
        quot as i128
    }
}

/// The semantic-tail layout of one file: how many dims, and how they map
/// to disk bytes. Derived from the file header; passed to every serialize/
/// deserialize site so the in-memory representation stays full-width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemLayout {
    /// Total semantic dimension count (header byte 7).
    pub dims: usize,
    /// `FLAG_QUANTIZED_SEMANTIC`: user dims stored as 2-byte TQ1.9 and the
    /// reserved region elided unless GACL is on.
    pub quantized: bool,
    /// `FLAG_GACL`: in a quantized file, the reserved dims 0–15 are present
    /// on disk at full width (access bands stay bit-exact).
    pub gacl: bool,
}

impl SemLayout {
    /// Layout described by a file header.
    pub fn from_header(h: &GeoHeader) -> Self {
        Self {
            dims: h.semantic_dims as usize,
            quantized: h.flags & FLAG_QUANTIZED_SEMANTIC != 0,
            gacl: h.flags & FLAG_GACL != 0,
        }
    }

    /// A plain (unquantized) layout — the pre-quantization disk format.
    pub fn plain(dims: usize) -> Self {
        Self { dims, quantized: false, gacl: false }
    }

    /// In-memory tail size: always full-width Q64.64.
    pub fn mem_bytes(&self) -> usize {
        self.dims * 16
    }

    /// Dims in the reserved GACL region (0–15) actually configured.
    pub fn reserved_dims(&self) -> usize {
        self.dims.min(DIM_USER_DEFINED_START)
    }

    /// User dims (16+).
    pub fn user_dims(&self) -> usize {
        self.dims.saturating_sub(DIM_USER_DEFINED_START)
    }

    /// On-disk tail size for one entry.
    pub fn disk_bytes(&self) -> usize {
        if !self.quantized {
            self.mem_bytes()
        } else {
            let reserved = if self.gacl { self.reserved_dims() * 16 } else { 0 };
            reserved + self.user_dims() * 2
        }
    }

    /// Raw Q64.64 value of dim `d` in a full-width tail (zero if short).
    fn dim_raw(full: &[u8], d: usize) -> i128 {
        let start = d * 16;
        let end = start + 16;
        if full.len() >= end {
            i128::from_le_bytes(full[start..end].try_into().unwrap())
        } else {
            0
        }
    }

    /// Encode a full-width in-memory tail into the quantized disk tail.
    /// Only called for quantized layouts; inputs shorter than `mem_bytes`
    /// are treated as zero-extended. Errors (rather than saturating or
    /// silently dropping) on out-of-range user dims and on nonzero reserved
    /// dims when the reserved region is elided.
    pub fn encode_tail(&self, full: &[u8]) -> HoronResult<Vec<u8>> {
        debug_assert!(self.quantized);
        let mut out = Vec::with_capacity(self.disk_bytes());
        for d in 0..self.reserved_dims() {
            let raw = Self::dim_raw(full, d);
            if self.gacl {
                out.extend_from_slice(&raw.to_le_bytes());
            } else if raw != 0 {
                return Err(HoronError::InvalidOperation(format!(
                    "quantized file without GACL stores no reserved dims — dim {} must be zero",
                    d
                )));
            }
        }
        for u in 0..self.user_dims() {
            let d = DIM_USER_DEFINED_START + u;
            let raw = Self::dim_raw(full, d);
            let q = quantize_raw(raw).ok_or_else(|| {
                HoronError::InvalidOperation(format!(
                    "semantic dim {} is outside the TQ1.9 range ±29524/19683 (≈±1.49987)",
                    d
                ))
            })?;
            out.extend_from_slice(&q.to_le_bytes());
        }
        Ok(out)
    }

    /// Decode a quantized disk tail into full-width in-memory bytes.
    /// Only called for quantized layouts; the input must be exactly
    /// `disk_bytes` long (it always is — tails are fixed-size).
    pub fn decode_tail(&self, disk: &[u8]) -> HoronResult<Vec<u8>> {
        debug_assert!(self.quantized);
        if disk.len() != self.disk_bytes() {
            return Err(HoronError::InvalidFormat(format!(
                "quantized semantic tail is {} bytes, expected {}",
                disk.len(),
                self.disk_bytes()
            )));
        }
        let mut out = vec![0u8; self.mem_bytes()];
        let mut pos = 0usize;
        if self.gacl {
            for d in 0..self.reserved_dims() {
                out[d * 16..d * 16 + 16].copy_from_slice(&disk[pos..pos + 16]);
                pos += 16;
            }
        }
        for u in 0..self.user_dims() {
            let d = DIM_USER_DEFINED_START + u;
            let q = i16::from_le_bytes(disk[pos..pos + 2].try_into().unwrap());
            pos += 2;
            out[d * 16..d * 16 + 16].copy_from_slice(&dequantize_raw(q).to_le_bytes());
        }
        Ok(out)
    }

    /// Canonicalize caller-supplied coordinates for a quantized file:
    /// zero-extend to full width, validate (reserved dims zero unless GACL;
    /// user dims in TQ1.9 range), and replace each user dim with its
    /// on-grid value `dequantize(quantize(v))` — write-through, so the
    /// in-memory store always equals the post-reload state.
    pub fn canonicalize(&self, coords: &mut Vec<u8>) -> HoronResult<()> {
        debug_assert!(self.quantized);
        if coords.len() > self.mem_bytes() {
            return Err(HoronError::InvalidOperation(format!(
                "coords cover {} bytes but the file has {} semantic dims ({} bytes)",
                coords.len(),
                self.dims,
                self.mem_bytes()
            )));
        }
        coords.resize(self.mem_bytes(), 0);
        if !self.gacl {
            for d in 0..self.reserved_dims() {
                if Self::dim_raw(coords, d) != 0 {
                    return Err(HoronError::InvalidOperation(format!(
                        "quantized file without GACL stores no reserved dims — dim {} must be zero",
                        d
                    )));
                }
            }
        }
        for u in 0..self.user_dims() {
            let d = DIM_USER_DEFINED_START + u;
            let raw = Self::dim_raw(coords, d);
            let q = quantize_raw(raw).ok_or_else(|| {
                HoronError::InvalidOperation(format!(
                    "semantic dim {} is outside the TQ1.9 range ±29524/19683 (≈±1.49987)",
                    d
                ))
            })?;
            coords[d * 16..d * 16 + 16].copy_from_slice(&dequantize_raw(q).to_le_bytes());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_matches_gmath() {
        use g_math::fixed_point::domains::balanced_ternary::trit_q1_9::SCALE_TQ1_9;
        assert_eq!(TQ19_SCALE, SCALE_TQ1_9 as i128);
    }

    #[test]
    fn roundtrip_exhaustive() {
        // Every valid TQ1.9 raw survives dequantize → quantize unchanged.
        for q in -TQ19_MAX_RAW..=TQ19_MAX_RAW {
            let raw = dequantize_raw(q);
            assert_eq!(quantize_raw(raw), Some(q), "round-trip failed at q={}", q);
        }
    }

    #[test]
    fn matches_gmath_from_rational() {
        use g_math::fixed_point::domains::balanced_ternary::TritQ1_9;
        // Cross-check the encoding against gMath's own quantizer on
        // rationals spanning the range, both signs, and the boundary.
        for &(n, d) in &[
            (1i64, 3u64),
            (-1, 3),
            (1, 7),
            (-5, 9),
            (1, 1),
            (-1, 1),
            (29524, 19683),
            (-29524, 19683),
            (0, 1),
            (4217, 10000),
        ] {
            let expected = TritQ1_9::from_rational(n, d).unwrap().raw();
            // Q64.64 raw of n/d, rounded half away from zero — within half
            // a ULP of the true rational, far below the TQ1.9 half-step.
            let mag = (n.unsigned_abs() as u128) << 64;
            let dv = d as u128;
            let (quot, rem) = (mag / dv, mag % dv);
            let quot = if rem * 2 >= dv { quot + 1 } else { quot };
            let raw = if n < 0 { -(quot as i128) } else { quot as i128 };
            assert_eq!(quantize_raw(raw), Some(expected), "mismatch at {}/{}", n, d);
        }
    }

    #[test]
    fn out_of_range_is_none() {
        // One step past the boundary value: (29524 + 0.5)/19683 in Q64.64.
        let boundary = dequantize_raw(TQ19_MAX_RAW);
        let half_step = (1i128 << 64) / (2 * TQ19_SCALE);
        assert_eq!(quantize_raw(boundary), Some(TQ19_MAX_RAW));
        assert_eq!(quantize_raw(boundary + 2 * half_step), None);
        assert_eq!(quantize_raw(-(boundary + 2 * half_step)), None);
        assert_eq!(quantize_raw(i128::MAX), None);
        assert_eq!(quantize_raw(i128::MIN), None);
        // 2.0 exactly (the early-bail edge) is out of range.
        assert_eq!(quantize_raw(2i128 << 64), None);
    }

    #[test]
    fn layout_sizes() {
        // 40 dims, quantized, no GACL: 24 user dims × 2 = 48 bytes.
        let l = SemLayout { dims: 40, quantized: true, gacl: false };
        assert_eq!(l.mem_bytes(), 640);
        assert_eq!(l.disk_bytes(), 48);
        // With GACL the 16 reserved dims stay full-width.
        let g = SemLayout { dims: 40, quantized: true, gacl: true };
        assert_eq!(g.disk_bytes(), 16 * 16 + 24 * 2);
        // Unquantized is unchanged.
        assert_eq!(SemLayout::plain(40).disk_bytes(), 640);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let l = SemLayout { dims: 20, quantized: true, gacl: false };
        let mut full = vec![0u8; l.mem_bytes()];
        // On-grid user-dim values round-trip exactly.
        for (i, q) in [(16usize, 1234i16), (17, -29524), (18, 0), (19, 29524)] {
            full[i * 16..i * 16 + 16].copy_from_slice(&dequantize_raw(q).to_le_bytes());
        }
        let disk = l.encode_tail(&full).unwrap();
        assert_eq!(disk.len(), l.disk_bytes());
        assert_eq!(l.decode_tail(&disk).unwrap(), full);
    }

    #[test]
    fn encode_rejects_reserved_and_range() {
        let l = SemLayout { dims: 20, quantized: true, gacl: false };
        // Nonzero reserved dim without GACL.
        let mut full = vec![0u8; l.mem_bytes()];
        full[0..16].copy_from_slice(&1i128.to_le_bytes());
        assert!(l.encode_tail(&full).is_err());
        // Out-of-range user dim (value 2.0).
        let mut full = vec![0u8; l.mem_bytes()];
        full[16 * 16..17 * 16].copy_from_slice(&(2i128 << 64).to_le_bytes());
        assert!(l.encode_tail(&full).is_err());
        // With GACL, reserved dims pass through at full width.
        let g = SemLayout { dims: 20, quantized: true, gacl: true };
        let mut full = vec![0u8; g.mem_bytes()];
        full[0..16].copy_from_slice(&12345i128.to_le_bytes());
        let disk = g.encode_tail(&full).unwrap();
        assert_eq!(g.decode_tail(&disk).unwrap(), full);
    }

    #[test]
    fn canonicalize_writes_through() {
        let l = SemLayout { dims: 20, quantized: true, gacl: false };
        // An arbitrary off-grid value snaps to its grid neighbor. (1/3 would
        // be a bad pick: thirds are exact on a 3⁹ grid — that's the point of
        // balanced ternary. 1/7 is not.)
        let mut coords = vec![0u8; l.mem_bytes()];
        let off_grid = (1i128 << 64) / 7;
        coords[16 * 16..17 * 16].copy_from_slice(&off_grid.to_le_bytes());
        l.canonicalize(&mut coords).unwrap();
        let snapped =
            i128::from_le_bytes(coords[16 * 16..17 * 16].try_into().unwrap());
        assert_ne!(snapped, off_grid);
        assert_eq!(snapped, dequantize_raw(quantize_raw(off_grid).unwrap()));
        // Canonical coords re-canonicalize to themselves (idempotent).
        let again = coords.clone();
        let mut coords2 = coords;
        l.canonicalize(&mut coords2).unwrap();
        assert_eq!(coords2, again);
        // Short input zero-extends; oversized input errors.
        let mut short = Vec::new();
        l.canonicalize(&mut short).unwrap();
        assert_eq!(short.len(), l.mem_bytes());
        let mut long = vec![0u8; l.mem_bytes() + 16];
        assert!(l.canonicalize(&mut long).is_err());
    }
}
