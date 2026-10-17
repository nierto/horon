//! Hilbert curve mapping for content-addressable chunk storage.
//!
//! Maps N-dimensional semantic coordinates to a 1D Hilbert index that
//! preserves spatial locality — nearby points in semantic space map to
//! nearby indices, so semantically similar data lands in nearby chunks.
//!
//! Uses the Skilling algorithm ("Programming the Hilbert curve", 2004)
//! which works in arbitrary dimensions via bit-level Gray code transforms
//! and axis rotations.

use g_math::fixed_point::Q64_64 as FixedPoint;

/// A Hilbert space-filling curve mapper.
///
/// Configurable dimensions and bits-per-axis. The total index space is
/// `2^(dims * bits)` cells. For practical use:
/// - 3 dims × 16 bits = 48-bit index (256T cells)
/// - 4 dims × 12 bits = 48-bit index
/// - 8 dims × 8 bits  = 64-bit index
#[derive(Clone, Debug)]
pub struct HilbertMapper {
    dims: usize,
    bits: u32,
    max_coord: u32, // 2^bits - 1
}

/// A Hilbert index with its raw value and the coordinates it was derived from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct HilbertIndex(pub u128);

impl HilbertIndex {
    /// Raw index value — the cell's position along the Hilbert curve.
    pub fn value(&self) -> u128 { self.0 }

    /// Interpret as a chunk offset: index × chunk_size gives byte offset.
    pub fn chunk_offset(&self, chunk_size: usize) -> u64 {
        (self.0 as u64).wrapping_mul(chunk_size as u64)
    }
}

impl HilbertMapper {
    /// Create a new mapper.
    ///
    /// - `dims`: number of dimensions to map (typically 3-8)
    /// - `bits`: bits per axis (resolution). Higher = finer grid, larger index space.
    ///
    /// Total index bits = dims × bits. Must be ≤ 128.
    pub fn new(dims: usize, bits: u32) -> Self {
        assert!(dims > 0 && dims <= 16, "dims must be 1-16");
        assert!(bits > 0 && bits <= 16, "bits must be 1-16");
        assert!((dims as u32) * bits <= 128, "total index bits must be ≤ 128");
        Self {
            dims,
            bits,
            max_coord: (1u32 << bits) - 1,
        }
    }

    /// Map normalized [0, 1] fixed-point coordinates to a Hilbert index.
    ///
    /// The exact path: coordinates arrive as Q64.64 and the axis scaling is a
    /// fixed-point multiply, so the address a node receives — and therefore
    /// its byte position in a meaning-addressed file — never passes through
    /// floating point. `to_int` floors, and normalized inputs are
    /// non-negative, so adding one half reproduces round-half-away-from-zero
    /// exactly.
    pub fn coords_to_index_fixed(&self, coords: &[FixedPoint]) -> HilbertIndex {
        let zero = FixedPoint::from_int(0);
        let one = FixedPoint::from_int(1);
        let half = one / FixedPoint::from_int(2);
        let scale = FixedPoint::from_int(self.max_coord as i32);

        let mut axes: Vec<u32> = (0..self.dims)
            .map(|i| {
                let v = coords.get(i).copied().unwrap_or(zero);
                let v = if v < zero { zero } else if v > one { one } else { v };
                (v * scale + half).to_int() as u32
            })
            .collect();

        self.axes_to_hilbert(&mut axes)
    }

    /// Map normalized [0.0, 1.0] coordinates to a Hilbert index.
    ///
    /// Values outside [0, 1] are clamped. Only the first `dims` values are used.
    ///
    /// Prefer [`coords_to_index_fixed`](Self::coords_to_index_fixed) on any
    /// path that decides file layout; this one exists for callers that only
    /// have floats.
    pub fn coords_to_index(&self, coords: &[f64]) -> HilbertIndex {
        let mut axes: Vec<u32> = (0..self.dims)
            .map(|i| {
                let v = coords.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                (v * self.max_coord as f64).round() as u32
            })
            .collect();

        self.axes_to_hilbert(&mut axes)
    }

    /// Map a Hilbert index back to normalized [0.0, 1.0] coordinates.
    pub fn index_to_coords(&self, index: &HilbertIndex) -> Vec<f64> {
        let axes = self.hilbert_to_axes(index);
        axes.iter()
            .map(|&a| a as f64 / self.max_coord as f64)
            .collect()
    }

    /// Resolution: total number of cells along one axis.
    pub fn resolution(&self) -> u32 { self.max_coord + 1 }

    /// Total bits in the Hilbert index.
    pub fn total_bits(&self) -> u32 { self.dims as u32 * self.bits }

    // -----------------------------------------------------------------------
    // Skilling algorithm: axes ↔ Hilbert index
    // -----------------------------------------------------------------------

    fn axes_to_hilbert(&self, axes: &mut Vec<u32>) -> HilbertIndex {
        let n = self.dims;
        let b = self.bits;

        // --- Skilling transform: axes → transpose ---
        let m = 1u32 << (b - 1);

        // Inverse undo
        let mut q = m;
        while q > 1 {
            let p = q - 1;
            for i in 0..n {
                if axes[i] & q != 0 {
                    axes[0] ^= p;
                } else {
                    let t = (axes[0] ^ axes[i]) & p;
                    axes[0] ^= t;
                    axes[i] ^= t;
                }
            }
            q >>= 1;
        }

        // Gray encode
        for i in 1..n {
            axes[i] ^= axes[i - 1];
        }
        let mut t = 0u32;
        let mut q = m;
        while q > 1 {
            if axes[n - 1] & q != 0 {
                t ^= q - 1;
            }
            q >>= 1;
        }
        for i in 0..n {
            axes[i] ^= t;
        }

        // --- Interleave bits into Hilbert index ---
        self.interleave(axes)
    }

    fn hilbert_to_axes(&self, index: &HilbertIndex) -> Vec<u32> {
        let n = self.dims;
        let b = self.bits;

        let mut axes = self.deinterleave(index);

        // --- Inverse Skilling transform: transpose → axes ---

        // Gray decode (undo the Gray encoding from axes_to_hilbert)
        let t = axes[n - 1] >> 1;
        for i in (1..n).rev() {
            axes[i] ^= axes[i - 1];
        }
        axes[0] ^= t;

        // Undo the inverse undo (reconstruct from transpose)
        let mut q = 2u32;
        while q != (1 << b) {
            let p = q - 1;
            let mut i = n - 1;
            loop {
                if axes[i] & q != 0 {
                    axes[0] ^= p;
                } else {
                    let tt = (axes[0] ^ axes[i]) & p;
                    axes[0] ^= tt;
                    axes[i] ^= tt;
                }
                if i == 0 { break; }
                i -= 1;
            }
            q <<= 1;
        }

        axes
    }

    // Interleave n axes of b bits each into a single Hilbert index.
    // Bit ordering: MSB-first, cycling through axes for each bit level.
    fn interleave(&self, axes: &[u32]) -> HilbertIndex {
        let mut index: u128 = 0;
        for bit in (0..self.bits).rev() {
            for dim in 0..self.dims {
                index <<= 1;
                if axes[dim] & (1 << bit) != 0 {
                    index |= 1;
                }
            }
        }
        HilbertIndex(index)
    }

    fn deinterleave(&self, index: &HilbertIndex) -> Vec<u32> {
        let mut axes = vec![0u32; self.dims];
        let mut val = index.0;
        for bit in 0..self.bits {
            for dim in (0..self.dims).rev() {
                if val & 1 != 0 {
                    axes[dim] |= 1 << bit;
                }
                val >>= 1;
            }
        }
        axes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_2d() {
        let h = HilbertMapper::new(2, 8);
        let coords = vec![0.3, 0.7];
        let idx = h.coords_to_index(&coords);
        let back = h.index_to_coords(&idx);
        assert!((back[0] - coords[0]).abs() < 0.01);
        assert!((back[1] - coords[1]).abs() < 0.01);
    }

    #[test]
    fn roundtrip_3d() {
        let h = HilbertMapper::new(3, 10);
        let coords = vec![0.5, 0.25, 0.75];
        let idx = h.coords_to_index(&coords);
        let back = h.index_to_coords(&idx);
        for i in 0..3 {
            assert!((back[i] - coords[i]).abs() < 0.002,
                "dim {} mismatch: {} vs {}", i, back[i], coords[i]);
        }
    }

    #[test]
    fn locality_2d() {
        // Nearby points should get nearby indices
        let h = HilbertMapper::new(2, 12);
        let a = h.coords_to_index(&[0.50, 0.50]);
        let b = h.coords_to_index(&[0.51, 0.50]);
        let c = h.coords_to_index(&[0.90, 0.10]); // far away

        let dist_ab = (a.0 as i128 - b.0 as i128).unsigned_abs();
        let dist_ac = (a.0 as i128 - c.0 as i128).unsigned_abs();

        assert!(dist_ab < dist_ac, "nearby points should have closer indices: ab={} ac={}", dist_ab, dist_ac);
    }

    #[test]
    fn origin_is_zero() {
        let h = HilbertMapper::new(3, 8);
        let origin = h.coords_to_index(&[0.0, 0.0, 0.0]);
        assert_eq!(origin.0, 0);
    }

    #[test]
    fn index_space_bounded() {
        // All indices must fit within dims*bits
        let h = HilbertMapper::new(3, 8);
        let max_index = (1u128 << (3 * 8)) - 1;
        for coords in [[1.0,1.0,1.0],[0.0,1.0,0.0],[1.0,0.0,1.0],[0.5,0.5,0.5]] {
            let idx = h.coords_to_index(&coords);
            assert!(idx.0 <= max_index, "index {} exceeds max {}", idx.0, max_index);
        }
    }

    #[test]
    fn chunk_offset() {
        let h = HilbertMapper::new(3, 8);
        let idx = h.coords_to_index(&[0.5, 0.5, 0.5]);
        let offset = idx.chunk_offset(4096); // 4KB chunks
        assert!(offset > 0);
    }

    #[test]
    fn deterministic() {
        // Same coords always produce same index
        let h = HilbertMapper::new(4, 10);
        let coords = vec![0.33, 0.66, 0.11, 0.88];
        let a = h.coords_to_index(&coords);
        let b = h.coords_to_index(&coords);
        assert_eq!(a, b);
    }
}
