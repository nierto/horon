//! Geometric Access Control (GACL) — cooperative, coordinate-encoded scoping.
//!
//! Access rules are encoded in a node's semantic coordinates: each of the six
//! access concepts (dims 0–11) is a `[lo, hi]` band, and a requester's
//! credential vector carries their position on each axis. Access is granted
//! when every credential falls within the corresponding band.
//!
//! **What this is, and is not.** GACL is *cooperative query-scoping*, checked
//! on access: [`Credentials::can_access`]/[`can_read`](Credentials::can_read)
//! compare the credential against a node's bands, and the Horon query layer
//! filters unreadable nodes out of results. It is **not** a security boundary
//! and does **not** make unauthorized nodes geometrically unreachable — the
//! nodes are stored normally and found by the index, then filtered. Anyone
//! with raw file access, or who calls without setting credentials (unless the
//! file is opened fail-closed), sees everything.
//!
//! A stronger "geometric invisibility" model — folding access into the
//! distance function so the spatial index never even visits unauthorized
//! nodes — is a research direction, not a shipped or planned guarantee; do not
//! rely on it. Use OS/filesystem permissions or encryption for real
//! confidentiality.

use g_math::fixed_point::Q64_64 as FixedPoint;

use crate::format::*;

/// An access band [lo, hi] for a single access concept.
#[derive(Debug, Clone, Copy)]
pub struct AccessBand {
    /// Lower bound of the band (inclusive).
    pub lo: FixedPoint,
    /// Upper bound of the band (inclusive).
    pub hi: FixedPoint,
}

impl AccessBand {
    /// Fully open band — grants access to any credential value.
    pub fn open() -> Self {
        Self {
            lo: FixedPoint::from_int(0),
            hi: FixedPoint::from_int(1),
        }
    }

    /// Fully closed band — grants access to nobody.
    ///
    /// Represented as an empty interval (`lo > hi`) so that `permits` is false
    /// for *every* credential, including `0.0`. A `[0, 0]` band would still
    /// admit a requester whose credential is exactly `0.0` (the default/unset
    /// value), silently granting access to the least-privileged identity.
    pub fn closed() -> Self {
        Self {
            lo: FixedPoint::from_int(1),
            hi: FixedPoint::from_int(0),
        }
    }

    /// Band from lo to hi.
    ///
    /// Bounds are exact: bands are compared against credentials in Q64.64,
    /// so accepting them at f64 precision would make a band unable to
    /// express a boundary the comparison can actually distinguish.
    pub fn new(lo: FixedPoint, hi: FixedPoint) -> Self {
        Self { lo, hi }
    }

    /// Convenience constructor from decimal literals. Converts at the
    /// boundary and is lossy for values f64 cannot represent exactly —
    /// prefer [`new`](Self::new) where the bound matters.
    pub fn from_f64(lo: f64, hi: f64) -> Self {
        Self::new(FixedPoint::from_f64(lo), FixedPoint::from_f64(hi))
    }

    /// Check if a credential value falls within this band.
    pub fn permits(&self, credential: FixedPoint) -> bool {
        credential >= self.lo && credential <= self.hi
    }

    /// Narrow this band by a parent's band (monotonic restriction).
    /// Returns the intersection — child can only be more restrictive.
    pub fn narrow(&self, parent: &AccessBand) -> Self {
        Self {
            lo: if self.lo > parent.lo { self.lo } else { parent.lo },
            hi: if self.hi < parent.hi { self.hi } else { parent.hi },
        }
    }
}

/// Credential vector for a requester (user or service).
#[derive(Debug, Clone)]
pub struct Credentials {
    /// Requester's position in the read access dimension.
    pub read: FixedPoint,
    /// Requester's position in the write access dimension.
    pub write: FixedPoint,
    /// Requester's position in the execute/invoke dimension.
    pub exec: FixedPoint,
    /// Requester's position in the organizational domain dimension.
    pub domain: FixedPoint,
    /// Requester's position in the data classification dimension.
    pub classification: FixedPoint,
    /// Requester's position in the user/service identity dimension.
    pub identity: FixedPoint,
}

impl Credentials {
    /// Root credentials — full access to everything.
    pub fn root() -> Self {
        let one = FixedPoint::from_int(1);
        Self {
            read: one,
            write: one,
            exec: one,
            domain: one,
            classification: one,
            identity: one,
        }
    }

    /// Build credentials from group positions (takes max per dimension).
    pub fn from_groups(groups: &[Credentials]) -> Self {
        let mut result = Self {
            read: FixedPoint::from_int(0),
            write: FixedPoint::from_int(0),
            exec: FixedPoint::from_int(0),
            domain: FixedPoint::from_int(0),
            classification: FixedPoint::from_int(0),
            identity: FixedPoint::from_int(0),
        };
        for g in groups {
            if g.read > result.read { result.read = g.read; }
            if g.write > result.write { result.write = g.write; }
            if g.exec > result.exec { result.exec = g.exec; }
            if g.domain > result.domain { result.domain = g.domain; }
            if g.classification > result.classification { result.classification = g.classification; }
            if g.identity > result.identity { result.identity = g.identity; }
        }
        result
    }

    /// Check if these credentials grant access to a node with the given bands.
    pub fn can_access(&self, bands: &NodeAccessBands) -> bool {
        bands.read.permits(self.read)
            && bands.write.permits(self.write)
            && bands.exec.permits(self.exec)
            && bands.domain.permits(self.domain)
            && bands.classification.permits(self.classification)
            && bands.identity.permits(self.identity)
    }

    /// Check read-only access.
    pub fn can_read(&self, bands: &NodeAccessBands) -> bool {
        bands.read.permits(self.read)
            && bands.domain.permits(self.domain)
            && bands.classification.permits(self.classification)
            && bands.identity.permits(self.identity)
    }
}

/// Access bands for a node (extracted from semantic dimensions 0-11).
#[derive(Debug, Clone)]
pub struct NodeAccessBands {
    /// Band governing read access (semantic dims 0-1).
    pub read: AccessBand,
    /// Band governing write access (semantic dims 2-3).
    pub write: AccessBand,
    /// Band governing execute/invoke access (semantic dims 4-5).
    pub exec: AccessBand,
    /// Organizational domain band (semantic dims 6-7).
    pub domain: AccessBand,
    /// Data classification band (semantic dims 8-9).
    pub classification: AccessBand,
    /// User/service identity band (semantic dims 10-11).
    pub identity: AccessBand,
}

impl NodeAccessBands {
    /// Fully open — public access.
    pub fn public() -> Self {
        Self {
            read: AccessBand::open(),
            write: AccessBand::open(),
            exec: AccessBand::open(),
            domain: AccessBand::open(),
            classification: AccessBand::open(),
            identity: AccessBand::open(),
        }
    }

    /// Extract access bands from raw semantic coordinate bytes.
    /// Expects at least 12 semantic dimensions (12 × 16 = 192 bytes).
    pub fn from_semantic_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 * 16 {
            return None;
        }
        Some(Self {
            read: AccessBand {
                lo: fp_from_bytes(&bytes[DIM_READ_LO * 16..]),
                hi: fp_from_bytes(&bytes[DIM_READ_HI * 16..]),
            },
            write: AccessBand {
                lo: fp_from_bytes(&bytes[DIM_WRITE_LO * 16..]),
                hi: fp_from_bytes(&bytes[DIM_WRITE_HI * 16..]),
            },
            exec: AccessBand {
                lo: fp_from_bytes(&bytes[DIM_EXEC_LO * 16..]),
                hi: fp_from_bytes(&bytes[DIM_EXEC_HI * 16..]),
            },
            domain: AccessBand {
                lo: fp_from_bytes(&bytes[DIM_DOMAIN_LO * 16..]),
                hi: fp_from_bytes(&bytes[DIM_DOMAIN_HI * 16..]),
            },
            classification: AccessBand {
                lo: fp_from_bytes(&bytes[DIM_CLASS_LO * 16..]),
                hi: fp_from_bytes(&bytes[DIM_CLASS_HI * 16..]),
            },
            identity: AccessBand {
                lo: fp_from_bytes(&bytes[DIM_IDENTITY_LO * 16..]),
                hi: fp_from_bytes(&bytes[DIM_IDENTITY_HI * 16..]),
            },
        })
    }

    /// Serialize access bands into semantic coordinate bytes (first 192 bytes).
    pub fn to_semantic_bytes(&self, total_semantic_dims: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; total_semantic_dims * 16];
        if total_semantic_dims < 12 {
            return bytes;
        }

        fp_to_bytes(&mut bytes[DIM_READ_LO * 16..], self.read.lo);
        fp_to_bytes(&mut bytes[DIM_READ_HI * 16..], self.read.hi);
        fp_to_bytes(&mut bytes[DIM_WRITE_LO * 16..], self.write.lo);
        fp_to_bytes(&mut bytes[DIM_WRITE_HI * 16..], self.write.hi);
        fp_to_bytes(&mut bytes[DIM_EXEC_LO * 16..], self.exec.lo);
        fp_to_bytes(&mut bytes[DIM_EXEC_HI * 16..], self.exec.hi);
        fp_to_bytes(&mut bytes[DIM_DOMAIN_LO * 16..], self.domain.lo);
        fp_to_bytes(&mut bytes[DIM_DOMAIN_HI * 16..], self.domain.hi);
        fp_to_bytes(&mut bytes[DIM_CLASS_LO * 16..], self.classification.lo);
        fp_to_bytes(&mut bytes[DIM_CLASS_HI * 16..], self.classification.hi);
        fp_to_bytes(&mut bytes[DIM_IDENTITY_LO * 16..], self.identity.lo);
        fp_to_bytes(&mut bytes[DIM_IDENTITY_HI * 16..], self.identity.hi);

        bytes
    }

    /// Narrow all bands by a parent's bands (monotonic restriction).
    ///
    /// This is an **authoring-time** helper: it computes the intersection so a
    /// caller can materialize a child's effective bands before writing them to
    /// the node's semantic coordinates. Access checks read each node's stored
    /// bands *as-is* and do **not** walk ancestors, so inheritance is not
    /// enforced automatically — a child written with wider bands than its
    /// parent will be checked against those wider bands. Apply this at write
    /// time (or precompute effective bands) if you want hierarchical
    /// restriction. (Consistent with GACL being cooperative query-scoping, not
    /// an enforced security boundary — see the module docs.)
    pub fn narrow(&self, parent: &NodeAccessBands) -> Self {
        Self {
            read: self.read.narrow(&parent.read),
            write: self.write.narrow(&parent.write),
            exec: self.exec.narrow(&parent.exec),
            domain: self.domain.narrow(&parent.domain),
            classification: self.classification.narrow(&parent.classification),
            identity: self.identity.narrow(&parent.identity),
        }
    }
}

/// Read a FixedPoint from 16 bytes (i128 LE).
fn fp_from_bytes(bytes: &[u8]) -> FixedPoint {
    let raw = i128::from_le_bytes(bytes[..16].try_into().unwrap());
    FixedPoint::from_raw(raw)
}

/// Write a FixedPoint as 16 bytes (i128 LE).
fn fp_to_bytes(dest: &mut [u8], fp: FixedPoint) {
    let raw = fp.raw();
    dest[..16].copy_from_slice(&raw.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_band_permits() {
        let band = AccessBand::from_f64(0.5, 1.0);
        assert!(band.permits(FixedPoint::from_f64(0.7)));
        assert!(band.permits(FixedPoint::from_f64(0.5)));
        assert!(band.permits(FixedPoint::from_f64(1.0)));
        assert!(!band.permits(FixedPoint::from_f64(0.3)));
    }

    #[test]
    fn test_open_band_permits_all() {
        let band = AccessBand::open();
        assert!(band.permits(FixedPoint::from_f64(0.0)));
        assert!(band.permits(FixedPoint::from_f64(0.5)));
        assert!(band.permits(FixedPoint::from_f64(1.0)));
    }

    #[test]
    fn test_closed_band_permits_none() {
        let band = AccessBand::closed();
        assert!(!band.permits(FixedPoint::from_f64(0.5)));
        assert!(!band.permits(FixedPoint::from_f64(1.0)));
        // A closed band must reject even the default/unset credential 0.0 —
        // otherwise the least-privileged identity gets in for free.
        assert!(!band.permits(FixedPoint::from_f64(0.0)));
    }

    #[test]
    fn test_closed_band_survives_narrow_and_roundtrip() {
        // Narrowing anything by a closed band stays closed.
        let closed = AccessBand::closed();
        let narrowed = AccessBand::open().narrow(&closed);
        assert!(!narrowed.permits(FixedPoint::from_f64(0.0)));
        assert!(!narrowed.permits(FixedPoint::from_f64(0.5)));

        // And a node whose stored bands are closed denies all credentials.
        let mut node = NodeAccessBands::public();
        node.read = AccessBand::closed();
        let bytes = node.to_semantic_bytes(12);
        let restored = NodeAccessBands::from_semantic_bytes(&bytes).unwrap();
        assert!(!restored.read.permits(FixedPoint::from_f64(0.0)));
        assert!(!restored.read.permits(FixedPoint::from_f64(1.0)));
    }

    #[test]
    fn test_narrow_restricts() {
        let parent = AccessBand::from_f64(0.3, 0.9);
        let child = AccessBand::from_f64(0.1, 1.0); // wider than parent
        let narrowed = child.narrow(&parent);

        // Should be intersection: [0.3, 0.9]
        assert!(!narrowed.permits(FixedPoint::from_f64(0.2)));
        assert!(narrowed.permits(FixedPoint::from_f64(0.5)));
        assert!(!narrowed.permits(FixedPoint::from_f64(0.95)));
    }

    #[test]
    fn test_credentials_root_accesses_everything() {
        let creds = Credentials::root();
        let bands = NodeAccessBands::public();
        assert!(creds.can_access(&bands));
    }

    #[test]
    fn test_credentials_from_groups() {
        let eng = Credentials {
            read: FixedPoint::from_f64(0.5),
            write: FixedPoint::from_f64(0.3),
            exec: FixedPoint::from_f64(0.5),
            domain: FixedPoint::from_f64(0.5),
            classification: FixedPoint::from_f64(0.3),
            identity: FixedPoint::from_f64(0.42),
        };
        let security = Credentials {
            read: FixedPoint::from_f64(0.7),
            write: FixedPoint::from_f64(0.7),
            exec: FixedPoint::from_f64(0.7),
            domain: FixedPoint::from_f64(0.3),
            classification: FixedPoint::from_f64(0.8),
            identity: FixedPoint::from_f64(0.42),
        };

        let combined = Credentials::from_groups(&[eng, security]);
        // Should take max per dimension
        assert_eq!(combined.read.to_f64(), FixedPoint::from_f64(0.7).to_f64());
        assert_eq!(combined.domain.to_f64(), FixedPoint::from_f64(0.5).to_f64());
        assert_eq!(combined.classification.to_f64(), FixedPoint::from_f64(0.8).to_f64());
    }

    #[test]
    fn test_node_access_bands_inheritance() {
        let parent = NodeAccessBands {
            read: AccessBand::from_f64(0.5, 1.0),
            write: AccessBand::from_f64(0.8, 1.0),
            exec: AccessBand::open(),
            domain: AccessBand::from_f64(0.3, 0.7),
            classification: AccessBand::from_f64(0.6, 1.0),
            identity: AccessBand::open(),
        };

        // Child tries to be more permissive — should be narrowed
        let child = NodeAccessBands::public();
        let effective = child.narrow(&parent);

        // read should be narrowed to [0.5, 1.0]
        assert!(!effective.read.permits(FixedPoint::from_f64(0.3)));
        assert!(effective.read.permits(FixedPoint::from_f64(0.7)));
    }
}
