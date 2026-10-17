//! Cross-pillar proof: temporal epochs × the semantic disk.
//!
//! A node's affinity coordinates drift across sealed epochs in a `.htt`
//! file; `HoronHistory::trajectory` reads the recorded movement back, and
//! `SemanticDisk::classify_trajectory` turns it into a symbolic concept
//! sequence — "the movement of the data manifold" rendered as discrete
//! meaning-states, end to end through both shipped pillars.

use horon::{DurabilityMode, Horon, HoronConfig, HistoryRetention, HoronHistory};
use horon_engine::SemanticDisk;
use g_math::fixed_point::Q64_64 as FixedPoint;
use std::path::PathBuf;
use tempfile::NamedTempFile;

fn temp_path() -> PathBuf {
    NamedTempFile::new().unwrap().into_temp_path().to_path_buf()
}

/// 21 semantic dims: 16 reserved + 5 category affinities (dims 16..21).
fn coords(vals: &[f64]) -> Vec<u8> {
    let mut out = vec![0u8; 16 * 16];
    for v in vals {
        out.extend_from_slice(&FixedPoint::from_f64(*v).raw().to_le_bytes());
    }
    out.resize(21 * 16, 0);
    out
}

#[test]
fn recorded_drift_classifies_into_a_symbolic_concept_sequence() {
    let path = temp_path();
    {
        let gf = Horon::open_with_config(
            &path,
            HoronConfig {
                semantic_dims: 21,
                auto_compact_threshold: 0,
                durability: DurabilityMode::Relaxed,
                history_retention: HistoryRetention::Archive,
                ..Default::default()
            },
        )
        .unwrap();

        gf.put("/course/emdr", b"x").unwrap();

        // Epoch 1: solidly trauma.
        gf.set_semantic("/course/emdr", coords(&[0.9, 0.05, 0.05, 0.0, 0.0])).unwrap();
        gf.seal_epoch().unwrap();

        // Epoch 2: still trauma (population shifts slightly). A compaction
        // in between proves the archived history feeds the readout too.
        gf.set_semantic("/course/emdr", coords(&[0.85, 0.05, 0.1, 0.0, 0.0])).unwrap();
        gf.seal_epoch().unwrap();
        assert!(gf.compact().unwrap());

        // Epoch 3: the field moved under it — now systemisch.
        gf.set_semantic("/course/emdr", coords(&[0.05, 0.05, 0.9, 0.0, 0.0])).unwrap();
        gf.seal_epoch().unwrap();
        gf.flush().unwrap();
    }

    // Readout: the epoch log gives the recorded movement; the disk names its meaning.
    let hist = HoronHistory::open(&path).unwrap();
    assert!(hist.is_complete());
    let samples = hist.trajectory("/course/emdr", &(16..21));
    assert_eq!(samples.len(), 3, "one sample per sealed epoch");

    let disk = SemanticDisk::build(&[
        ("/trauma", 16),
        ("/cgt", 17),
        ("/systemisch", 18),
        ("/kind_jeugd", 19),
        ("/ouderen", 20),
    ])
    .unwrap();

    let symbolic = disk.classify_trajectory(16, &samples);
    assert_eq!(
        symbolic,
        vec![
            (1, "/trauma".to_string()),
            (2, "/trauma".to_string()),
            (3, "/systemisch".to_string()),
        ],
        "recorded coordinate drift must replay as the expected meaning-states"
    );
}
