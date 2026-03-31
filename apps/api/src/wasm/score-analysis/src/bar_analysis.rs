//! Bar-aligned musical analysis engine for all 6 performance dimensions.
//!
//! Transforms model scores + AMT MIDI + score data + reference profiles into
//! structured musical facts per dimension. Two tiers of output:
//!   - Tier 1: Full bar-aligned analysis with score + reference comparison
//!   - Tier 2: Absolute MIDI analysis (no score context)

use crate::types::{
    BarMap, ChunkAnalysis, DimensionAnalysis, NoteAlignment, PerfNote, PerfPedalEvent,
    ReferenceProfile, ScoreBar, ScoreContext,
};

// Dimension index constants (matching model output order)
const DIM_DYNAMICS: usize = 0;
const DIM_TIMING: usize = 1;
const DIM_PEDALING: usize = 2;
const DIM_ARTICULATION: usize = 3;
const DIM_PHRASING: usize = 4;
const DIM_INTERPRETATION: usize = 5;

const DIMENSION_NAMES: [&str; 6] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
];

// --- Utility functions ---

fn mean_u8(vals: &[u8]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().map(|&v| f64::from(v)).sum::<f64>() / vals.len() as f64
}

fn mean_f64(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn std_f64(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let m = mean_f64(vals);
    let variance = vals.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64;
    variance.sqrt()
}

// --- Tier 1 per-dimension analyzers ---

fn analyze_dynamics_tier1(
    alignments: &[NoteAlignment],
    score_bars: &[ScoreBar],
    reference: Option<&ReferenceProfile>,
    tempo_markings: &[serde_json::Value],
    bar_range: (u32, u32),
    model_score: f64,
) -> DimensionAnalysis {
    let perf_velocities: Vec<u8> = alignments.iter().map(|a| a.perf_velocity).collect();
    let score_velocities: Vec<u8> = score_bars
        .iter()
        .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
        .flat_map(|b| b.notes.iter().map(|n| n.velocity))
        .collect();

    let perf_mean = mean_u8(&perf_velocities);
    let score_mean = mean_u8(&score_velocities);

    let half = perf_velocities.len() / 2;
    let first_half_mean = if half > 0 {
        mean_u8(&perf_velocities[..half])
    } else {
        perf_mean
    };
    let second_half_mean = if half > 0 {
        mean_u8(&perf_velocities[half..])
    } else {
        perf_mean
    };
    let shape_delta = second_half_mean - first_half_mean;

    let shape_desc = if shape_delta > 8.0 {
        " with a noticeable crescendo through the passage"
    } else if shape_delta < -8.0 {
        " with a noticeable diminuendo through the passage"
    } else {
        " with relatively even dynamic level"
    };

    let analysis = if score_velocities.is_empty() {
        format!("Mean velocity {perf_mean:.0}/127{shape_desc}. Model score: {model_score:.2}.")
    } else {
        let diff = perf_mean - score_mean;
        let comparison = if diff > 15.0 {
            "louder than notated"
        } else if diff < -15.0 {
            "softer than notated"
        } else {
            "close to notated dynamic"
        };
        format!(
            "Mean velocity {perf_mean:.0}/127 ({comparison}){shape_desc}. Score mean: {score_mean:.0}. Model score: {model_score:.2}."
        )
    };

    let score_marking = tempo_markings
        .first()
        .and_then(|v| v.as_str())
        .map(std::string::ToString::to_string);

    let reference_comparison = reference.and_then(|ref_profile| {
        let ref_bars: Vec<_> = ref_profile
            .bars
            .iter()
            .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
            .collect();
        if ref_bars.is_empty() {
            return None;
        }
        let ref_vel_means: Vec<f64> = ref_bars.iter().map(|b| b.velocity_mean).collect();
        let ref_vel_stds: Vec<f64> = ref_bars.iter().map(|b| b.velocity_std).collect();
        let ref_mean = mean_f64(&ref_vel_means);
        let ref_std = mean_f64(&ref_vel_stds);
        let deviation = perf_mean - ref_mean;
        let within = if deviation.abs() <= ref_std { "within" } else { "outside" };
        Some(format!(
            "Reference performers average {ref_mean:.0} (std {ref_std:.1}); student is {within} reference range."
        ))
    });

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_DYNAMICS].to_string(),
        analysis,
        score_marking,
        reference_comparison,
    }
}

fn analyze_timing_tier1(
    alignments: &[NoteAlignment],
    reference: Option<&ReferenceProfile>,
    bar_range: (u32, u32),
    model_score: f64,
) -> DimensionAnalysis {
    let deviations: Vec<f64> = alignments.iter().map(|a| a.onset_deviation_ms).collect();
    let mean_dev = mean_f64(&deviations);
    let std_dev = std_f64(&deviations);

    let timing_class = if mean_dev < -30.0 {
        "rushing ahead of the score"
    } else if mean_dev > 30.0 {
        "dragging behind the score"
    } else {
        "close to score timing"
    };

    let analysis = format!(
        "Mean onset deviation {mean_dev:.1}ms (std {std_dev:.1}ms): {timing_class}. Model score: {model_score:.2}."
    );

    let reference_comparison = reference.and_then(|ref_profile| {
        let ref_bars: Vec<_> = ref_profile
            .bars
            .iter()
            .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
            .collect();
        if ref_bars.is_empty() {
            return None;
        }
        let ref_dev_means: Vec<f64> = ref_bars.iter().map(|b| b.onset_deviation_mean_ms).collect();
        let ref_dev_stds: Vec<f64> = ref_bars.iter().map(|b| b.onset_deviation_std_ms).collect();
        let ref_mean = mean_f64(&ref_dev_means);
        let ref_std = mean_f64(&ref_dev_stds);
        Some(format!(
            "Reference performers: mean deviation {ref_mean:.1}ms (std {ref_std:.1}ms)."
        ))
    });

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_TIMING].to_string(),
        analysis,
        score_marking: None,
        reference_comparison,
    }
}

fn analyze_pedaling_tier1(
    perf_pedal: &[PerfPedalEvent],
    score_bars: &[ScoreBar],
    reference: Option<&ReferenceProfile>,
    bar_range: (u32, u32),
    model_score: f64,
) -> DimensionAnalysis {
    let pedal_ons: Vec<f64> = perf_pedal
        .iter()
        .filter(|e| e.value >= 64)
        .map(|e| e.time)
        .collect();
    let pedal_offs: Vec<f64> = perf_pedal
        .iter()
        .filter(|e| e.value < 64)
        .map(|e| e.time)
        .collect();
    let event_count = pedal_ons.len();

    let avg_duration: Option<f64> = if !pedal_ons.is_empty() && !pedal_offs.is_empty() {
        let pairs: Vec<f64> = pedal_ons
            .iter()
            .zip(pedal_offs.iter())
            .map(|(on, off)| (off - on).max(0.0))
            .collect();
        if pairs.is_empty() {
            None
        } else {
            Some(mean_f64(&pairs))
        }
    } else {
        None
    };

    let score_pedal_count: usize = score_bars
        .iter()
        .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
        .map(|b| b.pedal_events.len())
        .sum();

    let analysis = if score_pedal_count > 0 && event_count == 0 {
        format!(
            "No pedal detected in performance, but score has {score_pedal_count} pedal marking(s). Model score: {model_score:.2}."
        )
    } else if let Some(dur) = avg_duration {
        format!(
            "{event_count} pedal event(s) detected, average duration {dur:.2}s. Score has {score_pedal_count} marking(s). Model score: {model_score:.2}."
        )
    } else {
        format!(
            "{event_count} pedal event(s) detected. Score has {score_pedal_count} marking(s). Model score: {model_score:.2}."
        )
    };

    let reference_comparison = reference.and_then(|ref_profile| {
        let ref_bars: Vec<_> = ref_profile
            .bars
            .iter()
            .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
            .collect();
        if ref_bars.is_empty() {
            return None;
        }
        let ref_pedal_counts: Vec<f64> = ref_bars
            .iter()
            .filter_map(|b| b.pedal_changes.map(f64::from))
            .collect();
        if ref_pedal_counts.is_empty() {
            return None;
        }
        let ref_count = mean_f64(&ref_pedal_counts);
        Some(format!(
            "Reference performers average {ref_count:.1} pedal changes per bar."
        ))
    });

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_PEDALING].to_string(),
        analysis,
        score_marking: None,
        reference_comparison,
    }
}

fn analyze_articulation_tier1(
    alignments: &[NoteAlignment],
    perf_notes: &[PerfNote],
    score_bars: &[ScoreBar],
    reference: Option<&ReferenceProfile>,
    bar_range: (u32, u32),
    model_score: f64,
) -> DimensionAnalysis {
    let perf_durations: Vec<f64> = perf_notes
        .iter()
        .map(|n| n.offset - n.onset)
        .filter(|&d| d > 0.0)
        .collect();
    let mean_perf_dur = mean_f64(&perf_durations);

    let score_durations: Vec<f64> = score_bars
        .iter()
        .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
        .flat_map(|b| b.notes.iter().map(|n| n.duration_seconds))
        .filter(|&d| d > 0.0)
        .collect();
    let mean_score_dur = mean_f64(&score_durations);

    let (ratio, style) = if mean_score_dur > 0.0 && mean_perf_dur > 0.0 {
        let r = mean_perf_dur / mean_score_dur;
        let s = if r > 1.1 {
            "legato (notes held longer than written)"
        } else if r < 0.6 {
            "staccato (notes shortened significantly)"
        } else {
            "normal note length"
        };
        (r, s)
    } else {
        (1.0, "normal note length")
    };

    let _ = alignments; // used for future score-beat-level analysis

    let analysis = if mean_score_dur > 0.0 {
        format!(
            "Mean note duration {mean_perf_dur:.3}s vs score {mean_score_dur:.3}s (ratio {ratio:.2}x): {style}. Model score: {model_score:.2}."
        )
    } else {
        format!("Mean note duration {mean_perf_dur:.3}s: {style}. Model score: {model_score:.2}.")
    };

    let reference_comparison = reference.and_then(|ref_profile| {
        let ref_bars: Vec<_> = ref_profile
            .bars
            .iter()
            .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
            .collect();
        if ref_bars.is_empty() {
            return None;
        }
        let ref_ratios: Vec<f64> = ref_bars
            .iter()
            .map(|b| b.note_duration_ratio_mean)
            .collect();
        let ref_ratio = mean_f64(&ref_ratios);
        Some(format!(
            "Reference performer note duration ratio: {ref_ratio:.2}x."
        ))
    });

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_ARTICULATION].to_string(),
        analysis,
        score_marking: None,
        reference_comparison,
    }
}

fn analyze_phrasing_tier1(
    alignments: &[NoteAlignment],
    reference: Option<&ReferenceProfile>,
    bar_range: (u32, u32),
    model_score: f64,
) -> DimensionAnalysis {
    let deviations: Vec<f64> = alignments.iter().map(|a| a.onset_deviation_ms).collect();
    let n = deviations.len();

    let analysis = if n >= 3 {
        let third = n / 3;
        let first_third_mean = mean_f64(&deviations[..third]);
        let last_third_mean = mean_f64(&deviations[n - third..]);
        let shift = last_third_mean - first_third_mean;

        if shift.abs() > 50.0 {
            let direction = if shift > 0.0 {
                "pulling back"
            } else {
                "pressing forward"
            };
            format!(
                "Timing shape shows a {:.0}ms shift from start to end of passage ({direction}). Model score: {:.2}.",
                shift.abs(), model_score
            )
        } else {
            format!(
                "Consistent timing shape across the passage (shift {shift:.0}ms). Model score: {model_score:.2}."
            )
        }
    } else {
        format!("Too few notes to assess phrasing shape. Model score: {model_score:.2}.")
    };

    let reference_comparison = reference.and_then(|ref_profile| {
        let ref_bars: Vec<_> = ref_profile
            .bars
            .iter()
            .filter(|b| b.bar_number >= bar_range.0 && b.bar_number <= bar_range.1)
            .collect();
        if ref_bars.is_empty() {
            return None;
        }
        let ref_stds: Vec<f64> = ref_bars.iter().map(|b| b.onset_deviation_std_ms).collect();
        let ref_std = mean_f64(&ref_stds);
        Some(format!(
            "Reference timing spread: {ref_std:.1}ms std across passage."
        ))
    });

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_PHRASING].to_string(),
        analysis,
        score_marking: None,
        reference_comparison,
    }
}

fn analyze_interpretation_tier1(
    alignments: &[NoteAlignment],
    model_score: f64,
) -> DimensionAnalysis {
    let abs_deviations: Vec<f64> = alignments
        .iter()
        .map(|a| a.onset_deviation_ms.abs())
        .collect();
    let mean_abs_dev = mean_f64(&abs_deviations);

    let analysis = format!(
        "Average absolute onset deviation from score: {mean_abs_dev:.1}ms. Model score: {model_score:.2}."
    );

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_INTERPRETATION].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

// --- Tier 2 per-dimension analyzers (absolute, no score context) ---

fn analyze_dynamics_tier2(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    let velocities: Vec<u8> = perf_notes.iter().map(|n| n.velocity).collect();
    let vel_min = velocities.iter().copied().min().unwrap_or(0);
    let vel_max = velocities.iter().copied().max().unwrap_or(0);
    let vel_mean = mean_u8(&velocities);
    let dynamic_range = vel_max.saturating_sub(vel_min);

    let analysis = format!(
        "Velocity range {vel_min}-{vel_max} (mean {vel_mean:.0}, range {dynamic_range}). Model score: {model_score:.2}."
    );

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_DYNAMICS].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_timing_tier2(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    let iois: Vec<f64> = perf_notes
        .windows(2)
        .map(|w| (w[1].onset - w[0].onset).abs())
        .filter(|&d| d > 0.0)
        .collect();
    let ioi_std = std_f64(&iois);
    let ioi_mean = mean_f64(&iois);

    let regularity = if ioi_std < 0.05 {
        "very regular"
    } else if ioi_std < 0.15 {
        "moderately regular"
    } else {
        "irregular"
    };

    let analysis = format!(
        "Inter-onset interval mean {ioi_mean:.3}s, std {ioi_std:.3}s ({regularity} timing). Model score: {model_score:.2}."
    );

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_TIMING].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_pedaling_tier2(perf_pedal: &[PerfPedalEvent], model_score: f64) -> DimensionAnalysis {
    let event_count = perf_pedal.len();

    let analysis = format!("{event_count} pedal event(s) detected. Model score: {model_score:.2}.");

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_PEDALING].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_articulation_tier2(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    let durations: Vec<f64> = perf_notes
        .iter()
        .map(|n| n.offset - n.onset)
        .filter(|&d| d > 0.0)
        .collect();
    let mean_dur = mean_f64(&durations);

    let analysis = format!("Mean note duration {mean_dur:.3}s. Model score: {model_score:.2}.");

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_ARTICULATION].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_phrasing_tier2(perf_notes: &[PerfNote], model_score: f64) -> DimensionAnalysis {
    let note_count = perf_notes.len();

    let analysis = format!("{note_count} notes in passage. Model score: {model_score:.2}.");

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_PHRASING].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

fn analyze_interpretation_tier2(model_score: f64) -> DimensionAnalysis {
    let analysis = format!("Model score: {model_score:.2}.");

    DimensionAnalysis {
        dimension: DIMENSION_NAMES[DIM_INTERPRETATION].to_string(),
        analysis,
        score_marking: None,
        reference_comparison: None,
    }
}

// --- Public API ---

/// Full bar-aligned analysis with score and reference comparison (Tier 1).
///
/// Requires a `BarMap` from the score follower and a `ScoreContext`. Produces
/// one `DimensionAnalysis` per dimension with score markings and reference
/// comparisons where available.
pub fn analyze_tier1(
    bar_map: &BarMap,
    perf_notes: &[PerfNote],
    perf_pedal: &[PerfPedalEvent],
    scores: &[f64; 6],
    score_ctx: &ScoreContext,
) -> ChunkAnalysis {
    let bar_range = (bar_map.bar_start, bar_map.bar_end);
    let bar_range_str = if bar_map.bar_start == bar_map.bar_end {
        format!("bar {}", bar_map.bar_start)
    } else {
        format!("bars {}-{}", bar_map.bar_start, bar_map.bar_end)
    };

    let score_bars = &score_ctx.score.bars;
    let alignments = &bar_map.alignments;
    let reference = score_ctx.reference.as_ref();
    let tempo_markings = &score_ctx.score.tempo_markings;

    let dimensions = vec![
        analyze_dynamics_tier1(
            alignments,
            score_bars,
            reference,
            tempo_markings,
            bar_range,
            scores[DIM_DYNAMICS],
        ),
        analyze_timing_tier1(alignments, reference, bar_range, scores[DIM_TIMING]),
        analyze_pedaling_tier1(
            perf_pedal,
            score_bars,
            reference,
            bar_range,
            scores[DIM_PEDALING],
        ),
        analyze_articulation_tier1(
            alignments,
            perf_notes,
            score_bars,
            reference,
            bar_range,
            scores[DIM_ARTICULATION],
        ),
        analyze_phrasing_tier1(alignments, reference, bar_range, scores[DIM_PHRASING]),
        analyze_interpretation_tier1(alignments, scores[DIM_INTERPRETATION]),
    ];

    ChunkAnalysis {
        tier: 1,
        bar_range: Some(bar_range_str),
        dimensions,
    }
}

/// Absolute MIDI analysis without score context (Tier 2).
///
/// Used when no piece is identified or score alignment fails. Produces one
/// `DimensionAnalysis` per dimension based on raw MIDI statistics.
pub fn analyze_tier2(
    perf_notes: &[PerfNote],
    perf_pedal: &[PerfPedalEvent],
    scores: &[f64; 6],
) -> ChunkAnalysis {
    let dimensions = vec![
        analyze_dynamics_tier2(perf_notes, scores[DIM_DYNAMICS]),
        analyze_timing_tier2(perf_notes, scores[DIM_TIMING]),
        analyze_pedaling_tier2(perf_pedal, scores[DIM_PEDALING]),
        analyze_articulation_tier2(perf_notes, scores[DIM_ARTICULATION]),
        analyze_phrasing_tier2(perf_notes, scores[DIM_PHRASING]),
        analyze_interpretation_tier2(scores[DIM_INTERPRETATION]),
    ];

    ChunkAnalysis {
        tier: 2,
        bar_range: None,
        dimensions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BarMap, NoteAlignment, PerfNote, PerfPedalEvent};

    fn make_perf_note(pitch: u8, onset: f64, offset: f64, velocity: u8) -> PerfNote {
        PerfNote {
            pitch,
            onset,
            offset,
            velocity,
        }
    }

    fn make_alignment(perf_velocity: u8, onset_deviation_ms: f64) -> NoteAlignment {
        NoteAlignment {
            perf_onset: 0.0,
            perf_pitch: 60,
            perf_velocity,
            score_bar: 1,
            score_beat: 0.0,
            score_pitch: 60,
            onset_deviation_ms,
        }
    }

    fn make_bar_map(alignments: Vec<NoteAlignment>) -> BarMap {
        BarMap {
            chunk_index: 0,
            bar_start: 1,
            bar_end: 2,
            alignments,
            confidence: 0.9,
            is_reanchored: false,
        }
    }

    fn default_scores() -> [f64; 6] {
        [0.75, 0.80, 0.65, 0.70, 0.72, 0.68]
    }

    #[test]
    fn tier2_produces_all_6_dimensions() {
        let notes = vec![
            make_perf_note(60, 0.0, 0.4, 80),
            make_perf_note(62, 0.5, 0.9, 85),
            make_perf_note(64, 1.0, 1.4, 75),
        ];
        let pedal = vec![];
        let scores = default_scores();

        let result = analyze_tier2(&notes, &pedal, &scores);

        assert_eq!(result.tier, 2);
        assert!(result.bar_range.is_none());
        assert_eq!(result.dimensions.len(), 6);

        let dim_names: Vec<&str> = result
            .dimensions
            .iter()
            .map(|d| d.dimension.as_str())
            .collect();
        assert!(dim_names.contains(&"dynamics"));
        assert!(dim_names.contains(&"timing"));
        assert!(dim_names.contains(&"pedaling"));
        assert!(dim_names.contains(&"articulation"));
        assert!(dim_names.contains(&"phrasing"));
        assert!(dim_names.contains(&"interpretation"));
    }

    #[test]
    fn tier2_includes_model_scores() {
        let notes = vec![
            make_perf_note(60, 0.0, 0.4, 80),
            make_perf_note(62, 0.5, 0.9, 85),
        ];
        let pedal = vec![];
        let scores = default_scores();

        let result = analyze_tier2(&notes, &pedal, &scores);

        for dim in &result.dimensions {
            assert!(
                dim.analysis.contains("Model score"),
                "Dimension '{}' analysis missing 'Model score': {}",
                dim.dimension,
                dim.analysis
            );
        }
    }

    #[test]
    fn timing_detects_rushing() {
        let alignments = vec![
            make_alignment(80, -50.0),
            make_alignment(80, -60.0),
            make_alignment(80, -45.0),
            make_alignment(80, -55.0),
        ];

        let deviations: Vec<f64> = alignments.iter().map(|a| a.onset_deviation_ms).collect();
        let mean_dev: f64 = deviations.iter().sum::<f64>() / deviations.len() as f64;
        assert!(
            mean_dev < -30.0,
            "Expected mean deviation < -30ms for rushing, got {}",
            mean_dev
        );

        let scores = default_scores();
        let timing_dim = analyze_timing_tier1(
            &alignments,
            None,
            (1, 2),
            scores[DIM_TIMING],
        );

        assert!(
            timing_dim.analysis.contains("rushing"),
            "Expected 'rushing' in timing analysis, got: {}",
            timing_dim.analysis
        );
    }

    #[test]
    fn pedaling_detects_no_pedal_vs_score() {
        use crate::types::{ScoreBar, ScorePedalEvent};

        let score_bar_with_pedal = ScoreBar {
            bar_number: 1,
            start_tick: 0,
            start_seconds: 0.0,
            time_signature: "4/4".to_string(),
            notes: vec![],
            pedal_events: vec![
                ScorePedalEvent {
                    event_type: "on".to_string(),
                    tick: 0,
                    seconds: 0.0,
                },
                ScorePedalEvent {
                    event_type: "off".to_string(),
                    tick: 480,
                    seconds: 0.5,
                },
            ],
            note_count: 0,
            pitch_range: vec![],
            mean_velocity: 80,
        };

        let perf_pedal: Vec<PerfPedalEvent> = vec![];
        let scores = default_scores();

        let pedal_dim = analyze_pedaling_tier1(
            &perf_pedal,
            &[score_bar_with_pedal],
            None,
            (1, 1),
            scores[DIM_PEDALING],
        );

        assert!(
            pedal_dim.analysis.contains("No pedal detected"),
            "Expected 'No pedal detected' in analysis, got: {}",
            pedal_dim.analysis
        );
    }
}
