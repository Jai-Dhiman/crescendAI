//! Teaching moment selection: given a session of scored chunks, determine
//! which chunk is a teaching moment and which dimension to surface.
//!
//! Algorithm:
//! 1. For each chunk, find its worst dimension (largest negative deviation from baseline)
//! 2. Keep chunks whose worst dimension is below baseline (deviation < 0)
//! 3. Rank by deviation magnitude (blind-spot detection: "normally fine but bad today")
//! 4. De-duplicate against recent observations (skip if same dimension in last 3)
//! 5. Return top-1 `TeachingMoment`
//!
//! No-candidates fallback (positive teaching moment):
//! - When every chunk is at-or-above baseline on every dimension, no negative
//!   candidate exists. Return a positive moment instead.
//! - Find highest-scoring dimension or largest positive deviation from baseline
//! - Return positive moment with `is_positive`: true
//! - If fewer than 2 chunks, return None ("need more to listen to")

use crate::dims::DIMS_6;
use crate::types::{RecentObservation, ScoredChunk, StudentBaselines, TeachingMoment};

/// Minimum number of scored chunks before we can produce a teaching moment.
const MIN_CHUNKS: usize = 2;

/// How many recent observations to check for dimension dedup.
const DEDUP_WINDOW: usize = 3;

/// Select the top-1 teaching moment from a session's scored chunks.
///
/// Returns None only when fewer than `MIN_CHUNKS` are provided (the caller
/// should respond with "I need a bit more to listen to").
///
/// When every chunk is at-or-above baseline on every dimension, returns a
/// positive teaching moment acknowledging what the student did well.
pub fn select_teaching_moment(
    chunks: &[ScoredChunk],
    baselines: &StudentBaselines,
    recent_observations: &[RecentObservation],
) -> Option<TeachingMoment> {
    if chunks.len() < MIN_CHUNKS {
        return None;
    }

    let baseline_arr = baselines.as_array();

    // Find each chunk's worst dimension and keep chunks below baseline.
    let mut candidates: Vec<Candidate> = Vec::new();

    for chunk in chunks {
        // Find the blind-spot dimension: largest negative deviation from baseline.
        // "Normally fine but bad today" is more valuable than "always bad".
        let (dim_idx, deviation) = max_negative_deviation(&chunk.scores, &baseline_arr);

        // Only treat as a candidate if the student is actually below baseline somewhere.
        // If they are at-or-above baseline on every dimension, this chunk is fine -- no
        // candidate. If every chunk is fine, the positive-moment fallback fires below.
        if deviation >= 0.0 {
            continue;
        }

        candidates.push(Candidate {
            chunk_index: chunk.chunk_index,
            dim_idx,
            score: chunk.scores[dim_idx],
            baseline: baseline_arr[dim_idx],
            deviation,
        });
    }

    // Sort by deviation magnitude (largest negative deviation first).
    candidates.sort_by(|a, b| {
        a.deviation
            .partial_cmp(&b.deviation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Dedup: skip candidates whose top dimension matches recent observations.
    let recent_dims: Vec<&str> = recent_observations
        .iter()
        .take(DEDUP_WINDOW)
        .map(|o| o.dimension.as_str())
        .collect();

    for candidate in &candidates {
        let dim_name = DIMS_6[candidate.dim_idx];
        if recent_dims.contains(&dim_name) {
            continue;
        }
        return Some(candidate.to_teaching_moment(false));
    }

    // If all candidates were deduped, return the top one anyway
    // (better to repeat a dimension than say nothing when there's a real issue).
    if let Some(candidate) = candidates.first() {
        return Some(candidate.to_teaching_moment(false));
    }

    // No chunks were below baseline anywhere -> positive teaching moment fallback.
    Some(select_positive_moment(chunks, &baseline_arr))
}

/// Select up to `max` within-session teaching moments, ranked against a
/// caller-supplied within-session `reference` (typically the per-dimension
/// session mean) rather than a stored longitudinal baseline.
///
/// Algorithm:
/// 1. For each chunk, find its worst dimension vs `reference` (largest negative deviation).
/// 2. Keep chunks whose worst dimension is below the reference (deviation < 0).
/// 3. Rank by deviation magnitude (most-below-mean first).
/// 4. Walk ranked candidates, emitting at most one moment per distinct dimension, up to `max`.
/// 5. If no chunk is below the reference anywhere, return a single positive moment.
/// 6. If fewer than `MIN_CHUNKS` chunks, return an empty vec.
///
/// Reasoning strings are within-session phrased (reference-neutral), never longitudinal.
/// This function is additive; `select_teaching_moment` (the live path) is unaffected.
pub fn select_session_moments(
    chunks: &[ScoredChunk],
    reference: &[f64; 6],
    max: usize,
) -> Vec<TeachingMoment> {
    if chunks.len() < MIN_CHUNKS {
        return Vec::new();
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for chunk in chunks {
        let (dim_idx, deviation) = max_negative_deviation(&chunk.scores, reference);
        if deviation >= 0.0 {
            continue;
        }
        candidates.push(Candidate {
            chunk_index: chunk.chunk_index,
            dim_idx,
            score: chunk.scores[dim_idx],
            baseline: reference[dim_idx],
            deviation,
        });
    }

    candidates.sort_by(|a, b| {
        a.deviation
            .partial_cmp(&b.deviation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut moments: Vec<TeachingMoment> = Vec::new();
    let mut used_dims: Vec<usize> = Vec::new();
    for candidate in &candidates {
        if moments.len() >= max {
            break;
        }
        if used_dims.contains(&candidate.dim_idx) {
            continue;
        }
        used_dims.push(candidate.dim_idx);
        moments.push(candidate.to_session_moment());
    }

    if moments.is_empty() {
        moments.push(select_positive_moment(chunks, reference));
    }

    moments
}

/// Positive teaching moment: find what the student did best.
#[allow(clippy::needless_range_loop)] // i indexes both scores[i] and baselines[i] in parallel
fn select_positive_moment(chunks: &[ScoredChunk], baselines: &[f64; 6]) -> TeachingMoment {
    // Strategy: find the dimension with the largest positive deviation from baseline.
    // If no baseline deviations are positive, fall back to highest absolute score.
    let mut best_dim_idx = 0;
    let mut best_deviation = f64::NEG_INFINITY;
    let mut best_chunk_idx = 0;
    let mut best_score = 0.0;

    for chunk in chunks {
        for i in 0..6 {
            let deviation = chunk.scores[i] - baselines[i];
            if deviation > best_deviation {
                best_deviation = deviation;
                best_dim_idx = i;
                best_chunk_idx = chunk.chunk_index;
                best_score = chunk.scores[i];
            }
        }
    }

    // If no positive deviation found (all below baseline), use highest absolute score.
    if best_deviation <= 0.0 {
        for chunk in chunks {
            for i in 0..6 {
                if chunk.scores[i] > best_score {
                    best_score = chunk.scores[i];
                    best_dim_idx = i;
                    best_chunk_idx = chunk.chunk_index;
                    best_deviation = chunk.scores[i] - baselines[i];
                }
            }
        }
    }

    let dim_name = DIMS_6[best_dim_idx];
    TeachingMoment {
        chunk_index: best_chunk_idx,
        dimension: dim_name.to_string(),
        score: best_score,
        baseline: baselines[best_dim_idx],
        deviation: best_deviation,
        reasoning: format!(
            "No issues detected. {} scored {:.2} (baseline {:.2}, +{:.2} deviation). Positive moment.",
            dim_name, best_score, baselines[best_dim_idx], best_deviation,
        ),
        is_positive: true,
    }
}

/// Find the dimension with the largest negative deviation from baseline.
/// Returns (`dimension_index`, deviation). Deviation is negative when below baseline.
fn max_negative_deviation(scores: &[f64; 6], baselines: &[f64; 6]) -> (usize, f64) {
    let mut worst_idx = 0;
    let mut worst_dev = f64::MAX;

    for i in 0..6 {
        let dev = scores[i] - baselines[i];
        if dev < worst_dev {
            worst_dev = dev;
            worst_idx = i;
        }
    }

    (worst_idx, worst_dev)
}

struct Candidate {
    chunk_index: usize,
    dim_idx: usize,
    score: f64,
    baseline: f64,
    deviation: f64,
}

impl Candidate {
    fn to_teaching_moment(&self, is_positive: bool) -> TeachingMoment {
        let dim_name = DIMS_6[self.dim_idx];
        let reasoning = format!(
            "{} scored {:.2} vs baseline {:.2} (deviation {:.2}).",
            dim_name, self.score, self.baseline, self.deviation,
        );

        TeachingMoment {
            chunk_index: self.chunk_index,
            dimension: dim_name.to_string(),
            score: self.score,
            baseline: self.baseline,
            deviation: self.deviation,
            reasoning,
            is_positive,
        }
    }

    fn to_session_moment(&self) -> TeachingMoment {
        let dim_name = DIMS_6[self.dim_idx];
        let reasoning = format!(
            "{} was the weakest relative to the rest of this session: {:.2} vs session average {:.2} (deviation {:.2}).",
            dim_name, self.score, self.baseline, self.deviation,
        );
        TeachingMoment {
            chunk_index: self.chunk_index,
            dimension: dim_name.to_string(),
            score: self.score,
            baseline: self.baseline,
            deviation: self.deviation,
            reasoning,
            is_positive: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StudentBaselines;

    fn default_baselines() -> StudentBaselines {
        StudentBaselines {
            dynamics: 0.55,
            timing: 0.50,
            pedaling: 0.45,
            articulation: 0.54,
            phrasing: 0.52,
            interpretation: 0.50,
        }
    }

    #[test]
    fn selects_worst_chunk() {
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.55, 0.50, 0.45, 0.54, 0.52, 0.50],
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50],
            },
            ScoredChunk {
                chunk_index: 2,
                scores: [0.55, 0.50, 0.40, 0.54, 0.52, 0.50],
            },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &[]);
        let moment = result.expect("should return a teaching moment");
        assert_eq!(moment.chunk_index, 1, "should pick chunk with worst pedaling");
        assert_eq!(moment.dimension, "pedaling");
        assert!(!moment.is_positive);
        assert!(moment.deviation < -0.2);
    }

    #[test]
    fn dedup_skips_recent_dimension() {
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50],
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.20, 0.50, 0.15, 0.54, 0.52, 0.50],
            },
        ];

        let recent = vec![RecentObservation {
            dimension: "pedaling".to_string(),
        }];

        let result = select_teaching_moment(&chunks, &default_baselines(), &recent);
        let moment = result.expect("should return a teaching moment");
        assert_eq!(moment.dimension, "dynamics");
    }

    #[test]
    fn no_chunks_above_threshold_returns_positive() {
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.80, 0.75, 0.85, 0.54, 0.52, 0.75],
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.85, 0.78, 0.90, 0.55, 0.53, 0.80],
            },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &[]);
        let moment = result.expect("should return a positive moment");
        assert!(moment.is_positive);
        assert!(moment.deviation > 0.0);
    }

    #[test]
    fn too_few_chunks_returns_none() {
        let chunks = vec![ScoredChunk {
            chunk_index: 0,
            scores: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        }];
        let result = select_teaching_moment(&chunks, &default_baselines(), &[]);
        assert!(result.is_none(), "should return None with only 1 chunk");
    }

    #[test]
    fn single_chunk_above_threshold() {
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.80, 0.75, 0.85, 0.55, 0.53, 0.75],
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.10, 0.50, 0.10, 0.54, 0.52, 0.50],
            },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &[]);
        let moment = result.expect("should find the bad chunk");
        assert_eq!(moment.chunk_index, 1);
        assert!(!moment.is_positive);
    }

    #[test]
    fn dedup_exhausted_returns_top_anyway() {
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50],
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.55, 0.50, 0.12, 0.54, 0.52, 0.50],
            },
        ];

        let recent = vec![
            RecentObservation { dimension: "pedaling".to_string() },
            RecentObservation { dimension: "dynamics".to_string() },
            RecentObservation { dimension: "timing".to_string() },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &recent);
        let moment = result.expect("should return top candidate even if deduped");
        assert_eq!(moment.dimension, "pedaling");
    }

    #[test]
    fn positive_moment_picks_highest_improvement() {
        let baselines = StudentBaselines {
            dynamics: 0.50,
            timing: 0.50,
            pedaling: 0.40,
            articulation: 0.50,
            phrasing: 0.50,
            interpretation: 0.50,
        };
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.80, 0.75, 0.85, 0.55, 0.53, 0.75],
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.82, 0.78, 0.70, 0.55, 0.53, 0.80],
            },
        ];

        let result = select_teaching_moment(&chunks, &baselines, &[]);
        let moment = result.expect("should return positive moment");
        assert!(moment.is_positive);
        assert_eq!(moment.dimension, "pedaling");
        assert!(moment.deviation > 0.4);
    }

    fn reference_mean(chunks: &[ScoredChunk]) -> [f64; 6] {
        let mut sums = [0.0f64; 6];
        for c in chunks {
            for i in 0..6 {
                sums[i] += c.scores[i];
            }
        }
        let n = chunks.len() as f64;
        let mut mean = [0.0f64; 6];
        for i in 0..6 {
            mean[i] = sums[i] / n;
        }
        mean
    }

    #[test]
    fn session_moments_rank_by_magnitude_vs_session_mean() {
        // Chunk 1 is far below the session mean on pedaling -> should rank first.
        let chunks = vec![
            ScoredChunk { chunk_index: 0, scores: [0.55, 0.50, 0.50, 0.54, 0.52, 0.50] },
            ScoredChunk { chunk_index: 1, scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50] },
            ScoredChunk { chunk_index: 2, scores: [0.55, 0.50, 0.48, 0.54, 0.52, 0.50] },
        ];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 6);
        assert!(!moments.is_empty(), "should return at least one moment");
        assert_eq!(moments[0].dimension, "pedaling");
        assert_eq!(moments[0].chunk_index, 1);
        assert!(moments[0].deviation < 0.0, "weakest moment is below the session mean");
        assert!(!moments[0].is_positive);
        assert!(
            moments[0].reasoning.contains("this session"),
            "reasoning must be within-session phrased, got: {}",
            moments[0].reasoning
        );
    }

    #[test]
    fn session_moments_return_distinct_dimensions_up_to_max() {
        let chunks = vec![
            ScoredChunk { chunk_index: 0, scores: [0.10, 0.80, 0.80, 0.80, 0.80, 0.80] },
            ScoredChunk { chunk_index: 1, scores: [0.80, 0.10, 0.80, 0.80, 0.80, 0.80] },
            ScoredChunk { chunk_index: 2, scores: [0.80, 0.80, 0.10, 0.80, 0.80, 0.80] },
        ];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 2);
        assert_eq!(moments.len(), 2, "must respect max");
        assert_ne!(
            moments[0].dimension, moments[1].dimension,
            "moments must span distinct dimensions"
        );
    }

    #[test]
    fn session_moments_positive_fallback_when_all_at_mean() {
        // Every chunk identical -> deviation from mean is 0 everywhere -> positive fallback.
        let chunks = vec![
            ScoredChunk { chunk_index: 0, scores: [0.60, 0.60, 0.60, 0.60, 0.60, 0.60] },
            ScoredChunk { chunk_index: 1, scores: [0.60, 0.60, 0.60, 0.60, 0.60, 0.60] },
        ];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 6);
        assert_eq!(moments.len(), 1, "positive fallback returns a single moment");
        assert!(moments[0].is_positive);
    }

    #[test]
    fn session_moments_empty_when_too_few_chunks() {
        let chunks = vec![ScoredChunk { chunk_index: 0, scores: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3] }];
        let reference = reference_mean(&chunks);
        let moments = select_session_moments(&chunks, &reference, 6);
        assert!(moments.is_empty(), "fewer than 2 chunks -> empty");
    }
}
