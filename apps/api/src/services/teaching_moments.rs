//! Teaching moment selection: given a session of scored chunks, determine
//! which chunk is a teaching moment and which dimension to surface.
//!
//! Algorithm (from 02-pipeline.md Stage 3):
//! 1. Run STOP classifier on each chunk, filter by threshold
//! 2. For passing chunks, compute deviation from student baseline per dimension
//! 3. Rank by max deviation (blind-spot detection: "normally fine but bad today")
//! 4. De-duplicate against recent observations (skip if same dimension in last 3)
//! 5. Return top-1 `TeachingMoment`
//!
//! No-candidates fallback (positive teaching moment):
//! - Find highest-scoring dimension or largest positive deviation from baseline
//! - Return positive moment with `is_positive`: true
//! - If fewer than 2 chunks, return None ("need more to listen to")

use crate::practice::dims::DIMS_6;
use crate::services::stop;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScoredChunk {
    pub chunk_index: usize,
    pub scores: [f64; 6],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StudentBaselines {
    pub dynamics: f64,
    pub timing: f64,
    pub pedaling: f64,
    pub articulation: f64,
    pub phrasing: f64,
    pub interpretation: f64,
}

impl StudentBaselines {
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.dynamics,
            self.timing,
            self.pedaling,
            self.articulation,
            self.phrasing,
            self.interpretation,
        ]
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecentObservation {
    pub dimension: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TeachingMoment {
    pub chunk_index: usize,
    pub dimension: String,
    pub score: f64,
    pub baseline: f64,
    pub deviation: f64,
    pub stop_probability: f64,
    pub reasoning: String,
    pub is_positive: bool,
}

/// Minimum number of scored chunks before we can produce a teaching moment.
const MIN_CHUNKS: usize = 2;

/// How many recent observations to check for dimension dedup.
const DEDUP_WINDOW: usize = 3;

/// Select the top-1 teaching moment from a session's scored chunks.
///
/// Returns None only when fewer than `MIN_CHUNKS` are provided (the caller
/// should respond with "I need a bit more to listen to").
///
/// When no chunks pass the STOP threshold, returns a positive teaching
/// moment acknowledging what the student did well (per 02-pipeline.md
/// "No-Candidates Fallback").
pub fn select_teaching_moment(
    chunks: &[ScoredChunk],
    baselines: &StudentBaselines,
    recent_observations: &[RecentObservation],
) -> Option<TeachingMoment> {
    if chunks.len() < MIN_CHUNKS {
        return None;
    }

    let baseline_arr = baselines.as_array();

    // Score each chunk with STOP classifier and compute per-dimension deviations.
    let mut candidates: Vec<Candidate> = Vec::new();

    for chunk in chunks {
        let stop_result = stop::classify(&chunk.scores);
        if !stop_result.triggered {
            continue;
        }

        // Find the blind-spot dimension: largest negative deviation from baseline.
        // "Normally fine but bad today" is more valuable than "always bad".
        let (dim_idx, deviation) = max_negative_deviation(&chunk.scores, &baseline_arr);

        candidates.push(Candidate {
            chunk_index: chunk.chunk_index,
            dim_idx,
            score: chunk.scores[dim_idx],
            baseline: baseline_arr[dim_idx],
            deviation,
            stop_probability: stop_result.probability,
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
    // (better to repeat a dimension than say nothing when STOP fired).
    if let Some(candidate) = candidates.first() {
        return Some(candidate.to_teaching_moment(false));
    }

    // No chunks passed STOP threshold -> positive teaching moment fallback.
    Some(select_positive_moment(chunks, &baseline_arr))
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
        stop_probability: 0.0,
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
    stop_probability: f64,
}

impl Candidate {
    fn to_teaching_moment(&self, is_positive: bool) -> TeachingMoment {
        let dim_name = DIMS_6[self.dim_idx];
        let reasoning = format!(
            "{} scored {:.2} vs baseline {:.2} (deviation {:.2}). STOP probability: {:.2}.",
            dim_name, self.score, self.baseline, self.deviation, self.stop_probability,
        );

        TeachingMoment {
            chunk_index: self.chunk_index,
            dimension: dim_name.to_string(),
            score: self.score,
            baseline: self.baseline,
            deviation: self.deviation,
            stop_probability: self.stop_probability,
            reasoning,
            is_positive,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                scores: [0.55, 0.50, 0.45, 0.54, 0.52, 0.50], // at baseline
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50], // pedaling terrible
            },
            ScoredChunk {
                chunk_index: 2,
                scores: [0.55, 0.50, 0.40, 0.54, 0.52, 0.50], // slight dip
            },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &[]);
        let moment = result.expect("should return a teaching moment");
        assert_eq!(
            moment.chunk_index, 1,
            "should pick chunk with worst pedaling"
        );
        assert_eq!(moment.dimension, "pedaling");
        assert!(!moment.is_positive);
        assert!(moment.deviation < -0.2);
    }

    #[test]
    fn dedup_skips_recent_dimension() {
        // Both chunks have pedaling issues, but pedaling was recently observed.
        // Should skip pedaling and pick the next worst dimension.
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.55, 0.50, 0.10, 0.54, 0.52, 0.50], // pedaling bad
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.20, 0.50, 0.15, 0.54, 0.52, 0.50], // dynamics + pedaling bad
            },
        ];

        let recent = vec![RecentObservation {
            dimension: "pedaling".to_string(),
        }];

        let result = select_teaching_moment(&chunks, &default_baselines(), &recent);
        let moment = result.expect("should return a teaching moment");
        // Should pick dynamics (chunk 1) since pedaling is deduped
        assert_eq!(moment.dimension, "dynamics");
    }

    #[test]
    fn no_chunks_above_threshold_returns_positive() {
        // All scores are very high (good playing) -> STOP won't fire.
        // Should return a positive teaching moment.
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
        assert_eq!(moment.stop_probability, 0.0);
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
        // Two chunks but only one triggers STOP
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.80, 0.75, 0.85, 0.55, 0.53, 0.75], // good
            },
            ScoredChunk {
                chunk_index: 1,
                scores: [0.10, 0.50, 0.10, 0.54, 0.52, 0.50], // bad dynamics + pedaling
            },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &[]);
        let moment = result.expect("should find the bad chunk");
        assert_eq!(moment.chunk_index, 1);
        assert!(!moment.is_positive);
    }

    #[test]
    fn dedup_exhausted_returns_top_anyway() {
        // Only one candidate dimension, and it was recently observed.
        // Should still return it rather than nothing.
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
            RecentObservation {
                dimension: "pedaling".to_string(),
            },
            RecentObservation {
                dimension: "dynamics".to_string(),
            },
            RecentObservation {
                dimension: "timing".to_string(),
            },
        ];

        let result = select_teaching_moment(&chunks, &default_baselines(), &recent);
        let moment = result.expect("should return top candidate even if deduped");
        // Should be pedaling (worst deviation) even though it's in recent
        assert_eq!(moment.dimension, "pedaling");
    }

    #[test]
    fn positive_moment_picks_highest_improvement() {
        // No STOP triggers (good scores), but pedaling improved most above baseline.
        let baselines = StudentBaselines {
            dynamics: 0.50,
            timing: 0.50,
            pedaling: 0.40, // baseline is low
            articulation: 0.50,
            phrasing: 0.50,
            interpretation: 0.50,
        };
        let chunks = vec![
            ScoredChunk {
                chunk_index: 0,
                scores: [0.80, 0.75, 0.85, 0.55, 0.53, 0.75], // pedaling = 0.85, baseline 0.40 -> +0.45
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
}
