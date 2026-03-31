use std::collections::HashMap;

use super::practice_mode::PracticeMode;

/// A single teaching moment accumulated during a practice session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccumulatedMoment {
    pub chunk_index: usize,
    pub dimension: String,
    pub score: f64,
    pub baseline: f64,
    pub deviation: f64,
    pub is_positive: bool,
    pub reasoning: String,
    pub bar_range: Option<(u32, u32)>,
    pub analysis_tier: u8,
    pub timestamp_ms: u64,
    pub llm_analysis: Option<String>,
}

/// A record of a mode transition during a practice session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModeTransitionRecord {
    pub from: PracticeMode,
    pub to: PracticeMode,
    pub chunk_index: usize,
    pub timestamp_ms: u64,
    pub dwell_ms: u64,
}

/// A record of a drilling passage during a practice session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DrillingRecord {
    pub bar_range: Option<(u32, u32)>,
    pub repetition_count: usize,
    pub first_scores: [f64; 6],
    pub final_scores: [f64; 6],
    pub started_at_chunk: usize,
    pub ended_at_chunk: usize,
}

/// A timeline event representing a chunk of audio processed during a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimelineEvent {
    pub chunk_index: usize,
    pub timestamp_ms: u64,
    pub has_audio: bool,
}

/// Accumulates practice session signals during a student's playing session.
/// After the session ends, a single LLM call synthesizes all accumulated data
/// into cohesive feedback.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionAccumulator {
    pub teaching_moments: Vec<AccumulatedMoment>,
    pub mode_transitions: Vec<ModeTransitionRecord>,
    pub drilling_records: Vec<DrillingRecord>,
    pub timeline: Vec<TimelineEvent>,
}

impl SessionAccumulator {
    pub fn accumulate_moment(&mut self, moment: AccumulatedMoment) {
        self.teaching_moments.push(moment);
    }

    pub fn accumulate_mode_transition(&mut self, record: ModeTransitionRecord) {
        self.mode_transitions.push(record);
    }

    pub fn accumulate_drilling_record(&mut self, record: DrillingRecord) {
        self.drilling_records.push(record);
    }

    pub fn accumulate_timeline_event(&mut self, event: TimelineEvent) {
        self.timeline.push(event);
    }

    /// Returns true if there is any teaching content to synthesize.
    pub fn has_teaching_content(&self) -> bool {
        !self.teaching_moments.is_empty() || !self.drilling_records.is_empty()
    }

    /// Returns the top moments for synthesis.
    ///
    /// Algorithm:
    /// 1. For each dimension, select the moment with the highest |deviation| (top-1 per dim).
    /// 2. For each dimension, also select the top-1 positive moment if it differs from step 1.
    /// 3. Cap total at 8 moments.
    /// 4. Sort the final set by `chunk_index` ascending.
    pub fn top_moments(
        &self,
        dimension_weights: Option<&HashMap<String, f64>>,
    ) -> Vec<&AccumulatedMoment> {
        let dimensions = [
            "dynamics",
            "timing",
            "pedaling",
            "articulation",
            "phrasing",
            "interpretation",
        ];
        let mut selected: Vec<&AccumulatedMoment> = Vec::new();

        for dim in &dimensions {
            let dim_moments: Vec<&AccumulatedMoment> = self
                .teaching_moments
                .iter()
                .filter(|m| m.dimension == *dim)
                .collect();

            if dim_moments.is_empty() {
                continue;
            }

            // Top-1 per dimension by |deviation|
            let top_by_deviation = dim_moments.iter().copied().max_by(|a, b| {
                a.deviation
                    .abs()
                    .partial_cmp(&b.deviation.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(top) = top_by_deviation {
                // Use chunk_index as identity to avoid duplicate ptr checks
                if !selected.iter().any(|m| std::ptr::eq(*m, top)) {
                    selected.push(top);
                }

                // Top-1 positive per dimension if different from top_by_deviation
                let top_positive = dim_moments
                    .iter()
                    .copied()
                    .filter(|m| m.is_positive)
                    .max_by(|a, b| {
                        a.deviation
                            .abs()
                            .partial_cmp(&b.deviation.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                if let Some(pos) = top_positive {
                    if !std::ptr::eq(pos, top) && !selected.iter().any(|m| std::ptr::eq(*m, pos)) {
                        selected.push(pos);
                    }
                }
            }
        }

        // If dimension weights are provided, re-sort candidates by weighted deviation
        // before capping. This ensures high-weight dimensions (e.g., pedaling for Chopin)
        // are prioritized when more than 8 candidates exist.
        if let Some(weights) = dimension_weights {
            selected.sort_by(|a, b| {
                let ad = if a.deviation.is_nan() {
                    0.0
                } else {
                    a.deviation.abs()
                };
                let bd = if b.deviation.is_nan() {
                    0.0
                } else {
                    b.deviation.abs()
                };
                let aw = ad * weights.get(&a.dimension).copied().unwrap_or(1.0);
                let bw = bd * weights.get(&b.dimension).copied().unwrap_or(1.0);
                bw.partial_cmp(&aw).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Cap at 8
        selected.truncate(8);

        // Sort by chunk_index ascending
        selected.sort_by_key(|m| m.chunk_index);

        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_moment(chunk: usize, dim: &str, deviation: f64, positive: bool) -> AccumulatedMoment {
        AccumulatedMoment {
            chunk_index: chunk,
            dimension: dim.to_string(),
            score: 0.5 + deviation,
            baseline: 0.5,
            deviation,
            is_positive: positive,
            reasoning: format!("{} chunk {}", dim, chunk),
            bar_range: None,
            analysis_tier: 1,
            timestamp_ms: (chunk as u64) * 15_000,
            llm_analysis: None,
        }
    }

    #[test]
    fn test_accumulate_moment() {
        let mut acc = SessionAccumulator::default();
        assert!(!acc.has_teaching_content());

        acc.accumulate_moment(make_moment(0, "dynamics", 0.3, false));
        assert!(acc.has_teaching_content());
        assert_eq!(acc.teaching_moments.len(), 1);
    }

    #[test]
    fn test_top_moments_dedup_by_dimension() {
        let mut acc = SessionAccumulator::default();
        // Three dynamics moments with different deviations
        acc.accumulate_moment(make_moment(0, "dynamics", 0.1, false));
        acc.accumulate_moment(make_moment(1, "dynamics", 0.5, false));
        acc.accumulate_moment(make_moment(2, "dynamics", 0.3, false));

        let top = acc.top_moments(None);
        // Should select only the highest |deviation| = 0.5 (chunk 1)
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].chunk_index, 1);
        assert!((top[0].deviation - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_top_moments_cap_at_8() {
        let mut acc = SessionAccumulator::default();
        let dims = [
            "dynamics",
            "timing",
            "pedaling",
            "articulation",
            "phrasing",
            "interpretation",
        ];
        // 2 moments per dimension (one negative, one positive) = 12 candidates -> capped at 8
        for (i, dim) in dims.iter().enumerate() {
            acc.accumulate_moment(make_moment(i * 2, dim, -0.4, false));
            acc.accumulate_moment(make_moment(i * 2 + 1, dim, 0.3, true));
        }

        let top = acc.top_moments(None);
        assert!(top.len() <= 8);
    }

    #[test]
    fn test_top_moments_sorted_chronologically() {
        let mut acc = SessionAccumulator::default();
        // Add moments out of order across different dimensions
        acc.accumulate_moment(make_moment(5, "phrasing", 0.4, false));
        acc.accumulate_moment(make_moment(2, "dynamics", 0.4, false));
        acc.accumulate_moment(make_moment(8, "timing", 0.4, false));

        let top = acc.top_moments(None);
        let indices: Vec<usize> = top.iter().map(|m| m.chunk_index).collect();
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(
            indices, sorted,
            "top_moments should be sorted by chunk_index"
        );
    }

    #[test]
    fn test_accumulate_mode_transition() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_mode_transition(ModeTransitionRecord {
            from: PracticeMode::Warming,
            to: PracticeMode::Running,
            chunk_index: 3,
            timestamp_ms: 45_000,
            dwell_ms: 45_000,
        });
        assert_eq!(acc.mode_transitions.len(), 1);
        assert_eq!(acc.mode_transitions[0].from, PracticeMode::Warming);
        assert_eq!(acc.mode_transitions[0].to, PracticeMode::Running);
        assert_eq!(acc.mode_transitions[0].chunk_index, 3);
    }

    #[test]
    fn test_accumulate_drilling_record() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_drilling_record(DrillingRecord {
            bar_range: Some((5, 10)),
            repetition_count: 4,
            first_scores: [0.4, 0.5, 0.3, 0.6, 0.5, 0.4],
            final_scores: [0.6, 0.7, 0.5, 0.7, 0.6, 0.6],
            started_at_chunk: 2,
            ended_at_chunk: 6,
        });
        assert_eq!(acc.drilling_records.len(), 1);
        assert_eq!(acc.drilling_records[0].repetition_count, 4);
        assert_eq!(acc.drilling_records[0].bar_range, Some((5, 10)));
    }

    #[test]
    fn test_serde_round_trip() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_moment(make_moment(0, "dynamics", 0.3, false));
        acc.accumulate_mode_transition(ModeTransitionRecord {
            from: PracticeMode::Warming,
            to: PracticeMode::Drilling,
            chunk_index: 1,
            timestamp_ms: 15_000,
            dwell_ms: 15_000,
        });
        acc.accumulate_drilling_record(DrillingRecord {
            bar_range: Some((1, 4)),
            repetition_count: 3,
            first_scores: [0.5; 6],
            final_scores: [0.7; 6],
            started_at_chunk: 1,
            ended_at_chunk: 4,
        });
        acc.accumulate_timeline_event(TimelineEvent {
            chunk_index: 0,
            timestamp_ms: 0,
            has_audio: true,
        });

        let json = serde_json::to_string(&acc).expect("serialization failed");
        let restored: SessionAccumulator =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(restored.teaching_moments.len(), acc.teaching_moments.len());
        assert_eq!(restored.mode_transitions.len(), acc.mode_transitions.len());
        assert_eq!(restored.drilling_records.len(), acc.drilling_records.len());
        assert_eq!(restored.timeline.len(), acc.timeline.len());

        let orig_moment = &acc.teaching_moments[0];
        let rest_moment = &restored.teaching_moments[0];
        assert_eq!(rest_moment.chunk_index, orig_moment.chunk_index);
        assert_eq!(rest_moment.dimension, orig_moment.dimension);
        assert!((rest_moment.deviation - orig_moment.deviation).abs() < f64::EPSILON);

        let orig_trans = &acc.mode_transitions[0];
        let rest_trans = &restored.mode_transitions[0];
        assert_eq!(rest_trans.from, orig_trans.from);
        assert_eq!(rest_trans.to, orig_trans.to);
    }

    #[test]
    fn test_top_moments_none_weights_unchanged() {
        let mut acc = SessionAccumulator::default();
        acc.accumulate_moment(make_moment(0, "dynamics", -0.4, false));
        acc.accumulate_moment(make_moment(1, "timing", 0.3, true));
        acc.accumulate_moment(make_moment(2, "pedaling", -0.5, false));

        let top = acc.top_moments(None);
        assert_eq!(top.len(), 3);
        // Should be sorted by chunk_index
        assert_eq!(top[0].chunk_index, 0);
        assert_eq!(top[1].chunk_index, 1);
        assert_eq!(top[2].chunk_index, 2);
    }

    #[test]
    fn test_top_moments_with_weights() {
        let mut acc = SessionAccumulator::default();
        // 5 dimensions with 2 moments each = 10 candidates -> capped at 8
        acc.accumulate_moment(make_moment(0, "dynamics", -0.4, false));
        acc.accumulate_moment(make_moment(1, "dynamics", 0.2, true));
        acc.accumulate_moment(make_moment(2, "timing", -0.3, false));
        acc.accumulate_moment(make_moment(3, "timing", 0.2, true));
        acc.accumulate_moment(make_moment(4, "pedaling", -0.3, false));
        acc.accumulate_moment(make_moment(5, "pedaling", 0.2, true));
        acc.accumulate_moment(make_moment(6, "articulation", -0.3, false));
        acc.accumulate_moment(make_moment(7, "articulation", 0.2, true));
        acc.accumulate_moment(make_moment(8, "phrasing", -0.3, false));
        acc.accumulate_moment(make_moment(9, "phrasing", 0.2, true));

        // Without weights: 10 candidates, capped to 8
        let top_unweighted = acc.top_moments(None);
        assert!(top_unweighted.len() <= 8);

        // With high pedaling weight: pedaling moments should survive the cap
        let mut weights = HashMap::new();
        weights.insert("pedaling".to_string(), 3.0);
        let top_weighted = acc.top_moments(Some(&weights));
        assert!(top_weighted.len() <= 8);
        // Pedaling moments should be present
        assert!(top_weighted.iter().any(|m| m.dimension == "pedaling"));
    }
}
