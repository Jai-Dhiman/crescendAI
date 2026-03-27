//! STOP classifier: determines whether a scored audio chunk contains a teaching-worthy moment.
//!
//! Implements Option B from the pipeline design (02-pipeline.md):
//! logistic regression on 6-dim composite scores, running in the cloud worker (Rust/WASM). LOVO CV AUC = 0.649 (balanced).
//!
//! Weights extracted from sklearn LogisticRegression trained on 1,699 labeled masterclass segments with class_weight='balanced'.
//! Coefficient sign consistency = 1.0 across all 60 LOVO folds.
//!
//! The StandardScaler parameters (mean, std) are required because the logistic regression was trained on standardized features.

use crate::practice::dims::DIMS_6;

/// NOTE: These constants are also defined in config/stop_config.json (single source of truth).
/// The Rust side reads from hardcoded values because WASM builds cannot read filesystem at compile time.
/// If you update these values, update config/stop_config.json as well.
///
/// Dimension order must match the training pipeline.
/// [dynamics, timing, pedaling, articulation, phrasing, interpretation]
pub const SCALER_MEAN: [f64; 6] = [0.5450, 0.4848, 0.4594, 0.5369, 0.5188, 0.5064];
const SCALER_STD: [f64; 6] = [0.0689, 0.0388, 0.0791, 0.0154, 0.0186, 0.0555];

/// Balanced logistic regression coefficients.
/// Interpretation of signs:
/// - Negative (dynamics, pedaling, interpretation): lower score -> more likely to stop.
///   These are the "musicality" dimensions where poor performance triggers intervention.
/// - Positive (timing, articulation, phrasing): higher score -> more likely to stop.
///   Teachers stop students who play technically well but lack musical expression.
const WEIGHTS: [f64; 6] = [-0.5266, 0.3681, -0.5483, 0.4884, 0.2427, -0.1541];
const BIAS: f64 = 0.1147;

pub const DEFAULT_STOP_THRESHOLD: f64 = 0.5;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StopResult {
    pub probability: f64,
    pub triggered: bool,
    pub top_dimension: String,
    pub top_deviation: f64,
}

/// Compute STOP probability for a single chunk's 6-dim scores.
/// Applies StandardScaler normalization, then logistic regression.
/// Input order: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
pub fn stop_probability(scores: &[f64; 6]) -> f64 {
    let logit: f64 = scores
        .iter()
        .zip(SCALER_MEAN.iter())
        .zip(SCALER_STD.iter())
        .zip(WEIGHTS.iter())
        .map(|(((s, mean), std), w)| {
            let scaled = (s - mean) / std;
            scaled * w
        })
        .sum::<f64>()
        + BIAS;

    sigmoid(logit)
}

/// Classify a chunk: compute STOP probability and identify the top contributing dimension.
pub fn classify(scores: &[f64; 6]) -> StopResult {
    let prob = stop_probability(scores);

    // Find the dimension that contributed most to the STOP probability.
    // Contribution = |scaled_score * weight| (absolute magnitude).
    let mut top_idx = 0;
    let mut top_contribution = 0.0f64;

    for i in 0..6 {
        let scaled = (scores[i] - SCALER_MEAN[i]) / SCALER_STD[i];
        let contribution = (scaled * WEIGHTS[i]).abs();
        if contribution > top_contribution {
            top_contribution = contribution;
            top_idx = i;
        }
    }

    // Deviation: how far this score is from the training mean, in std units
    let top_deviation = (scores[top_idx] - SCALER_MEAN[top_idx]) / SCALER_STD[top_idx];

    StopResult {
        probability: prob,
        triggered: prob >= DEFAULT_STOP_THRESHOLD,
        top_dimension: DIMS_6[top_idx].to_string(),
        top_deviation,
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_scores_low_probability() {
        // Scores at the training mean should produce probability near sigmoid(BIAS).
        // BIAS = 0.1147, sigmoid(0.1147) ~ 0.529.
        // With balanced weights and scores at mean, all scaled features are 0,
        // so logit = BIAS only.
        let scores = SCALER_MEAN;
        let prob = stop_probability(&scores);
        let expected = sigmoid(BIAS);
        assert!(
            (prob - expected).abs() < 1e-6,
            "At mean scores, probability should be sigmoid(bias) = {:.4}, got {:.4}",
            expected,
            prob,
        );
    }

    #[test]
    fn low_dynamics_high_probability() {
        // Dynamics has negative weight (-0.53): low dynamics -> high STOP.
        // Set dynamics to 0.1 (very low), rest at mean.
        let mut scores = SCALER_MEAN;
        scores[0] = 0.1; // dynamics far below mean
        let result = classify(&scores);
        assert!(
            result.probability > 0.7,
            "Very low dynamics should trigger high STOP probability, got {:.4}",
            result.probability,
        );
        assert!(result.triggered);
        assert_eq!(result.top_dimension, "dynamics");
    }

    #[test]
    fn low_pedaling_high_probability() {
        // Pedaling has the strongest negative weight (-0.55).
        let mut scores = SCALER_MEAN;
        scores[2] = 0.1; // pedaling far below mean
        let result = classify(&scores);
        assert!(
            result.probability > 0.7,
            "Very low pedaling should trigger high STOP probability, got {:.4}",
            result.probability,
        );
        assert!(result.triggered);
        assert_eq!(result.top_dimension, "pedaling");
    }

    #[test]
    fn all_high_scores_moderate_probability() {
        // All scores high (0.9): positive-weight dims push toward stop,
        // negative-weight dims push away. Net effect depends on weight balance.
        let scores = [0.9; 6];
        let prob = stop_probability(&scores);
        // The result should be defined -- no panics, no NaN.
        assert!(prob.is_finite());
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn known_weight_combination() {
        // Manually compute expected value for a known input.
        let scores = [0.3, 0.5, 0.2, 0.55, 0.53, 0.4];

        // Manual calculation:
        let mut logit = BIAS;
        for i in 0..6 {
            let scaled = (scores[i] - SCALER_MEAN[i]) / SCALER_STD[i];
            logit += scaled * WEIGHTS[i];
        }
        let expected = sigmoid(logit);

        let prob = stop_probability(&scores);
        assert!(
            (prob - expected).abs() < 1e-10,
            "Expected {:.6}, got {:.6}",
            expected,
            prob,
        );
    }

    #[test]
    fn classify_returns_correct_top_dimension() {
        // Only pedaling is extreme, rest at mean.
        let mut scores = SCALER_MEAN;
        scores[2] = 0.05; // pedaling extremely low
        let result = classify(&scores);
        assert_eq!(result.top_dimension, "pedaling");
        // Deviation should be negative (below mean)
        assert!(result.top_deviation < -2.0);
    }

    #[test]
    fn threshold_boundary() {
        // Scores at training mean -> probability ~ 0.529 (just above 0.5).
        let scores = SCALER_MEAN;
        let result = classify(&scores);
        // With balanced bias 0.1147, sigmoid gives ~0.529, which is above threshold.
        assert!(result.triggered);
    }
}
