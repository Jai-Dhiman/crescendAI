use std::collections::HashMap;

/// Running statistics tracker per dimension.
#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct DimStats {
    pub counts: HashMap<String, u32>,
    pub means: HashMap<String, f64>,
    pub m2s: HashMap<String, f64>, // Welford's running variance
}

impl DimStats {
    /// Update running stats with new scores (Welford's online algorithm).
    pub fn update(&mut self, scores: &HashMap<String, f64>) {
        for (dim, &value) in scores {
            let count = self.counts.entry(dim.clone()).or_insert(0);
            *count += 1;
            let n = *count;

            let mean = self.means.entry(dim.clone()).or_insert(0.0);
            let m2 = self.m2s.entry(dim.clone()).or_insert(0.0);

            let delta = value - *mean;
            *mean += delta / n as f64;
            let delta2 = value - *mean;
            *m2 += delta * delta2;
        }
    }

    fn stddev(&self, dim: &str) -> f64 {
        let count = self.counts.get(dim).copied().unwrap_or(0);
        if count < 2 { return f64::MAX }
        let m2 = self.m2s.get(dim).copied().unwrap_or(0.0);
        (m2 / (count - 1) as f64).sqrt()
    }

    fn mean(&self, dim: &str) -> f64 {
        self.means.get(dim).copied().unwrap_or(0.0)
    }

    /// Check if any dimension deviates > threshold stddevs from running mean.
    /// Returns the most deviant dimension and its z-score, if any exceed threshold.
    /// Requires at least `min_chunks` data points before flagging.
    pub fn detect_outlier(
        &self,
        scores: &HashMap<String, f64>,
        threshold: f64,
        min_chunks: u32,
    ) -> Option<(String, f64)> {
        let mut worst: Option<(String, f64)> = None;

        for (dim, &value) in scores {
            let count = self.counts.get(dim.as_str()).copied().unwrap_or(0);
            if count < min_chunks { continue; }

            let std = self.stddev(dim);
            if std == 0.0 || std == f64::MAX { continue; }

            let z = (value - self.mean(dim)).abs() / std;
            if z > threshold {
                match &worst {
                    None => worst = Some((dim.clone(), z)),
                    Some((_, wz)) if z > *wz => worst = Some((dim.clone(), z)),
                    _ => {}
                }
            }
        }

        worst
    }
}
