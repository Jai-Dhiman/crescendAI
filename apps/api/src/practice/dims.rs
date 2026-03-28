use std::collections::HashMap;

/// The 6 teacher-grounded dimensions.
pub const DIMS_6: [&str; 6] = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
];

/// Map 19 `PerCePiano` dimensions to 6 teacher-grounded dimensions.
/// Each output dimension is the mean of its constituent raw dimensions.
#[allow(clippy::implicit_hasher)] // concrete HashMap is fine here
pub fn map_19_to_6(raw: &HashMap<String, f64>) -> HashMap<String, f64> {
    let avg = |keys: &[&str]| -> f64 {
        let (sum, count) = keys
            .iter()
            .fold((0.0, 0u32), |(s, c), k| match raw.get(*k) {
                Some(v) => (s + v, c + 1),
                None => (s, c),
            });
        if count == 0 {
            0.0
        } else {
            sum / f64::from(count)
        }
    };

    let mut mapped = HashMap::new();
    mapped.insert(
        "dynamics".to_string(),
        avg(&["dynamics_range", "timbre_loudness"]),
    );
    mapped.insert("timing".to_string(), avg(&["timing", "tempo"]));
    mapped.insert(
        "pedaling".to_string(),
        avg(&["pedal_amount", "pedal_clarity"]),
    );
    mapped.insert(
        "articulation".to_string(),
        avg(&["articulation_length", "articulation_touch"]),
    );
    mapped.insert("phrasing".to_string(), avg(&["space", "balance", "drama"]));
    mapped.insert(
        "interpretation".to_string(),
        avg(&[
            "timbre_variety",
            "timbre_depth",
            "timbre_brightness",
            "mood_valence",
            "mood_energy",
            "mood_imagination",
            "interpretation_sophistication",
            "interpretation_overall",
        ]),
    );
    mapped
}
