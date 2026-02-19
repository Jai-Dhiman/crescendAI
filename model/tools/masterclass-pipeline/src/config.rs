use std::collections::HashMap;

pub struct DimensionInfo {
    pub percepiano_dims: &'static [&'static str],
    pub keywords: &'static [&'static str],
}

pub fn teaching_dimensions() -> HashMap<&'static str, DimensionInfo> {
    let mut m = HashMap::new();

    m.insert(
        "dynamics",
        DimensionInfo {
            percepiano_dims: &["Dynamics", "Dynamic Range", "Dynamic Consistency"],
            keywords: &[
                "loud", "soft", "piano", "forte", "crescendo", "diminuendo",
                "fortissimo", "pianissimo", "mezzo", "sforzando", "volume",
                "dynamic", "power", "gentle", "whisper",
            ],
        },
    );

    m.insert(
        "timing",
        DimensionInfo {
            percepiano_dims: &["Tempo Stability", "Rhythmic Accuracy", "Tempo Choice"],
            keywords: &[
                "tempo", "rhythm", "rubato", "accelerando", "ritardando",
                "rushing", "dragging", "timing", "pulse", "beat", "meter",
                "time", "speed", "slow", "fast", "steady", "rush",
            ],
        },
    );

    m.insert(
        "articulation",
        DimensionInfo {
            percepiano_dims: &["Articulation", "Note Accuracy"],
            keywords: &[
                "legato", "staccato", "tenuto", "marcato", "portato",
                "accent", "slur", "detached", "connected", "crisp",
                "smooth", "separation", "attack",
            ],
        },
    );

    m.insert(
        "pedaling",
        DimensionInfo {
            percepiano_dims: &["Pedal Use"],
            keywords: &[
                "pedal", "sustain", "damper", "una corda", "sostenuto",
                "half pedal", "flutter pedal", "pedaling", "muddy", "blur",
                "clean", "wash",
            ],
        },
    );

    m.insert(
        "tone_color",
        DimensionInfo {
            percepiano_dims: &["Timbre", "Tone Quality", "Sound Quality"],
            keywords: &[
                "tone", "color", "timbre", "sound", "bright", "dark",
                "warm", "cold", "rich", "thin", "singing", "bell",
                "round", "harsh", "beautiful", "quality",
            ],
        },
    );

    m.insert(
        "phrasing",
        DimensionInfo {
            percepiano_dims: &["Phrasing", "Musical Flow"],
            keywords: &[
                "phrase", "line", "breath", "shape", "direction", "arc",
                "contour", "flow", "sentence", "gesture", "long line",
                "singing", "melodic",
            ],
        },
    );

    m.insert(
        "voicing",
        DimensionInfo {
            percepiano_dims: &["Voicing", "Balance"],
            keywords: &[
                "voice", "voicing", "balance", "melody", "accompaniment",
                "inner voice", "soprano", "bass", "top", "bottom",
                "bring out", "highlight", "layer", "texture",
            ],
        },
    );

    m.insert(
        "interpretation",
        DimensionInfo {
            percepiano_dims: &["Emotional Expression", "Stylistic Accuracy", "Overall Impression"],
            keywords: &[
                "interpretation", "expression", "emotion", "character",
                "mood", "style", "period", "composer", "intention",
                "meaning", "story", "feeling", "spirit", "musical",
            ],
        },
    );

    m.insert(
        "technique",
        DimensionInfo {
            percepiano_dims: &["Technique", "Evenness"],
            keywords: &[
                "technique", "finger", "hand", "wrist", "arm", "weight",
                "rotation", "position", "passage", "scale", "arpeggio",
                "octave", "trill", "jump", "leap", "coordination",
                "relaxation", "tension",
            ],
        },
    );

    m.insert(
        "structure",
        DimensionInfo {
            percepiano_dims: &["Form Awareness"],
            keywords: &[
                "structure", "form", "section", "development", "recapitulation",
                "exposition", "coda", "transition", "bridge", "theme",
                "variation", "contrast", "proportion", "architecture",
            ],
        },
    );

    m
}

pub const WHISPER_MODEL_URL_BASE: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

pub fn whisper_model_url(model_name: &str) -> String {
    format!("{}/ggml-{}.bin", WHISPER_MODEL_URL_BASE, model_name)
}

#[allow(dead_code)]
pub const DEFAULT_WHISPER_MODEL: &str = "large-v3";
#[allow(dead_code)]
pub const DEFAULT_LLM_MODEL: &str = "qwen2.5:32b";

pub const CHUNK_DURATION_SECS: f64 = 300.0;
pub const CHUNK_OVERLAP_SECS: f64 = 30.0;
pub const REPETITION_THRESHOLD: usize = 3;
pub const SIMILARITY_THRESHOLD: f64 = 0.8;

pub const MIN_VIDEO_DURATION_SECS: f64 = 600.0;
pub const MAX_VIDEO_DURATION_SECS: f64 = 7200.0;

pub const MASTERCLASS_KEYWORDS: &[&str] = &[
    "masterclass",
    "master class",
    "lesson",
    "teaches",
    "coaching",
    "piano class",
];

pub const SAMPLE_RATE: u32 = 16000;
pub const FFT_SIZE: usize = 2048;
pub const HOP_SIZE: usize = 512;

pub const MOMENT_DEDUP_THRESHOLD_SECS: f64 = 10.0;

pub const VALID_DIMENSIONS: &[&str] = &[
    "dynamics",
    "timing",
    "articulation",
    "pedaling",
    "tone_color",
    "phrasing",
    "voicing",
    "interpretation",
    "technique",
    "structure",
];

/// Normalize a musical dimension string returned by the LLM.
///
/// Handles case differences, whitespace, and common aliases.
/// Returns "interpretation" with a warning for unknown values.
pub fn normalize_dimension(raw: &str) -> String {
    let normalized = raw.trim().to_lowercase().replace(' ', "_");

    if VALID_DIMENSIONS.contains(&normalized.as_str()) {
        return normalized;
    }

    // Common aliases
    let mapped = match normalized.as_str() {
        "rhythm" | "tempo" | "rubato" => "timing",
        "pedal" | "pedals" => "pedaling",
        "tone" | "timbre" | "sound" | "tone_quality" | "sound_quality" => "tone_color",
        "volume" | "dynamic" | "dynamic_range" => "dynamics",
        "phrase" | "melodic_line" => "phrasing",
        "voice" | "balance" => "voicing",
        "expression" | "emotion" | "style" | "musical_expression" => "interpretation",
        "form" | "form_awareness" => "structure",
        "fingering" | "hand_position" => "technique",
        "legato" | "staccato" | "accent" | "accents" => "articulation",
        _ => {
            tracing::warn!(
                "Unknown musical dimension '{}', defaulting to 'interpretation'",
                raw
            );
            "interpretation"
        }
    };

    mapped.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_dimension_exact_match() {
        assert_eq!(normalize_dimension("dynamics"), "dynamics");
        assert_eq!(normalize_dimension("timing"), "timing");
        assert_eq!(normalize_dimension("tone_color"), "tone_color");
    }

    #[test]
    fn normalize_dimension_case_insensitive() {
        assert_eq!(normalize_dimension("Dynamics"), "dynamics");
        assert_eq!(normalize_dimension("TIMING"), "timing");
        assert_eq!(normalize_dimension("Tone_Color"), "tone_color");
    }

    #[test]
    fn normalize_dimension_whitespace_to_underscore() {
        assert_eq!(normalize_dimension("tone color"), "tone_color");
    }

    #[test]
    fn normalize_dimension_alias_mapping() {
        assert_eq!(normalize_dimension("rhythm"), "timing");
        assert_eq!(normalize_dimension("pedal"), "pedaling");
        assert_eq!(normalize_dimension("volume"), "dynamics");
        assert_eq!(normalize_dimension("timbre"), "tone_color");
        assert_eq!(normalize_dimension("expression"), "interpretation");
    }

    #[test]
    fn normalize_dimension_unknown_defaults() {
        assert_eq!(normalize_dimension("underwater_basket_weaving"), "interpretation");
    }

    #[test]
    fn normalize_dimension_trimmed() {
        assert_eq!(normalize_dimension("  dynamics  "), "dynamics");
    }
}
