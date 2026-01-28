"""Constants for M1c MuQ L9-12 inference handler."""

PERCEPIANO_DIMENSIONS = [
    "timing",
    "articulation_length",
    "articulation_touch",
    "pedal_amount",
    "pedal_clarity",
    "timbre_variety",
    "timbre_depth",
    "timbre_brightness",
    "timbre_loudness",
    "dynamics_range",
    "tempo",
    "space",
    "balance",
    "drama",
    "mood_valence",
    "mood_energy",
    "mood_imagination",
    "interpretation_sophistication",
    "interpretation_overall",
]

# M1c MuQ L9-12 configuration
# MuQ uses averaged layers 9-12 (1024 dim) with stats pooling (mean+std -> 2048)
MODEL_CONFIG = {
    # MuQ configuration (layers to average)
    "muq_layer_start": 9,
    "muq_layer_end": 13,  # Exclusive (layers 9, 10, 11, 12)
    "muq_dim": 1024,  # Per-layer hidden size
    "pooled_dim": 2048,  # After mean+std pooling
    # Head configuration
    "hidden_dim": 512,
    "num_labels": 19,
    "dropout": 0.2,
    "pooling_stats": "mean_std",
    # Audio processing
    "target_sr": 24000,
    "max_frames": 1000,
}

# Model info for response
MODEL_INFO = {
    "name": "MuQ L9-12",
    "type": "audio-muq-only",
    "r2": 0.537,
    "best_fold_r2": 0.586,
    "description": "MuQ layers 9-12 with Pianoteq ensemble for piano performance evaluation",
    "architecture": "MuQStatsModel (MuQ L9-12 avg -> mean+std -> 2048 -> 512 -> 19)",
}

# Number of folds for ensemble
N_FOLDS = 3

# MAESTRO zero-shot evaluation statistics (n=500 professional recordings)
# Used for calibrating raw predictions relative to professional performance benchmarks
# Source: C3_maestro_transfer experiment from paper validation
MAESTRO_CALIBRATION = {
    "timing": {
        "mean": 0.691,
        "std": 0.094,
        "p5": 0.537,  # 5th percentile (approximated from mean - 1.645*std)
        "p95": 0.846,  # 95th percentile (approximated from mean + 1.645*std)
    },
    "articulation_length": {
        "mean": 0.686,
        "std": 0.105,
        "p5": 0.513,
        "p95": 0.858,
    },
    "articulation_touch": {
        "mean": 0.459,
        "std": 0.100,
        "p5": 0.294,
        "p95": 0.623,
    },
    "pedal_amount": {
        "mean": 0.772,
        "std": 0.088,
        "p5": 0.627,
        "p95": 0.917,
    },
    "pedal_clarity": {
        "mean": 0.637,
        "std": 0.078,
        "p5": 0.509,
        "p95": 0.765,
    },
    "timbre_variety": {
        "mean": 0.656,
        "std": 0.050,
        "p5": 0.574,
        "p95": 0.739,
    },
    "timbre_depth": {
        "mean": 0.727,
        "std": 0.061,
        "p5": 0.627,
        "p95": 0.828,
    },
    "timbre_brightness": {
        "mean": 0.548,
        "std": 0.054,
        "p5": 0.459,
        "p95": 0.637,
    },
    "timbre_loudness": {
        "mean": 0.495,
        "std": 0.137,
        "p5": 0.269,
        "p95": 0.721,
    },
    "dynamics_range": {
        "mean": 0.416,
        "std": 0.098,
        "p5": 0.254,
        "p95": 0.578,
    },
    "tempo": {
        "mean": 0.594,
        "std": 0.071,
        "p5": 0.478,
        "p95": 0.711,
    },
    "space": {
        "mean": 0.566,
        "std": 0.115,
        "p5": 0.377,
        "p95": 0.756,
    },
    "balance": {
        "mean": 0.694,
        "std": 0.056,
        "p5": 0.601,
        "p95": 0.786,
    },
    "drama": {
        "mean": 0.687,
        "std": 0.059,
        "p5": 0.590,
        "p95": 0.784,
    },
    "mood_valence": {
        "mean": 0.714,
        "std": 0.049,
        "p5": 0.634,
        "p95": 0.795,
    },
    "mood_energy": {
        "mean": 0.550,
        "std": 0.051,
        "p5": 0.466,
        "p95": 0.635,
    },
    "mood_imagination": {
        "mean": 0.598,
        "std": 0.117,
        "p5": 0.406,
        "p95": 0.790,
    },
    "interpretation_sophistication": {
        "mean": 0.692,
        "std": 0.083,
        "p5": 0.556,
        "p95": 0.829,
    },
    "interpretation_overall": {
        "mean": 0.658,
        "std": 0.059,
        "p5": 0.561,
        "p95": 0.754,
    },
}
