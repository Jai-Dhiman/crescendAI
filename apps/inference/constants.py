"""Constants for inference handler."""

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

# Model configuration (must match training config)
MERT_CONFIG = {
    "input_dim": 1024,
    "hidden_dim": 512,
    "num_labels": 19,
    "dropout": 0.2,
    "pooling": "mean",
    "layer_start": 13,
    "layer_end": 25,
    "target_sr": 24000,
    "max_frames": 1000,
}

# Model info for response
MODEL_INFO = {
    "symbolic": {"name": "PercePiano", "r2": 0.395},
    "audio": {"name": "MERT-330M", "r2": 0.433},
    "fusion": {"name": "Late Fusion", "r2": 0.510},
}

# Checkpoint paths (relative to /app in container)
CHECKPOINT_PATHS = {
    "mert_folds": "/app/checkpoints/mert/fold{}_best.ckpt",
    "percepiano_folds": "/app/checkpoints/percepiano/fold{}_best.pt",
}

# Number of folds for ensemble
N_FOLDS = 4
