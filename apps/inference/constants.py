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

# Model info for response (audio-only model)
MODEL_INFO = {
    "name": "MERT-330M",
    "type": "audio",
    "r2": 0.487,
    "description": "Audio-only piano performance evaluation using MERT-330M embeddings",
}

# Checkpoint paths (relative to /app in container)
CHECKPOINT_PATHS = {
    "mert_folds": "/app/checkpoints/mert/fold{}_best.ckpt",
}

# Number of folds for ensemble
N_FOLDS = 4
