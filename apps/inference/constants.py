"""Constants for D9c AsymmetricGatedFusion inference handler."""

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

# D9c AsymmetricGatedFusion configuration
# MERT uses averaged layers 13-24 (1024 dim)
# MuQ uses last hidden state (1024)
MODEL_CONFIG = {
    # MERT configuration (layers to average)
    "mert_layer_start": 13,
    "mert_layer_end": 25,  # Exclusive
    "mert_dim": 1024,  # Averaged layer dimension
    "mert_hidden": 512,
    # MuQ configuration
    "muq_dim": 1024,  # Last hidden state
    # Fusion configuration
    "shared_dim": 512,
    "num_labels": 19,
    "dropout": 0.2,
    "pooling": "attention",
    # Audio processing
    "target_sr": 24000,
    "max_frames": 1000,
}

# Model info for response
MODEL_INFO = {
    "name": "D9c-AsymmetricGatedFusion",
    "type": "audio-dual-model",
    "r2": 0.531,
    "best_fold_r2": 0.587,
    "description": "MERT+MuQ fusion with per-dimension gating for piano performance evaluation",
    "architecture": "AsymmetricGatedFusion (MERT 6144->768->512, MuQ 1024->512, per-dim gates)",
}

# Number of folds for ensemble
N_FOLDS = 4
