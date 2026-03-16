"""Constants for A1-Max MuQ LoRA inference handler."""

PERCEPIANO_DIMENSIONS = [
    "dynamics",
    "timing",
    "pedaling",
    "articulation",
    "phrasing",
    "interpretation",
]

# A1-Max model configuration
# MuQ embeddings (1024 dim) with attention pooling -> encoder -> regression head
MODEL_CONFIG = {
    # MuQ configuration (layers to average)
    "muq_layer_start": 9,
    "muq_layer_end": 13,  # Exclusive (layers 9, 10, 11, 12)
    "muq_dim": 1024,  # Per-layer hidden size (= input_dim)
    # Head configuration
    "input_dim": 1024,
    "hidden_dim": 512,
    "num_labels": 6,
    "dropout": 0.2,
    # Audio processing
    "target_sr": 24000,
    "max_frames": 1000,
}

# Model info for response
MODEL_INFO = {
    "name": "A1-Max MuQ LoRA",
    "type": "audio-muq-lora",
    "pairwise": 0.7872,
    "description": "A1-Max: MuQ + LoRA with ListMLE, CCC, mixup, hard negative mining",
    "architecture": "MuQLoRAMaxModel (MuQ L9-12 avg -> attn pool -> encoder -> 6-dim regression)",
    "best_config": "A1max_r32_L7-12_ls0.1",
}

# Number of folds for ensemble
N_FOLDS = 4

# MAESTRO calibration stats: per-dimension distribution over 24,321 professional segments.
# Computed by model/scripts/compute_maestro_calibration.py using A1-Max 4-fold ensemble.
MAESTRO_CALIBRATION = {
    "dynamics": {
        "mean": 0.560947,
        "std": 0.021063,
        "p5": 0.526612,
        "p25": 0.546136,
        "p50": 0.560859,
        "p75": 0.575372,
        "p95": 0.59573,
    },
    "timing": {
        "mean": 0.531883,
        "std": 0.028791,
        "p5": 0.480467,
        "p25": 0.512976,
        "p50": 0.534302,
        "p75": 0.552652,
        "p95": 0.575376,
    },
    "pedaling": {
        "mean": 0.590465,
        "std": 0.030438,
        "p5": 0.534399,
        "p25": 0.572243,
        "p50": 0.593731,
        "p75": 0.611854,
        "p95": 0.635053,
    },
    "articulation": {
        "mean": 0.553624,
        "std": 0.014287,
        "p5": 0.53023,
        "p25": 0.543792,
        "p50": 0.553554,
        "p75": 0.563275,
        "p95": 0.577426,
    },
    "phrasing": {
        "mean": 0.550866,
        "std": 0.013717,
        "p5": 0.528466,
        "p25": 0.541541,
        "p50": 0.550801,
        "p75": 0.560116,
        "p95": 0.573567,
    },
    "interpretation": {
        "mean": 0.564377,
        "std": 0.023457,
        "p5": 0.522434,
        "p25": 0.549302,
        "p50": 0.566195,
        "p75": 0.580981,
        "p95": 0.599733,
    },
}
