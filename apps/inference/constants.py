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

# MAESTRO calibration stats not yet computed for 6-dim A1-Max model.
# Run MAESTRO zero-shot evaluation to populate.
MAESTRO_CALIBRATION = {}
