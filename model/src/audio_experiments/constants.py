"""Constants and configuration for audio experiments."""

SEED = 42

PERCEPIANO_DIMENSIONS = [
    "timing", "articulation_length", "articulation_touch",
    "pedal_amount", "pedal_clarity",
    "timbre_variety", "timbre_depth", "timbre_brightness", "timbre_loudness",
    "dynamic_range", "tempo", "space", "balance", "drama",
    "mood_valence", "mood_energy", "mood_imagination",
    "sophistication", "interpretation",
]

DIMENSION_CATEGORIES = {
    "timing": ["timing"],
    "articulation": ["articulation_length", "articulation_touch"],
    "pedal": ["pedal_amount", "pedal_clarity"],
    "timbre": ["timbre_variety", "timbre_depth", "timbre_brightness", "timbre_loudness"],
    "dynamics": ["dynamic_range"],
    "tempo_space": ["tempo", "space", "balance", "drama"],
    "emotion": ["mood_valence", "mood_energy", "mood_imagination"],
    "interpretation": ["sophistication", "interpretation"],
}

BASE_CONFIG = {
    "input_dim": 1024,
    "hidden_dim": 512,
    "num_labels": 19,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip_val": 1.0,
    "batch_size": 64,
    "max_epochs": 200,
    "patience": 15,
    "max_frames": 1000,
    "n_folds": 4,
    "num_workers": 2,
    "seed": SEED,
}
