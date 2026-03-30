"""Approach B: YAMNet audio event classifier.

Uses Google's YAMNet (MobileNet v1 on AudioSet, 521 classes).
Has explicit "Piano" class (index 345).
Same weights as the TF.js browser version.

NOTE: Python latency is for relative comparison only.
Browser TF.js performance will differ.
"""

from __future__ import annotations

import numpy as np


# YAMNet class indices for piano-related sounds (from AudioSet ontology)
PIANO_CLASSES = {
    "Piano": 148,
    "Electric piano": 149,
    "Keyboard (musical)": 147,
}

# Classes that indicate "music" broadly (fallback)
MUSIC_CLASSES = {
    "Music": 132,
    "Musical instrument": 133,
    "Classical music": 232,
}


def _load_yamnet():
    """Load YAMNet model. Lazy import to avoid TF startup cost."""
    try:
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        return model
    except ImportError:
        raise ImportError(
            "YAMNet requires tensorflow and tensorflow_hub. "
            "Install with: uv pip install tensorflow tensorflow-hub"
        )


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = _load_yamnet()
    return _model


def compute_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Run YAMNet and return piano-related class scores.

    YAMNet expects 16kHz mono audio.
    """
    import librosa

    # Resample to 16kHz if needed (YAMNet requirement)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Ensure float32 in [-1, 1]
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))

    model = _get_model()
    scores, embeddings, spectrogram = model(audio)
    scores = scores.numpy()  # (frames, 521)

    # Average across frames
    mean_scores = np.mean(scores, axis=0)

    # Get piano-related scores
    piano_score = max(mean_scores[idx] for idx in PIANO_CLASSES.values())
    music_score = max(mean_scores[idx] for idx in MUSIC_CLASSES.values())

    # Get top class
    top_idx = int(np.argmax(mean_scores))

    return {
        "piano_score": float(piano_score),
        "music_score": float(music_score),
        "top_class_idx": top_idx,
        "top_class_score": float(mean_scores[top_idx]),
    }


def classify(
    features: dict[str, float],
    piano_threshold: float = 0.3,
    music_threshold: float = 0.4,
) -> tuple[str, float]:
    """Classify based on YAMNet piano/music scores.

    Primary gate: piano class score above threshold.
    Fallback: music class score above threshold (catches piano-like sounds
    that YAMNet labels as generic "Music").
    """
    piano_score = features["piano_score"]
    music_score = features["music_score"]

    if piano_score >= piano_threshold:
        return "piano", piano_score
    elif music_score >= music_threshold:
        return "piano", music_score * 0.8  # lower confidence for generic music
    else:
        confidence = 1.0 - max(piano_score, music_score)
        return "not_piano", confidence


def sweep_thresholds(
    all_features: list[dict],
) -> list[dict]:
    """Sweep piano score thresholds."""
    thresholds = np.arange(0.05, 0.60, 0.05)
    results = []
    for t in thresholds:
        preds = [classify(f, piano_threshold=t)[0] for f in all_features]
        results.append({"threshold": float(t), "predictions": preds})
    return results
