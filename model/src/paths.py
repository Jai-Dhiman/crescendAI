"""Central data path definitions for the CrescendAI model pipeline.

All scripts should import paths from here instead of constructing them inline.
Paths resolve to directories (not files) unless noted otherwise.
"""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


# --- Raw datasets (downloaded, gitignored) ---
class Raw:
    root = DATA_ROOT / "raw"
    competition = root / "competition"
    masterclass = root / "masterclass"
    youtube = root / "youtube"
    maestro = root / "maestro"
    maestro_corrupted = root / "maestro_corrupted"
    room_irs = root / "room_irs"


# --- Embeddings (extracted, gitignored) ---
class Embeddings:
    root = DATA_ROOT / "embeddings"
    percepiano = root / "percepiano"
    maestro = root / "maestro"
    masterclass = root / "masterclass"
    competition = root / "competition"
    t5_muq = root / "t5_muq"
    t5_aria = root / "t5_aria"
    practice_corrupted = root / "practice_corrupted"


# --- MIDI (gitignored, regenerable) ---
class Midi:
    root = DATA_ROOT / "midi"
    percepiano = root / "percepiano"
    amt = root / "amt"


# --- LEGACY: Pretraining corpus (S2 GNN, replaced by Aria) ---
# Kept for import compatibility with old scripts. Data deleted.
class Pretraining:
    root = DATA_ROOT / "pretraining"
    tokens = root / "tokens"
    graphs = root / "graphs"
    features = root / "features"


# --- Labels and annotations (tracked) ---
class Labels:
    root = DATA_ROOT / "labels"
    composite = root / "composite"
    percepiano = root / "percepiano"
    stop_classifier_weights = root / "stop_classifier_weights.json"


# --- Data source manifests (tracked) ---
class Manifests:
    root = DATA_ROOT / "manifests"
    masterclass = root / "masterclass"
    youtube = root / "youtube"
    competition = root / "competition"


# --- Scores (deployed to R2, partially tracked) ---
class Scores:
    root = DATA_ROOT / "scores"


# --- Reference performance profiles (gitignored, deployed to R2) ---
class References:
    root = DATA_ROOT / "references"
    v1 = root / "v1"


# --- Evaluation data (partially tracked) ---
class Evals:
    root = DATA_ROOT / "evals"
    skill_eval = root / "skill_eval"
    inference_cache = root / "inference_cache"
    traces = root / "traces"
    youtube_amt = root / "youtube_amt"
    intermediate = root / "intermediate"
    ood_practice = root / "ood_practice"


# --- Train/val/test splits (tracked) ---
class Splits:
    root = DATA_ROOT / "splits"


# --- Outputs ---
class Checkpoints:
    root = DATA_ROOT / "checkpoints"
    model_improvement = root / "model_improvement"


class Results:
    root = DATA_ROOT / "results"


class Calibration:
    root = DATA_ROOT / "calibration"


class Weights:
    root = DATA_ROOT / "weights"
