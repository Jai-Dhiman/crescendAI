"""Central data path definitions for the CrescendAI model pipeline.

All scripts should import paths from here instead of constructing them inline.
Paths resolve to directories (not files) unless noted otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
REPO_ROOT = DATA_ROOT.parent


def ensure_local(path: Path | str) -> Path:
    """Assert that a data path exists locally, raising a rehydrate hint if not.

    If the path is listed in data/manifests/r2_offload.json as offloaded and
    missing locally, raises FileNotFoundError with the exact rclone command
    to pull it back. Silent auto-download is deliberately not implemented so
    that stale-cache bugs surface immediately.
    """
    p = Path(path)
    if p.exists():
        return p

    manifest_path = DATA_ROOT / "manifests" / "r2_offload.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{p} missing and no r2_offload manifest found")

    with manifest_path.open() as f:
        manifest = json.load(f)

    try:
        rel = str(p.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(p)

    entry = manifest.get("entries", {}).get(rel)
    if entry is None:
        raise FileNotFoundError(f"{p} missing and not registered in r2_offload.json")

    bucket = manifest["bucket"]
    remote = manifest["remote_name"]
    reason = entry.get("reason", "")
    if "r2_prefix" in entry:
        cmd = f"rclone copy {remote}:{bucket}/{entry['r2_prefix']} {rel}"
    elif "regen_command" in entry:
        cmd = entry["regen_command"]
    else:
        raise FileNotFoundError(f"{p} registered but lacks r2_prefix or regen_command")

    raise FileNotFoundError(
        f"{p} is offloaded ({reason}). Rehydrate with:\n    {cmd}"
    )


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
