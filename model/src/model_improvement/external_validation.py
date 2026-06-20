"""External held-out validation loaders (issue #77).

Ingests external piano datasets as HELD-OUT eval sets — never training data —
to squash the C1 assumption (3-piece PercePiano generalization). Licenses are
verified before any use:

  - NeuroPiano  (anusfoil/NeuroPiano-data, MIT): 13-axis expert ratings (0-6),
    audio only. Gives held-out per-dim pairwise + skill-rank.
  - PianoVAM    (PianoVAM/PianoVAM_v1.0, CC-BY-NC-SA-4.0): audio + native
    Disklavier MIDI + skill tiers. Non-commercial; used for held-out eval only.
    Its native MIDI also enables the D4 check (Aria on GT-MIDI vs AMT-MIDI).

Blocked from this path:
  - PERiScoPe   (CC-BY-NC-SA-4.0): NonCommercial + ShareAlike copyleft is
    incompatible with a commercial product, and it carries no quality ratings,
    so it is not folded into TRAINING here.
  - PianoCoRe   (license unconfirmed per #77): do not use.

The audio-bearing sets require MuQ embedding extraction before scoring; the
loaders produce keys + labels + skill, and run_external_eval scores a model
given an embeddings dir (mirrors OODDataset / evaluate_model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from scipy.stats import spearmanr

from model_improvement.evaluation import evaluate_model
from model_improvement.taxonomy import DIMENSIONS, NUM_DIMS


# ---------------------------------------------------------------------------
# License gate — verified 2026-06 against the HF dataset cards
# ---------------------------------------------------------------------------


class LicenseError(RuntimeError):
    """Raised when a dataset is not license-cleared for this held-out path."""


# Allowed as HELD-OUT eval. commercial_ok flags whether the license permits
# commercial use (NC sets are eval-only research use, no redistribution).
ALLOWED_EXTERNAL = {
    "neuropiano": {
        "hf_id": "anusfoil/NeuroPiano-data",
        "spdx": "MIT",
        "use": "held_out_eval",
        "commercial_ok": True,
    },
    "pianovam": {
        "hf_id": "PianoVAM/PianoVAM_v1.0",
        "spdx": "CC-BY-NC-SA-4.0",
        "use": "held_out_eval",
        "commercial_ok": False,
    },
}

# Explicitly blocked from this path, with the reason.
BLOCKED_EXTERNAL = {
    "periscope": (
        "CC-BY-NC-SA-4.0",
        "NonCommercial + ShareAlike copyleft is incompatible with a commercial "
        "product; no quality ratings -> not folded into training here",
    ),
    "pianocore": (
        "unconfirmed",
        "license unconfirmed per issue #77 -> do not use",
    ),
}


def verify_license(name: str) -> dict:
    """Return the license record for an allowed external set, else raise.

    Raises:
        LicenseError: for a blocked or unknown dataset.
    """
    key = name.lower()
    if key in BLOCKED_EXTERNAL:
        spdx, reason = BLOCKED_EXTERNAL[key]
        raise LicenseError(f"{name} blocked ({spdx}): {reason}")
    if key not in ALLOWED_EXTERNAL:
        raise LicenseError(
            f"{name} is not a license-cleared external set; allowed: "
            f"{sorted(ALLOWED_EXTERNAL)}"
        )
    return ALLOWED_EXTERNAL[key]


# ---------------------------------------------------------------------------
# NeuroPiano 13-axis -> 6-dim mapping
# ---------------------------------------------------------------------------

# Verified against the real dataset schema (q_eng per question_id). Questions 1
# and 2 are free-text good/bad prompts (no rating axis) and are excluded. The
# remaining axes (q3..q13) are 0-6 ratings. NeuroPiano has no pedaling axis, so
# `pedaling` is left uncovered (NaN) rather than fabricated.
NEUROPIANO_QUESTION_TO_DIM: dict[int, str] = {
    3: "articulation",   # Is the legato even?
    4: "timing",         # Are the note values uniform?
    5: "dynamics",       # How solid is the sound? (tone / sonic core)
    6: "articulation",   # How clean is the attack?
    7: "dynamics",       # Are the left and right hands balanced?
    8: "timing",         # Are the onset timings aligned across hands?
    9: "timing",         # Is it played with the correct rhythm?
    10: "timing",        # Is the tempo kept constant?
    11: "phrasing",      # Are the lines connected?
    12: "interpretation",  # Is it played with a sense of tonality?
    13: "dynamics",      # Is the dynamics change natural?
}

NEUROPIANO_SCORE_MAX = 6.0


def map_neuropiano_axes_to_dims(
    question_scores: dict[int, float],
) -> list[float]:
    """Map a recording's {question_id: 0-6 score} to our 6-dim label vector.

    Covered dims are the mean of their contributing axes, normalized to [0, 1].
    Dims with no contributing axis (pedaling) are NaN (uncovered) so downstream
    per-dim pairwise excludes them instead of scoring a fabricated value.
    """
    buckets: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    for qid, dim in NEUROPIANO_QUESTION_TO_DIM.items():
        if qid in question_scores and question_scores[qid] is not None:
            buckets[dim].append(float(question_scores[qid]) / NEUROPIANO_SCORE_MAX)

    vec: list[float] = []
    for d in DIMENSIONS:
        vals = buckets[d]
        vec.append(float(np.mean(vals)) if vals else float("nan"))
    return vec


# ---------------------------------------------------------------------------
# Held-out dataset container
# ---------------------------------------------------------------------------


@dataclass
class ExternalEvalDataset:
    """A held-out external eval set (NOT in folds.json, NOT training).

    Args:
        name: license-cleared dataset name (verified on construction).
        keys: recording ids.
        labels: key -> 6-dim label vector (NaN for uncovered dims).
        skill: key -> continuous quality/skill reference for skill-rank.
        embeddings_dir: optional dir of {key}.pt MuQ frame tensors.
        native_midi: optional key -> native MIDI path (PianoVAM, for D4).
    """

    name: str
    keys: list[str]
    labels: dict[str, list[float]]
    skill: dict[str, float] = field(default_factory=dict)
    embeddings_dir: Optional[Path] = None
    native_midi: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        verify_license(self.name)  # fail loudly if not license-cleared

    def __len__(self) -> int:
        return len(self.keys)

    def skill_tiers(self) -> dict[str, int] | None:
        """Bucket the continuous skill into integer tiers for Cohen's d."""
        if not self.skill:
            return None
        vals = np.array([self.skill[k] for k in self.keys if k in self.skill])
        if len(vals) < 2 or vals.std() == 0:
            return None
        # Tertiles -> 3 tiers (low/mid/high).
        q1, q2 = np.percentile(vals, [33.3, 66.6])
        tiers = {}
        for k in self.keys:
            if k not in self.skill:
                continue
            v = self.skill[k]
            tiers[k] = 0 if v <= q1 else (1 if v <= q2 else 2)
        return tiers


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_parquet_rows(parquet_dir: Path, columns: list[str]) -> list[dict]:
    """Read selected columns from all parquet shards in a dir as row dicts.

    Raises:
        RuntimeError: if pyarrow is not installed (explicit, no fallback).
        FileNotFoundError: if no parquet shards are present.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read external dataset parquet shards "
            "(`uv add pyarrow` or `uv run --with pyarrow ...`)."
        ) from exc

    shards = sorted(Path(parquet_dir).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {parquet_dir}")

    rows: list[dict] = []
    for shard in shards:
        table = pq.read_table(shard, columns=columns)
        rows.extend(table.to_pylist())
    return rows


def download_external(name: str, dest: str | Path, allow_patterns: list[str] | None = None) -> Path:
    """Download a license-cleared external dataset to `dest` via huggingface_hub.

    Verifies the license first. Use allow_patterns to fetch only tabular
    metadata (e.g. ["*.parquet", "*.json"]) without the large audio payloads.
    """
    rec = verify_license(name)
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=rec["hf_id"],
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )
    return dest


def load_neuropiano(
    parquet_dir: str | Path,
    embeddings_dir: str | Path | None = None,
) -> ExternalEvalDataset:
    """Load NeuroPiano (MIT) as a held-out eval set.

    The dataset is long-format (one row per recording x question). Rows are
    grouped by recording, question_id is pivoted to the 13-axis scores, then
    mapped to our 6 dims. The per-recording `mean` (0-6) is the skill reference.
    """
    verify_license("neuropiano")
    rows = _read_parquet_rows(
        Path(parquet_dir),
        columns=["playdata_id", "fname", "question_id", "score", "mean"],
    )

    by_rec: dict[str, dict] = {}
    for r in rows:
        key = r.get("playdata_id") or r.get("fname")
        if key is None:
            continue
        rec = by_rec.setdefault(key, {"scores": {}, "mean": None})
        qid = r.get("question_id")
        if qid is not None and r.get("score") is not None:
            rec["scores"][int(qid)] = r["score"]
        if r.get("mean") is not None:
            rec["mean"] = float(r["mean"])

    keys: list[str] = []
    labels: dict[str, list[float]] = {}
    skill: dict[str, float] = {}
    for key, rec in by_rec.items():
        vec = map_neuropiano_axes_to_dims(rec["scores"])
        # Require at least one covered dim.
        if all(np.isnan(v) for v in vec):
            continue
        keys.append(key)
        labels[key] = vec
        if rec["mean"] is not None:
            skill[key] = rec["mean"]

    return ExternalEvalDataset(
        name="neuropiano",
        keys=sorted(keys),
        labels=labels,
        skill=skill,
        embeddings_dir=Path(embeddings_dir) if embeddings_dir else None,
    )


def load_pianovam(
    metadata_path: str | Path,
    embeddings_dir: str | Path | None = None,
    native_midi_dir: str | Path | None = None,
) -> ExternalEvalDataset:
    """Load PianoVAM (CC-BY-NC-SA-4.0) as a held-out, non-commercial eval set.

    PianoVAM ships audio + native Disklavier MIDI + per-performer skill tiers but
    no perceptual per-dim ratings, so it is used for skill-rank + the D4 check
    (Aria GT-MIDI vs AMT-MIDI), not per-dim pairwise. `metadata_path` is a JSON
    mapping recording id -> {"skill": <float|int>, "midi": "<filename>"}.
    """
    verify_license("pianovam")
    meta_path = Path(metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"PianoVAM metadata not found: {meta_path}. Download per the loader "
            f"docs (hf download PianoVAM/PianoVAM_v1.0 --repo-type dataset)."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    keys = sorted(meta.keys())
    skill = {k: float(meta[k]["skill"]) for k in keys if "skill" in meta[k]}
    native_midi = {}
    if native_midi_dir is not None:
        midi_root = Path(native_midi_dir)
        for k in keys:
            fn = meta[k].get("midi")
            if fn:
                native_midi[k] = str(midi_root / fn)

    # No per-dim labels: empty label vectors mark perceptual dims uncovered.
    labels = {k: [float("nan")] * NUM_DIMS for k in keys}
    return ExternalEvalDataset(
        name="pianovam",
        keys=keys,
        labels=labels,
        skill=skill,
        embeddings_dir=Path(embeddings_dir) if embeddings_dir else None,
        native_midi=native_midi,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def skill_rank_spearman(
    pred_overall: np.ndarray | list[float],
    skill: np.ndarray | list[float],
) -> float:
    """Spearman rank correlation between predicted overall score and skill.

    Returns NaN if fewer than 2 points or a constant input.
    """
    a = np.asarray(pred_overall, dtype=float)
    b = np.asarray(skill, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


def _embeddings_get_input_fn(embeddings_dir: Path) -> Callable:
    def _inner(key: str):
        path = embeddings_dir / f"{key}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No embedding for {key} at {path}")
        emb = torch.load(path, map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        mask = torch.ones(emb.shape[:2], dtype=torch.bool)
        return emb, mask

    return _inner


def run_external_eval(
    model: torch.nn.Module,
    dataset: ExternalEvalDataset,
    encode_fn: Callable,
    compare_fn: Callable,
    predict_fn: Callable,
    get_input_fn: Optional[Callable] = None,
    num_dims: int = NUM_DIMS,
) -> dict:
    """Score a model on an external held-out set: per-dim pairwise + skill-rank.

    Reuses evaluate_model (per-dim pairwise via pairwise_detail, Cohen's d via
    skill_discrimination) and adds skill_rank_spearman of the predicted overall
    score against the dataset's continuous skill reference.

    If get_input_fn is None, embeddings are read from dataset.embeddings_dir.
    """
    if len(dataset) == 0:
        return {"skipped": "empty_external_dataset", "external_source": dataset.name}

    if get_input_fn is None:
        if dataset.embeddings_dir is None:
            raise ValueError(
                "no get_input_fn and dataset.embeddings_dir is unset; extract "
                "MuQ embeddings for the external set first"
            )
        get_input_fn = _embeddings_get_input_fn(dataset.embeddings_dir)

    result = evaluate_model(
        model=model,
        val_keys=dataset.keys,
        labels=dataset.labels,
        get_input_fn=get_input_fn,
        encode_fn=encode_fn,
        compare_fn=compare_fn,
        predict_fn=predict_fn,
        num_dims=num_dims,
        skill_tiers=dataset.skill_tiers(),
        return_predictions=True,
    )

    # skill-rank: predicted overall (mean across dims) vs skill reference.
    if "predictions" in result and dataset.skill:
        preds = np.asarray(result["predictions"], dtype=float)
        pred_keys = result.get("pred_keys", dataset.keys)
        pred_overall, skill_vals = [], []
        for i, k in enumerate(pred_keys):
            if k in dataset.skill:
                pred_overall.append(np.nanmean(preds[i, :num_dims]))
                skill_vals.append(dataset.skill[k])
        result["skill_rank_spearman"] = skill_rank_spearman(pred_overall, skill_vals)
    else:
        result["skill_rank_spearman"] = float("nan")

    result["external_source"] = dataset.name
    result["n_samples"] = len(dataset)
    return result


def d4_aria_gt_vs_amt(
    dataset: ExternalEvalDataset,
    aria_encode_midi_fn: Callable[[str], object],
    amt_midi_dir: str | Path | None = None,
) -> dict:
    """D4: Aria embeddings on native GT MIDI vs AMT-transcribed MIDI.

    PianoVAM's native Disklavier MIDI is ground truth; the AMT MIDI comes from
    the WS1 #72 pipeline. When the AMT MIDI is unavailable this returns a
    skipped sentinel naming the dependency rather than fabricating a comparison.

    Returns per-recording cosine similarity between Aria(GT) and Aria(AMT)
    embeddings, plus the mean, when both are present.
    """
    if not dataset.native_midi:
        return {"skipped": "no_native_midi", "external_source": dataset.name}
    if amt_midi_dir is None:
        return {
            "skipped": "amt_midi_unavailable (dep: WS1 #72 AMT extraction)",
            "external_source": dataset.name,
        }
    amt_root = Path(amt_midi_dir)
    if not amt_root.exists():
        return {
            "skipped": f"amt_midi_unavailable: {amt_root} missing (dep: #72)",
            "external_source": dataset.name,
        }

    sims: dict[str, float] = {}
    for key, gt_path in dataset.native_midi.items():
        amt_path = amt_root / f"{key}.mid"
        if not Path(gt_path).exists() or not amt_path.exists():
            continue
        z_gt = aria_encode_midi_fn(gt_path)
        z_amt = aria_encode_midi_fn(str(amt_path))
        if z_gt is None or z_amt is None:
            continue
        a = np.asarray(z_gt, dtype=float).ravel()
        b = np.asarray(z_amt, dtype=float).ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        sims[key] = float(np.dot(a, b) / denom)

    if not sims:
        return {"skipped": "no_paired_midi_scored", "external_source": dataset.name}
    return {
        "per_recording_cosine": sims,
        "mean_cosine": float(np.mean(list(sims.values()))),
        "n_paired": len(sims),
        "external_source": dataset.name,
    }
