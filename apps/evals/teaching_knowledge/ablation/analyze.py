# apps/evals/teaching_knowledge/ablation/analyze.py
"""Ablation analysis: cosine similarity, four-quadrant binning, decision rule."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def cosine_similarity(a: str, b: str) -> float:
    embeddings = _model().encode([a, b], convert_to_numpy=True, normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def compute_deltas(jsonl_path: Path) -> dict[str, float]:
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]

    session_scores: dict[str, dict[str, float]] = {}
    for row in rows:
        rid = row["recording_id"]
        cond = row["condition"]
        dims = row.get("judge_dimensions") or []
        scores = [float(d["score"]) for d in dims if isinstance(d.get("score"), (int, float))]
        if not scores:
            continue
        session_scores.setdefault(rid, {})[cond] = sum(scores) / len(scores)

    conditions = ("flip", "shuffle", "marginal")
    deltas: dict[str, float] = {}
    for cond in conditions:
        pairs = [
            (session_scores[sid]["real"], session_scores[sid][cond])
            for sid in session_scores
            if "real" in session_scores[sid] and cond in session_scores[sid]
        ]
        if not pairs:
            deltas[cond] = 0.0
            continue
        mean_real = sum(r for r, _ in pairs) / len(pairs)
        mean_cond = sum(c for _, c in pairs) / len(pairs)
        deltas[cond] = round(mean_real - mean_cond, 6)
    return deltas


def decide_verdict(deltas: dict[str, float], mean_sim_flip: float) -> str:
    flip = deltas.get("flip", 0.0)
    shuffle = deltas.get("shuffle", 0.0)
    marginal = deltas.get("marginal", 0.0)
    if flip <= 0.15 or mean_sim_flip >= 0.92:
        return "false"
    if flip > 0.3 and shuffle > 0.15 and marginal > 0.15 and mean_sim_flip < 0.85:
        return "true"
    return "equivocal"
