"""
Piano pedagogy relevance classifier using sentence-transformers cosine similarity.

Scores text passages against the centroid of known positive teaching moments.
Used to filter CPT corpus to concentrate it on actual piano pedagogy content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_POSITIVES_PATH = (
    Path(__file__).parent.parent
    / "teaching_knowledge"
    / "data"
    / "raw_teaching_db.json"
)
_NEGATIVES_PATH = (
    Path(__file__).parent / "data" / "negatives" / "curated_negatives.jsonl"
)
_CURATED_POSITIVES_PATH = (
    Path(__file__).parent / "data" / "curated_positives.jsonl"
)


@dataclass
class ClassifierMetrics:
    precision: float
    recall: float
    f1: float
    threshold: float
    n_positives: int
    n_negatives: int


def _load_positives(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["what_teacher_said"] for r in records if r.get("what_teacher_said")]
    if not texts:
        raise ValueError(f"No positive examples found in {path}")
    return texts


def _load_jsonl_texts(path: Path) -> list[str]:
    """Load text strings from a JSONL file with a 'text' key per record."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if text:
                texts.append(text)
    return texts


def _load_negatives(path: Path) -> list[str]:
    texts = _load_jsonl_texts(path)
    if not texts:
        raise ValueError(f"No negative examples found in {path}")
    return texts


class PedagogyRelevanceClassifier:
    """
    Cosine similarity classifier for piano pedagogy relevance.

    Computes the centroid embedding of positive examples (masterclass teaching
    moments + curated explanatory-style positives) and scores new text by
    cosine similarity to that centroid. Optimal classification threshold is
    found by maximizing F1 on the full labeled dataset (positives + negatives).
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float | None = None,
    ) -> None:
        self._model = SentenceTransformer(model_name)

        masterclass_positives = _load_positives(_POSITIVES_PATH)
        extra_positives = (
            _load_jsonl_texts(_CURATED_POSITIVES_PATH)
            if _CURATED_POSITIVES_PATH.exists()
            else []
        )
        self._positive_texts = masterclass_positives + extra_positives
        self._negative_texts = _load_negatives(_NEGATIVES_PATH)

        # Compute centroid over positive embeddings
        pos_embeddings = self._model.encode(
            self._positive_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Centroid of unit vectors, then renormalize for clean cosine sim
        centroid = pos_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            raise ValueError("Positive centroid is zero vector; cannot normalize")
        self._centroid: np.ndarray = centroid / norm

        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self._find_optimal_threshold()

    def _find_optimal_threshold(self) -> float:
        """Find threshold that maximizes F1 on the full labeled dataset."""
        all_texts = self._positive_texts + self._negative_texts
        labels = [1] * len(self._positive_texts) + [0] * len(self._negative_texts)

        scores = self.score_batch(all_texts)
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        # Evaluate all candidate thresholds
        candidates = np.linspace(0.0, 1.0, 201)
        best_f1 = -1.0
        best_thresh = 0.5

        for thresh in candidates:
            preds = (scores_arr >= thresh).astype(int)
            tp = int(((preds == 1) & (labels_arr == 1)).sum())
            fp = int(((preds == 1) & (labels_arr == 0)).sum())
            fn = int(((preds == 0) & (labels_arr == 1)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        return best_thresh

    def score(self, text: str) -> float:
        """Return cosine similarity of text to positive centroid (0.0-1.0)."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarity = float(np.dot(embedding, self._centroid))
        # Cosine similarity on unit vectors is in [-1, 1]; clip to [0, 1]
        return float(np.clip(similarity, 0.0, 1.0))

    def score_batch(self, texts: list[str]) -> list[float]:
        """Return cosine similarities for a batch of texts."""
        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = embeddings @ self._centroid
        return [float(np.clip(s, 0.0, 1.0)) for s in similarities]

    def is_relevant(self, text: str) -> bool:
        """Return True if text is classified as piano pedagogy content."""
        return self.score(text) >= self._threshold

    def validate(self) -> ClassifierMetrics:
        """Compute precision/recall/F1 on the full labeled dataset."""
        all_texts = self._positive_texts + self._negative_texts
        labels = [1] * len(self._positive_texts) + [0] * len(self._negative_texts)

        scores = self.score_batch(all_texts)
        scores_arr = np.array(scores)
        labels_arr = np.array(labels)

        preds = (scores_arr >= self._threshold).astype(int)
        tp = int(((preds == 1) & (labels_arr == 1)).sum())
        fp = int(((preds == 1) & (labels_arr == 0)).sum())
        fn = int(((preds == 0) & (labels_arr == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return ClassifierMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            threshold=self._threshold,
            n_positives=len(self._positive_texts),
            n_negatives=len(self._negative_texts),
        )
