"""FRONT 7b (#101): per-note score alignment + global affine detrend.

Fork of ``chroma_dtw_eval.amt_regen._build_pairs`` that KEEPS the per-note
correspondence it discards (that function reduces matches to anonymous monotone
anchor arrays for the pseudo-truth bar map).

CONTRACT (mirrors apps/evals .../measurers/onset_deviation.py): each matched note
gains ``score_onset`` = a * score_sec + b IN PERFORMANCE TIME, where (a, b) is the
least-squares fit over ALL matched notes. Raw score-seconds would be dominated by
the global tempo gap (a real performance runs ~1.5-2.2x the score's nominal
seconds); the full DTW warp would absorb the local rush/drag signal itself. Only
the global affine leaves ``perf_onset - score_onset`` = directional rush/drag.
``bar_number`` comes from the matched SCORE note's position in the bundle's
measure_table (score seconds, pre-affine), so bar-tier claims resolve against
score structure, not performance time.

Unmatched notes carry neither field; the measurer skips them (and abstains
legibly when nothing is aligned).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chroma_dtw_eval.amt_regen import (
    _amt_to_perf_na,
    _load_bach_json_score,
    _match,
)

SCORE_ALIGN_SCHEMA = "v1"
MS = 1000.0


class ScoreAlignError(RuntimeError):
    pass


def matched_note_pairs(
    score_na: np.ndarray, matches: list[dict], n_perf_notes: int
) -> list[tuple[int, int]]:
    """(perf_note_idx, score_row_idx) for every label=='match' entry.

    Perf ids are "p{i}" where i indexes the ORIGINAL notes list handed to
    _amt_to_perf_na (ids are assigned before its onset sort, so i is the bundle
    notes index). Unknown score ids are skipped (parangonar artifacts, mirrors
    _build_pairs); an out-of-range perf index is a caller bug and raises. When
    parangonar matches one perf note twice, the first entry wins.
    """
    score_id_to_idx = {str(s["id"]): i for i, s in enumerate(score_na)}
    by_perf_idx: dict[int, int] = {}
    for entry in matches:
        if entry.get("label") != "match":
            continue
        s_idx = score_id_to_idx.get(str(entry.get("score_id")))
        if s_idx is None:
            continue
        p_id = str(entry.get("performance_id"))
        if not p_id.startswith("p"):
            raise ScoreAlignError(f"unexpected performance id format: {p_id!r}")
        p_idx = int(p_id[1:])
        if not 0 <= p_idx < n_perf_notes:
            raise ScoreAlignError(
                f"perf index {p_idx} out of range for {n_perf_notes} notes"
            )
        by_perf_idx.setdefault(p_idx, s_idx)
    if not by_perf_idx:
        raise ScoreAlignError("parangonar produced zero note matches")
    return sorted(by_perf_idx.items())


def fit_affine(perf_onsets: np.ndarray, score_onsets: np.ndarray) -> tuple[float, float]:
    """Least-squares perf ~= a * score + b over all matched notes."""
    perf = np.asarray(perf_onsets, dtype=np.float64)
    score = np.asarray(score_onsets, dtype=np.float64)
    if perf.size < 2 or float(np.ptp(score)) <= 0.0:
        raise ScoreAlignError(
            f"affine fit needs >=2 distinct score onsets (got {perf.size} matches, "
            f"score span {float(np.ptp(score)) if score.size else 0.0:.4f}s)"
        )
    a, b = np.polyfit(score, perf, 1)
    if a <= 0:
        raise ScoreAlignError(
            f"affine fit has non-positive tempo ratio a={a:.4f}; alignment is degenerate"
        )
    return float(a), float(b)


def bar_number_for_score_sec(measure_table: list[dict], score_sec: float) -> int:
    rows = sorted(measure_table, key=lambda r: float(r["start_sec"]))
    starts = np.array([float(r["start_sec"]) for r in rows])
    idx = int(np.searchsorted(starts, score_sec, side="right")) - 1
    return int(rows[max(idx, 0)]["bar_number"])


def annotate_bundle(bundle: dict, score_na: np.ndarray, matches: list[dict]) -> dict:
    """Mutate `bundle` in place: per-note score_onset + bar_number, plus a
    score_align metadata block. Stale fields from a previous run are stripped
    first so unmatched notes never keep old correspondences."""
    notes = bundle["notes"]
    pairs = matched_note_pairs(score_na, matches, len(notes))
    perf = np.array([float(notes[p]["onset"]) for p, _ in pairs])
    score_sec = np.array([float(score_na[s]["onset_sec"]) for _, s in pairs])
    a, b = fit_affine(perf, score_sec)

    for n in notes:
        n.pop("score_onset", None)
        n.pop("bar_number", None)

    measure_table = bundle["measure_table"]
    residuals_ms = []
    for (p_idx, _s_idx), perf_onset, s_sec in zip(pairs, perf, score_sec):
        predicted = a * s_sec + b
        notes[p_idx]["score_onset"] = float(predicted)
        notes[p_idx]["bar_number"] = bar_number_for_score_sec(measure_table, s_sec)
        residuals_ms.append((perf_onset - predicted) * MS)

    bundle["score_align"] = {
        "schema": SCORE_ALIGN_SCHEMA,
        "affine_a": a,
        "affine_b": b,
        "n_matched": len(pairs),
        "n_notes": len(notes),
        "residual_rms_ms": float(np.sqrt(np.mean(np.square(residuals_ms)))),
    }
    return bundle


def align_bundle_file(bundle_path: Path, score_path: Path) -> dict:
    """Re-match a cached bundle's AMT notes against its score and rewrite the
    bundle (atomically) with score-aligned fields. Returns the score_align
    metadata block. Mirrors extractor.py: perf note array uses the SCORE's
    beat_sec, so the re-match reproduces the extraction-time correspondence."""
    bundle = json.loads(bundle_path.read_text())
    score_na, _measure_table, _score_sha, beat_sec = _load_bach_json_score(score_path)
    perf_na = _amt_to_perf_na(bundle["notes"], beat_sec)
    matches = _match(score_na, perf_na)
    annotate_bundle(bundle, score_na, matches)

    tmp = bundle_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle))
    tmp.replace(bundle_path)
    return bundle["score_align"]
