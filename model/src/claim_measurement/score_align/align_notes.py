"""FRONT 7b (#101): per-note score alignment + global affine detrend.

Fork of ``chroma_dtw_eval.amt_regen._build_pairs`` that KEEPS the per-note
correspondence it discards (that function reduces matches to anonymous monotone
anchor arrays for the pseudo-truth bar map).

CONTRACT (mirrors apps/evals .../measurers/onset_deviation.py): each matched note
gains ``score_onset`` = a_w * score_sec + b_w IN PERFORMANCE TIME, where (a_w, b_w)
is a least-squares fit PER FIXED PERF-TIME WINDOW (default 15s, mirroring the live
Rust per-chunk design). Raw score-seconds are dominated by the global tempo gap;
a single WHOLE-PIECE affine was tried first and empirically invalidated on the 10
real bundles (2026-07-07): rubato drift leaves residual RMS of SECONDS (Traumerei
clean core: global 2.2s median -> windowed-30s 0.30s median), 1-2 orders above any
plausible tau. The full DTW warp is the opposite failure (absorbs the rush/drag
signal). The windowed local affine is the working middle.

DEGENERACY (declared, not hidden): mean residual over the SAME notes an
LSQ-with-intercept fit used is zero BY CONSTRUCTION, so a whole-piece mean-d on
these fields is meaningless. The bundle's score_align.reference_frame field
declares this; the measurer abstains on whole_piece for same-set-LSQ frames.
Bar/region-tier d (a subset of each window) is the meaningful statistic.

``bar_number`` comes from the matched SCORE note's position in the bundle's
measure_table (score seconds, pre-affine), so bar-tier claims resolve against
score structure, not performance time.

Unmatched notes and notes in windows with <MIN_WINDOW_EVENTS matches (or a
degenerate/backward window fit) carry neither field; the measurer skips them
(and abstains legibly when nothing is aligned).
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

SCORE_ALIGN_SCHEMA = "v2"
REFERENCE_FRAME = "windowed_affine"  # same-set LSQ: whole-piece mean-d degenerate
DEFAULT_WINDOW_SEC = 15.0  # mirrors the live Rust per-chunk affine (note_align.rs)
MIN_WINDOW_EVENTS = 8
# Drop matches farther than this from the bundle's accepted pseudo-truth anchor
# envelope. parangonar's global matcher scatters matches across the score with no
# local constraint (#108 day-0 finding; confirmed here: 9/10 real bundles were
# match-scatter-limited, not rubato-limited). The anchors passed distributional
# acceptance at extraction (MIN_ANCHORS/span/gap), so they are the trusted coarse
# map. 1.5s = the pseudo-truth localization tolerance.
ANCHOR_GATE_SEC = 1.5
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


def _windowed_predictions(
    perf: np.ndarray, score_sec: np.ndarray, window_sec: float
) -> tuple[np.ndarray, int]:
    """Per-fixed-perf-window affine predictions. Returns (predictions with NaN
    for notes in unfittable windows, n_fitted_windows)."""
    preds = np.full(perf.size, np.nan)
    n_windows = 0
    w0 = float(perf.min())
    w_end = float(perf.max())
    while w0 <= w_end:
        mask = (perf >= w0) & (perf < w0 + window_sec)
        w0 += window_sec
        if int(mask.sum()) < MIN_WINDOW_EVENTS:
            continue
        try:
            a, b = fit_affine(perf[mask], score_sec[mask])
        except ScoreAlignError:
            continue  # degenerate window (flat score span / backward fit): skip
        preds[mask] = a * score_sec[mask] + b
        n_windows += 1
    return preds, n_windows


def annotate_bundle(
    bundle: dict,
    score_na: np.ndarray,
    matches: list[dict],
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> dict:
    """Mutate `bundle` in place: per-note score_onset + bar_number, plus a
    score_align metadata block. Stale fields from a previous run are stripped
    first so unmatched notes never keep old correspondences."""
    notes = bundle["notes"]
    pairs = matched_note_pairs(score_na, matches, len(notes))
    perf = np.array([float(notes[p]["onset"]) for p, _ in pairs])
    score_sec = np.array([float(score_na[s]["onset_sec"]) for _, s in pairs])

    # Anchor gate: keep only matches consistent with the accepted coarse map.
    anchors = bundle.get("anchors") or {}
    anchor_perf = np.asarray(anchors.get("perf_audio_sec") or [], dtype=np.float64)
    anchor_score = np.asarray(anchors.get("score_audio_sec") or [], dtype=np.float64)
    n_anchor_dropped = 0
    if anchor_perf.size >= 2:
        expected_score = np.interp(perf, anchor_perf, anchor_score)
        keep = np.abs(score_sec - expected_score) <= ANCHOR_GATE_SEC
        n_anchor_dropped = int((~keep).sum())
        if not keep.any():
            raise ScoreAlignError(
                f"all {len(pairs)} matches fall outside the anchor envelope "
                f"(+-{ANCHOR_GATE_SEC}s); match set and accepted coarse map disagree"
            )
        pairs = [pr for pr, k in zip(pairs, keep) if k]
        perf, score_sec = perf[keep], score_sec[keep]

    preds, n_windows = _windowed_predictions(perf, score_sec, window_sec)
    if n_windows == 0:
        raise ScoreAlignError(
            f"no window of {window_sec}s had >={MIN_WINDOW_EVENTS} fittable matches "
            f"({len(pairs)} matches over {perf.max() - perf.min():.1f}s)"
        )

    for n in notes:
        n.pop("score_onset", None)
        n.pop("bar_number", None)

    measure_table = bundle["measure_table"]
    residuals_ms = []
    for (p_idx, _s_idx), perf_onset, s_sec, predicted in zip(pairs, perf, score_sec, preds):
        if np.isnan(predicted):
            continue
        notes[p_idx]["score_onset"] = float(predicted)
        notes[p_idx]["bar_number"] = bar_number_for_score_sec(measure_table, s_sec)
        residuals_ms.append((perf_onset - predicted) * MS)

    bundle["score_align"] = {
        "schema": SCORE_ALIGN_SCHEMA,
        "reference_frame": REFERENCE_FRAME,
        "window_sec": float(window_sec),
        "n_windows": n_windows,
        "n_matched": len(pairs),
        "n_anchor_dropped": n_anchor_dropped,
        "n_annotated": len(residuals_ms),
        "n_notes": len(notes),
        "residual_rms_ms": float(np.sqrt(np.mean(np.square(residuals_ms)))),
        "median_abs_residual_ms": float(np.median(np.abs(residuals_ms))),
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
