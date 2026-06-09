# model/src/piece_id_eval/stage0b_dtwgate.py
"""Stage-0b: DTW-confirm open-set gate experiment (#26).

Stage-0 verdict was TUNE: C1 note-chroma ranks well (R@10=0.938) but open-set
rejection FAILS (wrong-piece chroma cosine overlaps correct-piece cosine; no
FA<=0.05 @ TA>=0.60 operating point exists). A wrong lock poisons the whole
session, so chroma alone is unshippable.

This experiment asks the gating question: does onset+pitch subsequence-DTW
alignment cost, applied to the TOP candidate(s) chroma returns, separate
"correct piece" (in-catalog) from "harmonically-similar wrong piece"
(leave-one-out) well enough to be an open-set gate?

SUCCESS CRITERION: an operating point with false-accept <= 0.05 at
true-accept >= 0.60 on the leave-one-out negatives.

Three gate-signal variants x two query modes:
  V1 top1_cost   : DTW cost of chroma top-1 candidate (score = -cost)
  V2 topk_best   : DTW re-rank chroma top-K, gate on best (min) cost (score = -cost)
  V3 topk_margin : DTW cost margin between best and 2nd-best of chroma top-K
                   (score = cost[1] - cost[0]; larger => more confident)
  modes          : full-piece query, 90s windows (5 random starts)

Reuses existing harness primitives only (no Rust, no production code):
  NoteChromaMatcher (C1 recall), DtwCeilingMatcher._subseq_dtw_cost (confirm),
  open_set.operating_points / best_point, windowing.sample_windows.

Run:  cd model && PYTHONUNBUFFERED=1 caffeinate -i uv run python -m piece_id_eval.stage0b_dtwgate
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from piece_id_eval.matchers.dtw_ceiling import DtwCeilingMatcher
from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note, load_amt_notes, load_score_notes
from piece_id_eval.open_set import best_point, operating_points
from piece_id_eval.windowing import sample_windows

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]  # crescendai/model
_PRACTICE_ROOT = _MODEL_ROOT / "data/evals/practice_eval"
_NOTES_ROOT = _MODEL_ROOT / "data/evals/practice_eval_pseudo"
_SCORES_ROOT = _MODEL_ROOT / "data/scores"
_PIECE_MAP = _MODEL_ROOT / "data/evals/piece_id/eval_piece_map.json"
_OUTPUT = _MODEL_ROOT / "data/evals/piece_id/stage0b_dtwgate_results.json"

_TOP_K = 5
_WINDOW_SECONDS = 90.0
_N_STARTS = 5
_SEED = 42
_MAX_FA = 0.05
_MIN_TA = 0.60


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _query_pitches(query: list[Note]) -> np.ndarray:
    """Onset-sorted pitch array, exactly as DtwCeilingMatcher.rank() builds it."""
    return np.array(
        [n.pitch for n in sorted(query, key=lambda n: n.onset)], dtype=np.float32
    )


def load_data() -> tuple[dict[str, list[Note]], dict[str, list[Note]]]:
    """Build catalog {piece_id: notes} and recordings {piece_id: notes}.

    Mirrors bakeoff._cli_main: catalog from data/scores/*.json, one approved AMT
    recording per slug from the practice_eval_pseudo cache. Explicit failures.
    """
    piece_map: dict[str, str] = json.loads(_PIECE_MAP.read_text())

    catalog: dict[str, list[Note]] = {}
    for score_json in _SCORES_ROOT.glob("*.json"):
        piece_id = score_json.stem
        notes = load_score_notes(score_json)
        if notes:
            catalog[piece_id] = notes
    if not catalog:
        raise RuntimeError(f"empty catalog from {_SCORES_ROOT}")

    recordings: dict[str, list[Note]] = {}
    for slug, piece_id in piece_map.items():
        candidates_file = _PRACTICE_ROOT / slug / "candidates.yaml"
        if not candidates_file.exists():
            _log(f"[SKIP] {slug}: no candidates.yaml")
            continue
        candidates = yaml.safe_load(candidates_file.open())
        approved = [r for r in (candidates.get("recordings") or []) if r.get("approved")]
        for rec in approved:
            video_id = rec["video_id"]
            notes_path = _NOTES_ROOT / slug / video_id / "amt_notes.json"
            if not notes_path.exists():
                continue
            recordings[piece_id] = load_amt_notes(notes_path)
            break
        if piece_id not in recordings:
            _log(f"[SKIP] {slug}: no cached amt_notes.json for any approved recording")

    if piece_id_count := len(recordings):
        _log(f"[load] catalog={len(catalog)} recordings={piece_id_count}")
    else:
        raise RuntimeError("no recordings loaded")
    return catalog, recordings


def _dtw_costs_for_candidates(
    q_pitches: np.ndarray,
    candidate_ids: list[str],
    dtw: DtwCeilingMatcher,
) -> list[tuple[str, float]]:
    """DTW cost for each candidate id (lower = better). Reuses dtw._pitches refs.

    Skips the slow 254x DtwCeilingMatcher.rank: only the given candidates align.
    """
    out: list[tuple[str, float]] = []
    for cid in candidate_ids:
        ref = dtw._pitches.get(cid)
        if ref is None or ref.size == 0:
            continue
        cost = dtw._subseq_dtw_cost(q_pitches, ref)
        out.append((cid, cost))
    out.sort(key=lambda x: x[1])  # ascending cost (best first)
    return out


def _chroma_topk(matcher: NoteChromaMatcher, query: list[Note], k: int) -> list[str]:
    return [r.piece_id for r in matcher.rank(query)[:k]]


def _collect(
    catalog: dict[str, list[Note]],
    recordings: dict[str, list[Note]],
    full_catalog_chroma: NoteChromaMatcher,
    dtw: DtwCeilingMatcher,
    window_seconds: float | None,
    mode_label: str,
) -> dict[str, dict]:
    """Collect per-variant in-catalog and leave-one-out signals for one query mode.

    Returns {variant: {"in": [...], "loo": [...], "in_correct_top1": int,
                       "in_dtw_correct_top1": int, "n_in": int}}.
    For V1/V2 the stored signal is COST (lower=better); negated at scoring time.
    For V3 the stored signal is the cost MARGIN (higher=more confident).
    """
    variants = ("v1_top1_cost", "v2_topk_best", "v3_topk_margin")
    acc: dict[str, dict] = {
        v: {"in": [], "loo": [], "in_correct_top1": 0, "in_dtw_correct_top1": 0, "n_in": 0}
        for v in variants
    }

    for true_id, notes in recordings.items():
        windows = sample_windows(notes, window_seconds, _N_STARTS, _SEED)
        windows = [w for w in windows if w]

        # LOO catalog/matcher depends only on true_id -> build once per recording.
        loo_catalog = {pid: n for pid, n in catalog.items() if pid != true_id}
        loo_chroma = NoteChromaMatcher(loo_catalog)

        t0 = time.time()
        for win in windows:
            q_pitches = _query_pitches(win)

            # ---- IN-CATALOG (should ACCEPT) ----
            in_topk = _chroma_topk(full_catalog_chroma, win, _TOP_K)
            in_costs = _dtw_costs_for_candidates(q_pitches, in_topk, dtw)
            if in_costs:
                top1_id = in_topk[0]
                top1_cost = next(c for cid, c in in_costs if cid == top1_id)
                best_id, best_cost = in_costs[0]
                margin = (in_costs[1][1] - in_costs[0][1]) if len(in_costs) > 1 else 0.0

                acc["v1_top1_cost"]["in"].append(top1_cost)
                acc["v2_topk_best"]["in"].append(best_cost)
                acc["v3_topk_margin"]["in"].append(margin)
                for v in variants:
                    acc[v]["n_in"] += 1
                    acc[v]["in_correct_top1"] += int(top1_id == true_id)
                    acc[v]["in_dtw_correct_top1"] += int(best_id == true_id)

            # ---- LEAVE-ONE-OUT (should REJECT) ----
            loo_topk = _chroma_topk(loo_chroma, win, _TOP_K)
            loo_costs = _dtw_costs_for_candidates(q_pitches, loo_topk, dtw)
            if loo_costs:
                loo_top1_id = loo_topk[0]
                loo_top1_cost = next(c for cid, c in loo_costs if cid == loo_top1_id)
                loo_best_cost = loo_costs[0][1]
                loo_margin = (loo_costs[1][1] - loo_costs[0][1]) if len(loo_costs) > 1 else 0.0

                acc["v1_top1_cost"]["loo"].append(loo_top1_cost)
                acc["v2_topk_best"]["loo"].append(loo_best_cost)
                acc["v3_topk_margin"]["loo"].append(loo_margin)

        _log(f"[{mode_label}] {true_id}: {len(windows)} window(s) in {time.time()-t0:.1f}s")

    return acc


def _dist(xs: list[float]) -> dict[str, float] | None:
    if not xs:
        return None
    return {
        "min": round(float(min(xs)), 4),
        "median": round(float(statistics.median(xs)), 4),
        "max": round(float(max(xs)), 4),
        "n": len(xs),
    }


def _score_and_gate(variant: str, in_sig: list[float], loo_sig: list[float]) -> dict:
    """Convert stored signals to scores (>= threshold => accept) and find best point.

    V1/V2: signal is cost (lower=better) -> score = -cost.
    V3:    signal is margin (higher=better) -> score = margin.
    """
    if variant == "v3_topk_margin":
        in_scores = list(in_sig)
        loo_scores = list(loo_sig)
    else:
        in_scores = [-c for c in in_sig]
        loo_scores = [-c for c in loo_sig]

    if not in_scores or not loo_scores:
        return {"best_point": None, "note": "insufficient data"}

    # Threshold sweep: every observed score plus epsilon-shifted endpoints so the
    # TA=1/FA=1 and TA=0/FA=0 extremes are represented.
    obs = sorted(set(in_scores + loo_scores))
    span = (obs[-1] - obs[0]) or 1.0
    eps = span * 1e-6
    thresholds = [obs[0] - eps] + [a + (b - a) / 2 for a, b in zip(obs, obs[1:])] + obs + [obs[-1] + eps]
    thresholds = sorted(set(thresholds))

    pts = operating_points(in_scores, loo_scores, thresholds)
    bp = best_point(pts, max_fa=_MAX_FA, min_ta=_MIN_TA)
    # Also report the max-TA point achievable at FA<=0.05 (more informative than bp,
    # which minimizes FA and may pick a low-TA corner).
    feasible = [p for p in pts if p.fa <= _MAX_FA]
    max_ta_at_fa = max(feasible, key=lambda p: p.ta) if feasible else None
    return {
        "best_point": None if bp is None
        else {"threshold_on_score": round(bp.threshold, 4), "fa": round(bp.fa, 4), "ta": round(bp.ta, 4)},
        "max_ta_at_fa<=0.05": None if max_ta_at_fa is None
        else {"threshold_on_score": round(max_ta_at_fa.threshold, 4), "fa": round(max_ta_at_fa.fa, 4), "ta": round(max_ta_at_fa.ta, 4)},
    }


def main() -> None:
    t_start = time.time()
    catalog, recordings = load_data()

    _log("[build] full-catalog C1 chroma + DTW ref index ...")
    full_chroma = NoteChromaMatcher(catalog)
    dtw = DtwCeilingMatcher(catalog)  # pre-extracts ._pitches for all catalog pieces

    modes = [(None, "full"), (_WINDOW_SECONDS, "90s")]
    results: dict[str, dict] = {}
    any_pass = False

    for window_seconds, mode_label in modes:
        _log(f"\n=== mode={mode_label} ===")
        acc = _collect(catalog, recordings, full_chroma, dtw, window_seconds, mode_label)
        mode_out: dict[str, dict] = {}
        for variant, data in acc.items():
            gate = _score_and_gate(variant, data["in"], data["loo"])
            passed = gate["best_point"] is not None
            any_pass = any_pass or passed
            mode_out[variant] = {
                "in_cost_or_margin": _dist(data["in"]),
                "loo_cost_or_margin": _dist(data["loo"]),
                "chroma_top1_recall": round(data["in_correct_top1"] / data["n_in"], 3) if data["n_in"] else None,
                "dtw_rerank_top1_recall": round(data["in_dtw_correct_top1"] / data["n_in"], 3) if data["n_in"] else None,
                "gate": gate,
                "passes_FA<=0.05_TA>=0.60": passed,
            }
            _log(
                f"  [{mode_label}/{variant}] in={_dist(data['in'])} loo={_dist(data['loo'])} "
                f"-> best_point={gate['best_point']}"
            )
        results[mode_label] = mode_out

    verdict = "PASS" if any_pass else "FAIL"
    verdict_line = (
        "PASS: DTW-confirm is a viable open-set gate -> Phase-1 = chroma-recall(top-K) -> DTW-confirm-gate"
        if any_pass
        else "FAIL: DTW-confirm does not separate correct vs harmonically-similar wrong piece "
        "(no FA<=0.05 @ TA>=0.60 point in any variant/mode) -> escalate to the MuQ/Aria embedding channel"
    )

    out = {
        "experiment": "stage0b_dtwgate",
        "question": "Does onset+pitch subsequence-DTW alignment cost on chroma's top-K "
        "candidates separate in-catalog (accept) from leave-one-out (reject) at FA<=0.05 @ TA>=0.60?",
        "catalog_pieces": len(catalog),
        "recordings": len(recordings),
        "top_k": _TOP_K,
        "window_seconds": _WINDOW_SECONDS,
        "n_starts": _N_STARTS,
        "criterion": "FA<=0.05 @ TA>=0.60 (max_fa/min_ta in open_set.best_point)",
        "dtw_signal": "DtwCeilingMatcher._subseq_dtw_cost (per-note L1 semitone deviation, query-length normalized; lower=better)",
        "variants": {
            "v1_top1_cost": "DTW cost of chroma top-1 candidate",
            "v2_topk_best": "DTW re-rank chroma top-K, gate on best (min) cost",
            "v3_topk_margin": "DTW cost margin between best and 2nd-best of chroma top-K",
        },
        "results": results,
        "verdict": verdict,
        "verdict_line": verdict_line,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    _OUTPUT.write_text(json.dumps(out, indent=2))
    _log(f"\nVERDICT: {verdict}")
    _log(verdict_line)
    _log(f"Wrote {_OUTPUT}")


if __name__ == "__main__":
    main()
