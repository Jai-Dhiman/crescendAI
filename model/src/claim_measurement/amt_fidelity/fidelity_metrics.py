"""Pure AMT-fidelity metrics core (no model / scipy / numpy dependency).

Splits the testable math out of the model-I/O shell (onset_duration_render.py).
Everything here is deterministic and unit-tested in test_fidelity_metrics.py.

The two reductions gate two grounding dimensions in apps/api/src/wasm/score-analysis:
  - onset_fidelity  -> timing  (NoteAlignment.onset_deviation_ms; +-30ms thresholds)
  - duration_fidelity -> articulation (mean(perf.offset-perf.onset)/mean(score.dur))

Note dicts follow aria-amt _transcribe(): {"pitch": int, "onset": float_s,
"offset": float_s, "velocity": int}. Ground-truth MIDI notes use the same shape.
"""
from __future__ import annotations

import math
from typing import Any


# ---- statistics (hand-rolled to keep this module dependency-free) -----------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pstd(xs: list[float]) -> float:
    """Population standard deviation (ddof=0)."""
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _rankdata(xs: list[float]) -> list[float]:
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation. Returns Pearson on the rank vectors."""
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    if len(xs) < 2:
        return float("nan")
    rx, ry = _rankdata(xs), _rankdata(ys)
    mx, my = _mean(rx), _mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / math.sqrt(vx * vy)


# ---- note matching ----------------------------------------------------------

def match_notes(
    gt_notes: list[dict[str, Any]],
    amt_notes: list[dict[str, Any]],
    onset_window_s: float = 0.1,
) -> dict[str, Any]:
    """Greedy AMT<->GT matching: pitch-exact, nearest-onset, within-window, unique.

    For each GT note (in onset order) pick the as-yet-unused AMT note of the
    SAME MIDI pitch whose onset is closest and within onset_window_s. Unmatched
    GT notes are recall misses (the AMT substrate dropped/displaced them).
    """
    used = [False] * len(amt_notes)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for gt in sorted(gt_notes, key=lambda n: n["onset"]):
        best_j, best_d = -1, onset_window_s
        for j, amt in enumerate(amt_notes):
            if used[j] or amt["pitch"] != gt["pitch"]:
                continue
            d = abs(amt["onset"] - gt["onset"])
            if d <= best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            pairs.append((gt, amt_notes[best_j]))
    n_gt = len(gt_notes)
    return {
        "pairs": pairs,
        "n_gt": n_gt,
        "n_amt": len(amt_notes),
        "n_matched": len(pairs),
        "recall": (len(pairs) / n_gt) if n_gt else 0.0,
    }


# ---- fidelity reductions ----------------------------------------------------

def onset_fidelity(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Per-note onset recovery error in ms. Gates timing (+-30ms thresholds).

    bias_ms  = mean signed (amt - gt) onset error -> a fixable calibration constant
    noise_ms = population std of the error -> the irreducible substrate scatter
    The decision quantity is noise_ms vs the 30ms rush/drag band.
    """
    if not pairs:
        return {"n": 0}
    errs = [(amt["onset"] - gt["onset"]) * 1000.0 for gt, amt in pairs]
    abs_errs = [abs(e) for e in errs]
    s = sorted(abs_errs)
    p90 = s[min(len(s) - 1, int(math.ceil(0.9 * len(s)) - 1))]
    med = _median(errs)
    return {
        "n": len(pairs),
        "bias_ms": _mean(errs),
        "noise_ms": _pstd(errs),
        "median_abs_ms": _median(abs_errs),
        "mad_ms": _median([abs(e - med) for e in errs]),
        "p90_abs_ms": p90,
    }


def duration_fidelity(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Per-note duration recovery. Gates articulation (legato/staccato ratio).

    spearman_dur preserves the *ordering* of note lengths (articulation depends
    on relative duration); median_ratio is the central amt_dur/gt_dur. This is
    the first test of the AMT *offset* head, which drives note duration.
    """
    gt_durs, amt_durs, ratios = [], [], []
    for gt, amt in pairs:
        gd = gt["offset"] - gt["onset"]
        ad = amt["offset"] - amt["onset"]
        if gd > 0.0:
            gt_durs.append(gd)
            amt_durs.append(ad)
            ratios.append(ad / gd)
    if not ratios:
        return {"n": 0}
    med = _median(ratios)
    return {
        "n": len(ratios),
        "median_ratio": med,
        "mad_ratio": _median([abs(r - med) for r in ratios]),
        "spearman_dur": spearman(amt_durs, gt_durs) if len(ratios) >= 2 else float("nan"),
        "mean_gt_dur": _mean(gt_durs),
        "mean_amt_dur": _mean(amt_durs),
    }
