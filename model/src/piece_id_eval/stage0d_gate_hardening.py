# model/src/piece_id_eval/stage0d_gate_hardening.py
"""Stage-0d: harden the Stage-0c elastic open-set gate (#26).

Stage-0c PASSED (pitch-only chord-Jaccard elastic-DTW margin gate; full-piece
TA=0.875 @ FA=0.0) but on a TINY negative set: n_loo=16 in-catalog leave-one-out
pieces (rule-of-three FA upper bound ~0.19). This experiment stress-tests that
gate before any Rust port.

The gate under test is FROZEN to the Stage-0c winner:
  cost   = onset-events (50ms) -> 12-bin pitch-class SETS; local cost = Jaccard(pc)
           ONLY (w_time=0 -- the log-IOI rhythm term HURT in Stage-0c);
           librosa.sequence.dtw(subseq=True); min cost / query-event count.
  signal = v3 margin: (2nd-best - best) elastic cost among chroma top-K.
           Accept iff margin >= threshold.

Two negative axes (the third, H3, needs non-local data and is deferred):
  H1  in-catalog leave-one-out: long full/90s queries, the HARDEST adversary
      (a harmonically-similar piece from the SAME catalog). Enlarged via windowing;
      CIs cluster-bootstrapped by recording (effective independent n is the 16
      recordings, NOT the window count -- windows of one recording are correlated).
  H2  TRUE out-of-catalog: PercePiano performance MIDIs of works ABSENT from the
      254-piece catalog (Schubert D960 mv2/mv3, Beethoven WoO80 variations).
      CAVEATS, stated loudly: (a) segments are SHORT (~5-30s) vs long positives, so
      short OOD is an EASIER reject -> FA_ood is OPTIMISTIC (a lower bound);
      (b) only 3 distinct WORKS -> low harmonic diversity. EXCLUDES Schubert
      D935 no.3 == Impromptu Op.142 no.3, which IS in the catalog.

SUCCESS CRITERION (unchanged): an operating point with TA >= 0.60 at FA <= 0.05.
Hardened bar: the FA upper 95% CI bound (not just the point estimate) <= 0.05.

Run:  cd model && PYTHONUNBUFFERED=1 caffeinate -i uv run python -m piece_id_eval.stage0d_gate_hardening
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import partitura as pa

from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note
from piece_id_eval.stage0c_elastic_dtwgate import (
    _TOP_K,
    _W_PITCH,
    ElasticGate,
    _notes_to_events,
    load_data,
)
from piece_id_eval.windowing import sample_windows

_MODULE_DIR = Path(__file__).resolve().parent
_MODEL_ROOT = _MODULE_DIR.parents[1]
_PERCEPIANO_DIR = _MODEL_ROOT / "data/midi/percepiano"
_OUTPUT = _MODEL_ROOT / "data/evals/piece_id/stage0d_gate_hardening_results.json"

_W_TIME = 0.0  # pitch-only: Stage-0c proved the rhythm term hurt
_WINDOW_SECONDS = 90.0
_N_STARTS = 10  # more starts than Stage-0c to enlarge the in-catalog sample
_SEED = 42
_MIN_TA = 0.60
_MAX_FA = 0.05
_N_BOOTSTRAP = 2000
# To certify FA <= 0.05 from ZERO observed false-accepts you need >= 60 independent
# negatives (rule of three: 3/60 = 0.05). Fewer clusters can never certify, no matter
# how the bootstrap looks -- a zero-count bootstrap CI is degenerately [0,0].
_MIN_CLUSTERS_TO_CERTIFY = 60


def _rule_of_three(n_independent: int) -> float:
    """95% upper bound on a rate when 0 events were observed in n independent trials."""
    return 3.0 / n_independent if n_independent > 0 else 1.0

# PercePiano work-prefixes confirmed ABSENT from the 254-piece catalog (true OOD).
# Schubert_D935_no.3 is DELIBERATELY excluded: D935 no.3 == Impromptu Op.142 no.3,
# which IS in the catalog (schubert.impromptu_op142.3) -> would miscount as false-accept.
_OOD_PREFIXES = ("Schubert_D960_mv2", "Schubert_D960_mv3", "Beethoven_WoO80")
_OOD_EXCLUDED = ("Schubert_D935_no.3 (== Impromptu Op.142 no.3, IN catalog)",)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_perf_midi(path: Path) -> list[Note]:
    """Load a performance MIDI to Note list via partitura (seconds-domain note array)."""
    ppart = pa.load_performance_midi(str(path))
    na = ppart.note_array()
    notes = [
        Note(
            onset=float(r["onset_sec"]),
            offset=float(r["onset_sec"]) + max(float(r["duration_sec"]), 1e-3),
            pitch=int(r["pitch"]),
            velocity=int(r["velocity"]),
        )
        for r in na
    ]
    notes.sort(key=lambda n: n.onset)
    return notes


def _work_label(stem: str) -> str:
    for p in _OOD_PREFIXES:
        if stem.startswith(p):
            return p
    return "other"


def _margin(
    query: list[Note],
    chroma: NoteChromaMatcher,
    gate: ElasticGate,
) -> tuple[float, str, int] | None:
    """Pitch-only elastic margin gate signal for one query.

    Returns (margin, best_elastic_id, n_query_events) or None if too short / <2 candidates.
    margin = 2nd-best - best elastic cost among chroma top-K (higher = more confident accept).
    """
    q_pc, q_li = _notes_to_events(query)
    n_ev = q_pc.shape[0]
    if n_ev < 2:
        return None
    topk = [r.piece_id for r in chroma.rank(query)[:_TOP_K]]
    costs: list[tuple[str, float]] = []
    for cid in topk:
        c = gate.cost(q_pc, q_li, cid, _W_PITCH, _W_TIME)
        if c is not None and np.isfinite(c):
            costs.append((cid, c))
    if len(costs) < 2:
        return None
    costs.sort(key=lambda x: x[1])
    margin = costs[1][1] - costs[0][1]
    return margin, costs[0][0], n_ev


def _collect_in_catalog(
    catalog: dict[str, list[Note]],
    recordings: dict[str, list[Note]],
    full_chroma: NoteChromaMatcher,
    gate: ElasticGate,
    window_seconds: float | None,
    mode_label: str,
) -> tuple[list[dict], list[dict]]:
    """Return (positives, loo_negatives), each a list of per-query dicts.

    positive dict: {margin, correct (best==true), rec (cluster id), n_ev}
    loo dict:      {margin, rec, n_ev}
    """
    positives: list[dict] = []
    loo_neg: list[dict] = []
    for true_id, notes in recordings.items():
        windows = [w for w in sample_windows(notes, window_seconds, _N_STARTS, _SEED) if w]
        loo_catalog = {pid: n for pid, n in catalog.items() if pid != true_id}
        loo_chroma = NoteChromaMatcher(loo_catalog)
        t0 = time.time()
        for win in windows:
            res_in = _margin(win, full_chroma, gate)
            if res_in is not None:
                m, best_id, n_ev = res_in
                positives.append({"margin": m, "correct": best_id == true_id, "rec": true_id, "n_ev": n_ev})
            res_loo = _margin(win, loo_chroma, gate)
            if res_loo is not None:
                m, _bid, n_ev = res_loo
                loo_neg.append({"margin": m, "rec": true_id, "n_ev": n_ev})
        _log(f"[{mode_label}] {true_id}: {len(windows)} window(s) {time.time()-t0:.1f}s")
    return positives, loo_neg


def _collect_ood(full_chroma: NoteChromaMatcher, gate: ElasticGate) -> list[dict]:
    """Each PercePiano OOD segment -> margin against the FULL catalog (any accept = false)."""
    files = sorted(
        f for f in _PERCEPIANO_DIR.glob("*.mid") if _work_label(f.stem) in _OOD_PREFIXES
    )
    if not files:
        raise RuntimeError(f"no OOD PercePiano files under {_PERCEPIANO_DIR}")
    out: list[dict] = []
    t0 = time.time()
    for i, f in enumerate(files):
        notes = _load_perf_midi(f)
        res = _margin(notes, full_chroma, gate)
        if res is not None:
            m, _bid, n_ev = res
            out.append({"margin": m, "work": _work_label(f.stem), "n_ev": n_ev})
        if (i + 1) % 200 == 0:
            _log(f"[ood] {i+1}/{len(files)} segments {time.time()-t0:.1f}s")
    _log(f"[ood] {len(out)} usable segments from {len(files)} files {time.time()-t0:.1f}s")
    return out


def _evcount_summary(items: list[dict]) -> dict:
    evs = [it["n_ev"] for it in items]
    return {"n": len(evs), "n_ev_median": int(statistics.median(evs)) if evs else None,
            "n_ev_min": min(evs) if evs else None, "n_ev_max": max(evs) if evs else None}


def _rates_at(threshold: float, positives: list[dict], loo: list[dict], ood: list[dict]) -> dict:
    ta = sum(1 for p in positives if p["margin"] >= threshold and p["correct"]) / len(positives) if positives else 0.0
    fa_loo = sum(1 for n in loo if n["margin"] >= threshold) / len(loo) if loo else 0.0
    fa_ood = sum(1 for n in ood if n["margin"] >= threshold) / len(ood) if ood else 0.0
    return {"threshold": round(threshold, 5), "ta_strict": round(ta, 4),
            "fa_loo": round(fa_loo, 4), "fa_ood": round(fa_ood, 4)}


def _cluster_bootstrap_fa(neg: list[dict], threshold: float, cluster_key: str, n_boot: int, seed: int) -> dict | None:
    """Cluster-bootstrap a false-accept rate: resample CLUSTERS (recordings/works) with
    replacement, so correlated within-cluster windows don't inflate the effective n.
    Returns {point, ci95_low, ci95_high, n_clusters}.
    """
    if not neg:
        return None
    clusters: dict[str, list[dict]] = {}
    for n in neg:
        clusters.setdefault(n[cluster_key], []).append(n)
    keys = list(clusters.keys())
    point = sum(1 for n in neg if n["margin"] >= threshold) / len(neg)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        drawn: list[dict] = []
        for _ in range(len(keys)):
            drawn.extend(clusters[keys[rng.randrange(len(keys))]])
        boots.append(sum(1 for n in drawn if n["margin"] >= threshold) / len(drawn))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    # Honest upper bound: a zero-count bootstrap is degenerately [0,0]; replace with
    # the rule-of-three bound on the INDEPENDENT cluster count. Certification also
    # requires enough independent clusters to support a <=0.05 claim at all.
    n_clusters = len(keys)
    upper_honest = _rule_of_three(n_clusters) if point == 0.0 else max(hi, _rule_of_three(n_clusters) if hi == 0.0 else hi)
    certifiable = n_clusters >= _MIN_CLUSTERS_TO_CERTIFY and upper_honest <= _MAX_FA
    return {"point": round(point, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "n_clusters": n_clusters, "upper_honest": round(upper_honest, 4), "certifiable": certifiable}


def _cluster_bootstrap_ta(pos: list[dict], threshold: float, n_boot: int, seed: int) -> dict | None:
    if not pos:
        return None
    clusters: dict[str, list[dict]] = {}
    for p in pos:
        clusters.setdefault(p["rec"], []).append(p)
    keys = list(clusters.keys())
    point = sum(1 for p in pos if p["margin"] >= threshold and p["correct"]) / len(pos)
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        drawn: list[dict] = []
        for _ in range(len(keys)):
            drawn.extend(clusters[keys[rng.randrange(len(keys))]])
        boots.append(sum(1 for p in drawn if p["margin"] >= threshold and p["correct"]) / len(drawn))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    return {"point": round(point, 4), "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "n_clusters": len(keys)}


def _operating_point(positives: list[dict], loo: list[dict], ood: list[dict]) -> dict:
    """Sweep margin thresholds; pick the one maximizing TA_strict s.t. fa_loo<=0.05 AND fa_ood<=0.05.
    Then attach cluster-bootstrap CIs. Returns the chosen point + the full curve summary.
    """
    margins = sorted({it["margin"] for it in (positives + loo + ood)})
    if len(margins) < 2:
        return {"chosen": None, "note": "degenerate margin set"}
    grid = [margins[0] - 1e-6] + [(a + b) / 2 for a, b in zip(margins, margins[1:])] + [margins[-1] + 1e-6]
    rows = [_rates_at(t, positives, loo, ood) for t in grid]

    feasible = [r for r in rows if r["fa_loo"] <= _MAX_FA and r["fa_ood"] <= _MAX_FA]
    chosen = max(feasible, key=lambda r: r["ta_strict"]) if feasible else None

    # Also: the max TA achievable holding ONLY the (harder, long) in-catalog constraint.
    feas_loo_only = [r for r in rows if r["fa_loo"] <= _MAX_FA]
    max_ta_loo_only = max(feas_loo_only, key=lambda r: r["ta_strict"]) if feas_loo_only else None

    out: dict = {
        "chosen_point": chosen,
        "max_ta_at_fa_loo<=0.05_only": max_ta_loo_only,
        "passes_point_estimate": bool(chosen and chosen["ta_strict"] >= _MIN_TA),
    }
    if chosen:
        thr = chosen["threshold"]
        out["bootstrap_at_chosen"] = {
            "ta_strict": _cluster_bootstrap_ta(positives, thr, _N_BOOTSTRAP, _SEED),
            "fa_loo": _cluster_bootstrap_fa(loo, thr, "rec", _N_BOOTSTRAP, _SEED),
            "fa_ood": _cluster_bootstrap_fa(ood, thr, "work", _N_BOOTSTRAP, _SEED),
        }
    return out


def main() -> None:
    t_start = time.time()
    catalog, recordings = load_data()
    _log(f"[build] chroma + elastic index over {len(catalog)} catalog pieces ...")
    full_chroma = NoteChromaMatcher(catalog)
    gate = ElasticGate(catalog)

    _log("\n=== H2: true out-of-catalog (PercePiano) ===")
    ood = _collect_ood(full_chroma, gate)
    ood_by_work: dict[str, int] = {}
    for it in ood:
        ood_by_work[it["work"]] = ood_by_work.get(it["work"], 0) + 1

    results: dict[str, dict] = {}
    for window_seconds, mode_label in [(None, "full"), (_WINDOW_SECONDS, "90s")]:
        _log(f"\n=== H1: in-catalog leave-one-out (mode={mode_label}) ===")
        positives, loo = _collect_in_catalog(catalog, recordings, full_chroma, gate, window_seconds, mode_label)
        op = _operating_point(positives, loo, ood)
        results[mode_label] = {
            "positives": _evcount_summary(positives),
            "loo_negatives": _evcount_summary(loo),
            "ood_negatives": _evcount_summary(ood),
            "operating_point": op,
        }
        _log(f"  [{mode_label}] chosen={op.get('chosen_point')} passes_pt={op.get('passes_point_estimate')}")
        if op.get("bootstrap_at_chosen"):
            _log(f"  [{mode_label}] bootstrap={op['bootstrap_at_chosen']}")

    # Verdict: certified iff some mode has TA>=0.60 AND both FA axes are CERTIFIABLE
    # (enough independent clusters + honest upper bound <= 0.05). A zero-count bootstrap
    # alone does NOT certify -- that is the bug this guards against.
    certified = False
    point_pass = False
    for _mode, r in results.items():
        op = r["operating_point"]
        if op.get("passes_point_estimate"):
            point_pass = True
        bs = op.get("bootstrap_at_chosen")
        if op.get("passes_point_estimate") and bs and bs.get("fa_loo") and bs.get("fa_ood"):
            if bs["fa_loo"]["certifiable"] and bs["fa_ood"]["certifiable"] and bs["ta_strict"]["point"] >= _MIN_TA:
                certified = True

    if certified:
        verdict = "PASS_CERTIFIED"
        verdict_line = (
            "PASS (certified): a pitch-only elastic margin gate holds TA>=0.60 with both in-catalog "
            "and true-OOD false-accept upper-95%-CI <= 0.05. Proceed to Phase-1 Rust port. "
            "NOTE H3 still open: OOD diversity is only 3 works and segments are short (FA_ood optimistic)."
        )
    elif point_pass:
        verdict = "PASS_POINT_ONLY"
        verdict_line = (
            "PASS (point estimate only): an operating point clears TA>=0.60 @ FA<=0.05, but the "
            "false-accept upper-95%-CI exceeds 0.05 (negatives still too few/narrow to certify). "
            "Do H3 (long + diverse true-OOD via ASAP/MAESTRO) before committing to Rust."
        )
    else:
        verdict = "FAIL"
        verdict_line = (
            "FAIL: no operating point holds TA>=0.60 at FA<=0.05 once the negative set is enlarged "
            "(in-catalog LOO + true-OOD). The Stage-0c PASS does not survive hardening; "
            "reconsider the gate (e.g. add the embedding channel) before any Rust."
        )

    out = {
        "experiment": "stage0d_gate_hardening",
        "gate_under_test": "Stage-0c winner: pitch-only (w_time=0) chord-Jaccard elastic subseq-DTW, v3 margin",
        "catalog_pieces": len(catalog),
        "recordings": len(recordings),
        "top_k": _TOP_K,
        "window_seconds": _WINDOW_SECONDS,
        "n_starts": _N_STARTS,
        "criterion": "TA>=0.60 @ FA<=0.05; certified iff FA upper-95%-CI <= 0.05 (cluster bootstrap)",
        "ood_source": {
            "dir": str(_PERCEPIANO_DIR.relative_to(_MODEL_ROOT)),
            "included_prefixes": list(_OOD_PREFIXES),
            "excluded": list(_OOD_EXCLUDED),
            "segments_by_work": ood_by_work,
            "distinct_works": len(ood_by_work),
        },
        "results": results,
        "verdict": verdict,
        "verdict_line": verdict_line,
        "caveats": {
            "in_catalog_diversity": "Only 16 eval recordings -> in-catalog LOO effective independent n is 16 "
            "pieces regardless of windowing. CIs are cluster-bootstrapped by recording to reflect this.",
            "ood_length": "PercePiano segments are short (see ood_negatives.n_ev_median vs positives) while "
            "positives are long. Short OOD is an EASIER reject, so FA_ood is OPTIMISTIC (a lower bound).",
            "ood_diversity": "OOD covers only 3 distinct works (Schubert D960 mv2/mv3, Beethoven WoO80). "
            "Large segment count does NOT imply harmonic diversity; bootstrap by work (n_clusters=3) is coarse.",
            "h3_remaining": "Full certification needs long + diverse true-OOD (ASAP/MAESTRO) at the production "
            "buffer length. That is the remaining gate before Rust if this run is PASS_POINT_ONLY.",
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    _OUTPUT.write_text(json.dumps(out, indent=2))
    _log(f"\nVERDICT: {verdict}")
    _log(verdict_line)
    _log(f"Wrote {_OUTPUT}")


if __name__ == "__main__":
    main()
