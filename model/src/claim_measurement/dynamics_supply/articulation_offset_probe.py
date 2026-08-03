# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.24.0"]
# ///
"""FRONT 9 articulation gate probe: is Transkun's note-OFFSET accuracy reliable enough to
UNGATE articulation? (#101).

Articulation is `status: gated_on_measurement` in the taxonomy: its statistic (key-overlap /
legato-staccato ratio) depends on note OFFSETS, and "AMT offsets are weaker than onsets (higher
quantization error)". Aria failed this (offset F1 ~0.37, #125); Transkun's offsets are far
stronger (0.79). This probe measures Transkun offset reliability DIRECTLY on the MAESTRO
real-audio bundles (which carry matched AMT `notes` + `gt_notes`), and tests whether the
articulation STATISTIC itself (per-window overlap ratio) tracks ground truth.

Two questions, two measurements:
  1. OFFSET GATE: match AMT<->GT notes (pitch + onset), compare offset error to onset error
     (the known-reliable baseline). If offset error is close to onset error, offsets are usable.
  2. STATISTIC FIDELITY: per-window articulation ratio = median(note_duration / IOI) over notes.
     Correlate AMT vs GT across windows. High corr => the offset-derived statistic is faithful.

This is a PROBE (unblock decision), not a routed faithfulness rate -- articulation has no measurer
in the orchestrator registry yet (that wiring is what ungating authorizes).

Run:
    python .../articulation_offset_probe.py \
      --bundles .../model/data/evals/maestro_indep_bundles \
      --out     .../model/data/results/articulation_offset_probe.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ONSET_MATCH_TOL = 0.05   # 50 ms: pair an AMT note to a GT note of the same pitch


def match_notes(amt: list[dict], gt: list[dict], tol: float = ONSET_MATCH_TOL) -> list[tuple[dict, dict]]:
    """Greedy nearest-onset match within pitch, each GT note used at most once."""
    by_pitch: dict[int, list[dict]] = {}
    for g in gt:
        by_pitch.setdefault(g["pitch"], []).append(g)
    for lst in by_pitch.values():
        lst.sort(key=lambda n: n["onset"])
    used: set[int] = set()
    pairs = []
    for a in sorted(amt, key=lambda n: n["onset"]):
        cands = by_pitch.get(a["pitch"], [])
        best, best_dt = None, tol
        for g in cands:
            gid = id(g)
            if gid in used:
                continue
            dt = abs(g["onset"] - a["onset"])
            if dt <= best_dt:
                best, best_dt = g, dt
        if best is not None:
            used.add(id(best))
            pairs.append((a, best))
    return pairs


def articulation_ratio(notes: list[dict]) -> float | None:
    """Median (note_duration / inter-onset-interval) over onset-sorted notes.
    >1 legato/overlapping, <1 detached/staccato. None if too few notes."""
    ns = sorted(notes, key=lambda n: n["onset"])
    ratios = []
    for i in range(len(ns) - 1):
        ioi = ns[i + 1]["onset"] - ns[i]["onset"]
        if ioi > 1e-3:
            ratios.append((ns[i]["offset"] - ns[i]["onset"]) / ioi)
    return float(np.median(ratios)) if len(ratios) >= 5 else None


def probe(bundles: list[dict]) -> dict:
    onset_errs, offset_errs, dur_amt, dur_gt = [], [], [], []
    art_amt, art_gt = [], []
    matched, total_amt = 0, 0
    for b in bundles:
        amt, gt = b["notes"], b.get("gt_notes", [])
        total_amt += len(amt)
        pairs = match_notes(amt, gt)
        matched += len(pairs)
        for a, g in pairs:
            onset_errs.append(abs(a["onset"] - g["onset"]))
            offset_errs.append(abs(a["offset"] - g["offset"]))
            dur_amt.append(a["offset"] - a["onset"])
            dur_gt.append(g["offset"] - g["onset"])
        ra, rg = articulation_ratio(amt), articulation_ratio(gt)
        if ra is not None and rg is not None:
            art_amt.append(ra)
            art_gt.append(rg)
    onset_errs, offset_errs = np.array(onset_errs), np.array(offset_errs)
    dur_amt, dur_gt = np.array(dur_amt), np.array(dur_gt)
    art_amt, art_gt = np.array(art_amt), np.array(art_gt)

    # Gate criterion: the question is not "are offsets as clean as onsets" (they never are --
    # note releases are acoustically ambiguous), but "does the offset-DERIVED articulation
    # statistic track ground truth". So the gate is statistic fidelity + a high match rate;
    # the offset-error p90 is reported as a caveat (it bounds the tau the measurer will need).
    onset_med = float(np.median(onset_errs)) if onset_errs.size else None
    offset_med = float(np.median(offset_errs)) if offset_errs.size else None
    dur_corr = float(np.corrcoef(dur_amt, dur_gt)[0, 1]) if dur_amt.size > 1 else None
    art_corr = float(np.corrcoef(art_amt, art_gt)[0, 1]) if art_amt.size > 1 else None
    mrate = matched / total_amt if total_amt else 0.0
    gate_pass = bool(art_corr is not None and art_corr >= 0.80 and mrate >= 0.95)
    return {
        "n_bundles": len(bundles),
        "matched_notes": matched,
        "amt_notes": total_amt,
        "match_rate": round(matched / total_amt, 4) if total_amt else None,
        "onset_error_ms": {"median": round(onset_med * 1000, 2) if onset_med is not None else None,
                           "p90": round(float(np.percentile(onset_errs, 90)) * 1000, 2) if onset_errs.size else None},
        "offset_error_ms": {"median": round(offset_med * 1000, 2) if offset_med is not None else None,
                            "p90": round(float(np.percentile(offset_errs, 90)) * 1000, 2) if offset_errs.size else None},
        "note_duration_corr_amt_vs_gt": round(dur_corr, 4) if dur_corr is not None else None,
        "articulation_ratio_corr_amt_vs_gt": round(art_corr, 4) if art_corr is not None else None,
        "n_windows_with_ratio": int(art_amt.size),
        "offset_gate_pass": gate_pass,
        "gate_criteria": "articulation_ratio_corr >= 0.80 AND match_rate >= 0.95",
        "offset_tail_caveat_ms": round(float(np.percentile(offset_errs, 90)) * 1000, 2) if offset_errs.size else None,
        "verdict": ("UNGATE articulation (qualified): transkun articulation statistic tracks GT; "
                    "offset p90 tail bounds the tau -- least-clean of the verifiable dims"
                    if gate_pass else "KEEP GATED: articulation statistic does not track GT"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.articulation_offset_probe")
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    bundles = [json.loads(p.read_text()) for p in sorted(args.bundles.glob("*.json"))
               if not p.name.endswith(".tmp")]
    if not bundles:
        raise SystemExit(f"no bundles in {args.bundles}")
    res = probe(bundles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print("\n=== articulation offset-gate probe (transkun, real audio) ===", flush=True)
    for k in ("matched_notes", "match_rate", "onset_error_ms", "offset_error_ms",
              "note_duration_corr_amt_vs_gt", "articulation_ratio_corr_amt_vs_gt", "offset_gate_pass", "verdict"):
        print(f"  {k}: {res[k]}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
