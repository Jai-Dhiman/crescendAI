"""Dynamics level-cue two-arm teacher-input builder (#101 / #67).

The dynamics analog of the FRONT 7a-bis timing supply test. Front 5 (G-D) found the
generator makes ~90% dynamic-CONTRAST claims and ~0 falsifiable whole-piece LOUDNESS-LEVEL
claims, so the G-B-validated mean-velocity LEVEL statistic had nothing to score (a
construct mismatch, not a supply-size or CI problem). This asks the front-7 question for
dynamics: is that gap the generator's VOICE (won't state overall loudness) or its INPUT
(never given an overall-loudness signal)?

Two paired arms over the SAME performances that carry real AMT bundles (front-5 gd_bundles),
so the arms differ by exactly one variable:
  - ARM A: muq_means scalars only (the ORIGINAL production input).
  - ARM B: muq_means + a MEASURED overall-loudness cue derived from THIS bundle's mean AMT
    note velocity (d = mean_vel - REFERENCE_VELOCITY; direction vs tau).

KEY DIFFERENCE from the timing probe: timing's ARM-B cue had to be INVENTED (raw AMT onsets
are texture-confounded; a trustworthy directional signal needed score alignment = 7b).
Dynamics' validated statistic is mean note VELOCITY, which is density-free and gain-robust on
the raw bundle -- so the ARM-B cue is a REAL self-relative measurement, not an invented
stimulus. CONSEQUENCE (the honest caveat, carried from 7a-bis): because the SAME mean-velocity
statistic both CUES and SCORES, any resulting faithfulness rate is a SIGNAL-FIDELITY rate
(the measure feeds and adjudicates itself), not an independent faithfulness rate. The measured
contributions are therefore (1) the supply LIFT (ARM A ~0 -> ARM B committed level claims,
proving the gap is promptable) and (2) the teacher's sign-fidelity vs the cue.

Interpretation (mirrors 7a-bis):
  - ARM A level-supply > 0            -> voice confabulates level from a scalar (ungrounded).
  - ARM A ~0 but ARM B level-supply>0 -> the model is honest (won't invent overall loudness)
                                         but CAN express it when given a measured signal; the
                                         production blocker is the INPUT. Supply is promptable.
  - both ~0                           -> fundamental voice/format block for dynamics-level.

Run:
    uv run python model/src/claim_measurement/dynamics_supply/build_teacher_inputs.py \
        --bundles model/data/evals/gd_bundles/chopin_ballade_1 \
        --baseline apps/evals/results/baseline_v1.jsonl \
        --out-a /ABS/scratchpad/dyn_arm_a_inputs.json \
        --out-b /ABS/scratchpad/dyn_arm_b_inputs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[4]

# Mirrors the SHIPPED verifier (source of truth: apps/evals/claim_taxonomy/verifier/
# measurers/dynamics.py REFERENCE_VELOCITY, and the dynamics tolerance tau in
# apps/evals/claim_taxonomy/claim_taxonomy.json). REFERENCE_VELOCITY is the corpus-median
# AMT note velocity over fixed-gain PercePiano renders (#101 G-B); tau was locked at 6.5 by
# FRONT 4 tau calibration. The cue uses the SAME tau boundary the scorer uses so a "balanced"
# cue (|d|<=tau) is self-consistent with a neutral-level verdict.
REFERENCE_VELOCITY = 51.5
TAU = 6.5


def mean_velocity(notes: list[dict]) -> float:
    """Mean AMT note velocity -- the G-B-validated perceived-loudness proxy."""
    if not notes:
        raise ValueError("bundle has no notes; cannot measure mean velocity")
    return fmean(float(n["velocity"]) for n in notes)


def level_label(d: float, tau: float = TAU) -> str:
    """Signed loudness direction vs the neutral reference, using the scorer's tau band."""
    if d > tau:
        return "loud"
    if d < -tau:
        return "soft"
    return "balanced"


def build_signal(notes: list[dict], reference: float = REFERENCE_VELOCITY,
                 tau: float = TAU) -> dict:
    """The measured overall-loudness signal for one performance (ARM B ground truth)."""
    mv = mean_velocity(notes)
    d = mv - reference
    return {
        "label": level_label(d, tau),
        "mean_velocity": round(mv, 2),
        "d": round(d, 2),
        "reference_velocity": reference,
        "tau": tau,
        "measured": True,
        "caveat": ("MEASURED self-relative loudness (mean AMT velocity). The scorer uses "
                   "the SAME statistic -> any faithfulness rate is signal-fidelity."),
    }


def cue_text(sig: dict) -> str:
    """One natural-language overall-loudness cue the ARM-B teacher may turn into a claim."""
    lbl = sig["label"]
    head = "Loudness analysis vs a neutral reference (overall level): "
    if lbl == "loud":
        return (f"{head}the performance plays at a LOUDER overall dynamic level than a "
                f"neutral baseline -- projected and full across the piece.")
    if lbl == "soft":
        return (f"{head}the performance plays at a SOFTER overall dynamic level than a "
                f"neutral baseline -- subdued and under-projected across the piece.")
    return (f"{head}the overall dynamic level tracks a neutral baseline -- balanced and "
            f"well-judged, neither pushed loud nor held back soft.")


def build_arms(bundles: list[dict], muq: dict[str, dict],
               piece: str) -> tuple[list[dict], list[dict]]:
    """Pure two-arm builder. `bundles` is a list of loaded bundle dicts (each with
    video_id + notes); `muq` maps recording_id -> muq_means. A bundle with no matching
    muq_means is skipped (mirrors the timing builder)."""
    arm_a, arm_b = [], []
    for b in bundles:
        rid = b.get("video_id")
        means = muq.get(rid) if rid is not None else None
        if means is None:
            continue
        sig = build_signal(b.get("notes") or [])
        base = {"recording_id": rid, "piece": piece, "muq_means": means}
        arm_a.append(dict(base))
        arm_b.append({**base, "dynamics_level_signal": sig,
                      "dynamics_level_cue": cue_text(sig)})
    return arm_a, arm_b


def _muq_by_id(baseline: Path, piece: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in baseline.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("piece_slug") != piece:
            continue
        rid = r["recording_id"]
        if rid not in out and r.get("muq_means"):
            out[rid] = r["muq_means"]
    return out


def _load_bundles(bundle_dir: Path) -> list[dict]:
    out = []
    for p in sorted(bundle_dir.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        out.append(json.loads(p.read_text()))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.build_teacher_inputs")
    ap.add_argument("--bundles", type=Path,
                    default=REPO / "model/data/evals/gd_bundles/chopin_ballade_1")
    ap.add_argument("--baseline", type=Path,
                    default=REPO / "apps/evals/results/baseline_v1.jsonl")
    ap.add_argument("--piece", default="chopin_ballade_1")
    ap.add_argument("--out-a", type=Path, required=True)
    ap.add_argument("--out-b", type=Path, required=True)
    args = ap.parse_args(argv)

    muq = _muq_by_id(args.baseline, args.piece)
    bundles = _load_bundles(args.bundles)
    arm_a, arm_b = build_arms(bundles, muq, args.piece)

    for out, arm in ((args.out_a, arm_a), (args.out_b, arm_b)):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(arm, indent=2))

    from collections import Counter
    labels = Counter(x["dynamics_level_signal"]["label"] for x in arm_b)
    print(f"ARM A: {len(arm_a)} inputs (scalar muq only) -> {args.out_a}", flush=True)
    print(f"ARM B: {len(arm_b)} inputs (+measured loudness cue) -> {args.out_b}", flush=True)
    print(f"cue labels: {dict(labels)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
