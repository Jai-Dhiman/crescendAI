"""FRONT 8c: build GT-label-cued teacher inputs for the END-TO-END (deployed) dynamics rate.

The teacher-arm complement to the FRONT-8b oracle rate. 8b measured the VERIFIER's substrate
faithfulness (claim = GT label directly, no LLM). This arm layers the teacher back in: the
teacher is CUED by the ground-truth loudness label (independent of the AMT statistic that will
score it), writes prose, and its extracted claims are scored by the real AMT measurer + frozen
router. The resulting rate is the deployed end-to-end faithfulness = substrate faithfulness
(8b, 0.919) x teacher sign-fidelity (front 8, ~0.88) -- measured empirically, not multiplied.

Non-circular: the cue source is GROUND-TRUTH MIDI velocity; the score source is AMT velocity.
(Contrast front 8, where cue and score were BOTH AMT velocity -> circular.)

Reads FRONT-8b bundles (render_percepiano_bundles.py: gt_mean_velocity + gt_corpus_median),
so the cue is a REAL ground-truth signal per segment. Reuses the front-8 cue text + label band.

Run:
    uv run python .../dynamics_supply/build_gt_cued_inputs.py \
        --bundles .../model/data/evals/percepiano_indep_bundles \
        --out /ABS/scratchpad/dyn_gt_cued_inputs.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from claim_measurement.dynamics_supply.build_teacher_inputs import cue_text
from claim_measurement.dynamics_supply.independent_rate import gt_polarity

TAU_GT = 6.5
_DIMS = ["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]
_POL_TO_LABEL = {"+": "loud", "-": "soft", "neutral": "balanced"}


def build_input(bundle: dict, tau_gt: float = TAU_GT) -> dict:
    """One GT-cued teacher input. The loudness cue is derived from GROUND-TRUTH MIDI velocity
    (independent of the AMT statistic that scores it); muq_means is a neutral stub (the tested
    signal is the cue, not the scalars)."""
    gt_vel = bundle["gt_mean_velocity"]
    med = bundle["gt_corpus_median"]
    pol = gt_polarity(gt_vel, med, tau_gt)
    label = _POL_TO_LABEL[pol]
    signal = {
        "label": label,
        "gt_mean_velocity": gt_vel,
        "gt_d": round(gt_vel - med, 2),
        "source": "ground_truth_midi",
        "measured": True,
    }
    return {
        "recording_id": bundle["video_id"],
        "piece": "percepiano",
        "muq_means": {d: 0.6 for d in _DIMS},  # neutral stub -- not the tested signal
        "dynamics_level_cue": cue_text({"label": label}),
        "dynamics_level_signal": signal,
    }


def build_inputs(bundles: list[dict], tau_gt: float = TAU_GT) -> list[dict]:
    return [build_input(b, tau_gt) for b in bundles]


def _load_bundles(bundle_dir: Path) -> list[dict]:
    out = []
    for p in sorted(bundle_dir.glob("*.json")):
        if p.name.endswith(".tmp"):
            continue
        b = json.loads(p.read_text())
        if "gt_mean_velocity" not in b:
            raise ValueError(f"{p} is not a FRONT-8b bundle (no gt_mean_velocity)")
        out.append(b)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dynamics_supply.build_gt_cued_inputs")
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tau-gt", type=float, default=TAU_GT)
    args = ap.parse_args(argv)

    bundles = _load_bundles(args.bundles)
    inputs = build_inputs(bundles, args.tau_gt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inputs, indent=2))

    labels = Counter(x["dynamics_level_signal"]["label"] for x in inputs)
    print(f"{len(inputs)} GT-cued teacher inputs -> {args.out}", flush=True)
    print(f"GT cue labels: {dict(labels)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
