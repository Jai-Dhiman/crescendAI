"""
Minimal stub for generating pseudo labels from symbolic (aligned MIDI) features.

This script reads a JSONL manifest and writes a new JSONL with pseudo_labels and pseudo_conf populated.
In a real pipeline, you would:
- Load aligned score/performance MIDI per segment
- Compute execution metrics (onset deviations, duration ratios, velocity curves, pedal CC usage)
- Map metrics to [0,1] per dimension (robust min-max with clipping)
Here we stub with simple placeholders to show structure.
"""
import json
from pathlib import Path
import random

from typing import Dict

DIMS_EXEC = [
    "timing_stability", "tempo_control", "rhythmic_accuracy",
    "articulation_length", "articulation_hardness",
    "pedal_density", "pedal_clarity",
    "dynamic_range", "dynamic_control",
    "balance_melody_vs_accomp",
]


def stub_metric_to_unit(v: float) -> float:
    # Replace with per-dimension calibrated mapping
    return max(0.0, min(1.0, v))


def main(inp: str, outp: str):
    rnd = random.Random(42)
    with open(inp) as fin, open(outp, "w") as fout:
        for line in fin:
            eg = json.loads(line)
            pseudo: Dict[str, float] = {}
            conf: Dict[str, float] = {}
            for d in eg.get("dims", DIMS_EXEC):
                if d in DIMS_EXEC:
                    # placeholder: random but stable-ish per segment
                    base = (hash(eg.get("segment_id", "")) % 1000) / 1000.0
                    noise = rnd.uniform(-0.05, 0.05)
                    val = stub_metric_to_unit(base + noise)
                    pseudo[d] = val
                    conf[d] = 0.5  # default medium confidence
            eg["pseudo_labels"] = pseudo
            eg["pseudo_conf"] = conf
            fout.write(json.dumps(eg) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input manifest JSONL")
    ap.add_argument("--out", dest="outp", required=True, help="Output manifest JSONL with pseudo labels")
    args = ap.parse_args()
    main(args.inp, args.outp)