"""
Fit per-dimension affine calibration from pseudo labels to human labels.

Input: a labeled manifest JSONL containing labels, label_mask, pseudo_labels.
Output: calibration JSON with a,b per dimension (y_human ≈ a*y_pseudo + b).
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def fit_affine(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray):
    N, D = preds.shape
    a = np.ones(D, dtype=np.float32)
    b = np.zeros(D, dtype=np.float32)
    for d in range(D):
        m = mask[:, d] > 0.5
        if m.sum() < 20:
            continue
        x = preds[m, d]
        y = targets[m, d]
        X = np.stack([x, np.ones_like(x)], axis=1)
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        a[d], b[d] = float(sol[0]), float(sol[1])
    return a, b


def main(inp: str, out_json: str, dim_list: List[str]):
    rows: List[Dict] = [json.loads(l) for l in open(inp)]
    D = len(dim_list)
    P = []
    T = []
    M = []
    for eg in rows:
        p = np.zeros(D, dtype=np.float32)
        t = np.zeros(D, dtype=np.float32)
        m = np.zeros(D, dtype=np.float32)
        for i, d in enumerate(dim_list):
            if eg.get("pseudo_labels") and d in eg["pseudo_labels"]:
                p[i] = float(eg["pseudo_labels"][d])
            if eg.get("labels") and d in eg["labels"]:
                t[i] = float(eg["labels"][d])
            if eg.get("label_mask") and d in eg["label_mask"]:
                m[i] = 1.0 if eg["label_mask"][d] else 0.0
        P.append(p)
        T.append(t)
        M.append(m)
    P = np.stack(P)
    T = np.stack(T)
    M = np.stack(M)

    a, b = fit_affine(P, T, M)
    obj: Dict[str, Dict[str, float]] = {dim_list[i]: {"a": float(a[i]), "b": float(b[i])} for i in range(D)}
    Path(out_json).write_text(json.dumps(obj, indent=2))


if __name__ == "__main__":
    import argparse

    default_dims = [
        "timing_stability", "tempo_control", "rhythmic_accuracy",
        "articulation_length", "articulation_hardness",
        "pedal_density", "pedal_clarity",
        "dynamic_range", "dynamic_control",
        "balance_melody_vs_accomp",
        "phrasing_continuity", "expressiveness_intensity", "energy_level",
        "timbre_brightness", "timbre_richness", "timbre_color_variety",
    ]

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input labeled manifest JSONL")
    ap.add_argument("--out", dest="out_json", required=True, help="Output calibration JSON")
    ap.add_argument("--dims", nargs="*", default=default_dims, help="Dimension names in order")
    args = ap.parse_args()
    main(args.inp, args.out_json, args.dims)