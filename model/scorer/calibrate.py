import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def fit_affine_per_dim(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit y' = a*y + b per dimension via least squares on masked entries.

    preds/targets/mask: [N, D]
    Returns arrays a[D], b[D]
    """
    N, D = preds.shape
    a = np.ones(D, dtype=np.float32)
    b = np.zeros(D, dtype=np.float32)
    for d in range(D):
        m = mask[:, d] > 0.5
        if m.sum() < 10:
            continue
        x = preds[m, d]
        y = targets[m, d]
        # Solve [x, 1] * [a, b] = y in least squares sense
        X = np.stack([x, np.ones_like(x)], axis=1)
        sol, *_ = np.linalg.lstsq(X, y, rcond=None)
        a[d], b[d] = float(sol[0]), float(sol[1])
    return a, b


def apply_affine(preds: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return preds * a[None, :] + b[None, :]


def save_calibration(a: np.ndarray, b: np.ndarray, path: str):
    obj: Dict[str, Dict[str, float]] = {str(i): {"a": float(a[i]), "b": float(b[i])} for i in range(len(a))}
    Path(path).write_text(json.dumps(obj, indent=2))