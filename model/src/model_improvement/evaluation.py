"""Shared evaluation logic for audio and symbolic comparison notebooks."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)

from model_improvement.metrics import (
    MetricsSuite,
    compute_robustness_metrics,
    competition_spearman,
)
from model_improvement.taxonomy import NUM_DIMS

ROBUSTNESS_VETO_THRESHOLD = 15.0


def evaluate_model(
    model: torch.nn.Module,
    val_keys: list[str],
    labels: dict[str, list[float] | np.ndarray],
    get_input_fn: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    encode_fn: Callable,
    compare_fn: Callable,
    predict_fn: Callable,
    num_dims: int = NUM_DIMS,
) -> dict:
    """Assess a model on one fold's validation keys.

    Args:
        model: The encoder model (in .eval() mode).
        val_keys: Segment keys in this fold's validation set.
        labels: segment_key -> label array (at least num_dims long).
        get_input_fn: key -> (input_tensor[1,...], mask[1,...]).
        encode_fn: (model, input, mask) -> z embedding [1, D].
        compare_fn: (model, z_a, z_b) -> ranking logits [1, num_dims].
        predict_fn: (model, input, mask) -> scores [1, num_dims].
        num_dims: Number of label dimensions.

    Returns:
        Dict with "pairwise", "pairwise_detail", "r2" keys.
    """
    suite = MetricsSuite(ambiguous_threshold=0.05)
    results = {}
    valid_keys = [k for k in val_keys if k in labels]
    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        # Pre-encode all keys once (O(n) instead of O(n^2) encode calls)
        print(f"  encoding {len(valid_keys)} keys...", flush=True)
        z_cache = {}
        _enc_warned = False
        for idx, key in enumerate(valid_keys):
            if (idx + 1) % 50 == 0:
                print(f"  encode: {idx + 1}/{len(valid_keys)}", flush=True)
            try:
                inp, mask = get_input_fn(key)
                inp = inp.to(device) if inp is not None else inp
                mask = mask.to(device) if mask is not None else mask
                z_cache[key] = encode_fn(model, inp, mask)
            except Exception as exc:
                if not _enc_warned:
                    logger.warning("evaluate_model encode: %s (suppressing further)", exc)
                    _enc_warned = True

        # Pairwise comparisons using cached encodings
        encoded_keys = [k for k in valid_keys if k in z_cache]
        all_logits, all_la, all_lb = [], [], []
        total_pairs = len(encoded_keys) * (len(encoded_keys) - 1) // 2
        pair_count = 0
        _pw_warned = False
        for i, key_a in enumerate(encoded_keys):
            for key_b in encoded_keys[i + 1:]:
                pair_count += 1
                if pair_count % 5000 == 0:
                    print(f"  pairwise: {pair_count}/{total_pairs} pairs ({100 * pair_count / total_pairs:.0f}%)", flush=True)
                try:
                    logits = compare_fn(model, z_cache[key_a], z_cache[key_b])
                    lab_a = torch.tensor(
                        labels[key_a][:num_dims], dtype=torch.float32
                    )
                    lab_b = torch.tensor(
                        labels[key_b][:num_dims], dtype=torch.float32
                    )
                    all_logits.append(logits)
                    all_la.append(lab_a.unsqueeze(0))
                    all_lb.append(lab_b.unsqueeze(0))
                except Exception as exc:
                    if not _pw_warned:
                        logger.warning("evaluate_model pairwise: %s (suppressing further)", exc)
                        _pw_warned = True
                    continue

        if all_logits:
            pw = suite.pairwise_accuracy(
                torch.cat(all_logits), torch.cat(all_la), torch.cat(all_lb)
            )
            results["pairwise"] = pw["overall"]
            results["pairwise_detail"] = pw

        # R2 regression using predict_fn (also benefits from being O(n))
        _r2_warned = False
        all_preds, all_targets = [], []
        for idx, key in enumerate(valid_keys):
            if idx % 100 == 0 and idx > 0:
                print(f"  r2: {idx}/{len(valid_keys)} keys", flush=True)
            try:
                inp, mask = get_input_fn(key)
                inp = inp.to(device) if inp is not None else inp
                mask = mask.to(device) if mask is not None else mask
                pred = predict_fn(model, inp, mask)
                target = torch.tensor(
                    labels[key][:num_dims], dtype=torch.float32
                ).unsqueeze(0)
                all_preds.append(pred)
                all_targets.append(target)
            except Exception as exc:
                if not _r2_warned:
                    logger.warning("evaluate_model r2: %s (suppressing further)", exc)
                    _r2_warned = True
                continue

        if all_preds:
            results["r2"] = suite.regression_r2(
                torch.cat(all_preds), torch.cat(all_targets)
            )

    return results


def aggregate_folds(fold_results: list[dict]) -> dict:
    """Compute mean and std across folds for all numeric metrics.

    Args:
        fold_results: List of per-fold result dicts.

    Returns:
        Dict with "{metric}_mean" and "{metric}_std" for each numeric key.
    """
    all_keys: set[str] = set()
    for fr in fold_results:
        for k, v in fr.items():
            if isinstance(v, (int, float)):
                all_keys.add(k)

    result = {}
    for key in sorted(all_keys):
        values = [
            fr[key]
            for fr in fold_results
            if key in fr and isinstance(fr[key], (int, float))
        ]
        if values:
            result[f"{key}_mean"] = float(np.mean(values))
            result[f"{key}_std"] = float(np.std(values))
    return result


def run_robustness_check(
    model: torch.nn.Module,
    val_keys: list[str],
    get_input_fn: Callable,
    predict_fn: Callable,
    noise_std: float = 0.05,
) -> dict:
    """Check robustness via clean vs noisy predictions.

    Adds Gaussian noise to inputs as a proxy when real augmented data
    is unavailable.
    """
    clean_scores, aug_scores = [], []
    model.eval()
    _rob_warned = False

    device = next(model.parameters()).device

    with torch.no_grad():
        for key in val_keys:
            try:
                inp, mask = get_input_fn(key)
                inp = inp.to(device) if inp is not None else inp
                mask = mask.to(device) if mask is not None else mask
                pred_clean = predict_fn(model, inp, mask)
                clean_scores.append(pred_clean)

                inp_aug = inp + torch.randn_like(inp) * noise_std
                pred_aug = predict_fn(model, inp_aug, mask)
                aug_scores.append(pred_aug)
            except Exception as exc:
                if not _rob_warned:
                    logger.warning("run_robustness_check: %s (suppressing further)", exc)
                    _rob_warned = True
                continue

    if not clean_scores:
        return {"pearson_r": 0.0, "score_drop_pct": 100.0}

    return compute_robustness_metrics(
        torch.cat(clean_scores), torch.cat(aug_scores)
    )


def select_winner(
    results: dict[str, dict],
    veto_threshold: float = ROBUSTNESS_VETO_THRESHOLD,
) -> str | None:
    """Select best model: primary pairwise, tiebreak R2, veto robustness.

    Args:
        results: {model_name: {pairwise_mean, r2_mean, score_drop_pct}}.
        veto_threshold: Max allowed score_drop_pct.

    Returns:
        Winning model name, or None if all vetoed.
    """
    candidates = []
    for name, metrics in results.items():
        drop = metrics.get("score_drop_pct", 0.0)
        if drop > veto_threshold:
            continue
        candidates.append((
            name,
            metrics.get("pairwise_mean", 0.0),
            metrics.get("r2_mean", 0.0),
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return candidates[0][0]
