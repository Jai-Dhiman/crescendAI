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
from model_improvement.taxonomy import DIMENSIONS, NUM_DIMS

ROBUSTNESS_VETO_THRESHOLD = 15.0
DIMENSION_COLLAPSE_WARN_THRESHOLD = 0.9


def per_dimension_correlation(predictions: torch.Tensor) -> np.ndarray:
    """6x6 Pearson correlation of predicted dims across held-out samples.

    Args:
        predictions: [N, NUM_DIMS] model outputs.

    Returns:
        [NUM_DIMS, NUM_DIMS] correlation matrix. NaN-filled if N < 3 or
        a column is constant.
    """
    preds_np = predictions.detach().cpu().numpy()
    if preds_np.ndim != 2 or preds_np.shape[0] < 3:
        return np.full((NUM_DIMS, NUM_DIMS), np.nan)
    with np.errstate(invalid="ignore"):
        return np.corrcoef(preds_np, rowvar=False)


def conditional_independence(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> np.ndarray:
    """Partial-correlation matrix of predicted dims conditioning on overall quality.

    Because composite labels are derived from a single ordinal, the per-dim
    mean label is used as the overall-quality anchor. Each prediction column is
    residualized on that anchor via OLS, then residuals are correlated. A
    near-zero matrix means dims carry independent info beyond overall quality;
    near-identity means they're all restatements of the same scalar.

    Args:
        predictions: [N, NUM_DIMS].
        labels: [N, NUM_DIMS] composite labels.

    Returns:
        [NUM_DIMS, NUM_DIMS] residualized correlation matrix.
    """
    preds_np = predictions.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    if preds_np.ndim != 2 or preds_np.shape[0] < 3:
        return np.full((NUM_DIMS, NUM_DIMS), np.nan)

    anchor = labels_np.mean(axis=1)
    if np.std(anchor) == 0.0:
        return per_dimension_correlation(predictions)

    residuals = np.zeros_like(preds_np)
    for d in range(preds_np.shape[1]):
        slope, intercept = np.polyfit(anchor, preds_np[:, d], 1)
        residuals[:, d] = preds_np[:, d] - (slope * anchor + intercept)

    with np.errstate(invalid="ignore"):
        return np.corrcoef(residuals, rowvar=False)


def dimension_collapse_score(predictions: torch.Tensor) -> float:
    """Scalar collapse indicator: mean absolute off-diagonal pairwise correlation.

    Values near 1.0 mean the 6-vector is effectively a scalar replicated 6x.
    Values near 0.0 mean the dims are genuinely independent. The harness uses
    this to decide whether to surface per-dim feedback or collapse to a single
    quality observation.
    """
    corr = per_dimension_correlation(predictions)
    if np.all(np.isnan(corr)):
        return float("nan")
    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = corr[mask]
    return float(np.nanmean(np.abs(off_diag)))


def skill_discrimination_report(
    predictions: torch.Tensor,
    skill_tier_labels: np.ndarray | list | None,
) -> dict:
    """Per-dim and overall Cohen's d between adjacent T5 skill tiers.

    Args:
        predictions: [N, NUM_DIMS].
        skill_tier_labels: Integer tier per sample (e.g. T5's 1-5 scale) or None.

    Returns:
        Dict with "per_tier_pair" mapping "{t_lo}->{t_hi}" to {per_dimension,
        overall, n_lo, n_hi}. Empty if fewer than 2 tiers present. Returns
        {"skipped": "no_tier_labels"} if labels are None so that upstream JSON
        always has a consistent shape.
    """
    if skill_tier_labels is None:
        return {"skipped": "no_tier_labels"}

    preds_np = predictions.detach().cpu().numpy()
    tiers = np.asarray(skill_tier_labels)
    unique_tiers = sorted(t for t in np.unique(tiers).tolist() if not np.isnan(t) if isinstance(t, float) or True)
    if len(unique_tiers) < 2:
        return {"per_tier_pair": {}, "note": "insufficient_tier_diversity"}

    per_tier_pair: dict = {}
    for t_lo, t_hi in zip(unique_tiers[:-1], unique_tiers[1:]):
        mask_lo = tiers == t_lo
        mask_hi = tiers == t_hi
        if mask_lo.sum() < 2 or mask_hi.sum() < 2:
            continue

        per_dim = {}
        for d in range(preds_np.shape[1]):
            lo = preds_np[mask_lo, d]
            hi = preds_np[mask_hi, d]
            pooled_std = float(np.sqrt((np.var(lo, ddof=1) + np.var(hi, ddof=1)) / 2))
            name = DIMENSIONS[d] if d < len(DIMENSIONS) else f"dim_{d}"
            per_dim[name] = float((hi.mean() - lo.mean()) / pooled_std) if pooled_std > 0 else 0.0

        lo_mean = preds_np[mask_lo].mean(axis=1)
        hi_mean = preds_np[mask_hi].mean(axis=1)
        pooled = float(np.sqrt((np.var(lo_mean, ddof=1) + np.var(hi_mean, ddof=1)) / 2))
        overall_d = float((hi_mean.mean() - lo_mean.mean()) / pooled) if pooled > 0 else 0.0

        per_tier_pair[f"{t_lo}->{t_hi}"] = {
            "per_dimension": per_dim,
            "overall": overall_d,
            "n_lo": int(mask_lo.sum()),
            "n_hi": int(mask_hi.sum()),
        }

    return {"per_tier_pair": per_tier_pair}


def gate_value_stats(gate_tensors: torch.Tensor | None) -> dict | None:
    """Mean/std/histogram of fusion gate activations per dimension.

    Returns None if the model has no gates (non-fusion models). Called
    separately from evaluate_model() because gate tensors are model-specific;
    see docs/model/08-uncertainty-and-diagnostics.md for when the harness
    should consume these.

    Args:
        gate_tensors: [N, NUM_DIMS] gate values, typically post-sigmoid in [0, 1].

    Returns:
        Per-dimension stats dict, or None if gate_tensors is None.
    """
    if gate_tensors is None:
        return None
    g = gate_tensors.detach().cpu().numpy()
    if g.ndim != 2:
        return None

    result: dict = {"per_dimension": {}}
    for d in range(g.shape[1]):
        name = DIMENSIONS[d] if d < len(DIMENSIONS) else f"dim_{d}"
        hist, _ = np.histogram(g[:, d], bins=10, range=(0.0, 1.0))
        result["per_dimension"][name] = {
            "mean": float(g[:, d].mean()),
            "std": float(g[:, d].std()),
            "histogram_10bins": hist.tolist(),
        }
    return result


def evaluate_model(
    model: torch.nn.Module,
    val_keys: list[str],
    labels: dict[str, list[float] | np.ndarray],
    get_input_fn: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    encode_fn: Callable,
    compare_fn: Callable,
    predict_fn: Callable,
    num_dims: int = NUM_DIMS,
    skill_tiers: dict[str, int | float] | None = None,
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
        skill_tiers: Optional map segment_key -> integer skill tier (T5 1-5).
            When given, Cohen's d between adjacent tiers is reported.

    Returns:
        Dict with keys: pairwise, pairwise_detail, r2,
        per_dimension_correlation (6x6 list), conditional_independence
        (6x6 list), dimension_collapse_score (float),
        skill_discrimination (dict).
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
        all_preds, all_sigmas, all_targets = [], [], []
        for idx, key in enumerate(valid_keys):
            if idx % 100 == 0 and idx > 0:
                print(f"  r2: {idx}/{len(valid_keys)} keys", flush=True)
            try:
                inp, mask = get_input_fn(key)
                inp = inp.to(device) if inp is not None else inp
                mask = mask.to(device) if mask is not None else mask
                pred_out = predict_fn(model, inp, mask)
                target = torch.tensor(
                    labels[key][:num_dims], dtype=torch.float32
                ).unsqueeze(0)
                if isinstance(pred_out, tuple):
                    mu, sigma = pred_out
                    all_preds.append(mu)
                    all_sigmas.append(sigma)
                else:
                    all_preds.append(pred_out)
                all_targets.append(target)
            except Exception as exc:
                if not _r2_warned:
                    logger.warning("evaluate_model r2: %s (suppressing further)", exc)
                    _r2_warned = True
                continue

        if all_preds:
            preds_cat = torch.cat(all_preds)
            targets_cat = torch.cat(all_targets)
            results["r2"] = suite.regression_r2(preds_cat, targets_cat)

            # Per-dimension independence + collapse diagnostics (Chunk A).
            # Computed from the same pre-encoded predictions so they add O(1)
            # overhead on top of the R2 pass.
            corr = per_dimension_correlation(preds_cat)
            cond_corr = conditional_independence(preds_cat, targets_cat)
            results["per_dimension_correlation"] = corr.tolist()
            results["conditional_independence"] = cond_corr.tolist()
            results["dimension_collapse_score"] = dimension_collapse_score(preds_cat)

            if all_sigmas:
                sigmas_cat = torch.cat(all_sigmas)
                from model_improvement.calibration import per_dim_calibration_report
                results["calibration"] = per_dim_calibration_report(
                    preds_cat, sigmas_cat, targets_cat
                )

            if skill_tiers is not None:
                tier_vec = [skill_tiers.get(k) for k in valid_keys if k in z_cache]
                tier_vec = [t for t in tier_vec if t is not None]
                if len(tier_vec) == preds_cat.shape[0]:
                    results["skill_discrimination"] = skill_discrimination_report(
                        preds_cat, tier_vec
                    )
                else:
                    results["skill_discrimination"] = {
                        "skipped": "tier_key_mismatch",
                        "n_preds": preds_cat.shape[0],
                        "n_tiers": len(tier_vec),
                    }
            else:
                results["skill_discrimination"] = {"skipped": "no_tier_labels"}

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
    clean_sigmas, aug_sigmas = [], []
    model.eval()
    _rob_warned = False

    device = next(model.parameters()).device

    with torch.no_grad():
        for key in val_keys:
            try:
                inp, mask = get_input_fn(key)
                inp = inp.to(device) if inp is not None else inp
                mask = mask.to(device) if mask is not None else mask
                clean_out = predict_fn(model, inp, mask)
                if isinstance(clean_out, tuple):
                    clean_scores.append(clean_out[0])
                    clean_sigmas.append(clean_out[1])
                else:
                    clean_scores.append(clean_out)

                inp_aug = inp + torch.randn_like(inp) * noise_std
                aug_out = predict_fn(model, inp_aug, mask)
                if isinstance(aug_out, tuple):
                    aug_scores.append(aug_out[0])
                    aug_sigmas.append(aug_out[1])
                else:
                    aug_scores.append(aug_out)
            except Exception as exc:
                if not _rob_warned:
                    logger.warning("run_robustness_check: %s (suppressing further)", exc)
                    _rob_warned = True
                continue

    if not clean_scores:
        return {"pearson_r": 0.0, "score_drop_pct": 100.0}

    results = compute_robustness_metrics(
        torch.cat(clean_scores), torch.cat(aug_scores)
    )

    if clean_sigmas and aug_sigmas:
        sigma_clean_mean = torch.cat(clean_sigmas).mean().item()
        sigma_aug_mean = torch.cat(aug_sigmas).mean().item()
        results["sigma_clean_mean"] = sigma_clean_mean
        results["sigma_aug_mean"] = sigma_aug_mean
        results["sigma_monotone"] = sigma_aug_mean > sigma_clean_mean

    return results


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
