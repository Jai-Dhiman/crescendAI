"""Shared evaluation logic for audio and symbolic comparison notebooks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from scipy.stats import pearsonr

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
    return_pairwise_masks: bool = False,
    return_predictions: bool = False,
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
            logits_cat = torch.cat(all_logits)
            la_cat = torch.cat(all_la)
            lb_cat = torch.cat(all_lb)
            pw = suite.pairwise_accuracy(logits_cat, la_cat, lb_cat)
            results["pairwise"] = pw["overall"]
            results["pairwise_detail"] = pw

            if return_pairwise_masks:
                # Per-pair overall correct / non-ambiguous masks, aligned to the
                # upper-triangular ordering of `encoded_keys`. Two model runs over
                # the SAME val keys produce identically-ordered masks, so their
                # error patterns can be correlated downstream (the G2 gate).
                correct, non_amb = pairwise_overall_masks(
                    logits_cat, la_cat, lb_cat,
                    threshold=suite.ambiguous_threshold,
                )
                results["pairwise_masks"] = {
                    "correct": correct.tolist(),
                    "non_ambiguous": non_amb.tolist(),
                    "keys": list(encoded_keys),
                }

        # R2 regression using predict_fn (also benefits from being O(n))
        _r2_warned = False
        all_preds, all_sigmas, all_targets = [], [], []
        pred_keys: list[str] = []
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
                pred_keys.append(key)
            except Exception as exc:
                if not _r2_warned:
                    logger.warning("evaluate_model r2: %s (suppressing further)", exc)
                    _r2_warned = True
                continue

        if all_preds:
            preds_cat = torch.cat(all_preds)
            targets_cat = torch.cat(all_targets)
            results["r2"] = suite.regression_r2(preds_cat, targets_cat)

            if return_predictions:
                # Per-key predictions (aligned to pred_keys) so callers can build
                # per-piece pairwise without re-running inference.
                results["predictions"] = preds_cat.detach().cpu().tolist()
                results["pred_keys"] = pred_keys

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


# ---------------------------------------------------------------------------
# WS2 validation gates (issue #75)
#
# Three numbers every sweep must emit:
#   1. dimension_collapse_score   (already computed in evaluate_model)
#   2. MuQ<->Aria error-correlation (G2) with an explicit <0.5 pass/fail
#   3. per-piece pairwise + bootstrap CI, flagging single-piece regressions
#
# The G2 gate is fine-tune-agnostic: it consumes correct/incorrect masks from
# whatever model produced them, so it runs identically on fine-tuned models and
# frozen probes. The decorrelation *verdict* across the two real streams is
# WS3 #80; this module wires the mechanism into the eval path.
# ---------------------------------------------------------------------------

G2_DECORRELATION_THRESHOLD = 0.5


def pairwise_overall_masks(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair overall correct / non-ambiguous masks for a set of comparisons.

    "Overall" collapses the per-dimension ranking to a single decision by
    averaging across dims, matching how the error-correlation gate reasons about
    whether two streams agree on which performance is better.

    Args:
        logits: [n_pairs, n_dims] ranking logits (positive => A > B).
        labels_a: [n_pairs, n_dims] ground-truth scores for A.
        labels_b: [n_pairs, n_dims] ground-truth scores for B.
        threshold: |mean label diff| below this marks the pair ambiguous.

    Returns:
        (correct, non_ambiguous): two boolean arrays of shape [n_pairs].
    """
    logits_np = logits.detach().cpu().numpy()
    diff = (labels_a.detach().cpu().numpy() - labels_b.detach().cpu().numpy())

    pred_overall = logits_np.mean(axis=1) > 0
    diff_overall = diff.mean(axis=1)
    true_overall = diff_overall > 0

    non_ambiguous = np.abs(diff_overall) >= threshold
    correct = pred_overall == true_overall
    return correct, non_ambiguous


def _normalize_fold_masks(masks) -> list[dict]:
    """Accept either a single {correct, non_ambiguous} dict or a list of them."""
    if isinstance(masks, dict):
        return [masks]
    return list(masks)


def _g2_verdict(phi_mean: float, threshold: float) -> str:
    if np.isnan(phi_mean):
        return "UNDEFINED: insufficient non-degenerate comparisons"
    if phi_mean < threshold:
        return "PASS: streams make different mistakes (decorrelated)"
    return "FAIL: streams make the same mistakes (redundant)"


def error_correlation_gate(
    masks_a,
    masks_b,
    threshold: float = G2_DECORRELATION_THRESHOLD,
) -> dict:
    """G2 gate: phi correlation between two streams' per-pair error patterns.

    For each fold, restrict to comparisons both streams treat as non-ambiguous,
    then compute the phi coefficient (Pearson r on the binary correct vectors).
    A fold whose shared correct vector is constant (all-correct or all-wrong for
    a stream) yields an undefined phi and is excluded rather than silently
    counted as zero.

    Args:
        masks_a / masks_b: a {correct, non_ambiguous} dict, or a per-fold list of
            them. Both arguments must describe the same folds in the same order
            over the same val-key ordering.
        threshold: decorrelation pass cutoff (G2 is <0.5).

    Returns:
        Dict with phi_per_fold, phi_mean, phi_std, n_pairs_per_fold,
        n_valid_folds, threshold, pass (bool), and a human-readable verdict.
    """
    folds_a = _normalize_fold_masks(masks_a)
    folds_b = _normalize_fold_masks(masks_b)
    if len(folds_a) != len(folds_b):
        raise ValueError(
            f"fold count mismatch: {len(folds_a)} streams_a vs {len(folds_b)} streams_b"
        )

    phi_per_fold: list[float] = []
    n_pairs_per_fold: list[int] = []
    for fa, fb in zip(folds_a, folds_b):
        ca = np.asarray(fa["correct"], dtype=bool)
        cb = np.asarray(fb["correct"], dtype=bool)
        na = np.asarray(fa["non_ambiguous"], dtype=bool)
        nb = np.asarray(fb["non_ambiguous"], dtype=bool)
        if not (len(ca) == len(cb) == len(na) == len(nb)):
            raise ValueError(
                "mask length mismatch within a fold; streams must score the "
                "same val keys in the same order"
            )

        shared = na & nb
        n_shared = int(shared.sum())
        n_pairs_per_fold.append(n_shared)
        if n_shared < 3:
            phi_per_fold.append(float("nan"))
            continue

        a = ca[shared].astype(float)
        b = cb[shared].astype(float)
        if a.std() == 0.0 or b.std() == 0.0:
            # One stream is uniformly right (or wrong) on the shared set; phi is
            # undefined. Surfacing nan keeps a degenerate fold from masquerading
            # as a decorrelated (passing) result.
            phi_per_fold.append(float("nan"))
            continue

        r, _ = pearsonr(a, b)
        phi_per_fold.append(float(r))

    valid = [p for p in phi_per_fold if not np.isnan(p)]
    phi_mean = float(np.mean(valid)) if valid else float("nan")
    phi_std = float(np.std(valid)) if valid else float("nan")
    passed = (not np.isnan(phi_mean)) and phi_mean < threshold

    return {
        "phi_per_fold": phi_per_fold,
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "n_pairs_per_fold": n_pairs_per_fold,
        "n_valid_folds": len(valid),
        "threshold": threshold,
        "pass": bool(passed),
        "verdict": _g2_verdict(phi_mean, threshold),
    }


def _overall_scores(
    predictions: torch.Tensor,
    keys: list[str],
    labels: dict[str, list[float] | np.ndarray],
    num_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean-across-dims scalar pred + label per key (aligned to `keys`)."""
    preds_np = predictions.detach().cpu().numpy()
    op = preds_np[:, :num_dims].mean(axis=1)
    ol = np.array(
        [np.asarray(labels[k][:num_dims], dtype=np.float64).mean() for k in keys]
    )
    return op, ol


def _pairwise_from_scalars(
    op: np.ndarray, ol: np.ndarray, threshold: float
) -> tuple[float, int]:
    """Vectorized overall pairwise accuracy over all unordered pairs."""
    n = len(op)
    if n < 2:
        return float("nan"), 0
    dp = op[:, None] - op[None, :]
    dl = ol[:, None] - ol[None, :]
    iu = np.triu_indices(n, k=1)
    dpu, dlu = dp[iu], dl[iu]
    na = np.abs(dlu) >= threshold
    if not na.any():
        return float("nan"), 0
    correct = (dpu[na] > 0) == (dlu[na] > 0)
    return float(correct.mean()), int(na.sum())


def per_piece_pairwise(
    predictions: torch.Tensor,
    keys: list[str],
    labels: dict[str, list[float] | np.ndarray],
    piece_of: dict[str, str],
    threshold: float = 0.05,
    num_dims: int = NUM_DIMS,
) -> dict:
    """Overall pairwise accuracy computed within each piece.

    PercePiano has only 3 pieces => 3 CV folds, too few to trust an aggregate.
    Reporting pairwise per piece exposes a single-piece regression that a flat
    fold mean would hide.

    Args:
        predictions: [N, >=num_dims] aligned to `keys`.
        keys: list of segment keys.
        labels: key -> label vector.
        piece_of: key -> piece id.

    Returns:
        piece_id -> {pairwise, n_pairs, n_keys}.
    """
    op, ol = _overall_scores(predictions, keys, labels, num_dims)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        piece = piece_of.get(k)
        if piece is not None:
            groups[piece].append(i)

    out: dict = {}
    for piece, idxs in groups.items():
        pw, n_pairs = _pairwise_from_scalars(op[idxs], ol[idxs], threshold)
        out[piece] = {"pairwise": pw, "n_pairs": n_pairs, "n_keys": len(idxs)}
    return out


def per_piece_pairwise_bootstrap(
    predictions: torch.Tensor,
    keys: list[str],
    labels: dict[str, list[float] | np.ndarray],
    piece_of: dict[str, str],
    n_boot: int = 1000,
    seed: int = 42,
    threshold: float = 0.05,
    num_dims: int = NUM_DIMS,
    ci: float = 0.95,
) -> dict:
    """per_piece_pairwise plus a bootstrap CI over keys within each piece.

    Returns piece_id -> {pairwise, n_pairs, n_keys, ci_low, ci_high}. The CI is
    the (1-ci)/2 and 1-(1-ci)/2 percentiles of the bootstrap distribution; the
    point estimate is the full-sample pairwise accuracy.
    """
    point = per_piece_pairwise(predictions, keys, labels, piece_of, threshold, num_dims)
    op, ol = _overall_scores(predictions, keys, labels, num_dims)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        piece = piece_of.get(k)
        if piece is not None:
            groups[piece].append(i)

    rng = np.random.RandomState(seed)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 - (1.0 - ci) / 2.0) * 100.0

    out: dict = {}
    for piece, idxs in groups.items():
        base = dict(point.get(piece, {}))
        idx_arr = np.asarray(idxs)
        n_keys = len(idx_arr)
        if n_keys < 2:
            out[piece] = {**base, "ci_low": float("nan"), "ci_high": float("nan")}
            continue

        boot_vals = []
        for _ in range(n_boot):
            sample = rng.choice(idx_arr, size=n_keys, replace=True)
            pw, _ = _pairwise_from_scalars(op[sample], ol[sample], threshold)
            if not np.isnan(pw):
                boot_vals.append(pw)

        if boot_vals:
            out[piece] = {
                **base,
                "ci_low": float(np.percentile(boot_vals, lo_q)),
                "ci_high": float(np.percentile(boot_vals, hi_q)),
            }
        else:
            out[piece] = {**base, "ci_low": float("nan"), "ci_high": float("nan")}
    return out


def flag_single_piece_regressions(
    current: dict,
    baseline: dict,
    tol: float = 0.02,
) -> list[dict]:
    """Flag pieces whose pairwise dropped more than `tol` below the baseline.

    Args:
        current / baseline: piece_id -> {"pairwise": float} (per_piece_pairwise
            output shape).
        tol: regression tolerance in absolute pairwise accuracy.

    Returns:
        List of {piece, current, baseline, delta} for regressed pieces.
    """
    flagged: list[dict] = []
    for piece, cur in current.items():
        if piece not in baseline:
            continue
        c = cur.get("pairwise")
        b = baseline[piece].get("pairwise")
        if c is None or b is None or np.isnan(c) or np.isnan(b):
            continue
        if c < b - tol:
            flagged.append(
                {"piece": piece, "current": c, "baseline": b, "delta": c - b}
            )
    return flagged


def build_validation_gate_block(
    fold_metrics: list[dict],
    *,
    muq_masks=None,
    aria_masks=None,
    per_piece: dict | None = None,
    per_piece_baseline: dict | None = None,
    g2_threshold: float = G2_DECORRELATION_THRESHOLD,
) -> dict:
    """Assemble the three WS2 validation gates into one JSON-serializable block.

    Args:
        fold_metrics: per-fold evaluate_model results (for dimension_collapse).
        muq_masks / aria_masks: per-fold pairwise masks from each stream. When
            both are present the G2 gate runs; otherwise it is explicitly marked
            skipped (single stream) rather than silently omitted.
        per_piece: per_piece_pairwise output for this run.
        per_piece_baseline: prior per_piece_pairwise to flag regressions against.
    """
    collapse_per_fold = [m.get("dimension_collapse_score") for m in fold_metrics]
    collapse_vals = [
        c for c in collapse_per_fold
        if isinstance(c, (int, float)) and not np.isnan(c)
    ]

    block: dict = {
        "dimension_collapse_per_fold": collapse_per_fold,
        "dimension_collapse_mean": (
            float(np.mean(collapse_vals)) if collapse_vals else None
        ),
    }

    if muq_masks is not None and aria_masks is not None:
        block["g2_error_correlation"] = error_correlation_gate(
            muq_masks, aria_masks, threshold=g2_threshold
        )
    else:
        block["g2_error_correlation"] = {
            "skipped": "single_stream",
            "note": (
                "provide muq_masks and aria_masks (post fine-tune) to run the "
                "G2 decorrelation gate"
            ),
        }

    if per_piece is not None:
        block["per_piece_pairwise"] = per_piece
        if per_piece_baseline is not None:
            block["single_piece_regressions"] = flag_single_piece_regressions(
                per_piece, per_piece_baseline
            )

    return block


def print_validation_gate_summary(block: dict) -> None:
    """Print the three gates, with an explicit G2 PASS/FAIL line."""
    print("=== WS2 validation gates ===")

    collapse = block.get("dimension_collapse_mean")
    collapse_str = f"{collapse:.4f}" if isinstance(collapse, float) else "n/a"
    print(f"  dimension_collapse_mean: {collapse_str}")

    g2 = block.get("g2_error_correlation", {})
    if g2.get("skipped"):
        print(f"  G2 MuQ<->Aria error correlation: SKIPPED ({g2['skipped']})")
    else:
        phi = g2.get("phi_mean", float("nan"))
        thr = g2.get("threshold", G2_DECORRELATION_THRESHOLD)
        status = "PASS" if g2.get("pass") else "FAIL"
        print(
            f"  G2 MuQ<->Aria error correlation: phi={phi:.4f} "
            f"(threshold {thr}) -> {status}"
        )
        print(f"      {g2.get('verdict', '')}")

    per_piece = block.get("per_piece_pairwise")
    if per_piece:
        print("  per-piece pairwise:")
        for piece, stats in sorted(per_piece.items()):
            pw = stats.get("pairwise", float("nan"))
            lo = stats.get("ci_low")
            hi = stats.get("ci_high")
            ci_str = (
                f" [{lo:.3f}, {hi:.3f}]"
                if isinstance(lo, float) and isinstance(hi, float) and not np.isnan(lo)
                else ""
            )
            print(f"      {piece}: {pw:.4f}{ci_str} (n={stats.get('n_keys')})")

    regressions = block.get("single_piece_regressions")
    if regressions:
        print("  SINGLE-PIECE REGRESSIONS:")
        for r in regressions:
            print(
                f"      {r['piece']}: {r['current']:.4f} vs baseline "
                f"{r['baseline']:.4f} (delta {r['delta']:+.4f})"
            )
