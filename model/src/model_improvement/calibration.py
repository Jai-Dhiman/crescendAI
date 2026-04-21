"""Calibration metrics for heteroscedastic output heads.

ECE computation follows Kuleshov et al. (2018) "Accurate Uncertainties for
Deep Learning Using Calibrated Regression": for each confidence level alpha,
the fraction of targets inside the Gaussian (1-alpha) CI should equal alpha.

Reference: https://arxiv.org/abs/1807.00263
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm

from model_improvement.taxonomy import DIMENSIONS


def expected_calibration_error(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
) -> list[float]:
    """Per-dimension ECE using Gaussian confidence intervals.

    For a well-calibrated Gaussian, the fraction of targets inside the
    (1 - alpha) CI equals alpha at every level. ECE is the mean absolute
    deviation from this ideal, averaged across n_bins confidence levels.

    Args:
        mu: Predicted means [N, num_dims].
        sigma: Predicted stds [N, num_dims].
        targets: Ground-truth values [N, num_dims].
        n_bins: Number of confidence levels to evaluate.

    Returns:
        ECE per dimension, length num_dims. Lower is better. Target <= 0.05.
    """
    mu_np = mu.detach().cpu().float().numpy()
    sigma_np = sigma.detach().cpu().float().numpy()
    targets_np = targets.detach().cpu().float().numpy()

    alphas = np.linspace(1.0 / n_bins, 1.0, n_bins)

    eces = []
    for d in range(mu_np.shape[1]):
        observed = []
        for alpha in alphas:
            z = norm.ppf((1.0 + alpha) / 2.0)
            in_ci = np.abs(targets_np[:, d] - mu_np[:, d]) <= z * sigma_np[:, d]
            observed.append(in_ci.mean())
        eces.append(float(np.mean(np.abs(np.array(observed) - alphas))))

    return eces


def reliability_diagram(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
    dim_idx: int = 0,
) -> dict:
    """Calibration curve data for a single dimension.

    Returns values suitable for plotting: x=alphas, y=observed fractions.
    Perfect calibration is the diagonal y=x.

    Args:
        mu: Predicted means [N, num_dims].
        sigma: Predicted stds [N, num_dims].
        targets: Ground-truth values [N, num_dims].
        n_bins: Number of confidence levels.
        dim_idx: Which dimension to compute the diagram for.

    Returns:
        Dict with 'alphas' list and 'observed' list (both length n_bins).
    """
    mu_np = mu.detach().cpu().float().numpy()[:, dim_idx]
    sigma_np = sigma.detach().cpu().float().numpy()[:, dim_idx]
    targets_np = targets.detach().cpu().float().numpy()[:, dim_idx]

    alphas = np.linspace(1.0 / n_bins, 1.0, n_bins)
    observed = []
    for alpha in alphas:
        z = norm.ppf((1.0 + alpha) / 2.0)
        in_ci = np.abs(targets_np - mu_np) <= z * sigma_np
        observed.append(float(in_ci.mean()))

    return {"alphas": alphas.tolist(), "observed": observed}


def per_dim_calibration_report(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
    ece_target: float = 0.05,
) -> dict:
    """Full calibration report: per-dimension ECE, mean sigma, pass/fail.

    Args:
        mu: Predicted means [N, num_dims].
        sigma: Predicted stds [N, num_dims].
        targets: Ground-truth values [N, num_dims].
        n_bins: Number of confidence levels for ECE.
        ece_target: Threshold for 'passes_target' flag.

    Returns:
        Dict keyed by dimension name with ECE, mean_sigma, and passes_target.
    """
    num_dims = mu.shape[1]
    eces = expected_calibration_error(mu, sigma, targets, n_bins)
    mean_sigmas = sigma.detach().cpu().float().mean(dim=0).tolist()

    report: dict = {}
    for i in range(num_dims):
        dim = DIMENSIONS[i] if i < len(DIMENSIONS) else f"dim_{i}"
        report[dim] = {
            "ece": eces[i],
            "mean_sigma": mean_sigmas[i],
            "passes_target": eces[i] <= ece_target,
        }

    return report
