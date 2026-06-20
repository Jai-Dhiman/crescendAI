from __future__ import annotations

from typing import Callable

import numpy as np


class SubstrateErrorEngine:
    """Seeded Monte-Carlo engine for substrate error propagation and bootstrap.

    All methods are deterministic given the same seed. The engine advances
    its RNG state per call, so call order must be consistent for reproducibility.
    """

    def __init__(self, seed: int = 42, n_samples: int = 500) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = n_samples

    def timing_onset_jitter_sec(self) -> np.ndarray:
        """n_samples of AMT onset jitter: Gaussian(mean=0, sigma=0.010s)."""
        return self._rng.normal(loc=0.0, scale=0.010, size=self._n)

    def dynamics_rms_jitter_db(self) -> np.ndarray:
        """n_samples of RMS frame variance: Gaussian(mean=0, sigma=0.3 dB)."""
        return self._rng.normal(loc=0.0, scale=0.3, size=self._n)

    def pedal_threshold_jitter(self) -> np.ndarray:
        """n_samples of CC threshold offset: Uniform(-10, +10)."""
        return self._rng.uniform(low=-10.0, high=10.0, size=self._n)

    def bootstrap_d(
        self, values: np.ndarray, stat_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Bootstrap stat_fn over values with replacement. Returns n_samples d values."""
        n = len(values)
        indices = self._rng.integers(0, n, size=(self._n, n))
        return np.array([stat_fn(values[idx]) for idx in indices])

    def alignment_uncertainty_sec(
        self,
        perf_audio_sec: np.ndarray,
        score_audio_sec: np.ndarray,
        bar_start_score_sec: float,
    ) -> float:
        """MC propagation of anchor jitter -> std of bar-start audio_sec estimates.

        Perturbs each anchor by timing_onset_jitter_sec, then interpolates
        bar_start_score_sec for each sample. Returns std of resulting estimates.
        """
        jitter = self.timing_onset_jitter_sec()
        estimates = np.empty(self._n)
        for i, j in enumerate(jitter):
            perturbed_perf = perf_audio_sec + j
            # np.interp requires sorted x; re-sort after perturbation
            order = np.argsort(perturbed_perf)
            estimates[i] = np.interp(
                bar_start_score_sec,
                score_audio_sec[order],
                perturbed_perf[order],
            )
        return float(np.std(estimates))
