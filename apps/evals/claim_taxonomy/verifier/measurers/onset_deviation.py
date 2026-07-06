"""FRONT 7b: signed onset-deviation-vs-score timing measurer (#101).

Score-RELATIVE timing statistic. Truth by construction: rushing = playing ahead of the
score; dragging = behind. Consumes a SCORE-ALIGNED bundle in which each note carries
``score_onset`` (sec) -- the aligned score time from the offline aria-AMT -> parangonar
bar-map. This replaces the degenerate self-relative IOI-CV (GATE 3), which measured tempo
SPREAD, not directional deviation.

Statistic:  d = mean_over_matched_notes( (perf_onset - score_onset) * 1000 )  [ms]
  d < 0  -> perf onsets EARLY  -> rushing   (frozen route_verdict polarity "-")
  d > 0  -> perf onsets LATE   -> dragging  (polarity "+")
The sign matches the shipped TimingMeasurer convention and the frozen router contract
(a "you rushed" claim, polarity "-", is SUPPORTED only when d < 0 and |d| > tau). Unit is
ms; route_verdict is unit-agnostic.

error_bar folds three independent sources in quadrature:
  1. sampling var  -- bootstrap over the matched per-note deviations,
  2. onset-jitter var -- AMT onset noise perturbs perf_onset (SubstrateErrorEngine),
  3. alignment var -- the score_onset itself is only known to +-region.alignment_
     uncertainty_sec; this is the honest cost of the parangonar bar-map and is the term
     that makes a low-coverage / weak alignment widen the bar.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

MINIMUM_EVENTS = 8
MS = 1000.0


@dataclass
class Measurement:
    d: float
    error_bar: float
    event_count: int
    substrate_failure: bool


class OnsetDeviationMeasurer:
    """Signed mean onset deviation (perf - score) in ms, whole_piece + bar tiers."""

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        notes = bundle.get("notes") or []
        if not notes:
            raise UnverifiableError("substrate_failure", "bundle contains zero notes")

        # matched = notes carrying a score correspondence; the rest have no direction.
        perf = []
        score = []
        for n in notes:
            so = n.get("score_onset")
            if so is None:
                continue
            perf.append(float(n["onset"]))
            score.append(float(so))
        if not perf:
            raise UnverifiableError(
                "substrate_failure",
                "no score-aligned notes (bundle carries no parangonar correspondence)",
            )
        perf_arr = np.asarray(perf, dtype=np.float64)
        score_arr = np.asarray(score, dtype=np.float64)

        if location != "whole_piece":
            mask = (perf_arr >= region.audio_start_sec) & (perf_arr < region.audio_end_sec)
            perf_arr = perf_arr[mask]
            score_arr = score_arr[mask]

        event_count = int(perf_arr.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} score-aligned onsets"
                + ("" if location == "whole_piece" else
                   f" in region [{region.audio_start_sec:.2f}, {region.audio_end_sec:.2f}s]")
                + f"; need >= {MINIMUM_EVENTS}",
            )

        deviations_ms = (perf_arr - score_arr) * MS
        d = float(np.mean(deviations_ms))

        error_bar = self._error_bar(deviations_ms, region, engine)
        return Measurement(d=d, error_bar=error_bar, event_count=event_count,
                           substrate_failure=False)

    def _error_bar(
        self, deviations_ms: np.ndarray, region: ResolvedRegion, engine: SubstrateErrorEngine
    ) -> float:
        # 1. sampling variance of the mean, via bootstrap over per-note deviations.
        bootstrapped = engine.bootstrap_d(deviations_ms, lambda x: float(np.mean(x)))
        sampling_var = float(np.var(bootstrapped))

        # 2. AMT onset-jitter: perturb every perf onset by a sampled jitter (constant
        # shift cancels in a pure mean, so sample a per-note jitter vector).
        jitters = engine.timing_onset_jitter_sec()  # seconds
        n = deviations_ms.size
        perturbed_means = np.empty(len(jitters))
        rng = np.random.default_rng(0)
        for i, scale in enumerate(np.abs(jitters)):
            noise = rng.normal(0.0, max(scale, 1e-9), size=n) * MS
            perturbed_means[i] = float(np.mean(deviations_ms + noise))
        onset_var = float(np.var(perturbed_means))

        # 3. alignment uncertainty: score_onset is known only to +-alignment_uncertainty;
        # a per-note alignment error of std=align_sec shifts the mean by ~align/sqrt(n).
        align_ms = float(region.alignment_uncertainty_sec) * MS
        align_var = (align_ms ** 2) / max(n, 1)

        return math.sqrt(sampling_var + onset_var + align_var)
