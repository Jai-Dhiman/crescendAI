from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

MINIMUM_EVENTS = 8
TAU_PERCENT = 8.0  # provisional; read from taxonomy at call time


@dataclass
class Measurement:
    d: float
    error_bar: float
    event_count: int
    substrate_failure: bool


class TimingMeasurer:
    """Measure signed tempo deviation for timing claims.

    Sign convention:
    - d < 0: region is faster than reference (rushed)
    - d > 0: region is slower than reference (dragging)
    - Whole-piece: d = CV% of local IOI-derived BPM (always >= 0; high = inconsistent tempo)
    """

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

        all_onsets = np.array([float(n["onset"]) for n in notes], dtype=np.float64)
        all_onsets.sort()

        if location == "whole_piece":
            return self._measure_whole_piece(all_onsets, engine)

        region_onsets = all_onsets[
            (all_onsets >= region.audio_start_sec) & (all_onsets < region.audio_end_sec)
        ]
        event_count = int(region_onsets.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} onsets in region [{region.audio_start_sec:.2f}, "
                f"{region.audio_end_sec:.2f}s]; need >= {MINIMUM_EVENTS}",
            )

        d, sampling_var = self._region_d_and_sampling_var(
            region_onsets, all_onsets, engine
        )
        substrate_var = self._substrate_var(region_onsets, d, engine)
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _ioi_to_bpm(self, onsets: np.ndarray) -> np.ndarray:
        ioi = np.diff(onsets)
        ioi = ioi[ioi > 0.01]
        if ioi.size == 0:
            return np.array([120.0])
        return 60.0 / ioi

    def _established_tempo(self, all_onsets: np.ndarray) -> float:
        bpms = self._ioi_to_bpm(all_onsets)
        return float(np.median(bpms))

    def _region_median_bpm(self, region_onsets: np.ndarray) -> float:
        bpms = self._ioi_to_bpm(region_onsets)
        return float(np.median(bpms))

    def _region_d_and_sampling_var(
        self,
        region_onsets: np.ndarray,
        all_onsets: np.ndarray,
        engine: SubstrateErrorEngine,
    ) -> tuple[float, float]:
        established = self._established_tempo(all_onsets)
        if established == 0.0:
            raise UnverifiableError("substrate_failure", "established_tempo is zero")

        def stat(onsets: np.ndarray) -> float:
            bpm = self._region_median_bpm(onsets)
            # Sign convention: rushed (higher BPM) -> negative d; dragging (lower BPM) -> positive d
            return (established - bpm) / established * 100.0

        d = stat(region_onsets)
        bootstrapped = engine.bootstrap_d(region_onsets, stat)
        sampling_var = float(np.var(bootstrapped))
        return d, sampling_var

    def _substrate_var(
        self, region_onsets: np.ndarray, d: float, engine: SubstrateErrorEngine
    ) -> float:
        jitters = engine.timing_onset_jitter_sec()
        established = self._established_tempo(region_onsets)
        perturbed_ds = np.empty(len(jitters))
        for i, j in enumerate(jitters):
            perturbed = region_onsets + j
            bpm = self._region_median_bpm(perturbed)
            if established > 0:
                perturbed_ds[i] = (bpm - established) / established * 100.0
            else:
                perturbed_ds[i] = 0.0
        return float(np.var(perturbed_ds))

    def _measure_whole_piece(
        self, all_onsets: np.ndarray, engine: SubstrateErrorEngine
    ) -> Measurement:
        event_count = int(all_onsets.size)
        if event_count < MINIMUM_EVENTS:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} onsets; need >= {MINIMUM_EVENTS}",
            )
        bpms = self._ioi_to_bpm(all_onsets)
        if bpms.mean() == 0:
            raise UnverifiableError("substrate_failure", "mean BPM is zero")
        d = float(bpms.std() / bpms.mean() * 100.0)

        bootstrapped = engine.bootstrap_d(
            bpms, lambda x: float(x.std() / x.mean() * 100.0) if x.mean() > 0 else 0.0
        )
        sampling_var = float(np.var(bootstrapped))
        jitters = engine.timing_onset_jitter_sec()
        perturbed_cvs = np.empty(len(jitters))
        for i, j in enumerate(jitters):
            perturbed = all_onsets + j
            pb = self._ioi_to_bpm(perturbed)
            perturbed_cvs[i] = float(pb.std() / pb.mean() * 100.0) if pb.mean() > 0 else 0.0
        substrate_var = float(np.var(perturbed_cvs))
        error_bar = math.sqrt(sampling_var + substrate_var)
        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)
