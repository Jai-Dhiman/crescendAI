from __future__ import annotations

import math

import numpy as np

from claim_taxonomy.verifier.measurers.timing import Measurement
from claim_taxonomy.verifier.models import UnverifiableError
from claim_taxonomy.verifier.location_resolver import ResolvedRegion
from claim_taxonomy.verifier.substrate_error import SubstrateErrorEngine

SR = 16000
HOP_LENGTH = 512
MINIMUM_FRAMES = 20  # ~640ms at 16kHz / 512 hop


class DynamicsMeasurer:
    """Measure RMS-based dynamic loudness for dynamics claims.

    Sign convention:
    - d < 0: region is quieter / flatter than whole-piece reference (narrow dynamics)
    - d > 0: region is louder / wider than reference
    - Whole-piece: d = RMS-contour std normalized by within-piece dynamic range (dispersion)
    """

    def measure(
        self,
        location: dict | str,
        bundle: dict,
        region: ResolvedRegion,
        engine: SubstrateErrorEngine,
    ) -> Measurement:
        audio_path = bundle.get("audio_path")
        if not audio_path:
            raise UnverifiableError("substrate_failure", "bundle missing audio_path")

        import librosa
        try:
            y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        except Exception as exc:
            raise UnverifiableError("substrate_failure", f"failed to load audio: {exc}") from exc

        rms_frames = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        rms_db = 10.0 * np.log10(rms_frames + 1e-9)
        frame_times = librosa.frames_to_time(
            np.arange(len(rms_db)), sr=SR, hop_length=HOP_LENGTH
        )

        if location == "whole_piece":
            return self._measure_whole_piece(rms_db, engine)

        mask = (frame_times >= region.audio_start_sec) & (frame_times < region.audio_end_sec)
        region_db = rms_db[mask]
        event_count = int(region_db.size)

        if event_count < MINIMUM_FRAMES:
            raise UnverifiableError(
                "region_too_short",
                f"only {event_count} RMS frames in region "
                f"[{region.audio_start_sec:.2f}, {region.audio_end_sec:.2f}s]; "
                f"need >= {MINIMUM_FRAMES}",
            )

        d = float(np.mean(region_db) - np.mean(rms_db))

        bootstrapped = engine.bootstrap_d(region_db, np.mean)
        sampling_var = float(np.var(bootstrapped - np.mean(rms_db)))

        jitters = engine.dynamics_rms_jitter_db()
        # Per-sample dB jitter j is a scalar offset, so
        # mean(region_db + j) - mean(whole_db) = d + j; var over samples reduces to var(jitters).
        substrate_var = float(np.var(jitters))
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)

    def _measure_whole_piece(
        self, rms_db: np.ndarray, engine: SubstrateErrorEngine
    ) -> Measurement:
        event_count = int(rms_db.size)
        if event_count < MINIMUM_FRAMES:
            raise UnverifiableError(
                "region_too_short",
                f"whole_piece has only {event_count} RMS frames; need >= {MINIMUM_FRAMES}",
            )
        dynamic_range = float(rms_db.max() - rms_db.min())
        if dynamic_range < 0.1:
            dynamic_range = 0.1
        d = float(rms_db.std() / dynamic_range)

        bootstrapped = engine.bootstrap_d(
            rms_db,
            lambda x: float(x.std() / max(x.max() - x.min(), 0.1)),
        )
        sampling_var = float(np.var(bootstrapped))
        jitters = engine.dynamics_rms_jitter_db()
        perturbed = np.array([
            float((rms_db + j).std() / max((rms_db + j).max() - (rms_db + j).min(), 0.1))
            for j in jitters
        ])
        substrate_var = float(np.var(perturbed))
        error_bar = math.sqrt(sampling_var + substrate_var)

        return Measurement(d=d, error_bar=error_bar, event_count=event_count, substrate_failure=False)
