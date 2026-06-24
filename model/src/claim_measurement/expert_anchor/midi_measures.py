"""MIDI-native deterministic measurements for the M2 expert-anchor study (#66).

PercePiano segments are MIDI, but the shipped verifier consumes an audio bundle
(librosa-RMS dynamics). These functions compute the whole_piece d-statistics directly
from MIDI so they can be correlated against PercePiano perceptual labels per dimension.
They mirror the verifier's whole_piece conventions (see
docs/model/claim-verifier-signed-d-conventions.md): timing CV%, dynamics dispersion,
pedaling presence. No LLM is involved.
"""
from __future__ import annotations

import numpy as np


def timing_cv_percent(onset_sec: np.ndarray) -> float:
    """Tempo coefficient-of-variation (%) from inter-onset intervals.

    Mirrors the whole_piece timing statistic (std(bpms)/mean(bpms)*100). Chords are
    collapsed (unique sorted onsets) so simultaneous notes do not yield zero IOIs.
    """
    onsets = np.unique(np.sort(np.asarray(onset_sec, dtype=np.float64)))
    iois = np.diff(onsets)
    iois = iois[iois > 1e-3]
    if iois.size < 2:
        raise ValueError("need >=2 distinct inter-onset intervals for timing CV")
    tempos = 60.0 / iois
    mean = float(np.mean(tempos))
    if mean <= 0:
        raise ValueError("non-positive mean tempo")
    return float(np.std(tempos) / mean * 100.0)


def timing_ioi_cv(onset_sec: np.ndarray) -> float:
    """Coefficient of variation of inter-onset intervals (std/mean). The GATE 2
    timing proxy (partial rho ~0.25 vs perceived timing) -- a better perceptual
    correlate than tempo CV. Chords collapsed."""
    onsets = np.unique(np.sort(np.asarray(onset_sec, dtype=np.float64)))
    iois = np.diff(onsets)
    iois = iois[iois > 1e-3]
    if iois.size < 2:
        raise ValueError("need >=2 distinct inter-onset intervals for IOI CV")
    mean = float(np.mean(iois))
    return float(np.std(iois) / mean) if mean > 0 else 0.0


def dynamics_mean_velocity(velocity: np.ndarray) -> float:
    """Mean MIDI velocity. The GATE 2 dynamics proxy: discriminant-valid partial
    rho ~0.56 vs perceived dynamics (anti-halo), the strongest validated proxy."""
    v = np.asarray(velocity, dtype=np.float64)
    if v.size < 1:
        raise ValueError("need >=1 note for mean velocity")
    return float(np.mean(v))


def dynamics_velocity_dispersion(velocity: np.ndarray) -> float:
    """Velocity dispersion (std of MIDI velocities) -- the velocity-based dynamics
    adapter. Higher = wider dynamic spread; flat playing -> 0. Mirrors the whole_piece
    dynamics dispersion statistic on the MIDI substrate (velocity ~ perceived loudness)."""
    v = np.asarray(velocity, dtype=np.float64)
    if v.size < 2:
        raise ValueError("need >=2 notes for dynamics dispersion")
    return float(np.std(v))


def pedaling_on_fraction(cc64_events: list[tuple[float, int]], total_dur: float) -> float:
    """Fraction of the segment duration with sustain pedal down (CC64 >= 64).

    Mirrors the whole_piece pedaling presence statistic on MIDI. `cc64_events` is a list
    of (time_sec, value) pairs; the pedal is down between a >=64 event and the next <64.
    """
    if total_dur <= 0:
        raise ValueError("total_dur must be > 0")
    events = sorted(cc64_events, key=lambda e: e[0])
    on_time = 0.0
    pedal_down_since: float | None = None
    for t, value in events:
        if value >= 64 and pedal_down_since is None:
            pedal_down_since = t
        elif value < 64 and pedal_down_since is not None:
            on_time += t - pedal_down_since
            pedal_down_since = None
    if pedal_down_since is not None:
        on_time += total_dur - pedal_down_since
    return float(min(on_time / total_dur, 1.0))
