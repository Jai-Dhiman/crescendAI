"""
Controlled degradation for piano performance quality variance generation.

Creates 4 quality tiers from pristine MAESTRO performances:
- Pristine (30%): Original, score 95-100
- Good (30%): Light degradation, score 80-95
- Moderate (25%): Moderate degradation, score 65-80
- Poor (15%): Heavy degradation, score 50-65

Implements MIDI and audio degradation functions to simulate
realistic amateur performance errors.

Reference: TRAINING_PLAN_v2.md Phase 1
"""

from enum import Enum
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import pretty_midi
import scipy.signal


class QualityTier(Enum):
    """Quality tier enumeration."""

    PRISTINE = "pristine"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"


# Quality tier parameters
QUALITY_TIER_PARAMS = {
    QualityTier.PRISTINE: {
        "score_range": (95, 100),
        "probability": 0.30,
        "midi_jitter_ms": 0,
        "wrong_note_rate": 0.0,
        "dynamics_compression": 0.0,
        "audio_noise_snr_db": None,
        "audio_filter_enabled": False,
    },
    QualityTier.GOOD: {
        "score_range": (80, 95),
        "probability": 0.30,
        "midi_jitter_ms": 10,
        "wrong_note_rate": 0.0,
        "dynamics_compression": 0.1,
        "audio_noise_snr_db": 35,
        "audio_filter_enabled": False,
    },
    QualityTier.MODERATE: {
        "score_range": (65, 80),
        "probability": 0.25,
        "midi_jitter_ms": 30,
        "wrong_note_rate": 0.02,
        "dynamics_compression": 0.3,
        "audio_noise_snr_db": 25,
        "audio_filter_enabled": False,
    },
    QualityTier.POOR: {
        "score_range": (50, 65),
        "probability": 0.15,
        "midi_jitter_ms": 60,
        "wrong_note_rate": 0.05,
        "dynamics_compression": 0.5,
        "audio_noise_snr_db": 20,
        "audio_filter_enabled": True,
    },
}


def degrade_midi_timing(
    midi_data: pretty_midi.PrettyMIDI, jitter_ms: float, seed: Optional[int] = None
) -> pretty_midi.PrettyMIDI:
    """
    Add timing jitter to MIDI note onsets and offsets.

    Simulates rhythmic imprecision by randomly shifting note times.
    Uses Gaussian distribution centered at 0 with std = jitter_ms.

    Args:
        midi_data: Input MIDI object
        jitter_ms: Standard deviation of timing jitter in milliseconds
        seed: Random seed for reproducibility

    Returns:
        New MIDI object with jittered timing
    """
    if jitter_ms <= 0:
        return midi_data

    if seed is not None:
        np.random.seed(seed)

    # Create new MIDI object
    degraded = pretty_midi.PrettyMIDI()

    # Convert jitter to seconds
    jitter_s = jitter_ms / 1000.0

    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum, name=instrument.name
        )

        # Jitter note times
        for note in instrument.notes:
            # Generate jitter values
            onset_jitter = np.random.normal(0, jitter_s)
            offset_jitter = np.random.normal(0, jitter_s)

            # Apply jitter (ensure onset < offset)
            new_start = max(0, note.start + onset_jitter)
            new_end = max(
                new_start + 0.01, note.end + offset_jitter
            )  # Minimum 10ms duration

            new_note = pretty_midi.Note(
                velocity=note.velocity, pitch=note.pitch, start=new_start, end=new_end
            )
            new_instrument.notes.append(new_note)

        # Copy control changes without jitter
        new_instrument.control_changes = instrument.control_changes.copy()

        degraded.instruments.append(new_instrument)

    return degraded


def inject_wrong_notes(
    midi_data: pretty_midi.PrettyMIDI,
    error_rate: float,
    pitch_offset_range: Tuple[int, int] = (-3, 3),
    seed: Optional[int] = None,
) -> pretty_midi.PrettyMIDI:
    """
    Inject wrong notes by randomly altering pitches.

    Simulates note accuracy errors by changing a fraction of notes
    to nearby pitches (typically ±1-3 semitones).

    Args:
        midi_data: Input MIDI object
        error_rate: Fraction of notes to alter (0.0-1.0)
        pitch_offset_range: Range of pitch alterations in semitones
        seed: Random seed for reproducibility

    Returns:
        New MIDI object with wrong notes injected
    """
    if error_rate <= 0:
        return midi_data

    if seed is not None:
        np.random.seed(seed)

    # Create new MIDI object
    degraded = pretty_midi.PrettyMIDI()

    for instrument in midi_data.instruments:
        new_instrument = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum, name=instrument.name
        )

        for note in instrument.notes:
            # Decide if this note should be altered
            if np.random.random() < error_rate:
                # Generate random pitch offset (exclude 0 to ensure change)
                offset = np.random.randint(
                    pitch_offset_range[0], pitch_offset_range[1] + 1
                )
                if offset == 0:
                    offset = 1 if np.random.random() < 0.5 else -1

                # Apply offset (keep within MIDI range 0-127)
                new_pitch = np.clip(note.pitch + offset, 0, 127)
            else:
                new_pitch = note.pitch

            new_note = pretty_midi.Note(
                velocity=note.velocity, pitch=new_pitch, start=note.start, end=note.end
            )
            new_instrument.notes.append(new_note)

        # Copy control changes
        new_instrument.control_changes = instrument.control_changes.copy()

        degraded.instruments.append(new_instrument)

    return degraded


def compress_midi_dynamics(
    midi_data: pretty_midi.PrettyMIDI, compression_factor: float
) -> pretty_midi.PrettyMIDI:
    """
    Compress MIDI velocity range to simulate poor dynamics control.

    Reduces velocity variance by moving all velocities toward the mean.
    compression_factor=0.0: no change
    compression_factor=1.0: all velocities become mean

    Args:
        midi_data: Input MIDI object
        compression_factor: Degree of compression (0.0-1.0)

    Returns:
        New MIDI object with compressed dynamics
    """
    if compression_factor <= 0:
        return midi_data

    # Create new MIDI object
    degraded = pretty_midi.PrettyMIDI()

    for instrument in midi_data.instruments:
        # Calculate mean velocity for this instrument
        velocities = [note.velocity for note in instrument.notes]
        if not velocities:
            degraded.instruments.append(instrument)
            continue

        mean_velocity = np.mean(velocities)

        new_instrument = pretty_midi.Instrument(
            program=instrument.program, is_drum=instrument.is_drum, name=instrument.name
        )

        for note in instrument.notes:
            # Compress velocity toward mean
            compressed_velocity = (
                note.velocity * (1 - compression_factor)
                + mean_velocity * compression_factor
            )
            compressed_velocity = int(np.clip(compressed_velocity, 1, 127))

            new_note = pretty_midi.Note(
                velocity=compressed_velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            )
            new_instrument.notes.append(new_note)

        # Copy control changes
        new_instrument.control_changes = instrument.control_changes.copy()

        degraded.instruments.append(new_instrument)

    return degraded


def degrade_audio_quality(
    audio_data: np.ndarray,
    sr: int,
    noise_snr_db: Optional[float] = None,
    apply_filtering: bool = False,
    filter_cutoffs: Tuple[float, float] = (100, 6000),
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Degrade audio quality by adding noise and/or filtering.

    Simulates poor recording conditions or audio artifacts.

    Args:
        audio_data: Input audio waveform
        sr: Sample rate
        noise_snr_db: SNR for additive noise (None = no noise)
        apply_filtering: Whether to apply bandpass filter
        filter_cutoffs: Bandpass filter cutoff frequencies (low, high) in Hz
        seed: Random seed for reproducibility

    Returns:
        Degraded audio waveform
    """
    degraded = audio_data.copy()

    if seed is not None:
        np.random.seed(seed)

    # Add noise
    if noise_snr_db is not None:
        # Calculate signal power
        signal_power = np.mean(audio_data**2)

        # Calculate noise power for desired SNR
        snr_linear = 10 ** (noise_snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio_data))

        # Add noise to signal
        degraded = degraded + noise

    # Apply bandpass filter (simulates poor recording equipment)
    if apply_filtering:
        nyquist = sr / 2
        low = filter_cutoffs[0] / nyquist
        high = filter_cutoffs[1] / nyquist

        # Design Butterworth bandpass filter
        b, a = scipy.signal.butter(4, [low, high], btype="band")

        # Apply filter
        degraded = scipy.signal.filtfilt(b, a, degraded)

    # Clip to prevent overflow
    degraded = np.clip(degraded, -1.0, 1.0)

    return degraded


def apply_quality_tier(
    midi_data: Optional[pretty_midi.PrettyMIDI],
    audio_data: Optional[np.ndarray],
    sr: int,
    quality_tier: QualityTier,
    seed: Optional[int] = None,
) -> Tuple[Optional[pretty_midi.PrettyMIDI], Optional[np.ndarray], Dict]:
    """
    Apply degradation corresponding to a quality tier.

    This is the main interface for creating quality variance.
    Applies all degradation functions according to tier parameters.

    Args:
        midi_data: Input MIDI object (None to skip MIDI degradation)
        audio_data: Input audio waveform (None to skip audio degradation)
        sr: Sample rate
        quality_tier: Quality tier to apply
        seed: Random seed for reproducibility

    Returns:
        Tuple of (degraded_midi, degraded_audio, metadata)
        metadata includes quality tier, score range, and applied degradations
    """
    params = QUALITY_TIER_PARAMS[quality_tier]

    degraded_midi = None
    degraded_audio = None

    # Apply MIDI degradations
    if midi_data is not None:
        degraded_midi = midi_data

        # 1. Timing jitter
        if params["midi_jitter_ms"] > 0:
            degraded_midi = degrade_midi_timing(
                degraded_midi, params["midi_jitter_ms"], seed=seed
            )

        # 2. Wrong notes
        if params["wrong_note_rate"] > 0:
            degraded_midi = inject_wrong_notes(
                degraded_midi,
                params["wrong_note_rate"],
                seed=seed + 1 if seed is not None else None,
            )

        # 3. Dynamics compression
        if params["dynamics_compression"] > 0:
            degraded_midi = compress_midi_dynamics(
                degraded_midi, params["dynamics_compression"]
            )

    # Apply audio degradations
    if audio_data is not None:
        degraded_audio = degrade_audio_quality(
            audio_data,
            sr,
            noise_snr_db=params["audio_noise_snr_db"],
            apply_filtering=params["audio_filter_enabled"],
            seed=seed + 2 if seed is not None else None,
        )

    # Generate quality score
    score_min, score_max = params["score_range"]
    quality_score = np.random.uniform(score_min, score_max)

    # Create metadata
    metadata = {
        "quality_tier": quality_tier.value,
        "quality_score": quality_score,
        "score_range": params["score_range"],
        "degradations_applied": {
            "midi_jitter_ms": params["midi_jitter_ms"],
            "wrong_note_rate": params["wrong_note_rate"],
            "dynamics_compression": params["dynamics_compression"],
            "audio_noise_snr_db": params["audio_noise_snr_db"],
            "audio_filter_enabled": params["audio_filter_enabled"],
        },
    }

    return degraded_midi, degraded_audio, metadata


def sample_quality_tier(seed: Optional[int] = None) -> QualityTier:
    """
    Sample a quality tier according to target distribution.

    Distribution:
    - Pristine: 30%
    - Good: 30%
    - Moderate: 25%
    - Poor: 15%

    Args:
        seed: Random seed for reproducibility

    Returns:
        Sampled quality tier
    """
    if seed is not None:
        np.random.seed(seed)

    probabilities = [QUALITY_TIER_PARAMS[tier]["probability"] for tier in QualityTier]
    tiers = list(QualityTier)

    chosen_tier = np.random.choice(tiers, p=probabilities)

    return chosen_tier


if __name__ == "__main__":
    print("Degradation module loaded successfully")
    print("\nQuality tiers:")
    for tier in QualityTier:
        params = QUALITY_TIER_PARAMS[tier]
        print(f"\n{tier.value.upper()}:")
        print(f"  Score range: {params['score_range']}")
        print(f"  Probability: {params['probability']:.0%}")
        print(f"  MIDI jitter: {params['midi_jitter_ms']}ms")
        print(f"  Wrong notes: {params['wrong_note_rate']:.1%}")
        print(f"  Dynamics compression: {params['dynamics_compression']:.1%}")
        print(
            f"  Audio SNR: {params['audio_noise_snr_db']} dB"
            if params["audio_noise_snr_db"]
            else "  Audio SNR: None"
        )
        print(f"  Audio filtering: {params['audio_filter_enabled']}")
