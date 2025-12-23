"""
Labeling functions for weak supervision of piano performance evaluation.

Implements labeling functions for 8 dimensions (TRAINING_PLAN_v2.md):

Technical (4):
- note_accuracy
- rhythmic_stability
- articulation_clarity
- pedal_technique

Timbre/Dynamics (2):
- tone_quality
- dynamic_range

Interpretive (2):
- musical_expression
- overall_interpretation

Each function operates on MIDI and/or audio features to estimate quality scores.
These imperfect signals are combined via weak supervision to create training labels.
"""

from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import pretty_midi


class LabelingFunction:
    """Base class for labeling functions."""

    def __init__(self, name: str, dimension: str, weight: float = 1.0):
        """
        Initialize labeling function.

        Args:
            name: Descriptive name of this labeling function
            dimension: Target dimension (note_accuracy, rhythmic_precision, etc.)
            weight: Relative weight/confidence of this function (0-1)
        """
        self.name = name
        self.dimension = dimension
        self.weight = weight

    def __call__(
        self,
        midi_data: Optional[pretty_midi.PrettyMIDI] = None,
        audio_data: Optional[np.ndarray] = None,
        sr: int = 24000,
        **kwargs,
    ) -> Optional[float]:
        """
        Compute pseudo-label score.

        Args:
            midi_data: PrettyMIDI object
            audio_data: Audio waveform
            sr: Sample rate
            **kwargs: Additional features

        Returns:
            Score in 0-100 range, or None if cannot compute
        """
        raise NotImplementedError


# ==================== NOTE ACCURACY ====================


class MIDINoteCountComplexity(LabelingFunction):
    """
    Note accuracy proxy based on MIDI note count and density.

    Assumption: More notes = higher technical difficulty = more opportunity for errors.
    Inversely correlate note density with accuracy (dense = lower accuracy).
    """

    def __init__(self):
        super().__init__(
            name="midi_note_count_complexity", dimension="note_accuracy", weight=0.7
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        # Count total notes across all instruments
        total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)

        # Duration in seconds
        duration = midi_data.get_end_time()

        if duration == 0:
            return None

        # Notes per second
        note_density = total_notes / duration

        # Heuristic: Higher density = lower expected accuracy
        # Typical range: 1-20 notes/sec for piano
        # Map to 0-100 scale (inverse relationship)
        if note_density < 1:
            score = 95.0
        elif note_density < 5:
            score = 90.0 - (note_density - 1) * 5.0
        elif note_density < 10:
            score = 70.0 - (note_density - 5) * 4.0
        elif note_density < 15:
            score = 50.0 - (note_density - 10) * 3.0
        else:
            score = 35.0

        return np.clip(score, 0, 100)


class MIDIPitchRangeComplexity(LabelingFunction):
    """
    Note accuracy proxy based on pitch range.

    Assumption: Wider pitch range = more hand movement = higher difficulty.
    """

    def __init__(self):
        super().__init__(
            name="midi_pitch_range_complexity", dimension="note_accuracy", weight=0.5
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        pitches = []
        for instrument in midi_data.instruments:
            pitches.extend([note.pitch for note in instrument.notes])

        if not pitches:
            return None

        pitch_range = max(pitches) - min(pitches)

        # Typical range: 12-60 semitones (1-5 octaves)
        # Map to 0-100 scale (inverse relationship)
        if pitch_range < 24:
            score = 95.0
        elif pitch_range < 48:
            score = 85.0 - (pitch_range - 24) * 0.5
        else:
            score = 73.0 - (pitch_range - 48) * 0.3

        return np.clip(score, 50, 100)


# ==================== RHYTHMIC PRECISION ====================


class MIDITimingVariance(LabelingFunction):
    """
    Rhythmic precision based on timing variance from grid.

    Assumption: Professional performances have small, intentional deviations.
    Large random deviations indicate poor timing.
    """

    def __init__(self):
        super().__init__(
            name="midi_timing_variance", dimension="rhythmic_precision", weight=0.8
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        # Get all note onset times
        onset_times = []
        for instrument in midi_data.instruments:
            onset_times.extend([note.start for note in instrument.notes])

        if len(onset_times) < 10:
            return None

        onset_times = np.array(sorted(onset_times))

        # Compute inter-onset intervals (IOIs)
        iois = np.diff(onset_times)

        # Estimate beat grid (assume 120 BPM = 0.5s per beat)
        median_ioi = np.median(iois)
        if median_ioi == 0:
            return None

        # Compute deviation from nearest grid point
        grid = median_ioi
        deviations = np.abs(onset_times - np.round(onset_times / grid) * grid)

        # Mean absolute deviation in milliseconds
        mad_ms = np.mean(deviations) * 1000

        # Heuristic scoring:
        # <10ms = excellent (95-100)
        # 10-30ms = good (80-95)
        # 30-50ms = moderate (60-80)
        # >50ms = poor (<60)
        if mad_ms < 10:
            score = 95.0 + (10 - mad_ms) * 0.5
        elif mad_ms < 30:
            score = 80.0 + (30 - mad_ms) * 0.75
        elif mad_ms < 50:
            score = 60.0 + (50 - mad_ms) * 1.0
        else:
            score = 60.0 - (mad_ms - 50) * 0.5

        return np.clip(score, 40, 100)


class MIDITempoStability(LabelingFunction):
    """
    Rhythmic precision based on tempo stability.

    Assumption: Good rhythm control shows consistent tempo (low variance).
    """

    def __init__(self):
        super().__init__(
            name="midi_tempo_stability", dimension="rhythmic_precision", weight=0.6
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        # Get tempo changes
        tempo_times, tempos = midi_data.get_tempo_changes()

        if len(tempos) < 2:
            # Constant tempo = perfect stability
            return 95.0

        # Coefficient of variation (CV = std / mean)
        tempo_cv = np.std(tempos) / np.mean(tempos)

        # Good performances: CV < 0.1 (10% variation)
        # Poor performances: CV > 0.3 (30% variation)
        if tempo_cv < 0.05:
            score = 95.0
        elif tempo_cv < 0.15:
            score = 85.0 - (tempo_cv - 0.05) * 100
        elif tempo_cv < 0.30:
            score = 75.0 - (tempo_cv - 0.15) * 50
        else:
            score = 67.5 - (tempo_cv - 0.30) * 25

        return np.clip(score, 50, 100)


# ==================== TONE QUALITY ====================


class AudioSpectralCentroid(LabelingFunction):
    """
    Tone quality based on spectral centroid (brightness).

    Assumption: Good tone has balanced spectral distribution.
    Too dark or too bright indicates poor tone control.
    """

    def __init__(self):
        super().__init__(
            name="audio_spectral_centroid", dimension="tone_quality", weight=0.7
        )

    def __call__(self, audio_data=None, sr=24000, **kwargs):
        if audio_data is None:
            return None

        # Compute spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        mean_centroid = np.mean(centroid)

        # Piano fundamental range: ~27Hz (A0) to ~4186Hz (C8)
        # Good tone: centroid around 1000-2000 Hz (balanced)
        # Too low (<500Hz): muddy
        # Too high (>3000Hz): harsh
        if 1000 <= mean_centroid <= 2000:
            score = 90.0 + (1500 - abs(mean_centroid - 1500)) / 50
        elif 500 <= mean_centroid < 1000:
            score = 70.0 + (mean_centroid - 500) / 25
        elif 2000 < mean_centroid <= 3000:
            score = 70.0 + (3000 - mean_centroid) / 50
        else:
            score = 60.0

        return np.clip(score, 50, 100)


class AudioInharmonicity(LabelingFunction):
    """
    Tone quality based on inharmonicity (deviation from pure harmonics).

    Assumption: Good piano tone has low inharmonicity in overtones.
    """

    def __init__(self):
        super().__init__(
            name="audio_inharmonicity", dimension="tone_quality", weight=0.5
        )

    def __call__(self, audio_data=None, sr=24000, **kwargs):
        if audio_data is None:
            return None

        # Compute spectral contrast (proxy for harmonic clarity)
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        mean_contrast = np.mean(contrast)

        # Higher contrast = clearer harmonics = better tone
        # Typical range: 15-35 dB
        if mean_contrast > 25:
            score = 85.0 + (mean_contrast - 25) * 0.5
        elif mean_contrast > 20:
            score = 75.0 + (mean_contrast - 20) * 2.0
        else:
            score = 65.0 + (mean_contrast - 15) * 2.0

        return np.clip(score, 60, 95)


# ==================== DYNAMICS CONTROL ====================


class MIDIVelocityRange(LabelingFunction):
    """
    Dynamics control based on velocity range.

    Assumption: Good dynamics shows wide, controlled range.
    Narrow range = monotone, poor control.
    """

    def __init__(self):
        super().__init__(
            name="midi_velocity_range", dimension="dynamics_control", weight=0.8
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        velocities = []
        for instrument in midi_data.instruments:
            velocities.extend([note.velocity for note in instrument.notes])

        if not velocities:
            return None

        velocity_range = max(velocities) - min(velocities)
        velocity_std = np.std(velocities)

        # MIDI velocity: 0-127
        # Good dynamics: range > 60, std > 15
        # Poor dynamics: range < 30, std < 10
        if velocity_range > 70 and velocity_std > 20:
            score = 90.0 + (velocity_range - 70) * 0.2
        elif velocity_range > 50 and velocity_std > 15:
            score = 75.0 + (velocity_range - 50) * 0.75
        elif velocity_range > 30:
            score = 60.0 + (velocity_range - 30) * 0.75
        else:
            score = 50.0 + velocity_range * 0.33

        return np.clip(score, 50, 100)


class MIDIVelocitySmoothing(LabelingFunction):
    """
    Dynamics control based on velocity transition smoothness.

    Assumption: Good dynamics shows smooth transitions (gradual changes).
    Erratic jumps indicate poor control.
    """

    def __init__(self):
        super().__init__(
            name="midi_velocity_smoothing", dimension="dynamics_control", weight=0.6
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        # Get all notes sorted by time
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        all_notes.sort(key=lambda n: n.start)

        if len(all_notes) < 10:
            return None

        velocities = np.array([note.velocity for note in all_notes])

        # Compute velocity changes (differences between consecutive notes)
        velocity_changes = np.abs(np.diff(velocities))

        # Mean absolute change
        mac = np.mean(velocity_changes)

        # Good control: MAC < 10 (smooth changes)
        # Poor control: MAC > 20 (erratic changes)
        if mac < 8:
            score = 90.0 + (8 - mac) * 1.0
        elif mac < 15:
            score = 75.0 + (15 - mac) * 2.14
        elif mac < 25:
            score = 60.0 + (25 - mac) * 1.5
        else:
            score = 50.0

        return np.clip(score, 50, 100)


# ==================== ARTICULATION ====================


class MIDINoteDurationVariance(LabelingFunction):
    """
    Articulation based on note duration variance.

    Assumption: Good articulation shows intentional variation (legato vs staccato).
    All notes same length = poor control.
    """

    def __init__(self):
        super().__init__(
            name="midi_note_duration_variance", dimension="articulation", weight=0.7
        )

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        durations = []
        for instrument in midi_data.instruments:
            durations.extend([note.end - note.start for note in instrument.notes])

        if not durations:
            return None

        # Coefficient of variation
        if np.mean(durations) == 0:
            return None

        duration_cv = np.std(durations) / np.mean(durations)

        # Good articulation: CV = 0.3-0.7 (varied but controlled)
        # Poor articulation: CV < 0.2 (monotone) or CV > 1.0 (erratic)
        if 0.3 <= duration_cv <= 0.7:
            score = 85.0 + (0.5 - abs(duration_cv - 0.5)) * 20
        elif 0.2 <= duration_cv < 0.3:
            score = 70.0 + (duration_cv - 0.2) * 150
        elif 0.7 < duration_cv <= 1.0:
            score = 70.0 - (duration_cv - 0.7) * 50
        else:
            score = 55.0

        return np.clip(score, 50, 95)


class AudioAttackTransients(LabelingFunction):
    """
    Articulation based on attack transient clarity.

    Assumption: Clear attacks = good articulation control.
    """

    def __init__(self):
        super().__init__(
            name="audio_attack_transients", dimension="articulation", weight=0.6
        )

    def __call__(self, audio_data=None, sr=24000, **kwargs):
        if audio_data is None:
            return None

        # Compute onset strength (attack clarity)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)

        # Metrics: mean and variance of onset strength
        mean_onset = np.mean(onset_env)
        std_onset = np.std(onset_env)

        # Good articulation: high mean (clear attacks) + high variance (varied)
        # Normalize to 0-100 scale
        attack_score = min(mean_onset * 10, 50)  # Mean contribution
        variance_score = min(std_onset * 5, 50)  # Variance contribution

        score = attack_score + variance_score

        return np.clip(score, 50, 100)


# ==================== PEDALING ====================


class MIDIPedalCoherence(LabelingFunction):
    """
    Pedaling technique based on CC64 (sustain pedal) coherence.

    Assumption: Good pedaling shows phrase-aligned changes (not random).
    """

    def __init__(self):
        super().__init__(name="midi_pedal_coherence", dimension="pedaling", weight=0.8)

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        # Extract CC64 (sustain pedal) events
        pedal_events = []
        for instrument in midi_data.instruments:
            for cc in instrument.control_changes:
                if cc.number == 64:  # Sustain pedal
                    pedal_events.append((cc.time, cc.value))

        if len(pedal_events) < 2:
            # No pedal usage or minimal usage
            return 70.0

        pedal_events.sort(key=lambda x: x[0])

        # Count pedal changes (transitions 0→127 or 127→0)
        pedal_changes = 0
        prev_state = 0
        for _, value in pedal_events:
            state = 1 if value > 63 else 0
            if state != prev_state:
                pedal_changes += 1
            prev_state = state

        duration = midi_data.get_end_time()
        if duration == 0:
            return None

        # Changes per minute
        changes_per_min = (pedal_changes / duration) * 60

        # Good pedaling: 10-30 changes per minute (phrase-aligned)
        # Poor: <5 (too static) or >50 (too frequent/random)
        if 10 <= changes_per_min <= 30:
            score = 85.0 + (20 - abs(changes_per_min - 20)) * 0.5
        elif 5 <= changes_per_min < 10:
            score = 70.0 + (changes_per_min - 5) * 3.0
        elif 30 < changes_per_min <= 50:
            score = 70.0 - (changes_per_min - 30) * 0.75
        else:
            score = 55.0

        return np.clip(score, 50, 95)


class MIDIPedalTiming(LabelingFunction):
    """
    Pedaling technique based on timing accuracy (alignment with notes).

    Assumption: Good pedaling changes align with phrase boundaries.
    """

    def __init__(self):
        super().__init__(name="midi_pedal_timing", dimension="pedaling", weight=0.6)

    def __call__(self, midi_data=None, **kwargs):
        if midi_data is None:
            return None

        # Get pedal change times
        pedal_times = []
        for instrument in midi_data.instruments:
            for cc in instrument.control_changes:
                if cc.number == 64:
                    pedal_times.append(cc.time)

        # Get note onset times
        onset_times = []
        for instrument in midi_data.instruments:
            onset_times.extend([note.start for note in instrument.notes])

        if len(pedal_times) < 2 or len(onset_times) < 10:
            return 75.0

        pedal_times = np.array(sorted(pedal_times))
        onset_times = np.array(sorted(onset_times))

        # For each pedal change, find distance to nearest note onset
        min_distances = []
        for pedal_time in pedal_times:
            distances = np.abs(onset_times - pedal_time)
            min_distances.append(np.min(distances))

        # Mean distance in milliseconds
        mean_dist_ms = np.mean(min_distances) * 1000

        # Good timing: <50ms from nearest note
        # Poor timing: >200ms (not aligned with phrases)
        if mean_dist_ms < 30:
            score = 90.0 + (30 - mean_dist_ms) * 0.33
        elif mean_dist_ms < 100:
            score = 75.0 + (100 - mean_dist_ms) * 0.21
        elif mean_dist_ms < 200:
            score = 60.0 + (200 - mean_dist_ms) * 0.15
        else:
            score = 55.0

        return np.clip(score, 50, 100)


# ==================== MUSICAL EXPRESSION ====================


class MIDIPhrasingCoherence(LabelingFunction):
    """
    Musical expression based on phrasing coherence.

    Assumption: Good musical expression shows phrase-level structure
    (grouping of notes with dynamic and timing contours).
    """

    def __init__(self):
        super().__init__(
            name="midi_phrasing_coherence", dimension="musical_expression", weight=0.7
        )

    def __call__(self, midi_data=None, audio_data=None, sr=24000, **kwargs):
        if midi_data is None:
            return None

        # Get all notes sorted by time
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        all_notes.sort(key=lambda n: n.start)

        if len(all_notes) < 20:
            return None

        # Analyze velocity and timing patterns for phrase structure
        velocities = np.array([note.velocity for note in all_notes])
        durations = np.array([note.end - note.start for note in all_notes])

        # Look for phrase-level patterns (alternating crescendo/decrescendo)
        # Compute local trends using moving average
        window_size = min(10, len(velocities) // 4)
        if window_size < 3:
            return 75.0

        velocity_trends = np.convolve(
            velocities, np.ones(window_size) / window_size, mode="valid"
        )

        # Count direction changes (phrase boundaries)
        direction_changes = np.sum(np.diff(np.sign(np.diff(velocity_trends))) != 0)

        # Good phrasing: 5-15 phrase boundaries per 100 notes
        phrase_density = direction_changes / len(velocity_trends) * 100

        if 5 <= phrase_density <= 15:
            score = 85.0 + (10 - abs(phrase_density - 10)) * 1.0
        elif 3 <= phrase_density < 5:
            score = 70.0 + (phrase_density - 3) * 7.5
        elif 15 < phrase_density <= 25:
            score = 70.0 - (phrase_density - 15) * 1.5
        else:
            score = 60.0

        return np.clip(score, 55, 95)


class AudioDynamicContour(LabelingFunction):
    """
    Musical expression based on audio dynamic contour.

    Assumption: Good expression shows smooth, intentional dynamic shaping
    over time (not flat or erratic).
    """

    def __init__(self):
        super().__init__(
            name="audio_dynamic_contour", dimension="musical_expression", weight=0.6
        )

    def __call__(self, audio_data=None, sr=24000, **kwargs):
        if audio_data is None:
            return None

        # Compute RMS energy in frames
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(
            y=audio_data, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Smooth RMS to get dynamic contour
        smoothed_rms = np.convolve(rms, np.ones(20) / 20, mode="same")

        # Measure variance (too flat = poor expression, too erratic = poor control)
        rms_cv = (
            np.std(smoothed_rms) / np.mean(smoothed_rms)
            if np.mean(smoothed_rms) > 0
            else 0
        )

        # Good expression: CV = 0.3-0.6 (moderate variance)
        if 0.3 <= rms_cv <= 0.6:
            score = 85.0 + (0.45 - abs(rms_cv - 0.45)) * 30
        elif 0.2 <= rms_cv < 0.3:
            score = 70.0 + (rms_cv - 0.2) * 150
        elif 0.6 < rms_cv <= 0.8:
            score = 70.0 - (rms_cv - 0.6) * 50
        else:
            score = 60.0

        return np.clip(score, 55, 95)


# ==================== OVERALL INTERPRETATION ====================


class CombinedPerformanceQuality(LabelingFunction):
    """
    Overall interpretation based on combined technical and expressive metrics.

    Holistic assessment that combines multiple aspects of performance quality.
    """

    def __init__(self):
        super().__init__(
            name="combined_performance_quality",
            dimension="overall_interpretation",
            weight=0.8,
        )

    def __call__(self, midi_data=None, audio_data=None, sr=24000, **kwargs):
        if midi_data is None or audio_data is None:
            return None

        # Combine multiple quality indicators
        scores = []

        # 1. Tempo stability (rhythmic control)
        tempo_times, tempos = midi_data.get_tempo_changes()
        if len(tempos) >= 2:
            tempo_cv = np.std(tempos) / np.mean(tempos)
            tempo_score = 90.0 if tempo_cv < 0.1 else 70.0
            scores.append(tempo_score)

        # 2. Dynamic range (expressive control)
        velocities = []
        for instrument in midi_data.instruments:
            velocities.extend([note.velocity for note in instrument.notes])
        if velocities:
            velocity_range = max(velocities) - min(velocities)
            dynamic_score = min(90.0, 50.0 + velocity_range * 0.5)
            scores.append(dynamic_score)

        # 3. Audio quality (tone and clarity)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
        contrast_score = min(90.0, 60.0 + np.mean(spectral_contrast))
        scores.append(contrast_score)

        # Average all indicators
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 75.0

        return np.clip(overall_score, 60, 95)


class AudioTemporalCoherence(LabelingFunction):
    """
    Overall interpretation based on temporal coherence.

    Measures consistency and flow of the performance over time.
    """

    def __init__(self):
        super().__init__(
            name="audio_temporal_coherence",
            dimension="overall_interpretation",
            weight=0.6,
        )

    def __call__(self, audio_data=None, sr=24000, **kwargs):
        if audio_data is None:
            return None

        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)

        # Compute autocorrelation to measure periodicity/coherence
        onset_autocorr = librosa.autocorrelate(onset_env)

        # Strong autocorrelation = good temporal structure
        # Weak autocorrelation = incoherent/random
        autocorr_strength = np.max(onset_autocorr[1 : len(onset_autocorr) // 4])

        # Normalize and score
        if autocorr_strength > 100:
            score = 90.0
        elif autocorr_strength > 50:
            score = 75.0 + (autocorr_strength - 50) * 0.3
        else:
            score = 60.0 + autocorr_strength * 0.3

        return np.clip(score, 60, 95)


# ==================== LABELING FUNCTION REGISTRY ====================


def get_all_labeling_functions() -> Dict[str, list]:
    """
    Get all labeling functions organized by dimension.

    Updated for TRAINING_PLAN_v2.md with 8 dimensions:
    - Technical (4): note_accuracy, rhythmic_stability, articulation_clarity, pedal_technique
    - Timbre/Dynamics (2): tone_quality, dynamic_range
    - Interpretive (2): musical_expression, overall_interpretation

    Returns:
        Dictionary mapping dimension names to lists of labeling functions
    """
    return {
        "note_accuracy": [
            MIDINoteCountComplexity(),
            MIDIPitchRangeComplexity(),
        ],
        "rhythmic_stability": [  # Renamed from rhythmic_precision
            MIDITimingVariance(),
            MIDITempoStability(),
        ],
        "articulation_clarity": [  # Renamed from articulation
            MIDINoteDurationVariance(),
            AudioAttackTransients(),
        ],
        "pedal_technique": [  # Renamed from pedaling
            MIDIPedalCoherence(),
            MIDIPedalTiming(),
        ],
        "tone_quality": [
            AudioSpectralCentroid(),
            AudioInharmonicity(),
        ],
        "dynamic_range": [  # Renamed from dynamics_control
            MIDIVelocityRange(),
            MIDIVelocitySmoothing(),
        ],
        "musical_expression": [  # NEW
            MIDIPhrasingCoherence(),
            AudioDynamicContour(),
        ],
        "overall_interpretation": [  # NEW
            CombinedPerformanceQuality(),
            AudioTemporalCoherence(),
        ],
    }


if __name__ == "__main__":
    print("Labeling functions module loaded successfully")
    print("\nAvailable labeling functions:")
    for dim, funcs in get_all_labeling_functions().items():
        print(f"\n{dim}:")
        for func in funcs:
            print(f"  - {func.name} (weight={func.weight})")
