"""Audio to MIDI transcription using basic-pitch."""

from typing import Optional, Tuple

import numpy as np
import pretty_midi


class TranscriptionError(Exception):
    """Raised when MIDI transcription fails."""

    pass


def transcribe_audio_to_midi(
    audio: np.ndarray,
    sr: int = 24000,
) -> Tuple[pretty_midi.PrettyMIDI, dict]:
    """Transcribe audio to MIDI using basic-pitch.

    Args:
        audio: Audio waveform
        sr: Sample rate of audio

    Returns:
        Tuple of (PrettyMIDI object, note_events dict)

    Raises:
        TranscriptionError: If transcription fails
    """
    try:
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        # Run basic-pitch inference
        model_output, midi_data, note_events = predict(
            audio,
            sr,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
        )

        return midi_data, note_events

    except ImportError:
        raise TranscriptionError("basic-pitch not installed")
    except Exception as e:
        raise TranscriptionError(f"MIDI transcription failed: {e}")


def extract_midi_features(
    midi_data: pretty_midi.PrettyMIDI,
) -> dict:
    """Extract features from MIDI for symbolic model.

    Args:
        midi_data: PrettyMIDI object from transcription

    Returns:
        Dict of extracted features
    """
    features = {
        "num_notes": 0,
        "duration": 0.0,
        "mean_pitch": 0.0,
        "pitch_range": 0,
        "mean_velocity": 0.0,
        "velocity_range": 0,
        "notes_per_second": 0.0,
    }

    # Collect all notes from all instruments
    all_notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    if not all_notes:
        return features

    # Extract basic statistics
    pitches = [n.pitch for n in all_notes]
    velocities = [n.velocity for n in all_notes]
    durations = [n.end - n.start for n in all_notes]

    features["num_notes"] = len(all_notes)
    features["duration"] = midi_data.get_end_time()
    features["mean_pitch"] = np.mean(pitches)
    features["pitch_range"] = max(pitches) - min(pitches)
    features["mean_velocity"] = np.mean(velocities)
    features["velocity_range"] = max(velocities) - min(velocities)
    features["mean_duration"] = np.mean(durations)

    if features["duration"] > 0:
        features["notes_per_second"] = features["num_notes"] / features["duration"]

    return features
