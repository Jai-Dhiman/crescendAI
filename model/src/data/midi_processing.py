import numpy as np
import mido
import pretty_midi
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional


class OctupleMIDITokenizer:
    """
    Tokenizer for OctupleMIDI representation.

    Each MIDI event is represented as an 8-tuple:
    1. Note type (note-on, note-off, time-shift)
    2. Beat (metrical position)
    3. Position (position within beat)
    4. Pitch (MIDI note number, 21-108 for piano)
    5. Duration (note length in ticks)
    6. Velocity (attack velocity, 0-127)
    7. Instrument (always 0 for piano)
    8. Bar (measure number)
    """

    def __init__(self, max_pitch: int = 108, min_pitch: int = 21):
        self.max_pitch = max_pitch
        self.min_pitch = min_pitch

        # Vocabulary sizes for each dimension
        self.vocab_sizes = {
            'type': 5,  # note-on, note-off, time-shift, pedal-on, pedal-off
            'beat': 16,  # up to 16 beats per measure
            'position': 16,  # 16 subdivisions per beat
            'pitch': max_pitch - min_pitch + 1,  # 88 piano keys
            'duration': 128,  # quantized durations
            'velocity': 128,  # MIDI velocity range
            'instrument': 1,  # piano only
            'bar': 512,  # up to 512 measures
        }

    def encode(self, midi_data: pretty_midi.PrettyMIDI) -> np.ndarray:
        """
        Encode MIDI data into OctupleMIDI tokens.

        Args:
            midi_data: PrettyMIDI object

        Returns:
            Array of shape [num_events, 8]
        """
        events = []

        # Get tempo changes
        tempo_changes = midi_data.get_tempo_changes()

        # Process each instrument (typically just one for piano)
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue

            # Sort notes by start time
            notes = sorted(instrument.notes, key=lambda x: x.start)

            for note in notes:
                # Calculate beat and bar information
                beat, position = self._time_to_beat(note.start, tempo_changes, midi_data)
                bar = int(beat // 4)  # Assuming 4/4 time
                beat_in_bar = beat % 4

                # Quantize duration
                duration_ticks = int((note.end - note.start) * 100)  # Rough quantization
                duration_ticks = min(duration_ticks, 127)

                # Create note-on event
                event = [
                    0,  # type: note-on
                    int(beat_in_bar),
                    int(position),
                    note.pitch,
                    duration_ticks,
                    note.velocity,
                    0,  # instrument: piano
                    bar,
                ]
                events.append(event)

        return np.array(events, dtype=np.int32)

    def _time_to_beat(
        self,
        time: float,
        tempo_changes: Tuple[np.ndarray, np.ndarray],
        midi_data: pretty_midi.PrettyMIDI
    ) -> Tuple[float, float]:
        """
        Convert time in seconds to beat and position.

        PRODUCTION: Properly handles tempo changes throughout the piece.
        """
        tempo_times, tempos = tempo_changes

        # Ensure arrays are properly shaped (flatten if needed)
        tempo_times = np.atleast_1d(tempo_times).flatten()
        tempos = np.atleast_1d(tempos).flatten()

        # If no tempo changes, use default
        if len(tempos) == 0:
            tempo = 120.0
            beat = time * (tempo / 60.0)
        else:
            # Find the appropriate tempo for this time
            # tempo_times are in seconds, tempos are in BPM
            beat = 0.0
            current_time = 0.0

            for i in range(len(tempo_times)):
                # Get tempo for this segment
                tempo = tempos[i]

                # Determine segment end time
                if i < len(tempo_times) - 1:
                    segment_end = tempo_times[i + 1]
                else:
                    segment_end = time  # Last segment extends to our target time

                # If our target time is before this segment starts, we're done
                if i > 0 and time < tempo_times[i]:
                    break

                # Calculate how much time to advance in this segment
                if time <= segment_end:
                    # Target time is in this segment
                    time_in_segment = time - current_time
                    beat += time_in_segment * (tempo / 60.0)
                    break
                else:
                    # Target time is after this segment, accumulate full segment
                    time_in_segment = segment_end - current_time
                    beat += time_in_segment * (tempo / 60.0)
                    current_time = segment_end

        # Calculate position within beat (16 subdivisions per beat)
        position = (beat % 1.0) * 16

        return beat, position


def load_midi(path: Union[str, Path]) -> pretty_midi.PrettyMIDI:
    """
    Load MIDI file using pretty_midi.

    Args:
        path: Path to MIDI file

    Returns:
        PrettyMIDI object
    """
    return pretty_midi.PrettyMIDI(str(path))


def align_midi_to_audio(
    midi: pretty_midi.PrettyMIDI,
    audio_duration: float,
    time_stretch_ratio: float = 1.0,
) -> pretty_midi.PrettyMIDI:
    """
    Align MIDI timing to match audio duration.

    Args:
        midi: PrettyMIDI object
        audio_duration: Target audio duration in seconds
        time_stretch_ratio: Ratio to stretch MIDI timing

    Returns:
        Aligned PrettyMIDI object
    """
    midi_duration = midi.get_end_time()

    if abs(midi_duration - audio_duration) < 0.1:
        return midi  # Already aligned

    # Calculate stretch ratio
    ratio = audio_duration / midi_duration * time_stretch_ratio

    # Adjust all note times
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.start *= ratio
            note.end *= ratio

    return midi


def extract_midi_features(midi: pretty_midi.PrettyMIDI) -> Dict[str, np.ndarray]:
    """
    Extract basic features from MIDI for pseudo-labeling.

    Features:
    - Note onset times
    - Note velocities
    - Pedal events (CC64)
    - Pitch range

    Args:
        midi: PrettyMIDI object

    Returns:
        Dictionary of features
    """
    features = {}

    # Collect all notes
    all_notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    # Sort by start time
    all_notes = sorted(all_notes, key=lambda x: x.start)

    # Extract onset times and velocities
    onset_times = np.array([note.start for note in all_notes])
    velocities = np.array([note.velocity for note in all_notes])
    pitches = np.array([note.pitch for note in all_notes])

    features['onset_times'] = onset_times
    features['velocities'] = velocities
    features['pitches'] = pitches

    # Extract pedal events
    pedal_events = []
    for instrument in midi.instruments:
        for control_change in instrument.control_changes:
            if control_change.number == 64:  # Sustain pedal
                pedal_events.append((control_change.time, control_change.value))

    if pedal_events:
        features['pedal_times'] = np.array([t for t, v in pedal_events])
        features['pedal_values'] = np.array([v for t, v in pedal_events])
    else:
        features['pedal_times'] = np.array([])
        features['pedal_values'] = np.array([])

    # Calculate statistics
    features['pitch_range'] = pitches.max() - pitches.min() if len(pitches) > 0 else 0
    features['velocity_mean'] = velocities.mean() if len(velocities) > 0 else 0
    features['velocity_std'] = velocities.std() if len(velocities) > 0 else 0
    features['note_count'] = len(all_notes)
    features['duration'] = midi.get_end_time()

    return features


def segment_midi(
    midi: pretty_midi.PrettyMIDI,
    segment_times: List[Tuple[float, float]],
) -> List[pretty_midi.PrettyMIDI]:
    """
    Segment MIDI data to match audio segments.

    Args:
        midi: PrettyMIDI object
        segment_times: List of (start_time, end_time) tuples

    Returns:
        List of PrettyMIDI objects for each segment
    """
    segments = []

    for start_time, end_time in segment_times:
        # Create new MIDI object for segment
        segment_midi = pretty_midi.PrettyMIDI()

        # Copy tempo information
        segment_midi.time_signature_changes = [
            ts for ts in midi.time_signature_changes
            if start_time <= ts.time < end_time
        ]

        # Copy notes within time range
        for instrument in midi.instruments:
            segment_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )

            for note in instrument.notes:
                # Include notes that overlap with segment
                if note.start < end_time and note.end > start_time:
                    # Adjust timing relative to segment start
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=max(0, note.start - start_time),
                        end=min(end_time - start_time, note.end - start_time)
                    )
                    segment_instrument.notes.append(new_note)

            # Copy pedal events
            for cc in instrument.control_changes:
                if start_time <= cc.time < end_time:
                    new_cc = pretty_midi.ControlChange(
                        number=cc.number,
                        value=cc.value,
                        time=cc.time - start_time
                    )
                    segment_instrument.control_changes.append(new_cc)

            segment_midi.instruments.append(segment_instrument)

        segments.append(segment_midi)

    return segments


def encode_octuple_midi(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    """
    Encode MIDI into OctupleMIDI token sequence.

    Args:
        midi: PrettyMIDI object

    Returns:
        Array of shape [num_events, 8] with token indices
    """
    tokenizer = OctupleMIDITokenizer()
    return tokenizer.encode(midi)


if __name__ == "__main__":
    print("MIDI processing module loaded successfully")
    print("Features:")
    print("- OctupleMIDI tokenization (8-dimensional representation)")
    print("- MIDI-audio alignment")
    print("- Feature extraction for pseudo-labeling")
    print("- Segmentation to match audio windows")
