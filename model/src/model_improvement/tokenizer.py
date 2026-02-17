"""REMI MIDI tokenizer and continuous feature extraction for symbolic encoders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import miditok
from symusic import Score


class PianoTokenizer:
    """REMI tokenizer for piano MIDI, wrapping miditok.

    Configures a REMI tokenizer with piano-specific settings:
    - Full piano pitch range (21-108, A0-C8)
    - 32 velocity bins
    - Tempo tracking enabled
    - Special tokens: PAD, BOS, EOS, MASK
    """

    def __init__(self, max_seq_len: int = 2048) -> None:
        self.max_seq_len = max_seq_len
        config = miditok.TokenizerConfig(
            pitch_range=(21, 109),
            num_velocities=32,
            use_velocities=True,
            use_tempos=True,
            num_tempos=32,
            tempo_range=(40, 250),
            special_tokens=["PAD", "BOS", "EOS", "MASK"],
        )
        self._tokenizer = miditok.REMI(tokenizer_config=config)

    def encode(self, midi_path: Path) -> list[int]:
        """Encode a MIDI file into a list of token IDs.

        Args:
            midi_path: Path to the MIDI file.

        Returns:
            List of integer token IDs.

        Raises:
            FileNotFoundError: If the MIDI file does not exist.
            ValueError: If the MIDI file produces no tokens.
        """
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")

        score = Score(str(midi_path))
        tok_sequences = self._tokenizer.encode(score)

        if not tok_sequences:
            raise ValueError(f"No tokens produced from MIDI file: {midi_path}")

        # REMI returns a list of TokSequence (one per instrument track).
        # For piano, we take the first (and typically only) track.
        ids = tok_sequences[0].ids

        # Truncate to max_seq_len if needed
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]

        return ids

    def decode(self, tokens: list[int], output_path: Path) -> None:
        """Decode token IDs back to a MIDI file.

        Args:
            tokens: List of integer token IDs.
            output_path: Path where the decoded MIDI file will be written.

        Raises:
            ValueError: If the token list is empty.
        """
        if not tokens:
            raise ValueError("Cannot decode an empty token list.")

        output_path = Path(output_path)

        # miditok REMI expects a 2D list: [instrument_track][tokens]
        score = self._tokenizer.decode([tokens])
        score.dump_midi(str(output_path))

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        """Token ID for padding."""
        return self._tokenizer.pad_token_id

    @property
    def mask_token_id(self) -> int:
        """Token ID for masking (used in MLM-style pretraining)."""
        # MASK is the 4th special token (index 3): PAD=0, BOS=1, EOS=2, MASK=3
        mask_ids = [
            tid
            for token_str, tid in self._tokenizer.vocab.items()
            if token_str.startswith("MASK")
        ]
        if not mask_ids:
            raise ValueError("MASK token not found in tokenizer vocabulary.")
        return mask_ids[0]


def extract_continuous_features(
    midi_path: Path, frame_rate: int = 50
) -> np.ndarray:
    """Extract continuous feature curves from MIDI for the S3 experiment.

    Computes frame-level features from a MIDI file at the given sample rate.

    Features (per frame):
        0: weighted average pitch (normalized to [0, 1] over piano range)
        1: weighted average velocity (normalized to [0, 1])
        2: note density (number of active notes)
        3: pedal status (1.0 if sustain pedal is active, 0.0 otherwise)
        4: inter-onset interval (seconds since last note onset)

    Args:
        midi_path: Path to the MIDI file.
        frame_rate: Number of frames per second.

    Returns:
        numpy array of shape [T, 5] where T = ceil(duration * frame_rate).

    Raises:
        FileNotFoundError: If the MIDI file does not exist.
        ValueError: If the MIDI file has zero duration.
    """
    import pretty_midi

    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    duration = pm.get_end_time()
    if duration <= 0:
        raise ValueError(f"MIDI file has zero or negative duration: {midi_path}")

    num_frames = int(np.ceil(duration * frame_rate))
    frame_times = np.arange(num_frames) / frame_rate

    # Feature dimensions: pitch, velocity, density, pedal, IOI
    num_features = 5
    features = np.zeros((num_frames, num_features), dtype=np.float32)

    # Piano pitch range for normalization
    pitch_min, pitch_max = 21, 108

    # Collect all notes across instruments
    all_notes = []
    for instrument in pm.instruments:
        all_notes.extend(instrument.notes)

    if not all_notes:
        return features

    # Build piano roll: (128, T) with velocity values -- O(N) instead of O(T*N)
    piano_roll = np.zeros((128, num_frames), dtype=np.float32)
    for note in all_notes:
        s = max(0, int(note.start * frame_rate))
        e = min(num_frames, int(np.ceil(note.end * frame_rate)))
        if s < e:
            piano_roll[note.pitch, s:e] = note.velocity

    # Density: number of active notes per frame
    active = piano_roll > 0
    density = active.sum(axis=0).astype(np.float32)
    has_notes = density > 0

    # Average velocity (normalized to [0, 1])
    vel_sum = piano_roll.sum(axis=0)
    features[has_notes, 1] = vel_sum[has_notes] / density[has_notes] / 127.0

    # Average pitch (normalized to [0, 1] over piano range)
    pitch_indices = np.arange(128, dtype=np.float32).reshape(-1, 1)
    pitch_sum = (active * pitch_indices).sum(axis=0).astype(np.float32)
    avg_pitch = np.zeros(num_frames, dtype=np.float32)
    avg_pitch[has_notes] = pitch_sum[has_notes] / density[has_notes]
    features[:, 0] = np.clip(
        (avg_pitch - pitch_min) / (pitch_max - pitch_min), 0.0, 1.0
    )

    features[:, 2] = density

    # Pedal status via searchsorted -- O(T) instead of O(T*P)
    pedal_events: list[tuple[float, int]] = []
    for instrument in pm.instruments:
        for cc in instrument.control_changes:
            if cc.number == 64:
                pedal_events.append((cc.time, cc.value))
    pedal_events.sort(key=lambda x: x[0])

    if pedal_events:
        pedal_times = np.array([t for t, _ in pedal_events])
        pedal_values = np.array(
            [1.0 if v >= 64 else 0.0 for _, v in pedal_events]
        )
        indices = np.searchsorted(pedal_times, frame_times, side="right") - 1
        valid = indices >= 0
        features[valid, 3] = pedal_values[indices[valid]]

    # IOI via searchsorted -- O(T) instead of O(T*O)
    onset_arr = np.array(sorted(set(note.start for note in all_notes)))
    if len(onset_arr) > 1:
        indices = np.searchsorted(onset_arr, frame_times, side="right") - 1
        valid = indices >= 1
        features[valid, 4] = onset_arr[indices[valid]] - onset_arr[indices[valid] - 1]

    return features
