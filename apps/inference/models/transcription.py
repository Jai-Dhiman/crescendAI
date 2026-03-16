"""ByteDance AMT (Automatic Music Transcription) wrapper.

Transcribes audio to MIDI using ByteDance's piano transcription model,
then parses the MIDI into a structured note list for downstream consumers.

    REQUEST FLOW:
    +------------------+     +------------------+     +------------------+
    | numpy audio      | --> | ByteDance AMT    | --> | pretty_midi      |
    | (24kHz float32)  |     | -> temp MIDI     |     | -> note list     |
    +------------------+     +------------------+     +------------------+
                                                             |
                                                             v
                                                      [{pitch, onset,
                                                        offset, velocity}]

Audio comes in at 24kHz (MuQ pipeline sample rate). ByteDance resamples
internally to 16kHz. Onset/offset timestamps are in seconds relative to
the original audio duration.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
from piano_transcription_inference import PianoTranscription


class TranscriptionError(Exception):
    """Raised when AMT transcription fails."""

    pass


class TranscriptionModel:
    """Wrapper around ByteDance PianoTranscription for inference endpoint use."""

    def __init__(self, device: str = "cuda"):
        """Load ByteDance transcription model.

        Args:
            device: "cuda" for GPU, "cpu" for CPU. Weights must be
                    pre-downloaded in Dockerfile for production use.
        """
        print(f"Loading ByteDance PianoTranscription on {device}...")
        load_start = time.time()
        self._transcriber = PianoTranscription(device=device)
        print(f"PianoTranscription loaded in {time.time() - load_start:.1f}s")

    def transcribe(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Transcribe audio to a sorted list of MIDI notes and pedal events.

        Args:
            audio: Mono float32 audio array (any sample rate -- ByteDance
                   resamples internally to 16kHz).
            sample_rate: Sample rate of the input audio.

        Returns:
            Tuple of (notes, pedal_events):
            - notes: List of notes sorted by onset time:
              [{"pitch": 60, "onset": 0.12, "offset": 0.45, "velocity": 78}, ...]
              Returns [] for silent audio (no notes detected).
            - pedal_events: List of CC64 sustain pedal events sorted by time:
              [{"time": 0.10, "value": 127}, {"time": 0.85, "value": 0}, ...]
              Returns [] if no pedal events are present.

        Raises:
            TranscriptionError: If transcription or MIDI parsing fails.
        """
        transcribe_start = time.time()
        temp_dir = tempfile.mkdtemp(prefix="amt_")

        try:
            midi_path = Path(temp_dir) / "transcription.mid"

            # ByteDance API: transcribe(audio_array, midi_output_path)
            print("Running ByteDance AMT transcription...")
            self._transcriber.transcribe(audio, str(midi_path))

            if not midi_path.exists():
                raise TranscriptionError("Transcriber did not produce a MIDI file")

            # Parse MIDI with pretty_midi
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            notes = []
            pedal_events = []
            for instrument in midi.instruments:
                for note in instrument.notes:
                    notes.append(
                        {
                            "pitch": int(note.pitch),
                            "onset": round(float(note.start), 4),
                            "offset": round(float(note.end), 4),
                            "velocity": int(note.velocity),
                        }
                    )
                for cc in instrument.control_changes:
                    if cc.number == 64:  # Sustain pedal only
                        pedal_events.append(
                            {
                                "time": round(float(cc.time), 4),
                                "value": int(cc.value),
                            }
                        )

            # Sort by onset time, then by pitch for simultaneous notes
            notes.sort(key=lambda n: (n["onset"], n["pitch"]))
            pedal_events.sort(key=lambda e: e["time"])

            elapsed_ms = int((time.time() - transcribe_start) * 1000)
            print(
                f"AMT complete: {len(notes)} notes, {len(pedal_events)} pedal events in {elapsed_ms}ms"
            )

            return notes, pedal_events

        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e
        finally:
            # Clean up temp directory (runs on success AND exception)
            shutil.rmtree(temp_dir, ignore_errors=True)
