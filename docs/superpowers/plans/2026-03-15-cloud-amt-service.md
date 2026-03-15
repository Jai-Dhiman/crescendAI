# Cloud AMT Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ByteDance piano transcription to the HF inference endpoint so every request returns both MuQ quality scores and a parsed MIDI note list.

**Architecture:** Extend the existing `handler.py` with a new `TranscriptionModel` in `models/transcription.py`. Sequential execution: MuQ scoring (~0.4s) then AMT transcription (~0.7s). Graceful degradation: AMT failure preserves MuQ scores.

**Tech Stack:** Python, piano-transcription-inference (ByteDance), pretty-midi, HuggingFace Inference Endpoints (Docker, CUDA)

**Design spec:** `docs/superpowers/specs/2026-03-15-cloud-amt-service-design.md`

---

## File Structure

```
apps/inference/
  models/
    transcription.py    # NEW  -- TranscriptionModel + TranscriptionError
    loader.py           # MODIFY -- add transcription model to cache
    __init__.py         # unchanged
    inference.py        # unchanged
    calibration.py      # unchanged
  handler.py            # MODIFY -- call AMT after MuQ, build combined response
  requirements.txt      # MODIFY -- add piano-transcription-inference, pretty-midi
  Dockerfile            # MODIFY -- pre-download AMT weights
  tests/
    test_transcription.py  # NEW -- unit tests for transcription module
```

---

## Chunk 1: Transcription Module + Tests

### Task 1: TranscriptionModel

**Files:**
- Create: `apps/inference/models/transcription.py`
- Create: `apps/inference/tests/test_transcription.py`

- [ ] **Step 1: Write failing tests for the transcription module**

```python
# apps/inference/tests/test_transcription.py
"""Tests for the ByteDance AMT transcription module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def test_transcribe_returns_sorted_note_list():
    """Transcription of audio returns notes sorted by onset."""
    from models.transcription import TranscriptionModel

    # Create a mock that simulates ByteDance writing a MIDI file
    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")

        # Mock the transcribe method to write a simple MIDI
        def fake_transcribe(audio, midi_path):
            # Write a minimal MIDI file using pretty_midi
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)
            inst.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=0.9))
            inst.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=0.1, end=0.4))
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=0.3, end=0.7))
            pm.instruments.append(inst)
            pm.write(str(midi_path))

        model._transcriber.transcribe = fake_transcribe

        audio = np.zeros(24000, dtype=np.float32)  # 1s silence at 24kHz
        notes = model.transcribe(audio, sample_rate=24000)

        assert len(notes) == 3
        # Sorted by onset
        assert notes[0]["onset"] == pytest.approx(0.1, abs=0.01)
        assert notes[1]["onset"] == pytest.approx(0.3, abs=0.01)
        assert notes[2]["onset"] == pytest.approx(0.5, abs=0.01)
        # Check all fields present
        for note in notes:
            assert "pitch" in note
            assert "onset" in note
            assert "offset" in note
            assert "velocity" in note


def test_transcribe_empty_audio_returns_empty_list():
    """Silent audio produces an empty note list (not null)."""
    from models.transcription import TranscriptionModel

    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")

        def fake_transcribe(audio, midi_path):
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            pm.instruments.append(pretty_midi.Instrument(program=0))
            pm.write(str(midi_path))

        model._transcriber.transcribe = fake_transcribe

        audio = np.zeros(24000 * 15, dtype=np.float32)
        notes = model.transcribe(audio, sample_rate=24000)

        assert notes == []
        assert isinstance(notes, list)


def test_transcribe_error_raises_transcription_error():
    """If ByteDance transcriber fails, TranscriptionError is raised."""
    from models.transcription import TranscriptionModel, TranscriptionError

    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")
        model._transcriber.transcribe = MagicMock(side_effect=RuntimeError("GPU OOM"))

        audio = np.zeros(24000, dtype=np.float32)
        with pytest.raises(TranscriptionError, match="GPU OOM"):
            model.transcribe(audio, sample_rate=24000)


def test_transcribe_cleans_up_temp_files():
    """Temp directory is cleaned up even on success."""
    from models.transcription import TranscriptionModel

    with patch("models.transcription.PianoTranscription") as mock_pt:
        model = TranscriptionModel(device="cpu")

        temp_dirs_created = []
        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            temp_dirs_created.append(d)
            return d

        def fake_transcribe(audio, midi_path):
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            pm.instruments.append(pretty_midi.Instrument(program=0))
            pm.write(str(midi_path))

        model._transcriber.transcribe = fake_transcribe

        with patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            audio = np.zeros(24000, dtype=np.float32)
            model.transcribe(audio, sample_rate=24000)

        # Temp dir should have been cleaned up
        for d in temp_dirs_created:
            assert not Path(d).exists(), f"Temp dir not cleaned up: {d}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/inference && python -m pytest tests/test_transcription.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.transcription'`

- [ ] **Step 3: Write the transcription module**

```python
# apps/inference/models/transcription.py
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

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> list[dict[str, Any]]:
        """Transcribe audio to a sorted list of MIDI notes.

        Args:
            audio: Mono float32 audio array (any sample rate -- ByteDance
                   resamples internally to 16kHz).
            sample_rate: Sample rate of the input audio.

        Returns:
            List of notes sorted by onset time:
            [{"pitch": 60, "onset": 0.12, "offset": 0.45, "velocity": 78}, ...]
            Returns [] for silent audio (no notes detected).

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
            for instrument in midi.instruments:
                for note in instrument.notes:
                    notes.append({
                        "pitch": int(note.pitch),
                        "onset": round(float(note.start), 4),
                        "offset": round(float(note.end), 4),
                        "velocity": int(note.velocity),
                    })

            # Sort by onset time, then by pitch for simultaneous notes
            notes.sort(key=lambda n: (n["onset"], n["pitch"]))

            elapsed_ms = int((time.time() - transcribe_start) * 1000)
            print(f"AMT complete: {len(notes)} notes in {elapsed_ms}ms")

            return notes

        except TranscriptionError:
            raise
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e
        finally:
            # Clean up temp directory (runs on success AND exception)
            shutil.rmtree(temp_dir, ignore_errors=True)
```

- [ ] **Step 4: Install dependencies for local testing**

Run: `cd apps/inference && pip install pretty-midi piano-transcription-inference`
(Or if using the project venv: `uv pip install pretty-midi piano-transcription-inference`)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd apps/inference && python -m pytest tests/test_transcription.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add apps/inference/models/transcription.py apps/inference/tests/test_transcription.py
git commit -m "feat(inference): add TranscriptionModel for ByteDance AMT"
```

---

## Chunk 2: Handler Integration + Deployment Config

### Task 2: Integrate AMT into handler.py

**Files:**
- Modify: `apps/inference/handler.py`

- [ ] **Step 1: Add transcription import and initialization**

Add to the imports at the top of `handler.py`:

```python
from models.transcription import TranscriptionModel, TranscriptionError
```

Add to `EndpointHandler.__init__`, after the MuQ cache initialization (after line 66):

```python
        # Initialize AMT transcription model
        print("Loading ByteDance AMT model...")
        self._transcription = TranscriptionModel(device="cuda")
```

- [ ] **Step 2: Add AMT call and combined response in `__call__`**

Replace the "Build response" section (lines 136-153) with:

```python
            # Run AMT transcription (after MuQ scoring, sequential)
            midi_notes = None
            transcription_info = None
            amt_error = None

            try:
                print("Running AMT transcription...")
                amt_start = time.time()
                midi_notes = self._transcription.transcribe(audio, 24000)
                amt_elapsed_ms = int((time.time() - amt_start) * 1000)

                pitches = [n["pitch"] for n in midi_notes]
                transcription_info = {
                    "note_count": len(midi_notes),
                    "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
                    "transcription_time_ms": amt_elapsed_ms,
                }
            except TranscriptionError as e:
                print(f"AMT failed (graceful degradation): {e}")
                amt_error = str(e)

            # Build combined response
            processing_time_ms = int((time.time() - start_time) * 1000)

            result = {
                "predictions": self._predictions_to_dict(predictions),
                "midi_notes": midi_notes,
                "transcription_info": transcription_info,
                "model_info": {
                    "name": MODEL_INFO["name"],
                    "type": MODEL_INFO["type"],
                    "pairwise": MODEL_INFO["pairwise"],
                    "architecture": MODEL_INFO["architecture"],
                    "ensemble_folds": len(self._cache.muq_heads),
                },
                "audio_duration_seconds": duration,
                "processing_time_ms": processing_time_ms,
            }

            if amt_error:
                result["amt_error"] = amt_error

            print(f"Inference complete in {processing_time_ms}ms")
            return result
```

- [ ] **Step 3: Verify the handler still works with existing tests (if any)**

Run any existing handler tests, or do a quick manual import check:
```bash
cd apps/inference && python -c "from handler import EndpointHandler; print('Import OK')"
```

- [ ] **Step 4: Commit**

```bash
git add apps/inference/handler.py
git commit -m "feat(inference): integrate AMT into endpoint handler with graceful degradation"
```

---

### Task 3: Update requirements.txt and Dockerfile

**Files:**
- Modify: `apps/inference/requirements.txt`
- Modify: `apps/inference/Dockerfile`

- [ ] **Step 1: Update requirements.txt**

Add these two lines at the end:

```
# AMT (Automatic Music Transcription)
piano-transcription-inference
pretty-midi>=0.2.10
```

- [ ] **Step 2: Update Dockerfile**

Add AMT weight pre-download after the existing MuQ download (after line 40). Insert before the "Copy application code" section:

```dockerfile
# Pre-download ByteDance AMT model weights (REQUIRED for cold start performance)
# device='cpu' is correct -- no GPU during Docker build. Constructor downloads weights only.
RUN python3 -c "\
print('Downloading ByteDance piano transcription model...'); \
from piano_transcription_inference import PianoTranscription; \
PianoTranscription(device='cpu'); \
print('Done!'); \
"
```

- [ ] **Step 3: Commit**

```bash
git add apps/inference/requirements.txt apps/inference/Dockerfile
git commit -m "feat(inference): add AMT dependencies and pre-download weights in Dockerfile"
```

---

### Task 4: Integration Test with Real Audio

**Files:**
- Create: `apps/inference/tests/test_integration_amt.py`

- [ ] **Step 1: Write integration test**

This test requires a real audio file. Use one of the MAESTRO recordings or any piano WAV.

```python
# apps/inference/tests/test_integration_amt.py
"""Integration test: verify full endpoint returns both scores and MIDI notes.

Requires:
- CUDA GPU or CPU with sufficient memory
- Piano audio file for testing

Skip this test in CI (requires GPU + model weights).
"""

from pathlib import Path

import numpy as np
import pytest


# Skip if piano-transcription-inference not installed
pytest.importorskip("piano_transcription_inference")


SAMPLE_AUDIO_PATH = Path(__file__).parent / "fixtures" / "piano_sample.wav"


@pytest.mark.skipif(
    not SAMPLE_AUDIO_PATH.exists(),
    reason="Sample audio not available (place a piano WAV at tests/fixtures/piano_sample.wav)",
)
def test_full_pipeline_returns_scores_and_notes():
    """End-to-end: real audio produces both predictions and midi_notes."""
    import librosa
    from models.transcription import TranscriptionModel

    audio, sr = librosa.load(str(SAMPLE_AUDIO_PATH), sr=24000, mono=True, duration=15.0)

    model = TranscriptionModel(device="cpu")
    notes = model.transcribe(audio, sample_rate=24000)

    assert isinstance(notes, list)
    # Real piano audio should produce at least some notes
    assert len(notes) > 0, "Expected notes from piano audio"

    # Verify note structure
    for note in notes:
        assert 0 <= note["pitch"] <= 127
        assert note["onset"] >= 0
        assert note["offset"] > note["onset"]
        assert 0 <= note["velocity"] <= 127

    # Verify sorted by onset
    for i in range(1, len(notes)):
        assert notes[i]["onset"] >= notes[i - 1]["onset"]


def test_transcription_model_loads_on_cpu():
    """TranscriptionModel can be initialized on CPU (for testing)."""
    from models.transcription import TranscriptionModel

    model = TranscriptionModel(device="cpu")
    assert model._transcriber is not None
```

- [ ] **Step 2: Create fixtures directory**

```bash
mkdir -p apps/inference/tests/fixtures
```

Optionally copy a short piano WAV for the integration test. If no sample is available, the test will skip.

- [ ] **Step 3: Run all tests**

```bash
cd apps/inference && python -m pytest tests/ -v
```

Expected: Unit tests PASS, integration test either PASS (if fixture exists) or SKIP.

- [ ] **Step 4: Commit**

```bash
git add apps/inference/tests/test_integration_amt.py
git commit -m "test(inference): add AMT integration test"
```

---

### Task 5: Deploy to HF Endpoint

- [ ] **Step 1: Push updated code to HF model repo**

The HF inference endpoint auto-rebuilds when the model repo is updated. Push the modified `handler.py`, `models/transcription.py`, `requirements.txt`, and `Dockerfile` to the HF model repository.

```bash
# This depends on how the HF repo is set up -- typically:
cd apps/inference
# Copy files to the HF model repo clone and push
# The exact commands depend on the repo structure
```

- [ ] **Step 2: Wait for endpoint rebuild**

Monitor the HF Inference Endpoints dashboard. The rebuild should take 5-10 minutes (downloading AMT weights adds ~2 minutes to the build).

- [ ] **Step 3: Verify endpoint returns AMT data**

```bash
# Send a test request (using the existing endpoint URL from wrangler.toml)
curl -X POST https://mxcyiqltad84v9w1.us-east4.gcp.endpoints.huggingface.cloud \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"audio_url": "YOUR_TEST_AUDIO_URL"}}' | python3 -m json.tool
```

Verify the response contains:
- `predictions` (6 dimension scores, same as before)
- `midi_notes` (list of notes, or `null` if AMT failed)
- `transcription_info` (note_count, pitch_range, transcription_time_ms)
- `processing_time_ms` < 2000

- [ ] **Step 4: Verify backward compatibility**

Confirm existing consumers (API worker) still work correctly -- they read `predictions` and should ignore the new fields.

- [ ] **Step 5: Commit any deployment fixes**

```bash
git commit -m "deploy: Cloud AMT service live on HF endpoint"
```
