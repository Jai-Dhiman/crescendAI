# Data Collection (T2-T4) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Collect audio data tiers T2-T4 to expand the audio training pipeline from 1,202 PercePiano segments to ~60,000+ segments with ordinal, contrastive, and invariance training signals.

**Architecture:** Shared audio utilities (Pedalboard-based I/O, 30s segmentation) underpin three tier-specific pipelines. Each pipeline follows the same pattern: download audio -> segment into 30s clips -> extract MuQ embeddings per segment -> write metadata.jsonl. T2 extends the existing competition pipeline with segmentation. T3 builds a new MAESTRO audio pipeline. T4 builds a YouTube piano pipeline with augmentation. All pipelines are idempotent and resumable.

**Tech Stack:** Pedalboard (audio I/O + augmentation), MuQ (embeddings), yt-dlp (YouTube downloads), soundfile (fallback I/O), pytest (testing)

**Implementation notes (Batch 1):**
- Worktree: `.worktrees/data-collection`, branch: `feature/data-collection-t2-t4`
- `load_audio` uses `AudioFile.resampled_to()` instead of `Resample` plugin (plan's approach doesn't actually resample)
- Mock patch path for MuQExtractor is `audio_experiments.extractors.muq.MuQExtractor` (not `model_improvement.*.MuQExtractor`) because it's imported inside function bodies
- Task 7 conflict: `src/model_improvement/augmentation.py` already exists with torchaudio-based `AudioAugmentor`. New Pedalboard augmentation needs a different filename or merge strategy.

---

## Existing Code Summary

**Already built (reuse as-is):**
- `src/audio_experiments/extractors/muq.py` -- `MuQExtractor` class with `extract_from_file()` and `extract_from_audio()` methods
- `src/model_improvement/competition.py` -- scrape results, discover YouTube URLs, download audio at 24kHz mono, extract full-recording MuQ embeddings
- `scripts/collect_competition_data.py` -- CLI runner for the competition pipeline
- `src/model_improvement/data.py` -- `CompetitionDataset`, `CompetitionPairSampler`, `AugmentedEmbeddingDataset`
- `src/model_improvement/datasets.py` -- `load_maestro_midi_files()` parses `maestro-v3.0.0.json`

**What's missing (this plan builds):**
1. Pedalboard-based audio utilities (load, segment, save)
2. T2: Segmentation of competition recordings into 30s clips with per-segment metadata and embeddings
3. T3: Full MAESTRO audio pipeline (download, segment, embed, contrastive pair mapping)
4. T4: YouTube piano pipeline with augmentation (lower priority, not a completion gate)

---

### Task 1: Add Pedalboard Dependency and Create Audio Utilities Module [DONE]

**Files:**
- Modify: `pyproject.toml:7` (add pedalboard to dependencies)
- Create: `src/model_improvement/audio_utils.py`
- Create: `tests/model_improvement/test_audio_utils.py`

**Step 1: Write failing tests for audio utilities**

```python
# tests/model_improvement/test_audio_utils.py
import numpy as np
import pytest
from pathlib import Path

from model_improvement.audio_utils import load_audio, segment_audio, save_audio


class TestLoadAudio:
    def test_loads_wav_as_mono_float32(self, tmp_path):
        # Create a synthetic stereo WAV file
        import soundfile as sf
        sr = 44100
        stereo = np.random.randn(sr * 2, 2).astype(np.float32)  # 2s stereo
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), stereo, sr)

        audio, out_sr = load_audio(wav_path, target_sr=24000)
        assert out_sr == 24000
        assert audio.ndim == 1  # mono
        assert audio.dtype == np.float32
        # 2s at 44100 -> 2s at 24000 = ~48000 samples
        assert abs(len(audio) - 48000) < 500

    def test_loads_mono_wav_without_conversion(self, tmp_path):
        import soundfile as sf
        sr = 24000
        mono = np.random.randn(sr).astype(np.float32)  # 1s mono
        wav_path = tmp_path / "mono.wav"
        sf.write(str(wav_path), mono, sr)

        audio, out_sr = load_audio(wav_path, target_sr=24000)
        assert out_sr == 24000
        assert audio.ndim == 1
        assert abs(len(audio) - sr) < 10

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_audio(Path("/nonexistent/file.wav"))


class TestSegmentAudio:
    def test_segments_into_expected_count(self):
        sr = 24000
        audio = np.random.randn(sr * 95).astype(np.float32)  # 95 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0, min_duration=5.0)
        # 95s -> 30, 30, 30, 5 = 4 segments (last one is 5s, meets min)
        assert len(segments) == 4

    def test_drops_short_tail(self):
        sr = 24000
        audio = np.random.randn(sr * 62).astype(np.float32)  # 62 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0, min_duration=5.0)
        # 62s -> 30, 30, 2 = 2 segments (2s tail dropped, below min_duration)
        assert len(segments) == 2

    def test_segment_metadata_correct(self):
        sr = 24000
        audio = np.random.randn(sr * 60).astype(np.float32)  # 60 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0)
        assert segments[0]["start_sec"] == 0.0
        assert segments[0]["end_sec"] == 30.0
        assert segments[1]["start_sec"] == 30.0
        assert segments[1]["end_sec"] == 60.0
        assert len(segments[0]["audio"]) == sr * 30

    def test_short_audio_returns_single_segment(self):
        sr = 24000
        audio = np.random.randn(sr * 10).astype(np.float32)  # 10 seconds
        segments = segment_audio(audio, sr=sr, segment_duration=30.0, min_duration=5.0)
        assert len(segments) == 1
        assert segments[0]["start_sec"] == 0.0
        assert abs(segments[0]["end_sec"] - 10.0) < 0.01


class TestSaveAudio:
    def test_save_and_reload(self, tmp_path):
        sr = 24000
        audio = np.random.randn(sr).astype(np.float32)
        path = tmp_path / "out.wav"
        save_audio(audio, path, sr=sr)
        assert path.exists()
        assert path.stat().st_size > 0

        loaded, loaded_sr = load_audio(path, target_sr=sr)
        assert loaded_sr == sr
        np.testing.assert_allclose(loaded, audio, atol=1e-4)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_improvement.audio_utils'`

**Step 3: Add pedalboard dependency**

In `pyproject.toml`, add `"pedalboard>=0.9.0"` to the dependencies list after the `soundfile` entry.

Then run: `cd /Users/jdhiman/Documents/crescendai/model && uv pip install -e ".[dev]"`

**Step 4: Write implementation**

```python
# src/model_improvement/audio_utils.py
"""Shared audio I/O and segmentation utilities.

Uses Pedalboard for fast audio I/O (4x faster than librosa at scale).
Falls back to soundfile if Pedalboard is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_audio(path: Path, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono, resample to target_sr.

    Args:
        path: Path to audio file (WAV, FLAC, MP3, etc.).
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (audio as 1D float32 ndarray, sample rate).

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    from pedalboard.io import AudioFile

    with AudioFile(str(path)) as f:
        sr = f.samplerate
        audio = f.read(f.frames)  # shape: (channels, samples)

    # Convert to mono if stereo/multi-channel
    if audio.ndim == 2 and audio.shape[0] > 1:
        audio = audio.mean(axis=0)
    elif audio.ndim == 2:
        audio = audio[0]

    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        from pedalboard import Resample

        resampler = Resample(target_sample_rate=target_sr)
        # Pedalboard expects (channels, samples) for processing
        audio_2d = audio.reshape(1, -1)
        audio = resampler(audio_2d, sample_rate=sr).squeeze(0)
        sr = target_sr

    return audio, sr


def segment_audio(
    audio: np.ndarray,
    sr: int,
    segment_duration: float = 30.0,
    min_duration: float = 5.0,
) -> list[dict]:
    """Split audio into fixed-duration segments.

    Args:
        audio: 1D audio array.
        sr: Sample rate in Hz.
        segment_duration: Duration of each segment in seconds.
        min_duration: Minimum duration for the last segment. Shorter tails
            are dropped.

    Returns:
        List of dicts with keys: audio (np.ndarray), start_sec (float),
        end_sec (float).
    """
    total_samples = len(audio)
    segment_samples = int(segment_duration * sr)
    min_samples = int(min_duration * sr)

    segments = []
    offset = 0

    while offset < total_samples:
        end = min(offset + segment_samples, total_samples)
        chunk = audio[offset:end]

        if len(chunk) < min_samples:
            break  # drop runt tail

        start_sec = offset / sr
        end_sec = end / sr

        segments.append({
            "audio": chunk,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        })

        offset = end

    return segments


def save_audio(audio: np.ndarray, path: Path, sr: int = 24000) -> None:
    """Write mono audio to WAV file.

    Args:
        audio: 1D float32 audio array.
        path: Output file path.
        sr: Sample rate in Hz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    from pedalboard.io import AudioFile

    audio_2d = audio.reshape(1, -1).astype(np.float32)
    with AudioFile(str(path), "w", samplerate=sr, num_channels=1) as f:
        f.write(audio_2d)
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_audio_utils.py -v`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/audio_utils.py tests/model_improvement/test_audio_utils.py pyproject.toml
git commit -m "feat: add Pedalboard-based audio utilities for segmentation and I/O"
```

---

### Task 2: Add Segmentation to T2 Competition Pipeline [DONE]

**Files:**
- Modify: `src/model_improvement/competition.py:548-571` (update extract_competition_embeddings)
- Modify: `scripts/collect_competition_data.py`
- Create: `tests/model_improvement/test_competition_segmentation.py`

**Step 1: Write failing tests for competition segmentation**

```python
# tests/model_improvement/test_competition_segmentation.py
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.competition import (
    segment_and_embed_competition,
    CompetitionRecord,
)


class TestSegmentAndEmbedCompetition:
    @pytest.fixture
    def setup_cache(self, tmp_path):
        """Create a mock competition cache with audio files and recording metadata."""
        cache_dir = tmp_path / "chopin2021"
        audio_dir = cache_dir / "audio"
        audio_dir.mkdir(parents=True)

        # Create synthetic WAV files (90s each = 3 segments of 30s)
        import soundfile as sf
        sr = 24000
        for i in range(2):
            audio = np.random.randn(sr * 90).astype(np.float32)
            sf.write(str(audio_dir / f"recording_{i}.wav"), audio, sr)

        # Create recordings metadata
        recordings = [
            {
                "recording_id": "recording_0",
                "competition": "chopin",
                "edition": 2021,
                "round": "stage2",
                "placement": 1,
                "performer": "Test Performer A",
                "piece": "Ballade No. 1",
                "audio_path": "audio/recording_0.wav",
                "duration_seconds": 90.0,
                "source_url": "https://youtube.com/watch?v=test0",
                "country": "Test",
            },
            {
                "recording_id": "recording_1",
                "competition": "chopin",
                "edition": 2021,
                "round": "stage2",
                "placement": 2,
                "performer": "Test Performer B",
                "piece": "Ballade No. 1",
                "audio_path": "audio/recording_1.wav",
                "duration_seconds": 90.0,
                "source_url": "https://youtube.com/watch?v=test1",
                "country": "Test",
            },
        ]
        import jsonlines
        with jsonlines.open(cache_dir / "recordings.jsonl", mode="w") as writer:
            for r in recordings:
                writer.write(r)

        return cache_dir

    @patch("model_improvement.competition.MuQExtractor")
    def test_creates_segment_metadata(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        # Mock MuQ to return fake embeddings
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        n = segment_and_embed_competition(cache_dir, segment_duration=30.0)

        metadata_path = cache_dir / "metadata.jsonl"
        assert metadata_path.exists()

        import jsonlines
        with jsonlines.open(metadata_path) as reader:
            segments = list(reader)

        # 2 recordings * 3 segments each = 6 segments
        assert len(segments) == 6
        assert n == 6

        # Check segment metadata schema
        seg = segments[0]
        assert "segment_id" in seg
        assert "recording_id" in seg
        assert "performer" in seg
        assert "piece" in seg
        assert "round" in seg
        assert "placement" in seg
        assert "segment_start" in seg
        assert "segment_end" in seg

    @patch("model_improvement.competition.MuQExtractor")
    def test_creates_per_segment_embeddings(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        segment_and_embed_competition(cache_dir, segment_duration=30.0)

        emb_dir = cache_dir / "muq_embeddings"
        assert emb_dir.exists()
        pt_files = list(emb_dir.glob("*.pt"))
        assert len(pt_files) == 6

    @patch("model_improvement.competition.MuQExtractor")
    def test_segments_inherit_recording_placement(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        segment_and_embed_competition(cache_dir)

        import jsonlines
        with jsonlines.open(cache_dir / "metadata.jsonl") as reader:
            segments = list(reader)

        # All segments from recording_0 should have placement=1
        r0_segs = [s for s in segments if s["recording_id"] == "recording_0"]
        assert all(s["placement"] == 1 for s in r0_segs)

        # All segments from recording_1 should have placement=2
        r1_segs = [s for s in segments if s["recording_id"] == "recording_1"]
        assert all(s["placement"] == 2 for s in r1_segs)

    @patch("model_improvement.competition.MuQExtractor")
    def test_is_idempotent(self, mock_extractor_cls, setup_cache):
        cache_dir = setup_cache
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_extractor_cls.return_value = mock_extractor

        n1 = segment_and_embed_competition(cache_dir)
        n2 = segment_and_embed_competition(cache_dir)

        assert n1 == 6
        assert n2 == 0  # All already cached
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_competition_segmentation.py -v`
Expected: FAIL with `ImportError: cannot import name 'segment_and_embed_competition'`

**Step 3: Write implementation**

Add to `src/model_improvement/competition.py` (replace `extract_competition_embeddings`):

```python
# Add to imports at top of competition.py
import torch
import numpy as np

# Add new function after extract_competition_embeddings (keep old function for backward compat)

def segment_and_embed_competition(
    cache_dir: Path,
    segment_duration: float = 30.0,
    min_segment_duration: float = 5.0,
) -> int:
    """Segment competition recordings into 30s clips and extract per-segment MuQ embeddings.

    Reads full recordings from cache_dir/audio/*.wav and recording-level metadata
    from cache_dir/recordings.jsonl. Produces:
    - cache_dir/metadata.jsonl with per-segment metadata
    - cache_dir/muq_embeddings/{segment_id}.pt per segment

    Returns count of newly processed segments.
    """
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio, segment_audio

    audio_dir = cache_dir / "audio"
    emb_dir = cache_dir / "muq_embeddings"
    metadata_path = cache_dir / "metadata.jsonl"
    recordings_path = cache_dir / "recordings.jsonl"

    if not audio_dir.exists():
        logger.warning("No audio directory at %s", audio_dir)
        return 0

    # Load recording-level metadata
    if not recordings_path.exists():
        # Fall back to old metadata.jsonl for backward compat
        recordings_path = cache_dir / "metadata.jsonl"
        if not recordings_path.exists():
            logger.warning("No recordings metadata at %s", recordings_path)
            return 0

    with jsonlines.open(recordings_path) as reader:
        recordings = list(reader)

    if not recordings:
        logger.warning("No recordings found in %s", recordings_path)
        return 0

    # Load already-processed segment IDs for idempotency
    existing_segments: set[str] = set()
    if metadata_path.exists() and metadata_path != recordings_path:
        with jsonlines.open(metadata_path) as reader:
            for seg in reader:
                existing_segments.add(seg["segment_id"])

    emb_dir.mkdir(parents=True, exist_ok=True)

    extractor = MuQExtractor(cache_dir=emb_dir)
    new_count = 0

    for recording in recordings:
        recording_id = recording["recording_id"]
        wav_path = audio_dir / f"{recording_id}.wav"

        if not wav_path.exists():
            logger.warning("Audio file not found: %s", wav_path)
            continue

        # Check if all segments for this recording are already done
        # (heuristic: if any segment_id starting with this recording_id exists)
        if any(sid.startswith(recording_id) for sid in existing_segments):
            logger.debug("Segments for %s already processed", recording_id)
            continue

        audio, sr = load_audio(wav_path, target_sr=24000)
        segments = segment_audio(
            audio, sr=sr,
            segment_duration=segment_duration,
            min_duration=min_segment_duration,
        )

        for i, seg in enumerate(segments):
            segment_id = f"{recording_id}_seg{i:03d}"

            if segment_id in existing_segments:
                continue

            # Extract MuQ embedding for this segment
            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)

            # Save embedding
            torch.save(embedding, emb_dir / f"{segment_id}.pt")

            # Write segment metadata
            seg_record = {
                "segment_id": segment_id,
                "recording_id": recording_id,
                "competition": recording.get("competition", "chopin"),
                "edition": recording.get("edition", 2021),
                "round": recording["round"],
                "placement": recording["placement"],
                "performer": recording["performer"],
                "piece": recording["piece"],
                "segment_start": seg["start_sec"],
                "segment_end": seg["end_sec"],
                "source_url": recording.get("source_url", ""),
                "country": recording.get("country", ""),
            }

            with jsonlines.open(metadata_path, mode="a") as writer:
                writer.write(seg_record)

            existing_segments.add(segment_id)
            new_count += 1

    del extractor

    logger.info("Processed %d new segments", new_count)
    return new_count
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_competition_segmentation.py -v`
Expected: All 4 tests PASS

**Step 5: Update the collection script**

In `scripts/collect_competition_data.py`, add Step 4a (segmentation) after the existing download step:

```python
# After Step 3 (download) and before Step 4 (embeddings), add:

    # Step 3b: Rename metadata.jsonl to recordings.jsonl if needed
    # (backward compat: old pipeline wrote metadata.jsonl at recording level)
    recordings_path = cache_dir / "recordings.jsonl"
    old_metadata = cache_dir / "metadata.jsonl"
    if old_metadata.exists() and not recordings_path.exists():
        old_metadata.rename(recordings_path)

    # Step 4: Segment and extract per-segment embeddings
    if args.skip_embeddings:
        logger.info("=" * 60)
        logger.info("Step 4: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 4: Segmenting audio and extracting MuQ embeddings...")
        from model_improvement.competition import segment_and_embed_competition
        n_segments = segment_and_embed_competition(cache_dir)
        logger.info("Processed %d new segments", n_segments)
```

**Step 6: Run the full test suite to check for regressions**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/competition.py tests/model_improvement/test_competition_segmentation.py scripts/collect_competition_data.py
git commit -m "feat: add 30s segmentation to T2 competition pipeline"
```

---

### Task 3: MAESTRO Audio Pipeline - Metadata and Segmentation [DONE]

**Files:**
- Create: `src/model_improvement/maestro.py`
- Create: `tests/model_improvement/test_maestro.py`

**Step 1: Write failing tests for MAESTRO metadata parsing and segment builder**

```python
# tests/model_improvement/test_maestro.py
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.maestro import (
    parse_maestro_audio_metadata,
    MaestroSegment,
)


class TestParseMaestroAudioMetadata:
    @pytest.fixture
    def maestro_dir(self, tmp_path):
        """Create a mock MAESTRO directory with metadata JSON."""
        maestro = tmp_path / "maestro-v3.0.0"
        maestro.mkdir()

        metadata = {
            "canonical_composer": {"0": "Chopin", "1": "Chopin", "2": "Beethoven"},
            "canonical_title": {
                "0": "Ballade No. 1",
                "1": "Ballade No. 1",
                "2": "Sonata No. 14",
            },
            "split": {"0": "train", "1": "train", "2": "validation"},
            "midi_filename": {
                "0": "2015/file_a.midi",
                "1": "2015/file_b.midi",
                "2": "2017/file_c.midi",
            },
            "audio_filename": {
                "0": "2015/file_a.wav",
                "1": "2015/file_b.wav",
                "2": "2017/file_c.wav",
            },
            "duration": {"0": 300.5, "1": 280.2, "2": 600.1},
        }

        with open(maestro / "maestro-v3.0.0.json", "w") as f:
            json.dump(metadata, f)

        return maestro

    def test_parses_all_records(self, maestro_dir):
        records = parse_maestro_audio_metadata(maestro_dir)
        assert len(records) == 3

    def test_record_schema(self, maestro_dir):
        records = parse_maestro_audio_metadata(maestro_dir)
        r = records[0]
        assert "audio_filename" in r
        assert "canonical_title" in r
        assert "canonical_composer" in r
        assert "split" in r
        assert "duration" in r

    def test_raises_on_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            parse_maestro_audio_metadata(Path("/nonexistent"))

    def test_raises_on_missing_json(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            parse_maestro_audio_metadata(empty_dir)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_maestro.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_improvement.maestro'`

**Step 3: Write implementation**

```python
# src/model_improvement/maestro.py
"""MAESTRO v3 audio pipeline: metadata, segmentation, MuQ embedding extraction.

Processes MAESTRO v3 audio recordings into 30s segments with MuQ embeddings
for cross-performer contrastive training (T3 data tier).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonlines

logger = logging.getLogger(__name__)


@dataclass
class MaestroSegment:
    segment_id: str
    audio_filename: str
    canonical_title: str
    canonical_composer: str
    split: str
    segment_start: float
    segment_end: float
    duration_seconds: float


def parse_maestro_audio_metadata(maestro_dir: Path) -> list[dict]:
    """Parse MAESTRO v3 metadata JSON for audio file entries.

    Args:
        maestro_dir: Path to MAESTRO root (contains maestro-v3.0.0.json).

    Returns:
        List of dicts with keys: audio_filename, canonical_title,
        canonical_composer, split, duration, midi_filename.

    Raises:
        FileNotFoundError: If maestro_dir or metadata JSON does not exist.
    """
    maestro_dir = Path(maestro_dir)
    if not maestro_dir.exists():
        raise FileNotFoundError(f"MAESTRO directory not found: {maestro_dir}")

    metadata_path = maestro_dir / "maestro-v3.0.0.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"MAESTRO metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        raw = json.load(f)

    # MAESTRO JSON is column-oriented: {col_name: {row_idx: value}}
    if isinstance(raw, dict) and "audio_filename" in raw:
        row_keys = list(raw["audio_filename"].keys())
        records = [
            {col: raw[col][k] for col in raw}
            for k in row_keys
        ]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError(
            f"Unexpected MAESTRO JSON format: top-level type={type(raw).__name__}"
        )

    # Filter to records that have audio_filename
    records = [r for r in records if r.get("audio_filename")]

    logger.info("Parsed %d MAESTRO audio records", len(records))
    return records


def segment_and_embed_maestro(
    maestro_dir: Path,
    cache_dir: Path,
    segment_duration: float = 30.0,
    min_segment_duration: float = 5.0,
) -> int:
    """Segment all MAESTRO audio and extract per-segment MuQ embeddings.

    For each audio file in MAESTRO, loads audio via Pedalboard, segments
    into 30s clips, extracts MuQ embeddings per segment, and writes:
    - cache_dir/metadata.jsonl with per-segment metadata
    - cache_dir/muq_embeddings/{segment_id}.pt per segment

    Args:
        maestro_dir: Path to MAESTRO root with audio files.
        cache_dir: Output directory for cached embeddings and metadata.
        segment_duration: Duration of each segment in seconds.
        min_segment_duration: Minimum duration for last segment.

    Returns:
        Count of newly processed segments.
    """
    import torch
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio, segment_audio

    metadata_path = cache_dir / "metadata.jsonl"
    emb_dir = cache_dir / "muq_embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    records = parse_maestro_audio_metadata(maestro_dir)

    # Load existing segment IDs for idempotency
    existing_segments: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for seg in reader:
                existing_segments.add(seg["segment_id"])

    extractor = MuQExtractor(cache_dir=emb_dir)
    new_count = 0

    for i, record in enumerate(records):
        audio_filename = record["audio_filename"]
        audio_path = maestro_dir / audio_filename

        if not audio_path.exists():
            logger.warning("Audio file not found: %s", audio_path)
            continue

        # Create a stable base ID from the audio filename
        base_id = audio_filename.replace("/", "_").replace(".", "_")
        base_id = f"maestro_{base_id}"

        # Skip if any segments for this recording already exist
        if any(sid.startswith(base_id) for sid in existing_segments):
            if i % 100 == 0:
                logger.debug("[%d/%d] Already processed: %s", i, len(records), base_id)
            continue

        logger.info("[%d/%d] Processing: %s", i + 1, len(records), audio_filename)

        audio, sr = load_audio(audio_path, target_sr=24000)
        segments = segment_audio(
            audio, sr=sr,
            segment_duration=segment_duration,
            min_duration=min_segment_duration,
        )

        for j, seg in enumerate(segments):
            segment_id = f"{base_id}_seg{j:03d}"

            if segment_id in existing_segments:
                continue

            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)
            torch.save(embedding, emb_dir / f"{segment_id}.pt")

            seg_record = MaestroSegment(
                segment_id=segment_id,
                audio_filename=audio_filename,
                canonical_title=record.get("canonical_title", "Unknown"),
                canonical_composer=record.get("canonical_composer", "Unknown"),
                split=record.get("split", "train"),
                segment_start=seg["start_sec"],
                segment_end=seg["end_sec"],
                duration_seconds=seg["end_sec"] - seg["start_sec"],
            )

            with jsonlines.open(metadata_path, mode="a") as writer:
                writer.write(asdict(seg_record))

            existing_segments.add(segment_id)
            new_count += 1

    del extractor
    logger.info("Processed %d new MAESTRO segments", new_count)
    return new_count


def build_piece_performer_mapping(cache_dir: Path) -> dict:
    """Build piece-to-segment mapping for contrastive pair generation.

    Groups segments by canonical_title. Pieces with 2+ distinct source
    recordings enable contrastive learning (same piece, different performer).

    Args:
        cache_dir: Directory containing metadata.jsonl.

    Returns:
        Dict mapping canonical_title -> list of segment_ids.
        Only includes pieces with 2+ distinct source recordings.
    """
    metadata_path = cache_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return {}

    # Group segments by piece, tracking which source recordings they come from
    piece_to_recordings: dict[str, set[str]] = {}
    piece_to_segments: dict[str, list[str]] = {}

    with jsonlines.open(metadata_path) as reader:
        for seg in reader:
            title = seg["canonical_title"]
            audio_file = seg["audio_filename"]
            segment_id = seg["segment_id"]

            if title not in piece_to_recordings:
                piece_to_recordings[title] = set()
                piece_to_segments[title] = []

            piece_to_recordings[title].add(audio_file)
            piece_to_segments[title].append(segment_id)

    # Filter to pieces with 2+ distinct recordings
    contrastive_mapping = {
        title: segments
        for title, segments in piece_to_segments.items()
        if len(piece_to_recordings[title]) >= 2
    }

    logger.info(
        "Contrastive mapping: %d pieces with 2+ recordings, %d total segments",
        len(contrastive_mapping),
        sum(len(s) for s in contrastive_mapping.values()),
    )

    return contrastive_mapping
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_maestro.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/maestro.py tests/model_improvement/test_maestro.py
git commit -m "feat: add MAESTRO audio metadata parser for T3 pipeline"
```

---

### Task 4: MAESTRO Segmentation, Embedding, and Contrastive Mapping Tests

**Files:**
- Modify: `tests/model_improvement/test_maestro.py` (add segmentation and mapping tests)

**Step 1: Write failing tests for segment_and_embed and contrastive mapping**

Append to `tests/model_improvement/test_maestro.py`:

```python
from model_improvement.maestro import (
    segment_and_embed_maestro,
    build_piece_performer_mapping,
)


class TestSegmentAndEmbedMaestro:
    @pytest.fixture
    def maestro_with_audio(self, tmp_path):
        """Create mock MAESTRO dir with metadata and audio files."""
        maestro = tmp_path / "maestro-v3.0.0"
        subdir = maestro / "2015"
        subdir.mkdir(parents=True)

        import soundfile as sf
        sr = 24000
        # Two recordings of the same piece (different performers)
        for fname in ["file_a.wav", "file_b.wav"]:
            audio = np.random.randn(sr * 65).astype(np.float32)  # 65s = 2 full + 1 short
            sf.write(str(subdir / fname), audio, sr)
        # One recording of a different piece
        audio = np.random.randn(sr * 35).astype(np.float32)
        sf.write(str(subdir / "file_c.wav"), audio, sr)

        metadata = {
            "canonical_composer": {"0": "Chopin", "1": "Chopin", "2": "Beethoven"},
            "canonical_title": {
                "0": "Ballade No. 1",
                "1": "Ballade No. 1",
                "2": "Sonata No. 14",
            },
            "split": {"0": "train", "1": "train", "2": "validation"},
            "midi_filename": {
                "0": "2015/file_a.midi",
                "1": "2015/file_b.midi",
                "2": "2015/file_c.midi",
            },
            "audio_filename": {
                "0": "2015/file_a.wav",
                "1": "2015/file_b.wav",
                "2": "2015/file_c.wav",
            },
            "duration": {"0": 65.0, "1": 65.0, "2": 35.0},
        }
        with open(maestro / "maestro-v3.0.0.json", "w") as f:
            json.dump(metadata, f)

        return maestro, tmp_path / "cache"

    @patch("model_improvement.maestro.MuQExtractor")
    def test_produces_segment_metadata(self, mock_cls, maestro_with_audio):
        maestro_dir, cache_dir = maestro_with_audio
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        n = segment_and_embed_maestro(maestro_dir, cache_dir, segment_duration=30.0)

        metadata_path = cache_dir / "metadata.jsonl"
        assert metadata_path.exists()

        with jsonlines.open(metadata_path) as reader:
            segments = list(reader)

        # file_a: 65s -> 2 full (30s) + 1 short (5s, meets min) = 3 segments
        # file_b: 65s -> 3 segments
        # file_c: 35s -> 1 full + 1 short (5s) = 2 segments
        assert len(segments) == 8
        assert n == 8

    @patch("model_improvement.maestro.MuQExtractor")
    def test_is_idempotent(self, mock_cls, maestro_with_audio):
        maestro_dir, cache_dir = maestro_with_audio
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        n1 = segment_and_embed_maestro(maestro_dir, cache_dir)
        n2 = segment_and_embed_maestro(maestro_dir, cache_dir)
        assert n1 > 0
        assert n2 == 0

    @patch("model_improvement.maestro.MuQExtractor")
    def test_contrastive_mapping(self, mock_cls, maestro_with_audio):
        maestro_dir, cache_dir = maestro_with_audio
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        segment_and_embed_maestro(maestro_dir, cache_dir)
        mapping = build_piece_performer_mapping(cache_dir)

        # "Ballade No. 1" has 2 recordings -> included
        assert "Ballade No. 1" in mapping
        assert len(mapping["Ballade No. 1"]) == 6  # 3 segments from each recording

        # "Sonata No. 14" has only 1 recording -> excluded
        assert "Sonata No. 14" not in mapping
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_maestro.py::TestSegmentAndEmbedMaestro -v`
Expected: FAIL (tests call functions that exist but haven't been tested with mocks yet -- verify mock wiring works)

**Step 3: Fix any issues, verify all pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_maestro.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add tests/model_improvement/test_maestro.py
git commit -m "test: add segmentation and contrastive mapping tests for MAESTRO"
```

---

### Task 5: MAESTRO Collection Script

**Files:**
- Create: `scripts/collect_maestro_audio.py`

**Step 1: Write the collection script**

```python
# scripts/collect_maestro_audio.py
"""Segment MAESTRO v3 audio and extract MuQ embeddings for T3 contrastive training.

Run from model/ directory:
    python scripts/collect_maestro_audio.py --maestro-dir data/maestro-v3.0.0
    python scripts/collect_maestro_audio.py --maestro-dir data/maestro-v3.0.0 --skip-embeddings

Expects MAESTRO v3 audio to be downloaded already (200GB WAV).
Download from: https://magenta.tensorflow.org/datasets/maestro

Each step is idempotent. Running again skips completed work.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.maestro import (
    build_piece_performer_mapping,
    parse_maestro_audio_metadata,
    segment_and_embed_maestro,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAESTRO audio segmentation and MuQ embedding extraction"
    )
    parser.add_argument(
        "--maestro-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "maestro-v3.0.0",
        help="Path to MAESTRO v3 root directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "maestro_cache",
        help="Output directory for cached segments and embeddings",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only parse metadata, skip segmentation and embedding",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=30.0,
        help="Segment duration in seconds (default: 30)",
    )
    args = parser.parse_args()

    logger.info("MAESTRO audio pipeline")
    logger.info("MAESTRO dir: %s", args.maestro_dir)
    logger.info("Cache dir: %s", args.cache_dir)
    t_start = time.time()

    # Step 1: Parse metadata
    logger.info("=" * 60)
    logger.info("Step 1: Parsing MAESTRO metadata...")
    records = parse_maestro_audio_metadata(args.maestro_dir)
    logger.info("Found %d audio records", len(records))

    # Count audio files that exist on disk
    n_exists = sum(
        1 for r in records
        if (args.maestro_dir / r["audio_filename"]).exists()
    )
    logger.info("Audio files on disk: %d / %d", n_exists, len(records))

    if n_exists == 0:
        logger.error(
            "No audio files found. Download MAESTRO v3 audio from: "
            "https://magenta.tensorflow.org/datasets/maestro"
        )
        sys.exit(1)

    # Step 2: Segment and extract embeddings
    if args.skip_embeddings:
        logger.info("=" * 60)
        logger.info("Step 2: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 2: Segmenting audio and extracting MuQ embeddings...")
        n_segments = segment_and_embed_maestro(
            args.maestro_dir, args.cache_dir,
            segment_duration=args.segment_duration,
        )
        logger.info("Processed %d new segments", n_segments)

    # Step 3: Build contrastive mapping
    logger.info("=" * 60)
    logger.info("Step 3: Building piece-performer contrastive mapping...")
    mapping = build_piece_performer_mapping(args.cache_dir)
    logger.info(
        "Contrastive pairs: %d pieces with 2+ recordings",
        len(mapping),
    )

    # Save mapping for training
    mapping_path = args.cache_dir / "contrastive_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("Saved mapping to %s", mapping_path)

    # Summary
    logger.info("=" * 60)
    metadata_path = args.cache_dir / "metadata.jsonl"
    emb_dir = args.cache_dir / "muq_embeddings"
    n_segments_total = 0
    if metadata_path.exists():
        import jsonlines
        with jsonlines.open(metadata_path) as reader:
            n_segments_total = sum(1 for _ in reader)
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0

    logger.info("Total segments: %d", n_segments_total)
    logger.info("Total embeddings: %d", n_emb)
    logger.info("Contrastive pieces: %d", len(mapping))
    logger.info(
        "Contrastive segments: %d",
        sum(len(v) for v in mapping.values()),
    )

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
```

**Step 2: Verify script is importable**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -c "import scripts.collect_maestro_audio" 2>&1 || echo "Script is standalone, not importable (expected)"`

**Step 3: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add scripts/collect_maestro_audio.py
git commit -m "feat: add MAESTRO audio collection script for T3 pipeline"
```

---

### Task 6: T4 YouTube Piano Pipeline - Channel Curation and Download

**Files:**
- Create: `src/model_improvement/youtube_piano.py`
- Create: `tests/model_improvement/test_youtube_piano.py`
- Create: `data/youtube_piano_cache/channels.yaml`

**Step 1: Write failing tests for YouTube piano discovery and download**

```python
# tests/model_improvement/test_youtube_piano.py
import json
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.youtube_piano import (
    load_channel_list,
    discover_channel_videos,
    download_piano_audio,
)


class TestLoadChannelList:
    def test_loads_yaml(self, tmp_path):
        channels_yaml = tmp_path / "channels.yaml"
        channels_yaml.write_text(
            "channels:\n"
            "  - url: https://youtube.com/@channel1\n"
            "    name: Channel One\n"
            "    category: recital\n"
            "  - url: https://youtube.com/@channel2\n"
            "    name: Channel Two\n"
            "    category: conservatory\n"
        )
        channels = load_channel_list(channels_yaml)
        assert len(channels) == 2
        assert channels[0]["name"] == "Channel One"

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_channel_list(Path("/nonexistent/channels.yaml"))


class TestDiscoverChannelVideos:
    @patch("model_improvement.youtube_piano.subprocess")
    def test_returns_video_list(self, mock_subprocess):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            '{"id": "abc123", "title": "Chopin Ballade No 1", "duration": 600}\n'
            '{"id": "def456", "title": "Beethoven Sonata No 14", "duration": 900}\n'
        )
        mock_subprocess.run.return_value = mock_result

        videos = discover_channel_videos("https://youtube.com/@channel1", max_videos=10)
        assert len(videos) == 2
        assert videos[0]["id"] == "abc123"

    @patch("model_improvement.youtube_piano.subprocess")
    def test_handles_yt_dlp_failure(self, mock_subprocess):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error"
        mock_subprocess.run.return_value = mock_result

        videos = discover_channel_videos("https://youtube.com/@bad", max_videos=10)
        assert videos == []


class TestDownloadPianoAudio:
    @patch("model_improvement.youtube_piano._download_audio_yt_dlp")
    def test_downloads_and_writes_metadata(self, mock_download, tmp_path):
        cache_dir = tmp_path / "youtube_piano_cache"

        # Mock download to create a fake WAV file
        def fake_download(url, output_path):
            import soundfile as sf
            audio = np.random.randn(24000 * 60).astype(np.float32)
            sf.write(str(output_path), audio, 24000)

        mock_download.side_effect = fake_download

        videos = [
            {"id": "abc123", "title": "Chopin Ballade", "duration": 600,
             "channel": "TestChannel", "url": "https://youtube.com/watch?v=abc123"},
        ]

        records = download_piano_audio(videos, cache_dir)
        assert len(records) == 1
        assert (cache_dir / "audio" / "abc123.wav").exists()

        # Metadata should be written
        import jsonlines
        with jsonlines.open(cache_dir / "recordings.jsonl") as reader:
            metadata = list(reader)
        assert len(metadata) == 1
        assert metadata[0]["video_id"] == "abc123"
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_youtube_piano.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_improvement.youtube_piano'`

**Step 3: Write implementation**

```python
# src/model_improvement/youtube_piano.py
"""T4: Unlabeled piano audio at scale from YouTube.

Downloads audio from curated piano YouTube channels, segments into 30s clips,
extracts clean MuQ embeddings, and optionally generates augmented pairs for
invariance training.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import jsonlines
import yaml

logger = logging.getLogger(__name__)


@dataclass
class PianoRecording:
    video_id: str
    title: str
    channel: str
    duration_seconds: float
    audio_path: str
    source_url: str


def load_channel_list(channels_path: Path) -> list[dict]:
    """Load curated YouTube piano channel list from YAML.

    Args:
        channels_path: Path to channels.yaml.

    Returns:
        List of dicts with keys: url, name, category.

    Raises:
        FileNotFoundError: If channels_path does not exist.
    """
    channels_path = Path(channels_path)
    if not channels_path.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_path}")

    with open(channels_path) as f:
        data = yaml.safe_load(f)

    return data.get("channels", [])


def discover_channel_videos(
    channel_url: str,
    max_videos: int = 100,
) -> list[dict]:
    """Discover videos from a YouTube channel using yt-dlp.

    Args:
        channel_url: YouTube channel URL.
        max_videos: Maximum number of videos to return.

    Returns:
        List of dicts with keys: id, title, duration, url.
    """
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                "--playlist-end", str(max_videos),
                channel_url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("yt-dlp failed for %s: %s", channel_url, e)
        return []

    if result.returncode != 0:
        logger.warning("yt-dlp error for %s: %s", channel_url, result.stderr[:200])
        return []

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = data.get("id", "")
        if not video_id:
            continue

        videos.append({
            "id": video_id,
            "title": data.get("title", ""),
            "duration": data.get("duration", 0),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "channel": data.get("channel", data.get("uploader", "")),
        })

    return videos


def _download_audio_yt_dlp(url: str, output_path: Path) -> None:
    """Download audio from YouTube as 24kHz mono WAV."""
    result = subprocess.run(
        [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ar 24000 -ac 1",
            "--output", str(output_path.with_suffix(".%(ext)s")),
            "--no-playlist",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (code {result.returncode}): {result.stderr[:500]}"
        )


def download_piano_audio(
    videos: list[dict],
    cache_dir: Path,
) -> list[PianoRecording]:
    """Download audio from YouTube videos.

    Args:
        videos: List of video dicts from discover_channel_videos().
        cache_dir: Output directory.

    Returns:
        List of PianoRecording for successfully downloaded videos.
    """
    audio_dir = cache_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "recordings.jsonl"

    # Load existing downloads for idempotency
    existing_ids: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for r in reader:
                existing_ids.add(r["video_id"])

    records: list[PianoRecording] = []

    for video in videos:
        video_id = video["id"]
        wav_path = audio_dir / f"{video_id}.wav"

        if video_id in existing_ids and wav_path.exists():
            logger.debug("Skipping %s (already downloaded)", video_id)
            continue

        url = video.get("url", f"https://www.youtube.com/watch?v={video_id}")

        logger.info("Downloading %s: %s", video_id, video.get("title", ""))

        try:
            _download_audio_yt_dlp(url, wav_path)
        except Exception as e:
            logger.error("Failed to download %s: %s", video_id, e)
            continue

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            logger.error("Download produced empty file: %s", video_id)
            continue

        import soundfile as sf
        try:
            info = sf.info(str(wav_path))
            duration = info.duration
        except Exception:
            duration = 0.0

        record = PianoRecording(
            video_id=video_id,
            title=video.get("title", ""),
            channel=video.get("channel", ""),
            duration_seconds=duration,
            audio_path=f"audio/{video_id}.wav",
            source_url=url,
        )
        records.append(record)

        with jsonlines.open(metadata_path, mode="a") as writer:
            writer.write(asdict(record))

    logger.info("Downloaded %d new recordings", len(records))
    return records


def segment_and_embed_piano(
    cache_dir: Path,
    segment_duration: float = 30.0,
    min_segment_duration: float = 5.0,
) -> int:
    """Segment YouTube piano audio and extract clean MuQ embeddings.

    Reads recordings from cache_dir/audio/*.wav and cache_dir/recordings.jsonl.
    Writes:
    - cache_dir/metadata.jsonl with per-segment metadata
    - cache_dir/muq_embeddings/{segment_id}.pt per segment

    Returns count of newly processed segments.
    """
    import torch
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio, segment_audio

    audio_dir = cache_dir / "audio"
    emb_dir = cache_dir / "muq_embeddings"
    metadata_path = cache_dir / "metadata.jsonl"
    recordings_path = cache_dir / "recordings.jsonl"

    if not recordings_path.exists():
        logger.warning("No recordings metadata at %s", recordings_path)
        return 0

    with jsonlines.open(recordings_path) as reader:
        recordings = list(reader)

    existing_segments: set[str] = set()
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            for seg in reader:
                existing_segments.add(seg["segment_id"])

    emb_dir.mkdir(parents=True, exist_ok=True)
    extractor = MuQExtractor(cache_dir=emb_dir)
    new_count = 0

    for i, recording in enumerate(recordings):
        video_id = recording["video_id"]
        wav_path = audio_dir / f"{video_id}.wav"

        if not wav_path.exists():
            continue

        base_id = f"yt_{video_id}"
        if any(sid.startswith(base_id) for sid in existing_segments):
            continue

        logger.info("[%d/%d] Segmenting %s", i + 1, len(recordings), video_id)

        audio, sr = load_audio(wav_path, target_sr=24000)
        segments = segment_audio(
            audio, sr=sr,
            segment_duration=segment_duration,
            min_duration=min_segment_duration,
        )

        for j, seg in enumerate(segments):
            segment_id = f"{base_id}_seg{j:03d}"

            if segment_id in existing_segments:
                continue

            audio_tensor = torch.from_numpy(seg["audio"]).float()
            embedding = extractor.extract_from_audio(audio_tensor)
            torch.save(embedding, emb_dir / f"{segment_id}.pt")

            seg_record = {
                "segment_id": segment_id,
                "video_id": video_id,
                "title": recording.get("title", ""),
                "channel": recording.get("channel", ""),
                "segment_start": seg["start_sec"],
                "segment_end": seg["end_sec"],
            }

            with jsonlines.open(metadata_path, mode="a") as writer:
                writer.write(seg_record)

            existing_segments.add(segment_id)
            new_count += 1

    del extractor
    logger.info("Processed %d new YouTube piano segments", new_count)
    return new_count
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_youtube_piano.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/youtube_piano.py tests/model_improvement/test_youtube_piano.py
git commit -m "feat: add T4 YouTube piano download and segmentation pipeline"
```

---

### Task 7: T4 Audio Augmentation Pipeline

**Files:**
- Create: `src/model_improvement/augmentation.py`
- Create: `tests/model_improvement/test_augmentation_pipeline.py`

**Step 1: Write failing tests for Pedalboard-based audio augmentation**

```python
# tests/model_improvement/test_augmentation_pipeline.py
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model_improvement.augmentation import (
    create_augmentation_chain,
    augment_audio,
    augment_and_embed_piano,
)


class TestCreateAugmentationChain:
    def test_returns_callable(self):
        chain = create_augmentation_chain(seed=42)
        assert callable(chain)

    def test_produces_different_output(self):
        chain = create_augmentation_chain(seed=42)
        audio = np.random.randn(24000 * 5).astype(np.float32)
        augmented = chain(audio, sample_rate=24000)
        assert augmented.shape == audio.shape
        # Should not be identical (augmentation applied)
        assert not np.allclose(audio, augmented, atol=1e-6)


class TestAugmentAudio:
    def test_augments_and_returns_ndarray(self):
        audio = np.random.randn(24000 * 10).astype(np.float32)
        augmented = augment_audio(audio, sr=24000, seed=42)
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == audio.shape
        assert augmented.dtype == np.float32


class TestAugmentAndEmbedPiano:
    @pytest.fixture
    def cache_with_segments(self, tmp_path):
        """Create mock cache with clean embeddings and metadata."""
        cache_dir = tmp_path / "youtube_piano_cache"
        emb_dir = cache_dir / "muq_embeddings"
        audio_dir = cache_dir / "audio"
        emb_dir.mkdir(parents=True)
        audio_dir.mkdir(parents=True)

        # Create fake audio and clean embeddings
        import soundfile as sf
        sr = 24000
        audio = np.random.randn(sr * 65).astype(np.float32)
        sf.write(str(audio_dir / "vid123.wav"), audio, sr)

        # Clean embeddings (from segment_and_embed_piano)
        for i in range(2):
            torch.save(torch.randn(93, 1024), emb_dir / f"yt_vid123_seg{i:03d}.pt")

        # Segment metadata
        import jsonlines
        segments = [
            {"segment_id": f"yt_vid123_seg{i:03d}", "video_id": "vid123",
             "segment_start": i * 30.0, "segment_end": (i + 1) * 30.0}
            for i in range(2)
        ]
        with jsonlines.open(cache_dir / "metadata.jsonl", mode="w") as writer:
            for s in segments:
                writer.write(s)

        # Recordings metadata
        with jsonlines.open(cache_dir / "recordings.jsonl", mode="w") as writer:
            writer.write({"video_id": "vid123", "title": "Test", "channel": "Test",
                          "duration_seconds": 65.0, "audio_path": "audio/vid123.wav",
                          "source_url": "https://youtube.com/watch?v=vid123"})

        return cache_dir

    @patch("model_improvement.augmentation.MuQExtractor")
    def test_creates_augmented_embeddings(self, mock_cls, cache_with_segments):
        cache_dir = cache_with_segments
        mock_extractor = MagicMock()
        mock_extractor.extract_from_audio.return_value = torch.randn(93, 1024)
        mock_cls.return_value = mock_extractor

        n = augment_and_embed_piano(cache_dir)

        aug_dir = cache_dir / "muq_embeddings_augmented"
        assert aug_dir.exists()
        assert len(list(aug_dir.glob("*.pt"))) == 2
        assert n == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_augmentation_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_improvement.augmentation'`

**Step 3: Write implementation**

```python
# src/model_improvement/augmentation.py
"""Audio augmentation pipeline using Pedalboard for T4 invariance training.

Generates augmented versions of piano audio for training the model to be
invariant to recording conditions (room acoustics, noise, compression,
different microphones/phones).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def create_augmentation_chain(seed: int | None = None):
    """Create a Pedalboard augmentation chain for piano audio.

    Applies a random subset of:
    - Reverb (room IR simulation)
    - Compression (dynamic range reduction)
    - Low-pass filter (phone mic simulation)
    - Pitch shift (slight tuning variation)
    - Parametric EQ (frequency coloring)
    - Gain (volume variation)

    Args:
        seed: Random seed for reproducible augmentation.

    Returns:
        Callable that takes (audio_ndarray, sample_rate) and returns augmented audio.
    """
    from pedalboard import (
        Compressor,
        Gain,
        LowpassFilter,
        Pedalboard,
        PitchShift,
        Reverb,
    )

    rng = np.random.RandomState(seed)

    # Randomly sample augmentation parameters
    effects = []

    # Reverb (60% chance) -- room acoustics variation
    if rng.random() < 0.6:
        effects.append(Reverb(
            room_size=rng.uniform(0.1, 0.7),
            wet_level=rng.uniform(0.05, 0.3),
        ))

    # Compression (50% chance) -- dynamic range reduction
    if rng.random() < 0.5:
        effects.append(Compressor(
            threshold_db=rng.uniform(-30, -10),
            ratio=rng.uniform(2.0, 6.0),
        ))

    # Low-pass filter (40% chance) -- phone/laptop mic simulation
    if rng.random() < 0.4:
        effects.append(LowpassFilter(
            cutoff_frequency_hz=rng.uniform(4000, 12000),
        ))

    # Pitch shift (30% chance) -- slight tuning variation
    if rng.random() < 0.3:
        effects.append(PitchShift(
            semitones=rng.uniform(-0.5, 0.5),
        ))

    # Gain (70% chance) -- volume variation
    if rng.random() < 0.7:
        effects.append(Gain(
            gain_db=rng.uniform(-6, 3),
        ))

    board = Pedalboard(effects)

    def augment(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        # Pedalboard expects (channels, samples)
        audio_2d = audio.reshape(1, -1).astype(np.float32)
        result = board(audio_2d, sample_rate=sample_rate)
        return result.squeeze(0).astype(np.float32)

    return augment


def augment_audio(
    audio: np.ndarray,
    sr: int = 24000,
    seed: int | None = None,
) -> np.ndarray:
    """Apply random augmentation to audio.

    Args:
        audio: 1D float32 audio array.
        sr: Sample rate.
        seed: Random seed.

    Returns:
        Augmented audio as 1D float32 ndarray (same shape as input).
    """
    chain = create_augmentation_chain(seed=seed)
    return chain(audio, sample_rate=sr)


def augment_and_embed_piano(
    cache_dir: Path,
    segment_duration: float = 30.0,
) -> int:
    """Generate augmented audio and extract MuQ embeddings for T4 invariance.

    For each segment in cache_dir/metadata.jsonl:
    1. Load the corresponding audio segment from the full recording
    2. Apply random augmentation
    3. Extract MuQ embedding from augmented audio
    4. Save to cache_dir/muq_embeddings_augmented/{segment_id}.pt

    Args:
        cache_dir: YouTube piano cache directory.
        segment_duration: Segment duration used during segmentation.

    Returns:
        Count of newly processed augmented segments.
    """
    import torch
    from audio_experiments.extractors.muq import MuQExtractor
    from model_improvement.audio_utils import load_audio

    import jsonlines

    metadata_path = cache_dir / "metadata.jsonl"
    audio_dir = cache_dir / "audio"
    aug_emb_dir = cache_dir / "muq_embeddings_augmented"

    if not metadata_path.exists():
        logger.warning("No segment metadata at %s", metadata_path)
        return 0

    aug_emb_dir.mkdir(parents=True, exist_ok=True)

    # Load segment metadata
    with jsonlines.open(metadata_path) as reader:
        segments = list(reader)

    # Check which augmented embeddings already exist
    existing = {p.stem for p in aug_emb_dir.glob("*.pt")}

    to_process = [s for s in segments if s["segment_id"] not in existing]
    if not to_process:
        logger.info("All %d augmented embeddings already cached", len(segments))
        return 0

    logger.info("Augmenting %d segments (%d already cached)", len(to_process), len(existing))

    extractor = MuQExtractor(cache_dir=aug_emb_dir)
    new_count = 0

    # Group segments by video_id to load audio once per recording
    from collections import defaultdict
    video_segments: dict[str, list[dict]] = defaultdict(list)
    for seg in to_process:
        video_segments[seg["video_id"]].append(seg)

    for video_id, segs in video_segments.items():
        wav_path = audio_dir / f"{video_id}.wav"
        if not wav_path.exists():
            logger.warning("Audio not found: %s", wav_path)
            continue

        audio, sr = load_audio(wav_path, target_sr=24000)

        for seg in segs:
            segment_id = seg["segment_id"]
            start_sample = int(seg["segment_start"] * sr)
            end_sample = int(seg["segment_end"] * sr)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < sr:  # less than 1 second
                continue

            # Apply augmentation with segment-specific seed for reproducibility
            seed = hash(segment_id) % (2**31)
            augmented = augment_audio(segment_audio, sr=sr, seed=seed)

            # Extract MuQ embedding from augmented audio
            audio_tensor = torch.from_numpy(augmented).float()
            embedding = extractor.extract_from_audio(audio_tensor)

            torch.save(embedding, aug_emb_dir / f"{segment_id}.pt")
            new_count += 1

    del extractor
    logger.info("Generated %d augmented embeddings", new_count)
    return new_count
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/model_improvement/test_augmentation_pipeline.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add src/model_improvement/augmentation.py tests/model_improvement/test_augmentation_pipeline.py
git commit -m "feat: add Pedalboard-based augmentation pipeline for T4 invariance"
```

---

### Task 8: YouTube Piano Collection Script and Channel List

**Files:**
- Create: `scripts/collect_youtube_piano.py`
- Create: `data/youtube_piano_cache/channels.yaml`

**Step 1: Create curated channel list**

```yaml
# data/youtube_piano_cache/channels.yaml
# Curated list of YouTube channels with professional piano performances.
# Categories: recital (solo recitals), conservatory (student/faculty),
# competition (other competitions), misc (mixed classical content).
channels:
  - url: https://www.youtube.com/@taborpiano
    name: Tabor Piano
    category: recital
  - url: https://www.youtube.com/@Medici.tv
    name: medici.tv
    category: recital
  - url: https://www.youtube.com/@ClassicalMusicOnly
    name: Classical Music Only
    category: recital
  - url: https://www.youtube.com/@BerlinPhil
    name: Berliner Philharmoniker
    category: recital
  - url: https://www.youtube.com/@wigaborehall
    name: Wigmore Hall
    category: recital
  - url: https://www.youtube.com/@CarnegieHall
    name: Carnegie Hall
    category: recital
  - url: https://www.youtube.com/@royalalberthall
    name: Royal Albert Hall
    category: recital
  - url: https://www.youtube.com/@JuilliardSchool
    name: Juilliard School
    category: conservatory
  - url: https://www.youtube.com/@CurtisInstitute
    name: Curtis Institute of Music
    category: conservatory
  - url: https://www.youtube.com/@RoyalAcademyofMusic
    name: Royal Academy of Music
    category: conservatory
  - url: https://www.youtube.com/@colaborapiano
    name: Colabora Piano
    category: conservatory
  - url: https://www.youtube.com/@CliburnCompetition
    name: Cliburn Competition
    category: competition
  - url: https://www.youtube.com/@LeedsCompetition
    name: Leeds International Piano Competition
    category: competition
  - url: https://www.youtube.com/@QueenElisabethCompetition
    name: Queen Elisabeth Competition
    category: competition
  - url: https://www.youtube.com/@ChopinInstitute
    name: Chopin Institute
    category: competition
```

**Step 2: Write the collection script**

```python
# scripts/collect_youtube_piano.py
"""Collect unlabeled piano audio from YouTube for T4 augmentation invariance training.

Run from model/ directory:
    python scripts/collect_youtube_piano.py
    python scripts/collect_youtube_piano.py --max-videos-per-channel 50
    python scripts/collect_youtube_piano.py --skip-download --skip-augmentation

Each step is idempotent. Running again skips completed work.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT / "src"))

from model_improvement.youtube_piano import (
    discover_channel_videos,
    download_piano_audio,
    load_channel_list,
    segment_and_embed_piano,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YouTube piano audio collection for T4 invariance training"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_ROOT / "data" / "youtube_piano_cache",
        help="Output directory",
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=MODEL_ROOT / "data" / "youtube_piano_cache" / "channels.yaml",
        help="Path to channels YAML file",
    )
    parser.add_argument(
        "--max-videos-per-channel",
        type=int,
        default=100,
        help="Max videos to discover per channel",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-augmentation", action="store_true")
    args = parser.parse_args()

    logger.info("YouTube piano collection pipeline")
    logger.info("Cache dir: %s", args.cache_dir)
    t_start = time.time()

    # Step 1: Load channel list
    logger.info("=" * 60)
    logger.info("Step 1: Loading channel list...")
    channels = load_channel_list(args.channels_file)
    logger.info("Found %d channels", len(channels))

    # Step 2: Discover videos
    logger.info("=" * 60)
    logger.info("Step 2: Discovering videos...")
    all_videos = []
    for ch in channels:
        logger.info("  Discovering: %s", ch["name"])
        videos = discover_channel_videos(
            ch["url"], max_videos=args.max_videos_per_channel,
        )
        for v in videos:
            v["channel"] = ch["name"]
        all_videos.extend(videos)
        logger.info("  Found %d videos", len(videos))
    logger.info("Total videos discovered: %d", len(all_videos))

    # Step 3: Download audio
    if args.skip_download:
        logger.info("=" * 60)
        logger.info("Step 3: SKIPPED (--skip-download)")
    else:
        logger.info("=" * 60)
        logger.info("Step 3: Downloading audio...")
        records = download_piano_audio(all_videos, args.cache_dir)
        logger.info("Downloaded %d new recordings", len(records))

    # Step 4: Segment and extract clean embeddings
    if args.skip_embeddings:
        logger.info("=" * 60)
        logger.info("Step 4: SKIPPED (--skip-embeddings)")
    else:
        logger.info("=" * 60)
        logger.info("Step 4: Segmenting and extracting clean MuQ embeddings...")
        n = segment_and_embed_piano(args.cache_dir)
        logger.info("Processed %d new segments", n)

    # Step 5: Augment and extract augmented embeddings
    if args.skip_augmentation:
        logger.info("=" * 60)
        logger.info("Step 5: SKIPPED (--skip-augmentation)")
    else:
        logger.info("=" * 60)
        logger.info("Step 5: Generating augmented embeddings...")
        from model_improvement.augmentation import augment_and_embed_piano
        n_aug = augment_and_embed_piano(args.cache_dir)
        logger.info("Generated %d augmented embeddings", n_aug)

    # Summary
    logger.info("=" * 60)
    import jsonlines
    metadata_path = args.cache_dir / "metadata.jsonl"
    emb_dir = args.cache_dir / "muq_embeddings"
    aug_dir = args.cache_dir / "muq_embeddings_augmented"

    n_segments = 0
    if metadata_path.exists():
        with jsonlines.open(metadata_path) as reader:
            n_segments = sum(1 for _ in reader)
    n_emb = len(list(emb_dir.glob("*.pt"))) if emb_dir.exists() else 0
    n_aug_emb = len(list(aug_dir.glob("*.pt"))) if aug_dir.exists() else 0

    logger.info("Total segments: %d", n_segments)
    logger.info("Clean embeddings: %d", n_emb)
    logger.info("Augmented embeddings: %d", n_aug_emb)

    elapsed = time.time() - t_start
    logger.info("Pipeline complete in %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add scripts/collect_youtube_piano.py data/youtube_piano_cache/channels.yaml
git commit -m "feat: add YouTube piano collection script and channel list for T4"
```

---

### Task 9: Update hatch build config and run full test suite

**Files:**
- Modify: `pyproject.toml:81` (add new packages to wheel config)

**Step 1: Verify all new modules are included in the wheel**

The existing `tool.hatch.build.targets.wheel` only lists specific packages. Verify all new modules are importable:

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -c "from model_improvement.audio_utils import load_audio; from model_improvement.maestro import parse_maestro_audio_metadata; from model_improvement.youtube_piano import load_channel_list; from model_improvement.augmentation import augment_audio; print('All imports OK')"`

These should already work since they're under `src/model_improvement/` which is already in the hatch packages list.

**Step 2: Run the full test suite**

Run: `cd /Users/jdhiman/Documents/crescendai/model && python -m pytest tests/ -v`
Expected: All tests PASS (no regressions)

**Step 3: Commit if any config changes were needed**

```bash
cd /Users/jdhiman/Documents/crescendai/model
git add pyproject.toml
git commit -m "chore: ensure all new modules included in build config"
```

---

## Execution Order and Dependencies

```
Task 1 (audio_utils)
  |
  +---> Task 2 (T2 competition segmentation)
  |
  +---> Task 3 (MAESTRO metadata)
  |       |
  |       +---> Task 4 (MAESTRO embed + mapping tests)
  |       |
  |       +---> Task 5 (MAESTRO collection script)
  |
  +---> Task 6 (YouTube piano pipeline)
          |
          +---> Task 7 (augmentation pipeline)
          |
          +---> Task 8 (YouTube collection script + channels)

Task 9 (final verification) -- depends on all above
```

Tasks 2, 3, and 6 can proceed in parallel after Task 1.
Tasks 3-5 (MAESTRO) and Tasks 6-8 (YouTube) are independent tracks.

## Priority

**Critical path (must complete):** Tasks 1-5 (audio utils + T2 segmentation + T3 MAESTRO)
**Lower priority (additive):** Tasks 6-8 (T4 YouTube piano + augmentation)
**Gate check:** Task 9 (full test suite verification)
