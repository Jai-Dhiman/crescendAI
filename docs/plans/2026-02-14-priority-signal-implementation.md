# Priority Signal Validation - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate whether masterclass STOP/CONTINUE labels can be predicted from audio, using 65 moments from 5 videos.

**Architecture:** Notebook orchestrator (`model/notebooks/masterclass_experiments/`) calls into library modules (`model/src/masterclass_experiments/`). Two classifiers: Model B (logistic regression on 19 PercePiano quality scores) and Model A (logistic regression on 2048-dim MuQ embeddings). Leave-one-video-out cross-validation.

**Tech Stack:** Python, scikit-learn, PyTorch (MuQ extractor + PercePiano inference), soundfile, numpy, matplotlib

**Design Doc:** `docs/plans/2026-02-14-priority-signal-validation-design.md`

---

### Task 1: Module Scaffolding

**Files:**

- Create: `model/src/masterclass_experiments/__init__.py`
- Create: `model/tests/masterclass_experiments/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p model/src/masterclass_experiments
mkdir -p model/tests/masterclass_experiments
mkdir -p model/notebooks/masterclass_experiments
mkdir -p model/data/masterclass_cache/segments
mkdir -p model/data/masterclass_cache/muq_embeddings
mkdir -p model/data/masterclass_cache/quality_scores
```

**Step 2: Create `__init__.py` files**

`model/src/masterclass_experiments/__init__.py`:

```python
"""Masterclass priority signal validation experiment."""
```

`model/tests/masterclass_experiments/__init__.py`:

```python
```

**Step 3: Verify the package is importable**

Run: `cd model && uv run python -c "import masterclass_experiments; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add model/src/masterclass_experiments/__init__.py model/tests/masterclass_experiments/__init__.py
git commit -m "scaffold masterclass_experiments module"
```

---

### Task 2: Data Loading - Parse Moments

**Files:**

- Create: `model/src/masterclass_experiments/data.py`
- Create: `model/tests/masterclass_experiments/test_data.py`

**Context:** Moments live in `tools/masterclass-pipeline/all_moments.jsonl`. Each line is JSON with fields: `moment_id`, `video_id`, `teacher`, `stop_timestamp`, `playing_before_start`, `playing_before_end`, `feedback_start`, `feedback_end`, `feedback_summary`, `musical_dimension`, `severity`, `piece`, `confidence`, etc.

**Step 1: Write failing test for moment loading**

`model/tests/masterclass_experiments/test_data.py`:

```python
import json
import tempfile
from pathlib import Path

from masterclass_experiments.data import Moment, load_moments


def _write_moments(path: Path, moments: list[dict]) -> None:
    with open(path, "w") as f:
        for m in moments:
            f.write(json.dumps(m) + "\n")


SAMPLE_MOMENT = {
    "moment_id": "abc123",
    "video_id": "7FTdGbVCPyQ",
    "video_title": "Test Masterclass",
    "teacher": "Test Teacher",
    "stop_timestamp": 619.6,
    "feedback_start": 619.6,
    "feedback_end": 649.6,
    "playing_before_start": 533.28,
    "playing_before_end": 549.90,
    "transcript_text": "Some feedback",
    "feedback_summary": "Summary",
    "musical_dimension": "tone_color",
    "secondary_dimensions": ["interpretation"],
    "severity": "moderate",
    "feedback_type": "suggestion",
    "piece": "Chopin Ballade No. 1",
    "composer": "Chopin",
    "passage_description": None,
    "student_level": None,
    "stop_order": 1,
    "total_stops": 16,
    "time_spent_seconds": 30.0,
    "demonstrated": False,
    "extracted_at": "2026-02-15T05:07:20.257864+00:00",
    "extraction_model": "gpt-4o",
    "confidence": 0.7,
}


def test_load_moments_parses_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "moments.jsonl"
        _write_moments(path, [SAMPLE_MOMENT])

        moments = load_moments(path)

        assert len(moments) == 1
        m = moments[0]
        assert m.moment_id == "abc123"
        assert m.video_id == "7FTdGbVCPyQ"
        assert m.playing_before_start == 533.28
        assert m.playing_before_end == 549.90
        assert m.feedback_end == 649.6
        assert m.musical_dimension == "tone_color"


def test_load_moments_sorts_by_video_and_timestamp():
    m1 = {**SAMPLE_MOMENT, "moment_id": "a", "video_id": "vid1", "stop_timestamp": 200.0}
    m2 = {**SAMPLE_MOMENT, "moment_id": "b", "video_id": "vid1", "stop_timestamp": 100.0}
    m3 = {**SAMPLE_MOMENT, "moment_id": "c", "video_id": "vid2", "stop_timestamp": 50.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "moments.jsonl"
        _write_moments(path, [m1, m2, m3])

        moments = load_moments(path)

        assert [m.moment_id for m in moments] == ["b", "a", "c"]
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_data.py -v`
Expected: FAIL (cannot import `load_moments`)

**Step 3: Write implementation**

`model/src/masterclass_experiments/data.py`:

```python
"""Moment parsing and audio segment extraction for masterclass experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Moment:
    """A single teaching moment from a masterclass video."""

    moment_id: str
    video_id: str
    teacher: str
    stop_timestamp: float
    playing_before_start: float
    playing_before_end: float
    feedback_start: float
    feedback_end: float
    feedback_summary: str
    musical_dimension: str
    severity: str
    piece: str
    confidence: float


def load_moments(jsonl_path: Path) -> list[Moment]:
    """Parse moments JSONL file, sorted by video_id then stop_timestamp."""
    moments = []
    with open(jsonl_path) as f:
        for line in f:
            raw = json.loads(line)
            moments.append(
                Moment(
                    moment_id=raw["moment_id"],
                    video_id=raw["video_id"],
                    teacher=raw["teacher"],
                    stop_timestamp=raw["stop_timestamp"],
                    playing_before_start=raw["playing_before_start"],
                    playing_before_end=raw["playing_before_end"],
                    feedback_start=raw["feedback_start"],
                    feedback_end=raw["feedback_end"],
                    feedback_summary=raw["feedback_summary"],
                    musical_dimension=raw["musical_dimension"],
                    severity=raw["severity"],
                    piece=raw["piece"],
                    confidence=raw["confidence"],
                )
            )
    moments.sort(key=lambda m: (m.video_id, m.stop_timestamp))
    return moments
```

**Step 4: Run test to verify it passes**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_data.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add model/src/masterclass_experiments/data.py model/tests/masterclass_experiments/test_data.py
git commit -m "add moment loading from masterclass JSONL"
```

---

### Task 3: Segment Extraction - Identify STOP/CONTINUE Windows

**Files:**

- Modify: `model/src/masterclass_experiments/data.py`
- Modify: `model/tests/masterclass_experiments/test_data.py`

**Context:** STOP segments use `playing_before_start`/`playing_before_end` from each moment. CONTINUE segments are gaps between consecutive moments in the same video: from one moment's `feedback_end` to the next moment's `playing_before_start`. Only include CONTINUE segments where the gap is >= 5 seconds (short gaps are likely just brief pauses, not sustained playing).

**Step 1: Write failing test for segment identification**

Add to `model/tests/masterclass_experiments/test_data.py`:

```python
from masterclass_experiments.data import Segment, identify_segments


def test_identify_segments_creates_stop_segments():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        )
    ]

    segments = identify_segments(moments)

    stops = [s for s in segments if s.label == "stop"]
    assert len(stops) == 1
    assert stops[0].start_time == 80.0
    assert stops[0].end_time == 100.0
    assert stops[0].video_id == "vid1"


def test_identify_segments_creates_continue_between_moments():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid1",
            teacher="T",
            stop_timestamp=200.0,
            playing_before_start=180.0,
            playing_before_end=200.0,
            feedback_start=200.0,
            feedback_end=230.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments)

    continues = [s for s in segments if s.label == "continue"]
    assert len(continues) == 1
    # CONTINUE window: feedback_end of moment a (130.0) to playing_before_start of moment b (180.0)
    assert continues[0].start_time == 130.0
    assert continues[0].end_time == 180.0


def test_identify_segments_skips_short_continue_gaps():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid1",
            teacher="T",
            stop_timestamp=133.0,
            playing_before_start=131.0,
            playing_before_end=133.0,
            feedback_start=133.0,
            feedback_end=140.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments, min_continue_duration=5.0)

    continues = [s for s in segments if s.label == "continue"]
    assert len(continues) == 0


def test_identify_segments_no_continue_across_videos():
    moments = [
        Moment(
            moment_id="a",
            video_id="vid1",
            teacher="T",
            stop_timestamp=100.0,
            playing_before_start=80.0,
            playing_before_end=100.0,
            feedback_start=100.0,
            feedback_end=130.0,
            feedback_summary="s",
            musical_dimension="timing",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
        Moment(
            moment_id="b",
            video_id="vid2",
            teacher="T",
            stop_timestamp=200.0,
            playing_before_start=180.0,
            playing_before_end=200.0,
            feedback_start=200.0,
            feedback_end=230.0,
            feedback_summary="s",
            musical_dimension="dynamics",
            severity="moderate",
            piece="piece",
            confidence=0.7,
        ),
    ]

    segments = identify_segments(moments)

    continues = [s for s in segments if s.label == "continue"]
    assert len(continues) == 0
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_data.py::test_identify_segments_creates_stop_segments -v`
Expected: FAIL (cannot import `identify_segments`)

**Step 3: Write implementation**

Add to `model/src/masterclass_experiments/data.py`:

```python
from itertools import groupby


@dataclass
class Segment:
    """An audio segment labeled as STOP or CONTINUE."""

    segment_id: str
    video_id: str
    label: str  # "stop" or "continue"
    start_time: float  # seconds into the WAV
    end_time: float
    moment_id: str | None = None  # linked moment for STOP segments


def identify_segments(
    moments: list[Moment],
    min_continue_duration: float = 5.0,
) -> list[Segment]:
    """Identify STOP and CONTINUE segments from moments.

    STOP segments: playing window before each teacher intervention.
    CONTINUE segments: gaps between consecutive moments in the same video
    where the student was playing but the teacher did not stop.
    """
    segments: list[Segment] = []
    seq = 0

    for video_id, group in groupby(moments, key=lambda m: m.video_id):
        video_moments = list(group)

        for i, m in enumerate(video_moments):
            # STOP segment: student playing before the teacher stopped
            segments.append(
                Segment(
                    segment_id=f"stop_{seq:04d}",
                    video_id=m.video_id,
                    label="stop",
                    start_time=m.playing_before_start,
                    end_time=m.playing_before_end,
                    moment_id=m.moment_id,
                )
            )
            seq += 1

            # CONTINUE segment: gap between this moment's feedback_end
            # and next moment's playing_before_start (same video only)
            if i < len(video_moments) - 1:
                next_m = video_moments[i + 1]
                gap_start = m.feedback_end
                gap_end = next_m.playing_before_start

                if gap_end - gap_start >= min_continue_duration:
                    segments.append(
                        Segment(
                            segment_id=f"cont_{seq:04d}",
                            video_id=m.video_id,
                            label="continue",
                            start_time=gap_start,
                            end_time=gap_end,
                        )
                    )
                    seq += 1

    return segments
```

**Step 4: Run all tests to verify they pass**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_data.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add model/src/masterclass_experiments/data.py model/tests/masterclass_experiments/test_data.py
git commit -m "add STOP/CONTINUE segment identification from moments"
```

---

### Task 4: Audio Segment Slicing

**Files:**

- Modify: `model/src/masterclass_experiments/data.py`
- Modify: `model/tests/masterclass_experiments/test_data.py`

**Context:** Given segments with start/end times and a directory of WAV files (named `{video_id}.wav`), slice audio into individual WAV files. Uses `soundfile` which is already a project dependency.

**Step 1: Write failing test**

Add to `model/tests/masterclass_experiments/test_data.py`:

```python
import numpy as np
import soundfile as sf

from masterclass_experiments.data import extract_audio_segments


def test_extract_audio_segments_creates_wav_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_dir = Path(tmpdir) / "audio"
        wav_dir.mkdir()
        out_dir = Path(tmpdir) / "segments"
        out_dir.mkdir()

        # Create a 10-second mono WAV at 16kHz
        sr = 16000
        audio = np.random.randn(sr * 10).astype(np.float32)
        sf.write(wav_dir / "vid1.wav", audio, sr)

        segments = [
            Segment(
                segment_id="stop_0000",
                video_id="vid1",
                label="stop",
                start_time=1.0,
                end_time=3.0,
                moment_id="a",
            ),
            Segment(
                segment_id="cont_0001",
                video_id="vid1",
                label="continue",
                start_time=5.0,
                end_time=8.0,
            ),
        ]

        extract_audio_segments(segments, wav_dir, out_dir)

        # Check files were created
        assert (out_dir / "stop_0000.wav").exists()
        assert (out_dir / "cont_0001.wav").exists()

        # Check durations
        data0, sr0 = sf.read(out_dir / "stop_0000.wav")
        assert len(data0) == sr * 2  # 2 seconds

        data1, sr1 = sf.read(out_dir / "cont_0001.wav")
        assert len(data1) == sr * 3  # 3 seconds
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_data.py::test_extract_audio_segments_creates_wav_files -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `model/src/masterclass_experiments/data.py`:

```python
import soundfile as sf


def extract_audio_segments(
    segments: list[Segment],
    wav_dir: Path,
    output_dir: Path,
) -> None:
    """Slice audio segments from source WAV files.

    Args:
        segments: Segments with start/end times.
        wav_dir: Directory containing {video_id}.wav files.
        output_dir: Directory to write individual segment WAVs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache file info per video to avoid re-reading
    wav_info: dict[str, tuple[int, int]] = {}  # video_id -> (sr, total_frames)

    for seg in segments:
        out_path = output_dir / f"{seg.segment_id}.wav"
        if out_path.exists():
            continue

        wav_path = wav_dir / f"{seg.video_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        if seg.video_id not in wav_info:
            info = sf.info(wav_path)
            wav_info[seg.video_id] = (info.samplerate, info.frames)

        sr, total_frames = wav_info[seg.video_id]
        start_frame = int(seg.start_time * sr)
        end_frame = int(seg.end_time * sr)

        # Clamp to file bounds
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        data, _ = sf.read(wav_path, start=start_frame, stop=end_frame, dtype="float32")
        sf.write(out_path, data, sr)
```

**Step 4: Run all tests**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_data.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add model/src/masterclass_experiments/data.py model/tests/masterclass_experiments/test_data.py
git commit -m "add audio segment extraction from WAV files"
```

---

### Task 5: MuQ Feature Extraction

**Files:**

- Create: `model/src/masterclass_experiments/features.py`
- Create: `model/tests/masterclass_experiments/test_features.py`

**Context:** Reuse `MuQExtractor` from `audio_experiments.extractors.muq`. For each segment WAV, extract MuQ embeddings and apply stats pooling (mean + std concatenation) to get a fixed 2048-dim vector. Cache results as `.pt` files.

**Step 1: Write failing test**

`model/tests/masterclass_experiments/test_features.py`:

```python
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from masterclass_experiments.data import Segment
from masterclass_experiments.features import extract_muq_features, stats_pool


def test_stats_pool_produces_correct_shape():
    embeddings = torch.randn(10, 1024)
    pooled = stats_pool(embeddings)
    assert pooled.shape == (2048,)


def test_stats_pool_values():
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pooled = stats_pool(embeddings)
    # mean = [2.0, 3.0], std = [sqrt(2), sqrt(2)] ~ [1.414, 1.414]
    assert torch.allclose(pooled[:2], torch.tensor([2.0, 3.0]))
    assert pooled.shape == (4,)


def test_extract_muq_features_returns_dict_of_pooled_vectors():
    with tempfile.TemporaryDirectory() as tmpdir:
        seg_dir = Path(tmpdir) / "segments"
        seg_dir.mkdir()
        cache_dir = Path(tmpdir) / "muq_cache"

        # Create a dummy WAV
        sr = 24000
        audio = np.random.randn(sr * 2).astype(np.float32)
        sf.write(seg_dir / "stop_0000.wav", audio, sr)

        segments = [
            Segment(
                segment_id="stop_0000",
                video_id="vid1",
                label="stop",
                start_time=0.0,
                end_time=2.0,
                moment_id="a",
            )
        ]

        # Mock MuQExtractor to avoid loading real model
        fake_embedding = torch.randn(10, 1024)  # [T, 1024]
        with patch("masterclass_experiments.features.MuQExtractor") as MockExtractor:
            instance = MockExtractor.return_value
            instance.extract_from_file.return_value = fake_embedding

            features = extract_muq_features(segments, seg_dir, cache_dir)

        assert "stop_0000" in features
        assert features["stop_0000"].shape == (2048,)  # mean + std
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_features.py::test_stats_pool_produces_correct_shape -v`
Expected: FAIL

**Step 3: Write implementation**

`model/src/masterclass_experiments/features.py`:

```python
"""Feature extraction for masterclass segments."""

from __future__ import annotations

from pathlib import Path

import torch

from audio_experiments.extractors.muq import MuQExtractor
from masterclass_experiments.data import Segment


def stats_pool(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean + std pooling over time dimension.

    Args:
        embeddings: [T, D] tensor.

    Returns:
        [2*D] tensor (mean concatenated with std).
    """
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    return torch.cat([mean, std])


def extract_muq_features(
    segments: list[Segment],
    segment_dir: Path,
    cache_dir: Path,
) -> dict[str, torch.Tensor]:
    """Extract stats-pooled MuQ embeddings for each segment.

    Args:
        segments: List of segments to process.
        segment_dir: Directory containing segment WAV files.
        cache_dir: Directory to cache raw MuQ embeddings.

    Returns:
        Dict mapping segment_id to [2048] pooled embedding tensor.
    """
    extractor = MuQExtractor(cache_dir=cache_dir)
    features: dict[str, torch.Tensor] = {}

    for seg in segments:
        wav_path = segment_dir / f"{seg.segment_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Segment WAV not found: {wav_path}")

        # MuQExtractor handles caching internally
        raw = extractor.extract_from_file(wav_path)  # [T, 1024]
        pooled = stats_pool(raw)  # [2048]
        features[seg.segment_id] = pooled

    return features
```

**Step 4: Run tests**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_features.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add model/src/masterclass_experiments/features.py model/tests/masterclass_experiments/test_features.py
git commit -m "add MuQ feature extraction with stats pooling"
```

---

### Task 6: PercePiano Quality Score Inference

**Files:**

- Modify: `model/src/masterclass_experiments/features.py`
- Modify: `model/tests/masterclass_experiments/test_features.py`

**Context:** Load a trained `MuQStatsModel` checkpoint from GDrive (`gdrive:crescendai_data/checkpoints/strongest_paper/checkpoints/A1c_stratified_fold/fold0_best.ckpt`). Run inference on each segment's raw MuQ embeddings to get 19 quality dimension scores. The model uses internal pooling (mean+std) and an MLP classifier with sigmoid output.

**Step 1: Write failing test**

Add to `model/tests/masterclass_experiments/test_features.py`:

```python
from masterclass_experiments.features import extract_quality_scores


def test_extract_quality_scores_returns_19dim_vectors():
    raw_embeddings = {
        "stop_0000": torch.randn(10, 1024),
        "cont_0001": torch.randn(15, 1024),
    }

    # Mock the model to return 19 scores
    fake_scores = torch.sigmoid(torch.randn(1, 19))
    with patch("masterclass_experiments.features.MuQStatsModel") as MockModel:
        instance = MockModel.load_from_checkpoint.return_value
        instance.eval.return_value = instance
        instance.to.return_value = instance
        instance.pool.return_value = torch.randn(1, 2048)
        instance.clf.return_value = fake_scores

        scores = extract_quality_scores(
            raw_embeddings,
            checkpoint_path=Path("/tmp/fake.ckpt"),
        )

    assert "stop_0000" in scores
    assert "cont_0001" in scores
    assert scores["stop_0000"].shape == (19,)
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_features.py::test_extract_quality_scores_returns_19dim_vectors -v`
Expected: FAIL

**Step 3: Write implementation**

Add to `model/src/masterclass_experiments/features.py`:

```python
from audio_experiments.models.muq_models import MuQStatsModel


@torch.no_grad()
def extract_quality_scores(
    raw_embeddings: dict[str, torch.Tensor],
    checkpoint_path: Path,
) -> dict[str, torch.Tensor]:
    """Run PercePiano model inference to get 19-dim quality scores.

    Args:
        raw_embeddings: Dict mapping segment_id to [T, 1024] raw MuQ embeddings.
        checkpoint_path: Path to trained MuQStatsModel checkpoint.

    Returns:
        Dict mapping segment_id to [19] quality score tensor.
    """
    model = MuQStatsModel.load_from_checkpoint(checkpoint_path)
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    scores: dict[str, torch.Tensor] = {}

    for seg_id, emb in raw_embeddings.items():
        # Model expects [B, T, D] with attention mask
        x = emb.unsqueeze(0).to(device)  # [1, T, 1024]
        mask = torch.ones(1, emb.shape[0], device=device)  # [1, T]

        pooled = model.pool(x, mask)  # [1, 2048]
        pred = model.clf(pooled)  # [1, 19]
        scores[seg_id] = pred.squeeze(0).cpu()

    return scores
```

**Step 4: Run tests**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_features.py -v`
Expected: All PASS

**Step 5: Download PercePiano checkpoint from GDrive**

```bash
mkdir -p model/data/checkpoints/percepiano
rclone copy "gdrive:crescendai_data/checkpoints/strongest_paper/checkpoints/A1c_stratified_fold/fold0_best.ckpt" model/data/checkpoints/percepiano/
```

Verify: `ls -la model/data/checkpoints/percepiano/fold0_best.ckpt`

**Step 6: Commit**

```bash
git add model/src/masterclass_experiments/features.py model/tests/masterclass_experiments/test_features.py
git commit -m "add PercePiano quality score inference from checkpoint"
```

---

### Task 7: Model Training - Classifiers

**Files:**

- Create: `model/src/masterclass_experiments/models.py`
- Create: `model/tests/masterclass_experiments/test_models.py`

**Context:** Two sklearn classifiers. Model B: logistic regression on 19 quality scores. Model A: logistic regression on 2048-dim MuQ embeddings. Both return prediction probabilities for evaluation.

**Step 1: Write failing test**

`model/tests/masterclass_experiments/test_models.py`:

```python
import numpy as np

from masterclass_experiments.models import train_classifier


def test_train_classifier_returns_predictions():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 5))
    y = rng.integers(0, 2, size=20)

    result = train_classifier(
        X, y, train_idx=list(range(15)), test_idx=list(range(15, 20))
    )

    assert "y_pred_proba" in result
    assert "y_pred" in result
    assert "y_true" in result
    assert "coefficients" in result
    assert len(result["y_pred_proba"]) == 5
    assert len(result["y_true"]) == 5


def test_train_classifier_coefficients_shape():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 19))
    y = rng.integers(0, 2, size=30)

    result = train_classifier(
        X, y, train_idx=list(range(25)), test_idx=list(range(25, 30))
    )

    assert result["coefficients"].shape == (19,)
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_models.py -v`
Expected: FAIL

**Step 3: Write implementation**

`model/src/masterclass_experiments/models.py`:

```python
"""Binary classifiers for STOP/CONTINUE prediction."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: list[int],
    test_idx: list[int],
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """Train logistic regression and return predictions on test set.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N] (1=stop, 0=continue).
        train_idx: Indices for training.
        test_idx: Indices for testing.
        C: Regularization strength (inverse).
        max_iter: Maximum iterations.

    Returns:
        Dict with y_true, y_pred, y_pred_proba, coefficients.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)

    return {
        "y_true": y_test,
        "y_pred": clf.predict(X_test),
        "y_pred_proba": clf.predict_proba(X_test)[:, 1],
        "coefficients": clf.coef_.squeeze(),
    }
```

**Step 4: Run tests**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add model/src/masterclass_experiments/models.py model/tests/masterclass_experiments/test_models.py
git commit -m "add logistic regression classifier for STOP/CONTINUE"
```

---

### Task 8: Evaluation - Leave-One-Video-Out CV

**Files:**

- Create: `model/src/masterclass_experiments/evaluation.py`
- Create: `model/tests/masterclass_experiments/test_evaluation.py`

**Context:** Leave-one-video-out cross-validation. For each of 5 videos, train on 4 and test on 1. Compute AUC-ROC, accuracy, precision, recall across all held-out predictions. Also produce per-segment results for qualitative analysis.

**Step 1: Write failing test**

`model/tests/masterclass_experiments/test_evaluation.py`:

```python
import numpy as np

from masterclass_experiments.evaluation import leave_one_video_out_cv


def test_lovo_cv_returns_aggregate_metrics():
    rng = np.random.default_rng(42)
    n = 30
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n)
    video_ids = np.array(["v1"] * 10 + ["v2"] * 10 + ["v3"] * 10)

    results = leave_one_video_out_cv(X, y, video_ids)

    assert "auc" in results
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "per_segment" in results
    assert isinstance(results["auc"], float)
    assert len(results["per_segment"]) == n


def test_lovo_cv_per_segment_has_required_fields():
    rng = np.random.default_rng(42)
    n = 20
    X = rng.standard_normal((n, 5))
    y = rng.integers(0, 2, size=n)
    video_ids = np.array(["v1"] * 10 + ["v2"] * 10)

    results = leave_one_video_out_cv(X, y, video_ids)

    seg = results["per_segment"][0]
    assert "video_id" in seg
    assert "y_true" in seg
    assert "y_pred_proba" in seg
```

**Step 2: Run test to verify it fails**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_evaluation.py -v`
Expected: FAIL

**Step 3: Write implementation**

`model/src/masterclass_experiments/evaluation.py`:

```python
"""Evaluation utilities for masterclass priority signal experiment."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from masterclass_experiments.models import train_classifier


def leave_one_video_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    video_ids: np.ndarray,
    segment_ids: np.ndarray | None = None,
) -> dict:
    """Leave-one-video-out cross-validation.

    Args:
        X: Feature matrix [N, D].
        y: Binary labels [N].
        video_ids: Video ID per sample [N].
        segment_ids: Optional segment IDs for qualitative analysis [N].

    Returns:
        Dict with aggregate metrics and per-segment predictions.
    """
    unique_videos = np.unique(video_ids)
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    per_segment = []

    for held_out_video in unique_videos:
        test_mask = video_ids == held_out_video
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0].tolist()
        test_idx = np.where(test_mask)[0].tolist()

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        # Check both classes exist in training set
        if len(np.unique(y[train_mask])) < 2:
            continue

        result = train_classifier(X, y, train_idx, test_idx)

        all_y_true.extend(result["y_true"])
        all_y_pred.extend(result["y_pred"])
        all_y_proba.extend(result["y_pred_proba"])

        for i, idx in enumerate(test_idx):
            per_segment.append({
                "video_id": video_ids[idx],
                "segment_id": segment_ids[idx] if segment_ids is not None else str(idx),
                "y_true": int(result["y_true"][i]),
                "y_pred_proba": float(result["y_pred_proba"][i]),
            })

    all_y_true_arr = np.array(all_y_true)
    all_y_pred_arr = np.array(all_y_pred)
    all_y_proba_arr = np.array(all_y_proba)

    # Compute aggregate metrics
    has_both_classes = len(np.unique(all_y_true_arr)) == 2
    auc = float(roc_auc_score(all_y_true_arr, all_y_proba_arr)) if has_both_classes else 0.5
    accuracy = float(accuracy_score(all_y_true_arr, all_y_pred_arr))
    precision = float(precision_score(all_y_true_arr, all_y_pred_arr, zero_division=0))
    recall = float(recall_score(all_y_true_arr, all_y_pred_arr, zero_division=0))

    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "n_samples": len(all_y_true_arr),
        "n_stop": int(all_y_true_arr.sum()),
        "n_continue": int((1 - all_y_true_arr).sum()),
        "per_segment": per_segment,
    }
```

**Step 4: Run tests**

Run: `cd model && uv run pytest tests/masterclass_experiments/test_evaluation.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add model/src/masterclass_experiments/evaluation.py model/tests/masterclass_experiments/test_evaluation.py
git commit -m "add leave-one-video-out cross-validation evaluation"
```

---

### Task 9: Notebook Orchestrator

**Files:**

- Create: `model/notebooks/masterclass_experiments/01_priority_signal_validation.ipynb`

**Context:** Notebook ties everything together. Calls into `masterclass_experiments.*` modules. Seven sections following the existing notebook conventions from `model/CLAUDE.md`.

**Step 1: Create the notebook with these cells**

**Cell 1 (Markdown):**

```markdown
# Priority Signal Validation Experiment

Validates whether masterclass teaching moments (STOP/CONTINUE) can be predicted from audio.

- **Model B:** Logistic regression on 19 PercePiano quality scores
- **Model A:** Logistic regression on 2048-dim MuQ embeddings
- **Evaluation:** Leave-one-video-out cross-validation (5 folds)

Design doc: `docs/plans/2026-02-14-priority-signal-validation-design.md`
```

**Cell 2 (Code) - Setup:**

```python
import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
MODEL_ROOT = Path("../..").resolve()
sys.path.insert(0, str(MODEL_ROOT / "src"))

from masterclass_experiments.data import load_moments, identify_segments, extract_audio_segments
from masterclass_experiments.features import extract_muq_features, extract_quality_scores, stats_pool
from masterclass_experiments.evaluation import leave_one_video_out_cv
```

**Cell 3 (Code) - Config & Paths:**

```python
MOMENTS_PATH = Path("../../../../tools/masterclass-pipeline/all_moments.jsonl").resolve()
WAV_DIR = Path("../../../../tools/masterclass-pipeline/data/audio").resolve()
CACHE_DIR = MODEL_ROOT / "data" / "masterclass_cache"
SEGMENT_DIR = CACHE_DIR / "segments"
MUQ_CACHE_DIR = CACHE_DIR / "muq_embeddings"
CHECKPOINT_PATH = MODEL_ROOT / "data" / "checkpoints" / "percepiano" / "fold0_best.ckpt"

for d in [SEGMENT_DIR, MUQ_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Moments: {MOMENTS_PATH}")
print(f"WAV dir: {WAV_DIR}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
```

**Cell 4 (Code) - Data Preparation:**

```python
# Load and explore moments
moments = load_moments(MOMENTS_PATH)
print(f"Loaded {len(moments)} moments")

# Identify segments
segments = identify_segments(moments)
stop_segments = [s for s in segments if s.label == "stop"]
cont_segments = [s for s in segments if s.label == "continue"]
print(f"STOP segments: {len(stop_segments)}")
print(f"CONTINUE segments: {len(cont_segments)}")

# Duration stats
for label, segs in [("STOP", stop_segments), ("CONTINUE", cont_segments)]:
    durations = [s.end_time - s.start_time for s in segs]
    print(f"{label}: mean={np.mean(durations):.1f}s, median={np.median(durations):.1f}s, "
          f"min={np.min(durations):.1f}s, max={np.max(durations):.1f}s")

# Extract audio segments
extract_audio_segments(segments, WAV_DIR, SEGMENT_DIR)
print(f"Audio segments saved to {SEGMENT_DIR}")
```

**Cell 5 (Code) - Feature Extraction:**

```python
# Extract MuQ embeddings (this takes a few minutes)
print("Extracting MuQ features...")
muq_features = extract_muq_features(segments, SEGMENT_DIR, MUQ_CACHE_DIR)
print(f"Extracted MuQ features for {len(muq_features)} segments")

# Also keep raw embeddings for PercePiano inference
from audio_experiments.extractors.muq import MuQExtractor
extractor = MuQExtractor(cache_dir=MUQ_CACHE_DIR)
raw_embeddings = {}
for seg in segments:
    wav_path = SEGMENT_DIR / f"{seg.segment_id}.wav"
    raw_embeddings[seg.segment_id] = extractor.extract_from_file(wav_path)

# Extract quality scores
print("Running PercePiano inference...")
quality_scores = extract_quality_scores(raw_embeddings, CHECKPOINT_PATH)
print(f"Extracted quality scores for {len(quality_scores)} segments")
```

**Cell 6 (Code) - Run Both Models:**

```python
from audio_experiments.constants import PERCEPIANO_DIMENSIONS

# Build feature matrices
segment_ids = np.array([s.segment_id for s in segments])
video_ids = np.array([s.video_id for s in segments])
labels = np.array([1 if s.label == "stop" else 0 for s in segments])

# Model B features: 19 quality scores
X_quality = np.stack([quality_scores[sid].numpy() for sid in segment_ids])
print(f"Model B features: {X_quality.shape}")

# Model A features: 2048-dim MuQ embeddings
X_muq = np.stack([muq_features[sid].numpy() for sid in segment_ids])
print(f"Model A features: {X_muq.shape}")

# Run leave-one-video-out CV
print("\n--- Model B (Quality Scores) ---")
results_b = leave_one_video_out_cv(X_quality, labels, video_ids, segment_ids)
print(f"AUC: {results_b['auc']:.3f}")
print(f"Accuracy: {results_b['accuracy']:.3f}")
print(f"Precision: {results_b['precision']:.3f}")
print(f"Recall: {results_b['recall']:.3f}")
print(f"Samples: {results_b['n_samples']} ({results_b['n_stop']} stop, {results_b['n_continue']} continue)")

print("\n--- Model A (MuQ Embeddings) ---")
results_a = leave_one_video_out_cv(X_muq, labels, video_ids, segment_ids)
print(f"AUC: {results_a['auc']:.3f}")
print(f"Accuracy: {results_a['accuracy']:.3f}")
print(f"Precision: {results_a['precision']:.3f}")
print(f"Recall: {results_a['recall']:.3f}")
```

**Cell 7 (Code) - Analysis & Visualization:**

```python
import matplotlib.pyplot as plt
from masterclass_experiments.models import train_classifier

# Comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

metrics = ["auc", "accuracy", "precision", "recall"]
model_b_vals = [results_b[m] for m in metrics]
model_a_vals = [results_a[m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35
axes[0].bar(x - width/2, model_b_vals, width, label="Model B (Quality)")
axes[0].bar(x + width/2, model_a_vals, width, label="Model A (MuQ)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].set_ylabel("Score")
axes[0].set_title("Model Comparison")
axes[0].legend()
axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
axes[0].set_ylim(0, 1)

# Qualitative: worst false negatives (teacher stopped but model said continue)
print("\n--- False Negatives (Model B) ---")
fn = [s for s in results_b["per_segment"] if s["y_true"] == 1 and s["y_pred_proba"] < 0.5]
fn.sort(key=lambda s: s["y_pred_proba"])
for s in fn[:5]:
    print(f"  {s['segment_id']} (video {s['video_id']}): proba={s['y_pred_proba']:.3f}")

print("\n--- False Positives (Model B) ---")
fp = [s for s in results_b["per_segment"] if s["y_true"] == 0 and s["y_pred_proba"] > 0.5]
fp.sort(key=lambda s: -s["y_pred_proba"])
for s in fp[:5]:
    print(f"  {s['segment_id']} (video {s['video_id']}): proba={s['y_pred_proba']:.3f}")

# Model B coefficient analysis
print("\n--- Model B: Which quality dimensions predict STOP? ---")
full_result = train_classifier(
    X_quality, labels, list(range(len(labels))), list(range(len(labels)))
)
coefs = full_result["coefficients"]
sorted_idx = np.argsort(np.abs(coefs))[::-1]
for i in sorted_idx:
    print(f"  {PERCEPIANO_DIMENSIONS[i]:25s}: {coefs[i]:+.3f}")

axes[1].barh(
    range(19),
    coefs[sorted_idx],
    tick_label=[PERCEPIANO_DIMENSIONS[i] for i in sorted_idx],
)
axes[1].set_xlabel("Coefficient")
axes[1].set_title("Model B: Quality Dimension Importance")
plt.tight_layout()
plt.show()
```

**Step 2: Commit**

```bash
git add model/notebooks/masterclass_experiments/01_priority_signal_validation.ipynb
git commit -m "add priority signal validation notebook"
```

---

### Task 10: End-to-End Smoke Test

**Step 1: Run all unit tests**

Run: `cd model && uv run pytest tests/masterclass_experiments/ -v`
Expected: All PASS

**Step 2: Run data pipeline on real data to verify it loads**

```bash
cd model && uv run python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from masterclass_experiments.data import load_moments, identify_segments

moments = load_moments(Path('../tools/masterclass-pipeline/all_moments.jsonl'))
segments = identify_segments(moments)
stops = [s for s in segments if s.label == 'stop']
conts = [s for s in segments if s.label == 'continue']
print(f'Moments: {len(moments)}, Stops: {len(stops)}, Continues: {len(conts)}')
for v in set(s.video_id for s in segments):
    v_stops = len([s for s in stops if s.video_id == v])
    v_conts = len([s for s in conts if s.video_id == v])
    print(f'  {v}: {v_stops} stops, {v_conts} continues')
"
```

Expected: `Moments: 65, Stops: 65, Continues: <some number>` with per-video breakdown

**Step 3: Final commit if any cleanup needed**

```bash
git add -A model/src/masterclass_experiments/ model/tests/masterclass_experiments/
git commit -m "priority signal validation experiment: complete"
```
