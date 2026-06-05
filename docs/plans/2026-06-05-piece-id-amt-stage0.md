# Piece-ID AMT Stage-0 Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Produce a defensible KILL/TUNE/PROCEED verdict for symbolic note-to-note piece identification on the 16 `practice_eval` recordings before any Rust/WASM is written.
**Spec:** docs/specs/2026-06-05-piece-id-amt-stage0-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / model/CLAUDE.md). Python = uv not pip; partitura not music21; explicit exceptions, no silent fallbacks; no emojis; Trackio for tracking.

---

## Task Groups

```
Group 0 (parallel): Task 1 (notes.py), Task 2 (transcribe.py)
  — foundation; every later module imports notes.py; transcribe.py unblocks the operational cache run
  [SHIPS INDEPENDENTLY: notes.py and transcribe.py usable standalone before bakeoff exists]

Group 1 (parallel, depends on Group 0): Task 3 (windowing.py), Task 4 (note_chroma.py), Task 5 (corruption.py), Task 6 (open_set.py), Task 7 (decision.py re-threshold)
  — independent utility modules; none imports the others

Group 2a (sequential, depends on Group 1): Task 8 (matchers/base.py + note_chroma_matcher.py C1)
  — must run first; rewrites matchers/base.py Matcher protocol to accept list[Note]; C2/C3/C4 depend on the new signature

Group 2b (parallel, depends on Group 2a): Task 9 (landmark.py C2), Task 10 (dtw_ceiling.py C3 note-based), Task 11 (chroma_seq_dtw.py C4)
  — matchers; each touches a different file; all import from updated matchers/base.py

Group 3 (sequential, depends on Group 2): Task 12 (bakeoff.py orchestrator + CLI + dead-file deletions)
  — integrates all matchers + all utilities; CLI smoke test; deletes superseded files
```

---

## Task 1: notes.py — Note substrate and loaders

**Group:** 0 (parallel with Task 2)

**Behavior being verified:** `load_amt_notes` converts a flat `{onset,offset,pitch,velocity}` JSON list to `list[Note]` sorted by onset; `load_score_notes` converts score JSON `bars[].notes[]` shape to the same `list[Note]`.

**Interface under test:** `Note`, `load_amt_notes(path) -> list[Note]`, `load_score_notes(path) -> list[Note]`

**Files:**
- Create: `model/src/piece_id_eval/notes.py`
- Create: `model/tests/piece_id_eval/test_notes.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_notes.py
"""Verify Note loaders through their public interface on committed fixtures."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from piece_id_eval.notes import Note, load_amt_notes, load_score_notes

REPO_ROOT = Path(__file__).resolve().parents[4]
BACH_SCORE = REPO_ROOT / "model/data/scores/bach.prelude.bwv_846.json"


def _write_amt_fixture(tmp_path: Path) -> Path:
    """Write a minimal flat AMT notes JSON with 3 notes out of onset order."""
    notes = [
        {"onset": 1.5, "offset": 2.0, "pitch": 62, "velocity": 70},
        {"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 80},
        {"onset": 3.0, "offset": 3.5, "pitch": 64, "velocity": 90},
    ]
    p = tmp_path / "amt_notes.json"
    p.write_text(json.dumps(notes))
    return p


def test_load_amt_notes_returns_notes_sorted_by_onset(tmp_path: Path) -> None:
    p = _write_amt_fixture(tmp_path)
    notes = load_amt_notes(p)
    assert len(notes) == 3
    onsets = [n.onset for n in notes]
    assert onsets == sorted(onsets), f"not sorted: {onsets}"


def test_load_amt_notes_fields_match_fixture(tmp_path: Path) -> None:
    p = _write_amt_fixture(tmp_path)
    notes = load_amt_notes(p)
    first = notes[0]  # onset=0.0 after sorting
    assert isinstance(first, Note)
    assert first.onset == pytest.approx(0.0)
    assert first.offset == pytest.approx(0.5)
    assert first.pitch == 60
    assert first.velocity == 80


def test_load_score_notes_returns_notes_sorted_by_onset() -> None:
    if not BACH_SCORE.exists():
        pytest.skip("bach score fixture not present")
    notes = load_score_notes(BACH_SCORE)
    assert len(notes) > 0
    onsets = [n.onset for n in notes]
    assert onsets == sorted(onsets), f"not sorted: {onsets[:5]}"


def test_load_score_notes_fields_plausible() -> None:
    if not BACH_SCORE.exists():
        pytest.skip("bach score fixture not present")
    notes = load_score_notes(BACH_SCORE)
    first = notes[0]
    assert isinstance(first, Note)
    assert first.onset >= 0.0
    assert first.offset > first.onset
    assert 21 <= first.pitch <= 108  # piano range
    assert 0 <= first.velocity <= 127


def test_load_amt_notes_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_amt_notes(tmp_path / "nonexistent.json")


def test_note_namedtuple_fields() -> None:
    n = Note(onset=0.0, offset=0.5, pitch=60, velocity=80)
    assert n.onset == 0.0
    assert n.offset == 0.5
    assert n.pitch == 60
    assert n.velocity == 80
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_notes.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.notes'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/notes.py
"""Symbolic note substrate and loaders.

Both AMT output (flat list of {onset,offset,pitch,velocity}) and score JSON
(bars[].notes[].{onset_seconds,duration_seconds,pitch,velocity}) are
normalised to the same Note NamedTuple, sorted ascending by onset.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple


class Note(NamedTuple):
    """A single symbolic note in seconds. Onset and offset are absolute seconds."""
    onset: float
    offset: float
    pitch: int
    velocity: int


def load_amt_notes(path: Path) -> list[Note]:
    """Load AMT notes from a flat JSON list of {onset,offset,pitch,velocity} dicts.

    Returns notes sorted ascending by onset.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the JSON is not a list.
    """
    if not path.exists():
        raise FileNotFoundError(f"AMT notes file not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list, got {type(raw).__name__}: {path}")
    notes = [
        Note(
            onset=float(d["onset"]),
            offset=float(d["offset"]),
            pitch=int(d["pitch"]),
            velocity=int(d.get("velocity", 80)),
        )
        for d in raw
        if isinstance(d, dict) and "onset" in d and "pitch" in d
    ]
    notes.sort(key=lambda n: n.onset)
    return notes


def load_score_notes(path: Path) -> list[Note]:
    """Load score notes from a score JSON with bars[].notes[] structure.

    Each note has onset_seconds, duration_seconds, pitch, velocity.
    Returns notes sorted ascending by onset.

    Raises:
        FileNotFoundError: if path does not exist.
        KeyError: if the JSON lacks expected structure.
    """
    if not path.exists():
        raise FileNotFoundError(f"Score JSON not found: {path}")
    body = json.loads(path.read_text())
    notes: list[Note] = []
    for bar in body.get("bars") or []:
        for n in bar.get("notes") or []:
            onset = float(n["onset_seconds"])
            duration = float(n.get("duration_seconds", 0.25))
            notes.append(Note(
                onset=onset,
                offset=onset + duration,
                pitch=int(n["pitch"]),
                velocity=int(n.get("velocity", 80)),
            ))
    notes.sort(key=lambda n: n.onset)
    return notes
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_notes.py -p no:cov -x
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/notes.py model/tests/piece_id_eval/test_notes.py && git commit -m "feat(piece-id): add Note substrate and AMT/score loaders (notes.py)"
```

---

## Task 2: transcribe.py — AMT-notes cache builder

**Group:** 0 (parallel with Task 1)

**Behavior being verified:** `ensure_amt_notes` writes an `amt_notes.json` file on first call, returns the path, and skips the HTTP request on a second call (idempotent). Tested against a stub AMT HTTP server using the same pattern as `test_amt_regen.py`.

**Interface under test:** `ensure_amt_notes(audio_path, out_path, amt_url, force=False) -> Path`

**Files:**
- Create: `model/src/piece_id_eval/transcribe.py`
- Create: `model/tests/piece_id_eval/test_transcribe.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_transcribe.py
"""Verify ensure_amt_notes against a stub AMT HTTP server.

Pattern copied from tests/chroma_dtw_eval/test_amt_regen.py.
"""
from __future__ import annotations

import http.server
import json
import socketserver
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from piece_id_eval.transcribe import ensure_amt_notes


def _write_stub_wav(path: Path, duration_seconds: float = 3.0, sr: int = 16000) -> None:
    """Write a silent mono 16kHz WAV fixture."""
    samples = np.zeros(int(duration_seconds * sr), dtype=np.float32)
    sf.write(str(path), samples, sr)


_CANNED_NOTES = [
    {"onset": 0.0, "offset": 0.5, "pitch": 60, "velocity": 80},
    {"onset": 0.5, "offset": 1.0, "pitch": 62, "velocity": 75},
    {"onset": 1.0, "offset": 1.5, "pitch": 64, "velocity": 70},
]


class _StubAmtHandler(http.server.BaseHTTPRequestHandler):
    call_count: int = 0

    def log_message(self, *a, **k) -> None:  # silence
        pass

    def do_POST(self) -> None:  # noqa: N802
        _StubAmtHandler.call_count += 1
        n = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(n)
        body = json.dumps({"midi_notes": _CANNED_NOTES}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def stub_amt_server():
    _StubAmtHandler.call_count = 0
    srv = socketserver.TCPServer(("127.0.0.1", 0), _StubAmtHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}/transcribe"
    srv.shutdown()
    srv.server_close()


def test_ensure_amt_notes_writes_file(tmp_path: Path, stub_amt_server: str) -> None:
    audio = tmp_path / "test.wav"
    _write_stub_wav(audio)
    out = tmp_path / "amt_notes.json"

    result = ensure_amt_notes(audio, out, amt_url=stub_amt_server)

    assert result == out
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) > 0
    assert "onset" in data[0] and "pitch" in data[0]


def test_ensure_amt_notes_is_idempotent(tmp_path: Path, stub_amt_server: str) -> None:
    audio = tmp_path / "test.wav"
    _write_stub_wav(audio)
    out = tmp_path / "amt_notes.json"

    ensure_amt_notes(audio, out, amt_url=stub_amt_server)
    calls_after_first = _StubAmtHandler.call_count

    # Second call must NOT hit the server again.
    ensure_amt_notes(audio, out, amt_url=stub_amt_server)
    assert _StubAmtHandler.call_count == calls_after_first, (
        "ensure_amt_notes made an HTTP request on second call (not idempotent)"
    )


def test_ensure_amt_notes_force_retranscribes(tmp_path: Path, stub_amt_server: str) -> None:
    audio = tmp_path / "test.wav"
    _write_stub_wav(audio)
    out = tmp_path / "amt_notes.json"

    ensure_amt_notes(audio, out, amt_url=stub_amt_server)
    calls_after_first = _StubAmtHandler.call_count

    ensure_amt_notes(audio, out, amt_url=stub_amt_server, force=True)
    assert _StubAmtHandler.call_count > calls_after_first, (
        "force=True did not retranscribe"
    )


def test_ensure_amt_notes_raises_on_missing_audio(tmp_path: Path, stub_amt_server: str) -> None:
    with pytest.raises(FileNotFoundError):
        ensure_amt_notes(
            tmp_path / "missing.wav",
            tmp_path / "out.json",
            amt_url=stub_amt_server,
        )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_transcribe.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.transcribe'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/transcribe.py
"""AMT-notes cache builder.

Calls the local AMT server to transcribe an audio file and writes the result
to a JSON cache. Idempotent: skips the HTTP call if the cache already exists
(unless force=True).

Reuses _read_wav_16k_mono and _transcribe_clip from chroma_dtw_eval.amt_regen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from chroma_dtw_eval.amt_regen import (
    DEFAULT_AMT_URL,
    _read_wav_16k_mono,
    _transcribe_clip,
)

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval"
DEFAULT_AMT_NOTES_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval_pseudo"


def ensure_amt_notes(
    audio_path: Path,
    out_path: Path,
    amt_url: str = DEFAULT_AMT_URL,
    force: bool = False,
) -> Path:
    """Transcribe audio_path via AMT and write notes to out_path as JSON.

    Idempotent: if out_path already exists and force=False, returns immediately.

    Args:
        audio_path: path to a WAV file (any sample rate; resampled to 16kHz).
        out_path: destination JSON path; parent directory is created if needed.
        amt_url: URL of the local AMT /transcribe endpoint.
        force: if True, re-transcribe even if out_path already exists.

    Returns:
        out_path (the written or pre-existing cache file).

    Raises:
        FileNotFoundError: if audio_path does not exist.
        AmtRegenError: if the AMT server fails after retries.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if out_path.exists() and not force:
        return out_path

    audio_16k = _read_wav_16k_mono(audio_path)
    notes = _transcribe_clip(audio_16k, amt_url)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notes, indent=2))
    return out_path


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe cached audio for piece-ID slugs via local AMT server."
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        required=True,
        help="Slug names under data/evals/practice_eval/ (e.g. bach_prelude_c_wtc1)",
    )
    parser.add_argument(
        "--amt-url",
        default=DEFAULT_AMT_URL,
        help=f"AMT endpoint URL (default: {DEFAULT_AMT_URL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if cache already exists.",
    )
    parser.add_argument(
        "--practice-root",
        type=Path,
        default=DEFAULT_PRACTICE_ROOT,
        help="Root of practice_eval audio directories.",
    )
    parser.add_argument(
        "--notes-root",
        type=Path,
        default=DEFAULT_AMT_NOTES_ROOT,
        help="Root of practice_eval_pseudo cache directories.",
    )
    args = parser.parse_args()

    import yaml  # pyyaml; already in pyproject.toml deps

    for slug in args.slugs:
        slug_dir = args.practice_root / slug
        candidates_file = slug_dir / "candidates.yaml"
        if not candidates_file.exists():
            print(f"[SKIP] {slug}: no candidates.yaml at {candidates_file}", file=sys.stderr)
            continue
        with candidates_file.open() as f:
            candidates = yaml.safe_load(f)
        recordings = [r for r in (candidates.get("recordings") or []) if r.get("approved")]
        for rec in recordings:
            video_id = rec["video_id"]
            audio_path = slug_dir / "audio" / f"{video_id}.wav"
            if not audio_path.exists():
                print(f"[SKIP] {slug}/{video_id}: audio not cached at {audio_path}", file=sys.stderr)
                continue
            out_path = args.notes_root / slug / video_id / "amt_notes.json"
            print(f"[transcribe] {slug}/{video_id} -> {out_path}")
            ensure_amt_notes(audio_path, out_path, amt_url=args.amt_url, force=args.force)
            print(f"[done]      {slug}/{video_id}")


if __name__ == "__main__":
    _cli_main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_transcribe.py -p no:cov -x
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/transcribe.py model/tests/piece_id_eval/test_transcribe.py && git commit -m "feat(piece-id): add AMT-notes cache builder with CLI (transcribe.py)"
```

---

## Task 3: windowing.py — Arbitrary-start subsequence sampler

**Group:** 1 (parallel with Tasks 4–7; depends on Group 0)

**Behavior being verified:** `sample_windows` returns `n_starts` windows of the requested duration (in seconds), each a sublist of `Note` with onset in the sampled range; `window_seconds=None` returns a single full-piece window; results are deterministic for the same seed.

**Interface under test:** `sample_windows(notes, window_seconds, n_starts, seed) -> list[list[Note]]`

**Files:**
- Create: `model/src/piece_id_eval/windowing.py`
- Create: `model/tests/piece_id_eval/test_windowing.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_windowing.py
"""Verify sample_windows through its public interface on synthetic notes."""
from __future__ import annotations

from piece_id_eval.notes import Note
from piece_id_eval.windowing import sample_windows


def _make_notes(n: int, spacing: float = 0.5) -> list[Note]:
    """Create n synthetic notes at regular spacing."""
    return [Note(onset=i * spacing, offset=i * spacing + 0.3, pitch=60, velocity=80) for i in range(n)]


def test_sample_windows_count_matches_n_starts() -> None:
    notes = _make_notes(100)  # 50 seconds at 0.5s spacing
    windows = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=42)
    assert len(windows) == 5


def test_sample_windows_notes_within_window() -> None:
    notes = _make_notes(100)
    windows = sample_windows(notes, window_seconds=10.0, n_starts=3, seed=0)
    for win in windows:
        if len(win) == 0:
            continue
        duration = win[-1].onset - win[0].onset
        assert duration <= 10.0 + 1e-6, f"window spans {duration:.3f}s > 10s"


def test_sample_windows_full_returns_single_window() -> None:
    notes = _make_notes(20)
    windows = sample_windows(notes, window_seconds=None, n_starts=5, seed=0)
    assert len(windows) == 1
    assert windows[0] == notes


def test_sample_windows_deterministic() -> None:
    notes = _make_notes(100)
    a = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=7)
    b = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=7)
    assert a == b


def test_sample_windows_different_seeds_differ() -> None:
    notes = _make_notes(100)
    a = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=1)
    b = sample_windows(notes, window_seconds=10.0, n_starts=5, seed=2)
    # Different seeds should (almost certainly) produce different start offsets
    starts_a = [w[0].onset if w else None for w in a]
    starts_b = [w[0].onset if w else None for w in b]
    assert starts_a != starts_b


def test_sample_windows_short_recording_returns_full() -> None:
    """If recording is shorter than window, return full recording as single window."""
    notes = _make_notes(4)  # 2 seconds total
    windows = sample_windows(notes, window_seconds=30.0, n_starts=5, seed=0)
    assert len(windows) == 1
    assert windows[0] == notes
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_windowing.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.windowing'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/windowing.py
"""Arbitrary-start subsequence window sampler.

Given a list of Notes and a window length in seconds, samples n_starts
uniformly random start offsets and returns subsets of notes falling within
each window. Deterministic via seed.

window_seconds=None -> single full-piece window (no start sampling).
If the recording duration < window_seconds, falls back to the full recording
as a single window.
"""
from __future__ import annotations

import random

from piece_id_eval.notes import Note


def sample_windows(
    notes: list[Note],
    window_seconds: float | None,
    n_starts: int,
    seed: int,
) -> list[list[Note]]:
    """Sample up to n_starts windows of window_seconds duration from notes.

    Args:
        notes: sorted list of Note (ascending onset).
        window_seconds: window duration in seconds. None -> full piece.
        n_starts: number of random start offsets to sample.
        seed: RNG seed for determinism.

    Returns:
        List of note sublists, each covering [start, start + window_seconds).
        If window_seconds is None or recording is shorter than window_seconds,
        returns [notes] (the full recording as a single window).
    """
    if not notes:
        return [[]]

    if window_seconds is None:
        return [notes]

    recording_duration = notes[-1].onset - notes[0].onset
    if recording_duration <= window_seconds:
        return [notes]

    rng = random.Random(seed)
    first_onset = notes[0].onset
    max_start = notes[-1].onset - window_seconds

    windows: list[list[Note]] = []
    for _ in range(n_starts):
        start = rng.uniform(first_onset, max_start)
        end = start + window_seconds
        window = [n for n in notes if start <= n.onset < end]
        windows.append(window)
    return windows
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_windowing.py -p no:cov -x
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/windowing.py model/tests/piece_id_eval/test_windowing.py && git commit -m "feat(piece-id): add arbitrary-start window sampler (windowing.py)"
```

---

## Task 4: note_chroma.py — Note-derived chroma features

**Group:** 1 (parallel with Tasks 3, 5, 6, 7; depends on Group 0)

**Behavior being verified:** `chroma_vector` on a list of pure-C notes produces a (12,) array with bin 0 dominant and L2 norm ~1.0; `chroma_sequence` on regularly-spaced notes produces a (12, T) array with T > 0.

**Interface under test:** `chroma_vector(notes) -> np.ndarray`, `chroma_sequence(notes, frame_seconds) -> np.ndarray`

**Files:**
- Create: `model/src/piece_id_eval/note_chroma.py`
- Create: `model/tests/piece_id_eval/test_note_chroma.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_note_chroma.py
"""Verify chroma_vector and chroma_sequence through their public interfaces."""
from __future__ import annotations

import numpy as np
import pytest

from piece_id_eval.note_chroma import chroma_sequence, chroma_vector
from piece_id_eval.notes import Note


def _c_notes(n: int = 8) -> list[Note]:
    """n notes all on C (pitch 60, pc=0)."""
    return [Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=60, velocity=80) for i in range(n)]


def _mixed_notes() -> list[Note]:
    """Notes covering pitch classes 0 (C), 4 (E), 7 (G)."""
    return [
        Note(onset=0.0, offset=0.4, pitch=60, velocity=80),  # C
        Note(onset=0.5, offset=0.9, pitch=64, velocity=80),  # E
        Note(onset=1.0, offset=1.4, pitch=67, velocity=80),  # G
    ]


def test_chroma_vector_shape() -> None:
    cv = chroma_vector(_c_notes())
    assert cv.shape == (12,), f"expected (12,), got {cv.shape}"


def test_chroma_vector_l2_normalized() -> None:
    cv = chroma_vector(_c_notes())
    norm = float(np.linalg.norm(cv))
    assert abs(norm - 1.0) < 1e-6, f"expected unit vector, norm={norm}"


def test_chroma_vector_dominant_bin_for_c_notes() -> None:
    cv = chroma_vector(_c_notes())
    # C notes -> pitch class 0 should be dominant
    assert np.argmax(cv) == 0, f"expected bin 0 dominant, got bin {np.argmax(cv)}"


def test_chroma_vector_mixed_notes_has_three_nonzero_bins() -> None:
    cv = chroma_vector(_mixed_notes())
    nonzero = int(np.sum(cv > 0))
    assert nonzero == 3, f"expected 3 nonzero bins, got {nonzero}"


def test_chroma_sequence_shape() -> None:
    notes = [Note(onset=i * 0.25, offset=i * 0.25 + 0.2, pitch=60 + (i % 12), velocity=80) for i in range(40)]
    cs = chroma_sequence(notes, frame_seconds=0.5)
    assert cs.shape[0] == 12, f"expected 12 rows, got {cs.shape[0]}"
    assert cs.shape[1] > 0, "expected at least 1 frame"


def test_chroma_sequence_each_frame_normalized() -> None:
    notes = [Note(onset=i * 0.25, offset=i * 0.25 + 0.2, pitch=60, velocity=80) for i in range(40)]
    cs = chroma_sequence(notes, frame_seconds=0.5)
    norms = np.linalg.norm(cs, axis=0)
    # Every non-zero frame should be unit normalized
    for t, norm in enumerate(norms):
        if norm > 0:
            assert abs(norm - 1.0) < 1e-5, f"frame {t} norm={norm}"


def test_chroma_vector_empty_notes_raises() -> None:
    with pytest.raises(ValueError):
        chroma_vector([])
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_note_chroma.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.note_chroma'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/note_chroma.py
"""Note-derived chroma features (key-dependent; no OTI/transposition invariance).

C1 (NoteChromaMatcher) uses chroma_vector for global bag-of-notes chroma.
C4 (ChromaSeqDtwMatcher) uses chroma_sequence for frame-level chroma.

Both functions are key-dependent: pitch-class 0 = C, no cyclic normalisation.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.notes import Note


def chroma_vector(notes: list[Note]) -> np.ndarray:
    """Aggregate all notes into a single 12-bin chroma vector, L2-normalised.

    Each note contributes its velocity (float) to its pitch-class bin.
    The result is the unit-norm version.

    Args:
        notes: non-empty list of Note.

    Returns:
        np.ndarray of shape (12,), float64, L2-normalised.

    Raises:
        ValueError: if notes is empty.
    """
    if not notes:
        raise ValueError("chroma_vector requires at least one note")
    cv = np.zeros(12, dtype=np.float64)
    for n in notes:
        cv[n.pitch % 12] += float(n.velocity)
    norm = np.linalg.norm(cv)
    if norm > 0:
        cv /= norm
    return cv


def chroma_sequence(notes: list[Note], frame_seconds: float) -> np.ndarray:
    """Compute a frame-level chroma matrix from notes.

    Time axis spans [notes[0].onset, notes[-1].offset) quantised to frame_seconds.
    Each note's velocity is added to all frames it overlaps (onset <= frame_start < offset).
    Each frame is L2-normalised independently; silent frames remain zero.

    Args:
        notes: sorted list of Note (ascending onset). May be empty.
        frame_seconds: frame hop and window length in seconds.

    Returns:
        np.ndarray of shape (12, T) where T = ceil(duration / frame_seconds).
        Returns shape (12, 0) for empty notes.
    """
    if not notes:
        return np.zeros((12, 0), dtype=np.float64)

    t_start = notes[0].onset
    t_end = max(n.offset for n in notes)
    duration = t_end - t_start
    n_frames = max(1, int(np.ceil(duration / frame_seconds)))

    cs = np.zeros((12, n_frames), dtype=np.float64)
    for note in notes:
        pc = note.pitch % 12
        f_start = int((note.onset - t_start) / frame_seconds)
        f_end = int(np.ceil((note.offset - t_start) / frame_seconds))
        f_start = max(0, f_start)
        f_end = min(n_frames, f_end)
        cs[pc, f_start:f_end] += float(note.velocity)

    # L2-normalise each frame independently
    norms = np.linalg.norm(cs, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    cs /= norms
    return cs
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_note_chroma.py -p no:cov -x
```
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/note_chroma.py model/tests/piece_id_eval/test_note_chroma.py && git commit -m "feat(piece-id): add note-derived chroma features (note_chroma.py)"
```

---

## Task 5: corruption.py — Synthetic note degradation

**Group:** 1 (parallel with Tasks 3, 4, 6, 7; depends on Group 0)

**Behavior being verified:** `corrupt_notes` at `deletion_rate=0.5` removes ~50% of notes; at `insertion_rate=0.5` adds new notes; at `jitter_seconds=0.1` shifts onsets/offsets; results are deterministic for same seed.

**Interface under test:** `corrupt_notes(notes, deletion_rate, insertion_rate, jitter_seconds, seed) -> list[Note]`

**Files:**
- Create: `model/src/piece_id_eval/corruption.py`
- Create: `model/tests/piece_id_eval/test_corruption.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_corruption.py
"""Verify corrupt_notes statistics at known rates/seeds."""
from __future__ import annotations

from piece_id_eval.corruption import corrupt_notes
from piece_id_eval.notes import Note


def _make_notes(n: int = 100) -> list[Note]:
    return [Note(onset=i * 0.5, offset=i * 0.5 + 0.3, pitch=60 + (i % 24), velocity=80) for i in range(n)]


def test_deletion_rate_removes_roughly_half() -> None:
    notes = _make_notes(200)
    corrupted = corrupt_notes(notes, deletion_rate=0.5, insertion_rate=0.0, jitter_seconds=0.0, seed=0)
    ratio = len(corrupted) / len(notes)
    assert 0.35 < ratio < 0.65, f"expected ~50% remaining, got {ratio:.2f}"


def test_insertion_rate_adds_notes() -> None:
    notes = _make_notes(100)
    corrupted = corrupt_notes(notes, deletion_rate=0.0, insertion_rate=0.5, jitter_seconds=0.0, seed=0)
    assert len(corrupted) > len(notes), "insertion_rate=0.5 should add notes"


def test_jitter_shifts_onsets() -> None:
    notes = _make_notes(50)
    corrupted = corrupt_notes(notes, deletion_rate=0.0, insertion_rate=0.0, jitter_seconds=0.05, seed=0)
    assert len(corrupted) == len(notes)
    original_onsets = [n.onset for n in notes]
    corrupted_onsets = [n.onset for n in corrupted]
    # At least some onsets should have changed
    changed = sum(abs(a - b) > 1e-9 for a, b in zip(original_onsets, corrupted_onsets))
    assert changed > 0, "jitter_seconds>0 should shift some onsets"


def test_no_corruption_is_identity() -> None:
    notes = _make_notes(20)
    corrupted = corrupt_notes(notes, deletion_rate=0.0, insertion_rate=0.0, jitter_seconds=0.0, seed=0)
    assert corrupted == notes


def test_deterministic_for_same_seed() -> None:
    notes = _make_notes(100)
    a = corrupt_notes(notes, deletion_rate=0.3, insertion_rate=0.2, jitter_seconds=0.05, seed=42)
    b = corrupt_notes(notes, deletion_rate=0.3, insertion_rate=0.2, jitter_seconds=0.05, seed=42)
    assert a == b


def test_corrupted_notes_sorted_by_onset() -> None:
    notes = _make_notes(50)
    corrupted = corrupt_notes(notes, deletion_rate=0.2, insertion_rate=0.3, jitter_seconds=0.05, seed=1)
    onsets = [n.onset for n in corrupted]
    assert onsets == sorted(onsets), "output must be sorted by onset"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_corruption.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.corruption'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/corruption.py
"""Synthetic note degradation for corruption-ablation experiments.

Three independent corruption modes applied in order:
1. Deletion: each note dropped independently with probability deletion_rate.
2. Jitter: onset/offset shifted by Uniform(-jitter_seconds, +jitter_seconds).
3. Insertion: for each surviving note, with probability insertion_rate, insert
   a random note nearby (pitch = original_pitch + Uniform(-12, 12), clamped
   to MIDI range 21-108; onset within ±1.0s of the original).

Output is sorted ascending by onset.
"""
from __future__ import annotations

import random

from piece_id_eval.notes import Note

_MIDI_MIN = 21
_MIDI_MAX = 108


def corrupt_notes(
    notes: list[Note],
    deletion_rate: float,
    insertion_rate: float,
    jitter_seconds: float,
    seed: int,
) -> list[Note]:
    """Apply deletion, jitter, and insertion corruption to notes.

    Args:
        notes: input note list (not modified in place).
        deletion_rate: probability in [0, 1] each note is dropped.
        insertion_rate: probability in [0, 1] a spurious note is inserted
            adjacent to each surviving note.
        jitter_seconds: max absolute onset/offset shift in seconds (Uniform).
        seed: RNG seed for full determinism.

    Returns:
        Corrupted list of Note, sorted ascending by onset.
    """
    rng = random.Random(seed)
    result: list[Note] = []

    for n in notes:
        # Deletion
        if deletion_rate > 0.0 and rng.random() < deletion_rate:
            continue

        # Jitter
        if jitter_seconds > 0.0:
            shift = rng.uniform(-jitter_seconds, jitter_seconds)
            onset = max(0.0, n.onset + shift)
            offset = max(onset + 0.01, n.offset + shift)
            n = Note(onset=onset, offset=offset, pitch=n.pitch, velocity=n.velocity)

        result.append(n)

        # Insertion
        if insertion_rate > 0.0 and rng.random() < insertion_rate:
            pitch_shift = rng.randint(-12, 12)
            new_pitch = max(_MIDI_MIN, min(_MIDI_MAX, n.pitch + pitch_shift))
            onset_shift = rng.uniform(-1.0, 1.0)
            new_onset = max(0.0, n.onset + onset_shift)
            new_offset = new_onset + rng.uniform(0.05, 0.3)
            new_velocity = max(1, min(127, n.velocity + rng.randint(-20, 20)))
            result.append(Note(onset=new_onset, offset=new_offset, pitch=new_pitch, velocity=new_velocity))

    result.sort(key=lambda n: n.onset)
    return result
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_corruption.py -p no:cov -x
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/corruption.py model/tests/piece_id_eval/test_corruption.py && git commit -m "feat(piece-id): add synthetic note degradation (corruption.py)"
```

---

## Task 6: open_set.py — Leave-one-out FA/TA curve

**Group:** 1 (parallel with Tasks 3, 4, 5, 7; depends on Group 0)

**Behavior being verified:** `operating_points` sweeps thresholds and returns FA/TA pairs; `best_point` finds the point with FA <= max_fa and TA >= min_ta, or returns None if none qualifies.

**Interface under test:** `operating_points(in_catalog_results, loo_results, thresholds) -> list[OperatingPoint]`, `best_point(points, max_fa, min_ta) -> OperatingPoint | None`

**Files:**
- Create: `model/src/piece_id_eval/open_set.py`
- Create: `model/tests/piece_id_eval/test_open_set.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_open_set.py
"""Verify operating_points and best_point through their public interfaces."""
from __future__ import annotations

import numpy as np
import pytest

from piece_id_eval.open_set import OperatingPoint, best_point, operating_points


def _perfect_results() -> tuple[list[float], list[float]]:
    """In-catalog queries score 1.0; out-of-catalog score 0.0."""
    in_scores = [1.0] * 10
    loo_scores = [0.0] * 10
    return in_scores, loo_scores


def _random_results(seed: int = 0) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    in_scores = rng.uniform(0.5, 1.0, 10).tolist()
    loo_scores = rng.uniform(0.0, 0.5, 10).tolist()
    return in_scores, loo_scores


def test_operating_points_returns_one_per_threshold() -> None:
    in_s, loo_s = _perfect_results()
    thresholds = [0.0, 0.5, 1.0]
    pts = operating_points(in_s, loo_s, thresholds)
    assert len(pts) == 3


def test_operating_points_perfect_separation() -> None:
    in_s, loo_s = _perfect_results()
    thresholds = [0.5]
    pts = operating_points(in_s, loo_s, thresholds)
    pt = pts[0]
    assert pt.fa == pytest.approx(0.0)
    assert pt.ta == pytest.approx(1.0)


def test_operating_points_all_accepted_at_zero_threshold() -> None:
    in_s, loo_s = _random_results()
    pts = operating_points(in_s, loo_s, [0.0])
    pt = pts[0]
    assert pt.fa == pytest.approx(1.0)
    assert pt.ta == pytest.approx(1.0)


def test_best_point_returns_none_when_no_point_qualifies() -> None:
    in_s = [0.3] * 5  # all low scores
    loo_s = [0.3] * 5
    pts = operating_points(in_s, loo_s, [0.5])
    # At threshold 0.5, no in-catalog accepted: TA=0 < 0.6
    result = best_point(pts, max_fa=0.05, min_ta=0.60)
    assert result is None


def test_best_point_finds_qualifying_point() -> None:
    in_s, loo_s = _perfect_results()
    thresholds = [0.0, 0.5, 0.9, 1.1]
    pts = operating_points(in_s, loo_s, thresholds)
    result = best_point(pts, max_fa=0.05, min_ta=0.60)
    assert result is not None
    assert result.fa <= 0.05
    assert result.ta >= 0.60


def test_operating_point_is_namedtuple() -> None:
    in_s, loo_s = _perfect_results()
    pts = operating_points(in_s, loo_s, [0.5])
    pt = pts[0]
    assert isinstance(pt, OperatingPoint)
    assert hasattr(pt, "threshold")
    assert hasattr(pt, "fa")
    assert hasattr(pt, "ta")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_open_set.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.open_set'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/open_set.py
"""Leave-one-out false-accept / true-accept curve for open-set evaluation.

In-catalog queries: the true piece IS in the catalog (standard recall).
LOO (leave-one-out) queries: the true piece has been removed from the catalog;
  any accept is a false accept (the only correct answer is "unknown").

A query is "accepted" if its top-1 score >= threshold.
"""
from __future__ import annotations

from typing import NamedTuple


class OperatingPoint(NamedTuple):
    """One point on the FA/TA curve at a given threshold."""
    threshold: float
    fa: float  # false-accept rate: fraction of LOO queries accepted
    ta: float  # true-accept rate: fraction of in-catalog queries accepted


def operating_points(
    in_catalog_scores: list[float],
    loo_scores: list[float],
    thresholds: list[float],
) -> list[OperatingPoint]:
    """Sweep thresholds and return one OperatingPoint per threshold.

    Args:
        in_catalog_scores: top-1 match score for each in-catalog query.
        loo_scores: top-1 match score for each LOO (out-of-catalog) query.
        thresholds: list of score thresholds to sweep (ascending order not required).

    Returns:
        List of OperatingPoint in the same order as thresholds.

    Raises:
        ValueError: if either score list is empty.
    """
    if not in_catalog_scores:
        raise ValueError("in_catalog_scores is empty")
    if not loo_scores:
        raise ValueError("loo_scores is empty")

    points: list[OperatingPoint] = []
    for t in thresholds:
        ta = sum(1 for s in in_catalog_scores if s >= t) / len(in_catalog_scores)
        fa = sum(1 for s in loo_scores if s >= t) / len(loo_scores)
        points.append(OperatingPoint(threshold=float(t), fa=fa, ta=ta))
    return points


def best_point(
    points: list[OperatingPoint],
    max_fa: float,
    min_ta: float,
) -> OperatingPoint | None:
    """Return the OperatingPoint with lowest FA that also satisfies TA >= min_ta,
    among all points with FA <= max_fa. Returns None if no such point exists.

    Args:
        points: list of OperatingPoint from operating_points().
        max_fa: maximum allowable false-accept rate.
        min_ta: minimum required true-accept rate.

    Returns:
        Best OperatingPoint, or None if no point qualifies.
    """
    qualifying = [p for p in points if p.fa <= max_fa and p.ta >= min_ta]
    if not qualifying:
        return None
    return min(qualifying, key=lambda p: p.fa)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_open_set.py -p no:cov -x
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/open_set.py model/tests/piece_id_eval/test_open_set.py && git commit -m "feat(piece-id): add leave-one-out FA/TA curve (open_set.py)"
```

---

## Task 7: decision.py — Re-threshold gate

**Group:** 1 (parallel with Tasks 3, 4, 5, 6; depends on Group 0)

**Behavior being verified:** `decide` returns KILL when `dtw_recall10 < 0.70`; PROCEED when `best_indexable_recall10 >= 0.85` AND `open_set_ok_flag`; TUNE otherwise. New open-set gate is FA <= 0.05 @ TA >= 0.60 (down from FA <= 0.10 @ TA >= 0.75).

**Interface under test:** `decide(dtw_recall10, best_indexable_recall10, open_set_ok_flag) -> str` (signature unchanged, only constants change)

**Files:**
- Modify: `model/src/piece_id_eval/decision.py`
- Modify: `model/tests/piece_id_eval/test_decision.py`

---

- [ ] **Step 1: Write the failing test**

Update `test_decision.py` to assert the new open-set gate criterion (FA <= 0.05 @ TA >= 0.60). The existing KILL/PROCEED/TUNE boundary tests for `dtw_recall10` and `best_indexable_recall10` already pass; add tests that exercise the new docstring and a truth-table comment documenting the new threshold.

```python
# model/tests/piece_id_eval/test_decision.py
"""Verify pre-registered KILL/TUNE/PROCEED rule.

Gate thresholds (updated for note-based bakeoff):
  KILL    if dtw_recall10 < 0.70
  PROCEED if best_indexable_recall10 >= 0.85 AND open_set_ok (FA<=0.05 @ TA>=0.60)
  TUNE    otherwise
"""
from piece_id_eval.decision import decide


def test_kill_when_dtw_below_threshold() -> None:
    assert decide(dtw_recall10=0.60, best_indexable_recall10=0.95, open_set_ok_flag=True) == "KILL"


def test_kill_at_exact_boundary() -> None:
    assert decide(dtw_recall10=0.699, best_indexable_recall10=0.90, open_set_ok_flag=True) == "KILL"


def test_proceed_when_all_criteria_met() -> None:
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_ok_but_indexable_low() -> None:
    assert decide(dtw_recall10=0.75, best_indexable_recall10=0.80, open_set_ok_flag=True) == "TUNE"


def test_tune_when_dtw_ok_indexable_ok_but_open_set_fails() -> None:
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.90, open_set_ok_flag=False) == "TUNE"


def test_proceed_at_exact_indexable_boundary() -> None:
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_exactly_at_boundary() -> None:
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.60, open_set_ok_flag=True) == "TUNE"


def test_docstring_reflects_new_open_set_thresholds() -> None:
    """The module docstring must mention the current FA/TA thresholds."""
    import piece_id_eval.decision as _mod
    doc = _mod.__doc__ or ""
    assert "0.05" in doc, "docstring missing FA threshold 0.05"
    assert "0.60" in doc, "docstring missing TA threshold 0.60"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_decision.py -p no:cov -x
```
Expected: FAIL — `AssertionError` on `test_docstring_reflects_new_open_set_thresholds` (current docstring says 0.10 / 0.75)

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/decision.py
"""Pre-registered KILL / TUNE / PROCEED gate for the piece-ID feasibility harness.

Rule (pre-registered before any real data is collected):
  KILL    if DtwCeilingMatcher recall@10 < 0.70
  PROCEED if some indexable matcher recall@10 >= 0.85
           AND open_set_ok (FA <= 0.05 at TA >= 0.60)
  TUNE    otherwise
"""
from __future__ import annotations

_DTW_KILL_THRESHOLD = 0.70
_INDEXABLE_PROCEED_THRESHOLD = 0.85


def decide(
    dtw_recall10: float,
    best_indexable_recall10: float,
    open_set_ok_flag: bool,
) -> str:
    """Return 'KILL', 'PROCEED', or 'TUNE' based on pre-registered thresholds.

    Args:
        dtw_recall10: recall@10 of DtwCeilingMatcher (the discrimination ceiling).
        best_indexable_recall10: max recall@10 across note-based indexable matchers
            (NoteChromaMatcher, LandmarkMatcher, ChromaSeqDtwMatcher).
        open_set_ok_flag: True iff an open-set threshold exists with FA<=0.05, TA>=0.60.

    Returns:
        'KILL' | 'PROCEED' | 'TUNE'
    """
    if dtw_recall10 < _DTW_KILL_THRESHOLD:
        return "KILL"
    if best_indexable_recall10 >= _INDEXABLE_PROCEED_THRESHOLD and open_set_ok_flag:
        return "PROCEED"
    return "TUNE"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_decision.py -p no:cov -x
```
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/decision.py model/tests/piece_id_eval/test_decision.py && git commit -m "feat(piece-id): re-threshold decision gate to FA<=0.05/TA>=0.60 (decision.py)"
```

---

## Task 8: matchers/base.py + note_chroma_matcher.py (C1)

**Group:** 2a (sequential after Group 1; 2b tasks depend on this)

**Behavior being verified:** The note-based `Matcher` Protocol accepts `list[Note]` not `np.ndarray`; `NoteChromaMatcher.rank` on a 3-piece synthetic catalog returns the queried piece first (recall@1 == 1.0).

**Interface under test:** `Matcher` Protocol, `NoteChromaMatcher(catalog: dict[str, list[Note]])`, `NoteChromaMatcher.rank(query: list[Note]) -> list[Ranked]`

**Files:**
- Modify: `model/src/piece_id_eval/matchers/base.py`
- Create: `model/src/piece_id_eval/matchers/note_chroma_matcher.py`
- Modify: `model/tests/piece_id_eval/test_matchers.py`

---

- [ ] **Step 1: Write the failing test**

Replace the contents of `test_matchers.py` with the note-based version (the old chroma-based tests reference `ChordNgramMatcher`, `TwoDFTMatcher`, and `build_score_chroma` which will be deleted).

```python
# model/tests/piece_id_eval/test_matchers.py
"""Verify note-based matchers implement the Matcher protocol and find the right
piece on a small synthetic catalog. Each matcher's first test is score
self-query recall@1 == 1.0.
"""
from __future__ import annotations

import pytest

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher
from piece_id_eval.notes import Note


def _catalog_3piece() -> dict[str, list[Note]]:
    """3-piece synthetic catalog with distinct pitch classes."""
    pitches = [60, 64, 67]  # C, E, G
    catalog: dict[str, list[Note]] = {}
    for i, p in enumerate(pitches):
        catalog[f"piece_{i}"] = [
            Note(onset=j * 0.5, offset=j * 0.5 + 0.4, pitch=p, velocity=80)
            for j in range(20)
        ]
    return catalog


def test_note_chroma_matcher_protocol_compliance() -> None:
    catalog = _catalog_3piece()
    m = NoteChromaMatcher(catalog)
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_note_chroma_matcher_self_query_recall_at_1() -> None:
    catalog = _catalog_3piece()
    m = NoteChromaMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert len(ranked) == 3
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id} not first: {ranked[0].piece_id}"
        )


def test_note_chroma_matcher_returns_ranked_list() -> None:
    catalog = _catalog_3piece()
    m = NoteChromaMatcher(catalog)
    ranked = m.rank(catalog["piece_0"])
    assert all(isinstance(r, Ranked) for r in ranked)
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True), f"not descending: {scores}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_matchers.py -p no:cov -x
```
Expected: FAIL — `ImportError` (NoteChromaMatcher does not exist) or `ModuleNotFoundError`

- [ ] **Step 3: Implement the minimum to make the test pass**

First update `matchers/base.py`:

```python
# model/src/piece_id_eval/matchers/base.py
"""Matcher protocol and Ranked result type (note-based)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable, NamedTuple

from piece_id_eval.notes import Note


class Ranked(NamedTuple):
    """A single (piece_id, score) result. Higher score = better match."""
    piece_id: str
    score: float


@runtime_checkable
class Matcher(Protocol):
    """Protocol for note-based piece-ID matchers."""

    @property
    def name(self) -> str:
        """Short identifier for this matcher (used in report tables)."""
        ...

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces against a query note list.

        Returns list of Ranked sorted descending by score (highest first).
        """
        ...
```

Then create `matchers/note_chroma_matcher.py`:

```python
# model/src/piece_id_eval/matchers/note_chroma_matcher.py
"""C1: Note-chroma cosine matcher.

Computes a single 12-bin key-dependent chroma vector per piece and query,
then ranks by cosine similarity. Fast O(N_catalog) search.

Hides: chroma_vector computation, cosine similarity, catalog indexing.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.note_chroma import chroma_vector
from piece_id_eval.notes import Note


class NoteChromaMatcher:
    """Cosine similarity over key-dependent note-chroma vectors (C1)."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        """Pre-compute chroma vectors for all catalog pieces.

        Args:
            catalog: {piece_id: list[Note]} for all catalog entries.
        """
        self._index: dict[str, np.ndarray] = {
            pid: chroma_vector(notes)
            for pid, notes in catalog.items()
            if notes
        }

    @property
    def name(self) -> str:
        return "note_chroma_cosine"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by cosine similarity to query chroma vector.

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by cosine similarity.

        Raises:
            ValueError: if query is empty.
        """
        if not query:
            raise ValueError("query is empty")
        q_vec = chroma_vector(query)
        results: list[Ranked] = []
        for piece_id, ref_vec in self._index.items():
            similarity = float(np.dot(q_vec, ref_vec))
            results.append(Ranked(piece_id=piece_id, score=similarity))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_matchers.py -p no:cov -x
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/matchers/base.py model/src/piece_id_eval/matchers/note_chroma_matcher.py model/tests/piece_id_eval/test_matchers.py && git commit -m "feat(piece-id): note-based Matcher protocol + C1 NoteChromaMatcher"
```

---

## Task 9: matchers/landmark.py (C2)

**Group:** 2b (parallel with Tasks 10, 11; depends on Task 8)

**Behavior being verified:** `LandmarkMatcher.rank` on a 3-piece synthetic catalog (each with distinct pitch-class sequence) returns the queried piece first (recall@1 == 1.0).

**Interface under test:** `LandmarkMatcher(catalog: dict[str, list[Note]])`, `LandmarkMatcher.rank(query: list[Note]) -> list[Ranked]`

**Files:**
- Create: `model/src/piece_id_eval/matchers/landmark.py`
- Create: `model/tests/piece_id_eval/test_landmark.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_landmark.py
"""Verify LandmarkMatcher (C2) through its public interface."""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.landmark import LandmarkMatcher
from piece_id_eval.notes import Note


def _make_piece(root_pitch: int, n: int = 30) -> list[Note]:
    """Piece with a repeating melodic pattern starting at root_pitch."""
    pattern = [0, 2, 4, 5, 7, 9, 11]
    return [
        Note(onset=i * 0.3, offset=i * 0.3 + 0.25, pitch=root_pitch + pattern[i % 7], velocity=80)
        for i in range(n)
    ]


def _catalog() -> dict[str, list[Note]]:
    return {
        "piece_c": _make_piece(60),   # C major scale pattern
        "piece_f": _make_piece(65),   # F major scale pattern
        "piece_g": _make_piece(67),   # G major scale pattern
    }


def test_landmark_matcher_protocol_compliance() -> None:
    m = LandmarkMatcher(_catalog())
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_landmark_self_query_recall_at_1() -> None:
    catalog = _catalog()
    m = LandmarkMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id}: expected first, got {ranked[0].piece_id}"
        )


def test_landmark_rank_returns_all_pieces() -> None:
    catalog = _catalog()
    m = LandmarkMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    assert len(ranked) == 3
    ids = {r.piece_id for r in ranked}
    assert ids == set(catalog.keys())


def test_landmark_scores_descending() -> None:
    catalog = _catalog()
    m = LandmarkMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_landmark.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.matchers.landmark'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/matchers/landmark.py
"""C2: Ordinal-landmark hash matcher.

Token: (pc_anchor, interval, ordinal_gap) where:
  - pc_anchor = anchor note pitch % 12 (absolute, key-dependent)
  - interval  = (target.pitch - anchor.pitch) clamped to [-12, 12]
  - ordinal_gap = target event index - anchor event index, in 1..MAX_GAP

The token is ordinal (event-index) not temporal: a student at half speed
produces the same tokens as at full speed. Inverted index maps token ->
list[piece_id]; rank by total hit count.

K=5 target notes per anchor; MAX_GAP=5 (per spec Open Questions defaults).
"""
from __future__ import annotations

from collections import defaultdict

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.notes import Note

_K = 5          # number of target notes per anchor
_MAX_GAP = 5    # maximum ordinal gap between anchor and target

Token = tuple[int, int, int]  # (pc_anchor, interval, ordinal_gap)


def _build_tokens(notes: list[Note]) -> list[Token]:
    """Generate all (pc_anchor, interval, ordinal_gap) tokens from a note list."""
    tokens: list[Token] = []
    for i, anchor in enumerate(notes):
        pc_anchor = anchor.pitch % 12
        for gap in range(1, _MAX_GAP + 1):
            j = i + gap
            if j >= len(notes):
                break
            target = notes[j]
            interval = max(-12, min(12, target.pitch - anchor.pitch))
            tokens.append((pc_anchor, interval, gap))
    return tokens


def _build_index(catalog: dict[str, list[Note]]) -> dict[Token, list[str]]:
    """Build inverted index: token -> [piece_id, ...]."""
    index: dict[Token, list[str]] = defaultdict(list)
    for piece_id, notes in catalog.items():
        seen: set[Token] = set()
        for token in _build_tokens(notes):
            if token not in seen:
                index[token].append(piece_id)
                seen.add(token)
    return dict(index)


class LandmarkMatcher:
    """Inverted landmark-token hit-count matcher (C2)."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        self._piece_ids = list(catalog.keys())
        self._index = _build_index(catalog)

    @property
    def name(self) -> str:
        return "landmark"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by landmark token hit count.

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by hit count (score).
        """
        hits: dict[str, int] = {pid: 0 for pid in self._piece_ids}
        for token in _build_tokens(query):
            for piece_id in self._index.get(token, []):
                if piece_id in hits:
                    hits[piece_id] += 1
        results = [Ranked(piece_id=pid, score=float(count)) for pid, count in hits.items()]
        results.sort(key=lambda r: r.score, reverse=True)
        return results
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_landmark.py -p no:cov -x
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/matchers/landmark.py model/tests/piece_id_eval/test_landmark.py && git commit -m "feat(piece-id): add C2 ordinal-landmark hash matcher (landmark.py)"
```

---

## Task 10: matchers/dtw_ceiling.py (C3 note-based)

**Group:** 2b (parallel with Tasks 9, 11; depends on Task 8)

**Behavior being verified:** The rebuilt `DtwCeilingMatcher` accepts `list[Note]` (not `np.ndarray`), computes onset-ordered pitch DTW normalized by query length, and ranks the correct piece first on a synthetic catalog (recall@1 == 1.0).

**Interface under test:** `DtwCeilingMatcher(catalog: dict[str, list[Note]])`, `DtwCeilingMatcher.rank(query: list[Note]) -> list[Ranked]`

**Files:**
- Modify: `model/src/piece_id_eval/matchers/dtw_ceiling.py`
- Create: `model/tests/piece_id_eval/test_dtw_ceiling.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_dtw_ceiling.py
"""Verify DtwCeilingMatcher (C3 note-based) through its public interface."""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.dtw_ceiling import DtwCeilingMatcher
from piece_id_eval.notes import Note


def _make_piece(pitches: list[int], onset_step: float = 0.5) -> list[Note]:
    return [
        Note(onset=i * onset_step, offset=i * onset_step + 0.3, pitch=p, velocity=80)
        for i, p in enumerate(pitches)
    ]


def _catalog() -> dict[str, list[Note]]:
    return {
        "ascending": _make_piece(list(range(60, 80))),
        "descending": _make_piece(list(range(79, 59, -1))),
        "constant": _make_piece([60] * 20),
    }


def test_dtw_ceiling_protocol_compliance() -> None:
    m = DtwCeilingMatcher(_catalog())
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_dtw_ceiling_self_query_recall_at_1() -> None:
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id}: expected first, got {ranked[0].piece_id}"
        )


def test_dtw_ceiling_ranks_all_pieces() -> None:
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    ranked = m.rank(catalog["ascending"])
    assert len(ranked) == 3
    ids = {r.piece_id for r in ranked}
    assert ids == set(catalog.keys())


def test_dtw_ceiling_scores_descending() -> None:
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    ranked = m.rank(catalog["ascending"])
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_dtw_ceiling_subsequence_query_ranks_correct_piece() -> None:
    """A short window (first 8 notes) of ascending should still match ascending."""
    catalog = _catalog()
    m = DtwCeilingMatcher(catalog)
    query = catalog["ascending"][:8]
    ranked = m.rank(query)
    assert ranked[0].piece_id == "ascending"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_dtw_ceiling.py -p no:cov -x
```
Expected: FAIL — `TypeError: rank() got unexpected argument list[Note]` or `AttributeError` (old signature takes `np.ndarray`)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the entire body of `dtw_ceiling.py`:

```python
# model/src/piece_id_eval/matchers/dtw_ceiling.py
"""C3: Subsequence onset-ordered pitch DTW ceiling matcher (note-based).

For each catalog piece, extracts the pitch sequence (sorted by onset) and
runs a subsequence DTW of the query pitch sequence against it. DTW cost is
normalized by query length and negated to produce a score (lower cost = higher
score).

This is the discrimination ceiling: if note-to-note DTW cannot separate
pieces, no indexable method will.

Hides: pitch extraction, subsequence DTW, query-length normalization.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.notes import Note


class DtwCeilingMatcher:
    """Subsequence onset-ordered pitch DTW over note sequences (C3)."""

    def __init__(self, catalog: dict[str, list[Note]]) -> None:
        """Pre-extract pitch sequences for all catalog pieces.

        Args:
            catalog: {piece_id: list[Note]} for all catalog entries.
        """
        self._pitches: dict[str, np.ndarray] = {
            pid: np.array([n.pitch for n in sorted(notes, key=lambda n: n.onset)], dtype=np.float32)
            for pid, notes in catalog.items()
            if notes
        }

    @property
    def name(self) -> str:
        return "dtw_ceiling"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by subsequence pitch DTW cost (lower = better).

        Score = -normalized_cost (higher = better match).

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by score.

        Raises:
            ValueError: if query is empty.
        """
        if not query:
            raise ValueError("query is empty")
        q_pitches = np.array(
            [n.pitch for n in sorted(query, key=lambda n: n.onset)], dtype=np.float32
        )
        results: list[Ranked] = []
        for piece_id, ref_pitches in self._pitches.items():
            cost = self._subseq_dtw_cost(q_pitches, ref_pitches)
            results.append(Ranked(piece_id=piece_id, score=-cost))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _subseq_dtw_cost(self, query: np.ndarray, ref: np.ndarray) -> float:
        """Subsequence DTW: slide query over ref; return min normalized path cost.

        If query is longer than ref, falls back to full DTW.
        Cost is normalized by query length.
        """
        Q = len(query)
        R = len(ref)
        if Q == 0:
            return 0.0
        if Q > R:
            return self._full_dtw(query, ref) / Q

        best = float("inf")
        for start in range(R - Q + 1):
            seg = ref[start : start + Q]
            cost = float(np.sum(np.abs(query - seg)))
            if cost < best:
                best = cost
        return best / Q

    def _full_dtw(self, query: np.ndarray, ref: np.ndarray) -> float:
        Q = len(query)
        R = len(ref)
        dp = np.full((Q + 1, R + 1), float("inf"))
        dp[0, 0] = 0.0
        for i in range(1, Q + 1):
            for j in range(1, R + 1):
                d = float(abs(query[i - 1] - ref[j - 1]))
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[Q, R])
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_dtw_ceiling.py -p no:cov -x
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/matchers/dtw_ceiling.py model/tests/piece_id_eval/test_dtw_ceiling.py && git commit -m "feat(piece-id): rebuild C3 DtwCeilingMatcher for note input (dtw_ceiling.py)"
```

---

## Task 11: matchers/chroma_seq_dtw.py (C4)

**Group:** 2b (parallel with Tasks 9, 10; depends on Task 8)

**Behavior being verified:** `ChromaSeqDtwMatcher.rank` on a 3-piece synthetic catalog returns the queried piece first (recall@1 == 1.0).

**Interface under test:** `ChromaSeqDtwMatcher(catalog: dict[str, list[Note]], frame_seconds=0.5)`, `ChromaSeqDtwMatcher.rank(query: list[Note]) -> list[Ranked]`

**Files:**
- Create: `model/src/piece_id_eval/matchers/chroma_seq_dtw.py`
- Create: `model/tests/piece_id_eval/test_chroma_seq_dtw.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_chroma_seq_dtw.py
"""Verify ChromaSeqDtwMatcher (C4) through its public interface."""
from __future__ import annotations

from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.matchers.chroma_seq_dtw import ChromaSeqDtwMatcher
from piece_id_eval.notes import Note


def _make_piece(pitch: int, n: int = 40) -> list[Note]:
    """All notes on a single pitch; distinct pitch -> distinct chroma."""
    return [Note(onset=i * 0.25, offset=i * 0.25 + 0.2, pitch=pitch, velocity=80) for i in range(n)]


def _catalog() -> dict[str, list[Note]]:
    return {
        "piece_c": _make_piece(60),   # pc=0
        "piece_e": _make_piece(64),   # pc=4
        "piece_g": _make_piece(67),   # pc=7
    }


def test_chroma_seq_dtw_protocol_compliance() -> None:
    m = ChromaSeqDtwMatcher(_catalog())
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_chroma_seq_dtw_self_query_recall_at_1() -> None:
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    for piece_id, notes in catalog.items():
        ranked = m.rank(notes)
        assert ranked[0].piece_id == piece_id, (
            f"self-query for {piece_id}: expected first, got {ranked[0].piece_id}"
        )


def test_chroma_seq_dtw_ranks_all_pieces() -> None:
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    assert len(ranked) == 3
    ids = {r.piece_id for r in ranked}
    assert ids == set(catalog.keys())


def test_chroma_seq_dtw_scores_descending() -> None:
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    ranked = m.rank(catalog["piece_c"])
    scores = [r.score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_chroma_seq_dtw_subsequence_query() -> None:
    """A 10-note window from piece_c should still match piece_c first."""
    catalog = _catalog()
    m = ChromaSeqDtwMatcher(catalog)
    query = catalog["piece_c"][:10]
    ranked = m.rank(query)
    assert ranked[0].piece_id == "piece_c"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_chroma_seq_dtw.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.matchers.chroma_seq_dtw'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/piece_id_eval/matchers/chroma_seq_dtw.py
"""C4: Subsequence DTW over note-derived chroma sequences.

Computes chroma_sequence (12, T) for both query and each catalog piece
at frame_seconds=0.5, then runs the same subsequence sliding-window DTW
as C3 but on 12-dim chroma columns instead of scalar pitch.
Cost normalized by query frame count; negated to score.

Hides: chroma_sequence computation, column-wise Euclidean DTW, normalization.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked
from piece_id_eval.note_chroma import chroma_sequence
from piece_id_eval.notes import Note

_DEFAULT_FRAME_SECONDS = 0.5


class ChromaSeqDtwMatcher:
    """Subsequence chroma-sequence DTW matcher (C4)."""

    def __init__(
        self,
        catalog: dict[str, list[Note]],
        frame_seconds: float = _DEFAULT_FRAME_SECONDS,
    ) -> None:
        self._frame_seconds = frame_seconds
        self._catalog_seq: dict[str, np.ndarray] = {
            pid: chroma_sequence(notes, frame_seconds)
            for pid, notes in catalog.items()
            if notes
        }

    @property
    def name(self) -> str:
        return "chroma_seq_dtw"

    def rank(self, query: list[Note]) -> list[Ranked]:
        """Rank catalog pieces by subsequence chroma-sequence DTW.

        Args:
            query: list of Note representing the query window.

        Returns:
            list of Ranked sorted descending by score (-normalized cost).

        Raises:
            ValueError: if query is empty.
        """
        if not query:
            raise ValueError("query is empty")
        q_seq = chroma_sequence(query, self._frame_seconds)  # (12, Q)
        Q = q_seq.shape[1]
        results: list[Ranked] = []
        for piece_id, ref_seq in self._catalog_seq.items():
            cost = self._subseq_dtw_cost(q_seq, ref_seq, Q)
            results.append(Ranked(piece_id=piece_id, score=-cost))
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _subseq_dtw_cost(
        self, query: np.ndarray, ref: np.ndarray, Q: int
    ) -> float:
        """Subsequence DTW of query (12, Q) against ref (12, R).

        Slides a window of Q frames over ref; returns minimum total
        column-wise Euclidean cost / Q.
        If Q > R, falls back to full DTW.
        """
        R = ref.shape[1]
        if Q == 0:
            return 0.0
        if Q > R:
            return self._full_dtw(query.T, ref.T) / Q

        best = float("inf")
        for start in range(R - Q + 1):
            seg = ref[:, start : start + Q]  # (12, Q)
            cost = float(np.sum(np.linalg.norm(query - seg, axis=0)))
            if cost < best:
                best = cost
        return best / Q

    def _full_dtw(self, q: np.ndarray, r: np.ndarray) -> float:
        """Full DTW on (Q, 12) and (R, 12) arrays."""
        Q, R = q.shape[0], r.shape[0]
        dp = np.full((Q + 1, R + 1), float("inf"))
        dp[0, 0] = 0.0
        for i in range(1, Q + 1):
            for j in range(1, R + 1):
                d = float(np.linalg.norm(q[i - 1] - r[j - 1]))
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[Q, R])
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_chroma_seq_dtw.py -p no:cov -x
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/matchers/chroma_seq_dtw.py model/tests/piece_id_eval/test_chroma_seq_dtw.py && git commit -m "feat(piece-id): add C4 chroma-sequence DTW matcher (chroma_seq_dtw.py)"
```

---

## Task 12: bakeoff.py + dead-file deletions + matchers/__init__.py

**Group:** 3 (sequential, depends on Group 2)

**Behavior being verified:** `bakeoff.run` on a 2-piece synthetic catalog produces a `BakeoffReport` with populated recall tables and a verdict; `python -m piece_id_eval.bakeoff --help` exits 0 with `--no-track`; dead audio-chroma files are deleted.

**Interface under test:** `run(catalog, recordings, ...) -> BakeoffReport`; CLI `python -m piece_id_eval.bakeoff --no-track`

**Files:**
- Create: `model/src/piece_id_eval/bakeoff.py`
- Create: `model/tests/piece_id_eval/test_bakeoff.py`
- Modify: `model/src/piece_id_eval/matchers/__init__.py`
- Delete: `model/src/piece_id_eval/matchers/chord_ngram.py`
- Delete: `model/src/piece_id_eval/matchers/twodft.py`
- Delete: `model/src/piece_id_eval/query_chroma.py`
- Delete: `model/src/piece_id_eval/score_chroma.py`
- Delete: `model/src/piece_id_eval/query_set.py`
- Delete: `model/src/piece_id_eval/cli.py`
- Delete: `model/src/piece_id_eval/report.py`
- Delete: `model/tests/piece_id_eval/test_query_chroma.py`
- Delete: `model/tests/piece_id_eval/test_score_chroma.py`
- Delete: `model/tests/piece_id_eval/test_query_set.py`
- Delete: `model/tests/piece_id_eval/test_cli_smoke.py`
- Delete: `model/tests/piece_id_eval/test_report.py`

---

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_bakeoff.py
"""Integration test: bakeoff.run on a 2-piece synthetic catalog.

Verifies BakeoffReport is populated with recall tables, verdict, and per-matcher
results. CLI smoke test verifies --no-track exits 0.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from piece_id_eval.bakeoff import BakeoffReport, run
from piece_id_eval.notes import Note

REPO_ROOT = Path(__file__).resolve().parents[4]


def _make_piece(root: int, n: int = 40) -> list[Note]:
    return [Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=root + (i % 7), velocity=80) for i in range(n)]


def _synthetic_catalog() -> dict[str, list[Note]]:
    return {
        "piece_a": _make_piece(60),
        "piece_b": _make_piece(67),
    }


def _synthetic_recordings() -> dict[str, list[Note]]:
    """One recording per piece (same as catalog for self-query test)."""
    return {
        "piece_a": _make_piece(60),
        "piece_b": _make_piece(67),
    }


def test_bakeoff_run_returns_report() -> None:
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None, 10.0],
        n_starts=2,
        corruption_grid=[
            {"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0},
            {"deletion_rate": 0.3, "insertion_rate": 0.0, "jitter_seconds": 0.0},
        ],
        seed=42,
        no_track=True,
    )
    assert isinstance(report, BakeoffReport)


def test_bakeoff_report_has_recall_table() -> None:
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None],
        n_starts=1,
        corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
        seed=0,
        no_track=True,
    )
    # recall_table is a dict keyed by (matcher_name, window_label)
    assert len(report.recall_table) > 0
    for key, val in report.recall_table.items():
        matcher_name, window_label = key
        assert isinstance(matcher_name, str)
        assert "recall@1" in val or "recall@10" in val


def test_bakeoff_report_has_verdict() -> None:
    catalog = _synthetic_catalog()
    recordings = _synthetic_recordings()
    report = run(
        catalog=catalog,
        recordings=recordings,
        window_lengths=[None],
        n_starts=1,
        corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
        seed=0,
        no_track=True,
    )
    assert report.verdict in {"KILL", "TUNE", "PROCEED"}


def test_bakeoff_cli_smoke(tmp_path: Path) -> None:
    """CLI help exits 0 without crashing."""
    result = subprocess.run(
        [sys.executable, "-m", "piece_id_eval.bakeoff", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "model"),
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"


def test_bakeoff_cli_no_track_synthetic(tmp_path: Path) -> None:
    """CLI runs with --no-track and --synthetic-only without needing real data."""
    result = subprocess.run(
        [sys.executable, "-m", "piece_id_eval.bakeoff", "--no-track", "--synthetic-only"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT / "model"),
    )
    assert result.returncode == 0, f"CLI failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "VERDICT:" in result.stdout
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run pytest tests/piece_id_eval/test_bakeoff.py -p no:cov -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval.bakeoff'`

- [ ] **Step 3: Implement the minimum to make the test pass**

First update `matchers/__init__.py`:

```python
# model/src/piece_id_eval/matchers/__init__.py
from piece_id_eval.matchers.chroma_seq_dtw import ChromaSeqDtwMatcher
from piece_id_eval.matchers.dtw_ceiling import DtwCeilingMatcher
from piece_id_eval.matchers.landmark import LandmarkMatcher
from piece_id_eval.matchers.note_chroma_matcher import NoteChromaMatcher

__all__ = [
    "DtwCeilingMatcher",
    "NoteChromaMatcher",
    "LandmarkMatcher",
    "ChromaSeqDtwMatcher",
]
```

Then create `bakeoff.py`:

```python
# model/src/piece_id_eval/bakeoff.py
"""Piece-ID bakeoff orchestrator.

Sweeps: window_lengths x matchers x corruption_grid.
For each (matcher, window_length, corruption): compute recall@{1,5,10} across recordings.
Computes leave-one-out open-set FA/TA curve.
Emits: BakeoffReport (recall_table, corruption_curves, open_set_point, verdict).
Writes sidecar JSON if --output-json is given.
Logs to Trackio unless --no-track.

CLI: python -m piece_id_eval.bakeoff [--no-track] [--synthetic-only] [--output-json PATH]
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from piece_id_eval.corruption import corrupt_notes
from piece_id_eval.decision import decide
from piece_id_eval.matchers import (
    ChromaSeqDtwMatcher,
    DtwCeilingMatcher,
    LandmarkMatcher,
    NoteChromaMatcher,
)
from piece_id_eval.matchers.base import Matcher
from piece_id_eval.metrics import Rankings, recall_at_k
from piece_id_eval.notes import Note
from piece_id_eval.open_set import best_point, operating_points
from piece_id_eval.windowing import sample_windows

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_PRACTICE_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval"
_DEFAULT_NOTES_ROOT = _MODULE_DIR.parents[2] / "data/evals/practice_eval_pseudo"
_DEFAULT_SCORES_ROOT = _MODULE_DIR.parents[2] / "data/scores"
_DEFAULT_PIECE_MAP = _MODULE_DIR.parents[2] / "data/evals/piece_id/eval_piece_map.json"

_WINDOW_LENGTHS: list[float | None] = [15.0, 30.0, 60.0, 90.0, None]
_N_STARTS = 5
_SEED = 42
_OPEN_SET_THRESHOLDS = [i / 20 for i in range(21)]  # 0.0, 0.05, ..., 1.0
_MAX_FA = 0.05
_MIN_TA = 0.60


@dataclass
class BakeoffReport:
    """Output of a bakeoff run."""
    # recall_table: {(matcher_name, window_label): {"recall@1": float, "recall@5": float, "recall@10": float}}
    recall_table: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    # corruption_curves: {(matcher_name, corruption_label): {"recall@10": float}}
    corruption_curves: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    # open_set_ok: True if a threshold exists with FA<=0.05, TA>=0.60
    open_set_ok: bool = False
    # verdict: KILL / TUNE / PROCEED
    verdict: str = "TUNE"
    # dtw_ceiling_recall10: best recall@10 for DtwCeilingMatcher (ceiling)
    dtw_ceiling_recall10: float = 0.0
    # best_indexable_recall10: max recall@10 across C1/C2/C4
    best_indexable_recall10: float = 0.0


def _window_label(window_seconds: float | None) -> str:
    return "full" if window_seconds is None else f"{int(window_seconds)}s"


def _corruption_label(grid_entry: dict[str, float]) -> str:
    return (
        f"del={grid_entry['deletion_rate']:.1f}"
        f"_ins={grid_entry['insertion_rate']:.1f}"
        f"_jit={grid_entry['jitter_seconds']:.2f}"
    )


def run(
    catalog: dict[str, list[Note]],
    recordings: dict[str, list[Note]],
    window_lengths: list[float | None] | None = None,
    n_starts: int = _N_STARTS,
    corruption_grid: list[dict[str, float]] | None = None,
    seed: int = _SEED,
    no_track: bool = False,
) -> BakeoffReport:
    """Run the full bakeoff and return a BakeoffReport.

    Args:
        catalog: {piece_id: list[Note]} for all 254 (or synthetic) catalog pieces.
        recordings: {piece_id: list[Note]} for each query recording (true piece_id as key).
        window_lengths: list of window durations in seconds; None = full piece.
            Defaults to [15, 30, 60, 90, None].
        n_starts: number of random start offsets per window length.
        corruption_grid: list of corruption parameter dicts with keys
            deletion_rate, insertion_rate, jitter_seconds.
            Defaults to a small preset grid.
        seed: RNG seed for windowing and corruption.
        no_track: if True, skip Trackio logging.

    Returns:
        BakeoffReport with recall tables, corruption curves, and verdict.
    """
    if window_lengths is None:
        window_lengths = _WINDOW_LENGTHS
    if corruption_grid is None:
        corruption_grid = [
            {"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0},
            {"deletion_rate": 0.2, "insertion_rate": 0.0, "jitter_seconds": 0.05},
            {"deletion_rate": 0.4, "insertion_rate": 0.1, "jitter_seconds": 0.1},
        ]

    matchers: list[Matcher] = [
        NoteChromaMatcher(catalog),
        LandmarkMatcher(catalog),
        DtwCeilingMatcher(catalog),
        ChromaSeqDtwMatcher(catalog),
    ]

    report = BakeoffReport()

    # Recall sweep: window x matcher (no corruption)
    clean_corruption = {"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}
    for window_seconds in window_lengths:
        wlabel = _window_label(window_seconds)
        for matcher in matchers:
            rankings: Rankings = []
            for true_id, notes in recordings.items():
                windows = sample_windows(notes, window_seconds, n_starts, seed)
                for win in windows:
                    if not win:
                        continue
                    ranked = matcher.rank(win)
                    rankings.append((true_id, [(r.piece_id, r.score) for r in ranked]))
            if not rankings:
                continue
            r1 = recall_at_k(rankings, 1)
            r5 = recall_at_k(rankings, 5)
            r10 = recall_at_k(rankings, 10)
            report.recall_table[(matcher.name, wlabel)] = {
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
            }

    # Corruption sweep: use full window; compare across matchers
    full_label = _window_label(None)
    for entry in corruption_grid:
        clabel = _corruption_label(entry)
        for matcher in matchers:
            rankings = []
            for true_id, notes in recordings.items():
                corrupted = corrupt_notes(
                    notes,
                    deletion_rate=entry["deletion_rate"],
                    insertion_rate=entry["insertion_rate"],
                    jitter_seconds=entry["jitter_seconds"],
                    seed=seed,
                )
                if not corrupted:
                    continue
                ranked = matcher.rank(corrupted)
                rankings.append((true_id, [(r.piece_id, r.score) for r in ranked]))
            if not rankings:
                continue
            report.corruption_curves[(matcher.name, clabel)] = {
                "recall@10": recall_at_k(rankings, 10)
            }

    # Compute ceiling + best indexable recall@10 (best across all window lengths)
    dtw_name = DtwCeilingMatcher(catalog).name  # "dtw_ceiling"
    indexable_names = {NoteChromaMatcher(catalog).name, LandmarkMatcher(catalog).name, ChromaSeqDtwMatcher(catalog).name}

    for (mname, wlabel), vals in report.recall_table.items():
        r10 = vals["recall@10"]
        if mname == dtw_name:
            report.dtw_ceiling_recall10 = max(report.dtw_ceiling_recall10, r10)
        if mname in indexable_names:
            report.best_indexable_recall10 = max(report.best_indexable_recall10, r10)

    # Open-set: leave-one-out (remove true piece from catalog, top-1 score)
    in_scores: list[float] = []
    loo_scores: list[float] = []
    best_matcher = matchers[2]  # DtwCeilingMatcher as open-set oracle
    for true_id, notes in recordings.items():
        # In-catalog: full catalog
        ranked = best_matcher.rank(notes)
        in_scores.append(ranked[0].score if ranked else 0.0)
        # LOO: catalog without true piece
        loo_catalog = {pid: n for pid, n in catalog.items() if pid != true_id}
        if loo_catalog:
            loo_matcher = DtwCeilingMatcher(loo_catalog)
            loo_ranked = loo_matcher.rank(notes)
            loo_scores.append(loo_ranked[0].score if loo_ranked else 0.0)

    if in_scores and loo_scores:
        pts = operating_points(in_scores, loo_scores, _OPEN_SET_THRESHOLDS)
        bp = best_point(pts, max_fa=_MAX_FA, min_ta=_MIN_TA)
        report.open_set_ok = bp is not None

    report.verdict = decide(
        dtw_recall10=report.dtw_ceiling_recall10,
        best_indexable_recall10=report.best_indexable_recall10,
        open_set_ok_flag=report.open_set_ok,
    )

    if not no_track:
        try:
            import trackio as tr
            tr.log({
                "dtw_ceiling_recall10": report.dtw_ceiling_recall10,
                "best_indexable_recall10": report.best_indexable_recall10,
                "open_set_ok": report.open_set_ok,
                "verdict": report.verdict,
            })
        except Exception:
            pass  # Trackio is optional; never fail the run on tracking errors

    return report


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Piece-ID note-based bakeoff. Emits VERDICT: KILL|TUNE|PROCEED."
    )
    parser.add_argument("--no-track", action="store_true", help="Skip Trackio logging.")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Run on a 2-piece synthetic catalog instead of real data (for CI smoke).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write BakeoffReport metrics to this JSON path.",
    )
    parser.add_argument(
        "--piece-map",
        type=Path,
        default=_DEFAULT_PIECE_MAP,
        help="JSON map of slug->piece_id.",
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=_DEFAULT_SCORES_ROOT,
        help="Directory of score JSON files.",
    )
    parser.add_argument(
        "--notes-root",
        type=Path,
        default=_DEFAULT_NOTES_ROOT,
        help="Root of practice_eval_pseudo cache directories.",
    )
    parser.add_argument(
        "--practice-root",
        type=Path,
        default=_DEFAULT_PRACTICE_ROOT,
        help="Root of practice_eval directories (for candidates.yaml).",
    )
    args = parser.parse_args()

    if args.synthetic_only:
        # 2-piece synthetic run for CI smoke
        def _p(root: int, n: int = 40) -> list[Note]:
            return [Note(onset=i * 0.5, offset=i * 0.5 + 0.4, pitch=root + (i % 7), velocity=80) for i in range(n)]

        catalog = {"piece_a": _p(60), "piece_b": _p(67)}
        recordings = {"piece_a": _p(60), "piece_b": _p(67)}
        report = run(
            catalog=catalog,
            recordings=recordings,
            window_lengths=[None, 10.0],
            n_starts=2,
            corruption_grid=[{"deletion_rate": 0.0, "insertion_rate": 0.0, "jitter_seconds": 0.0}],
            seed=42,
            no_track=True,
        )
    else:
        from piece_id_eval.notes import load_amt_notes, load_score_notes

        piece_map: dict[str, str] = json.loads(args.piece_map.read_text())

        # Load catalog from score JSONs
        catalog: dict[str, list[Note]] = {}
        for score_json in args.scores_root.glob("*.json"):
            piece_id = score_json.stem
            try:
                catalog[piece_id] = load_score_notes(score_json)
            except Exception as exc:
                print(f"[WARN] skipping {score_json.name}: {exc}", file=sys.stderr)

        # Load recordings from amt_notes.json cache
        import yaml

        recordings: dict[str, list[Note]] = {}
        for slug, piece_id in piece_map.items():
            slug_dir = args.practice_root / slug
            candidates_file = slug_dir / "candidates.yaml"
            if not candidates_file.exists():
                print(f"[SKIP] {slug}: no candidates.yaml", file=sys.stderr)
                continue
            with candidates_file.open() as f:
                candidates = yaml.safe_load(f)
            approved = [r for r in (candidates.get("recordings") or []) if r.get("approved")]
            for rec in approved:
                video_id = rec["video_id"]
                notes_path = args.notes_root / slug / video_id / "amt_notes.json"
                if not notes_path.exists():
                    print(f"[SKIP] {slug}/{video_id}: amt_notes.json missing (run transcribe first)", file=sys.stderr)
                    continue
                try:
                    notes = load_amt_notes(notes_path)
                    recordings[piece_id] = notes
                    break  # use first available approved recording per slug
                except Exception as exc:
                    print(f"[WARN] {slug}/{video_id}: {exc}", file=sys.stderr)

        if not recordings:
            print("ERROR: no recordings loaded. Run `python -m piece_id_eval.transcribe` first.", file=sys.stderr)
            sys.exit(1)

        report = run(
            catalog=catalog,
            recordings=recordings,
            no_track=args.no_track,
        )

    # Print recall table
    print("\n=== Recall Table ===")
    for (mname, wlabel), vals in sorted(report.recall_table.items()):
        print(f"  {mname:30s} window={wlabel:6s}  R@1={vals['recall@1']:.3f}  R@5={vals.get('recall@5', 0):.3f}  R@10={vals['recall@10']:.3f}")

    print(f"\nDTW ceiling recall@10: {report.dtw_ceiling_recall10:.3f}")
    print(f"Best indexable recall@10: {report.best_indexable_recall10:.3f}")
    print(f"Open-set ok (FA<=0.05 @ TA>=0.60): {report.open_set_ok}")
    print(f"\nVERDICT: {report.verdict}")

    if args.output_json:
        out: dict[str, Any] = {
            "verdict": report.verdict,
            "dtw_ceiling_recall10": report.dtw_ceiling_recall10,
            "best_indexable_recall10": report.best_indexable_recall10,
            "open_set_ok": report.open_set_ok,
            "recall_table": {f"{m}|{w}": v for (m, w), v in report.recall_table.items()},
            "corruption_curves": {f"{m}|{c}": v for (m, c), v in report.corruption_curves.items()},
        }
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\nSidecar JSON written to {args.output_json}")


if __name__ == "__main__":
    _cli_main()
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run pytest tests/piece_id_eval/test_bakeoff.py -p no:cov -x
```
Expected: PASS (5 tests)

- [ ] **Step 5: Delete dead files and commit**

```bash
git rm model/src/piece_id_eval/matchers/chord_ngram.py \
       model/src/piece_id_eval/matchers/twodft.py \
       model/src/piece_id_eval/query_chroma.py \
       model/src/piece_id_eval/score_chroma.py \
       model/src/piece_id_eval/query_set.py \
       model/src/piece_id_eval/cli.py \
       model/src/piece_id_eval/report.py \
       model/tests/piece_id_eval/test_query_chroma.py \
       model/tests/piece_id_eval/test_score_chroma.py \
       model/tests/piece_id_eval/test_query_set.py \
       model/tests/piece_id_eval/test_cli_smoke.py \
       model/tests/piece_id_eval/test_report.py

git add model/src/piece_id_eval/bakeoff.py \
        model/src/piece_id_eval/matchers/__init__.py \
        model/tests/piece_id_eval/test_bakeoff.py

git commit -m "feat(piece-id): add bakeoff orchestrator + CLI; delete dead audio-chroma harness"
```

---

## Post-Build Operational Step (not a code task — performed by researcher after all tasks pass)

After all tasks are committed and `cd model && uv run pytest tests/piece_id_eval/ -p no:cov` is green:

1. Start the local AMT server: `just amt`
2. Transcribe the 15 missing slugs (all except `chopin_ballade_1` which already has `amt_notes.json`):

```bash
cd model && uv run python -m piece_id_eval.transcribe \
  --slugs bach_invention_1 bach_prelude_c_wtc1 chopin_etude_op10no4 \
          chopin_waltz_csm clair_de_lune debussy_arabesque_1 \
          fantaisie_impromptu fur_elise liszt_liebestraum_3 \
          moonlight_sonata_mvt1 mozart_k545_mvt1 nocturne_op9no2 \
          pathetique_mvt2 rachmaninoff_prelude_csm schumann_traumerei
```

3. Run the real bakeoff:

```bash
cd model && uv run python -m piece_id_eval.bakeoff --output-json bakeoff_results.json
```

4. Read `VERDICT:` from stdout. If PROCEED, name the winning matcher and window length for the Rust/WASM Phase 1 plan.
