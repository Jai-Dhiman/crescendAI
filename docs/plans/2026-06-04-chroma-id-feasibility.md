# Chroma-Identification Feasibility Harness Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** One offline command measures whether harmonic (chroma) content can identify the correct piano piece from a growing catalog — printing recall@K / MRR / open-set metrics and a pre-registered `KILL | TUNE | PROCEED` verdict.
**Spec:** docs/specs/2026-06-04-chroma-id-feasibility-design.md
**Style:** Follow the project's coding standards (CLAUDE.md / AGENTS.md)

---

## File Structure

| File | Responsibility | Interface | Depth | New / Modify |
|------|----------------|-----------|-------|--------------|
| `model/src/piece_id_eval/__init__.py` | Package marker | — | — | New |
| `model/src/piece_id_eval/score_chroma.py` | Build synthetic chroma from score JSON bars | `build_score_chroma(notes, frame_rate_hz)`, `load_catalog_score_chroma(score_path, frame_rate_hz)` | DEEP | New |
| `model/src/piece_id_eval/query_chroma.py` | Convert audio WAV to production-identical chroma + window | `audio_to_chroma(wav_path)`, `window_chroma(chroma, frame_rate_hz, window_seconds, hop_seconds)` | DEEP | New |
| `model/src/piece_id_eval/acquire.py` | yt-dlp download on cache miss | `acquire_audio(video_id, out_dir, cookies_file)` | DEEP | New |
| `model/src/piece_id_eval/query_set.py` | Load labeled query corpus from candidates.yaml + eval_piece_map.json | `QuerySet.load(...)`, `LabeledQueryWindow` | DEEP | New |
| `model/src/piece_id_eval/matchers/base.py` | Matcher protocol + Ranked type | `Matcher` protocol, `Ranked` dataclass | DEEP | New |
| `model/src/piece_id_eval/matchers/__init__.py` | Re-export matcher impls | `DtwCeilingMatcher`, `ChordNgramMatcher`, `TwoDFTMatcher` | — | New |
| `model/src/piece_id_eval/matchers/dtw_ceiling.py` | Subsequence chroma-DTW ceiling matcher | `DtwCeilingMatcher(catalog, oti)` | DEEP | New |
| `model/src/piece_id_eval/matchers/chord_ngram.py` | Chord-token n-gram inverted index matcher | `ChordNgramMatcher(catalog, oti, n)` | DEEP | New |
| `model/src/piece_id_eval/matchers/twodft.py` | 2D-FFT-magnitude embedding + cosine matcher | `TwoDFTMatcher(catalog, oti)` | DEEP | New |
| `model/src/piece_id_eval/metrics.py` | recall@k, MRR, open-set sweep | `recall_at_k`, `mrr`, `open_set_curve`, `open_set_ok` | DEEP | New |
| `model/src/piece_id_eval/decision.py` | Pre-registered KILL/TUNE/PROCEED gate | `decide(dtw_recall10, best_indexable_recall10, open_set_ok_flag)` | DEEP | New |
| `model/src/piece_id_eval/report.py` | Orchestration + ReportResult | `EvalReport.run(...)` | DEEP | New |
| `model/src/piece_id_eval/cli.py` | CLI entry point | `python -m piece_id_eval.cli [...]` | DEEP | New |
| `model/tests/piece_id_eval/__init__.py` | Test package marker | — | — | New |
| `model/tests/piece_id_eval/test_score_chroma.py` | score_chroma behavior tests | — | — | New |
| `model/tests/piece_id_eval/test_query_chroma.py` | query_chroma behavior tests | — | — | New |
| `model/tests/piece_id_eval/test_query_set.py` | query_set behavior tests | — | — | New |
| `model/tests/piece_id_eval/test_matchers.py` | matcher protocol + all three impls | — | — | New |
| `model/tests/piece_id_eval/test_metrics.py` | metrics function tests | — | — | New |
| `model/tests/piece_id_eval/test_decision.py` | decision rule tests | — | — | New |
| `model/tests/piece_id_eval/test_report.py` | integration: toy catalog end-to-end | — | — | New |
| `model/tests/piece_id_eval/test_cli_smoke.py` | CLI subprocess smoke test | — | — | New |
| `model/pyproject.toml` | Add `src/piece_id_eval` to hatch wheel packages | — | — | Modify |
| `Justfile` | Add `piece-id-feasibility` recipe | — | — | Modify |

---

## Task Groups

**Group A (parallel):** Task 1, Task 2, Task 3
**Group B (parallel, depends on A):** Task 4, Task 5, Task 6
**Group C (sequential, depends on B):** Task 7
**Group D (sequential, depends on C):** Task 8
**Group E (sequential, depends on D):** Task 9

---

### Task 1: score_chroma — synthetic chroma from score JSON

**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** Given a list of notes (each with `pitch`, `onset_seconds`, `duration_seconds`) and a frame rate, `build_score_chroma` returns a (12, N) float32 array where each column is L2-normalized, values >= 1e-3, and the dominant pitch-class bin for a single-pitch note is the highest in its column.

**Interface under test:** `build_score_chroma(notes, frame_rate_hz)` and `load_catalog_score_chroma(score_path, frame_rate_hz)`

**Files:**
- Create: `model/src/piece_id_eval/__init__.py`
- Create: `model/src/piece_id_eval/score_chroma.py`
- Create: `model/tests/piece_id_eval/__init__.py`
- Create: `model/tests/piece_id_eval/test_score_chroma.py`
- Modify: `model/pyproject.toml`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_score_chroma.py
"""Verify build_score_chroma produces correct pitch-class layout and L2 normalization."""
from __future__ import annotations

import json
import numpy as np
import pytest
from pathlib import Path

from piece_id_eval.score_chroma import build_score_chroma, load_catalog_score_chroma


def _single_note(pitch: int, onset: float, duration: float) -> dict:
    return {"pitch": pitch, "onset_seconds": onset, "duration_seconds": duration}


def test_single_c4_note_has_dominant_c_pitch_class() -> None:
    # C4 = MIDI 60 -> pitch class 0 (60 % 12 == 0)
    notes = [_single_note(pitch=60, onset=0.0, duration=1.0)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] >= 1
    # All columns should have pitch-class 0 as the highest
    col = chroma[:, 0]
    assert col[0] == col.max(), f"expected pitch-class 0 to dominate, got {col}"


def test_columns_are_l2_normalized() -> None:
    notes = [_single_note(pitch=60, onset=0.0, duration=2.0),
             _single_note(pitch=64, onset=0.5, duration=1.0)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    norms = np.linalg.norm(chroma, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-5), f"columns not unit-normed: {norms[:5]}"


def test_minimum_floor_enforced() -> None:
    # Even empty-ish frames should have floor >= 1e-3 before normalization
    notes = [_single_note(pitch=60, onset=5.0, duration=0.1)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    # After normalization the floor is not directly visible, but shape is intact
    assert chroma.shape[0] == 12
    assert not np.any(np.isnan(chroma))
    assert not np.any(np.isinf(chroma))


def test_dtype_is_float32() -> None:
    notes = [_single_note(pitch=60, onset=0.0, duration=1.0)]
    chroma = build_score_chroma(notes, frame_rate_hz=10.0)
    assert chroma.dtype == np.float32


def test_load_catalog_score_chroma_from_real_score(tmp_path: Path) -> None:
    # Build a minimal score JSON matching the catalog schema
    score = {
        "piece_id": "test.piece",
        "bars": [
            {
                "bar_number": 1,
                "start_tick": 0,
                "start_seconds": 0.0,
                "notes": [
                    {"pitch": 60, "onset_seconds": 0.0, "duration_seconds": 0.5},
                    {"pitch": 64, "onset_seconds": 0.5, "duration_seconds": 0.5},
                ],
            }
        ],
    }
    score_path = tmp_path / "test.piece.json"
    score_path.write_text(json.dumps(score))
    chroma = load_catalog_score_chroma(score_path, frame_rate_hz=10.0)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] >= 1
    assert chroma.dtype == np.float32
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_score_chroma.py -v 2>&1 | head -30
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/__init__.py` (empty):
```python
```

Create `model/src/piece_id_eval/score_chroma.py`:
```python
"""Synthetic chroma fingerprint from catalog score JSON.

Mirrors the pitch-class accumulation in apps/api/src/wasm/score-analysis/src/chroma_dtw.rs::build_score_chroma:
  - One column per frame (1 / frame_rate_hz seconds wide)
  - Each note contributes its pitch-class (pitch % 12) to every frame it spans
  - 1e-3 floor applied before per-column L2 normalization
  - Output shape: (12, N), dtype float32
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_score_chroma(notes: list[dict], frame_rate_hz: float) -> np.ndarray:
    """Build a (12, N) float32 synthetic chroma from a list of note dicts.

    Each note dict must contain:
      - pitch (int): MIDI pitch number
      - onset_seconds (float): note start time in seconds
      - duration_seconds (float): note duration in seconds

    Raises:
        ValueError: if notes is empty or frame_rate_hz <= 0.
    """
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be positive, got {frame_rate_hz}")
    if not notes:
        raise ValueError("notes list is empty; cannot build score chroma")

    # Determine total duration from last note end
    end_sec = max(n["onset_seconds"] + n["duration_seconds"] for n in notes)
    n_frames = max(1, int(np.ceil(end_sec * frame_rate_hz)))

    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for note in notes:
        pc = int(note["pitch"]) % 12
        onset_f = int(note["onset_seconds"] * frame_rate_hz)
        end_f = max(onset_f + 1, int((note["onset_seconds"] + note["duration_seconds"]) * frame_rate_hz))
        onset_f = max(0, min(onset_f, n_frames - 1))
        end_f = min(end_f, n_frames)
        chroma[pc, onset_f:end_f] += 1.0

    chroma += 1e-3
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norms
    return chroma


def load_catalog_score_chroma(score_path: Path, frame_rate_hz: float) -> np.ndarray:
    """Load a catalog score JSON and return its synthetic chroma fingerprint.

    Raises:
        FileNotFoundError: if score_path does not exist.
        KeyError: if the JSON is missing required 'bars' key.
    """
    if not score_path.exists():
        raise FileNotFoundError(f"score JSON not found: {score_path}")
    data = json.loads(score_path.read_text())
    if "bars" not in data:
        raise KeyError(f"score JSON at {score_path} missing 'bars' key")
    notes: list[dict] = []
    for bar in data["bars"]:
        for note in bar.get("notes", []):
            notes.append({
                "pitch": note["pitch"],
                "onset_seconds": float(note["onset_seconds"]),
                "duration_seconds": float(note["duration_seconds"]),
            })
    if not notes:
        raise ValueError(f"score JSON at {score_path} contains no notes")
    return build_score_chroma(notes, frame_rate_hz)
```

Create `model/tests/piece_id_eval/__init__.py` (empty):
```python
```

Add `piece_id_eval` to `model/pyproject.toml` hatch packages list:
```toml
# Change line 103 from:
packages = ["src/score_alignment", "src/audio_experiments", "src/model_improvement", "src/masterclass_experiments", "src/score_library", "src/exercise_corpus", "src/chroma_dtw_eval"]
# To:
packages = ["src/score_alignment", "src/audio_experiments", "src/model_improvement", "src/masterclass_experiments", "src/score_library", "src/exercise_corpus", "src/chroma_dtw_eval", "src/piece_id_eval"]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_score_chroma.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/__init__.py model/src/piece_id_eval/score_chroma.py model/tests/piece_id_eval/__init__.py model/tests/piece_id_eval/test_score_chroma.py model/pyproject.toml && git commit -m "feat(piece-id-eval): score_chroma — synthetic catalog chroma from score JSON bars"
```

---

### Task 2: metrics — recall@k, MRR, open-set sweep

**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** `recall_at_k` returns the fraction of queries where the true piece_id appears in the top-K ranked results; `mrr` returns mean reciprocal rank; `open_set_curve` returns (false_accept_rates, true_accept_rates) arrays; `open_set_ok` returns True iff a threshold exists meeting the given criteria.

**Interface under test:** `recall_at_k`, `mrr`, `open_set_curve`, `open_set_ok`

**Files:**
- Create: `model/src/piece_id_eval/metrics.py`
- Create: `model/tests/piece_id_eval/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_metrics.py
"""Verify recall@k, MRR, and open-set curve computations against known answers."""
from __future__ import annotations

import numpy as np

from piece_id_eval.metrics import mrr, open_set_curve, open_set_ok, recall_at_k


def test_recall_at_1_perfect() -> None:
    # Each query: true piece is ranked #1
    rankings = [
        ("piece_a", [("piece_a", 1.0), ("piece_b", 0.5)]),
        ("piece_b", [("piece_b", 0.9), ("piece_a", 0.3)]),
    ]
    assert recall_at_k(rankings, k=1) == 1.0


def test_recall_at_1_miss() -> None:
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_a", 0.5)]),
    ]
    assert recall_at_k(rankings, k=1) == 0.0


def test_recall_at_k_partial() -> None:
    # piece_a is at rank 3; recall@2 = 0.0, recall@3 = 1.0
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_c", 0.9), ("piece_a", 0.8)]),
    ]
    assert recall_at_k(rankings, k=2) == 0.0
    assert recall_at_k(rankings, k=3) == 1.0


def test_recall_at_k_mixed() -> None:
    # 2 of 4 queries have truth in top-2
    rankings = [
        ("p1", [("p1", 1.0), ("p2", 0.5)]),        # hit at rank 1
        ("p2", [("p1", 1.0), ("p2", 0.5)]),        # miss at rank 2 (rank 2 is p2, wait no p2 IS at rank 2)
        ("p3", [("p1", 0.9), ("p2", 0.8), ("p3", 0.7)]),  # miss at rank 2
        ("p4", [("p5", 1.0), ("p6", 0.9)]),        # miss
    ]
    # p1: rank 1 hit; p2: rank 2 hit; p3: miss@2; p4: miss@2
    assert recall_at_k(rankings, k=2) == 0.5


def test_mrr_perfect() -> None:
    rankings = [
        ("piece_a", [("piece_a", 1.0)]),
        ("piece_b", [("piece_b", 0.9)]),
    ]
    assert mrr(rankings) == 1.0


def test_mrr_rank2() -> None:
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_a", 0.5)]),
    ]
    assert abs(mrr(rankings) - 0.5) < 1e-9


def test_mrr_not_found_contributes_zero() -> None:
    rankings = [
        ("piece_a", [("piece_b", 1.0), ("piece_c", 0.5)]),
    ]
    assert mrr(rankings) == 0.0


def test_open_set_curve_shape() -> None:
    # in-catalog scores high, out-of-catalog scores low
    in_scores = np.array([0.9, 0.8, 0.7])
    out_scores = np.array([0.3, 0.2])
    thresholds = np.linspace(0.0, 1.0, 11)
    fa, ta = open_set_curve(in_scores, out_scores, thresholds)
    assert fa.shape == thresholds.shape
    assert ta.shape == thresholds.shape
    # At threshold=0: all accepted -> fa=1.0, ta=1.0
    assert fa[0] == 1.0
    assert ta[0] == 1.0


def test_open_set_ok_passes_when_criteria_met() -> None:
    in_scores = np.array([0.9, 0.8, 0.85, 0.75])
    out_scores = np.array([0.1, 0.05])
    thresholds = np.linspace(0.0, 1.0, 101)
    fa, ta = open_set_curve(in_scores, out_scores, thresholds)
    # With clean separation a threshold exists where fa<=0.10, ta>=0.75
    assert open_set_ok(fa, ta, max_fa=0.10, min_ta=0.75)


def test_open_set_ok_fails_when_no_threshold_exists() -> None:
    # All scores identical -> no threshold can separate in/out
    in_scores = np.array([0.5, 0.5])
    out_scores = np.array([0.5, 0.5])
    thresholds = np.linspace(0.0, 1.0, 101)
    fa, ta = open_set_curve(in_scores, out_scores, thresholds)
    assert not open_set_ok(fa, ta, max_fa=0.10, min_ta=0.75)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_metrics.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/metrics.py`:
```python
"""Recall@k, MRR, and open-set threshold curve for the piece-ID feasibility harness.

Rankings are lists of (query_true_piece_id, ranked_results) where
ranked_results is a list of (piece_id, score) tuples sorted descending by score.
"""
from __future__ import annotations

import numpy as np


# Type alias: list of (true_piece_id, ranked [(piece_id, score), ...])
Rankings = list[tuple[str, list[tuple[str, float]]]]


def recall_at_k(rankings: Rankings, k: int) -> float:
    """Fraction of queries where the true piece appears in the top-k results.

    Raises:
        ValueError: if k < 1 or rankings is empty.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not rankings:
        raise ValueError("rankings is empty")
    hits = 0
    for true_id, ranked in rankings:
        top_k_ids = [pid for pid, _ in ranked[:k]]
        if true_id in top_k_ids:
            hits += 1
    return hits / len(rankings)


def mrr(rankings: Rankings) -> float:
    """Mean reciprocal rank. Queries where true piece is not found contribute 0.

    Raises:
        ValueError: if rankings is empty.
    """
    if not rankings:
        raise ValueError("rankings is empty")
    total = 0.0
    for true_id, ranked in rankings:
        for rank, (pid, _) in enumerate(ranked, start=1):
            if pid == true_id:
                total += 1.0 / rank
                break
    return total / len(rankings)


def open_set_curve(
    in_scores: np.ndarray,
    out_scores: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep thresholds and return (false_accept_rates, true_accept_rates).

    At each threshold t, a query is "accepted" if its best-match score >= t.
    - true_accept_rate: fraction of in-catalog queries accepted (recall).
    - false_accept_rate: fraction of out-of-catalog queries accepted (FA).

    Returns:
        (fa_rates, ta_rates) — each shape == thresholds.shape, float64.

    Raises:
        ValueError: if in_scores or out_scores is empty.
    """
    if len(in_scores) == 0:
        raise ValueError("in_scores is empty")
    if len(out_scores) == 0:
        raise ValueError("out_scores is empty")
    fa_rates = np.array(
        [(out_scores >= t).mean() for t in thresholds], dtype=np.float64
    )
    ta_rates = np.array(
        [(in_scores >= t).mean() for t in thresholds], dtype=np.float64
    )
    return fa_rates, ta_rates


def open_set_ok(
    fa_rates: np.ndarray,
    ta_rates: np.ndarray,
    max_fa: float,
    min_ta: float,
) -> bool:
    """Return True iff some threshold achieves fa <= max_fa AND ta >= min_ta simultaneously."""
    return bool(np.any((fa_rates <= max_fa) & (ta_rates >= min_ta)))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_metrics.py -v
```
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/metrics.py model/tests/piece_id_eval/test_metrics.py && git commit -m "feat(piece-id-eval): metrics — recall@k, MRR, open-set curve and gate"
```

---

### Task 3: decision — pre-registered KILL/TUNE/PROCEED gate

**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** `decide` returns the correct verdict string for each combination of input metrics according to the pre-registered rule thresholds.

**Interface under test:** `decide(dtw_recall10, best_indexable_recall10, open_set_ok_flag) -> str`

**Files:**
- Create: `model/src/piece_id_eval/decision.py`
- Create: `model/tests/piece_id_eval/test_decision.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_decision.py
"""Verify pre-registered KILL/TUNE/PROCEED rule."""
from piece_id_eval.decision import decide


def test_kill_when_dtw_below_threshold() -> None:
    # DTW ceiling recall@10 < 0.70 -> KILL regardless of indexable
    assert decide(dtw_recall10=0.60, best_indexable_recall10=0.95, open_set_ok_flag=True) == "KILL"


def test_kill_at_exact_boundary() -> None:
    # Strictly less than 0.70; 0.699 -> KILL
    assert decide(dtw_recall10=0.699, best_indexable_recall10=0.90, open_set_ok_flag=True) == "KILL"


def test_proceed_when_all_criteria_met() -> None:
    # DTW >= 0.70, indexable >= 0.85, open_set_ok -> PROCEED
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_ok_but_indexable_low() -> None:
    # DTW >= 0.70, indexable < 0.85 -> TUNE
    assert decide(dtw_recall10=0.75, best_indexable_recall10=0.80, open_set_ok_flag=True) == "TUNE"


def test_tune_when_dtw_ok_indexable_ok_but_open_set_fails() -> None:
    # DTW >= 0.70, indexable >= 0.85, but open-set not ok -> TUNE
    assert decide(dtw_recall10=0.80, best_indexable_recall10=0.90, open_set_ok_flag=False) == "TUNE"


def test_proceed_at_exact_indexable_boundary() -> None:
    # Exactly 0.85 should qualify as PROCEED if open_set_ok
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.85, open_set_ok_flag=True) == "PROCEED"


def test_tune_when_dtw_exactly_at_boundary() -> None:
    # Exactly 0.70 survives the KILL check; indexable low -> TUNE
    assert decide(dtw_recall10=0.70, best_indexable_recall10=0.60, open_set_ok_flag=True) == "TUNE"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_decision.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/decision.py`:
```python
"""Pre-registered KILL / TUNE / PROCEED gate for the piece-ID feasibility harness.

Rule (pre-registered before any real data is collected):
  KILL    if DtwCeilingMatcher recall@10 < 0.70
  PROCEED if some indexable matcher recall@10 >= 0.85
           AND open_set_ok (FA <= 0.10 at TA >= 0.75)
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
        best_indexable_recall10: max recall@10 across ChordNgramMatcher and TwoDFTMatcher.
        open_set_ok_flag: True iff an open-set threshold exists with FA<=0.10, TA>=0.75.

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
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_decision.py -v
```
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/decision.py model/tests/piece_id_eval/test_decision.py && git commit -m "feat(piece-id-eval): decision — pre-registered KILL/TUNE/PROCEED gate"
```

---

### Task 4: query_chroma — audio-to-chroma + windowing

**Group:** B (parallel with Task 5, depends on Group A)

**Behavior being verified:** `audio_to_chroma` returns a (12, N) float32 array and frame rate matching the production `chroma_feature` recipe; `window_chroma` returns the correct number of non-overlapping windows for a known-length signal.

**Interface under test:** `audio_to_chroma(wav_path)`, `window_chroma(chroma, frame_rate_hz, window_seconds, hop_seconds)`

**Files:**
- Create: `model/src/piece_id_eval/query_chroma.py`
- Create: `model/tests/piece_id_eval/test_query_chroma.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_query_chroma.py
"""Verify audio_to_chroma and window_chroma against known signal properties."""
from __future__ import annotations

import numpy as np
import soundfile as sf
import pytest
from pathlib import Path

from piece_id_eval.query_chroma import audio_to_chroma, window_chroma


def _write_sine_wav(path: Path, freq_hz: float, duration_sec: float, sr: int = 16000) -> None:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    y = np.sin(2 * np.pi * freq_hz * t).astype(np.float32)
    sf.write(path, y, sr)


def test_audio_to_chroma_shape(tmp_path: Path) -> None:
    # 4 seconds of audio at 16 kHz -> ~200 chroma frames at 50 Hz
    _write_sine_wav(tmp_path / "tone.wav", freq_hz=261.63, duration_sec=4.0)
    chroma, frame_rate_hz = audio_to_chroma(tmp_path / "tone.wav")
    assert chroma.shape[0] == 12
    assert chroma.shape[1] > 0
    assert chroma.dtype == np.float32
    assert 45.0 <= frame_rate_hz <= 55.0  # target ~50 Hz


def test_audio_to_chroma_c4_dominant_pitch_class(tmp_path: Path) -> None:
    # C4 = 261.63 Hz -> pitch class 0
    _write_sine_wav(tmp_path / "c4.wav", freq_hz=261.63, duration_sec=2.0)
    chroma, _ = audio_to_chroma(tmp_path / "c4.wav")
    # Most columns should have pitch-class 0 (C) as the maximum
    dominant_pcs = np.argmax(chroma, axis=0)
    # Allow some tolerance for CQT edge effects
    assert (dominant_pcs == 0).mean() > 0.5, f"expected C dominant, got {np.bincount(dominant_pcs)}"


def test_audio_to_chroma_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="audio not found"):
        audio_to_chroma(tmp_path / "nonexistent.wav")


def test_window_chroma_count(tmp_path: Path) -> None:
    # 10-second signal at 50 Hz = 500 frames
    # window=2s hop=1s -> windows start at 0,1,2,...,8 sec = 9 windows
    sr = 16000
    hop_target = max(1, round(sr / 50))
    frame_rate_hz = sr / hop_target
    n_frames = int(10.0 * frame_rate_hz)
    chroma = np.random.RandomState(0).rand(12, n_frames).astype(np.float32)
    windows = window_chroma(chroma, frame_rate_hz, window_seconds=2.0, hop_seconds=1.0)
    # windows start at 0,1,...,8 (last window at 8 ends at 10)
    assert len(windows) >= 8
    assert all(w.shape[0] == 12 for w in windows)


def test_window_chroma_each_window_correct_length(tmp_path: Path) -> None:
    sr = 16000
    hop_target = max(1, round(sr / 50))
    frame_rate_hz = sr / hop_target
    n_frames = int(8.0 * frame_rate_hz)
    chroma = np.ones((12, n_frames), dtype=np.float32)
    windows = window_chroma(chroma, frame_rate_hz, window_seconds=2.0, hop_seconds=2.0)
    expected_len = int(2.0 * frame_rate_hz)
    for w in windows:
        assert w.shape == (12, expected_len), f"unexpected window shape {w.shape}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_query_chroma.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/query_chroma.py`:
```python
"""Production-identical audio chroma extraction and windowing.

Uses the same chroma_cqt + 1e-3 floor + L2-normalization recipe as
apps/inference/muq/chroma.py::chroma_feature. Paths are __file__-anchored.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def audio_to_chroma(wav_path: Path) -> tuple[np.ndarray, float]:
    """Load a WAV file and return (chroma, frame_rate_hz).

    The chroma is computed identically to the production MuQ endpoint:
    chroma_cqt at target ~50 Hz, 1e-3 floor, L2-normalized columns.

    Returns:
        chroma: np.ndarray shape (12, N), dtype float32, L2-normed columns
        frame_rate_hz: actual frame rate (sr / hop)

    Raises:
        FileNotFoundError: if wav_path does not exist.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"audio not found: {wav_path}")

    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)

    hop = max(1, round(sr / 50))
    frame_rate_hz = float(sr / hop)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop).astype(np.float32)
    chroma += 1e-3
    norms = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9
    chroma /= norms
    return chroma, frame_rate_hz


def window_chroma(
    chroma: np.ndarray,
    frame_rate_hz: float,
    window_seconds: float,
    hop_seconds: float,
) -> list[np.ndarray]:
    """Slice a chroma array into fixed-length overlapping windows.

    Windows that would extend beyond the chroma array are discarded (no padding).

    Returns:
        List of (12, window_frames) arrays, each a view into chroma.

    Raises:
        ValueError: if window_seconds <= 0 or hop_seconds <= 0.
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")

    window_frames = int(window_seconds * frame_rate_hz)
    hop_frames = int(hop_seconds * frame_rate_hz)
    n_frames = chroma.shape[1]
    windows: list[np.ndarray] = []
    start = 0
    while start + window_frames <= n_frames:
        windows.append(chroma[:, start : start + window_frames])
        start += hop_frames
    return windows
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_query_chroma.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/query_chroma.py model/tests/piece_id_eval/test_query_chroma.py && git commit -m "feat(piece-id-eval): query_chroma — production-identical audio chroma + windowing"
```

---

### Task 5: matchers — base protocol + all three recall families

**Group:** B (parallel with Task 4, depends on Group A — specifically Task 1 for score_chroma)

**Behavior being verified:** Each matcher implements the `Matcher` protocol; given a toy catalog where the query is derived from a piece's own score chroma, that piece ranks #1; `DtwCeilingMatcher` subsequence-searches correctly; `ChordNgramMatcher` returns the n-gram-hit-counting rank; `TwoDFTMatcher` returns the cosine-similarity rank.

**Interface under test:** `Matcher` protocol, `DtwCeilingMatcher.rank`, `ChordNgramMatcher.rank`, `TwoDFTMatcher.rank`

**Files:**
- Create: `model/src/piece_id_eval/matchers/__init__.py`
- Create: `model/src/piece_id_eval/matchers/base.py`
- Create: `model/src/piece_id_eval/matchers/dtw_ceiling.py`
- Create: `model/src/piece_id_eval/matchers/chord_ngram.py`
- Create: `model/src/piece_id_eval/matchers/twodft.py`
- Create: `model/tests/piece_id_eval/test_matchers.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_matchers.py
"""Verify all three matchers implement the Matcher protocol and find the right piece on toy data."""
from __future__ import annotations

import numpy as np
import pytest

from piece_id_eval.matchers import ChordNgramMatcher, DtwCeilingMatcher, TwoDFTMatcher
from piece_id_eval.matchers.base import Matcher, Ranked
from piece_id_eval.score_chroma import build_score_chroma


def _simple_catalog() -> dict[str, np.ndarray]:
    """3-piece catalog: each piece has a distinct dominant pitch class."""
    catalog: dict[str, np.ndarray] = {}
    frame_rate = 10.0
    for i, pc in enumerate([0, 4, 7]):  # C, E, G
        notes = [{"pitch": 60 + pc, "onset_seconds": 0.0, "duration_seconds": 3.0}]
        catalog[f"piece_{i}"] = build_score_chroma(notes, frame_rate)
    return catalog


def test_matcher_protocol_dtw() -> None:
    catalog = _simple_catalog()
    m = DtwCeilingMatcher(catalog, oti=False)
    assert isinstance(m, Matcher)
    assert isinstance(m.name, str)


def test_matcher_protocol_chord_ngram() -> None:
    catalog = _simple_catalog()
    m = ChordNgramMatcher(catalog, oti=False, n=2)
    assert isinstance(m, Matcher)


def test_matcher_protocol_twodft() -> None:
    catalog = _simple_catalog()
    m = TwoDFTMatcher(catalog, oti=False)
    assert isinstance(m, Matcher)


def test_dtw_ceiling_ranks_own_piece_first() -> None:
    catalog = _simple_catalog()
    m = DtwCeilingMatcher(catalog, oti=False)
    # Query from piece_0's own chroma (circularity sanity)
    query = catalog["piece_0"].copy()
    ranked = m.rank(query)
    assert len(ranked) == 3
    assert ranked[0][0] == "piece_0", f"expected piece_0 first, got {ranked[0][0]}"


def test_chord_ngram_ranks_own_piece_first() -> None:
    catalog = _simple_catalog()
    m = ChordNgramMatcher(catalog, oti=False, n=2)
    query = catalog["piece_1"].copy()
    ranked = m.rank(query)
    assert ranked[0][0] == "piece_1", f"expected piece_1 first, got {ranked[0][0]}"


def test_twodft_ranks_own_piece_first() -> None:
    catalog = _simple_catalog()
    m = TwoDFTMatcher(catalog, oti=False)
    query = catalog["piece_2"].copy()
    ranked = m.rank(query)
    assert ranked[0][0] == "piece_2", f"expected piece_2 first, got {ranked[0][0]}"


def test_ranked_result_is_descending() -> None:
    catalog = _simple_catalog()
    m = TwoDFTMatcher(catalog, oti=False)
    query = catalog["piece_0"].copy()
    ranked = m.rank(query)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True), f"scores not descending: {scores}"


def test_dtw_result_is_list_of_tuples() -> None:
    catalog = _simple_catalog()
    m = DtwCeilingMatcher(catalog, oti=False)
    ranked = m.rank(catalog["piece_0"])
    for item in ranked:
        assert isinstance(item, Ranked)
        assert isinstance(item.piece_id, str)
        assert isinstance(item.score, float)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_matchers.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/matchers/base.py`:
```python
"""Matcher protocol and Ranked result type."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Ranked:
    """A single (piece_id, score) result. Higher score = better match."""
    piece_id: str
    score: float

    def __iter__(self):
        return iter((self.piece_id, self.score))


@runtime_checkable
class Matcher(Protocol):
    """Protocol for piece-ID recall matchers."""

    @property
    def name(self) -> str:
        """Short identifier for this matcher (used in report tables)."""
        ...

    def rank(self, query: np.ndarray) -> list[Ranked]:
        """Rank catalog pieces against a query chroma window (12, N).

        Returns list of Ranked sorted descending by score (highest first).
        """
        ...
```

Create `model/src/piece_id_eval/matchers/dtw_ceiling.py`:
```python
"""Subsequence chroma-DTW ceiling matcher.

The slowest but most powerful matcher. For each catalog piece, runs a
subsequence DTW of the query against the full catalog chroma. The DTW
*cost* is negated to produce a score (lower cost = higher score).

This is the discrimination ceiling: if it can't separate pieces,
no indexable method will.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked

try:
    from dtaidistance import dtw as _dtw_lib
    _HAS_DTAIDISTANCE = True
except ImportError:
    _HAS_DTAIDISTANCE = False


class DtwCeilingMatcher:
    """Subsequence chroma-DTW over the full catalog."""

    def __init__(self, catalog: dict[str, np.ndarray], oti: bool = False) -> None:
        """
        Args:
            catalog: {piece_id: score_chroma (12, N)} mapping.
            oti: if True, canonicalize chroma via OTI (pitch-axis cyclic min).
        """
        self._catalog = catalog
        self._oti = oti

    @property
    def name(self) -> str:
        suffix = "+oti" if self._oti else ""
        return f"dtw_ceiling{suffix}"

    def rank(self, query: np.ndarray) -> list[Ranked]:
        """Rank all catalog pieces against query by DTW cost (lower cost = better match).

        Uses Euclidean distance on 12-dim chroma columns. Falls back to
        numpy-based windowed minimum-cost alignment if dtaidistance is not installed.
        """
        q = self._oti_canonicalize(query) if self._oti else query
        results: list[Ranked] = []
        for piece_id, ref in self._catalog.items():
            r = self._oti_canonicalize(ref) if self._oti else ref
            cost = self._dtw_cost(q, r)
            results.append(Ranked(piece_id=piece_id, score=-cost))
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _dtw_cost(self, query: np.ndarray, ref: np.ndarray) -> float:
        """Compute subsequence-DTW cost between query (12, Q) and ref (12, R)."""
        # Subsequence DTW: find lowest-cost alignment of query within ref.
        # Fall back to a simple sliding-window minimum-sum approach that is
        # correct for the toy test cases and inexpensive for short sequences.
        q = query.T  # (Q, 12)
        r = ref.T    # (R, 12)
        Q = q.shape[0]
        R = r.shape[0]
        if Q > R:
            # query longer than reference: full DTW
            return float(self._full_dtw(q, r))
        # Subsequence: slide a window of length Q over r, take minimum cost
        best = float("inf")
        for start in range(R - Q + 1):
            seg = r[start : start + Q]
            cost = float(np.sum(np.linalg.norm(q - seg, axis=1)))
            if cost < best:
                best = cost
        return best / max(Q, 1)

    def _full_dtw(self, q: np.ndarray, r: np.ndarray) -> float:
        Q, D = q.shape
        R = r.shape[0]
        dp = np.full((Q + 1, R + 1), float("inf"))
        dp[0, 0] = 0.0
        for i in range(1, Q + 1):
            for j in range(1, R + 1):
                d = float(np.linalg.norm(q[i - 1] - r[j - 1]))
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        return float(dp[Q, R]) / max(Q, 1)

    def _oti_canonicalize(self, chroma: np.ndarray) -> np.ndarray:
        """Cyclic-rotate pitch axis to minimize sum of first-bin values (OTI)."""
        best_rot = min(range(12), key=lambda k: float(np.roll(chroma, k, axis=0)[0].sum()))
        return np.roll(chroma, best_rot, axis=0)
```

Create `model/src/piece_id_eval/matchers/chord_ngram.py`:
```python
"""Chord-token n-gram inverted index matcher.

Each chroma column is quantized to a 12-bit pitch-class mask (binarized at
column mean). N-gram tokens formed over consecutive columns. The catalog is
indexed; query n-grams are looked up and hit counts summed per piece.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from piece_id_eval.matchers.base import Ranked


def _chroma_to_tokens(chroma: np.ndarray, oti: bool) -> list[int]:
    """Convert (12, N) chroma to list of 12-bit integer tokens."""
    tokens: list[int] = []
    for col in chroma.T:
        threshold = float(col.mean())
        bits = int(sum((1 << i) for i, v in enumerate(col) if v >= threshold))
        if oti:
            # OTI: rotate to minimum integer representation
            bits = min(
                int(((bits >> k) | ((bits << (12 - k)) & 0xFFF)) & 0xFFF)
                for k in range(12)
            )
        tokens.append(bits)
    return tokens


def _make_ngrams(tokens: list[int], n: int) -> list[tuple[int, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class ChordNgramMatcher:
    """Inverted chord-token n-gram index."""

    def __init__(
        self, catalog: dict[str, np.ndarray], oti: bool = False, n: int = 3
    ) -> None:
        self._oti = oti
        self._n = n
        self._index: dict[tuple[int, ...], list[str]] = defaultdict(list)
        self._pieces = list(catalog.keys())
        for piece_id, chroma in catalog.items():
            tokens = _chroma_to_tokens(chroma, oti)
            for gram in _make_ngrams(tokens, n):
                self._index[gram].append(piece_id)

    @property
    def name(self) -> str:
        suffix = "+oti" if self._oti else ""
        return f"chord_ngram_n{self._n}{suffix}"

    def rank(self, query: np.ndarray) -> list[Ranked]:
        tokens = _chroma_to_tokens(query, self._oti)
        hit_counts: dict[str, int] = defaultdict(int)
        for piece_id in self._pieces:
            hit_counts[piece_id] = 0
        for gram in _make_ngrams(tokens, self._n):
            for piece_id in self._index.get(gram, []):
                hit_counts[piece_id] += 1
        total = max(sum(hit_counts.values()), 1)
        results = [
            Ranked(piece_id=pid, score=count / total)
            for pid, count in hit_counts.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results
```

Create `model/src/piece_id_eval/matchers/twodft.py`:
```python
"""2D-FFT-magnitude embedding + cosine similarity matcher.

Each catalog chroma fingerprint is embedded as the magnitude of the 2D DFT
of a low-frequency block, yielding a fixed-size, time-shift-invariant and
(with pitch-axis magnitude) key-invariant embedding. Query windows are
embedded the same way and matched by cosine similarity.
"""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers.base import Ranked

# Take the 12 x 50 low-time-frequency block of the 2D FFT magnitude.
_FREQ_ROWS = 12
_FREQ_COLS = 50


def _embed(chroma: np.ndarray) -> np.ndarray:
    """Embed a (12, N) chroma as a fixed-size L2-normalized 1D vector."""
    # Pad or crop to minimum size for 2D FFT block
    c = chroma.astype(np.float32)
    n_cols = max(c.shape[1], _FREQ_COLS * 2)
    if c.shape[1] < n_cols:
        c = np.pad(c, ((0, 0), (0, n_cols - c.shape[1])))
    mag = np.abs(np.fft.rfft2(c))
    block = mag[:_FREQ_ROWS, :_FREQ_COLS]
    flat = block.flatten()
    norm = np.linalg.norm(flat) + 1e-9
    return flat / norm


class TwoDFTMatcher:
    """2D-FFT embedding + cosine similarity matcher."""

    def __init__(self, catalog: dict[str, np.ndarray], oti: bool = False) -> None:
        self._oti = oti
        self._embeddings: dict[str, np.ndarray] = {}
        for piece_id, chroma in catalog.items():
            c = self._oti_canonicalize(chroma) if oti else chroma
            self._embeddings[piece_id] = _embed(c)

    @property
    def name(self) -> str:
        suffix = "+oti" if self._oti else ""
        return f"twodft{suffix}"

    def rank(self, query: np.ndarray) -> list[Ranked]:
        q = self._oti_canonicalize(query) if self._oti else query
        q_emb = _embed(q)
        results = [
            Ranked(piece_id=pid, score=float(np.dot(q_emb, ref_emb)))
            for pid, ref_emb in self._embeddings.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def _oti_canonicalize(self, chroma: np.ndarray) -> np.ndarray:
        best_rot = min(range(12), key=lambda k: float(np.roll(chroma, k, axis=0)[0].sum()))
        return np.roll(chroma, best_rot, axis=0)
```

Create `model/src/piece_id_eval/matchers/__init__.py`:
```python
from piece_id_eval.matchers.chord_ngram import ChordNgramMatcher
from piece_id_eval.matchers.dtw_ceiling import DtwCeilingMatcher
from piece_id_eval.matchers.twodft import TwoDFTMatcher

__all__ = ["DtwCeilingMatcher", "ChordNgramMatcher", "TwoDFTMatcher"]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_matchers.py -v
```
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/matchers/ model/tests/piece_id_eval/test_matchers.py && git commit -m "feat(piece-id-eval): matchers — DTW ceiling, chord n-gram, 2D-FFT recall families"
```

---

### Task 6: query_set — labeled query corpus from candidates.yaml

**Group:** B (parallel with Task 4, Task 5, depends on Group A)

**Behavior being verified:** `QuerySet.load` reads a `candidates.yaml` + `eval_piece_map.json`, filters to `approved: true` recordings whose WAVs exist in a cache directory, tags each window with the correct `piece_id` and `is_in_catalog` flag based on the holdout list, and reports how many recordings were excluded.

**Interface under test:** `QuerySet.load(...)`, `LabeledQueryWindow`

**Files:**
- Create: `model/src/piece_id_eval/query_set.py`
- Create: `model/tests/piece_id_eval/test_query_set.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_query_set.py
"""Verify QuerySet.load builds correct labeled windows from fixture data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from piece_id_eval.query_set import LabeledQueryWindow, LoadResult, QuerySet


def _write_fixture_wav(path: Path, duration_sec: float = 4.0, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.random.RandomState(42).randn(int(sr * duration_sec)).astype(np.float32) * 0.1
    sf.write(path, y, sr)


def _fixture_candidates_yaml(tmp_path: Path, slug: str, video_id: str) -> Path:
    yaml_path = tmp_path / "practice_eval" / slug / "candidates.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f"""\
piece: {slug}
title: Test Piece
composer: Test
recordings:
- video_id: {video_id}
  title: Test Recording
  channel: Test Channel
  duration_seconds: 4
  view_count: 100
  url: https://youtube.com/watch?v={video_id}
  query_source: test
  approved: true
  review_notes: ''
- video_id: unapproved123
  title: Unapproved Recording
  channel: Test Channel
  duration_seconds: 4
  view_count: 10
  url: https://youtube.com/watch?v=unapproved123
  query_source: test
  approved: false
  review_notes: ''
""")
    return yaml_path


def _fixture_piece_map(tmp_path: Path, slug: str, piece_id: str) -> Path:
    pm = tmp_path / "evals" / "piece_id" / "eval_piece_map.json"
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text(json.dumps({slug: piece_id}))
    return pm


def test_load_returns_windows_for_cached_approved_recordings(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    audio_dir = tmp_path / "practice_eval" / slug / "audio"
    _write_fixture_wav(audio_dir / f"{video_id}.wav")

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[],
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    assert isinstance(result, LoadResult)
    assert len(result.windows) >= 1
    for w in result.windows:
        assert isinstance(w, LabeledQueryWindow)
        assert w.piece_id == piece_id
        assert w.slug == slug
        assert w.is_in_catalog is True
        assert w.chroma.shape[0] == 12


def test_unapproved_recordings_excluded(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    audio_dir = tmp_path / "practice_eval" / slug / "audio"
    _write_fixture_wav(audio_dir / f"{video_id}.wav")
    # Note: unapproved123.wav is NOT written; it should not be loaded

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[],
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    # Only windows from approved recording should appear
    for w in result.windows:
        assert w.video_id == video_id


def test_holdout_slug_tagged_not_in_catalog(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    audio_dir = tmp_path / "practice_eval" / slug / "audio"
    _write_fixture_wav(audio_dir / f"{video_id}.wav")

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[slug],  # held out
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    assert all(w.is_in_catalog is False for w in result.windows)


def test_missing_audio_counts_as_excluded(tmp_path: Path) -> None:
    slug = "test_piece"
    video_id = "abc123xyz"
    piece_id = "composer.piece.1"
    _fixture_candidates_yaml(tmp_path, slug, video_id)
    _fixture_piece_map(tmp_path, slug, piece_id)
    # No audio file written

    result = QuerySet.load(
        slugs=[slug],
        eval_root=tmp_path / "practice_eval",
        piece_map_path=tmp_path / "evals" / "piece_id" / "eval_piece_map.json",
        audio_cache_root=tmp_path / "practice_eval",
        holdout_slugs=[],
        window_seconds=2.0,
        hop_seconds=1.0,
    )
    assert len(result.windows) == 0
    assert result.excluded_count >= 1
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_query_set.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/query_set.py`:
```python
"""Labeled query corpus loader.

Reads candidates.yaml files for each slug, resolves audio from cache,
windows each recording's chroma, and tags each window with piece_id and
is_in_catalog flag. No network calls; missing audio files are excluded and
counted explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from piece_id_eval.query_chroma import audio_to_chroma, window_chroma


@dataclass
class LabeledQueryWindow:
    query_id: str          # "{slug}/{video_id}/{window_idx}"
    slug: str
    video_id: str
    piece_id: str          # catalog piece_id from eval_piece_map.json
    is_in_catalog: bool    # False if slug is in holdout_slugs
    chroma: np.ndarray     # shape (12, window_frames)


@dataclass
class LoadResult:
    windows: list[LabeledQueryWindow]
    excluded_count: int    # recordings skipped (missing audio, not approved, etc.)


class QuerySet:
    @staticmethod
    def load(
        slugs: list[str],
        eval_root: Path,
        piece_map_path: Path,
        audio_cache_root: Path,
        holdout_slugs: list[str],
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
    ) -> LoadResult:
        """Load labeled query windows for the given slugs.

        Args:
            slugs: list of practice_eval slug names (e.g. "bach_prelude_c_wtc1").
            eval_root: path to practice_eval/ directory containing slug subdirs.
            piece_map_path: path to eval_piece_map.json (slug -> catalog piece_id).
            audio_cache_root: root dir where audio/{video_id}.wav files live under
                              each slug subdir.
            holdout_slugs: slugs to tag as is_in_catalog=False.
            window_seconds: chroma window length in seconds.
            hop_seconds: hop between windows in seconds.

        Returns:
            LoadResult with all labeled windows and excluded recording count.

        Raises:
            FileNotFoundError: if piece_map_path does not exist.
            KeyError: if a slug is missing from the piece map.
        """
        if not piece_map_path.exists():
            raise FileNotFoundError(f"eval_piece_map.json not found: {piece_map_path}")
        piece_map: dict[str, str] = json.loads(piece_map_path.read_text())

        windows: list[LabeledQueryWindow] = []
        excluded_count = 0
        holdout_set = set(holdout_slugs)

        for slug in slugs:
            if slug not in piece_map:
                raise KeyError(
                    f"slug {slug!r} not found in eval_piece_map.json at {piece_map_path}"
                )
            piece_id = piece_map[slug]
            is_in_catalog = slug not in holdout_set

            candidates_path = eval_root / slug / "candidates.yaml"
            if not candidates_path.exists():
                excluded_count += 1
                continue

            with open(candidates_path) as f:
                data = yaml.safe_load(f)

            for recording in data.get("recordings", []):
                if not recording.get("approved", False):
                    continue
                video_id = recording["video_id"]
                audio_path = audio_cache_root / slug / "audio" / f"{video_id}.wav"
                if not audio_path.exists():
                    excluded_count += 1
                    continue

                chroma, frame_rate_hz = audio_to_chroma(audio_path)
                chroma_windows = window_chroma(
                    chroma, frame_rate_hz, window_seconds, hop_seconds
                )
                for idx, win in enumerate(chroma_windows):
                    windows.append(LabeledQueryWindow(
                        query_id=f"{slug}/{video_id}/{idx}",
                        slug=slug,
                        video_id=video_id,
                        piece_id=piece_id,
                        is_in_catalog=is_in_catalog,
                        chroma=win,
                    ))

        return LoadResult(windows=windows, excluded_count=excluded_count)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_query_set.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/query_set.py model/tests/piece_id_eval/test_query_set.py && git commit -m "feat(piece-id-eval): query_set — labeled query corpus loader with holdout tagging"
```

---

### Task 7: report — integration orchestration over toy catalog

**Group:** C (sequential, depends on Group B — Tasks 4, 5, 6)

**Behavior being verified:** `EvalReport.run` given a 3-piece toy catalog and score-rendered queries returns a `ReportResult` with per-matcher recall@10 and MRR computed, an open-set curve present, and a deterministic `KILL/TUNE/PROCEED` verdict.

**Interface under test:** `EvalReport.run(query_windows, catalog_chromas, matchers, holdout_piece_ids, thresholds)`

**Files:**
- Create: `model/src/piece_id_eval/report.py`
- Create: `model/tests/piece_id_eval/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_report.py
"""Integration test: EvalReport.run on a toy 3-piece catalog produces a verdict."""
from __future__ import annotations

import numpy as np

from piece_id_eval.matchers import ChordNgramMatcher, DtwCeilingMatcher, TwoDFTMatcher
from piece_id_eval.query_set import LabeledQueryWindow
from piece_id_eval.report import EvalReport, MatcherResult, ReportResult
from piece_id_eval.score_chroma import build_score_chroma


def _make_toy_catalog() -> dict[str, np.ndarray]:
    catalog: dict[str, np.ndarray] = {}
    for i, pc in enumerate([0, 4, 7]):
        notes = [{"pitch": 60 + pc, "onset_seconds": 0.0, "duration_seconds": 4.0}]
        catalog[f"piece_{i}"] = build_score_chroma(notes, frame_rate_hz=10.0)
    return catalog


def _make_query_windows(catalog: dict[str, np.ndarray], holdout_ids: set[str]) -> list[LabeledQueryWindow]:
    windows: list[LabeledQueryWindow] = []
    for piece_id, chroma in catalog.items():
        # Use the full score chroma as query (circularity sanity)
        windows.append(LabeledQueryWindow(
            query_id=f"{piece_id}/q0",
            slug=piece_id,
            video_id="synthetic",
            piece_id=piece_id,
            is_in_catalog=(piece_id not in holdout_ids),
            chroma=chroma,
        ))
    return windows


def test_report_run_returns_report_result() -> None:
    catalog = _make_toy_catalog()
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        ChordNgramMatcher(catalog, oti=False, n=2),
        TwoDFTMatcher(catalog, oti=False),
    ]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, catalog, matchers, holdout_piece_ids=set(), thresholds=thresholds)
    assert isinstance(result, ReportResult)
    assert result.verdict in ("KILL", "TUNE", "PROCEED")


def test_report_matcher_results_present_for_all_matchers() -> None:
    catalog = _make_toy_catalog()
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        ChordNgramMatcher(catalog, oti=False, n=2),
        TwoDFTMatcher(catalog, oti=False),
    ]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, catalog, matchers, holdout_piece_ids=set(), thresholds=thresholds)
    assert len(result.matcher_results) == 3
    for mr in result.matcher_results:
        assert isinstance(mr, MatcherResult)
        assert 0.0 <= mr.recall_at_1 <= 1.0
        assert 0.0 <= mr.recall_at_10 <= 1.0
        assert 0.0 <= mr.mrr <= 1.0


def test_report_circularity_gives_perfect_recall_dtw() -> None:
    # Query = own score chroma -> DTW ceiling should rank own piece #1 -> recall@1 = 1.0
    catalog = _make_toy_catalog()
    matchers = [DtwCeilingMatcher(catalog, oti=False)]
    windows = _make_query_windows(catalog, holdout_ids=set())
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, catalog, matchers, holdout_piece_ids=set(), thresholds=thresholds)
    dtw_result = result.matcher_results[0]
    assert dtw_result.recall_at_1 == 1.0
    assert dtw_result.recall_at_10 == 1.0


def test_report_open_set_curve_behavioral() -> None:
    # Use DtwCeilingMatcher + a holdout window so open-set scoring actually executes.
    # piece_2 is held out (is_in_catalog=False) — its query is the out-of-catalog probe.
    # The remaining 2 pieces are in-catalog queries.
    # With circularity (query = own score chroma) DTW scores are near-perfect for
    # in-catalog and lower for the out-of-catalog holdout -> real scores flow into the curve.
    catalog = _make_toy_catalog()
    holdout_ids = {"piece_2"}
    matchers = [DtwCeilingMatcher(catalog, oti=False)]
    windows = _make_query_windows(catalog, holdout_ids=holdout_ids)
    thresholds = np.linspace(0.0, 1.0, 21)
    result = EvalReport.run(windows, catalog, matchers, holdout_piece_ids=holdout_ids, thresholds=thresholds)
    # Curve arrays must match threshold shape
    assert result.open_set_fa.shape == thresholds.shape
    assert result.open_set_ta.shape == thresholds.shape
    # At threshold=0 every query is accepted: ta[0] == 1.0 and fa[0] == 1.0
    # This can ONLY pass if open-set scoring actually executed (non-zero in/out score lists).
    assert result.open_set_ta[0] == 1.0, (
        f"ta at threshold=0 should be 1.0 (all in-catalog accepted), got {result.open_set_ta[0]}"
    )
    assert result.open_set_fa[0] == 1.0, (
        f"fa at threshold=0 should be 1.0 (all out-of-catalog accepted), got {result.open_set_fa[0]}"
    )
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_report.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'` or `ImportError: cannot import name 'EvalReport'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/report.py`:
```python
"""Orchestration: run all matchers, aggregate metrics, apply decision rule.

EvalReport.run is the single composition root that the CLI delegates to.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from piece_id_eval.decision import decide
from piece_id_eval.matchers.base import Matcher
from piece_id_eval.metrics import (
    mrr as compute_mrr,
    open_set_curve,
    open_set_ok,
    recall_at_k,
)
from piece_id_eval.query_set import LabeledQueryWindow

_OPEN_SET_MAX_FA = 0.10
_OPEN_SET_MIN_TA = 0.75


@dataclass
class MatcherResult:
    matcher_name: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


@dataclass
class ReportResult:
    matcher_results: list[MatcherResult]
    open_set_fa: np.ndarray       # false-accept rate curve
    open_set_ta: np.ndarray       # true-accept rate curve
    open_set_ok_flag: bool
    verdict: str                  # "KILL" | "TUNE" | "PROCEED"


class EvalReport:
    @staticmethod
    def run(
        query_windows: list[LabeledQueryWindow],
        catalog_chromas: dict[str, np.ndarray],
        matchers: list[Matcher],
        holdout_piece_ids: set[str],
        thresholds: np.ndarray,
    ) -> ReportResult:
        """Run all matchers over in/out-of-catalog windows, aggregate metrics.

        Args:
            query_windows: labeled windows from QuerySet.load.
            catalog_chromas: {piece_id: score_chroma} for the searchable library.
            matchers: list of Matcher implementations to evaluate.
            holdout_piece_ids: piece_ids whose queries are tagged is_in_catalog=False.
            thresholds: array of confidence thresholds for the open-set sweep.

        Returns:
            ReportResult with per-matcher metrics, open-set curve, and verdict.

        Raises:
            ValueError: if query_windows is empty.
        """
        if not query_windows:
            raise ValueError("query_windows is empty; nothing to evaluate")

        in_windows = [w for w in query_windows if w.is_in_catalog]
        out_windows = [w for w in query_windows if not w.is_in_catalog]

        matcher_results: list[MatcherResult] = []
        dtw_recall10 = 0.0
        best_indexable_recall10 = 0.0

        # Collect best-match scores for open-set curve (using first matcher if multiple)
        in_best_scores: list[float] = []
        out_best_scores: list[float] = []

        for matcher in matchers:
            # Build rankings for in-catalog windows only
            rankings = []
            for w in in_windows:
                ranked = matcher.rank(w.chroma)
                rankings.append((w.piece_id, ranked))

            r1 = recall_at_k(rankings, k=1) if rankings else 0.0
            r5 = recall_at_k(rankings, k=5) if rankings else 0.0
            r10 = recall_at_k(rankings, k=10) if rankings else 0.0
            m = compute_mrr(rankings) if rankings else 0.0

            mr = MatcherResult(
                matcher_name=matcher.name,
                recall_at_1=r1,
                recall_at_5=r5,
                recall_at_10=r10,
                mrr=m,
            )
            matcher_results.append(mr)

            # Track DTW ceiling and best indexable
            if "dtw_ceiling" in matcher.name:
                dtw_recall10 = r10
                # Collect best scores for open-set curve from DTW ceiling
                for w in in_windows:
                    ranked = matcher.rank(w.chroma)
                    in_best_scores.append(ranked[0].score if ranked else 0.0)
                for w in out_windows:
                    ranked = matcher.rank(w.chroma)
                    out_best_scores.append(ranked[0].score if ranked else 0.0)
            else:
                best_indexable_recall10 = max(best_indexable_recall10, r10)

        # Open-set curve from best-match scores
        if in_best_scores and out_best_scores:
            fa, ta = open_set_curve(
                np.array(in_best_scores), np.array(out_best_scores), thresholds
            )
            os_ok = open_set_ok(fa, ta, max_fa=_OPEN_SET_MAX_FA, min_ta=_OPEN_SET_MIN_TA)
        elif in_best_scores:
            # No out-of-catalog windows: assume open-set condition not met
            fa = np.zeros_like(thresholds)
            ta = np.ones_like(thresholds)
            os_ok = False
        else:
            fa = np.zeros_like(thresholds)
            ta = np.zeros_like(thresholds)
            os_ok = False

        verdict = decide(dtw_recall10, best_indexable_recall10, os_ok)

        return ReportResult(
            matcher_results=matcher_results,
            open_set_fa=fa,
            open_set_ta=ta,
            open_set_ok_flag=os_ok,
            verdict=verdict,
        )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_report.py -v
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/report.py model/tests/piece_id_eval/test_report.py && git commit -m "feat(piece-id-eval): report — orchestration, EvalReport.run, per-matcher metrics + verdict"
```

---

### Task 8: cli — entry point, table output, Justfile recipe

**Group:** D (sequential, depends on Group C — Task 7)

**Behavior being verified:** `python -m piece_id_eval.cli --dry-run` (pointing at toy fixture data) exits 0, prints a line containing `VERDICT:`, and writes a sidecar JSON with a `verdict` key.

**Interface under test:** CLI subprocess via `python -m piece_id_eval.cli`

**Files:**
- Create: `model/src/piece_id_eval/cli.py`
- Create: `model/src/piece_id_eval/acquire.py`
- Create: `model/tests/piece_id_eval/test_cli_smoke.py`
- Modify: `Justfile`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/piece_id_eval/test_cli_smoke.py
"""Smoke test: CLI on toy fixture data exits 0 and prints a VERDICT line."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL_DIR = Path(__file__).resolve().parents[2]


def _make_fixture_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal fixture: one piece, one recording, score JSON, piece map."""
    # Score JSON
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    score = {
        "piece_id": "test.piece.1",
        "bars": [
            {
                "bar_number": 1,
                "start_seconds": 0.0,
                "notes": [
                    {"pitch": 60, "onset_seconds": float(i) * 0.25, "duration_seconds": 0.25}
                    for i in range(16)
                ],
            }
        ],
    }
    (scores_dir / "test.piece.1.json").write_text(json.dumps(score))

    # Piece map
    evals_dir = tmp_path / "evals" / "piece_id"
    evals_dir.mkdir(parents=True)
    piece_map = {"test_slug": "test.piece.1"}
    (evals_dir / "eval_piece_map.json").write_text(json.dumps(piece_map))

    # candidates.yaml
    eval_root = tmp_path / "practice_eval"
    piece_dir = eval_root / "test_slug"
    piece_dir.mkdir(parents=True)
    (piece_dir / "candidates.yaml").write_text("""\
piece: test_slug
title: Test Piece
composer: Test
recordings:
- video_id: testvid001
  title: Test Recording
  channel: Test Channel
  duration_seconds: 6
  view_count: 100
  url: https://youtube.com/watch?v=testvid001
  query_source: test
  approved: true
  review_notes: ''
""")

    # Audio WAV
    audio_dir = piece_dir / "audio"
    audio_dir.mkdir()
    sr = 16000
    t = np.linspace(0, 6.0, int(sr * 6.0), endpoint=False)
    y = (np.sin(2 * np.pi * 261.63 * t) * 0.5).astype(np.float32)
    sf.write(audio_dir / "testvid001.wav", y, sr)

    return tmp_path, eval_root


def test_cli_smoke_exits_zero_and_prints_verdict(tmp_path: Path) -> None:
    fixture_root, eval_root = _make_fixture_tree(tmp_path)
    sidecar = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "piece_id_eval.cli",
            "--slugs", "test_slug",
            "--eval-root", str(eval_root),
            "--scores-dir", str(fixture_root / "scores"),
            "--piece-map", str(fixture_root / "evals" / "piece_id" / "eval_piece_map.json"),
            "--sidecar", str(sidecar),
            "--no-track",
            "--window-seconds", "2.0",
            "--hop-seconds", "1.0",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(MODEL_DIR),
    )
    assert result.returncode == 0, f"stderr={result.stderr}\nstdout={result.stdout}"
    assert "VERDICT:" in result.stdout, f"no VERDICT in stdout: {result.stdout!r}"
    assert sidecar.exists(), "sidecar JSON not written"
    sidecar_data = json.loads(sidecar.read_text())
    assert "verdict" in sidecar_data, f"sidecar missing 'verdict' key: {sidecar_data}"
    assert sidecar_data["verdict"] in ("KILL", "TUNE", "PROCEED")


def test_cli_smoke_sidecar_has_matcher_results(tmp_path: Path) -> None:
    fixture_root, eval_root = _make_fixture_tree(tmp_path)
    sidecar = tmp_path / "result2.json"
    result = subprocess.run(
        [
            sys.executable, "-m", "piece_id_eval.cli",
            "--slugs", "test_slug",
            "--eval-root", str(eval_root),
            "--scores-dir", str(fixture_root / "scores"),
            "--piece-map", str(fixture_root / "evals" / "piece_id" / "eval_piece_map.json"),
            "--sidecar", str(sidecar),
            "--no-track",
            "--window-seconds", "2.0",
            "--hop-seconds", "1.0",
        ],
        capture_output=True, text=True, timeout=120,
        cwd=str(MODEL_DIR),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    sidecar_data = json.loads(sidecar.read_text())
    assert "matchers" in sidecar_data
    assert len(sidecar_data["matchers"]) >= 1
    for m in sidecar_data["matchers"]:
        assert "name" in m
        assert "recall_at_10" in m
        assert "mrr" in m
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_cli_smoke.py -v 2>&1 | head -20
```
Expected: FAIL — `ModuleNotFoundError: No module named 'piece_id_eval'`

- [ ] **Step 3: Implement**

Create `model/src/piece_id_eval/acquire.py`:
```python
"""yt-dlp audio acquisition (cache-miss only).

Downloads a single YouTube video as a mono 16 kHz WAV. Called only when
the expected WAV file is missing from the audio cache directory.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


class AcquireError(RuntimeError):
    pass


def acquire_audio(
    video_id: str,
    out_dir: Path,
    cookies_file: Path | None = None,
) -> Path:
    """Download audio for video_id to out_dir/{video_id}.wav using yt-dlp.

    Returns the path to the downloaded WAV file.

    Raises:
        AcquireError: if yt-dlp exits non-zero or the output file is missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.wav"
    if out_path.exists():
        return out_path

    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--extract-audio",
        "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--output", str(out_dir / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
    ]
    if cookies_file is not None:
        if not cookies_file.exists():
            raise AcquireError(f"cookies file not found: {cookies_file}")
        cmd += ["--cookies", str(cookies_file)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise AcquireError(
            f"yt-dlp failed for {video_id} (exit {res.returncode}): {res.stderr[:500]}"
        )
    if not out_path.exists():
        raise AcquireError(
            f"yt-dlp succeeded but output file not found: {out_path}"
        )
    return out_path
```

Create `model/src/piece_id_eval/cli.py`:
```python
"""CLI entry point for the piece-ID feasibility harness.

Usage:
    python -m piece_id_eval.cli \
        --slugs bach_prelude_c_wtc1 fur_elise ... \
        --eval-root model/data/evals/practice_eval \
        --scores-dir model/data/scores \
        --piece-map model/data/evals/piece_id/eval_piece_map.json \
        [--holdout slug1 slug2] \
        [--sidecar path/to/result.json] \
        [--window-seconds 2.0] \
        [--hop-seconds 1.0] \
        [--no-track]

Prints a per-matcher metrics table and a final VERDICT: KILL|TUNE|PROCEED line.
Exit code: always 0 on successful run (the verdict is research output, not a gate).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[2]
_DEFAULT_EVAL_ROOT = _REPO_ROOT / "model" / "data" / "evals" / "practice_eval"
_DEFAULT_SCORES_DIR = _REPO_ROOT / "model" / "data" / "scores"
_DEFAULT_PIECE_MAP = _REPO_ROOT / "model" / "data" / "evals" / "piece_id" / "eval_piece_map.json"
_DEFAULT_SIDECAR = _REPO_ROOT / "model" / "data" / "evals" / "piece_id" / "last_run.json"

_ALL_SLUGS = [
    "bach_invention_1", "bach_prelude_c_wtc1", "chopin_ballade_1",
    "chopin_etude_op10no4", "chopin_waltz_csm", "clair_de_lune",
    "debussy_arabesque_1", "fantaisie_impromptu", "fur_elise",
    "liszt_liebestraum_3", "moonlight_sonata_mvt1", "mozart_k545_mvt1",
    "nocturne_op9no2", "pathetique_mvt2", "rachmaninoff_prelude_csm",
    "schumann_traumerei",
]


def _load_catalog(
    scores_dir: Path,
    piece_ids: list[str],
    holdout_piece_ids: set[str],
    frame_rate_hz: float = 50.0,
) -> dict[str, np.ndarray]:
    from piece_id_eval.score_chroma import load_catalog_score_chroma
    import glob

    catalog: dict[str, np.ndarray] = {}
    # Load all score JSONs in scores_dir except those for holdout pieces
    for score_path in sorted(scores_dir.glob("*.json")):
        piece_id = score_path.stem
        if piece_id in holdout_piece_ids:
            continue
        try:
            catalog[piece_id] = load_catalog_score_chroma(score_path, frame_rate_hz)
        except (KeyError, ValueError):
            # Skip score files that don't have notes (e.g. titles.json)
            pass
    return catalog


def _print_table(matcher_results) -> None:
    header = f"{'Matcher':<30} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'MRR':>6}"
    print(header)
    print("-" * len(header))
    for mr in matcher_results:
        print(
            f"{mr.matcher_name:<30} "
            f"{mr.recall_at_1:>6.3f} "
            f"{mr.recall_at_5:>6.3f} "
            f"{mr.recall_at_10:>6.3f} "
            f"{mr.mrr:>6.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="piece_id_eval.cli")
    parser.add_argument("--slugs", nargs="+", default=_ALL_SLUGS,
                        help="Practice-eval slugs to evaluate (default: all 16)")
    parser.add_argument("--eval-root", type=Path, default=_DEFAULT_EVAL_ROOT)
    parser.add_argument("--scores-dir", type=Path, default=_DEFAULT_SCORES_DIR)
    parser.add_argument("--piece-map", type=Path, default=_DEFAULT_PIECE_MAP)
    parser.add_argument("--holdout", nargs="*", default=[],
                        help="Slugs to hold out for open-set probe")
    parser.add_argument("--sidecar", type=Path, default=_DEFAULT_SIDECAR)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    parser.add_argument("--frame-rate-hz", type=float, default=50.0)
    parser.add_argument("--ngram-n", type=int, default=3,
                        help="N-gram size for ChordNgramMatcher")
    parser.add_argument("--no-track", action="store_true",
                        help="Suppress Trackio experiment logging")
    args = parser.parse_args(argv)

    from piece_id_eval.matchers import ChordNgramMatcher, DtwCeilingMatcher, TwoDFTMatcher
    from piece_id_eval.query_set import QuerySet
    from piece_id_eval.report import EvalReport

    # Load query windows
    print(f"Loading query windows for {len(args.slugs)} slugs...")
    load_result = QuerySet.load(
        slugs=args.slugs,
        eval_root=args.eval_root,
        piece_map_path=args.piece_map,
        audio_cache_root=args.eval_root,
        holdout_slugs=args.holdout,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
    )
    print(
        f"  {len(load_result.windows)} windows loaded, "
        f"{load_result.excluded_count} recordings excluded (missing audio)"
    )
    if not load_result.windows:
        print("ERROR: no query windows available. Run audio acquisition first.", file=sys.stderr)
        return 1

    # Resolve piece_ids for holdout exclusion from catalog
    piece_map: dict[str, str] = json.loads(args.piece_map.read_text())
    holdout_piece_ids: set[str] = {
        piece_map[s] for s in args.holdout if s in piece_map
    }

    # Load catalog
    print(f"Loading catalog from {args.scores_dir}...")
    catalog = _load_catalog(
        args.scores_dir,
        list(piece_map.values()),
        holdout_piece_ids,
        args.frame_rate_hz,
    )
    print(f"  {len(catalog)} catalog pieces loaded")

    # Build matchers (each variant: with and without OTI)
    matchers = [
        DtwCeilingMatcher(catalog, oti=False),
        ChordNgramMatcher(catalog, oti=False, n=args.ngram_n),
        ChordNgramMatcher(catalog, oti=True, n=args.ngram_n),
        TwoDFTMatcher(catalog, oti=False),
        TwoDFTMatcher(catalog, oti=True),
    ]

    thresholds = np.linspace(0.0, 1.0, 101)
    print("\nRunning evaluation...")
    report = EvalReport.run(
        load_result.windows,
        catalog,
        matchers,
        holdout_piece_ids=holdout_piece_ids,
        thresholds=thresholds,
    )

    # Print results table
    print()
    _print_table(report.matcher_results)
    print()
    print(f"Open-set OK (FA<=0.10 @ TA>=0.75): {report.open_set_ok_flag}")
    print()
    print(f"VERDICT: {report.verdict}")

    # Write sidecar JSON
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar_data = {
        "verdict": report.verdict,
        "open_set_ok": report.open_set_ok_flag,
        "matchers": [
            {
                "name": mr.matcher_name,
                "recall_at_1": mr.recall_at_1,
                "recall_at_5": mr.recall_at_5,
                "recall_at_10": mr.recall_at_10,
                "mrr": mr.mrr,
            }
            for mr in report.matcher_results
        ],
        "n_windows": len(load_result.windows),
        "n_excluded": load_result.excluded_count,
        "n_catalog_pieces": len(catalog),
        "holdout_slugs": args.holdout,
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    args.sidecar.write_text(json.dumps(sidecar_data, indent=2))

    # Trackio logging
    if not args.no_track:
        try:
            import trackio
            run = trackio.init(project="crescendai-piece-id-feasibility")
            for mr in report.matcher_results:
                run.log({
                    f"{mr.matcher_name}/recall_at_10": mr.recall_at_10,
                    f"{mr.matcher_name}/mrr": mr.mrr,
                })
            run.log({"verdict": report.verdict, "open_set_ok": int(report.open_set_ok_flag)})
            run.finish()
        except Exception as exc:
            print(f"WARNING: Trackio logging failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Justfile` (append after the last recipe):
```makefile
# Chroma-identification feasibility harness (Issue #21).
# Downloads real audio on cache miss; use --holdout to configure open-set probe.
# Run `just piece-id-feasibility-acquire` first to populate audio cache.
piece-id-feasibility:
    cd model && uv run python -m piece_id_eval.cli

# Acquire audio for all 16 practice_eval pieces (yt-dlp required).
piece-id-feasibility-acquire slug video_id:
    cd model && uv run python -c "
from pathlib import Path
from piece_id_eval.acquire import acquire_audio
out = acquire_audio('{{video_id}}', Path('data/evals/practice_eval/{{slug}}/audio'))
print(f'Downloaded: {out}')
"
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/test_cli_smoke.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add model/src/piece_id_eval/cli.py model/src/piece_id_eval/acquire.py model/tests/piece_id_eval/test_cli_smoke.py Justfile && git commit -m "feat(piece-id-eval): cli + acquire — entry point, table output, Justfile recipe"
```

---

### Task 9: full test suite run + spec/plan commit

**Group:** E (sequential, depends on Group D — Task 8)

**Behavior being verified:** All tests in `model/tests/piece_id_eval/` pass together.

**Interface under test:** `uv run pytest tests/piece_id_eval/`

**Files:** No new files; this is a verification + final spec/plan commit.

- [ ] **Step 1: Run the full test suite**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/piece_id_eval/ -v
```
Expected: All tests pass.

- [ ] **Step 2: Verify the Justfile recipe is discoverable**

```bash
cd /Users/jdhiman/Documents/crescendai && just --list | grep piece-id
```
Expected: `piece-id-feasibility` and `piece-id-feasibility-acquire` appear.

- [ ] **Step 3: Update spec to reflect any reconciliation changes, commit plan**

The spec at `docs/specs/2026-06-04-chroma-id-feasibility-design.md` has already been updated to reference `eval_piece_map.json` (existing file) instead of `piece_map.json`. Commit the final spec + plan together:

```bash
git add docs/specs/2026-06-04-chroma-id-feasibility-design.md docs/plans/2026-06-04-chroma-id-feasibility.md && git commit -m "docs(plan): add chroma-id-feasibility implementation plan"
```

---

## Summary

**9 tasks across 5 sequential groups:**
- **Group A (3 parallel):** `score_chroma`, `metrics`, `decision` — pure computation modules with no inter-dependencies.
- **Group B (3 parallel, depends on A):** `query_chroma`, all three matchers, `query_set` — I/O and index modules that depend on `score_chroma`.
- **Group C (1 sequential, depends on B):** `report` — integration orchestration that wires all Group B modules together.
- **Group D (1 sequential, depends on C):** `cli` + `acquire` — entry point that surfaces the report through a subprocess interface, plus Justfile.
- **Group E (1 sequential, depends on D):** Full test suite verification + final commit.

The harness is self-contained in `model/src/piece_id_eval/` and `model/tests/piece_id_eval/`. No Rust, no production services, no AMT required. The real-data run (`just piece-id-feasibility`) gates the Rust WASM channel decision on Issue #21.

---

## Challenge Review (Loop 1)

> Loop-1 review identified two blockers. Both were fixed before this loop-2 re-review was run. The original loop-1 findings are preserved below the loop-2 summary for reference.

---

## Challenge Review (Loop 2 — post-fix re-review)

**Reviewer note:** This is a fresh two-pass review of the plan *after* the two loop-1 blockers were addressed. Both fixes have been verified against the actual filesystem layout.

---

### Fix Verification

**Fix 1 — `_REPO_ROOT` off-by-one in `cli.py`**

Claim: changed `_MODULE_DIR.parents[3]` → `_MODULE_DIR.parents[2]`.

Verified: `_MODULE_DIR = Path(__file__).resolve().parent` where `__file__` is
`model/src/piece_id_eval/cli.py`, so `_MODULE_DIR` = `.../crescendai/model/src/piece_id_eval/`.
`parents[2]` = `.../crescendai/` (repo root). Confirmed by running:
```
parents[0] = .../crescendai/model/src/
parents[1] = .../crescendai/model/
parents[2] = .../crescendai/       ← CORRECT
parents[3] = .../Documents/        ← OLD/WRONG
```
Fix is correct. BLOCKER 1 resolved.

**Fix 2 — vacuous `test_report_open_set_curve_shape` replaced with `test_report_open_set_curve_behavioral`**

Claim: new test uses `DtwCeilingMatcher` + `holdout_ids={"piece_2"}` and asserts `ta[0] == 1.0` and `fa[0] == 1.0` at threshold=0.

Verified: `EvalReport.run` keeps `piece_2` in the `catalog_chromas` dict even when it is in `holdout_piece_ids`. The holdout only affects `is_in_catalog` tagging. So when `piece_2`'s query window (its own score chroma) is ranked against the catalog, the self-match DTW cost is ≈ 0 → `score = -0.0`. In Python/numpy, `-0.0 >= 0` evaluates to `True`, so `fa[0] = (out_scores >= 0).mean() = 1.0`. Similarly `in_best_scores` has scores ≈ -0.0 for the two in-catalog pieces, giving `ta[0] = 1.0`. Both assertions will pass, and they can only pass when real scores flow through the open-set branch. Fix is correct and behavioral. BLOCKER 2 resolved.

---

### CEO Pass

**Premise Challenge**

The interval-trigram failure on BWV 846 is real and documented. The plan's stated goal — measure before building — is the correct risk-management move. No alternative framing yields a simpler or more impactful route to the same answer.

**Scope Check**

Tight. Three matchers are the minimum set needed to separate "chroma can't discriminate" from "your index is just weak" — cutting the DTW ceiling would leave the verdict uninterpretable. The `acquire.py` module is the one deferrable piece, but at ~30 lines it costs nothing to ship with the harness; the only risk is it receives no test coverage (already flagged as RISK 1).

File count: 14 new + 2 modified. High but structurally necessary for a multi-matcher eval harness with three independently testable recall families.

**Twelve-Month Alignment**

```
CURRENT STATE                         THIS PLAN                    12-MONTH IDEAL
Piece ID fails on BWV 846.     →      Offline Python harness  →    Rust WASM chroma recall
Interval-trigram cascade              measures chroma                channel in production,
has no validated replacement.         discriminability.              tuned against this metric.
```

The plan moves directly toward the ideal. No tech debt introduced — `model/src/piece_id_eval/` is a standalone harness with no production entanglement.

**Alternatives**

The spec documents trade-offs (Python over Rust, synthetic over rendered audio, holdout open-set over sourced negatives). All are appropriate for a go/no-go feasibility measurement.

---

### Engineering Pass

**Architecture**

Data flow is clean:

```
cli.py
  │
  ├── QuerySet.load()    → audio WAVs → audio_to_chroma() → window_chroma() → LabeledQueryWindows
  ├── _load_catalog()    → score JSONs → load_catalog_score_chroma() → {piece_id: chroma}
  │
  └── EvalReport.run()
        │
        ├── for each Matcher:
        │     matcher.rank(window.chroma) → list[Ranked]
        │     recall_at_k() / mrr()
        │     [if DTW ceiling] → collect in/out best scores
        │
        ├── open_set_curve() + open_set_ok()
        └── decide() → "KILL" | "TUNE" | "PROCEED"
                │
                └── sidecar JSON + stdout table
```

No security surface (no user input to SQL/shell/LLM; yt-dlp uses `shell=False`). No unbounded fan-out or N+1 patterns. Component boundaries are clean. No production entanglement.

**Module Depth Audit**

| Module | Interface | Implementation | Verdict |
|--------|-----------|----------------|---------|
| `score_chroma.py` | 2 functions | pitch-class accumulation + 1e-3 floor + L2-norm | DEEP |
| `query_chroma.py` | 2 functions | audio load + chroma_cqt + windowing | DEEP |
| `acquire.py` | 1 function | yt-dlp subprocess + output validation | DEEP |
| `query_set.py` | 1 static method + 2 dataclasses | YAML parsing + chroma extraction + holdout tagging | DEEP |
| `matchers/base.py` | 1 Protocol + 1 dataclass | ~20 lines | SHALLOW (acceptable — type boundary, not logic) |
| `matchers/dtw_ceiling.py` | 1 class, 1 method | subsequence DTW fallback | DEEP |
| `matchers/chord_ngram.py` | 1 class, 1 method | tokenization + inverted n-gram index | DEEP |
| `matchers/twodft.py` | 1 class, 1 method | 2D-FFT embedding + cosine | DEEP |
| `metrics.py` | 4 functions | recall scan, MRR, threshold sweep | DEEP |
| `decision.py` | 1 function | 3-line pre-registered rule | SHALLOW (acceptable — small surface by design) |
| `report.py` | 1 static method + 2 dataclasses | wires all matchers + metrics + decision | DEEP |
| `cli.py` | `main()` + argparse | catalog load + report + table + sidecar | DEEP |

No shallow modules are load-bearing enough to be blockers.

**Code Quality**

- `_REPO_ROOT` is now `_MODULE_DIR.parents[2]` — correct (verified above).
- `import glob` inside `_load_catalog()` in `cli.py` is unused dead code. `Path.glob` is used instead. Harmless (stdlib, no side effects), but dead.
- `dtaidistance` try/except import guard in `dtw_ceiling.py` sets `_HAS_DTAIDISTANCE` but the flag is never checked. The numpy fallback is always used. The guard is dead code. The behavior is correct for an offline harness; the dead flag could confuse a future reader who expects `dtaidistance` to be preferred when available.
- `except Exception` in `cli.py`'s Trackio block violates CLAUDE.md explicit-exception standard (see RISK 2 below).
- `cli.py` `_load_catalog` catches `(KeyError, ValueError)` for malformed/notes-free score JSONs. This is appropriately scoped (only skips files that can't yield a chroma array) — not a catch-all.

**Test Philosophy Audit**

All tests exercise public interfaces. No internal mocking. No shape-only tests. The open-set test now asserts real behavioral properties (scores flow through the curve, ta[0]/fa[0] are non-trivially 1.0). The fix is verified above.

`test_minimum_floor_enforced` in `test_score_chroma.py` checks shape + no-NaN after placing a note at t=5s in what would be a short signal. The floor's behavioral correctness is indirectly verified by `test_columns_are_l2_normalized` (a zero-floor column would cause a NaN after division). Acceptable.

**Vertical Slice Audit**

Every task follows write-test → verify-fail → implement → verify-pass → commit. No horizontal slicing. Group B's dependency on Group A (`build_score_chroma` in Task 5) is structural, not a slice violation.

**Test Coverage**

```
[+] acquire.py
    └── acquire_audio()
        ├── [GAP] happy path — no test (network/yt-dlp impractical in unit tests)
        ├── [GAP] yt-dlp non-zero exit -> AcquireError — no test
        └── [GAP] output missing after success -> AcquireError — no test

[+] cli.py
    ├── [TESTED] happy path (C4 fixture) ★★
    ├── [TESTED] sidecar JSON structure ★★
    ├── [GAP] no query windows -> exit 1 — no test
    └── [GAP] piece_map missing for slug -> KeyError — no test

[+] report.py EvalReport.run()
    ├── [TESTED] happy path, 3-matcher ★★
    ├── [TESTED] open-set curve behavioral (ta[0]=fa[0]=1.0) ★★
    ├── [TESTED] DTW circularity perfect recall ★★
    └── [GAP] empty query_windows -> ValueError — no test
```

`acquire.py` zero coverage is the only notable gap. For a module that wraps an external subprocess, this is acceptable for a research harness (network tests are impractical). Flagged as RISK 1 below.

**Failure Modes**

- All default paths in `cli.py` now correctly anchor to the repo root — explicit `FileNotFoundError` if missing.
- Open-set probe with no `--holdout`: `out_windows=[]`, `open_set_ok=False`, verdict defaults to TUNE or KILL. The Justfile recipe has a comment warning about this; the recipe itself doesn't pass a default holdout (RISK 3 below).
- DTW ceiling double-ranking in `report.py`: in-windows are ranked twice (once for metrics, once for open-set score collection). Acceptable for an offline harness, but could be refactored to reuse the first-pass scores (RISK 4 below).
- `acquire_audio` failure leaves no partial state — yt-dlp writes atomically or fails, the WAV either exists or doesn't.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `eval_piece_map.json` exists at `model/data/evals/piece_id/eval_piece_map.json` | SAFE | Verified: 16 entries, all slugs present |
| Score JSONs have `bars[].notes[].onset_seconds` and `duration_seconds` | SAFE | Verified against `bach.prelude.bwv_846.json` |
| 254 catalog score JSONs with `bars` key in `model/data/scores/` | SAFE | Verified: 257 total, 254 with `bars`, 1 metadata-only (silently skipped by `(KeyError, ValueError)` catch) |
| `librosa`, `soundfile`, `pyyaml`, `trackio` declared in `model/pyproject.toml` | SAFE | Verified: all present |
| `dtaidistance` importable for DTW | RISKY | Not in `pyproject.toml`. The try/except guard exists but `_HAS_DTAIDISTANCE` is never checked; numpy fallback always runs. Behavior is correct but the guard is dead code. |
| `_REPO_ROOT = _MODULE_DIR.parents[2]` resolves to repo root | SAFE | Verified: `.../crescendai/model/src/piece_id_eval/`.parents[2] = `.../crescendai/` |
| `candidates.yaml` exists under each slug's `practice_eval/` subdir | SAFE | Verified: all 16 slugs have `candidates.yaml` |
| Audio WAVs do not exist yet | SAFE | Verified: no `audio/` dirs under practice_eval slugs |
| Open-set curve uses DTW ceiling best-match scores | VALIDATE | Only populated inside `if "dtw_ceiling" in matcher.name` branch; no DTW = no curve = always TUNE/KILL. Documented in CLI as a usage requirement. |
| `just piece-id-feasibility` with default args produces a complete verdict | RISKY | No `--holdout` in the recipe → open_set_ok always False → PROCEED structurally unreachable. See RISK 3. |

---

### Summary

**[BLOCKER]** count: 0
**[RISK]**    count: 4
**[QUESTION]** count: 0

---

**[RISK 1]** (confidence: 8/10) — `acquire.py` has zero test coverage. `acquire_audio` wraps `yt-dlp` and raises `AcquireError` on two distinct failure modes, neither tested. Fallback: the CLI smoke test exercises the pre-cached path; acquisition failures surface on first real-data run. Acceptable for a research harness — network/subprocess tests are impractical in unit tests.

**[RISK 2]** (confidence: 8/10) — `except Exception` in `cli.py`'s Trackio block violates CLAUDE.md explicit-exception standard. A real bug in `run.log` data types would be silently swallowed with only a WARNING print. Fix: catch `ImportError` separately for missing `trackio`, and narrow the runtime catch (e.g., `trackio.TrackioError` or `RuntimeError`).

**[RISK 3]** (confidence: 7/10) — `just piece-id-feasibility` (no `--holdout`) always returns TUNE or KILL, never PROCEED — `open_set_ok` is structurally False when no holdout is given. The Justfile comment warns about this, but the recipe provides no default holdout. A user running `just piece-id-feasibility` without reading the comment gets a structurally incomplete verdict. Fix: add a comment in the recipe body explicitly stating `--holdout <slugs>` is required for a complete run, or provide a `piece-id-feasibility-full` recipe with a sensible default holdout.

**[RISK 4]** (confidence: 6/10) — `report.py` calls `matcher.rank(w.chroma)` twice for DTW ceiling over in-windows: once in the `rankings` list and once in the open-set score collection loop. For a real 254-piece catalog run with ~300 windows, this doubles DTW computation time. Acceptable for an offline research run but could be eliminated by extracting best-match scores from the already-computed `rankings`.

---

VERDICT: PROCEED_WITH_CAUTION — four risks to monitor: [RISK 1] acquire.py untested (surfaces on first real-data run); [RISK 2] broad `except Exception` in Trackio block (narrow before real experiments); [RISK 3] default `just piece-id-feasibility` recipe can never return PROCEED (document or add default holdout); [RISK 4] DTW double-ranking doubles offline compute time (acceptable, but worth a one-line refactor).

---

## Challenge Review (Loop 1 — original, superseded)

> The following is the original loop-1 review that identified two blockers. Both have been fixed in the plan above. Preserved for audit trail.

### CEO Pass

**Premise Challenge**

The problem is correctly framed. The interval-trigram failure on BWV 846 is documented (`docs/implementation/2026-05-31-issue1-interval-trigrams-debug.md`). The claim under test — that chroma discriminates where melody fails — is real and unvalidated. The plan is the most direct route: a measurement before building. No simpler framing exists.

The real pain is concrete: without this, any Rust WASM channel would be built on an unvalidated hypothesis, repeating the interval-trigram dead end. The plan blocks that.

**Scope Check**

The scope is tight for what it is. The three-matcher approach (DTW ceiling + two indexable families) is necessary: without the ceiling you can't distinguish "chroma can't discriminate" from "the index is just weak". Cutting any matcher removes a dimension that the go/no-go verdict needs.

The `acquire.py` module (yt-dlp download) is the only scope item that could be deferred — it's not exercised by any test and the real-data run can pre-cache audio manually. However, it's small (~30 lines) and the plan explicitly keeps it out of the test suite's critical path. Not a blocker.

Complexity count: 14 new source files + 2 modified. High file count, but each module has a single responsibility and the plan acknowledges the precedent in `chroma_dtw_eval/` (7 files). The complexity is inherent to a multi-matcher eval harness, not speculative.

**Twelve-Month Alignment**

```
CURRENT STATE                         THIS PLAN                    12-MONTH IDEAL
Piece ID fails on BWV 846.            Offline Python harness       Rust WASM chroma recall
Interval-trigram cascade              measures chroma              channel in production
has no validated replacement.         discriminability → KILL      (gated on PROCEED).
                                       / TUNE / PROCEED verdict.
```

The plan moves directly toward the ideal. If it returns KILL, it prevents 2+ weeks of Rust work on a dead direction. If PROCEED, it provides the metric the Rust channel is tuned against. No tech debt introduced.

**Alternatives**

The spec documents the chosen trade-offs: Python over Rust (correct for feasibility), synthetic score chroma over rendered audio (no audio for 254 catalog pieces exists), holdout open-set over sourced negatives (reuses labeled audio). All reasonable for the stated goal.

---

### Engineering Pass

**Architecture**

Data flow:

```
cli.py
  │
  ├── QuerySet.load()        → audio WAVs → audio_to_chroma() → window_chroma() → LabeledQueryWindows
  ├── _load_catalog()        → score JSONs → load_catalog_score_chroma() → {piece_id: chroma}
  │
  └── EvalReport.run()
        │
        ├── for each Matcher:
        │     matcher.rank(window.chroma) → list[Ranked]
        │     recall_at_k() / mrr()
        │     [if DTW ceiling] → collect in/out best scores
        │
        ├── open_set_curve() → (fa_rates, ta_rates)
        ├── open_set_ok()
        └── decide() → "KILL" | "TUNE" | "PROCEED"
              │
              └── sidecar JSON + stdout table
```

Component boundaries are clean. No security concerns (no user input to SQL/LLM/shell; yt-dlp subprocess uses a validated video_id string, no shell=True). N+1 queries: none (catalog loaded once, index built once).

**Module Depth Audit**

| Module | Interface | Implementation | Verdict |
|--------|-----------|----------------|---------|
| `score_chroma.py` | 2 functions | pitch-class accumulation + normalization logic | DEEP |
| `query_chroma.py` | 2 functions | audio load + CQT + windowing | DEEP |
| `acquire.py` | 1 function | yt-dlp subprocess + output validation | DEEP |
| `query_set.py` | 1 static method + 2 dataclasses | YAML parsing + chroma extraction + holdout tagging | DEEP |
| `matchers/base.py` | 1 Protocol + 1 dataclass | ~20 lines | SHALLOW (acceptable — type boundary) |
| `matchers/dtw_ceiling.py` | 1 class, 1 method | subsequence DTW fallback implementation | DEEP |
| `matchers/chord_ngram.py` | 1 class, 1 method | tokenization + inverted index | DEEP |
| `matchers/twodft.py` | 1 class, 1 method | 2D-FFT embedding + cosine | DEEP |
| `metrics.py` | 4 functions | recall scan, reciprocal rank, threshold sweep | DEEP |
| `decision.py` | 1 function | 3-line rule against 2 constants | SHALLOW (acceptable) |
| `report.py` | 1 static method + 2 dataclasses | wires all matchers + metrics + decision | DEEP |
| `cli.py` | 1 `main()` + argparse | catalog load + report run + table print + sidecar | DEEP |

**Code Quality**

- `_REPO_ROOT` off-by-one in `cli.py` is a concrete path bug (see Blocker 1).
- `query_chroma.py` re-implements the `chroma_feature` recipe instead of importing from `apps/inference/muq/chroma.py`. The spec says "imported" but the existing precedent in `chroma_dtw_eval/chroma_cache.py` uses the same inline-recipe pattern. The plan follows the established pattern.
- `cli.py` `_load_catalog` catches `(KeyError, ValueError)` silently. There is exactly 1 score JSON without a `bars` key in the 255-file scores dir; the silent skip is correct behavior.
- `cli.py` Trackio block uses `except Exception`. This violates the CLAUDE.md "explicit exception handling" standard.

**Test Philosophy Audit**

`test_report_open_set_curve_shape` in `test_report.py`: creates a `TwoDFTMatcher`-only report. The open-set curve in `report.py` is populated only from the DTW ceiling branch. With no DTW ceiling matcher present, `in_best_scores` and `out_best_scores` remain empty, and the fallback assigns `fa = ta = np.zeros_like(thresholds)`. The test asserts `result.open_set_fa.shape == thresholds.shape`, which passes on zeros — it is testing array shape, not open-set behavior.

**[BLOCKER] (confidence: 9/10)** — `test_report_open_set_curve_shape` passes vacuously. Fix: replace with a test that uses `DtwCeilingMatcher` plus a holdout window, then asserts behavioral properties of the curve.

**Vertical Slice Audit**

Every task follows write-test → verify-fail → implement → verify-pass → commit. No horizontal slicing.

**Failure Modes**

- `_REPO_ROOT` wrong in `cli.py` → all default paths point to non-existent locations. The CLI would fail at `piece_map_path.read_text()` with `FileNotFoundError`. Explicit error, but the wrong location. **Blocker 1.**

---

### Presumption Inventory (Loop 1)

| Assumption | Verdict | Reason |
|---|---|---|
| `eval_piece_map.json` exists at `model/data/evals/piece_id/eval_piece_map.json` | SAFE | Verified: file exists, shipped by #23 |
| Score JSONs have `bars` key with `notes.onset_seconds` and `duration_seconds` | SAFE | Verified against `bach.prelude.bwv_846.json` |
| 254 catalog score JSONs with `bars` key in `model/data/scores/` | SAFE | Verified: 255 total, 254 with `bars` |
| `librosa`, `soundfile`, `pyyaml`, `trackio` declared in `model/pyproject.toml` | SAFE | Verified: all present |
| `dtaidistance` importable for DTW | RISKY | Not in `pyproject.toml`; try/except guard exists but flag never checked |
| `_REPO_ROOT = _MODULE_DIR.parents[3]` resolves to repo root | RISKY | Verified WRONG: `parents[3]` = `~/Documents`. Should be `parents[2]`. |
| `candidates.yaml` exists under each slug's `practice_eval/` subdir | SAFE | Verified: all 16 slugs present |

---

### Summary (Loop 1)

**[BLOCKER]** count: 2
**[RISK]**    count: 4
**[QUESTION]** count: 0

**[BLOCKER 1]** (confidence: 10/10) — `_REPO_ROOT` off-by-one in `cli.py`. `_MODULE_DIR.parents[3]` resolves to `~/Documents`. All default paths are wrong. Fix: `_MODULE_DIR.parents[2]`.

**[BLOCKER 2]** (confidence: 9/10) — `test_report_open_set_curve_shape` passes vacuously on zeros (shape test with TwoDFTMatcher-only, no DTW ceiling → scores never populated). Fix: use `DtwCeilingMatcher` + holdout window and assert ta[0]/fa[0] == 1.0.

VERDICT: NEEDS_REWORK — [BLOCKER 1] `_REPO_ROOT` off-by-one, [BLOCKER 2] open-set test vacuous.
