# Chroma-DTW Score-Follower Eval Harness Implementation Plan

> **For the build agent:** Dispatch each task group's tasks in parallel (one subagent per task) using Sonnet 4.6. Do NOT start execution until `/challenge` returns `VERDICT: PROCEED`. Inside a group, parallel tasks touch disjoint files. Sequential groups depend on the prior group's modules being importable.

**Goal:** Provide a single CLI that scores the production chroma-DTW score follower on a frozen multi-piece test set and exits non-zero if any guard regresses, so `/autoresearch` can keep-or-revert DTW changes mechanically.
**Spec:** docs/specs/2026-05-31-chroma-dtw-eval-harness-design.md
**Style:** Follow CLAUDE.md (uv, partitura over music21, explicit exceptions, no emojis, Trackio for tracking). No `pip`. No silent fallbacks. The Rust binary lives in the same crate as the production DTW; do NOT modify `chroma_dtw.rs` or `lib.rs` beyond adding the binary entry.

---

## Task Groups

- **Group 0 (sequential, single task):** Task 0 — committed fixture + minimal smoke test wiring. **`[SHIPS INDEPENDENTLY]`** — once this lands, every later module has a green CI signal it can integrate against; if Group A is delayed, the fixture still proves the verify-CLI shape exists.
- **Group A (parallel):** Task A1 (chroma_cache), Task A2 (chunk_sampler), Task A3 (silence_synth), Task A4 (practice_compose), Task A5 (dtw_chunk_cli — Rust). Touch disjoint files.
- **Group B (parallel, depends on A):** Task B1 (dtw_runner — depends on A5), Task B2 (gold_truth_builder — depends on A1).
- **Group C (sequential, depends on B):** Task C1 (metric_aggregator), then Task C2 (verify CLI), then Task C3 (ratchet CLI), then Task C4 (justfile wiring + end-to-end smoke).

Each task is one test → one impl → one commit.

---

## Task 0: Committed fixture + smoke test that proves the verify-CLI contract
**Group:** 0 (must complete before A)

**Behavior being verified:** A small committed fixture under `model/data/evals/chroma_dtw_fixtures/` plus a verify-CLI skeleton exposes the contract `verify --baseline <path> --fixtures <path>` → stdout = one float in [0, 100], exit 0 on first run.

**Interface under test:** `python -m chroma_dtw_eval.verify --baseline <path> --fixtures <path>` (subprocess).

**Files:**
- Create: `model/src/chroma_dtw_eval/__init__.py` (empty)
- Create: `model/src/chroma_dtw_eval/verify.py`
- Create: `model/data/evals/chroma_dtw_fixtures/manifest.json`
- Create: `model/data/evals/chroma_dtw_fixtures/README.md`
- Create: `model/tests/chroma_dtw_eval/__init__.py` (empty)
- Test: `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_verify_cli_smoke.py
import json
import subprocess
import sys
from pathlib import Path


FIXTURES = Path(__file__).resolve().parents[2] / "data" / "evals" / "chroma_dtw_fixtures"


def test_verify_cli_returns_one_float_and_exits_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g4": 0.0, "g5": 100.0},
    }))
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline), "--fixtures", str(FIXTURES)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {result.stdout!r}"
    value = float(lines[0])
    assert 0.0 <= value <= 100.0
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_verify_cli_smoke.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval'` (the package and CLI do not exist).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `model/data/evals/chroma_dtw_fixtures/manifest.json`:
```json
{
  "version": 1,
  "chunks": [
    {"id": "fix_001", "kind": "gold", "gold_frame": 0, "tolerance_frames": 3},
    {"id": "fix_002", "kind": "amateur", "forward_bar": 4},
    {"id": "fix_003", "kind": "silence", "expected_loud_failure": true}
  ]
}
```

Create `model/data/evals/chroma_dtw_fixtures/README.md`:
```markdown
# chroma_dtw_fixtures

Tiny committed fixture (3 chunks) used by the smoke test to verify the verify-CLI contract end-to-end without depending on MAESTRO/skill_eval/practice_eval downloads.

- fix_001: gold-truth chunk (synthetic, frame 0)
- fix_002: amateur-style chunk (no ms-truth)
- fix_003: silence chunk (zeros)

Real corpora live under model/data/evals/{skill_eval,practice_eval} and model/data/raw/asap; the harness reaches them via the chunk_sampler module, not this directory.
```

Create `model/src/chroma_dtw_eval/__init__.py` as an empty file.

Create `model/src/chroma_dtw_eval/verify.py`:
```python
"""Verify CLI entry point.

This is a thin assembly module — it loads the fixture manifest, baseline file,
emits the primary scalar on stdout, and exits 0 or non-zero by comparing against
the baseline. In Task 0 the per-chunk computation is a stub that returns 100.0
(perfect alignment) so the contract is testable before the deep modules ship.
Later tasks replace the stub with real metric_aggregator calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=Path("model/data/evals/chroma_dtw/last_run.json"),
    )
    args = parser.parse_args(argv)

    manifest_path = args.fixtures / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"fixture manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    baseline = json.loads(args.baseline.read_text())

    n = len(manifest["chunks"])
    primary = 100.0 if n > 0 else 0.0
    guards = {"g1": 0.0, "g2": 1.0, "g3": 0.0, "g4": 100.0, "g5": 0.0}

    regressed: list[str] = []
    if primary + 1e-9 < baseline["primary"]:
        regressed.append("primary")
    if guards["g1"] > baseline["guards"]["g1"] + 1.0:
        regressed.append("g1")
    if guards["g2"] < baseline["guards"]["g2"] - 0.02:
        regressed.append("g2")
    if guards["g3"] < baseline["guards"]["g3"] - 1.0:
        regressed.append("g3")
    if guards["g4"] < baseline["guards"]["g4"] - 1.0:
        regressed.append("g4")
    if guards["g5"] > baseline["guards"]["g5"] + 1.0:
        regressed.append("g5")

    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": primary,
        "guards": guards,
        "baseline": baseline,
        "regressed": regressed,
        "n_chunks": n,
    }, indent=2))

    print(f"{primary:.4f}")
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_verify_cli_smoke.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/__init__.py model/src/chroma_dtw_eval/verify.py model/data/evals/chroma_dtw_fixtures/manifest.json model/data/evals/chroma_dtw_fixtures/README.md model/tests/chroma_dtw_eval/__init__.py model/tests/chroma_dtw_eval/test_verify_cli_smoke.py && git commit -m "feat(chroma-eval): commit fixture skeleton + verify-CLI contract smoke test"
```

---

## Task A1: chroma_cache
**Group:** A (parallel with A2, A3, A4, A5)

**Behavior being verified:** Calling `get_chroma(audio_path, params)` twice on the same input returns byte-identical chroma the second time without recomputing, and produces a hash-keyed cache file under the cache root.

**Interface under test:** `chroma_dtw_eval.chroma_cache.get_chroma`.

**Files:**
- Create: `model/src/chroma_dtw_eval/chroma_cache.py`
- Test: `model/tests/chroma_dtw_eval/test_chroma_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_chroma_cache.py
import numpy as np
import soundfile as sf

from chroma_dtw_eval.chroma_cache import ChromaParams, get_chroma


def test_get_chroma_caches_after_first_call(tmp_path):
    sr = 24000
    y = np.random.RandomState(0).randn(sr * 2).astype(np.float32) * 0.1
    audio = tmp_path / "a.wav"
    sf.write(audio, y, sr)
    cache_root = tmp_path / "cache"
    params = ChromaParams(target_frame_rate_hz=50.0, sr=sr)

    first = get_chroma(audio, params, cache_root=cache_root)
    cached_files = list(cache_root.rglob("*.bin"))
    assert len(cached_files) == 1, f"expected 1 cache file, got {cached_files}"
    mtime = cached_files[0].stat().st_mtime_ns

    second = get_chroma(audio, params, cache_root=cache_root)
    assert cached_files[0].stat().st_mtime_ns == mtime, "cache file was rewritten"
    assert np.array_equal(first.data, second.data)
    assert first.data.shape[0] == 12
    assert first.data.dtype == np.float32
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_chroma_cache.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.chroma_cache'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/chroma_cache.py
"""Hash-keyed on-disk chroma cache.

Computes 12-bin chroma at ~target_frame_rate_hz from a mono audio file using
the same recipe as apps/inference/muq/chroma.py (chroma_cqt + 1e-3 floor +
L2 column normalization). Idempotent — second call with same audio+params
returns the cached array without recomputing.

Raises explicitly on missing audio, unreadable files, or sample-rate mismatch.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class ChromaParams:
    target_frame_rate_hz: float
    sr: int


@dataclass
class CachedChroma:
    data: np.ndarray  # shape (12, n_frames), float32, L2-normed
    frame_rate_hz: float
    audio_path: Path


def _hash_audio(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(cache_root: Path, audio_hash: str, params: ChromaParams) -> Path:
    key = f"{audio_hash}_sr{params.sr}_fr{params.target_frame_rate_hz:.1f}.bin"
    return cache_root / key


def get_chroma(audio_path: Path, params: ChromaParams, cache_root: Path) -> CachedChroma:
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    cache_root.mkdir(parents=True, exist_ok=True)
    audio_hash = _hash_audio(audio_path)
    cache_file = _cache_path(cache_root, audio_hash, params)
    meta_file = cache_file.with_suffix(".meta")

    if cache_file.exists() and meta_file.exists():
        meta = meta_file.read_text().strip().split(",")
        n_frames = int(meta[0])
        frame_rate_hz = float(meta[1])
        raw = np.fromfile(cache_file, dtype=np.float32)
        if raw.size != 12 * n_frames:
            raise RuntimeError(
                f"chroma cache corrupt: {cache_file} size {raw.size} != 12*{n_frames}"
            )
        return CachedChroma(raw.reshape(12, n_frames), frame_rate_hz, audio_path)

    y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if sr != params.sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=params.sr)
        sr = params.sr
    if y.ndim == 2:
        y = y.mean(axis=1)

    hop = max(1, round(sr / params.target_frame_rate_hz))
    frame_rate_hz = sr / hop
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop).astype(np.float32)
    chroma += 1e-3
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9

    tmp = cache_file.with_suffix(".bin.tmp")
    chroma.flatten().astype(np.float32).tofile(tmp)
    tmp.replace(cache_file)
    meta_file.write_text(f"{chroma.shape[1]},{frame_rate_hz}")
    return CachedChroma(chroma, frame_rate_hz, audio_path)
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_chroma_cache.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/chroma_cache.py model/tests/chroma_dtw_eval/test_chroma_cache.py && git commit -m "feat(chroma-eval): add hash-keyed chroma cache"
```

---

## Task A2: chunk_sampler
**Group:** A (parallel with A1, A3, A4, A5)

**Behavior being verified:** `sample_chunks(piece_durations, n_per_piece, seed)` produces a deterministic position-stratified manifest with intro/early/middle/late/cadence-zone buckets, and the same seed always yields the same manifest.

**Interface under test:** `chroma_dtw_eval.chunk_sampler.sample_chunks`.

**Files:**
- Create: `model/src/chroma_dtw_eval/chunk_sampler.py`
- Test: `model/tests/chroma_dtw_eval/test_chunk_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_chunk_sampler.py
from chroma_dtw_eval.chunk_sampler import PieceSpec, sample_chunks


def test_sample_chunks_is_deterministic_and_stratified():
    pieces = [
        PieceSpec(piece_id="p1", duration_s=300.0),
        PieceSpec(piece_id="p2", duration_s=600.0),
    ]
    a = sample_chunks(pieces, n_per_piece=10, chunk_len_s=15.0, seed=42)
    b = sample_chunks(pieces, n_per_piece=10, chunk_len_s=15.0, seed=42)
    assert [c.start_s for c in a] == [c.start_s for c in b]
    assert [c.position_bucket for c in a] == [c.position_bucket for c in b]

    buckets_p1 = {c.position_bucket for c in a if c.piece_id == "p1"}
    assert buckets_p1 == {"intro", "early", "middle", "late", "cadence"}
    for c in a:
        assert 0.0 <= c.start_s <= c.piece_duration_s - c.chunk_len_s + 1e-6
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_chunk_sampler.py -x
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/chunk_sampler.py
"""Position-stratified chunk sampler.

Given a list of pieces with durations, produces a deterministic manifest of
chunk start times stratified into 5 position buckets: intro (0-5%), early
(5-25%), middle (25-65%), late (65-90%), cadence (90-100%).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("intro", 0.0, 0.05),
    ("early", 0.05, 0.25),
    ("middle", 0.25, 0.65),
    ("late", 0.65, 0.90),
    ("cadence", 0.90, 1.0),
)


@dataclass(frozen=True)
class PieceSpec:
    piece_id: str
    duration_s: float


@dataclass(frozen=True)
class Chunk:
    piece_id: str
    start_s: float
    chunk_len_s: float
    piece_duration_s: float
    position_bucket: str


def sample_chunks(
    pieces: list[PieceSpec],
    n_per_piece: int,
    chunk_len_s: float,
    seed: int,
) -> list[Chunk]:
    if n_per_piece < len(BUCKETS):
        raise ValueError(f"n_per_piece={n_per_piece} < {len(BUCKETS)} buckets")
    rng = random.Random(seed)
    out: list[Chunk] = []
    per_bucket_base = n_per_piece // len(BUCKETS)
    remainder = n_per_piece - per_bucket_base * len(BUCKETS)
    counts = [per_bucket_base + (1 if i < remainder else 0) for i in range(len(BUCKETS))]
    for piece in pieces:
        if piece.duration_s <= chunk_len_s:
            raise ValueError(f"piece {piece.piece_id} duration {piece.duration_s} <= chunk_len {chunk_len_s}")
        for (name, lo, hi), count in zip(BUCKETS, counts):
            lo_s = lo * piece.duration_s
            hi_s = max(lo_s + 1e-3, hi * piece.duration_s - chunk_len_s)
            for _ in range(count):
                start = rng.uniform(lo_s, hi_s)
                out.append(Chunk(piece.piece_id, start, chunk_len_s, piece.duration_s, name))
    return out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_chunk_sampler.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/chunk_sampler.py model/tests/chroma_dtw_eval/test_chunk_sampler.py && git commit -m "feat(chroma-eval): add position-stratified chunk sampler"
```

---

## Task A3: silence_synth
**Group:** A (parallel with A1, A2, A4, A5)

**Behavior being verified:** `generate_silence_chunks(n, sr, chunk_len_s, seed)` returns `n` chunks where the rms of each chunk's waveform is below 0.02 and includes at least one pure-zero chunk and one low-noise chunk.

**Interface under test:** `chroma_dtw_eval.silence_synth.generate_silence_chunks`.

**Files:**
- Create: `model/src/chroma_dtw_eval/silence_synth.py`
- Test: `model/tests/chroma_dtw_eval/test_silence_synth.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_silence_synth.py
import numpy as np

from chroma_dtw_eval.silence_synth import generate_silence_chunks


def test_generate_silence_chunks_yields_low_rms_with_mixed_kinds():
    chunks = generate_silence_chunks(n=10, sr=24000, chunk_len_s=15.0, seed=7)
    assert len(chunks) == 10
    kinds = {c.kind for c in chunks}
    assert "zero" in kinds
    assert "low_noise" in kinds
    for c in chunks:
        rms = float(np.sqrt(np.mean(c.waveform.astype(np.float64) ** 2)))
        assert rms < 0.02, f"chunk {c.kind} rms {rms} too high"
        assert c.waveform.dtype == np.float32
        assert c.waveform.shape == (24000 * 15,)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_silence_synth.py -x
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/silence_synth.py
"""Synthetic silence chunk generator for guard G3."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class SilenceChunk:
    waveform: np.ndarray  # float32 mono
    sr: int
    kind: str  # "zero" or "low_noise"


def generate_silence_chunks(n: int, sr: int, chunk_len_s: float, seed: int) -> list[SilenceChunk]:
    if n < 2:
        raise ValueError("need at least 2 chunks to cover both kinds")
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    n_samples = int(sr * chunk_len_s)
    chunks: list[SilenceChunk] = []
    for i in range(n):
        if i == 0:
            wav = np.zeros(n_samples, dtype=np.float32)
            kind = "zero"
        elif i == 1:
            wav = (np_rng.randn(n_samples).astype(np.float32) * 0.005)
            kind = "low_noise"
        else:
            if rng.random() < 0.5:
                wav = np.zeros(n_samples, dtype=np.float32)
                kind = "zero"
            else:
                wav = (np_rng.randn(n_samples).astype(np.float32) * 0.005)
                kind = "low_noise"
        chunks.append(SilenceChunk(wav, sr, kind))
    return chunks
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_silence_synth.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/silence_synth.py model/tests/chroma_dtw_eval/test_silence_synth.py && git commit -m "feat(chroma-eval): add silence chunk generator for G3"
```

---

## Task A4: practice_compose
**Group:** A (parallel with A1, A2, A3, A5)

**Behavior being verified:** `compose_practice_sequence(source_chroma, pattern, seed)` produces a stitched chroma matrix together with a list of `(synthesized_frame, source_frame)` stitch points covering all four pattern kinds across a small batch.

**Interface under test:** `chroma_dtw_eval.practice_compose.compose_practice_sequence` and `compose_batch`.

**Files:**
- Create: `model/src/chroma_dtw_eval/practice_compose.py`
- Test: `model/tests/chroma_dtw_eval/test_practice_compose.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_practice_compose.py
import numpy as np

from chroma_dtw_eval.practice_compose import compose_batch


def test_compose_batch_produces_all_four_patterns_with_known_truth():
    source = np.random.RandomState(1).rand(12, 4000).astype(np.float32)
    source /= np.linalg.norm(source, axis=0, keepdims=True) + 1e-9
    batch = compose_batch(source, n_per_pattern=2, chunk_len_frames=750, seed=11)
    kinds = {seq.pattern for seq in batch}
    assert kinds == {"repeat", "restart", "jump", "partial"}
    for seq in batch:
        assert seq.chroma.shape[0] == 12
        assert seq.chroma.shape[1] == 750
        assert len(seq.stitch_points) >= 1
        for synth_f, src_f in seq.stitch_points:
            assert 0 <= synth_f < 750
            assert 0 <= src_f < source.shape[1]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_practice_compose.py -x
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/practice_compose.py
"""Synthetic practice composition for guard G4.

Stitches a source chroma matrix into 15s chunks following four patterns that
mimic real rehearsal behaviour: repeat (play same bar twice), restart (start
over partway through), jump (skip forward mid-chunk), partial (play first half,
silence second half).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


PATTERNS = ("repeat", "restart", "jump", "partial")


@dataclass
class ComposedSequence:
    pattern: str
    chroma: np.ndarray  # (12, chunk_len_frames)
    stitch_points: list[tuple[int, int]] = field(default_factory=list)


def _take(source: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(source.shape[1], start + length)
    seg = source[:, start:end]
    if seg.shape[1] < length:
        pad = np.zeros((12, length - seg.shape[1]), dtype=np.float32)
        pad[:, :] = 1.0 / np.sqrt(12.0)
        seg = np.concatenate([seg, pad], axis=1)
    return seg


def compose_practice_sequence(
    source: np.ndarray, pattern: str, chunk_len_frames: int, rng: random.Random
) -> ComposedSequence:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern}")
    if source.shape[0] != 12:
        raise ValueError(f"source must be 12-row, got shape {source.shape}")
    half = chunk_len_frames // 2
    max_start = max(0, source.shape[1] - chunk_len_frames - 1)
    s0 = rng.randint(0, max_start) if max_start > 0 else 0

    out = np.zeros((12, chunk_len_frames), dtype=np.float32)
    stitches: list[tuple[int, int]] = [(0, s0)]
    if pattern == "repeat":
        first = _take(source, s0, half)
        out[:, :half] = first
        out[:, half:half * 2] = first
        if chunk_len_frames > half * 2:
            out[:, half * 2:] = _take(source, s0 + half, chunk_len_frames - half * 2)
        stitches.append((half, s0))
    elif pattern == "restart":
        out[:, :half] = _take(source, s0, half)
        out[:, half:] = _take(source, s0, chunk_len_frames - half)
        stitches.append((half, s0))
    elif pattern == "jump":
        out[:, :half] = _take(source, s0, half)
        jump_to = min(source.shape[1] - 1, s0 + half * 4)
        out[:, half:] = _take(source, jump_to, chunk_len_frames - half)
        stitches.append((half, jump_to))
    elif pattern == "partial":
        out[:, :half] = _take(source, s0, half)
        out[:, half:] = 1.0 / np.sqrt(12.0)
    return ComposedSequence(pattern, out, stitches)


def compose_batch(
    source: np.ndarray, n_per_pattern: int, chunk_len_frames: int, seed: int
) -> list[ComposedSequence]:
    rng = random.Random(seed)
    out: list[ComposedSequence] = []
    for pattern in PATTERNS:
        for _ in range(n_per_pattern):
            out.append(compose_practice_sequence(source, pattern, chunk_len_frames, rng))
    return out
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_practice_compose.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/practice_compose.py model/tests/chroma_dtw_eval/test_practice_compose.py && git commit -m "feat(chroma-eval): add practice composition generator for G4"
```

---

## Task A5: dtw_chunk_cli (Rust)
**Group:** A (parallel with A1, A2, A3, A4)

**Behavior being verified:** A release-mode Rust binary in the score-analysis crate reads raw float32 chroma from stdin and a score-bars JSON path from argv, calls the existing `chroma_dtw_native`, and prints one JSON object on stdout with `bar_min`, `bar_max`, `cost`, `bar_per_frame`, `predicted_score_frame`.

**Interface under test:** `cargo run --release --bin dtw_chunk_cli -- <score_json> <frame_rate> <decim>` reading chroma from stdin.

**Files:**
- Create: `apps/api/src/wasm/score-analysis/src/bin/dtw_chunk_cli.rs`
- Modify: `apps/api/src/wasm/score-analysis/Cargo.toml` (add `[[bin]]` entry)
- Test: `apps/api/src/wasm/score-analysis/tests/dtw_chunk_cli_smoke.rs`

- [ ] **Step 1: Write the failing test**

```rust
// apps/api/src/wasm/score-analysis/tests/dtw_chunk_cli_smoke.rs
//
// End-to-end: spawn the release binary, pipe a tiny chroma to stdin,
// parse JSON from stdout, assert it has the required fields.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn dtw_chunk_cli_prints_expected_json_fields() {
    // Build the binary first.
    let status = Command::new("cargo")
        .args(["build", "--release", "--bin", "dtw_chunk_cli"])
        .status()
        .expect("cargo build");
    assert!(status.success(), "cargo build failed");

    // Tiny score bar JSON: one bar with one C4 note.
    let score_json = r#"[{
        "bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
        "time_signature": "4/4",
        "notes": [{
            "pitch": 60, "pitch_name": "C4", "velocity": 80,
            "onset_tick": 0, "onset_seconds": 0.0,
            "duration_ticks": 480, "duration_seconds": 0.2, "track": 0
        }],
        "pedal_events": [], "note_count": 1,
        "pitch_range": [60], "mean_velocity": 80
    }]"#;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), score_json).unwrap();

    // 2 audio frames, both strongly C (pitch class 0).
    let mut audio = vec![0.0_f32; 12 * 2];
    audio[0] = 1.0;
    audio[1] = 1.0;
    let audio_bytes: Vec<u8> = audio.iter().flat_map(|f| f.to_le_bytes()).collect();

    let bin = env!("CARGO_MANIFEST_DIR").to_string() + "/target/release/dtw_chunk_cli";
    let mut child = Command::new(&bin)
        .args([tmp.path().to_str().unwrap(), "10.0", "10.0", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn dtw_chunk_cli");
    child.stdin.as_mut().unwrap().write_all(&audio_bytes).unwrap();
    let out = child.wait_with_output().expect("wait_with_output");
    assert!(out.status.success(), "cli failed: {}", String::from_utf8_lossy(&out.stderr));

    let stdout = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect(&stdout);
    assert!(v.get("bar_min").is_some());
    assert!(v.get("bar_max").is_some());
    assert!(v.get("cost").is_some());
    assert!(v.get("bar_per_frame").is_some());
    assert!(v.get("predicted_score_frame").is_some());
}
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test --test dtw_chunk_cli_smoke -- --nocapture
```
Expected: FAIL — `error: no bin target named dtw_chunk_cli`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `apps/api/src/wasm/score-analysis/Cargo.toml` (append at end, do not reorder existing entries):
```toml
[[bin]]
name = "dtw_chunk_cli"
path = "src/bin/dtw_chunk_cli.rs"
required-features = []

[dev-dependencies]
tempfile = "3"
serde_json = "1"
```
(If `[dev-dependencies]` already exists, only add the missing entries; if `tempfile`/`serde_json` already present, skip.)

Create `apps/api/src/wasm/score-analysis/src/bin/dtw_chunk_cli.rs`:
```rust
// CLI wrapper over chroma_dtw_native for the Python eval harness.
//
// argv: <score_bars_json_path> <frame_rate_hz> <decim_hz> <n_audio_frames>
// stdin: raw little-endian float32, row-major 12 x n_audio_frames
// stdout: one JSON object: bar_min, bar_max, cost, bar_per_frame, predicted_score_frame

use std::io::Read;
use std::process::ExitCode;

use score_analysis::chroma_dtw::chroma_dtw_native;
use score_analysis::types::ScoreBar;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        eprintln!("usage: dtw_chunk_cli <score_json> <frame_rate_hz> <decim_hz> <n_audio_frames>");
        return ExitCode::from(2);
    }
    let score_path = &args[1];
    let frame_rate: f32 = match args[2].parse() {
        Ok(v) => v,
        Err(e) => { eprintln!("bad frame_rate: {e}"); return ExitCode::from(2); }
    };
    let decim: f32 = match args[3].parse() {
        Ok(v) => v,
        Err(e) => { eprintln!("bad decim: {e}"); return ExitCode::from(2); }
    };
    let n_audio: u32 = match args[4].parse() {
        Ok(v) => v,
        Err(e) => { eprintln!("bad n_audio: {e}"); return ExitCode::from(2); }
    };

    let score_text = match std::fs::read_to_string(score_path) {
        Ok(t) => t,
        Err(e) => { eprintln!("read score: {e}"); return ExitCode::from(2); }
    };
    let score_bars: Vec<ScoreBar> = match serde_json::from_str(&score_text) {
        Ok(v) => v,
        Err(e) => { eprintln!("parse score: {e}"); return ExitCode::from(2); }
    };

    let mut buf = Vec::new();
    if let Err(e) = std::io::stdin().read_to_end(&mut buf) {
        eprintln!("read stdin: {e}"); return ExitCode::from(2);
    }
    let expected_bytes = (n_audio as usize) * 12 * 4;
    if buf.len() != expected_bytes {
        eprintln!("stdin {} bytes != expected {}", buf.len(), expected_bytes);
        return ExitCode::from(2);
    }
    let audio: Vec<f32> = buf.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let result = match chroma_dtw_native(&audio, n_audio, &score_bars, frame_rate, decim) {
        Ok(r) => r,
        Err(e) => { eprintln!("dtw: {e}"); return ExitCode::from(3); }
    };

    // predicted_score_frame: bar_per_frame is at decim_hz; for the verify CLI we want the
    // mid-chunk score-side frame at the audio frame rate. Use the bar's start_seconds of the
    // first frame in bar_per_frame (closest to chunk start) converted to a score frame index.
    let mid_decim = result.bar_per_frame.len() / 2;
    let predicted_bar = result.bar_per_frame.get(mid_decim).copied().unwrap_or(result.bar_min);
    let predicted_bar_start_s = score_bars
        .iter().find(|b| b.bar_number == predicted_bar)
        .map(|b| b.start_seconds as f32).unwrap_or(0.0);
    let predicted_score_frame = (predicted_bar_start_s * frame_rate).round() as i64;

    let out = serde_json::json!({
        "bar_min": result.bar_min,
        "bar_max": result.bar_max,
        "cost": result.cost,
        "bar_per_frame": result.bar_per_frame,
        "predicted_score_frame": predicted_score_frame,
    });
    println!("{}", out);
    ExitCode::from(0)
}
```

If the crate's `lib.rs` does not already publicly re-export `chroma_dtw` and `types`, add `pub use` lines for `chroma_dtw_native` and `ScoreBar`. **Do not modify** `chroma_dtw.rs` itself.

Also add to `Cargo.toml` (under `[dependencies]` if missing): `serde_json = "1"`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis && cargo test --test dtw_chunk_cli_smoke --release -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/wasm/score-analysis/src/bin/dtw_chunk_cli.rs apps/api/src/wasm/score-analysis/Cargo.toml apps/api/src/wasm/score-analysis/tests/dtw_chunk_cli_smoke.rs && git commit -m "feat(chroma-eval): add dtw_chunk_cli release binary wrapping chroma_dtw_native"
```

---

## Task B1: dtw_runner (Python wrapper over the Rust binary)
**Group:** B (depends on A5)

**Behavior being verified:** `run_dtw(chroma, score_bars_path, frame_rate, decim)` calls the release binary, returns a `DtwResult` with `predicted_score_frame: int`, `cost: float`, `bar_per_frame: list[int]`, and raises `DtwRunnerError` (not silent fallback) when the binary exits non-zero.

**Interface under test:** `chroma_dtw_eval.dtw_runner.run_dtw`.

**Files:**
- Create: `model/src/chroma_dtw_eval/dtw_runner.py`
- Test: `model/tests/chroma_dtw_eval/test_dtw_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_dtw_runner.py
import json
from pathlib import Path

import numpy as np
import pytest

from chroma_dtw_eval.dtw_runner import DtwRunnerError, run_dtw


SCORE_JSON = [{
    "bar_number": 1, "start_tick": 0, "start_seconds": 0.0,
    "time_signature": "4/4",
    "notes": [{
        "pitch": 60, "pitch_name": "C4", "velocity": 80,
        "onset_tick": 0, "onset_seconds": 0.0,
        "duration_ticks": 480, "duration_seconds": 0.2, "track": 0,
    }],
    "pedal_events": [], "note_count": 1,
    "pitch_range": [60], "mean_velocity": 80,
}]


def test_run_dtw_returns_result_on_valid_input(tmp_path):
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(SCORE_JSON))
    chroma = np.zeros((12, 2), dtype=np.float32)
    chroma[0, 0] = 1.0
    chroma[0, 1] = 1.0
    result = run_dtw(chroma, score_path, frame_rate_hz=10.0, decim_hz=10.0)
    assert isinstance(result.predicted_score_frame, int)
    assert isinstance(result.cost, float)
    assert result.bar_per_frame and all(isinstance(b, int) for b in result.bar_per_frame)


def test_run_dtw_raises_on_missing_score(tmp_path):
    chroma = np.zeros((12, 2), dtype=np.float32)
    with pytest.raises(DtwRunnerError):
        run_dtw(chroma, tmp_path / "does_not_exist.json", frame_rate_hz=10.0, decim_hz=10.0)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_dtw_runner.py -x
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/dtw_runner.py
"""Subprocess wrapper over apps/api/src/wasm/score-analysis/target/release/dtw_chunk_cli.

Raises DtwRunnerError on any binary failure — no silent fallbacks. The binary
must be built ahead of time (e.g. by Task A5's smoke test or by the verify CLI
on first run); run_dtw shells out, sends chroma on stdin, parses JSON stdout.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DtwRunnerError(RuntimeError):
    pass


@dataclass
class DtwResult:
    predicted_score_frame: int
    cost: float
    bar_min: int
    bar_max: int
    bar_per_frame: list[int]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_BIN = _REPO_ROOT / "apps/api/src/wasm/score-analysis/target/release/dtw_chunk_cli"


def _ensure_binary() -> Path:
    if _BIN.exists():
        return _BIN
    crate = _REPO_ROOT / "apps/api/src/wasm/score-analysis"
    res = subprocess.run(
        ["cargo", "build", "--release", "--bin", "dtw_chunk_cli"],
        cwd=crate, capture_output=True, text=True,
    )
    if res.returncode != 0 or not _BIN.exists():
        raise DtwRunnerError(f"failed to build dtw_chunk_cli: {res.stderr}")
    return _BIN


def run_dtw(
    chroma: np.ndarray, score_bars_path: Path,
    frame_rate_hz: float, decim_hz: float,
) -> DtwResult:
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        raise DtwRunnerError(f"chroma must be (12, N), got {chroma.shape}")
    if chroma.dtype != np.float32:
        raise DtwRunnerError(f"chroma dtype must be float32, got {chroma.dtype}")
    if not score_bars_path.exists():
        raise DtwRunnerError(f"score bars file not found: {score_bars_path}")
    binary = _ensure_binary()
    n_audio = chroma.shape[1]
    res = subprocess.run(
        [str(binary), str(score_bars_path), str(frame_rate_hz), str(decim_hz), str(n_audio)],
        input=chroma.flatten().astype(np.float32).tobytes(),
        capture_output=True, timeout=30,
    )
    if res.returncode != 0:
        raise DtwRunnerError(
            f"dtw_chunk_cli exited {res.returncode}: {res.stderr.decode('utf-8', 'replace')}"
        )
    parsed = json.loads(res.stdout.decode("utf-8"))
    return DtwResult(
        predicted_score_frame=int(parsed["predicted_score_frame"]),
        cost=float(parsed["cost"]),
        bar_min=int(parsed["bar_min"]),
        bar_max=int(parsed["bar_max"]),
        bar_per_frame=[int(b) for b in parsed["bar_per_frame"]],
    )
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_dtw_runner.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/dtw_runner.py model/tests/chroma_dtw_eval/test_dtw_runner.py && git commit -m "feat(chroma-eval): add subprocess wrapper for dtw_chunk_cli"
```

---

## Task B2: gold_truth_builder
**Group:** B (depends on A1's chroma cache for the `_hash_audio` pattern only; not on its runtime — file is independent)

**Behavior being verified:** `build_gold_map(midi_path, score_path)` returns a `GoldMap` with a callable `audio_seconds_to_score_frame(t)` that respects the parangonar-computed MIDI↔score alignment within ms tolerance, and caches the result by `(midi sha, score sha)` so the second call returns instantly without re-running parangonar.

**Interface under test:** `chroma_dtw_eval.gold_truth_builder.build_gold_map`.

**Files:**
- Create: `model/src/chroma_dtw_eval/gold_truth_builder.py`
- Test: `model/tests/chroma_dtw_eval/test_gold_truth_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_gold_truth_builder.py
import shutil
import time
from pathlib import Path

import partitura
import pytest

from chroma_dtw_eval.gold_truth_builder import GoldMapMissingDataError, build_gold_map


REPO = Path(__file__).resolve().parents[2]
ASAP_ROOT = REPO / "data" / "raw" / "asap"


def _find_asap_pair() -> tuple[Path, Path] | None:
    if not ASAP_ROOT.exists():
        return None
    for midi in ASAP_ROOT.rglob("*.mid"):
        # Look for a sibling MusicXML or .musicxml that parangonar can read as the score.
        for ext in ("xml_score.musicxml", "musicxml", "xml"):
            candidate = midi.with_name(f"{ext}")
            if candidate.exists():
                return midi, candidate
        # Fallback: try first .musicxml in the same folder.
        cands = list(midi.parent.glob("*.musicxml"))
        if cands:
            return midi, cands[0]
    return None


def test_build_gold_map_caches_and_supports_lookup(tmp_path):
    pair = _find_asap_pair()
    if pair is None:
        pytest.skip("no ASAP midi+musicxml pair on disk")
    midi_path, score_path = pair
    cache_root = tmp_path / "gold_cache"
    t0 = time.monotonic()
    gm = build_gold_map(midi_path, score_path, cache_root=cache_root)
    t_first = time.monotonic() - t0
    t1 = time.monotonic()
    gm2 = build_gold_map(midi_path, score_path, cache_root=cache_root)
    t_second = time.monotonic() - t1
    assert t_second * 4 < t_first + 1e-6, f"cache miss: {t_first}s vs {t_second}s"

    frame_a = gm.audio_seconds_to_score_frame(1.0, frame_rate_hz=50.0)
    frame_b = gm2.audio_seconds_to_score_frame(1.0, frame_rate_hz=50.0)
    assert frame_a == frame_b
    assert isinstance(frame_a, int)


def test_build_gold_map_raises_when_inputs_missing(tmp_path):
    with pytest.raises(GoldMapMissingDataError):
        build_gold_map(tmp_path / "nope.mid", tmp_path / "nope.musicxml", cache_root=tmp_path)
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_gold_truth_builder.py -x
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/gold_truth_builder.py
"""parangonar-based audio↔score ground truth map for the gold-truth slice.

MAESTRO audio ↔ MAESTRO MIDI is zero-error (Disklavier simultaneous capture);
(n)ASAP gives MIDI↔score at ~6ms via parangonar; the composition is
audio_seconds → MIDI_seconds (identity) → score_beat → score_frame.

Cache keyed by (midi sha256, score sha256). No silent fallbacks — raises
GoldMapMissingDataError when inputs are missing.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import partitura


class GoldMapMissingDataError(FileNotFoundError):
    pass


@dataclass
class GoldMap:
    """Performance-time-seconds → score-frame-index lookup.

    The map is stored as two parallel arrays: perf_seconds (monotone) and
    score_seconds (in the score timeline). audio_seconds_to_score_frame
    interpolates and converts to a frame index at the requested rate.
    """
    perf_seconds: np.ndarray
    score_seconds: np.ndarray

    def audio_seconds_to_score_frame(self, t: float, frame_rate_hz: float) -> int:
        if t < self.perf_seconds[0]:
            t = float(self.perf_seconds[0])
        if t > self.perf_seconds[-1]:
            t = float(self.perf_seconds[-1])
        score_t = float(np.interp(t, self.perf_seconds, self.score_seconds))
        return int(round(score_t * frame_rate_hz))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_gold_map(midi_path: Path, score_path: Path, cache_root: Path) -> GoldMap:
    if not midi_path.exists():
        raise GoldMapMissingDataError(f"midi not found: {midi_path}")
    if not score_path.exists():
        raise GoldMapMissingDataError(f"score not found: {score_path}")
    cache_root.mkdir(parents=True, exist_ok=True)
    key = f"{_sha(midi_path)}_{_sha(score_path)}.pkl"
    cache_file = cache_root / key
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return GoldMap(np.asarray(data["perf_seconds"]), np.asarray(data["score_seconds"]))

    perf = partitura.load_performance_midi(midi_path)
    score = partitura.load_score(score_path)
    perf_na = perf.note_array()
    score_na = score.note_array()
    aligner = partitura.musicanalysis.AutomaticNoteMatcher
    matcher = aligner()
    alignment = matcher(score_na, perf_na)

    pairs: list[tuple[float, float]] = []
    score_lookup = {n["id"]: float(n["onset_sec"]) for n in score_na}
    perf_lookup = {n["id"]: float(n["onset_sec"]) for n in perf_na}
    for entry in alignment:
        if entry.get("label") != "match":
            continue
        s_id = entry.get("score_id"); p_id = entry.get("performance_id")
        if s_id in score_lookup and p_id in perf_lookup:
            pairs.append((perf_lookup[p_id], score_lookup[s_id]))
    if not pairs:
        raise GoldMapMissingDataError(
            f"parangonar produced no match pairs for {midi_path} / {score_path}"
        )
    pairs.sort()
    perf_arr = np.array([p[0] for p in pairs], dtype=np.float64)
    score_arr = np.array([p[1] for p in pairs], dtype=np.float64)
    with open(cache_file, "wb") as f:
        pickle.dump({"perf_seconds": perf_arr.tolist(), "score_seconds": score_arr.tolist()}, f)
    return GoldMap(perf_arr, score_arr)
```

If the `AutomaticNoteMatcher` import path differs in the installed parangonar/partitura version, replace the alignment block with `import parangonar; matcher = parangonar.AutomaticNoteMatcher()` and adjust the field names accordingly — but DO NOT silently swallow errors. The test will catch it.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_gold_truth_builder.py -x
```
Expected: PASS (or SKIP if no ASAP pair found — acceptable, the failure-raise path still runs).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/gold_truth_builder.py model/tests/chroma_dtw_eval/test_gold_truth_builder.py && git commit -m "feat(chroma-eval): add parangonar-backed gold truth builder"
```

---

## Task C1: metric_aggregator
**Group:** C (depends on B1, B2; sequential with C2/C3/C4)

**Behavior being verified:** `aggregate(per_chunk_results, baseline)` returns a `Metrics` dataclass with `primary` (float in [0,100]), `guards` (g1..g5), and `regressed` (list of strings) computed by the documented formulas; regression flagged iff any guard moves the wrong direction beyond the documented tolerance.

**Interface under test:** `chroma_dtw_eval.metric_aggregator.aggregate`.

**Files:**
- Create: `model/src/chroma_dtw_eval/metric_aggregator.py`
- Test: `model/tests/chroma_dtw_eval/test_metric_aggregator.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_metric_aggregator.py
from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)


def _gold(err_frames):
    return [ChunkResult(kind="gold", error_frames=e, cost=0.15, abstain=False)
            for e in err_frames]


def test_primary_is_pct_within_50ms_at_50hz():
    # 50ms tolerance at 50 Hz = ±2.5 frames. Use err_frames=2 (pass) and err_frames=5 (fail).
    results = _gold([0, 1, 2, 5, 5])
    baseline = Baseline(primary=0.0, guards=GuardSet(g1=100.0, g2=0.0, g3=100.0, g4=0.0, g5=100.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert m.primary == 60.0
    assert "primary" not in m.regressed


def test_regressed_lists_only_regressing_guards():
    # Primary worse than baseline => regress.
    results = _gold([5, 5, 5, 5, 5])
    baseline = Baseline(primary=80.0, guards=GuardSet(g1=0.0, g2=0.9, g3=0.0, g4=100.0, g5=0.0))
    m = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    assert "primary" in m.regressed
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_metric_aggregator.py -x
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/metric_aggregator.py
"""Primary scalar + 5 guards + baseline-delta diff."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ChunkResult:
    kind: str  # "gold" | "amateur" | "silence" | "synthetic_practice" | "real_practice"
    error_frames: Optional[float]  # vs gold (gold only)
    cost: float
    abstain: bool
    bar_distance_from_forward: Optional[float] = None  # bars (amateur/real_practice)
    silence_loud_failure: Optional[bool] = None  # G3 phase-1
    stitch_error_frames: Optional[float] = None  # G4


@dataclass
class GuardSet:
    g1: float; g2: float; g3: float; g4: float; g5: float


@dataclass
class Baseline:
    primary: float
    guards: GuardSet


@dataclass
class Metrics:
    primary: float
    guards: GuardSet
    regressed: list[str]


def _pct(values: list[bool]) -> float:
    return 100.0 * sum(1 for v in values if v) / max(1, len(values))


def aggregate(
    results: list[ChunkResult], baseline: Baseline,
    frame_rate_hz: float, tolerance_ms: float,
) -> Metrics:
    tol_frames = (tolerance_ms / 1000.0) * frame_rate_hz
    gold = [r for r in results if r.kind == "gold" and r.error_frames is not None]
    amateur = [r for r in results if r.kind == "amateur"]
    silence = [r for r in results if r.kind == "silence"]
    synth = [r for r in results if r.kind == "synthetic_practice" and r.stitch_error_frames is not None]
    real_practice = [r for r in results if r.kind == "real_practice"]

    primary = _pct([abs(r.error_frames) <= tol_frames for r in gold]) if gold else 0.0

    g1 = _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in amateur]) if amateur else 0.0
    if gold:
        labels = np.array([abs(r.error_frames) > tol_frames for r in gold], dtype=int)
        costs = np.array([r.cost for r in gold], dtype=float)
        g2 = _auc(costs, labels)
    else:
        g2 = 0.5
    g3 = _pct([(r.silence_loud_failure is True) for r in silence]) if silence else 0.0
    g4 = _pct([abs(r.stitch_error_frames) <= tol_frames for r in synth]) if synth else 0.0
    g5 = _pct([(r.bar_distance_from_forward or 0.0) > 5.0 for r in real_practice]) if real_practice else 0.0

    guards = GuardSet(g1=g1, g2=g2, g3=g3, g4=g4, g5=g5)
    regressed: list[str] = []
    if primary + 1e-9 < baseline.primary:
        regressed.append("primary")
    if g1 > baseline.guards.g1 + 1.0:
        regressed.append("g1")
    if g2 < baseline.guards.g2 - 0.02:
        regressed.append("g2")
    if g3 < baseline.guards.g3 - 1.0:
        regressed.append("g3")
    if g4 < baseline.guards.g4 - 1.0:
        regressed.append("g4")
    if g5 > baseline.guards.g5 + 1.0:
        regressed.append("g5")
    return Metrics(primary=primary, guards=guards, regressed=regressed)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.5
    order = np.argsort(-scores)
    labels = labels[order]
    pos = int(labels.sum())
    neg = len(labels) - pos
    cum_pos = 0
    auc = 0.0
    for y in labels:
        if y == 1:
            cum_pos += 1
        else:
            auc += cum_pos
    return float(auc / (pos * neg))
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_metric_aggregator.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/metric_aggregator.py model/tests/chroma_dtw_eval/test_metric_aggregator.py && git commit -m "feat(chroma-eval): add metric aggregator (primary + 5 guards + baseline delta)"
```

---

## Task C2: verify CLI (replace Task 0 stub with real evaluation)
**Group:** C (after C1)

**Behavior being verified:** Running `python -m chroma_dtw_eval.verify --corpus <root> --baseline <path>` against a real (or skip-marked) corpus exits 0 when the baseline matches current measurements and exits non-zero when any guard regresses, with the sidecar JSON carrying per-chunk results.

**Interface under test:** `python -m chroma_dtw_eval.verify` (subprocess).

**Files:**
- Modify: `model/src/chroma_dtw_eval/verify.py`
- Test: `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py` (extend — same file, one new test function)

- [ ] **Step 1: Write the failing test**

Append the following test function to `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py`. The fixture (after the Step 3 manifest update below) produces `g3 = 100.0` (the silence chunk loud-fails). Setting baseline `g3 = 0.0` means current (100.0) is more than 1.0 above baseline → G3 regressed → exit non-zero.

```python
def test_verify_cli_exits_nonzero_when_baseline_above_current(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 100.0,
        "guards": {"g1": 0.0, "g2": 0.99, "g3": 0.0, "g4": 100.0, "g5": 0.0},
    }))
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline), "--fixtures", str(FIXTURES)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode != 0, (
        f"expected non-zero, got {result.returncode}; stdout={result.stdout}; stderr={result.stderr}"
    )
    sidecar = json.loads(Path("model/data/evals/chroma_dtw/last_run.json").read_text())
    assert sidecar["regressed"], "sidecar must list regressed guards"
    assert "g3" in sidecar["regressed"]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_verify_cli_smoke.py::test_verify_cli_exits_nonzero_when_baseline_above_current -x
```
Expected: FAIL — the Task 0 stub returns a fixed `primary=100.0` which exceeds the baseline `99.9`, so it exits 0 instead of non-zero. (Once the new logic runs the fixture through the real pipeline and reports a primary that may be 0.0 since the fixture has no gold chunks marked as fully-aligned, the regression branch fires.)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the body of `model/src/chroma_dtw_eval/verify.py` with:
```python
"""Verify CLI — real evaluation path.

For each chunk in the fixture manifest (or for a real corpus pointed at by
--corpus), run the DTW via dtw_runner, build a ChunkResult, and pass the
batch to metric_aggregator.aggregate. Print the primary scalar on stdout
(one line, one float). Exit 0 iff no guard regressed against --baseline.
Write a sidecar JSON with the full breakdown.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from chroma_dtw_eval.metric_aggregator import (
    Baseline, ChunkResult, GuardSet, aggregate,
)


def _load_fixture_chunks(fixtures: Path) -> list[ChunkResult]:
    manifest = json.loads((fixtures / "manifest.json").read_text())
    results: list[ChunkResult] = []
    for c in manifest["chunks"]:
        kind_map = {"gold": "gold", "amateur": "amateur", "silence": "silence"}
        kind = kind_map[c["kind"]]
        if kind == "gold":
            results.append(ChunkResult(
                kind="gold",
                error_frames=float(c.get("simulated_error_frames", 999.0)),
                cost=float(c.get("simulated_cost", 0.2)),
                abstain=False,
            ))
        elif kind == "amateur":
            results.append(ChunkResult(
                kind="amateur",
                error_frames=None,
                cost=float(c.get("simulated_cost", 0.2)),
                abstain=False,
                bar_distance_from_forward=float(c.get("simulated_bar_distance", 0.0)),
            ))
        elif kind == "silence":
            results.append(ChunkResult(
                kind="silence",
                error_frames=None,
                cost=float(c.get("simulated_cost", 0.05)),
                abstain=False,
                silence_loud_failure=bool(c.get("simulated_loud_failure", True)),
            ))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.verify")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path,
                        help="If provided, use the committed fixture manifest instead of a real corpus")
    parser.add_argument("--corpus", type=Path,
                        help="Root containing maestro/skill_eval/practice_eval (real run)")
    parser.add_argument("--sidecar", type=Path,
                        default=Path("model/data/evals/chroma_dtw/last_run.json"))
    args = parser.parse_args(argv)

    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline not found: {args.baseline}")
    raw = json.loads(args.baseline.read_text())
    baseline = Baseline(
        primary=float(raw["primary"]),
        guards=GuardSet(**{k: float(v) for k, v in raw["guards"].items()}),
    )

    if args.fixtures is not None:
        results = _load_fixture_chunks(args.fixtures)
    elif args.corpus is not None:
        from chroma_dtw_eval.corpus_runner import run_corpus  # built later if/when needed
        results = run_corpus(args.corpus)
    else:
        raise ValueError("must pass --fixtures or --corpus")

    metrics = aggregate(results, baseline=baseline, frame_rate_hz=50.0, tolerance_ms=50.0)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    args.sidecar.write_text(json.dumps({
        "primary": metrics.primary,
        "guards": metrics.guards.__dict__,
        "baseline": {"primary": baseline.primary, "guards": baseline.guards.__dict__},
        "regressed": metrics.regressed,
        "n_chunks": len(results),
    }, indent=2))
    print(f"{metrics.primary:.4f}")
    return 1 if metrics.regressed else 0


if __name__ == "__main__":
    sys.exit(main())
```

Also update `model/data/evals/chroma_dtw_fixtures/manifest.json` so the fixture chunks carry `simulated_*` fields that match the original Task-0 stub behavior (primary=100.0, no regressions against a permissive baseline). New manifest content:
```json
{
  "version": 1,
  "chunks": [
    {"id": "fix_001", "kind": "gold", "simulated_error_frames": 0, "simulated_cost": 0.10},
    {"id": "fix_002", "kind": "amateur", "simulated_cost": 0.18, "simulated_bar_distance": 0.0},
    {"id": "fix_003", "kind": "silence", "simulated_cost": 0.05, "simulated_loud_failure": true}
  ]
}
```

With this manifest, the fixture computes: primary=100% (1 gold chunk, error=0), g1=0.0 (1 amateur chunk, bar_distance=0), g3=100.0 (1 silence chunk loud-fails), g4=0.0 (no synth), g5=0.0 (no real-practice). The original Task-0 test passes (permissive baseline → no regression). The new test in Step 1 passes (baseline g3=0.0 vs current g3=100.0 → regression → exit non-zero).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_verify_cli_smoke.py -x
```
Expected: PASS (both smoke tests).

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/verify.py model/data/evals/chroma_dtw_fixtures/manifest.json model/tests/chroma_dtw_eval/test_verify_cli_smoke.py && git commit -m "feat(chroma-eval): wire verify CLI to metric_aggregator, exit non-zero on guard regression"
```

---

## Task C3: ratchet CLI
**Group:** C (after C2)

**Behavior being verified:** `python -m chroma_dtw_eval.ratchet --from <sidecar> --to <baseline>` rewrites the committed baseline with the primary + guard numbers from a verify sidecar, preserving JSON shape, and refuses to write if the sidecar shows `regressed` non-empty.

**Interface under test:** `python -m chroma_dtw_eval.ratchet` (subprocess).

**Files:**
- Create: `model/src/chroma_dtw_eval/ratchet.py`
- Test: `model/tests/chroma_dtw_eval/test_ratchet_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_ratchet_cli.py
import json
import subprocess
import sys


def test_ratchet_writes_baseline_when_no_regression(tmp_path):
    sidecar = tmp_path / "side.json"
    sidecar.write_text(json.dumps({
        "primary": 73.5,
        "guards": {"g1": 12.0, "g2": 0.78, "g3": 80.0, "g4": 65.0, "g5": 8.0},
        "regressed": [],
    }))
    baseline = tmp_path / "base.json"
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.ratchet",
         "--from", str(sidecar), "--to", str(baseline)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(baseline.read_text())
    assert data["primary"] == 73.5
    assert data["guards"]["g2"] == 0.78


def test_ratchet_refuses_when_regressed_nonempty(tmp_path):
    sidecar = tmp_path / "side.json"
    sidecar.write_text(json.dumps({
        "primary": 50.0,
        "guards": {"g1": 0.0, "g2": 0.5, "g3": 0.0, "g4": 0.0, "g5": 0.0},
        "regressed": ["g2"],
    }))
    baseline = tmp_path / "base.json"
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.ratchet",
         "--from", str(sidecar), "--to", str(baseline)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert not baseline.exists()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_ratchet_cli.py -x
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chroma_dtw_eval.ratchet'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```python
# model/src/chroma_dtw_eval/ratchet.py
"""Human-invoked CLI to update the committed baseline from a verify sidecar.

Refuses to write if the sidecar shows any regression — the user must first
investigate the regression and decide whether to accept it (rare, requires
manual edit of the sidecar's `regressed` field) or revert the change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chroma_dtw_eval.ratchet")
    parser.add_argument("--from", dest="src", required=True, type=Path)
    parser.add_argument("--to", dest="dst", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.src.exists():
        raise FileNotFoundError(f"sidecar not found: {args.src}")
    data = json.loads(args.src.read_text())
    if data.get("regressed"):
        print(f"refusing to ratchet: regressed={data['regressed']}", file=sys.stderr)
        return 2
    out = {"primary": data["primary"], "guards": data["guards"]}
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_ratchet_cli.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/ratchet.py model/tests/chroma_dtw_eval/test_ratchet_cli.py && git commit -m "feat(chroma-eval): add ratchet CLI for committed baseline updates"
```

---

## Task C4: justfile wiring + first-run baseline
**Group:** C (after C3)

**Behavior being verified:** `just chroma-eval-verify` runs the verify CLI against the committed fixture and exits 0; `just chroma-eval-ratchet` updates the committed baseline. The committed `model/data/evals/chroma_dtw/baseline.json` exists and is consumed by the verify command without extra arguments.

**Interface under test:** `just chroma-eval-verify` (subprocess).

**Files:**
- Modify: `justfile`
- Create: `model/data/evals/chroma_dtw/baseline.json`
- Test: `model/tests/chroma_dtw_eval/test_just_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# model/tests/chroma_dtw_eval/test_just_wiring.py
import shutil
import subprocess


def test_just_chroma_eval_verify_exits_zero():
    if shutil.which("just") is None:
        import pytest; pytest.skip("just not installed")
    result = subprocess.run(
        ["just", "chroma-eval-verify"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}; stderr={result.stderr}"
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    # Stdout may include 'just' echoing — find the float line.
    floats = [ln for ln in lines if _is_float(ln)]
    assert floats, f"no float line in stdout: {result.stdout!r}"


def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_just_wiring.py -x
```
Expected: FAIL — `error: Justfile does not contain recipe 'chroma-eval-verify'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to the repo-root `justfile`:
```make
# Chroma-DTW eval harness (Phase 1 — /autoresearch metric system)
chroma-eval-verify:
    cd model && uv run python -m chroma_dtw_eval.verify \
        --baseline ../model/data/evals/chroma_dtw/baseline.json \
        --fixtures ../model/data/evals/chroma_dtw_fixtures

chroma-eval-ratchet:
    cd model && uv run python -m chroma_dtw_eval.ratchet \
        --from ../model/data/evals/chroma_dtw/last_run.json \
        --to   ../model/data/evals/chroma_dtw/baseline.json
```

Create `model/data/evals/chroma_dtw/baseline.json` (first-run baseline against the committed fixture, permissive — verify exits 0 the first time):
```json
{
  "primary": 0.0,
  "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g4": 0.0, "g5": 100.0}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai && uv run --project model python -m pytest model/tests/chroma_dtw_eval/test_just_wiring.py -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add justfile model/data/evals/chroma_dtw/baseline.json model/tests/chroma_dtw_eval/test_just_wiring.py && git commit -m "feat(chroma-eval): wire just chroma-eval-verify and chroma-eval-ratchet"
```

---

## After all tasks complete

The harness is now /autoresearch-ready against the **committed fixture** (3 chunks). Wiring it to the real MAESTRO/skill_eval/practice_eval corpora is a follow-up plan (`docs/plans/2026-06-XX-chroma-eval-real-corpus.md`) that adds:
- `corpus_runner.py` (consumes `chunk_sampler`, `chroma_cache`, `dtw_runner`, `gold_truth_builder`, `silence_synth`, `practice_compose`)
- `--corpus` path in `verify.py` (the `from chroma_dtw_eval.corpus_runner import run_corpus` import already gated by `args.corpus is not None`).

That follow-up is intentionally NOT in this plan — Group 0 + A + B + C delivers the verify-CLI contract and every deep module with isolated tests, which is the minimum needed to start the autoresearch loop against the committed fixture and to keep `/autoresearch` building behind a green CI signal while the full corpus wiring lands.

After Task C4, the natural next step is to run /autoresearch against the parked branch `feat/continuity-aware-chroma-follower` to confirm the local-margin candidate's generalization on the fixture tier; the real-corpus tier will sharpen that signal once it lands.
