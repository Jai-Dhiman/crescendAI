# Transkun Transcriber Migration Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task) where the group table allows. Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace aria-amt with Transkun repo-wide behind an unchanged `/transcribe` contract and a single shared shell-out helper, so every transcription consumer gets Transkun's offset/velocity/pedal fidelity with zero contract churn.
**Spec:** docs/specs/2026-07-23-transkun-transcriber-migration-design.md
**Issue/branch:** #128 / worktree `issue-128-transkun-adopt`
**Style:** Follow CLAUDE.md coding standards. Explicit exception handling over silent fallbacks. No emojis. All measurer/service scripts stay `uv`-run.

## Verified environment facts (do not re-derive)
- Transkun CLI: `transkun <audioPath> <outPath> [--device cpu]`; positional in/out; default `--device` is already `cpu`. Installed version = `2.0.1` (probed 2026-07-23).
- Invoke ONLY as `uv run --no-project --with transkun --python 3.11 transkun IN.wav OUT.wav.mid --device cpu`. A bare `uv run --with ...` from inside `model/` rebuilds the shared `model/.venv` (gotcha [[project_uv_run_mutates_model_venv]]); `--no-project` is mandatory. Nested uv-run inside a uv-run script works.
- Transkun on CPU ≈ 0.45× realtime; MPS is SLOWER — force CPU.
- `pretty_midi` reads `instrument.notes[*].velocity` and `instrument.control_changes` (`number==64` for sustain). Available in `model/.venv`; add to the service `/// deps`.
- HTTP `/transcribe` contract is frozen: `inference.ts`, `amt_regen.py`, `pieceid_amt_axis.py` must need ZERO edits.
- `audio_chunker.py` has 4 live importers (MuQ path) — DO NOT delete. `src/export_onnx.py` does not exist — only `apps/inference/amt/scripts/export_onnx.py`.

## Task Groups
```
Group 0 (deep module / harness, internal-sequential):  T1 -> T2 -> T3 -> T4        [SHIPS INDEPENDENTLY]
Group A (HTTP service, depends on 0):                   T5 -> T6 -> T7 -> (T8 || T9)
Group B (measurer swaps, depends on 0, parallel):       T10 || T11
Group C (chroma retune, depends on 0):                  T12
Group D (config + docs, independent, parallel):         T13 || T14
Group E (deletions, depends on A + B):                  T15
Group F (validation gates, depends on A+B+C+D+E, seq):  T16 -> T17 -> T18 -> T19 -> T20
```
Group 0 is `[SHIPS INDEPENDENTLY]`: once T1–T4 land, `transkun_cli` is a usable, tested, model-verified transcription helper on its own.

---

### Task 1: transkun_cli parses notes + velocity from a MIDI file
**Group:** 0 (first; sequential within group)

**Behavior being verified:** `midi_to_notes_and_pedals` turns a MIDI file's notes into `{pitch,onset,offset,velocity}` dicts, sorted by (onset,pitch), velocity carried through.
**Interface under test:** `transkun_cli.midi_to_notes_and_pedals(midi_path)`

**Files:**
- Create: `apps/inference/amt/transkun_cli.py`
- Test: `apps/inference/amt/test_transkun_cli.py`

- [x] **Step 1: Write the failing test**

```python
# apps/inference/amt/test_transkun_cli.py
"""Behavior tests for the shared Transkun shell-out helper.

Run: cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile \
        --with pytest pytest test_transkun_cli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pretty_midi
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transkun_cli


def _write_midi(path: Path, notes, pedal_ccs) -> None:
    """notes: list of (pitch, start_s, end_s, velocity). pedal_ccs: list of (time_s, value)."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end, vel in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )
    for t, v in pedal_ccs:
        inst.control_changes.append(
            pretty_midi.ControlChange(number=64, time=t, value=v)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_notes_carry_pitch_onset_offset_velocity(tmp_path):
    midi_path = tmp_path / "n.mid"
    _write_midi(
        midi_path,
        notes=[(60, 0.5, 1.0, 90), (67, 0.10, 0.40, 55), (60, 0.10, 0.30, 70)],
        pedal_ccs=[],
    )
    notes, pedals = transkun_cli.midi_to_notes_and_pedals(midi_path)

    assert pedals == []
    # sorted by (onset, pitch): (60,0.10),(67,0.10),(60,0.50)
    assert [(n["pitch"], round(n["onset"], 2)) for n in notes] == [
        (60, 0.10), (67, 0.10), (60, 0.50)
    ]
    first = notes[0]
    assert set(first) == {"pitch", "onset", "offset", "velocity"}
    assert first["velocity"] == 70
    assert round(first["offset"], 2) == 0.30
    assert all(isinstance(n["velocity"], int) for n in notes)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with pytest pytest test_transkun_cli.py::test_notes_carry_pitch_onset_offset_velocity -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'transkun_cli'` (or `AttributeError: midi_to_notes_and_pedals`).

- [x] **Step 3: Implement the minimum to make the test pass**

```python
# apps/inference/amt/transkun_cli.py
"""Shared Transkun transcription helper.

Deliberately does NOT import `transkun`: it shells out to an isolated env
(`uv run --no-project --with transkun --python 3.11 transkun IN OUT --device cpu`),
so this module is import-safe from BOTH the service env and model/.venv (whose
torch deps conflict with Transkun). Parses the output MIDI with pretty_midi.

Returns the exact dict shapes both surfaces already expect:
  notes:  {"pitch": int, "onset": float, "offset": float, "velocity": int}
  pedals: {"time": float, "value": int}   (CC64 >= 64 -> value 127 "on", else 0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pretty_midi


class TranskunError(RuntimeError):
    """Raised when Transkun transcription fails. Never return empty notes on error."""


def midi_to_notes_and_pedals(
    midi_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a Transkun-produced MIDI file into note and pedal-event lists."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    notes: list[dict[str, Any]] = []
    pedals: list[dict[str, Any]] = []  # CC64 parsing added in T2
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append({
                "pitch": int(n.pitch),
                "onset": round(float(n.start), 4),
                "offset": round(float(n.end), 4),
                "velocity": int(n.velocity),
            })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes, pedals
```
(Notes only — the CC64 pedal branch is added in T2 as its own red→green slice.)

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with pytest pytest test_transkun_cli.py::test_notes_carry_pitch_onset_offset_velocity -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/inference/amt/transkun_cli.py apps/inference/amt/test_transkun_cli.py && git commit -m "feat(amt): transkun_cli MIDI note parse with velocity (#128)"
```

---

### Task 2: transkun_cli maps CC64 sustain to pedal events
**Group:** 0 (depends on T1; same files)

**Behavior being verified:** `midi_to_notes_and_pedals` converts CC64 control changes to pedal events with value 127 when raw value ≥ 64, else 0.
**Interface under test:** `transkun_cli.midi_to_notes_and_pedals(midi_path)`

**Files:**
- Modify: `apps/inference/amt/transkun_cli.py`
- Test: `apps/inference/amt/test_transkun_cli.py`

- [x] **Step 1: Write the failing test**

```python
def test_cc64_maps_to_pedal_on_off(tmp_path):
    midi_path = tmp_path / "p.mid"
    _write_midi(
        midi_path,
        notes=[(60, 0.0, 1.0, 80)],
        pedal_ccs=[(0.20, 100), (0.80, 10), (0.90, 64), (1.10, 63)],
    )
    _notes, pedals = transkun_cli.midi_to_notes_and_pedals(midi_path)

    assert [(round(p["time"], 2), p["value"]) for p in pedals] == [
        (0.20, 127),  # 100 >= 64 -> on
        (0.80, 0),    # 10  <  64 -> off
        (0.90, 127),  # 64  >= 64 -> on (boundary)
        (1.10, 0),    # 63  <  64 -> off (boundary)
    ]
    assert all(p["value"] in (0, 127) for p in pedals)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with pytest pytest test_transkun_cli.py::test_cc64_maps_to_pedal_on_off -q
```
Expected: FAIL — T1 returns `pedals == []` (no CC handling yet), so the assert on 4 pedal events fails.

- [x] **Step 3: Implement the minimum to make the test pass** — add the CC64 branch inside the `for inst in pm.instruments` loop of `midi_to_notes_and_pedals`, and sort pedals:

```python
        for cc in inst.control_changes:
            if int(cc.number) != 64:
                continue
            pedals.append({
                "time": round(float(cc.time), 4),
                "value": 127 if int(cc.value) >= 64 else 0,
            })
```
and before the `return`, add `pedals.sort(key=lambda e: e["time"])`.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with pytest pytest test_transkun_cli.py -q
```
Expected: PASS (both tests)

- [x] **Step 5: Commit**

```bash
git add apps/inference/amt/transkun_cli.py apps/inference/amt/test_transkun_cli.py && git commit -m "test(amt): lock CC64>=64 pedal-on boundary in transkun_cli (#128)"
```

---

### Task 3: transkun_cli raises TranskunError on missing input (loud failure)
**Group:** 0 (depends on T2; same files)

**Behavior being verified:** `transcribe_wav` on a nonexistent WAV raises `TranskunError` before spawning any subprocess (fast, no model), never returns empty notes.
**Interface under test:** `transkun_cli.transcribe_wav(path)` and `transkun_cli.TranskunError`

**Files:**
- Modify: `apps/inference/amt/transkun_cli.py`
- Test: `apps/inference/amt/test_transkun_cli.py`

- [x] **Step 1: Write the failing test**

```python
def test_transcribe_wav_missing_input_raises(tmp_path):
    missing = tmp_path / "nope.wav"
    with pytest.raises(transkun_cli.TranskunError):
        transkun_cli.transcribe_wav(missing)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with pytest pytest test_transkun_cli.py::test_transcribe_wav_missing_input_raises -q
```
Expected: FAIL — `AttributeError: module 'transkun_cli' has no attribute 'transcribe_wav'`.

- [x] **Step 3: Implement the minimum to make the test pass** (append to `transkun_cli.py`)

```python
import subprocess
import tempfile

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
_TRANSKUN_TIMEOUT_S = 900


def _run_transkun(in_wav: Path, out_mid: Path) -> None:
    """Shell out to Transkun in an isolated env. Raise TranskunError on any failure."""
    cmd = [
        "uv", "run", "--no-project", "--with", "transkun", "--python", "3.11",
        "transkun", str(in_wav), str(out_mid), "--device", "cpu",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_TRANSKUN_TIMEOUT_S
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise TranskunError(f"transkun subprocess failed to run: {exc}") from exc
    if proc.returncode != 0:
        raise TranskunError(
            f"transkun exited {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', errors='replace')[-2000:]}"
        )
    if not out_mid.exists():
        raise TranskunError(
            f"transkun exited 0 but produced no MIDI at {out_mid}"
        )


def transcribe_wav(
    wav_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe a WAV file to (notes, pedals) via Transkun. Raise TranskunError on failure."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise TranskunError(f"input WAV does not exist: {wav_path}")
    with tempfile.TemporaryDirectory() as td:
        out_mid = Path(td) / "out.mid"
        _run_transkun(wav_path, out_mid)
        return midi_to_notes_and_pedals(out_mid)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with pytest pytest test_transkun_cli.py::test_transcribe_wav_missing_input_raises -q
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/inference/amt/transkun_cli.py apps/inference/amt/test_transkun_cli.py && git commit -m "feat(amt): transkun_cli transcribe_wav + loud TranskunError (#128)"
```

---

### Task 4: transcribe_pcm end-to-end on the real sample WAV (GATED: model)
**Group:** 0 (depends on T3; same files). This slice downloads Transkun weights on first run (~minutes) and is Gate 4's unit-level counterpart.

**Behavior being verified:** `transcribe_pcm(pcm)` writes a temp 16k WAV, runs real Transkun on CPU, and returns non-empty notes each carrying an integer velocity.
**Interface under test:** `transkun_cli.transcribe_pcm(pcm_16k)`

**Files:**
- Modify: `apps/inference/amt/transkun_cli.py`
- Test: `apps/inference/amt/test_transkun_cli.py`
- Fixture (ALREADY COMMITTED, do not regenerate): `apps/inference/amt/fixtures/piano_sample_5s_16k.wav` — a real ~5s mono 16kHz piano clip, force-added past the `*.wav` .gitignore rule as part of the challenge-fix commit. It is tracked, so a fresh checkout/worktree has it. This is the guaranteed-present clip that lets Task 4 (and Gate 4) exercise real Transkun instead of skipping.

- [x] **Step 1: Write the failing test**

```python
def test_transcribe_pcm_on_real_sample_returns_notes_with_velocity():
    """Real Transkun on the committed piano fixture. Slow: downloads weights once.

    The fixture `apps/inference/amt/fixtures/piano_sample_5s_16k.wav` is a real
    ~5s mono 16kHz piano clip committed to the repo (force-added past the `*.wav`
    .gitignore rule), so it is GUARANTEED present in every fresh checkout/worktree.
    This test FAILS HARD (never pytest.skip) when the fixture is absent — a missing
    fixture is a real regression that must break the build, not silently pass.
    """
    wav = Path(__file__).resolve().parent / "fixtures" / "piano_sample_5s_16k.wav"
    if not wav.exists():
        raise AssertionError(
            f"required committed fixture missing: {wav} "
            "(force-add it: git add -f apps/inference/amt/fixtures/piano_sample_5s_16k.wav)"
        )
    y, sr = sf.read(str(wav), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != transkun_cli.SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), transkun_cli.SAMPLE_RATE)
        y = resample_poly(y, transkun_cli.SAMPLE_RATE // g, int(sr) // g).astype("float32")

    notes, pedals = transkun_cli.transcribe_pcm(y)

    assert len(notes) > 0
    assert all(set(n) == {"pitch", "onset", "offset", "velocity"} for n in notes)
    assert all(isinstance(n["velocity"], int) and n["velocity"] > 0 for n in notes)
    assert all(isinstance(p["value"], int) and p["value"] in (0, 127) for p in pedals)
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with scipy --with pytest pytest test_transkun_cli.py::test_transcribe_pcm_on_real_sample_returns_notes_with_velocity -q
```
Expected: FAIL — `AttributeError: module 'transkun_cli' has no attribute 'transcribe_pcm'`.

- [x] **Step 3: Implement the minimum to make the test pass** (append to `transkun_cli.py`)

```python
def transcribe_pcm(
    pcm_16k: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Transcribe a 16kHz mono float32 PCM array to (notes, pedals) via Transkun."""
    pcm = np.ascontiguousarray(np.asarray(pcm_16k, dtype=np.float32))
    if pcm.size == 0:
        raise TranskunError("transcribe_pcm received empty PCM")
    with tempfile.TemporaryDirectory() as td:
        in_wav = Path(td) / "in.wav"
        sf.write(str(in_wav), pcm, SAMPLE_RATE, format="WAV", subtype="FLOAT")
        return transcribe_wav(in_wav)
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with pretty_midi --with numpy --with soundfile --with scipy --with pytest pytest test_transkun_cli.py::test_transcribe_pcm_on_real_sample_returns_notes_with_velocity -q
```
Expected: PASS (real Transkun; first run downloads weights).

- [x] **Step 5: Commit**

```bash
git add apps/inference/amt/transkun_cli.py apps/inference/amt/test_transkun_cli.py && git commit -m "feat(amt): transkun_cli transcribe_pcm end-to-end verified on sample (#128)"
```

---

### Task 5: HTTP response builder produces the frozen transcription_info shape
**Group:** A (depends on Group 0; first in group; sequential T5→T6→T7 on transcription.py)

**Behavior being verified:** `build_response` assembles the exact `/transcribe` response the frozen contract requires (`midi_notes`, `pedal_events`, `transcription_info` with note_count / pitch_range / pedal_event_count / transcription_time_ms / chunk_duration_s).
**Interface under test:** `transcription.build_response(notes, pedals, chunk_duration_s, elapsed_ms)`

**Files:**
- Modify: `apps/inference/amt/transcription.py`
- Test: `apps/inference/amt/test_transcription.py`

- [ ] **Step 1: Write the failing test** (replace the file's contents; the old tests target deleted helpers)

```python
# apps/inference/amt/test_transcription.py
"""Behavior tests for the Transkun-backed EndpointHandler + helpers.

Run: cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi \
        --with fastapi --with pytest pytest test_transcription.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcription


def test_build_response_has_frozen_contract_shape():
    notes = [
        {"pitch": 60, "onset": 0.1, "offset": 0.5, "velocity": 70},
        {"pitch": 72, "onset": 0.2, "offset": 0.6, "velocity": 90},
    ]
    pedals = [{"time": 0.15, "value": 127}]
    resp = transcription.build_response(notes, pedals, chunk_duration_s=15.0, elapsed_ms=42)

    assert resp["midi_notes"] == notes
    assert resp["pedal_events"] == pedals
    info = resp["transcription_info"]
    assert info["note_count"] == 2
    assert info["pitch_range"] == [60, 72]
    assert info["pedal_event_count"] == 1
    assert info["transcription_time_ms"] == 42
    assert info["chunk_duration_s"] == 15.0


def test_build_response_empty_notes_pitch_range_zero():
    resp = transcription.build_response([], [], chunk_duration_s=0.0, elapsed_ms=1)
    assert resp["transcription_info"]["pitch_range"] == [0, 0]
    assert resp["midi_notes"] == []
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi --with fastapi --with pytest pytest test_transcription.py::test_build_response_has_frozen_contract_shape -q
```
Expected: FAIL — `AttributeError: module 'transcription' has no attribute 'build_response'` (the current transcription.py imports aria at module load and has no build_response). If the import itself errors on aria (e.g. aria-amt not installed in the test env), that is expected — this task begins the rewrite, which removes all aria/torch module-level imports and the old class in Step 3.

- [ ] **Step 3: Implement the minimum to make the test pass** — begin the rewrite of `transcription.py`. Replace the entire aria machinery at the top of the file. Keep `decode_webm_to_pcm` (ffmpeg path) unchanged. Add:

  **IMPORT-ORDERING REQUIREMENT (do not defer to Task 7):** Deleting the module-level `import torch` (line 47) and `import amt.config` / `from amt...` imports (lines 55, 89-92) is NOT self-consistent unless the old aria `EndpointHandler` class (currently `class EndpointHandler` at line 324 through the end of the old class, ~line 706) is deleted in THIS task. Its `_transcribe` method carries an `@torch.inference_mode()` decorator (line 568) that is evaluated at CLASS-DEFINITION time — i.e. at `import transcription`. If the old class survives while `torch` is gone from module scope, `import transcription` raises `NameError: name 'torch' is not defined` and Step 4's `build_response` tests cannot even import the module. So in this task: delete the module-level torch/amt imports AND delete the entire old aria `EndpointHandler` class body (lines ~324-706, including `_transcribe`, `_setup_kv_cache`, and every method that references torch/amt). After this task the module has `build_response` (+ the kept `decode_webm_to_pcm`) and NO `EndpointHandler` — the Transkun `EndpointHandler` is ADDED fresh in Task 7. There must be ZERO import-time reference to `torch` or `amt` left. (The `from amt.inference.model import KVCache` on line 409 is a lazy in-method import inside the old class and is removed with the class.)

```python
# apps/inference/amt/transcription.py  (new top-of-file; delete aria imports,
# _AMT_CONFIG, _patched_load_config, _load_weight, midi_dict_to_notes_and_pedals,
# deduplicate_notes, advance_valid_note_groups, _setup_kv_cache, _find_checkpoint,
# AND the entire old aria `EndpointHandler` class — see the import-ordering note above)
from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensure transkun_cli importable
import transkun_cli

SAMPLE_RATE = 16000
FFMPEG_DECODE_TIMEOUT_S = 60
# ... keep decode_webm_to_pcm exactly as-is ...


def build_response(
    notes: list[dict[str, Any]],
    pedals: list[dict[str, Any]],
    chunk_duration_s: float,
    elapsed_ms: int,
) -> dict[str, Any]:
    """Assemble the frozen /transcribe response shape."""
    pitches = [n["pitch"] for n in notes]
    return {
        "midi_notes": notes,
        "pedal_events": pedals,
        "transcription_info": {
            "note_count": len(notes),
            "pitch_range": [min(pitches), max(pitches)] if pitches else [0, 0],
            "pedal_event_count": len(pedals),
            "transcription_time_ms": int(elapsed_ms),
            "chunk_duration_s": round(float(chunk_duration_s), 2),
        },
    }
```
(Add `import sys` near the top with the other imports.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi --with fastapi --with pytest pytest test_transcription.py -q
```
Expected: PASS (both build_response tests)

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt/transcription.py apps/inference/amt/test_transcription.py && git commit -m "feat(amt): build_response frozen contract shape; drop aria machinery (#128)"
```

---

### Task 6: EndpointHandler refuses to start when no Transkun path resolves
**Group:** A (depends on T5; same file transcription.py)

**Behavior being verified:** `resolve_transcriber()` returns a callable when a Transkun path is available, and raises `RuntimeError` (refuse to start) when NEITHER a warm in-process `transkun` import NOR the `uv` CLI is available — never a silent per-request fallback.
**Interface under test:** `transcription.resolve_transcriber()`

**Files:**
- Modify: `apps/inference/amt/transcription.py`
- Test: `apps/inference/amt/test_transcription.py`

- [ ] **Step 1: Write the failing test**

```python
def test_resolve_transcriber_refuses_when_no_path(monkeypatch):
    # Force the warm import to fail AND the CLI probe to fail.
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda _cmd: None)
    with pytest.raises(RuntimeError):
        transcription.resolve_transcriber()


def test_resolve_transcriber_falls_back_to_cli(monkeypatch):
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda cmd: "/usr/bin/uv")
    fn = transcription.resolve_transcriber()
    assert fn is transkun_cli.transcribe_pcm
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi --with fastapi --with pytest pytest test_transcription.py::test_resolve_transcriber_refuses_when_no_path -q
```
Expected: FAIL — `AttributeError: module 'transcription' has no attribute 'resolve_transcriber'`.

- [ ] **Step 3: Implement the minimum to make the test pass** (add to `transcription.py`)

```python
def _import_warm_transcriber() -> Callable[[np.ndarray], tuple[list, list]] | None:
    """Return an in-process transkun PCM->(notes,pedals) callable if transkun is
    importable in THIS env (warm, load-once), else None. No silent failure: any
    import problem returns None so the caller falls back to the CLI explicitly."""
    try:
        import transkun  # noqa: F401
    except ImportError:
        return None

    def _warm(pcm_16k: np.ndarray) -> tuple[list, list]:
        # Reuse the CLI helper's WAV+MIDI parse contract by writing a temp WAV and
        # invoking transkun's Python API. Kept minimal; the shared parse guarantees
        # identical output shape to the CLI path.
        import numpy as _np
        import soundfile as _sf
        pcm = _np.ascontiguousarray(_np.asarray(pcm_16k, dtype=_np.float32))
        if pcm.size == 0:
            raise transkun_cli.TranskunError("warm transcribe received empty PCM")
        with tempfile.TemporaryDirectory() as td:
            in_wav = Path(td) / "in.wav"
            out_mid = Path(td) / "out.mid"
            _sf.write(str(in_wav), pcm, transkun_cli.SAMPLE_RATE, format="WAV", subtype="FLOAT")
            transkun.transcribe.transcribe(str(in_wav), str(out_mid), device="cpu")
            return transkun_cli.midi_to_notes_and_pedals(out_mid)

    return _warm


def resolve_transcriber() -> Callable[[np.ndarray], tuple[list, list]]:
    """Resolve ONE transcriber at init. Prefer warm in-process transkun; else the
    CLI helper (requires `uv`); else raise so the service refuses to start."""
    warm = _import_warm_transcriber()
    if warm is not None:
        return warm
    if shutil.which("uv") is not None:
        return transkun_cli.transcribe_pcm
    raise RuntimeError(
        "No Transkun path available: transkun is not importable and `uv` is not on PATH."
    )
```
Note: `transkun.transcribe.transcribe(...)`'s exact Python-API signature must be confirmed against the installed package during build; if it differs, keep the warm path but adapt the call — the FALLBACK CLI path is the guaranteed one and is what the tests pin. If confirming the warm signature is not quick, return `None` from `_import_warm_transcriber` unconditionally (CLI-only) and record "warm in-process transkun path deferred" as a #128 follow-up. The refuse-to-start + CLI-fallback contract (the tested behavior) holds either way.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi --with fastapi --with pytest pytest test_transcription.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt/transcription.py apps/inference/amt/test_transcription.py && git commit -m "feat(amt): resolve_transcriber refuse-to-start + CLI fallback (#128)"
```

---

### Task 7: EndpointHandler decodes, transcribes, ignores context_audio, returns contract
**Group:** A (depends on T6; same file transcription.py)

**Behavior being verified:** `EndpointHandler.__call__` returns a `MISSING_CHUNK_AUDIO` error body when `chunk_audio` is absent (fast, no model), and its `__init__` wires a resolved transcriber without loading any aria model. `context_audio` is accepted but never concatenated.
**Interface under test:** `transcription.EndpointHandler(path).__call__(data)`

**Files:**
- Modify: `apps/inference/amt/transcription.py`
- Test: `apps/inference/amt/test_transcription.py`

- [ ] **Step 1: Write the failing test**

```python
def test_handler_missing_chunk_returns_error_body(monkeypatch):
    # CLI-fallback resolution keeps __init__ fast (no model load).
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda cmd: "/usr/bin/uv")
    handler = transcription.EndpointHandler(path="")
    out = handler({"inputs": {}})
    assert out["error"]["code"] == "MISSING_CHUNK_AUDIO"


def test_handler_transcribes_chunk_and_ignores_context(monkeypatch):
    monkeypatch.setattr(transcription, "_import_warm_transcriber", lambda: None)
    monkeypatch.setattr(transcription.shutil, "which", lambda cmd: "/usr/bin/uv")

    seen = {}
    def _fake_transcribe(pcm):
        seen["len"] = len(pcm)
        return ([{"pitch": 60, "onset": 0.0, "offset": 0.5, "velocity": 80}],
                [{"time": 0.1, "value": 127}])

    handler = transcription.EndpointHandler(path="")
    handler._transcribe_fn = _fake_transcribe
    monkeypatch.setattr(transcription, "decode_webm_to_pcm",
                        lambda b: np.zeros(transcription.SAMPLE_RATE, dtype=np.float32))

    out = handler({"inputs": {"chunk_audio": base64.b64encode(b"x").decode(),
                              "context_audio": base64.b64encode(b"y").decode()}})
    # context_audio must NOT be concatenated: transcriber sees exactly 1s of chunk PCM.
    assert seen["len"] == transcription.SAMPLE_RATE
    assert out["transcription_info"]["note_count"] == 1
    assert out["pedal_events"] == [{"time": 0.1, "value": 127}]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi --with fastapi --with pytest pytest test_transcription.py::test_handler_missing_chunk_returns_error_body -q
```
Expected: FAIL — `AttributeError: module 'transcription' has no attribute 'EndpointHandler'` (the old aria `EndpointHandler` was already deleted in Task 5; this task adds the new Transkun-backed one).

- [ ] **Step 3: Implement the minimum to make the test pass** — ADD the new Transkun-backed `EndpointHandler` class (the old aria `EndpointHandler` was deleted in Task 5, so this is a fresh class, not a body replacement):

```python
class EndpointHandler:
    """Transkun-backed transcription handler. Frozen /transcribe contract."""

    def __init__(self, path: str = "") -> None:
        # `path` retained for call-site compatibility; Transkun weights are managed
        # by the transkun package itself. Resolve the transcriber once (refuse to
        # start if none available).
        self._transcribe_fn = resolve_transcriber()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        start_time = time.time()
        try:
            inputs = data.get("inputs", data)
            if isinstance(inputs, str):
                inputs = {"chunk_audio": inputs}

            chunk_audio_b64 = inputs.get("chunk_audio")
            if not chunk_audio_b64:
                return {"error": {"code": "MISSING_CHUNK_AUDIO",
                                  "message": "chunk_audio field is required"}}

            chunk_pcm = decode_webm_to_pcm(base64.b64decode(chunk_audio_b64))
            chunk_duration_s = len(chunk_pcm) / SAMPLE_RATE
            # context_audio is accepted but IGNORED (Transkun is whole-piece; the
            # aria overlap/dedup semantics are gone). No concatenation.

            notes, pedals = self._transcribe_fn(chunk_pcm)
            elapsed_ms = int((time.time() - start_time) * 1000)
            return build_response(notes, pedals, chunk_duration_s, elapsed_ms)
        except Exception as e:
            return {"error": {"code": "TRANSCRIPTION_ERROR",
                              "message": str(e), "traceback": traceback.format_exc()}}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with numpy --with soundfile --with pretty_midi --with fastapi --with pytest pytest test_transcription.py -q
```
Expected: PASS (all transcription tests)

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt/transcription.py apps/inference/amt/test_transcription.py && git commit -m "feat(amt): Transkun EndpointHandler, context_audio ignored (#128)"
```

---

### Task 8: Local dev AMT server boots on Transkun and reports it at /health
**Group:** A (depends on T7; parallel with T9 — different file)

**Behavior being verified:** `amt_local_server.py`'s FastAPI app serves `/health` reporting `model == "transkun"`, and its `/// deps` no longer pull aria git deps.
**Interface under test:** `GET /health` on the FastAPI app.

**Files:**
- Modify: `apps/inference/amt/amt_local_server.py`
- Test: `apps/inference/amt/test_amt_local_server.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# apps/inference/amt/test_amt_local_server.py
"""Run: cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy \
        --with soundfile --with pretty_midi --with pytest pytest test_amt_local_server.py"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import amt_local_server


def test_health_reports_transkun_before_model_load():
    client = TestClient(amt_local_server.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "transkun"
    assert body["loaded"] is False  # _handler not initialized in-process
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy --with soundfile --with pretty_midi --with pytest pytest test_amt_local_server.py -q
```
Expected: FAIL — health returns `model == "aria-amt"`.

- [ ] **Step 3: Implement the minimum to make the test pass** — edit `amt_local_server.py`:
  - Replace the `/// script` dependency block:
    ```python
    # /// script
    # requires-python = ">=3.11"
    # dependencies = [
    #     "fastapi>=0.115.0",
    #     "uvicorn>=0.34.0",
    #     "numpy>=1.24.0",
    #     "soundfile>=0.12.0",
    #     "pretty_midi>=0.2.10",
    #     "transkun>=2.0.1",
    # ]
    # ///
    ```
  - Change the health handler: `return {"status": "ok", "model": "transkun", "loaded": _handler is not None}`.
  - Leave the `EndpointHandler` delegation in `_init_model`/`/transcribe` intact (it now resolves Transkun). Update the module docstring "Aria-AMT" → "Transkun".

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy --with soundfile --with pretty_midi --with pytest pytest test_amt_local_server.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt/amt_local_server.py apps/inference/amt/test_amt_local_server.py && git commit -m "feat(amt): local dev server on Transkun; /health reports transkun (#128)"
```

---

### Task 9: Container server delegates to Transkun and drops the ONNX split
**Group:** A (depends on T7; parallel with T8 — different file)

**Behavior being verified:** container `server.py`'s `/health` returns the healthy shape with no ONNX/aria decoder loaded; its transcription path delegates to the Transkun handler.
**Interface under test:** `GET /health` on the container FastAPI app.

**Files:**
- Modify: `apps/inference/amt/server.py`
- Test: `apps/inference/amt/test_server.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# apps/inference/amt/test_server.py
"""Run: cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy \
        --with soundfile --with pretty_midi --with pytest pytest test_server.py"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server


def test_health_shape_no_onnx():
    # Do not trigger lifespan model load; hit the route function directly.
    client = TestClient(server.app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "inference_count" in body
    # ONNX globals must be gone from the module surface.
    assert not hasattr(server, "_encoder_onnx")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy --with soundfile --with pretty_midi --with onnxruntime --with pytest pytest test_server.py -q
```
Expected: FAIL — `server` module imports `onnxruntime` / aria at load, or `_encoder_onnx` still exists.

- [ ] **Step 3: Implement the minimum to make the test pass** — rewrite `server.py`:
  - Delete: the `import onnxruntime as ort`, `import torch`, the aria `_amt_config`/`_patched_load_config` block, all aria `from amt...` imports, `from transcription import _load_weight, advance_valid_note_groups`, `_setup_kv_cache_for_decoder`, `load_models`, the ONNX `transcribe(...)`, `midi_dict_to_notes_and_pedals`, `deduplicate_notes`, and the `_encoder_onnx`/`_decoder`/`_audio_transform`/`_tokenizer` globals.
  - Keep the FastAPI app + `/transcribe` + `/health` routes, `decode_webm_to_pcm`, `_inference_count`, `_start_time`.
  - Replace transcription with the shared handler:
    ```python
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transcription import EndpointHandler, decode_webm_to_pcm, build_response  # noqa: E402

    _handler: EndpointHandler | None = None

    @asynccontextmanager
    async def lifespan(application):
        global _start_time, _handler
        _start_time = time.time()
        _handler = EndpointHandler(path=os.environ.get("CHECKPOINT_PATH", ""))
        yield

    # /transcribe: decode chunk (ignore context_audio), call _handler, return its body.
    # /health: {"status": "healthy", "model_loaded": _handler is not None,
    #           "inference_count": _inference_count, "uptime_s": ...}
    ```
  - `/health` must not require `_encoder_onnx`; report `model_loaded = _handler is not None`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with fastapi --with httpx --with numpy --with soundfile --with pretty_midi --with pytest pytest test_server.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt/server.py apps/inference/amt/test_server.py && git commit -m "feat(amt): container server delegates to Transkun; drop ONNX split (#128)"
```

---

### Task 10: gd_rate window transcriber swaps to transkun_cli (injectable callable)
**Group:** B (depends on Group 0; parallel with T11 — different file)

**Behavior being verified:** `_transcribe_windows` transcribes each window through an injected callable, offsets note/pedal times to clip-relative, and pools them; the produced bundle carries the Transkun substrate string.
**Interface under test:** `transcribe_bundles._transcribe_windows(transcribe, audio, starts)` and `_build_bundle(...)`

**Files:**
- Modify: `model/src/claim_measurement/gd_rate/transcribe_bundles.py`
- Test: `model/src/claim_measurement/gd_rate/test_transcribe_bundles.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# model/src/claim_measurement/gd_rate/test_transcribe_bundles.py
"""Run: cd model && uv run --with numpy --with pytest pytest \
        src/claim_measurement/gd_rate/test_transcribe_bundles.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transcribe_bundles as tb


def test_transcribe_windows_offsets_and_pools_via_injected_callable():
    calls = []
    def fake_transcribe(pcm):
        calls.append(len(pcm))
        return ([{"pitch": 60, "onset": 1.0, "offset": 1.5, "velocity": 70}],
                [{"time": 0.5, "value": 127}])

    audio = np.zeros(int(200 * tb.SAMPLE_RATE), dtype=np.float32)
    notes, pedals = tb._transcribe_windows(fake_transcribe, audio, [0.0, 30.0])

    assert len(calls) == 2
    # window starts 0.0 and 30.0 -> note onsets 1.0 and 31.0
    assert sorted(round(n["onset"], 1) for n in notes) == [1.0, 31.0]
    assert sorted(round(p["time"], 1) for p in pedals) == [0.5, 30.5]


def test_bundle_records_transkun_substrate():
    b = tb._build_bundle("chopin_ballade_1", "rid", notes=[], pedal_events=[],
                        duration_sec=10.0, window_starts=[0.0])
    assert b["substrate_versions"]["amt"].startswith("transkun/")
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run --with numpy --with pytest pytest src/claim_measurement/gd_rate/test_transcribe_bundles.py -q
```
Expected: FAIL — `_transcribe_windows` currently takes `handler` and calls `handler._transcribe`; and substrate is `aria-amt/...`.

- [ ] **Step 3: Implement the minimum to make the test pass** — edit `transcribe_bundles.py`:
  - Change `_transcribe_windows(handler, audio, starts)` signature to `_transcribe_windows(transcribe, audio, starts)`; replace `notes, pedals = handler._transcribe(pcm)` with `notes, pedals = transcribe(pcm)`.
  - In `_build_bundle`, change `"substrate_versions": {"amt": "aria-amt/piano-medium-double-1.0"}` → `{"amt": "transkun/2.0.1"}`.
  - In `main`: delete `sys.path.insert(0, REPO/"apps/inference/amt")` + `from transcription import EndpointHandler` + `handler = EndpointHandler(...)`; instead `sys.path.insert(0, str(REPO / "apps/inference/amt")); from transkun_cli import transcribe_pcm`; change the call site `notes, pedals = _transcribe_windows(handler, audio, starts)` → `_transcribe_windows(transcribe_pcm, audio, starts)`; drop the `--weights` reliance for model loading (keep the arg for back-compat or remove; if removed, delete its `add_argument`).
  - Replace the `/// script` deps: drop the two aria `git+` lines; keep `numpy`, `soundfile`, `scipy`, `pretty_midi`; drop `torch`/`safetensors`/`numba`/`llvmlite` (no longer needed). Update the "Truth-label purity: aria-amt is a non-LLM transcription model" line → "Transkun".

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run --with numpy --with pytest pytest src/claim_measurement/gd_rate/test_transcribe_bundles.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/gd_rate/transcribe_bundles.py model/src/claim_measurement/gd_rate/test_transcribe_bundles.py && git commit -m "feat(measure): gd_rate windows via transkun_cli; transkun substrate (#128)"
```

---

### Task 11: Remaining measurer render scripts swap EndpointHandler → transkun_cli
**Group:** B (depends on Group 0; parallel with T10 — different files)

**Behavior being verified:** no measurer render script imports `EndpointHandler` or calls `handler._transcribe` any more; each uses `transkun_cli.transcribe_pcm`.
**Interface under test:** static contract of the swapped files (guard test).

**Files:**
- Modify: `model/src/claim_measurement/ga_validation/amt_dynamics_ga_render.py`, `ga_validation/amt_dynamics_gb_gate.py`, `ga_validation/amt_pedaling_ga_render.py`, `gc_error_bars/gc_dynamics_render.py`, `tau_calibration/tau_pedaling_render.py`, `amt_fidelity/onset_duration_render.py`, `dynamics_supply/render_percepiano_bundles.py`, `apps/inference/extract_amt_midi.py`
- Test: `model/src/claim_measurement/test_no_aria_amt_handler.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# model/src/claim_measurement/test_no_aria_amt_handler.py
"""Guard: no measurer render script depends on the deleted aria EndpointHandler.
Run: cd model && uv run --with pytest pytest src/claim_measurement/test_no_aria_amt_handler.py"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SWAPPED = [
    "model/src/claim_measurement/ga_validation/amt_dynamics_ga_render.py",
    "model/src/claim_measurement/ga_validation/amt_dynamics_gb_gate.py",
    "model/src/claim_measurement/ga_validation/amt_pedaling_ga_render.py",
    "model/src/claim_measurement/gc_error_bars/gc_dynamics_render.py",
    "model/src/claim_measurement/tau_calibration/tau_pedaling_render.py",
    "model/src/claim_measurement/amt_fidelity/onset_duration_render.py",
    "model/src/claim_measurement/dynamics_supply/render_percepiano_bundles.py",
    "apps/inference/extract_amt_midi.py",
]


def test_no_endpointhandler_import_remains():
    offenders = []
    for rel in SWAPPED:
        text = (REPO / rel).read_text()
        if "EndpointHandler" in text or "handler._transcribe" in text:
            offenders.append(rel)
    assert offenders == [], f"still on aria handler: {offenders}"


def test_all_use_transkun_cli():
    missing = [rel for rel in SWAPPED
               if "transkun_cli" not in (REPO / rel).read_text()]
    assert missing == [], f"not swapped to transkun_cli: {missing}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run --with pytest pytest src/claim_measurement/test_no_aria_amt_handler.py -q
```
Expected: FAIL — every listed file still references `EndpointHandler`.

- [ ] **Step 3: Implement the minimum to make the test pass** — in EACH listed file apply the same mechanical swap:
  - Delete the `sys.path.insert(... "apps/inference/amt")` + `from transcription import EndpointHandler` + `handler = EndpointHandler(path=...)` lines.
  - Add `sys.path.insert(0, str(<REPO>/ "apps/inference/amt")); from transkun_cli import transcribe_pcm` (use each file's existing repo-root anchor; e.g. `Path(__file__).resolve().parents[N]`).
  - Replace every `handler._transcribe(pcm)` / `handler._transcribe(audio)` call with `transcribe_pcm(pcm)` / `transcribe_pcm(audio)`.
  - In `dynamics_supply/render_percepiano_bundles.py`, also change `"substrate_versions": {"amt": "aria-amt/piano-medium-double-1.0"}` → `{"amt": "transkun/2.0.1"}`.
  - Update each `/// script` deps block: drop the aria `git+` lines and `torch`/`safetensors`; keep/ add `numpy`, `soundfile`, `pretty_midi`, `scipy` as each file needs. `extract_amt_midi.py` transcribes WAV files — it may call `transkun_cli.transcribe_wav` (path) or load+`transcribe_pcm`; use `transcribe_wav` on the WAV path and keep its pretty_midi serialization.
  - Replace any "aria-amt" prose in docstrings/comments in these files with "Transkun".
  - **`onset_duration_render.py` — TRUNCATION RISK, verify number-neutrality at Gate 5 (do NOT redesign here):** this file has `AMT_WINDOW_S = 30.0 # aria-amt _transcribe hard-truncates to this` and passes the FULL clip audio to `handler._transcribe(audio)`, relying on aria's IMPLICIT 30s truncation. `transkun_cli.transcribe_pcm` does NOT truncate — it transcribes the whole clip. The explicit `crop(..., cutoff)` after transcription is expected to preserve the metric, but the mechanical swap does not prove it. After swapping to `transcribe_pcm`, keep an explicit 30s cap in this script (slice the PCM to `AMT_WINDOW_S * SAMPLE_RATE` before calling `transcribe_pcm`) so behavior stays equivalent to aria's implicit cap, and add a code comment noting the cap is now explicit because Transkun does not self-truncate. Flag this file for Gate 5: its rendered numbers MUST be confirmed number-neutral vs the aria baseline (any drift is a real regression, not metadata).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run --with pytest pytest src/claim_measurement/test_no_aria_amt_handler.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/claim_measurement/ga_validation model/src/claim_measurement/gc_error_bars model/src/claim_measurement/tau_calibration model/src/claim_measurement/amt_fidelity model/src/claim_measurement/dynamics_supply/render_percepiano_bundles.py apps/inference/extract_amt_midi.py model/src/claim_measurement/test_no_aria_amt_handler.py && git commit -m "feat(measure): swap all render scripts to transkun_cli (#128)"
```

---

### Task 12: Chroma pseudo-truth drops the Aria re-onset deduper
**Group:** C (depends on Group 0)

**Behavior being verified:** two same-pitch notes 0.05s apart are BOTH kept (Transkun does not emit the aria re-onset artifact, so no merging).
**Interface under test:** `amt_regen._dedup_amt_notes(notes)`

**Files:**
- Modify: `model/src/chroma_dtw_eval/amt_regen.py`
- Test: `model/src/chroma_dtw_eval/test_dedup_amt_notes.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# model/src/chroma_dtw_eval/test_dedup_amt_notes.py
"""Run: cd model && uv run --with numpy --with pytest pytest \
        src/chroma_dtw_eval/test_dedup_amt_notes.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chroma_dtw_eval import amt_regen


def test_same_pitch_close_onsets_are_not_merged():
    notes = [
        {"pitch": 60, "onset": 1.00, "offset": 1.20, "velocity": 70},
        {"pitch": 60, "onset": 1.05, "offset": 1.25, "velocity": 72},  # 50ms later
    ]
    out = amt_regen._dedup_amt_notes(notes)
    assert len(out) == 2  # Transkun has no re-onset artifact; keep both
    assert sorted(round(n["onset"], 2) for n in out) == [1.00, 1.05]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run --with numpy --with pytest pytest src/chroma_dtw_eval/test_dedup_amt_notes.py -q
```
Expected: FAIL — current `_dedup_amt_notes` merges the two (returns 1) because `DEDUP_WINDOW_S=0.08 > 0.05`.

- [ ] **Step 3: Implement the minimum to make the test pass** — in `amt_regen.py`:
  - Set `DEDUP_WINDOW_S = 0.0` and update its comment to: "Transkun does not emit the aria same-pitch re-onset artifact, so no merging is applied (window 0.0). Retained as a pass-through so the pipeline shape is unchanged; #128."
  - With `window_s = 0.0`, the `< window_s` guard is never true, so `_dedup_amt_notes` becomes an order-normalizing pass-through. Leave the function body otherwise intact.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run --with numpy --with pytest pytest src/chroma_dtw_eval/test_dedup_amt_notes.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/chroma_dtw_eval/amt_regen.py model/src/chroma_dtw_eval/test_dedup_amt_notes.py && git commit -m "feat(chroma): drop aria re-onset dedup for Transkun (#128)"
```

---

### Task 13: amt_version.json pins Transkun
**Group:** D (independent; parallel with T14)

**Behavior being verified:** the pseudo-truth version config names Transkun, changing the cache key so stale aria pseudo-truth is invalidated and regenerated by Gate 1.
**Interface under test:** `model/config/amt_version.json` contents.

**Files:**
- Modify: `model/config/amt_version.json`
- Test: `model/src/chroma_dtw_eval/test_amt_version_config.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# model/src/chroma_dtw_eval/test_amt_version_config.py
"""Run: cd model && uv run --with pytest pytest src/chroma_dtw_eval/test_amt_version_config.py"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[2] / "config/amt_version.json"


def test_amt_version_names_transkun():
    body = json.loads(CONFIG.read_text())
    assert body["model_name"] == "transkun"
    assert body["regen_source_default"] == "local:transkun"
    assert "transkun" in body["checkpoint_hash"]
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd model && uv run --with pytest pytest src/chroma_dtw_eval/test_amt_version_config.py -q
```
Expected: FAIL — `model_name == "aria-amt"`.

- [ ] **Step 3: Implement the minimum to make the test pass** — probe the installed version, then write the config:

```bash
# get the real installed version (should be 2.0.1)
uv run --no-project --with transkun --python 3.11 python -c "import importlib.metadata as m; print(m.version('transkun'))"
```
Set `model/config/amt_version.json`:
```json
{
  "checkpoint_hash": "transkun_v2.0.1_2026_07_23",
  "parangonar_version": "3.3.2",
  "regen_source_default": "local:transkun",
  "model_name": "transkun",
  "pinned_at": "2026-07-23",
  "notes": "checkpoint_hash is a stable label, not a cryptographic digest; bump when the Transkun version changes. parangonar_version must match the installed parangonar package."
}
```
(Use the actually-probed version in the label/hash if it differs from 2.0.1.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd model && uv run --with pytest pytest src/chroma_dtw_eval/test_amt_version_config.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/config/amt_version.json model/src/chroma_dtw_eval/test_amt_version_config.py && git commit -m "feat(config): pin AMT version to Transkun 2.0.1 (#128)"
```

---

### Task 14: Docs + Justfile prose updated aria-amt → Transkun
**Group:** D (independent; parallel with T13)

**Behavior being verified:** the transcriber-substrate prose across docs + Justfile recipe comments names Transkun, not aria-amt. (Docs-only task; verification is a grep guard, not a behavior test — noted per plan rules since no runtime behavior changes.)
**Interface under test:** file contents (grep guard test).

**Files:**
- Modify: `docs/apps/07-evaluation.md`, `docs/model/01-data.md`, `docs/model/claim-verifier-signed-d-conventions.md`, `docs/architecture.md`, `CLAUDE.md`, `Justfile`
- Test: `apps/inference/amt/test_docs_transkun.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# apps/inference/amt/test_docs_transkun.py
"""Guard: transcriber docs name Transkun. Run: cd apps/inference/amt && \
        uv run --with pytest pytest test_docs_transkun.py"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = [
    "docs/apps/07-evaluation.md",
    "docs/model/01-data.md",
    "docs/architecture.md",
]


def test_docs_mention_transkun():
    missing = [d for d in DOCS if "ranskun" not in (REPO / d).read_text()]
    assert missing == [], f"docs not updated to Transkun: {missing}"
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with pytest pytest test_docs_transkun.py -q
```
Expected: FAIL — none of the three docs mention Transkun yet.

- [ ] **Step 3: Implement the minimum to make the test pass** — edit the prose in each file:
  - `docs/apps/07-evaluation.md`, `docs/model/01-data.md`, `docs/model/claim-verifier-signed-d-conventions.md`, `docs/architecture.md`: where the pipeline describes the AMT transcriber / "aria-amt" as the non-LLM transcription substrate, change to "Transkun (MIT, ISMIR 2024)". Where truth-label-purity notes say "aria-amt is a non-LLM transcription model", change the model name to Transkun.
  - `CLAUDE.md` Model Strategy: clarify that the AMT MIDI feeding the symbolic (Aria encoder) stream + MPM features is produced by Transkun. Do NOT confuse the transcriber (Transkun) with the Aria 650M symbolic ENCODER — only the transcriber changed.
  - `Justfile`: in the `amt`, `amt-extract`, `amt-run`, `catalog-pieceid-amt-axis` recipe comments, replace "Aria-AMT"/"aria-amt" with "Transkun". Do NOT change recipe NAMES (the `amt` prefix stays; consumers rely on it).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with pytest pytest test_docs_transkun.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/apps/07-evaluation.md docs/model/01-data.md docs/model/claim-verifier-signed-d-conventions.md docs/architecture.md CLAUDE.md Justfile apps/inference/amt/test_docs_transkun.py && git commit -m "docs: aria-amt transcriber -> Transkun across docs + Justfile (#128)"
```

---

### Task 15: Delete ONNX export machinery + Dockerfile ONNX stage
**Group:** E (depends on Group A + T11 — nothing imports the ONNX path once server.py is rewritten)

**Behavior being verified:** the ONNX exporter is gone and the Dockerfile no longer builds/copies it; `audio_chunker.py` is untouched (has live importers).
**Interface under test:** repository file state (guard test).

**Files:**
- Delete: `apps/inference/amt/scripts/export_onnx.py`
- Modify: `apps/inference/amt/Dockerfile`
- Test: `apps/inference/amt/test_onnx_removed.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# apps/inference/amt/test_onnx_removed.py
"""Run: cd apps/inference/amt && uv run --with pytest pytest test_onnx_removed.py"""
from __future__ import annotations

from pathlib import Path

AMT = Path(__file__).resolve().parent
REPO = AMT.parents[2]


def test_export_onnx_deleted():
    assert not (AMT / "scripts/export_onnx.py").exists()


def test_dockerfile_has_no_onnx():
    df = (AMT / "Dockerfile").read_text()
    assert "export_onnx" not in df
    assert "onnxruntime" not in df


def test_audio_chunker_preserved():
    # audio_chunker has live importers (MuQ path) and must NOT be deleted.
    assert (REPO / "apps/inference/audio_chunker.py").exists()
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/inference/amt && uv run --with pytest pytest test_onnx_removed.py -q
```
Expected: FAIL — `export_onnx.py` still exists; Dockerfile still references it.

- [ ] **Step 3: Implement the minimum to make the test pass**
  - `git rm apps/inference/amt/scripts/export_onnx.py` (and remove the now-empty `scripts/` dir if empty).
  - Rewrite `apps/inference/amt/Dockerfile`: remove the builder stage that installs `onnx`/`onnxruntime` and runs `scripts/export_onnx.py`; remove `COPY scripts/export_onnx.py` and `COPY --from=builder /build/onnx_models/ /app/models/`. The runtime stage installs `transkun` (+ `pretty_midi`, `soundfile`, `numpy`, `fastapi`, `uvicorn`, `ffmpeg`) and runs `CMD ["python", "server.py"]`. Keep the `checkpoint.safetensors` COPY only if still referenced; since the aria decoder is gone, drop it and the `CHECKPOINT_PATH`/`MODEL_DIR` env that only fed the ONNX/decoder path.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/inference/amt && uv run --with pytest pytest test_onnx_removed.py -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/inference/amt/Dockerfile apps/inference/amt/scripts apps/inference/amt/test_onnx_removed.py && git commit -m "chore(amt): delete ONNX export machinery + Dockerfile stage (#128)"
```

---

### Task 16: Gate 4 — live service smoke on a real clip
**Group:** F (depends on A+B+C+D+E; first gate)

**Behavior being verified:** the swapped local service transcribes a real clip and returns velocity + pedals + notes.
**Interface under test:** `just amt` + `smoke_test_amt.py`

**Files:**
- Modify: `apps/inference/smoke_test_amt.py` (drop aria `git+` dep from `/// deps`; point `DEFAULT_WAV` at the committed fixture; keep the `{"chunk_audio": ..., "context_audio": None}` call — contract unchanged)

- [ ] **Step 1:** Update `smoke_test_amt.py`:
  - `/// script` deps: drop `aria-amt @ git+...` and `torch`/`safetensors`; keep `numpy`; add `soundfile`, `pretty_midi`. The `run_amt` path constructs `EndpointHandler` and calls it — unchanged (now Transkun-backed).
  - Repoint `DEFAULT_WAV` from the gitignored, untracked `Beethoven_WoO80_var27_8bars_3_15.wav` to the GUARANTEED-PRESENT committed fixture:
    ```python
    DEFAULT_WAV = str(Path(__file__).resolve().parent / "amt" / "fixtures" / "piano_sample_5s_16k.wav")
    ```
    (`smoke_test_amt.py` lives in `apps/inference/`, so `parent / "amt" / "fixtures" / ...` resolves to `apps/inference/amt/fixtures/piano_sample_5s_16k.wav`.) The existing `main()` already `sys.exit(1)`s when `args.wav` is absent — with the fixture tracked, the default now resolves in every fresh checkout, so the gate genuinely runs instead of exiting early. Do NOT relax that existence check.
  - Transkun manages its own weights (there is no aria checkpoint dir): DELETE the `DEFAULT_CHECKPOINT_DIR` existence gate in `main()` (the `if not Path(args.checkpoint).exists(): sys.exit(1)` block, and its aria-amt-weights default) so the smoke does not falsely abort on a missing aria weights dir. Keep passing whatever `path` `EndpointHandler` needs (it ignores it). The assertions on notes/velocity/pedals remain — the gate is not weakened.
- [ ] **Step 2: Run the smoke**

```bash
cd apps/inference && uv run smoke_test_amt.py
```
Expected: prints `Notes: <N>0`, `Pedal events: >=0`, first notes show integer velocity > 0. FAIL loudly if the service errors.

- [ ] **Step 3:** Boot the server and hit it over HTTP:

```bash
just amt   # starts amt_local_server on :8001
# in another shell:
curl -s localhost:8001/health   # -> {"model":"transkun","loaded":true}
```
Expected: `/health` reports `transkun`, `loaded:true`.

- [ ] **Step 4: Commit**

```bash
git add apps/inference/smoke_test_amt.py && git commit -m "test(amt): Gate 4 smoke on Transkun service (#128)"
```

---

### Task 17: Gate 3 — claim-verifier + model measurer tests green
**Group:** F (depends on T16)

**Behavior being verified:** the onset/dynamics/pedaling measurer suites and the model pedal-threading suites still pass (bundle shape unchanged by the swap).
**Interface under test:** existing pytest suites.

- [ ] **Step 1: Run the claim-taxonomy measurer suites**

```bash
cd apps/evals && uv run --with pytest pytest claim_taxonomy/tests/test_onset_deviation_measurer.py claim_taxonomy/tests/test_dynamics_measurer.py claim_taxonomy/tests/test_pedaling_measurer.py claim_taxonomy/tests/test_onset_deviation_integration.py -q
```
Expected: PASS (these use synthetic bundle fixtures; substrate-string change is metadata-only).

- [ ] **Step 2: Run the model measurer suites**

```bash
cd model && uv run --with numpy --with pytest pytest src/claim_measurement/tests/test_pedal_threading.py src/claim_measurement/tests/test_extract_bundle_pedals.py -q
```
Expected: PASS. If `test_pedal_threading` asserts aria-specific dedup behavior, update the expectation to the Transkun pass-through and record the change in the commit body.

- [ ] **Step 3: Commit** (only if a test expectation legitimately changed)

```bash
git commit -am "test(measure): Gate 3 measurer suites green under Transkun (#128)"
```

---

### Task 18: Gate 1 — chroma pseudo-truth regen + recall not regressed
**Group:** F (depends on T17). REQUIRES rehydrated practice audio + AMT service up.

**Behavior being verified:** regenerated Transkun pseudo-truth yields piece-ID recall not worse than the aria baseline.
**Interface under test:** `just amt-regen-pseudo-truth` + `just chroma-eval-verify`

- [ ] **Step 1:** Bring up the service (`just amt`) and regenerate pseudo-truth (the Transkun `checkpoint_hash` change invalidates the aria cache, forcing regen):

```bash
just amt-regen-pseudo-truth <piece> <video_id>   # for each verified piece/video
```
- [ ] **Step 2:** Run the full chroma eval:

```bash
just chroma-eval-verify
```
Expected: piece-ID recall  aria baseline in `baseline.json`.
- [ ] **Step 3 (conditional retune):** If recall regressed AND the `LowCoverageError` gate is the cause (Transkun's ~7% fewer notes push `n_anchors < MIN_ANCHORS` or `max_gap > MAX_ANCHOR_GAP_S`), lower `MIN_ANCHORS` and/or relax `MAX_ANCHOR_GAP_S` in `model/src/chroma_dtw_eval/amt_regen.py`, re-run, and record the before/after recall + the new thresholds on issue #128. Do NOT ratchet the baseline down silently.
- [ ] **Step 4: Commit** (only if thresholds were retuned)

```bash
git commit -am "fix(chroma): retune anchor gates for Transkun note density (#128)"
```

---

### Task 19: Gate 2 — unseen-generator recall stable
**Group:** F (depends on T18). REQUIRES piece-id audio cache.

**Behavior being verified:** the piece-ID feasibility harness recall is stable under Transkun.
**Interface under test:** `just piece-id-feasibility`

- [ ] **Step 1:** Ensure the audio cache is present (`just piece-id-feasibility-acquire` if needed), service up.
- [ ] **Step 2:** Run:

```bash
just piece-id-feasibility
```
Expected: recall stable vs the aria run; record the number on #128.
- [ ] **Step 3:** No commit unless a code change was required; if so, commit with `(#128)`.

---

### Task 20: Gate 5 — re-baseline evals hard-coding aria numbers
**Group:** F (depends on T19; final)

**Behavior being verified:** any eval baseline/fixture that hard-codes aria transcription numbers is re-baselined under Transkun, with deltas recorded on #128.
**Interface under test:** eval baseline artifacts.

- [ ] **Step 1:** Identify hard-coded aria baselines:

```bash
grep -rn "aria-amt\|aria_amt" model apps --include=*.json --include=*.py | grep -iv transkun
```
- [ ] **Step 2:** For each true baseline (e.g. `chroma_dtw_eval` `baseline.json`, any measurer golden), re-run its producer under Transkun and update the baseline; record old→new deltas in an issue #128 comment. **Explicitly re-run `amt_fidelity/onset_duration_render.py` and confirm its numbers are neutral vs the aria baseline** (Task 11 flagged it: aria implicitly truncated to 30s; the Transkun swap must keep the equivalent explicit cap — any drift here is a real regression, not metadata).
- [ ] **Step 3:** Post the STATE line:

```bash
gh issue comment 128 --body "STATE: Transkun migration complete; Gate deltas recorded (chroma recall <old>-><new>, piece-id <old>-><new>). Next: /review then /ship."
```
- [ ] **Step 4: Commit** any re-baselined artifacts:

```bash
git commit -am "test(eval): re-baseline transcription evals under Transkun (#128)"
```

---

## Follow-ups (recorded, out of the first slice — do NOT silently adopt)
- Collapse the measurers' 27s stratified windowing to whole-piece Transkun transcription — prove number-neutral first (spec simplification (a)).
- Unify dev `amt_local_server.py` with container `server.py` now the ONNX rationale is gone — prove behavior-neutral first (spec simplification (b)).
- If the warm in-process `transkun` Python-API signature could not be confirmed in T6, the service runs CLI-only; wire the warm path once confirmed.

---

## Challenge Review

### CEO Pass

**Premise — correct problem, direct path.** aria-amt's offset F1 0.37 vs Transkun 0.79 (#125 Stage 0) is a measured, load-bearing defect for a product whose thesis is expressive fidelity (offset/velocity/pedal). Replacing the transcriber behind a frozen `/transcribe` contract is the most direct route — verified: `inference.ts` reads only `midi_notes`/`pedal_events`/`error` (lines 58-59, 265-276), so the contract-freeze claim holds and no API/iOS/web churn is required. Existing coverage is correctly reused (the shared `transkun_cli` helper, `decode_webm_to_pcm` kept). No simpler framing found.

**Scope — appropriately cut.** The two genuine simplifications (collapse 27s windowing to whole-piece; unify `amt_local_server.py` with `server.py`) are explicitly DEFERRED as #128 follow-ups with "prove number/behavior-neutral first" gates — the right call. Complexity smell (touches ~25 files) is inherent to a repo-wide substrate swap, not gold-plating; most are mechanical one-line import swaps.

**12-month alignment — toward the ideal.** Transkun (MIT) removes two `git+` deps and the ONNX/PyTorch container split whose only rationale was aria CPU throughput. Net tech-debt reduction. Aligned.

**Alternatives.** Spec Q's document the version-label and threshold decisions with defaults; the shell-out-vs-import trade is stated. Adequate.

### Engineering Pass

**Architecture.** Data flow verified end to end: `chunk_audio` b64 → `decode_webm_to_pcm` (ffmpeg, kept) → `transcribe_fn(pcm)` → `build_response`. Both `amt_regen.py` (posts WAV as `chunk_audio`, `context_audio: None`, reads only `midi_notes`/`pedal_events`) and `pieceid_amt_axis.py` (reads only `midi_notes`) confirm the frozen shape is sufficient. Dropping `context_duration_s` from `transcription_info` is safe — no consumer reads it.

**Module depth.** `transkun_cli` (4-symbol surface hiding subprocess orchestration + pretty_midi CC64 parse) is genuinely DEEP. `transcription` rewrite (`EndpointHandler`/`build_response`/`resolve_transcriber`) is DEEP. No shallow-module smell.

**BLOCKER — Task 5 import-ordering breaks `import transcription` at its own Step 4.** `transcription.py` decorates the old `_transcribe` with `@torch.inference_mode()` at class-definition time (line 568), and imports `torch` (47) + `amt.config` (55) at module top. Task 5 Step 3 deletes those module-level imports but defers removing the old aria `EndpointHandler` to Task 7. Between T5 and T7 the decorator expression `torch.inference_mode()` is evaluated at import and raises `NameError`, so `import transcription` fails and T5 Step 4 ("both build_response tests PASS") cannot go green. Fix: fold the old-class removal (or at minimum the `@torch.inference_mode()` method) into Task 5, or keep the aria imports through Task 7.

**BLOCKER — the sole real-model verification silently skips; sample WAV is gitignored, not "checked-in."** `git check-ignore` confirms `apps/inference/Beethoven_WoO80_var27_8bars_3_15.wav` is ignored and `git ls-files` shows it untracked — absent in any fresh checkout/worktree. Task 4's `test_transcribe_pcm_on_real_sample...` does `if not wav.exists(): pytest.skip(...)`, so Group 0's "[SHIPS INDEPENDENTLY], model-verified" claim is unverified — the test skips. Gate 4 (`smoke_test_amt.py`, `DEFAULT_WAV` = same path) crashes on `read_bytes()` with no `--wav`. Result: NO task and NO gate actually exercises real Transkun on this tree. Fix: add a small real clip to the repo (or copy into the worktree / point at a rehydrated clip) AND make Task 4 fail-hard rather than skip when the sample is absent, so the model crux is genuinely gated.

**RISK (6/10) — "warm load-once" may not be warm.** `_import_warm_transcriber`'s `_warm` closure calls `transkun.transcribe.transcribe(in_wav, out_mid, device="cpu")` per request. If that top-level entry re-instantiates the model each call (unverified), the "warm in-process, load-once" perf rationale in the spec collapses and the service reloads weights every request. Fallback: confirm transkun caches the model globally, or hoist model construction into `__init__` and call a lower-level API. Verify this is actually an issue before relying on the warm path for live latency.

**RISK (7/10) — live latency / per-call weight reload under CLI fallback.** Transkun ≈ 0.45x realtime on CPU; the CLI path spawns a fresh `uv run --with transkun` (fresh interpreter + weight load, first-ever call downloads weights) per request. For the live 15s-chunk loop this is ~30s+ per chunk plus reload. If the warm path is deferred (T6's escape hatch), the live `/transcribe` is unusably slow. Acceptable pre-beta (zero users) but the plan never quantifies it. Fallback: keep warm path as a hard requirement for the live surface.

**RISK (7/10) — `extract_amt_midi.py` is not a mechanical `_transcribe → transcribe_pcm` swap.** It calls the FULL `handler({"chunk_audio":..., "context_audio":None})` (line 183) and reads `result["transcription_info"]["pitch_range"]` / `transcription_time_ms` for its `_sanity.jsonl`. Task 11 lumps it into "same mechanical swap" but it needs a bespoke rewrite to `transkun_cli.transcribe_wav(wav_path)` plus reconstructing/dropping the sanity fields. Under-specified; call it out as its own sub-step.

**RISK (6/10) — `onset_duration_render.py` depends on aria's implicit 30s truncation.** `AMT_WINDOW_S = 30.0 # aria-amt _transcribe hard-truncates to this`; it passes FULL clip audio to `handler._transcribe(audio)` and relies on the 30s cap. `transkun_cli.transcribe_pcm` does NOT truncate — it transcribes the whole clip (perf hit on long clips). The explicit `crop(..., cutoff)` afterward likely preserves the metric, but the mechanical swap in Task 11 does not verify number-neutrality. Watch at Gate 5.

**RISK (5/10) — Transkun CC64 density vs aria discrete pedal_msgs.** T2 maps EVERY CC64 control-change to a pedal event (incl. consecutive 127s from a continuous pedal stream), whereas aria emitted discrete on/off `pedal_msgs`. Pedal on-fraction measurers (`tau_pedaling_render`, `amt_pedaling_ga_render`) integrate these; a dense same-value stream could shift the statistic or mis-pair on→off. Only surfaced at Gate 3/5. Consider collapsing consecutive same-value CC64 events in the parse.

**RISK (6/10) — Task 11 guard is text-only.** `test_no_aria_amt_handler.py` asserts string presence/absence, not that the swapped scripts run or produce correct output. A broken swap (e.g. `extract_amt_midi`) passes the guard. These scripts need real audio+model to exercise, so a static guard is a pragmatic floor — but do not treat green as "swap works."

**RISK (6/10) — container `server.py` never runtime-validated.** Task 9 hits `/health` via TestClient without triggering lifespan/model load, and no gate boots the container. If the warm path is deferred to CLI-only and the T15 Dockerfile (which installs transkun but not `uv`) lacks `uv` on PATH, `resolve_transcriber()` raises and the container refuses to start — caught only at a future deploy. Ensure the Dockerfile provides whichever path `resolve_transcriber` will actually take.

**Test philosophy / vertical slice.** Tasks are mostly clean one-test→one-impl→one-commit. Minor: T5/T6/T7/T10 each land 2 closely-related tests in one slice — both implemented in the same commit, so not true horizontal slicing; acceptable. Guard/existence tests (T11/T14/T15) are shape tests by nature but are legitimate regression guards for a mechanical swap+deletion. Full-replace of `test_transcription.py` only drops tests for deleted helpers (`deduplicate_notes`, `midi_dict_to_notes_and_pedals`, `advance_valid_note_groups`); `decode_webm_to_pcm` (kept) was already untested, so no coverage regression.

**Failure modes.** Loud-failure design is sound: `TranskunError` on non-zero exit / missing MIDI / missing input; `resolve_transcriber` refuses to start rather than silent per-request fallback; `__call__` returns a TRANSCRIPTION_ERROR body (matching the existing Tier-3 degrade contract). No new silent failures introduced. Chroma `DEDUP_WINDOW_S=0.0` correctly degrades to an order-normalizing pass-through (guard `diff < 0.0` never fires).

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `/transcribe` contract readable-fields are frozen for all 3 HTTP consumers | SAFE | Verified: inference.ts / amt_regen / pieceid read only midi_notes+pedal_events(+error) |
| Beethoven sample WAV is "checked-in" and available to Task 4 / Gate 4 | RISKY | gitignored + untracked; absent in worktree → test skips, smoke crashes (BLOCKER) |
| Deleting torch/aria imports in T5 leaves a still-importable module until T7 | RISKY | `@torch.inference_mode()` at class-def evaluates at import → NameError (BLOCKER) |
| Warm in-process transkun path is "load-once" | VALIDATE | Closure calls top-level transcribe per request; caching unverified |
| `transkun.transcribe.transcribe(wav,out,device=)` signature | VALIDATE | Plan itself flags it unconfirmed; CLI fallback is the tested path |
| All 8 measurer swaps are mechanical `_transcribe→transcribe_pcm` | RISKY | extract_amt_midi uses full handler()+info; onset_duration relies on 30s trunc |
| Transkun ~7% under-transcription won't drop chroma below MIN_ANCHORS gate | VALIDATE | Only Gate 1 run resolves it; threshold retune path documented |
| CC64→event mapping preserves pedal on-fraction statistics | VALIDATE | Dense same-value stream differs from aria discrete pedal_msgs; Gate 3/5 |
| Offline measurer gates are runtime-feasible with per-window fresh-subprocess reload | VALIDATE | Weight reload per call; wall-clock unestimated |

### Summary
[BLOCKER] count: 2
[RISK]    count: 7
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — (1) Task 5 deletes module-level `torch`/`amt` imports while the `@torch.inference_mode()`-decorated old EndpointHandler remains until Task 7, breaking `import transcription` at T5 Step 4 — fold the old-class removal into Task 5 or keep imports through T7. (2) The sole real-model verification (Task 4) silently skips and Gate 4 smoke crashes because the Beethoven sample WAV is gitignored/untracked, not "checked-in" — supply a real clip (repo or worktree) and make Task 4 fail-hard instead of skip so real Transkun is genuinely gated.

---

## Challenge Re-Review (2026-07-25)

Re-review after the two prior blockers were reworked. Fresh CEO+ENG adversarial pass, verified against the live tree.

### Prior blockers — both RESOLVED (verified against code)

**Blocker 1 (import-ordering) — RESOLVED (confidence 9/10).** Task 5 Step 3 now carries an explicit **IMPORT-ORDERING REQUIREMENT** block instructing deletion of the module-level `torch`/`amt` imports AND the entire old aria `EndpointHandler` class in THIS task. Verified the code matches the plan's line references exactly: `import torch`@47, `import amt.config`@55, `from amt...`@89-92, `class EndpointHandler`@324 running to EOF@706, `@torch.inference_mode()`@568. The class is the last thing in the file, so "delete 324-706" is a clean tail removal; `decode_webm_to_pcm`@134 and `SAMPLE_RATE`@94 sit before it and survive. The remaining module-level aria references (`_AMT_CONFIG`@57, `_patched_load_config`@79, the `_amt_config.load_config = _patched_load_config` patch statement@87, `_load_weight`@100, `midi_dict_to_notes_and_pedals`@182, `deduplicate_notes`@227, `advance_valid_note_groups`@270) are all inside the contiguous aria block 47-133 / 182-323 that the task explicitly names for deletion and covers under "replace the entire aria machinery at the top of the file." After Task 5 there is ZERO import-time `torch`/`amt` reference, so `import transcription` succeeds at T5 Step 4. Consistency bonus: Task 7's red-state expectation ("`module 'transcription' has no attribute 'EndpointHandler'`") is now CORRECT because the old class is genuinely gone between T5 and T7.

**Blocker 2 (real-model verification) — RESOLVED (confidence 10/10).** Verified `apps/inference/amt/fixtures/piano_sample_5s_16k.wav` is: tracked (`git ls-files` lists it), committed to HEAD (`git cat-file -e HEAD:...` passes — not merely staged), a real 5.00s mono 16kHz 16-bit PCM WAV (156K). The `.gitignore:143 *.wav` rule genuinely ignores arbitrary `.wav` names, confirming the "force-added past .gitignore" narrative. Task 4's test now `raise AssertionError` (fail-hard) instead of `pytest.skip` when the fixture is absent (lines 351-355). Gate 4 (Task 16) repoints `smoke_test_amt.py`'s `DEFAULT_WAV` to the same fixture and keeps the existence check. Real Transkun is now genuinely exercised by both a unit test (T4) and a service gate (Gate 4) on every fresh checkout.

### New findings from the edits

No new BLOCKER introduced by either rework. One minor pre-existing gap surfaced:

[OBS] — Task 6's second test (`test_resolve_transcriber_falls_back_to_cli`, line 564) references `transkun_cli.transcribe_pcm`, but the test file header established in Task 5 (lines 424-438) imports only `transcription`, not `transkun_cli`. As written this NameErrors. It is NOT introduced by the two fixes and is self-correcting under strict TDD (the build agent hits the NameError at T6 Step 2 and adds `import transkun_cli`; `sys.path.insert` already makes it importable). Recommend the build agent add `import transkun_cli` to the test file when landing T6. Not a blocker.

### Standing risks (unchanged, still valid watch-items)

The 7 RISKs from the first review remain accurate and none were made worse by the edits: warm-path "load-once" unverified; CLI-fallback per-call weight reload latency; `extract_amt_midi.py` needs a bespoke (not mechanical) swap; `onset_duration_render.py` 30s implicit-truncation dependency (watch at Gate 5); CC64 density vs aria discrete pedal_msgs; Task 11 text-only guard; container `server.py` never runtime-booted (ensure the T15 Dockerfile provides whichever path `resolve_transcriber` takes). All are execution-time watch items, not plan-blocking.

### Re-Review Summary
[BLOCKER] count: 0
[RISK]    count: 7 (carried forward, unchanged)
[QUESTION] count: 0

VERDICT: PROCEED_WITH_CAUTION — both prior blockers verified resolved against the live tree (Task 5 folds the aria class deletion in with the torch/amt import removal; fixture is committed to HEAD and Task 4 + Gate 4 fail hard). Monitor during execution: (a) add `import transkun_cli` to the T6 test to avoid a NameError; (b) `extract_amt_midi.py` is a bespoke rewrite, not a mechanical swap; (c) `onset_duration_render.py`'s explicit 30s cap must be confirmed number-neutral at Gate 5; (d) confirm the warm in-process transkun path is truly load-once (else live `/transcribe` reloads weights per chunk); (e) ensure the T15 Dockerfile supplies the runtime path `resolve_transcriber` will actually take.
