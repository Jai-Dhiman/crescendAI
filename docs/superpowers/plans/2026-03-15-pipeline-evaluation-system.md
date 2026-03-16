# Pipeline Evaluation System (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 eval framework -- local inference runner, shared infrastructure (judge, reporting, cache, traces), observation quality eval, subagent reasoning eval, and the eval-mode endpoint in the Rust worker.

**Architecture:** Python eval harness reads cached inference results and either (a) sends them to wrangler dev via an eval-mode endpoint for full pipeline evaluation, or (b) calls Groq directly for isolated subagent reasoning evaluation. An LLM judge (Claude) scores observations with binary rubrics. All results flow to a shared JSON report format.

**Tech Stack:** Python 3.10+ (eval harness, uv), Rust/WASM (eval-chunk endpoint), Anthropic SDK (judge), Groq SDK (subagent eval), PyTorch (local inference)

**Spec:** `docs/superpowers/specs/2026-03-15-pipeline-evaluation-system-design.md`

---

## Chunk 1: Infrastructure Foundation

### Task 1: Device Auto-Detection in Inference Code

**Files:**
- Modify: `apps/inference/models/loader.py:106-111`
- Modify: `apps/inference/handler.py:66,70`
- Modify: `apps/inference/models/transcription.py:42,51`

- [ ] **Step 1: Fix ModelCache.initialize device fallback**

In `apps/inference/models/loader.py`, line 111 currently reads:
```python
self.device = torch.device(device if torch.cuda.is_available() else "cpu")
```

This ignores MPS and overrides the passed `device` argument when CUDA is unavailable. Replace with:
```python
import os

def _resolve_device(requested: str) -> torch.device:
    """Resolve device with env override and auto-detection.

    Supports: "cuda", "mps", "cpu", "auto".
    "auto" runs the full cascade: CUDA > MPS > CPU.
    CRESCEND_DEVICE env var overrides the requested device.
    """
    dev = os.environ.get("CRESCEND_DEVICE", requested)
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    elif dev == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    elif dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        dev = "cpu"
    return torch.device(dev)
```

Add this as a module-level function in `loader.py`. Then change line 111 to:
```python
self.device = _resolve_device(device)
```

- [ ] **Step 2: Parameterize device in EndpointHandler**

In `apps/inference/handler.py`, change lines 65-70 from:
```python
self._cache = get_model_cache()
self._cache.initialize(device="cuda", checkpoint_dir=checkpoint_dir)

# Initialize AMT transcription model
print("Loading ByteDance AMT model...")
self._transcription = TranscriptionModel(device="cuda")
```

To:
```python
import os
device = os.environ.get("CRESCEND_DEVICE", "cuda")

self._cache = get_model_cache()
self._cache.initialize(device=device, checkpoint_dir=checkpoint_dir)

# Initialize AMT transcription model
print("Loading ByteDance AMT model...")
try:
    self._transcription = TranscriptionModel(device=device)
except RuntimeError as e:
    if "mps" in str(e).lower() or "MPS" in str(e):
        print(f"AMT failed on {device}, falling back to CPU: {e}")
        self._transcription = TranscriptionModel(device="cpu")
    else:
        raise
```

The try/except handles the case where ByteDance's PianoTranscription doesn't support MPS.

- [ ] **Step 3: Verify locally**

Run: `cd apps/inference && CRESCEND_DEVICE=cpu python -c "from models.loader import _resolve_device; import torch; print(_resolve_device('cuda'))"`

Expected: `cpu` (no CUDA on Mac)

- [ ] **Step 4: Commit**

```bash
git add apps/inference/models/loader.py apps/inference/handler.py
git commit -m "feat(inference): add device auto-detection with CRESCEND_DEVICE env override"
```

---

### Task 2: Local Inference Runner

**Files:**
- Create: `apps/inference/eval_runner.py`
- Create: `apps/inference/audio_chunker.py`

- [ ] **Step 1: Create audio chunker utility**

`apps/inference/audio_chunker.py` -- splits audio files into 15s chunks matching production:

```python
"""Split audio files into 15-second chunks for eval inference."""

from __future__ import annotations

import numpy as np
from preprocessing.audio import preprocess_audio_from_bytes

CHUNK_DURATION_S = 15
SAMPLE_RATE = 24000
CHUNK_SAMPLES = CHUNK_DURATION_S * SAMPLE_RATE


def chunk_audio_file(file_path: str, max_duration: int = 300) -> list[np.ndarray]:
    """Load audio file and split into 15s chunks.

    Args:
        file_path: Path to audio file (any format ffmpeg supports).
        max_duration: Maximum audio duration in seconds.

    Returns:
        List of numpy arrays, each 15s of 24kHz mono float32.
    """
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    audio, duration = preprocess_audio_from_bytes(audio_bytes, max_duration=max_duration)

    chunks = []
    total_samples = len(audio)
    for start in range(0, total_samples, CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if len(chunk) >= SAMPLE_RATE:  # skip chunks < 1s
            chunks.append(chunk)

    return chunks
```

- [ ] **Step 2: Create the eval runner**

`apps/inference/eval_runner.py`:

```python
"""Local inference batch runner for pipeline evaluation.

Loads EndpointHandler with auto-detected device, processes audio files
in a directory, and writes versioned JSON cache.

Usage:
    CRESCEND_DEVICE=cpu python eval_runner.py --audio-dir ../../data/eval/youtube_amt/
    CRESCEND_DEVICE=mps python eval_runner.py  # uses defaults
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

# Set device before any torch imports (auto = CUDA > MPS > CPU)
os.environ.setdefault("CRESCEND_DEVICE", "auto")

from audio_chunker import chunk_audio_file
from handler import EndpointHandler

DEFAULT_CHECKPOINT_DIR = str(
    Path(__file__).parents[1].parent / "model" / "data" / "checkpoints" / "model_improvement" / "A1"
)
DEFAULT_AUDIO_DIR = str(Path(__file__).parents[1].parent / "data" / "eval" / "youtube_amt")
DEFAULT_CACHE_DIR = str(Path(__file__).parents[1].parent / "data" / "eval" / "inference_cache")


def get_git_sha() -> tuple[str, bool]:
    """Return (sha, is_dirty)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", True


def build_model_fingerprint(model_info: dict) -> str:
    """Build a cache directory name from model info.

    Format: {name}_{architecture} matching spec convention (e.g., "a1-max_muq-l9-12").
    """
    name = model_info.get("name", "unknown").lower().replace(" ", "-")
    arch = model_info.get("architecture", "unknown").lower().replace(" ", "-")
    return f"{name}_{arch}"


def run(
    checkpoint_dir: str,
    audio_dir: str,
    cache_dir: str,
) -> None:
    """Run batch inference on all audio files in audio_dir."""
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    audio_files = sorted(
        p for p in audio_path.iterdir()
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".webm", ".m4a", ".opus"}
    )
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_dir}")

    print(f"Found {len(audio_files)} audio files in {audio_dir}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Device: {os.environ.get('CRESCEND_DEVICE', 'auto')}")

    # Initialize handler
    print("Loading models...")
    handler = EndpointHandler(path=checkpoint_dir)

    # Determine cache directory from model fingerprint
    # Run a tiny inference to get model_info
    test_result = handler({"inputs": audio_files[0].read_bytes(), "parameters": {"max_duration_seconds": 5}})
    if "error" in test_result:
        raise RuntimeError(f"Test inference failed: {test_result['error']}")

    fingerprint = build_model_fingerprint(test_result.get("model_info", {}))
    versioned_cache = Path(cache_dir) / fingerprint
    versioned_cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {versioned_cache}")

    git_sha, git_dirty = get_git_sha()

    for i, audio_file in enumerate(audio_files):
        recording_id = audio_file.stem
        cache_file = versioned_cache / f"{recording_id}.json"

        if cache_file.exists():
            print(f"[{i+1}/{len(audio_files)}] {recording_id} -- cached, skipping")
            continue

        start = time.time()

        # Chunk audio into 15s segments
        try:
            chunks = chunk_audio_file(str(audio_file))
        except Exception as e:
            print(f"[{i+1}/{len(audio_files)}] {recording_id} -- SKIP (audio error: {e})")
            continue

        # Run inference on each chunk
        chunk_results = []
        for ci, chunk_audio in enumerate(chunks):
            buf = io.BytesIO()
            sf.write(buf, chunk_audio, 24000, format="WAV")
            audio_b64 = base64.b64encode(buf.getvalue()).decode()

            result = handler({"inputs": audio_b64})
            if "error" in result:
                print(f"  chunk {ci} failed: {result['error']}")
                continue

            chunk_results.append({
                "chunk_index": ci,
                "predictions": result.get("predictions", {}),
                "midi_notes": result.get("midi_notes", []),
                "pedal_events": result.get("pedal_events", []),
                "transcription_info": result.get("transcription_info"),
                "audio_duration_seconds": result.get("audio_duration_seconds", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
            })

        elapsed = time.time() - start

        # Write cache file
        cache_data = {
            "recording_id": recording_id,
            "model_fingerprint": fingerprint,
            "git_sha": git_sha,
            "chunks": chunk_results,
            "total_duration_seconds": sum(c["audio_duration_seconds"] for c in chunk_results),
            "total_chunks": len(chunk_results),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)

        print(f"[{i+1}/{len(audio_files)}] {recording_id} ({elapsed:.1f}s, {len(chunk_results)} chunks)")

    print(f"\nDone. Cache: {versioned_cache}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local inference batch runner")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    run(args.checkpoint_dir, args.audio_dir, args.cache_dir)
```

- [ ] **Step 3: Test with a single audio file**

```bash
cd apps/inference
mkdir -p ../../data/eval/youtube_amt
# Copy one test audio file to data/eval/youtube_amt/
CRESCEND_DEVICE=cpu python eval_runner.py --audio-dir ../../data/eval/youtube_amt/
```

Expected: Model loads on CPU, processes audio, writes JSON cache file.

- [ ] **Step 4: Commit**

```bash
git add apps/inference/eval_runner.py apps/inference/audio_chunker.py
git commit -m "feat(inference): add local eval runner with audio chunking and versioned cache"
```

---

### Task 3: Eval-Mode Endpoint in Rust Worker

**Files:**
- Modify: `apps/api/src/practice/session.rs:248-443`
- Modify: `apps/api/src/server.rs` (route registration)

- [ ] **Step 1: Extract downstream pipeline from handle_chunk_ready**

In `apps/api/src/practice/session.rs`, the function `handle_chunk_ready` (line 248) does steps 1-12. We need to extract steps 3-12 into a reusable function that both `handle_chunk_ready` and the new eval endpoint can call.

Create a new method `process_inference_result` on PracticeSession:

```rust
/// Process pre-computed inference results through the downstream pipeline.
/// Steps 3-12 of handle_chunk_ready: extract scores, update state, run STOP,
/// score following, analysis, and generate observation if triggered.
///
/// Used by both handle_chunk_ready (after HF inference) and the eval-chunk
/// endpoint (with cached inference results).
async fn process_inference_result(
    &self,
    ws: &WebSocket,
    index: usize,
    hf_response: serde_json::Value,
) -> Result<()> {
    // ... steps 3-12 from handle_chunk_ready, unchanged
}
```

Then refactor `handle_chunk_ready` to call it:
```rust
async fn handle_chunk_ready(&self, ws: &WebSocket, index: usize, r2_key: &str) -> Result<()> {
    // 1. Fetch audio from R2
    let audio_bytes = match self.fetch_audio_from_r2(r2_key).await {
        Ok(bytes) => bytes,
        Err(e) => {
            console_error!("R2 fetch failed for {}: {}", r2_key, e);
            self.send_zeroed_chunk_processed(ws, index)?;
            return Ok(());
        }
    };

    // 2. Call HF inference
    let hf_response = match self.call_hf_inference(&audio_bytes).await {
        Ok(resp) => resp,
        Err(e) => {
            console_error!("HF inference failed for chunk {}: {}", index, e);
            self.inner.borrow_mut().inference_failures += 1;
            self.send_zeroed_chunk_processed(ws, index)?;
            return Ok(());
        }
    };

    // 3-12. Downstream pipeline
    self.process_inference_result(ws, index, hf_response).await
}
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
cd apps/api && cargo test
```

Expected: All existing tests pass (refactor only, no behavior change).

- [ ] **Step 3: Add eval-chunk WebSocket message handler**

In the WebSocket message handler (wherever WS messages are dispatched in session.rs), add handling for a new message type:

```rust
// In the WS message dispatch logic:
"eval_chunk" => {
    // Only available in dev mode
    let is_dev = self.env.var("ENVIRONMENT")
        .map(|v| v.to_string() == "development")
        .unwrap_or(false);
    if !is_dev {
        let _ = ws.send_with_str(r#"{"type":"error","message":"eval_chunk only available in dev"}"#);
        return Ok(());
    }

    let chunk_index = msg.get("chunk_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build HF-response-shaped JSON from the eval payload
    let hf_response = serde_json::json!({
        "predictions": msg.get("predictions").cloned().unwrap_or_default(),
        "midi_notes": msg.get("midi_notes").cloned().unwrap_or_default(),
        "pedal_events": msg.get("pedal_events").cloned().unwrap_or_default(),
    });

    self.process_inference_result(ws, chunk_index, hf_response).await?;
}
```

This uses WebSocket messages (not HTTP POST) since the practice session DO already uses WS for all chunk communication. The eval harness connects via WS, sends `eval_chunk` messages, and receives observations on the same connection.

**Note:** This is a deliberate deviation from the spec, which describes `POST /api/practice/eval-chunk`. Using WS messages is simpler because the DO already dispatches everything via WS, and it avoids adding a second HTTP entrypoint.

Additionally, mark the session as an eval session when the first `eval_chunk` is received:

```rust
// At the top of the eval_chunk handler:
self.inner.borrow_mut().is_eval_session = true;
```

Add `is_eval_session: bool` to `SessionState` (default `false`). When `is_eval_session` is true, the `generate_observation` method includes an `eval_context` field in the observation WS message containing the full context the teacher LLM saw:

```rust
// In generate_observation, after building the observation response JSON:
if self.inner.borrow().is_eval_session {
    response["eval_context"] = serde_json::json!({
        "predictions": &scores_json,
        "baselines": &baselines_json,
        "recent_observations": &recent_obs_json,
        "analysis_facts": &analysis_json,
        "piece_name": &self.inner.borrow().piece_query,
    });
}
```

This gives the eval judge the same context the teacher LLM received, enabling meaningful accuracy and appropriateness assessment. The field costs nothing in production (flag is always false).

- [ ] **Step 4: Test with wrangler dev**

```bash
cd apps/api && npx wrangler dev
```

In another terminal, use websocat or a quick Python script to:
1. POST to `/api/practice/start` to create a session
2. Connect to WS
3. Send `{"type": "eval_chunk", "chunk_index": 0, "predictions": {"dynamics": 0.5, ...}, "midi_notes": [], "pedal_events": []}`
4. Verify response includes `chunk_processed` message

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/practice/session.rs
git commit -m "feat(api): add eval-chunk WS message and extract process_inference_result"
```

---

### Task 4: Shared Eval Infrastructure -- Reporting

**Files:**
- Create: `apps/api/evals/pyproject.toml`
- Create: `apps/api/evals/shared/__init__.py`
- Create: `apps/api/evals/shared/reporting.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "pipeline-eval"
version = "0.1.0"
description = "Pipeline evaluation harness for CrescendAI"
requires-python = ">=3.10"
dependencies = [
    "anthropic>=0.40.0",
    "groq>=0.4.0",
    "requests>=2.31.0",
    "websockets>=12.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["shared", "observation_quality", "subagent_reasoning"]
```

- [ ] **Step 2: Create reporting module**

`apps/api/evals/shared/reporting.py`:

```python
"""Shared eval reporting: JSON envelope + terminal summary table."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class MetricResult:
    mean: float
    std: float
    n: int
    pass_threshold: float | None = None

    @property
    def passed(self) -> bool | None:
        if self.pass_threshold is None:
            return None
        return self.mean >= self.pass_threshold


@dataclass
class EvalReport:
    eval_name: str
    eval_version: str
    dataset: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    worst_cases: list[dict] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        sha, dirty = _get_git_info()
        self.metadata.setdefault("git_sha", sha)
        self.metadata.setdefault("git_dirty", dirty)
        self.metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    def to_json(self) -> dict:
        data = {
            "eval_name": self.eval_name,
            "eval_version": self.eval_version,
            "dataset": self.dataset,
            "metrics": {},
            "pass_criteria": {},
            "worst_cases": self.worst_cases,
            "cost": self.cost,
            "metadata": self.metadata,
        }
        all_passed = True
        for name, m in self.metrics.items():
            data["metrics"][name] = {
                "mean": round(m.mean, 4),
                "std": round(m.std, 4),
                "n": m.n,
            }
            if m.pass_threshold is not None:
                data["metrics"][name]["pass"] = m.passed
                data["pass_criteria"][f"{name}_mean_gte"] = m.pass_threshold
                if not m.passed:
                    all_passed = False
        data["pass_criteria"]["all_passed"] = all_passed
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        print(f"Report saved: {path}")

    def print_summary(self) -> None:
        data = self.to_json()
        print(f"\n{'=' * 60}")
        print(f"  {self.eval_name} (v{self.eval_version})")
        print(f"  Dataset: {self.dataset}")
        print(f"  Git: {self.metadata.get('git_sha', '?')}"
              f"{'*' if self.metadata.get('git_dirty') else ''}")
        print(f"{'=' * 60}")
        print(f"  {'Metric':<30} {'Value':>8} {'Gate':>8} {'Status':>8}")
        print(f"  {'-' * 56}")
        for name, vals in data["metrics"].items():
            gate = data["pass_criteria"].get(f"{name}_mean_gte", "")
            gate_str = f">={gate}" if gate else "--"
            status = ""
            if "pass" in vals:
                status = "PASS" if vals["pass"] else "FAIL"
            print(f"  {name:<30} {vals['mean']:>8.3f} {gate_str:>8} {status:>8}")
        print(f"  {'-' * 56}")
        all_passed = data["pass_criteria"].get("all_passed", True)
        print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
        if self.cost:
            print(f"  Cost: ~${self.cost.get('estimated_usd', 0):.2f} "
                  f"({self.cost.get('judge_calls', 0)} judge calls)")
        print(f"{'=' * 60}\n")


def _get_git_info() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", True
```

- [ ] **Step 3: Verify module imports**

```bash
cd apps/api/evals && uv sync && uv run python -c "from shared.reporting import EvalReport, MetricResult; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add apps/api/evals/pyproject.toml apps/api/evals/shared/
git commit -m "feat(evals): add shared reporting module with JSON envelope and terminal summary"
```

---

### Task 5: Shared Eval Infrastructure -- Inference Cache

**Files:**
- Create: `apps/api/evals/shared/inference_cache.py`

- [ ] **Step 1: Create inference cache module**

`apps/api/evals/shared/inference_cache.py`:

```python
"""Read/write versioned inference cache with staleness detection."""

from __future__ import annotations

import json
from pathlib import Path


class StaleCacheError(Exception):
    """Raised when cache model fingerprint doesn't match expected version."""
    pass


def find_cache_dir(cache_root: Path) -> Path | None:
    """Find the most recent versioned cache directory.

    Returns None if no cache exists.
    """
    if not cache_root.exists():
        return None
    dirs = sorted(
        (d for d in cache_root.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


def load_recording(cache_dir: Path, recording_id: str) -> dict:
    """Load a cached recording's inference results.

    Raises:
        FileNotFoundError: If cache file doesn't exist.
        json.JSONDecodeError: If cache file is corrupt.
    """
    cache_file = cache_dir / f"{recording_id}.json"
    with open(cache_file) as f:
        return json.load(f)


def load_all_recordings(cache_dir: Path) -> list[dict]:
    """Load all cached recordings from a versioned cache directory."""
    recordings = []
    for f in sorted(cache_dir.glob("*.json")):
        with open(f) as fh:
            recordings.append(json.load(fh))
    return recordings


def validate_cache(cache_dir: Path, expected_fingerprint: str | None = None) -> str:
    """Validate cache integrity and return the model fingerprint.

    Args:
        cache_dir: Path to versioned cache directory.
        expected_fingerprint: If provided, raises StaleCacheError on mismatch.

    Returns:
        The model fingerprint from the cache.

    Raises:
        StaleCacheError: If fingerprint doesn't match expected.
        FileNotFoundError: If cache is empty.
    """
    files = list(cache_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Cache directory is empty: {cache_dir}")

    # Read fingerprint from first file
    with open(files[0]) as f:
        data = json.load(f)
    fingerprint = data.get("model_fingerprint", "unknown")

    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise StaleCacheError(
            f"Cache fingerprint mismatch: cache has '{fingerprint}', "
            f"expected '{expected_fingerprint}'. "
            f"Re-run inference with: cd apps/inference && python eval_runner.py"
        )

    return fingerprint
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/evals/shared/inference_cache.py
git commit -m "feat(evals): add inference cache reader with staleness detection"
```

---

### Task 6: Shared Eval Infrastructure -- LLM Judge

**Files:**
- Create: `apps/api/evals/shared/judge.py`
- Create: `apps/api/evals/shared/prompts/observation_quality_judge_v1.txt`
- Create: `apps/api/evals/shared/prompts/subagent_reasoning_judge_v1.txt`

- [ ] **Step 1: Create LLM judge client**

`apps/api/evals/shared/judge.py`:

```python
"""LLM-as-judge client for eval scoring.

Wraps Anthropic API with retry, structured output parsing, and
versioned prompt loading. All evals that need LLM judgment use this.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_RETRIES = 3
BASE_DELAY = 2.0  # seconds


@dataclass
class CriterionScore:
    criterion: str
    passed: bool | None  # None = parse failure or refusal
    evidence: str
    raw_response: str


@dataclass
class JudgeResult:
    scores: list[CriterionScore]
    model: str
    prompt_version: str
    latency_ms: int

    @property
    def pass_rate(self) -> float:
        scored = [s for s in self.scores if s.passed is not None]
        if not scored:
            return 0.0
        return sum(1 for s in scored if s.passed) / len(scored)


def load_prompt(prompt_name: str) -> str:
    """Load a versioned judge prompt from the prompts directory."""
    path = PROMPTS_DIR / prompt_name
    if not path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {path}")
    return path.read_text()


def judge_observation(
    observation_text: str,
    context: dict,
    prompt_file: str = "observation_quality_judge_v1.txt",
    model: str = "claude-sonnet-4-6",
) -> JudgeResult:
    """Score an observation using the LLM judge.

    Args:
        observation_text: The teacher observation to evaluate.
        context: Dict with keys: predictions, baselines, recent_observations,
                 analysis_facts, piece_name, bar_range.
        prompt_file: Versioned prompt file name.
        model: Anthropic model to use.

    Returns:
        JudgeResult with per-criterion scores.
    """
    prompt_template = load_prompt(prompt_file)

    # Format context into the prompt
    user_message = prompt_template.format(
        observation_text=observation_text,
        predictions=json.dumps(context.get("predictions", {}), indent=2),
        baselines=json.dumps(context.get("baselines", {}), indent=2),
        recent_observations=json.dumps(context.get("recent_observations", []), indent=2),
        analysis_facts=json.dumps(context.get("analysis_facts", {}), indent=2),
        piece_name=context.get("piece_name", "Unknown"),
        bar_range=context.get("bar_range", "Unknown"),
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    start = time.time()
    response = _call_with_retry(client, model, user_message)
    latency_ms = int((time.time() - start) * 1000)

    scores = _parse_judge_response(response)

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_file,
        latency_ms=latency_ms,
    )


def judge_subagent(
    subagent_output: dict,
    context: dict,
    prompt_file: str = "subagent_reasoning_judge_v1.txt",
    model: str = "claude-sonnet-4-6",
) -> JudgeResult:
    """Score a subagent reasoning output using the LLM judge.

    Args:
        subagent_output: Dict with keys: dimension, framing, reasoning_trace.
        context: Dict with keys: predictions, baselines, recent_observations.
        prompt_file: Versioned prompt file name.
        model: Anthropic model to use.

    Returns:
        JudgeResult with per-criterion scores.
    """
    prompt_template = load_prompt(prompt_file)

    user_message = prompt_template.format(
        dimension=subagent_output.get("dimension", ""),
        framing=subagent_output.get("framing", ""),
        reasoning_trace=subagent_output.get("reasoning_trace", ""),
        predictions=json.dumps(context.get("predictions", {}), indent=2),
        baselines=json.dumps(context.get("baselines", {}), indent=2),
        recent_observations=json.dumps(context.get("recent_observations", []), indent=2),
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    start = time.time()
    response = _call_with_retry(client, model, user_message)
    latency_ms = int((time.time() - start) * 1000)

    scores = _parse_judge_response(response)

    return JudgeResult(
        scores=scores,
        model=model,
        prompt_version=prompt_file,
        latency_ms=latency_ms,
    )


def _call_with_retry(client: anthropic.Anthropic, model: str, user_message: str) -> str:
    """Call Anthropic API with exponential backoff on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                system="You are an evaluation judge. For each criterion, you MUST format your response exactly as:\n\n**[Criterion Name]:** YES or NO\nEvidence: \"your evidence here\"\n\nDo not deviate from this format.",
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = BASE_DELAY * (2 ** attempt)
            print(f"  Rate limited, retrying in {delay}s...")
            time.sleep(delay)
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # Overloaded
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  API overloaded, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Exhausted retries")


def _parse_judge_response(response_text: str) -> list[CriterionScore]:
    """Parse the judge's response into per-criterion scores.

    Expected format per criterion:
    **[CRITERION_NAME]:** YES/NO
    Evidence: "quoted text or explanation"
    """
    scores = []
    lines = response_text.strip().split("\n")
    current_criterion = None
    current_evidence = []

    for line in lines:
        line = line.strip()
        if line.startswith("**") and ("YES" in line.upper() or "NO" in line.upper()):
            # Save previous criterion
            if current_criterion:
                scores.append(current_criterion)
                current_criterion = None

            # Parse this criterion
            try:
                # Extract criterion name between ** markers
                name_part = line.split("**")[1].rstrip(":").strip()
                passed = "YES" in line.upper().split("**")[-1]
                current_criterion = CriterionScore(
                    criterion=name_part,
                    passed=passed,
                    evidence="",
                    raw_response=line,
                )
                current_evidence = []
            except (IndexError, ValueError):
                current_criterion = CriterionScore(
                    criterion="parse_error",
                    passed=None,
                    evidence=line,
                    raw_response=line,
                )
        elif current_criterion and line.lower().startswith("evidence:"):
            current_evidence.append(line[len("evidence:"):].strip().strip('"'))
        elif current_criterion and line and not line.startswith("**"):
            current_evidence.append(line)

    # Save last criterion
    if current_criterion:
        current_criterion.evidence = " ".join(current_evidence)
        scores.append(current_criterion)

    # If parsing found nothing, return a single null-score entry
    if not scores:
        scores.append(CriterionScore(
            criterion="parse_failure",
            passed=None,
            evidence=response_text[:500],
            raw_response=response_text,
        ))

    return scores
```

- [ ] **Step 2: Create observation quality judge prompt**

`apps/api/evals/shared/prompts/observation_quality_judge_v1.txt`:

```
You are evaluating a piano teaching observation generated by an AI practice companion.

## Performance Context
- Piece: {piece_name} (bars {bar_range})
- Dimension scores: {predictions}
- Student baseline: {baselines}
- Recent observations: {recent_observations}

## Analysis Context
{analysis_facts}

## Observation Being Evaluated
"{observation_text}"

## Evaluate each criterion below. For each, answer YES or NO and quote evidence.

**Musical Accuracy:** Does the observation describe something that is actually true about this performance? Quote the specific claim and state whether the dimension scores and analysis context support it. Answer YES or NO.
Evidence:

**Specificity:** Does the observation reference a concrete musical moment (bar number, passage name, phrase boundary, specific notes)? Quote the reference. If it says only "your dynamics" without locating where, answer NO.
Evidence:

**Actionability:** Could the student change something specific in their next attempt based on this observation? Quote the suggested action. Generic advice like "practice more" or "keep working on it" = NO.
Evidence:

**Tone:** Is the tone warm and encouraging without being condescending or vague? Quote any problematic phrasing. If none, answer YES.
Evidence:

**Dimension Appropriateness:** Given the dimension scores and student baseline, was this the most valuable dimension to teach? If a different dimension had a larger negative deviation from baseline and was not covered in the recent observations listed above, answer NO and name which dimension should have been selected instead.
Evidence:
```

- [ ] **Step 3: Create subagent reasoning judge prompt**

`apps/api/evals/shared/prompts/subagent_reasoning_judge_v1.txt`:

```
You are evaluating a subagent's teaching moment analysis for a piano practice companion.

The subagent receives performance context and must select which dimension to teach, choose an appropriate framing, and provide reasoning.

## Performance Context
- Dimension scores: {predictions}
- Student baseline: {baselines}
- Recent observations (last 3 dimensions taught): {recent_observations}

## Subagent Output Being Evaluated
- Selected dimension: {dimension}
- Framing: {framing}
- Reasoning trace: {reasoning_trace}

## Evaluate each criterion below. For each, answer YES or NO and provide evidence.

**Dimension Selection:** Compute the deviation from baseline for each dimension (score - baseline). The subagent should select the dimension with the largest negative deviation that was NOT covered in the recent observations. Show all deviations. Does the selected dimension match? Answer YES or NO.
Evidence:

**Framing Match:** Given the student's trajectory on the selected dimension (compare score vs baseline: if score < baseline, student is declining; if score > baseline, improving), is the framing appropriate? Correction for decline, recognition for improvement, encouragement for stable, question for exploration. Answer YES or NO.
Evidence:

**Reasoning Coherence:** Does the reasoning trace logically support the dimension choice and framing? Quote any non-sequiturs, unsupported claims, or contradictions. If the reasoning is coherent, answer YES.
Evidence:
```

- [ ] **Step 4: Verify imports**

```bash
cd apps/api/evals && uv run python -c "from shared.judge import load_prompt; print(load_prompt('observation_quality_judge_v1.txt')[:100])"
```

- [ ] **Step 5: Commit**

```bash
git add apps/api/evals/shared/judge.py apps/api/evals/shared/prompts/
git commit -m "feat(evals): add LLM judge client with binary rubrics and versioned prompts"
```

---

### Task 7: Shared Eval Infrastructure -- Traces

**Files:**
- Create: `apps/api/evals/shared/traces.py`

- [ ] **Step 1: Create trace writer**

`apps/api/evals/shared/traces.py`:

```python
"""Pipeline trace writer for eval debuggability.

Writes one JSON file per observation capturing the full pipeline state
at each stage: inference -> STOP -> teaching moment -> analysis ->
subagent -> teacher -> judge scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PipelineTrace:
    observation_id: str
    recording_id: str
    chunk_index: int
    # Pipeline stages (populated incrementally)
    inference: dict = field(default_factory=dict)
    stop_score: float | None = None
    stop_triggered: bool = False
    teaching_moment: dict = field(default_factory=dict)
    analysis_facts: dict = field(default_factory=dict)
    subagent_output: dict = field(default_factory=dict)
    teacher_observation: str = ""
    judge_scores: list[dict] = field(default_factory=list)

    def save(self, traces_dir: Path) -> Path:
        """Write trace to JSON file. Returns the file path."""
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.observation_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def load_trace(traces_dir: Path, observation_id: str) -> dict:
    """Load a trace file by observation ID."""
    path = traces_dir / f"{observation_id}.json"
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/evals/shared/traces.py
git commit -m "feat(evals): add pipeline trace writer for observation debuggability"
```

---

### Task 8: Preflight Checks and run_all.py

**Files:**
- Create: `apps/api/evals/run_all.py`

- [ ] **Step 1: Create run_all.py with preflight checks**

`apps/api/evals/run_all.py`:

```python
"""Pipeline evaluation orchestrator.

Usage:
    uv run python run_all.py                    # run all Phase 1 evals
    uv run python run_all.py --suite obs        # observation quality only
    uv run python run_all.py --suite subagent   # subagent reasoning only
    uv run python run_all.py --skip-preflight   # skip preflight checks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parents[2].parent / "data" / "eval"
CACHE_DIR = DATA_DIR / "inference_cache"
TRACES_DIR = DATA_DIR / "traces"
REPORTS_DIR = Path(__file__).parent / "reports"
CALIBRATION_DIR = DATA_DIR / "calibration"

WRANGLER_URL = "http://localhost:8787"

SUITES = {
    "obs": "observation_quality",
    "subagent": "subagent_reasoning",
}


def preflight() -> bool:
    """Run preflight checks. Returns True if all pass."""
    checks = []

    # 1. Inference cache exists
    from shared.inference_cache import find_cache_dir
    cache = find_cache_dir(CACHE_DIR)
    if cache:
        n_files = len(list(cache.glob("*.json")))
        checks.append(("Inference cache", True, f"{cache.name} ({n_files} recordings)"))
    else:
        checks.append(("Inference cache", False,
                        "Not found. Run: cd apps/inference && CRESCEND_DEVICE=cpu python eval_runner.py"))

    # 2. Anthropic API key
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(("ANTHROPIC_API_KEY", has_anthropic,
                    "Set" if has_anthropic else "Missing. Export ANTHROPIC_API_KEY"))

    # 3. Groq API key
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    checks.append(("GROQ_API_KEY", has_groq,
                    "Set" if has_groq else "Missing. Export GROQ_API_KEY"))

    # 4. Wrangler dev responding (only needed for observation quality)
    wrangler_ok = False
    try:
        resp = requests.get(f"{WRANGLER_URL}/health", timeout=3)
        wrangler_ok = resp.status_code == 200
        checks.append(("wrangler dev", wrangler_ok, "Responding on :8787"))
    except requests.ConnectionError:
        checks.append(("wrangler dev", False,
                        "Not running. Start with: cd apps/api && npx wrangler dev"))

    # 5. D1 seeded with score catalog (query a known piece via worker)
    if wrangler_ok:
        try:
            # Use the exercises endpoint as a proxy for D1 seeding
            resp = requests.get(f"{WRANGLER_URL}/api/exercises", timeout=5)
            d1_ok = resp.status_code == 200
            checks.append(("D1 score catalog", d1_ok,
                            "Seeded" if d1_ok else "Empty. Seed with: cd apps/api && npx wrangler d1 execute ..."))
        except requests.ConnectionError:
            checks.append(("D1 score catalog", False, "Could not verify"))
    else:
        checks.append(("D1 score catalog", False, "Skipped (wrangler not running)"))

    # 6. Worker LLM keys (.dev.vars)
    dev_vars_path = Path(__file__).parents[1] / ".dev.vars"
    if dev_vars_path.exists():
        dev_vars_content = dev_vars_path.read_text()
        has_worker_keys = "GROQ_API_KEY" in dev_vars_content and "ANTHROPIC_API_KEY" in dev_vars_content
        checks.append(("Worker LLM keys (.dev.vars)", has_worker_keys,
                        "Both keys present" if has_worker_keys else "Missing GROQ_API_KEY or ANTHROPIC_API_KEY in apps/api/.dev.vars"))
    else:
        checks.append(("Worker LLM keys (.dev.vars)", False,
                        f"File not found: {dev_vars_path}. Create with GROQ_API_KEY and ANTHROPIC_API_KEY"))

    # Print results
    print("\nPreflight Checks:")
    print("-" * 60)
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            all_ok = False
    print("-" * 60)

    return all_ok


def run_observation_quality() -> None:
    """Run the observation quality eval suite."""
    print("\n>>> Running observation quality eval...")
    from observation_quality.eval_observation_quality import main as obs_main
    obs_main(
        cache_dir=CACHE_DIR,
        traces_dir=TRACES_DIR,
        reports_dir=REPORTS_DIR,
        wrangler_url=WRANGLER_URL,
    )


def run_subagent_reasoning() -> None:
    """Run the subagent reasoning eval suite."""
    print("\n>>> Running subagent reasoning eval...")
    from subagent_reasoning.eval_subagent_reasoning import main as sub_main
    sub_main(reports_dir=REPORTS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Pipeline evaluation runner")
    parser.add_argument("--suite", choices=list(SUITES.keys()), help="Run specific suite")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    args = parser.parse_args()

    if not args.skip_preflight:
        if not preflight():
            print("\nPreflight failed. Fix the issues above and retry.")
            sys.exit(1)
        print()

    if args.suite:
        if args.suite == "obs":
            run_observation_quality()
        elif args.suite == "subagent":
            run_subagent_reasoning()
    else:
        # Run all Phase 1 suites
        run_observation_quality()
        run_subagent_reasoning()

    print("\nAll evals complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/evals/run_all.py
git commit -m "feat(evals): add run_all.py orchestrator with preflight checks"
```

---

## Chunk 2: Observation Quality Eval (Phase 1a)

### Task 9: WebSocket Client for Eval Pipeline

**Files:**
- Create: `apps/api/evals/shared/pipeline_client.py`

- [ ] **Step 1: Create the pipeline client**

This client connects to wrangler dev via WebSocket, creates a practice session, sends eval-chunk messages, and collects observations.

`apps/api/evals/shared/pipeline_client.py`:

```python
"""WebSocket client for sending eval chunks to the local Rust pipeline.

Connects to wrangler dev, creates a practice session, sends pre-computed
inference results via eval_chunk messages, and collects observations.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

import requests
import websockets
import asyncio


@dataclass
class PipelineObservation:
    """An observation returned by the pipeline."""
    text: str
    dimension: str
    framing: str
    chunk_index: int
    score: float
    baseline: float
    reasoning_trace: str
    is_fallback: bool = False
    raw_message: dict = field(default_factory=dict)


@dataclass
class SessionResult:
    """Result of running a full recording through the pipeline."""
    session_id: str
    recording_id: str
    observations: list[PipelineObservation]
    chunk_responses: list[dict]  # chunk_processed messages
    errors: list[str]
    duration_ms: int


async def run_recording(
    wrangler_url: str,
    recording_cache: dict,
    student_id: str = "eval-student-001",
    piece_query: str | None = None,
) -> SessionResult:
    """Run a cached recording through the full pipeline via wrangler dev.

    Args:
        wrangler_url: Base URL for wrangler dev (e.g., http://localhost:8787).
        recording_cache: Loaded cache JSON for one recording.
        student_id: Student ID to use for the session.
        piece_query: Optional piece name for score matching.

    Returns:
        SessionResult with observations and chunk responses.
    """
    recording_id = recording_cache["recording_id"]
    chunks = recording_cache["chunks"]
    start = time.time()

    # 1. Create practice session via HTTP
    session_id = str(uuid.uuid4())
    resp = requests.post(
        f"{wrangler_url}/api/practice/start",
        json={
            "session_id": session_id,
            "student_id": student_id,
            "piece_query": piece_query,
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return SessionResult(
            session_id=session_id,
            recording_id=recording_id,
            observations=[],
            chunk_responses=[],
            errors=[f"Failed to start session: {resp.status_code} {resp.text}"],
            duration_ms=0,
        )

    # 2. Connect WebSocket (localhost only -- eval never targets remote)
    if "localhost" not in wrangler_url and "127.0.0.1" not in wrangler_url:
        raise ValueError(
            f"Eval pipeline client only connects to localhost, got: {wrangler_url}"
        )
    from urllib.parse import urlparse
    parsed = urlparse(wrangler_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/api/practice/ws/{session_id}"

    observations = []
    chunk_responses = []
    errors = []

    try:
        async with websockets.connect(ws_url) as ws:
            # 3. Send each chunk as eval_chunk
            for chunk in chunks:
                msg = {
                    "type": "eval_chunk",
                    "chunk_index": chunk["chunk_index"],
                    "predictions": chunk["predictions"],
                    "midi_notes": chunk.get("midi_notes", []),
                    "pedal_events": chunk.get("pedal_events", []),
                }
                await ws.send(json.dumps(msg))

                # Collect responses (chunk_processed + possible observation)
                # Wait for chunk_processed, with timeout
                try:
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        response = json.loads(raw)
                        msg_type = response.get("type", "")

                        if msg_type == "chunk_processed":
                            chunk_responses.append(response)
                            break
                        elif msg_type == "observation":
                            observations.append(PipelineObservation(
                                text=response.get("text", ""),
                                dimension=response.get("dimension", ""),
                                framing=response.get("framing", ""),
                                chunk_index=response.get("chunk_index", 0),
                                score=response.get("score", 0.0),
                                baseline=response.get("baseline", 0.0),
                                reasoning_trace=response.get("reasoning_trace", ""),
                                is_fallback=response.get("is_fallback", False),
                                raw_message=response,
                            ))
                        elif msg_type == "error":
                            errors.append(response.get("message", "unknown error"))
                            break
                except asyncio.TimeoutError:
                    errors.append(f"Timeout waiting for chunk {chunk['chunk_index']}")

            # 4. After all chunks, wait briefly for any trailing observations
            try:
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    response = json.loads(raw)
                    if response.get("type") == "observation":
                        observations.append(PipelineObservation(
                            text=response.get("text", ""),
                            dimension=response.get("dimension", ""),
                            framing=response.get("framing", ""),
                            chunk_index=response.get("chunk_index", 0),
                            score=response.get("score", 0.0),
                            baseline=response.get("baseline", 0.0),
                            reasoning_trace=response.get("reasoning_trace", ""),
                            is_fallback=response.get("is_fallback", False),
                            raw_message=response,
                        ))
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

    except Exception as e:
        errors.append(f"WebSocket error: {e}")

    duration_ms = int((time.time() - start) * 1000)

    return SessionResult(
        session_id=session_id,
        recording_id=recording_id,
        observations=observations,
        chunk_responses=chunk_responses,
        errors=errors,
        duration_ms=duration_ms,
    )
```

- [ ] **Step 2: Commit**

```bash
git add apps/api/evals/shared/pipeline_client.py
git commit -m "feat(evals): add WebSocket pipeline client for eval-chunk sessions"
```

---

### Task 10: Observation Quality Eval Runner

**Files:**
- Create: `apps/api/evals/observation_quality/__init__.py`
- Create: `apps/api/evals/observation_quality/eval_observation_quality.py`

- [ ] **Step 1: Create the eval runner**

`apps/api/evals/observation_quality/eval_observation_quality.py`:

```python
"""Observation quality evaluation.

Runs YouTube recordings through the full pipeline via wrangler dev,
then scores each observation with the LLM judge on 5 binary criteria.
"""

from __future__ import annotations

import asyncio
import statistics
from pathlib import Path

from shared.inference_cache import find_cache_dir, load_all_recordings, validate_cache
from shared.judge import judge_observation, CriterionScore
from shared.pipeline_client import run_recording, SessionResult
from shared.reporting import EvalReport, MetricResult
from shared.traces import PipelineTrace

CRITERIA = [
    "Musical Accuracy",
    "Specificity",
    "Actionability",
    "Tone",
    "Dimension Appropriateness",
]

PASS_THRESHOLDS = {
    "Musical Accuracy": 0.70,
    "Specificity": 0.60,
    "Actionability": 0.60,
    "Tone": 0.80,
    "Dimension Appropriateness": 0.60,
}


def main(
    cache_dir: Path,
    traces_dir: Path,
    reports_dir: Path,
    wrangler_url: str,
) -> EvalReport:
    """Run the observation quality eval."""
    # Load inference cache
    versioned_cache = find_cache_dir(cache_dir)
    if not versioned_cache:
        raise FileNotFoundError(f"No inference cache found in {cache_dir}")

    fingerprint = validate_cache(versioned_cache)
    recordings = load_all_recordings(versioned_cache)
    print(f"Loaded {len(recordings)} recordings from cache ({fingerprint})")

    # Run each recording through the pipeline
    all_scores: dict[str, list[bool]] = {c: [] for c in CRITERIA}
    worst_cases: list[dict] = []
    total_judge_calls = 0
    total_observations = 0
    no_observation_count = 0

    for i, recording in enumerate(recordings):
        recording_id = recording["recording_id"]
        print(f"  [{i+1}/{len(recordings)}] {recording_id}...", end=" ", flush=True)

        # Run through pipeline
        result: SessionResult = asyncio.run(
            run_recording(wrangler_url, recording)
        )

        if result.errors:
            print(f"ERRORS: {result.errors}")
            continue

        if not result.observations:
            print("no observations")
            no_observation_count += 1
            continue

        total_observations += len(result.observations)

        # Judge each observation
        for obs in result.observations:
            # Use the eval_context echoed back by the Rust pipeline.
            # This contains the exact context the teacher LLM saw:
            # predictions, baselines, recent_observations, analysis_facts, piece_name.
            eval_ctx = obs.raw_message.get("eval_context", {})
            context = {
                "predictions": eval_ctx.get("predictions", {}),
                "baselines": eval_ctx.get("baselines", {}),
                "recent_observations": eval_ctx.get("recent_observations", []),
                "analysis_facts": eval_ctx.get("analysis_facts", {}),
                "piece_name": eval_ctx.get("piece_name", recording_id),
                "bar_range": eval_ctx.get("bar_range", "full recording"),
            }

            judge_result = judge_observation(obs.text, context)
            total_judge_calls += 1

            # Write trace
            trace = PipelineTrace(
                observation_id=f"{recording_id}_chunk{obs.chunk_index}",
                recording_id=recording_id,
                chunk_index=obs.chunk_index,
                inference=recording["chunks"][obs.chunk_index]
                    if obs.chunk_index < len(recording["chunks"])
                    else {},
                teacher_observation=obs.text,
                judge_scores=[
                    {"criterion": s.criterion, "passed": s.passed, "evidence": s.evidence}
                    for s in judge_result.scores
                ],
            )
            trace.save(traces_dir)

            # Aggregate scores by criterion
            for score in judge_result.scores:
                if score.criterion in all_scores and score.passed is not None:
                    all_scores[score.criterion].append(score.passed)

                    # Track worst cases (any criterion that failed)
                    if not score.passed:
                        worst_cases.append({
                            "recording_id": recording_id,
                            "chunk_index": obs.chunk_index,
                            "criterion": score.criterion,
                            "observation": obs.text[:200],
                            "evidence": score.evidence[:200],
                            "trace_file": f"{recording_id}_chunk{obs.chunk_index}.json",
                        })

        print(f"{len(result.observations)} obs, {result.duration_ms}ms")

    # Build report
    report = EvalReport(
        eval_name="observation_quality",
        eval_version="1.0",
        dataset=f"youtube_amt_{len(recordings)}",
    )
    report.metadata["model_fingerprint"] = fingerprint
    report.metadata["total_observations"] = total_observations
    report.metadata["no_observation_recordings"] = no_observation_count

    for criterion in CRITERIA:
        scores = all_scores[criterion]
        if scores:
            mean = sum(scores) / len(scores)
            # std of binary values
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            report.metrics[criterion] = MetricResult(
                mean=mean,
                std=std,
                n=len(scores),
                pass_threshold=PASS_THRESHOLDS.get(criterion),
            )

    report.worst_cases = sorted(worst_cases, key=lambda x: x["criterion"])[:20]
    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": total_judge_calls * 0.003,  # rough estimate
    }

    report.save(reports_dir / "observation_quality.json")
    report.print_summary()
    return report
```

- [ ] **Step 2: Create __init__.py**

`apps/api/evals/observation_quality/__init__.py`: empty file.

- [ ] **Step 3: Commit**

```bash
git add apps/api/evals/observation_quality/
git commit -m "feat(evals): add observation quality eval with 5-criterion LLM judge"
```

---

## Chunk 3: Subagent Reasoning Eval (Phase 1b)

### Task 11: Hand-Crafted Subagent Scenarios

**Files:**
- Create: `apps/api/evals/subagent_reasoning/__init__.py`
- Create: `apps/api/evals/subagent_reasoning/scenarios/scenarios.json`
- Create: `apps/api/evals/subagent_reasoning/eval_subagent_reasoning.py`

- [ ] **Step 1: Create scenarios file**

`apps/api/evals/subagent_reasoning/scenarios/scenarios.json` -- starter set of 10 scenarios (expand to ~35 during calibration):

```json
[
  {
    "id": "obvious_dynamics_decline",
    "description": "Dynamics clearly worst dimension, large negative deviation",
    "predictions": {"dynamics": 0.30, "timing": 0.65, "pedaling": 0.60, "articulation": 0.70, "phrasing": 0.55, "interpretation": 0.50},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [],
    "expected_dimension": "dynamics",
    "expected_framing": "correction"
  },
  {
    "id": "all_improving",
    "description": "All dimensions above baseline, should pick positive recognition",
    "predictions": {"dynamics": 0.70, "timing": 0.72, "pedaling": 0.68, "articulation": 0.75, "phrasing": 0.65, "interpretation": 0.60},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [],
    "expected_dimension": null,
    "expected_framing": "recognition"
  },
  {
    "id": "dedup_dynamics_repeated",
    "description": "Dynamics worst but already covered 3 times, should pick next worst",
    "predictions": {"dynamics": 0.30, "timing": 0.65, "pedaling": 0.40, "articulation": 0.70, "phrasing": 0.55, "interpretation": 0.50},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [{"dimension": "dynamics"}, {"dimension": "dynamics"}, {"dimension": "dynamics"}],
    "expected_dimension": "pedaling",
    "expected_framing": "correction"
  },
  {
    "id": "borderline_stop",
    "description": "Scores very close to baseline, borderline STOP",
    "predictions": {"dynamics": 0.52, "timing": 0.58, "pedaling": 0.56, "articulation": 0.63, "phrasing": 0.50, "interpretation": 0.46},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [],
    "expected_dimension": "articulation",
    "expected_framing": "correction"
  },
  {
    "id": "mixed_signals",
    "description": "One dimension declining sharply, another improving sharply",
    "predictions": {"dynamics": 0.30, "timing": 0.80, "pedaling": 0.60, "articulation": 0.70, "phrasing": 0.55, "interpretation": 0.50},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [],
    "expected_dimension": "dynamics",
    "expected_framing": "correction"
  },
  {
    "id": "no_baseline_first_session",
    "description": "No baseline data, first session ever",
    "predictions": {"dynamics": 0.45, "timing": 0.55, "pedaling": 0.50, "articulation": 0.60, "phrasing": 0.40, "interpretation": 0.35},
    "baselines": {},
    "recent_observations": [],
    "expected_dimension": null,
    "expected_framing": "encouragement"
  },
  {
    "id": "phrasing_declining_after_recognition",
    "description": "Phrasing was recognized last time but now declining",
    "predictions": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.35, "interpretation": 0.48},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [{"dimension": "phrasing"}],
    "expected_dimension": "phrasing",
    "expected_framing": "correction"
  },
  {
    "id": "interpretation_consistently_low",
    "description": "Interpretation always low, never addressed",
    "predictions": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.25},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.30},
    "recent_observations": [{"dimension": "dynamics"}, {"dimension": "pedaling"}, {"dimension": "timing"}],
    "expected_dimension": "interpretation",
    "expected_framing": "correction"
  },
  {
    "id": "all_stable",
    "description": "Everything near baseline, no strong signal",
    "predictions": {"dynamics": 0.54, "timing": 0.59, "pedaling": 0.57, "articulation": 0.64, "phrasing": 0.51, "interpretation": 0.47},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [],
    "expected_dimension": null,
    "expected_framing": "encouragement"
  },
  {
    "id": "timing_worst_but_recently_covered",
    "description": "Timing worst deviation, covered once recently but not 3 times",
    "predictions": {"dynamics": 0.50, "timing": 0.35, "pedaling": 0.55, "articulation": 0.60, "phrasing": 0.50, "interpretation": 0.45},
    "baselines": {"dynamics": 0.55, "timing": 0.60, "pedaling": 0.58, "articulation": 0.65, "phrasing": 0.52, "interpretation": 0.48},
    "recent_observations": [{"dimension": "timing"}],
    "expected_dimension": "timing",
    "expected_framing": "correction"
  }
]
```

- [ ] **Step 2: Create eval runner**

`apps/api/evals/subagent_reasoning/eval_subagent_reasoning.py`:

```python
"""Subagent reasoning evaluation.

Tests the Groq subagent's ability to select the right dimension,
choose appropriate framing, and produce coherent reasoning from
hand-crafted scenarios.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

from groq import Groq

from shared.judge import judge_subagent
from shared.reporting import EvalReport, MetricResult

SCENARIOS_PATH = Path(__file__).parent / "scenarios" / "scenarios.json"


def load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def build_subagent_prompt(scenario: dict) -> str:
    """Build the subagent prompt matching production format.

    This should match the prompt in apps/api/src/services/prompts.rs
    as closely as possible.
    """
    predictions = scenario["predictions"]
    baselines = scenario.get("baselines", {})
    recent = scenario.get("recent_observations", [])

    recent_dims = ", ".join(o["dimension"] for o in recent) if recent else "none"

    # Compute deviations
    deviations = {}
    for dim, score in predictions.items():
        baseline = baselines.get(dim)
        if baseline is not None:
            deviations[dim] = round(score - baseline, 3)

    return f"""You are a piano teaching assistant analyzing a student's performance.

## Performance Scores (0-1, higher = better)
{json.dumps(predictions, indent=2)}

## Student Baseline (running average)
{json.dumps(baselines, indent=2) if baselines else "No baseline yet (first session)"}

## Deviations from Baseline
{json.dumps(deviations, indent=2) if deviations else "N/A (no baseline)"}

## Recently Covered Dimensions (last 3 observations)
{recent_dims}

## Your Task
Select ONE dimension to teach about and provide:
1. The dimension name
2. A framing: "correction" (student declining), "recognition" (student improving), "encouragement" (stable/first session), or "question" (exploratory)
3. A brief reasoning trace explaining your choice

Respond in this exact JSON format:
{{
  "dimension": "<dimension_name>",
  "framing": "<correction|recognition|encouragement|question>",
  "reasoning_trace": "<1-2 sentences explaining your selection>"
}}"""


def call_subagent(prompt: str) -> dict:
    """Call Groq with the subagent prompt."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500,
    )
    text = response.choices[0].message.content.strip()

    # Parse JSON from response (handle markdown code blocks)
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    return json.loads(text)


def main(reports_dir: Path) -> EvalReport:
    """Run the subagent reasoning eval."""
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")

    dimension_correct = []
    framing_correct = []
    reasoning_coherent = []
    total_judge_calls = 0

    for i, scenario in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] {scenario['id']}...", end=" ", flush=True)

        # Call the subagent
        prompt = build_subagent_prompt(scenario)
        try:
            output = call_subagent(prompt)
        except (json.JSONDecodeError, Exception) as e:
            print(f"SKIP (subagent error: {e})")
            continue

        # Quick check: dimension selection (deterministic, no judge needed)
        expected_dim = scenario.get("expected_dimension")
        if expected_dim is not None:
            dim_match = output.get("dimension") == expected_dim
            dimension_correct.append(dim_match)
        else:
            # expected_dimension is null = any dimension is acceptable
            dimension_correct.append(True)

        # Quick check: framing match (deterministic)
        expected_framing = scenario.get("expected_framing")
        if expected_framing:
            framing_match = output.get("framing") == expected_framing
            framing_correct.append(framing_match)

        # Judge reasoning coherence (needs LLM)
        judge_result = judge_subagent(output, scenario)
        total_judge_calls += 1

        coherence_scores = [s for s in judge_result.scores if "coherence" in s.criterion.lower()]
        if coherence_scores and coherence_scores[0].passed is not None:
            reasoning_coherent.append(coherence_scores[0].passed)

        status = "PASS" if (expected_dim is None or dim_match) else "FAIL"
        print(f"{status} (picked: {output.get('dimension')}, framing: {output.get('framing')})")

    # Build report
    report = EvalReport(
        eval_name="subagent_reasoning",
        eval_version="1.0",
        dataset=f"scenarios_{len(scenarios)}",
    )

    if dimension_correct:
        report.metrics["dimension_selection"] = MetricResult(
            mean=sum(dimension_correct) / len(dimension_correct),
            std=statistics.stdev(dimension_correct) if len(dimension_correct) > 1 else 0.0,
            n=len(dimension_correct),
            pass_threshold=0.80,
        )
    if framing_correct:
        report.metrics["framing_match"] = MetricResult(
            mean=sum(framing_correct) / len(framing_correct),
            std=statistics.stdev(framing_correct) if len(framing_correct) > 1 else 0.0,
            n=len(framing_correct),
            pass_threshold=0.75,
        )
    if reasoning_coherent:
        report.metrics["reasoning_coherence"] = MetricResult(
            mean=sum(reasoning_coherent) / len(reasoning_coherent),
            std=statistics.stdev(reasoning_coherent) if len(reasoning_coherent) > 1 else 0.0,
            n=len(reasoning_coherent),
            pass_threshold=0.70,
        )

    report.cost = {
        "judge_calls": total_judge_calls,
        "estimated_usd": total_judge_calls * 0.003,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    report.save(reports_dir / "subagent_reasoning.json")
    report.print_summary()
    return report
```

- [ ] **Step 3: Create __init__.py**

`apps/api/evals/subagent_reasoning/__init__.py`: empty file.

- [ ] **Step 4: Commit**

```bash
git add apps/api/evals/subagent_reasoning/
git commit -m "feat(evals): add subagent reasoning eval with 10 starter scenarios"
```

---

### Task 12: .gitignore and Data Directory Setup

**Files:**
- Modify: `.gitignore`
- Create: `data/eval/.gitkeep`
- Create: `data/eval/youtube_amt/.gitkeep`
- Create: `apps/api/evals/reports/.gitkeep`

- [ ] **Step 1: Update .gitignore**

Add to `.gitignore`:
```
# Eval data (large files, cached inference, traces)
data/eval/inference_cache/
data/eval/traces/
data/eval/calibration/
data/eval/youtube_amt/*.wav
data/eval/youtube_amt/*.mp3
data/eval/youtube_amt/*.flac
data/eval/youtube_amt/*.ogg
data/eval/youtube_amt/*.webm
data/eval/youtube_amt/*.m4a
data/eval/youtube_amt/*.opus

# Eval reports (regenerated)
apps/api/evals/reports/
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p data/eval/youtube_amt data/eval/inference_cache data/eval/traces data/eval/calibration
touch data/eval/.gitkeep data/eval/youtube_amt/.gitkeep
mkdir -p apps/api/evals/reports
touch apps/api/evals/reports/.gitkeep
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore data/eval/.gitkeep data/eval/youtube_amt/.gitkeep apps/api/evals/reports/.gitkeep
git commit -m "feat(evals): add data directory structure and gitignore for eval artifacts"
```

---

### Task 13: Final Integration Smoke Test

- [ ] **Step 1: Verify the full setup**

```bash
cd apps/api/evals && uv sync
uv run python -c "
from shared.reporting import EvalReport, MetricResult
from shared.inference_cache import find_cache_dir, StaleCacheError
from shared.traces import PipelineTrace
from shared.judge import load_prompt
print('All imports OK')

# Test reporting
r = EvalReport('test', '0.1', 'smoke_test')
r.metrics['accuracy'] = MetricResult(mean=0.85, std=0.1, n=10, pass_threshold=0.70)
r.print_summary()
print('Reporting OK')

# Test prompt loading
p = load_prompt('observation_quality_judge_v1.txt')
assert '{observation_text}' in p
print('Prompts OK')
"
```

- [ ] **Step 2: Verify preflight detects missing pieces**

```bash
cd apps/api/evals && uv run python run_all.py
```

Expected: Preflight should show FAIL for inference cache (not yet generated) and possibly wrangler dev (not running). This confirms preflight catches missing prerequisites.

- [ ] **Step 3: Commit any fixes from smoke test**

Only if needed. If smoke test passes cleanly, skip.

---

## Execution Notes

**Task dependencies:**
- Tasks 1-2 (inference changes) are independent of Tasks 3-12 (eval framework)
- Task 3 (eval-chunk endpoint) can be done in parallel with Tasks 4-8
- Tasks 9-10 (observation quality) depend on Tasks 3-8
- Task 11 (subagent reasoning) depends on Tasks 4, 6 only (no wrangler dev needed)

**Parallelization opportunities:**
- Agent A: Tasks 1-2 (inference device changes + runner)
- Agent B: Tasks 4-8 (shared infrastructure + run_all)
- Agent C: Task 3 (Rust eval-chunk endpoint)
- Agent D: Task 11 (subagent scenarios + eval)
- Sequential: Tasks 9-10 (after A, B, C complete), Task 12-13 (after all)

**After this plan:** Phase 1 is complete when `uv run python run_all.py` runs both evals and prints results. The calibration step (scoring 20 observations manually, computing kappa) is a human task that follows.
