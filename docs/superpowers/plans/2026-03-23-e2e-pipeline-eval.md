# E2E Pipeline Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a three-layer evaluation harness that tests the full CrescendAI pipeline (STOP -> teaching moments -> observations) on 312 T5 YouTube Skill recordings, measuring observation quality, STOP generalization, and piece identification accuracy.

**Architecture:** Extends existing eval framework with: (1) HTTP-based inference cache generation for T5 recordings via local MuQ/AMT servers, (2) T5 scenario files + judge v3 prompt + STOP/piece-ID metrics in the pipeline eval, (3) new analysis script for cross-cutting statistical analysis. All evaluation routes through the real system (wrangler dev via WebSocket).

**Tech Stack:** Python 3.12, PyYAML, scipy (bootstrap CIs, Spearman, Cohen's d), httpx (HTTP client for local inference), tqdm (progress bars), existing eval infrastructure (pipeline_client.py, judge.py, reporting.py)

**Spec:** `docs/superpowers/specs/2026-03-23-e2e-pipeline-eval-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `apps/evals/pipeline/practice_eval/generate_t5_scenarios.py` | Create | Read T5 manifests, write scenario YAMLs |
| `apps/evals/pipeline/practice_eval/scenarios/t5_*.yaml` | Create (4) | T5 scenario files (generated, not hand-written) |
| `apps/evals/shared/prompts/observation_quality_judge_v3.txt` | Create | Judge prompt with skill-appropriateness criterion |
| `apps/evals/shared/pipeline_client.py` | Modify | Capture `piece_identified` WebSocket messages |
| `apps/evals/inference/eval_runner.py` | Modify | Add `--auto-t5` flag with HTTP client mode |
| `apps/evals/pipeline/practice_eval/eval_practice.py` | Modify | Add `--scenarios` flag, checkpointing, STOP/piece-ID metric extraction |
| `apps/evals/pipeline/practice_eval/analyze_e2e.py` | Create | Statistical analysis (Spearman, Cohen's d, bootstrap CIs, confusion matrix) |
| `Justfile` | Modify | Add `eval-e2e`, `eval-cache`, `eval-pipeline`, `eval-analyze` commands |
| `apps/evals/tests/test_generate_scenarios.py` | Create | Unit tests for scenario generation |
| `apps/evals/tests/test_analyze_e2e.py` | Create | Unit tests for analysis computations |

---

### Task 1: T5 Scenario Generation Script

**Files:**
- Create: `apps/evals/pipeline/practice_eval/generate_t5_scenarios.py`
- Read: `model/data/evals/skill_eval/{piece}/manifest.yaml`
- Output: `apps/evals/pipeline/practice_eval/scenarios/t5_{piece}.yaml`
- Test: `apps/evals/tests/test_generate_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
# apps/evals/tests/test_generate_scenarios.py
"""Tests for T5 scenario generation."""
import tempfile
from pathlib import Path

import yaml


def test_manifest_to_scenario_basic():
    """A manifest entry with skill_bucket maps to scenario with skill_level."""
    from pipeline.practice_eval.generate_t5_scenarios import manifest_to_scenarios

    manifest = {
        "piece": "fur_elise",
        "recordings": [
            {
                "video_id": "abc123",
                "title": "Test Recording",
                "channel": "Test Channel",
                "skill_bucket": 3,
                "downloaded": True,
            },
        ],
    }
    result = manifest_to_scenarios(manifest)
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["video_id"] == "abc123"
    assert candidate["skill_level"] == 3
    assert candidate["include"] is True
    # No piece_query -- forces automatic piece identification
    assert "piece_query" not in result


def test_manifest_skips_not_downloaded():
    """Recordings that failed to download are excluded."""
    from pipeline.practice_eval.generate_t5_scenarios import manifest_to_scenarios

    manifest = {
        "piece": "fur_elise",
        "recordings": [
            {"video_id": "good", "title": "OK", "skill_bucket": 2, "downloaded": True},
            {"video_id": "bad", "title": "Failed", "skill_bucket": 3, "downloaded": False},
        ],
    }
    result = manifest_to_scenarios(manifest)
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["video_id"] == "good"


def test_generate_writes_yaml(tmp_path):
    """generate_scenario_file writes valid YAML to the output path."""
    from pipeline.practice_eval.generate_t5_scenarios import generate_scenario_file

    manifest = {
        "piece": "test_piece",
        "recordings": [
            {"video_id": "v1", "title": "T1", "skill_bucket": 1, "downloaded": True},
        ],
    }
    out = tmp_path / "t5_test_piece.yaml"
    generate_scenario_file(manifest, out)

    assert out.exists()
    data = yaml.safe_load(out.read_text())
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["skill_level"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd apps/evals && uv run python -m pytest tests/test_generate_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Write the scenario generation module**

```python
# apps/evals/pipeline/practice_eval/generate_t5_scenarios.py
"""Generate T5 scenario YAML files from skill_eval manifest.yaml files.

Reads manifest.yaml (with skill_bucket labels) and writes scenario YAML
in the format load_scenarios() expects (candidates with video_id, include,
skill_level). Deliberately omits piece_query to test automatic piece
identification via the N-gram pipeline.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.generate_t5_scenarios
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))
from paths import MODEL_DATA

T5_PIECES = [
    "bach_prelude_c_wtc1",
    "bach_invention_1",
    "fur_elise",
    "nocturne_op9no2",
]

MANIFEST_BASE = MODEL_DATA / "evals" / "skill_eval"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def manifest_to_scenarios(manifest: dict) -> dict:
    """Convert a T5 manifest dict to the scenario YAML format."""
    candidates = []
    for rec in manifest.get("recordings", []):
        if not rec.get("downloaded", False):
            continue
        candidates.append({
            "video_id": rec["video_id"],
            "include": True,
            "skill_level": rec["skill_bucket"],
            "title": rec.get("title", ""),
            "general_notes": f"T5 skill corpus, bucket {rec['skill_bucket']}",
        })
    # No piece_query -- forces automatic piece identification
    return {"candidates": candidates}


def generate_scenario_file(manifest: dict, output_path: Path) -> int:
    """Write a scenario YAML file. Returns number of candidates written."""
    scenario = manifest_to_scenarios(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(scenario, f, default_flow_style=False, sort_keys=False)
    return len(scenario["candidates"])


def main():
    total = 0
    for piece_id in T5_PIECES:
        manifest_path = MANIFEST_BASE / piece_id / "manifest.yaml"
        if not manifest_path.exists():
            print(f"  {piece_id}: no manifest.yaml, skipping")
            continue
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        out = SCENARIOS_DIR / f"t5_{piece_id}.yaml"
        n = generate_scenario_file(manifest, out)
        total += n
        print(f"  {piece_id}: {n} candidates -> {out.name}")
    print(f"\nTotal: {total} candidates across {len(T5_PIECES)} pieces")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/evals && uv run python -m pytest tests/test_generate_scenarios.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Generate the actual scenario files**

Run: `cd apps/evals && uv run python -m pipeline.practice_eval.generate_t5_scenarios`
Expected: Output showing 4 pieces with candidate counts

- [ ] **Step 6: Commit**

```bash
git add apps/evals/pipeline/practice_eval/generate_t5_scenarios.py \
       apps/evals/pipeline/practice_eval/scenarios/t5_*.yaml \
       apps/evals/tests/test_generate_scenarios.py
git commit -m "feat: T5 scenario generation for E2E pipeline eval"
```

---

### Task 2: Judge Prompt v3 (Skill-Appropriateness Criterion)

**Files:**
- Read: `apps/evals/shared/prompts/observation_quality_judge_v2.txt`
- Create: `apps/evals/shared/prompts/observation_quality_judge_v3.txt`

- [ ] **Step 1: Create v3 prompt by copying v2 and adding the skill-appropriateness criterion**

Copy `observation_quality_judge_v2.txt` to `observation_quality_judge_v3.txt`. Add these changes:

1. Add `Student skill level: {skill_level}` to the Performance Context section (after the baselines line)
2. Add this 6th criterion at the end:

```
**Skill Appropriateness:** Given that this student is at skill level {skill_level} (1=beginner, 2=early intermediate, 3=intermediate, 4=advanced, 5=professional), does the observation match their developmental stage? A beginner (1-2) needs fundamentals: hand position, rhythm accuracy, basic dynamics, note correctness. An intermediate (3) needs musical concepts: phrasing, pedaling technique, voicing, dynamic shaping. An advanced/professional (4-5) needs nuance: interpretive choices, stylistic references, subtle voicing, rubato, and sophisticated pedaling. Score YES if the observation's language complexity, technical vocabulary, and focus area are appropriate for the stated level. Score NO if the observation talks over the student's head or beneath their level.
Evidence:
```

- [ ] **Step 2: Verify the prompt template has all 6 criterion placeholders**

Run: `grep -c "^\*\*.*:\*\*" apps/evals/shared/prompts/observation_quality_judge_v3.txt`
Expected: `6`

- [ ] **Step 3: Commit**

```bash
git add apps/evals/shared/prompts/observation_quality_judge_v3.txt
git commit -m "feat: judge prompt v3 with skill-appropriateness criterion"
```

---

### Task 3: Pipeline Client Extension (piece_identified capture)

**Files:**
- Modify: `apps/evals/shared/pipeline_client.py`

- [ ] **Step 1: Add `piece_identification` field to `SessionResult`**

In `apps/evals/shared/pipeline_client.py`, add to the `SessionResult` dataclass (after line 42):

```python
@dataclass
class PieceIdentification:
    """Result of automatic piece identification."""
    piece_id: str
    confidence: float
    method: str
    notes_consumed: int = 0


@dataclass
class SessionResult:
    """Result of running a full recording through the pipeline."""
    session_id: str
    recording_id: str
    observations: list[PipelineObservation]
    chunk_responses: list[dict]
    errors: list[str]
    duration_ms: int
    piece_identification: PieceIdentification | None = None
```

- [ ] **Step 2: Capture `piece_identified` messages in the WebSocket receive loop**

In the WebSocket message handling loop (around line 170), add a handler for `piece_identified`:

```python
# Add a local variable before the WebSocket loop:
piece_id_result: PieceIdentification | None = None

# In the while True loop (after the observation check ~line 173):
elif msg_type == "piece_identified":
    piece_id_result = PieceIdentification(
        piece_id=response.get("pieceId", ""),
        confidence=response.get("confidence", 0.0),
        method=response.get("method", ""),
        notes_consumed=response.get("notesConsumed", 0),
    )
```

Also capture `piece_identified` in the trailing observation loop (around line 186):

```python
elif response.get("type") == "piece_identified":
    piece_id_result = PieceIdentification(
        piece_id=response.get("pieceId", ""),
        confidence=response.get("confidence", 0.0),
        method=response.get("method", ""),
        notes_consumed=response.get("notesConsumed", 0),
    )
```

Pass `piece_identification=piece_id_result` to the final `SessionResult` constructor (line 195).

- [ ] **Step 3: Commit**

```bash
git add apps/evals/shared/pipeline_client.py
git commit -m "feat: capture piece_identified WebSocket messages in pipeline client"
```

---

### Task 4: Inference Cache HTTP Client (eval_runner --auto-t5)

**Files:**
- Modify: `apps/evals/inference/eval_runner.py`

- [ ] **Step 1: Add HTTP inference functions and --auto-t5 argument parsing**

Add to `eval_runner.py` at the bottom (before `if __name__`), keeping all existing code intact:

```python
# --- HTTP client mode for --auto-t5 ---

import httpx
from tqdm import tqdm


def health_check_servers(muq_url: str, amt_url: str) -> None:
    """Verify both local inference servers are running."""
    for name, url in [("MuQ", muq_url), ("AMT", amt_url)]:
        try:
            resp = httpx.get(f"{url}/health", timeout=5.0)
            if resp.status_code != 200:
                raise RuntimeError(f"{name} server at {url} returned {resp.status_code}")
        except httpx.ConnectError:
            raise RuntimeError(
                f"{name} server not running at {url}. Start it with: just {'muq' if name == 'MuQ' else 'amt'}"
            )


def run_http_inference(
    audio_path: Path,
    muq_url: str,
    amt_url: str,
    chunk_seconds: float = 15.0,
) -> dict:
    """Run inference on an audio file via HTTP to local MuQ + AMT servers.

    Returns cache-format dict with chunks containing predictions + MIDI.
    """
    audio_data = chunk_audio_file(str(audio_path), chunk_seconds=chunk_seconds)

    chunks = []
    for i, chunk_audio in enumerate(audio_data):
        # MuQ quality scoring
        muq_resp = httpx.post(
            muq_url,
            content=chunk_audio.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=60.0,
        )
        muq_resp.raise_for_status()
        muq_result = muq_resp.json()

        # AMT transcription
        amt_resp = httpx.post(
            amt_url,
            content=chunk_audio.tobytes(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=60.0,
        )
        amt_resp.raise_for_status()
        amt_result = amt_resp.json()

        chunks.append({
            "chunk_index": i,
            "predictions": muq_result.get("predictions", {}),
            "midi_notes": amt_result.get("notes", []),
            "pedal_events": amt_result.get("pedal_events", []),
            "audio_duration_seconds": chunk_seconds,
        })

    return {"chunks": chunks, "total_chunks": len(chunks)}


def run_auto_t5(
    cache_dir: str,
    muq_url: str = "http://localhost:8000",
    amt_url: str = "http://localhost:8001",
) -> None:
    """Scan T5 manifests, generate inference cache for uncached recordings."""
    from datetime import datetime, timezone

    T5_PIECES = [
        "bach_prelude_c_wtc1",
        "bach_invention_1",
        "fur_elise",
        "nocturne_op9no2",
    ]
    MANIFEST_BASE = MODEL_DATA / "evals" / "skill_eval"

    health_check_servers(muq_url, amt_url)
    print("Both servers healthy.")

    fingerprint = "auto-t5_http"
    cache_path = Path(cache_dir) / fingerprint
    cache_path.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in cache_path.glob("*.json")}
    total_cached = 0
    total_skipped = 0

    for piece_id in T5_PIECES:
        manifest_path = MANIFEST_BASE / piece_id / "manifest.yaml"
        if not manifest_path.exists():
            print(f"  {piece_id}: no manifest.yaml, skipping")
            continue

        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)

        recordings = [
            r for r in manifest.get("recordings", [])
            if r.get("downloaded", False) and r["video_id"] not in existing
        ]
        if not recordings:
            print(f"  {piece_id}: all {len(manifest.get('recordings', []))} recordings cached")
            continue

        print(f"\n  {piece_id}: {len(recordings)} uncached recordings")
        audio_dir = MANIFEST_BASE / piece_id / "audio"

        for rec in tqdm(recordings, desc=f"  {piece_id}"):
            video_id = rec["video_id"]
            audio_path = audio_dir / f"{video_id}.wav"
            if not audio_path.exists():
                total_skipped += 1
                continue

            try:
                result = run_http_inference(audio_path, muq_url, amt_url)
                result["recording_id"] = video_id
                result["model_fingerprint"] = fingerprint
                git_sha, _ = get_git_sha()
                result["git_sha"] = git_sha
                result["cached_at"] = datetime.now(timezone.utc).isoformat()

                out_path = cache_path / f"{video_id}.json"
                out_path.write_text(json.dumps(result, indent=2) + "\n")
                total_cached += 1
            except Exception as e:
                print(f"    {video_id}: inference failed: {e}")
                total_skipped += 1

    print(f"\nDone. Cached: {total_cached}, Skipped: {total_skipped}")
```

- [ ] **Step 2: Add --auto-t5 to argument parser**

Add `import yaml` to the imports at top. Then extend the argument parser in the existing `if __name__ == "__main__":` block:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval inference runner")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--audio-dir", default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--auto-t5", action="store_true",
                        help="Scan T5 manifests and generate cache via local HTTP servers")
    parser.add_argument("--muq-url", default="http://localhost:8000")
    parser.add_argument("--amt-url", default="http://localhost:8001")
    args = parser.parse_args()

    if args.auto_t5:
        run_auto_t5(args.cache_dir, args.muq_url, args.amt_url)
    else:
        run(args.checkpoint_dir, args.audio_dir, args.cache_dir)
```

- [ ] **Step 3: Commit**

```bash
git add apps/evals/inference/eval_runner.py
git commit -m "feat: add --auto-t5 HTTP client mode to eval_runner"
```

**IMPORTANT -- before implementing Step 1:** Read `apps/inference/muq_local_server.py` and `apps/inference/amt_local_server.py` to check their endpoint signatures. The `run_http_inference` function above assumes raw audio bytes via POST body, but the servers may expect WAV file upload via multipart form or a different content type. Adjust the HTTP client code to match the actual server endpoints.

---

### Task 5: Extend eval_practice.py (--scenarios, checkpointing, metrics)

**Files:**
- Modify: `apps/evals/pipeline/practice_eval/eval_practice.py`

- [ ] **Step 1: Add --scenarios argument and piece auto-detection**

Replace the argument parser section (lines 76-81) to support `--scenarios`:

```python
def main():
    parser = argparse.ArgumentParser(description="Run practice recording eval")
    parser.add_argument("--wrangler-url", default="http://localhost:8787")
    parser.add_argument("--piece", default=None,
                        help="Run a specific piece (e.g., fur_elise)")
    parser.add_argument("--scenarios", default=None,
                        help="Scenario prefix to match (e.g., 't5' matches t5_*.yaml)")
    parser.add_argument("--checkpoint-file", default=None,
                        help="Path to checkpoint JSON for resume support")
    args = parser.parse_args()

    # Determine which pieces to run
    if args.scenarios:
        # Find all scenario files matching the prefix
        pattern = f"{args.scenarios}_*.yaml"
        scenario_files = sorted(SCENARIOS_DIR.glob(pattern))
        if not scenario_files:
            raise FileNotFoundError(f"No scenarios matching {pattern} in {SCENARIOS_DIR}")
        pieces = [f.stem.removeprefix(f"{args.scenarios}_") for f in scenario_files]
        # Use scenario prefix in filenames to find correct YAML
        scenario_prefix = args.scenarios + "_"
    elif args.piece:
        pieces = [args.piece]
        scenario_prefix = ""
    else:
        pieces = ["fur_elise", "nocturne_op9no2"]
        scenario_prefix = ""
```

- [ ] **Step 2: Update load_scenarios call to use prefix**

Update the piece loop to use the correct scenario file name:

```python
    for piece_id in pieces:
        scenario_file = f"{scenario_prefix}{piece_id}" if scenario_prefix else piece_id
        scenarios = load_scenarios(scenario_file)
```

- [ ] **Step 3: Add STOP and piece ID metric extraction to the observation loop**

Add these tracking variables alongside the existing ones (after line 89):

```python
    # STOP metrics
    stop_probabilities: list[tuple[float, int]] = []  # (probability, skill_level)

    # Piece ID metrics
    piece_id_results: list[dict] = []  # {expected, actual, confidence, notes, correct}
```

After `run_recording()` returns (around line 115), add piece ID tracking:

```python
            # Track piece identification (only for T5 scenarios without piece_query)
            if not scenario.get("piece_query") and result.piece_identification:
                expected_piece = piece_id
                actual_piece = result.piece_identification.piece_id
                piece_id_results.append({
                    "expected": expected_piece,
                    "actual": actual_piece,
                    "confidence": result.piece_identification.confidence,
                    "notes_consumed": result.piece_identification.notes_consumed,
                    "correct": expected_piece == actual_piece,
                })
```

Inside the observation loop (after extracting eval_ctx), add STOP tracking:

```python
                # Track STOP probability
                teaching_moment = eval_ctx.get("teaching_moment", {})
                stop_prob = teaching_moment.get("stop_probability")
                if stop_prob is not None:
                    stop_probabilities.append((stop_prob, skill_level))
```

- [ ] **Step 4: Use judge v3 prompt when skill_level is available**

Replace the JUDGE_PROMPT constant and update judge call:

```python
JUDGE_PROMPT_V2 = "observation_quality_judge_v2.txt"
JUDGE_PROMPT_V3 = "observation_quality_judge_v3.txt"
```

In the observation loop, select the prompt based on skill_level:

```python
                prompt = JUDGE_PROMPT_V3 if skill_level > 0 else JUDGE_PROMPT_V2
                context["skill_level"] = str(skill_level) if skill_level > 0 else "unknown"
                judge_result = judge_observation(obs.text, context, prompt_file=prompt)
```

- [ ] **Step 5: Add per-recording checkpointing**

Add checkpoint load/save logic. Before the main piece loop:

```python
    # Checkpointing
    checkpoint_path = Path(args.checkpoint_file) if args.checkpoint_file else REPORTS_DIR / ".eval_checkpoint.json"
    completed_ids: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        completed_ids = set(checkpoint_data.get("completed", []))
        all_observations = checkpoint_data.get("observations", [])
        # Restore counters from checkpoint
        total_recordings = checkpoint_data.get("total_recordings", 0)
        recordings_with_obs = checkpoint_data.get("recordings_with_obs", 0)
        print(f"Resuming from checkpoint: {len(completed_ids)} recordings already done")
```

Inside the recording loop, skip completed and save after each:

```python
            if video_id in completed_ids:
                print("checkpointed, skipping")
                total_recordings += 1
                continue

            # ... (existing run_recording + judge code) ...

            # Save checkpoint after each recording
            completed_ids.add(video_id)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "completed": list(completed_ids),
                    "observations": all_observations,
                    "total_recordings": total_recordings,
                    "recordings_with_obs": recordings_with_obs,
                }, f)
```

- [ ] **Step 6: Add STOP and piece ID metrics to the report**

After the existing metric aggregation, add:

```python
    # STOP metrics
    if stop_probabilities:
        from scipy.stats import spearmanr
        probs = [p for p, _ in stop_probabilities]
        levels = [l for _, l in stop_probabilities]
        rho, p_value = spearmanr(probs, levels)
        report.metadata["stop_probability_skill_correlation"] = {
            "spearman_rho": round(rho, 4),
            "p_value": round(p_value, 4),
            "n": len(stop_probabilities),
        }
        # Trigger rate by bucket
        from collections import defaultdict
        bucket_triggers = defaultdict(list)
        bucket_probs = defaultdict(list)
        for prob, level in stop_probabilities:
            bucket_triggers[level].append(prob >= 0.5)
            bucket_probs[level].append(prob)
        report.metadata["stop_trigger_rate_by_bucket"] = {
            str(k): {"rate": round(sum(v)/len(v), 3), "n": len(v)}
            for k, v in sorted(bucket_triggers.items())
        }
        # Raw STOP probabilities by bucket for Cohen's d in analyze_e2e.py
        report.metadata["stop_probabilities_by_bucket"] = {
            str(k): [round(p, 4) for p in v]
            for k, v in sorted(bucket_probs.items())
        }

    # Piece ID metrics
    if piece_id_results:
        correct = sum(1 for r in piece_id_results if r["correct"])
        report.metadata["piece_id"] = {
            "top1_accuracy": round(correct / len(piece_id_results), 3),
            "total": len(piece_id_results),
            "correct": correct,
            "mean_notes_to_identify": round(
                sum(r["notes_consumed"] for r in piece_id_results) / len(piece_id_results)
            ),
            "false_positives": sum(
                1 for r in piece_id_results
                if not r["correct"] and r["confidence"] > 0.8
            ),
        }

    # Model fingerprint for auditability (from cache loading)
    if INFERENCE_CACHE_BASE.exists():
        cache_subdirs = sorted(d.name for d in INFERENCE_CACHE_BASE.iterdir() if d.is_dir())
        report.metadata["inference_cache_fingerprints"] = cache_subdirs

    # Note: piece_id_top3_accuracy is out of scope -- the piece_identified
    # WebSocket message only includes the top-1 result. Adding top-3 would
    # require API changes to include candidate rankings.
```

- [ ] **Step 7: Clean up checkpoint on successful completion**

After saving the final report:

```python
    # Remove checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("  Checkpoint cleared (run completed successfully)")
```

- [ ] **Step 8: Commit**

```bash
git add apps/evals/pipeline/practice_eval/eval_practice.py
git commit -m "feat: extend eval_practice with T5 scenarios, checkpointing, STOP/piece-ID metrics"
```

---

### Task 6: Analysis Script (analyze_e2e.py)

**Files:**
- Create: `apps/evals/pipeline/practice_eval/analyze_e2e.py`
- Test: `apps/evals/tests/test_analyze_e2e.py`

- [ ] **Step 1: Write the failing tests**

```python
# apps/evals/tests/test_analyze_e2e.py
"""Tests for E2E analysis computations."""
import math


def test_cohens_d_identical_groups():
    """Cohen's d between identical groups is 0."""
    from pipeline.practice_eval.analyze_e2e import cohens_d
    assert cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.0


def test_cohens_d_distinct_groups():
    """Cohen's d between well-separated groups is large."""
    from pipeline.practice_eval.analyze_e2e import cohens_d
    d = cohens_d([0.1, 0.15, 0.2], [0.8, 0.85, 0.9])
    assert d > 2.0  # Very large effect


def test_bootstrap_ci_contains_mean():
    """Bootstrap CI should contain the sample mean."""
    from pipeline.practice_eval.analyze_e2e import bootstrap_ci
    data = [0.5, 0.6, 0.7, 0.8, 0.9]
    low, high = bootstrap_ci(data, n_bootstrap=1000, seed=42)
    mean = sum(data) / len(data)
    assert low <= mean <= high


def test_bootstrap_ci_small_sample_warning():
    """Bootstrap with N < 5 returns None (unreliable)."""
    from pipeline.practice_eval.analyze_e2e import bootstrap_ci
    result = bootstrap_ci([0.5, 0.6], n_bootstrap=100)
    assert result is None


def test_confusion_matrix_perfect():
    """Perfect identification gives diagonal-only matrix."""
    from pipeline.practice_eval.analyze_e2e import build_confusion_matrix
    results = [
        {"expected": "bach", "actual": "bach", "correct": True},
        {"expected": "fur_elise", "actual": "fur_elise", "correct": True},
    ]
    matrix = build_confusion_matrix(results)
    assert matrix["bach"]["bach"] == 1
    assert matrix["fur_elise"]["fur_elise"] == 1
    assert matrix.get("bach", {}).get("fur_elise", 0) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/evals && uv run python -m pytest tests/test_analyze_e2e.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write the analysis script**

```python
# apps/evals/pipeline/practice_eval/analyze_e2e.py
"""Cross-cutting analysis of E2E pipeline eval results.

Reads practice_eval.json and practice_eval_observations.json,
produces STOP generalization, observation quality, and piece ID reports.

No LLM calls -- pure computation on cached results.

Usage:
    cd apps/evals/
    uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json
    uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json --stop-only
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = math.sqrt(((len(group1) - 1) * s1**2 + (len(group2) - 1) * s2**2)
                        / (len(group1) + len(group2) - 2))
    if pooled == 0:
        return 0.0
    return float((m1 - m2) / pooled)


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float] | None:
    """Compute bootstrap confidence interval. Returns None if N < 5."""
    if len(data) < 5:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1 - confidence) / 2
    low = float(np.quantile(means, alpha))
    high = float(np.quantile(means, 1 - alpha))
    return (round(low, 4), round(high, 4))


def build_confusion_matrix(results: list[dict]) -> dict[str, dict[str, int]]:
    """Build a confusion matrix from piece ID results."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r["expected"]][r.get("actual", "unidentified")] += 1
    return {k: dict(v) for k, v in matrix.items()}


def print_stop_report(report: dict, observations: list[dict]) -> None:
    """Print STOP generalization analysis."""
    print("\n" + "=" * 60)
    print("STOP GENERALIZATION REPORT")
    print("=" * 60)

    meta = report.get("metadata", {})
    stop_corr = meta.get("stop_probability_skill_correlation", {})
    if stop_corr:
        rho = stop_corr.get("spearman_rho", "N/A")
        p = stop_corr.get("p_value", "N/A")
        n = stop_corr.get("n", 0)
        print(f"\nSpearman rho (STOP prob vs skill): {rho} (p={p}, n={n})")
        if isinstance(rho, float):
            if rho < -0.3:
                print("  -> Good: higher-skill students get lower STOP probability")
            elif rho > 0.1:
                print("  -> WARNING: STOP triggers MORE on skilled students (inverted)")
            else:
                print("  -> Weak/no correlation -- STOP may not generalize to this data")

    trigger_rates = meta.get("stop_trigger_rate_by_bucket", {})
    if trigger_rates:
        print(f"\n{'Bucket':<10} {'Trigger Rate':>15} {'N':>8} {'95% CI':>20}")
        print("-" * 55)
        for bucket in sorted(trigger_rates.keys(), key=lambda x: int(x)):
            info = trigger_rates[bucket]
            rate = info["rate"]
            n = info["n"]
            ci = bootstrap_ci([1.0 if rate > 0.5 else 0.0] * n)
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "N<5"
            flag = " *" if n < 5 else ""
            print(f"  {bucket:<8} {rate:>13.3f} {n:>8}{flag} {ci_str:>20}")

    # Cohen's d between adjacent buckets (from raw STOP probabilities)
    bucket_probs = meta.get("stop_probabilities_by_bucket", {})
    if bucket_probs:
        buckets_sorted = sorted(bucket_probs.keys(), key=int)
        print(f"\nCohen's d (STOP probability separation between adjacent buckets):")
        for i in range(len(buckets_sorted) - 1):
            b1, b2 = buckets_sorted[i], buckets_sorted[i + 1]
            d = cohens_d(bucket_probs[b1], bucket_probs[b2])
            effect = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            print(f"  Bucket {b1} vs {b2}: d={d:.3f} ({effect} effect)")

    print()


def print_observation_report(report: dict, observations: list[dict]) -> None:
    """Print observation quality dashboard by skill bucket."""
    print("\n" + "=" * 60)
    print("OBSERVATION QUALITY BY SKILL BUCKET")
    print("=" * 60)

    # Group observations by skill level
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for obs in observations:
        sl = obs.get("skill_level", 0)
        if sl > 0:
            by_bucket[sl].append(obs)

    # Collect all criteria
    all_criteria = set()
    for obs in observations:
        all_criteria.update(obs.get("judge_scores", {}).keys())
    criteria = sorted(all_criteria)

    if not criteria:
        print("\nNo judge scores found in observations.")
        return

    # Header
    header = f"{'Criterion':<25}"
    for b in sorted(by_bucket.keys()):
        header += f" {'B' + str(b) + f' (n={len(by_bucket[b])})':>15}"
    print(f"\n{header}")
    print("-" * len(header))

    # Per-criterion, per-bucket pass rate
    for criterion in criteria:
        row = f"{criterion:<25}"
        for b in sorted(by_bucket.keys()):
            scores = [
                obs["judge_scores"].get(criterion)
                for obs in by_bucket[b]
                if obs["judge_scores"].get(criterion) is not None
            ]
            if scores:
                rate = sum(scores) / len(scores)
                n = len(scores)
                flag = "*" if n < 5 else " "
                row += f" {rate:>13.3f}{flag}"
            else:
                row += f" {'---':>14}"
        print(row)

    print("\n* = N < 5, treat with caution")

    # Worst criterion per bucket
    print(f"\nWorst criterion per bucket:")
    for b in sorted(by_bucket.keys()):
        worst_criterion = None
        worst_rate = 1.0
        for criterion in criteria:
            scores = [
                obs["judge_scores"].get(criterion)
                for obs in by_bucket[b]
                if obs["judge_scores"].get(criterion) is not None
            ]
            if scores:
                rate = sum(scores) / len(scores)
                if rate < worst_rate:
                    worst_rate = rate
                    worst_criterion = criterion
        if worst_criterion:
            print(f"  Bucket {b}: {worst_criterion} ({worst_rate:.3f})")


def print_piece_id_report(report: dict) -> None:
    """Print piece ID accuracy analysis."""
    print("\n" + "=" * 60)
    print("PIECE IDENTIFICATION ACCURACY")
    print("=" * 60)

    meta = report.get("metadata", {})
    pid = meta.get("piece_id", {})
    if not pid:
        print("\nNo piece ID data (all scenarios used explicit piece_query)")
        return

    print(f"\nTop-1 accuracy: {pid.get('top1_accuracy', 'N/A')}")
    print(f"Total tested: {pid.get('total', 0)}")
    print(f"Correct: {pid.get('correct', 0)}")
    print(f"Mean notes to identify: {pid.get('mean_notes_to_identify', 'N/A')}")
    print(f"False positives (high confidence wrong): {pid.get('false_positives', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze E2E pipeline eval results")
    parser.add_argument("--report", required=True, help="Path to practice_eval.json")
    parser.add_argument("--stop-only", action="store_true", help="Only print STOP analysis")
    parser.add_argument("--piece-id-only", action="store_true", help="Only print piece ID analysis")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Report not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    # Load detailed observations if available
    obs_path = report_path.parent / "practice_eval_observations.json"
    observations = []
    if obs_path.exists():
        with open(obs_path) as f:
            observations = json.load(f)

    if args.stop_only:
        print_stop_report(report, observations)
    elif args.piece_id_only:
        print_piece_id_report(report)
    else:
        print_stop_report(report, observations)
        print_observation_report(report, observations)
        print_piece_id_report(report)

    print("\n" + "=" * 60)
    print(f"Report: {report_path}")
    print(f"Observations: {len(observations)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/evals && uv run python -m pytest tests/test_analyze_e2e.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add apps/evals/pipeline/practice_eval/analyze_e2e.py \
       apps/evals/tests/test_analyze_e2e.py
git commit -m "feat: add E2E analysis script with STOP, observation, piece ID reports"
```

---

### Task 7: Justfile Additions

**Files:**
- Modify: `Justfile`

- [ ] **Step 1: Add eval commands to Justfile**

Add after the existing `deploy-api` recipe:

```just
# --- E2E Pipeline Eval ---

# Run full E2E pipeline eval (cache -> pipeline -> analyze)
eval-e2e: eval-cache eval-pipeline eval-analyze

# Generate missing inference cache for T5 corpus (requires just muq + just amt running)
eval-cache:
    cd apps/evals && uv run python -m inference.eval_runner --auto-t5

# Run pipeline eval on T5 corpus (requires just api running)
eval-pipeline:
    cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice --scenarios t5

# Analyze eval results
eval-analyze:
    cd apps/evals && uv run python -m pipeline.practice_eval.analyze_e2e --report reports/practice_eval.json

# Generate T5 scenario files from manifests
eval-scenarios:
    cd apps/evals && uv run python -m pipeline.practice_eval.generate_t5_scenarios
```

- [ ] **Step 2: Commit**

```bash
git add Justfile
git commit -m "feat: add eval-e2e, eval-cache, eval-pipeline, eval-analyze commands"
```

---

### Task 8: Integration Validation (fur_elise baseline)

This task validates the harness works by running it on the 2 existing pieces before tackling the full T5 corpus.

**Files:** None new (uses everything built in Tasks 1-7)

- [ ] **Step 1: Generate T5 scenarios for the 2 pieces with existing cache**

Run: `just eval-scenarios`
Verify: `ls apps/evals/pipeline/practice_eval/scenarios/t5_fur_elise.yaml`

- [ ] **Step 2: Run pipeline eval on fur_elise only (requires wrangler dev)**

Run: `cd apps/evals && uv run python -m pipeline.practice_eval.eval_practice --scenarios t5 --piece fur_elise`
Expected: Observations collected, judge scores computed, STOP metrics in metadata

- [ ] **Step 3: Run analysis on results**

Run: `just eval-analyze`
Expected: Terminal output with STOP report, observation quality by bucket, no crashes

- [ ] **Step 4: Compare against existing practice_eval baseline**

Manually compare the observation quality metrics from the new run against any prior `practice_eval.json` results. The v3 judge adds one criterion but the existing 5 should produce similar pass rates.

- [ ] **Step 5: Commit any fixups discovered during integration**

```bash
git add -u
git commit -m "fix: integration fixes from fur_elise baseline validation"
```

---

## Dependency Graph

```
Task 1 (scenarios) ──────────────────────┐
Task 2 (judge v3) ───────────────────────┤
Task 3 (pipeline_client piece_id) ───────┼── Task 5 (eval_practice extension) ── Task 7 (Justfile)
Task 4 (eval_runner --auto-t5) ──────────┘                                           |
                                                                                      v
Task 6 (analyze_e2e.py) ─────────────────────────────────────────────────── Task 8 (integration)
```

Tasks 1-4 are independent and can be parallelized. Task 5 depends on 1-3. Task 6 is independent. Task 7 ties them together. Task 8 validates everything.
