# Pipeline Evaluation System

Comprehensive evaluation framework for every component of the CrescendAI pipeline, from audio inference through observation delivery. Answers three questions across three phases: "Is the product good?" (Phase 1), "Which piece is weak?" (Phase 2), "Did it regress?" (Phase 3).

## Problem

The CrescendAI pipeline has strong research-grade evaluation at the model layer (A1-Max 80.8% pairwise, STOP AUC 0.845, memory recall=1.0) and working integration tests at the pipeline layer (end-to-end web observations verified). But the middle layers -- where the system reasons about what to teach and generates the actual feedback -- have no isolated evaluation. We don't know if the observations the system delivers are actually good teaching.

Existing coverage:

| Tier | Components | Status |
|------|-----------|--------|
| Research-grade | A1-Max scoring, STOP classifier, AMT validation, Layer 1 gates | Solid |
| Dedicated eval | Memory system (38 scenarios + LoCoMo) | Solid |
| Integration tested | Practice session DO, score follower, teaching moments, full pipeline | Working |
| No isolated eval | Subagent reasoning, teacher observation quality, teaching moment ranking, bar-aligned analysis, STOP threshold sensitivity | Gap |

## Approach: Dual-Track

Two tracks in parallel:
1. **End-to-end observation quality** -- run YouTube recordings through the full pipeline, score observations with an LLM judge
2. **Component isolation** -- targeted evals for the highest-uncertainty components

Phased rollout: Phase 1 establishes the baseline, Phase 2 adds diagnostics, Phase 3 adds regression detection.

## Design Decisions (from CEO review)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Rust boundary bridge | `wrangler dev` + eval-mode endpoint | Evaluates actual production code; eval endpoint accepts pre-computed inference, bypassing HF call |
| Inference cache | Versioned with model fingerprint | Prevents silent staleness when model changes |
| Checkpoint weights | Local path (`model/data/checkpoints/model_improvement/A1/fold_{0-3}/`) | Already on disk from training |
| LLM judge format | Binary rubrics (YES/NO) with evidence quotes | Forces grounded decisions; avoids Likert clustering |
| Calibration | 20 human-scored observations, Cohen's kappa > 0.6 per criterion | Validates judge before trusting automation |
| Reproducibility | Git SHA + dirty flag in report metadata | Makes irreproducible runs visible without blocking dev iteration |
| Debuggability | Full pipeline traces per observation | Enables root-cause analysis when quality drops |
| Eval vs test boundary | Evals = quality judgments on non-deterministic outputs. Tests = deterministic correctness. | Keeps framework focused |

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        EVAL SYSTEM                                  │
  │                                                                     │
  │  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
  │  │ YouTube Audio │    │ model/data/checkpoints/A1/fold_{0-3}   │  │
  │  │ (50 files)    │    │ (local weights)                         │  │
  │  └──────┬───────┘    └──────────────┬──────────────────────────┘  │
  │         │                            │                             │
  │         ▼                            ▼                             │
  │  ┌──────────────────────────────────────┐                         │
  │  │ LOCAL INFERENCE RUNNER               │                         │
  │  │ apps/inference/eval_runner.py        │  Python, CPU/MPS        │
  │  │ --checkpoint-dir  --device auto      │                         │
  │  └──────────────┬───────────────────────┘                         │
  │                 │                                                   │
  │                 ▼                                                   │
  │  ┌──────────────────────────────────────┐                         │
  │  │ INFERENCE CACHE (versioned)          │                         │
  │  │ data/eval/inference_cache/           │                         │
  │  │   a1max_v1.0_amt_v1.0/              │                         │
  │  │     yt_001.json ... yt_050.json      │                         │
  │  │ + model fingerprint + git SHA        │                         │
  │  └──────────────┬───────────────────────┘                         │
  │                 │                                                   │
  │    ┌────────────┼────────────────┐                                │
  │    │            │                │                                 │
  │    ▼            ▼                ▼                                 │
  │  ┌────────┐  ┌────────────┐  ┌─────────────────────────────┐     │
  │  │Subagent│  │STOP Thresh │  │ WRANGLER DEV (localhost:8787)│     │
  │  │Reasoning│ │Sensitivity │  │ Full Rust pipeline:          │     │
  │  │Eval    │  │Eval        │  │  STOP → TeachMoment →        │     │
  │  │(1b)    │  │(2c)        │  │  ScoreFollow → Analysis →    │     │
  │  │        │  │            │  │  Subagent → Teacher           │     │
  │  │Groq    │  │Python      │  └──────────────┬───────────────┘     │
  │  │direct  │  │(reads      │                  │                     │
  │  └───┬────┘  │ cache)     │  ┌───────────────▼──────────────┐     │
  │      │       └─────┬──────┘  │ PIPELINE TRACES              │     │
  │      │             │         │ data/eval/traces/             │     │
  │      │             │         └───────────────┬──────────────┘     │
  │      ▼             ▼                         ▼                     │
  │  ┌──────────────────────────────────────────────────────────┐     │
  │  │ LLM JUDGE (shared/judge.py) → Claude API                │     │
  │  │ Binary rubrics + evidence quotes                         │     │
  │  │ Calibrated against 20 human-scored observations          │     │
  │  └──────────────────────────┬───────────────────────────────┘     │
  │                             ▼                                      │
  │  ┌──────────────────────────────────────────────────────────┐     │
  │  │ REPORT JSON + terminal summary table                     │     │
  │  │ metadata: {git_sha, dirty, model_version, timestamp}     │     │
  │  └──────────────────────────────────────────────────────────┘     │
  └─────────────────────────────────────────────────────────────────────┘
```

## Rust Bridge: Eval-Mode Endpoint

In production, `handle_chunk_ready` in `session.rs` fetches audio from R2 and calls the HF inference endpoint. For evals, we need to bypass those two steps and inject pre-computed inference results directly into the downstream pipeline (STOP -> teaching moments -> score following -> analysis -> subagent -> teacher).

**Solution: `POST /api/practice/eval-chunk`** -- A new endpoint on the DO, gated behind a `#[cfg(feature = "eval")]` flag (or an env-var check), that accepts pre-computed inference results as JSON instead of an R2 key:

```json
{
  "chunk_index": 2,
  "predictions": {"dynamics": 0.42, "timing": 0.71, ...},
  "midi_notes": [...],
  "pedal_events": [...]
}
```

This endpoint calls the same downstream code path as `handle_chunk_ready` starting from step 3 (extract scores), skipping steps 1-2 (R2 fetch + HF call). The downstream pipeline is identical to production.

**Why not mock the HF endpoint?** Mocking requires intercepting HTTP inside the WASM worker, which is fragile. The eval endpoint is a clean seam: it's 10-15 lines of Rust that call the same functions, and it only exists in dev builds.

**Wrangler dev setup for evals:**
1. Start `wrangler dev` (serves the worker locally with Miniflare D1)
2. Seed D1 with score catalog + test student records
3. Python eval harness creates a practice session via `POST /api/practice/start`
4. For each cached chunk, POST to `/api/practice/eval-chunk` with inference results
5. Collect observations from the WebSocket connection

**LLM keys in wrangler dev:** The worker needs `GROQ_API_KEY` and `ANTHROPIC_API_KEY` to call the subagent and teacher LLMs. These go in `.dev.vars` (wrangler's local secrets file, already gitignored). The preflight check verifies LLM key availability by making a test call through the worker, not just checking Python env vars.

## Inference Cache Schema

Each YouTube recording is chunked into 15-second segments (matching production) before inference. The cache stores one JSON file per recording containing an array of chunk results:

```json
{
  "recording_id": "yt_017",
  "model_fingerprint": "a1max_v1.0_amt_v1.0",
  "git_sha": "a1b2c3d",
  "chunks": [
    {
      "chunk_index": 0,
      "predictions": {"dynamics": 0.42, "timing": 0.71, "pedaling": 0.55, "articulation": 0.63, "phrasing": 0.48, "interpretation": 0.51},
      "midi_notes": [{"pitch": 60, "onset": 0.1, "offset": 0.5, "velocity": 80}, ...],
      "pedal_events": [{"time": 0.3, "value": 127}, ...],
      "transcription_info": {"note_count": 45, "pitch_range": [48, 84], "pedal_event_count": 3},
      "audio_duration_seconds": 15.0,
      "processing_time_ms": 12400
    },
    ...
  ],
  "total_duration_seconds": 180.5,
  "total_chunks": 12,
  "cached_at": "2026-03-15T14:30:00Z"
}
```

The eval runner chunks audio into 15s segments before calling the handler, matching the production `MediaRecorder.start(15000)` behavior. This means each recording produces multiple cache entries (one per chunk), grouped in a single file.

## Data Flow

The most expensive operation is model inference (~10-30s per chunk on CPU). All downstream evals consume cached inference output rather than re-running the model.

```
  YouTube audio files (50)
          |
          v
  [Local Inference Runner]  <-- runs once, ~25-40 min CPU
          |
          v
  inference_cache/            <-- versioned JSON per recording
    a1max_v1.0_amt_v1.0/
      yt_001.json ... yt_050.json
          |
          |---> STOP eval (reads predictions, applies threshold sweep)
          |---> Teaching moment eval (reads multi-chunk sequences)
          |---> Analysis eval (reads midi_notes + pedal_events + score)
          |---> Subagent eval (constructs context from cached data)
          |---> Observation quality eval (sends to wrangler dev, full pipeline)
          └---> Score follower eval (reads midi_notes vs score MIDI)
```

### Shadow Paths

Every data flow node has defined behavior for nil, empty, and error inputs:

| Node | Nil/Missing | Empty | Error | Slow |
|------|------------|-------|-------|------|
| Audio file | Skip + warn | Skip + warn (0-byte) | AudioProcessingError → skip + warn | Log time, continue |
| Inference cache | Trigger re-inference for that file | N/A | StaleCacheError → block run | N/A |
| Wrangler dev | Preflight catches → print instructions | N/A | Retry 1x, then skip + log | Log, continue |
| Pipeline observation | Record as "no_observation" | Record as "no_observation" | Record partial data | Log |
| LLM judge | Backoff + retry | Null score + warning | Null score + warning | Continue |
| Report | Partial suite failure → continue others | Report N, mean over non-null | Validation error → log | N/A |

## Components

### 1. Local Inference Runner (`apps/inference/eval_runner.py`)

Loads `EndpointHandler` with auto-detected device (CUDA > MPS > CPU). Accepts a directory of audio files, runs batch inference, writes versioned JSON cache.

Changes to inference code:
- Parameterize `device` in `EndpointHandler.__init__` (currently hardcoded `"cuda"` on line 66)
- Auto-detect: `"cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`
- Same change for `TranscriptionModel.__init__` (also hardcoded `"cuda"` on line 70)
- **ModelCache singleton guard:** `ModelCache.initialize()` has an early-return guard (`if self.muq_model is not None: return`). The eval runner must pass `device` before the first `initialize()` call, or bypass the singleton by constructing a fresh `ModelCache` instance directly. The runner will set `device` via environment variable (`CRESCEND_DEVICE`) read before any model loading.

The runner:
- Accepts `--checkpoint-dir` (default: `model/data/checkpoints/model_improvement/A1/`)
- Accepts `--audio-dir` (default: `data/eval/youtube_amt/`)
- Accepts `--cache-dir` (default: `data/eval/inference_cache/`)
- Computes model fingerprint from `model_info` in inference output
- Creates versioned cache subdirectory: `{cache_dir}/{model_name}_{model_version}/`
- Skips already-cached files (idempotent)
- Outputs progress: `[23/50] yt_023.json (14.2s, 3 chunks)`

### 2. Shared Infrastructure (`apps/api/evals/shared/`)

**`judge.py`** -- LLM-as-judge client:
- Wraps Anthropic API with retry (exponential backoff on 429), structured output parsing
- Loads versioned judge prompts from `shared/prompts/`
- Returns structured scores per criterion with evidence quotes
- Handles parse failures and refusals as null scores with warnings

**`reporting.py`** -- JSON report envelope:
```json
{
  "eval_name": "observation_quality",
  "eval_version": "1.0",
  "timestamp": "2026-03-15T14:30:00Z",
  "dataset": "youtube_amt_50",
  "git_sha": "a1b2c3d",
  "git_dirty": false,
  "model_version": "a1max_v1.0_amt_v1.0",
  "metrics": {
    "musical_accuracy": {"mean": 0.82, "std": 0.11, "n": 50, "pass": true},
    ...
  },
  "pass_criteria": {"musical_accuracy_mean_gte": 0.70, "all_passed": true},
  "worst_cases": [...],
  "cost": {"judge_calls": 750, "estimated_usd": 2.50}
}
```

**`inference_cache.py`** -- Cache read/write with version checking:
- Writes JSON per recording with model fingerprint
- On read, validates fingerprint matches expected version
- Raises `StaleCacheError` on mismatch with clear message

**`traces.py`** -- Pipeline trace writer:
- Writes one JSON file per observation to `data/eval/traces/`
- Captures: inference output, STOP score, teaching moment selection, analysis facts, subagent output, teacher observation, judge scores + evidence
- Report links to trace files for worst-case observations

### 3. Preflight Checks (`run_all.py`)

Before running any eval suite, verify:
- Checkpoint directory exists and contains fold weights
- Audio manifest/directory is present with expected file count
- `wrangler dev` is responding on localhost:8787 (HTTP GET /health)
- D1 is seeded: query a known piece from the score catalog via the worker
- `ANTHROPIC_API_KEY` environment variable is set (for LLM judge)
- `GROQ_API_KEY` environment variable is set (for subagent reasoning eval)
- Worker has LLM access: verify `.dev.vars` contains `GROQ_API_KEY` and `ANTHROPIC_API_KEY` (these are the worker's keys for the subagent and teacher LLM calls, separate from the Python env vars used by the judge)
- Git SHA + dirty flag captured for metadata

If any check fails, print a clear error message with the fix and exit.

### 4. Observation Quality Eval (`apps/api/evals/observation_quality/`) -- Phase 1a

**Input:** YouTube recordings run through the full pipeline via wrangler dev.

For each recording:
1. Read cached inference output (predictions + midi_notes + pedal_events per chunk)
2. Create a practice session via `POST localhost:8787/api/practice/start`
3. For each chunk, POST to `localhost:8787/api/practice/eval-chunk` with pre-computed inference results (bypasses R2 fetch + HF call, runs STOP → teaching moments → score following → analysis → subagent → teacher)
4. Collect observations from the WebSocket connection
5. Write pipeline trace (full state at each stage)
6. Send observation + context to LLM judge

**LLM Judge Rubric (5 binary criteria with evidence):**

| Criterion | Judge Prompt | Pass = |
|-----------|-------------|--------|
| Musical accuracy | "Does the observation describe something actually true about this performance? Quote the specific claim and state whether the audio/MIDI evidence supports it." | YES |
| Specificity | "Does the observation reference a concrete musical moment (bar number, passage, phrase boundary)? Quote the reference. If it says only 'your dynamics' without locating where, answer NO." | YES |
| Actionability | "Could the student change something specific in their next attempt? Quote the suggested action. Generic advice like 'practice more' = NO." | YES |
| Tone | "Is the tone warm and encouraging without being condescending or vague? Quote any problematic phrasing. If none, YES." | YES |
| Dimension appropriateness | "Given the score context (6 dimension scores, baseline, recent observations), was this the most valuable dimension to teach? If a different dimension had a larger deviation and wasn't recently covered, answer NO and name it." | YES |

**Context provided to judge:** The judge sees the same context the teacher LLM saw -- dimension scores, student baselines, recent observations, analysis facts, AMT summary. This catches "confidently wrong" observations that sound plausible but contradict the score data.

**Output:** Per-observation scores, aggregate pass rate per criterion, worst-case examples with trace links.

### 5. Subagent Reasoning Eval (`apps/api/evals/subagent_reasoning/`) -- Phase 1b

**Input:** ~35 hand-crafted scenarios. Each scenario is the exact JSON context the subagent receives: chunk scores, student baselines, recent observations, dimension analysis.

Scenarios cover:
- Obvious dimension selection (dynamics clearly worst → should pick dynamics)
- All dimensions improving (should pick positive recognition)
- Recent dimension repetition (same dimension flagged 3 times → should pick something else)
- Borderline STOP (should note uncertainty in reasoning)
- Mixed signals (one dimension declining, another improving → should prioritize decline)
- No baseline yet (first session → should frame as exploration, not correction)

**LLM Judge Rubric (3 binary criteria):**

| Criterion | Pass = |
|-----------|--------|
| Dimension selection | Selected dimension has the largest negative deviation not covered in last 3 observations |
| Framing match | Framing tracks trajectory: correction for decline, recognition for improvement |
| Reasoning coherence | Reasoning trace logically supports the conclusion, no non-sequiturs |

**Note:** This eval calls Groq directly (not through the Rust pipeline). It tests the LLM's reasoning, not the plumbing.

### 6. Teaching Moment Ranking Eval (`apps/api/evals/teaching_moments/`) -- Phase 2a

**Input:** 20-30 multi-chunk sessions with known score trajectories.

**Method:** For each session, the system selects a teaching moment. You (Jai) independently rank which chunk you'd teach from. Compare system ranking vs your ranking using Kendall's tau.

**Success gate:** tau >= 0.5 (better than random, tracks teacher intuition).

### 7. Analysis Accuracy Eval (`apps/api/evals/analysis_accuracy/`) -- Phase 2b

**Input:** 15-20 MAESTRO performances with obvious musical features (clear crescendos, pedal changes, tempo rubato).

**Method:** Run through local inference + analysis engine (via wrangler dev). Verify that stated facts (crescendo detected, pedal overlap, etc.) match what's actually in the performance.

**Success gate:** Fact precision >= 70% (most stated facts are true).

### 8. STOP Threshold Sensitivity Eval (`apps/api/evals/stop_sensitivity/`) -- Phase 2c

**Input:** Full YouTube set with STOP scores.

**Method:** Sweep thresholds (0.3, 0.4, 0.5, 0.6, 0.7). At each threshold, measure:
- Observations per session (how chatty is the system?)
- Dimension distribution (does one dimension dominate?)
- Observation quality (reuse Phase 1a's judge)

**Output:** No pass/fail gate. The sweep informs the optimal threshold choice.

### 9. Score Follower Robustness Eval -- Phase 3a

**Language:** Rust. Lives alongside the score follower code in `apps/api/src/practice/`, not in the Python eval directory. Runs via `cargo test --features eval` in the `apps/api/` crate.

**Integration with `run_all.py`:** The Python runner shells out to `cargo test --features eval --message-format json` in `apps/api/`, parses the JSON test output, and converts it to the shared report envelope format. This is the one eval that crosses the language boundary.

**Input:** Synthetic MIDI perturbations of ASAP scores: section skips, restarts, out-of-order practice, long sessions (>10 chunks).

**Success gates:**
- Clean input alignment: >= 95%
- Section skips: >= 70% (graceful degradation)

### 10. Observation Diversity Eval (`apps/api/evals/observation_diversity/`) -- Phase 3b

**Input:** Simulated 5-10 consecutive sessions for the same student/piece.

**Measures:**
- Dimension repetition rate: fraction of consecutive observation pairs that target the same dimension across the session sequence. Target: <= 40% over 5 sessions. Example: if 10 observations are generated and 4 consecutive pairs share a dimension, repetition rate = 4/9 = 0.44.
- Observation text similarity: mean pairwise cosine distance of sentence embeddings across all observations in the sequence. Higher = more diverse.
- Framing variety: entropy of the framing distribution (correction/recognition/encouragement/question). Higher entropy = more balanced.

### 11. Teacher Voice Consistency Eval (`apps/api/evals/teacher_voice/`) -- Phase 3c

**Input:** Same scenario run 10 times.

**Measures:** Variance in LLM judge scores across runs. Target: std <= 0.15 per criterion. Substance should be consistent even if phrasing varies.

## LLM Judge Calibration Process

1. Run Phase 1a on ~20 observations to generate observation + context pairs
2. You (Jai) score each observation on the 5 binary criteria with brief notes
3. Run the LLM judge on the same 20
4. Compute Cohen's kappa per criterion. Target: kappa > 0.6
5. If any criterion falls below 0.6, examine disagreements:
   - Judge too lenient → tighten prompt ("Only answer YES if...")
   - Judge too strict → loosen ("Answer YES even if...")
   - Genuine ambiguity → add examples to judge prompt
6. Iterate judge prompt (not rubric) until kappa > 0.6 on all criteria
7. Store final judge prompt as versioned file (`shared/prompts/observation_quality_judge_v1.txt`)
8. Lock the 20-observation calibration set as regression anchor. Re-run on calibration set before changing judge prompts.

## Directory Structure

```
apps/api/evals/
  ├── memory/                   <-- existing (untouched)
  ├── shared/
  │   ├── __init__.py
  │   ├── judge.py              <-- LLM-as-judge client (Claude API)
  │   ├── reporting.py          <-- JSON envelope + summary table
  │   ├── inference_cache.py    <-- read/write versioned cache
  │   ├── traces.py             <-- pipeline trace writer
  │   └── prompts/
  │       ├── observation_quality_judge_v1.txt
  │       └── subagent_reasoning_judge_v1.txt
  ├── observation_quality/      <-- Phase 1a
  │   ├── __init__.py
  │   └── eval_observation_quality.py
  ├── subagent_reasoning/       <-- Phase 1b
  │   ├── __init__.py
  │   ├── eval_subagent_reasoning.py
  │   └── scenarios/
  │       └── scenarios.json    <-- 35 hand-crafted scenarios
  ├── teaching_moments/         <-- Phase 2a
  ├── analysis_accuracy/        <-- Phase 2b
  ├── stop_sensitivity/         <-- Phase 2c
  ├── # score_follower lives in apps/api/src/practice/ (Rust, Phase 3a)
  ├── observation_diversity/    <-- Phase 3b
  ├── teacher_voice/            <-- Phase 3c
  ├── pyproject.toml
  └── run_all.py                <-- top-level runner

apps/inference/
  ├── eval_runner.py            <-- local inference batch runner
  └── ...existing handler code...

data/eval/                      <-- .gitignore'd, manifest for reproducibility
  ├── youtube_amt/              <-- 50 audio files (or URLs manifest)
  ├── inference_cache/          <-- versioned JSON cache
  │   └── a1max_v1.0_amt_v1.0/
  ├── traces/                   <-- per-observation pipeline traces
  └── calibration/              <-- human-scored observations
      └── human_scores.json
```

## Phasing and Success Criteria

### Phase 1: "Is the product good?"

**Delivers:**
- Local inference runner with device auto-detection
- Versioned inference cache for YouTube AMT set (50 recordings)
- Observation quality eval (5-criterion LLM judge + 20-observation calibration set)
- Subagent reasoning eval (~35 hand-crafted scenarios)
- Shared infrastructure (judge, reporting, cache, traces, preflight, run_all.py)

**Success gates:**

| Metric | Gate | Rationale |
|--------|------|-----------|
| Judge-human kappa (all 5 criteria) | >= 0.6 | Judge is trustworthy |
| Observation musical accuracy | >= 70% pass | System says true things |
| Observation specificity | >= 60% pass | Bar references work when score identified |
| Subagent dimension selection | >= 80% pass | Picks right dimension on obvious cases |
| Subagent framing match | >= 75% pass | Framing tracks trajectory |

**Done when:** `uv run python run_all.py` prints a summary table with Phase 1 evals reporting metrics. Calibration set scored, kappa documented.

### Phase 2: "Which piece is weak?"

**Delivers:**
- Teaching moment ranking eval (20-30 scenarios, your rankings as ground truth)
- Bar-aligned analysis accuracy eval (15-20 MAESTRO performances)
- STOP threshold sensitivity sweep

**Success gates:**

| Metric | Gate |
|--------|------|
| Teaching moment Kendall's tau | >= 0.5 |
| Analysis fact precision | >= 70% |
| STOP threshold sweep | Documented (no pass/fail) |

**Done when:** You can identify which component drags observation quality down.

### Phase 3: "Did it regress?"

**Delivers:**
- Score follower robustness eval (Rust, synthetic MIDI perturbations)
- Observation diversity eval (multi-session sequences)
- Teacher voice consistency eval (repeated runs)

**Success gates:**

| Metric | Gate |
|--------|------|
| Score follower (clean input) | >= 95% alignment |
| Score follower (section skips) | >= 70% |
| Dimension repetition (5 sessions) | <= 40% |
| Teacher variance (10 runs) | std <= 0.15 |

**Done when:** Pipeline changes can be validated against the full suite before shipping.

## Error & Rescue Map

| Error | Rescued? | Action | Eval sees |
|-------|----------|--------|-----------|
| Checkpoint missing | Y | Print path, exit | Clear error |
| CUDA OOM | Y | Fall back to CPU | Slower, works |
| Corrupt audio | Y | Skip + log | Warning in report |
| AMT 0 notes | Y | Flag as no_transcription | Tier 3 analysis |
| Cache version mismatch | Y | Warn + block | Must regenerate |
| Corrupt cache JSON | Y | Delete + re-inference | Warning |
| Wrangler not running | Y | Preflight catches | Clear error |
| D1 not seeded | Y | Preflight: query known piece | Clear error |
| Worker 500 | Y | Retry 1x, then skip | Warning in report |
| Empty observation | Y | Record as no_observation | Counted in metrics |
| Rate limited (Claude) | Y | Exponential backoff, 3x | Slower, works |
| API key missing | Y | Preflight catches | Blocks run |
| Judge parse failure | Y | Null score + log raw | Warning + null |
| Judge refusal | Y | Null score + log | Warning + null |
| Partial suite failure | Y | Continue others | Partial results |

## Out of Scope

| Item | Rationale |
|------|-----------|
| CI integration | Pre-launch; run manually |
| Real-user A/B testing | No users yet |
| iOS audio quality eval | Needs paired recordings |
| Synthesized facts eval | Feature not implemented |
| Exercise effectiveness | Needs weeks of student data |
| Dashboard / web UI | JSON + terminal table sufficient |
| Eval-driven prompt auto-tuning | Future capability |
| Memory eval changes | Existing eval untouched; semantic similarity fallback already planned separately |

## Performance Estimates

| Phase | Warm cache? | Estimated time |
|-------|-------------|---------------|
| Phase 1 full run | No (first time) | 45-90 min |
| Phase 1 full run | Yes | 20-35 min |
| Phase 1 judge only | Yes | 15-25 min |
| Subagent reasoning only | N/A (no inference) | 5-10 min |

Inference dominates cold runs. LLM judge calls can be parallelized with asyncio for ~2-3x speedup.
