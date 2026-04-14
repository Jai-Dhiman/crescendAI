# Eval Baseline Readiness Design

**Goal:** Land every eval-harness improvement that can be built without valid MuQ scores, so that when Model v2 training completes and the cache is refreshed, one command produces a locked Sonnet 4.6 baseline artifact with per-dim means, bootstrap CIs, provenance, and a holdout that was never touched during any prompt iteration.

**Not in scope:**
- Running the baseline itself (gated on Model v2 training finishing and cache re-inference)
- Real dual-judge calibration numbers (requires scored syntheses)
- Production trace flywheel (requires live beta sessions with valid scores)
- Phantom-criteria audit of actual example judgments (requires real judgments)
- Teacher model finetuning training
- Changes to the MuQ or AMT inference pipelines
- Deploying any judge or rubric change to prod (observation/chat paths stay on current prompts)

**Parent roadmap:** `docs/plans/2026-04-14-eval-improvements.md` (strategic P0–P3). This spec is the tactical TDD build plan for the P0 and early-P1/P2 code work.

---

## Problem

The synthesis quality eval at `apps/evals/teaching_knowledge/` is ~85% built as infrastructure but 0% validated as a quality gate. Concretely:

1. **`apps/evals/teaching_knowledge/results/` is empty.** `run_eval.py` has only been run in `--dry-run` mode. No locked baseline exists.
2. **Style injection is unwired.** `run_eval.py` lines 77–314 pass `composer` through as passive metadata but never load `data/playbook.yaml`'s `piece_style_dimension_rules` or inject them into the synthesis prompt. The previous smoke test scored 0/3 on Style-Consistent Musical Language as a direct consequence.
3. **No train/test split.** 890 cached recordings are pooled — any prompt iteration against this pool will invisibly overfit. The wiki principle "train/test split is very important — autonomous optimization overfits" (from *Better Harness: A Recipe for Harness Hill-Climbing*) has no enforcement in code.
4. **No dataset taxonomy.** Recordings carry no `composer_era`, `skill_bucket`, or `duration_bucket` tags, so targeted subset runs (the wiki's "tag evals by what they test, not where they came from") are impossible.
5. **No provenance.** Output JSONL rows lack `run_id` and `git_sha`, so two runs sitting in `results/` cannot be distinguished.
6. **No judge-family invariant.** Same-family judging (teacher-model family == judge-model family) causes phrasing-preference bias; the rule is currently uncoded and can be violated by accident in a CLI invocation.
7. **Rubric may contain phantom criteria.** The 7-dim rubric was LLM-derived by `derive_rubrics.py`. "Autonomy-Supporting Motivation" and "Scaffolded Guided Discovery" are suspect candidates for phantom criteria per the wiki principle. The judge schema also conflates process and outcome into one score, which the wiki warns against: "process and outcome rewards must be separated into independent signals."
8. **No aggregation.** Even when the baseline runs, output is raw JSONL — nothing reduces it to per-dim means with confidence intervals.
9. **No regression harness.** No code diffs two runs and flags regressions.
10. **No second judge.** The eval uses only `@cf/google/gemma-4-26b-a4b-it`. Cross-family variance cannot be measured, so judge bias is a silent failure mode.
11. **No teacher finetune A/B scaffold.** When the finetuned Qwen model lands, there is no one-command way to diff its synthesis quality against baseline Sonnet.

**Constraint:** Model v2 training is in progress. The 890 cached MuQ scores were produced by old model weights and must be treated as stale — we cannot trust their numeric outputs as baseline signal. Everything in this spec must be buildable and testable **without** valid MuQ scores, using fixture data, mocked cache entries, and synthetic judge outputs.

## Solution (from the user's perspective)

When Model v2 training finishes and the cache is re-inferred, the user runs:

```bash
cd apps/evals
uv run python -m teaching_knowledge.scripts.tag_dataset
uv run python -m teaching_knowledge.scripts.split --seed 42
uv run python -m teaching_knowledge.run_eval \
    --split train \
    --teacher-model claude-sonnet-4-6 \
    --judge-model "@cf/google/gemma-4-26b-a4b-it" \
    --out results/baseline_sonnet46_judge-gemma4_2026-04-14.jsonl
uv run python -m teaching_knowledge.scripts.aggregate \
    results/baseline_sonnet46_judge-gemma4_2026-04-14.jsonl
```

And receives:

- A JSONL file where every row carries `run_id`, `git_sha`, and dimension scores with `{process, outcome}` per dim
- An aggregate report with per-dim mean, bootstrap 95% CI, and stratified breakdowns by composer era and skill bucket
- A guarantee that the holdout split was never touched
- A hard error if the teacher and judge model families conflict (e.g., Sonnet as both)

Later, when a prompt change is proposed, the user runs `regression_check.py baseline.jsonl candidate.jsonl` and sees a per-dim delta table with non-overlap flags. When a finetuned teacher is ready, `teacher_model/eval_ab.py` produces a one-screen verdict comparing it to baseline.

## Design

### Approach: 16 vertical tracer bullets across 6 deep modules + 5 orchestrator edits

Every improvement is a vertical slice: one test → one implementation → one commit. No horizontal "write all tests then write all impls." No task exceeds a single public behavior.

### Key design decisions

**1. Style rules as a single JSON source of truth.**
`apps/evals/teaching_knowledge/data/playbook.yaml` has `piece_style_dimension_rules` embedded. We extract that section into a standalone JSON file `apps/evals/shared/data/style_rules.json`. Python reads it via `shared/style_rules.py`; TypeScript reads it via a mirror copy at `apps/api/src/lib/style-rules.json`. A pytest asserts hash equality between the two copies so drift is caught at test time rather than at production eval time. The playbook.yaml retains the section as human-readable documentation (marked with a `# SOURCE OF TRUTH: shared/data/style_rules.json` comment) but Python and TS never read from it for runtime style rules.

*Alternative considered:* Keep playbook.yaml as the source and have Python read from it directly. Rejected because TS cannot import YAML without adding a parser dependency to the Workers bundle, and keeping two parsers (Python YAML + TS YAML) is worse than two JSON copies.

**2. Style injection in user message, not system prompt.**
The style guidance for a recording depends on its composer — it varies per call. Injecting it in the user message keeps the system prompt cache-warm (Anthropic prompt caching matches on system prefix). Both `run_eval.py::build_synthesis_user_msg` and `apps/api/src/services/prompts.ts::buildSynthesisFraming` inject style guidance between `<session_data>` and `<task>`.

*Alternative considered:* Inject into the system prompt. Rejected because (a) breaks prompt caching, (b) per-call system prompts are awkward in the Anthropic SDK, (c) style guidance is data about THIS recording, which belongs in the user slot.

**3. `bootstrap_ci` moves to `shared/stats.py`.**
A working `bootstrap_ci` already exists at `pipeline/practice_eval/analyze_e2e.py` with tests at `tests/test_analyze_e2e.py`. We extract it (plus `cohens_d`) to `shared/stats.py`, leave a re-export in `analyze_e2e.py` so existing tests keep passing, and let `teaching_knowledge/scripts/aggregate.py` import directly from `shared/stats`. Zero behavior change to the existing pipeline eval.

**4. GPT-5.4-mini as a new `provider="openrouter"` on the existing `LLMClient`.**
Rather than a parallel client, we extend `teaching_knowledge/llm_client.py` to support a third provider via OpenRouter (`https://openrouter.ai/api/v1/chat/completions`). The public interface of `LLMClient.complete()` does not change. The `MODELS` table gets an `"openrouter"` entry with `openai/gpt-5.4-mini` as the model slug (OpenRouter's `vendor/model` naming convention). A new private method `_openrouter_complete()` mirrors `_workers_ai_complete()`. HTTP uses the existing `requests` dep. Benefit: one API key unlocks the full OpenRouter model catalog — swapping to Sonnet-as-judge in Phase 2 is a model-string change, not a new client. Auth env var is `OPENROUTER_API_KEY` (read from env or `apps/api/.dev.vars`). Family detection still treats `openai/gpt-*` as the `openai` family for the judge-compatibility check.

**5. Process/outcome split ships as a schema extension, not a replacement.**
`DimensionScore` in `shared/judge.py` gains two new fields (`process: int | None`, `outcome: int | None`). The existing `score: int | None` field remains for backwards compatibility with existing dry-run JSONL. The judge prompt (`synthesis_quality_judge_v2.txt`) is updated to request both `process` and `outcome` in its output. `_parse_v2_response` handles both legacy (single-score) and new (process+outcome) response shapes — legacy responses map `score` → both `process=score, outcome=score` for seamless transition. After the first run on the new schema, the aggregator treats them as independent signals.

**6. Judge-family compatibility encoded as a data table.**
`shared/judge_compatibility.py` exposes `assert_judge_compatible(teacher_model: str, judge_model: str) -> None`. It raises `ValueError` if the two models share a family. Families are derived from model-name prefixes (`claude-*` → anthropic, `@cf/openai/*` or `gpt-*` → openai, `@cf/google/*` → google, `@cf/qwen/*` → qwen). The function is called once at the top of `run_eval.run()` and once at the top of `dual_judge.run()`. New teacher/judge combinations only require adding a row to the family table.

**7. No new package dependencies.**
numpy is already transitively installed (used by analyze_e2e). requests is base. pytest is in dev-deps. subprocess (for git SHA) is stdlib. We add zero new entries to `pyproject.toml`.

**8. Fixture-driven tests; no real LLM calls.**
Every test in the plan is hermetic: it uses fixture JSONL, fixture manifests, or stub LLM clients. The GPT client test validates payload-building through a pure `_build_openai_payload` helper extracted from `_openai_complete`, avoiding network I/O. This means the entire plan can execute offline and in CI.

### Ordering principle

Foundation modules (composer→era, tag_dataset, GPT client) have no dependencies and ship first in parallel. Modules that layer on top (load_style_rules, split) come next. Then `run_eval.py` gets four sequential edits (all touch the same file, cannot parallelize). Judge schema and prompt edits touch different files and can parallelize. Analysis tooling (aggregate, regression_check) depends on the frozen output JSONL shape from the run_eval edits. Dual-judge and A/B wrappers depend on aggregate. This gives six natural task groups.

## Modules

### 1. `apps/evals/shared/style_rules.py` — **NEW, DEEP**

**Interface:**
```python
def composer_to_era(composer: str) -> str  # returns "Baroque" | "Classical" | ...
def get_style_guidance(composer: str) -> str  # returns formatted guidance text
```

**Hides:** The composer-substring matching rules (e.g., "Bach" → Baroque, "Chopin" → Romantic), the JSON loading, the fallback to "Unknown" era, the formatting of dimension priorities into a prose block injected into the synthesis prompt.

**Tested through:** Public `composer_to_era` and `get_style_guidance` only. No tests on the internal JSON structure or era table.

### 2. `apps/evals/shared/data/style_rules.json` — **NEW, data file**

The piece-style dimension rules extracted from `playbook.yaml`. Single source of truth for both Python and TypeScript. Schema:
```json
{
  "eras": {
    "Baroque": {
      "composer_patterns": ["Bach", "Handel", "Scarlatti", "Couperin"],
      "dimensions": {
        "articulation": "very high – clear separation, finger articulation, evenness.",
        "pedaling": "minimal or none; focus on finger legato.",
        ...
      }
    },
    ...
  }
}
```

### 3. `apps/api/src/lib/style-rules.json` — **NEW, data file (mirror)**

Byte-identical copy of `apps/evals/shared/data/style_rules.json`. Kept in sync via a pytest that hashes both and asserts equality.

### 4. `apps/evals/shared/stats.py` — **NEW, DEEP**

**Interface:**
```python
def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float] | None
def cohens_d(group1: list[float], group2: list[float]) -> float
```

**Hides:** The numpy random sampling, percentile computation, the small-sample guard (returns None for N<5), the pooled-stddev formula.

**Tested through:** Existing tests at `tests/test_analyze_e2e.py` (which get pointed at the new import path), plus new tests at `tests/test_stats.py` for any new behavior.

### 5. `apps/evals/shared/provenance.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class RunProvenance:
    run_id: str       # e.g., "2026-04-14T15-03-22Z_a1b2c3d"
    git_sha: str      # full SHA
    git_dirty: bool   # uncommitted changes?

def make_run_provenance(suffix: str | None = None) -> RunProvenance
```

**Hides:** The subprocess invocation for `git rev-parse HEAD`, dirty-tree detection via `git status --porcelain`, timestamp formatting (UTC, filesystem-safe), the graceful fallback when git binary is unavailable (returns `git_sha="unknown"`, `git_dirty=True`).

**Tested through:** Public `make_run_provenance()` — test that the returned object has all three fields populated and the run_id is filesystem-safe. Test the fallback by monkeypatching subprocess to raise FileNotFoundError.

### 6. `apps/evals/shared/judge_compatibility.py` — **NEW, DEEP**

**Interface:**
```python
def assert_judge_compatible(teacher_model: str, judge_model: str) -> None
    # Raises ValueError("judge family {f} matches teacher family — forbidden") if same family
```

**Hides:** The model-name-prefix → family mapping table (handling Anthropic-native names like `claude-*`, Workers AI `@cf/<vendor>/*` slugs, and OpenRouter `<vendor>/<model>` slugs like `openai/gpt-5.4-mini` or `anthropic/claude-sonnet-4-6`), the matching logic, the error message format.

**Tested through:** Public `assert_judge_compatible`. Tests cover: same-family raise (`claude-sonnet-4-6` + `anthropic/claude-sonnet-4-6` — tests that OpenRouter slug resolves to the same anthropic family as the native name), cross-family pass (`claude-sonnet-4-6` + `@cf/google/gemma-4-26b-a4b-it`), cross-family pass via OpenRouter (`claude-sonnet-4-6` + `openai/gpt-5.4-mini`), unknown model raise (bonus safety), Workers AI prefix resolution (`@cf/openai/gpt-oss-120b` resolves to openai family).

### 7. `apps/evals/teaching_knowledge/scripts/tag_dataset.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class RecordingTags:
    recording_id: str
    composer_era: str       # from shared/style_rules.composer_to_era
    skill_bucket: int       # copied from manifest
    duration_bucket: str    # "<30s" | "30-60s" | "60s+"

def tag_recording(recording_id: str, manifest_entry: dict, cache_entry: dict) -> RecordingTags
def build_dataset_index(manifest_lookup: dict, cache_dir: Path, out_path: Path) -> None
```

**Hides:** The era inference (delegates to `shared/style_rules`), the duration bucket boundaries, the cache-entry reading (`total_duration_seconds`), the JSONL output format.

**Tested through:** `tag_recording()` with fixture manifest entry + fixture cache entry — assert expected tags. `build_dataset_index()` with a temp directory — assert the JSONL file is produced and has the expected row count. No dependency on real cache files; tests construct their own.

### 8. `apps/evals/teaching_knowledge/scripts/split.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class Split:
    train: list[str]    # recording_ids
    holdout: list[str]  # recording_ids

def stratified_split(tags: list[RecordingTags], seed: int, holdout_ratio: float = 0.2) -> Split
def write_split(split: Split, out_path: Path) -> None
def load_split(split_path: Path, which: str) -> set[str]  # which in {"train","holdout","all"}
```

**Hides:** The stratification key (composer_era × skill_bucket), the random.Random(seed) deterministic sampling, the guarantee that every stratum contributes to both splits (or a documented exception for strata with N<2), the JSON persistence format.

**Tested through:** Public functions. Tests: deterministic output for same seed, 80/20 ratio on a 100-tag fixture, stratification preserves era distribution within tolerance, `load_split(which="holdout")` returns the correct set.

### 9. `apps/evals/teaching_knowledge/scripts/aggregate.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class DimensionAggregate:
    name: str
    mean_process: float | None
    ci_process: tuple[float, float] | None
    mean_outcome: float | None
    ci_outcome: tuple[float, float] | None
    n: int

@dataclass
class AggregateResult:
    dimensions: list[DimensionAggregate]
    composite_mean: float
    composite_ci: tuple[float, float] | None
    by_era: dict[str, dict[str, float]]      # era -> {dim -> mean}
    by_skill: dict[int, dict[str, float]]    # skill_bucket -> {dim -> mean}
    total_rows: int
    run_id: str

def aggregate_run(jsonl_path: Path, dataset_index_path: Path) -> AggregateResult
def write_aggregate(result: AggregateResult, out_path: Path) -> None
```

**Hides:** The JSONL streaming, the dim-name-to-values accumulator, the call to `shared.stats.bootstrap_ci`, the stratified breakdown joining on `dataset_index.jsonl`, the None-handling for unscored rows.

**Tested through:** `aggregate_run()` on a fixture 10-row JSONL — assert per-dim means match hand-computed values, CI is a valid tuple, stratification buckets are populated. No real runs needed.

### 10. `apps/evals/teaching_knowledge/scripts/regression_check.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class DimensionRegression:
    name: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    significant: bool    # CIs do not overlap
    direction: str       # "regressed" | "improved" | "null"

@dataclass
class RegressionReport:
    dimensions: list[DimensionRegression]
    composite_delta: float
    composite_significant: bool
    has_regression: bool  # any dim regressed with significance

def check_regression(baseline_agg: AggregateResult, candidate_agg: AggregateResult) -> RegressionReport
def format_report(report: RegressionReport) -> str
```

**Hides:** The CI overlap check, the direction classification, the text formatting for CLI output.

**Tested through:** Fixture `AggregateResult` instances — assert the report correctly identifies a known regression and a known improvement. No real runs.

### 11. `apps/evals/teaching_knowledge/scripts/dual_judge.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class DimensionAgreement:
    name: str
    spearman: float
    trust_level: str   # "high" (>0.7) | "uncertain" (0.4-0.7) | "low" (<0.4)

@dataclass
class DualJudgeReport:
    dimensions: list[DimensionAgreement]
    n_compared: int

def run_dual_judge(
    synthesis_jsonl: Path,
    judge_a_provider: str,
    judge_a_model: str,
    judge_b_provider: str,
    judge_b_model: str,
    out_path: Path,
) -> DualJudgeReport

def compute_agreement(judge_a_rows: list[dict], judge_b_rows: list[dict]) -> list[DimensionAgreement]
```

**Hides:** The parallel judge invocation, the per-dim Spearman computation (pure Python, rank-based, no scipy), the trust-level thresholds.

**Tested through:** `compute_agreement()` with fixture judge outputs — assert Spearman values and trust-level classifications. The full `run_dual_judge()` is exercised via a stub LLMClient subclass that returns canned JSON (no real API calls). The stub machinery proves the wiring works; real calibration numbers wait for Model v2.

### 12. `apps/evals/teacher_model/eval_ab.py` — **NEW, DEEP**

**Interface:**
```python
@dataclass
class ABReport:
    baseline_run_id: str
    candidate_run_id: str
    regression_report: RegressionReport
    efficiency_delta: dict[str, float]  # {"cost_usd": ..., "latency_ms": ..., "tokens": ...}
    verdict: str  # "CANDIDATE_WINS" | "CANDIDATE_LOSES" | "EQUIVALENT"

def run_ab(baseline_jsonl: Path, candidate_jsonl: Path, dataset_index: Path) -> ABReport
```

**Hides:** The double-aggregate, the delegation to `regression_check`, the efficiency deltas (computed from `synthesis_latency_ms` and `judge_latency_ms` fields already in the JSONL), the verdict logic (CANDIDATE_WINS requires composite improvement AND no dim regressed with significance).

**Tested through:** Fixture baseline + candidate JSONLs representing (a) a win, (b) a loss, (c) equivalence — assert the verdict matches expectation.

### 13. Modified: `apps/evals/teaching_knowledge/run_eval.py`

Four edits, each its own task because all touch the same file and must serialize:

- **T3:** `build_synthesis_user_msg` accepts the composer via `meta` (already does) and calls `shared.style_rules.get_style_guidance(meta["composer"])`, injecting the text between `<session_data>` and `<task>`. No new parameter needed — composer is already in `meta`.
- **T5:** `run()` accepts `split: str = "all"` and `split_path: Path | None = None`. Filters `cache_files` by `load_split(split_path, which=split)`. New CLI flags `--split` and `--split-file`.
- **T6:** Every row written to output JSONL gets `run_id` and `git_sha` fields from `make_run_provenance()`. CLI flag `--run-id-suffix` optional.
- **T7:** `run()` accepts `teacher_model` and `judge_model` CLI flags. Before the synthesis loop, `assert_judge_compatible(teacher_model, judge_model)` is called. Default values preserve existing behavior.

### 14. Modified: `apps/evals/shared/judge.py`

Two edits, different surfaces of the same file — cannot parallelize:

- **T10:** `DimensionScore` dataclass gains `process: int | None = None` and `outcome: int | None = None` fields. `_parse_v2_response()` handles both legacy (single `score`) and new (`process` + `outcome`) response shapes. Legacy responses map `score → process=score, outcome=score`. `JudgeResultV2.mean_score` unchanged (uses `score` field, which remains the aggregate).

### 15. Modified: `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`

- **T11:** Update the output schema in the prompt: `{ "criterion": "...", "process": <0-3 or N/A>, "outcome": <0-3 or N/A>, "evidence": "...", "reason": "..." }`. Add a short rubric paragraph explaining the split ("process = did the teacher notice/attempt the behavior; outcome = was the advice correct given the performance"). Example rows updated to match.

### 16. Modified: `apps/evals/teaching_knowledge/llm_client.py`

- **T14:** `MODELS` gains an `"openrouter"` entry with `"quality": "openai/gpt-5.4-mini"`, `"cheap": "openai/gpt-5.4-mini"`, `"judge": "openai/gpt-5.4-mini"`. New private method `_openrouter_complete()` mirrors `_workers_ai_complete()` but hits `https://openrouter.ai/api/v1/chat/completions` with `Authorization: Bearer $OPENROUTER_API_KEY`. New helper `_build_openrouter_payload()` is a pure function extracted from `_openrouter_complete` for testability (accepts model slug, messages, temperature; returns the dict body that is JSON-serialized onto the wire). `_load_openrouter_token()` reads `OPENROUTER_API_KEY` from env or `apps/api/.dev.vars`. Optional `HTTP-Referer` and `X-Title` headers are set to `https://crescend.ai` and `CrescendAI Evals` respectively (OpenRouter uses these for leaderboard attribution; harmless to include).

### 17. Modified: `apps/api/src/services/prompts.ts`

- **T4:** `buildSynthesisFraming` accepts a new required parameter `composer: string` (breaks the existing call signature — callers in `services/teacher.ts` must pass it through). A new internal helper `getStyleGuidance(composer)` reads `apps/api/src/lib/style-rules.json` and returns the formatted guidance text, which is inserted between `<session_data>` and `<task>`. A Python-side pytest hashes `apps/api/src/lib/style-rules.json` against `apps/evals/shared/data/style_rules.json` to catch drift.

### 18. Modified: `apps/evals/pipeline/practice_eval/analyze_e2e.py`

- **T0 (prep):** `bootstrap_ci` and `cohens_d` move to `shared/stats.py`. `analyze_e2e.py` re-exports them via `from shared.stats import bootstrap_ci, cohens_d`. Existing tests at `tests/test_analyze_e2e.py` continue to import from `pipeline.practice_eval.analyze_e2e` and keep passing. This is a refactor-only task (no behavior change) but ships as its own vertical slice because it unblocks T12 (aggregate.py).

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/shared/stats.py` | Bootstrap CI + Cohen's d (moved from analyze_e2e) | New |
| `apps/evals/shared/style_rules.py` | composer→era + get_style_guidance | New |
| `apps/evals/shared/data/style_rules.json` | Era dimension rules, single source of truth | New |
| `apps/evals/shared/provenance.py` | make_run_provenance() | New |
| `apps/evals/shared/judge_compatibility.py` | assert_judge_compatible() | New |
| `apps/evals/teaching_knowledge/scripts/__init__.py` | Package marker | New |
| `apps/evals/teaching_knowledge/scripts/tag_dataset.py` | Dataset tagging CLI | New |
| `apps/evals/teaching_knowledge/scripts/split.py` | Stratified split CLI | New |
| `apps/evals/teaching_knowledge/scripts/aggregate.py` | Aggregation + bootstrap CI CLI | New |
| `apps/evals/teaching_knowledge/scripts/regression_check.py` | Run diff CLI | New |
| `apps/evals/teaching_knowledge/scripts/dual_judge.py` | Dual-judge harness | New |
| `apps/evals/teacher_model/eval_ab.py` | Finetune A/B scaffold | New |
| `apps/api/src/lib/style-rules.json` | Byte-identical mirror of evals JSON | New |
| `apps/evals/tests/test_stats.py` | (covered by existing test_analyze_e2e + new tests for moved behavior) | New |
| `apps/evals/tests/test_style_rules.py` | composer→era + guidance tests | New |
| `apps/evals/tests/test_provenance.py` | make_run_provenance tests | New |
| `apps/evals/tests/test_judge_compatibility.py` | assert_judge_compatible tests | New |
| `apps/evals/tests/test_tag_dataset.py` | Tagging unit tests | New |
| `apps/evals/tests/test_split.py` | Stratified split tests | New |
| `apps/evals/tests/test_run_eval_style_injection.py` | build_synthesis_user_msg style tests | New |
| `apps/evals/tests/test_run_eval_split_flag.py` | --split behavior test | New |
| `apps/evals/tests/test_run_eval_provenance.py` | run_id/git_sha in output | New |
| `apps/evals/tests/test_run_eval_judge_family.py` | assert_judge_compatible wiring | New |
| `apps/evals/tests/test_judge_process_outcome.py` | _parse_v2_response schema | New |
| `apps/evals/tests/test_judge_prompt_schema.py` | prompt contains process+outcome | New |
| `apps/evals/tests/test_aggregate.py` | aggregate_run on fixture | New |
| `apps/evals/tests/test_regression_check.py` | check_regression on fixtures | New |
| `apps/evals/tests/test_llm_client_openrouter.py` | _build_openrouter_payload | New |
| `apps/evals/tests/test_dual_judge.py` | compute_agreement on fixtures | New |
| `apps/evals/tests/test_eval_ab.py` | run_ab verdict on fixtures | New |
| `apps/evals/tests/test_prompts_ts_style_rules.py` | prompts.ts contains era guidance + hash-match with evals JSON | New |
| `apps/evals/teaching_knowledge/run_eval.py` | Style injection, --split, provenance, judge-family assert | Modify |
| `apps/evals/shared/judge.py` | DimensionScore process/outcome fields, _parse_v2_response update | Modify |
| `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt` | process/outcome output schema | Modify |
| `apps/evals/teaching_knowledge/llm_client.py` | openai provider support | Modify |
| `apps/api/src/services/prompts.ts` | buildSynthesisFraming accepts composer, injects style | Modify |
| `apps/evals/pipeline/practice_eval/analyze_e2e.py` | Re-export stats from shared/stats | Modify |
| `apps/evals/teaching_knowledge/data/playbook.yaml` | Comment pointing to JSON source of truth | Modify |

## Open Questions

- **Q: Should prompts.ts change land now or defer until the finetune is ready?**
  Default: land now. The synthesis prompt in prod is what the beta users see; if the eval injects style guidance and prod does not, we are measuring a different prompt than we ship. Even though Sonnet is the teacher in Phase 1 and the finetune (Phase 2) will have its own prompt, keeping eval/prod aligned is the principle we do not want to violate. If landing the TS change causes friction (e.g., `services/teacher.ts` callers break), we can revisit — but the default is "land it."

- **Q: Does `buildSynthesisFraming`'s new required `composer` parameter break existing callers?**
  Default: yes, and we fix them in the same task. The callers are in `apps/api/src/services/teacher.ts` and possibly `apps/api/src/do/session-brain.ts`. Both already have composer in scope via piece metadata. Task T4 includes the caller update as part of the same test cycle.

- **Q: What if a composer in a manifest is "Unknown"?**
  Default: `composer_to_era` returns `"Unknown"` and `get_style_guidance` returns an empty string. The synthesis user message omits the `<style_guidance>` section entirely. This is tested.

- **Q: What if `git` is unavailable when `make_run_provenance` is called?**
  Default: fall back to `git_sha="unknown"`, `git_dirty=True`, log a warning to stderr. The run still produces a provenance-stamped artifact (with a visibly-degraded stamp). Tested via monkeypatch.

- **Q: Why OpenRouter instead of the OpenAI SDK directly?**
  Default: OpenRouter. One API key unlocks GPT-5.4-mini, Sonnet (for Phase 2), Gemini, and the full OpenRouter catalog — swapping judges becomes a model-string change instead of a new-client engineering task. The thin-proxy cost (~20–50ms added latency, small markup) is negligible for judge calls which are not user-facing. Raw HTTP via `requests` mirrors the existing Workers AI client, so zero new dependencies.

- **Q: Should `assert_judge_compatible` be a hard raise or a warning?**
  Default: hard raise (`ValueError`). Rationale: the invariant is a correctness property, not a suggestion. If a user genuinely wants to run same-family judging for research purposes, they can bypass with a `--allow-same-family` flag, which we deliberately do NOT add in this spec — it stays hidden until there is a real use case.

- **Q: Is the process/outcome split worth the judge-side complexity cost?**
  Default: yes, per the wiki principle and the specific failure mode we want to catch ("teacher seems fine but MuQ missed something"). The legacy-compatibility path means the change is low-risk — old fixtures and old responses keep working.
