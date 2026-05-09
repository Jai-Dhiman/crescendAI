# Stage 0 Capability Probe Design

**Status:** original 2026-05-07; amended 2026-05-08 to add Stage 1 prerequisite probes (tokenizer pinning, chitchat over-call sub-stratum, post-tool-result continuation degeneracy probe). Amendment changes are tagged `(added 2026-05-08)` inline.

**Goal:** Produce a defensible capability dossier that scores base Qwen3.6-35B-A3B at-ceiling / mid-tier / absent across the seven teacher capabilities (judgment, taste, integration, voice, vocabulary, tool-calling, adaptation), within a 1-2 day budget, with results that set training dosage for Stages 2-3 and decide whether Stage 5 is needed. The amendment additionally produces three Stage 1 prerequisite artifacts: a pinned tokenizer hash, a per-category tool-discipline breakdown including chitchat, and a post-tool-result continuation degeneracy rate.

**Not in scope:**
- Any training compute (Stage 0 is inference + judging only)
- Extending the existing 50-Q MCQ probe (runs as-is)
- A full paired comparison against the n=513 Sonnet baseline (we sample n=100)
- Production-format Anthropic tool_use testing (Stage 1 success metric, not Stage 0)
- Hosting / quantization decisions for finetuned models (Stage 0 hits OpenRouter)
- Rubric calibration / human inter-rater work (separate critical-path workstream)

## Problem

The Stage 0 entry in `apps/evals/teacher_model/TRAINING_PLAN.md` Section 4 says "score each capability at-ceiling / mid-tier / absent" but the existing artifacts don't measure all seven capabilities:

- The 50-Q MCQ (`domain_knowledge_probe.py`) tests piano-pedagogy facts. It maps weakly to *vocabulary* and *judgment* and not at all to *taste, voice, integration, adaptation, tool-calling*.
- The 7-dim synthesis judge (`synthesis_quality_judge_v2.txt`) covers *judgment, integration, voice, vocabulary* but has no signal for *taste, adaptation, tool-calling*.
- No tool-calling evaluation exists for the base model.

Without a per-capability verdict, Stages 1-3 dosage is guesswork. Stage 1 negative-example weighting, Stage 2 data volume, and Stage 3 hill-climb targets all depend on which capabilities are at-ceiling vs absent on the base model.

## Solution (from the user's perspective)

Run three commands. Get one dossier:

```bash
uv run python -m teacher_model.stage0 pin-tokenizer                                                   # (added 2026-05-08)
uv run python -m teacher_model.stage0 sample --n 100 --seed 42
uv run python -m teacher_model.stage0 synthesis    --provider openrouter --model qwen/qwen3.6-35b-a3b
uv run python -m teacher_model.stage0 tool         --provider openrouter --model qwen/qwen3.6-35b-a3b
uv run python -m teacher_model.stage0 continuation --provider openrouter --model qwen/qwen3.6-35b-a3b # (added 2026-05-08)
uv run python -m teacher_model.stage0 mcq          --provider openrouter --model qwen/qwen3.6-35b-a3b
uv run python -m teacher_model.stage0 aggregate
```

Reads `apps/evals/teacher_model/stage0/results/capability_dossier.md`:

```
## Capability dossier — qwen/qwen3.6-35b-a3b (instruct base)
| Capability         | Tier        | Primary signal | Value | vs Sonnet | CI         | Note |
|--------------------|-------------|----------------|-------|-----------|------------|------|
| Judgment           | mid-tier    | ASCF+SGD avg   | 1.42  | -0.38     | [1.32,1.51]|      |
| Taste              | absent      | Taste defens.  | 0.91  | n/a       | [0.78,1.04]| no anchor |
| Integration        | mid-tier    | CAP outcome    | 1.85  | -0.31     | [1.71,1.99]|      |
| Voice              | at-ceiling  | SPP+ATL+ASM    | 2.79  | -0.05     | [2.74,2.83]|      |
| Vocabulary         | at-ceiling  | SCML           | 2.95  | -0.05     | [2.90,3.00]|      |
| Tool-calling       | absent      | discipline%    | 33%   | n/a       | exact n=30 | no anchor |
| Adaptation         | absent      | Adapt. spec.   | 0.78  | n/a       | [0.62,0.94]| no anchor |
```

Each row shows the tier, the underlying number, the delta vs Sonnet (where anchored), the 95% CI, and any inconsistency flags between primary and corroborating signals. The dossier is the input to Stages 1-3 dosage decisions in TRAINING_PLAN Section 4.

## Design

### Approach

Three independent probes feeding one aggregator.

1. **Synthesis probe** (Pipeline A) — n=100 stratified-by-era×skill briefings drawn from the 890 in `model/data/eval/inference_cache/auto-t5_http/`. Uses the production synthesis system prompt (`apps/shared/teacher-style/synthesis_system.txt`) and the production briefing-construction logic (`build_synthesis_user_msg` from `teaching_knowledge/run_eval.py`). Each response is judged by an extended judge prompt that scores the 7 existing rubric dimensions plus 2 new dimensions: **Taste defensibility** and **Adaptation specificity**.

2. **Tool probe** (Pipeline B) — 40 hand-curated cases (20 positive, 20 negative) committed to `tool_probe_cases.jsonl`. The model is given the 6-tool palette (mirrored from `apps/api/src/services/tool-processor.ts`) via a Qwen-native system prompt with 3 few-shot examples. The scorer separates two metrics: **discipline accuracy** (call vs. don't-call decision) and **format-conditional schema validity** (when the model called, did args match the schema). **(amended 2026-05-08)** Each negative case carries a `category` tag (`chitchat | premature | ambiguous | already_recommended | out_of_scope | borderline_wrong_tool`); the aggregator emits per-category over-call rates so Stage 1 can confirm its Q4 contingency (chitchat-rate >70% triggers organic-narration negatives).

2b. **Continuation probe** (Pipeline B+, added 2026-05-08) — for each of the 20 positive cases on which the model successfully tool-called, construct a synthetic `tool_result` follow-up turn (canned plausible result for that tool) and re-prompt the model. Score the assistant's continuation against four degeneracy categories: **refusal** (explicit "I cannot continue"), **repetition** (re-emits the same tool_call), **format collapse** (output is not valid prose, e.g. raw JSON dump), **empty/truncated** (length below 10 tokens). Aggregator emits `continuation_degeneracy_rate` (any-of-4 / total) and a per-category breakdown. This rate gates the brainstormed Q2 emission-only choice for Stage 1 — degeneracy >30% triggers the contingency primer plan.

3. **MCQ probe** (Pipeline C) — runs the existing `apps/evals/teacher_model/domain_knowledge_probe.py` against OpenRouter with no changes other than adding `openrouter` to its `--provider` choices.

The **aggregator** combines all three result files into the capability dossier per the signal-mapping table (below). Tier classification uses Sonnet-anchored relative thresholds for the 4 capabilities the existing rubric covers, and absolute thresholds for the 3 capabilities with no Sonnet baseline (taste, adaptation, tool-calling) — these latter rows are explicitly flagged "no baseline anchor" in the dossier.

### Signal-to-capability mapping

| # | Capability | Primary signal | Corroborating signal | Anchor | Tier rule |
|---|---|---|---|---|---|
| 1 | Judgment | Synthesis: avg(ASCF, SGD) outcome | Tool probe: when-to-call discipline on negative cases | Sonnet relative | worst delta of {ASCF vs 1.387, SGD vs 2.195} |
| 2 | Taste | Synthesis: Taste defensibility (NEW dim, 0-3) | — | Absolute | ≥2.5 ceiling, 1.5-2.5 mid, <1.5 absent |
| 3 | Integration | Synthesis: CAP outcome | Synthesis: composite mean over 7 base dims | Sonnet relative | CAP delta vs 2.164; corroborator must agree within 1 tier |
| 4 | Voice | Synthesis: avg(SPP, ATL, ASM) outcome | — | Sonnet relative | delta vs Sonnet 3-dim mean (≈2.84) |
| 5 | Vocabulary | Synthesis: SCML outcome | MCQ: "concepts" topic accuracy | Sonnet relative | SCML delta vs 3.0; corroborator: MCQ concepts ≥60% required for "at-ceiling" |
| 6 | Tool-calling | Tool probe: discipline accuracy | Tool probe: format-conditional schema validity (reported, not tiered) | Absolute | discipline ≥80% ceiling, 50-80% mid, <50% absent |
| 7 | Adaptation | Synthesis: Adaptation specificity (NEW dim, 0-3) | — | Absolute | ≥2.5 ceiling, 1.5-2.5 mid, <1.5 absent |

Tier thresholds (Sonnet-anchored mode): within 0.25 of baseline = **at-ceiling**; 0.25-0.75 below = **mid-tier**; >0.75 below = **absent**. When a 95% CI straddles a tier boundary, the dossier emits a compound label (e.g., `mid_tier_with_ceiling_overlap`) rather than picking one side.

### Stratified sampling design

Strata = era (Romantic / Baroque / Classical / Impressionist via `composer_to_era`) × skill_bucket (the bucket values present in the manifests, restricted to those with ≥10 baseline samples). Approximately 12 cells × 8-9 each ≈ 100. Deterministic via fixed seed (default 42). The 100 briefing IDs commit to `stage0_holdout.jsonl` so all post-Stage-1 evals can hit the *same* recordings for paired comparison.

### Why this approach over alternatives

- **Why not extend MCQ to cover all 7 capabilities?** MCQs measure "can pick the right answer," not "can produce teacher behavior." The synthesis eval already does the latter; an MCQ proxy for integration / adaptation would add cost without fixing the proxy gap.
- **Why not the full n=513 paired comparison?** ~5x the inference + judge cost for 0.05-0.1 tighter CIs, which doesn't change at-ceiling/mid/absent classification. Save the paired n=513 for post-Stage-2 evaluation where deltas are smaller.
- **Why score tool-call discipline and format separately?** Stage 1's job is teaching native Anthropic `tool_use` format. Testing the base model on that target is rigged — it'll fail on syntax even with perfect when-to-call instinct, telling us nothing about whether Stage 1 needs heavy negative-example weighting OR just format SFT.
- **Why absolute thresholds for taste/adaptation/tool-calling?** No Sonnet baseline exists for those signals. We could synthesize one (run Sonnet on the same 30 tool cases, judge Sonnet outputs on the 2 new dims) but that adds ~half a day and biases the Stage 4 DPO baseline. Defer to absolute thresholds, flag explicitly.

### Trade-offs chosen

- **Speed over precision:** n=100 vs n=513 means ~±0.10 CIs vs ±0.025. Acceptable because tier classification is robust at that resolution.
- **Implementer drafts tool cases, founder reviews:** founder-from-scratch is highest signal quality but slows Stage 0 by ~half a day. Implementer-drafts-with-review keeps the budget.
- **Inference via OpenRouter:** zero infra setup, but OpenRouter routes across providers — the dossier records `routed_provider` from response headers so reproducibility is preserved. If OpenRouter only ships the instruct variant of qwen3.6-35b-a3b, the dossier labels it "instruct base" (which is closer to the actual Stage 1 starting checkpoint anyway).
- **Two new judge dimensions extend the existing prompt rather than running a separate judge call:** keeps cost flat and ensures every row has all 9 dim scores from a single judgment context.

## Modules

- **sampler** (`apps/evals/teacher_model/stage0/sampler.py`)
  - Interface: `sample_holdout(briefings_dir: Path, manifests: dict, n: int, seed: int) -> list[dict]`
  - Hides: era classification via `composer_to_era`, per-stratum bucket math, deterministic seeding, balance correction when a stratum is under-populated
  - Tested through: public function with the real 890-briefing pool; assertions on per-stratum counts and seed determinism
- **tool_scorer** (`apps/evals/teacher_model/stage0/tool_scorer.py`)
  - Interface: `score_response(raw: str, expected: ToolCase, schemas: dict) -> ToolProbeResult`
  - Hides: 3 format-tolerant extractors (Qwen `<tool_call>...</tool_call>`, raw JSON, prose-with-embedded-JSON), JSON-schema validation, discipline classification (call/no-call/wrong-tool)
  - Tested through: input fixtures covering each format × {present, absent} × {valid args, invalid args}
- **tier_classifier** (`apps/evals/teacher_model/stage0/tier_classifier.py`)
  - Interface: `classify_tier(value: float, baseline: float | None, mode: Literal['relative', 'absolute'], ci: tuple[float, float] | None) -> Tier`
  - Hides: threshold tables, CI overlap → compound-label logic
  - Tested through: table-driven boundary tests
- **judge_extended** (`apps/evals/teacher_model/stage0/judge_extended.py`)
  - Interface: `judge_extended(synthesis_text: str, context: dict, provider: str, model: str | None) -> JudgeResultV2Extended`
  - Hides: extended prompt (7 base + 2 new dims), 9-dim parsing
  - Tested through: parse a fixture judge response, assert correct dim count and structure
- **aggregator** (`apps/evals/teacher_model/stage0/aggregator.py`)
  - Interface: `build_dossier(synthesis_jsonl: Path, tool_jsonl: Path, mcq_json: Path, baseline_aggregate_json: Path, out_dir: Path) -> Dossier`
  - Hides: capability→signal map application, primary/corroborating cross-check + inconsistency flag, bootstrap CI on synthesis means, exact CI on tool discipline, dossier JSON + Markdown rendering, error-rate gate (>5% → refuse to emit)
  - Tested through: synthetic input JSONLs that exercise each tier path, the inconsistency flag, and the error-rate gate
- **run_synthesis** / **run_tool_probe** (intentionally shallow orchestration glue): retry/backoff, resume from existing JSONL, append-on-flush. Tested via a `--n 5` smoke test against a cheap model in CI.
- **continuation_probe** (`apps/evals/teacher_model/stage0/continuation_probe.py`, added 2026-05-08)
  - Interface: `score_continuation(initial_assistant: str, tool_result: dict, follow_up_response: str) -> ContinuationResult`
  - Hides: 4-category degeneracy classifier (refusal / repetition / format-collapse / empty), tool-result fixture builder per tool name, length thresholds
  - Tested through: fixture inputs covering each degeneracy category and a clean continuation; assert each is classified correctly
- **pin_tokenizer** (`apps/evals/teacher_model/stage0/pin_tokenizer.py`, added 2026-05-08)
  - Interface: `pin_tokenizer(model_id: str, out_path: Path) -> TokenizerPin`
  - Hides: tokenizer download via `transformers.AutoTokenizer.from_pretrained`, file-set hash computation (sha256 over sorted relative paths + contents of `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` if present, and any added vocab files), JSON serialization of the pin record
  - Tested through: invoke with a small public tokenizer (e.g. `gpt2`), assert hash is stable across two invocations and changes when one source file is mutated

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/evals/teacher_model/stage0/__init__.py` | Package marker | New |
| `apps/evals/teacher_model/stage0/sampler.py` | Stratified holdout sampling | New |
| `apps/evals/teacher_model/stage0/tool_scorer.py` | Tool-call extraction + scoring | New |
| `apps/evals/teacher_model/stage0/tier_classifier.py` | Sonnet-anchored + absolute tier classification | New |
| `apps/evals/teacher_model/stage0/judge_extended.py` | Judge v2 + 2 added dims | New |
| `apps/evals/teacher_model/stage0/aggregator.py` | Dossier builder | New |
| `apps/evals/teacher_model/stage0/run_synthesis.py` | Pipeline A runner | New |
| `apps/evals/teacher_model/stage0/run_tool_probe.py` | Pipeline B runner | New |
| `apps/evals/teacher_model/stage0/cli.py` | argparse front-end (`sample`/`synthesis`/`tool`/`mcq`/`aggregate` subcommands) | New |
| `apps/evals/teacher_model/stage0/prompts/judge_v2_extended.txt` | Extended judge prompt (9 dims) | New |
| `apps/evals/teacher_model/stage0/prompts/tool_probe_system.txt` | Qwen-native tool-use system prompt + 3 few-shot examples | New |
| `apps/evals/teacher_model/stage0/data/tool_probe_cases.jsonl` | 40 curated cases (20 positive, 20 negative across 6 categories); drafted by implementer, founder-reviewed | New |
| `apps/evals/teacher_model/stage0/continuation_probe.py` | Post-tool-result continuation degeneracy classifier (added 2026-05-08) | New |
| `apps/evals/teacher_model/stage0/pin_tokenizer.py` | Tokenizer download + hash pin (added 2026-05-08) | New |
| `apps/evals/teacher_model/stage0/run_continuation.py` | Pipeline B+ runner (synthetic tool_result follow-up + score) (added 2026-05-08) | New |
| `apps/evals/teacher_model/stage0/results/tokenizer_pin.json` | Generated artifact: pinned Qwen3.6-A3B tokenizer hash + file list (consumed by Stage 1) (added 2026-05-08) | New (generated, committed) |
| `apps/evals/teacher_model/stage0/tests/test_continuation_probe.py` | Degeneracy classification fixtures (added 2026-05-08) | New |
| `apps/evals/teacher_model/stage0/tests/test_pin_tokenizer.py` | Hash stability + mutation sensitivity (added 2026-05-08) | New |
| `apps/evals/teacher_model/stage0/data/stage0_holdout.jsonl` | Generated then committed (deterministic, seed 42) | New |
| `apps/evals/teacher_model/stage0/tests/test_sampler.py` | Stratification + determinism | New |
| `apps/evals/teacher_model/stage0/tests/test_tier_classifier.py` | Boundary tests | New |
| `apps/evals/teacher_model/stage0/tests/test_tool_scorer.py` | Format extraction + schema validation | New |
| `apps/evals/teacher_model/stage0/tests/test_aggregator.py` | Dossier shape + inconsistency flag + error-rate gate | New |
| `apps/evals/teacher_model/domain_knowledge_probe.py` | Add `"openrouter"` to `--provider` choices (1-line change) | Modify |

## Open Questions

- **Q:** Does OpenRouter ship `qwen/qwen3.6-35b-a3b` as raw base or instruct? **Default:** record whatever OpenRouter labels it (via `/api/v1/models/...`) and tag the dossier `instruct base` if instruct-tuned. Stage 1 LoRA usually starts from instruct anyway, so the comparison stays valid.
- **Q:** Should the dossier also score Sonnet on the 2 new dimensions (taste, adaptation) to give those rows a baseline anchor? **Default:** no — defer; flagging "no baseline anchor" is honest and adding it costs ~half a day. Revisit post-Stage-1 if those rows end up driving expensive Stage 3 decisions.
- **Q:** When OpenRouter routes the same model across multiple providers within one Stage 0 run, do we accept the mixed-provider result or pin to one provider? **Default:** record `routed_provider` per row, emit `provider_mix` in dossier meta, and flag "mixed-provider run" if >1 provider served the run. Re-run pinned only if the flag fires AND a capability sits on a tier boundary.
- **Q (added 2026-05-08):** What constitutes a "plausible" synthetic `tool_result` payload for the continuation probe across the 6 tools? **Default:** for each tool, hand-author one canonical successful payload (e.g. for `search_catalog`, return `{"matches": [{"pieceId": "chopin.ballades.1", "composer": "Chopin", "title": "Ballade No. 1", "barCount": 264}]}`). Commit these to `stage0/data/continuation_fixtures.json`. Re-use the same payload across all positive cases that called that tool — the probe measures continuation behavior, not result-handling sophistication.
- **Q (added 2026-05-08):** Should `pin-tokenizer` fail loudly if the model card lists no `chat_template.jinja`? **Default:** yes — Stage 1 cannot proceed without an explicit chat template (Q1 brainstorm decision: chat-template-native tool-call slot). The pin step writes the pin file only on success; absence of `chat_template.jinja` raises `MissingChatTemplateError` and the dossier flags `tokenizer_pin: failed`.
