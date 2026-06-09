# Teacher Model Training Plan -- Qwen3-30B-A3B

**Status:** active plan (2026-05-07). Supersedes the 2026-05-05 draft (which itself superseded the 4-stage CPT/SFT/GRPO/DPO plan from 2026-03-30).
**Origin:** capability-decomposition session 2026-05-07. Plan is now grounded in the explicit three-layer architecture (ear / harness / teacher) and the production interface contract.

---

## 1. The architectural frame

CrescendAI's teacher pipeline is three layers. Training only touches the third.

| Layer | Job | Implementation |
|---|---|---|
| **Ear** | Convert audio to structured signals | MuQ (quality) + Aria-AMT (transcription) + (planned) tone-color features |
| **Harness** | State, retrieval, tool execution, artifact rendering | Cloudflare Worker + Durable Object + Postgres + R2 + Rust WASM |
| **Teacher** | Judgment, taste, integration, voice, vocabulary, tool-calling, adaptation | Finetuned Qwen3-30B-A3B |

The teacher model owns exactly seven capabilities; everything else belongs to the ear or the harness. This decomposition is the basis for every training decision below.

### What the teacher MUST own

1. **Pedagogical judgment** -- symptom vs root cause; correct vs encourage; what to assign; what NOT to assign; when to drop a piece.
2. **Musical taste** -- phrasing, rubato, voicing priority, character, style. The territory where two excellent teachers thoughtfully disagree.
3. **Multi-signal integration** -- fusing ear signals + score-following + student history + piece style + session context into one coherent response.
4. **Voice and warmth** -- register, tone, encouragement-to-correction ratio, never-say rules, house style.
5. **Pedagogical vocabulary** -- "brighter", "from deeper in the key", "let the line breathe". Translating measured signals into teacher language.
6. **Tool-calling protocol** -- deciding to call a tool, constructing arguments correctly, knowing when NOT to call.
7. **Adaptation logic** -- bespoke modifications a static library can't pre-author ("Czerny op. 299 #14 but only mm. 1-8, hands separate, dotted rhythm").

### What the teacher MUST NOT own

1. Authoritative facts (exercise IDs, MEI, opus numbers, fingerings, recording URLs) -- harness via tools.
2. Session/student state -- harness injects as context.
3. Acoustic measurement -- ear.
4. Score arithmetic (alignment, DTW, bar mapping) -- harness/Rust WASM.
5. Artifact rendering (Verovio, audio playback) -- harness.

---

## 2. The interface contract (load-bearing)

Every training example matches the production interface exactly. If training shape != inference shape, the model learns artifacts.

**Teacher input (briefing):**
```
{
  audio_observations: { 6 dims + bar-level signals + (planned) tone-color features },
  score_position: { current_bar, piece_id, alignment_confidence },
  piece_metadata: { composer, era, style, edition_notes },
  student_state: { skill_bucket, recent_pieces, known_weaknesses, baseline_scores },
  session_history: { recent_sessions, current_session_arc, top_moments },
  available_tools: [recommend_exercise, fetch_recording, fetch_fingering, ...]
}
```

**Teacher output:**
```
{
  prose: string,              // warm, voiced, integrated response
  tool_calls: ToolCall[]      // zero or more; never invents IDs
}
```

This is the only shape that ever appears in training.

---

## 3. Capability -> intervention mapping

| Capability | Training intervention |
|---|---|
| Tool-calling protocol | Tool-format LoRA SFT + negative examples (when NOT to call) |
| Integration (briefing -> response) | Briefing-shaped LoRA SFT |
| Pedagogical judgment | Capability LoRA SFT + reasoning-trace distillation |
| Adaptation logic | SFT with reasoning traces |
| Vocabulary | Falls out of SFT (vocabulary appears in correct contexts) |
| Voice / warmth / register | DPO on rubric-judged + human-rated pairs |
| Taste (rubato, phrasing, character) | DPO; never RL on noisy reward |

---

## 4. The five-stage curriculum

One LoRA adapter, carried through Stages 1-4. Stage 5 is contingent.

### Stage 0 -- Capability mapping (no training compute, ~1-2 days)

**Goal:** establish baseline of base 35B-A3B against the seven teacher capabilities.

**Status:** harness implemented (`apps/evals/teacher_model/stage0/`). Run via:
```
cd apps/evals
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 pin-tokenizer --model Qwen/Qwen3-30B-A3B-Instruct-2507
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 sample --n 100
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 synthesis --model <openrouter-id>
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 tool --model <openrouter-id>
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 continuation --model <openrouter-id>
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 mcq --model <openrouter-id>
uv run --extra teacher-model-stage0 python -m teacher_model.stage0 aggregate
```

**Pipelines:**
- **A (synthesis):** n=100 stratified holdout, judge v2-extended (9 dims = 7 base + Taste + Adaptation).
- **B (tool-probe):** 40 hand-curated cases (20 positive / 20 negative across 6 negative categories).
- **B+ (continuation):** replays successful tool calls with synthetic `tool_result`, classifies degeneracy.
- **C (MCQ):** existing 50-Q domain knowledge probe (`domain_knowledge_probe.py`) with openrouter support.
- **Aggregator:** Sonnet-anchored + absolute tier classification; 5% error-rate gate; emits `capability_dossier.json` + `.md`.

**Output:** capability dossier that sets dosage for Stages 2-3 and decides whether Stage 5 is needed.

### Stage 1 -- Tool-format SFT (universally needed)

**Goal:** native emission of the 6-tool palette in Anthropic-compatible `tool_use` format with correct when-to-call discipline.

**Status:** data pipeline implemented (`apps/evals/teacher_model/stage1/`). Run via:
```
cd apps/evals
uv run python -m teacher_model.stage1 holdout --cache-dir <briefings-dir> --out holdout.jsonl
uv run python -m teacher_model.stage1 distill --shape synthesis --n 1000
uv run python -m teacher_model.stage1 render --out rendered.jsonl
uv run python -m teacher_model.stage1 harness --endpoint <vllm-url> --holdout holdout.jsonl --tokenizer-pin tokenizer_pin.json
```
Pydantic validators for all 6 tools locked against `tool-processor.ts` via SHA256 contract test (`test_schema_contract.py`).

**Data:** ~2K examples. ~30% **negative examples** (briefings where calling a tool would be wrong -- chitchat, ambiguous moments, low-confidence observations).

**Method:** LoRA SFT. Targets per Unsloth Qwen3.5 default: `q/k/v/o + gate/up/down`. Seq_len 4096. Log per-token expert entropy throughout.

**Why first:** every later stage assumes tool calls work. Also eliminates the TS-side translation layer (Agent 1 task).

**Side effect:** if this lands, the model speaks Anthropic-compatible `tool_use` natively -- no translation shim needed in production.

### Stage 2 -- Briefing-shaped SFT (the integrator)

**Goal:** model accepts the production briefing and produces coherent, integrated responses.

**Data:** ~5-10K examples. Build pipeline:
1. Pull real briefings from `model/data/eval/inference_cache/auto-t5_http/` (890 files available).
2. Generate ideal responses via Sonnet/GPT-5 on each briefing.
3. Filter through 8-dim rubric (judge v2): keep only chosen >= 2.5 composite.
4. Hand-curate ~500 golden cases for hard scenarios (sparse coverage areas).

**Method:** LoRA SFT on the same adapter as Stage 1.

**Critical:** every example is the exact production interface shape. No briefing-less examples. No raw text. No MEI.

### Stage 3 -- Capability SFT with reasoning traces (targeted hill-climb)

**Goal:** close the specific gaps Stage 0 identifies. Highest-confidence targets:
- **ASCF (Audible-Specific Corrective Feedback)** -- locked baseline 1.387 / 3.0; weakest dim.
- Root-cause vs symptom reasoning.
- Exercise selection logic.
- "What NOT to assign" calls.

**Data:** ~10-20K examples with reasoning traces.

```
<briefing>...</briefing>
<reasoning>
Signals: rushed in m.12 (timing 0.62), uneven 16ths (articulation 0.71),
LH dynamics drop (0.55).
Root cause hypothesis: LH tension -> rushing as compensation.
Pedagogical move: address tension before rhythm.
Exercise: slow LH-only practice + Pischna #5.
</reasoning>
<response>...warm, integrated teacher prose...</response>
<tool_call>recommend_exercise(id="pischna_5", rationale="...")</tool_call>
```

The `<reasoning>` block trains the integration logic. At inference it can be stripped or kept depending on UX.

**Sources:**
- Track A masterclass extractions (379 teaching moments).
- Sonnet/GPT-5 distillation through 8-dim rubric.
- ~500 hand-curated **adversarial briefings** (cases where naive teachers respond wrongly: over-correcting beginners, under-praising advanced, ignoring session arc, recommending too many things).

**Method:** LoRA SFT continuing the same adapter.

### Stage 4 -- DPO for voice, warmth, taste

**Goal:** consistent house voice; correct register for skill level; taste-level interpretation choices; warmth.

**Data:** ~5-15K pairs.
- ~1K human-rated pairs (top-tier discrimination, drawn from rubric calibration set).
- ~5-15K synthetic pairs (rubric-judged Sonnet-vs-Sonnet or Sonnet-vs-GPT-5 on same briefing).
- Hybrid approach: human for top-tier, synthetic for bulk.

**Hard prerequisite:** rubric human calibration Phase 1 (weighted κ ≥ 0.6 on 11/14 sub-scores, founder-rateable dims) must complete BEFORE this stage. Phase 2 (expert pianist rating on 3 remaining outcome dims: ASCF, Scaffolded, Style-Consistent) must also complete before DPO pairs on those dims are included -- without Phase 2, proceed with synthetic-only pairs on expert dims only. Without calibration, DPO is preference noise. This is the most under-attended item on the critical path.

**Method:** DPO on the same LoRA adapter.

**Why DPO not GRPO:**
- GRPO needs verifiable reward; 8-dim rubric is noisy.
- GRPO on MoE is unstable; KL drifts unpredictably across experts.
- DPO is simpler, MoE-stable, equivalent outcome.
- 2026 production narrow-domain models (medical, legal, financial) almost universally use SFT+DPO.

### Stage 5 -- CPT (contingent only, likely skipped)

Run only if Stage 0 + Stage 3 reveal a real knowledge gap that capability SFT didn't close.

If needed: **instruction-formatted CPT** (Q-A pairs about pedagogy), not raw text. Raw-text CPT on MoE has high routing-collapse risk and the harness fetches facts anyway.

**Most likely outcome:** not needed.

---

## 5. Three creative additions (built into stages above, not separate phases)

1. **Negative tool examples** (Stage 1) -- explicit "I'd just listen here" / "no exercise yet" briefings. Combats over-eager tool use.
2. **Voice anchors** (Stage 2/4) -- ~50-100 distilled exemplar teacher voices from masterclass transcripts. Train consistent embodiment of *house* voice instead of drift between teaching styles.
3. **Adversarial briefings** (Stage 3) -- ~500 hand-curated cases where naive teachers fail. Bends the policy where rubric data is sparse.

---

## 6. What we explicitly DO NOT do

- **Don't CPT raw pedagogy text.** The 100M-token clean corpus is *source material* for Stage 2/3 synthesis, not direct training data.
- **Don't train on MEI / Verovio / exercise IDs / opus numbers.** Facts go through tools. Every training example treats facts as tool outputs.
- **Don't run multiple LoRA adapters and merge.** Carry one adapter through Stages 1->2->3->4. Reduces compounding routing drift.
- **Don't run Stage 4 (DPO) before rubric calibration r >= 0.7.** Most common way preference training goes sideways.
- **Don't full-FT.** Re-densifies a sparse model; LoRA on `q/k/v/o + gate/up/down` is the MoE-aware compromise.
- **Don't use GRPO/PPO.** MoE instability + noisy reward = reward hacking.
- **Don't train a vision adapter.** Qwen3.6 is multimodal but our signals are structured, not images.

---

## 7. MoE-specific training discipline (cross-cutting)

- **Tooling:** Unsloth / Llama-Factory / Swift over bare TRL. They handle load-balance aux loss and expert-routing telemetry.
- **LoRA targets:** `q/k/v/o + gate/up/down` (Unsloth Qwen3.5 default). LoRA-ing all 256 experts re-densifies the model; this default touches the shared expert + router gates -- the MoE-aware compromise.
- **Expert-entropy monitoring:** log per-token expert entropy at every stage. If entropy collapses below threshold mid-training, abort and re-architect.
- **Sequence length:** 4096 for SFT/DPO. 262K native context is overkill for pedagogy texts and amplifies routing imbalance.
- **Single adapter carry-forward:** same LoRA across Stages 1-4. Reduces compounding routing drift.
- **Aux loss:** maintain load-balancing aux loss across all training stages.

---

## 8. Strategic gates

1. **Stage 0 baseline complete** -- gates dosage of Stages 2-3 and whether Stage 5 is needed.
2. **Rubric human calibration Phase 1 (weighted κ ≥ 0.6, 11/14 sub-scores)** -- gates Stage 2 quality filter and Stage 4 partial preference signal (founder-rateable dims). Phase 2 (expert pianist, 3 outcome dims: ASCF, Scaffolded, Style-Consistent) gates Stage 4 fully. On critical path twice.
3. **Tone-color ear features shipped (or scoped out)** -- gates whether vocabulary training in Stage 2/3 includes tone color.
4. **Blind A/B voice test vs. Sonnet** -- gates production deployment.
5. **200+ beta users with retention** -- gates serious compute commitment beyond LoRA stages.

---

## 9. Compute envelope

LoRA on 35B-A3B fits on 1-2 nodes. Per-stage rough estimate:

| Stage | Examples | Compute | Wall-clock |
|---|---|---|---|
| 0 | -- | inference only | 1-2 days |
| 1 | ~2K | LoRA SFT | 1-2 days train |
| 2 | ~5-10K | LoRA SFT | 3-5 days train |
| 3 | ~10-20K | LoRA SFT | 5-7 days train |
| 4 | ~5-15K pairs | LoRA DPO | 3-5 days train |
| 5 | (contingent) | LoRA instruct-CPT | 5-10 days |

**Critical path is data, not GPUs.** Data pipelines for Stages 2-4 dominate the timeline (8-13 weeks of data work vs ~3 weeks of training).

---

## 10. Workstream impact

| Workstream | Status under this plan |
|---|---|
| **CPT-corpus pipeline** (`apps/evals/teacher_model/data/corpus/`, 16M / 100M words) | Pivots from "CPT-ready dataset" to **"SFT data synthesis source corpus."** Same cleaning/dedup work; downstream consumer changes. Still ships as v1 dataset -- load-bearing infra regardless of training shape. |
| **Translation layer** (TS-side Anthropic <-> Qwen tool format) | **Likely obsolete after Stage 1.** Stops being needed once tool-format SFT lands. Stopgap for the pre-finetune period only. |
| **Domain probe** (`domain_knowledge_probe.py`) | **Promoted to Stage 0 anchor.** Probe + free-form synthesis eval on base model becomes the gating step for the rest of the plan. |
| **Rubric calibration** | **Phase 1 tooling SHIPPED** (`apps/evals/teacher_model/calibration/`). Stratified sample selection, founder rater CLI, per-sub-score weighted kappa, drift detection, and filter-recipe emission are code-complete with 40 passing tests. Ready to run founder ratings. Phase 2 (expert pianist, 3 outcome dims: ASCF, Scaffolded, Style-Consistent) deferred. Sits on critical path twice (Stage 2 filter + Stage 4 signal). |

---

## 11. Open questions

1. **Tone-color features:** scoped in or out? If in, ear-side feature engineering (spectral centroid trajectories, attack sharpness, harmonic-to-noise ratio, sustain-decay envelope) is a prerequisite for vocabulary training in Stages 2/3.
2. **Tool-format details for Qwen3.6:** model card mentioned `qwen3_coder` parser; verify chat-template format vs. Qwen3.5 before locking Stage 1 SFT data format.
3. **Reasoning-trace UX:** keep `<reasoning>` blocks at inference (transparent teacher) or strip them (clean teacher voice)? Affects Stage 3 data shape.
4. **Voice anchor sourcing:** which 50-100 teacher voices from masterclass corpus best represent house voice? Hand-curated by founder vs. clustered by judge?
5. **DPO data sourcing ratio:** human-to-synthetic mix needs sizing once rubric calibration completes.

---

## 12. References

- `apps/evals/teacher_model/stage0/` -- Stage 0 capability probe harness (shipped).
  - `cli.py` -- entry point (`python -m teacher_model.stage0 <subcommand>`)
  - `aggregator.py` -- builds `capability_dossier.json` + `.md`
  - `run_synthesis.py` / `run_tool_probe.py` / `run_continuation.py` -- pipeline runners
  - `stage0/data/tool_probe_cases.jsonl` -- 40 hand-curated tool-probe cases
  - `stage0/prompts/judge_v2_extended.txt` -- 9-dim judge prompt
- `apps/evals/teacher_model/domain_knowledge_probe.py` -- Stage 0 MCQ anchor (openrouter supported).
- `apps/evals/teacher_model/data/corpus/` -- 100M-token cleaned corpus (Stage 2/3 source material).
- `apps/api/src/services/tool-processor.ts` -- 5-tool palette (Stage 1 target format).
- `apps/api/src/services/teacher.ts` -- unified teacher service (deployment target).
- `apps/api/src/harness/loop/phase2.ts` (`buildPhase2Prompt`) + `apps/api/src/harness/artifacts/synthesis.ts` -- V6 synthesis prompt + artifact schema (briefing format anchor; replaced the deleted `synthesis_system.txt` when V6 became canon, #28).
- `model/data/eval/inference_cache/auto-t5_http/` -- 890 real briefings (Stage 2 source).
- `apps/evals/teaching_knowledge/data/raw_teaching_db.json` -- 379 masterclass moments (Stage 3 source).
- `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt` -- 8-dim rubric judge (Stage 2 filter, Stage 4 signal).
- `apps/evals/teacher_model/calibration/` -- Phase 1 calibration protocol (shipped). Entry points: `select_sample.select_sample()`, `rater_cli.capture_synthesis_ratings()`, `analyze_calibration.calibrate()`, `analyze_drift.analyze_drift()`, `emit_recipe.emit()`. Artifact output: `calibration/artifacts/filter_recipe.py`.
- Memory: `project_teacher_model_finetuning.md`, `project_teaching_knowledge_eval.md`, `project_corpus_pipeline.md`.

---

## Appendix -- plan history

- **2026-03-30:** original plan, Qwen3.5-27B dense, 4 stages (CPT + SFT + GRPO + DPO).
- **2026-05-05:** model switched to Qwen3-30B-A3B (sparse MoE). 4-stage plan invalidated due to MoE failure modes (expert collapse, GRPO instability). Reduced to 2-stage SFT + DPO with contingent CPT.
- **2026-05-07 (current):** plan reframed around explicit ear/harness/teacher decomposition and production interface contract. Five stages with one carry-forward LoRA adapter. Critical path identified as rubric calibration. Translation layer marked obsolete after Stage 1.
