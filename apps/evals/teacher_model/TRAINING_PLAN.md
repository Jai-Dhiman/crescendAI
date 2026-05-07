# Teacher Model Training Plan -- Qwen3.6-35B-A3B

**Status:** design draft (2026-05-05). Supersedes the prior 4-stage plan in `docs/superpowers/specs/2026-03-30-teacher-model-finetuning-design.md`.
**Origin:** brainstorm session 2026-05-05 during CPT-corpus dataset design; cross-referenced with two parallel brainstorms (translation layer, domain probe).

---

## Context: why the plan is changing

The original plan (CPT + SFT + GRPO + DPO) was designed for **Qwen3.5-27B dense** in March 2026. On 2026-05-05 the target model switched to **Qwen3.6-35B-A3B** (sparse MoE, 35B total / 3B active per token, 256 experts, 248K vocab, 262K native context, multimodal-capable, Apache 2.0, released April 2026).

That switch invalidates the canonical 4-stage plan, not because the stages are individually wrong, but because:
1. A 35B-class MoE has absorbed enormous pedagogy text in base training. CPT's marginal value-add is much smaller than for a 27B dense base.
2. MoE has training failure modes that don't exist in dense models: **expert collapse**, **load-balance drift**, **routing skew across stages**.
3. GRPO on MoE is unstable -- KL divergence drifts unpredictably across experts.
4. The 4-stage plan was over-engineered for the 27B dense world; on 35B-A3B it compounds risk without compounding capability.

This document captures the corrected plan: **what the teacher model is trying to be, what the base model already gives us for free, and what training stages are actually needed.**

---

## What the teacher model must demonstrate (5 capabilities)

1. **Pedagogical judgment** -- when to correct vs. encourage, when to challenge vs. support
2. **Bar-specific corrective feedback** grounded in audio observations (current weakest dim: ASCF outcome 1.387 / 3.0)
3. **Deep piano-pedagogy knowledge** -- fingering systems, voicing, technique schools, repertoire tradition
4. **Warmth + consistent voice** across sessions
5. **Independence from Anthropic** (structural moat)

---

## Three layers of "knowledge" -- decomposed

Critical insight for designing training stages on a 35B-A3B base:

### Layer 1 -- General piano knowledge (LATENT in base model)

Almost certainly already present in base 35B-A3B from pretraining:
- Technique schools (Russian / Leschetizky / Taubman)
- Repertoire eras and tradition
- ABRSM / RCM grade structure
- Hanon / Czerny / Pischna technique exercises
- Generic "if student struggles with X, try Y" mappings
- Pedagogy framing (correction/encouragement balance, learning-arc awareness)

**Implication:** no CPT needed for Layer 1 unless Phase 0 baseline proves otherwise.

### Layer 2 -- Pedagogical mapping (PARTIALLY LATENT)

Patterns like "uneven 16ths in m. 12 of Mozart K545 + history of 4th-finger weakness -> recommend slow-practice + Pischna pattern N" require both general knowledge AND grounding in observed signals. Base model can do half of this with strong prompting + few-shot. The gap is bridged by SFT, not CPT.

### Layer 3 -- System-specific knowledge (NOT LATENT, never will be from text)

- Your exercise library schema and IDs
- Your Verovio MEI snippets
- Your synthesis pipeline output structures (`AudibleObservation`, score-following bar analysis, 6-dim signals)
- Your tool palette in `services/tool-processor.ts`

**Implication:** Layer 3 is a *grounding* problem, not a knowledge-injection problem. No amount of pedagogy corpus will teach exercise IDs. This is taught via tool-calling SFT and tool-use protocol.

---

## Verovio / structured exercise output -- tool calling, not generation

The model **does not** generate Verovio MEI directly. MEI generation has hard validity constraints (XML well-formed, MEI schema valid, musically sensible, renders correctly). One bad attribute and Verovio throws.

Pattern:
- Model calls `recommend_exercise(exercise_id, rationale, context)` -- *selects* from your library
- Backend deterministically looks up the exercise's pre-authored MEI and renders it
- Model never sees or generates MEI

This means:
- **No MEI in CPT corpus** -- wasted tokens
- **No MEI in SFT data** -- same reason
- **Training signal is "given audio observations + student context, call the right tool with the right arguments"** -- pure tool-use SFT, ~5K-20K examples

Generating *novel* exercises from scratch is a separate, much harder problem and explicitly out of scope for v1. Stick with the curated library + tool selection.

---

## The corrected training plan -- 2 stages, 1 contingent

### Phase 0 -- Capability mapping (no training compute, ~1-2 days)

Establish what base 35B-A3B can already do across all 5 capabilities:
- Run domain probe (Agent 2's existing 50-Q MCQ) against base
- Run free-form synthesis eval (8-dim rubric) on a held-out set against base
- Run tool-calling capability test (system prompt + few-shot, can it call 5-tool palette correctly?)

For each capability: at-ceiling, mid-tier, or absent.

**This phase is the gating step.** Phases 2-4 are scoped from these results.

### Phase 1 -- Tool-calling format alignment (universally needed)

Regardless of Phase 0 outcomes, the model must natively emit your tool format. Two options:
- **Inference-only:** system prompt + few-shot, ~80-90% tool-call success rate
- **Tool-format SFT:** ~2K examples, ~99% success, removes system-prompt token overhead

**Recommended:** Tool-format SFT. Tiny, cheap, foundation everything else stacks on. **Sidenote:** if this lands well, it eliminates the need for a TS-side translation layer (Agent 1's task A) -- the model speaks Anthropic-compatible tool_use natively.

### Phase 2 -- Capability SFT (do *only* what Phase 0 says is missing)

Build instruction-formatted SFT data targeting specific gaps. Likely candidates:
- **Bar-specific corrective feedback** (your weakest dim). Format: `(audio_observation_summary, score_position, student_history) -> (bar_specific_critique + tool_call_to_recommend_exercise)`. ~5K-20K examples synthesized from Sonnet + curated YouTube transcripts.
- **Voice consistency across session length** (warmth + format)
- **Pedagogical judgment** (correct vs. encourage decisions)

**Critical:** SFT on instruction-formatted pedagogy gives knowledge + format + voice in *one* stage. The 100M-token corpus lives here as **source material from which SFT examples are synthesized**, not as direct training data.

### Phase 3 -- DPO for warmth/judgment polish (NOT GRPO)

Pairwise preferences (~5K-20K pairs), human-collected on chosen vs. rejected responses.

**Skip GRPO entirely.** Reasons:
- GRPO needs verifiable reward; 8-dim rubric is noisy and rewards subjective judgments -> RL on noisy reward = reward hacking
- GRPO on MoE is unstable; KL drifts unpredictably across experts
- DPO is simpler, MoE-stable, gives same outcome (preference alignment) at less risk
- 2026 production narrow-domain models (medical, legal, financial) almost universally use SFT+DPO, not SFT+RLHF

### Phase 4 -- CPT (contingent only, likely skipped)

Run only if Phase 0 + Phase 2 reveals genuine knowledge gaps. Format as instruction-CPT (completion examples), not raw-text CPT, to keep MoE routing balanced.

**Most likely:** not needed. Base model + capability SFT closes the gaps.

---

## What this means for the three current workstreams

| Workstream | Status under corrected plan |
|---|---|
| **CPT-corpus pipeline (this brainstorm)** | Pivots from "CPT-ready dataset" to **"SFT data synthesis source corpus."** Same cleaning/dedup work; downstream consumer changes. The 100M-token clean dataset becomes input to a Sonnet-driven SFT-data-generation pipeline. **Still ships as v1 dataset** -- load-bearing infra regardless of training shape. |
| **Agent 1 (translation layer)** | **Likely obsolete after Phase 1.** Tool-format SFT teaches the model to natively emit Anthropic-compatible tool_use. Layer becomes a stopgap for the pre-finetune period only. |
| **Agent 2 (domain probe)** | **Promoted to Phase 0 anchor.** Probe + free-form synthesis eval on base model becomes the gating step that determines what Phases 2-4 do. Reframe (delta-gate, regression set, routing telemetry) stays. |

---

## MoE-specific training considerations (cross-cutting)

- **Tooling:** Unsloth / Llama-Factory / Swift over bare TRL -- they handle load-balance aux loss and expert-routing telemetry.
- **LoRA targets:** Unsloth Qwen3.5 default is `q/k/v/o + gate/up/down`. LoRA-ing all 256 experts re-densifies the model; LoRA-ing none misses where domain knowledge lives. The `gate/up/down` targets touch the shared expert + router gates, which is the MoE-aware compromise.
- **Expert-entropy monitoring:** log per-token expert entropy at every training stage, not just probe. If entropy collapses below threshold mid-training, abort and re-architect.
- **Sequence length:** 4096 for SFT/DPO. 262K native context is overkill for pedagogy texts and amplifies routing imbalance.
- **Same LoRA carried forward:** if multiple stages, carry the same adapter across them rather than full-FT each stage. Reduces compounding routing drift.

---

## Open questions (still to resolve)

1. **CPT-vs-no-CPT definitive answer:** pending Phase 0 baseline. If base 35B-A3B clears 75%+ on existing probe AND scores >2.0 on free-form synthesis eval, CPT is almost certainly not needed.
2. **Tool-format details for Qwen3.6:** model card mentioned `qwen3_coder` parser; need to verify chat-template format vs. Qwen3.5 before locking Phase 1 SFT data format.
3. **DPO data source:** human-collected pairwise preferences vs. rubric-judged synthetic pairs vs. hybrid. Hybrid likely best (human for top-tier, synthetic for bulk), but quantity/quality tradeoff needs sizing.
4. **Rubric lock:** the 8-dim rubric is the training target for both Phase 2 (SFT-data quality filter) and Phase 3 (DPO preference signal). Human calibration must complete before Phase 2 begins. This is the **highest-leverage and lowest-attention item** across the whole plan.

---

## Strategic gates (revised)

Original plan had 4 gates: PMF + A/B + rubric + 7B-probe. Updated:

1. **Phase 0 baseline complete** (was: 7B probe pass) -- gates whether CPT phase 4 is even needed
2. **Rubric human calibration r >= 0.7** (unchanged) -- gates Phase 2 and Phase 3
3. **Blind A/B voice test vs. Sonnet** (unchanged) -- gates production deployment
4. **200+ beta users with retention** (unchanged) -- gates serious compute commitment

Compute envelope is dramatically smaller: SFT (~2K + 5K-20K examples) + DPO (~5K-20K pairs) on 35B-A3B fits on 1-2 nodes, not the 4-8 node spread the original 4-stage plan assumed.

---

## References

- `apps/evals/teacher_model/domain_knowledge_probe.py` -- existing probe (Phase 0 anchor)
- `apps/evals/teacher_model/data/corpus/` -- 100M-token raw corpus (SFT source)
- `apps/evals/teacher_model/cpt_pipeline/` -- deterministic preprocessing pipeline (ingest → filter → dedup → split → HF publish); run via `uv run python -m teacher_model.cpt_pipeline.pipeline run --corpus-dir ... --provenance-dir ... --out-dir ... --repo-id Jai-D/Crescendai-piano-pedagogy-cpt-v1`
- `apps/api/src/services/tool-processor.ts` -- 5-tool palette (Phase 1 target format)
- `apps/api/src/services/teacher.ts` -- unified teacher service (deployment target)
- Memory: `project_teacher_model_finetuning.md`, `project_beta_priorities.md`, `project_teaching_knowledge_eval.md`
