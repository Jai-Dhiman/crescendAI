# Teacher Model Finetuning Design

**Date:** 2026-03-30
**Status:** Reviewed (/autoplan). Pending implementation plan.
**Timeline:** 6 months, gated (data collection immediate, training gated on 4 conditions)
**Approach:** "The Prodigy" -- CPT + SFT + GRPO + DPO

## Motivation

Voice ownership. The teacher personality should live in model weights, not a rented system prompt. This provides:

1. A structural moat that competitors cannot replicate by using the same API
2. Independence from Anthropic model update risks (tone shifts, safety filter changes)
3. Deeper domain knowledge than a generalist model with a system prompt
4. Cost savings at scale (10x cheaper past 2K DAU)

Claude continues to serve production until the finetuned model clears a hard quality gate on the 7-dimension teaching rubric.

## Strategic Gates (from /autoplan CEO review)

Data collection starts immediately. Training is gated on these 4 conditions:

1. **PMF Signal:** Web beta has 200+ active users with 2+ sessions each and a measurable 30-day retention curve
2. **Voice A/B Test:** Blind test confirms users can distinguish finetuned voice quality from Claude-with-current-prompt (if they cannot, the voice moat premise is falsified)
3. **Rubric Validation:** Judge v2 rubric validated against 50-100 human ratings from piano students/teachers with Pearson r >= 0.7 per dimension
4. **CPT Probe:** Continual pretraining on Qwen3.5-7B with 20M tokens confirms knowledge injection (>= 60% on domain knowledge probe). Cost: ~$15, 2 hours.

Gates 1-2 require beta launch. Gates 3-4 can run during data collection phase. If any gate fails, reassess before committing $700 to the 27B CPT run.

## Must-Fix Before Implementation (from /autoplan Eng review)

These 3 items must be resolved before any training code is written:

1. **Qwen tool format translation:** Write a spec of Qwen's tool calling format in ChatML. Build a serializer that translates `create_exercise` from Anthropic schema to Qwen/OpenAI format. Test round-trip with 50 examples. Wrong format in SFT corrupts the entire downstream pipeline.
2. **Judge v2 scoring rule fix:** Change "absent dimension = 2" to "absent dimension = N/A (excluded from mean)" in `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`. Current rule lets GRPO game the average by hyper-optimizing 2-3 dimensions and omitting the rest.
3. **Relevance classifier specification:** Define architecture (e.g., sentence-transformers cosine similarity), curate 200-400 negative examples from same source domains, hold out 20% for validation, publish precision/recall curve before building the 100M token corpus.

## Competitive Risk

The durable moat is audio-based evaluation (Model v2: MuQ + Aria), not teacher voice. Voice ownership is valuable but not the primary differentiator. Competitors (Tonebase, Simply Piano/JoyTunes) could run a similar CPT pipeline with their own data. The proprietary session data flywheel from real students is a second-order moat that only compounds with scale.

## Legal Prerequisite

Before the data pipeline starts: $300-500 IP attorney consultation on YouTube ToS implications for model training, the specific scraping targets listed in the corpus table, and whether "research" characterization applies to a commercial product. YouTube "Free" is an access mechanism, not a legal status.

## Base Model

**Qwen3.5-27B (dense)**

- Apache 2.0, no commercial restrictions
- Best-in-class tool calling (BFCL-V4: 72.2)
- Dense architecture for stable CPT knowledge injection (vs MoE which concentrates knowledge in routing-dependent expert subsets)
- 27B fits on a single A100/H100 for inference
- Rich finetuning ecosystem (Unsloth, TRL, Axolotl)

Ruled out: Llama 4 (MoE, Meta license), DeepSeek (671B too large), Gemma 3 27B (close second but weaker tool calling, Gemma license vs Apache 2.0), Qwen3.5-122B-A10B (MoE, multi-GPU serving).

## Data Pipeline

### Transcription Engine

**Cohere Transcribe** (`cohere-transcribe-03-2026`) + **Pyannote speaker-diarization-3.1**

- Cohere: Apache 2.0, 2B params, 5.42% WER (#1 Open ASR Leaderboard), 525x real-time throughput
- Pyannote: Speaker diarization to separate teacher from student in masterclass audio
- Pipeline: `yt-dlp (audio) -> Cohere Transcribe -> Pyannote diarization -> merge -> quality filter -> clean`
- 3,000 hours of masterclasses transcribed in ~6 hours on a consumer GPU

### Corpus (Target: 100M base, 200M+ goal)

#### Tier 1: Masterclass Video Transcriptions (~80-120M tokens)

| Source | Content | Volume | Access |
|--------|---------|--------|--------|
| tonebase Piano | Expert masterclasses, Juilliard/Yale faculty | 603 videos, ~170 hrs | Free (YouTube) |
| Josh Wright | ProPractice technique series | 877 videos | Free (YouTube) |
| Nahre Sol | Theory + stylistic pedagogy | 350 videos | Free (YouTube) |
| Juilliard YouTube | Masterclasses (Perahia, Schiff) | 924 videos (subset) | Free (YouTube) |
| Festival channels | Verbier, Gilmore, Ruhr masterclasses | Hundreds of videos | Free (YouTube) |
| YouTube long tail | "piano masterclass", "piano lesson" searches | Thousands of videos | Free (YouTube) |
| Piano pedagogy podcasts | ~100+ episodes across multiple shows | ~50-100 hrs | Free (RSS + transcribe) |

Key YouTube channels verified: tonebase Piano (250K subs), Josh Wright (203K), Nahre Sol (827K), Juilliard (98.5K), UIPianoPed (~3,000 performance demos).

#### Tier 2: Pedagogy Literature (~40-60M tokens)

| Source | Content | Access |
|--------|---------|--------|
| Piano Pedagogy Forum | 22 volumes (1998-2021), open-access journal | Free PDF download |
| PQDT Open dissertations | Hundreds of 100-400pp dissertations on piano pedagogy | Free full text |
| ERIC open-access articles | 200-400 music education research papers | Free |
| Public domain books | Matthay (Act of Touch, First Principles, Piano Fallacies), Czerny (Letters to a Young Lady), C.P.E. Bach (Essay on True Art), Brower (Piano Mastery 2 vols), Jonas (Master School 7 vols), Leschetizky method (Bree, Prentner) | Internet Archive / Gutenberg |
| IMSLP pedagogy texts | Clementi, Schirmer Teacher's Guide, method books | Free PDF |
| RCM Piano Syllabus (2022) | Graded technical requirements, 10 levels | Free PDF |
| Bulletproof Musician | 600+ articles on practice psychology, deliberate practice | Free (web scrape) |

#### Tier 3: Performance Practice & Musicology (~30-40M tokens)

| Source | Content | Access |
|--------|---------|--------|
| Henle Urtext prefaces | Performance practice commentary per piece | Free (web scrape) |
| Chopin Review journal | Open-access performance style scholarship | Free |
| Yale OHAM transcripts | Oral histories from major pianists | Free PDF on request |
| MIT Music Oral History | Searchable interview transcripts | Free download |
| ISME conference proceedings | Piano pedagogy research papers | Free PDF |
| Piano biomechanics literature | PMC open-access papers on technique | Free |

#### Tier 4: CrescendAI Own Data (~5-10M tokens, growing)

| Source | Content | Access |
|--------|---------|--------|
| 73 teaching transcripts | Analyzed pedagogy research | Already have |
| 379 teaching moments | Extracted + classified | Already have |
| Playbook (5 clusters) | Product-grounded pedagogy principles | Already have |
| PercePiano text annotations | Expert performance quality descriptions | Already have |
| PIAST dataset | 9,673 tracks with text descriptions | GitHub download |
| Real session data | Accumulates during web beta (with consent) | Growing |

#### Optional Licensed Sources

| Source | What | Cost |
|--------|------|------|
| Gramophone archive | 50K+ reviews since 1923 | Institutional subscription |
| Pianist Magazine | 15-year how-to-play archive | Exact Editions subscription |
| Piano Inspires / Clavier Companion | Piano teaching periodical since 1991 | Institutional subscription |

### Data Quality Pipeline

```
Source -> Extract (Cohere Transcribe / PDF parse / HTTP scrape)
       -> Diarize (Pyannote, for audio sources)
       -> Language filter (English only v1)
       -> Relevance classifier (seeded from 379 teaching moments, threshold 0.3)
       -> Dedup (MinHash + LSH)
       -> Segment (paragraph/topic boundaries)
       -> Clean (remove ads, sponsorship, off-topic)
       -> Tag (source type, topic, difficulty level)
```

Relevance classifier: Binary classifier (e.g., sentence-transformers/all-MiniLM-L6-v2 cosine similarity against centroid) trained on 379 positive examples + 200-400 curated negatives from same source domains. Hold out 20% for validation. Publish precision/recall curve with chosen operating point. Threshold ~0.3 (adjust per validation results). Ensures CPT corpus is concentrated on actual piano teaching, not tangential content.

Corpus provenance: Every document tracked in a provenance manifest (`corpus_provenance.jsonl`): `{url, title, channel_or_publisher, download_timestamp, license_claimed, word_count, inclusion_threshold_score}`. Required for legal/compliance audits.

Transcription validation: Before full 3,000-hour pipeline, validate on 10-video sample per source channel: (a) Cohere Transcribe WER on 1-hour masterclass sample (target < 15%), (b) Pyannote diarization accuracy on piano masterclass audio (target < 20% speaker confusion). If thresholds not met, evaluate domain-adapted alternatives.

## Training Pipeline

### Stage 1: Continual Pretraining (CPT)

**Goal:** Inject piano pedagogy knowledge into model weights.

Staged approach (inspired by Cursor Composer 2 CPT design):
- **Stage 1a - Bulk training:** Full 200M token corpus (80% pedagogy + 20% SlimPajama CommonCrawl/Wikipedia, randomly interleaved). Standard next-token prediction. Builds broad domain knowledge.
- **Stage 1b - Quality narrowing:** Curated high-quality subset only (30-50M tokens: masterclass transcripts + classic pedagogy texts + PercePiano annotations). Higher signal-to-noise ratio narrows knowledge to the most valuable material.
- **Stage 1c - Brief SFT alignment:** Small set (~500 examples) of instruction-response pairs to re-stabilize instruction following before full SFT. Prevents CPT from drifting too far from chat format.

Training config:
- Full-parameter training (not LoRA) for maximum knowledge absorption
- Learning rate: Cosine schedule from 2e-5 warmup to 1e-5 final
- Epochs: 1 pass per stage (no data reuse, following Cursor's single-epoch finding)
- Checkpointing: Every 10% of training. Run domain knowledge probe at each checkpoint.
- Compute: ~8-16 hours on 4xH100, ~$400-700. Budget 2-3x for restarts.
- Test staged vs monolithic approach in 7B probe (Gate 4)

**CPT Gate:** Post-CPT perplexity on held-out pedagogy corpus must decrease. Perplexity on 1M-token SlimPajama general slice must not increase by more than 5%. Domain knowledge probe (100 piano pedagogy QA pairs) must score >= 60%. If gate fails: reduce LR to 5e-6 and re-run. If still failing, increase SlimPajama ratio to 30%. If still failing after 2 attempts, re-evaluate base model choice.

**Risk:** Catastrophic forgetting of conversational ability. Mitigated by the 80/20 interleaved data mix and cosine LR schedule.

### Stage 2: Supervised Fine-Tuning (SFT)

**Goal:** Teach the CrescendAI teacher format, voice, and tool usage.

Data (2-5K examples):

| Type | Source | Count |
|------|--------|-------|
| Observation responses | Teaching moments -> Claude Sonnet-generated gold responses + 100-200 human-written anchors from masterclass transcripts | ~1,000 |
| Chat conversations | Multi-turn dialogues: all 6 dimensions, all skill levels, all repertoire periods | ~1,000 |
| Tool use (exercises) | Scenarios requiring `create_exercise` tool | ~500 |
| Elaborations | "Tell me more" follow-ups with musical depth | ~300 |
| Edge cases | Frustration, plateaus, wrong piece, first session | ~200 |

Format: ChatML / Qwen chat template with system prompt, user context, assistant response. Tool calls in Qwen native format.

Training: QLoRA (rank 32-64) with Unsloth. Target all linear projections including MLP (gate_proj, up_proj, down_proj), not just attention. ~2-4 hours on 1xH100. ~$10-30. Tool calls formatted in Qwen native ChatML format (translation layer must be built first, see Must-Fix section).

**SFT Gate:** Tool call reliability must be >= 90% on 100 tool call test scenarios. If < 80%, add 200 more tool examples and re-run SFT only.

The SFT stage bakes in current system prompt rules: 1-3 sentences, no scores, no lists, specific musical references, actionable corrections, no markdown in observations.

**Environment-production parity (Composer 2 lesson):** SFT training data must use the exact production prompt templates serialized from `prompts.rs` (subagent system prompt, teacher system prompt, tool schema) in Qwen ChatML format. Training on approximations of the production prompts causes distribution mismatch at serving time.

### Stage 3: GRPO (Group Relative Policy Optimization)

**Goal:** Teach pedagogical judgment using the existing eval rubric as reward.

How it works:
1. Present a teaching scenario (student context + scores + piece)
2. Model generates N=4-8 candidate responses
3. Judge v2 (7-dimension rubric) scores each response
4. GRPO updates the model toward higher-scored responses

Reward signal (8 dimensions from `apps/evals/teaching_knowledge/`):
- Musical Accuracy
- Pedagogical Appropriateness
- Specificity & Actionability
- Student Awareness
- Style Consistency
- Observation Pacing
- Tool Use Appropriateness
- Tool Invocation Correctness (8th dimension: binary, 0/1 on scenarios where tool call is warranted)

Scoring rule: Absent dimensions scored as N/A (excluded from mean), NOT as 2. This prevents reward hacking via selective omission.

**Verifiable anti-pattern penalties (regex-checkable, no LLM judge needed):**
- Response contains numeric scores/ratings -> penalty
- Response contains markdown formatting (headers, bullet lists) -> penalty
- Response uses emojis -> penalty
- Response exceeds 500 characters -> graduated nonlinear penalty
- Tool call created but never referenced in observation text -> penalty
- Response contains "great job" or equivalent empty praise -> penalty

These complement the LLM-judged rubric dimensions with fast, deterministic signals.

**Nonlinear length penalty (from Cursor Composer 2):** Concave-down curve. On simple teaching scenarios (single-dimension correction), extra tokens are heavily penalized -- the teacher should be brief. On complex scenarios (multi-dimension, exercise creation, elaboration), the model is allowed to generate longer responses without disproportionate penalty. Calibrate the curve using the 1-3 sentence target for observations and the ~500 char max from the current production system.

**GRPO Gate:** If any rubric dimension drops > 0.5 points below SFT baseline, revert to SFT checkpoint and adjust GRPO reward weights. Checkpoint tool calling reliability every 100 GRPO steps.

Why GRPO: No critic model needed. Uses group-relative ranking. Proven by DeepSeek R1 and Cursor Composer 2 for subjective quality optimization. Simpler and cheaper than PPO/RLHF.

**GRPO config (Composer 2 lessons applied):**
- **Single-epoch regime:** Never train on the same prompt twice. Generate 1,500+ unique scenarios to ensure single-pass coverage. Prevents overfitting and maintains response diversity.
- **No advantage normalization by std:** Cursor found std normalization "massively upweighted tiny behavioral differences in groups where every rollout was equally correct." When our judge v2 scores cluster around 2.0-2.5, small score differences would get amplified. Skip std normalization.
- **No length standardization:** Rely on the nonlinear length penalty instead. Length standardization introduces bias toward a fixed response length, but our teacher needs variable length (1 sentence for recognition, 3 sentences + exercise for correction).

Training: 1,500 unique scenarios, 4-8 generations each, single epoch. ~4-8 hours on 1xH100 with TRL. ~$20-50.

**Risk:** Judge bias propagation. If the judge systematically favors corrections over encouragement, the model will too. Audit judge v2 for dimension-level bias before GRPO. Current smoke test scored 2.57/3.0.

### Stage 4: DPO (Direct Preference Optimization)

**Goal:** Final polish on voice warmth, tone, and subjective feel.

Data (500-1K preference pairs):
- Best vs worst GRPO outputs for each scenario (automated)
- Pre-GRPO SFT checkpoint outputs as explicit low-quality negatives (larger preference gap for stronger DPO gradient)
- Hand-curated pairs where human judges warmth/tone (the rubric catches quality, you catch feel)
- Anti-pattern pairs: generic responses, lecture-y responses, responses with scores/ratings, emoji, markdown in observations, tool call schema errors

Training: QLoRA with TRL DPOTrainer. ~2-4 hours on 1xH100. ~$10-30.

**DPO Gate:** If human preference for DPO model < 60% vs SFT+GRPO checkpoint, discard DPO and ship SFT+GRPO only.

### Compute Budget Summary

| Stage | Hardware | Duration | Cost |
|-------|----------|----------|------|
| CPT | 4xH100 (RunPod/Lambda) | 8-16 hrs | $400-700 |
| SFT | 1xH100 | 2-4 hrs | $10-30 |
| GRPO | 1xH100 | 4-8 hrs | $20-50 |
| DPO | 1xH100 | 2-4 hrs | $10-30 |
| **Total** | | **16-32 hrs** | **$440-810** |

## Hosting & Serving

### Phase 1: Serverless Validation (months 5-6)

Serve the finetuned model via Together.ai or Fireworks.ai:
- Together.ai: ~$0.20-0.50/M tokens, custom model upload
- Fireworks.ai: ~$0.50/M tokens, sub-100ms TTFT, LoRA at base cost
- Zero infrastructure management
- Integration: Add a third AI Gateway route (`crescendai-teacher-custom`) pointing to Together/Fireworks via OpenAI-compatible API

### Phase 2: Self-Hosted (post-validation, when volume justifies)

When daily volume exceeds ~2M tokens/day:
- vLLM on RunPod ($1.49/hr A100) or Modal (scale-to-zero, per-second billing)
- Qwen3.5-27B AWQ 4-bit quantization: single A100/H100
- Sub-500ms latency with speculative decoding (vs current 1.5s from Anthropic)

### Subagent Architecture

Keep the two-stage split initially:
- Groq subagent (structured reasoning, $0.50-0.60/M tokens) stays unchanged
- Custom teacher model replaces Anthropic for voice generation
- Evaluate collapsing to single model after GRPO training proves pedagogical reasoning quality

### Cost at Scale

| DAU | Claude Sonnet 4.6 | Together.ai Serverless | Self-Hosted (A100 24/7) |
|-----|-------------------|----------------------|------------------------|
| 100 | ~$270/mo | ~$10-15/mo | $1,100/mo (overkill) |
| 1,000 | ~$2,700/mo | ~$100-150/mo | $1,100/mo |
| 10,000 | ~$27,000/mo | ~$1,000-1,500/mo | $1,100/mo (add GPUs) |

## Evaluation Strategy

### Quality Gate (non-negotiable)

The finetuned model replaces Claude **only** when it wins on the 7-dimension teaching rubric across a representative eval set. Specifically:

1. Run the finetuned model and Claude on identical scenarios from the teaching knowledge eval
2. Judge v2 scores both blindly
3. Finetuned model must match or exceed Claude's average rubric score
4. No single dimension can regress by more than 0.3 points (on 0-3 scale)
5. Tool use reliability (exercise creation) must be >= 95%

### Required Eval Infrastructure (build during data collection phase)

1. **Domain Knowledge Probe** (`apps/evals/model/domain_knowledge_probe.py`): 100 factual MCQ questions from the pedagogy literature. Score pre-CPT, post-CPT, post-SFT. CPT model must score >= 60%.
2. **Tool Calling Regression Tests** (`apps/evals/shared/judge.py` -> `judge_tool_calls()`): 100 scenarios (50 tool-required, 50 tool-not-required). Check: tool call present, name matches, JSON validates against schema.
3. **Perplexity Proxy** (for CPT gate): Compute perplexity on held-out pedagogy slice (must decrease) and general text slice (must not increase > 5%).
4. **Latency Benchmark**: P50 TTFT <= 800ms, P95 TTFT <= 2000ms, measured under 10 concurrent requests on Together.ai.

### Ongoing Evaluation

- A/B test with real students before full cutover
- Monitor per-dimension scores weekly
- Track tool use failure rate
- Compare student engagement metrics (session length, return rate) between Claude and finetuned model

### Data Consent (Tier 4, real session data)

Before collecting any real session data for training:
- Opt-in consent during signup ("your sessions may improve the AI teacher")
- Automatic PII scrubbing before writing to training corpus
- Deletion endpoint for consent withdrawal
- Required for GDPR compliance

### Model Weight Security

- All intermediate checkpoints in private HF repo under crescendai org
- HF fine-grained tokens: write-only for upload, read-only for inference
- Together.ai API keys in Cloudflare Workers secrets (same mechanism as ANTHROPIC_API_KEY)

## Timeline (Gated)

| Month | Data Pipeline | Gates & Eval | Training |
|-------|--------------|-------------|----------|
| 1 | IP attorney consultation. Build scraping/transcription pipeline. Validate Cohere+Pyannote on 10-video sample. Start YouTube transcription (tonebase, Josh Wright, Nahre Sol). Download public domain books. | Build relevance classifier (with negatives). Build domain knowledge probe. Build tool calling test harness. | Write Qwen tool format spec + translator. Fix judge v2 absent=2 scoring rule. |
| 2 | Continue YouTube long tail. Parse PDFs (Piano Pedagogy Forum, PQDT dissertations). Scrape Bulletproof Musician. Build provenance manifest. | **Gate 3 (rubric validation):** Run judge v2 on 73 transcripts, validate against human ratings. Get human sign-off on 7-dim rubric. **Gate 4 (CPT probe):** Run 7B CPT on 20M tokens, measure domain knowledge probe. | -- |
| 3 | Complete Tier 1-3 collection. Quality filtering + dedup. Assemble CPT corpus. | Check **Gate 1 (PMF):** 200+ active beta users? Check **Gate 2 (A/B):** Design blind voice quality test. | If all 4 gates pass: begin 27B CPT. If gates fail: reassess. |
| 4 | Continue accumulating. Curate SFT dataset (2-5K examples, incl. human anchors). | Run CPT gate (perplexity + domain probe). Per-stage tool calling eval. | CPT complete (if started). Begin SFT. |
| 5 | Generate GRPO scenarios. Create DPO preference pairs. | GRPO reward monitoring. Per-stage checkpoints. | SFT complete. GRPO training. DPO polish. |
| 6 | Collect real session data from beta (with consent). | Quality gate eval. Latency benchmark. | Upload to Together.ai. A/B test. Conditional production switch. |

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Quality regression vs Claude | High | Hard quality gate on 7-dim rubric. Claude stays as fallback. |
| Catastrophic forgetting during CPT | Medium | 80/20 data mix, conservative LR, MT-Bench before/after. |
| Tool use reliability | Medium | Extra GRPO iterations on tool scenarios. 95% threshold. |
| Judge bias in GRPO | Medium | Audit judge v2 per-dimension before training. |
| Engineering time (180-270 hrs) | Medium | Automate data pipeline. Training infra reusable for Model v2. |
| Base model obsolescence | Low | Qwen ecosystem active. Re-CPT on Qwen3.6+ if needed. |
| Legal risk from training data | Low | All Tier 1-3 sources are public/open-access. No copyrighted ebooks. Forum data used for research only. |

## Opportunity Cost

180-270 hours over 6 months. This is time not spent on:
- Web beta features (free tier gating, observation pacing, polish)
- iOS app (follows web beta)
- Model v2 training (contrastive pretraining, LoRA fine-tuning)
- User acquisition and marketing

Counterbalance: The data pipeline and training infrastructure are reusable. ML engineering skills compound. The data corpus itself is a permanent asset.

## Decision: When to NOT Do This

Abandon or defer if:
- Web beta fails to find product-market fit (the teacher voice doesn't matter if nobody uses the product)
- Claude Haiku 4.5 proves good enough at $450/mo for 1K DAU (cost motivation disappears)
- Anthropic releases a model fine-tuning API that gives you voice ownership without the engineering burden
- The quality gate is not met after 2 training iterations (signals the 27B model cannot match Claude for this task)

<!-- /autoplan restore point: /Users/jdhiman/.gstack/projects/Jai-Dhiman-crescendAI/main-autoplan-restore-20260331-014113.md -->

## /autoplan Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale |
|---|-------|----------|---------------|-----------|-----------|
| 1 | CEO | Restructure with 4 strategic gates | USER DECISION | N/A | User chose gated approach over fixed 6-month timeline |
| 2 | CEO | Add A/B blind test before training | Mechanical | P1 | Validates core premise at zero cost |
| 3 | CEO | Add RAG comparison to alternatives | Taste | P3 | RAG vs CPT is a close call; CPT has higher ceiling but RAG is reversible |
| 4 | CEO | Add rubric validation vs human preference | Mechanical | P1 | Prevents GRPO from optimizing for wrong signal |
| 5 | CEO | Add 7B CPT probe before 27B run | Mechanical | P3 | $15 experiment saves potential $700 waste |
| 6 | CEO | Add per-stage tool calling gates | Mechanical | P1 | 85% after CPT, 90% after SFT, 95% final |
| 7 | CEO | Add IP attorney consultation ($300-500) | Mechanical | P5 | YouTube ToS risk is real, not theoretical |
| 8 | CEO | Add competitive risk section | Mechanical | P3 | Audio eval is the durable moat, not voice |
| 9 | Eng | Define relevance classifier architecture | Mechanical | P1 | 379 positives + no negatives = noisy corpus |
| 10 | Eng | Formalize CPT gate (perplexity proxy) | Mechanical | P5 | MT-Bench requires serving; perplexity is cheaper |
| 11 | Eng | Add human-written SFT anchors (100-200) | Taste | P1 | Breaks circularity of Claude generating Claude's replacement data |
| 12 | Eng | Add tool calling as 8th GRPO dimension | Mechanical | P1 | Silent regression path otherwise |
| 13 | Eng | Specify 80/20 mix interleaving + LR schedule | Mechanical | P5 | Underspecified = non-reproducible |
| 14 | Eng | Write Qwen tool format translation layer | Mechanical | P5 | Must-fix-first: wrong schema corrupts entire pipeline |
| 15 | Eng | Fix absent=2 scoring rule in judge v2 | Mechanical | P5 | Must-fix-first: highest-probability reward hack |
| 16 | Eng | Include explicit low-quality DPO negatives | Mechanical | P3 | GRPO outputs too similar for strong DPO gradient |
| 17 | Eng | Add rollback decision trees at each stage | Mechanical | P1 | No rollback = no recovery from bad checkpoints |
| 18 | Eng | Build domain knowledge probe (100 QA pairs) | Mechanical | P1 | Only way to verify CPT worked |
| 19 | Eng | Add tool calling regression test harness | Mechanical | P1 | 95% threshold needs automated measurement |
| 20 | Eng | Define latency SLA (P50 <= 800ms, P95 <= 2s) | Mechanical | P5 | Production cutover needs measurable threshold |
| 21 | Eng | Build corpus provenance manifest | Mechanical | P1 | Legal compliance requires audit trail |
| 22 | Eng | Specify model weight access control | Mechanical | P5 | Private HF repo + scoped tokens |
| 23 | Eng | Specify consent + anonymization for Tier 4 | Mechanical | P1 | GDPR requirement before collecting session data |
| 24 | Eng | Validate Pyannote on 10-video sample first | Mechanical | P3 | Piano masterclasses have unique audio challenges |
| 25 | Eng | Spot-check Cohere WER on 1hr masterclass | Mechanical | P3 | Benchmark WER != real-world WER |
| 26 | Eng | Specify LoRA target_modules (include MLP) | Mechanical | P5 | MLP exclusion risks tool calling ability |
| 27 | Eng | Get human sign-off on 7-dim rubric | Mechanical | P1 | LLM-generated rubric needs domain expert review |

## Cross-Phase Themes

**Theme 1: Rubric reliability** -- flagged in CEO (Finding 4: circular dependency) and Eng (Findings E-3, T-5, H-5). The 7-dimension rubric is simultaneously the GRPO reward signal, the quality gate, and the eval metric. It was LLM-generated from LLM-extracted teaching moments. Three independent concerns converge: it may not correlate with student preference (CEO), it has a gameable scoring rule (Eng), and it needs human validation (Eng). High-confidence signal that rubric hardening is a prerequisite for training.

**Theme 2: Tool calling fragility** -- flagged in CEO (Finding 6) and Eng (Findings A-4, E-2, T-3, H-3). Tool calling must survive 4 training stages, translate between Anthropic and Qwen formats, and maintain 95% reliability. No per-stage gates, no regression tests, no format translation layer. This is the highest-risk integration surface.

**Theme 3: Data quality uncertainty** -- flagged in CEO (Finding 9: legal risk) and Eng (Findings A-1, H-1, H-2, S-1). The corpus assembly pipeline has: an undersized relevance classifier, unvalidated transcription quality on real masterclass audio, unvalidated diarization, no provenance tracking, and legal exposure. All solvable but all must be solved before committing $700 to CPT.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | issues_open | 9 findings (3 critical, 4 high, 2 medium). Plan restructured with 4 strategic gates. |
| Eng Review | `/plan-eng-review` | Architecture & tests | 1 | issues_open | 23 findings (2 critical, 10 high, 10 medium, 1 low). 3 must-fix-first items. |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | skipped | No UI scope in plan. |
| Codex Review | `codex` | Independent 2nd opinion | 0 | unavailable | Codex CLI not installed. |

**VERDICT:** REVIEWED with 27 auto-decided issues (25 mechanical, 2 taste). Plan needs restructuring before implementation. See Final Approval Gate below.
