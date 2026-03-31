# Teacher Model Finetuning Design

**Date:** 2026-03-30
**Status:** Design approved, pending implementation plan
**Timeline:** 6 months (parallel track alongside web beta)
**Approach:** "The Prodigy" -- CPT + SFT + GRPO + DPO

## Motivation

Voice ownership. The teacher personality should live in model weights, not a rented system prompt. This provides:

1. A structural moat that competitors cannot replicate by using the same API
2. Independence from Anthropic model update risks (tone shifts, safety filter changes)
3. Deeper domain knowledge than a generalist model with a system prompt
4. Cost savings at scale (10x cheaper past 2K DAU)

Claude continues to serve production until the finetuned model clears a hard quality gate on the 7-dimension teaching rubric.

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

Relevance classifier: Binary classifier trained on the 379 teaching moments as positive examples. Scores each passage 0-1 for pedagogical relevance. Discard below 0.3. This ensures CPT corpus is concentrated on actual piano teaching, not tangential content.

## Training Pipeline

### Stage 1: Continual Pretraining (CPT)

**Goal:** Inject piano pedagogy knowledge into model weights.

- Standard next-token prediction on the 100-200M token corpus
- Full-parameter training (not LoRA) for maximum knowledge absorption
- Learning rate: 1e-5 to 2e-5 (10-20x below original pretraining)
- Data mix: 80% piano pedagogy corpus + 20% general text (SlimPajama subset) to prevent catastrophic forgetting
- Epochs: 1-2 passes over the full corpus
- Compute: ~8-16 hours on 4xH100, ~$400-700

**Risk:** Catastrophic forgetting of conversational ability. Mitigated by the 80/20 data mix and conservative learning rate. Validated by running general conversation benchmarks (MT-Bench) before and after CPT.

### Stage 2: Supervised Fine-Tuning (SFT)

**Goal:** Teach the CrescendAI teacher format, voice, and tool usage.

Data (2-5K examples):

| Type | Source | Count |
|------|--------|-------|
| Observation responses | Teaching moments -> Claude-generated gold responses grounded in CPT knowledge | ~1,000 |
| Chat conversations | Multi-turn dialogues: all 6 dimensions, all skill levels, all repertoire periods | ~1,000 |
| Tool use (exercises) | Scenarios requiring `create_exercise` tool | ~500 |
| Elaborations | "Tell me more" follow-ups with musical depth | ~300 |
| Edge cases | Frustration, plateaus, wrong piece, first session | ~200 |

Format: ChatML / Qwen chat template with system prompt, user context, assistant response. Tool calls in Qwen native format.

Training: QLoRA (rank 32-64) with Unsloth. ~2-4 hours on 1xH100. ~$10-30.

The SFT stage bakes in current system prompt rules: 1-3 sentences, no scores, no lists, specific musical references, actionable corrections, no markdown in observations.

### Stage 3: GRPO (Group Relative Policy Optimization)

**Goal:** Teach pedagogical judgment using the existing eval rubric as reward.

How it works:
1. Present a teaching scenario (student context + scores + piece)
2. Model generates N=4-8 candidate responses
3. Judge v2 (7-dimension rubric) scores each response
4. GRPO updates the model toward higher-scored responses

Reward signal (7 dimensions from `apps/evals/teaching_knowledge/`):
- Musical Accuracy
- Pedagogical Appropriateness
- Specificity & Actionability
- Student Awareness
- Style Consistency
- Observation Pacing
- Tool Use Appropriateness

Why GRPO: No critic model needed. Uses group-relative ranking. Proven by DeepSeek R1 for subjective quality optimization. Simpler and cheaper than PPO/RLHF.

Training: 500-1,000 scenarios, 4-8 generations each. ~4-8 hours on 1xH100 with TRL. ~$20-50.

**Risk:** Judge bias propagation. If the judge systematically favors corrections over encouragement, the model will too. Audit judge v2 for dimension-level bias before GRPO. Current smoke test scored 2.57/3.0.

### Stage 4: DPO (Direct Preference Optimization)

**Goal:** Final polish on voice warmth, tone, and subjective feel.

Data (500-1K preference pairs):
- Best vs worst GRPO outputs for each scenario (automated)
- Hand-curated pairs where human judges warmth/tone (the rubric catches quality, you catch feel)
- Anti-pattern pairs: generic responses, lecture-y responses, responses with scores/ratings, emoji, markdown in observations

Training: QLoRA with TRL DPOTrainer. ~2-4 hours on 1xH100. ~$10-30.

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

### Ongoing Evaluation

- A/B test with real students before full cutover
- Monitor per-dimension scores weekly
- Track tool use failure rate
- Compare student engagement metrics (session length, return rate) between Claude and finetuned model

## Timeline

| Month | Data Pipeline | Training | Integration |
|-------|--------------|----------|-------------|
| 1 | Build scraping/transcription pipeline. Start YouTube transcription (tonebase, Josh Wright, Nahre Sol). Download public domain books. | -- | -- |
| 2 | Continue YouTube long tail. Parse PDFs (Piano Pedagogy Forum, PQDT dissertations). Scrape Bulletproof Musician. Build relevance classifier. | -- | -- |
| 3 | Complete Tier 1-3 collection. Quality filtering + dedup. Assemble final CPT corpus. | Begin CPT on first 100M checkpoint. | -- |
| 4 | Continue accumulating. Curate SFT dataset (2-5K examples). | CPT complete. Begin SFT. | -- |
| 5 | Generate GRPO scenarios. Audit judge v2 for bias. Create DPO preference pairs. | SFT complete. GRPO training. DPO polish. | Upload to Together.ai. Integration test with AI Gateway. |
| 6 | Collect real session data from beta (with consent). | Iterate based on eval results. | Quality gate eval. A/B test. Conditional production switch. |

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
