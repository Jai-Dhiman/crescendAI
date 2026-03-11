# Slice 11: Teacher Voice Fine-Tuning Strategy

See `docs/architecture.md` for the full system architecture.
See `docs/apps/06a-subagent-architecture.md` for the two-stage subagent pipeline.
See `docs/apps/06-teacher-llm-prompt.md` for the current teacher persona prompt.

**Status:** RESEARCH / FUTURE (not implemented)
**Last verified:** 2026-03-04
**Date:** 2026-03-04
**Notes:** Strategy document. Depends on v1 pipeline being live with real student interactions. Provider decisions (Groq for subagent, Anthropic direct for teacher) are immediate. Fine-tuning is a 6-12 month track.

**Goal:** Train a domain-specific teacher voice model on real piano pedagogy -- masterclass transcripts, lesson recordings, pedagogical literature -- so the teacher LLM speaks with the vocabulary, pacing, and specificity of an experienced piano teacher, not a generalist LLM approximating one.

---

## Provider Architecture (Immediate)

Updated from the original OpenRouter-only design. The pipeline now uses direct provider APIs for each stage, optimized for latency and quality respectively.

### Stage 1: Analysis Subagent -- Groq

**Provider:** Groq (direct API)
**Model:** Llama 3.3 70B or Llama 4 Maverick (evaluate both)
**Why Groq:** Groq's LPU inference engine runs Llama 70B at 450-800 tokens/second. The subagent outputs ~200 tokens of structured JSON + narrative. At 800 tok/s, that completes in ~0.25s. Combined with network overhead, expect ~0.3-0.5s total for stage 1.

**Why this matters for the pipeline:** The subagent is the latency bottleneck that blocks stage 2. Cutting subagent time from ~0.5s (Haiku via OpenRouter) to ~0.3s (Groq direct) reclaims budget for the teacher model.

**Cost:** ~$0.50-0.60 per million tokens (input and output). At 80 calls/user/month with ~1000 tokens per call, this is ~$0.004/user/month. Effectively free.

**Quality note:** Groq runs standard Llama weights -- identical outputs to Together.ai or any other Llama host. The LPU accelerates inference without degrading quality. The subagent task (structured reasoning over JSON, selecting from 3-5 candidates) is well within Llama 70B's capabilities.

### Stage 2: Teacher LLM -- Anthropic Direct

**Provider:** Anthropic API (direct, not via OpenRouter)
**Model:** Claude Sonnet 4.6
**Why direct Anthropic:** Eliminates the OpenRouter routing hop (~50-100ms). Prompt caching works natively with the Anthropic API -- the teacher persona system prompt stays cached across all students and sessions.

**Why Sonnet:** The teacher persona prompt (see `docs/apps/06-teacher-llm-prompt.md`) requires nuanced instruction following: be specific but brief, be warm but honest, reference the exact musical moment, suggest what to try. Sonnet 4.6 is measurably better at this than Llama 70B or GPT-4o for persona-constrained short-form generation. The teacher output is 1-3 sentences -- quality per token matters more than cost per token.

**Cost:** $3.00 input / $15.00 output per million tokens. At 80 calls/user/month with ~500 tokens per call (300 input from subagent handoff + 100 output), this is ~$0.01/user/month. Negligible.

**Prompt caching:** The teacher persona system prompt (~500 tokens) is identical across all requests. With Anthropic's prompt caching, this prefix is cached after the first request and subsequent requests pay reduced rates for the cached portion. For high-volume usage, this effectively halves the input cost for the static portion.

### Stage 3: UI Subagent -- Groq

Same provider and model as Stage 1. Only invoked ~20-30% of the time (when the teacher declares a non-text modality). Outputs a small JSON component configuration. Adds ~0.2-0.3s.

### Fallback

**Primary fallback:** If Groq is down, route subagent calls through OpenRouter to any available Llama provider (Together.ai, Fireworks, etc.). If Anthropic is down, route teacher calls through OpenRouter to Claude Sonnet.

**Emergency fallback:** Cloudflare Workers AI with Llama 3.1 70B for both stages. Co-located with the Workers backend, zero external dependency. Lower quality but functional.

### Updated Latency Budget

| Stage | Provider | Expected Latency |
|---|---|---|
| Subagent (analysis) | Groq, Llama 70B | ~0.3s |
| Teacher (observation) | Anthropic, Sonnet 4.6 | ~1.0-1.5s |
| UI subagent (optional) | Groq, Llama 70B | ~0.2s |
| **Total (text-only)** | | **~1.3-1.8s** |
| **Total (with component)** | | **~1.5-2.0s** |

Well within the <3s target. The latency improvement over OpenRouter (~0.3-0.5s saved) gives headroom for prompt caching misses or network variability.

### UX Pacing

Groq's speed creates a UX consideration for the teacher stage specifically. At 800 tok/s, a 75-token teacher observation would render in ~94ms -- essentially instant. This feels like a canned response, not a thoughtful teacher.

**Solution:** Client-side pacing in the iOS app. Buffer the streamed response and reveal tokens at ~4-5 words/second (~250 words/minute, natural reading speed) with a subtle typing animation. The response *starts* immediately (no perceived latency) but *appears* at a deliberate pace that matches the teacher persona.

This only applies to the teacher output (stage 2), which the student sees. The subagent output (stage 1) is consumed programmatically by the Worker -- speed is pure upside there.

Note: Since the teacher model is Sonnet 4.6 (not on Groq), its natural generation speed (~100-150 tok/s) already produces a reasonable streaming pace. Client-side pacing is a refinement, not a necessity. It becomes critical if/when the teacher model moves to a faster provider (e.g., after fine-tuning, running on Groq).

---

## Why Fine-Tune the Teacher Voice

The current teacher persona prompt (Slice 06) works well with Sonnet. But there is a ceiling to what prompt engineering alone can achieve for domain-specific voice:

**What prompts can do:**

- Constrain tone (warm, brief, honest)
- Constrain format (1-3 sentences, no bullets, no scores)
- Set behavioral rules (pick ONE thing, be specific, suggest what to try)

**What prompts cannot do:**

- Teach the LLM vocabulary it does not have ("drop into the key," "let the arm weight carry through," "the voicing is muddy in the inner voices")
- Encode the specific way experienced piano teachers give tactile, embodied feedback ("imagine your fingers are sinking into warm sand" vs. "play with more legato")
- Replicate the diagnostic specificity of someone who has heard thousands of students make the same mistake ("this sounds like you're pedaling on autopilot -- your ear isn't tracking the harmonic changes")
- Capture stylistic pedagogy conventions ("in Chopin, the rubato lives in the right hand while the left hand keeps time" -- a teacher would say this naturally, an LLM might not)

Fine-tuning encodes this domain language into the model's weights, so the teacher voice emerges naturally rather than being constrained into it.

---

## Data Sources for Fine-Tuning

The fine-tuning dataset trains the model to speak like a piano teacher giving real-time feedback. Each training example is an instruction-response pair: given a musical scenario, produce a teacher-quality observation.

### Source 1: Masterclass Transcripts

**What:** Transcripts of piano masterclasses where a teacher listens to a student play and gives feedback. The teacher's verbal responses -- what they say, how they phrase it, what they focus on -- are the target outputs.

**Where to find them:**

- Tonebase Piano (subscription, extensive masterclass library with world-class pedagogues)
- YouTube masterclasses (auto-captions, cleaned up): Barenboim, Lang Lang, Brendel, Uchida, Schiff, Perahia
- International competition masterclasses (Cliburn, Chopin Competition, Leeds)
- University/conservatory lecture recordings (many are publicly available)

**Volume estimate:** 50-100 masterclass videos at ~45 min each = 40-75 hours of transcription. Not all content is usable -- extracting the feedback moments (teacher speaking after student plays) requires segmentation.

**Processing pipeline:**

1. Transcribe via Whisper (or use existing captions)
2. Segment: identify "student plays, teacher responds" episodes
3. For each episode, extract:
   - What the student played (piece, passage, approximate bars)
   - What the teacher said (verbatim)
   - What dimension the feedback targets (dynamics, pedaling, phrasing, etc.)
   - The framing (correction, suggestion, demonstration request, encouragement)
4. Convert to instruction-tuning format (see Training Format below)

**Quality considerations:**

- Masterclass feedback is often for advanced students. The vocabulary and expectations differ from beginner feedback. Tag each example with the apparent student level.
- Some masterclass feedback is visual/demonstrative ("let me show you" + teacher plays). These are not directly usable as text examples but the verbal framing before/after the demonstration is valuable.
- Teacher personality varies enormously: Barenboim is philosophical, Lang Lang is enthusiastic, Brendel is analytical. The fine-tuned model should capture a range, with the CrescendAI persona selecting from within that range.

### Source 2: Piano Pedagogy Literature

**What:** Written descriptions of how to teach specific techniques, common student problems, and diagnostic advice. Not direct feedback transcripts, but the knowledge base that informs teacher intuition.

**Sources:**

- *The Art of Piano Playing* (Neuhaus)
- *On Piano Playing* (Sandor)
- *Fundamentals of Piano Practice* (Chang) -- freely available online
- *The Pianist's Guide to Pedaling* (Banowetz)
- Method book teacher guides (Faber, Alfred, Bastien) -- these explicitly describe common student errors and how to address them
- Piano pedagogy journals (American Music Teacher, Piano Pedagogy Forum)

**Processing:** Extract passages that describe specific feedback scenarios. Convert to instruction-tuning pairs:

- Input: "A student is playing a Chopin Nocturne and their pedaling sounds blurry in the transition between phrases."
- Output: (synthesized from the pedagogical source) "You're holding the pedal through the phrase change -- the harmonies are bleeding into each other. Try a half-pedal at the barline: lift just enough to clear the bass, but keep some sustain in the upper register."

**Volume estimate:** 20-30 books/sources, extracting 10-30 usable examples each = 200-900 training pairs. These are higher effort per example but high quality.

### Source 3: CrescendAI Golden Set (Accumulated Over Time)

**What:** The best teacher observations generated by Sonnet during real CrescendAI sessions. Curated by quality: observations that sound genuinely like a teacher, are specific and actionable, and use embodied/tactile language.

**Collection mechanism:**

- Manual curation during beta: Jai reviews all teacher outputs, stars the ones that sound right
- User signal (future): "helpful" / "not helpful" buttons on observations
- A/B testing signal: when testing prompt variants, the winning variants' outputs enter the golden set

**Volume:** Starts at zero. Grows with usage. Target: 500+ golden examples before attempting fine-tuning. At 80 observations/user/month with 10 beta users, that is ~800 observations/month. If 20% are golden, that is 160/month. Reach 500 in ~3 months of beta.

**Why this matters:** This is the distillation play. Once you have enough golden examples of CrescendAI's specific teacher voice (not just generic good piano teaching), you can fine-tune a smaller, cheaper model to replicate that exact voice.

### Source 4: Synthetic Data from Structured Scenarios

**What:** Use Sonnet to generate teacher observations from carefully constructed scenarios, then filter for quality. This bootstraps the dataset before real user interactions exist.

**Method:**

1. Define 50-100 canonical teaching scenarios spanning:
   - All 6 dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation)
   - All 3 student levels (beginner, intermediate, advanced)
   - Multiple framings (correction, recognition, encouragement, question)
   - Multiple composers/styles (Chopin, Bach, Beethoven, Debussy, Liszt, Ravel)
2. For each scenario, generate 5-10 variant observations using Sonnet with the teacher persona prompt
3. Human-filter: keep only the ones that sound like a real teacher
4. Include both the "good" examples (what the model should say) and "bad" examples (what it should avoid) for DPO training

**Volume:** 100 scenarios x 5 variants = 500 raw examples. After filtering, expect 200-350 usable pairs.

**Cost:** ~$0.50-1.00 in API calls. The human filtering time is the real cost.

---

## Training Format

### Supervised Fine-Tuning (SFT) Pairs

Each training example follows the instruction-tuning format the target model expects. For Llama-family models:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a piano teacher who has been listening to your student practice. Give ONE specific observation about what you just heard.
[Full teacher persona prompt from Slice 06]
<|eot_id|><|start_header_id|>user<|end_header_id|>

## What I heard
Teaching moment: bars 20-24, second phrase of Chopin Nocturne Op. 9 No. 2
Dimension: pedaling (score: 0.35, student's usual: 0.62)
Learning arc: polishing (session 14)
Musical context: Romantic nocturne -- pedaling is essential for the singing legato line
Framing: correction (blind spot -- pedaling usually a strength)

## What to say
Give one observation about the pedaling issue. Be specific about what you heard and what to try.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The pedaling in the second phrase got away from you -- you're catching the notes but not the harmonic changes. Listen for the shift at bar 22: that's where the bass moves, and the pedal needs to clear before the new harmony settles in. Try a half-change there instead of a full lift -- you'll keep the legato but lose the blur.
<|eot_id|>
```

### DPO (Direct Preference Optimization) Pairs

For each scenario, provide a "chosen" (good) and "rejected" (bad) response:

**Chosen (sounds like a teacher):**
> "The pedaling in the second phrase got away from you -- you're catching the notes but not the harmonic changes. Try a half-change at bar 22 where the bass moves."

**Rejected (sounds like an LLM):**
> "I noticed that your pedaling technique in measures 20-24 could benefit from some refinement. The sustain pedal appears to be held too long through harmonic transitions, which creates a blurring effect. Consider implementing half-pedaling at key harmonic change points."

The DPO training teaches the model to prefer the first style: direct, embodied, specific, conversational. Together.ai supports [advanced DPO variants](https://www.together.ai/fine-tuning) including IPO and KTO.

---

## Training Infrastructure

### Provider: Together.ai

**Why Together.ai for training:**

- Supports fine-tuning Llama 3.3 70B and larger models (up to 100B+)
- LoRA fine-tuning at $2.90/hour for 70B+ models
- DPO training supported with advanced variants
- Serverless inference for the fine-tuned model included
- 2-4x longer context training at no extra cost (useful for masterclass transcripts with longer context)

**Training cost estimate:**

- Dataset: ~1000-2000 training pairs (combining all 4 sources)
- LoRA fine-tuning on Llama 3.3 70B: ~2-4 hours of training = $6-12
- DPO pass: ~1-2 additional hours = $3-6
- **Total training cost: ~$10-20 per training run**

The training cost is trivial. The dataset creation is the real investment (weeks of human effort for curation and quality filtering).

### Target Models for Fine-Tuning

**Primary target: Llama 3.3 70B (or current equivalent at training time)**

- Large enough to capture nuanced pedagogical voice
- Small enough for affordable inference ($0.88/M tokens on Together.ai)
- Can be served on Groq after fine-tuning for fast inference

**Stretch target: Llama 3.2 8B (or current equivalent)**

- If the 70B fine-tune works well, attempt distillation to 8B
- 8B on Groq runs at ~1200 tok/s, $0.06/M tokens
- Would enable fully offline teacher voice on device (Apple Foundation Models integration path)

**Why not fine-tune Sonnet/Claude?**

- Anthropic does not currently offer fine-tuning for Claude models
- Even if available, the model weights stay with Anthropic -- no self-hosting, no Groq deployment
- Open-weight models (Llama) provide full control: host anywhere, quantize, deploy on-device

### Evaluation

Before deploying a fine-tuned model, evaluate against the current Sonnet baseline:

**Automated metrics:**

- Persona adherence: does the output follow the format rules? (1-3 sentences, no bullets, no scores, no jargon)
- Specificity: does it reference the specific musical moment from the input?
- Actionability: does it suggest something to try? (Simple keyword/pattern check)

**Human evaluation (the one that matters):**

- Blind A/B test: show 50 observation pairs (Sonnet vs. fine-tuned) to 3-5 evaluators (piano teachers or advanced students)
- Rate on: "Which sounds more like a real piano teacher?" and "Which is more helpful?"
- Target: fine-tuned model wins or ties >= 60% of the time

**Domain-specific checks:**

- Does the model use embodied/tactile vocabulary naturally? ("arm weight," "singing tone," "voicing the inner line")
- Does it avoid LLM-isms? ("I noticed that," "your technique could benefit from," "consider implementing")
- Does it contextualize for composer/style? (Different language for Chopin vs. Bach vs. Debussy)

---

## Timeline and Milestones

### Phase 0: Immediate (Now)

- Switch subagent to Groq + Llama 70B (or Maverick)
- Switch teacher to Anthropic direct API + Sonnet 4.6
- Implement client-side pacing layer in iOS
- Update `docs/architecture.md` to reflect new provider architecture

### Phase 1: Dataset Collection (Months 1-3 of v1 launch)

- Begin masterclass transcript collection and segmentation
- Generate synthetic scenarios from the Slice 06 prompt
- Start accumulating the golden set from real CrescendAI sessions
- Extract pedagogy literature examples
- **Milestone:** 500+ curated training pairs

### Phase 2: First Fine-Tune (Month 4-5)

- SFT on Llama 70B using Together.ai
- DPO pass with chosen/rejected pairs
- Evaluate against Sonnet baseline (blind A/B)
- **Milestone:** Fine-tuned model ties or beats Sonnet on teacher voice quality in >= 60% of blind comparisons

### Phase 3: Deployment Decision (Month 5-6)

If Phase 2 succeeds:

- Deploy fine-tuned Llama 70B on Groq for inference (fast + cheap + good)
- Keep Sonnet as fallback
- Monitor user engagement metrics (observation helpfulness, "tell me more" rate) for regressions

If Phase 2 falls short:

- Identify gaps (which scenarios did the fine-tuned model fail on?)
- Expand dataset in weak areas
- Re-train with more examples
- Keep Sonnet as primary, iterate on fine-tuning

### Phase 4: Distillation to 8B (Month 6-12)

- Use the fine-tuned 70B as a teacher model to generate training data for 8B
- Fine-tune Llama 8B (or equivalent) on the 70B's outputs
- Evaluate: can 8B match 70B quality for this narrow domain?
- If yes: deploy on Groq ($0.06/M tokens, ~1200 tok/s) or explore on-device deployment via Apple Foundation Models

---

## Cost Comparison: Current vs. Fine-Tuned

Per-user costs at 80 LLM calls/month:

| Configuration | Subagent Cost | Teacher Cost | Total/User/Month |
|---|---|---|---|
| **Current plan:** Groq + Sonnet | $0.004 | $0.01 | ~$0.014 |
| **Fine-tuned 70B on Groq (both stages)** | $0.004 | $0.006 | ~$0.010 |
| **Fine-tuned 8B on Groq (both stages)** | $0.0003 | $0.0004 | ~$0.001 |
| **Fine-tuned 8B on-device** | $0.00 | $0.00 | $0.00 |

The cost savings are not the motivation -- even Sonnet is only $0.01/user/month. The motivations are:

1. **Voice quality:** A fine-tuned model that speaks like a piano teacher by default, not by instruction
2. **Latency:** Fine-tuned 70B on Groq gives Sonnet-quality voice at Groq speed
3. **Independence:** Open weights, deployable anywhere, no single-provider dependency
4. **On-device path:** Fine-tuned 8B opens the door to fully offline teacher responses

---

## Risks and Mitigations

**Risk: Masterclass transcripts are copyrighted.**
Mitigation: Use transcripts for training only (transformative use). The model does not memorize or reproduce transcripts -- it learns vocabulary patterns and response style. Consult legal counsel before using paid platform content (Tonebase, Masterclass.com). Prioritize freely available sources (YouTube, university lectures, public domain pedagogy texts).

**Risk: Fine-tuned model loses generality.**
Mitigation: LoRA fine-tuning preserves the base model's general capabilities while adding domain-specific behavior. Evaluate on both piano-specific scenarios AND general instruction-following to ensure no regression.

**Risk: Dataset too small or biased.**
Mitigation: Start with synthetic data (Source 4) to establish a baseline, then improve with real data (Sources 1-3). Monitor for style biases (e.g., dataset dominated by one teacher's style). Include diverse pedagogues in the training set.

**Risk: Fine-tuned model sounds uncanny -- almost right but detectably off.**
Mitigation: DPO training specifically addresses this by teaching the model to prefer natural teacher language over LLM-isms. Human evaluation is mandatory before deployment. If the model is 80% there but has a telltale "AI" quality, keep Sonnet and iterate on the dataset.

---

## Open Questions

1. **Tonebase licensing:** Can masterclass transcripts be used for ML training under their terms of service? Need to review. Alternatively, approach Tonebase as a potential partner (their pedagogues' voices powering CrescendAI could be a mutual value proposition).

2. **Teacher diversity vs. consistency:** Should the fine-tuned model capture a range of teaching styles, or converge on a single CrescendAI voice? Range provides flexibility but may feel inconsistent. A single voice is more product-coherent but loses the richness of different pedagogical approaches.

3. **Evaluation set size:** How many blind A/B pairs are needed to make a statistically meaningful quality comparison? With 50 pairs and 3 evaluators, is that enough signal?

4. **On-device fine-tuned model:** Apple Foundation Models (iOS 18.2+) supports custom adapters. Can a LoRA adapter trained on Together.ai be converted and applied to Apple's on-device model? This is the ultimate endgame (zero-cost, zero-latency teacher) but the technical path is unclear.

5. **Continuous improvement:** As the golden set grows, should the model be re-trained periodically (monthly? quarterly?) or is a single fine-tune sufficient? How much does marginal data improve quality after the initial 1000-2000 examples?

6. **Multi-stage fine-tuning:** Should the subagent also be fine-tuned for musical reasoning, or is generic Llama 70B sufficient for structured analysis? The subagent task (JSON reasoning over 6 dimension scores) seems well within base model capabilities, but domain-specific reasoning patterns might improve selection quality.
