# CrescendAI Research & Product Timeline

> **Status (2026-03-19):** Clean-fold baseline established and optimized. **A1-Max optimized: 79.85% pairwise, R2=0.336** (4-fold mean, clean folds). Loss weight autoresearch (8 iterations) found contrastive=0.6, regression=0.8. Combined with best-checkpoint loading fix, this recovers most of the leaked-fold performance (was 80.8% on leaked folds, now 79.9% on clean folds). **Aria validation complete (Phase A):** Frozen Aria 59.6% pairwise, frozen MuQ 62.2%. Error correlation phi=0.043. Next: Phase B (contrastive pretraining) + Phase C (LoRA fine-tuning).

> **FOLD LEAK WARNING:** All pairwise accuracy numbers in the Legacy Results section below were computed with leaked folds (segments from the same piece appearing in both train and val splits). These numbers are INVALID for model comparison. Clean piece-stratified folds are now in place. All future results use piece-stratified CV only.

*Core question: "How well is the student playing what the score asks for?"*

Target user: Sarah -- 3 years playing, no teacher, records on her phone, wants direction on what to work on next.

North star: Give Sarah one piece of useful feedback on one passage she's working on. Not perfect. Not comprehensive. One thing a teacher would actually say after hearing her play.

---

## Aria Discovery (2026-03-18)

**Aria** is a 650M-parameter LLaMA-architecture model pretrained by EleutherAI on 820K piano MIDI performances (60K hours). Apache 2.0 license. SOTA on 6 MIR benchmarks.

**Why this changes everything:**

- Phase 3 (Symbolic FM, 6-12 months, HIGH risk) planned to pretrain a Transformer on 370K+ MIDI performances. Aria already did this at 2x scale with better data curation.
- S2 GNN (71.3% pairwise on leaked folds) was our best symbolic encoder, trained on 24,220 graphs. Aria's pretrained representations should dominate.
- The ISMIR paper's fusion failure (error correlation r=0.738) was partly because S2 lacked pretrained inductive bias. Aria closes this gap.
- Same architecture encodes both performance MIDI and score MIDI -- score conditioning is native, not bolted on.

**Decision:** Replace ALL custom symbolic encoders (S2 GNN, S2H, S3, S1) with Aria. Eliminate Phase 3 entirely. Integrate Aria into model v2 from day one.

---

## Key Decisions

### Inference: Cloud-Only (decided 2026-03-14)

All MuQ inference runs on the HuggingFace inference endpoint. No on-device ML / Core ML conversion planned. This simplifies the iOS architecture -- the app captures audio and sends chunks to the API worker, which forwards to HF and runs STOP classifier logic server-side.

### Fusion Strategy: Separate-Then-Fuse (decided 2026-03-18)

Train MuQ (audio) and Aria (symbolic) independently. Measure error correlation on clean folds. Fuse with per-dimension learned gates.

Previous attempt (ISMIR paper): fusion R2=0.524 < audio-only R2=0.537 (on leaked folds). Error correlation r=0.738. Root cause was S2 lacking pretrained inductive bias -- both encoders derived from identical MIDI source data without distinct learned representations. Aria's 820K-MIDI pretraining should break this correlation.

### Aria Validation Complete -- Phase A (2026-03-19)

Linear probe on frozen embeddings, 4-fold piece-stratified CV on PercePiano (clean folds). No fine-tuning, no LoRA -- pure representation quality test.

| Model | Pairwise | R2 | dynamics | timing | pedaling | artic. | phrasing | interp. |
|-------|----------|------|----------|--------|----------|--------|----------|---------|
| Aria-Embedding (512d) | 59.6% | -4.28 | 65.8% | 55.8% | 58.6% | 54.2% | 57.9% | 61.2% |
| Aria-Base (1536d) | 59.6% | -4.78 | 62.5% | 58.0% | 60.7% | 54.3% | 54.8% | 62.3% |
| MuQ (1024d, mean-pooled) | 62.2% | -1.90 | 72.4% | 67.5% | 66.6% | 54.7% | 60.9% | 63.9% |

**Error correlation phi = 0.043** (near-zero). This dramatically validates the fusion strategy: S2's correlation was r=0.738 (redundant), Aria's is 0.043 (independent). The 18x reduction in error correlation means fusion should produce substantial gains.

**Interpretation:**
- Both Aria variants perform identically (59.6%). Contrastive fine-tuning for the embedding variant neither helped nor hurt quality sensitivity.
- Aria is marginal frozen (55-60% band) but significantly above chance (50%). Quality signal exists; fine-tuning should unlock it.
- MuQ dominates on dynamics and timing. Aria's advantage should emerge after quality-aware training.
- Negative R2 values indicate frozen embeddings predict poorly in absolute terms (expected for a linear probe on representations not trained for this task).

**Decision:** Proceed to Phase B (contrastive pretraining) and Phase C (LoRA fine-tuning). The marginal frozen performance combined with near-zero error correlation confirms Aria is worth investing in for fusion.

Scripts: `model/src/model_improvement/aria_embeddings.py`, `aria_linear_probe.py`. Results: `model/data/results/aria_validation.json`.

### Fold Leak Fix (decided 2026-03-18)

All previous CV used segment-level splits that allowed segments from the same piece into both train and val. New piece-stratified folds ensure no piece appears in both splits. All legacy numbers are preserved for reference but marked INVALID.

### Phone Audio: Pseudo-Validated (decided 2026-03-14)

The YouTube AMT validation (79.9% agreement on 50 mediocre recordings) serves as proxy phone audio validation. These recordings include phone-quality audio, home pianos, digital keyboards, and varying room acoustics -- representative of real user conditions. No longer treated as an "existential risk." Formal paired recordings (studio + iPhone) remain a Wave 2 nice-to-have for quantifying the exact gap.

---

## Model v2 Plan (5-6 weeks)

### Training Strategy

**Symmetric contrastive pretraining:** Both MuQ and Aria get quality-aware contrastive training on T2 competition + T5 YouTube Skill data before fine-tuning. This teaches both encoders a shared quality-ordering space.

**Score conditioning from day one:** Aria encodes both performance MIDI (z_perf) and score MIDI (z_score). delta = z_perf - z_score captures expressive deviation. No separate "score conditioning phase" -- it is baked into the architecture.

**Multi-tier training mix:** PercePiano as anchor (20%), ordinal-dominated training (80%).

| Tier | Segments | Signal | Loss | Mix Weight |
|------|----------|--------|------|------------|
| T1: PercePiano | 1,202 | 6-dim regression + ranking | BCE + ListMLE + CCC | 20% (anchor) |
| T2: Competition (expanded) | ~11,000 | Ordinal placement ranking | ListMLE | 40% |
| T5: YouTube Skill | ~3,100 | Ordinal skill-level ranking (5 buckets) | ListMLE | 30% |
| T3: MAESTRO | 24,321 | Contrastive pairs (same piece, diff performer) | InfoNCE | 10% |

### Architecture

**Audio path (MuQ):** Same MuQ + LoRA backbone, attention pooling, per-dimension heads. Contrastive pretrain on T2+T5, then fine-tune on all tiers.

**Symbolic path (Aria):** Aria 650M frozen or LoRA-adapted. Takes performance MIDI + score MIDI as dual input. Outputs z_perf, z_score, and delta = z_perf - z_score. Contrastive pretrain on T2+T5 (using AMT MIDI), then fine-tune.

**Fusion:** Per-dimension learned gates. `quality_d = gate_d * f_audio(z_muq) + (1 - gate_d) * f_symbolic(z_perf, z_score, delta_d)`. Train gates on T1 (PercePiano) with both encoders frozen.

### Evaluation Framework

1. **Skill discrimination (primary):** Spearman rho between skill bucket and predicted score on held-out T5 recordings. Must be monotonically increasing from Bucket 1 to 5.
2. **Within-level pairwise (secondary):** Pairwise accuracy on PercePiano (piece-stratified folds) and within-bucket YouTube pairs.
3. **Per-dimension skill sensitivity:** Which dims show strongest skill-level signal?
4. **Cross-piece generalization:** Hold out entire pieces from training -- does skill discrimination transfer?
5. **Error decorrelation (fusion):** Measure error correlation between MuQ and Aria paths. Fusion is only worth pursuing if correlation drops below ~0.5 (was 0.738 with S2). **Phase A result: phi=0.043 -- gate passed decisively.**

### Timeline (5-6 weeks)

**Week 1-2: Data preparation + contrastive pretraining**

- T5 YouTube Skill Corpus: collect + curate remaining 14 pieces
- T2 expansion: Chopin 2015, Cliburn 2022/Amateur, Queen Elisabeth 2024
- Contrastive pretraining of MuQ and Aria on T2+T5

**Week 3-4: Independent encoder training**

- MuQ fine-tune on all tiers (piece-stratified CV)
- Aria fine-tune on all tiers (performance MIDI + score MIDI input)
- Skill discrimination checkpoint after each

**Week 5: Fusion + evaluation**

- Measure error correlation between MuQ and Aria
- If decorrelated: train per-dimension gates on T1
- Full evaluation: skill discrimination, pairwise, cross-piece generalization

**Week 6: Deploy + iterate**

- Deploy best model (single or fused) to HF endpoint
- Run pipeline evals with new model
- ISMIR paper update with clean-fold results

---

## Skill-Level Evaluation (2026-03-18) -- FAIL

Evaluated A1-Max on 26 YouTube Fur Elise recordings across 5 human-labeled skill buckets (beginner through professional). **The model shows zero skill-level discrimination:**

```
Overall mean by bucket:
  Bucket 1 (beginner):      0.558 (n=4)
  Bucket 2 (early intermed): 0.566 (n=6)
  Bucket 3 (intermediate):   0.560 (n=5)
  Bucket 4 (advanced):       0.561 (n=5)
  Bucket 5 (professional):   0.565 (n=6)
```

Total range: 0.008. No dimension shows monotonic skill-level trend. The model cannot distinguish a 1-year beginner from Lang Lang.

**Root cause:** Training data (PercePiano) is 100% advanced-level performers. The regression head is calibrated to a narrow quality band (~0.4-0.7). Beginner audio features are out-of-distribution, so the head defaults to the mean.

**Fix:** Multi-tier training with YouTube Skill Corpus (T5) + Aria symbolic path that spans the full quality spectrum.

---

## Legacy Results (INVALID -- fold leak)

> **WARNING:** All numbers in this section were computed with leaked folds. Segments from the same piece appeared in both train and val. These results are NOT valid for model comparison or publication. Kept for historical reference only.

### LEGACY: Encoder Training Results

For full encoder details, see `docs/model/03-encoders.md`.

Trained on 1,202 PercePiano segments with 6 teacher-grounded composite dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation). PercePiano contains only 3 works and ~22 advanced-level performers -- no skill-level diversity.

| Model | Modality | Strategy | Pairwise Acc (LEAKED) | R2 (LEAKED) |
|-------|----------|----------|----------------------|-------------|
| A1-Max (ensemble) | Audio | MuQ + LoRA rank-32, ListMLE, CCC | 80.8% | 0.50 |
| A1 | Audio | MuQ + LoRA rank-16 | 73.9% | 0.40 |
| A2 | Audio | MuQ staged adaptation | 71.4% | 0.42 |
| A3 | Audio | MuQ full unfreeze | 69.9% | 0.28 |
| S2 (LEGACY) | Symbolic | GNN on score graph | 71.3% | 0.32 |
| S2H (LEGACY) | Symbolic | Heterogeneous GNN | 70.2% | 0.36 |
| S3 (LEGACY) | Symbolic | CNN + Transformer | 70.0% | 0.37 |
| S1 (LEGACY) | Symbolic | Transformer on REMI tokens | 68.4% | 0.33 |

**Deployed at time of leak discovery:** A1-Max 4-fold ensemble on HF inference endpoint (cloud-only).

**LEGACY symbolic encoders (S1, S2, S2H, S3) are fully replaced by Aria.** No further development on custom symbolic encoders.

### LEGACY: Per-Dimension Complementarity

| Dimension | A1 Audio (LEAKED) | S2 GNN (LEAKED) | Winner |
|-----------|-------------------|-----------------|--------|
| dynamics | ~0.70 | ~0.77 | S2 |
| timing | ~0.77 | ~0.65 | A1 |
| pedaling | ~0.72 | ~0.72 | Tie |
| articulation | ~0.66 | ~0.70 | S2 |
| phrasing | ~0.63 | ~0.63 | Tie |
| interpretation | ~0.74 | ~0.77 | S2 |

Audio wins timing decisively. Symbolic wins dynamics, articulation, and interpretation. These patterns may hold with clean folds but exact numbers are invalid.

### Layer 1 Validation (complete, pre-leak-discovery)

Four experiments validating assumptions. For detailed results, see `03-encoders.md` Layer 1 section.

| Experiment | Gate | Result | Decision |
|-----------|------|--------|----------|
| Competition correlation (Chopin 2021, n=11) | rho > 0.3 | rho=+0.704 | **PASS** -- model quality signal is real |
| AMT degradation (ByteDance vs GT MIDI) | drop < 10% | 0.0% drop | **PASS** -- symbolic path viable through AMT |
| Dynamic range (intermediate vs professional) | diagnostic | Cohen's d=0.47 | Usable for within-student tracking |
| MIDI-as-context feedback (LLM judge) | B wins > 65% | B wins 45% | **SKIP** -- raw MIDI stats don't help LLM |

**YouTube AMT validation (2026-03-13):** 50 mediocre-quality recordings, 1,225 pairs, 79.9% A1-vs-S2 agreement. All 6 dimensions above 72%. This also serves as proxy phone audio validation -- these recordings span phone audio, home pianos, digital keyboards, and varying room acoustics.

**Key insights from Layer 1:**

- Pedaling (rho=0.887) and phrasing (rho=0.803) are strongest predictors of competition placement
- Dynamics has *negative* correlation with placement -- model captures "amount" not "appropriateness." Score conditioning (now via Aria delta) fixes this at the model level. Phase 1's bar-aligned analysis partially addresses via LLM reasoning ("score says pp")
- Raw MIDI statistics hurt the LLM (judge called them "false precision"), but bar-aligned passage-specific context is untested and remains the right goal

### Prior Results (historical context)

**Early baselines (Jan 2025, 19 dimensions):**

- MuQ regression: R2=0.537 on 19 redundant PercePiano dimensions (PCA showed only 4 significant factors)
- Contrastive ranking (E2a): 84% pairwise on 19 dims -- inflated by redundancy and "quality halo" (PC1 = 47% variance)
- Current 6-dim results (73.9% pairwise) are on independent, teacher-grounded dimensions -- lower numbers, higher signal

**Score alignment experiments (failed):**

- Standard DTW on MuQ embeddings: ~18s onset error (MuQ encodes semantic content, not temporal features)
- Learned MLP projection: representation collapse
- **Conclusion:** Use the right representation for each sub-problem. MuQ for quality assessment. Spectral/symbolic features for alignment. Combine at the feedback layer.

---

## Training Roadmap

### Phase 0+3 (merged): Model v2 with Aria (5-6 weeks)

Phase 0 (multi-tier training) and Phase 3 (symbolic FM) have been merged. Aria eliminates the need for custom symbolic FM pretraining. Score conditioning is immediate via Aria's dual-input architecture.

See "Model v2 Plan" section above for full details.

**0a. YouTube Skill Corpus collection**

- Collect ~775 recordings across 16 pieces, 5 skill buckets
- Human curation pass (~6.5 hours): 5-bucket classification per recording
- Download audio, extract MuQ embeddings + AMT MIDI for Aria

**0b. Competition expansion**

- Add Chopin 2015, Cliburn 2022, Cliburn Amateur, Queen Elisabeth 2024 to T2 pipeline
- Target ~11,000 competition segments

**0c. Symmetric contrastive pretraining**

- Contrastive pretrain both MuQ and Aria on T2+T5
- Quality-aware ordering: higher-placed performers as positives

**0d. Independent encoder fine-tuning**

- MuQ: LoRA fine-tune on T1+T2+T3+T5 (piece-stratified CV)
- Aria: LoRA fine-tune on T1+T2+T3+T5 (performance MIDI + score MIDI)

**0e. Fusion evaluation**

- Measure error correlation between MuQ and Aria
- If decorrelated: per-dimension learned gates
- If still correlated: ship best single encoder, revisit fusion later

### Phase 1: Score Infrastructure (COMPLETE)

**Goal:** Transform LLM input quality. Same A1-Max model, dramatically better feedback. This delivers 80% of the user-facing improvement.

**1a. Score MIDI library -- COMPLETE**

- V1: 242 ASAP score MIDIs deployed to D1 + R2, bar-centric JSON with notes/pedal/time sigs
- V2: Expand to MAESTRO (external score sourcing), IMSLP/MuseScore, MusicXML for richer annotations (future)
- Graceful degradation to absolute scoring (Tier 2) for unknown pieces

**1b. Cloud AMT service -- COMPLETE**

- ByteDance piano transcription alongside MuQ on HF endpoint
- Single upload, two outputs (scores + MIDI + pedal CC64 events)
- Validated: 0% pairwise drop (MAESTRO), 79.9% agreement (YouTube)

**1c. Score following -- COMPLETE**

- Onset+pitch subsequence DTW between AMT output and score MIDI (`apps/api/src/practice/score_follower.rs`)
- Cross-chunk continuity via FollowerState (last_known_bar)
- Re-anchoring when cost > threshold (student skips ahead or restarts)
- Median onset offset correction isolates true timing deviations from alignment artifacts
- Fuzzy piece matching against catalog (bigram Dice coefficient) with demand tracking for unmatched pieces

**1d. Bar-aligned musical analysis engine -- COMPLETE**

- All 6 dimensions analyzed per chunk (`apps/api/src/practice/analysis.rs`)
- Tier 1 (score context): bar-aligned facts with score + reference comparison
- Tier 2 (no score): absolute MIDI statistics (velocity, IOI, pedal events, note duration)
- Tier 3 (no AMT): scores only (current behavior, graceful degradation)
- Enriched piece_context flows to subagent prompt with `<musical_analysis>` per-dimension facts

**1e. Reference performance cache -- SCRIPT COMPLETE, DATA PENDING**

- Generation script: `model/src/score_library/reference_cache.py` (full DTW alignment, per-bar stats)
- Per-bar statistics: velocity, onset deviation, pedal duration, note duration ratio, performer count
- Requires running on MAESTRO recordings and uploading profiles to R2 (offline job, not yet executed)

### Phase 2: Temporal + Practice Intelligence (2-3 months, after model v2)

**Goal:** System becomes a practice partner, not just a judge.

**2a. Rubato detection**

- Onset deviation analysis with compensatory return check
- Confidence threshold: only flag timing when clearly NOT rubato (ratio > 0.7, deviation > 100ms, persists 2+ phrases)
- Otherwise: silence on timing (trust preservation)

**2b. Passage repetition tracking**

- Detect overlapping bar ranges across chunks (>60% overlap = repetition)
- Track per-attempt scores and improvement trajectory
- "Your pedaling improved from 0.28 to 0.38 across 5 attempts"

**2c. Practice mode auto-detection**

- One-hand (>80% notes above/below C4), slow practice (<60% marked tempo), section drill (same bars 3x+)
- Auto-detect with subtle UI confirmation badge

**2d. Within-session trajectory**

- Warm-up detection (first 3-5 min), fatigue detection (after ~40 min)
- "Consider taking a break -- your articulation is getting less precise"

**2e. Teaching moment upgrades**

- Positive moment detection (breakthroughs, recoveries, passage mastery)
- Novelty constraint (recency penalty per dimension)
- Musical priority weighting (Chopin -> pedaling 2x, Bach -> articulation 2x)

### Phase 4: Real Audio + Expert Labels (6+ months, after model v2)

**Goal:** Model hears real pianos, not synthesized audio.

**4a. Recording collection**

- 2K-5K segments: university partnerships, user opt-in, YouTube, commissioned
- 3+ skill levels, 5+ piano types, 5+ acoustic environments

**4b. Expert annotation campaign**

- 3-5 piano teachers, 6-dim rubric with score context
- Active learning: prioritize uncertain segments
- Cost: ~$50-100K

**4c. Retrain with acoustic diversity**

- Fixes Pianoteq domain gap, calibrates across skill levels
- Unlocks pedal resonance, piano character, room acoustics

### Legacy Waves (superseded by Phase roadmap)

The original Wave 2 (new data + robustness) and Wave 3 (score-conditioned model) have been incorporated into the Phase roadmap above. Wave 2's data collection maps to Phase 4. Wave 3's score conditioning is now handled by Aria's dual-input architecture in Phase 0+3.

---

## Open Research Questions

### Answered

- **Can one representation serve both alignment and quality?** No. MuQ encodes semantic content, not temporal features. Use MuQ for quality, spectral/symbolic for alignment.
- **Can a fine-tuned audio model learn alignment without catastrophic forgetting?** No. A3 showed catastrophic forgetting (R2=0.059 on fold 0). LoRA (<1% params) is the right strategy.
- **Can a MIDI encoder rival MuQ?** Nearly. S2 (71.3%) trails A1 (73.9%) by only 2.6pp (on leaked folds). Gap reflects pretraining scale, not modality choice. Aria (820K MIDI pretraining) should close this gap entirely.
- **Does masterclass feedback transfer to intermediate students?** Partially yes. LLM generates good intermediate-level feedback from 6-dim scores without modification.
- **Does symbolic signal survive AMT?** Yes. Studio: 0% drop. YouTube mediocre audio: 79.9% A1-vs-S2 agreement. AMT is production-viable.
- **Does the model work on phone/mediocre audio?** Pseudo-validated. YouTube AMT test (50 recordings from phone/home setups) shows 79.9% cross-encoder agreement, all dimensions > 72%.
- **Does audio-symbolic fusion help?** No, with S2. ISMIR paper: fusion R2=0.524 < audio-only 0.537. Error correlation r=0.738. Revisiting with Aria (pretrained symbolic FM should decorrelate errors).
- **Do we need to build our own symbolic FM?** No. Aria (EleutherAI, 650M params, 820K MIDIs, Apache 2.0) is SOTA on 6 MIR benchmarks. Phase 3 eliminated.

### Open

- **Aria error decorrelation:** Does Aria's pretrained representation produce errors decorrelated from MuQ? If correlation stays >0.6, fusion still won't help. Critical experiment in Week 5 of model v2 plan.
- ~~**Clean-fold baseline:**~~ **ANSWERED (2026-03-19).** A1-Max on clean piece-stratified folds: 77.5% pairwise with original weights, **79.85% with optimized weights** (4-fold mean). R2 improved from 0.119 to 0.336. Optimized: contrastive=0.6, regression=0.8. Best-checkpoint loading fix also contributed. Only ~1pp below leaked 80.8%, confirming leak inflation was modest (~1pp, not the feared ~3pp).
- **Rubato detection algorithm:** Compensatory return analysis designed (see `04-north-star.md`). Decision: confidence threshold + silence (only flag timing when clearly NOT rubato). Needs validation on real student recordings. Phase 2.
- **Dynamics "amount" vs "appropriateness":** Competition correlation is inverted (rho=-0.917). Aria's score conditioning (delta = z_perf - z_score) should fix this at the model level. Phase 1's bar-aligned analysis partially addresses via LLM reasoning ("score says pp").
- **Annotation noise ceiling:** PercePiano uses crowdsourced IRT. Expert annotation (Phase 4) may reveal model is already near label quality ceiling.
- **Fold variance:** A1 pairwise ranged 70.3-77.7% across leaked folds (~7pp spread). Expect different spread on clean piece-stratified folds. More labeled pieces should stabilize.
- **Real-time AMT quality:** ByteDance AMT validated on pre-recorded audio. Streaming inference at phone-audio quality untested. Phase 1.
- **Score following robustness:** How well does onset-based DTW handle section skips, restarts, and out-of-order practice? Phase 1.
- **Practice mode detection accuracy:** Can AMT reliably distinguish one-hand practice, slow practice, and section drilling? Phase 2.
- **Aria LoRA vs frozen:** How much adaptation does Aria need for quality prediction? Full LoRA fine-tune vs frozen backbone + projection head. Experiment in Week 3-4.

### Product & UX (answered or addressed)

- **What does Sarah want to hear?** Bar-aligned musical facts, not numbers. "The crescendo in bars 12-16 doesn't reach the forte Chopin marked" over "dynamics score 0.35." Phase 1.
- **Curated piece library vs open-ended?** Start with curated (~500 pieces from MAESTRO + ASAP). Degrade gracefully to absolute scoring for unknown pieces. Score-conditioned model (Aria delta) enables fully open-ended.
- **What's the right feedback cadence?** Novelty constraint prevents repeating the same dimension. Positive/corrective ratio target: 25-35% positive. Phase 2.

---

## Data Inventory

For detailed dataset specs, see `01-data.md`.

### Have

| Dataset | Segments | Signal | Status |
|---------|----------|--------|--------|
| PercePiano (T1) | 1,202 | 6-dim composite labels, within-piece ranking pairs | COMPLETE |
| Chopin 2021 competition (T2) | 2,293 | Ordinal placement (11 performers, 3 rounds) | COMPLETE |
| MAESTRO (T3) | 24,321 | Contrastive pairs (204 pieces) | COMPLETE |
| LEGACY: Symbolic pretraining corpus | 24,220 graphs | Link prediction pretraining (replaced by Aria) | LEGACY |
| Intermediate YouTube | 629 | Diagnostic + AMT validation | COMPLETE |
| Masterclass moments | 2,136 | Taxonomy derivation, quote bank | COMPLETE |
| Composite labels | 1,202 | 6 teacher-grounded dimensions | COMPLETE |

### Need (ordered by phase)

**Phase 0+3 (model v2):**

1. T5 YouTube Skill Corpus: remaining 14 pieces (~720 recordings, 2 curated)
2. T2 expansion: Chopin 2015, Cliburn 2022/Amateur, Queen Elisabeth 2024 (~8,700 segments)
3. AMT MIDI extraction for all T2+T5 recordings (Aria input)

**Phase 1 (score infrastructure):**

1. Score MIDI library V2 (MAESTRO + ASAP = ~500 pieces, expand via IMSLP/MuseScore)
2. Reference performance cache (per-bar statistics from MAESTRO professional recordings)

**Phase 4 (real audio):**

1. Diverse skill-level recordings (beginner through advanced, 5+ piano types, 5+ environments)
2. Expert annotations (3-5 teachers, 2K-5K segments, 6-dim rubric with score context)
