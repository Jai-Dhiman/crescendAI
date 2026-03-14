# CrescendAI Research & Product Timeline

> **Status (2026-03-13):** Layer 1 validation COMPLETE (all gates pass). YouTube AMT validation COMPLETE (79.9% A1-vs-S2 agreement on mediocre audio -- symbolic path confirmed viable). A1-max training IN PROGRESS. Wave 1 data prep COMPLETE (24,220 graphs, MAESTRO metadata + per-recording graphs, Thunder Compute stubs ready). Next: S2-max pretraining, fusion experiments, Core ML conversion.

*Core question: "How well is the student playing what the score asks for?"*

Target user: Sarah -- 3 years playing, no teacher, records on her phone, wants direction on what to work on next.

North star: Give Sarah one piece of useful feedback on one passage she's working on. Not perfect. Not comprehensive. One thing a teacher would actually say after hearing her play.

---

## Current Results (2026-03-11)

For full training details and per-fold breakdowns, see `docs/model/04-training-results.md`.

### Encoder Training (complete)

Trained on 1,202 PercePiano segments with 6 teacher-grounded composite dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation). 4-fold piece-stratified CV.

| Model | Modality | Strategy | Pairwise Acc | R2 |
|-------|----------|----------|-------------|-----|
| **A1** | **Audio** | **MuQ + LoRA rank-16** | **73.9%** | **0.40** |
| A2 | Audio | MuQ staged adaptation | 71.4% | 0.42 |
| A3 | Audio | MuQ full unfreeze | 69.9% | 0.28 |
| **S2** | **Symbolic** | **GNN on score graph** | **71.3%** | 0.32 |
| S2H | Symbolic | Heterogeneous GNN | 70.2% | 0.36 |
| S3 | Symbolic | CNN + Transformer | 70.0% | 0.37 |
| S1 | Symbolic | Transformer on REMI tokens | 68.4% | 0.33 |

**Winners:** A1 (audio), S2 (symbolic). Gap is only 2.6 percentage points.

### Per-Dimension Complementarity

| Dimension | A1 (Audio) | S2 (GNN) | Winner |
|-----------|-----------|----------|--------|
| dynamics | ~0.70 | ~0.77 | S2 |
| timing | ~0.77 | ~0.65 | A1 |
| pedaling | ~0.72 | ~0.72 | Tie |
| articulation | ~0.66 | ~0.70 | S2 |
| phrasing | ~0.63 | ~0.63 | Tie |
| interpretation | ~0.74 | ~0.77 | S2 |

Audio wins timing decisively. Symbolic wins dynamics, articulation, and interpretation. This validates gated per-dimension fusion (F3).

### Layer 1 Validation (complete)

Four experiments validating assumptions before investing in further model work. For detailed results, see `04-training-results.md` Layer 1 section.

| Experiment | Gate | Result | Decision |
|-----------|------|--------|----------|
| Competition correlation (Chopin 2021, n=11) | rho > 0.3 | rho=+0.704 | **PASS** -- model quality signal is real |
| AMT degradation (ByteDance vs GT MIDI) | drop < 10% | 0.0% drop | **PASS** -- symbolic path viable through AMT. YouTube follow-up: 79.9% A1-vs-S2 agreement on 50 mediocre recordings |
| Dynamic range (intermediate vs professional) | diagnostic | Cohen's d=0.47 | Usable for within-student tracking |
| MIDI-as-context feedback (LLM judge) | B wins > 65% | B wins 45% | **SKIP** -- raw MIDI stats don't help LLM |

**Key insights from Layer 1:**
- Pedaling (rho=0.887) and phrasing (rho=0.803) are strongest predictors of competition placement
- Dynamics has *negative* correlation with placement -- model captures "amount" not "appropriateness." Needs score conditioning (Wave 3) or better labels (Wave 2) to fix
- AMT 0% drop confirmed at scale: YouTube follow-up (50 recordings, 1,225 pairs) shows 79.9% A1-vs-S2 agreement on mediocre audio
- Raw MIDI statistics hurt the LLM (judge called them "false precision"), but bar-aligned passage-specific context is untested and remains the right goal for Wave 3

### Prior Results (historical context)

**Early baselines (Jan 2025, 19 dimensions):**
- MuQ regression: R2=0.537 on 19 redundant PercePiano dimensions (PCA showed only 4 significant factors)
- Contrastive ranking (E2a): 84% pairwise on 19 dims -- inflated by redundancy and "quality halo" (PC1 = 47% variance)
- Current 6-dim results (73.9% pairwise) are on independent, teacher-grounded dimensions -- lower numbers, higher signal

**Score alignment experiments (failed):**
- Standard DTW on MuQ embeddings: ~18s onset error (MuQ encodes semantic content, not temporal features)
- Learned MLP projection: representation collapse
- **Conclusion:** Use the right representation for each sub-problem. MuQ for quality assessment. Spectral/symbolic features for alignment. Combine at the feedback layer. Confirmed by Experiment 4 (MIDI-as-context): raw alignment stats don't help the LLM; bar-level musical context is needed.

---

## Training Roadmap

Three training waves, each validating the assumptions of the next.

### Wave 1: Push Both Encoders + Fusion (current)

**Goal:** Maximize both encoders, test fusion, convert to Core ML.

**2a. A1-max: audio encoder quick wins (IN PROGRESS)**
- Hard negative mining for pair sampling (expected +3-5% pairwise)
- Listwise ranking loss (ListMLE/LambdaRank, expected +2-4%)
- LoRA rank and layer ablation ({4, 8, 16, 32, 64} x layer ranges)
- Label smoothing, mixup on embeddings, loss weight tuning
- Target: push A1 from 73.9% toward ~78-80% pairwise

**2b. Data preparation (COMPLETE locally, Thunder Compute stubs ready)**
- GIANTMIDI-Piano: COMPLETE (8,278 graphs in shards 0075-0240)
- MAESTRO contrastive: COMPLETE (24,321 MuQ embeddings, metadata.jsonl, 1,123 per-recording graphs in shards 0241-0263, contrastive mapping for 204 pieces)
- Total pretrain corpus: 24,220 graphs across 6 sources (was 14,821)
- Competition MIDI: STUB READY (`scripts/transcribe_competition_midi.py`, needs Thunder Compute GPU for ByteDance transcription of 66 Chopin 2021 recordings)
- Harder AMT test: STUB READY (`scripts/build_amt_test_set.py`, needs Thunder Compute GPU for dual-system transcription of 25 selected pieces)

**2c. S2-max: symbolic encoder with expanded data**
- Pretrain on 24,220 graphs (link prediction) -- up from 14,821
- Finetune on PercePiano + MAESTRO contrastive + Competition ordinal (3 data sources, up from 1)
- Architecture exploration: edge-type embedding in GATConv attention (S2H's information without its overfitting), multi-scale graph pooling

**2d. AMT validation v2: harder test on improved S2**
- YouTube AMT validation (2026-03-13): 50 recordings, 1,225 pairs, 79.9% A1-vs-S2 agreement. All dimensions pass. See `04-training-results.md`.
- Remaining: test with improved S2-max after pretraining on 24K graphs, 2 AMT systems, close-quality pairs
- Gate: per-dimension drop < 10% still holds
- If fails: reassess symbolic path complexity vs. audio-only + LLM reasoning

**2e. F3 fusion: gated per-dimension routing**
- Freeze A1-max + S2-max encoders, train only fusion module on PercePiano
- Initialize gates from per-dimension findings (route timing->audio, dynamics/articulation/interpretation->symbolic)
- Compare: A1-max alone vs. A1-max + S2-max fused
- Decision: is the symbolic path worth production complexity (two encoder paths, AMT on device)?

**2f. Core ML conversion of winner**
- A1-max alone is simplest path (mostly frozen MuQ + small LoRA adapters)
- If fusion proves valuable, convert fused model as second step (adds AMT dependency on device)

### Wave 2: New Data + Robustness (months)

**Goal:** Break through the data bottleneck. Address skill level bias, recording condition bias, and label quality.

**3a. Phone audio paired recordings (start early -- no annotation needed)**
- 30-50 pieces, studio mic + iPhone simultaneous recording
- Tests full production chain: phone mic -> MuQ (A1) and phone -> AMT -> S2
- Highest-risk item for product -- existential if phone audio degrades quality signal
- Success metric: pairwise ranking accuracy on phone recordings within 10% of studio

**3b. Diverse skill-level recordings**
- Beginner through advanced, real practice conditions (apartments, studios, various pianos)
- Addresses dynamic range narrowness (d=0.47 between intermediate and professional)
- Addresses PercePiano advanced-level bias (all training data is advanced performances)
- Start with PercePiano's pieces for cross-validation against existing labels

**3c. Expert annotation campaign**
- 3-5 piano teachers annotate 6 dimensions using rubric from `composite_labels/teacher_rubric.json`
- Target: 2,000-5,000 annotated segments across 200+ pieces
- More pieces with multiple performances = quadratic growth in ranking pairs
- Active learning: use A1/S2 uncertainty (high prediction variance, regression/ranking disagreement) to prioritize annotation
- Inter-rater reliability measurement (currently unknown)
- Include negative examples: "nothing worth mentioning" for STOP classifier

**3d. Retrain both encoders on expanded data**
- Should fix dynamics inversion (negative competition correlation) -- current labels conflate "amount" with "quality"
- Should fix label quality issues (PercePiano's crowdsourced IRT vs expert annotation)
- Phone audio domain adaptation if gap > 10%: simulation augmentation or domain-adversarial training with gradient reversal

### Wave 3: Score-Conditioned Model (months-year)

**Goal:** The highest-ceiling architecture. Model learns quality relative to what the score asks for.

**Prerequisites:** Score alignment pipeline operational. 2K+ labeled (performance, score, quality) triples. Working encoder paths from Wave 1.

**4a. Score alignment pipeline**
- Chroma DTW for measure-level alignment (sub-second accuracy, solved problem)
- Bar-number mapping for teacher context
- User-assisted for MVP (Sarah selects piece and passage), automatic later

**4b. Timing direction as cheap teacher context win**
- Extract systematic rushing/dragging from onset deviations
- Single flag in teacher prompt -- Exp 4 showed the LLM judge found timing direction data consistently valuable, even when other MIDI stats were rejected as "false precision"
- Implementable without full score alignment

**4c. Score-conditioned architecture**
- `quality = f(z_performance_audio, z_score_midi)`
- Performance embedding from A1 (or fused model). Score embedding from S2 or dedicated encoder.
- Cross-attention or gated fusion between performance and score representations

**4d. Bar-aligned LLM context**
- The RIGHT version of Exp 4's concept: passage-specific, musically grounded
- Not: "velocity MAE = 15, onset deviation = 40ms"
- Instead: "In bars 12-16, velocity drops to pp but the score asks for mf crescendo"
- Enables: score-aware dynamics feedback, rubato detection, difficulty-aware teaching

**What Wave 3 enables:**
- "The dynamics are correct for this passage" vs "the dynamics don't match what Chopin wrote"
- Rubato detection: deviation from score timing + compensatory pattern = intentional expression
- Difficulty-aware feedback: "this passage is technically demanding, focus on X"
- Open-ended piece support (any piece with available MIDI score)

---

## Open Research Questions

### Answered

- **Can one representation serve both alignment and quality?** No. Confirmed by alignment failure (MuQ encodes semantic content, not temporal features) and by Exp 4 (raw MIDI stats don't help the LLM -- separate representations needed). Use MuQ for quality, spectral/symbolic for alignment.
- **Can a fine-tuned audio model learn alignment without catastrophic forgetting?** No. A3 (full unfreeze) showed catastrophic forgetting (R2=0.059 on fold 0). LoRA's minimal adaptation (A1, <1% params) is the right strategy.
- **Can a MIDI encoder rival MuQ?** Nearly. S2 (71.3% pairwise) trails A1 (73.9%) by only 2.6pp. May close with more pretraining data (GIANTMIDI) and multi-source finetuning.
- **Does masterclass feedback transfer to intermediate students?** Partially yes. 6-dim taxonomy derived from 2,136 advanced masterclass moments. Exp 4 showed the LLM generates good intermediate-level feedback from these dimensions without modification.
- **Does symbolic signal survive AMT?** Yes. Studio audio: 0% pairwise drop (ByteDance on 107 pairs, 4 MAESTRO pieces). YouTube mediocre audio: 79.9% A1-vs-S2 agreement across 50 recordings, 1,225 pairs, all 6 dimensions above 72%. AMT is production-viable.

### Open

- **Rubato detection:** Compensatory timing structure (slow down -> catch up) vs uncontrolled fluctuation. Requires score context + stylistic knowledge. Wave 3 score-conditioned model is the planned approach. Can IOI curve analysis in symbolic space provide a useful intermediate signal?
- **Dynamics "amount" vs "appropriateness":** Competition correlation shows dynamics is inverted (better players score lower). The model captures how much dynamic variation, not whether dynamics match the score. Score conditioning (Wave 3) or better labels (Wave 2) needed.
- **Phone audio gap:** All training is on Pianoteq-rendered audio. Phone recordings are an existential risk. 30-50 paired recordings (Wave 2, item 3a) will quantify this.
- **Annotation noise ceiling:** PercePiano uses crowdsourced IRT. What's the ceiling on prediction quality with these labels? Expert annotation (Wave 2, item 3c) may reveal that current model performance is near the label quality ceiling.
- **Edge-type information without overfitting:** S2H (heterogeneous GNN, 4 edge types) had better R2 but worse pairwise than S2 (homogeneous). Can edge-type embeddings in GATConv attention capture this information without the parameter explosion?
- **Fold variance:** A1 pairwise ranges 70.3-77.7% across folds (~7pp spread). Driven by which of 61 multi-performance pieces land in validation. More labeled pieces (Wave 2) should stabilize this.

### Product & UX

- What does Sarah actually want to hear? "Your dynamics are flat" vs "try exaggerating the crescendo" vs "listen to how Horowitz shapes this phrase"? (Exp 4 suggests the LLM already produces good pedagogical language from dimension scores alone.)
- Curated piece library vs open-ended? (Score-conditioned model in Wave 3 enables open-ended.)
- What's the right feedback cadence?

---

## Data Inventory & Needs

### Have

| Dataset | Segments | Signal | Status |
|---------|----------|--------|--------|
| PercePiano (T1) | 1,202 | 6-dim composite labels, within-piece ranking pairs | COMPLETE, on disk |
| Chopin 2021 competition (T2) | 2,293 | Ordinal placement (11 performers, 3 rounds) | COMPLETE (audio embeddings), MIDI transcription stub ready |
| MAESTRO (T3) | 24,321 | Contrastive pairs (204 pieces, 12,181 segments) | COMPLETE (embeddings + metadata + 1,123 per-recording graphs) |
| Symbolic pretraining corpus | 24,220 graphs | Link prediction pretraining | COMPLETE, on disk (shards 0000-0263) |
| Intermediate YouTube | 629 | Diagnostic (dynamic range analysis) + AMT validation (51 transcribed MIDI) | COMPLETE |
| Masterclass moments | 2,136 | Taxonomy derivation, quote bank | COMPLETE |
| Composite labels | 1,202 | 6 teacher-grounded dimensions | COMPLETE |

### Need (ordered by training wave)

**Wave 1 (data prep):**
1. ~~GIANTMIDI-Piano MIDI files -> graph conversion~~ COMPLETE (8,278 graphs)
2. ~~MAESTRO contrastive pairs~~ COMPLETE (24,321 embeddings, metadata, 1,123 per-recording graphs)
3. Competition MIDI: STUB READY, needs Thunder Compute run (66 Chopin 2021 recordings)
4. Harder AMT test: STUB READY, needs Thunder Compute run (25 pieces, 2 AMT systems)

**Wave 2 (new recordings, months):**
1. Phone audio paired recordings (30-50 pieces, studio + iPhone simultaneous)
2. Diverse skill-level recordings (beginner through advanced, real conditions)
3. Expert annotations (3-5 teachers, 2K+ segments, 6-dim rubric)

**Wave 3 (score alignment, months):**
1. Score alignment pipeline (chroma DTW, bar-level mapping)
2. Score-performance-quality triples (audio + score MIDI + labels)
