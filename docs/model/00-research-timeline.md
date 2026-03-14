# CrescendAI Research & Product Timeline

> **Status (2026-03-14):** Audio encoder COMPLETE (A1-Max deployed, 80.8% ensemble pairwise). Symbolic encoder COMPLETE (S2 GNN, 71.3% pairwise). Layer 1 validation COMPLETE (all gates pass). YouTube AMT validation COMPLETE (79.9% agreement on mediocre audio). Fusion DEFERRED (failed in ISMIR paper). All inference cloud-only via HF endpoint.

*Core question: "How well is the student playing what the score asks for?"*

Target user: Sarah -- 3 years playing, no teacher, records on her phone, wants direction on what to work on next.

North star: Give Sarah one piece of useful feedback on one passage she's working on. Not perfect. Not comprehensive. One thing a teacher would actually say after hearing her play.

---

## Current Results (2026-03-14)

For full encoder details, see `docs/model/03-encoders.md`.

### Encoder Training (complete)

Trained on 1,202 PercePiano segments with 6 teacher-grounded composite dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation). 4-fold piece-stratified CV.

| Model | Modality | Strategy | Pairwise Acc | R2 |
|-------|----------|----------|-------------|-----|
| **A1-Max (ensemble)** | **Audio** | **MuQ + LoRA rank-32, ListMLE, CCC** | **80.8%** | **0.50** |
| A1 | Audio | MuQ + LoRA rank-16 | 73.9% | 0.40 |
| A2 | Audio | MuQ staged adaptation | 71.4% | 0.42 |
| A3 | Audio | MuQ full unfreeze | 69.9% | 0.28 |
| **S2** | **Symbolic** | **GNN on score graph** | **71.3%** | 0.32 |
| S2H | Symbolic | Heterogeneous GNN | 70.2% | 0.36 |
| S3 | Symbolic | CNN + Transformer | 70.0% | 0.37 |
| S1 | Symbolic | Transformer on REMI tokens | 68.4% | 0.33 |

**Deployed:** A1-Max 4-fold ensemble on HF inference endpoint (cloud-only). **Winners:** A1-Max (audio), S2 (symbolic).

### Per-Dimension Complementarity

| Dimension | A1 (Audio) | S2 (GNN) | Winner |
|-----------|-----------|----------|--------|
| dynamics | ~0.70 | ~0.77 | S2 |
| timing | ~0.77 | ~0.65 | A1 |
| pedaling | ~0.72 | ~0.72 | Tie |
| articulation | ~0.66 | ~0.70 | S2 |
| phrasing | ~0.63 | ~0.63 | Tie |
| interpretation | ~0.74 | ~0.77 | S2 |

Audio wins timing decisively. Symbolic wins dynamics, articulation, and interpretation.

### Layer 1 Validation (complete)

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
- Dynamics has *negative* correlation with placement -- model captures "amount" not "appropriateness." Needs score conditioning (Wave 3) or better labels (Wave 2) to fix
- Raw MIDI statistics hurt the LLM (judge called them "false precision"), but bar-aligned passage-specific context is untested and remains the right goal for Wave 3

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

## Key Decisions

### Inference: Cloud-Only (decided 2026-03-14)

All MuQ inference runs on the HuggingFace inference endpoint. No on-device ML / Core ML conversion planned. This simplifies the iOS architecture -- the app captures audio and sends chunks to the API worker, which forwards to HF and runs STOP classifier logic server-side.

### Fusion: Deferred (decided 2026-03-14)

The ISMIR paper (`paper/ismir_v2/main.tex`) tested audio-symbolic fusion and found it underperforms audio-only:
- Fusion R2 = 0.524 vs audio-only R2 = 0.537
- Error correlation between modalities: r = 0.738 -- both fail on the same samples
- Root cause: audio's advantage is pretrained inductive bias (MuQ pretrained on 160K hours), not information content. Both derive from identical MIDI source data.
- **Implication:** Fusion becomes viable when symbolic foundation models (pretrained on large MIDI corpora) close the pretraining gap. See `04-north-star.md`.

### Phone Audio: Pseudo-Validated (decided 2026-03-14)

The YouTube AMT validation (79.9% agreement on 50 mediocre recordings) serves as proxy phone audio validation. These recordings include phone-quality audio, home pianos, digital keyboards, and varying room acoustics -- representative of real user conditions. No longer treated as an "existential risk." Formal paired recordings (studio + iPhone) remain a Wave 2 nice-to-have for quantifying the exact gap.

---

## Training Roadmap

### Wave 1: Push Encoders (current)

**Goal:** Maximize both encoders independently.

**2a. A1-Max: audio encoder improvements -- COMPLETE**
- ListMLE ranking loss, CCC regression, mixup, hard negatives, LoRA sweep (18 configs x 4 folds)
- Result: 80.8% ensemble pairwise (+6.8pp vs A1 baseline). Deployed on HF endpoint.

**2b. Data preparation -- COMPLETE**
- GIANTMIDI-Piano: 8,278 graphs
- MAESTRO contrastive: 24,321 MuQ embeddings, 1,123 per-recording graphs
- Total pretrain corpus: 24,220 graphs across 6 sources

**2c. S2-Max: symbolic encoder with expanded data**
- Pretrain on 24,220 graphs (link prediction) -- up from 14,821
- Finetune on PercePiano + MAESTRO contrastive + Competition ordinal
- Architecture exploration: edge-type embedding in GATConv attention, multi-scale graph pooling

**2d. AMT validation v2: harder test on improved S2**
- YouTube AMT: COMPLETE (79.9% agreement). MAESTRO AMT: COMPLETE (0% drop).
- Remaining: test improved S2-Max after pretraining on 24K graphs

### Wave 2: New Data + Robustness (months)

**Goal:** Break through the data bottleneck. Address skill level bias, recording condition bias, and label quality.

**3a. Diverse skill-level recordings**
- Beginner through advanced, real practice conditions (apartments, studios, various pianos)
- Addresses dynamic range narrowness (d=0.47 between intermediate and professional)
- Addresses PercePiano advanced-level bias

**3b. Expert annotation campaign**
- 3-5 piano teachers annotate 6 dimensions using rubric from `composite_labels/teacher_rubric.json`
- Target: 2,000-5,000 annotated segments across 200+ pieces
- Active learning: use A1/S2 uncertainty to prioritize annotation

**3c. Retrain both encoders on expanded data**
- Should fix dynamics inversion (negative competition correlation)
- Should fix label quality issues (PercePiano's crowdsourced IRT vs expert annotation)

### Wave 3: Score-Conditioned Model (months-year)

**Goal:** The highest-ceiling architecture. Model learns quality relative to what the score asks for. See `04-north-star.md` for the full vision.

**Prerequisites:** Score alignment pipeline operational. 2K+ labeled (performance, score, quality) triples.

**4a. Score alignment pipeline**
- Chroma DTW for measure-level alignment
- Bar-number mapping for teacher context

**4b. Timing direction as cheap teacher context win**
- Extract systematic rushing/dragging from onset deviations
- Single flag in teacher prompt -- implementable without full score alignment

**4c. Score-conditioned architecture**
- `quality = f(z_performance_audio, z_score_midi)`

**4d. Bar-aligned LLM context**
- Passage-specific, musically grounded feedback: "In bars 12-16, velocity drops to pp but the score asks for mf crescendo"

---

## Open Research Questions

### Answered

- **Can one representation serve both alignment and quality?** No. MuQ encodes semantic content, not temporal features. Use MuQ for quality, spectral/symbolic for alignment.
- **Can a fine-tuned audio model learn alignment without catastrophic forgetting?** No. A3 showed catastrophic forgetting (R2=0.059 on fold 0). LoRA (<1% params) is the right strategy.
- **Can a MIDI encoder rival MuQ?** Nearly. S2 (71.3%) trails A1 (73.9%) by only 2.6pp. Gap reflects pretraining scale, not modality choice.
- **Does masterclass feedback transfer to intermediate students?** Partially yes. LLM generates good intermediate-level feedback from 6-dim scores without modification.
- **Does symbolic signal survive AMT?** Yes. Studio: 0% drop. YouTube mediocre audio: 79.9% A1-vs-S2 agreement. AMT is production-viable.
- **Does the model work on phone/mediocre audio?** Pseudo-validated. YouTube AMT test (50 recordings from phone/home setups) shows 79.9% cross-encoder agreement, all dimensions > 72%.
- **Does audio-symbolic fusion help?** No, currently. ISMIR paper: fusion R2=0.524 < audio-only 0.537. Error correlation r=0.738. Deferred until symbolic foundation models exist.

### Open

- **Rubato detection:** Compensatory timing structure vs uncontrolled fluctuation. Requires score context. Wave 3.
- **Dynamics "amount" vs "appropriateness":** Competition correlation is inverted. Score conditioning (Wave 3) or better labels (Wave 2) needed.
- **Annotation noise ceiling:** PercePiano uses crowdsourced IRT. Expert annotation (Wave 2) may reveal model is near label quality ceiling.
- **Edge-type information without overfitting:** Can edge-type embeddings in GATConv attention capture S2H's information without parameter explosion?
- **Fold variance:** A1 pairwise ranges 70.3-77.7% across folds (~7pp spread). More labeled pieces should stabilize.

### Product & UX

- What does Sarah actually want to hear? "Your dynamics are flat" vs "try exaggerating the crescendo" vs "listen to how Horowitz shapes this phrase"?
- Curated piece library vs open-ended? (Score-conditioned model in Wave 3 enables open-ended.)
- What's the right feedback cadence?

---

## Data Inventory

For detailed dataset specs, see `01-data.md`.

### Have

| Dataset | Segments | Signal | Status |
|---------|----------|--------|--------|
| PercePiano (T1) | 1,202 | 6-dim composite labels, within-piece ranking pairs | COMPLETE |
| Chopin 2021 competition (T2) | 2,293 | Ordinal placement (11 performers, 3 rounds) | COMPLETE |
| MAESTRO (T3) | 24,321 | Contrastive pairs (204 pieces) | COMPLETE |
| Symbolic pretraining corpus | 24,220 graphs | Link prediction pretraining | COMPLETE |
| Intermediate YouTube | 629 | Diagnostic + AMT validation | COMPLETE |
| Masterclass moments | 2,136 | Taxonomy derivation, quote bank | COMPLETE |
| Composite labels | 1,202 | 6 teacher-grounded dimensions | COMPLETE |

### Need (ordered by training wave)

**Wave 2 (new recordings, months):**
1. Diverse skill-level recordings (beginner through advanced, real conditions)
2. Expert annotations (3-5 teachers, 2K+ segments, 6-dim rubric)

**Wave 3 (score alignment, months):**
1. Score alignment pipeline (chroma DTW, bar-level mapping)
2. Score-performance-quality triples (audio + score MIDI + labels)
