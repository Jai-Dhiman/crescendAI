# CrescendAI Research & Product Timeline

**Status:** Historical reference. The phase structure below (Phases 1-7) has been superseded by the implementation slices (`docs/apps/00-10`). The research results, failed experiments, open questions, and data needs remain valuable context.

*Restructured Feb 2025: Organized around the core question -- "How well is the student playing what the score asks for?"*

Target user: Sarah -- 3 years playing, no teacher, records on her phone, wants direction on what to work on next.

North star: Give Sarah one piece of useful feedback on one passage she's working on. Not perfect. Not comprehensive. One thing a teacher would actually say after hearing her play.

---

## Current Results (Jan 2025)

### Audio Foundation Model (MuQ)

- R² = 0.537 on PercePiano (19 quality dimensions)
- 55% improvement over symbolic baseline
- Validated: cross-soundfont, PSyllabus difficulty, MAESTRO zero-shot
- Key limitation: captures piece characteristics more than performer quality

### Contrastive Pairwise Ranking (Phase 1 -- complete)

- Model E2a: 84% overall pairwise accuracy (Kendall's tau = 0.681)
- MuQ baseline: 70.3% -> disentanglement adds +13.7 percentage points
- Best dimensions: drama (90.6%), articulation_length (87.7%)
- Trained on ASAP/MAESTRO same-piece, multi-performer pairs
- Successfully disentangles piece characteristics from performer quality

### Score Alignment Experiments (failed -- lessons learned)

- **Experiment A:** Standard DTW on raw MuQ cosine embeddings
  - Subsampled 5x, Sakoe-Chiba radius=500
  - ~18s weighted mean onset error, only 30% within 30ms threshold
  - 214 ASAP validation performances

- **Experiment B:** Learned MLP projection (1024->512->256) with soft-DTW loss
  - 22 epochs -> representation collapse to ~12s uniform error, 0% within 30ms
  - Failure mode: soft-DTW divergence optimizes global sequence similarity, not temporal correspondence -> MLP collapsed embeddings to narrow region, flattening the cost matrix

- **Core insight:** MuQ layers 9-12 encode slow-changing semantic content ("this sounds like Chopin"), not sharp temporal/harmonic features. These embeddings are fundamentally wrong for frame-level alignment.

- **Experiment C** (measure-level pooling): not run -- pooling frames to measures only loses resolution; the bottleneck is the feature representation itself.

**Conclusion from alignment work:** Don't make one model do everything. Use the right representation for each sub-problem. MuQ embeddings for quality assessment. Spectral/symbolic features for alignment. Combine at the feedback layer.

---

## Revised Phase Structure

> These phases predate the implementation slices (docs/00-10) and represent the original research roadmap. See the implementation slices for current engineering plans.

### Phase 1: Musical Intelligence Layer

*Promotes old Phase 3 (Pedagogy). Now comes first because it's what makes Sarah trust the product and solves the rubato problem.*

**Solves:** Without this, the system flags every expressive choice as an error. Sarah plays rubato, system says "timing is off," Sarah never comes back. The core challenge isn't detecting deviations -- it's distinguishing intentional expression from error and deciding what's worth mentioning.

**Core question:** "How well is the student playing what the score asks for?" NOT: "How much does this deviate from the score?"

**Approach:** Option B -- Priority Signal (selected from three candidates)
- Not "what's wrong" but "what's WORTH MENTIONING"
- A teacher hears ten things, chooses to stop for one
- Learning the filtering/attention function
- This is exactly what prevents rubato from being flagged as error: the system *notices* tempo deviation, *recognizes* it as musically coherent, and *chooses not to mention it*

**Why Option B over A and C:**
- Option A (Dimension Labels): clean ML formulation but imposes artificial structure on fluid teaching. Outputs classification, not prioritization.
- Option C (Transformation Directions): elegant but requires paired examples of same passage with different qualities. Data-hungry and hard to validate.
- Option B matches how teachers actually work: selective, not exhaustive. The filtering function IS the product.

**Musical context encoding** (Jai's domain expertise as technical moat):
- Score-aware interpretation: low dynamics in a pianissimo passage = fine; low dynamics in a fortissimo passage = problem worth mentioning
- Phrase structure awareness: tempo deviation at phrase boundaries = likely intentional; mid-phrase = likely error
- Compensatory patterns: rubato has structure (slow down -> catch up); uncontrolled fluctuation doesn't
- Stylistic norms: Chopin permits different freedoms than Bach

**Data source:** Masterclass recordings (YouTube, Tonebase, conservatory archives)
- Stopping points = implicit "this matters enough to halt momentum"
- Verbal feedback = dimension vocabulary grounded in real teaching
- Sequence of feedback = prioritization signal
- Student level + repertoire = context

**Deliverable:** A working priority model that, given a quality assessment of a passage and its musical context, outputs 1-3 things worth saying -- and correctly stays silent on things that aren't worth mentioning.

**Success metric:** Present system feedback alongside teacher feedback on same performances. Do teachers agree the system mentioned the right thing?

**Open questions:**
- Is teacher mental reference consistent across student levels?
- Does masterclass feedback (advanced students) transfer to Sarah's level?
- What's the minimum viable masterclass dataset to test this?
- How to handle "noticed but didn't mention" -- negative examples are critical for training priority signal but hard to collect
- Can this be bootstrapped with rules/heuristics from Jai's expertise before needing a learned model?

### Phase 2: Coarse Alignment

*Demotes old Phase 2 (Score Alignment). No longer a blocker. No longer requires sub-30ms onset accuracy.*

**Solves:** Need to know roughly where in the piece the student is playing so feedback can be localized to "the transition section" or "measures 20-24."

**Key reframe:** We do NOT need frame-level audio alignment. We need to answer "what part of the piece is this?" at measure or phrase resolution. This is a solved problem that doesn't require novel research.

**Approach (tiered):**
1. User-assisted (MVP): Sarah selects the piece and passage she's working on. Zero alignment needed.
2. Coarse automatic: Chroma/CQT features via librosa -> standard DTW against score. Sub-second accuracy is trivially achievable.
3. Refined (later): If needed, improve with beat-tracking + symbolic alignment.

### Phase 3: Real Audio & Phone Recording Bridge

*Promotes old Phase 5. Now comes before MVP because it's an existential risk.*

**Solves:** All training is on synthetic Pianoteq audio. Sarah records on her iPhone in an apartment with street noise.

**Approach:**
- Noise augmentation during training (room impulse responses, additive noise)
- Evaluate existing MuQ model on phone recordings to measure the gap
- Collect 50-100 phone recordings
- Domain adaptation if gap is significant

**Success metric:** Pairwise ranking accuracy on phone recordings within 10% of synthetic audio performance.

### Phase 4: MVP -- One Useful Piece of Feedback

**Core experience:**
1. Sarah selects piece and passage (user-assisted alignment for v1)
2. Uploads phone recording
3. System evaluates quality across dimensions
4. Musical intelligence layer filters and prioritizes
5. Sarah receives 1-3 pieces of actionable feedback with musical context

**Success metric:** >60% of feedback items rated "useful" or "very useful" by students.

### Phase 5: Temporal Refinement & Automated Alignment

Automatic piece identification, beat-level alignment, multi-scale feedback. Gate: only pursue after validating Phase 4 feedback is useful.

### Phase 6: Student Data & Longitudinal Tracking

Partner with music schools, student recordings with paired teacher annotations, privacy/consent framework, progress tracking, personalization.

### Phase 7: Real-Time Practice Companion (long term)

Seamless open recording session with AI filling gaps between playing with guidance. Key challenge: "when to speak" is harder than "what to say."

---

## Open Research Questions

### Architecture & Representation
- Is there a representation that serves BOTH alignment and quality? (Current evidence says no.)
- Could a fine-tuned audio model learn alignment without catastrophic forgetting of quality signal?
- How much musical nuance do we lose in symbolic space for piano? Pedaling, timbral intent, dynamic envelope -- do these matter for Sarah's level?
- Could we build a piano MIDI encoder that rivals MuQ quality results?

### The Rubato Problem (expression vs. error)
- Rubato has compensatory timing structure. Can this be detected automatically in symbolic space via IOI curves?
- Is the distinction between "intentional rubato" and "uncontrolled fluctuation" learnable, or does it require score context + stylistic knowledge?
- Different periods/composers permit different expressive freedoms. How much stylistic knowledge needs to be encoded explicitly vs learned?

### Pedagogy & Priority
- Is teacher mental reference consistent across student levels?
- Does masterclass feedback (to advanced students) transfer to intermediate students with basic errors?
- What's the minimum viable masterclass dataset to test priority signal?
- Which is more learnable: "what to say" or "when to stay silent"?
- Can Jai's own teaching intuitions be encoded as rules/heuristics to bootstrap the priority model?

### Quality Assessment
- How much of MuQ R²=0.537 reflects recording quality vs playing quality?
- Can contrastive disentanglement be validated on pieces outside ASAP/MAESTRO?
- What quality dimensions matter most for intermediate students?
- Annotation noise in PercePiano: what's the performance ceiling?

### Score Alignment (archived)
- Modern piano transcription gets 95%+ note F1 -- is the audio->MIDI conversion loss meaningful for Sarah's use case?
- Could an expressive rendering (style transfer) solve the synthesized-audio-comparison problem?
- Chroma/CQT DTW for coarse alignment: what's the actual accuracy on student recordings with wrong notes and restarts?

### Product & UX
- What does Sarah actually want to hear? "Your dynamics are flat" vs "try exaggerating the crescendo" vs "listen to how Horowitz shapes this phrase"?
- Curated piece library vs open-ended?
- What's the right feedback cadence?

---

## Data Needs Summary

**Have:**
- PercePiano: 1,202 segments with perceptual ratings (19 dimensions)
- ASAP/MAESTRO: multi-performer data, enables contrastive training
- PSyllabus: difficulty ratings
- Contrastive model E2a: trained, 84% pairwise accuracy

**Need (ordered by priority):**
1. Phone recordings of piano performances (50+ to characterize synthetic->real gap)
2. Masterclass corpus with timestamps + transcriptions (start with 10-20)
3. Teacher validation set (give teachers student recordings, collect their top 1-3 feedback items)
4. Student recordings with paired teacher feedback (partner with music schools)
