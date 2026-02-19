# Teacher-Grounded Feedback Taxonomy Design

## Motivation

The PercePiano data audit (2026-02-15) revealed three structural problems with training on the 19 PercePiano quality dimensions:

1. **Redundancy:** 31/171 dimension pairs have |r| > 0.7. PCA shows only 4 statistically significant factors, with PC1 (a "quality halo") capturing 47% of variance alone.
2. **Low audibility:** Even the best nonlinear probe (MLP) achieves R2 = +0.28 at best. Most dimensions are barely predictable from MuQ audio embeddings.
3. **Misaligned feedback:** The 19 dimensions don't map to what piano teachers actually tell students. Predicting "timbre_depth" or "sophistication" doesn't produce actionable guidance.

Meanwhile, raw MuQ embeddings (AUC 0.936) massively outperform PercePiano labels (AUC 0.814) for STOP prediction. The labels are a lossy, noisy bottleneck -- but we need them for interpretability: the model must explain *why* the teacher would stop, not just *that* they would.

The solution is to replace the 19 PercePiano dimensions with a smaller set of feedback categories derived from what teachers actually say, validated against multiple signals.

## Goals

1. Scale the masterclass dataset to 1000+ categorized STOP moments
2. Empirically derive 5-8 feedback dimensions from teacher behavior
3. Validate dimensions against model predictability, STOP utility, and PercePiano coverage
4. Build a bridge from PercePiano labels to the new taxonomy for training data
5. Produce a spec for rewriting the model improvement training plan (Part 2, separate design)

## Non-Goals (Deferred to Part 2)

- Rewriting the model improvement training plan
- Changing audio/symbolic encoder architecture
- Any model training or fine-tuning
- Production feedback UI

## Data Scaling Strategy

### Current State

- 65 extracted teaching moments from ~10 processed videos
- ~50 curated source videos in `sources.yaml`
- Pipeline: YouTube download -> Whisper transcription -> audio segmentation -> GPT-4o extraction
- Extraction currently forces moments into 10 predefined categories (defined in `config.rs`)

### Target

- 1000+ teaching moments
- 150+ source videos
- 30+ teachers, 50+ pieces, 15+ composers
- Balanced across teaching styles (demonstrative vs verbal, technical vs interpretive)

### Phase A: Process Existing Sources

Run the pipeline on all ~50 curated sources. Expected yield: ~350 moments.

No engineering changes needed.

### Phase B: Expand Source Library

Scale to 150+ videos by mining:

- High-yield channels: Berliner Klavierfesttage, Juilliard Master Classes, tonebase Piano, JMC
- Underrepresented eras: Baroque (Bach, Handel), 20th century (Bartok, Messiaen, Ligeti)
- Deliberately diverse teaching styles

Diversity targets: 30+ teachers, 50+ pieces, 15+ composers.

### Phase C: Open-Ended Re-Extraction

This is the critical methodological change. The current pipeline constrains extraction to 10 predefined categories, which creates a circular dependency: we can't discover what teachers talk about if we pre-decide the categories.

**Two-pass extraction:**

**Pass 1 (open-ended):** Re-extract all moments with a modified LLM prompt that asks for free-text description of the musical aspect being addressed. No predefined categories. Prompt: "What musical aspect is the teacher commenting on? Describe in 2-5 words."

**Pass 2 (clustering):** After collecting 1000+ open-ended descriptions, cluster them to discover natural categories. The existing 10 categories from `config.rs` become a hypothesis to validate, not a constraint.

## Dimension Derivation Methodology

### Step 1: Bottom-Up Clustering

- Embed the 1000+ free-text descriptions using a sentence transformer
- Cluster with HDBSCAN (data-driven cluster count, surfaces outliers)
- Manual review and naming of clusters
- Expected: 8-15 raw clusters, some merging on review

### Step 2: Multi-Signal Scoring

For each candidate dimension, compute four scores:

| Signal | Weight | Method | Meaning |
|--------|--------|--------|---------|
| Teacher frequency | Primary | Cluster size / total moments | How often teachers address this |
| MuQ predictability | Soft | Map to nearest PercePiano dims, use MLP probing R2 from audit | Can the audio model detect this? |
| STOP contribution | Soft | STOP AUC with vs without this dimension's PercePiano proxy | Does it help explain teacher stops? |
| PercePiano grounding | Softest | Correlation between cluster membership and PercePiano scores | Is there labeled training data? |

### Step 3: Selection Criteria

**Keep** a dimension if:

- Teacher frequency > 5% of moments (teachers care about it)
- AND at least one soft signal is positive (model can learn it OR it predicts STOP)

**Drop** a dimension if:

- Teacher frequency < 3% AND no soft signal supports it
- OR fully redundant with another kept dimension (r > 0.85 on PercePiano proxies)

### Step 4: Hierarchy Construction

Group surviving dimensions into 3-5 top-level categories for summary feedback, with 1-3 specific dimensions underneath for drill-down. Grouping derived from the clustering dendrogram.

Speculative example (to be validated by data):

```
Sound Quality
  - dynamics
  - tone_color
Musical Shaping
  - phrasing
  - timing
Technical Control
  - pedaling
  - articulation
Interpretive Choices
  - interpretation
  - voicing
```

### Step 5: Quote Bank Construction

For each final dimension, collect from teaching moments:

- 5-10 representative teacher quotes (feedback_summary + transcript excerpt)
- Organized by severity and feedback_type
- These power the feedback delivery: "Here's how Arie Vardi addressed pedaling in Chopin Ballade No. 1..."

## PercePiano Bridge

### Problem

The teacher-derived dimensions are grounded in pedagogy but lack dense training labels. PercePiano has 1,202 labeled segments but uses a different (redundant, noisy) dimension space. We need to translate.

### Approach: Soft Label Aggregation

For each teacher-derived dimension, create a composite PercePiano score:

1. Identify which of the 19 PercePiano dims load onto this teacher category (via correlation structure from audit + semantic alignment)
2. Weight by MLP probing R2 (predictable dims get higher weight -- the model can actually learn from them)
3. Average into a single composite score per teacher dimension

Example for a hypothetical "Sound Quality" composite:

```
sound_quality = w1 * dynamic_range + w2 * timbre_depth + w3 * timbre_variety + w4 * timbre_loudness
```

Where weights are proportional to max(0, MLP_R2) and normalized.

### Dimensions Without PercePiano Proxy

If a teacher dimension has no PercePiano mapping (e.g., "technique", "structure"):

- Use masterclass STOP moments as weak binary labels (teacher flagged this vs didn't)
- Accept sparse supervision for this dimension in Part 2
- Potentially rely on the symbolic encoder for score-readable aspects

### Validation

Re-run the STOP prediction experiment from the audit using new composite dimensions. Gate: composite dim STOP AUC >= 0.80 (within ~0.02 of the 19-dim baseline of 0.814).

## Deliverables

1. **Expanded masterclass dataset** -- 1000+ STOP moments with open-ended descriptions, 150+ videos, 30+ teachers, 15+ composers
2. **Empirical dimension taxonomy** -- 5-8 dimensions in a 2-level hierarchy, each with a multi-signal score card
3. **PercePiano bridge** -- Weighted mapping producing composite labels for all 1,202 training segments
4. **Quote bank** -- 5-10 real teacher quotes per dimension, organized by severity and feedback type
5. **Validation report** -- STOP prediction AUC, independence checks, coverage analysis

## Validation Gates

All must pass before Part 2 (training plan rewrite) begins.

| Gate | Criterion | Rationale |
|------|-----------|-----------|
| Data sufficiency | 1000+ moments, no category < 30 moments | Enough to trust clustering |
| STOP preservation | Composite dim STOP AUC >= 0.80 | Don't lose predictive power |
| Independence | No pair of final dims with r > 0.80 | Avoid PercePiano's redundancy |
| Actionability | Every dim maps to >= 5 real teacher quotes | Must be explainable |
| Coverage | Final dims cover >= 80% of teacher moments | Taxonomy shouldn't orphan most feedback |

## What Part 2 Receives

A spec: "Train your model to predict these N dimensions. Here are composite labels for PercePiano. Here are the aggregation weights. Here's what each modality should own. Here's the STOP AUC to beat."
