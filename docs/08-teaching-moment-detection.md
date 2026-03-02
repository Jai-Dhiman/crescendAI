# Slice 4: Teaching Moment / Priority Logic

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Given a session of analyzed audio chunks, identify which chunks are "teaching moments" (where a teacher would intervene) and which dimension to surface as the blind spot.

**Architecture:** Deploy the existing STOP classifier from masterclass experiments. Each chunk's MuQ embeddings or 6-dim scores are scored for STOP probability. When the student asks "how was that?", rank chunks by STOP probability, select the top moment, and identify the most relevant dimension.

**Tech Stack:** Existing STOP classifier (logistic regression), Python inference, MuQ embeddings

---

## Context

The masterclass experiments already built and validated:
- 1,707 labeled teaching moments (STOP/CONTINUE)
- STOP classifier: AUC 0.936 on MuQ embeddings, AUC 0.845 on 6-dim composite scores
- 6 validated dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Quote bank of 60 real teacher quotes

The task is to deploy this in the practice companion context: scoring each chunk during a session and surfacing the top teaching moment when the student asks.

## Design

### STOP Scoring Per Chunk

Two options, in order of preference:

**Option A: Score on MuQ embeddings directly (AUC 0.936)**
- The HF inference endpoint already extracts MuQ embeddings
- Add the STOP classifier weights to the inference handler
- Return STOP probability alongside dimension scores per chunk
- Pro: highest accuracy. Con: requires modifying the HF inference handler.

**Option B: Score on 6-dim composite scores (AUC 0.845)**
- The 6 dimension scores are already returned by the inference endpoint
- Run STOP classifier on the server side (Cloudflare Workers)
- Logistic regression is trivial to implement in Rust: `sigmoid(dot(weights, scores) + bias)`
- Pro: no changes to inference handler. Con: lower accuracy.

**Recommendation: Start with Option B** (simple, no HF changes), upgrade to Option A if needed.

### Logistic Regression in Rust

The STOP classifier is a logistic regression with 6 features (or 2048 for MuQ embeddings). For Option B:

```rust
struct StopClassifier {
    weights: [f64; 6],  // one per dimension
    bias: f64,
}

impl StopClassifier {
    fn predict(&self, dimensions: &[f64; 6]) -> f64 {
        let logit: f64 = self.weights.iter()
            .zip(dimensions.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>() + self.bias;
        1.0 / (1.0 + (-logit).exp())  // sigmoid
    }
}
```

Weights are extracted from the trained sklearn model in `model/src/masterclass_experiments/models.py` and hardcoded (6 floats + 1 bias).

### Teaching Moment Selection (when student asks "how was that?")

1. Collect all chunks from the current session (or since last query)
2. Each chunk has: 6-dim scores + STOP probability
3. **Filter:** Only consider chunks with STOP probability > 0.5 (would a teacher actually stop?)
4. **Rank:** Sort by STOP probability descending
5. **Select:** Take the top-1 chunk as the teaching moment
6. **Identify dimension:** Within the selected chunk, find the dimension that deviates most from the student's baseline (if student model exists) or the dimension with the lowest absolute score (cold start)

### Blind Spot Dimension Selection

For the selected teaching moment chunk, determine WHICH dimension to talk about:

**With student history (warm):**
- Compare each dimension to the student's rolling average for that dimension
- The most surprising negative deviation = the blind spot
- "Surprising" means: the student is usually fine on this, but it dipped

**Without student history (cold start):**
- Use the STOP classifier's feature importance (logistic regression coefficients)
- The dimension that contributed most to the STOP prediction for this chunk
- Alternatively: the dimension with the lowest score

**Tie-breaking:**
- Prefer dimensions that are harder to self-diagnose from the player's seat
- Blind-spot prior: voicing/balance > pedaling > phrasing > timing > dynamics > articulation
- (Dynamics and articulation are easier to feel while playing; voicing and pedaling effects are harder to hear from the bench)

### Output to Teacher LLM (Slice 6)

The priority logic passes to the LLM:

```json
{
    "teaching_moment": {
        "chunk_index": 7,
        "start_offset_sec": 105.0,
        "stop_probability": 0.87,
        "dimension": "pedaling",
        "dimension_score": 0.35,
        "student_baseline": 0.62,
        "deviation": -0.27,
        "context": "This chunk had the highest teaching moment probability in the session. Pedaling deviated significantly below the student's typical level."
    },
    "session_summary": {
        "total_chunks": 12,
        "chunks_above_threshold": 3,
        "dominant_weak_dimension": "pedaling",
        "session_duration_min": 3.0
    }
}
```

### What This Slice Does NOT Include

- The teacher LLM prompt that turns this into natural language (Slice 6)
- Student model persistence (Slice 5 -- cold start is fine for now)
- Piece identification
- Focus mode triggers

### Tasks

**Task 1: Extract STOP classifier weights**
- Load the trained sklearn model from masterclass experiments
- Extract the 6 coefficients + bias
- Save as a JSON file for hardcoding into the Rust backend
- If model was trained on MuQ embeddings (2048-dim), retrain on 6-dim composite scores

**Task 2: Implement STOP classifier in Rust**
- Add StopClassifier struct to the Workers backend
- Hardcode weights from Task 1
- Score each chunk after inference completes (Slice 3 stores results)
- Store stop_probability in session_chunks table

**Task 3: Implement teaching moment selection**
- Endpoint: `POST /api/sessions/{session_id}/ask`
- Collects all chunks, filters by STOP threshold, ranks, selects top-1
- Identifies the blind-spot dimension (cold start logic first)
- Returns structured teaching moment JSON

**Task 4: Add blind-spot prior**
- Implement dimension difficulty ranking for tie-breaking
- Test with synthetic session data: verify that voicing issues are surfaced over timing issues when scores are similar

**Task 5: Validate against masterclass data**
- Run the pipeline on the 1,707 labeled moments
- For STOP moments: does the system flag them? Which dimension does it surface?
- For CONTINUE moments: does the system correctly not flag them?
- Target: >80% alignment with original STOP labels

### Open Questions

1. The STOP classifier was trained on masterclass audio (professional students, concert pianos). Will it generalize to beginner/intermediate students on upright pianos recorded with phones? Likely needs recalibration.
2. Should there be a minimum STOP threshold below which the system says "sounded good, keep going" instead of always finding something to criticize?
3. The blind-spot prior (voicing > pedaling > ...) is a hypothesis. Validate with real user testing.
