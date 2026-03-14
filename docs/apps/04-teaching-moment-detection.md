# Slice 4: Teaching Moment / Priority Logic

**Status:** NOT STARTED
**Last verified:** 2026-03-03
**Notes:** No STOP classifier service in iOS codebase. Masterclass data pipeline exists in `model/` but Swift classifier extraction hasn't been done.

See `docs/architecture.md` for the full system architecture.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Given a session of analyzed audio chunks, identify which chunks are "teaching moments" (where a teacher would intervene) and which dimension to surface as the blind spot.

**Architecture:** Run the STOP classifier in the cloud worker (API). Each chunk's 6-dim scores (returned from HF inference) are scored for STOP probability. When the student asks "how was that?", the cloud worker ranks chunks by STOP probability, selects the top moment, identifies the most relevant dimension, and returns the pre-selected teaching moment to the client.

**Tech Stack:** Existing STOP classifier (logistic regression), Cloudflare Workers (Rust/WASM)

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

**Option A: Score on MuQ embeddings in HF endpoint (AUC 0.936)**

- Add STOP classifier as an additional output head in the HF inference endpoint
- Return STOP probability alongside dimension scores per chunk
- Pro: highest accuracy. Con: requires modifying the inference endpoint.

**Option B: Score on 6-dim composite scores in cloud worker (AUC 0.845)**

- The 6 dimension scores are returned by the HF endpoint
- Run STOP classifier in the cloud worker (Rust/WASM)
- Logistic regression is trivial: `sigmoid(dot(weights, scores) + bias)`
- Pro: no changes to HF endpoint. Con: lower accuracy.

**Recommendation: Start with Option B** (simple, runs in cloud worker on returned scores), upgrade to Option A if the 9% AUC gap matters in practice.

### Logistic Regression in Swift

The STOP classifier is a logistic regression with 6 features. For Option B:

```swift
struct StopClassifier {
    let weights: [Double]  // 6 weights, one per dimension
    let bias: Double

    func predict(dimensions: [Double]) -> Double {
        let logit = zip(weights, dimensions).map(*).reduce(0, +) + bias
        return 1.0 / (1.0 + exp(-logit))
    }
}
```

Weights are extracted from the trained sklearn model in `model/src/masterclass_experiments/models.py` and hardcoded (6 floats + 1 bias).

### Teaching Moment Selection (cloud worker, triggered when student taps "How was that?")

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

### Positive Teaching Moments

Teaching moment detection can also flag improvements and breakthroughs -- not just problems. When a dimension score is significantly above the student's baseline, or a previously flagged weakness shows measurable improvement, the system can surface a positive moment. Positive observations are real teaching: "Your pedaling in the second phrase has gotten much smoother." The cloud pipeline tags chunks where `dimension_score > baseline + threshold` as positive candidates alongside STOP-flagged chunks. The analysis subagent (see `docs/apps/06a-subagent-architecture.md`) decides whether to use a positive or corrective framing.

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
        "context": "This chunk had the highest teaching moment probability in the session. Pedaling deviated significantly below the student's typical level.",
        "section_label": "second phrase",
        "bar_range": "bars 20-24",
        "is_positive": false
    },
    "session_summary": {
        "total_chunks": 12,
        "chunks_above_threshold": 3,
        "dominant_weak_dimension": "pedaling",
        "session_duration_min": 3.0
    }
}
```

### Section/Passage Awareness

When the student reports what piece and section they are working on, the output to the subagent should include musical structure context. Instead of just chunk index and timestamp, include section labels ("A section", "transition", "coda") and approximate bar ranges ("bars 7-12"). This lets the teacher say "at bar 7" instead of "at 0:04." See `docs/apps/06a-subagent-architecture.md` for score alignment details. The `section_label`, `bar_range`, and `is_positive` fields in the JSON output above are optional -- populated when piece/section info is available.

### What This Slice Does NOT Include

- The teacher LLM prompt that turns this into natural language (Slice 6)
- Student model persistence (Slice 5 -- cold start is fine for now)
- Piece identification
- Focus mode triggers

### Tasks

**Task 1: Extract STOP classifier weights**

- Load the trained sklearn model from masterclass experiments
- Extract the 6 coefficients + bias
- Save as a JSON file for hardcoding into Swift
- If model was trained on MuQ embeddings (2048-dim), retrain on 6-dim composite scores

**Task 2: Implement STOP classifier in Swift**

- Add StopClassifier struct to the iOS app
- Hardcode weights from Task 1
- Score each chunk after inference completes (Slice 3 stores results)
- Store stopProbability in the SwiftData ChunkResult model

**Task 3: Implement teaching moment selection**

- Cloud worker function called when student taps "How was that?"
- Collects all chunk scores from the session, filters by STOP threshold, ranks, selects top-1
- Identifies the blind-spot dimension (cold start logic first)
- Returns structured teaching moment data for the LLM prompt (Slice 6)

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
