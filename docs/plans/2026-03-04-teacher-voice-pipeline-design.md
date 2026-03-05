# Teacher Voice Data Pipeline + Eval Framework

**Date:** 2026-03-04
**Status:** Design complete, prototype built

## Problem

CrescendAI's teacher LLM (Claude Sonnet 4.6) converts structured teaching moment data into natural, actionable piano feedback. Before investing in fine-tuning an open-weight model (Llama 70B on Together.ai), we need to:

1. Measure Claude's current ceiling with varying context richness
2. Determine whether the gap is in the LLM's voice or the harness's context
3. Build a reusable data pipeline that serves eval, fine-tuning, AND retrieval

## Key Insight

**The harness is the product.** Music interpretation is subjective -- even masters disagree about "good" playing. The system's value comes from detecting teaching moments, aligning to the score, and surfacing the right context to the LLM. If the harness provides rich enough context (piece, passage, dimension, student level, what went wrong), Claude's existing piano pedagogy knowledge may be sufficient.

Fine-tuning is worth pursuing only if Claude with perfect context still lacks:
- Domain vocabulary (embodied, tactile language)
- Diagnostic specificity
- Pedagogical framing

The eval measures this empirically.

## Architecture

A unified data pipeline with three consumers:

```
Raw Sources (masterclass, pedagogy, synthetic)
    |
    v
TeachingRecords (unified format)
    |
    +---> Eval Benchmark (test Claude's ceiling)
    +---> SFT/DPO Pairs (eventual Llama fine-tuning)
    +---> Retrieval Corpus (prompt enrichment at inference time)
```

## Data Sources (Priority Order)

| # | Source | Records | Effort | What it provides |
|---|--------|---------|--------|------------------|
| 1 | Masterclass moments (2,149 extracted) | ~2,100 usable | Low | Real teacher feedback language, diagnostics |
| 2 | Quote bank (60 per dimension) | 360 | Very low | Immediate retrieval corpus |
| 3 | Synthetic scenarios (Claude-generated) | 200-500 | Medium | Fill gaps: beginner/intermediate, specific pieces |
| 4 | Pedagogy literature (Chang, Neuhaus, etc.) | 200-400 | High | Embodied vocabulary, technique fundamentals |
| 5 | YouTube educational content | 300-600 | High | Piece-specific practice guidance |
| 6 | CrescendAI golden set (future) | Grows with usage | Ongoing | Exact product voice and quality bar |

### Dual-Use Data

The same data serves both the teacher voice AND the exercise database (Slice 07). Masterclass quotes about technique are both teacher language examples AND retrievable content for when a student is playing that piece. Text search + lightweight vector ranking over this corpus could power exercise retrieval without a full vector DB.

## Eval Methodology

### Three Context Levels

For each scenario, test Claude with increasing context richness:

1. **Bare:** dimension + generic framing only (baseline)
2. **Rich:** dimension + piece + passage + student level + feedback type
3. **Retrieved:** Rich + top-3 relevant quotes from the quote bank

### Rating Criteria (1-5 scale)

Priority order:
1. **Accuracy:** Does it correctly identify the musical issue?
2. **Actionability:** Does it tell the student what to do?
3. **Voice:** Does it sound like a real piano teacher? (secondary)

### Decision Criteria

- Claude + Rich >= 4.0 on accuracy + actionability: fine-tuning is marginal, invest in harness
- Claude + Retrieved jumps significantly over Rich: build retrieval pipeline, defer fine-tuning
- Claude plateaus below 3.5 regardless of context: fine-tuning has real headroom

## Unified Record Format

```python
@dataclass
class TeachingRecord:
    id: str
    source: str           # masterclass | pedagogy_book | synthetic | golden_set
    piece: str | None
    composer: str | None
    bars: str | None
    passage_description: str | None
    dimension: str        # one of 6: dynamics, timing, pedaling, articulation, phrasing, interpretation
    student_level: str    # beginner | intermediate | advanced
    feedback_type: str    # correction | suggestion | encouragement | explanation | question
    scenario: str         # what the student did (LLM input context)
    teacher_response: str # what the teacher said (target output)
    teacher_name: str | None
    source_id: str
    raw_transcript: str | None
    quality_score: float | None
    has_embodied_language: bool
    has_piece_specificity: bool
    is_actionable: bool
    metadata: dict
```

### Dimension Mapping

Raw masterclass pipeline uses 10 dimensions that map to our 6-dim taxonomy:

| Raw | Mapped |
|-----|--------|
| dynamics, timing, pedaling, articulation, phrasing, interpretation | Direct match |
| technique | articulation |
| voicing, tone_color | interpretation |
| structure | phrasing |

## Conversion Stats (from 2,149 raw moments)

- 2,136 usable records
- Dimension distribution: interpretation (562), articulation (453), timing (436), phrasing (298), dynamics (260), pedaling (127)
- Student level: 98% advanced, 2% intermediate, 0% beginner (confirms need for synthetic data)
- Quality: 31% use embodied language, 78% are piece-specific, 51% are actionable

## Early Observations (Smoke Test)

Tested Claude Sonnet 4.6 with all three context levels on a Zimerman masterclass moment about articulation in Chopin Ballade No. 1:

- **Bare:** Fabricated a scenario ("fingers getting sticky in a scalar run") -- not useful without real context
- **Rich:** Better but still invented specific details ("chord descent") that may not match the actual passage
- **Retrieved:** Noticeably closer to real teacher language, picking up embodied vocabulary from the quote bank ("contact with the keys first, then pressing through them with weight")

The retrieved variant's improvement over rich is a signal worth measuring systematically.

## Implementation

### Built (prototype)

- `model/src/teacher_voice/records.py` -- TeachingRecord dataclass, serialization, dimension mapping
- `model/src/teacher_voice/converters.py` -- Masterclass moment converter with quality heuristics
- `model/src/teacher_voice/benchmark.py` -- Benchmark runner (3 variants, Claude API, resumable)
- `model/src/teacher_voice/synthetic.py` -- 20 scenario templates covering all gaps, Claude-generated
- `model/src/teacher_voice/rate.py` -- CLI rating tool with summary statistics

### To Run

```bash
cd model

# 1. Convert masterclass moments (already done, saved to data/teacher_voice_eval/)
uv run python -m teacher_voice.converters

# 2. Generate synthetic scenarios
uv run python -m teacher_voice.synthetic

# 3. Run benchmark (50 masterclass + 20 synthetic = 210 API calls)
uv run python -m teacher_voice.benchmark

# 4. Rate outputs
uv run python -m teacher_voice.rate

# 5. View summary
uv run python -m teacher_voice.rate summary
```

### Next Steps

1. Run the full 50+20 benchmark
2. Rate outputs (Jai)
3. Analyze results: does retrieval close the gap? Where does Claude fall short?
4. Based on results, decide: retrieval pipeline vs. fine-tuning vs. both
5. If fine-tuning: convert full 2,136 records to SFT/DPO format for Together.ai
