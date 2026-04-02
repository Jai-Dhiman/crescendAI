# Teaching Knowledge Extraction & Research-Driven Eval

**Date:** 2026-03-26
**Status:** Draft
**Scope:** Pre-beta eval strategy grounded in music pedagogy research

## Problem

CrescendAI's pipeline produces session synthesis feedback, but we have no grounded definition of what "good teaching" looks like. The existing 7-capability eval framework (docs/apps/06-capabilities.md) measures technical correctness of individual capabilities in isolation. But the user only cares about one thing: was the teacher's feedback good?

Three compounding problems:
1. The pipeline's teaching model (detect problem -> pick worst dimension -> tell student to fix it) is a corrective feedback model. Great teachers do much more: encouragement, guided discovery, scaffolding, restraint.
2. Eval metrics (grounding, actionability, etc.) are proxy metrics that measure surface features, not musical or pedagogical correctness. A synthesis can score well on "grounding" and be completely wrong musically.
3. The founder is an amateur musician. Without extracting expert knowledge systematically, the teaching quality ceiling is bounded by amateur intuition.

## Key Insight

Instead of pre-defining eval metrics and hoping they correlate with quality, extract teaching expertise at scale from real teachers and pedagogy research. Let the metrics emerge from the evidence.

The sequence: **research what good teaching IS -> define metrics from research -> evaluate pipeline against those metrics -> iterate.**

## Architecture (Parallel Tracks)

Two independent tracks that converge when the playbook is ready:

```
TRACK A: Teaching Knowledge Research          TRACK B: Eval Infrastructure
═══════════════════════════════════          ════════════════════════════════

PHASE 0: Extract                             PHASE 0B: Fix Plumbing
  Sources -> LLM Filter ->                     Fix eval client routing
  4-field extraction ->                        Add accumulator snapshot
  Raw Teaching Database                        Complete inference cache
                                               Smoke test: 10 recordings
PHASE 0.5: Synthesize                          through current pipeline
  Raw Database ->                              (validates infra, not quality)
  LLM Synthesis ->
  Teaching Playbook +
  Eval Rubrics

              ┌──────── CONVERGENCE ────────┐
              │                             │
              ▼                             ▼
PHASE 1: Pipeline Alignment
  Playbook -> Revise synthesis prompt, teaching moment selection, STOP

PHASE 2: Eval Execution (Calibrated Funnel)
  361 T5 recordings -> Pipeline -> LLM Judge (rubric-grounded) + Human review (50)
  -> Calibrate judge -> Identify systematic failures -> Trace to capability

PHASE 3: Iterate
  Fix failures -> Re-run -> Re-judge -> Pass B (no piece context) -> Beta readiness
```

## Phase 0: Teaching Knowledge Extraction

### Sources

**Source A: T2 Masterclass Transcripts**
- 9,059 segments from piano competition masterclasses (Cliburn, etc.)
- Each segment has audio and performer context
- Transcript acquisition: `yt-dlp --write-auto-sub` on source YouTube videos. T2 metadata includes video IDs. Same quality gate as Source B applies: English only, min 500 words, discard if >30% [inaudible]/[Music] tokens.
- Risk: masterclass audio often has piano over speech. If auto-captions are unusable for >50% of T2 videos, fall back to Whisper transcription on the raw audio (slower but higher quality on speech-over-music).
- Target: extract teaching behavior from real expert-to-student feedback

**Source B: YouTube Piano Lessons (new data)**
- Search queries: "piano masterclass", "piano lesson feedback", "piano teacher critique", "piano tutorial pedagogical"
- Filter: videos with dialogue (not performance-only), educational content, varied skill levels
- Target: 200-500 transcripts covering beginner through advanced
- Method: `yt-dlp --write-auto-sub` for captions, or YouTube transcript API
- Transcript quality gate: English only, minimum 500 words, discard auto-captions with >30% [inaudible]/[Music] tokens. If auto-captions are too noisy for a source, fall back to manual transcription of the highest-value 20-30 videos.

**Source C: Pedagogy Literature**
- Web search for piano pedagogy papers, teaching methodology summaries
- Key frameworks: Suzuki method, ABRSM syllabus, RCM structure, Taubman approach
- Research on motor learning, deliberate practice (Ericsson), skill acquisition stages
- Method: LLM synthesis of principles from multiple sources

### Extraction Pipeline

**LLM Pass 1 (Filter):** Classify each transcript segment:
- Is this a real teaching/feedback moment? (vs. performance-only, Q&A, announcements)
- What type? (masterclass critique, lesson instruction, tutorial explanation)
- Confidence score for filtering

**LLM Pass 2 (Structured Extraction):** Two tiers of extraction.

**Core schema (4 extracted fields + source key, run on everything):**

```yaml
source_id: youtube_video_id or paper_citation  # metadata key, not extracted
# The 4 extracted fields:
what_teacher_said: "verbatim or close paraphrase of the teaching moment"
dimension_focus: dynamics | timing | pedaling | articulation | phrasing | interpretation | general
student_skill_estimate: beginner | early_intermediate | intermediate | advanced | professional
feedback_type: corrective | encouraging | modeling | guided_discovery | scaffolding | motivational
```

These 4 fields are high-confidence extractable from noisy transcripts and sufficient to build the playbook.

**Enrichment schema (additional fields, run on high-quality subset):**

For transcripts that pass a quality threshold (manual subtitles, clean auto-captions, or Whisper-transcribed), extract additional fields:

```yaml
piece: { composer, title }
specificity: piece_specific | passage_specific | technique_general
language_register: warm | direct | technical | metaphorical
what_teacher_did_NOT_say: "notable omissions"
implied_priority: "why this feedback over other possibilities"
student_response: "how the student reacted (if visible)"
```

Start with the core schema. If calibration shows high reliability on enrichment fields, expand. If not, the core 4 fields are sufficient.

### Extraction Calibration

Before running the full pipeline, validate extraction quality:
1. Manually annotate 20 transcripts (10 masterclass, 10 lesson) using the schema above
2. Run LLM extraction on the same 20
3. Measure agreement on key fields: strategy, dimension_focus, student_skill_estimate
4. Target: >80% agreement on categorical fields, qualitative alignment on free-text fields
5. If extraction quality is insufficient, simplify the schema (drop subtle fields like `what_teacher_did_NOT_say`) and re-test

### Minimum Viable Corpus

Gate between Phase 0 and Phase 0.5:
- Source A (masterclass): minimum 200 usable extracted moments (from 9K segments, even 5% yield = 450)
- Source B (lessons): minimum 50 usable extracted moments
- Source C (literature): minimum 10 distinct pedagogical frameworks or principles
- If any source falls short, proceed with available data but note the gap in the playbook

### Output: Raw Teaching Database

Structured YAML/JSON database of extracted teaching moments + pedagogy principles. This is the evidence base for everything downstream.

## Phase 0.5: Synthesis + Metric Derivation

### Teaching Playbook

Multi-round synthesis from the raw database, with founder review at each round:

**Round 1: Clustering.** LLM groups extracted teaching moments by student skill level and feedback type. Output: natural clusters and patterns that emerge from the data. Do NOT pre-assume an organizing principle (tiers, moves, or anything else).

**Round 2: Pattern extraction.** For each cluster, LLM identifies: dominant teaching strategies, dimension priorities by repertoire style, language patterns, recurring good/bad feedback shapes, and meta-patterns (what distinguishes great feedback from mediocre). Output: draft playbook structure with evidence citations.

**Round 3: Founder review + curation.** Jai reviews the emergent structure against his experience as a self-learner and masterclass observer. Validates, adjusts, or proposes alternative organization. Output: final playbook.

**Playbook quality gate:** The playbook must contain at least 3 distinct feedback patterns per skill level with cited evidence from the extracted data. Jai must identify at least one pattern that surprises him (something the current pipeline does not do). If the playbook feels like a restatement of what the pipeline already does, the research hasn't gone deep enough -- add more source data or revise the synthesis.

**Organizing principle:** The playbook's structure emerges from the extracted data. It might organize by skill level (tiers), by teaching move (encourage, correct, scaffold), by context (after first correct play-through, during drilling, at session end), or by some combination. The research will reveal the natural structure that real teachers use. Do not pre-commit.

**What the playbook must contain (regardless of structure):**
- When to use each type of feedback and when not to
- Piece-style dimension rules (e.g., which dimensions matter for Bach vs. Chopin)
- Good and bad feedback examples extracted from real teachers
- What great teachers do differently from mediocre teachers
- Language patterns and register appropriate for different contexts

### Research-Derived Eval Metrics

The metric dimensions emerge from Phase 0 findings. The process:

1. Cluster the extracted quality markers -- what patterns distinguish great feedback from mediocre?
2. Identify the dimensions that matter (these may NOT be the obvious ones like "actionability")
3. For each dimension, define a scale grounded in extracted examples
4. Build judge prompts with real good/bad examples as calibration

**What we pre-commit to:**
- A north star composite score for synthesis quality
- Dimensions derived from evidence, not assumptions
- Each dimension has rubric language grounded in real examples
- Judge calibrated against those examples
- Human review validates the judge

**What we explicitly do NOT pre-commit to:**
- The specific dimensions (maybe it's 3, maybe it's 7)
- The scale (maybe 0-3, maybe pass/fail, maybe something else)
- The weights
- The scoring language

## Track B: Eval Infrastructure (parallel with Track A)

Runs concurrently with Phase 0 and 0.5. Goal: eval pipeline is tested and ready when the playbook arrives.

### Phase 0B: Fix Plumbing

1. **Eval client routing.** Remove `is_eval_session` flag, route through production accumulation path.
2. **Accumulator snapshot.** Add full accumulator state (teaching moments, mode transitions, baselines) to synthesis WebSocket response.
3. **Inference cache completion.** Resolve AMT server crash, complete cache for all T5 recordings.
4. **Smoke test (10 recordings).** Run 10 recordings through the current pipeline end-to-end. Do NOT judge output quality. Purpose: validate that the client connects, chunks play, STOP fires, synthesis generates, and accumulator state is captured. Fix any plumbing issues found.

### Phase 0B Output

- Working eval pipeline (client -> API -> synthesis -> capture)
- Complete inference cache
- Confidence that when the playbook arrives, we can run 361 recordings without infrastructure failures

## Phase 1: Pipeline Alignment

The Teaching Playbook informs revisions to three pipeline components:

1. **Synthesis prompt** (`apps/api/src/services/prompts.rs` SESSION_SYNTHESIS_SYSTEM): Encode tier-aware strategies, piece-style rules, and the teaching posture appropriate for each skill level.

2. **Teaching moment selection** (`apps/api/src/practice/accumulator.rs` top_moments): Incorporate piece-style dimension priorities so the pipeline doesn't flag irrelevant dimensions.

3. **STOP behavior** (`apps/api/src/services/stop.rs`): Consider tier-aware patience -- newcomers may need more tolerance before STOP triggers.

### Skill Tier Detection at Runtime (separate work item)

This is a prerequisite for tier-aware synthesis but is a distinct implementation task, not part of the eval pipeline itself. Scoped separately in Phase 1.

For beta, skill tier is determined by a combination of:
- MuQ score distribution across the session (mean + variance of 6-dim scores correlates with skill)
- T5 skill labels provide the calibration data for this mapping
- Fallback: user self-report during onboarding ("How long have you been playing?")
- The tier detection does not need to be perfect -- it's a coarse signal (3-4 buckets) that shifts the synthesis prompt's teaching posture. A mis-tier produces mediocre feedback, not wrong feedback.
- For eval purposes (Phase 2), tier is known from T5 metadata (skill_bucket), so tier detection quality does not block the eval.
- For Phase 1 prompt revision, use T5 skill_bucket as a hardcoded tier input. The runtime tier detector is needed for production but not for eval. Phase 1 revises the prompt to be tier-aware using the known skill level, deferring automatic detection to a later task.

## Phase 2: Eval Execution

### Calibrated Funnel

1. **Run full T5 corpus** (361 recordings, Pass A with piece context) through the revised pipeline
2. **LLM judges score every synthesis** against playbook-derived rubrics
3. **Human review ~50 outputs**, stratified:
   - 10 worst-scored by judge
   - 10 best-scored by judge
   - 10 mid-range
   - ~15 targeted by failure hypothesis (per-piece, per-skill-level, edge cases)
   - 5 specifically testing piece-style awareness
4. **Calibrate judge** against human judgment -- where do they diverge?
5. **Trace failures** to responsible capability:
   - Wrong bars referenced -> score following
   - Wrong dimension flagged -> teaching moment selection or piece-style rules
   - Wrong skill-level tone -> tier detection
   - Generic/undifferentiated -> synthesis prompt or accumulator data quality

### Two-Pass Design (existing)

- Pass A: with piece_query (full score context)
- Pass B: without piece_query (zero-config, piece ID from AMT)
- Delta: quantifies the value of piece context for synthesis quality
- Pass B is informational, not a beta gate. It measures piece-context impact to inform prioritization of zero-config piece ID, but beta can ship with manual piece selection if the delta is large.

## Phase 3: Iterate

1. Fix systematic failures identified in Phase 2
2. Re-run pipeline, re-judge, spot-check
3. Run Pass B to measure piece-context delta
4. Beta readiness assessment: no critical failures + composite score meets research-derived threshold. Threshold is set after Phase 2 human review calibration, approved by founder. The bar is: "no synthesis output that a musically literate user would call wrong, and clear differentiation from generic LLM advice on 80%+ of outputs."

## What Already Exists vs. What's New

| Component | Status | Work |
|-----------|--------|------|
| T5 corpus (361 recordings) | Exists | None |
| Inference cache (MuQ + AMT) | Partially exists | Resolve AMT crash, complete cache generation (Track B) |
| Eval client (WebSocket) | Exists | Fix plumbing (routing, accumulator snapshot) |
| Two-pass runner | Exists | Validate, minor fixes |
| Judge framework | Exists | Replace proxy criteria with research-derived rubrics |
| Analysis script | Exists | Rewrite for new metric structure |
| T2 masterclass data (9K segments) | Exists | Need transcripts extracted |
| Teaching knowledge extraction pipeline | **New** | LLM pipeline for transcript analysis |
| YouTube lesson corpus | **New** | Source, download transcripts, filter |
| Pedagogy literature synthesis | **New** | Web research + LLM synthesis |
| Raw Teaching Database | **New** | Output of extraction pipeline |
| Teaching Playbook | **New** | Synthesis from raw database |
| Research-derived eval rubrics | **New** | Derive from Phase 0 findings |
| Revised synthesis prompt | **New** | Align to playbook |
| Revised teaching moment selection | **New** | Piece-style dimension priorities |

## Relationship to Existing Eval Framework

The 7-capability framework (docs/apps/06-capabilities.md) remains as the diagnostic decomposition. When synthesis quality fails, we trace back through the capabilities to find the root cause. But synthesis quality is the north star metric -- the capabilities are not evaluated independently.

The existing eval methodology (docs/apps/07-evaluation.md) provides infrastructure (T5 corpus, two-pass design, eval client) that this plan builds on. The key change is the source of eval criteria: research-derived rubrics instead of pre-assumed proxy metrics.

## Prerequisites

- **Inference cache complete.** All 361 T5 recordings must have MuQ + AMT cache entries before Phase 2. Current state: cache generation was in progress but AMT server crashed at ~2/269. This must be resolved first -- either by restarting cache generation or scoping Phase 2 to the cached subset.
- **Eval client plumbing fixed.** The eval client must route through the production accumulation path (not the legacy `is_eval_session` flag) and return accumulator snapshots with synthesis responses.

## Effort Estimates

| Phase | Track | Work | Estimate |
|-------|-------|------|----------|
| Phase 0 | A (Research) | Transcript acquisition, 4-field LLM extraction, calibration. Includes 2-day buffer for transcript quality triage. Early checkpoint after 50 T2 videos to decide whether to pivot heavier to Source B. | ~1 week |
| Phase 0B | B (Infra) | Fix eval plumbing, complete cache, smoke test 10 recordings | ~3-4 days |
| Phase 0.5 | A (Research) | Playbook synthesis (3 rounds + founder review), metric derivation | ~3-5 days |
| Phase 1 | Converged | Prompt/threshold revisions + skill detection prototype | ~3-4 days |
| Phase 2 | Converged | Full eval run + judge scoring + 50-output human review | ~3-4 days |
| Phase 3 | Converged | Iteration (fix -> re-run -> re-judge cycles) | Ongoing |
| **Total to first eval results** | | Track A + B parallel, then converge | **~2.5 weeks** |

Track B (infra) completes during Track A's Phase 0, so no idle time. The playbook arrives to a tested, ready pipeline.

LLM cost estimate: ~$50-150 for extraction (depends on transcript volume and re-extraction after schema simplification), ~$30 for judge passes, ~$20 for synthesis/iteration. Total < $250. All LLM passes use Claude Sonnet 4.6 unless cost requires Haiku for high-volume filtering (Pass 1). Extraction (Pass 2), synthesis (Phase 0.5), and judging (Phase 2) use Sonnet for quality.

Human review: Jai reviews all 50 stratified outputs in Phase 2 (~8-10 hours of listening + annotation). This is a bottleneck but intentional -- the founder's judgment is the calibration source until expert teachers are available post-beta.

## References

- `docs/references/amateur-learns-fantaisie-impromptu.md` -- Target user persona and self-learner behavior patterns
- `docs/references/HowwebuildevalsforDeepAgents.md` -- "Every eval is a vector that shifts behavior. More evals != better agents. Build targeted evals that reflect desired behaviors in production."
- `docs/apps/06-capabilities.md` -- 7-capability definitions (diagnostic decomposition)
- `docs/apps/07-evaluation.md` -- Existing eval methodology and T5 corpus
