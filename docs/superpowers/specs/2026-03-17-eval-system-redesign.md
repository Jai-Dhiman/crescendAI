# Eval System Redesign: Empirical Criteria + Two-Eval Framework

> **Status (2026-03-17):** Design approved. Three phases: derive feedback quality criteria from masterclass data, build skill-level eval (model quality) and practice recording eval (pipeline quality), run and analyze.

## Motivation

The current pipeline eval (observation quality, n=227) produces misleading results:

- **Specificity: 0.4%, Musical Accuracy: 20.7%, Dimension Appropriateness: 0.9%** -- all failing
- Root cause: eval runs YouTube performances (competent pianists) without piece context
- STOP classifier never triggers, analysis engine never runs, subagent never called
- Teacher generates observations from zero context, judged against full-context criteria
- The eval tests the empty-context fallback path, not the core pipeline

Additionally, the 5 eval criteria (Musical Accuracy, Specificity, Actionability, Tone, Dimension Appropriateness) were invented without empirical grounding. The 6 model dimensions were derived from masterclass analysis -- the eval criteria should be too.

## Design Overview

Three phases, three weeks:

```
Phase 1: CRITERIA DERIVATION (Week 1-2)
  Input:  2,136 masterclass teaching moments
  Method: Open-ended LLM extraction -> HDBSCAN clustering -> effectiveness validation
  Output: 5-8 empirically grounded feedback quality criteria + judge prompt v2

Phase 2: BUILD BOTH EVALS (Week 2-3)
  Eval 1 (Skill Level): Model scores vs playing quality across skill levels
  Eval 2 (Practice Recording): Full pipeline on real student practice sessions

Phase 3: RUN & ANALYZE (Week 3)
  Output: "Is the bottleneck the model, the pipeline, or both?"
```

---

## Phase 1: Feedback Quality Criteria Derivation

### Goal

Derive 5-8 empirically grounded criteria for evaluating whether an AI-generated piano teaching observation is effective. Same methodology as the 6 model dimensions (open-ended extraction from masterclass data, clustering, multi-signal validation).

### Data Source

2,136 teaching moments across ~60 masterclass videos. Each moment has:

- `feedback_summary`, `transcript_text` -- what the teacher said
- `feedback_type` -- correction, suggestion, explanation, demonstration
- `severity` -- minor, moderate, significant
- `demonstrated` -- whether teacher played to show what they mean
- `passage_description` -- where in the piece
- `stop_order`, `stop_group`, `stop_in_group`, `group_size` -- temporal sequence
- `time_spent_seconds` -- investment in this point

Location: `model/data/raw/masterclass/teaching_moments/*.jsonl`

### Effectiveness Signal

Three proxy signals for intervention quality (no student outcome data needed):

1. **Non-repetition (weak signal, ~16.7% base rate):** If the same musical issue is addressed only once, the intervention may have been sufficient. If the same issue recurs across multiple stop groups, the first intervention may not have landed. "Same issue" matching uses sentence-transformer cosine similarity (threshold >= 0.75) on `(passage_description, musical_dimension)` pairs -- not exact string match, since `passage_description` is free-text LLM output. **Caveat:** non-repetition is ambiguous: the teacher may not return because the student improved, because time ran out, or because other issues took priority. Only ~16.7% of moments appear in repeated combos, making this a thin signal. Weight it below time investment.

2. **Time investment (primary signal):** `time_spent_seconds` correlates with pedagogical significance. A 110-second explanation vs a 20-second correction are different investments. Less ambiguous than non-repetition because time allocation directly reflects the teacher's judgment of importance. Weight this most heavily in validation.

3. **Feedback type distribution (supporting signal):** `correction` vs `explanation` vs `demonstration` vs `suggestion` have different pedagogical functions. Distribution across effective (non-repeated) vs. ineffective (repeated) interventions reveals which approaches work.

### Derivation Pipeline

**Step 1: Feature Extraction (LLM, open-ended)**

For each of the 2,136 moments, prompt an LLM:

```
Given this piano masterclass teaching moment:

Teacher: {teacher}
Piece: {composer} - {piece}
What the teacher said: {feedback_summary}
Transcript excerpt: {transcript_text (truncated to ~500 chars)}
Feedback type: {feedback_type}
Teacher demonstrated: {demonstrated}
Severity: {severity}

What qualities make this teaching intervention effective or ineffective?
List 2-5 specific qualities, each in 2-5 words.
Focus on qualities expressible in text (not physical demonstration).

Format: one quality per line, no numbering.
```

Output: free-text quality descriptors per moment.
Cost: ~$5 (2,136 calls at ~$0.002/call).

**Step 2: Clustering (HDBSCAN)**

- Embed all quality descriptors with sentence-transformer (same as taxonomy derivation)
- Cluster with HDBSCAN (min_cluster_size tuned, same approach as taxonomy)
- Manual review and naming of clusters
- Expected: 10-20 raw clusters, merging to 5-8 on review

**Step 3: Effectiveness Validation**

For each cluster, compute:

| Signal | Method | Selection Threshold |
|---|---|---|
| Frequency | cluster_size / total_moments | > 5% |
| Repetition correlation | Presence in non-repeated vs repeated interventions | Positive correlation |
| Severity correlation | Presence in significant vs minor interventions | Any signal |

Keep criteria that meet frequency threshold AND at least one validity signal.

**Step 4: Rubric Writing**

For each selected criterion:
- Definition (1-2 sentences)
- Pass/fail boundary with examples from masterclass data
- 3 example PASS moments, 3 example FAIL moments (quoted from transcripts)
- Judge prompt template section

### Constraint: AI Text Context

Masterclass teachers can demonstrate, ask Socratic questions, and have multi-turn dialogue. Our pipeline produces a single 1-3 sentence text observation. The extraction prompt explicitly asks for qualities "expressible in text" to filter out criteria that require physicality.

### Methodological Note: Factual vs Evaluative Extraction

The taxonomy derivation (02-teacher-grounded-taxonomy.md) asked "what musical aspect is the teacher commenting on?" -- a factual extraction. This derivation asks "what qualities make this intervention effective or ineffective?" -- an evaluative judgment. LLMs are generally better at the former. Extracted quality descriptors may be noisier, leading to fuzzier clusters. Mitigation: during manual cluster review (Step 2), be aggressive about merging near-duplicate clusters and discarding noise clusters. If cluster coherence is poor, fall back to manual coding of a 200-moment sample instead of automated extraction on all 2,136.

### Output Artifacts

- `apps/evals/pipeline/criteria_derivation/data/qualities_raw.jsonl` -- extracted descriptors
- `apps/evals/pipeline/criteria_derivation/data/clusters.json` -- cluster assignments
- `apps/evals/pipeline/criteria_derivation/data/validation_report.json` -- effectiveness signals
- `apps/evals/shared/prompts/observation_quality_judge_v2.txt` -- derived judge prompt

---

## Phase 2A: Eval 1 -- Skill Level Eval (Model Quality)

### Goal

Validate that A1-Max model scores correlate with playing quality across skill levels. Answers: "Are the scores right?" independent of the pipeline.

### Dataset

Two pieces, manifests already built:

- **Fur Elise** (Beethoven): 33 recordings, buckets 1-5
- **Chopin Nocturne Op. 9 No. 2**: 30 recordings, buckets 1-5

Both in ASAP 242-piece catalog.

Location: `model/data/evals/skill_eval/{piece}/manifest.yaml`

### Label Curation

Manual pass on all 63 recordings:
- Verify skill bucket (fix misclassifications: Rousseau, Lisitsa, Leonskaja, Yundi Li, Tiffany Poon, etc. are bucket 3 but should be 4-5)
- Flag non-piece recordings (wrong piece, medley, tutorial)
- Flag audio quality issues
- Add `label_source: "manual"` field

### Inference Pipeline

Existing infrastructure (`apps/evals/model/skill_eval/`):
1. `collect.py` -- YouTube search + yt-dlp download (24kHz mono WAV)
2. `run_inference.py` -- chunk to 15s, run through HF endpoint or local server
3. `analyze.py` -- Spearman rho, Cohen's d, confusion rate, bootstrap CIs, plots

### Analysis Metrics

| Metric | What It Tells You | Pass Threshold |
|---|---|---|
| Overall Spearman rho | Score ordering matches skill ordering | > 0.3 |
| Per-dimension rho | Which dimensions track skill | Informational |
| Confusion rate | % of cross-bucket pairs with inverted scores | < 0.40 |
| Low vs High Cohen's d | Separation between extremes | > 0.5 |
| Per-bucket means | Score distributions at each level | Monotonic increase |

### What This Eval Does NOT Test

- Absolute score correctness (no expert annotations)
- Pipeline quality (no STOP, subagent, teacher)
- Feedback quality

---

## Phase 2B: Eval 2 -- Practice Recording Eval (Pipeline Quality)

### Goal

Validate that the full pipeline produces useful feedback when a real student practices a known piece. Answers: "Is the feedback useful?" end-to-end.

### Dataset: YouTube Practice Recordings

**Collection:** Search YouTube for practice/progress videos of Fur Elise and Nocturne Op. 9 No. 2. Filter for actual practice sessions, not polished performances.

**Search queries:**
- "{piece} practice session piano"
- "{piece} learning slow practice"
- "{piece} practicing hands separate"
- "{piece} working on [section]"

**Target:** 15-25 recordings. Bias toward beginner/intermediate (buckets 1-3) where STOP triggers.

**Selection criteria (human review):**
- Student is practicing, not performing
- Audio quality sufficient for AMT
- Piece is identifiable
- Mistakes are audible

**Fallback if practice videos are scarce:** Genuine practice sessions with audible mistakes from identifiable pieces may be rarer on YouTube than polished performances. If fewer than 10 suitable recordings are found: (1) use beginner-level skill eval recordings (bucket 1-2) as practice proxies -- these have real mistakes even if they're performing, not practicing; (2) broaden search to "piano progress" videos showing before/after of specific passages; (3) consider recording 3-5 sessions yourself as a supplement.

### Inference Step

Practice recordings follow the same pattern as the current eval: audio is first run through local inference (`apps/evals/inference/eval_runner.py` on MPS) to get predictions, midi_notes, and pedal_events. Results are cached as `{video_id}.json` in the inference cache directory. The pipeline client then sends pre-computed results via `eval_chunk` WebSocket messages to `wrangler dev`. This avoids re-running inference on every eval iteration and ensures reproducibility. Only the LLM stages (Groq subagent + Anthropic teacher) require API calls during the pipeline run.

### Annotation: Two-Pass

**Pass 1 (light, during collection):**

```yaml
recording_id: practice_fur_elise_beginner_03
video_id: <youtube_id>
piece_query: "beethoven fur elise"
skill_level: 2
general_notes: "Struggles with B section timing, pedaling muddy in A returns"
audio_quality: good
expected_stop: true
```

**Pass 2 (detailed, after first run):** Add bar-level annotations for interesting recordings:

```yaml
known_issues:
  - dimension: pedaling
    location: "A section returns"
    description: "pedal held through harmony changes"
  - dimension: timing
    location: "B section sixteenths"
    description: "rushing the fast passage"
```

### Pipeline Client Changes

Modify `apps/evals/shared/pipeline_client.py`:

1. **Send `set_piece`** with `piece_query` from scenario card (enables Tier 1 analysis)
2. **Real baselines only** -- precompute baselines from skill eval recordings at the same piece and approximate skill level. For example, if the practice recording is a bucket-2 student playing Fur Elise, compute baselines from the mean scores of bucket-2 Fur Elise skill eval recordings. Do NOT artificially inflate baselines to force STOP triggers -- that would test whether the teacher generates coherent text from fabricated context, not whether the pipeline produces useful feedback from real context. If no suitable baseline recordings exist for a skill level, run without baselines and accept that STOP may not trigger. That itself is useful data (indicates calibration needs).
3. **Tier tagging** -- tag each observation with analysis tier (1/2/3)
4. **Framing tagging** -- tag corrective vs positive in report

### Judge

Uses derived criteria from Phase 1. Judge receives:

- `observation_text`
- `eval_context` from Rust pipeline: predictions, baselines, analysis_facts, piece_name, bar_range, teaching_moment, subagent_output
- `scenario_card` from human annotations

### Analysis Metrics

| Metric | What It Tells You |
|---|---|
| Per-criterion pass rate | Which feedback qualities work/fail |
| STOP trigger rate | % of recordings where pipeline activates |
| Tier distribution | Tier 1 vs 2 vs 3 |
| Framing distribution | Corrective vs positive vs encouragement |
| Pass rate by tier | Does Tier 1 produce better observations? |
| Pass rate by framing | Are corrective observations better? |
| Pass rate by skill level | Does pipeline work for beginners? |

### Pass Criteria

Thresholds set after Phase 1. Each criterion gets a threshold based on masterclass analysis: if 80% of effective interventions exhibit a quality, pipeline threshold is aspirational (e.g., 60% for v1).

### Human Calibration Set

To catch systematic LLM judge bias, hand-score 10-15 observations from the first eval run. For each, Jai rates each criterion as pass/fail independently. If the LLM judge disagrees with human judgment on more than 30% of cases for any criterion, the judge prompt for that criterion needs revision before results are trusted. This is a one-time calibration step after the first run, not an ongoing requirement.

---

## Changes to Existing Code

| File | Change |
|---|---|
| `apps/evals/shared/pipeline_client.py` | Pass `piece_query`, add baseline injection, tag tier/framing |
| `apps/evals/pipeline/observation_quality/eval_observation_quality.py` | Support judge v2, add tier/framing segmentation |
| `apps/evals/shared/judge.py` | Support loading different prompt versions |
| `model/data/evals/skill_eval/*/manifest.yaml` | Curated skill bucket labels |

## New Code

| Component | Purpose |
|---|---|
| `apps/evals/pipeline/criteria_derivation/` (3 scripts + data) | Phase 1: derive criteria |
| `apps/evals/pipeline/practice_eval/` (2 scripts + scenarios) | Eval 2: practice collection + runner |
| `apps/evals/shared/prompts/observation_quality_judge_v2.txt` | Derived criteria judge prompt |

## Inference: Local Only

All inference runs locally on Jai's M4 Air (32GB) via MPS. No HF endpoint costs.

Two local inference paths exist:
- **`apps/inference/local_server.py`** -- HTTP server (~31s/chunk on MPS), used by pointing the Worker at `localhost:8000` via `HF_INFERENCE_ENDPOINT=http://localhost:8000` in `apps/api/.dev.vars`
- **`apps/evals/inference/eval_runner.py`** -- batch runner that loads models directly, caches results to JSON files. Does not require the Worker or local server running.

**For Eval 1 (skill eval):** Use the batch runner (`eval_runner.py`). It caches per-chunk predictions + MIDI + pedal to `model/data/evals/inference_cache/`. Run once, reuse cache across analysis iterations. The existing `run_inference.py` in the skill eval module should call the batch runner or load from its cache.

**For Eval 2 (practice eval):** Same batch runner to generate the inference cache for practice recordings. The pipeline client then sends cached results via `eval_chunk` WebSocket messages to `wrangler dev` (which still needs Groq + Anthropic API keys for the LLM stages).

**Latency budget:** ~31s per 15s chunk on MPS. A typical recording (3-5 min = 12-20 chunks) takes ~6-10 minutes for inference. 63 skill eval recordings: ~6-10 hours total. 20 practice recordings: ~2-3 hours. Run overnight or in batches.

## Dependencies

- Phase 1: existing masterclass data, Anthropic API, sentence-transformers, hdbscan
- Eval 1: yt-dlp, local inference (MuQ + AMT models on MPS)
- Eval 2: Phase 1 output, yt-dlp, local inference, wrangler dev, Groq + Anthropic APIs (LLM stages only)

## Cost Estimate

| Component | Cost |
|---|---|
| Phase 1 LLM extraction (2,136 calls) | ~$5 |
| Eval 1 inference (63 recordings) | $0 (local MPS) |
| Eval 2 inference (20 recordings) | $0 (local MPS) |
| Eval 2 LLM pipeline (20 recordings) | ~$2 (Groq subagent + Anthropic teacher) |
| Eval 2 judge (20 recordings) | ~$0.60 |
| **Total** | **~$8** |

## Timeline

| Week | Work | Output |
|---|---|---|
| Week 1 | Phase 1: LLM extraction, embed, cluster | Raw clusters + effectiveness validation |
| Week 2 | Phase 1: select criteria, write rubrics. Eval 1: curate labels, download | Criteria doc + curated manifests |
| Week 3 | Eval 1: run + analyze. Eval 2: collect, build, run | Both eval reports |

## Decision Point

After both evals complete:

- If Eval 1 fails (scores don't separate skill levels): model is the bottleneck. Prioritize training data diversity, skill-level calibration.
- If Eval 1 passes but Eval 2 fails: pipeline is the bottleneck. Prioritize prompt engineering, subagent reasoning, teacher context flow.
- If both pass: ready for product testing. Prioritize web polish, iOS wiring, exercises.
- If both fail: fundamental rethink needed.

## Known Limitations

1. **Two-piece generalization:** Both pieces (Fur Elise, Nocturne Op. 9 No. 2) are well-known and likely over-represented in audio model pretraining data. Results may not transfer to less common repertoire. Acceptable for v1; expand piece selection if results look suspiciously good.

2. **Masterclass-to-AI transfer gap:** Criteria derived from masterclass interventions describe what makes human teaching effective. Our pipeline is a single-shot text observation, not a multi-turn dialogue with demonstrations. The "expressible in text" filter mitigates this but does not eliminate the gap.

3. **Skill labels are ordinal only:** The skill eval validates score ordering (beginner < advanced), not absolute accuracy. A model that gives all beginners 0.60 and all professionals 0.65 would pass the ordering test despite tiny separation. Cohen's d catches this partially, but expert annotations would be needed for absolute validation.

4. **YouTube audio diversity:** YouTube recordings span wildly different audio setups (phone mic, room mic, direct digital output). Model performance may vary by recording quality independent of playing quality. The skill eval cannot distinguish "model fails on beginners" from "model fails on phone audio."
