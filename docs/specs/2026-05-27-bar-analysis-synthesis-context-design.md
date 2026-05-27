# Bar-Analysis-in-Synthesis Context Design

**Goal:** The teacher LLM receives structured per-bar facts (velocity, onset deviation, articulation ratio, pedal events, reference comparisons) for the chunk that triggered each top teaching moment, so its feedback can be bar-specific and actionable.

**Not in scope:**
- New Rust features (no tempo/rubato split, no dynamics curve fitting). Existing `analyze_tier1` / `analyze_tier2` outputs are sufficient.
- Schema changes to the `observations` table beyond what already exists (`reasoning_trace` already holds free-form text).
- Production deployment, migration, or rollout sequencing.
- The teacher finetune Stage 1+ work. This change is upstream of training data quality, not part of training.
- Validation gates from the MPM Bucket Enrichment hypothesis. That work treats MPM features as aux supervision; this change uses already-computed features as inference-time context. The two are independent.

## Problem

`apps/api/src/wasm/score-analysis/src/bar_analysis.rs::analyze_tier1` already produces six `DimensionAnalysis` records per chunk — e.g. `"Mean onset deviation -45.0ms (std 12.3ms): rushing ahead of the score. Reference performers: mean deviation -5.2ms (std 8.1ms)."` — and the result is wired up to `apps/api/src/do/session-brain.ts` at lines 614 and 1081. But only two fields of that result are extracted (`analysis.tier`, `analysis.bar_range`); the `analysis.dimensions` array is discarded.

In `session-brain.ts` lines 974 and 1240, every `AccumulatedMoment` is built with `llmAnalysis: null`. The synthesis prompt assembled in `apps/api/src/services/prompts.ts::buildSynthesisFraming` therefore contains aggregate MuQ scores and deviations from baseline, but zero measured musical facts. The teacher LLM cannot ground bar-specific feedback in evidence it does not have.

This is the root cause of the locked ASCF outcome baseline of 1.387/3.0 on the Teaching Knowledge eval (n=513). The eval prompt mirrors prod (`apps/evals/teaching_knowledge/run_eval.py::build_synthesis_user_msg`), so it shares the same deficit.

## Solution (from the user's perspective)

A student finishes a session containing one or more 15-second chunks where the model flagged a problem. In the synthesis response, the teacher names the specific bars, the specific measurable issue (e.g. "rushing by about 45 ms ahead of the score"), and where relevant cites a correlated dimension from the same passage (e.g. "and your notes are clipped — that's about half their written length"). Praise also gets specific: "in bars 12–14 your dynamics matched the notated shape closely, including the crescendo through the second half."

This is observable both in production (deployed prompt) and in the eval harness (the ASCF outcome score, which the eval judge assigns specifically based on bar/dimension specificity in the synthesis text).

## Design

**Approach (Option C from brainstorm, with deviation threshold 0.15 and cap 3):**

For each teaching moment the WASM `select_teaching_moment` selects, the system already computed a `ChunkAnalysis` containing all six `DimensionAnalysis` entries for that chunk. A new pure function `buildBarAnalysisFacts` filters that array down to:

- `selected`: the `DimensionAnalysis` for the dimension that triggered the moment (always included; never null).
- `correlated`: up to two additional `DimensionAnalysis` entries from the same chunk where `|score[i] − baseline[i]| ≥ 0.15`, sorted by absolute deviation descending, excluding the selected dimension.

The structured object is attached to `AccumulatedMoment.llmAnalysis` (a type-change from `string | null` to `BarAnalysisFacts | null`). The synthesis prompt in `buildSynthesisFraming` includes the structured object as a `bar_analysis` field on each top moment in the `session_data` JSON it hands to the teacher.

**Why 0.15 threshold:** The synthesis system prompt already calibrates the teacher with "deviation of 0.1 is noise; 0.2+ is meaningful." 0.15 sits in the middle — strict enough to suppress dimensions that are only marginally off baseline, lenient enough to include genuinely correlated problems.

**Why cap at 3:** The `SESSION_SYNTHESIS_SYSTEM` instruction tells the teacher to "focus on what matters most for THIS session" and "do NOT list all dimensions." Even at 3, the prompt-level discipline against laundry-list responses is preserved. Six dimensions per moment would create structural pressure against that discipline.

**Why structured JSON rather than concatenated prose:** The `DimensionAnalysis` records are already English sentences from Rust. Structured nesting (`selected: {...}, correlated: [{...}]`) lets the teacher distinguish "the main thing" from "also notable" without re-deriving that ranking from prose. JSON object boundaries make the field skippable when null without prompt-template branching.

**Why no Rust changes:** The existing analyzers already produce the right shape. Adding tempo/rubato split or dynamics curve fitting before validating that *any* bar-level data moves ASCF would be speculative. Order: validate hypothesis first, enrich features second.

**Eval mirror (B3 with Tier-2 fallback for non-matched pieces):**

The Python eval harness `run_eval.py` operates on a cached inference dataset (`model/data/eval/inference_cache/auto-t5_http/*.json`). Each cache file has `chunks[*].midi_notes`, `chunks[*].pedal_events`, and `chunks[*].predictions`. The harness gets:

1. `piece_score_map.py` — a hand-curated lookup mapping each eval `piece_slug` (the 17 directories in `model/data/evals/skill_eval/`) to a file under `model/data/scores/`. Pieces without a confident match map to `None`.
2. `bar_analysis_local.py` — a Python port of the Rust Tier-2 statistics (velocity stats, IOI stats, articulation duration mean, pedal event count) and a Python port of Tier-1 (adds notated comparison + delta) keyed off the score JSON when available. Identifies the worst-deviating chunk per recording and produces the same `BarAnalysisFacts` shape used in TS.
3. `run_eval.py::build_synthesis_user_msg` injects `bar_analysis` into each top moment when available, falling back to `None` when not.

Per-bar score-aligned reference profiles (`ReferenceProfile`) are not populated for any piece in production or in `model/data/scores/`. Tier-1 in this work means "performance vs notated score" only — the `reference_comparison` field stays absent. That parity is intentional: production also has `reference: null` today.

## Modules

### `bar-analysis-facts.ts` (TS, new)
- **Interface:** `buildBarAnalysisFacts(analysis: ChunkAnalysis, scoresArray: [number,...x6], baselines: StudentBaselines, selectedDimension: Dimension): BarAnalysisFacts | null`
- **Hides:** Dimension-name → index lookup, threshold (0.15), cap (3 total), absolute-deviation sort, exclusion of the selected dimension from correlated, null-when-`analysis.dimensions`-empty handling.
- **Tested through:** The public function. Inputs are constructed `ChunkAnalysis` records and baseline objects; output is asserted as a shaped struct. No mocks.

### `bar_analysis_local.py` (Python, new)
- **Interface:** `build_bar_analysis(chunks: list[dict], baselines: dict[str, float], score_json: dict | None) → dict | None`
- **Hides:** Worst-deviating-chunk selection across the recording, Python implementation of Tier-2 stat formulas (velocity mean/range, IOI mean/std, articulation ratio, pedal event count) and Tier-1 deltas (perf vs notated mean velocity, ratio of perf-to-score note duration), the threshold/cap filtering identical to the TS module.
- **Tested through:** The public function. Inputs are fixture chunk lists and a fixture score JSON (or None). No mocks.

### `piece_score_map.py` (Python, new)
- **Interface:** `get_score_path_for_piece(piece_slug: str) → Path | None`
- **Hides:** The hand-curated mapping table. Pure data.
- **Justification:** Shallow but unavoidable — it is a lookup table, and explicit listing is the simplest viable approach. Tested via one unit test that asserts known mappings.
- **Tested through:** The public function.

## Verification Architecture

- **Canonical success state:** Running `apps/evals/teaching_knowledge/run_eval.py` against the full 513-recording cache after the change produces an ASCF outcome mean ≥ 1.687 (= 1.387 + 0.30) with a 95% bootstrap CI whose lower bound exceeds the locked baseline's upper bound (2.507 composite → ASCF-equivalent threshold). Secondary signal: the Tier-1 subset (recordings whose piece has score JSON) has a larger ASCF lift than the Tier-2-only subset.
- **Automated check:** `cd apps/evals && uv run python -m teaching_knowledge.run_eval --limit 513 --out results/bar_analysis_run.jsonl` followed by the bootstrap CI computation in the existing eval analysis modules. The eval is deterministic given a fixed Sonnet seed (the harness already pins this).
- **Harness:** Already exists (`teaching_knowledge.run_eval`). **Task Group 0** runs the existing eval on 10 recordings on the current main commit to confirm the baseline ASCF outcome is reproducible from this branch before changes land. No new harness code required.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/services/bar-analysis-facts.ts` | New module: `buildBarAnalysisFacts` + `BarAnalysisFacts` type | New |
| `apps/api/src/services/bar-analysis-facts.test.ts` | Unit tests for `buildBarAnalysisFacts` | New |
| `apps/api/src/services/accumulator.ts` | `AccumulatedMoment.llmAnalysis: BarAnalysisFacts \| null` (type change) | Modify |
| `apps/api/src/do/session-brain.ts` | At lines ~614 and ~1081 keep `analysis` in scope; at lines ~963 and ~1227 call `buildBarAnalysisFacts` and assign to `accMoment.llmAnalysis` | Modify |
| `apps/api/src/services/prompts.ts` | `buildSynthesisFraming` accepts top moments with optional `bar_analysis` and includes it in the JSON | Modify |
| `apps/api/src/services/prompts.test.ts` | New test asserting the synthesis user msg contains `bar_analysis` JSON when provided | New (or extend existing) |
| `apps/api/src/services/synthesis.ts` | `persistAccumulatedMoments` writes `JSON.stringify(moment.llmAnalysis)` to `reasoning_trace` when `llmAnalysis` is non-null | Modify |
| `apps/evals/teaching_knowledge/piece_score_map.py` | New module with curated `PIECE_SCORE_MAP` and `get_score_path_for_piece` | New |
| `apps/evals/teaching_knowledge/tests/test_piece_score_map.py` | Unit tests for the lookup | New |
| `apps/evals/teaching_knowledge/bar_analysis_local.py` | New module: Python port of Tier-1/2 + worst-chunk selection + facts filtering | New |
| `apps/evals/teaching_knowledge/tests/test_bar_analysis_local.py` | Unit tests for the Python port | New |
| `apps/evals/teaching_knowledge/run_eval.py` | Inject `bar_analysis` into top moments in `build_synthesis_user_msg` | Modify |

## Open Questions

- **Q:** Should the synthesis prompt mention bar numbers explicitly in its instructions, or rely on the teacher to derive them from `bar_analysis`?
  **Default:** Rely on the teacher. The `bar_range` field is already present per moment ("bars 12-14"), and the prompt's existing line "Reference specific musical details (bars, sections, dimensions) when the data supports it" already invites it. Re-evaluate if eval shows the teacher still says "in this passage" instead of bar numbers.

- **Q:** When `analyzeTier2` runs (no score context), the `DimensionAnalysis` records have no `reference_comparison` and the `analysis` strings are sparser (e.g. just "Mean note duration 0.31s"). Is that enough signal to ship for Tier-2 recordings?
  **Default:** Yes. The hypothesis being tested is "any measured fact moves ASCF." Tier-2 facts are still measured facts. The Tier-1-vs-Tier-2 lift comparison built into the success criterion will tell us if the answer differs.

- **Q:** The ASCF lift target ≥ 0.3 was picked as "meaningful enough to be unambiguous." What if the lift is 0.15–0.29 — partial signal, but below threshold?
  **Default:** Treat as inconclusive. Do not ship to no-op the change; investigate whether the threshold (0.15) or cap (3) should be tuned and re-run. Do not invest in tempo/rubato split until the lift is unambiguous.
