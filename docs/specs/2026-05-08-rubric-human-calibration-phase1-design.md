# Rubric Human Calibration Protocol — Phase 1 (Founder-Only) Design

**Goal:** Produce a frozen `filter_recipe.py` artifact that tells the Stage 2 SFT data pipeline which of the 14 v2-rubric sub-scores are trustworthy, at what weight, and with what bias correction — derived from founder ratings on 200 stratified syntheses against the LLM judge.

**Not in scope:**
- Phase 2 expert rater layer (ASCF outcome / Scaffolded outcome / Style-Consistent outcome).
- Pairwise preference agreement for Stage 4 DPO.
- Any rewrite of `apps/evals/shared/prompts/synthesis_quality_judge_v2.txt`. The rubric is frozen for the protocol's duration; rewrites trigger a full re-rate, not an in-protocol edit.
- Re-generation of the `apps/evals/results/baseline_v1.jsonl` source pool.
- Stage 2 SFT data pipeline itself — this protocol emits a recipe; consumption is a downstream workstream.

## Problem

`apps/evals/teacher_model/TRAINING_PLAN.md` (Section 8, Strategic Gates) places "rubric human calibration r ≥ 0.7" on the critical path twice — gating Stage 2 (SFT data quality filter) and Stage 4 (DPO preference signal). Today there is no protocol, no rater workflow, no statistical method spec, and no contract between calibration output and downstream consumers. Without this artifact:

- Stage 2 cannot filter SFT training data — it would either accept all judge-rated syntheses (importing judge bias and noise wholesale) or filter on an arbitrary cutoff with no defensibility.
- Stage 4 DPO would train on synthetic preference pairs whose signal direction is unverified — preference noise instead of preference signal.
- The "Sonnet baseline 2.483/3.0 (n=513)" reference in `apps/evals/results/baseline_v1_aggregate.json` is referenced in 6+ places across `TRAINING_PLAN.md` but has never been validated against any human judgment.

## Solution (from the user's perspective)

The founder runs a CLI rater over 4-6 weeks (~35-45 hours rating time) on 200 stratified syntheses drawn from `apps/evals/results/baseline_v1.jsonl`. At protocol end, three artifacts exist:

- `apps/evals/teacher_model/calibration/artifacts/calibration_report.json` — per-sub-score weighted κ + bootstrap CI + threshold-decision agreement + bucket assignment (TRUSTED / TRUSTED-WITH-OFFSET / CEILING-ARTIFACT / UNTRUSTED).
- `apps/evals/teacher_model/calibration/artifacts/drift_report.json` — intra-rater κ on anchor duplicates (drift gate) + judge-vs-judge κ on day-1 vs day-30 re-runs.
- `apps/evals/teacher_model/calibration/artifacts/filter_recipe.py` — the public API: `COMPOSITE_PASS_THRESHOLD`, `WEIGHTED_SUB_SCORES`, `BIAS_CORRECTIONS`, `SANITY_FILTERS`. The Stage 2 SFT pipeline imports this and nothing else from the protocol.

A go/no-go decision lives in the calibration report: aggregate gate is **≥7 of 11 Phase-1 sub-scores in TRUSTED or TRUSTED-WITH-OFFSET buckets**.

## Design

### The two-phase split

The brainstorm explicitly partitioned the 14 sub-scores (7 dims × {process, outcome}):

- **Phase 1 (this design): 11 sub-scores rated by founder.** All 7 process scores + Autonomy outcome + Tone outcome (pure structural) + Concrete Artifact outcome + Specific Praise outcome (constrained-rubric, using the briefing's `muq_means` and `skill_bucket` as ground truth for level-appropriateness checks).
- **Phase 2 (deferred, separate design): 3 sub-scores requiring expert pianist judgment.** ASCF outcome, Scaffolded outcome, Style-Consistent outcome.

Founder is an amateur musician; ceiling on expert-judgment dims is bounded. Calibrating those dims with founder ratings would encode amateur error as ground truth — the worst failure mode for downstream DPO. The partition prevents this by construction.

### Statistic choice

The plan as written says "r ≥ 0.7" — Pearson r is the wrong statistic for both downstream uses:

- The data is 0-3 ordinal, not interval — Pearson overstates agreement when ratings cluster.
- Stage 2 needs **threshold decision agreement** (does the synthesis pass the 2.5 composite cutoff?), not score-level agreement.
- Stage 4 needs pairwise preference agreement — deferred to Phase 2.

This protocol uses **Cohen's weighted κ (quadratic weights)** for per-sub-score agreement and **threshold-decision agreement** (κ on the binary ≥2.5 composite outcome) as a separate statistic. Pairwise preference is deferred.

### Four-bucket failure handling

| Bucket | Criterion | Filter consumption |
|---|---|---|
| TRUSTED | weighted κ ≥ 0.6 AND threshold agreement ≥ 80% | Full weight in `WEIGHTED_SUB_SCORES`. |
| TRUSTED-WITH-OFFSET | κ ≥ 0.6 BUT threshold agreement < 80% AND mean offset ≥ 0.3 with low offset variance | Recorded in `BIAS_CORRECTIONS`; corrected score used for threshold. |
| CEILING-ARTIFACT | rater score variance < 0.2 OR judge score variance < 0.2 | Recorded in `SANITY_FILTERS` as never-say-violation auto-rejects only; no κ-based weight. |
| UNTRUSTED | κ < 0.6 with non-systematic disagreement | Excluded from filter entirely. Triggers documented rubric-rewrite recommendation. |

### Sample design

- **Source:** `apps/evals/results/baseline_v1.jsonl` — 2271 rows, 920 valid (1351 are baseline-side API errors and skipped). Seven judge dimensions per row, each with `process` + `outcome` + `score` (composite). Composers in data: Bach, Beethoven, Chopin, Debussy.
- **Total drawn from pool:** 230 (200 main + 30 reserved holdout, never rated, used for OFFSET re-validation).
- **Score-band stratification (within main 200):** 40% threshold band (composite 2.3-2.7, n=80) + 20% high (≥2.7, n=40) + 20% low (≤2.0, n=40) + 20% weak-dim band (ASCF process score ≤ 1.5, n=40).
- **Era min-quotas (across main 200):** Baroque/Bach ≥30, Classical/Beethoven ≥30, Romantic/Chopin ≥30, Impressionist/Debussy ≥30. Modern absent in data; no quota.
- **Skill-bucket min-quotas (across main 200):** beginner (skill_bucket 1-2) ≥50, intermediate (3) ≥50, advanced (4-5) ≥50.
- **Anchor injection:** ~20 silent duplicates within the main 200 (founder must not recognize). Used for intra-rater κ drift gate.
- **Pilot:** 30 of the 200 are flagged for early analysis at end of week 1. Rubric rewrite is the only allowed adjustment if pilot finds the protocol unrateable; partition adjustments require restart.

### Drift controls

- **Anchor re-rating** for intra-rater κ ≥ 0.7 gate (computed on ~20 duplicates).
- **Judge re-run** at day 1 and day 30 on the same 20 anchors via existing `judge_synthesis_v2`. Gate: judge-vs-judge κ ≥ 0.85.
- **Session cap:** 15 ratings per sitting, 2 sittings per day, hard-enforced by CLI.
- **Rubric re-read** prompt every 50 ratings.

### Public API contract (filter_recipe.py)

```python
COMPOSITE_PASS_THRESHOLD: float = 2.5

WEIGHTED_SUB_SCORES: dict[str, float] = {
    # sub_score_id -> weight in [0.0, 1.0]
    # Populated only with TRUSTED and TRUSTED-WITH-OFFSET sub-scores.
}

BIAS_CORRECTIONS: dict[str, float] = {
    # sub_score_id -> additive offset applied to raw judge score before threshold.
    # Populated only with TRUSTED-WITH-OFFSET sub-scores.
}

SANITY_FILTERS: list[str] = [
    # CEILING-ARTIFACT sub-scores demoted to never-say-violation auto-rejects.
]

# Sub-score IDs used: "{dim_slug}_process" or "{dim_slug}_outcome"
# Examples: "ascf_process", "tone_outcome", "concrete_artifact_outcome".
```

### Trade-offs chosen

- **Single rater, ceiling-bounded.** Accepts founder ceiling explicitly via Phase 1 / Phase 2 partition rather than implicitly via low trust scores. Trade-off: Stage 4 DPO blocks until Phase 2 OR uses synthetic-only pairs on expert dims.
- **Cohen's weighted κ over Pearson r.** Trade-off: numerically distinct from the plan's stated "r ≥ 0.7"; gate rewritten to "weighted κ ≥ 0.6" to match the data type and downstream use.
- **40% threshold-band oversample over uniform sampling.** Trade-off: less representative of the 920-row pool's distribution; deliberately weights mass at the filter's actual decision boundary.
- **Per-sub-score buckets over single aggregate gate.** Trade-off: more complex artifact emission; one stubborn sub-score doesn't block Stage 2.
- **Holdout-30 reserved at sample-selection time.** Trade-off: 30 fewer slots in the rating pool; OFFSET corrections become verifiable rather than circular.

## Modules

| Module | Interface | Hides | Tested through |
|---|---|---|---|
| `era_lookup.py` | `composer_to_era(composer: str) -> Era` | A 4-row composer→era table + the contract for unknown composers ("Other"). | Public function only. |
| `select_sample.py` | `select_sample(source_path, target_n, holdout_n, anchor_n, seed) -> dict` | Multi-criteria stratification under conflicting quotas, holdout reservation, anchor injection schedule. | Returned manifest dict (band proportions, era quotas, skill quotas, holdout disjoint, anchor flags). |
| `rater_cli.py` | `python -m ... rate --manifest M --output O` (CLI), `present_synthesis()` (testable inner function) | Judge-score blinding, anchor masking, session cap enforcement, resume-from-crash via append-only jsonl. | CLI-level: stdout/stderr/log scan for blinding leaks. Inner-function level: rating capture + cap enforcement + resume on synthetic input. |
| `judge_rerun.py` | `rerun_anchors(anchor_ids, baseline_path, output_path, run_label) -> None` | Iteration over anchor IDs, calls to `judge_synthesis_v2`, jsonl serialization. | Output file shape on a stub-judged run. |
| `analyze_drift.py` | `analyze_drift(ratings_path, judge_runs_path) -> DriftReport` | Intra-rater κ on duplicate-pair extraction; judge-vs-judge κ on day1/day30 pairing. | Report fields against synthetic ratings with known intra-rater κ. |
| `analyze_calibration.py` | `calibrate(ratings_path, baseline_path) -> CalibrationReport` | Cohen's weighted κ implementation, bootstrap CI, threshold-decision binary κ, mean-offset computation, 4-bucket routing. | Report bucket assignments against synthetic data with known κ values (0.3, 0.6, 0.8) and known offset patterns. |
| `emit_recipe.py` | `emit(calibration_report, drift_report, output_path) -> None` | Mapping from bucket → recipe field; bias-correction encoding; recipe Python source generation. | Round-trip: emit recipe, import the resulting module, verify Stage-2-style filtering decisions on synthetic syntheses. |

All tests verify behavior through these public interfaces. No tests mock internal collaborators; no tests assert on private state.

## File Changes

| File | Change | Type |
|---|---|---|
| `apps/evals/teacher_model/calibration/__init__.py` | empty package marker | New |
| `apps/evals/teacher_model/calibration/era_lookup.py` | composer→era classifier | New |
| `apps/evals/teacher_model/calibration/select_sample.py` | stratified sampler | New |
| `apps/evals/teacher_model/calibration/rater_cli.py` | rater CLI with blinding/cap/resume | New |
| `apps/evals/teacher_model/calibration/judge_rerun.py` | judge re-run scheduler | New |
| `apps/evals/teacher_model/calibration/analyze_drift.py` | drift analyzer | New |
| `apps/evals/teacher_model/calibration/analyze_calibration.py` | calibration analyzer with weighted κ + bucket routing | New |
| `apps/evals/teacher_model/calibration/emit_recipe.py` | filter_recipe.py emitter | New |
| `apps/evals/teacher_model/calibration/artifacts/.gitkeep` | runtime artifact dir | New |
| `apps/evals/teacher_model/calibration/tests/__init__.py` | test package marker | New |
| `apps/evals/teacher_model/calibration/tests/test_era_lookup.py` | era lookup tests | New |
| `apps/evals/teacher_model/calibration/tests/test_select_sample.py` | sampler stratification + quotas + holdout + anchors | New |
| `apps/evals/teacher_model/calibration/tests/test_rater_cli.py` | blinding security boundary + capture + cap + resume | New |
| `apps/evals/teacher_model/calibration/tests/test_judge_rerun.py` | rerun jsonl shape | New |
| `apps/evals/teacher_model/calibration/tests/test_analyze_drift.py` | intra-rater κ on synthetic duplicates | New |
| `apps/evals/teacher_model/calibration/tests/test_analyze_calibration.py` | weighted κ + threshold agree + bucket routing on synthetic ground-truth | New |
| `apps/evals/teacher_model/calibration/tests/test_emit_recipe.py` | round-trip recipe import + filter behavior | New |

No existing files modified.

## Open Questions

- Q: When the holdout-30 is consumed for OFFSET re-validation, does the protocol formally re-emit `filter_recipe.py` (versioned v2), or do OFFSET corrections that fail the holdout downgrade in-place to UNTRUSTED in the original recipe?
  Default: downgrade in-place. The recipe is regenerated once at protocol end; in-place downgrade is a single emission step rather than a versioning workflow.

- Q: If the judge re-run gate fails (judge κ < 0.85 day1 vs day30), does Phase 1 invalidate the entire protocol or proceed with a documented degradation?
  Default: invalidate. Judge drift means the entire `baseline_v1.jsonl` reference is suspect, which is a workstream concern beyond this protocol; this protocol pauses and escalates rather than papering over it.

- Q: Are the era classification edges (Beethoven as Classical vs Early Romantic) worth a config-file lookup vs hardcoded constant?
  Default: hardcoded constant in `era_lookup.py`. With only 4 composers and no plan to add more before Phase 2, a config file would be over-engineering.
