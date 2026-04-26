# Eval Harness v4 — Playbook Wiring + Signal Ablation + Atomic-Skill Rubrics

**Goal:** Add three eval-harness capabilities — top-2 cluster-based teacher-voice injection from a single source of truth, a 4-condition signal ablation that tests whether MuQ + AMT signals are load-bearing in synthesis, and a binary 8×5 atomic-skill rubric matrix that decomposes synthesis failures into specific pedagogical moves.

**Not in scope:**
- Replacing the locked 7-dim composite rubric (atomic matrix augments, never replaces).
- Live-LLM tests in the test suite (judge calls use recorded fixtures).
- Phase 2 finetuned-Qwen teacher A/B work (covered by `project_teacher_model_finetuning.md`).
- iOS surfaces (web/API only).
- Stratified session sampling for the ablation (random sample from holdout split, seeded).
- Persisting atomic-matrix results to a separate file (rows are added to the existing JSONL).

## Problem

1. **Style injection is incomplete.** `prompts.ts:126` injects only `style-rules.json` (era → dim weights). `playbook.yaml`'s richer pedagogical content (5 teaching-style clusters with `language_patterns`, exemplars, `when_to_use`) lives only in `apps/evals/teaching_knowledge/data/` and is not referenced by either prod synthesis or `run_eval.py`. The prod teacher voice is style-aware but cluster-blind.

2. **Signals may not be load-bearing.** Per the MuChoMusic finding, audio-LLMs frequently generate from language priors regardless of audio input. We have not tested whether our teacher's outputs change when MuQ + AMT signals are corrupted. If they don't, the synthesis is decorative grounding.

3. **7-dim composite is opaque on failure.** When a synthesis scores poorly on the locked rubric, we cannot localize *which pedagogical move* failed (e.g., "the voicing diagnosis was generic" vs "the exercise lacked tempo target"). Diagnostic decomposition does not exist.

## Solution (from the user's perspective)

- **Founder/eng:** edit one canonical `playbook.yaml` to update teacher voice. `just compile-playbook` produces the prod JSON. Both `apps/api` synthesis and `apps/evals` eval reflect the change on the next call.
- **Founder/eng:** run `uv run python -m teaching_knowledge.ablation.run_ablation` once. Receive a verdict `signals_load_bearing: true|false|equivocal` plus per-condition score deltas and four-quadrant counts.
- **Founder/eng:** run `uv run python -m teaching_knowledge.run_eval` as before. When a synthesis scores below threshold on the 7-dim composite, the JSONL row also contains an `atomic_matrix` field — an 8×5 boolean grid plus per-move "attempted" flag — so failure mode is inspectable post-hoc.

## Design

### Approach summary

- **Sub-goal 1 (playbook wiring):** Augment `playbook.yaml` with a `triggers:` block per cluster (signal-derived scoring formula). Move file to `shared/teacher-style/playbook.yaml`. `just compile-playbook` precompiles to `apps/api/src/lib/playbook.json` (committed). Two parallel modules — `apps/api/src/services/teacher_style.ts` (zod-validated) and `apps/evals/shared/teacher_style.py` (pydantic-validated) — load the playbook, run identical cluster-scoring math against signals, return top-2 selection, format two prompt blocks (`<teacher_voice>` for primary cluster, `<also_consider>` for secondary). Both `prompts.ts:buildSynthesisFraming` and `run_eval.py:build_synthesis_user_msg` append the blocks after the existing `<style_guidance>` block. A behavior-parity test feeds N fixture signal vectors through both implementations and asserts identical cluster selection.

- **Sub-goal 2 (signal ablation):** New module `apps/evals/teaching_knowledge/ablation/`. `corrupt_signals.py` provides three deterministic, seeded corruption functions (`shuffle`, `marginal`, `flip`). `run_ablation.py` selects 20 holdout sessions via seeded random sample, runs synthesis under all 4 conditions (real + 3 corruptions), invokes the locked dual-judge pipeline (Gemma-4 + GPT-5.4-mini) via `judge_synthesis_v2`, persists 80 rows to JSONL. `analyze.py` computes mean composite score per condition, mean cosine similarity (sentence-transformers `all-MiniLM-L6-v2`) between real and corrupted outputs of the same session, four-quadrant counts, and applies the pre-registered decision rule. Stop-rule: equivocal `Δ_flipped` triggers expansion to N=50 (manual re-invoke with `--n 50`).

- **Sub-goal 3 (atomic-skill rubrics):** New static content `apps/evals/shared/prompts/atomic_skill_rubrics.json` defining 8 rubrics (one per pedagogical move) × 5 binary outcome criteria each. New `apps/evals/shared/judge_atomic.py` provides `judge_atomic_matrix(synthesis_text, context, provider, model) → AtomicMatrixResult`. Single-judge call (Gemma-4 only) returns per-move `attempted: bool` plus `criteria: [bool; 5]` per move. Threshold gate in `run_eval.py`: when 7-dim composite mean < 2.0 (out of 3.0), call atomic judge and embed result in the row.

### Hardest decisions resolved

1. **Triggers live inside `playbook.yaml`** alongside `when_to_use` prose, not in a separate file. Prevents drift between prose and executable rules. Validated on load (zod/pydantic).

2. **Top-2 cluster cardinality.** Real sessions warrant multiple teaching moves; one is too few, three+ fragments voice. Top-2 is bounded prompt budget (~500 tokens) and clean for ablation.

3. **Diagnostic-only atomic matrix.** Single-judge run only when 7-dim drops below threshold. Cost stays close to current.

4. **Battery corruption (3 types), not single corruption.** Each tests a different failure mode (shuffle = uses signals at all; marginal = uses signal structure; flip = sign-following).

5. **Embedding similarity + dual-judge score, not pairwise judging.** Reuses locked judges; new pairwise prompt would need calibration debt. Four-quadrant analysis catches the "high-score-but-same-text" failure mode.

### Drafted content

#### `playbook.yaml` cluster triggers (auditable)

The selection function computes a score per cluster from these signals derived from session_data:

- `max_neg_dev` = max(0, max over dims of `-dev`) where `dev = score - baseline` and `dev < 0`. Captures worst dimension below baseline. `0` if no negatives ≥ 0.05.
- `max_pos_dev` = same construction for `dev > 0`.
- `n_significant` = count of dims with `|dev| ≥ 0.1`.
- `drilling_present` = `len(drilling_records) > 0`.
- `drilling_improved` = drilling final score − drilling first score > 0.15 (only when `drilling_present`).
- `duration_min` = `duration_seconds / 60`.
- `mode_count` = number of unique modes in `practice_pattern.modes` (or `1` for `"continuous_play"` string).
- `has_piece` = `piece.title` and `piece.title != "Unknown"`.

Top-2 selection: compute all 5 cluster scores, take the two highest. Tie-break priority order: `technical-corrective > positive-encouragement > artifact-based > guided-discovery > motivational`. If both top scores < 0.3 (no cluster fired confidently), fall back to `(technical-corrective, positive-encouragement)` with primary first.

The 5 trigger formulas (committed to the YAML as YAML expressions in a constrained DSL — see schema below):

```yaml
clusters:
  - name: "Technical-corrective feedback"
    triggers:
      score: "1.5 * max_neg_dev + 0.3 * n_significant - 0.5 * (1 if drilling_improved else 0)"
  - name: "Artifact-based teaching"
    triggers:
      score: "(1.0 * max_neg_dev if max_neg_dev >= 0.15 else 0) + 0.5 * (1 if duration_min < 20 else 0) + 0.5 * (1 if has_piece else 0)"
  - name: "Positive-encouragement / praise"
    triggers:
      score: "1.5 * max_pos_dev + 1.5 * (1 if drilling_improved else 0) + 0.5 * (1 if max_neg_dev < 0.1 else 0)"
  - name: "Motivational / autonomy-supportive statements"
    triggers:
      score: "0.5 * (1 if duration_min < 10 else 0) + 0.5 * (1 if max_neg_dev < 0.1 and max_pos_dev < 0.1 else 0)"
  - name: "Guided-discovery / scaffolding feedback"
    triggers:
      score: "0.5 * (1 if duration_min > 30 else 0) + 0.5 * (1 if mode_count >= 3 else 0) + 1.0 * (1 if drilling_present and not drilling_improved else 0)"
```

The DSL is a restricted subset: arithmetic + comparisons + boolean conditionals over the named signals. Implemented in TS via a hand-written evaluator over a parsed AST (no `eval`); in Python via a hand-written evaluator over the same AST. Schema validation rejects formulas referencing names outside the allowed signal set.

#### Injected prompt block format

After the existing `<style_guidance>` block, append:

```
<teacher_voice cluster="positive-encouragement">
Register: warm, expressive, often using adjectives (beautiful, dramatic, lovely).
Tone: enthusiastic, supportive, personal.
Exemplar: "Your 16th notes were really even — I was really impressed with your control."
</teacher_voice>

<also_consider cluster="technical-corrective">
Apply when: a measured dimension deviates from stylistic expectations.
Exemplar: "In bars 8-12, your pedaling is blurring the harmonies — try a half-pedal change on beat 3."
</also_consider>
```

The exemplar shown is the *first* item from the cluster's `good_examples` list (deterministic, not random). `language_patterns.register` and `language_patterns.tone` come directly from YAML.

#### Atomic-skill rubrics (drafted, auditable)

8 rubrics, 5 binary outcome criteria each. Each criterion is "Y/N" — does the synthesis text exhibit this property?

**1. voicing_diagnosis** — applies when synthesis addresses balance between voices/hands.
1. Names a specific voice or hand (melody / inner / bass / LH / RH).
2. References a specific bar number or time range (e.g., "bars 3–6", "the second phrase").
3. Prescribes a concrete dynamic or touch target (e.g., "below mp", "lighter").
4. Avoids vague balance language without specifics ("better balance", "more musical").
5. Names exactly one actionable practice strategy (e.g., "play LH alone", "exaggerate the contrast").

**2. pedal_triage** — applies when synthesis addresses pedaling.
1. Names which pedal phenomenon (over-pedaling, late change, blurred harmony, dry, flutter).
2. References a specific bar / harmony change / beat where the issue lives.
3. Prescribes a concrete pedal action (half-pedal, change on beat 3, release before next chord).
4. Connects pedaling to an audible musical outcome (clarity, resonance, color).
5. Includes a practice strategy (no-pedal first, listen-only pass, mark the pedal in score).

**3. rubato_coaching** — applies when synthesis addresses tempo flexibility.
1. Distinguishes expressive rubato from unintentional drift.
2. Names where in the phrase the rubato lives (peak, approach, recovery).
3. References style or composer expectation for rubato use.
4. Avoids generic "play with feeling" framing; prescribes a measurable shape.
5. Includes a practice strategy (metronome → freed, conduct the breath, sing the line).

**4. phrasing_arc_analysis** — applies when synthesis addresses musical sentence shape.
1. Names at least 2 of {start, peak, resolution} of the phrase explicitly.
2. References a specific bar or beat where the peak / breath sits.
3. Connects phrase shape to a dynamic, agogic, or tonal-color decision.
4. Addresses the line as a unit (does not isolate notes).
5. Includes a practice strategy (sing the line, slur in groups, breathe at phrase end).

**5. tempo_stability_triage** — applies when synthesis addresses pulse/timing.
1. Distinguishes drift (gradual) from rushing (acute) from dragging (acute) — not just "uneven".
2. Names where in the piece the instability appears (specific bars or transition).
3. Identifies a likely cause (technical demand, hand-coordination, breath, performance anxiety).
4. Prescribes a tempo target or a stable reference (metronome BPM, internal subdivision).
5. Includes a practice strategy (subdivision, slow-fast-slow, hands-separately at tempo).

**6. dynamic_range_audit** — applies when synthesis addresses dynamics.
1. Names the actual range observed (compressed, narrow, mostly mf, lacks pp, lacks ff).
2. References specific bars where the issue is sharpest.
3. Connects range to expressive function (rising arc, climax, surprise contrast).
4. Avoids "play louder/softer" without target dynamic level or context.
5. Includes a practice strategy (extreme contrasts as exercise, listen for ceiling/floor).

**7. articulation_clarity_check** — applies when synthesis addresses note attack/release.
1. Names the articulation type at issue (legato break, staccato length, accent placement, slur shape).
2. References specific bars or a specific figure (e.g., "the 16th-note runs").
3. Connects articulation to era/style expectation (Baroque clarity, Romantic blend).
4. Avoids vague "clearer" framing; prescribes a touch or finger-level technique.
5. Includes a practice strategy (slow detached, dotted rhythms, hands separately listening).

**8. exercise_proposal** — applies when synthesis proposes a concrete drill.
1. Names the specific skill the exercise targets (one dim or a specific gesture).
2. Specifies passage scope (exact bars / hands / tempo).
3. Specifies progression (start tempo / target tempo, or simple → complex chain).
4. Specifies a stop criterion (% accuracy, comfort, "until even").
5. Avoids generic "practice slowly" framing without measurable terms.

The judge prompt instructs Gemma-4 to first emit `attempted: Y/N` per move, then if `Y`, emit `criteria: [Y, Y, N, Y, N]` (length 5) for that move. If `attempted: N`, criteria are recorded as `null`. Output schema validated on parse.

### Decision rule (signal ablation)

Pre-registered. Let `Δ_score(c) = mean(real_composite) − mean(corrupted_composite_c)` per condition `c ∈ {shuffle, marginal, flip}`. Let `mean_sim_flip` = mean cosine similarity (sentence-transformers) between real and flipped outputs of the same session.

**Verdict `signals_load_bearing: true`** iff all hold:
- `Δ_score(flip) > 0.3` (out of 3.0 composite scale)
- `Δ_score(shuffle) > 0.15` AND `Δ_score(marginal) > 0.15`
- `mean_sim_flip < 0.85`

**Verdict `signals_load_bearing: false`** iff `Δ_score(flip) ≤ 0.15` OR `mean_sim_flip ≥ 0.92`.

**Verdict `signals_load_bearing: equivocal`** in the gap. Triggers the stop-rule: `analyze.py` prints "expand to N=50" guidance; founder re-runs `run_ablation.py --n 50`.

Both judges (Gemma-4 + GPT-5.4-mini) score every condition; verdict uses the *minimum* `Δ_score` across the two judges per condition (conservative). `mean_sim_flip` does not depend on judge.

### Threshold for atomic-matrix gate

When `mean(judge_dimensions[].score) < 2.0` (mean across the 7 dims, single judge is fine), call `judge_atomic_matrix`. Threshold rationale: the 7-dim rubric scale is 0–3; 2.0 is the boundary between "adequate" and "weak" per the existing rubric definition. Configurable via `--atomic-threshold` flag (default 2.0). Set to `4.0` (always-fire) or `0.0` (never-fire) for special runs.

Atomic-matrix call adds one Workers AI call per gated session. At a 30% gate-fire rate (conservative estimate based on current judge means), the cost overhead is ~30% of single-judge baseline — within budget.

## Modules

**`shared/teacher-style/playbook.yaml`**
- Interface: YAML schema — `clusters: [{name, note?, dominant_strategies, when_to_use, dimension_priorities, language_patterns: {register, tone, examples}, good_examples: [{text, source_id}], bad_examples?, distinguishing_features?, triggers: {score: <DSL string>}}]`
- Hides: 5 clusters' research-derived content + executable selection logic adjacent to its prose.
- Tested through: TS + Python loaders parse and validate; mismatches against schema fail loud.

**`apps/api/src/services/teacher_style.ts`**
- Interface: `selectClusters(signals: ClusterSignals) → {primary: ClusterRef, secondary: ClusterRef}`; `formatTeacherVoiceBlocks(selection) → string`.
- Hides: DSL evaluator (tokenize → parse AST → eval against signals dict), tie-break ordering, fallback default, exemplar selection (first `good_examples` item).
- Tested through: vitest unit tests over fixture signal vectors.

**`apps/evals/shared/teacher_style.py`**
- Interface: `select_clusters(signals: ClusterSignals) → ClusterSelection`; `format_teacher_voice_blocks(selection) → str`.
- Hides: same DSL evaluator implementation (Python AST evaluator).
- Tested through: pytest unit tests + parity test with TS fixtures.

**`scripts/compile_playbook.py`**
- Interface: `python scripts/compile_playbook.py` reads `shared/teacher-style/playbook.yaml`, writes `apps/api/src/lib/playbook.json`. Returns nonzero if YAML invalid or output unchanged drift detected (CI flag `--check`).
- Hides: YAML → JSON serialization rules (preserves ordered keys, UTF-8, no anchors).
- Tested through: pytest invokes CLI, compares output to a committed snapshot fixture.

**`apps/evals/teaching_knowledge/ablation/corrupt_signals.py`**
- Interface: `corrupt(top_moments, mode, seed, all_top_moments) → top_moments'`. `mode ∈ {"shuffle", "marginal", "flip"}`. `all_top_moments` is the corpus needed for shuffle/marginal sampling.
- Hides: per-mode corruption logic, RNG seeding for determinism.
- Tested through: pytest determinism (same seed → same output), boundary tests (flip preserves dim names), distributional test (marginal samples come from observed empirical distribution).

**`apps/evals/teaching_knowledge/ablation/run_ablation.py`**
- Interface: CLI: `python -m teaching_knowledge.ablation.run_ablation [--n 20] [--seed 42] [--out ablation_v1.jsonl] [--judges gemma,gpt]`.
- Hides: holdout session sampling, 4-condition orchestration per session, dual-judge invocation, JSONL persistence, resume-safety.
- Tested through: pytest end-to-end on a 2-session fixture with mocked LLM clients (recorded responses).

**`apps/evals/teaching_knowledge/ablation/analyze.py`**
- Interface: CLI: `python -m teaching_knowledge.ablation.analyze --in ablation_v1.jsonl [--out report.md]`. Prints verdict + table.
- Hides: cosine similarity computation, four-quadrant binning, decision-rule application, stop-rule guidance.
- Tested through: pytest on synthetic JSONL fixtures covering true/false/equivocal verdicts.

**`apps/evals/shared/prompts/atomic_skill_rubrics.json`**
- Interface: static JSON, 8 entries, each `{move_id, applies_when, criteria: [{id, text}; 5]}`.
- Hides: full 40-criterion content.
- Tested through: schema validation test (loaded into `judge_atomic` and asserts shape).

**`apps/evals/shared/judge_atomic.py`**
- Interface: `judge_atomic_matrix(synthesis_text, context, provider, model) → AtomicMatrixResult` where result is `{moves: [{move_id, attempted, criteria: [bool; 5] | null}; 8], judge_model, latency_ms}`.
- Hides: prompt construction, provider call, response parsing, schema validation.
- Tested through: pytest with recorded judge response fixture; schema-error path tested with malformed response.

**`apps/api/src/services/prompts.ts` (modify)**
- Change: `buildSynthesisFraming` accepts the existing `topMoments` and `drillingRecords` and additionally calls `selectClusters` + `formatTeacherVoiceBlocks`, appends the result after `<style_guidance>`. No signature change visible to callers (the data is already passed in).
- Tested through: extension of `prompts.test.ts` — assert `<teacher_voice>` appears, primary cluster matches expectation for fixture inputs, omitted when `formatTeacherVoiceBlocks` returns empty (it never does — fallback default ensures non-empty).

**`apps/evals/teaching_knowledge/run_eval.py` (modify)**
- Change A (sub-goal 1): `build_synthesis_user_msg` calls `select_clusters` + `format_teacher_voice_blocks`, appends after `get_style_guidance`.
- Change B (sub-goal 3): after judge call, if `mean(judge_dimensions[].score) < atomic_threshold`, call `judge_atomic_matrix`, attach result to row under `atomic_matrix` key. CLI flag `--atomic-threshold 2.0`.
- Tested through: pytest on a 1-session fixture with recorded LLM responses, asserts `atomic_matrix` present below threshold, absent above.

**`justfile` (modify)**
- Add: `compile-playbook` recipe. Add: `check-playbook-sync` recipe (runs `compile_playbook.py --check`).
- Tested through: shell test invokes recipe, asserts JSON output.

## File Changes

| File | Change | Type |
|------|--------|------|
| `shared/teacher-style/playbook.yaml` | Move from old location, add `triggers:` per cluster | New |
| `apps/evals/teaching_knowledge/data/playbook.yaml` | Remove (moved) | Delete |
| `apps/api/src/lib/playbook.json` | Compiled artifact, committed | New |
| `apps/api/src/services/teacher_style.ts` | Cluster selection + formatting | New |
| `apps/api/src/services/teacher_style.test.ts` | Behavior tests | New |
| `apps/evals/shared/teacher_style.py` | Mirror of TS module | New |
| `apps/evals/shared/test_teacher_style.py` | Pytest + cross-language parity | New |
| `apps/api/src/services/prompts.ts` | Append `<teacher_voice>` blocks | Modify |
| `apps/api/src/services/prompts.test.ts` | Add tests for cluster blocks | Modify |
| `apps/evals/teaching_knowledge/run_eval.py` | Append blocks; gate atomic call | Modify |
| `apps/evals/teaching_knowledge/test_run_eval.py` | Add atomic-gate test | New |
| `scripts/compile_playbook.py` | YAML → JSON precompile | New |
| `scripts/test_compile_playbook.py` | CLI test | New |
| `justfile` | `compile-playbook` + `check-playbook-sync` recipes | Modify |
| `apps/evals/teaching_knowledge/ablation/__init__.py` | Empty | New |
| `apps/evals/teaching_knowledge/ablation/corrupt_signals.py` | 3 corruption fns | New |
| `apps/evals/teaching_knowledge/ablation/test_corrupt_signals.py` | Determinism + correctness | New |
| `apps/evals/teaching_knowledge/ablation/run_ablation.py` | Orchestrator | New |
| `apps/evals/teaching_knowledge/ablation/test_run_ablation.py` | Fixture-based E2E | New |
| `apps/evals/teaching_knowledge/ablation/analyze.py` | Cosine + verdict | New |
| `apps/evals/teaching_knowledge/ablation/test_analyze.py` | Verdict tests | New |
| `apps/evals/shared/prompts/atomic_skill_rubrics.json` | 8×5 rubric content | New |
| `apps/evals/shared/judge_atomic.py` | Single-judge atomic call | New |
| `apps/evals/shared/test_judge_atomic.py` | Recorded-response test | New |

## Open Questions

- Q: Does `apps/evals/teaching_knowledge/data/playbook.yaml` have any other importers I missed? Default: `grep -r "playbook.yaml" apps/` will run as Task 0 of the plan; if hits, those callsites are updated to point at the new path.
- Q: Should the atomic-matrix threshold default to mean-across-7dim or the *minimum* across 7dim? Default: mean. Minimum would fire too aggressively (one weak dim shouldn't trigger full decomposition).
- Q: Should ablation reuse the inference cache (deterministic synthesis given fixed signals) or re-run synthesis live? Default: re-run live — corrupted signals don't have cache entries by construction. Real-condition synthesis can reuse cache only when corruption is "real" (passthrough). Implementation always re-runs for consistency.
- Q: Should `compile-playbook` be enforced via pre-commit hook or CI check? Default: CI check via `just check-playbook-sync` in the GitHub Actions workflow; do not add a pre-commit hook (founder workflow).
