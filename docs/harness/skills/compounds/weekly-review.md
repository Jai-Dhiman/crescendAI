---
name: weekly-review
tier: compound
description: |
  Aggregates the past week of sessions into one longitudinal SynthesisArtifact (scope=weekly) with a mandatory recurring_pattern. fires on OnWeeklyReview scheduled hook (default Sunday evening). fires when student manually requests a week-in-review summary. fires when 7+ sessions have accumulated since last weekly review. fires when student transitions to a new piece (review of prior week). fires when teacher requests a longitudinal report. does NOT fire on sessions newer than the window cutoff. does NOT call other compounds. does NOT bypass the single-write rule.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student session store for the past 7 days; per-session SynthesisArtifacts and DiagnosisArtifacts'
  artifacts: [SynthesisArtifact, DiagnosisArtifact]
writes: SynthesisArtifact
depends_on: [voicing-diagnosis, pedal-triage, rubato-coaching, phrasing-arc-analysis, tempo-stability-triage, dynamic-range-audit, articulation-clarity-check, exercise-proposal, prioritize-diagnoses, fetch-session-history]
triggered_by: OnWeeklyReview
---

## When-to-fire
On OnWeeklyReview hook firing for a student with at least 3 sessions in the past 7 days. The compound aggregates session-level SynthesisArtifacts plus their underlying DiagnosisArtifacts.

## When-NOT-to-fire
Skip when fewer than 3 sessions in window (insufficient longitudinal signal). Skip when a weekly-review SynthesisArtifact already exists for the same window (idempotency).

## Procedure
PHASE 1 (longitudinal fetch):
1. Call atom fetch-session-history(student_id, window_days=7) -> sessions.

PHASE 2 (aggregate diagnoses):
2. Flatten all DiagnosisArtifacts across the week's sessions.
3. Re-dispatch any of the 7 diagnosis molecules where the per-session DiagnosisArtifacts are insufficient (e.g., a passage that recurs across sessions but was never analyzed at session-level depth) -- this re-extends prior coverage.

PHASE 3 (prioritize):
4. Call atom prioritize-diagnoses(all_week_diagnoses) -> ranked list.
5. Detect dominant patterns: group by (primary_dimension, piece_id) and count.

PHASE 4 (recurring_pattern derivation - MANDATORY for weekly):
6. recurring_pattern is REQUIRED. Derive: pick the (primary_dimension, piece_id) pair appearing in the most sessions; compose a one-sentence pattern (e.g., "Pedaling regressed in 4 of 7 Chopin Ballade sessions this week, concentrated in slow-movement passages"). If no pattern repeats >= 3 sessions, recurring_pattern says "No single recurring issue this week; coverage was distributed across phrasing, timing, and pedaling."

PHASE 5 (exercise proposals for top recurring patterns):
7. For each of top 2 recurring focus_areas, call exercise-proposal(representative diagnosis) per Option B.

PHASE 6 (single write):
8. Compose SynthesisArtifact: session_id (synthetic, e.g., 'weekly:stu_42:2026-W17'), synthesis_scope='weekly', strengths (top 2 across week), focus_areas (top 3), proposed_exercises (top 2 from PHASE 5), dominant_dimension, recurring_pattern (mandatory non-null), next_session_focus, diagnosis_refs (all collected), headline derived LAST in week-summary teacher voice (300-500 chars).

## Concrete example
Input: student stu_42, 5 sessions on Chopin Ballade No 1 over 7 days, pedaling issue in bars 12-16 appeared in 4 of 5 sessions.
Output: SynthesisArtifact { session_id:'weekly:stu_42:2026-W17', synthesis_scope:'weekly', strengths:[{dimension:'phrasing', one_liner:'Phrase shape improved across the week.'},{dimension:'dynamics', one_liner:'Dynamic range opened up in the second theme.'}], focus_areas:[{dimension:'pedaling', one_liner:'Over-pedaled in slow movement bars 12-16 across 4 sessions.', severity:'significant'}], proposed_exercises:['ex:weekly_1','ex:weekly_2'], dominant_dimension:'pedaling', recurring_pattern:'Pedaling regressed in 4 of 5 Chopin Ballade sessions this week, concentrated in slow-movement passages bars 12-16.', next_session_focus:'Pedal-release drill bars 12-16 before any full playthrough.', diagnosis_refs:['diag:w_1','diag:w_2','diag:w_3','diag:w_4','diag:w_5'], headline:'This week was a strong one for shape. The phrase contours opened up across all five sessions and the second theme found its dynamic range. The thread to pull is the pedal in the slow movement -- four of five sessions saw the same blur in bars 12-16. Spend the first ten minutes of every session next week on the no-pedal-pass drill there before doing a full playthrough.' }.

## Post-conditions
Output validates as SynthesisArtifact with synthesis_scope='weekly' AND recurring_pattern non-null (Zod-enforced). Exactly ONE artifact written. headline derived from structured fields.
