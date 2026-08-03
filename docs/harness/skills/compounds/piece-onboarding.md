---
name: piece-onboarding
tier: compound
description: |
  Runs once when a student plays a new piece for the first time; orients them by comparing initial performance to reference cohort dynamics and phrasing. Writes one SynthesisArtifact (scope=piece_onboarding) where focus_areas all have severity='minor' (orientation, not diagnosis). fires on OnPieceDetected first-time hook (zero-config piece ID confirms a piece never seen before in this student's repertoire). fires when student opts into "introduce me to this piece" UI. fires after the first complete passage of the piece is captured. fires when reference performances are available for the piece. fires when the piece has explicit phrase/dynamic markings to compare against. does NOT fire on subsequent plays of the same piece (use session-synthesis instead). does NOT call other compounds. does NOT bypass the single-write rule.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'first-pass enrichment cache for the new piece; reference performance MuQ scores and AMT for the same piece (cohort)'
  artifacts: []
writes: SynthesisArtifact
depends_on: [phrasing-arc-analysis, dynamic-range-audit, fetch-reference-percentile, prioritize-diagnoses]
triggered_by: OnPieceDetected
---

## When-to-fire
On OnPieceDetected hook firing for a student where the detected piece is not present in the student's repertoire history AND the piece has reference performances loaded AND the student has played at least one complete passage.

## When-NOT-to-fire
Skip when piece has been played before by this student. Skip when no reference performance is available. Skip when the captured passage is shorter than 30 seconds.

## Procedure
PHASE 1 (orientation diagnoses):
1. In parallel, dispatch:
   a. phrasing-arc-analysis on each marked phrase in the captured passage (compared against reference cohort).
   b. dynamic-range-audit on the captured passage (compared against reference cohort).
2. For each diagnosis molecule, OVERRIDE the severity field to 'minor' regardless of computed value (this is orientation, not diagnosis -- per the SynthesisArtifact piece_onboarding contract).

PHASE 2 (cohort comparison):
3. For each of the 6 dimensions, call atom fetch-reference-percentile(dimension, student's MuQ score, piece_level) to position the student in the cohort.

PHASE 3 (prioritize and propose):
4. Call atom prioritize-diagnoses on the orientation diagnoses (which all have severity='minor').
5. Route any introductory prescription through the `ExerciseRoutingDecision` contract; this compound does not call a proposal molecule.

PHASE 4 (single write):
6. Compose SynthesisArtifact: session_id, synthesis_scope='piece_onboarding', strengths (any cohort comparisons where student is above 60th percentile), focus_areas (all severity='minor', up to 3), prescribed_exercise (at most one decision from PHASE 3), dominant_dimension (lowest cohort percentile), recurring_pattern=null (no longitudinal data for new piece), next_session_focus (one suggestion for the next play of this piece), diagnosis_refs, headline derived LAST in orientation/excitement teacher voice (300-500 chars), e.g., "Welcome to the Chopin Ballade -- here is where the music sits for you right now and where to grow into it."

## Concrete example
Input: student stu_42 plays first 60s of Chopin Ballade Op 23 (never before recorded), reference cohort exists.
Output: SynthesisArtifact { session_id:'sess_43', synthesis_scope:'piece_onboarding', strengths:[{dimension:'timing', one_liner:'Tempo sat naturally in the opening; you are at the 70th cohort percentile.'}], focus_areas:[{dimension:'pedaling', one_liner:'Pedaling sits at the 35th cohort percentile -- worth attention as you grow into the piece.', severity:'minor'},{dimension:'phrasing', one_liner:'The second-theme arc is asking for a clearer peak.', severity:'minor'}], prescribed_exercise:{kind:'own_passage_loop',target_dimension:'pedaling',bar_range:[12,16],tempo_factor:0.7}, dominant_dimension:'pedaling', recurring_pattern:null, next_session_focus:'Spend the first 5 minutes hearing the slow-movement pedal release.', diagnosis_refs:['diag:onb_1','diag:onb_2'], headline:'Welcome to the Chopin Ballade -- this is a piece that rewards slow listening before fast playing. Your tempo settled naturally in the opening, which is the harder thing. Where to grow into it: the slow movement asks for very particular pedal handling, and the second theme wants a clearer peak in the line. Both are habits that develop over weeks.' }.

## Post-conditions
Output validates as SynthesisArtifact with synthesis_scope='piece_onboarding' AND every focus_areas[].severity='minor' (Zod-enforced). recurring_pattern is null (no longitudinal data for new piece). Exactly ONE artifact written.
