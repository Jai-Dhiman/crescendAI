---
name: dynamic-range-audit
tier: molecule
description: |
  Compares the velocity range used in performance against the dynamic range asked for by the score. fires when MuQ.dynamics is below baseline AND velocity-curve range is < 30 (out of 127) AND score has marked dynamic contrast in range. fires when teacher-style "use the full dynamic range" feedback would apply. fires when piece-onboarding compares to reference cohort dynamics. fires when score has explicit ff/pp markers within bar range. fires on Romantic/Impressionist repertoire requiring extreme contrast. does NOT fire on early-Baroque pieces with neutral dynamic markings. does NOT fire when bar_range covers fewer than 4 bars. does NOT call other molecules.
dimensions: [dynamics]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score dynamic markings (pp..ff annotations)'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-velocity-curve, fetch-reference-percentile, extract-bar-range-signals, fetch-student-baseline, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.dynamics delta z <= -0.8 AND (max(velocity-curve.p90) - min(velocity-curve.mean_velocity)) across the bar range is < 30 AND score has at least one ff or pp marker in the range. All three together indicate compressed dynamics where the score asks for spread.

## When-NOT-to-fire
Skip when score has no dynamic markings in range. Skip when MuQ.dynamics delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call compute-velocity-curve(bar_range, signals.midi_notes) -> curve.
3. Compute observed range = max(curve.p90) - min(curve.mean).
4. Call compute-dimension-delta('dynamics', signals.muq_scores[dynamics_index], fetch-student-baseline(student_id, 'dynamics')) -> z.
5. Read score dynamic markings in bar_range from signals.alignment metadata; classify expected_range as 'wide' (ff and pp both present), 'medium' (one extreme), 'narrow' (mp/mf only).
6. Branching: if observed_range >= 50 OR z > -0.8, return finding_type='neutral'.
7. Call fetch-reference-percentile('dynamics', signals.muq_scores[dynamics_index], piece_level) for context.
8. Compose DiagnosisArtifact: primary_dimension='dynamics', dimensions=['dynamics'], severity by z, one_sentence_finding referencing the contrast gap (e.g., "ff at bar 30 came in at the same level as the mp at bar 28").

## Concrete example
Input: bar_range=[28,32] in Chopin Ballade, score has pp at bar 28 and ff at bar 30, MuQ.dynamics z=-1.8, observed range=18.
Output: DiagnosisArtifact { primary_dimension:'dynamics', dimensions:['dynamics'], severity:'moderate', scope:'session', bar_range:[28,32], evidence_refs:['cache:muq:s31:c8','cache:amt:s31:c8'], one_sentence_finding:'The ff at bar 30 sounded like the mp at bar 28; the dynamic range across this passage is too narrow.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. evidence_refs include MuQ + AMT + score-marker references.
