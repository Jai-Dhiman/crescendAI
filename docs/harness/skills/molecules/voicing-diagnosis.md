---
name: voicing-diagnosis
tier: molecule
description: |
  Diagnoses imbalance between melody and accompaniment voicing in homophonic textures. fires when MuQ dynamics is below baseline by >= 1 stddev AND AMT velocity-curve shows top-voice/bass-voice ratio is inverted. fires when teacher-style "bring out the melody" feedback would apply. fires when phrasing-arc-analysis flags a missing dynamic peak in the melodic line. fires when exercise-proposal needs a voicing-rooted issue. fires on a passage with >= 4 simultaneous notes. does NOT fire on monophonic passages. does NOT fire when the texture is not predominantly homophonic. does NOT call other molecules.
dimensions: [dynamics, phrasing]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score-alignment for the bar range'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-velocity-curve, fetch-student-baseline, fetch-reference-percentile, fetch-similar-past-observation, extract-bar-range-signals, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.dynamics dimension delta <= -1.0 (below student baseline) AND AMT-derived per-bar top-voice mean velocity is within 5 of bass-voice mean velocity (inverted or flat balance) for >= 60 percent of bars in range AND score-alignment indicates >= 4 simultaneous notes per bar (homophonic). Single-threshold variants of any of these are insufficient.

## When-NOT-to-fire
Skip when score texture is monophonic or 2-voice contrapuntal (compute_key_overlap suggests independent voices). Skip when bar range covers fewer than 2 bars. Skip when MuQ.dynamics delta is non-negative (no audible deficit).

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) to get signals.
2. Call compute-velocity-curve(bar_range, signals.midi_notes) -> velocity curve per bar.
3. Project velocity curve into top-voice and bass-voice means per bar by pitch (top quartile vs bottom quartile of pitches).
4. Call compute-dimension-delta(dimension='dynamics', current=signals.muq_scores[dynamics_index], baseline=fetch-student-baseline(student_id, 'dynamics')) -> z.
5. Branching: if z > -1.0, return early with finding_type='neutral' and a brief note that voicing was within baseline.
6. Compute fraction of bars with |top - bass| < 5; if < 0.6, return early with finding_type='neutral'.
7. Call fetch-reference-percentile('dynamics', signals.muq_scores[dynamics_index], piece_level) for context.
8. Call fetch-similar-past-observation(student_id, 'dynamics', piece_id, bar_range) -> may yield prior context for confidence calibration.
9. Compose DiagnosisArtifact: primary_dimension='dynamics', dimensions=['dynamics','phrasing'], severity = severity_from_z(z) where {-1..-1.5: minor, -1.5..-2.0: moderate, < -2.0: significant}, scope=passed-in scope, bar_range, evidence_refs from signals, one_sentence_finding (max 200 chars, no hedging) describing the voicing imbalance, confidence='high' if past-observation matched else 'medium', finding_type='issue'.

## Concrete example
Input: bar_range=[24,32], MuQ.dynamics=0.42 vs baseline.mean=0.60 stddev=0.08 -> z=-2.25; bars 24-32 show top-voice mean velocity 78 vs bass-voice mean 76 (flat) for 7 of 9 bars; texture has 5+ notes per bar.
Output: DiagnosisArtifact { primary_dimension: 'dynamics', dimensions: ['dynamics','phrasing'], severity: 'significant', scope: 'session', bar_range: [24,32], evidence_refs: ['cache:muq:s31:c12','cache:amt:s31:c12'], one_sentence_finding: 'Melody and accompaniment are voiced almost equally across bars 24-32; the top line is not coming through.', confidence: 'high', finding_type: 'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact (Zod schema) and primary_dimension is 'dynamics'. evidence_refs is non-empty. The procedure terminates with a single artifact regardless of branching.
