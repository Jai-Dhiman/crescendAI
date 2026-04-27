---
name: tempo-stability-triage
tier: molecule
description: |
  Distinguishes tempo drift, intentional rubato, and loss of pulse. fires when MuQ.timing is below baseline AND IOI correlation with score is low AND signed drift trends monotonically. fires when teacher-style "find the pulse again" feedback would apply. fires on technical or motoric passages where pulse stability is the goal. fires when bar-rate slowdown over the passage exceeds 5 percent. fires when the student is in non-rubato repertoire. does NOT fire on Romantic repertoire with notated rubato (use rubato-coaching). does NOT fire on cadenza or improvisatory sections. does NOT call other molecules.
dimensions: [timing]
reads:
  signals: 'AMT midi_notes, score-alignment, MuQ 6-dim scores'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-ioi-correlation, compute-onset-drift, align-performance-to-score, extract-bar-range-signals, fetch-student-baseline, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.timing delta z <= -1.0 AND compute-ioi-correlation r < 0.4 AND signed onset drift trend (linear regression slope across notes) is monotonic (>= 80 percent same-sign drift). All three together separate drift from rubato.

## When-NOT-to-fire
Skip when piece metadata indicates rubato repertoire. Skip when fewer than 12 aligned notes. Skip when MuQ.timing delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
3. Call compute-onset-drift(bar_range, signals.midi_notes, alignment) -> drift_per_note.
4. Call compute-ioi-correlation(signals.midi_notes, alignment) -> r.
5. Call compute-dimension-delta('timing', signals.muq_scores[timing_index], fetch-student-baseline(student_id, 'timing')) -> z.
6. Compute fraction of notes with same-sign drift (sign of signed drift); if < 0.8, return finding_type='neutral' (drift is non-monotonic, likely rubato; defer to rubato-coaching).
7. Classify subtype: 'slowing' (positive trend), 'rushing' (negative trend), 'unstable' (high variance, low correlation).
8. Compose DiagnosisArtifact: primary_dimension='timing', dimensions=['timing'], severity by z, one_sentence_finding referencing subtype and bar count.

## Concrete example
Input: bar_range=[1,16] (motoric Bach prelude), z=-1.6, r=0.2, drift trend +5,+10,+18,+30,+45,... (monotonic positive).
Output: DiagnosisArtifact { primary_dimension:'timing', dimensions:['timing'], severity:'significant', scope:'session', bar_range:[1,16], evidence_refs:['cache:muq:s31:c1','cache:amt:s31:c1'], one_sentence_finding:'The pulse slowed gradually across bars 1-16; by the end you were 30 percent under tempo.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type='neutral' when drift is non-monotonic. evidence_refs is non-empty.
