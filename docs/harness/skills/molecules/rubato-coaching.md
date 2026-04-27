---
name: rubato-coaching
tier: molecule
description: |
  Distinguishes intentional, returned rubato from uncompensated drift. fires when MuQ.timing is below baseline AND IOI correlation with score is low AND drift does not net to zero across the phrase. fires when teacher-style "let it breathe but come back" feedback would apply. fires on Romantic-period repertoire with notated rubato cues. fires when score has fermata or ritardando markers in the range. fires after a clear cadence to assess phrase-shape completion. does NOT fire on metronomically rigid passages with neutral MuQ.timing. does NOT fire on first-pass sight-reading where timing is incidental. does NOT call other molecules.
dimensions: [timing, phrasing, interpretation]
reads:
  signals: 'AMT midi_notes, score-alignment with phrase boundaries, MuQ 6-dim scores'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-onset-drift, compute-ioi-correlation, align-performance-to-score, extract-bar-range-signals, fetch-student-baseline, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.timing delta z <= -0.8 AND compute-ioi-correlation r < 0.3 (low correlation between performer and score IOIs) AND signed onset drift does not return to within 50ms of zero by phrase end. All three together indicate rubato that wandered without resolution.

## When-NOT-to-fire
Skip when fewer than 8 aligned notes (compute-ioi-correlation will be unstable). Skip when score has no phrase boundary inside the bar range (no return point to evaluate). Skip when MuQ.timing delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
3. Call compute-onset-drift(bar_range, signals.midi_notes, alignment) -> drift_per_note.
4. Call compute-ioi-correlation(signals.midi_notes, alignment) -> r.
5. Call compute-dimension-delta('timing', signals.muq_scores[timing_index], fetch-student-baseline(student_id, 'timing')) -> z.
6. Branching: if z > -0.8 OR r >= 0.3, return finding_type='neutral'.
7. Compute net signed drift at phrase end (last note's signed drift); if abs(net) <= 50ms, return finding_type='strength' with one_sentence_finding praising returned rubato.
8. Otherwise: classify subtype as 'rushed' (mean signed < 0) or 'dragged' (mean signed > 0).
9. Compose DiagnosisArtifact: primary_dimension='timing', dimensions=['timing','phrasing','interpretation'], severity by |z|, one_sentence_finding referencing subtype.

## Concrete example
Input: bar_range=[40,48] (8-bar phrase ending in cadence), z=-1.5, r=0.1, signed drift trend +30,+45,+80,+120 ms -> dragged, no return.
Output: DiagnosisArtifact { primary_dimension:'timing', dimensions:['timing','phrasing','interpretation'], severity:'moderate', scope:'session', bar_range:[40,48], evidence_refs:['cache:muq:s31:c14','cache:amt:s31:c14'], one_sentence_finding:'The rubato through bars 40-48 stretched without coming back; the phrase loses its shape.', confidence:'medium', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type can be 'issue', 'strength', or 'neutral' per branching above. evidence_refs is non-empty.
