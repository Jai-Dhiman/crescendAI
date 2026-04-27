---
name: phrasing-arc-analysis
tier: molecule
description: |
  Assesses dynamic and timing arc shape across a complete phrase. fires when MuQ.phrasing is below baseline AND velocity-curve does not exhibit a single peak per phrase AND signed onset drift does not return at phrase end. fires when teacher-style "where is the peak of this phrase" feedback would apply. fires when score has explicit phrase-boundary markers in the bar range. fires when piece-onboarding compares to reference performances. fires on long lyrical lines with rising-falling shape. does NOT fire on through-composed passages without phrase boundaries. does NOT fire on technical etude passages where shape is secondary. does NOT call other molecules.
dimensions: [phrasing, dynamics]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score phrase-boundary markers, score-alignment'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-velocity-curve, compute-onset-drift, align-performance-to-score, extract-bar-range-signals, fetch-reference-percentile, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.phrasing delta z <= -0.8 AND velocity-curve over the phrase does not exhibit a unimodal peak (peak-bar is at start or end, or the curve has multiple peaks of similar height) AND signed onset drift across the phrase does not converge to <= 50ms at phrase end. All three together indicate weak arc shape.

## When-NOT-to-fire
Skip when bar_range does not contain at least one complete phrase as marked in the score. Skip when the phrase is shorter than 4 bars. Skip when MuQ.phrasing delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call compute-velocity-curve(bar_range, signals.midi_notes) -> curve.
3. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
4. Call compute-onset-drift(bar_range, signals.midi_notes, alignment) -> drift.
5. Call compute-dimension-delta('phrasing', signals.muq_scores[phrasing_index], baseline_or_percentile) -> z.
6. Detect peak: argmax(curve.mean_velocity); if peak is at index 0 or last, or if there are multiple bars within 5 velocity of the peak, mark shape='flat' or 'multi-peaked'.
7. Compute drift convergence: abs(drift[last].signed); if > 50ms AND shape is flat/multi-peaked, return finding_type='issue'.
8. If z > -0.8 AND shape is unimodal, return finding_type='strength' praising arc.
9. Compose DiagnosisArtifact: primary_dimension='phrasing', dimensions=['phrasing','dynamics'], severity by z, one_sentence_finding describing where the peak should sit.

## Concrete example
Input: bar_range=[16,24] (8-bar phrase, score peak marked at bar 20), curve peaks at bar 17 (early peak), z=-1.2.
Output: DiagnosisArtifact { primary_dimension:'phrasing', dimensions:['phrasing','dynamics'], severity:'moderate', scope:'session', bar_range:[16,24], evidence_refs:['cache:muq:s31:c10','cache:amt:s31:c10'], one_sentence_finding:'The phrase peaks at bar 17 instead of bar 20; the climax of the line is arriving early.', confidence:'medium', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type may be 'issue' or 'strength' per branching. evidence_refs include both MuQ and AMT cache keys.
