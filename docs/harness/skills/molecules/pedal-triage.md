---
name: pedal-triage
tier: molecule
description: |
  Distinguishes over-pedaling, under-pedaling, and pedal-timing issues by combining MuQ pedaling delta with AMT pedal CC overlap ratio against score-aligned harmonic boundaries. fires when MuQ.pedaling delta below baseline AND pedal overlap ratio is outside expected band. fires when teacher-style "release the pedal" feedback would apply. fires when slow-movement playthrough shows muddy harmony. fires when score-alignment indicates harmony changes that the pedal did not respect. fires when student baseline shows recurring pedal issues. does NOT fire on harpsichord-style pieces with no pedal expectation. does NOT fire when AMT pedal CC stream is missing. does NOT call other molecules.
dimensions: [pedaling]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, AMT pedal CC64 timeline, score-alignment, score harmony-change markers'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-pedal-overlap-ratio, align-performance-to-score, fetch-student-baseline, extract-bar-range-signals, compute-dimension-delta, fetch-similar-past-observation]
---

## When-to-fire
Cross-modal pattern: MuQ.pedaling delta vs student baseline <= -1.0 AND pedal overlap ratio is either > 0.85 (over) or < 0.30 (under) for the bar range. Single-threshold variants are insufficient.

## When-NOT-to-fire
Skip when AMT pedal CC stream is unavailable. Skip when piece metadata indicates no pedal (early Baroque). Skip when MuQ.pedaling delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call compute-pedal-overlap-ratio(bar_range, signals.midi_notes, signals.pedal_cc) -> ratio.
3. Call compute-dimension-delta('pedaling', signals.muq_scores[pedaling_index], fetch-student-baseline(student_id, 'pedaling')) -> z.
4. Branching:
   a. If z > -1.0, return finding_type='neutral'.
   b. If ratio > 0.85, classify subtype='over_pedal'.
   c. If ratio < 0.30, classify subtype='under_pedal'.
   d. Otherwise, call align-performance-to-score(signals.midi_notes, score) and check whether pedal release coincides with harmony change markers within 100ms; if not, classify subtype='timing'.
5. Call fetch-similar-past-observation(student_id, 'pedaling', piece_id, bar_range) -> raises confidence to 'high' if matched within 14 days.
6. Compose DiagnosisArtifact: primary_dimension='pedaling', dimensions=['pedaling'], severity by z (same buckets as voicing-diagnosis), one_sentence_finding referencing the subtype in plain teacher language ("over-pedaled", "dry", "released late"), evidence_refs include the pedal_cc cache key.

## Concrete example
Input: bar_range=[12,16] in slow movement, MuQ.pedaling z=-2.1, pedal overlap ratio=0.92 -> over_pedal.
Output: DiagnosisArtifact { primary_dimension:'pedaling', dimensions:['pedaling'], severity:'significant', scope:'session', bar_range:[12,16], evidence_refs:['cache:muq:s31:c5','cache:amt-pedal:s31:c5'], one_sentence_finding:'Over-pedaled through bars 12-16; the harmonies are blurring into one wash.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact and primary_dimension is 'pedaling'. The molecule terminates with one artifact regardless of branching.
