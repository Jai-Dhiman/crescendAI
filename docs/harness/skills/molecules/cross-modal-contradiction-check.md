---
name: cross-modal-contradiction-check
tier: molecule
description: |
  Flags cases where MuQ dimension scores and AMT-derived structural features disagree on a passage. The highest-signal teacher diagnostic per the How-to-grep-video wiki finding (cross-modal queries beat single-signal triggers). fires when MuQ.timing is high but onset-drift is large. fires when MuQ.pedaling is high but pedal-overlap-ratio is at extremes. fires when MuQ.articulation is high but key-overlap-ratio direction contradicts score. fires when MuQ.dynamics is high but velocity-curve range is compressed. fires on any chunk where two extractions of the same musical content disagree by >= 1.5 stddev. does NOT fire on chunks with missing AMT (AMT is required for the cross-modal check). does NOT fire on monophonic test signals or sine-wave inputs. does NOT call other molecules.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, AMT pedal CC, score-alignment for the bar range'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [extract-bar-range-signals, align-performance-to-score, compute-dimension-delta, compute-onset-drift, compute-pedal-overlap-ratio, compute-key-overlap-ratio]
---

## When-to-fire
For each of the 4 cross-modal pairs (timing/onset-drift, pedaling/overlap-ratio, articulation/key-overlap, dynamics/velocity-range), check whether MuQ score is in the top quartile (z >= +0.5 vs cohort) AND the corresponding AMT-derived feature is in a contradictory direction. Fire when at least one pair contradicts. Cross-modal pattern is the trigger by definition.

## When-NOT-to-fire
Skip when AMT transcription is unavailable. Skip when bar_range covers fewer than 2 bars (cross-modal requires meaningful sample). Skip when score-alignment is unavailable (timing/articulation pairs need it).

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
3. Compute four pair checks:
   a. timing-pair: compute-dimension-delta('timing', signals.muq_scores[timing_index], cohort_baseline) >= +0.5 AND mean(compute-onset-drift) > 80 ms -> contradicts.
   b. pedaling-pair: compute-dimension-delta('pedaling', ...) >= +0.5 AND compute-pedal-overlap-ratio < 0.30 OR > 0.85 -> contradicts.
   c. articulation-pair: compute-dimension-delta('articulation', ...) >= +0.5 AND compute-key-overlap-ratio direction opposes score articulation in >= 50 percent of bars -> contradicts.
   d. dynamics-pair: compute-dimension-delta('dynamics', ...) >= +0.5 AND velocity-range across the passage is < 25 -> contradicts.
4. If no pair contradicts, return finding_type='neutral'.
5. Pick the most severe contradiction (largest |delta|) as primary_dimension.
6. Compose DiagnosisArtifact: primary_dimension=picked, dimensions=all pairs that contradicted, severity='significant' (cross-modal contradictions are inherently high-severity), one_sentence_finding describing the specific contradiction in teacher language ("MuQ said timing was clean but the onsets drifted 90 ms"), confidence='high', finding_type='issue'. evidence_refs include both the MuQ cache key and the AMT cache key for full audit.

## Concrete example
Input: bar_range=[20,28], MuQ.pedaling delta=+0.8 (excellent), pedal overlap ratio=0.92 (over-pedaled).
Output: DiagnosisArtifact { primary_dimension:'pedaling', dimensions:['pedaling'], severity:'significant', scope:'stop_moment', bar_range:[20,28], evidence_refs:['cache:muq:s31:c7','cache:amt-pedal:s31:c7'], one_sentence_finding:'MuQ rates pedaling clean here, but the pedal was held over harmonic changes for 92 percent of the passage -- the model and the score disagree.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. evidence_refs MUST include at least one MuQ ref AND at least one AMT-derived ref (the cross-modal evidence chain). primary_dimension matches the contradicting pair with the largest delta.
