---
name: articulation-clarity-check
tier: molecule
description: |
  Identifies execution mismatches between notated articulation (slurs vs staccato) and observed key-overlap behavior. fires when MuQ.articulation is below baseline AND key-overlap-ratio direction does not match score articulation markings. fires when teacher-style "make the staccato shorter" feedback would apply. fires on contrapuntal repertoire requiring voice independence. fires when score has explicit slurs or staccato dots in range. fires on Bach or Mozart fast passagework. does NOT fire on freely interpretive Romantic passages without notated articulation. does NOT fire when score articulation is missing. does NOT call other molecules.
dimensions: [articulation]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score articulation markings (slur, staccato, tenuto)'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-key-overlap-ratio, align-performance-to-score, extract-bar-range-signals, fetch-reference-percentile, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.articulation delta z <= -0.8 AND compute-key-overlap-ratio direction (positive=legato, negative=staccato) is opposite to score articulation in >= 50 percent of bars. Single-threshold MuQ-only triggers are insufficient because legato/staccato direction matters.

## When-NOT-to-fire
Skip when bar_range has no notated articulation markings. Skip on improvisatory or free-form passages. Skip when MuQ.articulation delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment with articulation markings.
3. For each bar in range, project to monophonic top voice and call compute-key-overlap-ratio -> ratio_per_bar.
4. Classify per-bar score articulation: 'slur' (overlap expected), 'staccato' (gap expected), 'detache' (~zero expected).
5. Compute per-bar mismatch: bar mismatches if score=slur AND ratio<=0, OR score=staccato AND ratio>=0.
6. If mismatch fraction < 0.5 OR z > -0.8, return finding_type='neutral'.
7. Compose DiagnosisArtifact: primary_dimension='articulation', dimensions=['articulation'], severity by z, one_sentence_finding referencing the dominant mismatch direction.

## Concrete example
Input: bar_range=[5,12] in Bach prelude, score marks all 8 bars staccato, observed ratios are +0.10 to +0.20 (legato) for 6 of 8 bars, MuQ.articulation z=-1.4.
Output: DiagnosisArtifact { primary_dimension:'articulation', dimensions:['articulation'], severity:'moderate', scope:'session', bar_range:[5,12], evidence_refs:['cache:muq:s31:c2','cache:amt:s31:c2'], one_sentence_finding:'The staccato bars 5-12 are sustaining into each other; the notes are blurring rather than separating.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type may be 'issue' or 'neutral'. evidence_refs is non-empty.
