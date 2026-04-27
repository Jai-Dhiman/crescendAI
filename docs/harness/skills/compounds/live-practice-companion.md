---
name: live-practice-companion
tier: compound
description: |
  Continuous in-session companion that dispatches stop-moment diagnosis on every high-probability STOP chunk. Single write per STOP event: one DiagnosisArtifact (scope=stop_moment) consumed later by session-synthesis. fires on OnRecordingActive hook for every chunk where classify-stop-moment probability >= 0.65. fires for live web-app sessions with WebSocket push. fires for iOS sessions when student requests "How was that?". fires when STOP-classifier is enabled in the session config. fires when at least one diagnosis molecule's preconditions are met. does NOT fire on chunks below STOP probability threshold. does NOT call other compounds. does NOT mutate prior artifacts.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'streaming MuQ 6-dim scores, AMT midi_notes, AMT pedal CC, score-alignment for each 15s chunk as it arrives'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [cross-modal-contradiction-check, rubato-coaching, classify-stop-moment]
triggered_by: OnRecordingActive
---

## When-to-fire
For each newly-written enrichment cache chunk during an active recording, call atom classify-stop-moment(muq_scores). If probability >= 0.65, fire the dispatch in the procedure below.

## When-NOT-to-fire
Skip chunks below threshold. Skip when session is in non-coaching mode (e.g., warm-up). Skip when AMT transcription is missing for the chunk (the cross-modal molecule cannot run). Do not duplicate-fire on the same chunk.

## Procedure
PHASE 1 (atom check):
1. Call atom classify-stop-moment(chunk.muq_scores) -> p_stop. If < 0.65, return without writing.

PHASE 2 (parallel cross-modal dispatch):
2. In parallel, dispatch:
   a. cross-modal-contradiction-check on chunk.bar_range (always; cross-modal is the highest-signal teacher diagnostic per the grep-video wiki).
   b. rubato-coaching on chunk.bar_range IF the score has a phrase boundary inside this chunk (skip otherwise).

PHASE 3 (single write):
3. Collect any non-neutral DiagnosisArtifact returned by the dispatched molecules.
4. If multiple non-neutral artifacts, pick the one with primary_dimension matching the lowest MuQ dimension (most-broken signal wins).
5. Write exactly ONE DiagnosisArtifact with scope='stop_moment' and bar_range derived from the chunk's bar coverage.
6. Push to the WebSocket channel (web) or queue for "How was that?" response (iOS) -- delivery is a runtime concern, not part of this compound's logic.

## Concrete example
Chunk c12 covers bars 20-28, MuQ scores [0.5, 0.7, 0.4, 0.6, 0.5, 0.5] (low pedaling). classify-stop-moment returns 0.78. cross-modal-contradiction-check fires and finds contradiction on pedaling pair.
Output: DiagnosisArtifact { primary_dimension:'pedaling', dimensions:['pedaling'], severity:'significant', scope:'stop_moment', bar_range:[20,28], evidence_refs:['cache:muq:s31:c12','cache:amt-pedal:s31:c12'], one_sentence_finding:'MuQ rates pedaling clean here, but the pedal held through three harmonic changes.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact with scope='stop_moment'. Exactly one artifact per STOP event (single write). bar_range is non-null. The compound never modifies prior artifacts.
