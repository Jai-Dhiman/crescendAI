---
name: session-synthesis
tier: compound
description: |
  Orchestrates end-of-session diagnosis, prioritization, exercise proposal, and writes one SynthesisArtifact. fires on OnSessionEnd hook (DO alarm, all exit paths, deferred recovery). fires when student stops practicing for 60+ seconds (current pacing rule). fires when student explicitly ends session via UI. fires when 30-minute hard cap is reached. fires when DO state is being checkpointed for sync. does NOT fire mid-session (use live-practice-companion instead). does NOT call other compounds. does NOT bypass the single-write rule.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'all enrichment cache entries for the session, plus prior live-practice-companion stop_moment DiagnosisArtifacts written during the session (per Option B compound-overlap policy)'
  artifacts: [DiagnosisArtifact]
writes: SynthesisArtifact
depends_on: [voicing-diagnosis, pedal-triage, rubato-coaching, phrasing-arc-analysis, tempo-stability-triage, dynamic-range-audit, articulation-clarity-check, exercise-proposal, prioritize-diagnoses, fetch-session-history]
triggered_by: OnSessionEnd
---

## When-to-fire
On OnSessionEnd hook firing for a session with at least 60 seconds of accumulated audio (configurable). The compound consumes whatever stop_moment DiagnosisArtifacts live-practice-companion already wrote during the session, plus the full enrichment cache.

## When-NOT-to-fire
Skip when session has < 60s of audio (no signal). Skip when session has already produced a SynthesisArtifact (idempotency). Skip when the runtime is replaying for eval (eval harness handles its own dispatch).

## Procedure
PHASE 1 (parallel diagnosis sweep):
1. List all session chunks; for each, read prior live-practice-companion stop_moment DiagnosisArtifacts.
2. In parallel, dispatch the 7 diagnosis molecules across plausible bar ranges:
   - voicing-diagnosis on homophonic passages
   - pedal-triage on slow-movement passages
   - rubato-coaching on phrase-end passages in rubato repertoire
   - phrasing-arc-analysis on each marked phrase
   - tempo-stability-triage on motoric passages in non-rubato repertoire
   - dynamic-range-audit on passages with dynamic markings
   - articulation-clarity-check on contrapuntal or fast-articulation passages
3. Collect all DiagnosisArtifacts written by molecules above PLUS the prior live-companion stop_moment artifacts into one list.

PHASE 2 (prioritize):
4. Call atom prioritize-diagnoses(all_diagnoses) -> ranked list.
5. Take top 3 issues for focus_areas; take top 2 strengths for strengths; remaining go into diagnosis_refs without surfacing.

PHASE 3 (exercise proposal, sequential per top diagnosis):
6. For each of top 3 focus_area diagnoses (severity in {moderate, significant}, finding_type='issue'), call exercise-proposal(diagnosis) per Option B (artifact passed as input).
7. Collect ExerciseArtifacts; cap at 3.

PHASE 4 (longitudinal hook):
8. Call atom fetch-session-history(student_id, window_days=14).
9. Detect recurring_pattern: if 2+ of the past 5 sessions had a DiagnosisArtifact with the same primary_dimension as today's top focus_area, write one-sentence pattern (e.g., "third session in a row over-pedaling slow movements"). Otherwise null.

PHASE 5 (single write):
10. Compose SynthesisArtifact: session_id, synthesis_scope='session', strengths (top 2), focus_areas (top 3 with {dimension, one_liner=diagnosis.one_sentence_finding, severity}), proposed_exercises (artifact ids), dominant_dimension=top focus_area's primary_dimension, recurring_pattern, next_session_focus (derived from top focus_area + exercise), diagnosis_refs (all collected), headline derived LAST from structured fields by composing a 300-500 char teacher-voice paragraph that opens with a strength, names the dominant focus, references the proposed exercise, and closes encouragingly.

## Concrete example
Input: 25-minute session of Chopin Ballade No 1, 4 stop_moment artifacts already written by live-practice-companion.
Output: SynthesisArtifact { session_id:'sess_42', synthesis_scope:'session', strengths:[{dimension:'phrasing', one_liner:'Clean shape across the second theme.'}], focus_areas:[{dimension:'pedaling', one_liner:'Over-pedaled bars 12-16 in slow movement.', severity:'significant'},{dimension:'timing', one_liner:'Tempo dragged through bars 40-48.', severity:'moderate'}], proposed_exercises:['ex:abc1','ex:abc2'], dominant_dimension:'pedaling', recurring_pattern:'Third session in a row over-pedaling slow movements.', next_session_focus:'Pedal release timing in the slow movement.', diagnosis_refs:['diag:1','diag:2','diag:3','diag:4','diag:5','diag:6','diag:7'], headline:'You played with real shape in the second theme today and the climax landed where it should. The thing pulling the picture out of focus is the pedal in the slow passages -- this is the third session in a row -- and we are going to spend tomorrow on releasing it cleanly between phrases. Your hands know what they want; the foot just needs to catch up.' }.

## Post-conditions
Output validates as SynthesisArtifact (Zod schema, synthesis_scope='session'). Exactly ONE artifact is written by this compound (single-write). headline is derived from structured fields, never written before them. dominant_dimension equals focus_areas[0].dimension when focus_areas is non-empty.
