---
name: fetch-session-history
tier: atom
description: |
  Fetches a student's prior SynthesisArtifacts and aggregated DiagnosisArtifacts within a date window. fires when weekly-review needs longitudinal context. fires when session-synthesis populates recurring_pattern. fires when piece-onboarding compares to past piece onboardings. fires when live-practice-companion checks for repeating issues across recent sessions. fires when fetch-similar-past-observation needs broader scope. does NOT fire across students. does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student session store: SynthesisArtifacts and DiagnosisArtifacts indexed by (student_id, created_at)'
  artifacts: []
writes: 'scalar:SessionHistory = { sessions: { session_id: string, created_at: number, synthesis: SynthesisArtifact, diagnoses: DiagnosisArtifact[] }[] }'
depends_on: []
---

## When-to-fire
Caller passes a student_id and date window (default: last 7 days). Atom returns all sessions in window with their synthesis and diagnoses.

## When-NOT-to-fire
Do not invoke for cross-student queries. Do not invoke without an explicit window (default applies, but caller should pass one for clarity).

## Procedure
1. Query student session store for sessions where created_at falls in window.
2. For each session, fetch its SynthesisArtifact (one) and all associated DiagnosisArtifacts.
3. Return ordered by created_at descending (most recent first).

## Concrete example
Input: student_id='stu_42', window_days=7.
Output: { sessions: [ { session_id: 'sess_31', created_at: 1714003200000, synthesis: {...}, diagnoses: [...] }, ... ] } -- 4 sessions in the past week.

## Post-conditions
Output sessions are ordered by created_at descending; each session has exactly one synthesis; diagnoses lists may be empty but are always present.
