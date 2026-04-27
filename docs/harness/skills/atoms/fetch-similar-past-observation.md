---
name: fetch-similar-past-observation
tier: atom
description: |
  Fetches the most similar prior diagnosis artifact for a given student, dimension, and bar context. fires when a diagnosis molecule wants to know if this finding is recurring. fires when session-synthesis needs to surface a repeating issue. fires when exercise-proposal wants prior-exercise context for the same diagnosis pattern. fires when weekly review groups recurring patterns. fires when piece-onboarding looks for analogous prior pieces. does NOT fire for cross-student lookups (privacy boundary). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student diagnosis artifact store, indexed by (student_id, dimension, piece_id, bar_range_overlap)'
  artifacts: []
writes: 'scalar:PastObservation = { artifact_id: string, session_id: string, days_ago: number, similarity_score: number } | null'
depends_on: []
---

## When-to-fire
Caller passes a student_id, dimension, piece_id, and bar_range. Atom returns the single most similar past DiagnosisArtifact for that student or null if no match exceeds similarity threshold 0.5.

## When-NOT-to-fire
Do not invoke for cross-student lookups. Do not invoke when the dimension is not in the 6 teacher-grounded dimensions.

## Procedure
1. Query student diagnosis store for prior artifacts matching (student_id, dimension).
2. Compute similarity = 0.5 * (piece_id == match) + 0.5 * (bar_range overlap fraction).
3. Return the highest-scoring match if score >= 0.5, else null.
4. Include days_ago = (now - artifact.created_at) / 86400_000.

## Concrete example
Input: student_id='stu_42', dimension='pedaling', piece_id='chopin_op23', bar_range=[12,16].
Output: { artifact_id: 'diag:abc789', session_id: 'sess_31', days_ago: 5, similarity_score: 0.85 }.

## Post-conditions
Returned value is null OR similarity_score >= 0.5. days_ago is non-negative.
