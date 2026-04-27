---
name: fetch-student-baseline
tier: atom
description: |
  Fetches a student's per-dimension MuQ score baseline (rolling mean + stddev over last N sessions). fires when compute-dimension-delta needs a baseline. fires when any diagnosis molecule needs to know what is normal for this student. fires when weekly review computes regression direction. fires when piece onboarding compares current to prior pieces. fires when prioritize-diagnoses weights by personalized severity. does NOT fire when student has fewer than 3 prior sessions (caller falls back to cohort percentile). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student memory layer (V7 surface, V5 stub): per-dimension session-mean MuQ scores indexed by student_id and dimension'
  artifacts: []
writes: 'scalar:Baseline = { dimension: Dim, mean: number, stddev: number, n_sessions: number } | null'
depends_on: []
---

## When-to-fire
Caller passes a student_id and dimension. Atom returns rolling mean + stddev over the last 10 sessions for that dimension, or null if fewer than 3 sessions are available.

## When-NOT-to-fire
Do not invoke when student_id is unknown. Do not invoke for the cohort baseline -- that is fetch-reference-percentile.

## Procedure
1. Look up student's per-session MuQ mean for the requested dimension across last 10 sessions.
2. If n < 3, return null.
3. Compute mean and stddev across the n session-means.
4. Return { dimension, mean, stddev, n_sessions: n }.

## Concrete example
Input: student_id='stu_42', dimension='pedaling'.
Output: { dimension: 'pedaling', mean: 0.65, stddev: 0.10, n_sessions: 8 }.

## Post-conditions
Returned value is null OR has all four fields populated; mean is in [0, 1]; stddev is non-negative; n_sessions >= 3 when not null.
