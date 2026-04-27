---
name: detect-passage-repetition
tier: atom
description: |
  Detects whether the same score bar range was practiced multiple times in close temporal succession within a session. fires when session-synthesis aggregates per-passage attempts. fires when exercise-proposal checks if the student is already drilling a passage. fires when live-practice-companion identifies repeated trouble spots. fires when weekly-review counts repetitions across sessions. fires when piece-onboarding observes practice strategy. does NOT fire across sessions (use session-history for that). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'score-alignment entries within a session, grouped by chunk_id and timestamped'
  artifacts: []
writes: 'scalar:RepetitionList = { bar_range: [number, number], attempt_count: number, first_attempt_ms: number, last_attempt_ms: number }[]'
depends_on: []
---

## When-to-fire
Caller passes a session_id. Atom returns all bar ranges practiced 2+ times within the session.

## When-NOT-to-fire
Do not invoke across sessions (use fetch-session-history). Do not invoke before score alignment has completed for the session.

## Procedure
1. List all aligned bar ranges in the session, ordered by start time.
2. Group by overlapping bar_range (>= 50% bar overlap counts as the same passage).
3. For each group with attempt_count >= 2, emit { bar_range, attempt_count, first_attempt_ms, last_attempt_ms }.
4. Sort output by attempt_count descending.

## Concrete example
Input: session with three plays of bars 12-16 at t=0, t=30000, t=75000, plus one play of bars 20-24 at t=90000.
Output: [{ bar_range: [12,16], attempt_count: 3, first_attempt_ms: 0, last_attempt_ms: 75000 }].

## Post-conditions
Output entries have attempt_count >= 2; bar_range is a contiguous range; first_attempt_ms <= last_attempt_ms.
