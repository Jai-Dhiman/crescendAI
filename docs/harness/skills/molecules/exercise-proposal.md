---
name: exercise-proposal
tier: molecule
description: |
  Generates one targeted ExerciseArtifact from a single DiagnosisArtifact passed in by the compound. fires when session-synthesis selects a top diagnosis to address. fires when live-practice-companion wants to interrupt with a drill. fires when weekly-review converts a recurring pattern into next-week practice. fires when piece-onboarding scaffolds first-week drills from initial diagnosis. fires when student-memory longitudinal trend triggers a remedial drill. does NOT fire without a diagnosis input (Option B contract). does NOT propose multiple exercises in one call (caller invokes N times). does NOT call other molecules.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'AMT midi_notes for the diagnosis bar_range; student baseline for the diagnosed dimension'
  artifacts: [DiagnosisArtifact]
writes: ExerciseArtifact
depends_on: [extract-bar-range-signals, fetch-similar-past-observation, fetch-student-baseline]
---

## When-to-fire
Caller (a compound) passes exactly one DiagnosisArtifact whose finding_type='issue' and severity is in {moderate, significant}. Skip 'minor' and 'strength' and 'neutral' diagnoses unless the compound explicitly opts in.

## When-NOT-to-fire
Do not fire on finding_type='strength' or 'neutral' (no remedy needed). Do not fire when bar_range is null (need a concrete drill target). Do not propose for diagnoses already addressed by an exercise within the past 3 days (call fetch-similar-past-observation to check).

## Procedure
1. Read the input DiagnosisArtifact (passed in by the compound).
2. Call extract-bar-range-signals(session_id, diagnosis.bar_range) -> signals.
3. Call fetch-similar-past-observation(student_id, diagnosis.primary_dimension, piece_id, diagnosis.bar_range) -> may return prior; if a prior exercise within 3 days targeted the same diagnosis, return null (caller handles).
4. Pick exercise_type from a fixed mapping by primary_dimension and severity:
   - pedaling + moderate -> 'pedal_isolation'
   - pedaling + significant -> 'pedal_isolation' with subtype 'no-pedal-pass'
   - timing + any -> 'segment_loop' with subtype 'metronome-locked'
   - dynamics + moderate -> 'dynamic_exaggeration'
   - dynamics + significant -> 'dynamic_exaggeration' with subtype 'extreme-contrast'
   - articulation + any -> 'isolated_hands' with subtype matching score articulation
   - phrasing + any -> 'slow_practice' with subtype 'shape-vocally'
   - interpretation + any -> 'slow_practice' with subtype 'imitate-reference'
5. Estimate minutes by severity: minor=2, moderate=5, significant=8.
6. Compose action_binding ONLY if exercise_type in {segment_loop, isolated_hands, pedal_isolation}: { tool: <future tool name>, args: { bar_range, ... } }.
7. Compose ExerciseArtifact: diagnosis_ref=diagnosis.id, diagnosis_summary=diagnosis.one_sentence_finding (frozen denorm), target_dimension=diagnosis.primary_dimension, exercise_type, exercise_subtype, bar_range=diagnosis.bar_range, instruction (<= 400 chars, imperative addressed to student), success_criterion (<= 200 chars), estimated_minutes, action_binding.

## Concrete example
Input DiagnosisArtifact: { primary_dimension:'pedaling', severity:'significant', bar_range:[12,16], one_sentence_finding:'Over-pedaled through bars 12-16; harmonies blurring.' ... }.
Output ExerciseArtifact: { diagnosis_ref:'diag:abc789', diagnosis_summary:'Over-pedaled through bars 12-16; harmonies blurring.', target_dimension:'pedaling', exercise_type:'pedal_isolation', exercise_subtype:'no-pedal-pass', bar_range:[12,16], instruction:'Play bars 12-16 three times with no sustain pedal at all. Listen for whether the line still sustains itself in your fingers.', success_criterion:'Three consecutive clean repetitions with no pedal where harmonies remain audibly distinct.', estimated_minutes:8, action_binding:{ tool:'mute_pedal', args:{ bars:[12,16] } } }.

## Post-conditions
Output validates as ExerciseArtifact (Zod schema). target_dimension equals input diagnosis.primary_dimension. action_binding is non-null when exercise_type is in {segment_loop, isolated_hands, pedal_isolation}. diagnosis_summary matches input diagnosis.one_sentence_finding character-for-character.
