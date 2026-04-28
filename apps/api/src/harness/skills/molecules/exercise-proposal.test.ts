import { test, expect } from 'vitest'
import { exerciseProposal } from './exercise-proposal'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { ExerciseArtifact } from '../../artifacts/exercise'

const pedaling_diagnosis = DiagnosisArtifactSchema.parse({
  primary_dimension: 'pedaling',
  dimensions: ['pedaling'],
  severity: 'significant',
  scope: 'session',
  bar_range: [12, 16],
  evidence_refs: ['cache:muq:s1:c5'],
  one_sentence_finding: 'Over-pedaled through bars 12-16; harmonies blurring.',
  confidence: 'high',
  finding_type: 'issue',
})

test('exerciseProposal: pedaling+significant with no prior exercise returns pedal_isolation/no-pedal-pass', async () => {
  const input = {
    diagnosis: pedaling_diagnosis,
    diagnosis_ref: 'diag:abc789',
    midi_notes: [
      { pitch: 60, onset_ms: 0, duration_ms: 500, velocity: 70, bar: 12 },
    ],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await exerciseProposal.invoke(input) as ExerciseArtifact
  expect(result.exercise_type).toBe('pedal_isolation')
  expect(result.exercise_subtype).toBe('no-pedal-pass')
  expect(result.target_dimension).toBe('pedaling')
  expect(result.bar_range).toEqual([12, 16])
  expect(result.action_binding).not.toBeNull()
  expect(result.estimated_minutes).toBe(8)
  expect(result.diagnosis_summary).toBe('Over-pedaled through bars 12-16; harmonies blurring.')
})
