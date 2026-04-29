import { test, expect } from 'vitest'
import { voicingDiagnosis } from './voicing-diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('voicingDiagnosis: flat top/bass voicing with z=-2.2 returns issue/significant', async () => {
  // bar 1 and bar 2: top pitch (72,70) vel=76, bass pitch (45,40) vel=74 → |76-74|=2 < 5 for 2/2 bars
  // session_means=[0.50,0.60,0.70] → baseline mean=0.60 stddev≈0.082 → z=(0.42-0.60)/0.082≈-2.2
  const input = {
    bar_range: [1, 2] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1', 'cache:amt:s1:c1'],
    muq_scores: [0.42, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 72, onset_ms: 0,    duration_ms: 500, velocity: 76, bar: 1 },
      { pitch: 70, onset_ms: 250,  duration_ms: 500, velocity: 76, bar: 1 },
      { pitch: 45, onset_ms: 0,    duration_ms: 500, velocity: 74, bar: 1 },
      { pitch: 40, onset_ms: 250,  duration_ms: 500, velocity: 74, bar: 1 },
      { pitch: 71, onset_ms: 1000, duration_ms: 500, velocity: 76, bar: 2 },
      { pitch: 69, onset_ms: 1250, duration_ms: 500, velocity: 76, bar: 2 },
      { pitch: 44, onset_ms: 1000, duration_ms: 500, velocity: 74, bar: 2 },
      { pitch: 41, onset_ms: 1250, duration_ms: 500, velocity: 74, bar: 2 },
    ],
    session_means_dynamics: [0.50, 0.60, 0.70],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await voicingDiagnosis.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.dimensions).toContain('phrasing')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})

test('voicingDiagnosis: z above neutral threshold returns neutral', async () => {
  const input = {
    bar_range: [1, 2] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1'],
    muq_scores: [0.54, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [{ pitch: 72, onset_ms: 0, duration_ms: 500, velocity: 90, bar: 1 }],
    session_means_dynamics: [0.54, 0.54, 0.54],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await voicingDiagnosis.invoke(input) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('voicingDiagnosis: insufficient session history throws', async () => {
  const input = {
    bar_range: [1, 2] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1'],
    muq_scores: [0.42, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [{ pitch: 72, onset_ms: 0, duration_ms: 500, velocity: 76, bar: 1 }],
    session_means_dynamics: [0.54, 0.60],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  await expect(voicingDiagnosis.invoke(input)).rejects.toThrow('insufficient session history')
})
