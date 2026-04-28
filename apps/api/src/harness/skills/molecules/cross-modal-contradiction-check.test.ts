import { test, expect } from 'vitest'
import { crossModalContradictionCheck } from './cross-modal-contradiction-check'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('crossModalContradictionCheck: MuQ pedaling high (z=+3.3) but overlap ratio=1.0 returns issue/significant/pedaling', async () => {
  const input = {
    bar_range: [20, 28] as [number, number],
    scope: 'stop_moment' as const,
    evidence_refs: ['cache:muq:s1:c7', 'cache:amt-pedal:s1:c7'],
    muq_scores: [0.54, 0.48, 0.72, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 1000, velocity: 70, bar: 20 },
      { pitch: 62, onset_ms: 1000, duration_ms: 1000, velocity: 70, bar: 22 },
      { pitch: 64, onset_ms: 2000, duration_ms: 1000, velocity: 70, bar: 24 },
      { pitch: 65, onset_ms: 3000, duration_ms: 1000, velocity: 70, bar: 26 },
    ],
    pedal_cc: [{ time_ms: 0, value: 127 }],
    alignment: [
      { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 20 },
      { perf_index: 1, score_index: 1, expected_onset_ms: 1000, bar: 22 },
      { perf_index: 2, score_index: 2, expected_onset_ms: 2000, bar: 24 },
      { perf_index: 3, score_index: 3, expected_onset_ms: 3000, bar: 26 },
    ],
    mono_notes_per_bar: [],
    score_articulation_per_bar: [],
    cohort_baselines: {
      dynamics:        { mean: 0.54, stddev: 0.07 },
      timing:          { mean: 0.48, stddev: 0.04 },
      pedaling:        { mean: 0.46, stddev: 0.08 },
      articulation:    { mean: 0.54, stddev: 0.02 },
    },
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await crossModalContradictionCheck.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
})
