import { test, expect } from 'vitest'
import { rubatoCoaching } from './rubato-coaching'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('rubatoCoaching: constant score IOIs vs stretching perf IOIs with large drift returns issue/significant/dragged', async () => {
  const midi_notes = [
    { pitch: 60, onset_ms: 0,    duration_ms: 400, velocity: 70, bar: 40 },
    { pitch: 62, onset_ms: 700,  duration_ms: 400, velocity: 70, bar: 40 },
    { pitch: 64, onset_ms: 1600, duration_ms: 400, velocity: 70, bar: 41 },
    { pitch: 65, onset_ms: 2700, duration_ms: 400, velocity: 70, bar: 42 },
    { pitch: 67, onset_ms: 4000, duration_ms: 400, velocity: 70, bar: 43 },
    { pitch: 69, onset_ms: 5500, duration_ms: 400, velocity: 70, bar: 44 },
    { pitch: 71, onset_ms: 7200, duration_ms: 400, velocity: 70, bar: 45 },
    { pitch: 72, onset_ms: 9100, duration_ms: 400, velocity: 70, bar: 46 },
  ]
  const alignment = [
    { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 40 },
    { perf_index: 1, score_index: 1, expected_onset_ms: 500,  bar: 40 },
    { perf_index: 2, score_index: 2, expected_onset_ms: 1000, bar: 41 },
    { perf_index: 3, score_index: 3, expected_onset_ms: 1500, bar: 42 },
    { perf_index: 4, score_index: 4, expected_onset_ms: 2000, bar: 43 },
    { perf_index: 5, score_index: 5, expected_onset_ms: 2500, bar: 44 },
    { perf_index: 6, score_index: 6, expected_onset_ms: 3000, bar: 45 },
    { perf_index: 7, score_index: 7, expected_onset_ms: 3500, bar: 46 },
  ]
  const input = {
    bar_range: [40, 48] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c14', 'cache:amt:s1:c14'],
    muq_scores: [0.54, 0.38, 0.46, 0.54, 0.52, 0.51],
    midi_notes,
    alignment,
    session_means_timing: [0.48, 0.54, 0.60],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await rubatoCoaching.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('timing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.dimensions).toContain('phrasing')
  expect(result.dimensions).toContain('interpretation')
})
