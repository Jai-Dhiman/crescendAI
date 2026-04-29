import { test, expect } from 'vitest'
import { phrasingArcAnalysis } from './phrasing-arc-analysis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('phrasingArcAnalysis: early velocity peak + no drift convergence with z=-1.75 returns issue/moderate', async () => {
  const midi_notes = [
    { pitch: 65, onset_ms: 0,    duration_ms: 400, velocity: 90, bar: 1 },
    { pitch: 65, onset_ms: 1000, duration_ms: 400, velocity: 70, bar: 2 },
    { pitch: 65, onset_ms: 2000, duration_ms: 400, velocity: 65, bar: 3 },
    { pitch: 65, onset_ms: 3000, duration_ms: 400, velocity: 60, bar: 4 },
    { pitch: 65, onset_ms: 4000, duration_ms: 400, velocity: 55, bar: 5 },
    { pitch: 65, onset_ms: 5000, duration_ms: 400, velocity: 50, bar: 6 },
    { pitch: 65, onset_ms: 6000, duration_ms: 400, velocity: 45, bar: 7 },
    { pitch: 65, onset_ms: 7000, duration_ms: 400, velocity: 40, bar: 8 },
  ]
  const alignment = [
    { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 1 },
    { perf_index: 1, score_index: 1, expected_onset_ms: 500,  bar: 2 },
    { perf_index: 2, score_index: 2, expected_onset_ms: 1000, bar: 3 },
    { perf_index: 3, score_index: 3, expected_onset_ms: 1500, bar: 4 },
    { perf_index: 4, score_index: 4, expected_onset_ms: 2000, bar: 5 },
    { perf_index: 5, score_index: 5, expected_onset_ms: 2500, bar: 6 },
    { perf_index: 6, score_index: 6, expected_onset_ms: 3000, bar: 7 },
    { perf_index: 7, score_index: 7, expected_onset_ms: 3500, bar: 8 },
  ]
  const input = {
    bar_range: [1, 8] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c10', 'cache:amt:s1:c10'],
    muq_scores: [0.54, 0.48, 0.46, 0.54, 0.38, 0.51],
    midi_notes,
    alignment,
    cohort_table_phrasing: [{ p: 50, value: 0.52 }, { p: 84, value: 0.60 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await phrasingArcAnalysis.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('phrasing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('moderate')
  expect(result.dimensions).toContain('dynamics')
})

test('phrasingArcAnalysis: z above neutral threshold returns neutral', async () => {
  const input = {
    bar_range: [1, 4] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c10'],
    muq_scores: [0.54, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [{ pitch: 65, onset_ms: 0, duration_ms: 400, velocity: 70, bar: 1 }],
    alignment: [{ perf_index: 0, score_index: 0, expected_onset_ms: 0, bar: 1 }],
    cohort_table_phrasing: [{ p: 50, value: 0.52 }, { p: 84, value: 0.60 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await phrasingArcAnalysis.invoke(input) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})
