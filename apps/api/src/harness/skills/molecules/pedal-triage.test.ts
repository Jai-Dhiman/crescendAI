import { test, expect } from 'vitest'
import { pedalTriage } from './pedal-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('pedalTriage: pedal held throughout (ratio>0.85) with z=-3.0 returns issue/significant/over_pedal', async () => {
  // 4 notes spanning 0-4000ms; pedal on at time_ms=0 value=127 → ratio=1.0 > 0.85
  // session_means_pedaling=[0.50,0.60,0.70] → mean=0.60 stddev≈0.082 → z=(0.35-0.60)/0.082≈-3.05
  const input = {
    bar_range: [12, 16] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c5', 'cache:amt-pedal:s1:c5'],
    muq_scores: [0.54, 0.48, 0.35, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 1000, velocity: 70, bar: 12 },
      { pitch: 62, onset_ms: 1000, duration_ms: 1000, velocity: 70, bar: 13 },
      { pitch: 64, onset_ms: 2000, duration_ms: 1000, velocity: 70, bar: 14 },
      { pitch: 65, onset_ms: 3000, duration_ms: 1000, velocity: 70, bar: 15 },
    ],
    pedal_cc: [{ time_ms: 0, value: 127 }],
    alignment: [
      { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 12 },
      { perf_index: 1, score_index: 1, expected_onset_ms: 1000, bar: 13 },
      { perf_index: 2, score_index: 2, expected_onset_ms: 2000, bar: 14 },
      { perf_index: 3, score_index: 3, expected_onset_ms: 3000, bar: 15 },
    ],
    harmony_changes: [],
    session_means_pedaling: [0.50, 0.60, 0.70],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await pedalTriage.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})
