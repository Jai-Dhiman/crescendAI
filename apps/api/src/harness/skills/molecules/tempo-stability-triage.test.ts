import { test, expect } from 'vitest'
import { tempoStabilityTriage } from './tempo-stability-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('tempoStabilityTriage: monotonic positive drift 14 notes, r=0, z=-3.3 returns issue/significant/slowing', async () => {
  const notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[] = []
  const alignment: { perf_index: number; score_index: number; expected_onset_ms: number; bar: number }[] = []
  for (let idx = 0; idx < 14; idx++) {
    const expected = idx * 500
    const drift = idx * 120
    notes.push({ pitch: 60 + idx, onset_ms: expected + drift, duration_ms: 400, velocity: 70, bar: Math.floor(idx / 2) + 1 })
    alignment.push({ perf_index: idx, score_index: idx, expected_onset_ms: expected, bar: Math.floor(idx / 2) + 1 })
  }
  const input = {
    bar_range: [1, 16] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1', 'cache:amt:s1:c1'],
    muq_scores: [0.54, 0.38, 0.46, 0.54, 0.52, 0.51],
    midi_notes: notes,
    alignment,
    session_means_timing: [0.48, 0.54, 0.60],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await tempoStabilityTriage.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('timing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
})
