import { test, expect } from 'vitest'
import { dynamicRangeAudit } from './dynamic-range-audit'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('dynamicRangeAudit: compressed velocities [60-65] with wide score markings and z=-3.3 returns issue/significant', async () => {
  const input = {
    bar_range: [28, 32] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c8', 'cache:amt:s1:c8'],
    muq_scores: [0.38, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 500, velocity: 60, bar: 28 },
      { pitch: 62, onset_ms: 500,  duration_ms: 500, velocity: 62, bar: 29 },
      { pitch: 64, onset_ms: 1000, duration_ms: 500, velocity: 63, bar: 30 },
      { pitch: 65, onset_ms: 1500, duration_ms: 500, velocity: 65, bar: 31 },
      { pitch: 67, onset_ms: 2000, duration_ms: 500, velocity: 61, bar: 32 },
    ],
    score_marking_type: 'wide' as const,
    session_means_dynamics: [0.48, 0.54, 0.60],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await dynamicRangeAudit.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})

test('dynamicRangeAudit: score_marking_type=none returns neutral regardless of z', async () => {
  const input = {
    bar_range: [28, 32] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c8'],
    muq_scores: [0.38, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,   duration_ms: 500, velocity: 60, bar: 28 },
      { pitch: 62, onset_ms: 500, duration_ms: 500, velocity: 90, bar: 29 },
    ],
    score_marking_type: 'none' as const,
    session_means_dynamics: [0.48, 0.54, 0.60],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await dynamicRangeAudit.invoke(input) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('dynamicRangeAudit: insufficient session history throws', async () => {
  const input = {
    bar_range: [28, 32] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c8'],
    muq_scores: [0.38, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [{ pitch: 60, onset_ms: 0, duration_ms: 500, velocity: 60, bar: 28 }],
    score_marking_type: 'wide' as const,
    session_means_dynamics: [0.48, 0.54],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  await expect(dynamicRangeAudit.invoke(input)).rejects.toThrow('insufficient session history')
})
