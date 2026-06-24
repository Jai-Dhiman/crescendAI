// dynamic-range-audit.test.ts (after refactor — selectors-only contract, score_marking_type removed)
import { test, expect } from 'vitest'
import { dynamicRangeAudit } from './dynamic-range-audit'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(
  sessionMeansN: number,
  muqDynamics: number,
  midiNotes?: GroundedDigest['chunks_adapted'][0]['midi_notes'],
): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = Array.from({ length: sessionMeansN }, () => 0.5)
    within_session_means[d] = 0.52
  }
  // Default: compressed velocity range [60-65]
  const defaultNotes = [
    { pitch: 60, onset_ms: 0,    duration_ms: 500, velocity: 60, bar: 28 },
    { pitch: 62, onset_ms: 500,  duration_ms: 500, velocity: 62, bar: 29 },
    { pitch: 64, onset_ms: 1000, duration_ms: 500, velocity: 63, bar: 30 },
    { pitch: 65, onset_ms: 1500, duration_ms: 500, velocity: 65, bar: 31 },
    { pitch: 67, onset_ms: 2000, duration_ms: 500, velocity: 61, bar: 32 },
  ]
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [28, 32] as [number, number],
      muq_scores: [muqDynamics, 0.48, 0.46, 0.54, 0.52, 0.51],
      midi_notes: midiNotes ?? defaultNotes,
      pedal_cc: [],
      alignment: [],
    }],
    mono_notes_per_bar: [],
    now_ms: 1000,
    cohort: {
      dynamics:       { mean: 0.55, stddev: 0.10 },
      timing:         { mean: 0.48, stddev: 0.12 },
      pedaling:       { mean: 0.46, stddev: 0.12 },
      articulation:   { mean: 0.54, stddev: 0.11 },
      phrasing:       { mean: 0.52, stddev: 0.11 },
      interpretation: { mean: 0.51, stddev: 0.13 },
    },
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary: 'chunk:0 bars 28-32',
    piece_id: 'test-piece',
  }
  return {
    env: {} as unknown as import('../../../lib/types').Bindings,
    studentId: 'stu-1', sessionId: 'sess-1', conversationId: null,
    digest: digest as unknown as Record<string, unknown>,
    waitUntil: () => {},
    pieceId: 'test-piece',
    trigger: 'synthesis',
    turnCap: 10,
  }
}

test('dynamicRangeAudit: compressed velocities [60-65] with z<-0.8 returns issue/significant', async () => {
  // session_means=[0.48,0.54,0.60] → baseline mean≈0.54 → z=(0.38-0.54)/stddev ≈ -3.3
  const ctx = makeCtx(3, 0.38)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.48, 0.54, 0.60]
  const selectors = {
    bar_range: [28, 32] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c8', 'cache:amt:s1:c8'],
  }
  const result = await dynamicRangeAudit.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})

test('dynamicRangeAudit: z above neutral threshold returns neutral', async () => {
  const ctx = makeCtx(3, 0.54)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.54, 0.54, 0.54]
  const selectors = { bar_range: [28, 32] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c8'] }
  const result = await dynamicRangeAudit.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('dynamicRangeAudit: wide velocity range (>=30) with z<-0.8 returns neutral (adequate range)', async () => {
  // velocity range 40-90 = 50 ≥ 30 → neutral despite bad z
  const wideNotes = [
    { pitch: 60, onset_ms: 0,   duration_ms: 500, velocity: 40, bar: 28 },
    { pitch: 62, onset_ms: 500, duration_ms: 500, velocity: 90, bar: 29 },
  ]
  const ctx = makeCtx(3, 0.38, wideNotes)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.48, 0.54, 0.60]
  const selectors = { bar_range: [28, 32] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c8'] }
  const result = await dynamicRangeAudit.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('dynamicRangeAudit: insufficient history uses within-session baseline and returns artifact (not throw)', async () => {
  const ctx = makeCtx(0, 0.38)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = []
  ;(ctx.digest as unknown as GroundedDigest).within_session_means.dynamics = 0.60
  const selectors = { bar_range: [28, 32] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c8'] }
  const result = await dynamicRangeAudit.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
})

test('dynamicRangeAudit: bar_range=null resolves full session and returns artifact', async () => {
  const ctx = makeCtx(3, 0.54)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.54, 0.54, 0.54]
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: ['cache:muq:s1:c8'] }
  const result = await dynamicRangeAudit.invoke(selectors, ctx) as DiagnosisArtifact
  expect(['neutral','issue','strength']).toContain(result.finding_type)
})

test('dynamicRangeAudit: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(dynamicRangeAudit.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
