import { test, expect } from 'vitest'
import { tempoStabilityTriage } from './tempo-stability-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(sessionMeansN: number, muqTiming: number, opts?: { monotonicDrift?: boolean }): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = Array.from({ length: sessionMeansN }, () => 0.5)
    within_session_means[d] = 0.52
  }
  // 14 notes with monotonic positive drift (each note progressively later than expected)
  const N = 14
  const midi_notes = Array.from({ length: N }, (_, k) => ({
    pitch: 60 + (k % 5), onset_ms: k * 500 + (opts?.monotonicDrift ? k * 40 : 0), duration_ms: 500, velocity: 70, bar: 1 + Math.floor(k / 4),
  }))
  const alignment = Array.from({ length: N }, (_, k) => ({
    perf_index: k, score_index: k, expected_onset_ms: k * 500, bar: 1 + Math.floor(k / 4),
  }))
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [1, 16] as [number, number],
      muq_scores: [0.54, muqTiming, 0.46, 0.54, 0.52, 0.51],
      midi_notes,
      pedal_cc: [],
      alignment,
    }],
    mono_notes_per_bar: [],
    now_ms: 1000,
    cohort: {
      dynamics:{mean:0.55,stddev:0.10}, timing:{mean:0.48,stddev:0.12}, pedaling:{mean:0.46,stddev:0.12},
      articulation:{mean:0.54,stddev:0.11}, phrasing:{mean:0.52,stddev:0.11}, interpretation:{mean:0.51,stddev:0.13},
    },
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary: 'chunk:0 bars 1-16',
    piece_id: 'test-piece',
  }
  return {
    env: {} as unknown as import('../../../lib/types').Bindings,
    studentId: 'stu-1', sessionId: 'sess-1', conversationId: null,
    digest: digest as unknown as Record<string, unknown>,
    waitUntil: () => {}, pieceId: 'test-piece', trigger: 'synthesis', turnCap: 10,
  }
}

test('tempoStabilityTriage: monotonic positive drift returns issue/significant/slowing', async () => {
  const ctx = makeCtx(3, 0.30, { monotonicDrift: true })
  ;(ctx.digest as unknown as GroundedDigest).session_means.timing = [0.50, 0.60, 0.70]
  const selectors = { bar_range: [1, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await tempoStabilityTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('timing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
})

test('tempoStabilityTriage: z above neutral threshold returns neutral', async () => {
  const ctx = makeCtx(3, 0.48)
  ;(ctx.digest as unknown as GroundedDigest).session_means.timing = [0.48, 0.48, 0.48]
  const selectors = { bar_range: [1, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await tempoStabilityTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('tempoStabilityTriage: empty/sparse alignment returns neutral (not throw)', async () => {
  const ctx = makeCtx(3, 0.48)
  ;(ctx.digest as unknown as GroundedDigest).chunks_adapted[0].alignment = []
  const selectors = { bar_range: [1, 4] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await tempoStabilityTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
  expect(result.primary_dimension).toBe('timing')
})

test('tempoStabilityTriage: insufficient history uses within-session baseline and returns artifact (not throw)', async () => {
  const ctx = makeCtx(0, 0.30, { monotonicDrift: true })
  ;(ctx.digest as unknown as GroundedDigest).session_means.timing = [0.48]
  const selectors = { bar_range: [1, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await tempoStabilityTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('timing')
})

test('tempoStabilityTriage: bar_range=null returns artifact', async () => {
  const ctx = makeCtx(3, 0.48)
  ;(ctx.digest as unknown as GroundedDigest).session_means.timing = [0.48, 0.48, 0.48]
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: ['cache:muq:s1:c0'] }
  const result = await tempoStabilityTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(['neutral','issue','strength']).toContain(result.finding_type)
})

test('tempoStabilityTriage: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(tempoStabilityTriage.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
