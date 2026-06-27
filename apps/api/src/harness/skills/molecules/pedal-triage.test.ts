// pedal-triage.test.ts (after refactor)
import { test, expect } from 'vitest'
import { pedalTriage } from './pedal-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(sessionMeansN: number, muqPedaling: number): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = Array.from({ length: sessionMeansN }, () => 0.5)
    within_session_means[d] = 0.52
  }
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [12, 16] as [number, number],
      muq_scores: [0.54, 0.48, muqPedaling, 0.54, 0.52, 0.51],
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
    compact_signal_summary: 'chunk:0 bars 12-16',
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

test('pedalTriage: over-pedal (ratio>0.85, z<-1) with sufficient history returns issue/significant', async () => {
  const selectors = {
    bar_range: [12, 16] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c5', 'cache:amt-pedal:s1:c5'],
  }
  const ctx = makeCtx(3, 0.35)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.50, 0.60, 0.70]
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})

test('pedalTriage: z above neutral threshold (-1.0) returns neutral', async () => {
  const selectors = { bar_range: [12, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c5'] }
  const ctx = makeCtx(3, 0.46)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.46, 0.46, 0.46]
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('pedalTriage: insufficient session history (n<3) uses within-session baseline and returns artifact (not throw)', async () => {
  const selectors = { bar_range: [12, 16] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c5'] }
  const ctx = makeCtx(0, 0.35)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.50]
  ;(ctx.digest as unknown as GroundedDigest).within_session_means.pedaling = 0.60
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
})

test('pedalTriage: bar_range=null resolves full session bundle and returns artifact', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: ['cache:muq:s1:c5'] }
  const ctx = makeCtx(3, 0.46)
  ;(ctx.digest as unknown as GroundedDigest).session_means.pedaling = [0.46, 0.46, 0.46]
  const result = await pedalTriage.invoke(selectors, ctx) as DiagnosisArtifact
  expect(['neutral','issue','strength']).toContain(result.finding_type)
})

test('pedalTriage: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(pedalTriage.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
