// voicing-diagnosis.test.ts (after refactor — selectors-only contract)
import { test, expect } from 'vitest'
import { voicingDiagnosis } from './voicing-diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(sessionMeansN: number, muqDynamics: number, midiNotes?: GroundedDigest['chunks_adapted'][0]['midi_notes']): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = Array.from({ length: sessionMeansN }, () => 0.5)
    within_session_means[d] = 0.52
  }
  // Default: flat voicing (top and bass nearly equal velocity)
  const defaultNotes = [
    { pitch: 72, onset_ms: 0,    duration_ms: 500, velocity: 76, bar: 1 },
    { pitch: 70, onset_ms: 250,  duration_ms: 500, velocity: 76, bar: 1 },
    { pitch: 45, onset_ms: 0,    duration_ms: 500, velocity: 74, bar: 1 },
    { pitch: 40, onset_ms: 250,  duration_ms: 500, velocity: 74, bar: 1 },
    { pitch: 71, onset_ms: 1000, duration_ms: 500, velocity: 76, bar: 2 },
    { pitch: 69, onset_ms: 1250, duration_ms: 500, velocity: 76, bar: 2 },
    { pitch: 44, onset_ms: 1000, duration_ms: 500, velocity: 74, bar: 2 },
    { pitch: 41, onset_ms: 1250, duration_ms: 500, velocity: 74, bar: 2 },
  ]
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [1, 2] as [number, number],
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
    compact_signal_summary: 'chunk:0 bars 1-2',
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

test('voicingDiagnosis: flat top/bass voicing with z<-1 returns issue/significant', async () => {
  // session_means=[0.50,0.60,0.70] → baseline mean≈0.60 → z=(0.42-0.60)/stddev ≈ -2.2
  const ctx = makeCtx(3, 0.42)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.50, 0.60, 0.70]
  const selectors = {
    bar_range: [1, 2] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1', 'cache:amt:s1:c1'],
  }
  const result = await voicingDiagnosis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.dimensions).toContain('phrasing')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})

test('voicingDiagnosis: z above neutral threshold returns neutral', async () => {
  const ctx = makeCtx(3, 0.54)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.54, 0.54, 0.54]
  const selectors = { bar_range: [1, 2] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c1'] }
  const result = await voicingDiagnosis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('voicingDiagnosis: insufficient history uses within-session baseline and returns artifact (not throw)', async () => {
  const ctx = makeCtx(0, 0.42)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = []
  ;(ctx.digest as unknown as GroundedDigest).within_session_means.dynamics = 0.60
  const selectors = { bar_range: [1, 2] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c1'] }
  const result = await voicingDiagnosis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
})

test('voicingDiagnosis: bar_range=null resolves full session and returns artifact', async () => {
  const ctx = makeCtx(3, 0.54)
  ;(ctx.digest as unknown as GroundedDigest).session_means.dynamics = [0.54, 0.54, 0.54]
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: ['cache:muq:s1:c1'] }
  const result = await voicingDiagnosis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(['neutral','issue','strength']).toContain(result.finding_type)
})

test('voicingDiagnosis: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(voicingDiagnosis.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
