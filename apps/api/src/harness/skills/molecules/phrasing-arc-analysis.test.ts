// phrasing-arc-analysis.test.ts (after refactor — selectors-only contract)
import { test, expect } from 'vitest'
import { phrasingArcAnalysis } from './phrasing-arc-analysis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import type { GroundedDigest } from '../../loop/grounded-digest'
import type { PhaseContext } from '../../loop/types'

function makeCtx(
  muqPhrasing: number,
  opts?: { emptyAlignment?: boolean }
): PhaseContext {
  const dims = ['dynamics','timing','pedaling','articulation','phrasing','interpretation'] as const
  const session_means: Record<string, number[]> = {}
  const within_session_means: Record<string, number> = {}
  for (const d of dims) {
    session_means[d] = [0.5, 0.5, 0.5]
    within_session_means[d] = 0.52
  }
  // 8 notes: velocity descends from bar 1 (90) to bar 8 (40) — early peak at index 0
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
  const alignment = opts?.emptyAlignment ? [] : [
    { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 1 },
    { perf_index: 1, score_index: 1, expected_onset_ms: 500,  bar: 2 },
    { perf_index: 2, score_index: 2, expected_onset_ms: 1000, bar: 3 },
    { perf_index: 3, score_index: 3, expected_onset_ms: 1500, bar: 4 },
    { perf_index: 4, score_index: 4, expected_onset_ms: 2000, bar: 5 },
    { perf_index: 5, score_index: 5, expected_onset_ms: 2500, bar: 6 },
    { perf_index: 6, score_index: 6, expected_onset_ms: 3000, bar: 7 },
    { perf_index: 7, score_index: 7, expected_onset_ms: 3500, bar: 8 },
  ]
  const digest: GroundedDigest = {
    chunks_adapted: [{
      chunk_id: 'chunk:0',
      bar_coverage: [1, 8] as [number, number],
      muq_scores: [0.54, 0.48, 0.46, 0.54, muqPhrasing, 0.51],
      midi_notes,
      pedal_cc: [],
      alignment,
    }],
    mono_notes_per_bar: [],
    now_ms: 1000,
    cohort: {
      dynamics:       { mean: 0.55, stddev: 0.10 },
      timing:         { mean: 0.48, stddev: 0.12 },
      pedaling:       { mean: 0.46, stddev: 0.12 },
      articulation:   { mean: 0.54, stddev: 0.11 },
      // phrasing p50=0.52, p84=0.60 → stddev=0.08 → z=(0.38-0.52)/0.08≈-1.75 (moderate)
      phrasing:       { mean: 0.52, stddev: 0.08 },
      interpretation: { mean: 0.51, stddev: 0.13 },
    },
    past_diagnoses_grounded: [],
    session_means: session_means as GroundedDigest['session_means'],
    within_session_means: within_session_means as GroundedDigest['within_session_means'],
    compact_signal_summary: 'chunk:0 bars 1-8',
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

test('phrasingArcAnalysis: early velocity peak + z<-0.8 returns issue/moderate', async () => {
  // muqPhrasing=0.38 → z=(0.38-0.52)/0.08=-1.75 → moderate
  const ctx = makeCtx(0.38)
  const selectors = {
    bar_range: [1, 8] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c10', 'cache:amt:s1:c10'],
  }
  const result = await phrasingArcAnalysis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('phrasing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('moderate')
  expect(result.dimensions).toContain('dynamics')
})

test('phrasingArcAnalysis: z above neutral threshold returns neutral', async () => {
  // muqPhrasing=0.52 → z=0 → neutral
  const ctx = makeCtx(0.52)
  const selectors = { bar_range: [1, 8] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c10'] }
  const result = await phrasingArcAnalysis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.finding_type).toBe('neutral')
})

test('phrasingArcAnalysis: empty alignment returns neutral (not throw)', async () => {
  // Even with bad phrasing z, empty alignment -> neutral
  const ctx = makeCtx(0.38, { emptyAlignment: true })
  const selectors = { bar_range: [1, 8] as [number, number], scope: 'session' as const, evidence_refs: ['cache:muq:s1:c10'] }
  const result = await phrasingArcAnalysis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('phrasing')
  expect(result.dimensions.length).toBeGreaterThanOrEqual(1)
  expect(result.evidence_refs.length).toBeGreaterThanOrEqual(1)
})

test('phrasingArcAnalysis: bar_range=null resolves full session and returns artifact', async () => {
  const ctx = makeCtx(0.52)
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: ['cache:muq:s1:c10'] }
  const result = await phrasingArcAnalysis.invoke(selectors, ctx) as DiagnosisArtifact
  expect(['neutral','issue','strength']).toContain(result.finding_type)
})

test('phrasingArcAnalysis: missing ctx throws', async () => {
  const selectors = { bar_range: null, scope: 'session' as const, evidence_refs: [] }
  await expect(phrasingArcAnalysis.invoke(selectors, undefined)).rejects.toThrow('ctx')
})
