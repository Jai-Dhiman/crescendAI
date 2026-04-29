import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import type { Bindings } from '../lib/types'
import type { ServiceContext } from '../lib/types'
import { synthesizeV6 } from './teacher'
import type { SynthesisArtifact } from '../harness/artifacts/synthesis'

const MOCK_BINDINGS = {
  AI_GATEWAY_TEACHER: 'https://gw.example',
  ANTHROPIC_API_KEY: 'test-key',
} as unknown as Bindings

const VALID_ARTIFACT: SynthesisArtifact = {
  session_id: 'sess-1',
  synthesis_scope: 'session',
  strengths: [],
  focus_areas: [{ dimension: 'pedaling', one_liner: 'Work on pedal timing.', severity: 'moderate' }],
  proposed_exercises: [],
  dominant_dimension: 'pedaling',
  recurring_pattern: null,
  next_session_focus: null,
  diagnosis_refs: [],
  headline: 'Today you played with intention across the full piece. The phrasing held up well in the opening section, and the dynamics showed real contrast. Pedaling is the area to focus on next — a few spots had overlapping notes that muddied the texture. Keep your foot ready to clear between phrases. We will zero in on bars three through five next time.',
}

const ENRICHED_CHUNK = {
  chunkIndex: 0,
  muq_scores: [0.6, 0.5, 0.7, 0.55, 0.6, 0.65],
  midi_notes: [{ pitch: 60, onset_ms: 1000, duration_ms: 500, velocity: 80 }],
  pedal_cc: [{ time_ms: 1000, value: 100 }],
  alignment: [{ perf_index: 0, score_index: 0, expected_onset_ms: 985, bar: 3 }],
  bar_coverage: [3, 4] as [number, number],
}

describe('synthesizeV6 digest shape', () => {
  const fetchSpy = vi.fn()

  beforeEach(() => {
    fetchSpy.mockReset()
    vi.stubGlobal('fetch', fetchSpy)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('passes chunks, baselines, cohort_tables, session_history, past_diagnoses into digest', async () => {
    let capturedBody: string | null = null

    // Call 1: Phase 1 — capture body, return end_turn immediately
    fetchSpy.mockImplementationOnce(async (_url: string, opts: RequestInit) => {
      if (capturedBody === null) capturedBody = opts.body as string
      return new Response(
        JSON.stringify({ content: [{ type: 'text', text: 'no tools' }], stop_reason: 'end_turn' }),
        { status: 200 },
      )
    })
    // Call 2: Phase 2 — write SynthesisArtifact
    fetchSpy.mockImplementationOnce(async () =>
      new Response(
        JSON.stringify({
          content: [{ type: 'tool_use', id: 'tu_1', name: 'write_synthesis_artifact', input: VALID_ARTIFACT }],
          stop_reason: 'tool_use',
        }),
        { status: 200 },
      )
    )

    const ctx: ServiceContext = { db: {} as never, env: MOCK_BINDINGS }
    const events = []
    for await (const ev of synthesizeV6(ctx, {
      studentId: 'stu-1',
      conversationId: null,
      sessionDurationMs: 60000,
      practicePattern: '{}',
      topMoments: [],
      drillingRecords: [],
      pieceMetadata: null,
      enrichedChunks: [ENRICHED_CHUNK],
      baselines: { dynamics: 0.5, timing: 0.48, pedaling: 0.46, articulation: 0.54, phrasing: 0.52, interpretation: 0.51 },
      sessionHistory: [],
      pastDiagnoses: [],
    }, 'sess-1')) {
      events.push(ev)
    }

    expect(capturedBody).not.toBeNull()
    const parsed = JSON.parse(capturedBody!)
    const userMsg = parsed.messages?.find((m: { role: string }) => m.role === 'user')
    expect(userMsg).toBeDefined()
    const digestText = typeof userMsg?.content === 'string' ? userMsg.content : JSON.stringify(userMsg?.content)
    expect(digestText).toContain('chunks')
    expect(digestText).toContain('baselines')
    expect(digestText).toContain('cohort_tables')
  })

  it('passes null baselines through to digest when baselines is null', async () => {
    let capturedBody: string | null = null

    // Call 1: Phase 1 — capture body, return end_turn
    fetchSpy.mockImplementationOnce(async (_url: string, opts: RequestInit) => {
      if (capturedBody === null) capturedBody = opts.body as string
      return new Response(JSON.stringify({ content: [{ type: 'text', text: 'x' }], stop_reason: 'end_turn' }), { status: 200 })
    })
    // Call 2: Phase 2 — write SynthesisArtifact
    fetchSpy.mockImplementationOnce(async () =>
      new Response(
        JSON.stringify({ content: [{ type: 'tool_use', id: 'tu_2', name: 'write_synthesis_artifact', input: VALID_ARTIFACT }], stop_reason: 'tool_use' }),
        { status: 200 },
      )
    )

    const ctx: ServiceContext = { db: {} as never, env: MOCK_BINDINGS }
    for await (const _ of synthesizeV6(ctx, {
      studentId: 'stu-1', conversationId: null, sessionDurationMs: 0, practicePattern: '{}',
      topMoments: [], drillingRecords: [], pieceMetadata: null,
      enrichedChunks: [], baselines: null, sessionHistory: [], pastDiagnoses: [],
    }, 'sess-1')) { /* drain */ }

    expect(capturedBody).not.toBeNull()
    // Parse the request body and extract the digest text from the user message content.
    const parsedBody = JSON.parse(capturedBody!)
    const userContent = parsedBody.messages?.find((m: { role: string }) => m.role === 'user')?.content as string
    expect(userContent).toBeDefined()
    expect(userContent).toMatch(/"baselines":\s*null/)
  })
})
