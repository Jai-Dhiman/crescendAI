import { test, expect, vi } from 'vitest'
import { validateCatalog } from '../validator'
import { runHook } from '../../loop/runHook'
import type { HookContext, HookEvent } from '../../loop/types'
import type { SynthesisArtifact } from '../../artifacts/synthesis'
import type { Bindings } from '../../../lib/types'

test('full catalog: validateCatalog returns no errors', async () => {
	const r = await validateCatalog('docs/harness/skills')
	expect(r.errors).toEqual([])
	expect(r.valid).toBe(true)
})

// E2E: real extract-bar-range-signals atom invoked through full harness pipeline

const MOCK_BINDINGS = {
	AI_GATEWAY_TEACHER: 'https://gw.example',
	ANTHROPIC_API_KEY: 'test-key',
} as unknown as Bindings

const FIXTURE_CHUNK = {
	chunkIndex: 0,
	muq_scores: [0.55, 0.48, 0.62, 0.51, 0.58, 0.60],
	midi_notes: [
		{ pitch: 60, onset_ms: 500, duration_ms: 400, velocity: 75, bar: 3 },
		{ pitch: 64, onset_ms: 900, duration_ms: 350, velocity: 70, bar: 3 },
	],
	pedal_cc: [
		{ time_ms: 500, value: 100 },
		{ time_ms: 800, value: 0 },
		{ time_ms: 900, value: 100 },
	],
	alignment: [
		{ perf_index: 0, score_index: 0, expected_onset_ms: 490, bar: 3 },
		{ perf_index: 1, score_index: 1, expected_onset_ms: 895, bar: 3 },
	],
	bar_coverage: [3, 4] as [number, number],
}

const FIXTURE_DIGEST = {
	sessionDurationMs: 120000,
	practicePattern: '{"mode":"practice"}',
	topMoments: [],
	drillingRecords: [],
	pieceMetadata: null,
	chunks: [FIXTURE_CHUNK],
	baselines: { dynamics: 0.54, timing: 0.48, pedaling: 0.46, articulation: 0.54, phrasing: 0.52, interpretation: 0.51 },
	cohort_tables: {
		pedaling: [{ p: 50, value: 0.46 }, { p: 75, value: 0.61 }],
	},
	session_history: [],
	past_diagnoses: [],
}

// Phase 1 LLM response: calls extract-bar-range-signals with fixture bar_range + chunks
const PHASE1_TOOL_CALL_RESPONSE = {
	content: [
		{
			type: 'tool_use',
			id: 'tu_1',
			name: 'extract-bar-range-signals',
			input: {
				bar_range: [3, 4],
				chunks: [
					{
						chunk_id: 'chunk:0',
						bar_coverage: [3, 4],
						muq_scores: FIXTURE_CHUNK.muq_scores,
						midi_notes: FIXTURE_CHUNK.midi_notes,
						pedal_cc: FIXTURE_CHUNK.pedal_cc,
						alignment: FIXTURE_CHUNK.alignment,
					},
				],
			},
		},
	],
	stop_reason: 'tool_use',
}

// Phase 1 LLM end_turn after tool result
const PHASE1_END_TURN = {
	content: [{ type: 'text', text: 'Diagnosis complete.' }],
	stop_reason: 'end_turn',
}

const VALID_SYNTHESIS_ARTIFACT: SynthesisArtifact = {
	session_id: 'sess-e2e',
	synthesis_scope: 'session',
	strengths: [],
	focus_areas: [{ dimension: 'pedaling', one_liner: 'Pedal releases were slightly late at bars 3-4.', severity: 'minor' }],
	proposed_exercises: [],
	dominant_dimension: 'pedaling',
	recurring_pattern: null,
	next_session_focus: null,
	diagnosis_refs: ['tu_1'],
	headline: "Today's session showed solid commitment to the phrase shapes throughout the piece. The dynamics held steady and your timing in the opening section was notably clean. The one area to refine is the pedaling at bars three and four, where the sustain was held a beat longer than the harmony called for. This muddied the transition slightly. Next time, try lifting the pedal on beat three and listen for how the texture clears. Small adjustment, big difference in the overall color.",
}

// Phase 2: write_synthesis_artifact
const PHASE2_ARTIFACT_RESPONSE = {
	content: [
		{
			type: 'tool_use',
			id: 'tu_2',
			name: 'write_synthesis_artifact',
			input: VALID_SYNTHESIS_ARTIFACT,
		},
	],
	stop_reason: 'tool_use',
}

test('E2E: extract-bar-range-signals atom invoked via real runHook pipeline', async () => {
	const fetchSpy = vi.fn()
	vi.stubGlobal('fetch', fetchSpy)

	try {
		// Call 1: Phase 1 turn 1 — LLM calls extract-bar-range-signals
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(PHASE1_TOOL_CALL_RESPONSE), { status: 200 }),
		)
		// Call 2: Phase 1 turn 2 — LLM ends turn after seeing tool result
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(PHASE1_END_TURN), { status: 200 }),
		)
		// Call 3: Phase 2 — writes SynthesisArtifact
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(PHASE2_ARTIFACT_RESPONSE), { status: 200 }),
		)

		const ctx: HookContext = {
			env: MOCK_BINDINGS,
			studentId: 'stu-e2e',
			sessionId: 'sess-e2e',
			conversationId: null,
			digest: FIXTURE_DIGEST,
			waitUntil: () => {},
		}

		const events: HookEvent<SynthesisArtifact>[] = []
		for await (const ev of runHook('OnSessionEnd', ctx)) {
			events.push(ev)
		}

		// Phase 1 tool result with ok:true must exist
		// extract-bar-range-signals returns a SignalBundle with muq_scores: number[][] (2D array)
		const toolResultEv = events.find((e) => e.type === 'phase1_tool_result')
		expect(toolResultEv).toBeDefined()
		expect(toolResultEv).toMatchObject({ type: 'phase1_tool_result', ok: true, tool: 'extract-bar-range-signals' })

		// SignalBundle output has muq_scores: number[][] — one 6-vector per chunk
		if (toolResultEv && toolResultEv.type === 'phase1_tool_result' && toolResultEv.ok) {
			const output = toolResultEv.output as { muq_scores: unknown[][] }
			expect(Array.isArray(output.muq_scores)).toBe(true)
			expect(output.muq_scores.length).toBeGreaterThan(0)
			// Each entry is the 6-vector from the chunk
			expect(Array.isArray(output.muq_scores[0])).toBe(true)
			expect((output.muq_scores[0] as number[]).length).toBe(6)
		}

		// Artifact has non-empty focus_areas
		const artifactEv = events.find((e) => e.type === 'artifact')
		expect(artifactEv).toBeDefined()
		if (artifactEv && artifactEv.type === 'artifact') {
			expect(artifactEv.value.focus_areas.length).toBeGreaterThan(0)
		}

		expect(fetchSpy).toHaveBeenCalledTimes(3)
	} finally {
		vi.unstubAllGlobals()
	}
})
