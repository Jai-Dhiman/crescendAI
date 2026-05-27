import { describe, expect, it } from "vitest";
import { buildV6WsPayload } from "./session-brain";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";
import { parseMuqResponse } from "../services/inference";
import { wsOutgoingMessageSchema } from "./session-brain.schema";
import { SessionAccumulator, type AccumulatedMoment } from "../services/accumulator";
import { buildBarAnalysisFacts } from "../services/bar-analysis-facts";
import type { ChunkAnalysis } from "../services/wasm-bridge";

const HEADLINE =
	"You showed up and put in real work today. The session was short but focused, and we'll keep building from here. There is plenty to dig into next time, and I'll be ready when you are. Keep listening for the shape of each phrase as you play. " +
	"Tomorrow we'll come at it fresh with one specific thing to chase down.";

const ARTIFACT: SynthesisArtifact = {
	session_id: "sess_42",
	synthesis_scope: "session",
	strengths: [],
	focus_areas: [],
	proposed_exercises: [],
	dominant_dimension: "phrasing",
	recurring_pattern: null,
	next_session_focus: null,
	diagnosis_refs: [],
	headline: HEADLINE,
	assigned_loops: [],
};

const ARTIFACT_WITH_LOOP: SynthesisArtifact = {
	...ARTIFACT,
	assigned_loops: [
		{ id: "loop-1", pieceId: "piece-abc", barsStart: 1, barsEnd: 4 },
	],
};

describe("buildV6WsPayload", () => {
	it("maps artifact.headline to WebSocket text field", () => {
		const payload = buildV6WsPayload(ARTIFACT);
		expect(payload.type).toBe("synthesis");
		expect(payload.text).toBe(HEADLINE);
		expect(payload.components).toEqual([]);
		expect(payload.isFallback).toBe(false);
	});

	it("uses artifact.headline as text, not a free-form string", () => {
		const customHeadline =
			"Different headline for this test, long enough to pass schema validation at three hundred characters minimum so lets add some more content here to ensure we exceed the threshold easily and have a proper test fixture that will not fail Zod checks now.";
		const payload = buildV6WsPayload({ ...ARTIFACT, headline: customHeadline });
		expect(payload.text).toBe(customHeadline);
		expect(payload.text).not.toBe(""); // guard: never produces empty text
	});

	it("always returns empty components array in V6 when no loopComponents passed", () => {
		const payload = buildV6WsPayload({
			...ARTIFACT,
			proposed_exercises: ["ex1", "ex2"],
		});
		expect(payload.components).toEqual([]); // Plan 4 fills in real exercise component construction
	});

	it("buildV6WsPayload with loop components returns them in components array", () => {
		const loopComp = { type: "segment_loop", config: { id: "loop-1" } };
		const payload = buildV6WsPayload(ARTIFACT_WITH_LOOP, [loopComp]);
		expect(payload.components).toHaveLength(1);
		expect(payload.components[0]?.type).toBe("segment_loop");
	});
});

describe("parseMuqResponse chroma extraction", () => {
	it("returns chromaBytes=null when response has no chroma_b64", () => {
		const raw = {
			predictions: {
				dynamics: 0.6,
				timing: 0.7,
				pedaling: 0.5,
				articulation: 0.8,
				phrasing: 0.65,
				interpretation: 0.72,
			},
		};
		const result = parseMuqResponse(raw);
		expect(result.chromaBytes).toBeNull();
		expect(result.chromaFrames).toBe(0);
	});

	it("returns decoded chromaBytes when chroma_b64 present", () => {
		const nFrames = 5;
		const buf = new Uint8Array(12 * nFrames * 4).fill(127);
		const b64 = btoa(String.fromCharCode(...buf));
		const raw = {
			predictions: {
				dynamics: 0.6,
				timing: 0.7,
				pedaling: 0.5,
				articulation: 0.8,
				phrasing: 0.65,
				interpretation: 0.72,
			},
			chroma_b64: b64,
			chroma_frames: nFrames,
			chroma_frame_rate_hz: 50.0,
		};
		const result = parseMuqResponse(raw);
		expect(result.chromaBytes).not.toBeNull();
		expect(result.chromaBytes!.byteLength).toBe(12 * nFrames * 4);
		expect(result.chromaFrames).toBe(nFrames);
		expect(result.chromaFrameRateHz).toBe(50.0);
	});

	it("throws InferenceError when MuQ response missing a dimension", () => {
		const raw = {
			predictions: {
				dynamics: 0.6,
				// timing missing
				pedaling: 0.5,
				articulation: 0.8,
				phrasing: 0.65,
				interpretation: 0.72,
			},
		};
		expect(() => parseMuqResponse(raw as Parameters<typeof parseMuqResponse>[0])).toThrow(
			"MuQ response missing dimensions: timing",
		);
	});
});

describe("chunk_bar_map WebSocket message", () => {
	it("DO sends chunk_bar_map message with correct shape after successful chroma alignment", () => {
		const barMapMsg = {
			type: "chunk_bar_map",
			chunk_index: 2,
			bar_min: 5,
			bar_max: 9,
			bar_per_frame: [5, 6, 7, 8, 9],
		};

		// Should parse without throwing — asserts the schema accepts the new variant.
		expect(() => wsOutgoingMessageSchema.parse(barMapMsg)).not.toThrow();

		const parsed = wsOutgoingMessageSchema.parse(barMapMsg);
		expect(parsed.type).toBe("chunk_bar_map");
		expect(parsed.chunk_index).toBe(2);
		expect(parsed.bar_min).toBe(5);
		expect(parsed.bar_max).toBe(9);
		expect(parsed.bar_per_frame).toEqual([5, 6, 7, 8, 9]);
	});
});

describe("session-brain accumulation contract", () => {
	it("buildBarAnalysisFacts result is the shape session-brain must attach to accMoment.llmAnalysis", () => {
		const analysis: ChunkAnalysis = {
			tier: 1,
			bar_range: "4-7",
			dimensions: [
				{ dimension: "timing", analysis: "rushing" },
				{ dimension: "dynamics", analysis: "soft" },
			],
		};
		const baselines = {
			dynamics: 0.5, timing: 0.5, pedaling: 0.5,
			articulation: 0.5, phrasing: 0.5, interpretation: 0.5,
		};
		const scores: [number, number, number, number, number, number] = [
			0.20, 0.30, 0.50, 0.50, 0.50, 0.50,
		];
		const facts = buildBarAnalysisFacts(analysis, scores, baselines, "timing");

		const moment: AccumulatedMoment = {
			chunkIndex: 0,
			dimension: "timing",
			score: 0.30,
			baseline: 0.50,
			deviation: -0.20,
			isPositive: false,
			reasoning: "rushing",
			barRange: [4, 7],
			analysisTier: 1,
			timestampMs: 0,
			llmAnalysis: facts,
		};
		const acc = new SessionAccumulator();
		acc.accumulateMoment(moment);
		const top = acc.topMoments();
		expect(top[0]?.llmAnalysis).not.toBeNull();
		expect(top[0]?.llmAnalysis?.selected.dimension).toBe("timing");
		expect(top[0]?.llmAnalysis?.correlated.map((d) => d.dimension)).toContain("dynamics");
	});
});
