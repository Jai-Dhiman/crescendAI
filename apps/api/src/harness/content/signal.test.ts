import { describe, expect, test } from "vitest";
import { SignalSchema } from "./signal";

const baseHeaders = {
	chunk_id: "chunk-abc",
	producer: "muq-handler",
	producer_version: "muq.v2.1",
	created_at: "2026-04-25T10:00:00.000Z",
};

describe("SignalSchema", () => {
	test("parses a valid MuQQuality signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "MuQQuality",
			payload: {
				dynamics: 0.6,
				timing: 0.55,
				pedaling: 0.5,
				articulation: 0.62,
				phrasing: 0.58,
				interpretation: 0.6,
			},
		});
		expect(result.success).toBe(true);
	});

	test("fails when MuQQuality payload is missing a dimension", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "MuQQuality",
			payload: {
				dynamics: 0.6,
				timing: 0.55,
				pedaling: 0.5,
				articulation: 0.62,
				phrasing: 0.58,
			},
		});
		expect(result.success).toBe(false);
	});

	test("parses a valid AMTTranscription signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "AMTTranscription",
			payload: {
				midi_notes: [
					{ pitch: 60, onset_ms: 0, offset_ms: 500, velocity: 80 },
				],
				pedals: [{ onset_ms: 0, offset_ms: 1200, type: "sustain" }],
			},
		});
		expect(result.success).toBe(true);
	});

	test("parses a valid StopMoment signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "StopMoment",
			payload: {
				probability: 0.82,
				dimension: "pedaling",
				bar_range: { start: 12, end: 16 },
			},
		});
		expect(result.success).toBe(true);
	});

	test("parses a valid ScoreAlignment signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "ScoreAlignment",
			payload: {
				alignments: [
					{ chunk_offset_ms: 100, score_offset_ms: 95, confidence: 0.9 },
				],
			},
		});
		expect(result.success).toBe(true);
	});

	test("fails for an unknown schema_name", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "Unknown",
			payload: {},
		});
		expect(result.success).toBe(false);
	});
});
