import { describe, expect, test } from "vitest";
import { ObservationSchema } from "./observation";

describe("ObservationSchema", () => {
	test("parses a valid Observation row", () => {
		const result = ObservationSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc",
			sessionId: "22222222-3333-4444-5555-666666666666",
			chunkIndex: 4,
			dimension: "pedaling",
			observationText: "Pedal held a bit long into the next harmony at bar 14.",
			elaborationText: null,
			reasoningTrace: null,
			framing: "correction",
			dimensionScore: 0.42,
			studentBaseline: 0.55,
			pieceContext: null,
			learningArc: "mid-learning",
			isFallback: false,
			createdAt: "2026-04-25T10:05:00.000Z",
			messageId: null,
			conversationId: null,
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when dimension is not one of the 6 known", () => {
		const result = ObservationSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc",
			sessionId: "22222222-3333-4444-5555-666666666666",
			dimension: "unknown",
			observationText: "x",
			isFallback: false,
			createdAt: "2026-04-25T10:05:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});
