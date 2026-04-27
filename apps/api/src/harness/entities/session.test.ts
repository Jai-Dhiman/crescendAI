import { describe, expect, test } from "vitest";
import { SessionSchema } from "./session";

describe("SessionSchema", () => {
	test("parses a valid Session row", () => {
		const result = SessionSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc123",
			startedAt: "2026-04-25T10:00:00.000Z",
			endedAt: "2026-04-25T10:30:00.000Z",
			avgDynamics: 0.6,
			avgTiming: 0.55,
			avgPedaling: 0.5,
			avgArticulation: 0.62,
			avgPhrasing: 0.58,
			avgInterpretation: 0.6,
			observationsJson: null,
			chunksSummaryJson: null,
			conversationId: null,
			accumulatorJson: null,
			needsSynthesis: false,
		});
		expect(result.success).toBe(true);
	});

	test("parses a valid in-progress Session with endedAt null", () => {
		const result = SessionSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc123",
			startedAt: "2026-04-25T10:00:00.000Z",
			endedAt: null,
			needsSynthesis: false,
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when endedAt is before startedAt", () => {
		const result = SessionSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc123",
			startedAt: "2026-04-25T10:30:00.000Z",
			endedAt: "2026-04-25T10:00:00.000Z",
			needsSynthesis: false,
		});
		expect(result.success).toBe(false);
	});
});
