import { describe, expect, it, vi } from "vitest";

vi.mock("./llm", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./llm")>();
	return { ...actual, callAnthropic: vi.fn() };
});
vi.mock("./memory", () => ({
	buildMemoryContext: vi.fn().mockResolvedValue(""),
}));

import { callAnthropic } from "./llm";
import { synthesize, type SynthesisInput } from "./teacher";
import type { ServiceContext } from "../lib/types";

const mockCallAnthropic = vi.mocked(callAnthropic);

function baseInput(referenceMode: "within_session" | null): SynthesisInput {
	return {
		studentId: "stu_1",
		conversationId: null,
		sessionDurationMs: 120_000,
		practicePattern: "continuous_play",
		topMoments: [{ dimension: "timing", score: 0.3 }],
		drillingRecords: [],
		pieceMetadata: { composer: "Chopin", title: "Etude" },
		enrichedChunks: [],
		baselines: null,
		sessionHistory: [],
		pastDiagnoses: [],
		pieceId: null,
		referenceMode,
	};
}

async function captureSystemBlocks(
	referenceMode: "within_session" | null,
): Promise<Array<{ type: string; text?: string }>> {
	mockCallAnthropic.mockReset();
	mockCallAnthropic.mockResolvedValue({
		content: [{ type: "text", text: "Your session." }],
		stop_reason: "end_turn",
		usage: { input_tokens: 0, output_tokens: 0 },
	});
	const ctx = { db: {}, env: {} } as unknown as ServiceContext;
	await synthesize(ctx, baseInput(referenceMode));
	const callArgs = mockCallAnthropic.mock.calls[0];
	const request = callArgs?.[1] as { system: Array<{ type: string; text?: string }> };
	return request.system;
}

describe("synthesize referenceMode threading", () => {
	it("forwards within_session reference mode into the teacher framing", async () => {
		const system = await captureSystemBlocks("within_session");
		// Match the angle-bracketed `<session_data>` tag, which is unique to the
		// framing block. UNIFIED_TEACHER_SYSTEM contains only the bare
		// `show_session_data` substring (prompts.ts:99), so a plain
		// `includes("session_data")` would resolve to systemBlocks[0] instead.
		const framing = system.find((b) => b.text?.includes("<session_data>"));
		expect(framing?.text).toContain("This is the student's first session");
		expect(framing?.text).toContain('"reference_mode"');
	});

	it("omits the first-session guardrail when referenceMode is null", async () => {
		const system = await captureSystemBlocks(null);
		// See note above: match `<session_data>` to select the framing block, not
		// UNIFIED_TEACHER_SYSTEM (which contains the bare `show_session_data`).
		const framing = system.find((b) => b.text?.includes("<session_data>"));
		expect(framing?.text).not.toContain("This is the student's first session");
		expect(framing?.text).not.toContain("reference_mode");
	});
});
