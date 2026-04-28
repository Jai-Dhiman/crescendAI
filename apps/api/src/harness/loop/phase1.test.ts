import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { SynthesisArtifactSchema } from "../artifacts/synthesis";
import type { CompoundBinding, PhaseContext, Phase1Event } from "./types";
import type { Bindings } from "../../lib/types";
import { runPhase1 } from "./phase1";

const MOCK_BINDINGS = {
	AI_GATEWAY_TEACHER: "https://gw.example",
	ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const EMPTY_BINDING: CompoundBinding = {
	compoundName: "session-synthesis",
	procedurePrompt: "test",
	tools: [],
	artifactSchema: SynthesisArtifactSchema,
	artifactToolName: "write_synthesis_artifact",
};

const PHASE_CTX: PhaseContext = {
	env: MOCK_BINDINGS,
	studentId: "stu_1",
	sessionId: "sess_1",
	conversationId: null,
	digest: { topMoments: [], drillingRecords: [], modeTransitions: [] },
	waitUntil: () => {},
	turnCap: 8,
};

const ANTHROPIC_END_TURN = {
	id: "msg_test",
	type: "message",
	role: "assistant",
	content: [{ type: "text", text: "Nothing to dispatch." }],
	stop_reason: "end_turn",
	usage: { input_tokens: 10, output_tokens: 5 },
};

describe("runPhase1 empty registry", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("yields phase1_done with zero tools when binding.tools is empty", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(ANTHROPIC_END_TURN), { status: 200 }),
		);

		const events: Phase1Event[] = [];
		for await (const ev of runPhase1(PHASE_CTX, EMPTY_BINDING)) {
			events.push(ev);
		}

		expect(events).toHaveLength(1);
		expect(events[0]).toEqual({
			type: "phase1_done",
			toolCallCount: 0,
			turnCount: 1,
		});
		expect(fetchSpy).toHaveBeenCalledTimes(1);
	});

	it("returns the collected DiagnosisArtifacts (empty when no tools called)", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(ANTHROPIC_END_TURN), { status: 200 }),
		);

		const events: Phase1Event[] = [];
		for await (const ev of runPhase1(PHASE_CTX, EMPTY_BINDING)) {
			events.push(ev);
		}
		const toolResultCount = events.filter(
			(e) => e.type === "phase1_tool_result",
		).length;
		expect(toolResultCount).toBe(0);
	});
});
