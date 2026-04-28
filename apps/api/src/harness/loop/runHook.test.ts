import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { Bindings } from "../../lib/types";
import type { HookContext, HookEvent } from "./types";
import { runHook } from "./runHook";
import type { SynthesisArtifact } from "../artifacts/synthesis";

const MOCK_BINDINGS = {
	AI_GATEWAY_TEACHER: "https://gw.example",
	ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const VALID_ARTIFACT: SynthesisArtifact = {
	session_id: "sess_42",
	synthesis_scope: "session",
	strengths: [],
	focus_areas: [],
	proposed_exercises: [],
	dominant_dimension: "phrasing",
	recurring_pattern: null,
	next_session_focus: null,
	diagnosis_refs: [],
	headline:
		"You showed up and put in real work today. The session was short but focused, and we'll keep building from here. There is plenty to dig into next time, and I'll be ready when you are. Keep listening for the shape of each phrase as you play. " +
		"Tomorrow we'll come at it fresh with one specific thing to chase down.",
};

const HOOK_CTX: HookContext = {
	env: MOCK_BINDINGS,
	studentId: "stu_1",
	sessionId: "sess_42",
	conversationId: "conv_1",
	digest: { topMoments: [], drillingRecords: [], modeTransitions: [] },
	waitUntil: () => {},
};

describe("runHook OnSessionEnd", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("yields phase1_done, phase2_started, artifact in order with empty registry", async () => {
		// Phase 1: end_turn with no tools
		fetchSpy.mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					content: [{ type: "text", text: "no diagnoses to dispatch" }],
					stop_reason: "end_turn",
				}),
				{ status: 200 },
			),
		);
		// Phase 2: forced write returns valid artifact
		fetchSpy.mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					content: [
						{
							type: "tool_use",
							id: "tu_1",
							name: "write_synthesis_artifact",
							input: VALID_ARTIFACT,
						},
					],
					stop_reason: "tool_use",
				}),
				{ status: 200 },
			),
		);

		const events: HookEvent<SynthesisArtifact>[] = [];
		for await (const ev of runHook("OnSessionEnd", HOOK_CTX)) {
			events.push(ev);
		}

		const types = events.map((e) => e.type);
		expect(types).toEqual(["phase1_done", "phase2_started", "artifact"]);
		const artifactEv = events.find((e) => e.type === "artifact");
		expect(artifactEv).toBeDefined();
		if (artifactEv && artifactEv.type === "artifact") {
			expect(artifactEv.value.session_id).toBe("sess_42");
		}
		expect(fetchSpy).toHaveBeenCalledTimes(2);
	});

	it("yields phase_error and stops if hook is unbound", async () => {
		const events: HookEvent<unknown>[] = [];
		for await (const ev of runHook("OnChatMessage" as const, HOOK_CTX)) {
			events.push(ev);
		}
		expect(events).toHaveLength(1);
		expect(events[0]?.type).toBe("phase_error");
	});
});
