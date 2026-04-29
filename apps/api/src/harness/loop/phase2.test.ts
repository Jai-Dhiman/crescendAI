import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { CompoundBinding, PhaseContext, HookEvent } from "./types";
import type { Bindings } from "../../lib/types";
import {
	SynthesisArtifactSchema,
	type SynthesisArtifact,
} from "../artifacts/synthesis";
import { runPhase2 } from "./phase2";

const MOCK_BINDINGS = {
	AI_GATEWAY_TEACHER: "https://gw.example",
	ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const BINDING: CompoundBinding = {
	compoundName: "session-synthesis",
	procedurePrompt: "test",
	tools: [],
	mode: "buffered",
	phases: 2,
	artifactSchema: SynthesisArtifactSchema,
	artifactToolName: "write_synthesis_artifact",
};

const PHASE_CTX: PhaseContext = {
	env: MOCK_BINDINGS,
	studentId: "stu_1",
	sessionId: "sess_1",
	conversationId: null,
	digest: {},
	waitUntil: () => {},
	turnCap: 8,
};

const VALID_ARTIFACT: SynthesisArtifact = {
	session_id: "sess_1",
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

describe("runPhase2 happy path", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("yields phase2_started then artifact when forced tool returns valid payload", async () => {
		const anthropicResp = {
			content: [
				{
					type: "tool_use",
					id: "tu_1",
					name: "write_synthesis_artifact",
					input: VALID_ARTIFACT,
				},
			],
			stop_reason: "tool_use",
		};
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(anthropicResp), { status: 200 }),
		);

		const events: HookEvent<unknown>[] = [];
		for await (const ev of runPhase2(PHASE_CTX, BINDING, [])) {
			events.push(ev);
		}

		expect(events[0]).toEqual({ type: "phase2_started" });
		expect(events[1]).toEqual({ type: "artifact", value: VALID_ARTIFACT });
		expect(events).toHaveLength(2);
		expect(fetchSpy).toHaveBeenCalledTimes(1);
		const callBody = JSON.parse(
			(fetchSpy.mock.calls[0][1] as { body: string }).body,
		);
		expect(callBody.tool_choice).toEqual({
			type: "tool",
			name: "write_synthesis_artifact",
		});
	});
});

describe("runPhase2 validation failure", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("yields validation_error and not artifact when input fails Zod", async () => {
		const malformed = {
			session_id: "",
			synthesis_scope: "session",
			strengths: [],
			focus_areas: [],
			proposed_exercises: [],
			dominant_dimension: "phrasing",
			recurring_pattern: null,
			next_session_focus: null,
			diagnosis_refs: [],
			headline: "too short",
		};
		const anthropicResp = {
			content: [
				{
					type: "tool_use",
					id: "tu_1",
					name: "write_synthesis_artifact",
					input: malformed,
				},
			],
			stop_reason: "tool_use",
		};
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(anthropicResp), { status: 200 }),
		);

		const events: HookEvent<unknown>[] = [];
		for await (const ev of runPhase2(PHASE_CTX, BINDING, [])) {
			events.push(ev);
		}

		expect(events[0]).toEqual({ type: "phase2_started" });
		expect(events[1]?.type).toBe("validation_error");
		expect(events.find((e) => e.type === "artifact")).toBeUndefined();
	});
});
