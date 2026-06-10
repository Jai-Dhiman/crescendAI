import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bindings } from "../../lib/types";
import {
	type SynthesisArtifact,
	SynthesisArtifactSchema,
} from "../artifacts/synthesis";
import { buildPhase2Prompt, runPhase2 } from "./phase2";
import type { CompoundBinding, HookEvent, PhaseContext } from "./types";

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

const VALID_ARTIFACT = {
	session_id: "sess_1",
	synthesis_scope: "session",
	strengths: [],
	focus_areas: [],
	prescribed_exercise: null,
	dominant_dimension: "phrasing",
	recurring_pattern: null,
	next_session_focus: null,
	diagnosis_refs: [],
	headline:
		"You showed up and put in real work today. The session was short but focused, and we'll keep building from here. There is plenty to dig into next time, and I'll be ready when you are. Keep listening for the shape of each phrase as you play. " +
		"Tomorrow we'll come at it fresh with one specific thing to chase down.",
	assigned_loops: [],
} as unknown as SynthesisArtifact;

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

	it("yields validation_error and not artifact when every repair attempt fails Zod", async () => {
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
		// Model keeps returning an invalid artifact on every repair attempt.
		// Fresh Response per call (a Response body can only be read once).
		fetchSpy.mockImplementation(
			async () => new Response(JSON.stringify(anthropicResp), { status: 200 }),
		);

		const events: HookEvent<unknown>[] = [];
		for await (const ev of runPhase2(PHASE_CTX, BINDING, [])) {
			events.push(ev);
		}

		// Only after the repair budget is exhausted does it surface validation_error.
		expect(events[0]).toEqual({ type: "phase2_started" });
		expect(events.at(-1)?.type).toBe("validation_error");
		expect(events.find((e) => e.type === "artifact")).toBeUndefined();
		// Initial call + repair retries (the model was given the zod error to fix).
		expect(fetchSpy.mock.calls.length).toBeGreaterThan(1);
	});
});

describe("buildPhase2Prompt — reflection+prescribe instructions", () => {
	const digest = { dominant_dimension: "dynamics", duration_minutes: 30 };
	const diagnoses = [{ id: "d1" }];

	it("instructs headline to be 2-4 sentences ending in one directional question", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, "");
		expect(prompt).toContain("2-4 sentences");
		expect(prompt).toContain("directional question");
		expect(prompt).toContain("dominant_dimension");
	});

	it("instructs prescribed_exercise to target dominant_dimension as own_passage_loop or corpus_drill", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, "");
		expect(prompt).toContain("prescribed_exercise");
		expect(prompt).toContain("own_passage_loop");
		expect(prompt).toContain("corpus_drill");
		expect(prompt).toContain("dominant_dimension");
		expect(prompt).not.toContain("proposed_exercises[0]");
		expect(prompt).not.toContain("proposed_exercises");
	});

	it("still passes all Task 1 invariants after the new instructions", () => {
		const guardrail =
			"This is the student's first session -- describe only what happened within this session; do not reference past sessions or claim improvement over time.";
		const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
		expect(prompt).toContain(JSON.stringify(digest, null, 2));
		expect(prompt).toContain(`(${diagnoses.length})`);
		expect(prompt).toContain(guardrail);
		expect(prompt).toContain("write_synthesis_artifact");
	});
});

describe("buildPhase2Prompt — current text invariants", () => {
	const digest = { dominant_dimension: "phrasing", duration_minutes: 20 };
	const diagnoses = [{ id: "d1" }, { id: "d2" }];
	const guardrail =
		"This is the student's first session -- describe only what happened within this session; do not reference past sessions or claim improvement over time.";

	it("contains the digest JSON", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
		expect(prompt).toContain(JSON.stringify(digest, null, 2));
	});

	it("contains the diagnoses count", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
		expect(prompt).toContain(`(${diagnoses.length})`);
	});

	it("contains the guardrail when provided", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
		expect(prompt).toContain(guardrail);
	});

	it("omits the guardrail when empty string", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, "");
		expect(prompt).not.toContain("first session");
	});

	it("contains the write_synthesis_artifact tool name", () => {
		const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
		expect(prompt).toContain("write_synthesis_artifact");
	});
});
