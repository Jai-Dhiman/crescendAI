import { test, expect, vi } from "vitest";
import { runHook } from "./runHook";
import type { HookContext, HookEvent } from "./types";
import type { SynthesisArtifact } from "../artifacts/synthesis";
import type { Bindings } from "../../lib/types";

// Regression for issue #28: V6 Phase 2 built its forced-tool input_schema with
// zodToJsonSchema(..., { target: "openApi3" }), emitting constructs the Anthropic
// Messages API rejects under JSON Schema draft 2020-12 ($ref dedup, nullable: true,
// boolean exclusiveMinimum) -> every Phase 2 call returned HTTP 400 -> V6 never
// emitted a SynthesisArtifact. This test pins the schema the API actually accepts
// by asserting the outgoing tool schema is free of those illegal constructs, and
// that a valid artifact flows through. It drives the real runHook pipeline and
// mocks only the LLM HTTP boundary.

const MOCK_BINDINGS = {
	AI_GATEWAY_TEACHER: "https://gw.example",
	ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const SPARSE_DIGEST = {
	sessionDurationMs: 90000,
	practicePattern: '{"mode":"continuous_play"}',
	topMoments: [
		{ dimension: "dynamics", deviation: -1.31, is_positive: false, reasoning: "Flatter than session mean." },
	],
	drillingRecords: [],
	pieceMetadata: null,
	chunks: [],
	baselines: null,
	cohort_tables: {},
	session_history: [],
	past_diagnoses: [],
	reference_mode: "within_session",
};

const VALID_ARTIFACT: SynthesisArtifact = {
	session_id: "sess-reg",
	synthesis_scope: "session",
	strengths: [],
	focus_areas: [{ dimension: "dynamics", one_liner: "Dynamics flattened in the middle section.", severity: "minor" }],
	proposed_exercises: [],
	dominant_dimension: "dynamics",
	recurring_pattern: null,
	next_session_focus: null,
	diagnosis_refs: [],
	headline:
		"This session held a steady pulse and the phrases breathed naturally for the most part. The one place to lean into next is dynamics: the middle section flattened out where it wanted more shape and contrast. Try exaggerating the swells and the quiet moments until the difference feels almost too big, then pull back. Want a drill targeting that?",
	assigned_loops: [],
};

// Recursively collect any illegal-under-2020-12 constructs in the tool schema.
function findIllegalConstructs(node: unknown, path = "$"): string[] {
	const issues: string[] = [];
	if (Array.isArray(node)) {
		node.forEach((v, i) => issues.push(...findIllegalConstructs(v, `${path}[${i}]`)));
		return issues;
	}
	if (node && typeof node === "object") {
		for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
			if (k === "$ref") issues.push(`${path}.$ref (internal ref not accepted as tool input_schema)`);
			if (k === "nullable") issues.push(`${path}.nullable (OpenAPI 3.0 keyword; 2020-12 wants type: [..,"null"])`);
			if (k === "exclusiveMinimum" && typeof v === "boolean") issues.push(`${path}.exclusiveMinimum is boolean (2020-12 requires a number)`);
			if (k === "exclusiveMaximum" && typeof v === "boolean") issues.push(`${path}.exclusiveMaximum is boolean (2020-12 requires a number)`);
			issues.push(...findIllegalConstructs(v, `${path}.${k}`));
		}
	}
	return issues;
}

test("Phase 2 forced-tool input_schema is draft-2020-12-clean (Anthropic-acceptable)", async () => {
	const fetchSpy = vi.fn();
	vi.stubGlobal("fetch", fetchSpy);
	try {
		// Phase 1: model dispatches no molecules on sparse data, ends turn.
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify({ content: [{ type: "text", text: "No AMT signals; nothing to diagnose." }], stop_reason: "end_turn" }), { status: 200 }),
		);
		// Phase 2: model writes a valid artifact.
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify({ content: [{ type: "tool_use", id: "tu_art", name: "write_synthesis_artifact", input: VALID_ARTIFACT }], stop_reason: "tool_use" }), { status: 200 }),
		);

		const ctx: HookContext = {
			env: MOCK_BINDINGS,
			studentId: "stu-reg",
			sessionId: "sess-reg",
			conversationId: null,
			digest: SPARSE_DIGEST,
			waitUntil: () => {},
			trigger: "synthesis",
		};

		const events: HookEvent<SynthesisArtifact>[] = [];
		for await (const ev of runHook("OnSessionEnd", ctx)) events.push(ev);

		// Locate the Phase 2 request (forced write_synthesis_artifact tool_choice).
		const phase2Call = fetchSpy.mock.calls.find((c) => {
			const body = JSON.parse((c[1] as RequestInit).body as string) as { tool_choice?: { name?: string } };
			return body.tool_choice?.name === "write_synthesis_artifact";
		});
		expect(phase2Call).toBeDefined();

		const body = JSON.parse((phase2Call?.[1] as RequestInit).body as string) as {
			tools: { name: string; input_schema: unknown }[];
		};
		const inputSchema = body.tools[0].input_schema;
		const illegal = findIllegalConstructs(inputSchema);
		expect(illegal).toEqual([]);

		// And the valid artifact must flow through to an artifact event.
		const artifactEv = events.find((e) => e.type === "artifact");
		expect(artifactEv).toBeDefined();
	} finally {
		vi.unstubAllGlobals();
	}
});

test("Phase 2 repairs a sub-300-char headline via a validation-feedback retry", async () => {
	const SHORT_HEADLINE_ARTIFACT = {
		...VALID_ARTIFACT,
		headline: "Good steady session; work the dynamics next. Want a drill?",
	};
	expect(SHORT_HEADLINE_ARTIFACT.headline.length).toBeLessThan(300);

	const fetchSpy = vi.fn();
	vi.stubGlobal("fetch", fetchSpy);
	try {
		// Phase 1: no molecules dispatched on sparse data.
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify({ content: [{ type: "text", text: "Nothing to diagnose." }], stop_reason: "end_turn" }), { status: 200 }),
		);
		// Phase 2 attempt 1: headline too short -> validation fails -> repair turn.
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify({ content: [{ type: "tool_use", id: "tu_short", name: "write_synthesis_artifact", input: SHORT_HEADLINE_ARTIFACT }], stop_reason: "tool_use" }), { status: 200 }),
		);
		// Phase 2 attempt 2: valid artifact.
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify({ content: [{ type: "tool_use", id: "tu_ok", name: "write_synthesis_artifact", input: VALID_ARTIFACT }], stop_reason: "tool_use" }), { status: 200 }),
		);

		const ctx: HookContext = {
			env: MOCK_BINDINGS,
			studentId: "stu-reg",
			sessionId: "sess-reg",
			conversationId: null,
			digest: SPARSE_DIGEST,
			waitUntil: () => {},
			trigger: "synthesis",
		};

		const events: HookEvent<SynthesisArtifact>[] = [];
		for await (const ev of runHook("OnSessionEnd", ctx)) events.push(ev);

		// Recovered: an artifact event is emitted, not a validation_error.
		expect(events.find((e) => e.type === "artifact")).toBeDefined();
		expect(events.find((e) => e.type === "validation_error")).toBeUndefined();

		// The repair turn fed the zod error back to the model (3 calls total: phase1 + 2 phase2).
		expect(fetchSpy).toHaveBeenCalledTimes(3);
		const repairBody = JSON.parse((fetchSpy.mock.calls[2][1] as RequestInit).body as string) as {
			messages: { role: string; content: unknown }[];
		};
		// The repair must carry a tool_result for the rejected tool_use_id, or the
		// Anthropic API rejects the follow-up (tool_use without tool_result -> 400).
		const repairUserMsg = repairBody.messages[repairBody.messages.length - 1];
		expect(repairUserMsg.role).toBe("user");
		const block = (repairUserMsg.content as { type: string; tool_use_id: string; content: string }[])[0];
		expect(block.type).toBe("tool_result");
		expect(block.tool_use_id).toBe("tu_short");
		expect(block.content).toContain("failed validation");
		expect(block.content).toContain("300");
	} finally {
		vi.unstubAllGlobals();
	}
});
