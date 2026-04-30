import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { parseAnthropicStream, stripAnalysis, synthesizeV6, chatV6 } from "./teacher";
import type { InlineComponent } from "./tool-processor";
import type { SynthesisInput } from "./teacher";
import type { ServiceContext } from "../lib/types";
import type { HookEvent } from "../harness/loop/types";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";

vi.mock("./llm", async (importOriginal) => {
	const actual = await importOriginal() as Record<string, unknown>;
	return { ...actual, callAnthropicStream: vi.fn() };
});
vi.mock("./memory", () => ({ buildMemoryContext: vi.fn().mockResolvedValue("") }));

// ---------------------------------------------------------------------------
// Helpers for building fake SSE streams
// ---------------------------------------------------------------------------

function sseLines(...events: Array<{ event: string; data: unknown }>): string {
	return (
		events
			.map((e) => `event: ${e.event}\ndata: ${JSON.stringify(e.data)}`)
			.join("\n\n") + "\n\n"
	);
}

function makeStream(content: string): ReadableStream {
	const encoder = new TextEncoder();
	const bytes = encoder.encode(content);
	return new ReadableStream({
		start(controller) {
			controller.enqueue(bytes);
			controller.close();
		},
	});
}

// ---------------------------------------------------------------------------
// Test: stream parser with text only
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - text only", () => {
	it("yields delta events and done with fullText", async () => {
		const sse = sseLines(
			{
				event: "message_start",
				data: { type: "message_start", message: { id: "msg_1" } },
			},
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "" },
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: "Hello" },
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: " world" },
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 0 },
			},
			{
				event: "message_delta",
				data: { type: "message_delta", delta: { stop_reason: "end_turn" } },
			},
			{
				event: "message_stop",
				data: { type: "message_stop" },
			},
		);

		const stream = makeStream(sse);
		const noopProcess = vi
			.fn()
			.mockResolvedValue({ name: "noop", componentsJson: [], isError: false });

		const events = [];
		for await (const event of parseAnthropicStream(stream, noopProcess)) {
			events.push(event);
		}

		// Should have two delta events and one done
		expect(events).toHaveLength(3);
		expect(events[0]).toEqual({ type: "delta", text: "Hello" });
		expect(events[1]).toEqual({ type: "delta", text: " world" });
		const doneEvent = events[2];
		expect(doneEvent.type).toBe("done");
		if (doneEvent.type === "done") {
			expect(doneEvent.fullText).toBe("Hello world");
			expect(doneEvent.allComponents).toEqual([]);
		}
		expect(noopProcess).not.toHaveBeenCalled();
	});
});

// ---------------------------------------------------------------------------
// Test: stream parser with text + tool_use
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - text + tool_use", () => {
	it("yields delta + tool_result + done with allComponents", async () => {
		const mockComponents: InlineComponent[] = [
			{
				type: "keyboard_guide",
				config: { title: "Scale", description: "C major", hands: "right" },
			},
		];
		const processToolFn = vi.fn().mockResolvedValue({
			name: "keyboard_guide",
			componentsJson: mockComponents,
			isError: false,
		});

		const sse = sseLines(
			{
				event: "message_start",
				data: { type: "message_start", message: { id: "msg_2" } },
			},
			// Text block starts
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "" },
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: "Try this:" },
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 0 },
			},
			// Tool use block starts
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 1,
					content_block: {
						type: "tool_use",
						id: "tool_abc",
						name: "keyboard_guide",
					},
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 1,
					delta: {
						type: "input_json_delta",
						partial_json: '{"title":"Scale","description":"C major",',
					},
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 1,
					delta: { type: "input_json_delta", partial_json: '"hands":"right"}' },
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 1 },
			},
			{
				event: "message_stop",
				data: { type: "message_stop" },
			},
		);

		const stream = makeStream(sse);
		const events = [];
		for await (const event of parseAnthropicStream(stream, processToolFn)) {
			events.push(event);
		}

		// tool_start + tool_result + done — intermediate narration ("Try this:") is discarded
		// because it appeared before a tool_use block in the same turn (Fix C).
		expect(events).toHaveLength(3);
		expect(events[0]).toEqual({ type: "tool_start", name: "keyboard_guide" });

		const toolEvent = events[1];
		expect(toolEvent.type).toBe("tool_result");
		if (toolEvent.type === "tool_result") {
			expect(toolEvent.name).toBe("keyboard_guide");
			expect(toolEvent.componentsJson).toEqual(mockComponents);
		}

		const doneEvent = events[2];
		expect(doneEvent.type).toBe("done");
		if (doneEvent.type === "done") {
			expect(doneEvent.fullText).toBe("Try this:");
			expect(doneEvent.allComponents).toEqual(mockComponents);
		}

		// processToolFn should have been called once with parsed JSON input
		expect(processToolFn).toHaveBeenCalledOnce();
		expect(processToolFn).toHaveBeenCalledWith("keyboard_guide", {
			title: "Scale",
			description: "C major",
			hands: "right",
		});
	});
});

// ---------------------------------------------------------------------------
// Test: stream parser with failed tool (isError: true)
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - failed tool", () => {
	it("yields tool_error (not tool_result) when processToolFn returns isError: true", async () => {
		const processToolFn = vi.fn().mockResolvedValue({
			name: "create_exercise",
			componentsJson: [],
			isError: true,
			errorMessage: "Exercises array must not be empty.",
		});

		const sse = sseLines(
			{
				event: "message_start",
				data: { type: "message_start", message: { id: "msg_3" } },
			},
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "" },
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: "Some text" },
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 0 },
			},
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 1,
					content_block: {
						type: "tool_use",
						id: "tool_xyz",
						name: "create_exercise",
					},
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 1,
					delta: {
						type: "input_json_delta",
						partial_json:
							'{"source_passage":"bar 1","target_skill":"legato","exercises":[]}',
					},
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 1 },
			},
			{
				event: "message_stop",
				data: { type: "message_stop" },
			},
		);

		const stream = makeStream(sse);
		const events = [];
		for await (const event of parseAnthropicStream(stream, processToolFn)) {
			events.push(event);
		}

		// tool_start + tool_error + done -- intermediate narration discarded (Fix C),
		// tool_error replaces tool_result because isError: true
		expect(events).toHaveLength(3);
		expect(events[0]).toEqual({ type: "tool_start", name: "create_exercise" });
		expect(events[1]).toEqual({
			type: "tool_error",
			name: "create_exercise",
			message: "Exercises array must not be empty.",
		});
		expect(events[2].type).toBe("done");

		const doneEvent = events[2];
		if (doneEvent.type === "done") {
			// allComponents should be empty since the tool failed
			expect(doneEvent.allComponents).toEqual([]);
			// fullText is still tracked internally even though the delta was not emitted
			expect(doneEvent.fullText).toBe("Some text");
		}
	});
});

// ---------------------------------------------------------------------------
// Test: split TCP chunks — SSE boundary falls across two read() calls
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - split TCP chunks", () => {
	it("buffers partial SSE messages and produces correct events when boundary is split", async () => {
		const encoder = new TextEncoder();

		// Build the full SSE sequence as individual event strings
		const blockStartEvent = `event: content_block_start\ndata: ${JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } })}\n\n`;
		const deltaEvent = `event: content_block_delta\ndata: ${JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "split" } })}\n\n`;
		const blockStopEvent = `event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`;
		const messageStopEvent = `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`;

		// Chunk 1: complete block_start + partial delta (cut before closing \n\n)
		const fullDelta = deltaEvent;
		const splitPoint = fullDelta.indexOf('", "index"') + 5; // cut mid-data
		const chunk1 = blockStartEvent + fullDelta.slice(0, splitPoint);
		const chunk2 =
			fullDelta.slice(splitPoint) + blockStopEvent + messageStopEvent;

		const stream = new ReadableStream({
			start(controller) {
				controller.enqueue(encoder.encode(chunk1));
				controller.enqueue(encoder.encode(chunk2));
				controller.close();
			},
		});

		const noopProcess = vi
			.fn()
			.mockResolvedValue({ name: "noop", componentsJson: [], isError: false });

		const events = [];
		for await (const event of parseAnthropicStream(stream, noopProcess)) {
			events.push(event);
		}

		// Should have one delta and one done, even though the delta arrived split across chunks
		const deltaEvents = events.filter((e) => e.type === "delta");
		expect(deltaEvents).toHaveLength(1);
		expect(deltaEvents[0]).toEqual({ type: "delta", text: "split" });

		const doneEvent = events.find((e) => e.type === "done");
		expect(doneEvent).toBeDefined();
		if (doneEvent && doneEvent.type === "done") {
			expect(doneEvent.fullText).toBe("split");
		}
	});
});

// ---------------------------------------------------------------------------
// Test: tool continuation — parseAnthropicStream reports tool calls for continuation
// ---------------------------------------------------------------------------

describe("parseAnthropicStream - tool call continuation info", () => {
	it("done event includes toolCalls with id, name, input, result when stop_reason is tool_use", async () => {
		const mockComponents: InlineComponent[] = [
			{
				type: "search_catalog_result",
				config: {
					matches: [
						{
							pieceId: "abc-123",
							composer: "Chopin",
							title: "Waltz Op. 64 No. 2",
						},
					],
				},
			},
		];
		const processToolFn = vi.fn().mockResolvedValue({
			name: "search_catalog",
			componentsJson: mockComponents,
			isError: false,
		});

		const sse = sseLines(
			{
				event: "message_start",
				data: { type: "message_start", message: { id: "msg_tool" } },
			},
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "" },
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: "Let me search for that." },
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 0 },
			},
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 1,
					content_block: {
						type: "tool_use",
						id: "toolu_abc123",
						name: "search_catalog",
					},
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 1,
					delta: {
						type: "input_json_delta",
						partial_json: '{"query":"Chopin waltz"}',
					},
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 1 },
			},
			{
				event: "message_delta",
				data: {
					type: "message_delta",
					delta: { stop_reason: "tool_use" },
				},
			},
			{
				event: "message_stop",
				data: { type: "message_stop" },
			},
		);

		const stream = makeStream(sse);
		const events = [];
		for await (const event of parseAnthropicStream(stream, processToolFn)) {
			events.push(event);
		}

		const doneEvent = events.find((e) => e.type === "done");
		expect(doneEvent).toBeDefined();
		if (doneEvent && doneEvent.type === "done") {
			// Must report tool calls for continuation
			expect(doneEvent.toolCalls).toHaveLength(1);
			expect(doneEvent.toolCalls[0].id).toBe("toolu_abc123");
			expect(doneEvent.toolCalls[0].name).toBe("search_catalog");
			expect(doneEvent.toolCalls[0].input).toEqual({ query: "Chopin waltz" });
			expect(doneEvent.toolCalls[0].result.componentsJson).toEqual(
				mockComponents,
			);

			// Must report stop_reason so caller knows to continue
			expect(doneEvent.stopReason).toBe("tool_use");
		}
	});

	it("done event has empty toolCalls and stopReason end_turn for text-only responses", async () => {
		const sse = sseLines(
			{
				event: "message_start",
				data: { type: "message_start", message: { id: "msg_text" } },
			},
			{
				event: "content_block_start",
				data: {
					type: "content_block_start",
					index: 0,
					content_block: { type: "text", text: "" },
				},
			},
			{
				event: "content_block_delta",
				data: {
					type: "content_block_delta",
					index: 0,
					delta: { type: "text_delta", text: "Just text" },
				},
			},
			{
				event: "content_block_stop",
				data: { type: "content_block_stop", index: 0 },
			},
			{
				event: "message_delta",
				data: {
					type: "message_delta",
					delta: { stop_reason: "end_turn" },
				},
			},
			{
				event: "message_stop",
				data: { type: "message_stop" },
			},
		);

		const stream = makeStream(sse);
		const noopProcess = vi
			.fn()
			.mockResolvedValue({ name: "noop", componentsJson: [], isError: false });

		const events = [];
		for await (const event of parseAnthropicStream(stream, noopProcess)) {
			events.push(event);
		}

		const doneEvent = events.find((e) => e.type === "done");
		expect(doneEvent).toBeDefined();
		if (doneEvent && doneEvent.type === "done") {
			expect(doneEvent.toolCalls).toHaveLength(0);
			expect(doneEvent.stopReason).toBe("end_turn");
		}
	});
});

// ---------------------------------------------------------------------------
// Test: stripAnalysis
// ---------------------------------------------------------------------------

describe("stripAnalysis", () => {
	it("strips a single analysis block", () => {
		const input =
			"<analysis>This is my reasoning.</analysis>Here is my response.";
		expect(stripAnalysis(input)).toBe("Here is my response.");
	});

	it("strips multiple analysis blocks", () => {
		const input =
			"<analysis>First reasoning block.</analysis>Middle text.<analysis>Second reasoning block.</analysis>Final response.";
		expect(stripAnalysis(input)).toBe("Middle text.Final response.");
	});

	it("returns text unchanged when no analysis block", () => {
		const input = "Just a normal response with no analysis tags.";
		expect(stripAnalysis(input)).toBe(
			"Just a normal response with no analysis tags.",
		);
	});

	it("handles empty analysis block", () => {
		const input = "<analysis></analysis>The response.";
		expect(stripAnalysis(input)).toBe("The response.");
	});

	it("handles multiline analysis block", () => {
		const input = "<analysis>\nLine 1\nLine 2\n</analysis>\nThe response.";
		expect(stripAnalysis(input)).toBe("The response.");
	});

	it("trims surrounding whitespace", () => {
		const input = "  <analysis>reasoning</analysis>  response  ";
		expect(stripAnalysis(input)).toBe("response");
	});
});

// ---------------------------------------------------------------------------
// Test: synthesizeV6 adapter
// ---------------------------------------------------------------------------

const V6_VALID_ARTIFACT: SynthesisArtifact = {
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
	assigned_loops: [],
};

describe("synthesizeV6 adapter", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("yields phase1_done, phase2_started, artifact for an empty session", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					content: [{ type: "text", text: "" }],
					stop_reason: "end_turn",
				}),
				{ status: 200 },
			),
		);
		fetchSpy.mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					content: [
						{
							type: "tool_use",
							id: "tu_1",
							name: "write_synthesis_artifact",
							input: V6_VALID_ARTIFACT,
						},
					],
					stop_reason: "tool_use",
				}),
				{ status: 200 },
			),
		);

		const ctx = {
			db: {} as ServiceContext["db"],
			env: {
				AI_GATEWAY_TEACHER: "https://gw.example",
				ANTHROPIC_API_KEY: "k",
			} as ServiceContext["env"],
		};

		const input: SynthesisInput = {
			studentId: "stu_1",
			conversationId: "conv_1",
			sessionDurationMs: 60_000,
			practicePattern: "[]",
			topMoments: [],
			drillingRecords: [],
			pieceMetadata: null,
			enrichedChunks: [],
			baselines: null,
			sessionHistory: [],
			pastDiagnoses: [],
		};

		const events: HookEvent<SynthesisArtifact>[] = [];
		for await (const ev of synthesizeV6(ctx, input, "sess_42")) {
			events.push(ev);
		}

		const types = events.map((e) => e.type);
		expect(types).toEqual(["phase1_done", "phase2_started", "artifact"]);
		expect(
			(events[2] as { type: "artifact"; value: SynthesisArtifact }).value,
		).toEqual(V6_VALID_ARTIFACT);
	});
});

// ---------------------------------------------------------------------------
// Test: chatV6 through harness path
// ---------------------------------------------------------------------------

function makeSSEStream(text: string): ReadableStream {
	const encoder = new TextEncoder();
	const sseText = [
		`event: content_block_start\ndata: {"index":0,"content_block":{"type":"text"}}\n\n`,
		`event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":${JSON.stringify(text)}}}\n\n`,
		`event: content_block_stop\ndata: {"index":0}\n\n`,
		`event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n`,
		`event: message_stop\ndata: {}\n\n`,
	].join("");
	return new ReadableStream({
		start(controller) {
			controller.enqueue(encoder.encode(sseText));
			controller.close();
		},
	});
}

const stubCtx = {
	env: {} as never,
	db: {} as never,
} as unknown as ServiceContext;

describe("chatV6", () => {
	it("yields delta and done events through the harness path", async () => {
		const { callAnthropicStream } = await import("./llm");
		vi.mocked(callAnthropicStream).mockResolvedValue(makeSSEStream("Hi"));

		const events = [];
		for await (const e of chatV6(stubCtx, "student1", [{ role: "user", content: "hello" }], "")) {
			events.push(e);
		}

		expect(events.some((e) => e.type === "delta")).toBe(true);
		expect(events.at(-1)?.type).toBe("done");
	});
});
