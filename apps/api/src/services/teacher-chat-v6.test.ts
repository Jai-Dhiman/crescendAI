import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { Bindings } from "../lib/types";
import type { TeacherEvent } from "./teacher";
import { runPhase1Streaming } from "./teacher";
import { getCompoundBinding } from "../harness/loop/compound-registry";

const MOCK_ENV = {
	AI_GATEWAY_TEACHER: "https://gw.example",
	ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;


function makeSseResponse(sseText: string): Response {
	return new Response(new TextEncoder().encode(sseText), { status: 200 });
}

const TOOL_USE_SSE = [
	'event: message_start\ndata: {"type":"message_start"}\n\n',
	'event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"search_catalog"}}\n\n',
	'event: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"composer\\":\\"Chopin\\"}"}}\n\n',
	'event: content_block_stop\ndata: {"index":0}\n\n',
	'event: message_delta\ndata: {"delta":{"stop_reason":"tool_use"}}\n\n',
	"event: message_stop\ndata: {}\n\n",
].join("");

const TEXT_ONLY_SSE = [
	'event: message_start\ndata: {"type":"message_start"}\n\n',
	'event: content_block_start\ndata: {"index":0,"content_block":{"type":"text"}}\n\n',
	'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"Hello world"}}\n\n',
	'event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"}}\n\n',
	"event: message_stop\ndata: {}\n\n",
].join("");

const PHASE_CTX = {
	env: MOCK_ENV,
	studentId: "stu_1",
	sessionId: "",
	conversationId: null as null,
	digest: {} as Record<string, unknown>,
	waitUntil: (_p: Promise<unknown>) => {},
	turnCap: 5,
};


describe("runPhase1Streaming — text-only turn", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("yields delta events and done with fullText for a text-only response", async () => {
		fetchSpy.mockImplementationOnce(() =>
			Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)),
		);

		const binding = getCompoundBinding("OnChatMessage")!;
		const systemBlocks = [
			{ type: "text" as const, text: "You are a teacher." },
		];
		const messages = [
			{ role: "user" as const, content: "What should I practice?" },
		];
		const processToolFn = vi.fn();

		const events: TeacherEvent[] = [];
		for await (const ev of runPhase1Streaming(
			PHASE_CTX,
			binding,
			systemBlocks,
			messages,
			processToolFn,
		)) {
			events.push(ev);
		}

		const deltas = events.filter((e) => e.type === "delta");
		expect(deltas).toHaveLength(1);
		expect((deltas[0] as { type: "delta"; text: string }).text).toBe(
			"Hello world",
		);

		const done = events.find((e) => e.type === "done");
		expect(done).toBeDefined();
		if (done && done.type === "done") {
			expect(done.fullText).toBe("Hello world");
			expect(done.stopReason).toBe("end_turn");
			expect(done.allComponents).toEqual([]);
		}

		expect(fetchSpy).toHaveBeenCalledTimes(1);
		expect(processToolFn).not.toHaveBeenCalled();
	});
});

describe("runPhase1Streaming — tool continuation", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("dispatches tool call and continues to a second text turn, accumulating components", async () => {
		fetchSpy
			.mockImplementationOnce(() =>
				Promise.resolve(makeSseResponse(TOOL_USE_SSE)),
			)
			.mockImplementationOnce(() =>
				Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)),
			);

		const mockResult = {
			name: "search_catalog",
			componentsJson: [
				{ type: "search_catalog_result", config: { matches: [] } },
			],
			isError: false,
		};
		const processToolFn = vi.fn().mockResolvedValue(mockResult);

		const binding = getCompoundBinding("OnChatMessage")!;
		const systemBlocks = [
			{ type: "text" as const, text: "You are a teacher." },
		];
		const messages = [
			{ role: "user" as const, content: "Find me some Chopin." },
		];

		const events: TeacherEvent[] = [];
		for await (const ev of runPhase1Streaming(
			PHASE_CTX,
			binding,
			systemBlocks,
			messages,
			processToolFn,
		)) {
			events.push(ev);
		}

		const toolStart = events.find((e) => e.type === "tool_start");
		expect(toolStart).toBeDefined();
		if (toolStart && toolStart.type === "tool_start") {
			expect(toolStart.name).toBe("search_catalog");
		}

		const deltas = events.filter((e) => e.type === "delta");
		expect(deltas.length).toBeGreaterThan(0);

		const done = events.findLast((e) => e.type === "done");
		expect(done).toBeDefined();
		if (done && done.type === "done") {
			expect(done.fullText).toBe("Hello world");
			expect(done.allComponents).toEqual(mockResult.componentsJson);
		}

		expect(processToolFn).toHaveBeenCalledTimes(1);
		expect(processToolFn).toHaveBeenCalledWith("search_catalog", {
			composer: "Chopin",
		});
		expect(fetchSpy).toHaveBeenCalledTimes(2);
	});
});

const ASSIGN_LOOP_SSE = [
	'event: message_start\ndata: {"type":"message_start"}\n\n',
	'event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tu_loop","name":"assign_segment_loop"}}\n\n',
	'event: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"piece_id\\":\\"chopin.ballades.1\\",\\"bars_start\\":12,\\"bars_end\\":16,\\"required_correct\\":5}"}}\n\n',
	'event: content_block_stop\ndata: {"index":0}\n\n',
	'event: message_delta\ndata: {"delta":{"stop_reason":"tool_use"}}\n\n',
	"event: message_stop\ndata: {}\n\n",
].join("");

// Known gap: processToolFn is mocked here so this suite tests that runPhase1Streaming
// correctly routes assign_segment_loop through the processToolFn hook. The actual
// chatV6 intercept (assignSegmentLoopAtom) is not exercised — no test DB available.
// assignSegmentLoopAtom validation is covered by assign-segment-loop.test.ts.
describe("runPhase1Streaming — assign_segment_loop intercept", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("tool_result event carries segment_loop component when processToolFn intercepts assign_segment_loop", async () => {
		// Two fetch calls: tool_use turn + forced text continuation
		fetchSpy
			.mockImplementationOnce(() => Promise.resolve(makeSseResponse(ASSIGN_LOOP_SSE)))
			.mockImplementationOnce(() => Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)));

		const mockComponent = {
			type: "segment_loop",
			config: { id: "loop-test-1", pieceId: "chopin.ballades.1", status: "pending" },
		};

		// processToolFn mirrors the intercept logic chatV6 installs
		const processToolFn = vi.fn().mockImplementation(
			async (name: string, _input: unknown) => {
				if (name === "assign_segment_loop") {
					return { name, componentsJson: [mockComponent], isError: false };
				}
				return { name, componentsJson: [], isError: false };
			},
		);

		const binding = getCompoundBinding("OnChatMessage")!;
		const systemBlocks = [{ type: "text" as const, text: "You are a teacher." }];
		const messages = [{ role: "user" as const, content: "Practice bars 12-16." }];

		const events: TeacherEvent[] = [];
		for await (const ev of runPhase1Streaming(
			PHASE_CTX,
			binding,
			systemBlocks,
			messages,
			processToolFn,
		)) {
			events.push(ev);
		}

		// processToolFn called with the parsed tool input
		expect(processToolFn).toHaveBeenCalledWith("assign_segment_loop", {
			piece_id: "chopin.ballades.1",
			bars_start: 12,
			bars_end: 16,
			required_correct: 5,
		});

		// tool_result event is emitted with the segment_loop component
		const toolResult = events.find(
			(e): e is Extract<TeacherEvent, { type: "tool_result" }> =>
				e.type === "tool_result" && (e as Extract<TeacherEvent, { type: "tool_result" }>).name === "assign_segment_loop",
		);
		expect(toolResult).toBeDefined();
		expect(toolResult?.componentsJson).toHaveLength(1);
		expect(toolResult?.componentsJson[0]?.type).toBe("segment_loop");

		// component accumulates into the final done event
		const done = events.findLast((e) => e.type === "done");
		expect(done?.type).toBe("done");
		if (done?.type === "done") {
			expect(done.allComponents[0]?.type).toBe("segment_loop");
		}
	});
});

describe("runPhase1Streaming — turn cap exhaustion", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("issues a forced tool_choice:none call after turnCap and yields forced_text_after_max_turns", async () => {
		const capCtx = { ...PHASE_CTX, turnCap: 2 };
		const processToolFn = vi.fn().mockResolvedValue({
			name: "search_catalog",
			componentsJson: [],
			isError: false,
		});

		// Two tool_use turns exhaust the cap; the third call is the forced text call
		fetchSpy
			.mockImplementationOnce(() =>
				Promise.resolve(makeSseResponse(TOOL_USE_SSE)),
			)
			.mockImplementationOnce(() =>
				Promise.resolve(makeSseResponse(TOOL_USE_SSE)),
			)
			.mockImplementationOnce(() =>
				Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)),
			);

		const binding = getCompoundBinding("OnChatMessage")!;
		const systemBlocks = [
			{ type: "text" as const, text: "You are a teacher." },
		];
		const messages = [
			{ role: "user" as const, content: "Find me some Chopin." },
		];

		const events: TeacherEvent[] = [];
		for await (const ev of runPhase1Streaming(
			capCtx,
			binding,
			systemBlocks,
			messages,
			processToolFn,
		)) {
			events.push(ev);
		}

		// Three fetch calls: turnCap tool turns + 1 forced call
		expect(fetchSpy).toHaveBeenCalledTimes(3);

		// The forced call sends tool_choice: none
		const forcedCallBody = JSON.parse(
			fetchSpy.mock.calls[2][1].body as string,
		) as { tool_choice: { type: string } };
		expect(forcedCallBody.tool_choice).toEqual({ type: "none" });

		// Final done event carries the forced stopReason
		const done = events.findLast((e) => e.type === "done");
		expect(done).toBeDefined();
		if (done && done.type === "done") {
			expect(done.stopReason).toBe("forced_text_after_max_turns");
			expect(done.fullText).toBe("Hello world");
		}
	});
});

