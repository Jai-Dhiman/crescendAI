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

describe("HARNESS_V6_CHAT_ENABLED Bindings type", () => {
	it("Bindings type includes HARNESS_V6_CHAT_ENABLED as a string field", () => {
		// Pick<Bindings, 'HARNESS_V6_CHAT_ENABLED'> fails to compile if the field is absent.
		const flag = { HARNESS_V6_CHAT_ENABLED: "false" } as Pick<
			Bindings,
			"HARNESS_V6_CHAT_ENABLED"
		>;
		expect(flag.HARNESS_V6_CHAT_ENABLED).toBe("false");
	});
});

