import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { Bindings, Db, ServiceContext } from "../lib/types";
import type { TeacherEvent } from "./teacher";
import { buildChatBinding, runPhase1Streaming } from "./teacher";
import { TOOL_REGISTRY } from "./tool-processor";

const MOCK_ENV = {
  AI_GATEWAY_TEACHER: "https://gw.example",
  ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const MOCK_CTX: ServiceContext = {
  db: {} as Db,
  env: MOCK_ENV,
};

function makeSseResponse(sseText: string): Response {
  return new Response(new TextEncoder().encode(sseText), { status: 200 });
}

const TOOL_USE_SSE = [
  'event: message_start\ndata: {"type":"message_start"}\n\n',
  'event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"search_catalog"}}\n\n',
  'event: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"composer\\":\\"Chopin\\"}"}}\n\n',
  'event: content_block_stop\ndata: {"index":0}\n\n',
  'event: message_delta\ndata: {"delta":{"stop_reason":"tool_use"}}\n\n',
  'event: message_stop\ndata: {}\n\n',
].join("");

const TEXT_ONLY_SSE = [
  'event: message_start\ndata: {"type":"message_start"}\n\n',
  'event: content_block_start\ndata: {"index":0,"content_block":{"type":"text"}}\n\n',
  'event: content_block_delta\ndata: {"index":0,"delta":{"type":"text_delta","text":"Hello world"}}\n\n',
  'event: message_delta\ndata: {"delta":{"stop_reason":"end_turn"}}\n\n',
  'event: message_stop\ndata: {}\n\n',
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

describe("buildChatBinding", () => {
  it("returns mode:'streaming' and phases:1", () => {
    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    expect(binding.mode).toBe("streaming");
    expect(binding.phases).toBe(1);
  });

  it("includes all TOOL_REGISTRY tools by name", () => {
    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    const expected = Object.keys(TOOL_REGISTRY).sort();
    const actual = binding.tools.map((t) => t.name).sort();
    expect(actual).toEqual(expected);
  });

  it("each tool has an object input_schema and an invoke function", () => {
    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    for (const tool of binding.tools) {
      expect(typeof tool.input_schema).toBe("object");
      expect(typeof tool.invoke).toBe("function");
    }
  });
});

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

    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    const systemBlocks = [{ type: "text" as const, text: "You are a teacher." }];
    const messages = [{ role: "user" as const, content: "What should I practice?" }];
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
    expect((deltas[0] as { type: "delta"; text: string }).text).toBe("Hello world");

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
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TOOL_USE_SSE)))
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)));

    const mockResult = {
      name: "search_catalog",
      componentsJson: [{ type: "search_catalog_result", config: { matches: [] } }],
      isError: false,
    };
    const processToolFn = vi.fn().mockResolvedValue(mockResult);

    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    const systemBlocks = [{ type: "text" as const, text: "You are a teacher." }];
    const messages = [{ role: "user" as const, content: "Find me some Chopin." }];

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
    expect(processToolFn).toHaveBeenCalledWith("search_catalog", { composer: "Chopin" });
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });
});
