import { vi, describe, expect, it, beforeEach } from "vitest";
import { ConfigError } from "../../lib/errors";
import { runStreamingHook } from "./runStreamingHook";
import type { HookContext } from "./types";

vi.mock("../../services/llm", async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return { ...actual, callAnthropicStream: vi.fn(), callWorkersAIStream: vi.fn() };
});

import { callAnthropicStream, callWorkersAIStream } from "../../services/llm";

const stubHookCtx: HookContext = {
  env: { TEACHER_PROVIDER: "anthropic" } as never,
  studentId: "s1",
  sessionId: "",
  conversationId: null,
  digest: {},
  waitUntil: () => {},
};

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

describe("runStreamingHook error paths", () => {
  it("throws ConfigError when hook has no registered binding", async () => {
    const gen = runStreamingHook("OnStop", stubHookCtx, async () => ({} as never), [], []);
    await expect(gen.next()).rejects.toBeInstanceOf(ConfigError);
  });

  it("throws ConfigError when binding mode is not streaming", async () => {
    const gen = runStreamingHook("OnSessionEnd", stubHookCtx, async () => ({} as never), [], []);
    await expect(gen.next()).rejects.toBeInstanceOf(ConfigError);
  });
});

describe("runStreamingHook happy path", () => {
  it("yields delta and done events for OnChatMessage with a streaming response", async () => {
    vi.mocked(callAnthropicStream).mockResolvedValue(makeSSEStream("Hello"));

    const events: import("../../services/teacher").TeacherEvent[] = [];
    for await (const e of runStreamingHook(
      "OnChatMessage",
      stubHookCtx,
      async () => ({ name: "noop", componentsJson: [], isError: false }),
      [],
      [{ role: "user", content: "test" }],
    )) {
      events.push(e);
    }

    expect(events.some((e) => e.type === "delta" && e.text === "Hello")).toBe(true);
    expect(events.at(-1)?.type).toBe("done");
  });
});

describe("runStreamingHook — Workers AI path", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("uses callWorkersAIStream when env.TEACHER_PROVIDER is not 'anthropic'", async () => {
    const encoder = new TextEncoder();
    const sseText = [
      `data: {"choices":[{"delta":{"role":"assistant","content":"Ciao"},"finish_reason":null}]}\n\n`,
      `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n`,
      `data: [DONE]\n\n`,
    ].join("");
    const fakeStream = new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(sseText));
        controller.close();
      },
    });

    vi.mocked(callWorkersAIStream).mockResolvedValue(fakeStream);
    vi.mocked(callAnthropicStream).mockRejectedValue(new Error("should not be called"));

    const waiEnv = {} as never;

    const events: import("../../services/teacher").TeacherEvent[] = [];
    for await (const e of runStreamingHook(
      "OnChatMessage",
      { ...stubHookCtx, env: waiEnv },
      async () => ({ name: "noop", componentsJson: [], isError: false }),
      [],
      [{ role: "user", content: "Hello" }],
    )) {
      events.push(e);
    }

    expect(callWorkersAIStream).toHaveBeenCalledOnce();
    expect(callAnthropicStream).not.toHaveBeenCalled();
    expect(events.some((e) => e.type === "delta" && e.text === "Ciao")).toBe(true);
    expect(events.at(-1)?.type).toBe("done");
  });
});
