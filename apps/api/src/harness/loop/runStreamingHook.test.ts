import { vi, describe, expect, it } from "vitest";
import { ConfigError } from "../../lib/errors";
import { runStreamingHook } from "./runStreamingHook";
import type { HookContext } from "./types";

vi.mock("../../services/llm", async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return { ...actual, callAnthropicStream: vi.fn() };
});

import { callAnthropicStream } from "../../services/llm";

const stubHookCtx: HookContext = {
  env: {} as never,
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
