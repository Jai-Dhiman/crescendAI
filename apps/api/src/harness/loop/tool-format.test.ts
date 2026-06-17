// apps/api/src/harness/loop/tool-format.test.ts
import { describe, expect, it } from "vitest";
import { toOpenAIChatRequest, toAnthropicResponse } from "./tool-format";

describe("toOpenAIChatRequest — tool definition mapping", () => {
  it("converts input_schema to function.parameters inside type:function wrapper", () => {
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "Hello" }],
      tools: [
        {
          name: "my_tool",
          description: "does something",
          input_schema: { type: "object", properties: { x: { type: "string" } } },
        },
      ],
      tool_choice: { type: "auto" } as const,
    };

    const out = toOpenAIChatRequest(req);

    expect(out.tools).toHaveLength(1);
    expect(out.tools[0]).toEqual({
      type: "function",
      function: {
        name: "my_tool",
        description: "does something",
        parameters: { type: "object", properties: { x: { type: "string" } } },
      },
    });
  });

  it("converts tool_choice {type:'auto'} to string 'auto'", () => {
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "Hello" }],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };
    const out = toOpenAIChatRequest(req);
    expect(out.tool_choice).toBe("auto");
  });

  it("converts tool_choice {type:'tool',name:'foo'} to {type:'function',function:{name:'foo'}}", () => {
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "Hello" }],
      tools: [],
      tool_choice: { type: "tool", name: "foo" } as const,
    };
    const out = toOpenAIChatRequest(req);
    expect(out.tool_choice).toEqual({ type: "function", function: { name: "foo" } });
  });
});

describe("toOpenAIChatRequest — message mapping", () => {
  it("maps a string user content to a single user message", () => {
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "Play Bach." }],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };
    const out = toOpenAIChatRequest(req);
    expect(out.messages).toHaveLength(1);
    expect(out.messages[0]).toEqual({ role: "user", content: "Play Bach." });
  });

  it("maps an assistant message with tool_use content blocks to tool_calls", () => {
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [
        {
          role: "assistant" as const,
          content: [
            {
              type: "tool_use" as const,
              id: "tu_1",
              name: "my_tool",
              input: { x: "val" },
            },
          ],
        },
      ],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };
    const out = toOpenAIChatRequest(req);
    expect(out.messages).toHaveLength(1);
    const msg = out.messages[0] as { role: string; tool_calls: unknown[] };
    expect(msg.role).toBe("assistant");
    expect(msg.tool_calls).toHaveLength(1);
    expect(msg.tool_calls[0]).toEqual({
      id: "tu_1",
      type: "function",
      function: { name: "my_tool", arguments: JSON.stringify({ x: "val" }) },
    });
  });

  it("maps tool_use blocks from a mixed [text, tool_use] assistant message; leading text is intentionally dropped", () => {
    // Documents the current contract: Qwen may emit a text preamble before its
    // tool_calls. We map the tool_use into tool_calls; the text preamble is NOT
    // carried into OpenAI history. Tool extraction is unaffected. If preserving
    // the text proves necessary for Phase 1 quality, that is a follow-up.
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [
        {
          role: "assistant" as const,
          content: [
            { type: "text" as const, text: "Let me check the signals." },
            {
              type: "tool_use" as const,
              id: "tu_1",
              name: "my_tool",
              input: { x: "val" },
            },
          ],
        },
      ],
      tools: [],
      tool_choice: { type: "auto" as const },
    };
    const out = toOpenAIChatRequest(req);
    expect(out.messages).toHaveLength(1);
    const msg = out.messages[0] as { role: string; content: string | null; tool_calls: unknown[] };
    expect(msg.role).toBe("assistant");
    expect(msg.tool_calls).toHaveLength(1);
    expect(msg.tool_calls[0]).toEqual({
      id: "tu_1",
      type: "function",
      function: { name: "my_tool", arguments: JSON.stringify({ x: "val" }) },
    });
    // The text preamble is dropped (content is null), not surfaced as a separate message.
    expect(msg.content).toBeNull();
  });

  it("maps user tool_result blocks to individual role:tool messages", () => {
    const req = {
      model: "@cf/qwen/qwen3-30b-a3b-fp8",
      max_tokens: 2048,
      messages: [
        {
          role: "user" as const,
          content: [
            {
              type: "tool_result" as const,
              tool_use_id: "tu_1",
              content: '{"score":0.8}',
            },
            {
              type: "tool_result" as const,
              tool_use_id: "tu_2",
              content: '{"score":0.6}',
            },
          ],
        },
      ],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };
    const out = toOpenAIChatRequest(req);
    expect(out.messages).toHaveLength(2);
    expect(out.messages[0]).toEqual({
      role: "tool",
      tool_call_id: "tu_1",
      content: '{"score":0.8}',
    });
    expect(out.messages[1]).toEqual({
      role: "tool",
      tool_call_id: "tu_2",
      content: '{"score":0.6}',
    });
  });
});

describe("toAnthropicResponse — OpenAI → Anthropic response mapping", () => {
  it("maps tool_calls to tool_use content blocks, JSON-parsing arguments into input", () => {
    const oaiRes = {
      choices: [
        {
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "write_synthesis_artifact",
                  arguments: '{"headline":"Great session","focus_areas":[]}',
                },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    };

    const out = toAnthropicResponse(oaiRes);

    expect(out.stop_reason).toBe("tool_use");
    expect(out.content).toHaveLength(1);
    expect(out.content[0]).toEqual({
      type: "tool_use",
      id: "call_1",
      name: "write_synthesis_artifact",
      input: { headline: "Great session", focus_areas: [] },
    });
  });

  it("maps text content to {type:'text'} blocks and derives stop_reason 'end_turn'", () => {
    const oaiRes = {
      choices: [
        {
          message: {
            role: "assistant",
            content: "Here is my response.",
            tool_calls: null,
          },
          finish_reason: "stop",
        },
      ],
    };

    const out = toAnthropicResponse(oaiRes);

    expect(out.stop_reason).toBe("end_turn");
    expect(out.content).toHaveLength(1);
    expect(out.content[0]).toEqual({ type: "text", text: "Here is my response." });
  });

  it("derives stop_reason 'tool_use' when any tool_calls are present", () => {
    const oaiRes = {
      choices: [
        {
          message: {
            role: "assistant",
            content: "Some text too",
            tool_calls: [
              {
                id: "c1",
                type: "function",
                function: { name: "foo", arguments: '{}' },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    };
    const out = toAnthropicResponse(oaiRes);
    expect(out.stop_reason).toBe("tool_use");
  });

  it("returns empty content array when message has no content and no tool_calls", () => {
    const oaiRes = {
      choices: [
        {
          message: {
            role: "assistant",
            content: null,
            tool_calls: null,
          },
          finish_reason: "stop",
        },
      ],
    };
    const out = toAnthropicResponse(oaiRes);
    expect(out.content).toEqual([]);
    expect(out.stop_reason).toBe("end_turn");
  });
});
