# V6 Teacher on Workers AI (Qwen3-30B-A3B) Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Run V6 session-end teacher synthesis end-to-end locally with zero Anthropic credits by routing Phase 1 + Phase 2 to Workers AI `@cf/qwen/qwen3-30b-a3b-fp8` through the same authenticated AI Gateway.
**Spec:** docs/specs/2026-06-16-v6-teacher-workers-ai-design.md
**Style:** Follow `apps/api/TS_STYLE.md` for all code in `apps/api/`.

## Build-env prerequisites

Before running any test command:
```bash
cd /path/to/worktree/apps/api && bun install
```

Run tests with:
```bash
cd apps/api && bunx vitest run <files>
```
Never use `bun test` — it runs the native Bun runner, not Vitest, and produces false failures.

Type check with:
```bash
cd apps/api && bun run typecheck
```

Final manual verification (not a task — do after all tasks are committed): start `just dev` then run `CRESCEND_COOKIE=… uv run /tmp/crescend-e2e/drive_cookie.py --wav model/data/evals/practice_eval/nocturne_op9no2/audio/_aySCutsVVQ.wav --piece "nocturne op 9 no 2" --chunks 8` and confirm the WS emits a `synthesis` event with a `write_synthesis_artifact` payload that has a 300–500-char headline, `focus_areas`, and `prescribed_exercise`.

## Runtime config & challenge-risk notes (read before executing)

The unit tests stub `Bindings` with `as unknown as Bindings`, so they pass regardless of `wrangler.toml`/`.dev.vars`. The runtime vars are sourced at `just dev` time from `apps/api/.dev.vars` in the **primary checkout** (not committed, not in this worktree). The post-2026-06-10 unification already defines `AI_GATEWAY_ENDPOINT` + `AI_GATEWAY_TOKEN` there; `CLOUDFLARE_API_TOKEN` is also already present (used by the existing `callWorkersAI`). `TEACHER_PROVIDER` is intentionally **unset** locally → `routeModel` defaults to `workers-ai`, which is the goal. The `[vars]` block in `wrangler.toml` binds none of these (correct — they are secrets/.dev.vars, not plaintext vars). **No `wrangler.toml` edit is required for local runs.** If a future deploy is wanted, `TEACHER_PROVIDER` and the gateway secrets are set via `wrangler secret`, out of scope here.

Three risks from /challenge, addressed in-plan:
1. **`callAnthropicStream` auth change is untested (Task 7).** The streaming chat path has no unit test in this plan (SSE translation is out of scope). After Task 7, the build agent MUST run the live chat smoke (see Task 7 Step 4b) before declaring done — a misconfigured `AI_GATEWAY_TOKEN` would break chat silently otherwise.
2. **`wrangler.toml` runtime-var drift.** Mitigated by the config note above + the early grep check added to Task 4 Step 4b.
3. **`toOpenAIChatRequest` drops text blocks from mixed `[text, tool_use]` assistant messages.** Qwen may emit a text preamble before tool_calls in Phase 1. Tool extraction is unaffected (context history is merely incomplete). Task 1 adds an explicit test pinning the *current, intentional* behavior (tool_use blocks are mapped; any leading text is dropped) so the decision is documented, not accidental. If live Phase 1 quality is poor, preserving text is a follow-up — NOT a blocker for credit-free runs.

---

## Task Groups

- **Group A (parallel):** Task 1 (`tool-format.ts`), Task 2 (`gateway-client.ts`), Task 3 (`route-model.ts`), Task 4 (`types.ts` — additive only)
- **Group B (parallel, depends on A):** Task 5 (`phase1.ts`), Task 6 (`phase2.ts`), Task 7 (`llm.ts`)
- **Group C (sequential, depends on B):** Task 8 (dead-var cleanup in `types.ts` + all remaining test-mock updates)

---

## Task 1: tool-format — Anthropic request → OpenAI request

**Group:** A (parallel with Tasks 2, 3, 4)

**Behavior being verified:** `toOpenAIChatRequest` converts an Anthropic Messages API request body (tools, tool_choice, messages with tool_use/tool_result content blocks) into the OpenAI `chat/completions` shape that Workers AI accepts. `toAnthropicResponse` converts an OpenAI `chat/completions` response back to the Anthropic `{content, stop_reason}` shape, JSON-parsing `function.arguments` into `tool_use.input`.

**Interface under test:** `toOpenAIChatRequest(req)` and `toAnthropicResponse(res)` — both exported from `apps/api/src/harness/loop/tool-format.ts`.

**Files:**
- Create: `apps/api/src/harness/loop/tool-format.ts`
- Create: `apps/api/src/harness/loop/tool-format.test.ts`

---

- [ ] **Step 1: Write the failing test**

```typescript
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run src/harness/loop/tool-format.test.ts
```
Expected: FAIL — `Cannot find module './tool-format'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/loop/tool-format.ts

export interface AnthropicToolDef {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface AnthropicToolChoiceAuto {
  type: "auto";
}

export interface AnthropicToolChoiceTool {
  type: "tool";
  name: string;
}

export type AnthropicToolChoice = AnthropicToolChoiceAuto | AnthropicToolChoiceTool;

export interface AnthropicToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
}

export interface AnthropicTextBlock {
  type: "text";
  text: string;
}

export interface AnthropicToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}

export type AnthropicContentBlock =
  | AnthropicToolUseBlock
  | AnthropicTextBlock
  | AnthropicToolResultBlock;

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

export interface AnthropicChatRequest {
  model: string;
  max_tokens: number;
  messages: AnthropicMessage[];
  tools?: AnthropicToolDef[];
  tool_choice?: AnthropicToolChoice;
}

export interface AnthropicMessageResponse {
  content: Array<AnthropicTextBlock | AnthropicToolUseBlock>;
  stop_reason: string;
}

// ---------------------------------------------------------------------------
// toOpenAIChatRequest
// ---------------------------------------------------------------------------

interface OpenAIToolDef {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

type OpenAIToolChoice =
  | "auto"
  | "none"
  | { type: "function"; function: { name: string } };

interface OpenAIToolCallFunction {
  name: string;
  arguments: string;
}

interface OpenAIToolCall {
  id: string;
  type: "function";
  function: OpenAIToolCallFunction;
}

type OpenAIMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: OpenAIToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };

interface OpenAIChatRequest {
  model: string;
  max_tokens: number;
  messages: OpenAIMessage[];
  tools: OpenAIToolDef[];
  tool_choice: OpenAIToolChoice;
}

export function toOpenAIChatRequest(req: AnthropicChatRequest): OpenAIChatRequest {
  const tools: OpenAIToolDef[] = (req.tools ?? []).map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
    },
  }));

  let tool_choice: OpenAIToolChoice = "auto";
  if (req.tool_choice) {
    if (req.tool_choice.type === "auto") {
      tool_choice = "auto";
    } else if (req.tool_choice.type === "tool") {
      tool_choice = { type: "function", function: { name: (req.tool_choice as AnthropicToolChoiceTool).name } };
    }
  }

  const messages: OpenAIMessage[] = [];
  for (const msg of req.messages) {
    if (typeof msg.content === "string") {
      messages.push({ role: msg.role as "user" | "assistant", content: msg.content });
    } else if (Array.isArray(msg.content)) {
      if (msg.role === "assistant") {
        const toolCalls: OpenAIToolCall[] = [];
        for (const block of msg.content) {
          if (block.type === "tool_use") {
            toolCalls.push({
              id: block.id,
              type: "function",
              function: {
                name: block.name,
                arguments: JSON.stringify(block.input),
              },
            });
          }
        }
        messages.push({ role: "assistant", content: null, tool_calls: toolCalls });
      } else if (msg.role === "user") {
        for (const block of msg.content) {
          if (block.type === "tool_result") {
            messages.push({
              role: "tool",
              tool_call_id: block.tool_use_id,
              content: block.content,
            });
          }
        }
      }
    }
  }

  return { model: req.model, max_tokens: req.max_tokens, messages, tools, tool_choice };
}

// ---------------------------------------------------------------------------
// toAnthropicResponse
// ---------------------------------------------------------------------------

interface OpenAIChatResponse {
  choices: Array<{
    message: {
      role: string;
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
      }> | null;
    };
    finish_reason: string;
  }>;
}

export function toAnthropicResponse(res: OpenAIChatResponse): AnthropicMessageResponse {
  const choice = res.choices[0];
  if (!choice) {
    return { content: [], stop_reason: "end_turn" };
  }

  const { message } = choice;
  const content: Array<AnthropicTextBlock | AnthropicToolUseBlock> = [];

  if (message.tool_calls && message.tool_calls.length > 0) {
    for (const tc of message.tool_calls) {
      content.push({
        type: "tool_use",
        id: tc.id,
        name: tc.function.name,
        input: JSON.parse(tc.function.arguments),
      });
    }
    return { content, stop_reason: "tool_use" };
  }

  if (message.content) {
    content.push({ type: "text", text: message.content });
  }

  return { content, stop_reason: "end_turn" };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run src/harness/loop/tool-format.test.ts
```
Expected: PASS (all tests green)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/tool-format.ts apps/api/src/harness/loop/tool-format.test.ts && git commit -m "feat(harness): add tool-format pure translation functions (Anthropic↔OpenAI)"
```

---

## Task 2: gateway-client — callModel provider routing + auth

**Group:** A (parallel with Tasks 1, 3, 4)

**Behavior being verified:** `callModel` routes to the correct gateway URL and sets the correct auth headers for both providers, returns an Anthropic-shaped response, and throws `InferenceError` on non-2xx.

**Interface under test:** `callModel(env, client, body)` exported from `apps/api/src/harness/loop/gateway-client.ts`.

**Files:**
- Create: `apps/api/src/harness/loop/gateway-client.ts`
- Create: `apps/api/src/harness/loop/gateway-client.test.ts`

---

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/loop/gateway-client.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bindings } from "../../lib/types";
import { callModel } from "./gateway-client";
import { InferenceError } from "../../lib/errors";

const BASE_ENV = {
  AI_GATEWAY_ENDPOINT: "https://gw.example.com",
  AI_GATEWAY_TOKEN: "gw-token-abc",
  CLOUDFLARE_API_TOKEN: "cf-token-xyz",
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;

const WORKERS_AI_ENV = {
  ...BASE_ENV,
  TEACHER_PROVIDER: "workers-ai",
} as unknown as Bindings;

const ANTHROPIC_RESPONSE = {
  content: [{ type: "text", text: "Hello from Anthropic" }],
  stop_reason: "end_turn",
  usage: { input_tokens: 10, output_tokens: 5 },
};

const WORKERS_AI_OPENAI_RESPONSE = {
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
              arguments: '{"headline":"Good session","focus_areas":[]}',
            },
          },
        ],
      },
      finish_reason: "tool_calls",
    },
  ],
};

describe("callModel — anthropic provider", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to /anthropic/v1/messages with cf-aig-authorization and anthropic-version headers", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify(ANTHROPIC_RESPONSE), { status: 200 }),
    );

    const client = { provider: "anthropic" as const, model: "claude-sonnet-4-20250514" };
    const body = {
      model: client.model,
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "test" }],
      tools: [],
      tool_choice: { type: "auto" as const },
    };

    const result = await callModel(BASE_ENV, client, body);

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://gw.example.com/anthropic/v1/messages");
    const headers = init.headers as Record<string, string>;
    expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token-abc");
    expect(headers["anthropic-version"]).toBe("2023-06-01");
    expect(headers["Content-Type"]).toBe("application/json");
    expect(result.content).toHaveLength(1);
    expect(result.stop_reason).toBe("end_turn");
  });

  it("throws InferenceError on non-2xx from anthropic path", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("credit balance too low", { status: 400 }),
    );

    const client = { provider: "anthropic" as const, model: "claude-sonnet-4-20250514" };
    const body = {
      model: client.model,
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "test" }],
      tools: [],
      tool_choice: { type: "auto" as const },
    };

    await expect(callModel(BASE_ENV, client, body)).rejects.toThrow(InferenceError);
  });
});

describe("callModel — workers-ai provider", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("POSTs to /workers-ai/v1/chat/completions with cf-aig-authorization + Authorization Bearer headers", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify(WORKERS_AI_OPENAI_RESPONSE), { status: 200 }),
    );

    const client = { provider: "workers-ai" as const, model: "@cf/qwen/qwen3-30b-a3b-fp8" };
    const body = {
      model: client.model,
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "test" }],
      tools: [
        {
          name: "write_synthesis_artifact",
          description: "Write artifact",
          input_schema: { type: "object" },
        },
      ],
      tool_choice: { type: "tool" as const, name: "write_synthesis_artifact" },
    };

    const result = await callModel(WORKERS_AI_ENV, client, body);

    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://gw.example.com/workers-ai/v1/chat/completions");
    const headers = init.headers as Record<string, string>;
    expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token-abc");
    expect(headers["Authorization"]).toBe("Bearer cf-token-xyz");
    expect(headers["Content-Type"]).toBe("application/json");
    // Response is translated back to Anthropic shape
    expect(result.stop_reason).toBe("tool_use");
    expect(result.content).toHaveLength(1);
    const block = result.content[0] as { type: string; name: string; input: unknown };
    expect(block.type).toBe("tool_use");
    expect(block.name).toBe("write_synthesis_artifact");
    expect(block.input).toEqual({ headline: "Good session", focus_areas: [] });
  });

  it("throws InferenceError on non-2xx from workers-ai path", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("Unauthorized", { status: 401 }),
    );

    const client = { provider: "workers-ai" as const, model: "@cf/qwen/qwen3-30b-a3b-fp8" };
    const body = {
      model: client.model,
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "test" }],
      tools: [],
      tool_choice: { type: "auto" as const },
    };

    await expect(callModel(WORKERS_AI_ENV, client, body)).rejects.toThrow(InferenceError);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run src/harness/loop/gateway-client.test.ts
```
Expected: FAIL — `Cannot find module './gateway-client'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/loop/gateway-client.ts
import { InferenceError } from "../../lib/errors";
import type { Bindings } from "../../lib/types";
import type { AnthropicChatRequest, AnthropicMessageResponse } from "./tool-format";
import { toOpenAIChatRequest, toAnthropicResponse } from "./tool-format";

export interface ModelClient {
  provider: "anthropic" | "workers-ai";
  model: string;
}

export async function callModel(
  env: Bindings,
  client: ModelClient,
  body: AnthropicChatRequest,
): Promise<AnthropicMessageResponse> {
  if (client.provider === "anthropic") {
    const url = `${env.AI_GATEWAY_ENDPOINT}/anthropic/v1/messages`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      throw new InferenceError(
        `callModel anthropic failed: ${res.status} ${await res.text()}`,
      );
    }
    return (await res.json()) as AnthropicMessageResponse;
  }

  // workers-ai: translate request, call, translate response
  const url = `${env.AI_GATEWAY_ENDPOINT}/workers-ai/v1/chat/completions`;
  const oaiBody = toOpenAIChatRequest(body);
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
      Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
    },
    body: JSON.stringify(oaiBody),
  });
  if (!res.ok) {
    throw new InferenceError(
      `callModel workers-ai failed: ${res.status} ${await res.text()}`,
    );
  }
  const oaiRes = await res.json();
  return toAnthropicResponse(oaiRes as Parameters<typeof toAnthropicResponse>[0]);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run src/harness/loop/gateway-client.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/gateway-client.ts apps/api/src/harness/loop/gateway-client.test.ts && git commit -m "feat(harness): add gateway-client callModel with provider routing and tool-format delegation"
```

---

## Task 3: route-model — return {provider, model} with env toggle

**Group:** A (parallel with Tasks 1, 2, 4)

**Behavior being verified:** `routeModel(kind, env)` returns `{provider:"workers-ai", model:"@cf/qwen/qwen3-30b-a3b-fp8"}` by default (when `env.TEACHER_PROVIDER` is undefined or `"workers-ai"`), and returns `{provider:"anthropic", model:"claude-sonnet-4-20250514"}` when `env.TEACHER_PROVIDER === "anthropic"`.

**Interface under test:** `routeModel(kind, env)` exported from `apps/api/src/harness/loop/route-model.ts`.

**Files:**
- Modify: `apps/api/src/harness/loop/route-model.ts`
- Modify: `apps/api/src/harness/loop/route-model.test.ts`

---

- [ ] **Step 1: Write the failing test**

Replace the content of `apps/api/src/harness/loop/route-model.test.ts`:

```typescript
// apps/api/src/harness/loop/route-model.test.ts
import { describe, expect, it } from "vitest";
import type { Bindings } from "../../lib/types";
import { routeModel } from "./route-model";

const WORKERS_AI_ENV = {
  TEACHER_PROVIDER: "workers-ai",
} as unknown as Bindings;

const ANTHROPIC_ENV = {
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;

const NO_PROVIDER_ENV = {} as unknown as Bindings;

describe("routeModel — workers-ai provider (default)", () => {
  it("returns workers-ai provider and Qwen model when TEACHER_PROVIDER=workers-ai", () => {
    const client = routeModel("phase1_analysis", WORKERS_AI_ENV);
    expect(client.provider).toBe("workers-ai");
    expect(client.model).toBe("@cf/qwen/qwen3-30b-a3b-fp8");
  });

  it("defaults to workers-ai when TEACHER_PROVIDER is not set", () => {
    const client = routeModel("phase2_voice", NO_PROVIDER_ENV);
    expect(client.provider).toBe("workers-ai");
    expect(client.model).toBe("@cf/qwen/qwen3-30b-a3b-fp8");
  });
});

describe("routeModel — anthropic provider (toggle)", () => {
  it("returns anthropic provider and Sonnet model when TEACHER_PROVIDER=anthropic", () => {
    const client = routeModel("phase1_analysis", ANTHROPIC_ENV);
    expect(client.provider).toBe("anthropic");
    expect(client.model).toBe("claude-sonnet-4-20250514");
  });

  it("returns anthropic provider for phase2_voice when TEACHER_PROVIDER=anthropic", () => {
    const client = routeModel("phase2_voice", ANTHROPIC_ENV);
    expect(client.provider).toBe("anthropic");
    expect(client.model).toBe("claude-sonnet-4-20250514");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run src/harness/loop/route-model.test.ts
```
Expected: FAIL — tests assert `client.provider` and `client.model` but existing `routeModel` returns `GatewayClient` with `gatewayUrlVar`, not `provider`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/api/src/harness/loop/route-model.ts`:

```typescript
// apps/api/src/harness/loop/route-model.ts
import type { Bindings } from "../../lib/types";

export type TaskKind = "phase1_analysis" | "phase2_voice";

export interface ModelClient {
  provider: "anthropic" | "workers-ai";
  model: string;
}

const WORKERS_AI_CLIENT: ModelClient = {
  provider: "workers-ai",
  model: "@cf/qwen/qwen3-30b-a3b-fp8",
};

const ANTHROPIC_CLIENT: ModelClient = {
  provider: "anthropic",
  model: "claude-sonnet-4-20250514",
};

export function routeModel(_kind: TaskKind, env: Bindings): ModelClient {
  if (env.TEACHER_PROVIDER === "anthropic") {
    return ANTHROPIC_CLIENT;
  }
  return WORKERS_AI_CLIENT;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run src/harness/loop/route-model.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/route-model.ts apps/api/src/harness/loop/route-model.test.ts && git commit -m "feat(harness): route-model returns {provider,model} with TEACHER_PROVIDER env toggle"
```

---

## Task 4: types.ts — add new Bindings fields (additive only)

**Group:** A (parallel with Tasks 1, 2, 3)

**Behavior being verified:** `Bindings` in `lib/types.ts` declares `AI_GATEWAY_ENDPOINT`, `AI_GATEWAY_TOKEN`, and `TEACHER_PROVIDER` so the codebase compiles after Tasks 1-3 are merged. The old fields (`AI_GATEWAY_TEACHER`, `AI_GATEWAY_BACKGROUND`, `ANTHROPIC_API_KEY`) are left in place until Task 8 removes them (keeping each intermediate commit compilable).

**Interface under test:** TypeScript type check passes — `bun run typecheck` exits 0.

**Files:**
- Modify: `apps/api/src/lib/types.ts`

---

- [ ] **Step 1: Write the failing test**

The "test" here is a typecheck. After adding the new fields to Bindings, `callModel` (which reads `env.AI_GATEWAY_ENDPOINT` and `env.AI_GATEWAY_TOKEN`) and `routeModel` (which reads `env.TEACHER_PROVIDER`) must compile without errors.

First, verify the current typecheck state (it may already pass or fail — we need to know the baseline):

```bash
cd apps/api && bun run typecheck 2>&1 | head -30
```

Then add the three fields. After adding them (but before removing old ones), typecheck must still pass.

- [ ] **Step 2: Run typecheck — note current state**

```bash
cd apps/api && bun run typecheck
```

Note the output. The step in the build agent's job is: after editing types.ts, run typecheck and confirm it still passes (no new errors introduced by the additive change).

- [ ] **Step 3: Add the three new fields to Bindings**

Edit `apps/api/src/lib/types.ts` — add the three new fields immediately after `ANTHROPIC_API_KEY`:

```typescript
export interface Bindings {
  HYPERDRIVE: Hyperdrive;
  CHUNKS: R2Bucket;
  SCORES: R2Bucket;
  ENVIRONMENT: string;
  ALLOWED_ORIGIN: string;
  APPLE_BUNDLE_ID: string;
  APPLE_WEB_SERVICES_ID: string;
  GOOGLE_CLIENT_ID: string;
  BETTER_AUTH_URL: string;
  AUTH_SECRET: string;
  APPLE_CLIENT_SECRET: string;
  GOOGLE_CLIENT_SECRET: string;
  SENTRY_DSN: string;
  HF_INFERENCE_ENDPOINT: string;
  ANTHROPIC_API_KEY: string;
  AI_GATEWAY_TEACHER: string;
  AI_GATEWAY_BACKGROUND: string;
  AI_GATEWAY_ENDPOINT: string;
  AI_GATEWAY_TOKEN: string;
  TEACHER_PROVIDER?: string;
  CLOUDFLARE_API_TOKEN: string;
  MUQ_ENDPOINT: string;
  AMT_ENDPOINT: string;
  SESSION_BRAIN: DurableObjectNamespace;
  HARNESS_V6_ENABLED: string;
  ALLOW_EVAL_STUDENT_OVERRIDE: string;
  EVAL_SHARED_SECRET: string;
}
```

- [ ] **Step 4: Run typecheck — verify it PASSES**

```bash
cd apps/api && bun run typecheck
```
Expected: exits 0 (no new errors)

- [ ] **Step 4b: Confirm runtime-var sourcing (challenge RISK #2)**

The new vars are NOT plaintext `[vars]` in `wrangler.toml` — they come from `.dev.vars` (local) / `wrangler secret` (prod). Confirm `wrangler.toml`'s `[vars]` block does NOT redundantly bind the old gateway vars (it should not bind any gateway var):

```bash
grep -nE "AI_GATEWAY_TEACHER|AI_GATEWAY_BACKGROUND|AI_GATEWAY_ENDPOINT|AI_GATEWAY_TOKEN|TEACHER_PROVIDER" apps/api/wrangler.toml
```
Expected: no output (gateway config is sourced from `.dev.vars`/secrets, not `wrangler.toml`). If any old var IS bound there, leave it — `wrangler.toml` is owned by the deploy flow, not this issue — but note it for the final manual verify.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/lib/types.ts && git commit -m "feat(types): add AI_GATEWAY_ENDPOINT, AI_GATEWAY_TOKEN, TEACHER_PROVIDER to Bindings"
```

---

## Task 5: phase1.ts — replace local callAnthropicMessage with callModel

**Group:** B (depends on Group A — needs route-model.ts and gateway-client.ts)

**Behavior being verified:** `runPhase1` sends requests through `callModel` instead of the local `callAnthropicMessage`. The existing phase1 tests pass unchanged (they mock global `fetch` and the URL that `callModel` hits is the same fetch boundary). The `MOCK_BINDINGS` in the test must be updated to use `AI_GATEWAY_ENDPOINT`/`AI_GATEWAY_TOKEN` instead of `AI_GATEWAY_TEACHER`/`ANTHROPIC_API_KEY`.

**Interface under test:** `runPhase1(ctx, binding)` exported from `apps/api/src/harness/loop/phase1.ts`, exercised via the existing `phase1.test.ts` test suite.

**Files:**
- Modify: `apps/api/src/harness/loop/phase1.ts`
- Modify: `apps/api/src/harness/loop/phase1.test.ts`

---

- [ ] **Step 1: Write the failing test**

Update the mock bindings and type imports in `phase1.test.ts` — change `AI_GATEWAY_TEACHER`/`ANTHROPIC_API_KEY` to `AI_GATEWAY_ENDPOINT`/`AI_GATEWAY_TOKEN`, and add `TEACHER_PROVIDER: "anthropic"` so that after `routeModel` returns `{provider:"anthropic"}`, `callModel` routes to the anthropic path and hits the mocked fetch at `https://gw.example/anthropic/v1/messages`.

Replace the `MOCK_BINDINGS` constant at the top of `apps/api/src/harness/loop/phase1.test.ts`:

```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_ENDPOINT: "https://gw.example",
  AI_GATEWAY_TOKEN: "test-gw-token",
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;
```

Keep all test bodies identical — they already mock global `fetch` and assert on behavior (`fetchSpy` call count, event types), not on headers. The URL changes from `undefined/anthropic/v1/messages` to `https://gw.example/anthropic/v1/messages` which is what the mock already intercepts.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run src/harness/loop/phase1.test.ts
```
Expected: FAIL — the existing implementation still calls the local `callAnthropicMessage` which reads `env[client.gatewayUrlVar]` (now undefined for the new env keys) and `env.ANTHROPIC_API_KEY` (also undefined).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/api/src/harness/loop/phase1.ts`:

```typescript
import { withRetries, wrapToolCall } from "./middleware";
import { callModel } from "./gateway-client";
import { routeModel } from "./route-model";
import type {
  CompoundBinding,
  PhaseContext,
  Phase1Event,
  ToolDefinition,
} from "./types";

function buildPhase1Tools(tools: ToolDefinition[]): unknown[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema,
  }));
}

export async function* runPhase1(
  ctx: PhaseContext,
  binding: CompoundBinding,
): AsyncGenerator<Phase1Event> {
  const messages: Array<{
    role: "user" | "assistant";
    content:
      | string
      | Array<
          | { type: "tool_use"; id: string; name: string; input: unknown }
          | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }
        >;
  }> = [
    {
      role: "user",
      content:
        `Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` +
        binding.procedurePrompt,
    },
  ];
  const toolMap = new Map(binding.tools.map((t) => [t.name, t]));
  const client = routeModel("phase1_analysis", ctx.env);
  let toolCallCount = 0;
  let turnCount = 0;

  while (turnCount < ctx.turnCap) {
    turnCount++;
    const response = await withRetries(() =>
      callModel(ctx.env, client, {
        model: client.model,
        max_tokens: 2048,
        messages,
        tools: buildPhase1Tools(binding.tools) as Array<{
          name: string;
          description: string;
          input_schema: Record<string, unknown>;
        }>,
        tool_choice: { type: "auto" },
      }),
    );

    const toolUses = response.content.filter(
      (
        b,
      ): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
        b.type === "tool_use",
    );

    if (toolUses.length === 0) {
      yield { type: "phase1_done", toolCallCount, turnCount };
      return;
    }

    messages.push({ role: "assistant", content: response.content });
    const toolResults: Array<{
      type: "tool_result";
      tool_use_id: string;
      content: string;
      is_error?: boolean;
    }> = [];

    for (const tu of toolUses) {
      toolCallCount++;
      yield {
        type: "phase1_tool_call",
        id: tu.id,
        tool: tu.name,
        input: tu.input,
      };
      const def = toolMap.get(tu.name);
      if (!def) {
        const error = `unknown tool: ${tu.name}`;
        yield {
          type: "phase1_tool_result",
          id: tu.id,
          tool: tu.name,
          ok: false,
          error,
        };
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: error,
          is_error: true,
        });
        continue;
      }
      try {
        const output = await wrapToolCall(tu.name, ctx, () =>
          def.invoke(tu.input, ctx),
        );
        yield {
          type: "phase1_tool_result",
          id: tu.id,
          tool: tu.name,
          ok: true,
          output,
        };
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: JSON.stringify(output),
        });
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        yield {
          type: "phase1_tool_result",
          id: tu.id,
          tool: tu.name,
          ok: false,
          error: message,
        };
        toolResults.push({
          type: "tool_result",
          tool_use_id: tu.id,
          content: message,
          is_error: true,
        });
      }
    }

    messages.push({ role: "user", content: toolResults });
  }

  yield { type: "phase1_done", toolCallCount, turnCount };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run src/harness/loop/phase1.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/phase1.ts apps/api/src/harness/loop/phase1.test.ts && git commit -m "feat(harness): phase1 delegates to callModel, removes local callAnthropicMessage"
```

---

## Task 6: phase2.ts — replace local callAnthropicMessage with callModel

**Group:** B (parallel with Task 5 and Task 7 — different files)

**Behavior being verified:** `runPhase2` sends requests through `callModel`. All existing phase2 tests pass with updated mock bindings. The 3-attempt Zod repair loop and forced `tool_choice` logic are byte-for-byte unchanged.

**Interface under test:** `runPhase2(ctx, binding, diagnoses)` exported from `apps/api/src/harness/loop/phase2.ts`, exercised via the existing `phase2.test.ts` test suite.

**Files:**
- Modify: `apps/api/src/harness/loop/phase2.ts`
- Modify: `apps/api/src/harness/loop/phase2.test.ts`

---

- [ ] **Step 1: Write the failing test**

Update `MOCK_BINDINGS` at the top of `apps/api/src/harness/loop/phase2.test.ts`:

```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_ENDPOINT: "https://gw.example",
  AI_GATEWAY_TOKEN: "test-gw-token",
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;
```

Keep all test bodies identical — they mock global `fetch`, assert on event types, and inspect the serialized request body `callBody.tool_choice`. This is unaffected by the URL change; `callModel` for the anthropic provider passes the body through verbatim.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run src/harness/loop/phase2.test.ts
```
Expected: FAIL — `env[client.gatewayUrlVar]` is undefined (the old `gatewayUrlVar` field no longer exists on `ModelClient`), producing `undefined/anthropic/v1/messages` URL which does not match the mock's expectation for `res.ok`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `apps/api/src/harness/loop/phase2.ts`:

```typescript
import { zodToJsonSchema } from "zod-to-json-schema";
import { FIRST_SESSION_GUARDRAIL } from "../../services/prompts";
import { withRetries } from "./middleware";
import { callModel } from "./gateway-client";
import { routeModel } from "./route-model";
import type { HookEvent, Phase2Binding, PhaseContext } from "./types";

export function buildPhase2Prompt(
  digest: Record<string, unknown>,
  diagnoses: unknown[],
  guardrail: string,
): string {
  const reflectionInstruction =
    "Headline instructions: write a light reflection in 2-4 sentences about what happened " +
    "in this session, ending in exactly one directional question about the dominant_dimension " +
    "(e.g. 'Want a drill targeting that?'). The headline must be 300-500 characters total. " +
    "Do not list all dimensions; focus on the one area that matters most.\n\n";

  const exerciseInstruction =
    "Exercise instructions: set prescribed_exercise to a single routing decision that targets " +
    "the dominant_dimension. Use kind='own_passage_loop' when the student has been identified " +
    "playing a specific piece and you want them to loop a bar range from it; use " +
    "kind='corpus_drill' when no piece is identified or a general technique drill would be " +
    "more appropriate. Set prescribed_exercise to null if no exercise is warranted. " +
    "Do NOT put a pieceId in prescribed_exercise — that is bound at the serving layer.\n\n";

  return (
    `Session digest:\n${JSON.stringify(digest, null, 2)}\n\n` +
    `Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
    guardrail +
    reflectionInstruction +
    exerciseInstruction +
    `Write the SynthesisArtifact now using the write_synthesis_artifact tool.`
  );
}

// Initial forced-tool call + up to (MAX_PHASE2_ATTEMPTS - 1) validation-repair
// turns. On sparse cold-start sessions the model reliably undershoots the
// headline 300-char minimum; a single forced call with no repair path makes any
// validation miss fatal (the historical "strict validation no-ops on sparse
// state" no-op). Showing the model its rejected artifact + the zod error lets it
// fix the flagged fields while preserving the 300-500 char product contract.
const MAX_PHASE2_ATTEMPTS = 3;

export async function* runPhase2(
  ctx: PhaseContext,
  binding: Phase2Binding,
  diagnoses: unknown[],
): AsyncGenerator<HookEvent<unknown>> {
  yield { type: "phase2_started" };

  const writeTool = {
    name: binding.artifactToolName,
    description:
      "Write the final compound artifact. Call this exactly once with the structured fields.",
    input_schema: artifactInputSchema(binding.artifactSchema),
  };

  const guardrail =
    ctx.digest.reference_mode === "within_session"
      ? `${FIRST_SESSION_GUARDRAIL}\n\n`
      : "";

  const userPrompt = buildPhase2Prompt(ctx.digest, diagnoses, guardrail);
  const client = routeModel("phase2_voice", ctx.env);
  const messages: Array<{ role: "user" | "assistant"; content: unknown }> = [
    { role: "user", content: userPrompt },
  ];

  let lastInvalid: { raw: unknown; zodError: string } | null = null;

  for (let attempt = 1; attempt <= MAX_PHASE2_ATTEMPTS; attempt++) {
    const response = await withRetries(() =>
      callModel(ctx.env, client, {
        model: client.model,
        max_tokens: 2048,
        messages: messages as Parameters<typeof callModel>[2]["messages"],
        tools: [writeTool],
        tool_choice: { type: "tool", name: binding.artifactToolName },
      }),
    );

    const toolUse = response.content.find(
      (
        b,
      ): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
        b.type === "tool_use" && b.name === binding.artifactToolName,
    );

    if (!toolUse) {
      yield {
        type: "phase_error",
        phase: 2,
        error: "no tool_use returned despite forced tool_choice",
      };
      return;
    }

    const parsed = binding.artifactSchema.safeParse(toolUse.input);
    if (parsed.success) {
      yield { type: "artifact", value: parsed.data };
      return;
    }

    lastInvalid = { raw: toolUse.input, zodError: parsed.error.message };

    if (attempt < MAX_PHASE2_ATTEMPTS) {
      // A user turn following an assistant tool_use must carry a tool_result
      // for that tool_use_id (Anthropic API contract), or the call 400s.
      messages.push({ role: "assistant", content: response.content });
      messages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: toolUse.id,
            is_error: true,
            content:
              `The artifact you wrote failed validation:\n${parsed.error.message}\n\n` +
              `Fix ONLY the flagged fields and call ${binding.artifactToolName} again. ` +
              `In particular, the headline must be between 300 and 500 characters — ` +
              `expand the reflection with concrete detail from the digest if it is too short.`,
          },
        ],
      });
    }
  }

  yield {
    type: "validation_error",
    raw: lastInvalid?.raw,
    zodError: lastInvalid?.zodError ?? "unknown validation failure",
  };
}

function artifactInputSchema(
  schema: Phase2Binding["artifactSchema"],
): Record<string, unknown> {
  // The Anthropic Messages API requires tool input_schema to be valid JSON
  // Schema draft 2020-12. The `openApi3` target emits OpenAPI-3.0 constructs
  // (nullable: true, boolean exclusiveMinimum) and `$ref`-dedups repeated
  // subschemas — all rejected with HTTP 400. The default (draft-07) target
  // with refs inlined produces a self-contained schema the API accepts.
  return zodToJsonSchema(schema, { $refStrategy: "none" }) as Record<
    string,
    unknown
  >;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run src/harness/loop/phase2.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/phase2.ts apps/api/src/harness/loop/phase2.test.ts && git commit -m "feat(harness): phase2 delegates to callModel, removes local callAnthropicMessage"
```

---

## Task 7: llm.ts — fix gateway var/auth drift for callWorkersAI and callAnthropic

**Group:** B (parallel with Tasks 5 and 6 — different files)

**Behavior being verified:** `callWorkersAI` uses `${env.AI_GATEWAY_ENDPOINT}/workers-ai/...` and sends `cf-aig-authorization: Bearer ${env.AI_GATEWAY_TOKEN}` (in addition to the existing `Authorization: Bearer ${env.CLOUDFLARE_API_TOKEN}`). `callAnthropic` and `callAnthropicStream` use `${env.AI_GATEWAY_ENDPOINT}/anthropic/...` and send `cf-aig-authorization: Bearer ${env.AI_GATEWAY_TOKEN}` (replacing `x-api-key`/`ANTHROPIC_API_KEY`). The `llm.test.ts` assertions are updated to match.

**Interface under test:** `callWorkersAI(env, model, messages, maxTokens)` exported from `apps/api/src/services/llm.ts`, exercised via the existing `llm.test.ts`.

**Files:**
- Modify: `apps/api/src/services/llm.ts`
- Modify: `apps/api/src/services/llm.test.ts`

---

- [ ] **Step 1: Write the failing test**

Update `apps/api/src/services/llm.test.ts` — change `AI_GATEWAY_BACKGROUND` to `AI_GATEWAY_ENDPOINT`, add `AI_GATEWAY_TOKEN`, and assert `cf-aig-authorization` header. The existing URL assertion must change from using `AI_GATEWAY_BACKGROUND` env value to the new `AI_GATEWAY_ENDPOINT` value.

```typescript
// apps/api/src/services/llm.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { callWorkersAI } from "./llm";
import type { Bindings } from "../lib/types";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("callWorkersAI", () => {
  it("sends Authorization Bearer + cf-aig-authorization headers to AI_GATEWAY_ENDPOINT/workers-ai path", async () => {
    const mockFetch = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: "Test title" } }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );

    const mockEnv = {
      AI_GATEWAY_ENDPOINT: "https://gateway.example.com",
      AI_GATEWAY_TOKEN: "gw-token-abc",
      CLOUDFLARE_API_TOKEN: "test-cf-token-abc123",
    } as unknown as Bindings;

    const result = await callWorkersAI(
      mockEnv,
      "@cf/google/gemma-4-26b-a4b-it",
      [{ role: "user", content: "Generate a title" }],
      30,
    );

    expect(mockFetch).toHaveBeenCalledOnce();
    const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(
      "https://gateway.example.com/workers-ai/v1/chat/completions",
    );

    const headers = init.headers as Record<string, string>;
    expect(headers["Authorization"]).toBe("Bearer test-cf-token-abc123");
    expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token-abc");
    expect(result).toBe("Test title");
  });

  it("throws InferenceError on non-ok response", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response("Unauthorized", { status: 401 }),
    );

    const mockEnv = {
      AI_GATEWAY_ENDPOINT: "https://gateway.example.com",
      AI_GATEWAY_TOKEN: "bad-gw-token",
      CLOUDFLARE_API_TOKEN: "bad-token",
    } as unknown as Bindings;

    await expect(
      callWorkersAI(mockEnv, "@cf/test-model", [
        { role: "user", content: "hi" },
      ]),
    ).rejects.toThrow("Workers AI error: 401");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run src/services/llm.test.ts
```
Expected: FAIL — current `callWorkersAI` uses `env.AI_GATEWAY_BACKGROUND` (not `env.AI_GATEWAY_ENDPOINT`) so the URL assertion fails; also `cf-aig-authorization` header is missing.

- [ ] **Step 3: Implement the minimum to make the test pass**

Edit `apps/api/src/services/llm.ts` — update `callWorkersAI`, `callAnthropic`, and `callAnthropicStream`. The exact changes:

1. `callWorkersAI`: change `${env.AI_GATEWAY_BACKGROUND}/workers-ai/...` to `${env.AI_GATEWAY_ENDPOINT}/workers-ai/...`; add `"cf-aig-authorization": \`Bearer ${env.AI_GATEWAY_TOKEN}\`` to headers (keep `Authorization`).
2. `callAnthropic`: change `${env.AI_GATEWAY_TEACHER}/anthropic/...` to `${env.AI_GATEWAY_ENDPOINT}/anthropic/...`; replace `"x-api-key": env.ANTHROPIC_API_KEY` with `"cf-aig-authorization": \`Bearer ${env.AI_GATEWAY_TOKEN}\``.
3. `callAnthropicStream`: same as `callAnthropic` — endpoint + auth swap.

```typescript
// apps/api/src/services/llm.ts
import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

// ---------------------------------------------------------------------------
// Content block types for multi-turn tool_use conversations
// ---------------------------------------------------------------------------

export type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | {
      type: "tool_result";
      tool_use_id: string;
      content: string;
      is_error?: boolean;
    };

interface LlmMessage {
  role: "system" | "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

export interface AnthropicSystemBlock {
  type: "text";
  text: string;
  cache_control?: { type: "ephemeral" };
}

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  system?: string | AnthropicSystemBlock[];
  messages: LlmMessage[];
  stream?: boolean;
  tools?: unknown[];
  tool_choice?: unknown;
}

interface AnthropicResponse {
  content: Array<{
    type: string;
    text?: string;
    id?: string;
    name?: string;
    input?: unknown;
  }>;
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number };
}

export async function callAnthropic(
  env: Bindings,
  request: AnthropicRequest,
): Promise<AnthropicResponse> {
  const url = `${env.AI_GATEWAY_ENDPOINT}/anthropic/v1/messages`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
      "anthropic-version": "2023-06-01",
      "anthropic-beta": "prompt-caching-2024-07-31",
    },
    body: JSON.stringify({ ...request, stream: false }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(`Anthropic request failed: ${res.status} ${text}`);
  }

  return res.json() as Promise<AnthropicResponse>;
}

export async function callAnthropicStream(
  env: Bindings,
  request: AnthropicRequest,
): Promise<ReadableStream> {
  const url = `${env.AI_GATEWAY_ENDPOINT}/anthropic/v1/messages`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
      "anthropic-version": "2023-06-01",
      "anthropic-beta": "prompt-caching-2024-07-31",
    },
    body: JSON.stringify({ ...request, stream: true }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(
      `Anthropic stream request failed: ${res.status} ${text}`,
    );
  }

  if (!res.body) {
    throw new InferenceError("Anthropic stream response has no body");
  }

  return res.body;
}

export async function callWorkersAI(
  env: Bindings,
  model: string,
  messages: Array<{ role: string; content: string }>,
  maxTokens: number = 100,
  chatTemplateKwargs?: { enable_thinking?: boolean; clear_thinking?: boolean },
): Promise<string> {
  const url = `${env.AI_GATEWAY_ENDPOINT}/workers-ai/v1/chat/completions`;
  const body: Record<string, unknown> = {
    model,
    messages,
    max_tokens: maxTokens,
  };
  if (chatTemplateKwargs) {
    body.chat_template_kwargs = chatTemplateKwargs;
  }
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
      Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new InferenceError(`Workers AI error: ${res.status}`);
  }
  const data = (await res.json()) as {
    choices: Array<{ message: { content: string } }>;
  };
  if (!data.choices?.[0]?.message?.content) {
    throw new InferenceError(
      `Workers AI returned no content. model=${model} body=${JSON.stringify(data).slice(0, 500)}`,
    );
  }
  return data.choices[0].message.content;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run src/services/llm.test.ts
```
Expected: PASS

- [ ] **Step 4b: Live chat smoke — `callAnthropicStream` auth is untested by unit tests (challenge RISK #1)**

`callAnthropicStream` had its auth swapped from `x-api-key`/`ANTHROPIC_API_KEY` to `cf-aig-authorization`/`AI_GATEWAY_TOKEN` with no SSE unit test (SSE translation is out of scope for this plan). Because the streaming chat path is the only currently-working Anthropic path, the build agent MUST manually confirm it still streams before declaring this task done. NOTE: chat stays on the Anthropic provider, so it requires the BYOK Anthropic account to have credits — if it returns `400 credit balance too low`, that is the known out-of-credits state (the entire reason for this issue), NOT an auth regression. Distinguish: a `401 AiGatewayError`/missing-`cf-aig-authorization` failure IS a regression and blocks; a `400 credit balance too low` is expected and does not block.

```bash
# With `just dev` running, send one chat message and confirm SSE tokens stream
# (or a 400 credit error — both prove the gateway auth header is accepted):
CRESCEND_COOKIE=… curl -N -sS https://localhost:8787/api/chat \
  -H "Content-Type: application/json" -H "Cookie: $CRESCEND_COOKIE" \
  -d '{"message":"hi","conversationId":null}' | head -5
```
Expected: SSE `event:`/`data:` lines stream, OR a `400 credit balance too low` (accepted). A `401`/`AiGatewayError` is a FAIL — re-check the `cf-aig-authorization` header.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/llm.ts apps/api/src/services/llm.test.ts && git commit -m "fix(llm): fix gateway var/auth drift — use AI_GATEWAY_ENDPOINT + cf-aig-authorization for all providers"
```

---

## Task 8: dead-var cleanup + remaining test mock updates

**Group:** C (sequential, depends on Group B — all implementation files are settled)

**Behavior being verified:** The full test suite passes with the updated mock bindings. Dead Bindings fields (`AI_GATEWAY_TEACHER`, `AI_GATEWAY_BACKGROUND`, `ANTHROPIC_API_KEY`) are removed from `lib/types.ts` now that no production code references them. All test files that previously used `AI_GATEWAY_TEACHER`/`ANTHROPIC_API_KEY` in their `MOCK_BINDINGS` are updated to `AI_GATEWAY_ENDPOINT`/`AI_GATEWAY_TOKEN`.

**Interface under test:** The entire harness test suite — all five affected test files plus the full typecheck.

**Files:**
- Modify: `apps/api/src/lib/types.ts` (remove `AI_GATEWAY_TEACHER`, `AI_GATEWAY_BACKGROUND`, `ANTHROPIC_API_KEY`)
- Modify: `apps/api/src/harness/loop/phase2-schema.test.ts`
- Modify: `apps/api/src/harness/loop/runHook.test.ts`
- Modify: `apps/api/src/services/teacher-synthesize-v6.test.ts`
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`
- Modify: `apps/api/src/harness/skills/__catalog__/integration.test.ts`

---

- [ ] **Step 1: Write the failing test**

Attempt to remove the three dead fields from `apps/api/src/lib/types.ts` Bindings:

```typescript
// Remove these three lines from Bindings:
// ANTHROPIC_API_KEY: string;
// AI_GATEWAY_TEACHER: string;
// AI_GATEWAY_BACKGROUND: string;
```

After removing them, run the full test suite. It fails because the five test files still set `AI_GATEWAY_TEACHER` and `ANTHROPIC_API_KEY` in their `MOCK_BINDINGS`.

```bash
cd apps/api && bun run typecheck
```
Expected: type errors in the test files referencing the removed fields.

- [ ] **Step 2: Run typecheck — verify it FAILS with the dead fields removed**

```bash
cd apps/api && bun run typecheck 2>&1 | grep -E "AI_GATEWAY_TEACHER|AI_GATEWAY_BACKGROUND|ANTHROPIC_API_KEY" | head -20
```
Expected: lines referencing the removed fields in the five test files.

- [ ] **Step 3: Update all five test files + run full suite**

Update `MOCK_BINDINGS` in each of the five test files:

**`apps/api/src/harness/loop/phase2-schema.test.ts`** — change:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_TEACHER: "https://gw.example",
  ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;
```
to:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_ENDPOINT: "https://gw.example",
  AI_GATEWAY_TOKEN: "test-gw-token",
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;
```

**`apps/api/src/harness/loop/runHook.test.ts`** — change:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_TEACHER: "https://gw.example",
  ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;
```
to:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_ENDPOINT: "https://gw.example",
  AI_GATEWAY_TOKEN: "test-gw-token",
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;
```

**`apps/api/src/services/teacher-synthesize-v6.test.ts`** — change:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_TEACHER: 'https://gw.example',
  ANTHROPIC_API_KEY: 'test-key',
} as unknown as Bindings
```
to:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_ENDPOINT: 'https://gw.example',
  AI_GATEWAY_TOKEN: 'test-gw-token',
  TEACHER_PROVIDER: 'anthropic',
} as unknown as Bindings
```

**`apps/api/src/services/teacher-chat-v6.test.ts`** — change:
```typescript
const MOCK_ENV = {
  AI_GATEWAY_TEACHER: "https://gw.example",
  ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;
```
to:
```typescript
const MOCK_ENV = {
  AI_GATEWAY_ENDPOINT: "https://gw.example",
  AI_GATEWAY_TOKEN: "test-gw-token",
  TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;
```

**`apps/api/src/harness/skills/__catalog__/integration.test.ts`** — change:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_TEACHER: 'https://gw.example',
  ANTHROPIC_API_KEY: 'test-key',
} as unknown as Bindings
```
to:
```typescript
const MOCK_BINDINGS = {
  AI_GATEWAY_ENDPOINT: 'https://gw.example',
  AI_GATEWAY_TOKEN: 'test-gw-token',
  TEACHER_PROVIDER: 'anthropic',
} as unknown as Bindings
```

- [ ] **Step 4: Run all affected tests — verify they PASS**

```bash
cd apps/api && bunx vitest run src/harness/loop/phase2-schema.test.ts src/harness/loop/runHook.test.ts src/services/teacher-synthesize-v6.test.ts src/services/teacher-chat-v6.test.ts "src/harness/skills/__catalog__/integration.test.ts" && bun run typecheck
```
Expected: all tests PASS, typecheck exits 0.

Run the complete automated check from the spec's verification architecture:

```bash
cd apps/api && bunx vitest run src/harness/loop/tool-format.test.ts src/harness/loop/gateway-client.test.ts src/harness/loop/route-model.test.ts src/harness/loop/phase1.test.ts src/harness/loop/phase2.test.ts src/services/llm.test.ts src/harness/loop/phase2-schema.test.ts src/harness/loop/runHook.test.ts src/services/teacher-synthesize-v6.test.ts src/services/teacher-chat-v6.test.ts "src/harness/skills/__catalog__/integration.test.ts"
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/lib/types.ts apps/api/src/harness/loop/phase2-schema.test.ts apps/api/src/harness/loop/runHook.test.ts apps/api/src/services/teacher-synthesize-v6.test.ts apps/api/src/services/teacher-chat-v6.test.ts "apps/api/src/harness/skills/__catalog__/integration.test.ts" && git commit -m "feat(types): remove dead AI_GATEWAY_TEACHER/AI_GATEWAY_BACKGROUND/ANTHROPIC_API_KEY, update all test mocks"
```

---

## Challenge Review

### CEO Pass

**Premise.** The problem is real and precisely scoped: after the 2026-06-10 gateway unification, `phase1.ts` and `phase2.ts` build `undefined/anthropic/v1/messages` because `AI_GATEWAY_TEACHER` is no longer in `.dev.vars`, and the Anthropic account is out of credits. V6 synthesis has been silently dead for every local session. This plan is the direct fix.

**Scope.** Eight files modified, two created (plus two new test files). Every change traces to the stated goal — no scope creep detected. The one non-obvious inclusion (`llm.ts` / `callWorkersAI` gateway-var fix) is a dependency cleanup correctly bundled in Task 7; it would break if left to drift.

**12-Month alignment.**

```
CURRENT STATE                    THIS PLAN                      12-MONTH IDEAL
phase1/phase2 dead locally   →   callModel shim, TEACHER_     →   provider toggle enables
(undefined URL, no credits)       PROVIDER env toggle              Qwen→finetuned-Qwen swap
                                  workers-ai default               with zero harness rewrites
```

The design explicitly keeps the Anthropic path selectable (`TEACHER_PROVIDER=anthropic`) so finetune A/B is a one-var swap. Aligned.

**Alternatives.** The spec documents the chosen approach (shim at client boundary, preserving Anthropic-shaped types end-to-end) and explains the rejection of alternatives (JSON-mode prompting, hard swap). Covered.

---

### Engineering Pass

**Architecture.**

Data flow for `runPhase1` after the change:

```
runPhase1(ctx, binding)
  └─ routeModel("phase1_analysis", ctx.env)  → {provider:"workers-ai"|"anthropic", model}
  └─ callModel(env, client, body)             → AnthropicMessageResponse
       ├── anthropic path: POST /anthropic/v1/messages, cf-aig-authorization
       └── workers-ai path: toOpenAIChatRequest(body)
                            → POST /workers-ai/v1/chat/completions
                            → toAnthropicResponse(oaiRes)
                            → AnthropicMessageResponse
```

Component boundaries are clean. `tool-format` is stateless and pure. `gateway-client` owns all HTTP + auth concern. `route-model` owns the env toggle. `phase1`/`phase2` are untouched in logic.

**Critical type collision — `ModelClient` exported from two modules.**

Both `gateway-client.ts` (Task 2, Step 3) and `route-model.ts` (Task 3, Step 3) define and export `interface ModelClient { provider: "anthropic" | "workers-ai"; model: string }`. When `phase1.ts` (Task 5) imports `routeModel` from `route-model` and passes the result to `callModel` from `gateway-client`, TypeScript will structurally unify them (they are identical shapes), so there is no compile error. However, the build agent may import `ModelClient` from either source when writing `phase1.ts`/`phase2.ts`, creating an ambiguous nominal dependency. This is not a type-error blocker but is an OBS: the canonical `ModelClient` should live in one file and be re-exported from the other, or both files should import from a shared location. The plan's provided `phase1.ts` implementation only imports from `gateway-client` and `route-model` without re-importing `ModelClient` by name, so in practice this is harmless — the return type of `routeModel` is inferred. Noting as OBS.

**Phase1 test URL matching after Task 5.**

After the `MOCK_BINDINGS` change to `AI_GATEWAY_ENDPOINT: "https://gw.example"` + `TEACHER_PROVIDER: "anthropic"`, `callModel` will POST to `https://gw.example/anthropic/v1/messages`. The existing `phase1.test.ts` mocks mock `global.fetch` unconditionally — they do not assert on the URL, only on call count and event types. So the URL change is transparent to those tests. Safe.

**Phase2 test: `tool_choice` assertion may break.**

`phase2.test.ts` line 88–94 asserts `callBody.tool_choice` equals `{ type: "tool", name: "write_synthesis_artifact" }`. After Task 6, the workers-ai path would translate this to `{ type: "function", function: { name: "write_synthesis_artifact" } }` before sending. However, Task 6 sets `TEACHER_PROVIDER: "anthropic"` in `MOCK_BINDINGS`, so the anthropic path is taken — the body is passed verbatim to fetch, and `tool_choice` is not translated. The assertion holds. Safe.

**Task 7 — `callAnthropicStream` not tested.**

The plan updates `callAnthropic` and `callAnthropicStream` in `llm.ts` but the test (`llm.test.ts`) only covers `callWorkersAI`. The `callAnthropicStream` gateway-var/auth change is applied in the implementation but has no test coverage for the new `cf-aig-authorization` header or the `AI_GATEWAY_ENDPOINT` URL path. The spec explicitly excludes streaming chat from the Workers AI scope, but the auth fix to `callAnthropicStream` is still a behavior change to a live path (the `/api/chat` SSE endpoint) with zero test coverage of the changed auth header.

**`wrangler.toml` not mentioned.**

The plan adds `AI_GATEWAY_ENDPOINT`, `AI_GATEWAY_TOKEN`, `TEACHER_PROVIDER`, and `CLOUDFLARE_API_TOKEN` to `Bindings` in `types.ts`. `wrangler.toml` must declare these vars for the worker to see them at runtime. The spec does not mention updating `wrangler.toml`, and the plan has no task for it. For `TEACHER_PROVIDER` (the new toggle) this means a local `.dev.vars` entry is required — if absent, `routeModel` defaults to `workers-ai` (the desired default), so it is not a blocker for the Workers AI goal. But if `.dev.vars` does not have `AI_GATEWAY_ENDPOINT`/`AI_GATEWAY_TOKEN`, the worker fails at runtime even though types compile. This was the original bug; the plan addresses the code but not the runtime binding config.

---

### Module Depth Audit

| Module | Exported interface | Implementation | Verdict |
|--------|-------------------|----------------|---------|
| `tool-format.ts` | 2 functions + ~8 type exports | ~100 LOC, handles all Anthropic↔OpenAI mapping including content-block fan-out, tool_choice variants, JSON parsing | DEEP |
| `gateway-client.ts` | 1 function (`callModel`) + `ModelClient` type | ~40 LOC, two provider branches, fetch, error handling, delegates to tool-format | DEEP |
| `route-model.ts` | 1 function (`routeModel`) + `ModelClient` + `TaskKind` | ~15 LOC, two constant returns | SHALLOW — but this is acceptable: the function is a pure config lookup and its simplicity is intentional. The interface is trivial because the logic is trivial. Not a smell here. |

---

### Test Philosophy Audit

All tests mock only the external HTTP boundary (`global.fetch`). No internal collaborators are mocked. Tests assert on observable behavior: returned Anthropic-shaped response, outgoing URL, outgoing headers, event sequences, call counts. No private methods are called, no internal state is inspected. Philosophy is sound.

**`phase2.test.ts` `tool_choice` body inspection.** Line 88–94 in the existing test parses the fetch body and asserts `callBody.tool_choice`. This is testing what went over the wire (the HTTP boundary), not internal state — this is acceptable. However, after Task 6 switches to `callModel`, the body sent to fetch is now the Anthropic-format body (for the anthropic path). The assertion `callBody.tool_choice.type === "tool"` remains correct because the anthropic path passes the body verbatim. If the path were `workers-ai`, `toOpenAIChatRequest` translates `{type:"tool",name:X}` to `{type:"function",function:{name:X}}` and the assertion would fail. The `TEACHER_PROVIDER: "anthropic"` in the updated mock prevents this. This coupling is fragile — if someone later runs the test against a `workers-ai` mock env, the assertion silently changes meaning.

---

### Vertical Slice Audit

Each task has exactly one failing-test step, one implementation step, one commit. No horizontal slicing. The group dependency (A → B → C) is correctly sequenced: Tasks 1-4 create the new modules; Tasks 5-7 replace the call sites; Task 8 removes dead vars and mops up the remaining test mocks that were left in place by Tasks 5-7 to avoid compile errors mid-build.

The "write failing test" for Task 4 is a typecheck rather than a runtime test. This is appropriate — the behavior being gated is "these new fields must exist in Bindings so that subsequent Tasks compile." A typecheck is the correct instrument here.

---

### Test Coverage Gaps

```
[+] tool-format.ts
    ├── toOpenAIChatRequest()
    │   ├── [TESTED] tool def mapping (input_schema → function.parameters) ★★★
    │   ├── [TESTED] tool_choice auto → "auto" ★★★
    │   ├── [TESTED] tool_choice tool → {type:"function",function:{name}} ★★★
    │   ├── [TESTED] string user content ★★
    │   ├── [TESTED] assistant tool_use blocks → tool_calls ★★★
    │   ├── [TESTED] user tool_result blocks → role:tool messages ★★★
    │   └── [GAP]   assistant message with MIXED text + tool_use blocks
    │               (text block is silently dropped in current impl — no test covers this)
    └── toAnthropicResponse()
        ├── [TESTED] tool_calls → tool_use blocks, arguments JSON-parsed ★★★
        ├── [TESTED] text content → text block, end_turn ★★★
        ├── [TESTED] tool_calls present → stop_reason tool_use ★★
        └── [TESTED] null content + null tool_calls → empty content ★★

[+] gateway-client.ts
    ├── callModel() — anthropic
    │   ├── [TESTED] URL, cf-aig-authorization, anthropic-version headers ★★★
    │   ├── [TESTED] non-2xx → InferenceError ★★★
    │   └── [GAP]   response body JSON parse error (malformed JSON from gateway)
    └── callModel() — workers-ai
        ├── [TESTED] URL, cf-aig-authorization, Authorization headers ★★★
        ├── [TESTED] response translated to Anthropic shape ★★★
        └── [TESTED] non-2xx → InferenceError ★★★

[+] llm.ts — callAnthropic / callAnthropicStream
    ├── [GAP] cf-aig-authorization header — no test (only callWorkersAI is tested)
    └── [GAP] AI_GATEWAY_ENDPOINT URL path for Anthropic — no test
```

The mixed text+tool_use gap in `toOpenAIChatRequest` is the only functionally significant gap. In the current implementation (Task 1, Step 3), when an assistant message has `content: [{type:"text",...}, {type:"tool_use",...}]`, the `for (const block of msg.content)` loop only appends `tool_calls` — the text block is dropped silently. In practice, Phase 1 multi-atom turns will have assistant messages with both text preamble and tool calls. If Qwen emits a text preamble before the tool_use (which it sometimes does), the tool_calls still come through correctly (the only thing dropped is the text preamble on the assistant turn), so this is a correctness gap but not a synthesis-breaking bug. The Phase 2 forced-tool path always returns tool_calls only (no preamble expected).

---

### Failure Modes

| Task | Failure scenario | Recovery |
|------|-----------------|----------|
| T1: tool-format | JSON.parse of malformed `function.arguments` throws | Propagates as uncaught exception through `callModel` → `withRetries` catches `InferenceError` only; a SyntaxError from JSON.parse would propagate as an unhandled exception through `runPhase2` and surface as a DO-level error, visible in logs. Not silent. |
| T2: gateway-client | `res.text()` on error response is awaited inside `InferenceError` constructor throw | `res.text()` is called before the throw, correctly consumes the body. Safe. |
| T3: route-model | `TEACHER_PROVIDER` undefined → defaults to workers-ai | Correct intended behavior. |
| T7: llm.ts | `callAnthropicStream` used by chat path — auth change from `x-api-key` to `cf-aig-authorization` | If `AI_GATEWAY_TOKEN` is undefined in `.dev.vars`, stream requests fail with 401 from the gateway. This is the same root cause as the original bug, now for the chat path. Since `.dev.vars` was updated as part of the gateway unification, this should already be populated — but it is an assumption. |
| T8: dead-var removal | TypeScript errors in files not listed in the plan | The plan lists 5 test files. If any other file references `AI_GATEWAY_TEACHER`/`ANTHROPIC_API_KEY`/`AI_GATEWAY_BACKGROUND`, typecheck in Step 4 will catch it before commit. Explicit failure, not silent. |

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|-----------|---------|--------|
| `.dev.vars` already contains `AI_GATEWAY_ENDPOINT`, `AI_GATEWAY_TOKEN`, `CLOUDFLARE_API_TOKEN` | VALIDATE | These were added in the 2026-06-10 gateway unification. Plan does not verify them. If absent, runtime fails even though tests pass. |
| `TEACHER_PROVIDER` not being in `.dev.vars` causes `routeModel` to default to `workers-ai` | SAFE | `env.TEACHER_PROVIDER === "anthropic"` is the only branch that returns anthropic; any other value (undefined, empty, etc.) returns workers-ai. |
| Qwen3-30B-A3B's `tool_choice:{type:"function",function:{name}}` forced call always returns `tool_calls` (never a bare text response) | VALIDATE | Proven via live curl for the non-agentic case, but Phase 2's repair loop depends on this holding across 3 attempts. If Qwen ignores the forced tool_choice on a repair turn, `toolUse` is undefined and `phase_error` is emitted — explicit failure, not silent. |
| The existing `phase1.test.ts` tests pass unchanged (besides `MOCK_BINDINGS`) after `callAnthropicMessage` is replaced by `callModel` | SAFE | Tests mock global fetch unconditionally and assert on event types and call counts, not on URL or headers. The URL changes, but tests do not check it. |
| The `phase2.test.ts` `tool_choice` body assertion holds after Task 6 | SAFE | Confirmed: `TEACHER_PROVIDER: "anthropic"` causes `callModel` to take the anthropic path, which sends the body verbatim without `toOpenAIChatRequest` translation. The assertion value is unchanged. |
| `wrangler.toml` does not need updating | RISKY | `wrangler.toml` defines which vars are bound. If it still lists `AI_GATEWAY_TEACHER`/`AI_GATEWAY_BACKGROUND` but not `AI_GATEWAY_ENDPOINT`/`AI_GATEWAY_TOKEN`, `wrangler dev` will log unknown-binding warnings and the runtime will miss the new vars. Plan does not check or update `wrangler.toml`. |
| No other production code references `AI_GATEWAY_TEACHER`/`AI_GATEWAY_BACKGROUND`/`ANTHROPIC_API_KEY` beyond the 5 test files listed in Task 8 | VALIDATE | Typecheck in Task 8 Step 4 will surface any missed references. But the build agent should run `grep -r AI_GATEWAY_TEACHER apps/api/src` before removing the fields, not rely on typecheck alone discovering them mid-step. |

---

### Summary

[BLOCKER] count: 0
[RISK]    count: 3
[QUESTION] count: 0
[OBS]     count: 2

**[RISK] (confidence: 8/10)** — `callAnthropicStream` in `llm.ts` (Task 7) has its auth header changed from `x-api-key: ANTHROPIC_API_KEY` to `cf-aig-authorization: Bearer AI_GATEWAY_TOKEN` with no test coverage of the new header or URL. The live SSE chat path (`/api/chat`) is the only user-visible path currently on Anthropic. If `AI_GATEWAY_TOKEN` is misconfigured, chat breaks silently in `.dev` — with no test to catch it. Fallback: manual `curl` the chat endpoint after Task 7 to confirm stream headers.

**[RISK] (confidence: 7/10)** — `wrangler.toml` is not updated in the plan. If it still binds the old `AI_GATEWAY_TEACHER`/`AI_GATEWAY_BACKGROUND` vars and not the new `AI_GATEWAY_ENDPOINT`/`AI_GATEWAY_TOKEN`, `wrangler dev` may warn or produce incorrect runtime behavior even though all tests pass (tests use `as unknown as Bindings` stubs, not wrangler-resolved env). The final manual verification step (`just dev` + driver) will catch this, but it should be checked earlier. Fallback: add a one-line check `grep AI_GATEWAY_ENDPOINT apps/api/wrangler.toml` to Task 4 before the commit.

**[RISK] (confidence: 6/10)** — `toOpenAIChatRequest` silently drops text blocks from mixed assistant messages (content: `[{type:"text",...}, {type:"tool_use",...}]`). Phase 1 multi-atom agentic turns may produce such responses from Qwen. The tool_calls are still extracted correctly, so synthesis is not broken — but the text preamble is lost from the reconstructed conversation context, which may cause context drift on multi-turn Phase 1 reasoning. No test covers this. Fallback: if Phase 1 quality is poor on live runs, inspect the Phase 1 fetch bodies for text+tool_use mixed responses.

**[OBS]** — `ModelClient` is defined and exported from both `route-model.ts` and `gateway-client.ts`. Both definitions are identical. TypeScript structural typing means this compiles without error, but the build agent should import `ModelClient` from one canonical source only — preferably `route-model.ts` (it is the producer) and import it into `gateway-client.ts`.

**[OBS]** — The `phase2.test.ts` `tool_choice` body assertion (`callBody.tool_choice.type === "tool"`) holds only because `TEACHER_PROVIDER: "anthropic"` is set in `MOCK_BINDINGS`, which causes the body to pass through verbatim. If that mock is ever changed to test the workers-ai path, the assertion would fail with a confusing mismatch (`{type:"function",...}` vs `{type:"tool",...}`). Consider adding a comment noting this coupling.

---

VERDICT: PROCEED_WITH_CAUTION — [RISK: callAnthropicStream auth change untested; RISK: wrangler.toml not updated for new vars; RISK: mixed text+tool_use assistant messages silently drop text blocks]
