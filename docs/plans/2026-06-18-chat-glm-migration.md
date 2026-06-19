# Chat GLM Migration Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Migrate V6 chat teacher from Anthropic claude-sonnet-4 streaming to
glm-4.7-flash on Workers AI via the AI Gateway, zero Anthropic spend, preserving SSE
contract.
**Spec:** docs/specs/2026-06-18-chat-glm-migration-design.md
**Style:** Follow `apps/api/TS_STYLE.md`. Use `bun run test` (not `bun test`). No emojis.
Explicit exception handling. Surgical changes only — no reformatting of untouched lines.
**Issue:** Closes #69

---

## Task Groups

```
Group A (parallel): Task 1a, Task 1b
Group B (sequential, depends on A): Task 2
Group C (sequential, depends on B): Task 3
Group D (sequential, depends on C): Task 4
```

Tasks 1a and 1b touch the same file (`tool-format.ts`). They are listed as parallel in
description but **must be executed sequentially** by the build agent because they share
one file. Dispatch Task 1b only after Task 1a commits.

Revised dispatch order:
```
Task 1a → Task 1b → Task 2 → Task 3 → Task 4
```

---

### Task 1a: Translate system blocks into a leading OpenAI system message

**Group:** A (first in sequential chain)

**Behavior being verified:** When `toOpenAIChatRequest` is called with a `system` field
containing Anthropic system blocks (or a plain string), the resulting OpenAI message list
begins with a `{role:"system", content:"<joined text>"}` message, and `cache_control` is
stripped.

**Interface under test:** `toOpenAIChatRequest` exported from
`apps/api/src/harness/loop/tool-format.ts`

**Files:**
- Modify: `apps/api/src/harness/loop/tool-format.ts`
- Modify: `apps/api/src/harness/loop/tool-format.test.ts`

---

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/harness/loop/tool-format.test.ts`, inside a new
`describe("toOpenAIChatRequest — system block translation", ...)` block:

```typescript
describe("toOpenAIChatRequest — system block translation", () => {
  it("prepends an array of system blocks as a leading role:system message, stripping cache_control", () => {
    const req = {
      model: "@cf/zai-org/glm-4.7-flash",
      max_tokens: 2048,
      system: [
        { type: "text" as const, text: "You are a helpful teacher.", cache_control: { type: "ephemeral" as const } },
        { type: "text" as const, text: "<student_memory>some facts</student_memory>" },
      ],
      messages: [{ role: "user" as const, content: "How do I improve my dynamics?" }],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };

    const out = toOpenAIChatRequest(req);

    expect(out.messages[0]).toEqual({
      role: "system",
      content: "You are a helpful teacher.\n\n<student_memory>some facts</student_memory>",
    });
    expect(out.messages[1]).toEqual({ role: "user", content: "How do I improve my dynamics?" });
    expect(out.messages).toHaveLength(2);
  });

  it("prepends a plain string system field as a leading role:system message", () => {
    const req = {
      model: "@cf/zai-org/glm-4.7-flash",
      max_tokens: 2048,
      system: "You are a piano teacher.",
      messages: [{ role: "user" as const, content: "Hello" }],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };

    const out = toOpenAIChatRequest(req);

    expect(out.messages[0]).toEqual({ role: "system", content: "You are a piano teacher." });
    expect(out.messages[1]).toEqual({ role: "user", content: "Hello" });
  });

  it("does not prepend a system message when system field is absent", () => {
    const req = {
      model: "@cf/zai-org/glm-4.7-flash",
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "Hello" }],
      tools: [],
      tool_choice: { type: "auto" } as const,
    };

    const out = toOpenAIChatRequest(req);

    expect(out.messages[0]).toEqual({ role: "user", content: "Hello" });
    expect(out.messages).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/harness/loop/tool-format.test.ts
```

Expected: FAIL — `system` property does not exist on type `AnthropicChatRequest` (TypeScript
error) or test asserting `out.messages[0].role === "system"` fails because no system message
is prepended.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/harness/loop/tool-format.ts`:

**3a. Add `system` to `AnthropicChatRequest`** (this type is in the same file):

```typescript
// Add this import at the top of tool-format.ts (it already has AnthropicSystemBlock from llm.ts — import it):
// AnthropicSystemBlock is defined locally in tool-format.ts? No — check: it is NOT defined there.
// We need to add a local definition or import from llm.ts.
// tool-format.ts does NOT currently import from llm.ts. Define inline to avoid circular imports:
```

In `tool-format.ts`, add `AnthropicSystemBlock` to the local types (before `AnthropicChatRequest`):

```typescript
export interface AnthropicSystemBlock {
  type: "text";
  text: string;
  cache_control?: { type: "ephemeral" };
}
```

Then update `AnthropicChatRequest`:

```typescript
export interface AnthropicChatRequest {
  model: string;
  max_tokens: number;
  system?: string | AnthropicSystemBlock[];
  messages: AnthropicMessage[];
  tools?: AnthropicToolDef[];
  tool_choice?: AnthropicToolChoice;
}
```

**3b. Add a helper to resolve the system string** (inside the file, before `toOpenAIChatRequest`):

```typescript
function resolveSystemText(system: string | AnthropicSystemBlock[]): string {
  if (typeof system === "string") {
    return system;
  }
  return system.map((b) => b.text).join("\n\n");
}
```

**3c. Prepend the system message in `toOpenAIChatRequest`** (at the start of the
`messages` array construction):

Replace:

```typescript
  const messages: OpenAIMessage[] = [];
  for (const msg of req.messages) {
```

With:

```typescript
  const messages: OpenAIMessage[] = [];
  if (req.system !== undefined) {
    messages.push({ role: "system", content: resolveSystemText(req.system) });
  }
  for (const msg of req.messages) {
```

Note: `OpenAIMessage` needs a system variant. Add it to the union:

```typescript
type OpenAIMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: OpenAIToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/harness/loop/tool-format.test.ts
```

Expected: All tests in `tool-format.test.ts` PASS (existing tests + 3 new ones).

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration && git add apps/api/src/harness/loop/tool-format.ts apps/api/src/harness/loop/tool-format.test.ts && git commit -m "fix(tool-format): translate system blocks to leading OpenAI system message

AnthropicChatRequest gains optional system field. toOpenAIChatRequest
joins text blocks (stripping cache_control) and prepends a role:system
message. Without this, UNIFIED_TEACHER_SYSTEM and student_memory are
silently dropped on the Workers AI path. Refs #69."
```

---

### Task 1b: Map `tool_choice:{type:"none"}` to `"none"`

**Group:** A (second — run after Task 1a commits)

**Behavior being verified:** When `toOpenAIChatRequest` receives `tool_choice:{type:"none"}`,
the output `tool_choice` is the string `"none"`.

**Interface under test:** `toOpenAIChatRequest` exported from
`apps/api/src/harness/loop/tool-format.ts`

**Files:**
- Modify: `apps/api/src/harness/loop/tool-format.ts`
- Modify: `apps/api/src/harness/loop/tool-format.test.ts`

---

- [ ] **Step 1: Write the failing test**

Add to the existing `describe("toOpenAIChatRequest — tool definition mapping", ...)` block
in `apps/api/src/harness/loop/tool-format.test.ts`:

```typescript
  it("converts tool_choice {type:'none'} to string 'none'", () => {
    const req = {
      model: "@cf/zai-org/glm-4.7-flash",
      max_tokens: 2048,
      messages: [{ role: "user" as const, content: "Just respond." }],
      tools: [],
      tool_choice: { type: "none" } as const,
    };
    const out = toOpenAIChatRequest(req);
    expect(out.tool_choice).toBe("none");
  });
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/harness/loop/tool-format.test.ts
```

Expected: FAIL — TypeScript error: `{type:"none"}` is not assignable to
`AnthropicToolChoice` (which is `AnthropicToolChoiceAuto | AnthropicToolChoiceTool`), OR
if TypeScript is relaxed, the test fails because `out.tool_choice` is `"auto"` not `"none"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/harness/loop/tool-format.ts`:

**3a. Add `AnthropicToolChoiceNone` to the type union:**

```typescript
export interface AnthropicToolChoiceNone {
  type: "none";
}

export type AnthropicToolChoice =
  | AnthropicToolChoiceAuto
  | AnthropicToolChoiceTool
  | AnthropicToolChoiceNone;
```

**3b. Handle `type:"none"` in the `tool_choice` translation block inside
`toOpenAIChatRequest`:**

Replace:

```typescript
  let tool_choice: OpenAIToolChoice = "auto";
  if (req.tool_choice) {
    if (req.tool_choice.type === "auto") {
      tool_choice = "auto";
    } else if (req.tool_choice.type === "tool") {
      tool_choice = {
        type: "function",
        function: { name: req.tool_choice.name },
      };
    }
  }
```

With:

```typescript
  let tool_choice: OpenAIToolChoice = "auto";
  if (req.tool_choice) {
    if (req.tool_choice.type === "auto") {
      tool_choice = "auto";
    } else if (req.tool_choice.type === "none") {
      tool_choice = "none";
    } else if (req.tool_choice.type === "tool") {
      tool_choice = {
        type: "function",
        function: { name: req.tool_choice.name },
      };
    }
  }
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/harness/loop/tool-format.test.ts
```

Expected: All tests in `tool-format.test.ts` PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration && git add apps/api/src/harness/loop/tool-format.ts apps/api/src/harness/loop/tool-format.test.ts && git commit -m "fix(tool-format): map tool_choice {type:none} to 'none' for Workers AI

runPhase1Streaming forced-final-turn passes tool_choice:{type:'none'}.
Without this mapping it fell through to 'auto', potentially causing
the model to keep calling tools after the turn cap. Refs #69."
```

---

### Task 2: Add `callWorkersAIStream` to `llm.ts`

**Group:** B (depends on Task 1b — needs the updated `AnthropicChatRequest` with `system`)

**Behavior being verified:** `callWorkersAIStream` sends a POST request to the Workers AI
completions endpoint with `stream:true`, both required auth headers, and returns the
response body as a `ReadableStream`. When the upstream returns a non-OK response it throws
`InferenceError`. When the response body is null it throws `InferenceError`.

**Interface under test:** `callWorkersAIStream` exported from
`apps/api/src/services/llm.ts`

**Files:**
- Modify: `apps/api/src/services/llm.ts`
- New: `apps/api/src/services/llm.test.ts`

---

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/services/llm.test.ts`:

```typescript
import { describe, expect, it, vi, beforeEach } from "vitest";
import { callWorkersAIStream } from "./llm";
import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

const mockEnv: Pick<Bindings, "AI_GATEWAY_ENDPOINT" | "AI_GATEWAY_TOKEN" | "CLOUDFLARE_API_TOKEN"> = {
  AI_GATEWAY_ENDPOINT: "https://gateway.example.com",
  AI_GATEWAY_TOKEN: "gw-token",
  CLOUDFLARE_API_TOKEN: "cf-token",
};

const stubBody: import("../harness/loop/tool-format").AnthropicChatRequest = {
  model: "@cf/zai-org/glm-4.7-flash",
  max_tokens: 2048,
  messages: [{ role: "user", content: "Hello" }],
  tools: [],
  tool_choice: { type: "auto" },
};

describe("callWorkersAIStream", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("POSTs to workers-ai completions endpoint with stream:true and both auth headers, returns res.body", async () => {
    const fakeBody = new ReadableStream();
    vi.mocked(fetch).mockResolvedValue(
      new Response(fakeBody, { status: 200 }),
    );

    const result = await callWorkersAIStream(mockEnv as Bindings, stubBody);

    expect(fetch).toHaveBeenCalledOnce();
    const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://gateway.example.com/workers-ai/v1/chat/completions");
    const headers = init.headers as Record<string, string>;
    expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token");
    expect(headers["Authorization"]).toBe("Bearer cf-token");
    const sentBody = JSON.parse(init.body as string) as Record<string, unknown>;
    expect(sentBody.stream).toBe(true);
    expect(result).toBe(fakeBody);
  });

  it("throws InferenceError when upstream returns non-OK status", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("upstream error", { status: 503 }),
    );

    await expect(callWorkersAIStream(mockEnv as Bindings, stubBody)).rejects.toBeInstanceOf(InferenceError);
  });

  it("throws InferenceError when response body is null", async () => {
    vi.mocked(fetch).mockResolvedValue(
      Object.assign(new Response(null, { status: 200 }), { body: null }),
    );

    await expect(callWorkersAIStream(mockEnv as Bindings, stubBody)).rejects.toBeInstanceOf(InferenceError);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/services/llm.test.ts
```

Expected: FAIL — `callWorkersAIStream` is not exported from `./llm`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `apps/api/src/services/llm.ts` (after the existing `callAnthropicStream` export,
before `callWorkersAI`):

```typescript
import type { AnthropicChatRequest } from "../harness/loop/tool-format";
import { toOpenAIChatRequest } from "../harness/loop/tool-format";

export async function callWorkersAIStream(
  env: Bindings,
  body: AnthropicChatRequest,
): Promise<ReadableStream> {
  const url = `${env.AI_GATEWAY_ENDPOINT}/workers-ai/v1/chat/completions`;
  const oaiBody = toOpenAIChatRequest(body);
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
      Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
    },
    body: JSON.stringify({ ...oaiBody, stream: true }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new InferenceError(
      `Workers AI stream request failed: ${res.status} ${text}`,
    );
  }

  if (!res.body) {
    throw new InferenceError("Workers AI stream response has no body");
  }

  return res.body;
}
```

Note: The import of `AnthropicChatRequest` and `toOpenAIChatRequest` must be placed at
the top of the file alongside existing imports. Check whether a circular import would
result: `llm.ts` → `tool-format.ts` → `simplify-schema.ts`. No cycle exists (tool-format
does not import llm). The import is safe.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/services/llm.test.ts
```

Expected: All 3 tests in `llm.test.ts` PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration && git add apps/api/src/services/llm.ts apps/api/src/services/llm.test.ts && git commit -m "feat(llm): add callWorkersAIStream for Workers AI streaming

POSTs to workers-ai/v1/chat/completions with stream:true and both
CF auth headers (cf-aig-authorization + Authorization). Reuses
toOpenAIChatRequest so system blocks and tool_choice fixes from Tasks
1a/1b apply automatically. Refs #69."
```

---

### Task 3: Add `parseOpenAIStream` to `teacher.ts`

**Group:** C (depends on Task 2)

**Behavior being verified:** `parseOpenAIStream` reads an OpenAI-format SSE stream and
yields `TeacherEvent` values identical in shape to those from `parseAnthropicStream`:
`delta` events for text, `tool_start`/`tool_result`/`tool_error` events for tool calls,
and a terminal `done` event with `fullText`, `allComponents`, `toolCalls`, `stopReason`.
It handles three SSE shapes: text-only stream, tool-call delivered in streaming fragments
(one chunk per `index` field per `delta.tool_calls` entry), and tool-call delivered in a
single non-delta chunk (all tool_calls fields present at once).

**Interface under test:** `parseOpenAIStream` exported from
`apps/api/src/services/teacher.ts`

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- New: `apps/api/src/services/teacher.test.ts`

---

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/services/teacher.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { parseOpenAIStream } from "./teacher";
import type { TeacherEvent } from "./teacher";

// Helper: build a ReadableStream from a sequence of SSE text chunks.
function makeStream(chunks: string[]): ReadableStream {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

// Noop tool processor — never called in text-only tests.
const noopTool = async (_name: string, _input: unknown) => ({
  name: "noop",
  componentsJson: [],
  isError: false as const,
});

// Tool processor that returns a fake component for "prescribe_exercise".
const exerciseTool = async (name: string, _input: unknown) => ({
  name,
  componentsJson: [{ type: "exercise_card", data: { id: "ex-1" } }],
  isError: false as const,
});

describe("parseOpenAIStream — text-only response", () => {
  it("yields delta events for each text fragment and a done event with fullText", async () => {
    // OpenAI SSE for a simple text-only response, delivered in two chunks.
    const sseChunks = [
      `data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n`,
      `data: {"choices":[{"delta":{"content":", world"},"finish_reason":null}]}\n\n`,
      `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n`,
      `data: [DONE]\n\n`,
    ];

    const events: TeacherEvent[] = [];
    for await (const ev of parseOpenAIStream(makeStream(sseChunks), noopTool)) {
      events.push(ev);
    }

    const deltas = events.filter((e): e is Extract<TeacherEvent, { type: "delta" }> => e.type === "delta");
    expect(deltas.map((d) => d.text).join("")).toBe("Hello, world");

    const done = events.at(-1);
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.fullText).toBe("Hello, world");
      expect(done.stopReason).toBe("stop");
      expect(done.toolCalls).toHaveLength(0);
    }
  });
});

describe("parseOpenAIStream — streamed tool-call (fragment accumulation)", () => {
  it("accumulates tool_call fragments by index and calls processToolFn, yielding tool_start then tool_result then done", async () => {
    // OpenAI streaming tool-call: id/name in first delta, arguments in subsequent deltas.
    const sseChunks = [
      `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"prescribe_exercise","arguments":""}}]},"finish_reason":null}]}\n\n`,
      `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"drills\\":"}}]},"finish_reason":null}]}\n\n`,
      `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"[\\"scale\\"]}"}}]},"finish_reason":null}]}\n\n`,
      `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n`,
      `data: [DONE]\n\n`,
    ];

    const events: TeacherEvent[] = [];
    for await (const ev of parseOpenAIStream(makeStream(sseChunks), exerciseTool)) {
      events.push(ev);
    }

    expect(events.some((e) => e.type === "tool_start" && e.name === "prescribe_exercise")).toBe(true);
    expect(events.some((e) => e.type === "tool_result" && e.name === "prescribe_exercise")).toBe(true);

    const done = events.at(-1);
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.toolCalls).toHaveLength(1);
      expect(done.toolCalls[0].name).toBe("prescribe_exercise");
      expect(done.stopReason).toBe("tool_calls");
    }
  });
});

describe("parseOpenAIStream — single-chunk tool-call (no streaming fragments)", () => {
  it("handles a model that emits the entire tool_call in one non-delta chunk", async () => {
    // Some models emit tool_calls in finish_reason chunk rather than streaming them.
    const sseChunks = [
      `data: {"choices":[{"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"prescribe_exercise","arguments":"{\\"drills\\":[\\"arpeggio\\"]}"}}]},"finish_reason":"tool_calls"}]}\n\n`,
      `data: [DONE]\n\n`,
    ];

    const events: TeacherEvent[] = [];
    for await (const ev of parseOpenAIStream(makeStream(sseChunks), exerciseTool)) {
      events.push(ev);
    }

    expect(events.some((e) => e.type === "tool_start" && e.name === "prescribe_exercise")).toBe(true);
    expect(events.some((e) => e.type === "tool_result" && e.name === "prescribe_exercise")).toBe(true);

    const done = events.at(-1);
    expect(done?.type).toBe("done");
    if (done?.type === "done") {
      expect(done.toolCalls).toHaveLength(1);
      expect(done.toolCalls[0].id).toBe("call_xyz");
      expect(done.stopReason).toBe("tool_calls");
    }
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/services/teacher.test.ts
```

Expected: FAIL — `parseOpenAIStream` is not exported from `./teacher`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `parseOpenAIStream` to `apps/api/src/services/teacher.ts` **after**
`parseAnthropicStream` (around line 383) and **before** `stripAnalysis`. Do not modify
any existing function.

```typescript
// ---------------------------------------------------------------------------
// parseOpenAIStream
// ---------------------------------------------------------------------------

interface OpenAIStreamToolCallAccumulator {
  id: string;
  name: string;
  argumentsAccumulator: string;
}

export async function* parseOpenAIStream(
  stream: ReadableStream,
  processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
  const decoder = new TextDecoder();
  const reader = stream.getReader();

  const toolAccumulators = new Map<number, OpenAIStreamToolCallAccumulator>();
  const state = {
    fullText: "",
    allComponents: [] as InlineComponent[],
    toolCalls: [] as ToolCallRecord[],
    stopReason: "stop",
    pendingTextDeltas: [] as string[],
    hasToolCallThisTurn: false,
  };

  let textBuffer = "";

  // SSE line parser for OpenAI format: each message is "data: <json>\n\n"
  // or "data: [DONE]\n\n".
  function* parseLines(raw: string): Generator<Record<string, unknown>> {
    const messages = raw.split(/\n\n/);
    for (const message of messages) {
      for (const line of message.split("\n")) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice("data:".length).trim();
        if (payload === "[DONE]") continue;
        let parsed: Record<string, unknown>;
        try {
          parsed = JSON.parse(payload) as Record<string, unknown>;
        } catch (err) {
          console.error(
            JSON.stringify({
              level: "error",
              message: "parseOpenAIStream: failed to parse SSE data",
              payload,
              error: err instanceof Error ? err.message : String(err),
            }),
          );
          Sentry.captureException(err, {
            tags: { service: "teacher", operation: "openai_sse_parse" },
            extra: { payload },
          });
          continue;
        }
        yield parsed;
      }
    }
  }

  // Process a parsed OpenAI SSE chunk. Returns events to yield.
  async function* processChunk(
    parsed: Record<string, unknown>,
  ): AsyncGenerator<TeacherEvent> {
    const choices = parsed["choices"] as Array<Record<string, unknown>> | undefined;
    if (!choices || choices.length === 0) return;

    const choice = choices[0];
    const delta = choice["delta"] as Record<string, unknown> | undefined;
    const finishReason = choice["finish_reason"] as string | null | undefined;

    if (delta) {
      // Text content
      const content = delta["content"] as string | null | undefined;
      if (content) {
        state.fullText += content;
        state.pendingTextDeltas.push(content);
      }

      // Tool calls (streaming fragment accumulation)
      const toolCalls = delta["tool_calls"] as
        | Array<{
            index: number;
            id?: string;
            type?: string;
            function?: { name?: string; arguments?: string };
          }>
        | undefined;

      if (toolCalls) {
        for (const tc of toolCalls) {
          const idx = tc.index;
          if (!toolAccumulators.has(idx)) {
            toolAccumulators.set(idx, {
              id: tc.id ?? "",
              name: tc.function?.name ?? "",
              argumentsAccumulator: "",
            });
            // Discard buffered text — it was intermediate narration
            state.pendingTextDeltas = [];
            state.hasToolCallThisTurn = true;
            const accum = toolAccumulators.get(idx)!;
            if (accum.name) {
              yield { type: "tool_start", name: accum.name };
            }
          } else {
            const accum = toolAccumulators.get(idx)!;
            if (tc.id) accum.id = tc.id;
            if (tc.function?.name) accum.name = tc.function.name;
          }
          if (tc.function?.arguments) {
            toolAccumulators.get(idx)!.argumentsAccumulator += tc.function.arguments;
          }
        }
      }
    }

    // On finish_reason, flush tool accumulators and/or text
    if (finishReason) {
      state.stopReason = finishReason;

      if (finishReason === "tool_calls" || toolAccumulators.size > 0) {
        // Finalize all accumulated tool calls
        const indices = Array.from(toolAccumulators.keys()).sort((a, b) => a - b);
        for (const idx of indices) {
          const accum = toolAccumulators.get(idx)!;

          // Emit tool_start if the name was only delivered in the finish chunk
          // (single-chunk delivery where tool_start wasn't emitted yet because
          //  hasToolCallThisTurn was false at accumulator creation time)
          if (!state.hasToolCallThisTurn) {
            state.hasToolCallThisTurn = true;
            state.pendingTextDeltas = [];
            yield { type: "tool_start", name: accum.name };
          }

          let toolInput: unknown;
          try {
            toolInput = JSON.parse(accum.argumentsAccumulator);
          } catch (err) {
            const parseMsg = err instanceof Error ? err.message : String(err);
            console.error(
              JSON.stringify({
                level: "error",
                message: "parseOpenAIStream: failed to parse tool arguments",
                toolName: accum.name,
                accumulated: accum.argumentsAccumulator,
                error: parseMsg,
              }),
            );
            yield {
              type: "tool_error",
              name: accum.name,
              message: `The model sent malformed input for ${accum.name}: ${parseMsg}`,
            };
            continue;
          }

          const result = await processToolFn(accum.name, toolInput);
          state.toolCalls.push({
            id: accum.id,
            name: accum.name,
            input: toolInput,
            result,
          });
          if (!result.isError) {
            state.allComponents.push(...result.componentsJson);
            yield {
              type: "tool_result",
              name: result.name,
              componentsJson: result.componentsJson,
            };
          } else {
            yield {
              type: "tool_error",
              name: result.name,
              message: result.errorMessage ?? "Tool call failed.",
            };
          }
        }
        toolAccumulators.clear();
      } else {
        // Text finish — flush pending deltas
        if (state.pendingTextDeltas.length > 0) {
          for (const text of state.pendingTextDeltas) {
            yield { type: "delta", text };
          }
          state.pendingTextDeltas = [];
        }
      }
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      textBuffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages; keep partial last message in buffer
      const lastDoubleNewline = textBuffer.lastIndexOf("\n\n");
      if (lastDoubleNewline === -1) continue;

      const toProcess = textBuffer.slice(0, lastDoubleNewline + 2);
      textBuffer = textBuffer.slice(lastDoubleNewline + 2);

      for (const parsed of parseLines(toProcess)) {
        for await (const ev of processChunk(parsed)) {
          yield ev;
        }
      }
    }

    // Flush remaining buffer
    if (textBuffer.trim()) {
      for (const parsed of parseLines(textBuffer)) {
        for await (const ev of processChunk(parsed)) {
          yield ev;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  yield {
    type: "done",
    fullText: state.fullText,
    allComponents: state.allComponents,
    toolCalls: state.toolCalls,
    stopReason: state.stopReason,
  };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/services/teacher.test.ts
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration && git add apps/api/src/services/teacher.ts apps/api/src/services/teacher.test.ts && git commit -m "feat(teacher): add parseOpenAIStream for Workers AI SSE

Reads OpenAI chat/completions SSE (choices[].delta.content,
choices[].delta.tool_calls[]). Accumulates tool-call fragments by
index. Handles single-chunk tool-call delivery (finish_reason chunk
carries full tool_calls). Yields identical TeacherEvent shapes to
parseAnthropicStream. Refs #69."
```

---

### Task 4: Branch `runPhase1Streaming` on provider

**Group:** D (depends on Task 3)

**Behavior being verified:** When `routeModel(ctx.env)` returns `provider:"workers-ai"`,
`runPhase1Streaming` uses `callWorkersAIStream` + `parseOpenAIStream` instead of
`callAnthropicStream` + `parseAnthropicStream`. The multi-turn tool loop and forced-final-
turn (with `tool_choice:"none"`) work identically on both providers. The Anthropic path is
unchanged.

**Interface under test:** `runPhase1Streaming` exported from
`apps/api/src/services/teacher.ts`, exercised through `runStreamingHook` (which is the
call site from `chatV6`).

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Modify: `apps/api/src/harness/loop/runStreamingHook.test.ts`

---

- [ ] **Step 1: Write the failing test**

Add a new `describe` block to
`apps/api/src/harness/loop/runStreamingHook.test.ts`. The existing tests mock
`callAnthropicStream`. Add a parallel mock for `callWorkersAIStream` and verify the
Workers AI path routes correctly.

At the top of `runStreamingHook.test.ts`, the existing mock is:

```typescript
vi.mock("../../services/llm", async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return { ...actual, callAnthropicStream: vi.fn() };
});
import { callAnthropicStream } from "../../services/llm";
```

Replace with (to mock both functions):

```typescript
vi.mock("../../services/llm", async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return { ...actual, callAnthropicStream: vi.fn(), callWorkersAIStream: vi.fn() };
});
import { callAnthropicStream, callWorkersAIStream } from "../../services/llm";
```

Then add this new describe block after the existing ones:

```typescript
describe("runStreamingHook — Workers AI path", () => {
  it("uses callWorkersAIStream when env.TEACHER_PROVIDER is not 'anthropic'", async () => {
    // OpenAI SSE for a simple text response
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
    // Ensure the Anthropic path is NOT called
    vi.mocked(callAnthropicStream).mockRejectedValue(new Error("should not be called"));

    const waiEnv = {
      // TEACHER_PROVIDER absent → routeModel returns workers-ai
    } as never;

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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/harness/loop/runStreamingHook.test.ts
```

Expected: FAIL — `callWorkersAIStream` is called zero times (the current code always
calls `callAnthropicStream`), assertion `expect(callWorkersAIStream).toHaveBeenCalledOnce()` fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/teacher.ts`:

**3a. Add imports** (at the top, alongside existing imports):

```typescript
import { callWorkersAIStream } from "./llm";
import { routeModel } from "../harness/loop/route-model";
import type { AnthropicChatRequest } from "../harness/loop/tool-format";
```

**3b. Modify `runPhase1Streaming`** to branch on provider.

The function currently calls `callAnthropicStream` in two places:
1. The main loop body (line ~417)
2. The forced-final-turn block (line ~510)

Replace both with a provider-branching helper. The full modified section of
`runPhase1Streaming` (only the call sites change — the rest of the function is untouched):

Replace the main-loop call (currently):

```typescript
    const stream = await callAnthropicStream(ctx.env, {
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      system: systemBlocks,
      messages: currentMessages,
      tools: toolSchemas,
      tool_choice: { type: "auto" },
    });

    let doneEvent: TeacherEvent | null = null;
    for await (const event of parseAnthropicStream(stream, processToolFn)) {
```

With:

```typescript
    const client = routeModel("phase1_analysis", ctx.env);
    const chatBody: AnthropicChatRequest = {
      model: client.model,
      max_tokens: 2048,
      system: systemBlocks,
      messages: currentMessages,
      tools: toolSchemas,
      tool_choice: { type: "auto" },
    };
    const stream =
      client.provider === "workers-ai"
        ? await callWorkersAIStream(ctx.env, chatBody)
        : await callAnthropicStream(ctx.env, chatBody);

    let doneEvent: TeacherEvent | null = null;
    const parseStream =
      client.provider === "workers-ai"
        ? parseOpenAIStream(stream, processToolFn)
        : parseAnthropicStream(stream, processToolFn);
    for await (const event of parseStream) {
```

Replace the forced-final-turn call (currently):

```typescript
    const forcedStream = await callAnthropicStream(ctx.env, {
      model: "claude-sonnet-4-20250514",
      max_tokens: 2048,
      system: systemBlocks,
      messages: currentMessages,
      tools: toolSchemas,
      tool_choice: { type: "none" },
    });

    let forcedDone: TeacherEvent | null = null;
    for await (const event of parseAnthropicStream(
      forcedStream,
      processToolFn,
    )) {
```

With:

```typescript
    const forcedClient = routeModel("phase1_analysis", ctx.env);
    const forcedBody: AnthropicChatRequest = {
      model: forcedClient.model,
      max_tokens: 2048,
      system: systemBlocks,
      messages: currentMessages,
      tools: toolSchemas,
      tool_choice: { type: "none" },
    };
    const forcedStream =
      forcedClient.provider === "workers-ai"
        ? await callWorkersAIStream(ctx.env, forcedBody)
        : await callAnthropicStream(ctx.env, forcedBody);

    let forcedDone: TeacherEvent | null = null;
    const parseForcedStream =
      forcedClient.provider === "workers-ai"
        ? parseOpenAIStream(forcedStream, processToolFn)
        : parseAnthropicStream(forcedStream, processToolFn);
    for await (const event of parseForcedStream) {
```

**Note:** The old hardcoded `model: "claude-sonnet-4-20250514"` is replaced with
`client.model` which will be `@cf/zai-org/glm-4.7-flash` on the Workers AI path
(or the value of `TEACHER_MODEL` env var if overridden).

**Tools reliability decision point:**
After this task is implemented and committed, perform live verification (Step 4b below).
If glm streaming tool-calls are unreliable, add `tool_choice: { type: "none" }` to the
Workers AI path's main loop call and file a follow-up issue. The forced-final-turn already
uses `none`, so only the main loop would need patching.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test src/harness/loop/runStreamingHook.test.ts
```

Expected: All tests in `runStreamingHook.test.ts` PASS (existing Anthropic tests + new WAI test).

**Step 4b: Run full test suite — verify no regressions**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration/apps/api && bun run test
```

Expected: Full suite passes. Existing `callAnthropicStream` mock in the original
`runStreamingHook.test.ts` "happy path" test continues to pass because `TEACHER_PROVIDER`
is absent in `stubHookCtx.env` (which is `{} as never`), so `routeModel` returns
`workers-ai` — update the original happy-path test to use
`{ ...stubHookCtx, env: { TEACHER_PROVIDER: "anthropic" } as never }` to keep it
exercising the Anthropic path explicitly. Or add `TEACHER_PROVIDER: "anthropic"` to
`stubHookCtx` — whichever is minimally invasive.

**Step 4c: Live verification (gated on `just api` running)**

```bash
# Start the API only (no MuQ/AMT needed for chat):
# In a separate terminal: just api

# 1. Auth (get a session cookie)
curl -s -c /tmp/crescend-cookies.txt \
  -X POST http://localhost:8787/api/auth/debug \
  -H "Content-Type: application/json" \
  -d '{}' | jq .

# 2. Send a chat message — verify SSE stream arrives and no Anthropic call is made
curl -N -s -b /tmp/crescend-cookies.txt \
  -X POST http://localhost:8787/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, what should I practice today?"}' 2>&1 | head -30

# Expected output includes lines like:
# event: start
# data: {"type":"start","conversationId":"..."}
#
# event: delta
# data: {"type":"delta","text":"..."}
#
# event: done
# data: {"type":"done"}

# 3. Seed a synthesized_facts row for memory-recall test
# Replace <STUDENT_ID> with the id returned by /api/auth/debug above
psql "postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" -c "
INSERT INTO synthesized_facts (id, student_id, dimension, fact_text, is_active, created_at)
VALUES (gen_random_uuid(), '<STUDENT_ID>', 'dynamics',
        'You consistently play the opening phrase too softly.',
        true, now())
ON CONFLICT DO NOTHING;
"

# 4. Memory recall turn
curl -N -s -b /tmp/crescend-cookies.txt \
  -X POST http://localhost:8787/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What have you noticed about my dynamics?"}' 2>&1 | head -40

# Expected: teacher response mentions "opening phrase" or "too softly" — proving
# student_memory was passed to glm.
```

**Tools reliability decision point:** In the live session, send a message that would
trigger `prescribe_exercise` (e.g. "Can you assign me an exercise for my left hand
arpeggios?"). If the response includes a `tool_result` SSE event with an exercise card,
streaming tool-calls work. If the response is text-only with no tool invocation, or if the
stream errors, set `tool_choice: { type: "none" }` on the Workers AI path's main loop
chatBody and file a follow-up issue titled "glm-4.7-flash streaming tool-calls unreliable
— chat text-only mode".

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-69-chat-glm-migration && git add apps/api/src/services/teacher.ts apps/api/src/harness/loop/runStreamingHook.test.ts && git commit -m "feat(teacher): route chat to Workers AI when TEACHER_PROVIDER != anthropic

runPhase1Streaming branches on routeModel(ctx.env).provider:
- workers-ai: callWorkersAIStream + parseOpenAIStream
- anthropic: callAnthropicStream + parseAnthropicStream (unchanged)
Forced-final-turn uses tool_choice:none on both paths (already mapped
correctly by Task 1b). Closes #69."
```

---

## Final Verification Checklist

After all tasks are committed:

- [ ] `cd apps/api && bun run test` — full suite green, zero regressions
- [ ] `TEACHER_PROVIDER` absent (or set to anything other than `"anthropic"`) → chat routes to glm
- [ ] `TEACHER_PROVIDER=anthropic` → chat still routes to Anthropic (regression-safe)
- [ ] Live chat turn surfaces `<student_memory>` facts (memory recall works)
- [ ] No Anthropic API calls appear in the CF AI Gateway dashboard for chat turns
- [ ] If streaming tool-calls work live: document in issue #69 comment. If not: file follow-up and apply `tool_choice:"none"` on WAI path.
