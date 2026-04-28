# Spec X — Harness Streaming + OnChatMessage Migration Implementation Plan

> **For the build agent:** All task groups are sequential — each group depends on the previous. Dispatch one subagent per task in order.
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Route `POST /api/chat` through the V6 harness loop's streaming primitive so both synthesis and chat share a single tool-registration surface, with zero change to the SSE wire protocol the web client consumes.
**Spec:** docs/specs/2026-04-27-spec-x-harness-streaming-design.md
**Style:** Follow `apps/api/TS_STYLE.md`. Never destructure `c.env`. ServiceContext for DI. Explicit exception handling, no silent fallbacks.

---

## Task Groups

```
Group A: Task 1 (CompoundBinding type extension — harness/loop/ only)
Group B (depends on A): Task 2 (buildChatBinding factory — teacher.ts)
Group C (depends on B): Task 3 (runPhase1Streaming text-only — teacher.ts)
Group D (depends on C): Task 4 (runPhase1Streaming tool continuation — teacher.ts)
Group E (depends on D): Task 5 (runPhase1Streaming turn cap — teacher.ts)
Group F (depends on E): Task 6 (chatV6 + equivalence oracle — teacher.ts + new test)
Group G (depends on F): Task 7 (flag dispatch — routes/chat.ts + lib/types.ts + wrangler.toml)
```

---

## Task 1: CompoundBinding Type Extension

**Group:** A

**Behavior being verified:** A `CompoundBinding` object carrying explicit `mode: 'buffered'` and `phases: 2` fields passes through `runPhase1` and produces the same `phase1_done` event as before; and `runHook` skips phase 2 dispatch when the binding lacks `artifactSchema`/`artifactToolName`.

**Interface under test:** `runPhase1(ctx, binding)` accepting the new required fields; `runHook` event sequence unchanged for `OnSessionEnd`.

**Files:**
- Modify: `apps/api/src/harness/loop/types.ts`
- Modify: `apps/api/src/harness/loop/compound-registry.ts`
- Modify: `apps/api/src/harness/loop/runHook.ts`
- Modify: `apps/api/src/harness/loop/phase2.ts`
- Modify: `apps/api/src/harness/loop/phase1.test.ts`
- Modify: `apps/api/src/harness/loop/phase2.test.ts`

- [ ] **Step 1: Write the failing test**

Add this test case inside the existing `describe("runPhase1 empty registry")` block in `apps/api/src/harness/loop/phase1.test.ts`:

```typescript
it("accepts a CompoundBinding with explicit mode and phases fields", async () => {
  const binding: CompoundBinding = {
    compoundName: "session-synthesis",
    procedurePrompt: "test",
    tools: [],
    mode: "buffered",
    phases: 2,
    artifactSchema: SynthesisArtifactSchema,
    artifactToolName: "write_synthesis_artifact",
  };
  fetchSpy.mockResolvedValueOnce(
    new Response(JSON.stringify(ANTHROPIC_END_TURN), { status: 200 }),
  );
  const events: Phase1Event[] = [];
  for await (const ev of runPhase1(PHASE_CTX, binding)) {
    events.push(ev);
  }
  expect(events).toHaveLength(1);
  expect(events[0]).toEqual({ type: "phase1_done", toolCallCount: 0, turnCount: 1 });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test harness/loop/phase1.test.ts
```

Expected: FAIL — TypeScript compile error: `Object literal may only specify known properties, and 'mode' does not exist in type 'CompoundBinding'`

- [ ] **Step 3: Implement the minimum to make the test pass**

**`apps/api/src/harness/loop/types.ts`** — replace `CompoundBinding` and add `Phase2Binding`:

```typescript
export interface CompoundBinding {
  compoundName: string;
  procedurePrompt: string;
  tools: ToolDefinition[];
  mode: "streaming" | "buffered";
  phases: 1 | 2;
  artifactSchema?: ZodTypeAny;
  artifactToolName?: string;
}

export type Phase2Binding = CompoundBinding & {
  artifactSchema: ZodTypeAny;
  artifactToolName: string;
};
```

**`apps/api/src/harness/loop/compound-registry.ts`** — add `mode` and `phases` to the `OnSessionEnd` binding:

```typescript
const REGISTRY: Map<HookKind, CompoundBinding> = new Map([
  [
    "OnSessionEnd" as const,
    {
      compoundName: "session-synthesis",
      procedurePrompt: SESSION_SYNTHESIS_PROCEDURE,
      tools: [],
      mode: "buffered" as const,
      phases: 2 as const,
      artifactSchema: SynthesisArtifactSchema,
      artifactToolName: "write_synthesis_artifact",
    },
  ],
]);
```

**`apps/api/src/harness/loop/runHook.ts`** — add `Phase2Binding` import and `isPhase2Binding` guard, wrap phase2 dispatch:

```typescript
import type { HookContext, HookEvent, HookKind, PhaseContext, Phase2Binding } from "./types";

// (add after imports, before runHook)
function isPhase2Binding(b: CompoundBinding): b is Phase2Binding {
  return b.phases === 2 && b.artifactSchema !== undefined && b.artifactToolName !== undefined;
}

// In runHook body: replace the unconditional runPhase2 try/catch with:
  if (isPhase2Binding(binding)) {
    try {
      for await (const ev of runPhase2(phaseCtx, binding, collectedDiagnoses)) {
        yield ev as HookEvent<ArtifactFor<H>>;
      }
    } catch (err) {
      yield {
        type: "phase_error",
        phase: 2,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }
```

The existing import `import type { HookContext, HookEvent, HookKind, PhaseContext } from "./types"` must also add `Phase2Binding`. Add `CompoundBinding` to the type import as well (it's needed for `isPhase2Binding`'s parameter):

```typescript
import type { CompoundBinding, HookContext, HookEvent, HookKind, PhaseContext, Phase2Binding } from "./types";
```

**`apps/api/src/harness/loop/phase2.ts`** — change the `binding` parameter from `CompoundBinding` to `Phase2Binding`:

```typescript
import type { Phase2Binding, HookEvent, PhaseContext } from "./types";

export async function* runPhase2(
  ctx: PhaseContext,
  binding: Phase2Binding,
  diagnoses: unknown[],
): AsyncGenerator<HookEvent<unknown>> {
```

The rest of `phase2.ts` is unchanged — `binding.artifactSchema` and `binding.artifactToolName` are now non-optional on `Phase2Binding`, so TypeScript accepts them without assertions.

**`apps/api/src/harness/loop/phase1.test.ts`** — update `EMPTY_BINDING` and `capBinding` fixtures to include `mode` and `phases`:

```typescript
const EMPTY_BINDING: CompoundBinding = {
  compoundName: "session-synthesis",
  procedurePrompt: "test",
  tools: [],
  mode: "buffered",
  phases: 2,
  artifactSchema: SynthesisArtifactSchema,
  artifactToolName: "write_synthesis_artifact",
};
```

Inside the turn-cap test, update `capBinding`:

```typescript
const capBinding: CompoundBinding = {
  compoundName: "session-synthesis",
  procedurePrompt: "test",
  tools: [
    {
      name: "dummy_tool",
      description: "test",
      input_schema: { type: "object" },
      invoke: async () => ({ ok: true }),
    },
  ],
  mode: "buffered",
  phases: 2,
  artifactSchema: SynthesisArtifactSchema,
  artifactToolName: "write_synthesis_artifact",
};
```

**`apps/api/src/harness/loop/phase2.test.ts`** — update `BINDING` fixture:

```typescript
const BINDING: CompoundBinding = {
  compoundName: "session-synthesis",
  procedurePrompt: "test",
  tools: [],
  mode: "buffered",
  phases: 2,
  artifactSchema: SynthesisArtifactSchema,
  artifactToolName: "write_synthesis_artifact",
};
```

Note: `phase2.ts` now accepts `Phase2Binding`, but `CompoundBinding` with `mode: 'buffered', phases: 2, artifactSchema, artifactToolName` satisfies `Phase2Binding` structurally. The test fixture annotation `const BINDING: CompoundBinding` still compiles; TypeScript checks structural compatibility when it is passed to `runPhase2(ctx, BINDING, [])`.

Note: `runHook.test.ts` requires no fixture changes — its `HOOK_CTX` is unchanged, and the `OnSessionEnd` binding is fetched from the registry (which is already updated).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test harness/loop/
```

Expected: PASS — all existing loop tests plus the new `mode`/`phases` test pass.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/types.ts \
        apps/api/src/harness/loop/compound-registry.ts \
        apps/api/src/harness/loop/runHook.ts \
        apps/api/src/harness/loop/phase2.ts \
        apps/api/src/harness/loop/phase1.test.ts \
        apps/api/src/harness/loop/phase2.test.ts && \
git commit -m "feat(harness): add mode/phases to CompoundBinding; add Phase2Binding type guard"
```

---

## Task 2: buildChatBinding Factory

**Group:** B (depends on Task 1)

**Behavior being verified:** `buildChatBinding(ctx, studentId)` returns a `CompoundBinding` with `mode: 'streaming'`, `phases: 1`, and one harness `ToolDefinition` entry for each key in `TOOL_REGISTRY`, each with a non-empty `input_schema` object and an `invoke` function.

**Interface under test:** `buildChatBinding` (new export from `services/teacher.ts`)

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Create: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

Create `apps/api/src/services/teacher-chat-v6.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import type { Bindings, Db, ServiceContext } from "../lib/types";
import { buildChatBinding } from "./teacher";
import { TOOL_REGISTRY } from "./tool-processor";

const MOCK_ENV = {
  AI_GATEWAY_TEACHER: "https://gw.example",
  ANTHROPIC_API_KEY: "test-key",
} as unknown as Bindings;

const MOCK_CTX: ServiceContext = {
  db: {} as Db,
  env: MOCK_ENV,
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: FAIL — `SyntaxError: The requested module './teacher' does not provide an export named 'buildChatBinding'`

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/teacher.ts`, update the import from `"../harness/loop/types"` (line 5) to add `CompoundBinding` and `PhaseContext`:

```typescript
import type { CompoundBinding, HookContext, HookEvent, PhaseContext } from "../harness/loop/types";
```

Add a new import for `routeModel` after the existing harness imports:

```typescript
import { routeModel } from "../harness/loop/route-model";
```

Update the import from `"./tool-processor"` (lines 15–20) to add `TOOL_REGISTRY`:

```typescript
import {
  getAnthropicToolSchemas,
  type InlineComponent,
  processToolUse,
  TOOL_REGISTRY,
  type ToolResult,
} from "./tool-processor";
```

Add `buildChatBinding` after the existing `// Types` section and before `chat`:

```typescript
// ---------------------------------------------------------------------------
// buildChatBinding
// ---------------------------------------------------------------------------

export function buildChatBinding(ctx: ServiceContext, studentId: string): CompoundBinding {
  return {
    compoundName: "chat-response",
    procedurePrompt: "",
    mode: "streaming",
    phases: 1,
    tools: Object.values(TOOL_REGISTRY).map((t) => ({
      name: t.name,
      description: t.description,
      // input_schema cast: AnthropicToolSchema.input_schema is an object subset of Record<string, unknown>
      input_schema: t.anthropicSchema.input_schema as Record<string, unknown>,
      // invoke satisfies the binding contract; the streaming path uses processToolFn instead
      // to preserve the ToolResult.componentsJson shape required for SSE rendering.
      invoke: async (input: unknown) => processToolUse(ctx, studentId, t.name, input),
    })),
  };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts \
        apps/api/src/services/teacher-chat-v6.test.ts && \
git commit -m "feat(teacher): add buildChatBinding factory for OnChatMessage harness binding"
```

---

## Task 3: runPhase1Streaming — Text-Only Turn

**Group:** C (depends on Task 2)

**Behavior being verified:** `runPhase1Streaming` called with a mocked Anthropic response containing a single text turn yields one `delta` event with the token text and one `done` event with `fullText`, `stopReason: 'end_turn'`, and empty `allComponents`.

**Interface under test:** `runPhase1Streaming` (new export from `services/teacher.ts`)

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

Add the following to `apps/api/src/services/teacher-chat-v6.test.ts` — new imports at the top, then new describe block:

```typescript
// Add to imports:
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import type { TeacherEvent } from "./teacher";
import { buildChatBinding, runPhase1Streaming } from "./teacher";

// Add shared helpers and constants (before the describe blocks):
function makeSseResponse(sseText: string): Response {
  return new Response(new TextEncoder().encode(sseText), { status: 200 });
}

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

// New describe block:
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: FAIL — `SyntaxError: The requested module './teacher' does not provide an export named 'runPhase1Streaming'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `runPhase1Streaming` to `apps/api/src/services/teacher.ts` after `buildChatBinding`. This is the minimum single-turn implementation (multi-turn loop added in Task 4):

```typescript
// ---------------------------------------------------------------------------
// runPhase1Streaming
// ---------------------------------------------------------------------------

export async function* runPhase1Streaming(
  ctx: PhaseContext,
  binding: CompoundBinding,
  systemBlocks: AnthropicSystemBlock[],
  initialMessages: Array<{ role: "user" | "assistant"; content: string | AnthropicContentBlock[] }>,
  processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
  const client = routeModel("phase1_analysis");
  const toolSchemas = binding.tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema,
  }));

  const stream = await callAnthropicStream(ctx.env, {
    model: client.model,
    max_tokens: 2048,
    system: systemBlocks,
    messages: initialMessages,
    tools: toolSchemas,
    tool_choice: { type: "auto" },
  });

  let doneEvent: TeacherEvent | null = null;
  for await (const event of parseAnthropicStream(stream, processToolFn)) {
    if (event.type === "done") {
      doneEvent = event;
    } else {
      yield event;
    }
  }

  if (doneEvent && doneEvent.type === "done") {
    yield doneEvent;
  }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts \
        apps/api/src/services/teacher-chat-v6.test.ts && \
git commit -m "feat(teacher): add runPhase1Streaming — text-only single-turn streaming loop"
```

---

## Task 4: runPhase1Streaming — Tool Continuation

**Group:** D (depends on Task 3)

**Behavior being verified:** When the first Anthropic stream contains a `tool_use` block, `runPhase1Streaming` dispatches the tool via `processToolFn`, builds continuation messages from the result, calls Anthropic a second time, and accumulates tool components into the final `done` event.

**Interface under test:** `runPhase1Streaming` (extended to multi-turn)

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

Add constants and a new describe block to `apps/api/src/services/teacher-chat-v6.test.ts`:

```typescript
// Add after TEXT_ONLY_SSE constant:
const TOOL_USE_SSE = [
  'event: message_start\ndata: {"type":"message_start"}\n\n',
  'event: content_block_start\ndata: {"index":0,"content_block":{"type":"tool_use","id":"tu_1","name":"search_catalog"}}\n\n',
  'event: content_block_delta\ndata: {"index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"composer\\":\\"Chopin\\"}"}}\n\n',
  'event: content_block_stop\ndata: {"index":0}\n\n',
  'event: message_delta\ndata: {"delta":{"stop_reason":"tool_use"}}\n\n',
  'event: message_stop\ndata: {}\n\n',
].join("");

// New describe block:
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: FAIL — the current single-turn implementation exits after the first stream; `fetchSpy` is called only once instead of twice, and `allComponents` is empty.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the `runPhase1Streaming` function body in `apps/api/src/services/teacher.ts` with the multi-turn loop (the turn-cap forced-call path is added in Task 5):

```typescript
export async function* runPhase1Streaming(
  ctx: PhaseContext,
  binding: CompoundBinding,
  systemBlocks: AnthropicSystemBlock[],
  initialMessages: Array<{ role: "user" | "assistant"; content: string | AnthropicContentBlock[] }>,
  processToolFn: ProcessToolFn,
): AsyncGenerator<TeacherEvent> {
  const client = routeModel("phase1_analysis");
  const toolSchemas = binding.tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.input_schema,
  }));

  let currentMessages = initialMessages;
  const accumulatedComponents: InlineComponent[] = [];

  for (let turn = 0; turn < ctx.turnCap; turn++) {
    const stream = await callAnthropicStream(ctx.env, {
      model: client.model,
      max_tokens: 2048,
      system: systemBlocks,
      messages: currentMessages,
      tools: toolSchemas,
      tool_choice: { type: "auto" },
    });

    let doneEvent: TeacherEvent | null = null;
    for await (const event of parseAnthropicStream(stream, processToolFn)) {
      if (event.type === "done") {
        doneEvent = event;
      } else {
        yield event;
      }
    }

    if (!doneEvent || doneEvent.type !== "done") break;

    accumulatedComponents.push(...doneEvent.allComponents);

    if (doneEvent.toolCalls.length === 0 || doneEvent.stopReason !== "tool_use") {
      yield { ...doneEvent, allComponents: accumulatedComponents };
      return;
    }

    const assistantContent: AnthropicContentBlock[] = [];
    if (doneEvent.fullText) {
      assistantContent.push({ type: "text", text: doneEvent.fullText });
    }
    for (const tc of doneEvent.toolCalls) {
      assistantContent.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.input });
    }

    const GENERIC_TOOL_ERROR =
      "Tool call failed validation. For piece_id, pass through the exact slug returned by search_catalog (e.g. 'chopin.ballades.1'); do not transform or invent it. Check all required fields and try again.";

    const toolResultContent: AnthropicContentBlock[] = doneEvent.toolCalls.map((tc) => {
      if (tc.result.isError) {
        return {
          type: "tool_result" as const,
          tool_use_id: tc.id,
          is_error: true,
          content: tc.result.errorMessage ?? GENERIC_TOOL_ERROR,
        };
      }
      return {
        type: "tool_result" as const,
        tool_use_id: tc.id,
        content: JSON.stringify(tc.result.componentsJson),
      };
    });

    currentMessages = [
      ...currentMessages,
      { role: "assistant" as const, content: assistantContent },
      { role: "user" as const, content: toolResultContent },
    ];

    console.log(
      JSON.stringify({
        level: "info",
        message: "streaming chat tool continuation",
        turn: turn + 1,
        toolCount: doneEvent.toolCalls.length,
        toolNames: doneEvent.toolCalls.map((tc) => tc.name),
      }),
    );
  }

  // Turn cap exhausted — placeholder yield; replaced in Task 5 with forced call
  yield {
    type: "done",
    fullText: "I had trouble putting that together — could you ask again?",
    allComponents: accumulatedComponents,
    toolCalls: [],
    stopReason: "max_tool_turns",
  };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts \
        apps/api/src/services/teacher-chat-v6.test.ts && \
git commit -m "feat(teacher): extend runPhase1Streaming to multi-turn tool continuation loop"
```

---

## Task 5: runPhase1Streaming — Turn Cap Exhaustion

**Group:** E (depends on Task 4)

**Behavior being verified:** When `ctx.turnCap` tool-calling turns are exhausted, `runPhase1Streaming` issues one additional `callAnthropicStream` call with `tool_choice: { type: 'none' }` and yields a final `done` event with `stopReason: 'forced_text_after_max_turns'`.

**Interface under test:** `runPhase1Streaming`

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/services/teacher-chat-v6.test.ts`:

```typescript
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
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TOOL_USE_SSE)))
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TOOL_USE_SSE)))
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)));

    const binding = buildChatBinding(MOCK_CTX, "stu_1");
    const systemBlocks = [{ type: "text" as const, text: "You are a teacher." }];
    const messages = [{ role: "user" as const, content: "Find me some Chopin." }];

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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: FAIL — `fetchSpy` is called only twice (the forced call never happens); `stopReason` is `'max_tool_turns'` not `'forced_text_after_max_turns'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/services/teacher.ts`, replace the tail of `runPhase1Streaming` — the comment and the placeholder `yield` after the `for` loop — with the forced-call path:

```typescript
  // Turn cap exhausted — force a text response with tool_choice: none
  try {
    const forcedStream = await callAnthropicStream(ctx.env, {
      model: client.model,
      max_tokens: 2048,
      system: systemBlocks,
      messages: currentMessages,
      tools: toolSchemas,
      tool_choice: { type: "none" },
    });

    let forcedDone: TeacherEvent | null = null;
    for await (const event of parseAnthropicStream(forcedStream, processToolFn)) {
      if (event.type === "done") {
        forcedDone = event;
      } else {
        yield event;
      }
    }

    if (forcedDone && forcedDone.type === "done" && forcedDone.fullText) {
      yield {
        type: "done",
        fullText: forcedDone.fullText,
        allComponents: accumulatedComponents,
        toolCalls: [],
        stopReason: "forced_text_after_max_turns",
      };
      return;
    }
  } catch (err) {
    console.error(
      JSON.stringify({
        level: "error",
        message: "forced final call failed after max tool turns",
        error: err instanceof Error ? err.message : String(err),
        stack: err instanceof Error ? err.stack : undefined,
      }),
    );
    Sentry.captureException(err, {
      tags: { service: "teacher", operation: "streaming_forced_final_call" },
    });
  }

  yield {
    type: "done",
    fullText: "I had trouble putting that together — could you ask again?",
    allComponents: accumulatedComponents,
    toolCalls: [],
    stopReason: "max_tool_turns",
  };
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts \
        apps/api/src/services/teacher-chat-v6.test.ts && \
git commit -m "feat(teacher): add turn-cap exhaustion forced-call path to runPhase1Streaming"
```

---

## Task 6: chatV6 + Equivalence Oracle

**Group:** F (depends on Task 5)

**Behavior being verified:** `chatV6(ctx, studentId, messages, dynamicContext)` produces an identical `TeacherEvent` array to `chat(ctx, studentId, messages, dynamicContext)` when both consume the same mocked Anthropic SSE stream.

**Interface under test:** `chatV6` (new export from `services/teacher.ts`); compared against `chat` via oracle.

**Files:**
- Modify: `apps/api/src/services/teacher.ts`
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/services/teacher-chat-v6.test.ts`:

```typescript
// Add to imports at top of file:
import { buildChatBinding, chat, chatV6, runPhase1Streaming } from "./teacher";

// New describe block:
describe("chatV6 equivalence oracle", () => {
  const fetchSpy = vi.fn();

  beforeEach(() => {
    fetchSpy.mockReset();
    vi.stubGlobal("fetch", fetchSpy);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("produces the same TeacherEvent sequence as chat() for a text-only turn", async () => {
    // Each call needs its own Response — ReadableStream bodies are consumed once.
    // makeSseResponse creates a fresh Uint8Array-backed body each time.
    fetchSpy
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)))
      .mockImplementationOnce(() => Promise.resolve(makeSseResponse(TEXT_ONLY_SSE)));

    const studentId = "stu_1";
    const messages = [{ role: "user" as const, content: "What should I practice?" }];
    const dynamicContext = "Student level: beginner.";

    const legacyEvents: TeacherEvent[] = [];
    for await (const ev of chat(MOCK_CTX, studentId, messages, dynamicContext)) {
      legacyEvents.push(ev);
    }

    const harnessEvents: TeacherEvent[] = [];
    for await (const ev of chatV6(MOCK_CTX, studentId, messages, dynamicContext)) {
      harnessEvents.push(ev);
    }

    expect(harnessEvents).toHaveLength(legacyEvents.length);
    for (let i = 0; i < legacyEvents.length; i++) {
      expect(harnessEvents[i]).toEqual(legacyEvents[i]);
    }
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: FAIL — `SyntaxError: The requested module './teacher' does not provide an export named 'chatV6'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `chatV6` to `apps/api/src/services/teacher.ts` after `runPhase1Streaming`. The constant `MAX_TOOL_TURNS` is already defined at line 351 in teacher.ts; `chatV6` reuses it:

```typescript
// ---------------------------------------------------------------------------
// chatV6
// ---------------------------------------------------------------------------

export async function* chatV6(
  ctx: ServiceContext,
  studentId: string,
  messages: Array<{ role: "user" | "assistant"; content: string | AnthropicContentBlock[] }>,
  dynamicContext: string,
): AsyncGenerator<TeacherEvent> {
  const systemBlocks: AnthropicSystemBlock[] = [
    {
      type: "text",
      text: UNIFIED_TEACHER_SYSTEM,
      cache_control: { type: "ephemeral" },
    },
    ...(dynamicContext.trim()
      ? [{ type: "text" as const, text: dynamicContext }]
      : []),
  ];

  const processToolFn: ProcessToolFn = async (name, input) =>
    processToolUse(ctx, studentId, name, input);

  const binding = buildChatBinding(ctx, studentId);
  const phaseCtx: PhaseContext = {
    env: ctx.env,
    studentId,
    sessionId: "",
    conversationId: null,
    digest: {},
    waitUntil: (_p: Promise<unknown>) => {},
    turnCap: MAX_TOOL_TURNS,
  };

  yield* runPhase1Streaming(phaseCtx, binding, systemBlocks, messages, processToolFn);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: PASS — both paths yield identical `TeacherEvent` arrays.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts \
        apps/api/src/services/teacher-chat-v6.test.ts && \
git commit -m "feat(teacher): add chatV6 adapter; equivalence oracle confirms identical TeacherEvent output"
```

---

## Task 7: Flag Dispatch

**Group:** G (depends on Task 6)

**Behavior being verified:** `HARNESS_V6_CHAT_ENABLED` is a valid key in the `Bindings` type; and when the flag equals `'true'`, the chat route dispatches to `teacherService.chatV6` rather than `teacherService.chat`.

**Interface under test:** `Bindings` type (compile-time); `routes/chat.ts` dispatch branch.

**Files:**
- Modify: `apps/api/src/lib/types.ts`
- Modify: `apps/api/wrangler.toml`
- Modify: `apps/api/src/routes/chat.ts`
- Modify: `apps/api/src/services/teacher-chat-v6.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `apps/api/src/services/teacher-chat-v6.test.ts`:

```typescript
// Add to top-level imports:
import type { Bindings } from "../lib/types";

// New describe block:
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Expected: FAIL — TypeScript compile error: `Type '"HARNESS_V6_CHAT_ENABLED"' does not satisfy the constraint 'keyof Bindings'`

- [ ] **Step 3: Implement the minimum to make the test pass**

**`apps/api/src/lib/types.ts`** — add `HARNESS_V6_CHAT_ENABLED` after `HARNESS_V6_ENABLED` (line 28):

```typescript
  HARNESS_V6_ENABLED: string;
  HARNESS_V6_CHAT_ENABLED: string;
```

**`apps/api/wrangler.toml`** — add after line 28 (`HARNESS_V6_ENABLED = "false"`):

```toml
HARNESS_V6_CHAT_ENABLED = "false"
```

**`apps/api/src/routes/chat.ts`** — replace the call to `teacherService.chat` in the `streamSSE` handler with a flag-gated dispatch. The current code (lines 43–48):

```typescript
// Before:
for await (const event of teacherService.chat(
  ctx,
  studentId,
  messages,
  dynamicContext,
)) {
```

Replace with:

```typescript
// After:
const teacherFn =
  c.env.HARNESS_V6_CHAT_ENABLED === "true"
    ? teacherService.chatV6
    : teacherService.chat;

for await (const event of teacherFn(
  ctx,
  studentId,
  messages,
  dynamicContext,
)) {
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test services/teacher-chat-v6.test.ts
```

Then run the full test suite to verify no regressions:

```bash
cd apps/api && bun run test
```

Expected: PASS — all tests including the new Bindings type test and all pre-existing harness loop tests.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/lib/types.ts \
        apps/api/wrangler.toml \
        apps/api/src/routes/chat.ts \
        apps/api/src/services/teacher-chat-v6.test.ts && \
git commit -m "feat(chat): add HARNESS_V6_CHAT_ENABLED flag; route dispatches to chatV6 when enabled"
```
