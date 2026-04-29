# Spec X — Chat Harness Migration Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Route live chat through the V6 harness registry so that `chatV6` calls `runStreamingHook("OnChatMessage")` and the legacy `chat()` and `buildChatBinding()` functions are deleted.
**Spec:** docs/specs/2026-04-29-spec-x-chat-harness-migration-design.md
**Style:** Follow apps/api/TS_STYLE.md for all code under apps/api/src/.

## Task Groups
Group A (parallel): Task 1, Task 2
Group B (sequential, depends on A): Task 3
Group C (sequential, depends on B): Task 4
Group D (sequential, depends on C): Task 5
Group E (sequential, depends on D): Task 6

---

### Task 1: Add ConfigError to lib/errors.ts
**Group:** A (parallel with Task 2)

**Behavior being verified:** `ConfigError` is a `DomainError` subclass whose `name` property is `"ConfigError"`.
**Interface under test:** `new ConfigError(message)` constructor

**Files:**
- Modify: `apps/api/src/lib/errors.ts`
- Test: `apps/api/src/lib/errors.test.ts` (new)

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it } from "vitest";
import { ConfigError, DomainError } from "./errors";

describe("ConfigError", () => {
  it("is a DomainError with name ConfigError", () => {
    const err = new ConfigError("missing binding");
    expect(err).toBeInstanceOf(DomainError);
    expect(err.name).toBe("ConfigError");
    expect(err.message).toBe("missing binding");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/lib/errors.test.ts
```
Expected: FAIL — `ConfigError is not exported from "./errors"`

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to end of `apps/api/src/lib/errors.ts`:

```typescript
export class ConfigError extends DomainError {
	constructor(message: string) {
		super(message);
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/lib/errors.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/lib/errors.ts apps/api/src/lib/errors.test.ts && git commit -m "feat(errors): add ConfigError domain error class"
```

---

### Task 2: Add OnChatMessage binding to compound-registry
**Group:** A (parallel with Task 1)

**Behavior being verified:** `getCompoundBinding("OnChatMessage")` returns a binding with `mode: "streaming"`, `phases: 1`, and 6 tools matching the TOOL_REGISTRY keys.
**Interface under test:** `getCompoundBinding("OnChatMessage")`

**Files:**
- Modify: `apps/api/src/harness/loop/compound-registry.ts`
- Modify: `apps/api/src/harness/loop/compound-registry.test.ts`

- [ ] **Step 1: Write the failing test**

Add a new `it` block inside the existing `describe("compound-registry")` in `apps/api/src/harness/loop/compound-registry.test.ts`:

```typescript
it("returns a streaming binding for OnChatMessage with 6 tools", () => {
  const binding = getCompoundBinding("OnChatMessage");
  expect(binding).toBeDefined();
  expect(binding?.compoundName).toBe("chat-response");
  expect(binding?.mode).toBe("streaming");
  expect(binding?.phases).toBe(1);
  expect(binding?.tools).toHaveLength(6);
  const names = binding!.tools.map((t) => t.name);
  expect(new Set(names).size).toBe(names.length);
  expect(names).toContain("create_exercise");
  expect(names).toContain("search_catalog");
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/harness/loop/compound-registry.test.ts
```
Expected: FAIL — `expect(binding).toBeDefined()` fails (returns undefined)

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the import and map entry to `apps/api/src/harness/loop/compound-registry.ts`:

At the top of the file, after the existing imports, add:
```typescript
import { TOOL_REGISTRY } from "../../services/tool-processor";
```

Change the `REGISTRY` declaration from `const REGISTRY: Map<HookKind, CompoundBinding> = new Map([` and add the OnChatMessage entry. The full updated REGISTRY:

```typescript
const REGISTRY: Map<HookKind, CompoundBinding> = new Map([
	[
		"OnSessionEnd" as const,
		{
			compoundName: "session-synthesis",
			procedurePrompt: SESSION_SYNTHESIS_PROCEDURE,
			tools: [...ALL_MOLECULES],
			mode: "buffered" as const,
			phases: 2 as const,
			artifactSchema: SynthesisArtifactSchema,
			artifactToolName: "write_synthesis_artifact",
		},
	],
	[
		"OnChatMessage" as const,
		{
			compoundName: "chat-response",
			procedurePrompt: "",
			tools: Object.values(TOOL_REGISTRY).map((t) => ({
				name: t.name,
				description: t.description,
				input_schema: t.anthropicSchema.input_schema as Record<string, unknown>,
				invoke: async (_input: unknown): Promise<unknown> => ({}),
			})),
			mode: "streaming" as const,
			phases: 1 as const,
		},
	],
]);
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/harness/loop/compound-registry.test.ts
```
Expected: PASS (all 4 tests including the existing ones)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/compound-registry.ts apps/api/src/harness/loop/compound-registry.test.ts && git commit -m "feat(registry): add OnChatMessage streaming binding"
```

---

### Task 3: Create runStreamingHook.ts with error guards
**Group:** B (depends on Group A)

**Behavior being verified:** `runStreamingHook` throws `ConfigError` when called with an unregistered hook, and when the registered binding has `mode !== "streaming"`.
**Interface under test:** `runStreamingHook(hook, hookCtx, processToolFn, systemBlocks, initialMessages)`

**Files:**
- Create: `apps/api/src/harness/loop/runStreamingHook.ts`
- Create: `apps/api/src/harness/loop/runStreamingHook.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it, vi } from "vitest";
import { ConfigError } from "../../lib/errors";
import { runStreamingHook } from "./runStreamingHook";
import type { HookContext } from "./types";

vi.mock("./compound-registry", () => ({
  getCompoundBinding: vi.fn(),
}));

import { getCompoundBinding } from "./compound-registry";

const stubHookCtx: HookContext = {
  env: {} as never,
  studentId: "s1",
  sessionId: "",
  conversationId: null,
  digest: {},
  waitUntil: () => {},
};

describe("runStreamingHook error paths", () => {
  it("throws ConfigError when hook has no registered binding", async () => {
    vi.mocked(getCompoundBinding).mockReturnValue(undefined);
    const gen = runStreamingHook("OnStop", stubHookCtx, async () => ({} as never), [], []);
    await expect(gen.next()).rejects.toBeInstanceOf(ConfigError);
  });

  it("throws ConfigError when binding mode is not streaming", async () => {
    vi.mocked(getCompoundBinding).mockReturnValue({
      compoundName: "x",
      procedurePrompt: "",
      tools: [],
      mode: "buffered",
      phases: 2,
      artifactSchema: undefined,
      artifactToolName: undefined,
    } as never);
    const gen = runStreamingHook("OnSessionEnd", stubHookCtx, async () => ({} as never), [], []);
    await expect(gen.next()).rejects.toBeInstanceOf(ConfigError);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/harness/loop/runStreamingHook.test.ts
```
Expected: FAIL — `Cannot find module './runStreamingHook'`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/api/src/harness/loop/runStreamingHook.ts`:

```typescript
import { ConfigError } from "../../lib/errors";
import { getCompoundBinding } from "./compound-registry";
import type { HookKind, HookContext, PhaseContext } from "./types";
import type { ToolResult } from "../../services/tool-processor";
import type { AnthropicContentBlock, AnthropicSystemBlock } from "../../services/llm";
import type { TeacherEvent } from "../../services/teacher";

type ProcessToolFn = (name: string, input: unknown) => Promise<ToolResult>;

const DEFAULT_TURN_CAP = 5;

export async function* runStreamingHook(
	hook: HookKind,
	hookCtx: HookContext,
	processToolFn: ProcessToolFn,
	systemBlocks: AnthropicSystemBlock[],
	initialMessages: Array<{
		role: "user" | "assistant";
		content: string | AnthropicContentBlock[];
	}>,
): AsyncGenerator<TeacherEvent> {
	const binding = getCompoundBinding(hook);
	if (!binding) {
		throw new ConfigError(
			`runStreamingHook: no binding registered for hook "${hook}"`,
		);
	}
	if (binding.mode !== "streaming") {
		throw new ConfigError(
			`runStreamingHook: binding for "${hook}" has mode "${binding.mode}", expected "streaming"`,
		);
	}
	// Happy path delegation added in Task 4.
	const _phaseCtx: PhaseContext = {
		...hookCtx,
		turnCap: DEFAULT_TURN_CAP,
	};
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/harness/loop/runStreamingHook.test.ts
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/runStreamingHook.ts apps/api/src/harness/loop/runStreamingHook.test.ts && git commit -m "feat(harness): add runStreamingHook with ConfigError guards"
```

---

### Task 4: Complete runStreamingHook happy path
**Group:** C (depends on Group B — Task 3)

**Behavior being verified:** `runStreamingHook("OnChatMessage", ...)` with a valid streaming binding yields `TeacherEvent` delta and done events forwarded from the underlying Anthropic stream.
**Interface under test:** `runStreamingHook("OnChatMessage", hookCtx, processToolFn, systemBlocks, messages)`

**Files:**
- Modify: `apps/api/src/harness/loop/runStreamingHook.ts`
- Modify: `apps/api/src/harness/loop/runStreamingHook.test.ts`

- [ ] **Step 1: Write the failing test**

Add the following helper and test to `apps/api/src/harness/loop/runStreamingHook.test.ts`.

First add a new import at the top of the file (keep existing imports):
```typescript
import { vi, describe, expect, it, beforeEach } from "vitest";
```

Add the following at the end of the test file (do NOT replace the existing mocks and tests):

```typescript
// ---------------------------------------------------------------------------
// Happy path — uses real compound-registry binding, mocks Anthropic HTTP
// ---------------------------------------------------------------------------

vi.mock("../../services/llm", async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return { ...actual, callAnthropicStream: vi.fn() };
});

import { callAnthropicStream } from "../../services/llm";

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

describe("runStreamingHook happy path", () => {
  beforeEach(() => {
    vi.mocked(getCompoundBinding).mockRestore?.();
  });

  it("yields delta and done events for OnChatMessage with a streaming response", async () => {
    // Use the REAL compound-registry binding (OnChatMessage, mode: streaming)
    const { getCompoundBinding: real } = await vi.importActual<typeof import("./compound-registry")>("./compound-registry");
    vi.mocked(getCompoundBinding).mockImplementation(real);
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
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/harness/loop/runStreamingHook.test.ts
```
Expected: FAIL — happy path test does not yield any events (generator returns immediately without yielding)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the `runStreamingHook.ts` body after the mode check. Remove the stub `_phaseCtx` block and add the delegation import + yield*:

Add to imports in `runStreamingHook.ts`:
```typescript
import { runPhase1Streaming } from "../../services/teacher";
```

Replace the body after `if (binding.mode !== "streaming")` check with:
```typescript
	const phaseCtx: PhaseContext = {
		...hookCtx,
		turnCap: DEFAULT_TURN_CAP,
	};
	yield* runPhase1Streaming(
		phaseCtx,
		binding,
		systemBlocks,
		initialMessages,
		processToolFn,
	);
```

The complete final `runStreamingHook.ts`:

```typescript
import { ConfigError } from "../../lib/errors";
import { getCompoundBinding } from "./compound-registry";
import { runPhase1Streaming } from "../../services/teacher";
import type { HookKind, HookContext, PhaseContext } from "./types";
import type { ToolResult } from "../../services/tool-processor";
import type { AnthropicContentBlock, AnthropicSystemBlock } from "../../services/llm";
import type { TeacherEvent } from "../../services/teacher";

type ProcessToolFn = (name: string, input: unknown) => Promise<ToolResult>;

const DEFAULT_TURN_CAP = 5;

export async function* runStreamingHook(
	hook: HookKind,
	hookCtx: HookContext,
	processToolFn: ProcessToolFn,
	systemBlocks: AnthropicSystemBlock[],
	initialMessages: Array<{
		role: "user" | "assistant";
		content: string | AnthropicContentBlock[];
	}>,
): AsyncGenerator<TeacherEvent> {
	const binding = getCompoundBinding(hook);
	if (!binding) {
		throw new ConfigError(
			`runStreamingHook: no binding registered for hook "${hook}"`,
		);
	}
	if (binding.mode !== "streaming") {
		throw new ConfigError(
			`runStreamingHook: binding for "${hook}" has mode "${binding.mode}", expected "streaming"`,
		);
	}
	const phaseCtx: PhaseContext = {
		...hookCtx,
		turnCap: DEFAULT_TURN_CAP,
	};
	yield* runPhase1Streaming(
		phaseCtx,
		binding,
		systemBlocks,
		initialMessages,
		processToolFn,
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/harness/loop/runStreamingHook.test.ts
```
Expected: PASS (4 tests: 2 error paths + 1 happy path)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/loop/runStreamingHook.ts apps/api/src/harness/loop/runStreamingHook.test.ts && git commit -m "feat(harness): complete runStreamingHook streaming delegation"
```

---

### Task 5: Wire chatV6 through runStreamingHook; delete buildChatBinding and chat()
**Group:** D (depends on Group C — Task 4)

**Behavior being verified:** `chatV6` delegates through `runStreamingHook` and yields `TeacherEvent` delta and done events; `buildChatBinding` and `chat` are no longer exported.
**Interface under test:** `chatV6(ctx, studentId, messages, dynamicContext)`

**Files:**
- Modify: `apps/api/src/services/teacher.ts`

No new test file is needed — add a test to the existing teacher test suite if one exists, otherwise create:
- Test: `apps/api/src/services/teacher.test.ts` (new, if it doesn't exist)

- [ ] **Step 1: Write the failing test**

Check whether `apps/api/src/services/teacher.test.ts` exists. If not, create it. Add:

```typescript
import { describe, expect, it, vi } from "vitest";

vi.mock("../services/llm", async (importOriginal) => {
  const actual = await importOriginal() as Record<string, unknown>;
  return { ...actual, callAnthropicStream: vi.fn() };
});
vi.mock("../services/memory", () => ({ buildMemoryContext: vi.fn().mockResolvedValue("") }));

import { callAnthropicStream } from "../services/llm";
import { chatV6 } from "../services/teacher";
import type { ServiceContext } from "../lib/types";

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
    vi.mocked(callAnthropicStream).mockResolvedValue(makeSSEStream("Hi"));

    const events = [];
    for await (const e of chatV6(stubCtx, "student1", [{ role: "user", content: "hello" }], "")) {
      events.push(e);
    }

    expect(events.some((e) => e.type === "delta")).toBe(true);
    expect(events.at(-1)?.type).toBe("done");
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun vitest run src/services/teacher.test.ts
```
Expected: PASS on current code (chatV6 already works with buildChatBinding), OR if teacher.test.ts was new, it passes. 

**Important:** The goal of this task is structural — delete `buildChatBinding` and `chat()`, and re-wire `chatV6` to use `runStreamingHook`. Run the test after the implementation to confirm the wiring still works.

Actually run `bun tsc --noEmit` first to verify baseline:
```bash
cd apps/api && bun tsc --noEmit 2>&1 | head -20
```

- [ ] **Step 3: Implement**

In `apps/api/src/services/teacher.ts`:

**3a. Add import at the top of the file** (after existing imports):
```typescript
import { runStreamingHook } from "../harness/loop/runStreamingHook";
```

**3b. Replace `chatV6` with the new implementation** (lines ~591–632):

```typescript
export async function* chatV6(
	ctx: ServiceContext,
	studentId: string,
	messages: Array<{
		role: "user" | "assistant";
		content: string | AnthropicContentBlock[];
	}>,
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

	const hookCtx: HookContext = {
		env: ctx.env,
		studentId,
		sessionId: "",
		conversationId: null,
		digest: {},
		waitUntil: (_p: Promise<unknown>) => {},
	};

	yield* runStreamingHook(
		"OnChatMessage",
		hookCtx,
		processToolFn,
		systemBlocks,
		messages,
	);
}
```

**3c. Delete `buildChatBinding`** (lines ~392–412, the entire exported function).

**3d. Delete `chat()`** (lines ~634–end of function, the entire exported function).

**3e. Delete `const MAX_TOOL_TURNS = 5;`** (line ~589, now unused).

**3f. Remove any imports made unused by the deletions.** Specifically:
- `HookContext` is now used by chatV6 → keep the import
- `PhaseContext` was used by buildChatBinding and runPhase1Streaming params → still needed (runPhase1Streaming signature)
- `CompoundBinding` was used by buildChatBinding → still needed (runPhase1Streaming param type)
- `getAnthropicToolSchemas` was used by `chat()` → remove from import if no longer used
- Check: is `getAnthropicToolSchemas` used anywhere else in the file? If no, remove from the tool-processor import line.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun vitest run src/services/teacher.test.ts && bun tsc --noEmit
```
Expected: PASS (vitest), no type errors (tsc)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/teacher.ts apps/api/src/services/teacher.test.ts && git commit -m "feat(chat): wire chatV6 through runStreamingHook, delete buildChatBinding and chat()"
```

---

### Task 6: Flip HARNESS_V6_CHAT_ENABLED and update status doc
**Group:** E (depends on Group D — Task 5)

**Behavior being verified:** Production flag is enabled; the V6 integration plan reflects Spec X as complete.
**Interface under test:** `wrangler.toml` config value + `docs/apps/00-status.md`

**Files:**
- Modify: `apps/api/wrangler.toml`
- Modify: `docs/apps/00-status.md`

- [ ] **Step 1: Write the failing test**

```bash
grep 'HARNESS_V6_CHAT_ENABLED' apps/api/wrangler.toml
```
Expected current output: `HARNESS_V6_CHAT_ENABLED = "false"`

- [ ] **Step 2: Verify current state**

```bash
grep 'HARNESS_V6_CHAT_ENABLED\|HARNESS_V6_ENABLED' apps/api/wrangler.toml
```
Expected: both lines visible with their current values.

- [ ] **Step 3: Implement**

In `apps/api/wrangler.toml`, change:
```toml
HARNESS_V6_CHAT_ENABLED = "false"
```
to:
```toml
HARNESS_V6_CHAT_ENABLED = "true"
```

In `docs/apps/00-status.md`, find the Spec X / OnChatMessage / chat harness migration status entry and mark it complete. If the entry reads "in-progress" or "planned", change it to "done" or "complete". Add note: "chatV6 routes through runStreamingHook("OnChatMessage"); legacy chat() and buildChatBinding deleted."

- [ ] **Step 4: Verify**

```bash
grep 'HARNESS_V6_CHAT_ENABLED' apps/api/wrangler.toml
```
Expected: `HARNESS_V6_CHAT_ENABLED = "true"`

- [ ] **Step 5: Commit**

```bash
git add apps/api/wrangler.toml docs/apps/00-status.md && git commit -m "feat(config): enable HARNESS_V6_CHAT_ENABLED, mark Spec X complete"
```
