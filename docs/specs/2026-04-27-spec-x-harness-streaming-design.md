# Spec X — Harness Streaming + OnChatMessage Migration

**Goal:** Route `POST /api/chat` through the V6 harness loop's streaming primitive so both synthesis and chat share a single tool-registration surface, with zero change to the SSE wire protocol the web client consumes.

**Not in scope:**
- New tools, new artifact types, new pedagogy
- V6 Plans 2–4 (synthesis atoms, molecules, integration — proceed in parallel, not a blocker)
- V8a action tools (`assign_segment_loop`, `render_annotation`, `schedule_followup_interrupt`)
- Synthesis path changes (`runPhase1`, `runPhase2`, `runHook` remain untouched)
- Mobile surfaces
- Shadow mode / gradual traffic splitting — binary flag flip only
- Mid-stream Anthropic event types (`content_block_start`, `input_json_delta`) exposed to the wire client — these stay internal to `parseAnthropicStream`

---

## Problem

Two parallel agent loops exist today with no shared registration surface:

1. **`teacher.chat()`** — a streaming multi-turn loop in `services/teacher.ts` that calls `callAnthropicStream`, parses token-level events via `parseAnthropicStream`, and dispatches tools through `processToolUse` (6 tools in `TOOL_REGISTRY`).
2. **`teacher.synthesizeV6()`** — a buffered two-phase loop that calls `runHook("OnSessionEnd")`, uses a `CompoundBinding` with an empty `tools: []` list (atoms not yet built), and yields `HookEvent<SynthesisArtifact>`.

Every new tool must be registered in both `TOOL_REGISTRY` (chat path) and the synthesis `CompoundBinding` (harness path) independently, with different invocation semantics. V8a's `assign_segment_loop` action tool needs to reach the chat path — but since the chat path is not harness-aware, V8a would add a third registry entry with its own gating logic duplicated outside the `wrapToolCall` middleware the harness was designed to own. The divergence compounds with each new tool.

Concretely: `services/teacher.ts:chat()` at line 378 calls `getAnthropicToolSchemas()` which reads `TOOL_REGISTRY` unconditionally. `harness/loop/compound-registry.ts:14` has `tools: []`. These are two codepaths that will diverge further unless unified now, before V8a lands.

---

## Solution (from the user's perspective)

No visible change to the web client. Internally: when `HARNESS_V6_CHAT_ENABLED=true`, `POST /api/chat` calls `teacher.chatV6()` instead of `teacher.chat()`. Both functions have identical signatures and yield identical `TeacherEvent` sequences for the same input. The SSE events (`delta`, `tool_start`, `tool_result`, `tool_error`, `done`) are unchanged. `saveAssistantMessage` receives the same payload. The equivalence oracle in the test suite asserts this.

V8a's contribution is: add `assign_segment_loop` to the `OnChatMessage` binding's tool list. One registration site, one invocation path, one permission gate.

---

## Design

### Why the streaming path lives in `services/teacher.ts`, not in `harness/loop/`

`runPhase1Streaming` reuses `parseAnthropicStream` (already in `teacher.ts`) and must yield `TeacherEvent` (which carries `componentsJson: InlineComponent[]` for UI rendering). If the streaming path lived in `harness/loop/`, it would need to import `TeacherEvent`, `InlineComponent`, and `parseAnthropicStream` from the services layer — a harness→services dependency that inverts the intended direction. Keeping the streaming path in `teacher.ts` preserves the harness as a pure infrastructure concern.

### `CompoundBinding` type extension

`mode: 'streaming' | 'buffered'` and `phases: 1 | 2` are added as **required fields**. `artifactSchema` and `artifactToolName` become **optional** — only required when `phases === 2`. The existing `OnSessionEnd` binding gets `mode: 'buffered', phases: 2` added explicitly. All test fixtures update accordingly. `runHook` asserts `phases === 2` before calling `runPhase2`; if `phases === 1`, it stops after phase 1 (this path is not used by `OnSessionEnd` today, but it exists for V8a to characterize the chat binding in documentation).

### `buildChatBinding` — per-request factory

The `OnChatMessage` binding cannot be a static registry entry because its tools need `ServiceContext` (db, env) and `studentId` closed over at call time. `buildChatBinding(ctx, studentId)` constructs a `CompoundBinding` from `TOOL_REGISTRY`: for each entry, it wraps `processToolUse(ctx, studentId, name, input)` as the `invoke` closure and uses the tool's `anthropicSchema.input_schema` as the harness `ToolDefinition.input_schema`. The binding declares `mode: 'streaming', phases: 1`. `artifactSchema` and `artifactToolName` are omitted (not needed for a phase-1-only binding).

### `runPhase1Streaming` — streaming agentic loop

`runPhase1Streaming(ctx, binding, systemBlocks, initialMessages, processToolFn)` is the streaming equivalent of `runPhase1`. It mirrors the `chat()` multi-turn loop structure:

1. Call `callAnthropicStream` with `binding.tools` projected to Anthropic tool schemas.
2. Pass the stream to `parseAnthropicStream(stream, processToolFn)`.
3. Yield all non-`done` events (deltas, tool_start, tool_result, tool_error) to the caller.
4. On `done`: if `stopReason !== 'tool_use'` or no tool calls, yield the final `done` event (with `allComponents` accumulated across turns) and return.
5. Otherwise, build continuation messages from the assistant's content and tool results, and loop.
6. When `ctx.turnCap` exhausted: issue one final `callAnthropicStream` with `tool_choice: { type: 'none' }` to force a text response.

The critical difference from `runPhase1`: `runPhase1Streaming` uses the binding's `tools` array **for schemas only** (to build the Anthropic API tool list). Actual invocation goes through `processToolFn`, not `binding.tools[].invoke`. This is because chat tools return `ToolResult` (with `componentsJson` for UI rendering), which is incompatible with the generic `invoke: (input) => Promise<unknown>` shape. The `binding.tools[].invoke` closures are defined (they call `processToolUse`) but are not called by the streaming path — they exist for future buffered-chat use cases.

### `chatV6` — the public adapter

`chatV6(ctx, studentId, messages, dynamicContext)` has an identical signature to `chat()`. It:
1. Builds `systemBlocks` from `UNIFIED_TEACHER_SYSTEM` + `dynamicContext` (same as `chat()`).
2. Calls `buildChatBinding(ctx, studentId)`.
3. Constructs `PhaseContext` from `ctx.env`, `studentId`, and `turnCap: MAX_TOOL_TURNS` (5).
4. Delegates to `runPhase1Streaming`.

### Route dispatch

`routes/chat.ts` checks `c.env.HARNESS_V6_CHAT_ENABLED === 'true'`. When true, it calls `teacherService.chatV6`; when false, `teacherService.chat`. The SSE translation layer is unchanged — both functions yield `TeacherEvent`, which maps to the same SSE wire events.

### Equivalence oracle

A dedicated test runs both `chat()` and `chatV6()` against the same mock Anthropic stream fixture (one text-only turn + one tool-use turn with continuation). It asserts:
- The `TeacherEvent` arrays are identical element-by-element.
- Any divergence surfaces as a test failure before the flag is enabled in production.

### Flag discipline

`HARNESS_V6_CHAT_ENABLED` is a separate flag from `HARNESS_V6_ENABLED` (synthesis). Independent bisection: a chat regression does not require rolling back synthesis, and vice versa.

### Trade-offs

- **Streaming loop in services, not harness.** Avoids the harness→services dependency. The cost: the streaming primitive is not reusable for a hypothetical future streaming compound. If synthesis ever needs streaming, that's a new spec.
- **`processToolFn` injection over `binding.invoke`.** Prevents an `InlineComponent` type from leaking into `harness/loop/types.ts`. The cost: `binding.tools[].invoke` is defined but unused by the streaming path, which is mildly confusing. A comment documents this.
- **Binary flag, no shadow mode.** Simpler to implement and reason about. The equivalence oracle provides pre-production confidence in lieu of traffic splitting.
- **`buildChatBinding` as a per-request factory, not a static registry entry.** Necessary because chat tools close over `ServiceContext`. The cost: `OnChatMessage` is not visible in the compound registry's static `Map`, which V8a must know when adding its tool to the list.

---

## Modules

### `harness/loop/types.ts — CompoundBinding`
- **Interface:** `CompoundBinding` type with `mode`, `phases`, optional `artifactSchema`/`artifactToolName`
- **Hides:** The discriminated semantics of streaming vs buffered bindings; the validation constraint that `artifactSchema` is only required for phase-2 bindings
- **Tested through:** Existing `runPhase1` and `runPhase2` tests (regression); `buildChatBinding` test (new fields used correctly)
- **Depth:** DEEP — stable type contract, substantial semantics behind optional fields

### `services/teacher.ts — buildChatBinding`
- **Interface:** `(ctx: ServiceContext, studentId: string) => CompoundBinding`
- **Hides:** Mapping from `TOOL_REGISTRY` to harness `ToolDefinition` shape; closing `processToolUse` over ctx and studentId
- **Tested through:** Direct call, asserting tool names, mode, phases
- **Depth:** SHALLOW — justified as a necessary seam for V8a to extend; no significant hidden complexity

### `services/teacher.ts — runPhase1Streaming`
- **Interface:** `(ctx: PhaseContext, binding: CompoundBinding, systemBlocks: AnthropicSystemBlock[], initialMessages: Array<{role: 'user'|'assistant'; content: string | AnthropicContentBlock[]}>, processToolFn: ProcessToolFn) => AsyncGenerator<TeacherEvent>`
- **Hides:** Multi-turn streaming loop; continuation message construction; forced-final-call path on turn cap exhaustion; accumulated `allComponents` across tool turns
- **Tested through:** Direct call with mock fetch (SSE stream fixture) and mock processToolFn
- **Depth:** DEEP — complex state machine hidden behind a simple generator interface

### `services/teacher.ts — chatV6`
- **Interface:** Identical to `chat()`: `(ctx, studentId, messages, dynamicContext) => AsyncGenerator<TeacherEvent>`
- **Hides:** Binding construction, phaseCtx construction, routing to runPhase1Streaming
- **Tested through:** Equivalence oracle comparing output to `chat()`
- **Depth:** SHALLOW — justified as the surface callers use; thin adapter by design

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/loop/types.ts` | Add required `mode`, `phases`; make `artifactSchema`, `artifactToolName` optional | Modify |
| `apps/api/src/harness/loop/compound-registry.ts` | Add `mode: 'buffered', phases: 2` to `OnSessionEnd` binding | Modify |
| `apps/api/src/harness/loop/phase1.test.ts` | Add `mode: 'buffered', phases: 2` to `EMPTY_BINDING` and `capBinding` fixtures | Modify |
| `apps/api/src/harness/loop/phase2.test.ts` | Add `mode: 'buffered', phases: 2` to `BINDING` fixture | Modify |
| `apps/api/src/harness/loop/runHook.test.ts` | No fixture changes needed (HOOK_CTX unchanged; binding is fetched from registry) | Modify |
| `apps/api/src/services/teacher.ts` | Export `buildChatBinding`, `runPhase1Streaming`; add `chatV6` | Modify |
| `apps/api/src/services/teacher-chat-v6.test.ts` | New: streaming unit tests + equivalence oracle | New |
| `apps/api/src/routes/chat.ts` | Add `HARNESS_V6_CHAT_ENABLED` flag branch | Modify |
| `apps/api/src/lib/types.ts` | Add `HARNESS_V6_CHAT_ENABLED: string` to `Bindings` | Modify |
| `apps/api/wrangler.toml` | Add `HARNESS_V6_CHAT_ENABLED = "false"` | Modify |

---

## Open Questions

- **Q: Should `buildChatBinding` be in `compound-registry.ts` or `teacher.ts`?** Default: `teacher.ts`. Reason: it needs `ServiceContext` (import from `lib/types`), which would create a harness→lib dependency in compound-registry if moved there. The factory is chat-specific; it belongs alongside the chat service code.
- **Q: Should `runPhase1Streaming` be exported?** Default: yes. It is a meaningful unit with an injectable `processToolFn` parameter, testable in isolation without a real DB. Exporting it enables direct testing without a complex route-level harness.
