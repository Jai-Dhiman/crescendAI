# Spec X — Chat Harness Migration Design

**Goal:** Route live chat through the V6 harness registry (`runStreamingHook("OnChatMessage")`) so that both chat and session-end flows share a single hook-dispatch architecture.
**Not in scope:** Changing the SSE wire protocol, adding new chat tools, multi-phase chat, or migrating the OnSessionEnd compound.

## Problem

`chatV6()` in `apps/api/src/services/teacher.ts` calls `buildChatBinding()` and `runPhase1Streaming()` directly, bypassing the compound-registry. This means:
- The registry's `OnChatMessage` slot is permanently unbound (`getCompoundBinding("OnChatMessage")` returns `undefined`)
- Chat tool definitions are duplicated outside the registry
- `buildChatBinding()` is an in-function factory that can't be independently tested or swapped
- The legacy `chat()` (5-turn inline loop) still ships alongside `chatV6()`

The flag `HARNESS_V6_CHAT_ENABLED` exists in `wrangler.toml` and `chat.ts` already routes to `chatV6` when true, but the gap between `chatV6` and the harness registry means the flag delivers no architectural benefit.

## Solution (from the user's perspective)

No visible change. The same SSE events (`delta`, `tool_start`, `tool_result`, `tool_error`, `done`) stream to the client. Tool use (create_exercise, search_catalog, etc.) continues to work. `HARNESS_V6_CHAT_ENABLED` now fully controls whether chat flows through the registry.

## Design

**Option chosen: full runHook routing via registry.** `chatV6` is re-wired to call a new thin function `runStreamingHook(hook, ...)` that reads the binding from the registry, asserts `mode === "streaming"`, and delegates to `runPhase1Streaming`. `buildChatBinding` and the legacy `chat()` are deleted.

**Key decisions:**

1. `runStreamingHook` lives in `apps/api/src/harness/loop/runStreamingHook.ts`. It imports `runPhase1Streaming` from `teacher.ts`, creating a soft circular: `teacher.ts` → `runStreamingHook.ts` → `teacher.ts`. Safe with esbuild — no module-level side effects; both files export named functions whose call graph is acyclic at runtime.

2. `OnChatMessage` binding in the registry uses the 6 tools from `TOOL_REGISTRY` (`tool-processor.ts`). Each is wrapped into a harness `ToolDefinition`. `invoke` is a stub (`async (_input) => ({})`) because `runPhase1Streaming` never calls `binding.tools[i].invoke` — it routes all invocation through the `processToolFn` parameter.

3. `ConfigError` is added to `lib/errors.ts` as a `DomainError` subclass. Thrown (not yielded) when `runStreamingHook` is called with an unregistered hook or a non-streaming binding.

4. `sessionId` fix: `chatV6` currently passes `hookCtx.sessionId = conversationId ?? ""`. This preserves existing behavior — no silent undefined in logs.

5. `HARNESS_V6_CHAT_ENABLED` is flipped to `"true"` only in the final task (Task 6), after all tests pass.

**Why not use `runHook` from `runHook.ts` directly?** `runHook` only calls the buffered `runPhase1` (non-streaming). Adding streaming support to `runHook` would require changing its return type to `AsyncGenerator<HookEvent | TeacherEvent>`, which breaks the OnSessionEnd callsite. A dedicated `runStreamingHook` keeps the two modes cleanly separated.

**Wire protocol preservation:** `chat.ts` route is unchanged. It still maps `TeacherEvent` → SSE. `runPhase1Streaming` still emits `TeacherEvent`. No format change.

## Modules

### `runStreamingHook` (new)
- **Interface:** `async function* runStreamingHook(hook, hookCtx, processToolFn, systemBlocks, initialMessages): AsyncGenerator<TeacherEvent>`
- **Hides:** registry lookup, mode assertion, `PhaseContext` construction, delegation to `runPhase1Streaming`
- **Tested through:** public function signature — error paths (no binding, wrong mode) and happy path (yields forwarded TeacherEvent stream)
- **Depth verdict:** DEEP — one-liner callers get a full registry-backed streaming hook, hiding three failure modes and context assembly

### `OnChatMessage` binding in `compound-registry`
- **Interface:** `getCompoundBinding("OnChatMessage")` → `CompoundBinding`
- **Hides:** TOOL_REGISTRY-to-ToolDefinition mapping, stub invoke wiring
- **Tested through:** existing compound-registry tests (add one assertion: returns defined binding with mode "streaming" and phases 1)
- **Depth verdict:** DEEP — callers see one object, implementation hides N tool-wrapping steps

### `ConfigError` in `lib/errors`
- **Interface:** `class ConfigError extends DomainError`
- **Hides:** nothing — pure type marker
- **Depth verdict:** SHALLOW — justified: it's a marker type, not a hiding module. Used for instanceof checks in error middleware.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/loop/runStreamingHook.ts` | New streaming hook dispatcher | New |
| `apps/api/src/harness/loop/runStreamingHook.test.ts` | Tests for error paths + happy path | New |
| `apps/api/src/harness/loop/compound-registry.ts` | Add OnChatMessage binding (TOOL_REGISTRY tools, mode streaming, phases 1) | Modify |
| `apps/api/src/harness/loop/compound-registry.test.ts` | Add assertion: OnChatMessage binding defined with correct shape | Modify |
| `apps/api/src/lib/errors.ts` | Add ConfigError extends DomainError | Modify |
| `apps/api/src/services/teacher.ts` | chatV6 calls runStreamingHook; delete buildChatBinding + chat() | Modify |
| `apps/api/wrangler.toml` | HARNESS_V6_CHAT_ENABLED = "true" | Modify |
| `docs/apps/00-status.md` | Mark Spec X complete | Modify |

## Open Questions

- Q: Should `ConfigError` be a 500 or 503 at the HTTP layer?  Default: 500 (misconfiguration, not transient).
