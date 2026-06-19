# Chat GLM Migration Design

**Goal:** Migrate the V6 chat teacher from Anthropic claude-sonnet-4 streaming to
glm-4.7-flash on Cloudflare Workers AI via the AI Gateway, with zero Anthropic spend
while preserving the existing SSE contract, system-block memory injection, and tool-calls.

**Not in scope:**
- Migrating `synthesize()` or `synthesizeV6()` (those remain on Anthropic / V6 harness)
- Migrating `callAnthropic()` (non-streaming path, not used by chat)
- Changing the web SSE event contract (`start/delta/tool_start/tool_result/tool_error/done`)
- Any UI or iOS changes
- Production deploy (`just deploy-api` is a separate, deliberate step)

---

## Problem

`runPhase1Streaming` in `apps/api/src/services/teacher.ts` (lines 397–562) always calls
`callAnthropicStream` (in `llm.ts`), which sends every chat turn to Anthropic. This
blocks the goal of zero-Anthropic-spend for routine chat while the full eval e2e harness
is being built.

Three concrete gaps block the Workers AI path for streaming:

1. **`toOpenAIChatRequest` in `tool-format.ts` drops the `system` field entirely.**
   `AnthropicChatRequest` has no `system` field in its type, so the `UNIFIED_TEACHER_SYSTEM`
   prompt and the `<student_memory>` dynamic context block — both passed as `systemBlocks`
   in `chatV6` — are silently omitted when translating to OpenAI format. Memory recall and
   teacher persona both fail silently.

2. **`toOpenAIChatRequest` does not map `tool_choice: {type:"none"}` to `"none"`.**
   The forced-final-turn call in `runPhase1Streaming` (line 511) passes
   `tool_choice: {type:"none"}` to prevent further tool calls after the turn cap is
   exhausted. The current translation falls through to `"auto"`, causing the model to
   potentially loop on tools even in the forced final turn.

3. **There is no streaming path for Workers AI.**
   `gateway-client.ts callModel()` is non-streaming only. `runPhase1Streaming` has no
   branch for `provider === "workers-ai"`.

---

## Solution (from the user's perspective)

When `TEACHER_PROVIDER` is not set to `"anthropic"` (or is absent and `TEACHER_MODEL`
points to `@cf/zai-org/glm-4.7-flash`), all chat turns route through Workers AI via the
AI Gateway. The SSE stream arriving at the browser is byte-for-byte identical in event
shape. Memory-recall questions surface `synthesized_facts` stored in Postgres. Tool calls
(e.g. `prescribe_exercise`) work if glm handles streamed tool-calls reliably; if streaming
tool-calls prove unreliable for glm, a `CHAT_TOOLS_ENABLED=false` env switch disables
tools (text-only mode) and files a follow-up issue.

---

## Design

### Approach

Two-layer fix:

**Layer 1 — `tool-format.ts` data translation fixes (Tasks 1a, 1b).**
`AnthropicChatRequest` gains a `system` field (`string | AnthropicSystemBlock[]`) that
matches the shape used in `llm.ts`. `toOpenAIChatRequest` joins text blocks (stripping
`cache_control`) and prepends a `{role:"system", content: "..."}` message to the
converted message list. A second fix maps `tool_choice: {type:"none"}` → `"none"`.

**Layer 2 — streaming plumbing (Tasks 2, 3, 4).**
`llm.ts` gains `callWorkersAIStream(env, body)` that POSTs with `stream:true` and returns
`res.body`. `teacher.ts` gains `parseOpenAIStream` that reads OpenAI SSE
(`choices[].delta.*`) and yields the same `TeacherEvent` shapes as `parseAnthropicStream`.
`runPhase1Streaming` branches on `routeModel(ctx.env).provider`: Anthropic path is
unchanged; Workers AI path uses the new functions.

### Key decisions

- **`AnthropicChatRequest` gets a `system` field** rather than creating a new type. The
  existing `llm.ts` `AnthropicRequest` already has this field; aligning the harness type
  is the minimal surgical change.
- **System blocks are joined by `\n\n`**, `cache_control` stripped, before becoming the
  leading system message. This is correct for OpenAI-compat APIs.
- **Tool-call streaming accumulator is robust to both streaming (fragment) and
  non-streaming (single-chunk) tool-call delivery**, since glm may differ from strict
  OpenAI streaming behavior.
- **Tools decision point is explicit.** After Task 4 is committed, verify live that glm
  streams tool-calls. If unreliable, set `tool_choice:"none"` on the Workers AI path
  (text-only) and file a follow-up issue — memory recall and Q&A work without tools.
- **`routeModel` is reused as-is.** It already returns `{provider:"workers-ai",
  model:"@cf/zai-org/glm-4.7-flash"}` when `TEACHER_PROVIDER != "anthropic"`.

---

## Modules

### `apps/api/src/harness/loop/tool-format.ts` (Modify)

**Interface:**
- `toOpenAIChatRequest(req: AnthropicChatRequest): OpenAIChatRequest`
  — `AnthropicChatRequest.system` added: `string | AnthropicSystemBlock[]`
  — system blocks joined and prepended as `{role:"system"}` message
  — `tool_choice: {type:"none"}` → `"none"`
- `toAnthropicResponse` — unchanged

**Hides:** Anthropic-to-OpenAI schema translation including system block join logic and
tool choice normalization.

**Depth verdict:** DEEP — one function hides the entire cross-format translation surface.

---

### `apps/api/src/services/llm.ts` (Modify)

**Interface:**
- `callWorkersAIStream(env: Bindings, body: AnthropicChatRequest): Promise<ReadableStream>`
  — POSTs `{...toOpenAIChatRequest(body), stream:true}` to workers-ai endpoint
  — Returns `res.body` (raw ReadableStream of SSE bytes)

**Hides:** Auth headers, endpoint construction, response validation, stream body extraction.

**Depth verdict:** DEEP — identical signature simplicity to `callAnthropicStream`.

---

### `apps/api/src/services/teacher.ts` (Modify)

**Interface (new export):**
- `parseOpenAIStream(stream: ReadableStream, processToolFn: ProcessToolFn): AsyncGenerator<TeacherEvent>`
  — Reads OpenAI SSE (`choices[].delta.content`, `choices[].delta.tool_calls[]`)
  — Accumulates tool-call fragments by index
  — Handles single-chunk tool-call delivery (no fragments, entire tool_call in one chunk)
  — Yields same `TeacherEvent` union as `parseAnthropicStream`

**Interface (modified):**
- `runPhase1Streaming` — branches on `routeModel(ctx.env).provider`:
  - `"anthropic"`: unchanged `callAnthropicStream` + `parseAnthropicStream`
  - `"workers-ai"`: `callWorkersAIStream` + `parseOpenAIStream`
  - Forced-final-turn works correctly because `tool_choice:{type:"none"}` now maps to `"none"` (Task 1b)

**Hides:** Provider routing, SSE parsing differences between Anthropic and OpenAI SSE
formats, tool-call fragment accumulation, text delta buffering.

**Depth verdict:** DEEP — caller (and all tests) see only `TeacherEvent` regardless of
which provider is active.

---

## Verification Architecture

**Canonical success state:**
1. Offline: `bun run test` in `apps/api/` passes with zero regressions.
2. Live: A chat turn at `http://localhost:8787/api/chat` returns an SSE stream with
   `event: delta` lines containing teacher text, without any Anthropic request appearing
   in the AI Gateway log.
3. Memory recall: A chat turn asking "what did you notice last time?" surfaces a seeded
   `synthesized_facts` row from local Postgres.

**Automated checks (offline):**
- `cd apps/api && bun run test src/harness/loop/tool-format.test.ts` — two new tests
  (system block translation, `tool_choice:none` mapping)
- `cd apps/api && bun run test src/services/teacher.test.ts` — three new tests
  (parseOpenAIStream: text-only, streamed tool-call, single-chunk tool-call)
- `cd apps/api && bun run test` — full suite green

**Harness:** No separate harness buildable before the feature. Live verification is manual
(curl or browser) after `just api` starts. Instructions in Task 4 verification step.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/loop/tool-format.ts` | Add `system` to `AnthropicChatRequest`; translate system blocks to leading system message; map `tool_choice:{type:"none"}` to `"none"` | Modify |
| `apps/api/src/harness/loop/tool-format.test.ts` | Add two tests: system block → leading message; `none` mapping | Modify |
| `apps/api/src/services/llm.ts` | Add `callWorkersAIStream(env, body)` | Modify |
| `apps/api/src/services/teacher.ts` | Add `parseOpenAIStream`; branch `runPhase1Streaming` on provider | Modify |
| `apps/api/src/services/teacher.test.ts` | Add `parseOpenAIStream` tests (3 SSE fixture scenarios) | New |

---

## Open Questions

- **Q:** Will glm-4.7-flash reliably stream tool-calls in fragments (vs. a single chunk)?
  **Default:** Handle both (fragment accumulator + single-chunk detection). After Task 4,
  verify live with a `prescribe_exercise` trigger. If unreliable, switch to `"none"` on
  the Workers AI path (text-only) and file a follow-up issue — this is an explicit
  decision point in Task 4.
