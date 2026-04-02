# Unified Teacher Voice

**Date:** 2026-03-31
**Status:** Design approved, pending implementation plan
**Scope:** Backend (API) + Frontend (Web) — no iOS, no model training

## Problem

CrescendAI has three separate LLM interaction paths that should be one unified teacher:

1. **`services/chat.ts`** — stateless text chat, no tools, no artifacts
2. **`services/ask.ts`** — two-stage Groq+Anthropic pipeline with `create_exercise` tool_use, completely dead (zero callers)
3. **SessionBrain DO synthesis** — post-session recap via `callSynthesisLlm`, no tools, no memory context

The teacher's strongest moat features (exercises, score rendering, artifacts) exist in code but are not wired to any live path. The student experiences a fragmented interaction: text chat with no tools, practice observations with stub placeholder text, and synthesis with no structured artifacts.

## Solution

A single **unified teacher service** (`teacher.ts`) that is the sole interface between CrescendAI and the teacher LLM. Both chat and post-session synthesis paths call `teacher.ts` with different contexts but the same tool palette and processing logic.

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM architecture | Single-stage (drop Groq) | Simpler, one fewer dependency. Anthropic now, Qwen3.5-27B fine-tune later |
| Tool protocol | Native tool_use (OpenAI-compatible) | Survives SFT on Qwen3.5-27B. De facto standard across Anthropic, Qwen, Mistral |
| SSE protocol | Typed events: `start`, `delta`, `tool_result`, `done` | 5 tools + structured synthesis need richer transport than text-only |
| Post-session delivery | Structured summary card with embedded artifacts | Not a text paragraph. Top moments, score highlights, exercises, personalized summary |
| Mid-practice LLM enrichment | Deferred to Phase 2 | DO accumulates WASM observations during practice. LLM enrichment adds latency + DO complexity |
| Fine-tune base model | Qwen3.5-27B | Apache 2.0, native FC, hybrid thinking, 262K context, best fine-tuning ecosystem |
| Prompt caching | Explicit cache boundary (static/dynamic split) | CC pattern: byte-identical static section cached across all students |
| Synthesis quality | `<analysis>` scratchpad (stripped before persistence) | CC pattern: model reasons in scratchpad, improves output, scratchpad never stored |

---

## 1. Unified Teacher Service (`teacher.ts`)

### Interface

```typescript
type TeacherContext =
  | ChatContext         // { memoryContext, studentProfile }
  | SynthesisContext    // { accumulator, sessionDuration, practicePattern, pieceMetadata, memoryContext }
  | EnrichmentContext;  // Phase 2: { teachingMoment, recentObservations }

// Streaming events yielded by teacher.chat()
type TeacherEvent =
  | { type: "delta"; text: string }
  | { type: "tool_result"; name: string; componentsJson: InlineComponent[] }
  | { type: "done"; fullText: string; allComponents: InlineComponent[] };

// Non-streaming response from teacher.synthesize()
type TeacherResponse = {
  text: string;                            // <analysis> already stripped
  toolResults: ToolResult[];               // processed, persisted, ready for frontend
};
```

### Entry Points

- **`teacher.chat(ctx, studentId, input)`** — Returns `AsyncGenerator<TeacherEvent>`. Internally: builds `ChatContext`, assembles prompt, calls `callAnthropicStream`, parses the Anthropic SSE stream (buffering `input_json_delta` per tool block, processing tools at `content_block_stop`), yields typed events. The route iterates this generator and writes SSE events. This keeps the route as a thin transport layer and makes the stream parser independently testable.
- **`teacher.synthesize(ctx, synthContext)`** — Non-streaming. Builds `SynthesisContext` from DO accumulator, calls LLM with full tool palette, returns `TeacherResponse`. Strips `<analysis>` scratchpad from text before returning. Throws on LLM failure (does NOT swallow errors or return fallback text — the caller handles retries).

### DO Caller Pattern

The DO creates `ServiceContext` manually (`{ db: createDb(env.HYPERDRIVE), env }`) since it has no Hono middleware. This is intentional and matches the existing pattern in `session-brain.ts`. `teacher.ts` accepts `ServiceContext` and is agnostic to how it was created.

### Prompt Assembly

```
[Static — cached across all students, cache_control: { type: "ephemeral" }]
  Teacher identity + pedagogy principles
  Dimension definitions (dynamics, timing, pedaling, articulation, phrasing, interpretation)
  Calibration notes (MuQ R2~0.5, deviation thresholds)
  All 5 tool schemas + usage guidance
  Teaching voice guidelines

TEACHER_PROMPT_DYNAMIC_BOUNDARY

[Dynamic — per-student, per-session, no cache control]
  Student profile (level, goals, baselines)
  Memory context (synthesized facts + recent observations)
  Mode-specific framing:
    - Chat: "you are in a conversation with your student"
    - Synthesis: "you just listened to a practice session" + accumulator data + <analysis> instruction
    - Enrichment (Phase 2): "enrich this observation"
```

Tool schemas live in the static section (all 5 always present). The mode-specific section guides WHEN to use them, not WHICH ones exist. This maximizes prompt cache hit rate.

### Model Abstraction

A `callTeacher` function wraps `callAnthropic` / `callAnthropicStream`. When swapping to Qwen3.5-27B, this is the ONE function that changes. The tool schema format is OpenAI-compatible, so the same `tools` array works with both providers.

For title generation (currently Groq), switch to Workers AI via the existing `AI_GATEWAY_BACKGROUND` HTTP path.

---

## 2. Tool Definitions and Processing

### Tool Registry

```typescript
type ToolDefinition = {
  name: string;
  schema: ZodSchema;              // Zod validation
  anthropicSchema: object;        // JSON Schema for LLM
  description: string;            // shown to LLM
  concurrencySafe: boolean;       // read-only = true, mutations = false
  maxResultChars?: number;        // overflow protection
  process: (ctx: ServiceContext, input: unknown) => Promise<ToolResult>;
};

const TEACHER_TOOLS: Record<string, ToolDefinition> = {
  create_exercise:   { concurrencySafe: false, maxResultChars: undefined, ... },
  score_highlight:   { concurrencySafe: true,  maxResultChars: 5000, ... },
  keyboard_guide:    { concurrencySafe: true,  maxResultChars: 2000, ... },
  show_session_data: { concurrencySafe: true,  maxResultChars: 10000, ... },
  reference_browser: { concurrencySafe: true,  maxResultChars: 2000, ... },
};
```

### Tool Descriptions

**`create_exercise`** — Create focused practice exercises. Hybrid catalog (25 curated) + LLM-generated fallback, persisted to DB.
- Schema: `{ source_passage: string, target_skill: string, exercises: [{ title, instruction, focus_dimension, hands? }] }`
- Returns: `{ type: "exercise_set", config: { sourcePassage, targetSkill, exercises } }`

**`score_highlight`** — Render annotated score passages for specific bars.
- Schema: `{ bars: z.string().regex(/^\d+(-\d+)?$/), annotations?: string[], piece_id?: z.string().uuid().optional() }`
- Server: validates `piece_id` against score catalog before R2 lookup (no path traversal). If `piece_id` absent or not in catalog, returns text-only annotation card (no rendered score).
- Returns: `{ type: "score_highlight", config: { bars, scoreData, annotations } }`

**`keyboard_guide`** — Fingering and hand position visualization.
- Schema: `{ title: string, description: string, fingering?: string, hands: "left" | "right" | "both" }`
- Returns: `{ type: "keyboard_guide", config }`

**`show_session_data`** — Pull up past practice data on demand.
- Schema: `{ query_type: "dimension_history" | "recent_sessions" | "session_detail", dimension?: z.enum(DIMS_6).optional(), session_id?: z.string().uuid().optional(), limit?: z.number().int().min(1).max(50).default(20) }`
- Server: ALL queries enforce `AND student_id = $studentId` (the LLM cannot access other students' data). DB queries use `LIMIT 50` ceiling regardless of input. Queries `observations`, `sessions` tables, accumulator JSON.
- Returns: `{ type: "session_data", config: { queryType, data } }`
- Truncated at `maxResultChars: 10000` with notice: "[Showing most recent data. Ask me to narrow the time range for more detail.]"

**`reference_browser`** — Reference performances and recordings.
- Schema: `{ piece_id?: z.string().uuid().optional(), passage?: string, description: string }`
- Server: validates `piece_id` against score catalog before lookup (same as `score_highlight`)
- Returns: `{ type: "reference_browser", config }`

### Processing Pipeline

1. LLM returns `tool_use` content block
2. Validate input against Zod schema
3. Check `concurrencySafe` — partition into parallel (read-only) and serial (mutations) batches
4. Execute tool logic (DB queries, R2 lookups, exercise persistence)
5. Check `maxResultChars` — truncate with summary if exceeded
6. Return `ToolResult` with `componentsJson` ready for frontend
7. **Guarantee**: every `tool_use` gets a `tool_result`, even on failure (`is_error: true`)

### What Migrates from ask.ts

- `processExerciseToolCall` → `processExercise` (same logic, cleaner interface)
- `lookupCatalogExercises` → used by `processExercise`
- `persistGeneratedExercise` → used by `processExercise`
- `postProcessObservation` → text cleanup utility
- `dimensionDescription` → dimension label utility
- Everything else deleted (Groq pipeline, subagent types, `handleAskInner`)

---

## 3. SSE Protocol and Chat Route

### Evolved SSE Events

```
Event: start        → { conversationId: string }
Event: delta        → raw text chunk
Event: tool_result  → { name: string, componentsJson: string }
Event: done         → { messageId: string }
Event: error        → { message: string }
```

### Streaming Tool Execution

Following the Claude Code `StreamingToolExecutor` pattern: tool processing starts at `content_block_stop`, not at `message_stop`. When Anthropic's stream emits a completed `tool_use` block:

1. Buffer `input_json_delta` as raw string during streaming (no partial parsing)
2. At `content_block_stop`: parse the full JSON, process through tool registry
3. Emit `tool_result` SSE event immediately — the student sees artifacts appear while the teacher is still "typing"
4. Continue streaming remaining text blocks

### Chat Route Changes

`routes/chat.ts` stays thin (transport layer):
- Calls `teacher.chat(ctx, studentId, body)` which returns `AsyncGenerator<TeacherEvent>`
- The route iterates the generator inside `streamSSE()`, mapping each `TeacherEvent` to an SSE event
- At `done` event: the generator yields `{ fullText, allComponents }` — the route passes these to `saveAssistantMessage` in `waitUntil` for persistence (text + `componentsJson` as proper JSONB)
- Title generation switches from Groq to Workers AI (`Qwen3-30b-a3b-fp8` via `AI_GATEWAY_BACKGROUND`)

### Multi-Turn History with tool_use

For beta, conversation history replayed to Anthropic includes only `role: "user" | "assistant"` text content. Tool_use/tool_result pairs are NOT replayed in history. The teacher doesn't need to remember which tools it called — the student's context and memory provide continuity. Tool results are visible in the frontend via persisted `componentsJson`. Revisit if context length becomes a concern.

### Streaming Error Recovery

If the stream errors mid-response:
- Send SSE `error` event
- In `waitUntil`: persist accumulated text (partial is better than nothing)
- Do NOT retry — client already received partial deltas

---

## 4. Post-Session Synthesis Path

### Flow

```
Student ends listening mode
  → DO alarm fires
  → DO builds SynthesisContext from accumulator + calls buildMemoryContext
  → DO calls teacher.synthesize(ctx, synthContext)  // non-streaming
  → teacher.ts assembles prompt (with <analysis> instruction) + all 5 tools
  → LLM responds with <analysis>...</analysis> + text + tool_use
  → teacher.ts strips <analysis>, processes tools through registry
  → Returns TeacherResponse { text, toolResults }
  → DO persists message with componentsJson
  → DO sends WS synthesis event { text, components, isFallback }
  → Frontend renders structured summary card
```

### DO Changes

- Replace `callSynthesisLlm(env, promptContext)` with `teacher.synthesize(ctx, synthContext)`
- Wire `buildMemoryContext(ctx, studentId)` into synthesis (currently a TODO)
- The claim-before-await / state versioning pattern stays identical
- Synthesis WS event shape evolves: `{ type: "synthesis", text, components: InlineComponent[], isFallback }`

### `<analysis>` Scratchpad

Synthesis mode prompt instructs the model to write `<analysis>` first (reason about which moments matter, how to frame feedback, what exercises would help). `teacher.ts` strips it:

```typescript
text = text.replace(/<analysis>[\s\S]*?<\/analysis>/g, "").trim();
```

The analysis improves synthesis quality without bloating conversation history. When we switch to Qwen3.5's thinking mode, the native `<think>` tags serve the same purpose.

### Deferred Synthesis

`POST /api/practice/synthesize` routes through the DO — calls `stub.fetch()` which invokes the DO's synthesis method. The DO owns the `synthesisCompleted` flag atomically, eliminating the race condition between the alarm path and the deferred HTTP path. The DO calls `teacher.synthesize()` in both cases. This means the deferred path also gets memory context (via `buildMemoryContext` inside the DO).

### Synthesis Token Budget

`max_tokens: 2048` for synthesis mode (up from 1024). The `<analysis>` scratchpad adds ~300-500 tokens of reasoning before the actual response, and tool calls add structured output.

### Synthesis Timeout

Add an `AbortController` with 25-second timeout on the `fetch()` inside `callAnthropic` for synthesis calls. CF Workers has a 30-second CPU time limit — the 25s timeout leaves headroom for persistence.

---

## 5. Frontend Changes

### SSE Handler Evolution

`api.chat.send()` handles 4 event types (was 3):

- **`start`**, **`delta`** — unchanged (RAF-batched text append)
- **`tool_result`** — parse `componentsJson` into `InlineComponent[]`, append to streaming message's `components` array. Existing artifact rendering kicks in immediately.
- **`done`** — unchanged (invalidate queries, clear transients)

### New InlineComponent Implementations

Replace `PlaceholderCard` stubs:

- **`ScoreHighlightCard`** — annotated score passage via OSMD (promote existing dev-only `ScorePanel` to production)
- **`KeyboardGuideCard`** — styled card with fingering description + hand indicator (visual piano keyboard is follow-up)
- **`ReferenceBrowserCard`** — styled card linking to reference performance (embedded player is follow-up)
- **`SessionDataCard`** — new type: dimension trend sparklines, session list, or session detail depending on `queryType`

### Post-Session Summary Card (`SynthesisCard`)

New component for `messageType: "synthesis"` messages with `components[]`:

- Header: piece name, session duration, practice mode summary
- Top moments: dimension pills, each with inline artifacts (score highlights, exercises)
- Teacher's personalized summary text
- Suggested exercises (from tool_use results)

Built from the `synthesis` WS event (text + components) + session metadata.

### Handle `piece_identified` WS Event

Add the missing case in `usePracticeSession.ts`:
```typescript
case "piece_identified":
  setPieceInfo({ composer: msg.composer, title: msg.title, confidence: msg.confidence });
  break;
```

Surface in `ListeningMode` UI.

### Cleanup

- Consolidate 3x `API_BASE` into single `lib/config.ts` export
- Consolidate 4x `invalidateQueries` call sites into `invalidateConversation(id)` helper
- Remove dead: `generateMockSession()`, `ObservationThrottle.queued`, `onSynthesis` callback

---

## 6. Dead Code Removal

### Backend Deletions

| Target | Reason |
|---|---|
| `services/ask.ts` (entire file) | Dead. Useful parts migrate to `teacher.ts` / `tool-processor.ts` |
| `SUBAGENT_SYSTEM`, `TEACHER_SYSTEM` in prompts.ts | Replaced by unified teacher prompt |
| `buildSubagentUserPrompt`, `buildTeacherUserPrompt` in prompts.ts | Groq-specific |
| `exerciseToolDefinition()` in prompts.ts | Migrates to tool registry |
| `ExerciseTool`, `ObservationRow`, `CatalogExercise` in prompts.ts | Migrate to teacher.ts types |
| `classifyStop`, `computeRerankFeatures`, `matchPieceText`, `initWasm` in wasm-bridge.ts | Exported, never imported |
| `StopResult` in wasm-bridge.ts | Only used by dead `classifyStop` |
| `callGroq` in llm.ts | Groq removed entirely |
| `SessionState.isEval` | Stored, never read |
| `SessionState.pieceQuery` | Written, never read |
| `topMoments(dimensionWeights?)` unused parameter | Never passed |
| `callMuqEndpoint`/`callAmtEndpoint` `sessionEnding` parameter | Always false |
| `tryIdentifyPiece(_totalNotes)` unused parameter | Never used |

### DB Migration: `componentsJson` Storage Format

The `messages.componentsJson` column is `jsonb` but currently stores a doubly-serialized string (`JSON.stringify(component)` stored inside JSONB). Fix: store as proper JSONB (parsed object). Add a Drizzle migration to deserialize existing data.

### DB Columns Flagged (Not Removed)

- `observations.elaborationText`, `.learningArc`, `.messageId` — never written
- `teachingApproaches` table — only written by dead `storeTeachingApproach`

These get `TODO: drop in next migration` comments. Schema changes require Drizzle migration.

### Frontend Deletions

- `generateMockSession()` in `mock-session.ts`
- `ObservationThrottle.queued` field
- `onSynthesis` callback from `PracticeSessionOptions`
- Three `PlaceholderCard` branches (replaced by real components)

---

## 7. Error Handling

### Teacher Service Errors

| Failure Mode | Chat Path | Synthesis Path |
|---|---|---|
| LLM call fails | SSE `error` event. No fallback text. | DO: `needsSynthesis` stays true. Deferred path retries. |
| Tool processing fails | Non-fatal. Text still delivers. `is_error: true` result sent to Anthropic (multi-turn correctness) but NOT emitted as `tool_result` SSE to frontend. Log structured error. | Non-fatal. Text synthesis still persists. Failed tool omitted from components. |
| Memory context unavailable | Proceed without memory. Log warning. | Same. |
| Student not found | Throw `NotFoundError` (domain error). | Same. |
| Stream errors mid-response | SSE `error` event. Persist partial text in `waitUntil`. No retry. | N/A (non-streaming). |
| Tool result overflow | Truncate at `maxResultChars` with notice. | Same. |

### Guarantees

- **Anthropic protocol**: Every `tool_use` block gets a `tool_result` sent back to Anthropic (even on error: `is_error: true`). This is required for multi-turn correctness.
- **Frontend SSE**: Only successful tool results are emitted as `tool_result` SSE events. Failed tools are silent to the student.
- Tool failures never prevent text delivery
- Partial streaming responses are persisted (better than nothing)
- `teacher.synthesize()` throws on LLM failure (no silent fallbacks per project principles). The caller handles retries.

---

## 8. Testing Strategy

### Unit Tests (Vitest + PGlite)

1. **Stream parser** — Feed synthetic Anthropic SSE bytes (interleaved text + tool blocks, partial JSON deltas, error mid-stream) to the `AsyncGenerator<TeacherEvent>` and verify correct event sequence. This is the highest-risk new code.
2. **Tool registry** — each processor independently: Zod validation, DB persistence, error cases, overflow truncation, `session_id` ownership enforcement, `piece_id` catalog validation
3. **Prompt assembly** — static section byte-identical across contexts, dynamic section includes correct mode-specific data, `<analysis>` instruction present in synthesis only
4. **Tool concurrency** — read-only tools marked safe, mutations marked unsafe, partitioning correct
5. **`<analysis>` stripping** — regex handles: normal case, multiple blocks (non-greedy correctness), missing close tag, empty analysis
6. **SSE protocol** — correct event shapes for all 5 types (start, delta, tool_result, done, error)

### Integration Tests

1. **Chat with tool_use** — mock Anthropic, verify SSE stream emits `delta` + `tool_result` in order, `componentsJson` persisted as proper JSONB with message
2. **Chat with failed tool** — mock Anthropic with tool_use that fails Zod validation, verify `is_error` sent to Anthropic but NOT emitted as SSE `tool_result`, text still delivered
3. **Synthesis with tool_use** — mock Anthropic, verify DO persists message with `componentsJson`, WS event includes `components`, `<analysis>` stripped
4. **Synthesis race** — fire alarm path and deferred path concurrently, verify exactly one synthesis message persisted (deferred routes through DO)
5. **Deferred synthesis with memory** — verify `POST /api/practice/synthesize` routes through DO, calls `teacher.synthesize()` with accumulator AND memory context
6. **Streaming error mid-response** — simulate stream failure after partial deltas, verify partial text persisted in `waitUntil`, SSE `error` event sent

---

## Phase 2 (Future)

- **Mid-practice LLM enrichment**: DO calls `teacher.enrichment()` during practice to enrich WASM teaching moments. Fire-and-forget with graceful degradation.
- **Student Practice Memory**: Structured per-student markdown (CC session memory pattern), extracted during synthesis, loaded into dynamic prompt section.
- **Qwen3.5-27B swap**: Change `callTeacher` function, include tool-call examples in SFT dataset, preserve chat template.
- **Visual piano keyboard**: Real keyboard rendering for `keyboard_guide` component.
- **Embedded audio player**: For `reference_browser` component.
- **Feature flags**: Gate major changes (`CRESCEND_TEACHER_FINETUNED`, `CRESCEND_SYNTHESIS_V2`) via wrangler env vars.
